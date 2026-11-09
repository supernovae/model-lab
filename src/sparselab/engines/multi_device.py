"""Multi-device microbatch-parallel execution helpers and process coordinator."""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing import resource_tracker
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import torch
from torch.nn import functional

from sparselab.config.models import AdamWConfig, RunConfig
from sparselab.engines.base import (
    EngineNonFiniteError,
    Microbatch,
    UpdateResult,
)
from sparselab.model.inspection import architecture_metrics, named_tensor_inventory
from sparselab.model.transformer import DenseLM
from sparselab.research.portability import apply_trainable_parameter_filter
from sparselab.runtime import (
    _auto_backend,
    precision_context,
    synchronize,
    torch_device_for,
)
from sparselab.training.optimizer import learning_rate_for_step
from sparselab.training.pilot_progress import current_pilot_progress

if TYPE_CHECKING:
    from sparselab.engines.pytorch import PyTorchEngine


def _default_output_projection_cross_entropy(
    hidden: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Project in BF16 and accumulate cross-entropy in FP32, as in training."""
    with torch.autocast(device_type=hidden.device.type, dtype=torch.bfloat16):
        logits = functional.linear(hidden, weight)
    return functional.cross_entropy(
        logits.float().flatten(0, 1),
        targets.flatten(),
        ignore_index=-100,
        reduction="sum",
    )


@dataclass
class SecondaryReplica:
    """Replica model bound to a secondary execution device."""

    device: torch.device
    model: DenseLM
    compiled_output_head_loss: Any = None


def execute_replica_microbatches(
    replica: SecondaryReplica,
    config: RunConfig,
    pairs: list[tuple[Microbatch, int]],
    valid_targets: int,
    output_head_loss_fn: Any = _default_output_projection_cross_entropy,
) -> dict[str, Any]:
    """Run forward and backward on a slice of microbatches on a replica device."""
    device = replica.device
    model = replica.model
    model.zero_grad(set_to_none=True)
    language_sum = torch.zeros((), device=device)
    aux_sum = torch.zeros((), device=device)
    diagnostic_sums: dict[str, torch.Tensor] = {}
    losses_finite = torch.ones((), device=device, dtype=torch.bool)
    recomputed = 0.0
    executed_microbatches = 0
    trainable_inputs = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    for batch, chunk_valid in pairs:
        if not chunk_valid:
            continue
        executed_microbatches += 1
        x = torch.from_numpy(batch.inputs).to(device)
        y = torch.from_numpy(batch.targets).to(device)
        addresses = (
            None
            if batch.byte_addresses is None
            else torch.from_numpy(batch.byte_addresses).to(device)
        )
        valid_target_mask = y != -100
        with precision_context(config, device):
            logits, auxiliary = model.forward_with_aux(
                x,
                byte_addresses=addresses,
                semantic_queries=None,
                memory_mask=None,
                valid_target_mask=valid_target_mask,
                activation_checkpointing=config.runtime.memory.activation_checkpointing.enabled,
                diagnostics=config.logging.architecture_diagnostics,
                return_hidden=config.runtime.compile_output_head_loss,
            )
        diagnostics = architecture_metrics(model)
        if config.runtime.compile_output_head_loss:
            loss_fn = replica.compiled_output_head_loss or output_head_loss_fn
            ce_sum = loss_fn(logits, y, model.output.weight)
        else:
            ce_sum = functional.cross_entropy(
                logits.float().flatten(0, 1),
                y.flatten(),
                ignore_index=-100,
                reduction="sum",
            )
        auxiliary = auxiliary.float()
        losses_finite = (
            losses_finite & torch.isfinite(ce_sum) & torch.isfinite(auxiliary)
        )
        backward_loss = (ce_sum + auxiliary * chunk_valid) / valid_targets
        torch.autograd.backward(backward_loss, inputs=trainable_inputs)
        recomputed += model.recomputed_block_call_ratio
        language_sum = language_sum + ce_sum.detach()
        aux_sum = aux_sum + auxiliary.detach() * chunk_valid
        for name, value in diagnostics.items():
            diagnostic_sums[name] = (
                diagnostic_sums.get(name, value.new_zeros(())) + value * chunk_valid
            )

    synchronize(device)
    return {
        "language_sum": language_sum,
        "aux_sum": aux_sum,
        "losses_finite": losses_finite,
        "recomputed": recomputed,
        "executed_microbatches": executed_microbatches,
        "diagnostic_sums": diagnostic_sums,
    }


def _build_shm_specs(
    config: RunConfig,
) -> tuple[
    list[tuple[str, tuple[int, ...], int, int]],
    list[tuple[str, tuple[int, ...], int, int]],
    int,
    int,
]:
    inventory = named_tensor_inventory(
        config.model,
        config.attention,
        trainable_parameters=config.training.trainable_parameters,
    )
    weight_specs: list[tuple[str, tuple[int, ...], int, int]] = []
    grad_specs: list[tuple[str, tuple[int, ...], int, int]] = []
    weight_offset = 0
    grad_offset = 0
    for name, spec in inventory.items():
        if spec.alias_of is not None:
            continue
        numel = spec.numel
        weight_specs.append((name, spec.shape, weight_offset, numel))
        weight_offset += numel
        if spec.trainable:
            grad_specs.append((name, spec.shape, grad_offset, numel))
            grad_offset += numel
    return weight_specs, grad_specs, max(1, weight_offset), max(1, grad_offset)


def _unpack_shm_to_model(
    model: DenseLM,
    weight_view: np.ndarray,
    weight_specs: list[tuple[str, tuple[int, ...], int, int]],
    device: torch.device,
) -> None:
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, shape, offset, numel in weight_specs:
            param = named_params[name]
            arr = weight_view[offset : offset + numel].reshape(shape)
            param.copy_(torch.from_numpy(arr).to(device=device, dtype=param.dtype))


def _coordinator_worker_loop(
    config_payload: dict[str, Any],
    worker_device_index: int,
    worker_env_overrides: dict[str, str],
    weight_shm_name: str,
    total_weight_floats: int,
    grad_shm_name: str,
    total_grad_floats: int,
    weight_specs: list[tuple[str, tuple[int, ...], int, int]],
    grad_specs: list[tuple[str, tuple[int, ...], int, int]],
    conn: Connection,
) -> None:
    weight_shm: SharedMemory | None = None
    grad_shm: SharedMemory | None = None
    weight_view: np.ndarray | None = None
    grad_view: np.ndarray | None = None
    try:
        for key, value in worker_env_overrides.items():
            os.environ[str(key)] = str(value)
        config = RunConfig.model_validate(config_payload)
        backend = (
            _auto_backend()
            if config.runtime.backend == "auto"
            else config.runtime.backend
        )
        device = torch_device_for(backend, worker_device_index)
        weight_shm = SharedMemory(name=weight_shm_name, create=False)
        grad_shm = SharedMemory(name=grad_shm_name, create=False)
        for shm in (weight_shm, grad_shm):
            with contextlib.suppress(OSError, ValueError, KeyError):
                resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
        weight_view = np.ndarray(
            (total_weight_floats,), dtype=np.float32, buffer=weight_shm.buf
        )
        grad_view = np.ndarray(
            (total_grad_floats,), dtype=np.float32, buffer=grad_shm.buf
        )

        model = DenseLM(config.model, config.attention).to(device)
        apply_trainable_parameter_filter(model, config, None)
        compiled_loss = None
        if config.runtime.compile_output_head_loss:
            compiled_loss = torch.compile(
                _default_output_projection_cross_entropy, dynamic=False
            )
        if config.runtime.compile_transformer_blocks:
            for block in model.blocks:
                block.forward_with_aux = torch.compile(
                    block.forward_with_aux, dynamic=False
                )
        replica = SecondaryReplica(
            device=device,
            model=model,
            compiled_output_head_loss=compiled_loss,
        )
        conn.send(("ready", None))

        while True:
            message = conn.recv()
            command = message[0]
            if command == "shutdown":
                break
            if command == "preload_weights":
                _unpack_shm_to_model(model, weight_view, weight_specs, device)
                synchronize(device)
                conn.send(("preloaded", None))
                continue
            if command == "run_step":
                _, worker_pairs, valid_targets, need_unpack = message
                if need_unpack:
                    _unpack_shm_to_model(model, weight_view, weight_specs, device)
                out = execute_replica_microbatches(
                    replica,
                    config,
                    worker_pairs,
                    valid_targets,
                    _default_output_projection_cross_entropy,
                )
                named_params = dict(model.named_parameters())
                for name, _shape, offset, numel in grad_specs:
                    grad = named_params[name].grad
                    if grad is None:
                        grad_view[offset : offset + numel].fill(0.0)
                    else:
                        grad_view[offset : offset + numel] = (
                            grad.detach().float().cpu().contiguous().numpy().reshape(-1)
                        )
                model.zero_grad(set_to_none=True)
                conn.send(
                    (
                        "step_done",
                        {
                            "language_sum": out["language_sum"].detach().cpu().numpy(),
                            "aux_sum": out["aux_sum"].detach().cpu().numpy(),
                            "losses_finite": bool(
                                out["losses_finite"].detach().cpu().item()
                            ),
                            "recomputed": float(out["recomputed"]),
                            "executed_microbatches": int(out["executed_microbatches"]),
                            "diagnostic_sums": {
                                k: v.detach().cpu().numpy()
                                for k, v in out["diagnostic_sums"].items()
                            },
                        },
                    )
                )
                continue
            raise ValueError(f"unknown coordinator worker command: {command!r}")
    except Exception as error:  # noqa: BLE001
        with contextlib.suppress(OSError, ValueError):
            conn.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        del weight_view
        del grad_view
        if weight_shm is not None:
            with contextlib.suppress(OSError, ValueError):
                weight_shm.close()
        if grad_shm is not None:
            with contextlib.suppress(OSError, ValueError):
                grad_shm.close()
        with contextlib.suppress(OSError, ValueError):
            conn.send(("closed", None))
        with contextlib.suppress(OSError, ValueError):
            conn.close()


class MultiProcessMicrobatchCoordinator:
    """Process-isolated 2-device microbatch coordinator using POSIX shared memory."""

    def __init__(
        self,
        *,
        config: RunConfig,
        worker_device_index: int = 0,
        worker_env_overrides: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.worker_device_index = worker_device_index
        self.worker_env_overrides = dict(worker_env_overrides or {})
        (
            self._weight_specs,
            self._grad_specs,
            self._total_weight_floats,
            self._total_grad_floats,
        ) = _build_shm_specs(config)

        self.weight_shm = SharedMemory(create=True, size=self._total_weight_floats * 4)
        try:
            self.grad_shm = SharedMemory(create=True, size=self._total_grad_floats * 4)
        except BaseException:
            self.weight_shm.close()
            self.weight_shm.unlink()
            raise

        self._weight_view: np.ndarray | None = np.ndarray(
            (self._total_weight_floats,),
            dtype=np.float32,
            buffer=self.weight_shm.buf,
        )
        self._grad_view: np.ndarray | None = np.ndarray(
            (self._total_grad_floats,),
            dtype=np.float32,
            buffer=self.grad_shm.buf,
        )

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        self._conn: Connection = parent_conn
        self.proc = ctx.Process(
            target=_coordinator_worker_loop,
            args=(
                config.model_dump(mode="json"),
                worker_device_index,
                self.worker_env_overrides,
                self.weight_shm.name,
                self._total_weight_floats,
                self.grad_shm.name,
                self._total_grad_floats,
                self._weight_specs,
                self._grad_specs,
                child_conn,
            ),
            daemon=True,
        )
        self._closed = False
        self._weights_packed = False
        self._weights_need_worker_unpack = True
        self._preload_pending = False
        try:
            self.proc.start()
            child_conn.close()
            status, payload = self._conn.recv()
            if status != "ready":
                raise RuntimeError(f"microbatch worker failed to start: {payload}")
        except BaseException:
            with contextlib.suppress(OSError, ValueError):
                child_conn.close()
            self.close()
            raise

    def _wait_preload(self) -> None:
        if not self._preload_pending:
            return
        status, payload = self._conn.recv()
        self._preload_pending = False
        if status != "preloaded":
            raise RuntimeError(f"worker weight preload failed: {payload}")
        self._weights_need_worker_unpack = False

    def pack_weights_to_shm(
        self, model: DenseLM, *, async_preload: bool = False
    ) -> None:
        """Copy primary model parameters into shared memory."""
        if self._closed or self._weight_view is None:
            raise RuntimeError("MultiProcessMicrobatchCoordinator is closed")
        self._wait_preload()
        named_params = dict(model.named_parameters())
        with torch.no_grad():
            for name, _shape, offset, numel in self._weight_specs:
                param = named_params[name]
                self._weight_view[offset : offset + numel] = (
                    param.detach().float().cpu().contiguous().numpy().reshape(-1)
                )
        self._weights_packed = True
        if async_preload:
            self._conn.send(("preload_weights",))
            self._preload_pending = True
            self._weights_need_worker_unpack = False
        else:
            self._weights_need_worker_unpack = True

    def train_update(
        self,
        engine: PyTorchEngine,
        microbatches: list[Microbatch],
        update_index: int,
        valid_targets: int,
    ) -> UpdateResult:
        """Execute a microbatch-parallel update across primary and worker processes."""
        if self._closed or self._grad_view is None:
            raise RuntimeError("MultiProcessMicrobatchCoordinator is closed")
        config, device, model, optimizer, monitor = engine._ready()
        engine._last_portability_row_gradient_evidence = None
        chunk_counts = [
            int(np.count_nonzero(batch.targets != -100)) for batch in microbatches
        ]
        observed_targets = sum(chunk_counts)
        if observed_targets <= 0:
            raise ValueError("update needs at least one supervised target")
        if valid_targets != observed_targets:
            raise ValueError(
                "update valid-target count differs from the supplied microbatches"
            )

        active_pairs = [
            (batch, chunk_valid)
            for batch, chunk_valid in zip(microbatches, chunk_counts, strict=True)
            if chunk_valid
        ]
        allocation_enabled = config.dataset.allocation_manifest_path is not None
        portability_enabled = engine.portability_run is not None
        if (
            len(active_pairs) < 2
            or allocation_enabled
            or portability_enabled
            or engine.scaler is not None
            or engine.offload is not None
        ):
            self._wait_preload()
            result = engine.train_update(microbatches, update_index, valid_targets)
            self.pack_weights_to_shm(model, async_preload=True)
            return result

        if self._preload_pending:
            self._wait_preload()
            need_unpack = False
        elif not self._weights_packed:
            self.pack_weights_to_shm(model, async_preload=False)
            need_unpack = True
        else:
            need_unpack = self._weights_need_worker_unpack
        self._weights_need_worker_unpack = False

        primary_pairs = active_pairs[0::2]
        worker_pairs = active_pairs[1::2]

        lr = learning_rate_for_step(
            update_index,
            config.training.max_steps,
            config.optimizer.warmup_steps,
            config.optimizer.peak,
            config.optimizer.floor,
            decay_steps=(
                config.optimizer.decay_steps
                if isinstance(config.optimizer, AdamWConfig)
                else None
            ),
        )
        synchronize(device)
        started = time.perf_counter()
        monitor.begin_update()
        optimizer.zero_grad(set_to_none=True)

        self._conn.send(("run_step", worker_pairs, valid_targets, need_unpack))

        language_sum = torch.zeros((), device=device)
        aux_sum = torch.zeros((), device=device)
        diagnostic_sums: dict[str, torch.Tensor] = {}
        losses_finite = torch.ones((), device=device, dtype=torch.bool)
        recomputed = 0.0
        executed_microbatches = 0
        pilot_progress = current_pilot_progress()
        trainable_inputs = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]

        for batch, chunk_valid in primary_pairs:
            executed_microbatches += 1
            x = torch.from_numpy(batch.inputs).to(device)
            y = torch.from_numpy(batch.targets).to(device)
            addresses = (
                None
                if batch.byte_addresses is None
                else torch.from_numpy(batch.byte_addresses).to(device)
            )
            valid_target_mask = y != -100
            with precision_context(config, device):
                logits, auxiliary = model.forward_with_aux(
                    x,
                    byte_addresses=addresses,
                    semantic_queries=None,
                    memory_mask=None,
                    valid_target_mask=valid_target_mask,
                    activation_checkpointing=config.runtime.memory.activation_checkpointing.enabled,
                    diagnostics=config.logging.architecture_diagnostics,
                    return_hidden=config.runtime.compile_output_head_loss,
                )
            diagnostics = architecture_metrics(model)
            monitor.sample("forward")
            if config.runtime.compile_output_head_loss:
                ce_sum = engine._compiled_output_head_loss(
                    logits, y, model.output.weight
                )
            else:
                ce_sum = functional.cross_entropy(
                    logits.float().flatten(0, 1),
                    y.flatten(),
                    ignore_index=-100,
                    reduction="sum",
                )
            auxiliary = auxiliary.float()
            losses_finite = (
                losses_finite & torch.isfinite(ce_sum) & torch.isfinite(auxiliary)
            )
            backward_loss = (ce_sum + auxiliary * chunk_valid) / valid_targets
            torch.autograd.backward(backward_loss, inputs=trainable_inputs)
            monitor.sample("backward")
            recomputed += model.recomputed_block_call_ratio
            language_sum = language_sum + ce_sum.detach()
            aux_sum = aux_sum + auxiliary.detach() * chunk_valid
            for name, value in diagnostics.items():
                diagnostic_sums[name] = (
                    diagnostic_sums.get(name, value.new_zeros(())) + value * chunk_valid
                )

        status, worker_payload = self._conn.recv()
        if status != "step_done":
            raise RuntimeError(f"microbatch worker step failed: {worker_payload}")

        language_sum = language_sum + torch.from_numpy(
            worker_payload["language_sum"]
        ).to(device=device, dtype=language_sum.dtype)
        aux_sum = aux_sum + torch.from_numpy(worker_payload["aux_sum"]).to(
            device=device, dtype=aux_sum.dtype
        )
        losses_finite = losses_finite & torch.tensor(
            worker_payload["losses_finite"], device=device, dtype=torch.bool
        )
        recomputed += float(worker_payload["recomputed"])
        executed_microbatches += int(worker_payload["executed_microbatches"])
        for name, arr in worker_payload["diagnostic_sums"].items():
            val_dev = torch.from_numpy(arr).to(device=device)
            diagnostic_sums[name] = (
                diagnostic_sums.get(name, val_dev.new_zeros(())) + val_dev
            )

        named_parameters = dict(model.named_parameters())
        for name, shape, offset, numel in self._grad_specs:
            param = named_parameters[name]
            grad_arr = self._grad_view[offset : offset + numel].reshape(shape)
            worker_grad = torch.from_numpy(grad_arr).to(
                device=device, dtype=param.dtype
            )
            if param.grad is None:
                param.grad = worker_grad.clone()
            else:
                param.grad.add_(worker_grad)

        if pilot_progress is not None:
            pilot_progress.emit(
                "progress",
                "optimizer_update",
                current_step=update_index,
                counter="items",
                value=executed_microbatches,
                total=len(microbatches),
                subject=f"update_{update_index}_backward_dispatches",
            )

        gradient_names = tuple(
            name
            for name, parameter in named_parameters.items()
            if parameter.grad is not None
        )
        if config.training.trainable_parameters is not None:
            allowed_names = set(config.training.trainable_parameters)
            unexpected_gradients = set(gradient_names) - allowed_names
            if unexpected_gradients:
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    "gradients escaped the explicit trainable inventory: "
                    f"{sorted(unexpected_gradients)}"
                )
        engine._last_gradient_parameter_names = gradient_names

        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.grad_clip_norm,
            error_if_nonfinite=False,
            foreach=False,
        )
        gradients_finite = torch.isfinite(norm)
        synchronize(device)
        if not bool(losses_finite):
            optimizer.zero_grad(set_to_none=True)
            raise EngineNonFiniteError("nonfinite training loss")
        if not bool(gradients_finite):
            optimizer.zero_grad(set_to_none=True)
            raise EngineNonFiniteError("nonfinite unscaled gradients")
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        engine._replicas_need_sync = True
        monitor.sample("optimizer")
        synchronize(device)
        memory_metrics = monitor.end_update()
        elapsed = time.perf_counter() - started
        engine._last_update_parameter_names = gradient_names

        self.pack_weights_to_shm(model, async_preload=True)

        metrics = {
            "train/loss": float(language_sum) / valid_targets,
            "moe/router_auxiliary_loss": float(aux_sum) / valid_targets,
            "optimizer/learning_rate": lr,
            "optimizer/grad_norm": float(norm),
            "performance/step_seconds": elapsed,
            "performance/tokens_per_second": valid_targets / max(elapsed, 1e-9),
            "batch/micro_batch_size": float(config.training.micro_batch_size),
            "batch/accumulation_steps": float(len(microbatches)),
            "batch/effective_batch_size": float(
                sum(batch.inputs.shape[0] for batch in microbatches)
            ),
            "batch/effective_tokens_per_update": float(valid_targets),
            "recompute/block_call_ratio": recomputed / executed_microbatches,
            **{
                name: float(value) / valid_targets
                for name, value in diagnostic_sums.items()
            },
            **memory_metrics,
        }
        return UpdateResult(
            "APPLIED",
            float(language_sum),
            valid_targets,
            valid_targets,
            elapsed,
            metrics,
        )

    def close(self) -> None:
        """Terminate the worker process and unlink shared memory segments."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(OSError, ValueError, RuntimeError):
            self._wait_preload()
        if self.proc.is_alive():
            with contextlib.suppress(OSError, ValueError, EOFError):
                self._conn.send(("shutdown",))
                if self._conn.poll(2.0):
                    self._conn.recv()
            self.proc.join(timeout=5.0)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=5.0)
        with contextlib.suppress(OSError, ValueError):
            self._conn.close()
        self._weight_view = None
        self._grad_view = None
        for shm in (self.weight_shm, self.grad_shm):
            with contextlib.suppress(OSError, ValueError):
                shm.close()
            with contextlib.suppress(OSError, ValueError, FileNotFoundError):
                shm.unlink()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
