"""Multi-device microbatch-parallel training: one process per secondary device.

The primary process keeps the model, optimizer and checkpoints. Each secondary
device gets a replica in its own process, started with only that device
visible (:mod:`klab.device_visibility`): on a host with several Arc cards a
process that sees every card also holds its device memory in host RAM. Weights
and gradients cross between processes through shared host tensors, and
microbatches and results through a pipe. Gradients are summed on the primary
in replica order, so an update is the same as on one device.
"""

from __future__ import annotations

import multiprocessing.connection
import os
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as torch_multiprocessing
from torch.nn import functional

from klab.config.models import RunConfig
from klab.device_visibility import child_environment
from klab.engines.base import Microbatch
from klab.engines.output_head import chunked_output_head_cross_entropy
from klab.model.inspection import architecture_metrics
from klab.model.transformer import DenseLM
from klab.runtime import precision_context, synchronize

# Seconds between checks that a replica process is still alive while waiting.
_POLL_SECONDS = 1.0
# Seconds a replica process may take to build its model and report ready.
_READY_SECONDS = 1800.0


@dataclass
class SecondaryReplica:
    """A model replica bound to one secondary device, inside its own process."""

    device: torch.device
    model: DenseLM
    compiled_output_head_loss: Any = None
    trainable_params: tuple[torch.nn.Parameter, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.trainable_params:
            self.trainable_params = tuple(
                p for p in self.model.parameters() if p.requires_grad
            )


def microbatch_position_mask(
    batch: Microbatch, device: torch.device, chunk_valid: int
) -> tuple[torch.Tensor | None, int]:
    """Validate the common serial/replica right-padding contract."""
    mask = batch.position_mask
    if mask is None:
        return None, chunk_valid
    if mask.dtype != np.bool_ or mask.shape != batch.inputs.shape:
        raise ValueError("position mask must be boolean and match inputs")
    if (
        np.any((batch.targets != -100) & ~mask)
        or np.any(mask[:, 1:] & ~mask[:, :-1])
        or np.any(~mask[:, 0])
    ):
        raise ValueError(
            "position mask must be nonempty right-padded prefixes covering targets"
        )
    return torch.from_numpy(mask).to(device), int(mask.sum())


def diagnostic_is_count(name: str, batches: list[Microbatch]) -> bool:
    return (
        any(batch.position_mask is not None for batch in batches)
        and name.startswith("engram/")
        and name.rsplit("/", 1)[-1]
        in {"lookup_count", "unique_addresses", "collision_count"}
    )


def accumulate_microbatch_diagnostics(
    sums: dict[str, torch.Tensor],
    weights: dict[str, int],
    diagnostics: dict[str, torch.Tensor],
    batch: Microbatch,
    chunk_valid: int,
    position_count: int,
) -> None:
    """Keep count totals and position-weighted Engram diagnostics replica invariant."""
    for name, value in diagnostics.items():
        count_metric = diagnostic_is_count(name, [batch])
        padded_engram = batch.position_mask is not None and name.startswith("engram/")
        weight = 1 if count_metric else position_count if padded_engram else chunk_valid
        sums[name] = sums.get(name, value.new_zeros(())) + value * weight
        weights[name] = 1 if count_metric else weights.get(name, 0) + weight


def execute_replica_microbatches(
    replica: SecondaryReplica,
    config: RunConfig,
    pairs: list[tuple[Microbatch, int]],
    valid_targets: int,
    output_head_loss_fn: Any,
) -> dict[str, Any]:
    """Run one replica's slice, leaving its gradients on its parameters."""
    started = time.perf_counter()
    device = replica.device
    model = replica.model
    model.zero_grad(set_to_none=True)
    language_sum = torch.zeros((), device=device)
    aux_sum = torch.zeros((), device=device)
    diagnostic_sums: dict[str, torch.Tensor] = {}
    diagnostic_weights: dict[str, int] = {}
    losses_finite = torch.ones((), device=device, dtype=torch.bool)
    recomputed = 0.0
    executed_microbatches = 0
    trainable_inputs = list(replica.trainable_params)
    head_chunk = config.runtime.output_head_chunk_tokens

    for batch, chunk_valid in pairs:
        if not chunk_valid and batch.position_mask is None:
            continue
        executed_microbatches += 1
        x = torch.from_numpy(batch.inputs).to(device)
        y = torch.from_numpy(batch.targets).to(device)
        addresses = (
            None
            if batch.byte_addresses is None
            else torch.from_numpy(batch.byte_addresses).to(device)
        )
        position_mask, position_count = microbatch_position_mask(
            batch, device, chunk_valid
        )
        valid_target_mask = y != -100
        with precision_context(config, device):
            logits, auxiliary = model.forward_with_aux(
                x,
                byte_addresses=addresses,
                valid_target_mask=valid_target_mask,
                activation_checkpointing=config.runtime.memory.activation_checkpointing.enabled,
                diagnostics=config.logging.architecture_diagnostics,
                return_hidden=config.runtime.compile_output_head_loss
                or head_chunk is not None,
                position_mask=position_mask,
            )
        diagnostics = architecture_metrics(model)
        if config.runtime.compile_output_head_loss:
            loss_fn = replica.compiled_output_head_loss or output_head_loss_fn
            ce_sum = loss_fn(logits, y, model.output.weight)
        elif head_chunk is not None:
            ce_sum = chunked_output_head_cross_entropy(
                logits,
                y,
                model.output.weight,
                head_chunk,
                lambda: precision_context(config, device),
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
        recomputed += model.recomputed_block_call_ratio
        language_sum = language_sum + ce_sum.detach()
        aux_sum = aux_sum + auxiliary.detach() * chunk_valid
        accumulate_microbatch_diagnostics(
            diagnostic_sums,
            diagnostic_weights,
            diagnostics,
            batch,
            chunk_valid,
            position_count,
        )

    synchronize(device)
    computed = time.perf_counter()
    return {
        "compute_seconds": computed - started,
        "finished_at": computed,
        "language_sum": language_sum,
        "aux_sum": aux_sum,
        "losses_finite": losses_finite,
        "recomputed": recomputed,
        "executed_microbatches": executed_microbatches,
        "diagnostic_sums": diagnostic_sums,
        "diagnostic_weights": diagnostic_weights,
    }


def build_replica(
    config: RunConfig, device: torch.device, state: dict[str, torch.Tensor]
) -> SecondaryReplica:
    """Build the replica the primary's model would be on ``device``."""
    from klab.engines.pytorch import (
        apply_recomputation_strategy,
        output_projection_cross_entropy,
    )
    from klab.model.rope import RoPE
    from klab.training.trainable import apply_trainable_parameter_filter

    model = DenseLM(config.model, config.attention).to(device)
    model.load_state_dict(state)
    apply_trainable_parameter_filter(model, config)
    apply_recomputation_strategy(model, config)
    compiled_loss = None
    runtime = config.runtime
    if runtime.compile_output_head_loss:
        compiled_loss = torch.compile(
            output_projection_cross_entropy,
            dynamic=runtime.compile_dynamic_shapes,
            fullgraph=runtime.compile_dynamic_shapes,
        )
    if runtime.compile_transformer_blocks:
        if runtime.compile_dynamic_shapes:
            for module in model.modules():
                if isinstance(module, RoPE):
                    module.reserve(config.model.max_seq_len, device)
        for block in model.blocks:
            block.forward_with_aux = torch.compile(
                block.forward_with_aux,
                dynamic=runtime.compile_dynamic_shapes,
                fullgraph=runtime.compile_dynamic_shapes,
            )
    return SecondaryReplica(
        device=device, model=model, compiled_output_head_loss=compiled_loss
    )


def _report_failure(connection: multiprocessing.connection.Connection) -> None:
    """Send the exception itself, so the primary raises the same kind of error."""
    error = sys.exc_info()[1]
    trace = traceback.format_exc()
    try:
        connection.send(("error", (error, trace)))
    except Exception:  # noqa: BLE001 - an unpicklable error still reaches the primary
        connection.send(("error", (RuntimeError(str(error)), trace)))


def _host(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _host(item) for key, item in value.items()}
    return value


def _replica_main(
    connection: multiprocessing.connection.Connection,
    config: RunConfig,
    backend: str,
    device_index: int,
    state: dict[str, torch.Tensor],
    weights: list[torch.Tensor],
    gradients: list[torch.Tensor],
    has_gradient: torch.Tensor,
) -> None:
    """A secondary device's process: build the replica, then run slices on request."""
    try:
        from klab.engines.pytorch import output_projection_cross_entropy
        from klab.runtime import torch_device_for

        device = torch_device_for(backend, device_index)
        replica = build_replica(config, device, state)
        del state
        shapes = [tuple(parameter.shape) for parameter in replica.trainable_params]
        if shapes != [tuple(weight.shape) for weight in weights]:
            raise ValueError("replica trainable parameters differ from the primary's")
        version = -1
        connection.send(("ready", None))
    except Exception:  # noqa: BLE001 - the primary reports the replica's error
        _report_failure(connection)
        return
    while True:
        message = connection.recv()
        if message[0] == "stop":
            return
        _, weights_version, pairs, valid_targets = message
        try:
            if weights_version != version:
                with torch.no_grad():
                    for parameter, weight in zip(
                        replica.trainable_params, weights, strict=True
                    ):
                        parameter.copy_(weight)
                version = weights_version
            result = execute_replica_microbatches(
                replica, config, pairs, valid_targets, output_projection_cross_entropy
            )
            with torch.no_grad():
                for index, parameter in enumerate(replica.trainable_params):
                    has_gradient[index] = parameter.grad is not None
                    if parameter.grad is not None:
                        gradients[index].copy_(parameter.grad)
            replica.model.zero_grad(set_to_none=True)
            connection.send(("done", _host(result)))
        except Exception:  # noqa: BLE001 - the primary reports the replica's error
            _report_failure(connection)


_ENVIRONMENT_LOCK = threading.Lock()


@contextmanager
def _environment(environment: dict[str, str]) -> Iterator[None]:
    """Start a child with ``environment``; a spawned child reads it at start."""
    with _ENVIRONMENT_LOCK:
        saved = dict(os.environ)
        os.environ.clear()
        os.environ.update(environment)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(saved)


@dataclass
class _ReplicaProcess:
    device_index: int
    process: Any
    connection: multiprocessing.connection.Connection
    gradients: list[torch.Tensor]
    has_gradient: torch.Tensor

    def receive(self, timeout: float | None = None) -> Any:
        waited = 0.0
        while not self.connection.poll(_POLL_SECONDS):
            waited += _POLL_SECONDS
            if not self.process.is_alive():
                raise RuntimeError(
                    f"replica process for device {self.device_index} exited "
                    f"with status {self.process.exitcode}"
                )
            if timeout is not None and waited >= timeout:
                raise TimeoutError(
                    f"replica process for device {self.device_index} did not "
                    f"answer within {timeout:.0f} seconds"
                )
        kind, payload = self.connection.recv()
        if kind == "error":
            error, trace = payload
            error.add_note(f"in the replica process for device {self.device_index}:")
            error.add_note(trace)
            raise error
        return payload


class ReplicaPool:
    """The secondary devices of one run, each in its own process."""

    def __init__(
        self,
        config: RunConfig,
        backend: str,
        model: DenseLM,
        trainable_params: list[torch.nn.Parameter],
    ) -> None:
        context = torch_multiprocessing.get_context("spawn")
        self.trainable_params = trainable_params
        self.weights = [
            torch.empty(p.shape, dtype=p.dtype).share_memory_()
            for p in trainable_params
        ]
        # The replicas' starting state, shared once and released when all are ready.
        state = {
            name: tensor.detach().cpu().share_memory_()
            for name, tensor in model.state_dict().items()
        }
        self.replicas: list[_ReplicaProcess] = []
        self.version = 0
        self.stale = True
        try:
            for device_index in config.runtime.secondary_device_indices:
                gradients = [torch.empty_like(w).share_memory_() for w in self.weights]
                has_gradient = torch.zeros(
                    len(self.weights), dtype=torch.bool
                ).share_memory_()
                parent, child = context.Pipe()
                process = context.Process(
                    target=_replica_main,
                    args=(
                        child,
                        config,
                        backend,
                        device_index,
                        state,
                        self.weights,
                        gradients,
                        has_gradient,
                    ),
                    name=f"klab-replica-{device_index}",
                    daemon=True,
                )
                with _environment(child_environment(backend, device_index)):
                    process.start()
                child.close()
                self.replicas.append(
                    _ReplicaProcess(
                        device_index, process, parent, gradients, has_gradient
                    )
                )
            for replica in self.replicas:
                replica.receive(timeout=_READY_SECONDS)
        except Exception:
            self.close()
            raise

    def __len__(self) -> int:
        return len(self.replicas)

    def invalidate(self) -> None:
        """The primary's weights changed; publish them before the next slice."""
        self.stale = True

    def dispatch(
        self,
        *,
        active_pairs: list[tuple[Microbatch, int]],
        valid_targets: int,
        primary_fn: Any,
        primary_device: torch.device,
    ) -> tuple[Any, list[dict[str, Any]], dict[str, float]]:
        """Run each replica's slice in its process while the caller runs the primary's.

        Microbatches are dealt round-robin: the primary takes every
        ``1 + len(self)``-th one starting at 0, replica ``i`` those starting at
        ``i + 1``. Replica gradients are added to the primary's in replica
        order. The timings say how long publishing weights took, how long each
        side computed, how long finished devices waited for the slowest, and
        how long merging gradients took.
        """
        world_size = 1 + len(self.replicas)
        primary_pairs = active_pairs[0::world_size]
        tasks = [
            (replica, active_pairs[index + 1 :: world_size])
            for index, replica in enumerate(self.replicas)
            if active_pairs[index + 1 :: world_size]
        ]
        started = time.perf_counter()
        if self.stale:
            with torch.no_grad():
                for weight, parameter in zip(
                    self.weights, self.trainable_params, strict=True
                ):
                    weight.copy_(parameter)
            self.version += 1
            self.stale = False
        published = time.perf_counter()
        for replica, pairs in tasks:
            replica.connection.send(("run", self.version, pairs, valid_targets))
        failure: BaseException | None = None
        try:
            primary_out = primary_fn(primary_pairs)
            synchronize(primary_device)
        finally:
            primary_finished = time.perf_counter()
            replica_outs = []
            for replica, _ in tasks:
                try:
                    replica_outs.append(replica.receive())
                except Exception as error:  # noqa: BLE001 - raised after every replica answers
                    failure = failure or error
        if failure is not None:
            raise failure
        collected: list[torch.Tensor | None] = [None] * len(self.trainable_params)
        with torch.no_grad():
            for replica, _ in tasks:
                for index, gradient in enumerate(replica.gradients):
                    if not bool(replica.has_gradient[index]):
                        continue
                    moved = gradient.to(primary_device)
                    held = collected[index]
                    collected[index] = moved if held is None else held.add_(moved)
            for parameter, gradient in zip(
                self.trainable_params, collected, strict=True
            ):
                if gradient is None:
                    continue
                if parameter.grad is None:
                    parameter.grad = gradient
                else:
                    parameter.grad.add_(gradient)
        synchronize(primary_device)
        finished = time.perf_counter()
        all_computed = max(
            [primary_finished, *(out["finished_at"] for out in replica_outs)]
        )
        timings = {
            "performance/weight_publish_seconds": published - started,
            "performance/primary_compute_seconds": primary_finished - published,
            "performance/replica_compute_seconds": max(
                (out["compute_seconds"] for out in replica_outs), default=0.0
            ),
            # Time finished devices spent waiting for the slowest one, summed.
            "performance/device_wait_seconds": sum(
                max(0.0, all_computed - out["finished_at"]) for out in replica_outs
            )
            + (all_computed - primary_finished),
            "performance/gradient_merge_seconds": finished - all_computed,
        }
        return primary_out, replica_outs, timings

    def close(self) -> None:
        for replica in self.replicas:
            try:
                replica.connection.send(("stop", None))
            except BrokenPipeError, OSError:
                pass
        for replica in self.replicas:
            replica.process.join(timeout=30)
            if replica.process.is_alive():
                replica.process.terminate()
                replica.process.join(timeout=10)
            replica.connection.close()
        self.replicas = []
