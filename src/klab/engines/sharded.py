"""Sharded data parallelism over real collectives: ZeRO-1, ZeRO-2 and FSDP.

The devices of ``runtime.device_index`` and ``runtime.secondary_device_indices``
form one :mod:`torch.distributed` group, one process per device (the primary's
is the trainer's own, rank 0; :mod:`klab.engines.multi_device` starts the rest).
Parameters are wrapped with FSDP2 (``fully_shard``), one group per transformer
block plus one for everything else; the three modes are settings of it:

``zero1``
    Optimizer state is sharded. Parameters stay gathered through the update and
    gradients accumulate whole, so one reduce-scatter per update.
``zero2``
    Gradients are also sharded: each microbatch's are reduce-scattered into
    the shard right after its backward.
``fsdp``
    Parameters are also sharded: each block is gathered for its forward and
    again for its backward, then released.

Every mode sums gradients (divide factor 1 with a plain SUM reduction): the loss is
already normalized by the update's target count, so the result equals one
device's, up to summation order. Gradient clipping and the finite checks see the
whole gradient. Checkpoints hold the consolidated, unsharded tensors in the
layout of a single-device run, so a run resumes on any device count and mode.

All ranks must run every collective in the same order, so the primary sends each
worker a command and runs the same operation itself (:meth:`ShardGroup.run`).
"""

from __future__ import annotations

import math
import socket
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor

from klab.config.models import RunConfig
from klab.engines.base import (
    EngineNonFiniteError,
    EvaluationResult,
    Microbatch,
)
from klab.engines.multi_device import (
    ReplicaProcess,
    SecondaryReplica,
    build_replica,
    execute_replica_microbatches,
    merge_replica_outputs,
    report_failure,
    spawn_replica_process,
    stop_replica_processes,
    to_host,
)
from klab.model.transformer import DenseLM
from klab.runtime import release_cached_memory, synchronize, torch_device_for
from klab.training.optimizer import build_optimizer, set_optimizer_learning_rate

PROCESS_GROUP_BACKEND = {
    "cpu": "gloo",
    "xpu": "xccl",
    "cuda": "nccl",
    "rocm": "nccl",
}
# A rank that dies or stalls leaves the others inside a collective; this bounds it.
_GROUP_TIMEOUT = timedelta(minutes=20)
# Seconds a worker may take to build its model and report ready.
_READY_SECONDS = 1800.0


@dataclass(frozen=True)
class TensorShape:
    """Stands in for a tensor a rank only receives a shard of."""

    shape: tuple[int, ...]
    dtype: torch.dtype


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _route_through_call(module: torch.nn.Module) -> None:
    """Make ``module(...)`` the way ``forward_with_aux`` runs.

    FSDP gathers and releases parameters in hooks on ``forward``; the model and
    its blocks are run through ``forward_with_aux``, so the hooks would never
    fire. Inside a sharded run the module's forward *is* ``forward_with_aux``.
    """
    module.forward = module.forward_with_aux
    module.forward_with_aux = module.__call__


def shard_model(model: DenseLM, mode: str) -> DenseLM:
    """Wrap ``model`` in place for ``mode`` (``zero1``, ``zero2`` or ``fsdp``)."""
    gather_per_use = mode == "fsdp"
    for block in model.blocks:
        fully_shard(block, reshard_after_forward=gather_per_use)
        _route_through_call(block)
    # Token memory (Engram tables) is gathered and released like a block; the
    # embedding, norm and output head, which the loss reads after the forward,
    # are gathered once and held through backward.
    for memory in (model.memory, *model.extra_memories.values()):
        if memory is not None:
            fully_shard(memory, reshard_after_forward=gather_per_use)
    fully_shard(model, reshard_after_forward=False)
    _route_through_call(model)
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)
            module.set_force_sum_reduction_for_comms(True)
            module.set_reshard_after_backward(gather_per_use)
    return model


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


_LOST_SHARD = 1
_NON_FINITE = 2
_INTEGER_OF_WIDTH = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _same_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Bit-for-bit equality, so a NaN equals itself."""
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    width = _INTEGER_OF_WIDTH[left.element_size()]
    return torch.equal(left.contiguous().view(width), right.contiguous().view(width))


class ShardedRank:
    """One rank's sharded model and optimizer, and the collectives that drive them."""

    def __init__(
        self,
        config: RunConfig,
        backend: str,
        rank: int,
        world_size: int,
        init_method: str,
        model: DenseLM,
        device: torch.device,
        compiled_output_head_loss: Any = None,
    ) -> None:
        self.config = config
        self.mode = config.runtime.parallel.mode
        self.rank = rank
        self.world_size = world_size
        self.device = device
        if device.type != "cpu":
            getattr(torch, device.type).set_device(device)
        dist.init_process_group(
            PROCESS_GROUP_BACKEND[backend],
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=_GROUP_TIMEOUT,
        )
        from klab.engines.pytorch import state_dict_aliases

        # Tied weights and other aliases are saved once, as in a single device.
        self.aliases = state_dict_aliases(model)
        self.model = shard_model(model, self.mode)
        self.optimizer = build_optimizer(self.model, config)
        self.replica = SecondaryReplica(
            device,
            self.model,
            compiled_output_head_loss=compiled_output_head_loss,
            sharded=True,
        )

    def release(self) -> None:
        """Free parameters a mode keeps gathered between collectives."""
        for module in self.model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

    def update(
        self,
        pairs: list[tuple[Microbatch, int]],
        valid_targets: int,
        lr: float,
        pad_to: int,
        pad_template: Microbatch,
    ) -> dict[str, Any]:
        """Run this rank's microbatches, then clip and apply the update."""
        from klab.engines.pytorch import output_projection_cross_entropy

        config = self.config
        model = self.model
        self.optimizer.zero_grad(set_to_none=True)

        def before_backward(index: int, count: int) -> None:
            # ZeRO-1 keeps whole gradients and reduces them once, on the last.
            if self.mode == "zero1":
                model.set_requires_gradient_sync(index == count - 1)

        try:
            out = execute_replica_microbatches(
                self.replica,
                config,
                pairs,
                valid_targets,
                output_projection_cross_entropy,
                pad_to=pad_to,
                pad_template=pad_template,
                before_backward=before_backward,
            )
            # Gradients live on the sharded parameters, which a mode that keeps
            # parameters gathered through backward only shows again once released.
            self.release()
            finite = out["losses_finite"].to(torch.int32).to(self.device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.grad_clip_norm,
                error_if_nonfinite=False,
                foreach=False,
            )
            norm_value = float(_full(norm))
            out["gradient_names"] = tuple(
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            )
            synchronize(self.device)
            if not bool(finite):
                raise EngineNonFiniteError("nonfinite training loss")
            if not math.isfinite(norm_value):
                raise EngineNonFiniteError("nonfinite unscaled gradients")
            set_optimizer_learning_rate(self.optimizer, lr)
            self.optimizer.step()
            out["grad_norm"] = norm_value
            return out
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        finally:
            model.set_requires_gradient_sync(True)
            self.release()
            synchronize(self.device)

    def evaluate(
        self,
        batches: list[Microbatch],
        positions: Any = None,
        observer: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    ) -> EvaluationResult:
        from klab.engines.pytorch import evaluate_batches

        try:
            return evaluate_batches(
                self.config, self.device, self.model, batches, positions, observer
            )
        finally:
            self.release()

    def gather_weights(self) -> dict[str, torch.Tensor]:
        """Whole tensors by name, on rank 0 only; every rank must call."""
        gathered: dict[str, torch.Tensor] = {}
        for name, tensor in self.model.state_dict().items():
            if name in self.aliases:
                continue
            whole = _full(tensor)
            if self.rank == 0:
                gathered[name] = whole.detach().to("cpu", copy=True).contiguous()
        self.release()
        return gathered

    def gather_optimizer_state(self) -> dict[str, Any] | None:
        """The optimizer's ``state_dict`` with whole tensors, on rank 0 only."""
        state = self.optimizer.state_dict()
        whole: dict[int, dict[str, Any]] = {}
        for index in sorted(state["state"]):
            slots = state["state"][index]
            whole[index] = {}
            for key in sorted(slots):
                value = slots[key]
                if isinstance(value, DTensor):
                    value = value.full_tensor()
                    value = (
                        value.detach().to("cpu", copy=True) if self.rank == 0 else None
                    )
                whole[index][key] = value
        if self.rank != 0:
            return None
        return {"state": whole, "param_groups": state["param_groups"]}

    def _check_restored(
        self, key: str, index: int, shard: DTensor, source: torch.Tensor | None
    ) -> None:
        """Fail on every rank if a restored shard differs from the checkpoint.

        ``source`` is the checkpoint's tensor on the host (rank 0 only). The shards are
        gathered back, rank 0 compares their bits with it and broadcasts the verdict,
        so every rank raises together. Bits, not values: a NaN already in the
        checkpoint is not mistaken for a lost shard, and is reported as what it is. A
        shard that never arrived would otherwise train on stale memory.
        """
        gathered = shard.full_tensor()
        verdict = torch.zeros(1, dtype=torch.int64, device=self.device)
        if self.rank == 0:
            assert source is not None
            if source.is_floating_point() and not bool(torch.isfinite(source).all()):
                verdict[0] = _NON_FINITE
            elif not _same_bits(gathered.cpu(), source):
                verdict[0] = _LOST_SHARD
        del gathered
        dist.broadcast(verdict, src=0)
        if int(verdict.item()) == _NON_FINITE:
            raise ValueError(
                f"checkpoint optimizer state {key!r} of parameter {index} is "
                "non-finite; it cannot be restored"
            )
        if int(verdict.item()) == _LOST_SHARD:
            raise ValueError(
                f"restored optimizer state {key!r} of parameter {index} differs "
                "from the checkpoint: a shard did not arrive intact"
            )

    def load_optimizer_state(self, state: dict[str, Any]) -> None:
        """Take this rank's shards from ``state`` (whole tensors on rank 0 only).

        Other ranks pass :class:`TensorShape` where rank 0 has a tensor.
        """
        from klab.training.checkpoints import restore_optimizer_state

        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]
        sharded: dict[int, dict[str, Any]] = {}
        for index in sorted(state["state"]):
            parameter = parameters[index]
            slots = {}
            for key in sorted(state["state"][index]):
                value = state["state"][index][key]
                if (
                    isinstance(value, TensorShape | torch.Tensor)
                    and tuple(value.shape) == tuple(parameter.shape)
                    and isinstance(parameter, DTensor)
                ):
                    # Rank 0 holds the tensor; the others only need a placeholder
                    # of the right shape, made on the device so the host never holds
                    # a whole copy. One unsharded tensor lives at a time.
                    source = value if isinstance(value, torch.Tensor) else None
                    whole = (
                        torch.empty(value.shape, dtype=value.dtype, device=self.device)
                        if isinstance(value, TensorShape)
                        else value.to(self.device)
                    )
                    value = distribute_tensor(
                        whole,
                        parameter.device_mesh,
                        parameter.placements,
                        src_data_rank=0,
                    )
                    # The scatter may still be running on the communication stream;
                    # freeing or releasing memory under it would corrupt a shard.
                    synchronize(self.device)
                    # Free the placeholder before the check gathers a whole tensor
                    # back, so each rank still peaks at one.
                    del whole
                    self._check_restored(key, index, value, source)
                    # Temporaries of many sizes stay reserved; return them each time so
                    # the peak is one tensor, not the whole optimizer state.
                    release_cached_memory(self.device)
                elif isinstance(value, TensorShape):
                    value = torch.empty(value.shape, dtype=value.dtype)
                slots[key] = value
            sharded[index] = slots
        restore_optimizer_state(
            self.optimizer,
            {"state": sharded, "param_groups": state["param_groups"]},
        )
        release_cached_memory(self.device)

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


def _shard_worker_main(
    connection: Any,
    config: RunConfig,
    backend: str,
    device_index: int,
    rank: int,
    world_size: int,
    init_method: str,
    state: dict[str, torch.Tensor],
) -> None:
    """A non-primary rank: build its shard of the model, then obey commands."""
    try:
        device = torch_device_for(backend, device_index)
        replica = build_replica(config, device, state)
        del state
        connection.send(("built", None))
        worker = ShardedRank(
            config,
            backend,
            rank,
            world_size,
            init_method,
            replica.model,
            device,
            replica.compiled_output_head_loss,
        )
        connection.send(("ready", None))
    except Exception:  # noqa: BLE001 - the primary reports the rank's error
        report_failure(connection)
        return
    while True:
        message = connection.recv()
        command = message[0]
        if command == "stop":
            worker.close()
            return
        try:
            result = None
            if command == "update":
                result = to_host(worker.update(*message[1:]))
            elif command == "evaluate":
                worker.evaluate(message[1])
            elif command == "weights":
                worker.gather_weights()
            elif command == "optimizer":
                worker.gather_optimizer_state()
            elif command == "restore":
                worker.load_optimizer_state(message[1])
            else:
                raise ValueError(f"unknown shard command {command!r}")
            connection.send(("done", result))
        except Exception:  # noqa: BLE001 - the primary reports the rank's error
            report_failure(connection)


def _for_worker(value: Any) -> Any:
    """What a worker is sent of one optimizer-state entry.

    Sending a tensor over the pipe moves its storage to shared memory, so a
    scalar such as Adam's step count is sent as a copy (the worker would
    otherwise increment the primary's), and a large tensor not at all: the
    worker receives its shard through the collective.
    """
    if not isinstance(value, torch.Tensor):
        return value
    if value.ndim:
        return TensorShape(tuple(value.shape), value.dtype)
    return value.clone()


class ShardGroup:
    """The primary's handle on a sharded run: rank 0 plus one process per other device."""

    def __init__(
        self,
        config: RunConfig,
        backend: str,
        model: DenseLM,
        device: torch.device,
        compiled_output_head_loss: Any = None,
    ) -> None:
        indices = config.runtime.secondary_device_indices
        self.world_size = 1 + len(indices)
        init_method = f"tcp://127.0.0.1:{_free_port()}"
        state = {
            name: tensor.detach().cpu().share_memory_()
            for name, tensor in model.state_dict().items()
        }
        self.replicas: list[ReplicaProcess] = []
        self.rank: ShardedRank | None = None
        try:
            for rank, device_index in enumerate(indices, start=1):
                self.replicas.append(
                    spawn_replica_process(
                        _shard_worker_main,
                        (
                            config,
                            backend,
                            device_index,
                            rank,
                            self.world_size,
                            init_method,
                            state,
                        ),
                        backend,
                        device_index,
                    )
                )
            # Joining the group blocks until every rank has; only join once
            # each worker is known to be alive and built.
            for replica in self.replicas:
                replica.receive(timeout=_READY_SECONDS)
            self.rank = ShardedRank(
                config,
                backend,
                0,
                self.world_size,
                init_method,
                model,
                device,
                compiled_output_head_loss,
            )
            for replica in self.replicas:
                replica.receive(timeout=_READY_SECONDS)
        except Exception:
            self.close()
            raise
        self.config = config
        self.device = device

    @property
    def model(self) -> DenseLM:
        assert self.rank is not None
        return self.rank.model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        assert self.rank is not None
        return self.rank.optimizer

    def run(
        self,
        command: str,
        local: Callable[[], Any],
        arguments: list[tuple[Any, ...]] | None = None,
    ) -> tuple[Any, list[Any]]:
        """Run ``local`` here and ``command`` on every worker, as one collective step.

        ``arguments`` has one tuple per worker; the default is no arguments.
        Raises the first rank's error once every rank has answered.
        """
        for index, replica in enumerate(self.replicas):
            replica.connection.send(
                (command, *(arguments[index] if arguments is not None else ()))
            )
        failure: BaseException | None = None
        local_result = None
        try:
            local_result = local()
        except Exception as error:  # noqa: BLE001 - raised after every rank answers
            failure = error
        replies = []
        for replica in self.replicas:
            try:
                replies.append(replica.receive())
            except Exception as error:  # noqa: BLE001 - raised after every rank answers
                failure = failure or error
        if failure is not None:
            raise failure
        return local_result, replies

    def update(
        self,
        active_pairs: list[tuple[Microbatch, int]],
        valid_targets: int,
        lr: float,
        microbatches: list[Microbatch],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """One optimizer update over ``active_pairs``, dealt round-robin to the ranks.

        Every rank runs ``ceil(len / world)`` microbatches, the short ones
        padded with masked copies, since each forward and backward is a
        collective. Returns the summed slice results (as
        :func:`~klab.engines.multi_device.execute_replica_microbatches` reports
        them, plus ``grad_norm`` and ``gradient_names``) and timings.
        """
        assert self.rank is not None
        world = self.world_size
        pad_to = -(-len(active_pairs) // world)
        template = active_pairs[0][0]
        slices = [active_pairs[index::world] for index in range(world)]
        started = time.perf_counter()
        local, replies = self.run(
            "update",
            lambda: self.rank.update(slices[0], valid_targets, lr, pad_to, template),
            [
                (slices[index + 1], valid_targets, lr, pad_to, template)
                for index in range(len(self.replicas))
            ],
        )
        merged = merge_replica_outputs(local, replies, self.device, microbatches)
        merged["grad_norm"] = local["grad_norm"]
        merged["gradient_names"] = local["gradient_names"]
        return merged, {
            "performance/rank_compute_seconds": max(
                local["compute_seconds"], *(r["compute_seconds"] for r in replies)
            ),
            "performance/update_seconds": time.perf_counter() - started,
        }

    def evaluate(
        self,
        batches: Iterable[Microbatch],
        positions: Any,
        observer: Callable[[torch.Tensor, torch.Tensor], None] | None,
    ) -> EvaluationResult:
        assert self.rank is not None
        held = list(batches)
        result, _ = self.run(
            "evaluate",
            lambda: self.rank.evaluate(held, positions, observer),
            [(held,)] * len(self.replicas),
        )
        return result

    def weights(self) -> dict[str, torch.Tensor]:
        assert self.rank is not None
        gathered, _ = self.run("weights", self.rank.gather_weights)
        return gathered

    def optimizer_state(self) -> dict[str, Any]:
        assert self.rank is not None
        state, _ = self.run("optimizer", self.rank.gather_optimizer_state)
        return state

    def restore_optimizer_state(self, state: dict[str, Any]) -> None:
        assert self.rank is not None
        shapes = {
            "state": {
                index: {key: _for_worker(value) for key, value in slots.items()}
                for index, slots in state["state"].items()
            },
            "param_groups": state["param_groups"],
        }
        self.run(
            "restore",
            lambda: self.rank.load_optimizer_state(state),
            [(shapes,)] * len(self.replicas),
        )

    def close(self) -> None:
        stop_replica_processes(self.replicas)
        self.replicas = []
        if self.rank is not None:
            self.rank.close()
            self.rank = None
