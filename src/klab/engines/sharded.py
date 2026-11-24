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
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor, Shard

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


_INTEGER_OF_WIDTH = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
_CHECKSUM_MASK = 2**64 - 1


def _checksum(tensor: torch.Tensor, offset: int = 0) -> int:
    """Position-weighted sum of a tensor's raw bits, modulo 2**64.

    Element ``i`` (in row-major order, counted from ``offset``) contributes its bits as
    an integer times ``offset + i + 1``. That is additive over shards cut along the
    first dimension when each is given its own offset, so the ranks' checksums add up
    to the whole tensor's without any rank seeing another's data, and it changes if
    shards are swapped, shifted or reordered, not only if bits are lost. Bits, not
    values: a NaN counts as itself.
    """
    if not tensor.numel():
        return 0
    width = _INTEGER_OF_WIDTH[tensor.element_size()]
    raw = tensor.contiguous().view(width).to(torch.int64).flatten()
    weights = torch.arange(
        offset + 1, offset + 1 + raw.numel(), dtype=torch.int64, device=raw.device
    )
    return int((raw * weights).sum().item()) & _CHECKSUM_MASK


def _shard_slices(
    value: torch.Tensor, parameter: Any, world_size: int
) -> list[torch.Tensor] | None:
    """``value`` cut into one slice per rank, or ``None`` if it is not sharded that way.

    A parameter sharded along its first dimension holds contiguous rows in rank order
    (``torch.chunk``, with empty slices when there are fewer rows than ranks). Each rank
    checks the slice it is handed against its own shard's shape, so a layout this does
    not match is refused instead of restored wrongly.
    """
    if (
        not isinstance(parameter, DTensor)
        or tuple(value.shape) != tuple(parameter.shape)
        or not value.ndim
        or [p for p in parameter.placements if isinstance(p, Shard)] != [Shard(0)]
    ):
        return None
    slices = list(torch.chunk(value, world_size, dim=0))
    slices += [value[:0]] * (world_size - len(slices))
    return slices


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
        # Optimizer-state entries taken by restore_slot, until restore_finish.
        self._restoring: dict[int, dict[str, Any]] = {}
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

    def restore_slot(
        self, index: int, key: str, value: Any, offset: int | None = None
    ) -> int | None:
        """Take one optimizer-state entry; return the checksum of this rank's shard.

        With ``offset`` the value is this rank's own slice of a sharded tensor, starting
        at that element of the whole. No rank is handed more than it keeps and nothing
        is exchanged between the cards: a collective would leave every buffer it moved
        resident in host memory for the life of the process, which cost a resumed
        1.2B-parameter run about 13 GiB of host memory per rank.
        """
        parameter = self._restore_parameters()[index]
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        checksum = None
        if offset is not None:
            assert isinstance(parameter, DTensor)
            expected = tuple(parameter.to_local().shape)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"optimizer state {key!r} of parameter {index}: rank {self.rank} was "
                    f"handed a slice of shape {tuple(value.shape)}, its shard is {expected}"
                )
            local = value.to(self.device)
            checksum = _checksum(local, offset)
            value = DTensor.from_local(
                local,
                parameter.device_mesh,
                parameter.placements,
                run_check=False,
                shape=parameter.shape,
                stride=parameter.stride(),
            )
        self._restoring.setdefault(index, {})[key] = value
        return checksum

    def restore_finish(self, param_groups: Any) -> None:
        """Load the entries taken by :meth:`restore_slot` into the optimizer."""
        from klab.training.checkpoints import restore_optimizer_state

        restore_optimizer_state(
            self.optimizer, {"state": self._restoring, "param_groups": param_groups}
        )
        self._restoring = {}
        # One unsharded temporary was made per entry; return the cached blocks.
        release_cached_memory(self.device)

    def _restore_parameters(self) -> list[torch.nn.Parameter]:
        return [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_worker_rank(
    config: RunConfig,
    backend: str,
    device: torch.device,
    rank: int,
    world_size: int,
    init_method: str,
    state: dict[str, torch.Tensor],
    *,
    built: Callable[[], None] = lambda: None,
) -> ShardedRank:
    """Build a non-primary rank's model on ``device`` and shard it.

    Nothing may outlive this call that refers to the unsharded parameters:
    ``SecondaryReplica`` keeps a tuple of them, and sharding replaces the model's
    parameters without freeing the originals, so a replica kept past this point
    pins a whole second copy of the weights on the device (4.7 GiB a card for a
    1.2B-parameter model, which was the headroom a resume needed).
    """
    replica = build_replica(config, device, state)
    built()
    model, compiled_loss = replica.model, replica.compiled_output_head_loss
    del replica
    return ShardedRank(
        config, backend, rank, world_size, init_method, model, device, compiled_loss
    )


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
        worker = build_worker_rank(
            config,
            backend,
            device,
            rank,
            world_size,
            init_method,
            state,
            built=lambda: connection.send(("built", None)),
        )
        # Not ``del``: the spawned process's arguments hold this same dict, and its
        # tensors are the primary's weights in shared memory, a whole host copy that
        # would otherwise stay mapped for the life of the worker.
        state.clear()
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
            elif command == "restore_slot":
                result = worker.restore_slot(*message[1:])
            elif command == "restore_finish":
                worker.restore_finish(message[1])
            else:
                raise ValueError(f"unknown shard command {command!r}")
            connection.send(("done", result))
        except Exception:  # noqa: BLE001 - the primary reports the rank's error
            report_failure(connection)


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
        """Hand every rank its part of each entry of ``state``; no collective runs.

        A sharded tensor is cut on the host and each rank gets only its own slice, as
        bytes over its pipe, so nothing is sent or copied to a device that the rank
        does not keep and at most one tensor is in flight. Each rank answers with the
        checksum of what it holds; they must add up to the checkpoint tensor's.
        """
        assert self.rank is not None
        workers = len(self.replicas)
        parameters = self.rank._restore_parameters()
        for index in sorted(state["state"]):
            slots = state["state"][index]
            for key in sorted(slots):
                value = slots[key]
                if not isinstance(value, torch.Tensor):
                    self.run(
                        "restore_slot",
                        lambda index=index, key=key, value=value: (
                            self.rank.restore_slot(index, key, value)
                        ),
                        [(index, key, value)] * workers,
                    )
                    continue
                value = value.detach().cpu()
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise ValueError(
                        f"checkpoint optimizer state {key!r} of parameter {index} "
                        "is non-finite; it cannot be restored"
                    )
                slices = _shard_slices(value, parameters[index], self.world_size)
                if slices is None:
                    # Bytes, not a tensor: pickling a tensor moves its storage into
                    # shared memory, and a step count sent that way would be the
                    # primary's own (the worker would increment it).
                    self.run(
                        "restore_slot",
                        lambda index=index, key=key, value=value: (
                            self.rank.restore_slot(index, key, value)
                        ),
                        [(index, key, value.numpy())] * workers,
                    )
                    continue
                offsets, start = [], 0
                for piece in slices:
                    offsets.append(start)
                    start += piece.numel()
                mine, theirs = self.run(
                    "restore_slot",
                    lambda index=index, key=key, piece=slices[0]: (
                        self.rank.restore_slot(index, key, piece, 0)
                    ),
                    [
                        (index, key, slices[rank].contiguous().numpy(), offsets[rank])
                        for rank in range(1, self.world_size)
                    ],
                )
                if (sum([mine, *theirs]) & _CHECKSUM_MASK) != _checksum(value):
                    raise ValueError(
                        f"restored optimizer state {key!r} of parameter {index} differs "
                        "from the checkpoint: a shard did not arrive intact"
                    )
        self.run(
            "restore_finish",
            lambda: self.rank.restore_finish(state["param_groups"]),
            [(state["param_groups"],)] * workers,
        )

    def close(self) -> None:
        stop_replica_processes(self.replicas)
        self.replicas = []
        if self.rank is not None:
            self.rank.close()
            self.rank = None
