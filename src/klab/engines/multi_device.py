"""Multi-device microbatch-parallel replica execution and flat-buffer reduction."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.nn import functional

from klab.config.models import RunConfig
from klab.engines.base import Microbatch
from klab.model.inspection import architecture_metrics
from klab.model.transformer import DenseLM
from klab.runtime import precision_context, synchronize


@dataclass
class SecondaryReplica:
    """In-process model replica bound to a secondary device."""

    device: torch.device
    model: DenseLM
    compiled_output_head_loss: Any = None
    trainable_params: tuple[torch.nn.Parameter, ...] = field(default=())
    # True until the replica copies the primary's parameters at the start of a
    # slice; set again after every applied optimizer step or state restore.
    stale: bool = True

    def __post_init__(self) -> None:
        if not self.trainable_params:
            self.trainable_params = tuple(
                p for p in self.model.parameters() if p.requires_grad
            )


def pull_replica_weights(
    primary_params: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
    replica: SecondaryReplica,
) -> None:
    """Copy the primary's trainable parameters onto one replica, tensor by tensor.

    Called from the replica's own thread at the start of its slice, so the copy
    overlaps with the primary's compute instead of delaying it.
    """
    with torch.no_grad():
        for replica_param, primary_param in zip(
            replica.trainable_params, primary_params, strict=True
        ):
            replica_param.copy_(primary_param, non_blocking=True)
    replica.stale = False


class GradientReduction:
    """Collect replica gradients on the primary device while it is still computing.

    Each replica thread pushes its gradients when its slice ends. Pushes are
    applied in replica order so the sum is the same on every run, and the first
    push hands over its transferred tensors, so the primary holds at most one
    extra copy of the gradients. ``merge`` adds the collected sum to the
    primary's own gradients once its slice has finished.
    """

    def __init__(
        self,
        primary_params: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
        primary_device: torch.device,
        pushes: int,
    ) -> None:
        self.primary_params = primary_params
        self.primary_device = primary_device
        self.collected: list[torch.Tensor | None] = [None] * len(primary_params)
        self.turns = [threading.Event() for _ in range(pushes)]

    def push(self, turn: int, replica: SecondaryReplica) -> None:
        try:
            if turn:
                self.turns[turn - 1].wait()
            with torch.no_grad():
                for index, parameter in enumerate(replica.trainable_params):
                    if parameter.grad is None:
                        continue
                    moved = parameter.grad.detach().to(
                        self.primary_device, non_blocking=True
                    )
                    held = self.collected[index]
                    if held is None:
                        self.collected[index] = moved
                    else:
                        held.add_(moved)
            replica.model.zero_grad(set_to_none=True)
        finally:
            self.turns[turn].set()

    def merge(self) -> None:
        with torch.no_grad():
            for parameter, collected in zip(
                self.primary_params, self.collected, strict=True
            ):
                if collected is None:
                    continue
                if parameter.grad is None:
                    parameter.grad = collected
                else:
                    parameter.grad.add_(collected)
        self.collected = [None] * len(self.primary_params)


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
    *,
    primary_params: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
    reduction: GradientReduction,
    turn: int,
) -> dict[str, Any]:
    """Run one replica's slice: pull weights, forward/backward, push gradients."""
    started = time.perf_counter()
    device = replica.device
    model = replica.model
    model.zero_grad(set_to_none=True)
    if replica.stale:
        pull_replica_weights(primary_params, replica)
    language_sum = torch.zeros((), device=device)
    aux_sum = torch.zeros((), device=device)
    diagnostic_sums: dict[str, torch.Tensor] = {}
    diagnostic_weights: dict[str, int] = {}
    losses_finite = torch.ones((), device=device, dtype=torch.bool)
    recomputed = 0.0
    executed_microbatches = 0
    trainable_inputs = list(replica.trainable_params)

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
                return_hidden=config.runtime.compile_output_head_loss,
                position_mask=position_mask,
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
    reduction.push(turn, replica)
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


def _replica_slice(
    replica: SecondaryReplica,
    config: RunConfig,
    pairs: list[tuple[Microbatch, int]],
    valid_targets: int,
    output_head_loss_fn: Any,
    primary_params: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
    reduction: GradientReduction,
    turn: int,
) -> dict[str, Any]:
    try:
        return execute_replica_microbatches(
            replica,
            config,
            pairs,
            valid_targets,
            output_head_loss_fn,
            primary_params=primary_params,
            reduction=reduction,
            turn=turn,
        )
    finally:
        # A failed slice must not leave later replicas waiting for its turn.
        reduction.turns[turn].set()


def dispatch_replicas_and_primary(
    *,
    replicas: list[SecondaryReplica],
    config: RunConfig,
    active_pairs: list[tuple[Microbatch, int]],
    valid_targets: int,
    output_head_loss_fn: Any,
    primary_fn: Any,
    primary_params: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
    primary_device: torch.device,
    executor: ThreadPoolExecutor,
) -> tuple[Any, list[dict[str, Any]], dict[str, float]]:
    """Run each replica's slice on its own thread while the caller runs the primary's.

    Compiled and eager kernels are dispatched the same way. A backward pass can
    block its host thread (embedding backward on XPU reads a count back per
    microbatch), so slices issued in turn from one thread run one device at a
    time; a thread per device keeps every device busy. Compilation itself is
    serialized by TorchDynamo's own lock.

    Each replica thread also copies the updated weights before its slice and
    moves its gradients to the primary after it, so that exchange overlaps with
    the primary's compute. The returned timings say how long each side computed
    and how long finished replicas waited for the primary.
    """
    world_size = 1 + len(replicas)
    primary_pairs = active_pairs[0::world_size]
    replica_tasks: list[tuple[SecondaryReplica, list[tuple[Microbatch, int]]]] = []
    for rep_idx, replica in enumerate(replicas):
        rep_pairs = active_pairs[rep_idx + 1 :: world_size]
        if rep_pairs:
            replica_tasks.append((replica, rep_pairs))
    reduction = GradientReduction(primary_params, primary_device, len(replica_tasks))
    started = time.perf_counter()
    futures = [
        executor.submit(
            _replica_slice,
            replica,
            config,
            rep_pairs,
            valid_targets,
            output_head_loss_fn,
            primary_params,
            reduction,
            turn,
        )
        for turn, (replica, rep_pairs) in enumerate(replica_tasks)
    ]
    try:
        primary_out = primary_fn(primary_pairs)
        synchronize(primary_device)
    finally:
        primary_finished = time.perf_counter()
        replica_outs = [future.result() for future in futures]
    reduction.merge()
    synchronize(primary_device)
    finished = time.perf_counter()
    all_computed = max(
        [primary_finished, *(out["finished_at"] for out in replica_outs)]
    )
    timings = {
        "performance/primary_compute_seconds": primary_finished - started,
        "performance/replica_compute_seconds": max(
            (out["compute_seconds"] for out in replica_outs), default=0.0
        ),
        # Time finished devices spent waiting for the slowest one, summed.
        "performance/device_wait_seconds": sum(
            all_computed - out["finished_at"] for out in replica_outs
        )
        + (all_computed - primary_finished),
        "performance/gradient_merge_seconds": finished - all_computed,
    }
    return primary_out, replica_outs, timings
