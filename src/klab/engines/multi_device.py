"""Multi-device microbatch-parallel replica execution and flat-buffer reduction."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
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

    def __post_init__(self) -> None:
        if not self.trainable_params:
            self.trainable_params = tuple(
                p for p in self.model.parameters() if p.requires_grad
            )


def sync_replicas_weights(
    primary_params: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
    replicas: list[SecondaryReplica],
) -> None:
    """Broadcast trainable parameters to all replicas using one flat copy per replica."""
    if not primary_params or not replicas:
        return
    with torch.no_grad():
        flat_primary = _flatten_dense_tensors([p.detach() for p in primary_params])
        for replica in replicas:
            flat_replica = flat_primary.to(replica.device, non_blocking=True)
            for rep_param, unflattened in zip(
                replica.trainable_params,
                _unflatten_dense_tensors(flat_replica, replica.trainable_params),
                strict=True,
            ):
                rep_param.copy_(unflattened, non_blocking=True)


def reduce_replica_gradients(
    primary_params: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
    replica: SecondaryReplica,
    primary_device: torch.device,
) -> None:
    """Reduce a replica's gradients onto the primary device in a single flat transfer."""
    if (
        not primary_params
        or not replica.trainable_params
        or all(p.grad is None for p in replica.trainable_params)
    ):
        return
    rep_grads = [
        p.grad.detach()
        if p.grad is not None
        else torch.zeros_like(p, memory_format=torch.preserve_format)
        for p in replica.trainable_params
    ]
    with torch.no_grad():
        flat_rep_grads = _flatten_dense_tensors(rep_grads)
        flat_on_primary = flat_rep_grads.to(primary_device, non_blocking=True)
        for primary_param, grad_slice in zip(
            primary_params,
            _unflatten_dense_tensors(flat_on_primary, primary_params),
            strict=True,
        ):
            if primary_param.grad is None:
                primary_param.grad = grad_slice.clone()
            else:
                primary_param.grad.add_(grad_slice)
    replica.model.zero_grad(set_to_none=True)


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
    sync_device: bool = True,
) -> dict[str, Any]:
    """Execute forward and backward on a replica's microbatch slice."""
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

    for batch, chunk_valid in pairs:
        if not chunk_valid and batch.position_mask is None:
            continue
        if batch.semantic_queries is not None or batch.semantic_mask is not None:
            raise ValueError(
                "secondary_device_indices cannot be combined with semantic query sidecars"
            )
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
                semantic_queries=None,
                memory_mask=None,
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

    if sync_device:
        synchronize(device)
    return {
        "language_sum": language_sum,
        "aux_sum": aux_sum,
        "losses_finite": losses_finite,
        "recomputed": recomputed,
        "executed_microbatches": executed_microbatches,
        "diagnostic_sums": diagnostic_sums,
        "diagnostic_weights": diagnostic_weights,
    }


def dispatch_replicas_and_primary(
    *,
    replicas: list[SecondaryReplica],
    config: RunConfig,
    active_pairs: list[tuple[Microbatch, int]],
    valid_targets: int,
    output_head_loss_fn: Any,
    primary_fn: Any,
    executor: ThreadPoolExecutor | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Launch replica slices before the primary slice, overlapping on eager runtimes."""
    world_size = 1 + len(replicas)
    primary_pairs = active_pairs[0::world_size]
    replica_tasks: list[tuple[SecondaryReplica, list[tuple[Microbatch, int]]]] = []
    for rep_idx, replica in enumerate(replicas):
        rep_pairs = active_pairs[rep_idx + 1 :: world_size]
        if rep_pairs:
            replica_tasks.append((replica, rep_pairs))

    compiled = (
        config.runtime.compile_transformer_blocks
        or config.runtime.compile_output_head_loss
    )
    if executor is not None and replica_tasks and not compiled:
        futures = [
            executor.submit(
                execute_replica_microbatches,
                replica,
                config,
                rep_pairs,
                valid_targets,
                output_head_loss_fn,
                sync_device=False,
            )
            for replica, rep_pairs in replica_tasks
        ]
        primary_out = primary_fn(primary_pairs)
        replica_outs = [fut.result() for fut in futures]
        for replica, _ in replica_tasks:
            synchronize(replica.device)
        return primary_out, replica_outs

    # When torch.compile is active, dispatch replica kernels before primary kernels
    # and synchronize once after all devices have been queued.
    replica_outs = [
        execute_replica_microbatches(
            replica,
            config,
            rep_pairs,
            valid_targets,
            output_head_loss_fn,
            sync_device=False,
        )
        for replica, rep_pairs in replica_tasks
    ]
    primary_out = primary_fn(primary_pairs)
    for replica, _ in replica_tasks:
        synchronize(replica.device)
    return primary_out, replica_outs
