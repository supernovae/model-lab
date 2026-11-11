"""Operational checkpoint cadence and incremental disk planning for a locked cell."""

from __future__ import annotations

from dataclasses import asdict
from math import ceil
from typing import TYPE_CHECKING, Any

from klab.config.models import RunConfig
from klab.model.inspection import inspection_report
from klab.workspace_preflight import (
    projected_data_bytes,
    training_storage_checks,
    verified_prepared_footprint,
)

if TYPE_CHECKING:
    from klab.data.verification import VerifiedPreparedData


def checkpoint_generation_bounds(
    config: RunConfig, *, completed_steps: int = 0, completed_tokens: int = 0
) -> dict[str, Any]:
    """Bound checkpoints from trainer watermarks and manager prune semantics."""
    # Plan retention does not alter CheckpointManager's per-run pruning policy.
    training = config.training
    # Loss masking and partial batches can commit fewer targets than the nominal
    # batch size. max_tokens / nominal batch is not an upper bound on updates.
    steps = training.max_steps
    tokens_per_update = (
        training.seq_len * training.micro_batch_size * training.gradient_accumulation
    )
    checkpoints = config.checkpoint
    explicit = tuple(step for step in checkpoints.steps if 0 < step < steps)
    scheduled_steps = (
        (steps - 1) // checkpoints.every_steps if checkpoints.every_steps else 0
    )
    scheduled_tokens = (
        ((steps - 1) * tokens_per_update) // checkpoints.every_tokens
        if checkpoints.every_tokens
        else 0
    )
    milestone_events = (
        len(checkpoints.retention.milestones) if checkpoints.retention else 0
    )
    validation_steps = (steps - 1) // config.evaluation.every_steps
    minute_steps = steps - 1 if checkpoints.every_minutes is not None else 0
    # With no other trigger to shift the watermark, a step cadence divisible
    # by validation cadence fires only at validation boundaries.
    disjoint_step_events = scheduled_steps
    if (
        checkpoints.every_steps is not None
        and checkpoints.every_steps % config.evaluation.every_steps == 0
        and not explicit
        and not milestone_events
        and checkpoints.every_tokens is None
        and checkpoints.every_minutes is None
    ):
        disjoint_step_events = 0
    # Initial and terminal saves are unconditional. The terminal evaluation's
    # best event coincides with the terminal write, as do all simultaneous events.
    # A target-token cap may terminate before a declared explicit step.
    lower = 2
    upper = 2 + min(
        steps - 1,
        len(explicit)
        + milestone_events
        + disjoint_step_events
        + scheduled_tokens
        + validation_steps
        + minute_steps,
    )
    if completed_steps:
        # A new resumed run stores only its future generations. Existing parent
        # bytes already reduce free space and must not be projected a second time.
        steps = training.max_steps - completed_steps
        remaining_tokens = training.max_tokens - completed_tokens
        if steps < 1 or remaining_tokens < 1:
            raise ValueError("continuation storage requires remaining training budget")
        explicit = tuple(
            s for s in checkpoints.steps if completed_steps < s < training.max_steps
        )
        scheduled_steps = (
            ((steps - 1) // checkpoints.every_steps + 1)
            if checkpoints.every_steps
            else 0
        )
        # Matched-document updates are not seq_len * micro_batch_size targets.
        # Use the remaining target cap and allow an immediately due inherited watermark.
        scheduled_tokens = (
            (
                (remaining_tokens + checkpoints.every_tokens - 1)
                // checkpoints.every_tokens
                + 1
            )
            if checkpoints.every_tokens
            else 0
        )
        validation_steps = (
            (training.max_steps - 1) // config.evaluation.every_steps
            - completed_steps // config.evaluation.every_steps
        )
        minute_steps = steps - 1 if checkpoints.every_minutes is not None else 0
        upper = 2 + min(
            steps - 1,
            len(explicit)
            + milestone_events
            + scheduled_steps
            + scheduled_tokens
            + validation_steps
            + minute_steps,
        )
    keep_periodic = checkpoints.keep_periodic
    retained_upper = upper if keep_periodic else min(upper, 3)
    # Manager prunes *after* committing and projecting a new generation. Reserve
    # a fourth transient generation when latest, previous and best are distinct.
    peak_generations = upper if keep_periodic else min(upper, 4)
    if checkpoints.retention is not None:
        retained_upper = min(upper, checkpoints.retention.retained_upper)
        peak_generations = min(upper, checkpoints.retention.retained_upper + 1)
    return {
        "steps": steps,
        "explicit": explicit,
        "scheduled_steps": scheduled_steps,
        "scheduled_tokens": scheduled_tokens,
        "validation_steps": validation_steps,
        "lower": min(lower, upper),
        "upper": upper,
        "retained_upper": retained_upper,
        "peak_generations": peak_generations,
    }


def storage_preview(
    config: RunConfig,
    retention: dict[str, bool] | None = None,
    *,
    verified_prepared: VerifiedPreparedData | None = None,
) -> dict[str, Any]:
    """Bound writes separately from peak future growth and existing durable data.

    The trainer writes at most once per optimizer update, including coincident
    validation, step, token and explicit triggers. Every successful write resets
    all cadence watermarks; therefore each interval's maximum event count can
    only decrease when other events force extra writes. Wall-clock cadence has
    no update-time lower bound and may fire at every update if configured.
    Plan retention does not alter the trainer's checkpoint manager.
    """
    bounds = checkpoint_generation_bounds(config)
    steps = bounds["steps"]
    explicit = bounds["explicit"]
    scheduled_steps = bounds["scheduled_steps"]
    scheduled_tokens = bounds["scheduled_tokens"]
    validation_steps = bounds["validation_steps"]
    lower = bounds["lower"]
    upper = bounds["upper"]
    retained_upper = bounds["retained_upper"]
    peak_generations = bounds["peak_generations"]
    checkpoints = config.checkpoint
    checkpoint_bytes = int(inspection_report(config)["estimated_checkpoint_bytes"])
    checkpoint_with_overhead = ceil(1.25 * checkpoint_bytes)
    checks = [
        asdict(check)
        for check in training_storage_checks(
            config,
            checkpoint_generations_upper=peak_generations,
            verified_prepared=verified_prepared,
        )
    ]
    if verified_prepared is None:
        existing_prepared_bytes = None
        future_cache_bytes = projected_data_bytes(config)
        future_run_copy_bytes = future_cache_bytes
    else:
        # The preflight verifies the sealed receipt, fresh fingerprints and
        # manifest before any proof sizes are used here.
        existing_prepared_bytes, _ = verified_prepared_footprint(verified_prepared)
        future_cache_bytes = 0
        future_run_copy_bytes = existing_prepared_bytes
    return {
        "initial_generation": 0,
        "terminal_generation": steps,
        "explicit_checkpoint_steps": explicit,
        "checkpoint_every_steps": checkpoints.every_steps,
        "checkpoint_every_tokens": checkpoints.every_tokens,
        "checkpoint_every_minutes": checkpoints.every_minutes,
        "validation_every_steps": config.evaluation.every_steps,
        "periodic_step_trigger_upper": scheduled_steps,
        "periodic_token_trigger_upper": scheduled_tokens,
        "validation_triggered_best_upper": validation_steps + 1,
        "write_count_lower": min(lower, upper),
        "write_count_upper": upper,
        "retained_generations_lower": 1,
        "retained_generations_upper": retained_upper,
        "peak_generations_upper": peak_generations,
        "requested_plan_retention": retention or {},
        "estimated_checkpoint_bytes": checkpoint_bytes,
        "estimated_total_write_bytes_upper": upper * checkpoint_with_overhead,
        "estimated_retained_bytes_upper": retained_upper * checkpoint_with_overhead,
        "estimated_peak_checkpoint_bytes_upper": (
            peak_generations * checkpoint_with_overhead
        ),
        "estimated_retained_inodes_upper": retained_upper * 16,
        "estimated_packed_cache_bytes": future_cache_bytes,
        "existing_prepared_bytes": existing_prepared_bytes,
        "future_cache_growth_bytes": future_cache_bytes,
        "future_run_copy_bytes": future_run_copy_bytes,
        "available_storage": checks,
        "estimated_free_bytes_after_retention": min(
            (
                check["available_bytes"]
                - check["projected_bytes"]
                - check["reserve_bytes"]
                for check in checks
            ),
            default=None,
        ),
    }
