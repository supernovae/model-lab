"""Checkpoint-scoped prepared-data exposure; no model work or new counters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sparselab.canonical import canonical_digest

if TYPE_CHECKING:
    from sparselab.data.packing import BatchCursor, TokenBlockDataset
    from sparselab.training.manifest import ArtifactIdentity


def exposure_context(
    dataset: TokenBlockDataset,
    artifacts: tuple[ArtifactIdentity, ...],
    *,
    seed: int,
    run_id: str,
    continuation: str,
    prior: dict[str, Any] | None,
) -> dict[str, Any]:
    """Capture the denominator and already-verified input identities once."""
    inventory = {
        item.relative_path.replace("\\", "/"): item.sha256 for item in artifacts
    }
    identities = {
        "prepared_manifest_sha256": inventory["data/manifest.json"],
        "train_tokens_sha256": inventory["data/train.npy"],
        "tokenizer_sha256": inventory["tokenizer.json"],
        "supervision_sha256": inventory["data/train_supervision.npy"]
        if dataset.supervision is not None
        else None,
        "supervision": "masked" if dataset.supervision is not None else "all_tokens",
    }
    protocol = {
        "kind": "packed-next-token-supervised-exposure-v1",
        "sequence_length": dataset.seq_len,
        "shuffle_seed": seed,
        "identities": identities,
    }
    from sparselab.data.story_packing import MatchedStoryDataset

    if isinstance(dataset, MatchedStoryDataset):
        protocol.update(
            {
                "kind": "matched-eos-stories-v1",
                "layout": dataset.layout,
                "targets_per_update": dataset.targets_per_update,
                "order": "source",
                "initial_context": "virtual-eos",
                "group_offsets_sha256": canonical_digest(
                    dataset.group_offsets.tolist()
                ),
            }
        )
        protocol.pop("shuffle_seed")
    resumed = continuation == "RESUMED"
    return {
        "version": 1,
        "protocol": protocol,
        "protocol_sha256": canonical_digest(protocol),
        "supply": dataset.exposure_supply(),
        "scope": {
            "continuation": continuation,
            "run_id": run_id,
            "origin_run_id": (prior or {}).get("scope", {}).get("origin_run_id")
            if resumed
            else run_id,
            "prior_checkpoint_exposure_available": bool(prior) if resumed else None,
            "counter_basis": "resumed cumulative" if resumed else "new run",
        },
    }


def checkpoint_exposure(
    context: dict[str, Any],
    *,
    tokens: int,
    cursor: BatchCursor,
    wall_seconds: float,
    invocation_seconds: float,
    update_seconds: float,
    wall_complete: bool,
    updates_complete: bool,
) -> dict[str, Any]:
    supply = context["supply"]
    blocks = supply["eligible_blocks"]
    targets = supply["usable_supervised_targets_per_pass"]
    if blocks <= 0 or targets <= 0 or not 0 <= cursor.next_block <= blocks:
        raise ValueError("invalid exposure supply or cursor")
    return {
        **context,
        "committed_supervised_targets": tokens,
        "effective_passes": tokens / targets,
        "cursor": {
            "epoch": cursor.epoch,
            "next_block": cursor.next_block,
            "completed_loader_passes": cursor.epoch + cursor.next_block // blocks,
            "next_eligible_block": cursor.next_block % blocks,
        },
        "finer_consumed_positions": None,
        "timing": {
            "optimizer_update_seconds": update_seconds if updates_complete else None,
            "training_loop_elapsed_seconds": wall_seconds if wall_complete else None,
            "invocation_elapsed_seconds_at_checkpoint": invocation_seconds,
            "other_training_overhead_seconds": None,
            "optimizer_basis": "applied updates, including their overflow retries; excludes wholly failed updates",
        },
    }
