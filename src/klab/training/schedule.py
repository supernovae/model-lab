"""Author fresh matched-group schedules from cold-verified prepared inputs.

This is config authoring, not checkpoint continuation. The source input bundle
remains immutable; publish a new bundle with the derived configuration before use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from klab.config import RunConfig, load_config

if TYPE_CHECKING:
    from klab.data.story_packing import MatchedStoryDataset


def schedule_settings(config: RunConfig, plan: MatchedStoryDataset) -> dict[str, Any]:
    """Compute exact schedule boundaries without changing targets or warmup."""
    budget = config.training.max_tokens
    if plan.target_group_transition or (
        config.training.matched_stories is not None
        and config.training.matched_stories.target_group_transition is not None
    ):
        raise ValueError(
            "fresh scheduling cannot author a checkpoint-bound group transition"
        )
    if budget > len(plan.ids):
        raise ValueError(
            "fresh schedule target budget exceeds prepared supply; repeated exposure is not allowed"
        )
    steps = plan.horizon(budget)
    settings: dict[str, Any] = {
        "training.max_steps": steps,
        "optimizer.decay_steps": steps,
    }
    if config.optimizer.name != "adamw":
        raise ValueError("matched schedule derivation requires AdamW")
    policy = config.checkpoint.retention
    if policy is not None:
        milestones = []
        for milestone in policy.milestones:
            if milestone.step > steps:
                raise ValueError("checkpoint milestone exceeds derived horizon")
            targets = (
                min(budget, plan.targets_for_updates(milestone.step))
                if milestone.step
                else 0
            )
            milestones.append({**milestone.model_dump(), "committed_targets": targets})
        settings["checkpoint.retention"] = {
            **policy.model_dump(mode="json"),
            "milestones": milestones,
            "terminal_step": steps,
            "terminal_targets": budget,
        }
    return settings


def derive_schedule(source: Path, bundle: Path, output: Path) -> dict[str, Any]:
    """Cold-verify actual packed bytes and publish a native config derivation."""
    import numpy as np

    from klab.data.story_packing import MatchedStoryDataset
    from klab.data.tokenizer import load_tokenizer
    from klab.derivation import derive_config
    from klab.staging import verify_prepared_inputs

    config = load_config(source)
    matched = config.training.matched_stories
    if matched is None:
        raise ValueError("schedule derivation requires matched_stories")
    if matched.target_group_transition is not None:
        raise ValueError(
            "fresh scheduling cannot author a checkpoint-bound group transition"
        )
    verified = verify_prepared_inputs(bundle, config, verification_mode="cold")
    assets = Path(verified["assets_root"])
    tokenizer = load_tokenizer(assets / "tokenizer.json")
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("matched schedule requires EOS")
    if (assets / "data/train_supervision.npy").exists():
        raise ValueError("matched schedule does not support supervision masks")
    ids = np.load(assets / "data/train.npy", mmap_mode="r", allow_pickle=False)
    plan = MatchedStoryDataset(
        ids,
        config.training.seq_len,
        eos,
        layout=matched.layout,
        targets_per_update=matched.targets_per_update,
        unpadded_documents=matched.unpadded_documents,
        frozen_update_ends=matched.frozen_update_ends,
        target_group_transition=(
            matched.target_group_transition.model_dump()
            if matched.target_group_transition
            else None
        ),
    )
    settings = schedule_settings(config, plan)
    receipt = derive_config(source, output, settings)
    steps = settings["training.max_steps"]
    previous_targets = plan.targets_for_updates(steps - 1) if steps > 1 else 0
    complete_targets = plan.targets_for_updates(steps)
    return {
        "status": "ok",
        "prepared_inputs_sha256": verified["sha256"],
        "groups_per_pass": len(plan),
        "targets_per_pass": len(ids),
        "max_steps": settings["training.max_steps"],
        "max_tokens": config.training.max_tokens,
        "committed_targets": config.training.max_tokens,
        "final_group_targets": config.training.max_tokens - previous_targets,
        "final_group_complete_targets": complete_targets - previous_targets,
        "final_group_clipped_targets": complete_targets - config.training.max_tokens,
        "derivation": receipt,
        "requires_prepared_inputs_publication": True,
    }


def register_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "schedule", help="Derive a fresh schedule from verified matched groups"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--prepared-inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.set_defaults(handler=run_schedule)


def run_schedule(args: argparse.Namespace) -> None:
    print(json.dumps(derive_schedule(args.config, args.prepared_inputs, args.output)))
