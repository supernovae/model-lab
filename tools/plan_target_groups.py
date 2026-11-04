"""CPU-only first-pass continuation planner; never opens optimizer/model tensors.

Run only when a full sequential read of the prepared train stream is acceptable.
The generated config is for a NEW run from the supplied completed generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from sparselab.config.loading import load_config
from sparselab.config.models import RunConfig
from sparselab.data.story_packing import MatchedStoryDataset
from sparselab.data.tokenizer import load_tokenizer
from sparselab.training.mlx_checkpoints import strict_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--targets", type=int, choices=(32768, 65536, 131072), default=131072
    )
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint.resolve()
    manifest = strict_json(checkpoint / "manifest.json")
    root = checkpoint.parent.parent
    tokenizer = load_tokenizer(root / "tokenizer.json")
    eos = tokenizer.token_to_id("<eos>")
    source = config.training.matched_stories
    if source is None or source.target_group_transition is not None:
        raise ValueError("requires an original matched-document config")
    if manifest["step"] >= config.training.max_steps:
        raise ValueError("the selected checkpoint has no unconsumed budget")
    policy = {
        "after_tokens": manifest["tokens_seen"],
        "previous_targets_per_update": source.targets_per_update,
        "parent_checkpoint_sha256": manifest["sha256"],
    }
    ids = np.load(root / "data/train.npy", mmap_mode="r", allow_pickle=False)
    plan = MatchedStoryDataset(
        ids,
        config.training.seq_len,
        eos,
        layout=source.layout,
        targets_per_update=args.targets,
        unpadded_documents=True,
        frozen_update_ends=source.frozen_update_ends,
        target_group_transition=policy,
    )
    # Configuration stage only. Native full-state continuation verifies all source
    # bytes, checkpoint binding, optimizer schedule and the cursor before adoption.
    if int(plan.group_offsets[manifest["step"]]) != manifest["tokens_seen"]:
        raise ValueError(
            "checkpoint cursor does not name the preserved prefix boundary"
        )
    raw = config.model_dump(mode="json")
    raw["name"] = f"{config.name}-targets-{args.targets}"
    raw["training"]["max_steps"] = plan.horizon(config.training.max_tokens)
    raw["training"]["matched_stories"].update(
        targets_per_update=args.targets,
        target_group_transition=policy,
        max_padded_slots=8192,
        max_attention_cells=16777216,
    )
    # Preserve approximately the old periodic recovery distance in targets.
    # Best/terminal/cancel saves remain independently due in the trainer.
    raw["checkpoint"].update(every_steps=None, every_tokens=5000000)
    result = RunConfig.model_validate(raw)
    args.output.write_text(
        yaml.safe_dump(result.model_dump(mode="json"), sort_keys=False)
    )
    print(
        json.dumps(
            {
                "config": str(args.output),
                "checkpoint": str(checkpoint),
                "source_step": manifest["step"],
                "next_token_offset": manifest["tokens_seen"],
                "new_max_steps": result.training.max_steps,
                "remaining_updates": result.training.max_steps - manifest["step"],
                "remaining_targets": result.training.max_tokens
                - manifest["tokens_seen"],
                "lr_policy": "retain absolute-step floor; retain Adam moments and step count",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
