"""CPU-only derivation from the immutable completed 100-update replay plan."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from sparselab.canonical import canonical_digest
from sparselab.config.models import RunConfig
from sparselab.data.document_batching import collate_document_update
from sparselab.data.story_packing import MatchedStoryDataset
from sparselab.engines.base import Microbatch
from sparselab.training.continuation import runtime_optimization_transition
from sparselab.training.manifest import config_sha256, sha256_file, source_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError(
            "Commit the reviewed implementation before preparing a source-pinned replay"
        )
    plan = json.loads(args.parent_plan.read_text())
    parent_digest = plan.pop("plan_sha256")
    if canonical_digest(plan) != parent_digest:
        raise ValueError("Original immutable plan changed")
    for path, expected in [
        (plan["config"], plan["config_file_sha256"]),
        (plan["data"], plan["data_sha256"]),
        (
            Path(plan["checkpoint"]) / "manifest.json",
            plan["checkpoint_manifest_file_sha256"],
        ),
    ]:
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"Original input changed: {path}")
    checkpoint_manifest = json.loads(
        (Path(plan["checkpoint"]) / "manifest.json").read_text()
    )
    run_manifest = json.loads(
        (Path(plan["checkpoint"]).parent.parent / "manifest.json").read_text()
    )
    parent = RunConfig.model_validate(run_manifest["effective_config"])
    if (
        config_sha256(parent.model_dump(mode="json"))
        != checkpoint_manifest["config_sha256"]
    ):
        raise ValueError("Checkpoint and saved parent config identity differ")
    plan["parent_source_identity_sha256"] = checkpoint_manifest[
        "source_identity_sha256"
    ]
    plan["candidate_source_identity_sha256"] = source_identity()["sha256"]
    plan["source_drift"] = {
        "scope": "explicit disposable replay of tested document-batching implementation",
        "parent": plan["parent_source_identity_sha256"],
        "candidate": plan["candidate_source_identity_sha256"],
        "production_adoption": "Requires native --allow-runtime-drift and --allow-runtime-optimization; no authority granted by benchmark",
    }
    plan["parent_config"] = parent.model_dump(mode="json")
    plan["parent_source_commit"] = plan["source_commit"]
    plan["parent_plan_sha256"] = parent_digest
    plan["source_commit"] = revision
    plan["arms"] = {}
    plan["transitions"] = {}
    plan["shape_audit"] = {}
    ids = np.load(plan["data"], mmap_mode="r")
    matched = parent.training.matched_stories
    dataset = MatchedStoryDataset(
        ids,
        parent.training.seq_len,
        plan["eos"],
        layout=matched.layout,
        targets_per_update=matched.targets_per_update,
        unpadded_documents=matched.unpadded_documents,
        frozen_update_ends=matched.frozen_update_ends,
    )
    for arm, width in [("baseline", 1), ("padded_mb1", 1), ("mb4", 4), ("mb8", 8)]:
        raw = parent.model_dump(mode="json")
        raw["runtime"]["memory"]["activation_checkpointing"]["enabled"] = False
        raw["runtime"]["compile_transformer_blocks"] = False
        raw["runtime"]["compile_output_head_loss"] = False
        raw["training"]["micro_batch_size"] = width
        raw["evaluation"]["batch_size"] = 1
        if arm != "baseline":
            raw["training"]["matched_stories"]["document_batching"] = "padded"
            raw["training"]["matched_stories"]["padding_multiple"] = 128
        cfg = RunConfig.model_validate(raw)
        plan["arms"][arm] = cfg.model_dump(mode="json")
        plan["transitions"][arm] = runtime_optimization_transition(
            parent, cfg, plan["checkpoint_sha256"]
        )
        audit = []
        for group in plan["groups"]:
            rows = list(dataset.update_rows(group["index"], len(ids)))
            if [len(r[0]) for r in rows] != group["lengths"]:
                raise ValueError("Original document membership changed")
            batches = (
                [Microbatch(r[0][None], r[1][None]) for r in rows]
                if arm == "baseline"
                else collate_document_update(
                    rows,
                    width=width,
                    eos=plan["eos"],
                    max_length=cfg.training.seq_len,
                    padding_multiple=128,
                )
            )
            if sum(int((b.targets != -100).sum()) for b in batches) != group["targets"]:
                raise ValueError("Original target masks changed")
            audit.append(
                {
                    "group": group["index"],
                    "compute_slots": sum(b.inputs.size for b in batches),
                    "real_positions": sum(len(r[0]) for r in rows),
                    "attention_pairs": sum(
                        b.inputs.shape[0] * b.inputs.shape[1] ** 2 for b in batches
                    ),
                    "shapes": [list(b.inputs.shape) for b in batches],
                }
            )
        plan["shape_audit"][arm] = {
            "updates": audit,
            "compute_slots": sum(r["compute_slots"] for r in audit),
            "attention_pairs": sum(r["attention_pairs"] for r in audit),
            "distinct_shapes": sorted({tuple(s) for r in audit for s in r["shapes"]}),
        }
    plan["plan_sha256"] = canonical_digest(plan)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    for arm, cfg in plan["arms"].items():
        receipt = {
            "decision": "explicit_unqualified_deviation",
            "qualified": False,
            "reason": "Isolated same-checkpoint document microbatch throughput experiment; CPU semantics tested, GPU qualification pending.",
            "plan_sha256": plan["plan_sha256"],
            "arm": arm,
            "source_commit": revision,
            "config_sha256": canonical_digest(cfg),
            "runtime_profile_sha256": sha256_file(args.runtime_profile),
        }
        (args.output / f"{arm}-admission.json").write_text(
            json.dumps(receipt, indent=2)
        )
    print(
        json.dumps(
            {
                "source_commit": revision,
                "plan_sha256": plan["plan_sha256"],
                "targets": sum(g["targets"] for g in plan["groups"]),
                "shape_audit": {
                    arm: {k: v for k, v in a.items() if k != "updates"}
                    for arm, a in plan["shape_audit"].items()
                },
            }
        )
    )
    import torch

    assert not torch.xpu.is_initialized()


if __name__ == "__main__":
    main()
