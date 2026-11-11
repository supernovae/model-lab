"""Authenticate native terminal counters before reconciling an attempt contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from klab.canonical import canonical_digest, canonical_json
from klab.training.manifest import read_manifest

NativeReceiptKind = Literal[
    "train", "panel", "campaign_run", "campaign_panel", "campaign_evaluation", "none"
]


def _training(
    path: Path, parent_checkpoint_path: Path | None
) -> tuple[int, int, int, int]:
    from klab.evaluation.evidence import experiment_evidence

    if not path.is_dir() or path.is_symlink():
        raise ValueError("native training run directory is missing or linked")
    manifest = read_manifest(path / "manifest.json")
    progress_path = path / "progress.json"
    if progress_path.is_symlink():
        raise ValueError("native training progress is linked")
    progress = json.loads(progress_path.read_text())
    manifest_sha = canonical_digest(manifest)
    evidence = experiment_evidence(path, verification_mode="cold")
    if (
        manifest.get("run_id") != path.name
        or evidence.get("run_id") != path.name
        or evidence.get("verified_checkpoints") is not True
        or progress.get("status") != "completed"
        or progress.get("manifest_sha256") != manifest_sha
    ):
        raise ValueError("native training completion or checkpoint evidence unverified")
    latest = progress.get("latest")
    records = evidence["checkpoints"]
    if not isinstance(latest, dict) or not isinstance(records, list):
        raise TypeError("native terminal checkpoint is missing")
    candidates = [
        row
        for row in records
        if row.get("path") == latest.get("relative_path")
        and row.get("digest") == latest.get("manifest_sha256")
    ]
    if len(candidates) != 1:
        raise ValueError("native terminal checkpoint differs from progress")
    terminal = candidates[0]
    if (
        type(terminal.get("step")) is not int
        or type(terminal.get("tokens_seen")) is not int
        or terminal["step"] <= 0
        or terminal["tokens_seen"] <= 0
        or (terminal["step"], terminal["tokens_seen"])
        != (progress.get("step"), progress.get("tokens_seen"))
    ):
        raise ValueError("native terminal counters differ from verified checkpoint")
    continuation = manifest.get("continuation_kind")
    if continuation == "FRESH":
        if (
            parent_checkpoint_path is not None
            or manifest.get("parent_run_id") is not None
        ):
            raise ValueError("fresh run cannot claim a parent checkpoint")
        prior_step = prior_tokens = 0
    elif continuation in {"RESUMED", "PROMOTED"}:
        if parent_checkpoint_path is None or not parent_checkpoint_path.is_absolute():
            raise ValueError("continuation requires a pinned parent checkpoint")
        from klab.training.checkpoints import CheckpointManager

        parent_run = parent_checkpoint_path.parent.parent
        parent_manifest = read_manifest(parent_run / "manifest.json")
        parent_manifest_sha = hashlib.sha256(
            canonical_json(parent_manifest)
        ).hexdigest()
        manager = CheckpointManager(parent_run, manifest_sha256=parent_manifest_sha)
        parent_report = manager.verify(
            parent_checkpoint_path,
            parent_manifest_sha,
            require_training_state=True,
            verification_mode="cold",
        )
        if not parent_report.valid or parent_checkpoint_path.is_symlink():
            raise ValueError("continuation parent checkpoint is unverified")
        parent_generation = manager._resolve(parent_checkpoint_path)
        parent = json.loads((parent_generation / "manifest.json").read_text())
        if (
            manifest.get("parent_run_id") != parent_run.name
            or manifest.get("checkpoint_sha256") != parent.get("sha256")
            or type(parent.get("step")) is not int
            or type(parent.get("tokens_seen")) is not int
        ):
            raise ValueError(
                "continuation parent identity differs from native manifest"
            )
        prior_step, prior_tokens = (
            (parent["step"], parent["tokens_seen"])
            if continuation == "RESUMED"
            else (0, 0)
        )
    else:
        raise ValueError("unknown native continuation kind")
    if terminal["step"] <= prior_step or terminal["tokens_seen"] <= prior_tokens:
        raise ValueError("native run advanced no optimizer frontier")
    return terminal["step"] - prior_step, terminal["tokens_seen"] - prior_tokens, 0, 0


def _panel(path: Path) -> tuple[int, int, int, int]:
    from klab.evaluation.panel import verify_panel_result

    panel = verify_panel_result(path, verification_mode="cold")
    rows = panel["rows"]
    if not rows or any(row["status"] != "COMPLETED" for row in rows):
        # A failed row can have unrecorded partial output. The full reservation
        # stays charged, with actual counters deliberately unestablished.
        raise ValueError("panel has incomplete or failed generation attempts")
    return 0, 0, len(rows), sum(len(row["token_ids"]) for row in rows)


def _campaign(
    source: Path, work_dir: Path, stage_id: str, *, expected_content_sha256: str
) -> tuple[str, Path]:
    from klab.campaign.engine import CampaignEngine

    engine = CampaignEngine(source, work_dir, cold_verify=True)
    status = engine.inspect("status")
    rows = {row["id"]: row for row in status["stages"]}
    if stage_id not in rows or rows[stage_id]["state"] != "COMPLETE":
        raise ValueError("contracted Campaign stage is not complete")
    stage = engine.stages[stage_id]
    if stage.kind not in {"experiment_run", "generation_panel", "evaluation"}:
        raise ValueError("contracted Campaign stage cannot provide native counters")
    run_stage = (
        stage
        if stage.kind == "experiment_run"
        else engine.stages[engine.stages[stage.collect].run]
    )
    lock = engine._lock(rows, run_stage.plan)
    if lock.scientific_sha256 != expected_content_sha256:
        raise ValueError("Campaign scientific lock differs from attempt contract")
    location = rows[stage_id].get("availability", {}).get("path")
    if not isinstance(location, str):
        raise TypeError("completed Campaign stage lacks native receipt path")
    return stage.kind, Path(location)


def verify_native_phase_counters(
    kind: NativeReceiptKind,
    receipt_path: Path | None,
    *,
    campaign_stage: str | None = None,
    campaign_work_dir: Path | None = None,
    parent_checkpoint_path: Path | None = None,
    expected_content_sha256: str,
) -> tuple[int, int, int, int]:
    """Return (updates, nonmasked targets, calls, completion IDs), or fail closed."""
    if kind == "none":
        if (
            receipt_path is not None
            or campaign_stage is not None
            or parent_checkpoint_path is not None
        ):
            raise ValueError("counter-free phase must not claim a native receipt")
        return 0, 0, 0, 0
    if receipt_path is None or not receipt_path.is_absolute():
        raise ValueError("absolute native receipt path required")
    if (
        kind in {"panel", "campaign_panel", "campaign_evaluation"}
        and parent_checkpoint_path is not None
    ):
        raise ValueError("panel receipt cannot claim a training parent")
    if kind in {"campaign_run", "campaign_panel", "campaign_evaluation"}:
        if campaign_stage is None or campaign_work_dir is None:
            raise ValueError("Campaign receipt requires stage and work directory")
        actual_kind, path = _campaign(
            receipt_path,
            campaign_work_dir,
            campaign_stage,
            expected_content_sha256=expected_content_sha256,
        )
        expected_kind = {
            "campaign_run": "experiment_run",
            "campaign_panel": "generation_panel",
            "campaign_evaluation": "evaluation",
        }[kind]
        if actual_kind != expected_kind:
            raise ValueError("Campaign receipt stage kind changed")
        if kind == "campaign_run":
            return _training(path, parent_checkpoint_path)
        if kind == "campaign_panel":
            return _panel(path)
        from klab.evaluation.suite import verify_evaluation_index

        verify_evaluation_index(path, verification_mode="cold")
        return 0, 0, 0, 0
    if campaign_stage is not None or campaign_work_dir is not None:
        raise ValueError("direct native receipt cannot specify Campaign binding")
    if kind == "train":
        return _training(receipt_path, parent_checkpoint_path)
    if kind == "panel":
        return _panel(receipt_path)
    raise ValueError("unknown native receipt kind")
