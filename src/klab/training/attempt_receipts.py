"""Authenticate native terminal counters before reconciling an attempt contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from klab.canonical import canonical_digest, canonical_json
from klab.training.manifest import read_manifest

NativeReceiptKind = Literal["train", "none"]


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


def verify_native_phase_counters(
    kind: NativeReceiptKind,
    receipt_path: Path | None,
    *,
    parent_checkpoint_path: Path | None = None,
) -> tuple[int, int, int, int]:
    """Return (updates, nonmasked targets, calls, completion IDs), or fail closed."""
    if kind == "none":
        if receipt_path is not None or parent_checkpoint_path is not None:
            raise ValueError("counter-free phase must not claim a native receipt")
        return 0, 0, 0, 0
    if receipt_path is None or not receipt_path.is_absolute():
        raise ValueError("absolute native receipt path required")
    if kind == "train":
        return _training(receipt_path, parent_checkpoint_path)
    raise ValueError("unknown native receipt kind")
