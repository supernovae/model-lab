"""Select a fixed-slice checkpoint from already recorded validation evidence."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from klab.campaign.state import publish_immutable
from klab.canonical import canonical_digest
from klab.config.loading import load_config
from klab.evaluation.evidence import experiment_evidence
from klab.training.checkpoints import CheckpointManager
from klab.training.manifest import read_manifest, sha256_file

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SCORE = re.compile(r"fixed-slices-([0-9a-f]{64})\.json\Z")
_STRATA = ("general_prose", "explanatory_prose", "incident_response_docs")


class FixedSelectionDeclaration(BaseModel):
    """Frozen score identities; checkpoint generations resolve from verified evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    run_id: str
    runs_dir: Path
    config: Path
    config_sha256: str
    profile: Path
    profile_sha256: str
    steps: tuple[int, ...]
    final_step: int = Field(gt=0)
    target_positions: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_identity(self) -> FixedSelectionDeclaration:
        if not _RUN.fullmatch(self.run_id):
            raise ValueError("invalid fixed-selection run ID")
        if any(
            not _SHA.fullmatch(value)
            for value in (self.config_sha256, self.profile_sha256)
        ):
            raise ValueError("invalid fixed-selection digest")
        if any(
            not path.is_absolute() or path.is_symlink()
            for path in (self.runs_dir, self.config, self.profile)
        ):
            raise ValueError("fixed-selection paths must be absolute and nonsymlink")
        if (
            len(self.steps) != 11
            or self.steps[0] != 0
            or self.steps[-1] != self.final_step
            or tuple(sorted(set(self.steps))) != self.steps
        ):
            raise ValueError(
                "fixed selection requires eleven unique ordered checkpoints"
            )
        return self


def _load_declaration(path: Path) -> FixedSelectionDeclaration:
    if path.is_symlink() or not path.is_file():
        raise ValueError("fixed-selection declaration must be a regular file")
    return FixedSelectionDeclaration.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _score_files(run: Path, *, before_test: bool) -> list[Path]:
    root = run / "evaluations"
    paths = sorted(root.glob("fixed-slices-*.json"))
    if before_test and len(paths) != 11:
        raise ValueError("selection must precede test/utility/continuation output")
    return paths


def _validated_scores(
    run: Path,
    declaration: FixedSelectionDeclaration,
    checkpoints: dict[int, dict[str, Any]],
    expected_items: dict[str, str],
    expected_config: dict[str, Any],
    expected_source_identity: str,
    expected_tokenizer_sha256: str,
    *,
    before_test: bool,
) -> list[dict[str, Any]]:
    scores: dict[int, dict[str, Any]] = {}
    for path in _score_files(run, before_test=before_test):
        match = _SCORE.fullmatch(path.name)
        if path.is_symlink() or match is None or sha256_file(path) != match.group(1):
            raise ValueError("fixed-selection score file identity mismatch")
        score = json.loads(path.read_text(encoding="utf-8"))
        if score.get("mode") != "validation":
            if before_test:
                raise ValueError(
                    "nonvalidation fixed score predates checkpoint selection"
                )
            continue
        if (
            score.get("profile_sha256") != declaration.profile_sha256
            or score.get("forward_input_positions") != 3084
            or score.get("scored_targets") != 3072
        ):
            raise ValueError("fixed validation profile or target coverage mismatch")
        identity = score.get("identity")
        if (
            not isinstance(identity, dict)
            or identity.get("run_id") != declaration.run_id
        ):
            raise ValueError("fixed validation run identity mismatch")
        step = identity.get("step")
        if type(step) is not int or step not in checkpoints or step in scores:
            raise ValueError("duplicate or undeclared fixed validation checkpoint")
        checkpoint = checkpoints[step]
        if (
            identity.get("checkpoint_sha256") != checkpoint["digest"]
            or identity.get("checkpoint_relative_path")
            != "checkpoints/" + checkpoint["path"]
            or identity.get("tokens_seen") != checkpoint["tokens_seen"]
            or identity.get("config") != expected_config
            or identity.get("source_identity_sha256") != expected_source_identity
            or identity.get("tokenizer_sha256") != expected_tokenizer_sha256
        ):
            raise ValueError("fixed validation checkpoint identity mismatch")
        items = score.get("items")
        if not isinstance(items, list) or len(items) != 12:
            raise ValueError("fixed validation must cover twelve windows")
        seen: set[str] = set()
        total_nll = 0.0
        by_stratum: Counter[str] = Counter()
        for item in items:
            if not isinstance(item, dict):
                raise TypeError("invalid fixed validation item")
            item_id = item.get("id")
            stratum = item.get("stratum")
            nll = item.get("nll_nats")
            if (
                not isinstance(item_id, str)
                or item_id not in expected_items
                or item_id in seen
                or stratum != expected_items[item_id]
                or item.get("targets") != 256
                or type(nll) not in (int, float)
                or not math.isfinite(nll)
                or nll < 0
            ):
                raise ValueError("fixed validation item identity or loss mismatch")
            seen.add(item_id)
            by_stratum[stratum] += 1
            total_nll += float(nll)
        if seen != expected_items.keys() or by_stratum != Counter(
            {name: 4 for name in _STRATA}
        ):
            raise ValueError("fixed validation omitted or substituted a window")
        loss = score.get("loss")
        if (
            type(loss) not in (int, float)
            or not math.isfinite(loss)
            or not math.isclose(loss, total_nll / 3072, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError("fixed validation aggregate loss is not token weighted")
        scores[step] = {
            "step": step,
            "checkpoint": checkpoint["path"],
            "checkpoint_sha256": checkpoint["digest"],
            "loss": float(loss),
            "score_path": str(path),
            "score_sha256": match.group(1),
        }
    if set(scores) != set(declaration.steps):
        raise ValueError("fixed validation checkpoint set incomplete")
    return [scores[step] for step in declaration.steps]


def _selection_payload(declaration_path: Path, *, before_test: bool) -> dict[str, Any]:
    declaration = _load_declaration(declaration_path)
    if (
        sha256_file(declaration.config) != declaration.config_sha256
        or sha256_file(declaration.profile) != declaration.profile_sha256
    ):
        raise ValueError("fixed-selection config or profile bytes changed")
    profile = json.loads(declaration.profile.read_text(encoding="utf-8"))
    validation_items = [
        row for row in profile["loss_slices"] if row["split"] == "validation"
    ]
    expected_items = {row["id"]: row["stratum"] for row in validation_items}
    if (
        len(validation_items) != 12
        or len(expected_items) != 12
        or Counter(expected_items.values()) != Counter({name: 4 for name in _STRATA})
    ):
        raise ValueError("fixed-selection profile validation coverage mismatch")
    run = declaration.runs_dir / declaration.run_id
    if run.is_symlink() or not run.is_dir():
        raise ValueError("fixed-selection run is absent or symlinked")
    requested_config = load_config(declaration.config)
    effective_path = run / "resolved_config.yaml"
    effective_config = load_config(effective_path)
    manifest = read_manifest(run / "manifest.json")
    if (
        manifest.get("run_id") != declaration.run_id
        or manifest.get("requested_config") != requested_config.model_dump(mode="json")
        or manifest.get("effective_config") != effective_config.model_dump(mode="json")
    ):
        raise ValueError("fixed-selection run/config identity mismatch")
    progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
    if (
        progress.get("status") != "completed"
        or progress.get("step") != declaration.final_step
        or progress.get("tokens_seen") != declaration.target_positions
        or progress.get("parent_run_id") is not None
    ):
        raise ValueError("fixed-selection fresh run is incomplete")
    evidence = experiment_evidence(run)
    rows = evidence.get("checkpoints")
    observations = evidence.get("quality_observations")
    if (
        evidence.get("run_id") != declaration.run_id
        or not evidence.get("verified_checkpoints")
        or evidence.get("missing_reports")
        or evidence.get("rejected_reports")
        or not isinstance(rows, list)
        or not isinstance(observations, list)
        or len(rows) != 11
        or len(observations) != 11
    ):
        raise ValueError("native checkpoint/validation evidence incomplete")
    checkpoints = {row["step"]: row for row in rows}
    by_operational = {row["step"]: row for row in observations}
    if (
        len(checkpoints) != 11
        or len(by_operational) != 11
        or set(checkpoints) != set(declaration.steps)
        or set(by_operational) != set(declaration.steps)
    ):
        raise ValueError("native checkpoint/validation cadence mismatch")
    for step in declaration.steps:
        row = checkpoints[step]
        op = by_operational[step]
        if (
            not row.get("verified")
            or row.get("errors")
            or op.get("checkpoint") != row["path"]
            or op.get("checkpoint_sha256") != row["digest"]
            or op.get("batches") != 1
            or op.get("max_batches") != 1
            or type(op.get("loss")) not in (int, float)
            or not math.isfinite(op["loss"])
        ):
            raise ValueError("native checkpoint or one-batch validation invalid")
    scores = _validated_scores(
        run,
        declaration,
        checkpoints,
        expected_items,
        effective_config.model_dump(mode="json"),
        manifest["source_identity"]["sha256"],
        profile["tokenizer_sha256"],
        before_test=before_test,
    )
    selected = min(scores, key=lambda row: (row["loss"], row["step"]))
    manifest_sha = canonical_digest(manifest)
    report = CheckpointManager(run, manifest_sha256=manifest_sha).verify(
        run / "checkpoints" / selected["checkpoint"],
        expected_manifest=manifest_sha,
        require_training_state=True,
        expected_config=effective_config,
    )
    if not report.valid or report.resume_level != "full":
        raise ValueError("selected checkpoint lacks verified full state")
    return {
        "format": "klab-fixed-validation-selection-v1",
        "declaration_sha256": sha256_file(declaration_path),
        "run_id": declaration.run_id,
        "run_manifest_sha256": sha256_file(run / "manifest.json"),
        "progress_sha256": sha256_file(run / "progress.json"),
        "config_sha256": declaration.config_sha256,
        "effective_config_sha256": sha256_file(effective_path),
        "profile_sha256": declaration.profile_sha256,
        "validation_steps": list(declaration.steps),
        "scores": scores,
        "selected_step": selected["step"],
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_fixed_validation_loss": selected["loss"],
        "selection_rule": "minimum finite token-weighted fixed validation loss; earliest step on tie",
    }


def select_fixed_validation(declaration: Path, output: Path) -> dict[str, Any]:
    """Publish once, refusing a run with prior test or prose score files."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"fixed selection output already exists: {output}")
    payload = _selection_payload(declaration, before_test=True)
    digest = canonical_digest(payload)
    receipt = {**payload, "sha256": digest}
    publish_immutable(output, receipt)
    return receipt


def verify_fixed_selection(declaration: Path, receipt_path: Path) -> dict[str, Any]:
    """Cold-reconcile a stored selection against the same run and score bytes."""
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("fixed-selection receipt is missing or symlinked")
    saved = json.loads(receipt_path.read_text(encoding="utf-8"))
    actual = _selection_payload(declaration, before_test=False)
    digest = canonical_digest(actual)
    if saved != {**actual, "sha256": digest}:
        raise ValueError("fixed-selection receipt differs from verified evidence")
    return saved
