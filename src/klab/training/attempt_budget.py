"""Persistent, conservative update and wall-time limits for bounded test attempts.

Reservations charge the full declared maximum before work starts. Failed or
interrupted attempts keep that charge, so retries cannot regain uncertain updates.
This does not replace native run counters or checkpoint verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from klab.training.attempt_commands import AttemptCommand, classify_attempt_command

if TYPE_CHECKING:
    from klab.operational_monitor import MonitorPolicy, WorkspaceBaseline

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class AttemptContract(BaseModel):
    """Canonical public v1 contract; durable ledger schema remains version 2."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    contract_version: Literal[1]
    max_optimizer_updates: int = Field(strict=True, ge=0)
    max_actual_target_positions: int = Field(strict=True, ge=0)
    max_generation_calls: int = Field(strict=True, ge=0)
    max_generated_tokens: int = Field(strict=True, ge=0)
    max_fixed_profile_forward_positions: int | None = Field(
        default=None, strict=True, ge=0, exclude_if=lambda value: value is None
    )
    max_nontraining_forward_positions: int | None = Field(
        default=None, strict=True, ge=0, exclude_if=lambda value: value is None
    )
    max_operational_validation_batches: int | None = Field(
        default=None, strict=True, ge=0, exclude_if=lambda value: value is None
    )
    max_operational_validation_forward_positions: int | None = Field(
        default=None, strict=True, ge=0, exclude_if=lambda value: value is None
    )
    max_wall_seconds: float = Field(gt=0)
    content_identity_sha256: str
    monitor_policy_sha256: str
    workspace_baseline_sha256: str
    acquisition_project_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    preparation_acquisition_identity_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    preparation_normalizer: (
        Literal["normalizer-structure-v2", "normalizer-structure-v3"] | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    admission_lock_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    admission_policy_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    admission_selection_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    fixed_profile_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    fixed_family_inventory_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    preparation_monitor_policy_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    evaluation_monitor_policy_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    require_resolved_train_config_binding: bool = Field(
        default=False, exclude_if=lambda value: value is False
    )
    require_evaluation_baseline_binding: bool = Field(
        default=False, exclude_if=lambda value: value is False
    )
    require_release_acceptance_binding: bool = Field(
        default=False, exclude_if=lambda value: value is False
    )
    require_admission_inspection_binding: bool = Field(
        default=False, exclude_if=lambda value: value is False
    )
    require_preledger_monitor_binding: bool = Field(
        default=False, exclude_if=lambda value: value is False
    )
    offline_retained_sources_only: bool = Field(
        default=False, exclude_if=lambda value: value is False
    )
    preparation_only: bool = Field(
        default=False, exclude_if=lambda value: value is False
    )

    @field_validator(
        "content_identity_sha256",
        "monitor_policy_sha256",
        "workspace_baseline_sha256",
    )
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("contract identities must be lowercase SHA-256 digests")
        return value

    @field_validator(
        "preparation_monitor_policy_sha256",
        "evaluation_monitor_policy_sha256",
        "acquisition_project_sha256",
        "preparation_acquisition_identity_sha256",
        "admission_lock_sha256",
        "admission_policy_sha256",
        "admission_selection_sha256",
        "fixed_profile_sha256",
        "fixed_family_inventory_sha256",
    )
    @classmethod
    def valid_optional_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("monitor identities must be lowercase SHA-256 digests")
        return value

    @model_validator(mode="after")
    def complete_forward_limits(self) -> AttemptContract:
        if (self.preparation_acquisition_identity_sha256 is None) != (
            self.preparation_normalizer is None
        ):
            raise ValueError("preparation identity and normalizer must be paired")
        if self.require_admission_inspection_binding and any(
            value is None
            for value in (
                self.admission_lock_sha256,
                self.admission_policy_sha256,
                self.admission_selection_sha256,
                self.preparation_normalizer,
            )
        ):
            raise ValueError(
                "inspected admission requires lock, policy, selection and normalizer identities"
            )
        fixed = self.max_fixed_profile_forward_positions
        total = self.max_nontraining_forward_positions
        if (fixed is None) != (total is None):
            raise ValueError("both forward-input limits must be declared together")
        if fixed is not None and total is not None and fixed > total:
            raise ValueError("fixed-profile forward-input limit exceeds total")
        batches = self.max_operational_validation_batches
        validation = self.max_operational_validation_forward_positions
        if (batches is None) != (validation is None):
            raise ValueError(
                "both operational validation limits must be declared together"
            )
        if validation is not None and (total is None or validation > total):
            raise ValueError(
                "operational validation limit exceeds forward-input allocation"
            )
        return self


def load_attempt_contract(
    path: Path, expected_sha256: str | None = None
) -> AttemptContract:
    if not path.is_absolute() or path.is_symlink():
        raise AttemptBudgetError("contract path must be absolute and not a symlink")
    try:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise AttemptBudgetError("attempt contract digest mismatch")
        return AttemptContract.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise AttemptBudgetError(f"invalid attempt contract: {error}") from error


def validate_attempt_paths(
    command: AttemptCommand,
    *,
    ledger: Path,
    completion: Path,
    monitor_log_dir: Path,
    monitor_policy: Path,
    baseline: Path,
) -> None:
    """Reject output aliases before any attempt reservation or child launch.

    Resolving existing ancestors catches symlink aliases, while the existing
    dispatch checks continue to enforce their own path and write rules.
    """
    outputs = {
        "outer completion": completion,
        "outer ready": completion.with_name(completion.name + ".ready"),
        "outer attempt receipt": completion.with_name(
            completion.name + ".attempt.json"
        ),
        "outer monitor tree": monitor_log_dir,
    }
    if "--output" in command.options:
        outputs["leaf output"] = Path(str(command.options["--output"]))
    if command.operation == "tokenizer train":
        from klab.config.loading import load_tokenizer_config

        tokenizer_config = load_tokenizer_config(Path(command.args[2]))
        outputs["tokenizer output tree"] = tokenizer_config.output_dir
    if command.operation == "train" and "--runs-dir" in command.options:
        outputs["training runs tree"] = Path(str(command.options["--runs-dir"]))
    inputs = {
        "attempt ledger": ledger,
        "attempt ledger WAL": ledger.with_name(ledger.name + "-wal"),
        "attempt ledger SHM": ledger.with_name(ledger.name + "-shm"),
        "workspace baseline": baseline,
        "whole monitor policy": monitor_policy,
    }
    if command.monitor is not None:
        outputs["inner monitor tree"] = Path(command.monitor["--log-dir"])
        inputs["inner monitor policy"] = Path(command.monitor["--policy"])
        inputs["inner monitor baseline"] = Path(command.monitor["--baseline"])
    input_options = {
        "--template",
        "--workspace-baseline",
        "--policy-document",
        "--draft",
        "--selection",
        "--inventory",
        "--clusters",
        "--splits",
        "--prior-release",
        "--candidate-release",
        "--prior-inventory",
        "--candidate-inventory",
        "--profile",
        "--suite",
        "--tokenizer",
        "--tokenizer-origin-release",
        "--family-inventory",
        "--policy",
        "--resource-envelope",
        "--prepared-inputs",
        "--stage-bundle",
        "--release",
        "--tokenizer-config",
        "--checkpoint",
        "--config",
        "--base-run-config",
    }
    for option in input_options & command.options.keys():
        inputs[f"leaf {option}"] = Path(str(command.options[option]))
    if command.operation not in {"monitor-baseline", "workspace preflight"}:
        first = len(command.operation.split())
        if len(command.args) > first:
            inputs["leaf positional input"] = Path(command.args[first])
        if command.operation in {
            "data prepared-inputs verify",
            "evaluation fixed-slices verify-selection",
            "evaluation fixed-slices score",
            "evaluation fixed-slices continuations",
        }:
            inputs["leaf second positional input"] = Path(command.args[first + 1])
    canonical_outputs = {name: path.resolve() for name, path in outputs.items()}
    canonical_inputs = {name: path.resolve() for name, path in inputs.items()}
    for name, path in canonical_outputs.items():
        for other_name, other in canonical_outputs.items():
            if name >= other_name:
                continue
            if path == other or path in other.parents or other in path.parents:
                raise AttemptBudgetError(
                    f"attempt output path collision: {name} / {other_name}"
                )
        for input_name, input_path in canonical_inputs.items():
            if (
                path == input_path
                or path in input_path.parents
                or input_path in path.parents
            ):
                raise AttemptBudgetError(
                    f"attempt output aliases protected input: {name} / {input_name}"
                )


class AttemptBudgetError(RuntimeError):
    """The shared attempt budget is missing, invalid, or exhausted."""


def _verify_admission_contract_binding(
    contract: AttemptContract, review: dict[str, object]
) -> None:
    if not contract.require_admission_inspection_binding:
        return
    if review.get("format") != "klab-admission-review-v2":
        raise AttemptBudgetError("preparation requires inspected admission")
    inspection = json.loads(
        Path(str(review["inspection_path"])).read_text(encoding="utf-8")
    )
    if (
        inspection.get("acquisition_lock_sha256") != contract.admission_lock_sha256
        or inspection.get("policy_sha256") != contract.admission_policy_sha256
        or inspection.get("selection_sha256") != contract.admission_selection_sha256
        or inspection.get("normalizer") != contract.preparation_normalizer
    ):
        raise AttemptBudgetError("inspected admission differs from contract identities")


def validate_contract_monitor_inputs(
    contract: AttemptContract,
    *,
    monitor_policy_path: Path,
    workspace_baseline_path: Path,
    workspace_root: Path,
) -> tuple[WorkspaceBaseline, MonitorPolicy]:
    """Cold-check one contract against its actual monitor policy and baseline."""
    from klab.operational_monitor import (
        load_monitor_policy,
        load_workspace_baseline,
    )

    if any(
        not path.is_absolute() or path.is_symlink()
        for path in (monitor_policy_path, workspace_baseline_path, workspace_root)
    ):
        raise AttemptBudgetError("monitor binding paths must be absolute and direct")
    try:
        if (
            hashlib.sha256(monitor_policy_path.read_bytes()).hexdigest()
            != contract.monitor_policy_sha256
        ):
            raise AttemptBudgetError("monitor policy identity differs from contract")
        baseline = load_workspace_baseline(workspace_baseline_path, workspace_root)
        policy = load_monitor_policy(monitor_policy_path)
        if (
            hashlib.sha256(monitor_policy_path.read_bytes()).hexdigest()
            != contract.monitor_policy_sha256
        ):
            raise AttemptBudgetError("monitor policy changed during validation")
        if baseline.sha256 != contract.workspace_baseline_sha256:
            raise AttemptBudgetError(
                "workspace baseline identity differs from contract"
            )
        return baseline, policy
    except OSError as error:
        raise AttemptBudgetError(f"contract input unavailable: {error}") from error


class AttemptBudget:
    def __init__(self, path: Path) -> None:
        if not path.is_absolute() or path.is_symlink():
            raise AttemptBudgetError("budget path must be absolute and not a symlink")
        self.path = path

    def bind_resolved_artifact(
        self,
        *,
        kind: Literal[
            "train_config",
            "evaluation_baseline",
            "pre_freeze_project",
            "build_project",
            "release_acceptance",
        ],
        path: Path,
        expected_sha256: str,
        content_identity_sha256: str,
        workspace_root: Path,
    ) -> dict[str, str]:
        """Seal one late-produced input without changing the original contract."""
        if kind not in {
            "train_config",
            "evaluation_baseline",
            "pre_freeze_project",
            "build_project",
            "release_acceptance",
        }:
            raise AttemptBudgetError("unsupported resolved artifact kind")
        if not _SHA256.fullmatch(expected_sha256):
            raise AttemptBudgetError("resolved artifact SHA-256 is invalid")
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise AttemptBudgetError(
                "resolved artifact must be an existing absolute file"
            )
        root = workspace_root.resolve(strict=True)
        if root != path.parent.resolve() and root not in path.parent.resolve().parents:
            raise AttemptBudgetError("resolved artifact is outside attempt workspace")
        self.remaining_seconds()
        pre_freeze_binding = (
            self._resolved_artifact("pre_freeze_project", workspace_root=root)
            if kind == "build_project"
            else None
        )
        build_binding = (
            self._resolved_artifact("build_project", workspace_root=root)
            if kind == "release_acceptance"
            else None
        )
        accepted_binding = (
            self._resolved_artifact("release_acceptance", workspace_root=root)
            if kind == "train_config"
            else None
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            contract, _, _, _ = self._check_v2(connection)
            if content_identity_sha256 != contract.content_identity_sha256:
                raise AttemptBudgetError("resolved artifact contract identity mismatch")
            if kind in {"pre_freeze_project", "build_project"}:
                from klab.corpus.acquisition import _project_sha
                from klab.corpus.project import load_project, project_path
                from klab.corpus.release_review import verify_admission_review
                from klab.training.manifest import sha256_file

                project = load_project(path)
                if (
                    contract.preparation_acquisition_identity_sha256 is None
                    or _project_sha(project)
                    != contract.preparation_acquisition_identity_sha256
                    or project.release.schema_version != 2
                    or project.release.record_admission is None
                    or project.release.normalizer != contract.preparation_normalizer
                    or sha256_file(
                        project_path(
                            project.root, project.release.record_admission.path
                        )
                    )
                    != project.release.record_admission.sha256
                ):
                    raise AttemptBudgetError("preparation project identity differs")
                if (
                    contract.require_release_acceptance_binding
                    or contract.require_admission_inspection_binding
                ):
                    admission_review = verify_admission_review(
                        project_path(
                            project.root, project.release.record_admission.path
                        ),
                        root,
                    )
                    _verify_admission_contract_binding(contract, admission_review)
                if kind == "build_project":
                    if pre_freeze_binding is None:
                        raise AttemptBudgetError("pre-freeze project is not bound")
                    pre_freeze = load_project(pre_freeze_binding[0])
                    if (
                        project.release != pre_freeze.release
                        or project.splits == pre_freeze.splits
                    ):
                        raise AttemptBudgetError(
                            "build project differs from reviewed pre-freeze binding"
                        )
                    splits_path = project_path(project.root, project.config.splits)
                    receipt_path = splits_path.with_name(
                        splits_path.stem + ".receipt.json"
                    )
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if (
                        receipt.get("split_sha256") != sha256_file(splits_path)
                        or receipt.get("admission_sha256")
                        != project.release.record_admission.sha256
                    ):
                        raise AttemptBudgetError(
                            "build project lacks matching frozen family receipt"
                        )
            elif kind == "release_acceptance":
                from klab.corpus.release_review import verify_release_review

                if build_binding is None:
                    raise AttemptBudgetError("build project is not bound")
                verify_release_review(path, root, build_project=build_binding[0])
            elif kind == "train_config":
                from klab.config.loading import load_config

                load_config(path)
                if contract.require_release_acceptance_binding:
                    from klab.corpus.release_review import (
                        verify_accepted_prepared_config,
                    )

                    if accepted_binding is None:
                        raise AttemptBudgetError("exact release has not been accepted")
                    verify_accepted_prepared_config(
                        path, json.loads(accepted_binding[0].read_text())
                    )
            else:
                from klab.operational_monitor import load_workspace_baseline

                load_workspace_baseline(path, root)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected_sha256:
                raise AttemptBudgetError("resolved artifact bytes differ")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS resolved_artifacts ("
                "kind TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL, "
                "content_identity_sha256 TEXT NOT NULL, bound_ns INTEGER NOT NULL)"
            )
            try:
                connection.execute(
                    "INSERT INTO resolved_artifacts VALUES (?, ?, ?, ?, ?)",
                    (kind, str(path), actual, content_identity_sha256, time.time_ns()),
                )
            except sqlite3.IntegrityError as error:
                raise AttemptBudgetError(
                    "resolved artifact is already bound"
                ) from error
        return {"kind": kind, "path": str(path), "sha256": actual}

    def _resolved_artifact(
        self, kind: str, *, workspace_root: Path
    ) -> tuple[Path, str] | None:
        with self._connect() as connection:
            contract, _, _, _ = self._check_v2(connection)
            present = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='resolved_artifacts'"
            ).fetchone()
            if not present:
                return None
            row = connection.execute(
                "SELECT path, sha256, content_identity_sha256 FROM resolved_artifacts "
                "WHERE kind=?",
                (kind,),
            ).fetchone()
        if row is None:
            return None
        path, digest, identity = row
        root = workspace_root.resolve(strict=True)
        if (
            identity != contract.content_identity_sha256
            or not isinstance(path, str)
            or not isinstance(digest, str)
            or not Path(path).is_absolute()
            or Path(path).is_symlink()
            or (
                root != Path(path).parent.resolve()
                and root not in Path(path).parent.resolve().parents
            )
            or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
        ):
            raise AttemptBudgetError("resolved artifact identity changed")
        if kind == "evaluation_baseline":
            from klab.operational_monitor import load_workspace_baseline

            load_workspace_baseline(Path(path), root)
        if kind in {"pre_freeze_project", "build_project"}:
            from klab.corpus.acquisition import _project_sha
            from klab.corpus.project import load_project, project_path
            from klab.corpus.release_review import verify_admission_review
            from klab.training.manifest import sha256_file

            project = load_project(Path(path))
            reference = project.release.record_admission
            if (
                _project_sha(project)
                != contract.preparation_acquisition_identity_sha256
                or project.release.schema_version != 2
                or project.release.normalizer != contract.preparation_normalizer
                or reference is None
                or sha256_file(project_path(project.root, reference.path))
                != reference.sha256
            ):
                raise AttemptBudgetError("bound preparation project identity changed")
            if (
                contract.require_release_acceptance_binding
                or contract.require_admission_inspection_binding
            ):
                admission_review = verify_admission_review(
                    project_path(project.root, reference.path), root
                )
                _verify_admission_contract_binding(contract, admission_review)
        if kind == "release_acceptance":
            from klab.corpus.release_review import verify_release_review

            build_binding = self._resolved_artifact(
                "build_project", workspace_root=root
            )
            if build_binding is None:
                raise AttemptBudgetError("build project is not bound")
            verify_release_review(Path(path), root, build_project=build_binding[0])
        return Path(path), digest

    @classmethod
    def create_contract(
        cls,
        path: Path,
        *,
        contract_path: Path,
        expected_sha256: str,
        monitor_policy_path: Path | None = None,
        workspace_baseline_path: Path | None = None,
        workspace_root: Path | None = None,
    ) -> AttemptBudget:
        """Create an exclusive v2 ledger from exact authenticated contract bytes."""
        contract = load_attempt_contract(contract_path, expected_sha256)
        provided = (
            monitor_policy_path,
            workspace_baseline_path,
            workspace_root,
        )
        if any(value is not None for value in provided):
            if not all(value is not None for value in provided):
                raise AttemptBudgetError("incomplete pre-ledger monitor binding")
            assert monitor_policy_path is not None
            assert workspace_baseline_path is not None
            assert workspace_root is not None
            validate_contract_monitor_inputs(
                contract,
                monitor_policy_path=monitor_policy_path,
                workspace_baseline_path=workspace_baseline_path,
                workspace_root=workspace_root,
            )
        elif contract.require_preledger_monitor_binding:
            raise AttemptBudgetError("pre-ledger monitor binding is required")
        raw = contract_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise AttemptBudgetError("attempt contract changed during initialization")
        try:
            raw_text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AttemptBudgetError("attempt contract is not UTF-8") from error
        budget = cls(path)
        if not path.parent.is_dir():
            raise AttemptBudgetError("budget parent directory must already exist")
        started_ns = time.time_ns()
        deadline_ns = started_ns + int(contract.max_wall_seconds * 1_000_000_000)
        if deadline_ns <= started_ns or deadline_ns > 2**63 - 1:
            raise ValueError("max_wall_seconds is outside the ledger clock range")
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(descriptor)
        with budget._connect() as connection:
            connection.execute(
                "CREATE TABLE budget ("
                "id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL, "
                "max_updates INTEGER NOT NULL, used_updates INTEGER NOT NULL, "
                "started_ns INTEGER NOT NULL, deadline_ns INTEGER NOT NULL, "
                "last_checked_ns INTEGER NOT NULL, "
                "max_targets INTEGER NOT NULL, used_targets INTEGER NOT NULL, "
                "max_calls INTEGER NOT NULL, used_calls INTEGER NOT NULL, "
                "max_tokens INTEGER NOT NULL, used_tokens INTEGER NOT NULL, "
                "contract_path TEXT NOT NULL, contract_sha256 TEXT NOT NULL, "
                "contract_json TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE reservations ("
                "id INTEGER PRIMARY KEY, label TEXT NOT NULL UNIQUE, "
                "updates INTEGER NOT NULL, target_positions INTEGER NOT NULL, "
                "generation_calls INTEGER NOT NULL, generated_tokens INTEGER NOT NULL, "
                "content_identity_sha256 TEXT NOT NULL, reserved_ns INTEGER NOT NULL, "
                "actual_updates INTEGER, actual_targets INTEGER, actual_calls INTEGER, "
                "actual_tokens INTEGER, completed_ns INTEGER)"
            )
            connection.execute(
                "INSERT INTO budget VALUES (1, 2, ?, 0, ?, ?, ?, ?, 0, ?, 0, ?, 0, ?, ?, ?)",
                (
                    contract.max_optimizer_updates,
                    started_ns,
                    deadline_ns,
                    started_ns,
                    contract.max_actual_target_positions,
                    contract.max_generation_calls,
                    contract.max_generated_tokens,
                    str(contract_path),
                    expected_sha256,
                    raw_text,
                ),
            )
            if contract.max_nontraining_forward_positions is not None:
                connection.execute(
                    "CREATE TABLE forward_reservations ("
                    "id INTEGER PRIMARY KEY, label TEXT NOT NULL UNIQUE, "
                    "kind TEXT NOT NULL, positions INTEGER NOT NULL, "
                    "content_identity_sha256 TEXT NOT NULL, reserved_ns INTEGER NOT NULL)"
                )
        return budget

    @staticmethod
    def _forward_state(
        connection: sqlite3.Connection, contract: AttemptContract
    ) -> tuple[int, int, int, int] | None:
        """Verify the optional append-only counters, including old-ledger absence."""
        try:
            present = (
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                    "AND name = 'forward_reservations'"
                ).fetchone()
                is not None
            )
            enabled = contract.max_nontraining_forward_positions is not None
            if present != enabled:
                raise AttemptBudgetError(
                    "forward-input ledger schema disagrees with contract"
                )
            if not enabled:
                return None
            rows = connection.execute(
                "SELECT label, kind, positions, content_identity_sha256, reserved_ns "
                "FROM forward_reservations"
            ).fetchall()
        except sqlite3.DatabaseError as error:
            raise AttemptBudgetError("invalid forward-input ledger") from error
        if any(
            not isinstance(label, str)
            or not label.strip()
            or kind not in {"fixed_profile", "operational_validation", "generation"}
            or type(positions) is not int
            or positions <= 0
            or identity != contract.content_identity_sha256
            or type(reserved_ns) is not int
            or reserved_ns <= 0
            for label, kind, positions, identity, reserved_ns in rows
        ) or len({row[0] for row in rows}) != len(rows):
            raise AttemptBudgetError("invalid forward-input reservations")
        fixed = sum(row[2] for row in rows if row[1] == "fixed_profile")
        total = sum(row[2] for row in rows)
        validation_rows = [row for row in rows if row[1] == "operational_validation"]
        validation = sum(row[2] for row in validation_rows)
        if (
            contract.max_fixed_profile_forward_positions is None
            or fixed > contract.max_fixed_profile_forward_positions
            or total > contract.max_nontraining_forward_positions
            or (
                contract.max_operational_validation_batches is not None
                and len(validation_rows) > contract.max_operational_validation_batches
            )
            or (
                contract.max_operational_validation_forward_positions is not None
                and validation > contract.max_operational_validation_forward_positions
            )
        ):
            raise AttemptBudgetError("forward-input reservation total exceeds contract")
        return fixed, total, len(validation_rows), validation

    @staticmethod
    def _version(connection: sqlite3.Connection) -> int:
        try:
            row = connection.execute(
                "SELECT version FROM budget WHERE id = 1"
            ).fetchone()
        except sqlite3.DatabaseError as error:
            raise AttemptBudgetError("invalid budget ledger") from error
        if row is None or row[0] != 2:
            raise AttemptBudgetError("unsupported budget ledger version")
        return row[0]

    @staticmethod
    def _row_v2(
        connection: sqlite3.Connection,
    ) -> tuple[
        AttemptContract, tuple[int, int, int, int], tuple[int, int, int, int], int, int
    ]:
        try:
            row = connection.execute(
                "SELECT version, max_updates, used_updates, started_ns, deadline_ns, last_checked_ns, "
                "max_targets, used_targets, max_calls, used_calls, max_tokens, used_tokens, "
                "contract_path, contract_sha256, contract_json FROM budget WHERE id = 1"
            ).fetchone()
            charges = connection.execute(
                "SELECT COALESCE(SUM(updates), 0), COALESCE(SUM(target_positions), 0), "
                "COALESCE(SUM(generation_calls), 0), COALESCE(SUM(generated_tokens), 0) "
                "FROM reservations"
            ).fetchone()
            reservation_rows = connection.execute(
                "SELECT updates, target_positions, generation_calls, generated_tokens, "
                "content_identity_sha256, actual_updates, actual_targets, actual_calls, "
                "actual_tokens, completed_ns FROM reservations"
            ).fetchall()
        except sqlite3.DatabaseError as error:
            raise AttemptBudgetError("invalid budget ledger") from error
        if (
            row is None
            or len(row) != 15
            or any(type(row[i]) is not int for i in range(12))
            or row[0] != 2
            or not all(isinstance(row[i], str) for i in (12, 13, 14))
            or row[4] <= row[5]
        ):
            raise AttemptBudgetError("invalid or expired v2 budget ledger")
        maximum = (row[1], row[6], row[8], row[10])
        used = (row[2], row[7], row[9], row[11])
        if any(
            limit < 0 or not 0 <= charged <= limit
            for limit, charged in zip(maximum, used)
        ):
            raise AttemptBudgetError("invalid v2 budget counters")
        try:
            raw = row[14].encode("utf-8")
            if hashlib.sha256(raw).hexdigest() != row[13]:
                raise AttemptBudgetError("stored attempt contract digest mismatch")
            contract = AttemptContract.model_validate_json(raw)
            if load_attempt_contract(Path(row[12]), row[13]) != contract:
                raise AttemptBudgetError("attempt contract content changed")
        except (UnicodeError, ValidationError) as error:
            raise AttemptBudgetError("invalid stored attempt contract") from error
        if maximum != (
            contract.max_optimizer_updates,
            contract.max_actual_target_positions,
            contract.max_generation_calls,
            contract.max_generated_tokens,
        ):
            raise AttemptBudgetError("contract limits disagree with ledger")
        if row[4] - row[3] != int(contract.max_wall_seconds * 1_000_000_000):
            raise AttemptBudgetError("contract deadline disagrees with ledger")
        if charges != used or any(
            any(type(value) is not int or value < 0 for value in reservation[:4])
            or reservation[4] != contract.content_identity_sha256
            or (
                any(value is not None for value in reservation[5:9])
                and (
                    any(type(value) is not int for value in reservation[5:9])
                    or any(
                        not 0 <= actual <= reserved
                        for actual, reserved in zip(reservation[5:9], reservation[:4])
                    )
                    or type(reservation[9]) is not int
                )
            )
            or (
                all(value is None for value in reservation[5:9])
                and reservation[9] is not None
            )
            for reservation in reservation_rows
        ):
            raise AttemptBudgetError(
                "reservation total or identity disagrees with ledger"
            )
        AttemptBudget._forward_state(connection, contract)
        return contract, maximum, used, row[4], row[5]

    def _connect(self) -> sqlite3.Connection:
        if not self.path.is_file() or self.path.is_symlink():
            raise AttemptBudgetError("budget ledger is missing or is a symlink")
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _check_v2(
        self, connection: sqlite3.Connection
    ) -> tuple[
        AttemptContract, tuple[int, int, int, int], tuple[int, int, int, int], int
    ]:
        contract, maximum, used, deadline_ns, last_checked_ns = self._row_v2(connection)
        now_ns = time.time_ns()
        if now_ns < last_checked_ns:
            raise AttemptBudgetError("clock moved backwards; budget fails closed")
        if now_ns >= deadline_ns:
            raise AttemptBudgetError("shared wall-time limit reached")
        connection.execute(
            "UPDATE budget SET last_checked_ns = ? WHERE id = 1", (now_ns,)
        )
        return contract, maximum, used, deadline_ns - now_ns

    def remaining_seconds(self) -> float:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._version(connection)
            _, _, _, remaining_ns = self._check_v2(connection)
        return remaining_ns / 1_000_000_000

    def reserve_vector(
        self,
        label: str,
        *,
        updates: int = 0,
        target_positions: int = 0,
        generation_calls: int = 0,
        generated_tokens: int = 0,
        content_identity_sha256: str,
    ) -> dict[str, int]:
        if not label.strip() or any(
            type(value) is not int or value < 0
            for value in (updates, target_positions, generation_calls, generated_tokens)
        ):
            raise ValueError(
                "vector reservation requires a label and nonnegative integer counters"
            )
        requested = (updates, target_positions, generation_calls, generated_tokens)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._version(connection) != 2:
                raise AttemptBudgetError("vector reservations require a v2 ledger")
            contract, maximum, used, _ = self._check_v2(connection)
            if content_identity_sha256 != contract.content_identity_sha256:
                raise AttemptBudgetError(
                    "content identity differs from attempt contract"
                )
            if connection.execute(
                "SELECT 1 FROM reservations WHERE label = ?", (label,)
            ).fetchone():
                raise AttemptBudgetError("duplicate phase reservation label")
            for name, requested_value, used_value, maximum_value in zip(
                (
                    "optimizer updates",
                    "actual target positions",
                    "generation calls",
                    "generated tokens",
                ),
                requested,
                used,
                maximum,
            ):
                if used_value + requested_value > maximum_value:
                    raise AttemptBudgetError(
                        f"shared {name} limit: {used_value} charged + {requested_value} requested > {maximum_value} approved"
                    )
            next_used = tuple(prior + amount for prior, amount in zip(used, requested))
            connection.execute(
                "UPDATE budget SET used_updates = ?, used_targets = ?, used_calls = ?, "
                "used_tokens = ? WHERE id = 1",
                next_used,
            )
            connection.execute(
                "INSERT INTO reservations (label, updates, target_positions, "
                "generation_calls, generated_tokens, content_identity_sha256, reserved_ns) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (label, *requested, content_identity_sha256, time.time_ns()),
            )
        return dict(
            zip(
                ("updates", "target_positions", "generation_calls", "generated_tokens"),
                (limit - charged for limit, charged in zip(maximum, next_used)),
            )
        )

    def reserve_forward_positions(
        self,
        label: str,
        *,
        kind: Literal["fixed_profile", "operational_validation", "generation"],
        positions: int,
        content_identity_sha256: str,
    ) -> dict[str, int]:
        """Charge nontraining model inputs before the forward; never refund failures."""
        if (
            not label.strip()
            or kind not in {"fixed_profile", "operational_validation", "generation"}
            or type(positions) is not int
            or positions <= 0
        ):
            raise ValueError(
                "forward reservation needs a label, kind and positive positions"
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._version(connection) != 2:
                raise AttemptBudgetError("forward reservations require a v2 ledger")
            contract, _, _, _ = self._check_v2(connection)
            if content_identity_sha256 != contract.content_identity_sha256:
                raise AttemptBudgetError(
                    "content identity differs from attempt contract"
                )
            state = self._forward_state(connection, contract)
            if state is None:
                raise AttemptBudgetError(
                    "attempt contract has no forward-input allocation"
                )
            fixed, total, validation_batches, validation = state
            if connection.execute(
                "SELECT 1 FROM forward_reservations WHERE label = ?", (label,)
            ).fetchone():
                raise AttemptBudgetError("duplicate forward-input reservation label")
            next_fixed = fixed + (positions if kind == "fixed_profile" else 0)
            next_total = total + positions
            fixed_limit = contract.max_fixed_profile_forward_positions
            total_limit = contract.max_nontraining_forward_positions
            assert fixed_limit is not None and total_limit is not None
            if next_fixed > fixed_limit:
                raise AttemptBudgetError(
                    "shared fixed-profile forward-input limit reached"
                )
            if next_total > total_limit:
                raise AttemptBudgetError(
                    "shared nontraining forward-input limit reached"
                )
            if kind == "operational_validation" and (
                (
                    contract.max_operational_validation_batches is not None
                    and validation_batches + 1
                    > contract.max_operational_validation_batches
                )
                or (
                    contract.max_operational_validation_forward_positions is not None
                    and validation + positions
                    > contract.max_operational_validation_forward_positions
                )
            ):
                raise AttemptBudgetError("shared operational validation limit reached")
            connection.execute(
                "INSERT INTO forward_reservations "
                "(label, kind, positions, content_identity_sha256, reserved_ns) "
                "VALUES (?, ?, ?, ?, ?)",
                (label, kind, positions, content_identity_sha256, time.time_ns()),
            )
        return {
            "fixed_profile_forward_positions": fixed_limit - next_fixed,
            "nontraining_forward_positions": total_limit - next_total,
        }

    @staticmethod
    def forward_allocation_active_from_environment() -> bool:
        """Discover forward limits from the current attempt contract."""
        path = os.environ.get("KLAB_ATTEMPT_BUDGET_LEDGER")
        if not path:
            return False
        budget = AttemptBudget(Path(path))
        with budget._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if budget._version(connection) != 2:
                raise AttemptBudgetError(
                    "forward allocation requires a current contract ledger"
                )
            contract, _, _, _ = budget._check_v2(connection)
            return contract.max_nontraining_forward_positions is not None

    @staticmethod
    def reserve_forward_from_environment(
        label: str,
        *,
        kind: Literal["fixed_profile", "operational_validation", "generation"],
        positions: int,
    ) -> dict[str, int]:
        """Bind a child-side reservation to the native owned attempt phase."""
        path = os.environ.get("KLAB_ATTEMPT_BUDGET_LEDGER")
        identity = os.environ.get("KLAB_ATTEMPT_CONTENT_IDENTITY_SHA256")
        phase = os.environ.get("KLAB_ATTEMPT_PHASE_LABEL")
        if not path or not identity or not phase:
            raise AttemptBudgetError("forward-input reservation lacks attempt binding")
        return AttemptBudget(Path(path)).reserve_forward_positions(
            f"{phase}:{label}",
            kind=kind,
            positions=positions,
            content_identity_sha256=identity,
        )

    @staticmethod
    def reserve_generation_from_environment(
        label: str, *, requested_tokens: int, forward_positions: int
    ) -> dict[str, int]:
        """Precharge one request and its worst-case inputs in the shared ledger."""
        path = os.environ.get("KLAB_ATTEMPT_BUDGET_LEDGER")
        identity = os.environ.get("KLAB_ATTEMPT_CONTENT_IDENTITY_SHA256")
        phase = os.environ.get("KLAB_ATTEMPT_PHASE_LABEL")
        if not path or not identity or not phase:
            raise AttemptBudgetError("generation reservation lacks attempt binding")
        budget = AttemptBudget(Path(path))
        budget.reserve_vector(
            f"{phase}:{label}:request",
            generation_calls=1,
            generated_tokens=requested_tokens,
            content_identity_sha256=identity,
        )
        return budget.reserve_forward_positions(
            f"{phase}:{label}:input",
            kind="generation",
            positions=forward_positions,
            content_identity_sha256=identity,
        )

    @staticmethod
    def require_fixed_evaluation_allocation_from_environment(
        *, fixed_positions: int, total_positions: int, calls: int, tokens: int
    ) -> None:
        """Refuse contracted model loading when its durable allowance is absent."""
        keys = (
            "KLAB_ATTEMPT_BUDGET_LEDGER",
            "KLAB_ATTEMPT_CONTENT_IDENTITY_SHA256",
            "KLAB_ATTEMPT_PHASE_LABEL",
            "KLAB_ATTEMPT_ACTIVITY",
        )
        present = [bool(os.environ.get(key)) for key in keys]
        if not any(present):
            return  # Historical direct evaluation has no attempt contract.
        if not all(present) or os.environ[keys[3]] != "evaluate":
            raise AttemptBudgetError("fixed evaluation lacks an owned attempt binding")
        budget = AttemptBudget(Path(os.environ[keys[0]]))
        with budget._connect() as connection:
            contract, _, _, _ = budget._check_v2(connection)
            forward = budget._forward_state(connection, contract)
        status = budget.status()
        if (
            os.environ[keys[1]] != contract.content_identity_sha256
            or not any(
                row["label"] == os.environ[keys[2]] and row["actual_updates"] is None
                for row in status["reservations"]
            )
            or forward is None
            or contract.max_fixed_profile_forward_positions is None
            or contract.max_nontraining_forward_positions is None
            or forward[0] + fixed_positions
            > contract.max_fixed_profile_forward_positions
            or forward[1] + total_positions > contract.max_nontraining_forward_positions
            or status["charged_generation_calls"] + calls
            > contract.max_generation_calls
            or status["charged_generated_tokens"] + tokens
            > contract.max_generated_tokens
        ):
            raise AttemptBudgetError("fixed evaluation lacks declared model allocation")

    def _record_verified_actual(
        self,
        label: str,
        *,
        updates: int,
        target_positions: int,
        generation_calls: int,
        generated_tokens: int,
        content_identity_sha256: str,
    ) -> None:
        """Store counters obtained from a native verifier; never refund charges."""
        actual = (updates, target_positions, generation_calls, generated_tokens)
        if not label.strip() or any(
            type(value) is not int or value < 0 for value in actual
        ):
            raise ValueError("actual counters must be nonnegative integers")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._version(connection) != 2:
                raise AttemptBudgetError("actual counters require a v2 ledger")
            contract, _, _, _ = self._check_v2(connection)
            if content_identity_sha256 != contract.content_identity_sha256:
                raise AttemptBudgetError(
                    "content identity differs from attempt contract"
                )
            row = connection.execute(
                "SELECT updates, target_positions, generation_calls, generated_tokens, "
                "actual_updates, actual_targets, actual_calls, actual_tokens "
                "FROM reservations WHERE label = ?",
                (label,),
            ).fetchone()
            if row is None:
                raise AttemptBudgetError("phase reservation is missing")
            if any(observed > reserved for observed, reserved in zip(actual, row[:4])):
                raise AttemptBudgetError("actual counter exceeds phase reservation")
            if any(value is not None for value in row[4:]):
                if tuple(row[4:]) != actual:
                    raise AttemptBudgetError(
                        "actual counters differ from recorded result"
                    )
                return
            connection.execute(
                "UPDATE reservations SET actual_updates = ?, actual_targets = ?, "
                "actual_calls = ?, actual_tokens = ?, completed_ns = ? WHERE label = ?",
                (*actual, time.time_ns(), label),
            )

    def status(self) -> dict[str, object]:
        with self._connect() as connection:
            self._version(connection)
            contract, maximum, used, deadline_ns, _ = self._row_v2(connection)
            forward = self._forward_state(connection, contract)
            started_ns = connection.execute(
                "SELECT started_ns FROM budget WHERE id = 1"
            ).fetchone()[0]
            rows = connection.execute(
                "SELECT label, updates, target_positions, generation_calls, "
                "generated_tokens, actual_updates, actual_targets, actual_calls, "
                "actual_tokens FROM reservations ORDER BY id"
            ).fetchall()
            status = {
                "version": 2,
                "contract_sha256": connection.execute(
                    "SELECT contract_sha256 FROM budget WHERE id = 1"
                ).fetchone()[0],
                "content_identity_sha256": contract.content_identity_sha256,
                "max_updates": maximum[0],
                "charged_updates": used[0],
                "max_actual_target_positions": maximum[1],
                "charged_actual_target_positions": used[1],
                "max_generation_calls": maximum[2],
                "charged_generation_calls": used[2],
                "max_generated_tokens": maximum[3],
                "charged_generated_tokens": used[3],
                "max_wall_seconds": contract.max_wall_seconds,
                "started_at_utc": datetime.fromtimestamp(
                    started_ns / 1e9, UTC
                ).isoformat(),
                "deadline_utc": datetime.fromtimestamp(
                    deadline_ns / 1e9, UTC
                ).isoformat(),
                "remaining_wall_seconds": max(
                    0.0, (deadline_ns - time.time_ns()) / 1e9
                ),
                "reservations": [
                    {
                        "label": label,
                        "updates": updates,
                        "target_positions": target_positions,
                        "generation_calls": calls,
                        "generated_tokens": tokens,
                        "actual_updates": actual_updates,
                        "actual_target_positions": actual_targets,
                        "actual_generation_calls": actual_calls,
                        "actual_generated_tokens": actual_tokens,
                    }
                    for (
                        label,
                        updates,
                        target_positions,
                        calls,
                        tokens,
                        actual_updates,
                        actual_targets,
                        actual_calls,
                        actual_tokens,
                    ) in rows
                ],
            }
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='resolved_artifacts'"
            ).fetchone():
                status["resolved_artifacts"] = [
                    {"kind": kind, "path": path, "sha256": digest}
                    for kind, path, digest in connection.execute(
                        "SELECT kind, path, sha256 FROM resolved_artifacts ORDER BY kind"
                    )
                ]
            if forward is not None:
                forward_rows = connection.execute(
                    "SELECT label, kind, positions FROM forward_reservations ORDER BY id"
                ).fetchall()
                status.update(
                    max_fixed_profile_forward_positions=contract.max_fixed_profile_forward_positions,
                    charged_fixed_profile_forward_positions=forward[0],
                    max_nontraining_forward_positions=contract.max_nontraining_forward_positions,
                    charged_nontraining_forward_positions=forward[1],
                    forward_reservations=[
                        {"label": label, "kind": kind, "positions": positions}
                        for label, kind, positions in forward_rows
                    ],
                )
                if contract.max_operational_validation_batches is not None:
                    status.update(
                        max_operational_validation_batches=contract.max_operational_validation_batches,
                        charged_operational_validation_batches=forward[2],
                        max_operational_validation_forward_positions=contract.max_operational_validation_forward_positions,
                        charged_operational_validation_forward_positions=forward[3],
                    )
            return status

    def run_contract(
        self,
        command: list[str],
        *,
        activity: Literal["train", "warmup", "validate", "evaluate", "inspect"],
        label: str,
        content_identity_sha256: str,
        monitor_policy_path: Path,
        workspace_baseline_path: Path,
        workspace_root: Path,
        completion: Path,
        updates: int = 0,
        target_positions: int = 0,
        generation_calls: int = 0,
        generated_tokens: int = 0,
        grace_seconds: float = 1.0,
        native_receipt_kind: Literal[
            "train",
            "panel",
            "campaign_run",
            "campaign_panel",
            "campaign_evaluation",
            "none",
        ] = "none",
        native_receipt_path: Path | None = None,
        campaign_stage: str | None = None,
        campaign_work_dir: Path | None = None,
        parent_checkpoint_path: Path | None = None,
    ) -> int:
        """Reserve before work, then run one phase under the owned supervisor."""
        if not command or not completion.is_absolute() or completion.exists():
            raise ValueError("command and unused absolute completion path required")
        resolved_root = workspace_root.resolve(strict=True)
        resolved_parent = completion.parent.resolve(strict=True)
        if not resolved_root.is_dir() or not (
            resolved_parent == resolved_root or resolved_root in resolved_parent.parents
        ):
            raise ValueError(
                "completion must be inside the existing common workspace root"
            )
        monitor_log_dir = completion.with_name(completion.name + ".monitor")
        if monitor_log_dir.exists():
            raise AttemptBudgetError("native monitor log directory already exists")
        if not 0 <= grace_seconds <= 10 or not math.isfinite(grace_seconds):
            raise ValueError("invalid owned-process grace")
        with self._connect() as connection:
            if self._version(connection) != 2:
                raise AttemptBudgetError("run_contract requires a v2 ledger")
            contract, _, _, _, _ = self._row_v2(connection)
        if content_identity_sha256 != contract.content_identity_sha256:
            raise AttemptBudgetError("content identity differs from attempt contract")
        try:
            classified = classify_attempt_command(command)
        except ValueError as error:
            raise AttemptBudgetError(str(error)) from error
        validate_attempt_paths(
            classified,
            ledger=self.path,
            completion=completion,
            monitor_log_dir=monitor_log_dir,
            monitor_policy=monitor_policy_path,
            baseline=workspace_baseline_path,
        )
        if contract.preparation_only and (
            classified.effect not in {"preparation", "inspection"}
            or activity != "inspect"
        ):
            raise AttemptBudgetError("preparation-only attempt forbids model work")
        if (
            classified.work_dir is not None
            and classified.work_dir.resolve() != resolved_root
        ):
            raise AttemptBudgetError("native work directory differs from attempt root")
        verified_retained_copy = False
        if contract.offline_retained_sources_only:
            if classified.operation in {"corpus budget-init", "corpus alias-snapshot"}:
                raise AttemptBudgetError(
                    "this attempt permits verified offline reuse only"
                )
            if classified.operation == "corpus acquire" and not classified.options.get(
                "--offline"
            ):
                from klab.corpus.project import (
                    RETAINED_MAX_BYTES,
                    RETAINED_MAX_ROWS,
                    load_project,
                )

                reuse_project = load_project(Path(classified.args[2]))
                if (
                    contract.acquisition_project_sha256 is None
                    or reuse_project.config.transport_budget is not None
                    or not (
                        (
                            reuse_project.config.source_effects
                            and all(
                                effect.effect == "reuse_only"
                                for effect in reuse_project.config.source_effects
                            )
                        )
                        or (
                            reuse_project.config.source_effects is None
                            and all(
                                source.kind == "huggingface_dataset"
                                and source.acquisition.retained_subset is not None
                                for source in reuse_project.sources
                            )
                            and sum(
                                s.acquisition.max_rows for s in reuse_project.sources
                            )
                            <= RETAINED_MAX_ROWS
                            and sum(
                                s.acquisition.max_bytes for s in reuse_project.sources
                            )
                            <= RETAINED_MAX_BYTES
                        )
                    )
                ):
                    raise AttemptBudgetError(
                        "this attempt permits verified offline reuse only"
                    )
                if reuse_project.config.source_effects is None:
                    from klab.corpus.acquisition import _project_sha

                    if (
                        contract.preparation_acquisition_identity_sha256 is None
                        or _project_sha(reuse_project)
                        != contract.preparation_acquisition_identity_sha256
                    ):
                        raise AttemptBudgetError(
                            "retained subset acquisition identity differs from contract"
                        )
                verified_retained_copy = True
        if "--output" in classified.options:
            output = Path(str(classified.options["--output"]))
            if not output.is_absolute() or (
                output.parent.resolve() != resolved_root
                and resolved_root not in output.parent.resolve().parents
            ):
                raise AttemptBudgetError("native output is outside the attempt root")
        if classified.monitor is not None:
            log_dir = Path(classified.monitor["--log-dir"])
            if (
                log_dir.parent.resolve() != resolved_root
                and resolved_root not in log_dir.parent.resolve().parents
            ):
                raise AttemptBudgetError(
                    "nested monitor logs are outside the attempt root"
                )
        if (
            classified.operation == "monitor-baseline"
            and Path(classified.args[1]).resolve() != resolved_root
        ):
            raise AttemptBudgetError("monitor baseline root differs from attempt root")
        if (
            classified.operation == "attempt status"
            and Path(str(classified.options["--ledger"])) != self.path
        ):
            raise AttemptBudgetError("nested attempt status points at another ledger")
        if (
            classified.operation
            in {
                "corpus budget-init",
                "corpus budget-status",
                "corpus alias-snapshot",
                "corpus acquire",
                "corpus admission-draft",
                "corpus inspect-admission",
            }
            and contract.acquisition_project_sha256 is not None
            and hashlib.sha256(Path(classified.args[2]).read_bytes()).hexdigest()
            != contract.acquisition_project_sha256
        ):
            raise AttemptBudgetError("acquisition project differs from contract")
        if (
            contract.require_admission_inspection_binding
            and classified.operation == "corpus inspect-admission"
        ):
            from klab.corpus.project import load_project

            project = load_project(Path(classified.args[2]))
            lock_path = (
                resolved_root / "corpora" / project.config.id / "acquisition.json"
            )
            if (
                hashlib.sha256(
                    Path(str(classified.options["--selection"])).read_bytes()
                ).hexdigest()
                != contract.admission_selection_sha256
                or hashlib.sha256(
                    Path(str(classified.options["--policy-document"])).read_bytes()
                ).hexdigest()
                != contract.admission_policy_sha256
                or hashlib.sha256(lock_path.read_bytes()).hexdigest()
                != contract.admission_lock_sha256
            ):
                raise AttemptBudgetError(
                    "admission inspection input differs from contract"
                )
        if classified.operation == "tokenizer train":
            import yaml

            from klab.config.loading import load_tokenizer_config
            from klab.corpus.export import verify_release_export

            declaration = yaml.safe_load(Path(classified.args[2]).read_text())
            declared_output = declaration.get("output_dir")
            if (
                not isinstance(declared_output, str)
                or not Path(declared_output).is_absolute()
            ):
                raise AttemptBudgetError("tokenizer output must be declared absolute")
            tokenizer_config = load_tokenizer_config(Path(classified.args[2]))
            destination = tokenizer_config.output_dir
            if tokenizer_config.sampling is None:
                raise AttemptBudgetError(
                    "tokenizer preparation requires declared bounded TRAIN sampling"
                )
            if (
                destination.is_symlink()
                or destination.resolve() == resolved_root
                or not destination.resolve().is_relative_to(resolved_root)
            ):
                raise AttemptBudgetError(
                    "tokenizer output is outside the attempt root or linked"
                )
            if destination.exists():
                raise AttemptBudgetError(
                    "tokenizer preparation requires a new output directory"
                )
            verify_release_export(tokenizer_config.dataset)
        preparation_kind = {
            "corpus split-inventory": "pre_freeze_project",
            "corpus freeze-splits": "pre_freeze_project",
            "corpus build": "build_project",
        }.get(classified.operation)
        if preparation_kind and contract.preparation_acquisition_identity_sha256:
            bound_project = self._resolved_artifact(
                preparation_kind, workspace_root=resolved_root
            )
            if bound_project is None or Path(classified.args[2]) != bound_project[0]:
                raise AttemptBudgetError(
                    "preparation command requires its bound project declaration"
                )
        elif preparation_kind and contract.acquisition_project_sha256 is not None:
            # Historical contracts used the acquisition project for these
            # commands. Keep that check until a reviewed variant opts in.
            if (
                classified.operation != "corpus build"
                and hashlib.sha256(Path(classified.args[2]).read_bytes()).hexdigest()
                != contract.acquisition_project_sha256
            ):
                raise AttemptBudgetError("acquisition project differs from contract")
        accepted_operations = {
            "tokenizer train",
            "corpus export",
            "corpus measure-tokens",
            "corpus materialize-mixture",
            "corpus verify-mixture",
            "data prepared-inputs publish",
            "data prepared-inputs verify",
            "stage",
            "train",
        }
        if (
            contract.require_release_acceptance_binding
            and classified.operation in accepted_operations
        ):
            accepted = self._resolved_artifact(
                "release_acceptance", workspace_root=resolved_root
            )
            if accepted is None:
                raise AttemptBudgetError("exact release has not been accepted")
            review = json.loads(accepted[0].read_text(encoding="utf-8"))
            if (
                classified.operation == "tokenizer train"
                and tokenizer_config.dataset.corpus_release_path.resolve()
                != Path(review["release_path"]).resolve()
            ):
                raise AttemptBudgetError("tokenizer release differs from acceptance")
            if classified.operation in {"corpus measure-tokens", "corpus export"}:
                from klab.corpus.cli import _release_path

                measured = _release_path(classified.args[2], resolved_root)
                if measured.resolve() != Path(review["release_path"]).resolve():
                    raise AttemptBudgetError("measured release differs from acceptance")
            if classified.operation in {
                "corpus materialize-mixture",
                "corpus verify-mixture",
            }:
                import yaml

                mixture = yaml.safe_load(Path(classified.args[2]).read_text())
                if (
                    not isinstance(mixture, dict)
                    or Path(mixture.get("release_path", "")).resolve()
                    != Path(review["release_path"]).resolve()
                    or Path(mixture.get("family_inventory", "")).resolve()
                    != Path(review["family_inventory_path"]).resolve()
                ):
                    raise AttemptBudgetError("mixture differs from accepted release")
            if classified.operation in {
                "data prepared-inputs publish",
                "data prepared-inputs verify",
            }:
                from klab.corpus.release_review import (
                    verify_accepted_prepared_config,
                )

                verify_accepted_prepared_config(Path(classified.args[3]), review)
        if classified.effect in {"fixed_score", "continuation"} and (
            (
                contract.fixed_profile_sha256 is not None
                and classified.options["--expected-profile-sha256"]
                != contract.fixed_profile_sha256
            )
            or (
                contract.fixed_family_inventory_sha256 is not None
                and classified.options["--expected-family-sha256"]
                != contract.fixed_family_inventory_sha256
            )
        ):
            raise AttemptBudgetError("fixed evaluation identity differs from contract")
        if classified.effect == "train" and activity != "train":
            raise AttemptBudgetError("training command requires training activity")
        if (
            classified.effect in {"fixed_score", "continuation"}
            and activity != "evaluate"
        ):
            raise AttemptBudgetError(
                "fixed model evaluation requires evaluation activity"
            )
        if classified.effect not in {"train", "campaign"} and activity == "train":
            raise AttemptBudgetError("training activity requires native training")
        if classified.effect == "campaign" and native_receipt_kind not in {
            "campaign_run",
            "campaign_panel",
            "campaign_evaluation",
        }:
            raise AttemptBudgetError("Campaign work requires a native stage receipt")
        if classified.monitor is not None:
            nested = classified.monitor
            policy_digest = hashlib.sha256(
                Path(nested["--policy"]).read_bytes()
            ).hexdigest()
            preparation_wrapper = (
                policy_digest == contract.preparation_monitor_policy_sha256
                and Path(nested["--baseline"]) == workspace_baseline_path
                and classified.effect not in {"fixed_score", "continuation"}
            )
            evaluation_baseline = self._resolved_artifact(
                "evaluation_baseline", workspace_root=resolved_root
            )
            evaluation_wrapper = (
                policy_digest == contract.evaluation_monitor_policy_sha256
                and evaluation_baseline is not None
                and Path(nested["--baseline"]) == evaluation_baseline[0]
            )
            if Path(nested["--workspace"]).resolve() != resolved_root or not (
                preparation_wrapper or evaluation_wrapper
            ):
                raise AttemptBudgetError("nested monitor policy or baseline is unbound")
        if (
            classified.effect in {"fixed_score", "continuation"}
            and contract.require_evaluation_baseline_binding
            and (
                classified.monitor is None
                or hashlib.sha256(
                    Path(classified.monitor["--policy"]).read_bytes()
                ).hexdigest()
                != contract.evaluation_monitor_policy_sha256
            )
        ):
            raise AttemptBudgetError(
                "fixed evaluation requires the bound output monitor"
            )
        if (
            classified.operation == "corpus acquire"
            and not classified.options.get("--offline")
            and not verified_retained_copy
        ):
            from klab.corpus.project import load_project
            from klab.corpus.transport_budget import TransportBudget

            project = load_project(Path(classified.args[2]))
            if project.config.transport_budget is None:
                raise AttemptBudgetError(
                    "live corpus acquisition requires a transport budget"
                )
            TransportBudget(
                resolved_root
                / "corpora"
                / project.config.id
                / "transport-budget.sqlite",
                project,
            )
        train_binding = self._resolved_artifact(
            "train_config", workspace_root=resolved_root
        )
        if (
            classified.operation
            in {
                "evaluation fixed-slices select",
                "evaluation fixed-slices verify-selection",
            }
            and contract.fixed_profile_sha256 is not None
        ):
            try:
                declaration = json.loads(Path(classified.args[3]).read_text())
            except (OSError, ValueError) as error:
                raise AttemptBudgetError(
                    "fixed selection declaration cannot be read"
                ) from error
            if not isinstance(declaration, dict) or (
                declaration.get("profile_sha256") != contract.fixed_profile_sha256
                or (
                    contract.require_resolved_train_config_binding
                    and (
                        train_binding is None
                        or declaration.get("config_sha256") != train_binding[1]
                        or declaration.get("config") != str(train_binding[0])
                    )
                )
            ):
                raise AttemptBudgetError(
                    "fixed selection declaration differs from attempt binding"
                )
        if (
            classified.effect in {"stage", "train"}
            and contract.require_resolved_train_config_binding
            and (train_binding is None or Path(classified.args[1]) != train_binding[0])
        ):
            raise AttemptBudgetError("resolved training config is not bound")
        if (
            contract.require_resolved_train_config_binding
            and classified.effect == "stage"
            and not {
                "--prepared-inputs",
                "--cold-verify",
                "--runtime",
                "--resource-envelope",
            }.issubset(classified.options)
        ):
            raise AttemptBudgetError("stage differs from resolved packet binding")
        if (
            contract.require_resolved_train_config_binding
            and classified.effect == "train"
            and not {"--stage-bundle", "--runtime", "--resource-envelope"}.issubset(
                classified.options
            )
        ):
            raise AttemptBudgetError("train differs from resolved packet binding")
        train_identity = (
            train_binding[1]
            if classified.effect == "train" and train_binding is not None
            else content_identity_sha256
        )
        if activity not in {"train", "warmup", "validate", "evaluate", "inspect"}:
            raise ValueError("unknown attempt activity")
        if contract.max_optimizer_updates == 0 and activity in ("train", "warmup"):
            raise AttemptBudgetError("zero-update contract refuses training or warmup")
        if activity in ("train", "warmup") and updates == 0:
            raise AttemptBudgetError(
                "training or warmup requires a positive update reservation"
            )
        if activity in ("train", "warmup") and target_positions == 0:
            raise AttemptBudgetError(
                "training or warmup requires a positive target-position reservation"
            )
        if generated_tokens > 0 and generation_calls == 0:
            raise AttemptBudgetError(
                "generated tokens require reserved generation calls"
            )
        if activity in ("validate", "evaluate", "inspect") and updates != 0:
            raise AttemptBudgetError(
                "nontraining activity cannot reserve optimizer updates"
            )
        reserved = (updates, target_positions, generation_calls, generated_tokens)
        if native_receipt_kind == "none" and any(reserved):
            raise AttemptBudgetError("nonzero reservation requires a native receipt")
        metered_fixed = classified.effect in {"fixed_score", "continuation"}
        if native_receipt_kind == "none" and not (
            self._approved_counter_free_command(command) or metered_fixed
        ):
            raise AttemptBudgetError(
                "counter-free phase requires an explicit supported native command"
            )
        if metered_fixed:
            status = self.status()
            required_fixed = (
                int(classified.options["--max-forward-positions"])
                if classified.effect == "fixed_score"
                else 0
            )
            required_total = required_fixed if required_fixed else 8 * 6112
            if (
                contract.max_nontraining_forward_positions is None
                or contract.max_fixed_profile_forward_positions is None
                or status["charged_nontraining_forward_positions"] + required_total
                > contract.max_nontraining_forward_positions
                or status["charged_fixed_profile_forward_positions"] + required_fixed
                > contract.max_fixed_profile_forward_positions
                or (
                    classified.effect == "continuation"
                    and (
                        status["charged_generation_calls"] + 8
                        > contract.max_generation_calls
                        or status["charged_generated_tokens"] + 512
                        > contract.max_generated_tokens
                    )
                )
            ):
                raise AttemptBudgetError(
                    "fixed evaluation lacks declared model allocation"
                )
        if activity in ("train", "warmup") and native_receipt_kind not in {
            "train",
            "campaign_run",
        }:
            raise AttemptBudgetError("training requires a native run receipt")
        if (generation_calls or generated_tokens) and native_receipt_kind not in {
            "panel",
            "campaign_panel",
        }:
            raise AttemptBudgetError("generation requires a native panel receipt")
        if native_receipt_kind in {"train", "panel"}:
            if native_receipt_path is None or native_receipt_path.exists():
                raise AttemptBudgetError("direct native receipt must be a fresh path")
            receipt_parent = native_receipt_path.parent.resolve(strict=True)
            if (
                receipt_parent != resolved_root
                and resolved_root not in receipt_parent.parents
            ):
                raise AttemptBudgetError("native receipt must be inside common root")
        if native_receipt_kind in {
            "campaign_run",
            "campaign_panel",
            "campaign_evaluation",
        }:
            if (
                campaign_stage is None
                or campaign_work_dir is None
                or native_receipt_path is None
            ):
                raise AttemptBudgetError("Campaign receipt binding is incomplete")
            campaign_root = campaign_work_dir.resolve(strict=True)
            if (
                campaign_root != resolved_root
                and resolved_root not in campaign_root.parents
            ):
                raise AttemptBudgetError(
                    "Campaign work directory is outside common root"
                )
            stage_limits = self._preflight_campaign_receipt(
                native_receipt_path,
                campaign_work_dir,
                campaign_stage,
                expected_kind=(
                    "experiment_run"
                    if native_receipt_kind == "campaign_run"
                    else "generation_panel"
                    if native_receipt_kind == "campaign_panel"
                    else "evaluation"
                ),
                content_identity_sha256=content_identity_sha256,
                contract_sha256=self.status()["contract_sha256"],
            )
        else:
            stage_limits = None
        if native_receipt_kind != "none":
            self._preflight_native_command(
                command,
                native_receipt_kind=native_receipt_kind,
                native_receipt_path=native_receipt_path,
                campaign_stage=campaign_stage,
                stage_limits=stage_limits,
                reserved=reserved,
                parent_checkpoint_path=parent_checkpoint_path,
                content_identity_sha256=train_identity,
            )
        if (
            contract.max_optimizer_updates == 0
            and native_receipt_kind not in {"campaign_panel", "campaign_evaluation"}
            and not self._approved_zero_update_command(command)
        ):
            raise AttemptBudgetError(
                "zero-update contract refuses unknown or training entry point"
            )
        from klab import operational_monitor, owned_process

        baseline, policy = validate_contract_monitor_inputs(
            contract,
            monitor_policy_path=monitor_policy_path,
            workspace_baseline_path=workspace_baseline_path,
            workspace_root=workspace_root,
        )
        ready = completion.with_name(completion.name + ".ready")
        if ready.exists():
            raise AttemptBudgetError("owned readiness path already exists")
        self.reserve_vector(
            label,
            updates=updates,
            target_positions=target_positions,
            generation_calls=generation_calls,
            generated_tokens=generated_tokens,
            content_identity_sha256=content_identity_sha256,
        )
        # Anchor before the wall-clock read: its transaction/return latency must
        # not extend the helper deadline beyond the parent's later budget read.
        deadline_anchor = time.monotonic()
        remaining = self.remaining_seconds()
        deadline = deadline_anchor + remaining
        environment = os.environ.copy()
        environment["KLAB_ATTEMPT_BUDGET_LEDGER"] = str(self.path)
        environment["KLAB_ATTEMPT_CONTRACT_SHA256"] = self.status()["contract_sha256"]
        environment["KLAB_ATTEMPT_CONTENT_IDENTITY_SHA256"] = content_identity_sha256
        environment["KLAB_ATTEMPT_PHASE_LABEL"] = label
        environment["KLAB_ATTEMPT_ACTIVITY"] = activity
        invocation = [
            sys.executable,
            "-m",
            "klab.owned_process",
            "--deadline",
            repr(deadline),
            "--completion",
            str(completion),
            "--grace",
            str(grace_seconds),
            "--ready",
            str(ready),
            "--",
            sys.executable,
            "-m",
            "klab.training.attempt_budget",
            "_monitored_phase",
            "--policy",
            str(monitor_policy_path),
            "--policy-sha256",
            contract.monitor_policy_sha256,
            "--baseline",
            str(workspace_baseline_path),
            "--baseline-sha256",
            contract.workspace_baseline_sha256,
            "--workspace",
            str(workspace_root),
            "--log-dir",
            str(monitor_log_dir),
            "--",
            *command,
        ]
        process = subprocess.Popen(invocation, env=environment)
        timed_out = False
        try:
            rc = process.wait(timeout=max(0.001, self.remaining_seconds()))
        except BaseException as error:
            timed_out = isinstance(
                error, (subprocess.TimeoutExpired, AttemptBudgetError)
            )
            deadline_stop = timed_out and time.monotonic() >= deadline
            # The helper already owns this deadline, including refusing to
            # launch after it. TERM during interpreter startup can kill it
            # before its cleanup handler exists, losing the shutdown receipt.
            # Give it only the existing bounded cleanup window to certify exit.
            # An earlier parent timeout (e.g. a wall-clock jump) still needs TERM.
            if process.poll() is None and not deadline_stop:
                process.terminate()  # The isolated subreaper owns TERM-to-KILL cleanup.
            try:
                rc = process.wait(timeout=grace_seconds + 4)
            except subprocess.TimeoutExpired as cleanup_error:
                raise AttemptBudgetError(
                    "owned-process shutdown deadline exhausted"
                ) from cleanup_error
            self._verify_owned_completion(completion, rc)
            storage_error: AttemptBudgetError | None = None
            try:
                self._final_storage_receipt(
                    completion=completion,
                    workspace_root=workspace_root,
                    baseline=baseline,
                    policy=policy,
                    label=label,
                    native_status="INTERRUPTED",
                )
            except AttemptBudgetError as final_error:
                storage_error = final_error
            if timed_out:
                raise AttemptBudgetError("shared wall-time limit reached") from None
            if storage_error is not None:
                raise storage_error
            raise
        self._verify_owned_completion(completion, rc)
        try:
            native = json.loads((monitor_log_dir / "completion.json").read_text())
            if (
                native["kind"] != "completion"
                or native["launch"]["command"] != command
                or native["launch"]["baseline_sha256"]
                != contract.workspace_baseline_sha256
                or native["launch"]["policy_sha256"] != contract.monitor_policy_sha256
                or native["launch"]["source_sha256"]
                != hashlib.sha256(
                    Path(operational_monitor.__file__).read_bytes()
                ).hexdigest()
                or native["launch"]["owned_source_sha256"]
                != hashlib.sha256(Path(owned_process.__file__).read_bytes()).hexdigest()
                or (native["status"] == "COMPLETE") != (rc == 0)
            ):
                raise ValueError("native monitor completion differs from phase binding")
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._final_storage_receipt(
                completion=completion,
                workspace_root=workspace_root,
                baseline=baseline,
                policy=policy,
                label=label,
                native_status="UNVERIFIED",
            )
            raise AttemptBudgetError("native monitor completion unverified") from error
        # Identity drift after the phase cannot turn a receipt into a valid success.
        if (
            hashlib.sha256(monitor_policy_path.read_bytes()).hexdigest()
            != contract.monitor_policy_sha256
        ):
            raise AttemptBudgetError("monitor policy changed during phase")
        from klab.operational_monitor import load_workspace_baseline

        if (
            load_workspace_baseline(workspace_baseline_path, workspace_root).sha256
            != contract.workspace_baseline_sha256
        ):
            raise AttemptBudgetError("workspace baseline changed during phase")
        actual: tuple[int, int, int, int] | None = None
        if rc == 0:
            if metered_fixed:
                charged = self.status()
                phase_forward = [
                    row
                    for row in charged.get("forward_reservations", [])
                    if row["label"].startswith(f"{label}:")
                ]
                expected = (
                    int(classified.options["--max-forward-positions"])
                    if classified.effect == "fixed_score"
                    else 8 * 6112
                )
                kind = (
                    "fixed_profile"
                    if classified.effect == "fixed_score"
                    else "generation"
                )
                if sum(row["positions"] for row in phase_forward) != expected or any(
                    row["kind"] != kind for row in phase_forward
                ):
                    self._final_storage_receipt(
                        completion=completion,
                        workspace_root=workspace_root,
                        baseline=baseline,
                        policy=policy,
                        label=label,
                        native_status="UNVERIFIED",
                    )
                    raise AttemptBudgetError(
                        "fixed evaluation forward charges unverified"
                    )
                if classified.effect == "continuation":
                    phase_requests = [
                        row
                        for row in charged["reservations"]
                        if row["label"].startswith(f"{label}:")
                    ]
                    if (
                        sum(row["generation_calls"] for row in phase_requests) != 8
                        or sum(row["generated_tokens"] for row in phase_requests) != 512
                    ):
                        raise AttemptBudgetError("fixed generation charges unverified")
            from klab.training.attempt_receipts import verify_native_phase_counters
            from klab.training.manifest import sha256_file

            try:
                if native_receipt_kind == "train":
                    config_source = Path(self._native_command_args(command)[1])
                    if (
                        config_source.is_symlink()
                        or sha256_file(config_source) != train_identity
                    ):
                        raise ValueError("direct train config changed during phase")
                actual = verify_native_phase_counters(
                    native_receipt_kind,
                    native_receipt_path,
                    campaign_stage=campaign_stage,
                    campaign_work_dir=campaign_work_dir,
                    parent_checkpoint_path=parent_checkpoint_path,
                    expected_content_sha256=train_identity,
                )
            except (OSError, ValueError, TypeError, KeyError) as error:
                self._final_storage_receipt(
                    completion=completion,
                    workspace_root=workspace_root,
                    baseline=baseline,
                    policy=policy,
                    label=label,
                    native_status="UNVERIFIED",
                )
                raise AttemptBudgetError("native phase counters unverified") from error
            try:
                self._record_verified_actual(
                    label,
                    updates=actual[0],
                    target_positions=actual[1],
                    generation_calls=actual[2],
                    generated_tokens=actual[3],
                    content_identity_sha256=content_identity_sha256,
                )
            except AttemptBudgetError:
                self._final_storage_receipt(
                    completion=completion,
                    workspace_root=workspace_root,
                    baseline=baseline,
                    policy=policy,
                    label=label,
                    native_status="COUNTER_CAP_EXCEEDED",
                )
                raise
        self.remaining_seconds()
        self._final_storage_receipt(
            completion=completion,
            workspace_root=workspace_root,
            baseline=baseline,
            policy=policy,
            label=label,
            native_status=native["status"],
        )
        self._check_final_deadline()
        return rc

    def _check_final_deadline(self) -> None:
        """Read-only terminal check, after the storage receipt has been counted."""
        with self._connect() as connection:
            _, _, _, deadline_ns, last_checked_ns = self._row_v2(connection)
        now_ns = time.time_ns()
        if now_ns < last_checked_ns:
            raise AttemptBudgetError("clock moved backwards; budget fails closed")
        if now_ns >= deadline_ns:
            raise AttemptBudgetError("shared wall-time limit reached")

    @staticmethod
    def _preflight_campaign_receipt(
        source: Path,
        work_dir: Path,
        stage_id: str,
        *,
        expected_kind: str,
        content_identity_sha256: str,
        contract_sha256: str,
    ) -> tuple[int, int]:
        from klab.campaign.engine import CampaignEngine

        engine = CampaignEngine(source, work_dir, cold_verify=True)
        projection = engine.inspect("next")
        action = projection["next_action"]
        if (
            action.get("stage") != stage_id
            or action.get("action") not in {"apply", "resume"}
            or engine.stages[stage_id].kind != expected_kind
        ):
            raise AttemptBudgetError("Campaign stage is not the next unused action")
        rows = {row["id"]: row for row in projection["stages"]}
        stage = engine.stages[stage_id]
        run_stage = (
            stage
            if expected_kind == "experiment_run"
            else engine.stages[engine.stages[stage.collect].run]
        )
        lock = engine._lock(rows, run_stage.plan)
        if lock.scientific_sha256 != content_identity_sha256:
            raise AttemptBudgetError("Campaign scientific lock differs from contract")
        reference = lock.execution.get("attempt_contract")
        if (
            not isinstance(reference, dict)
            or reference.get("sha256") != contract_sha256
        ):
            raise AttemptBudgetError("Campaign lock lacks the pinned attempt contract")
        if expected_kind == "experiment_run":
            runtime = engine.stages[stage.runtime]
            if runtime.worker is not None:
                raise AttemptBudgetError(
                    "contracted Campaign run requires an owned local runtime profile"
                )
            cells = [cell for cell in lock.cells if cell.id == stage.cell]
            if len(cells) != 1:
                raise AttemptBudgetError("Campaign run cell selection is ambiguous")
            training = cells[0].config.training
            return training.max_steps, training.max_tokens
        if expected_kind == "generation_panel":
            from klab.evaluation.panel import load_panel

            panel = load_panel(engine._path(stage.panel))
            return len(panel.prompts), len(panel.prompts) * panel.decoder.max_new_tokens
        from klab.evaluation.suite import load_suite

        suite = load_suite(engine._path(stage.suite))
        if any(
            item.kind not in {"heldout_lm", "surface_review"}
            for item in suite.evaluations
        ):
            raise AttemptBudgetError("Campaign evaluation has unaccounted generation")
        return 0, 0

    @staticmethod
    def _native_command_args(command: list[str]) -> list[str]:
        try:
            return list(classify_attempt_command(command).args)
        except ValueError as error:
            raise AttemptBudgetError(str(error)) from error

    @classmethod
    def _preflight_native_command(
        cls,
        command: list[str],
        *,
        native_receipt_kind: str,
        native_receipt_path: Path | None,
        campaign_stage: str | None,
        stage_limits: tuple[int, int] | None,
        reserved: tuple[int, int, int, int],
        parent_checkpoint_path: Path | None,
        content_identity_sha256: str,
    ) -> None:
        args = cls._native_command_args(command)
        if native_receipt_kind in {
            "campaign_run",
            "campaign_panel",
            "campaign_evaluation",
        }:
            if (
                len(args) < 3
                or args[:2] != ["campaign", "apply"]
                or Path(args[2]).resolve() != native_receipt_path.resolve()
                or args.count("--only-stage") != 1
                or args[args.index("--only-stage") + 1 : args.index("--only-stage") + 2]
                != [campaign_stage]
                or stage_limits is None
            ):
                raise AttemptBudgetError("Campaign command differs from phase binding")
            if native_receipt_kind == "campaign_run":
                if (
                    "--execute-runs" not in args
                    or reserved[0] < stage_limits[0]
                    or reserved[1] < stage_limits[1]
                    or reserved[2] != 0
                    or reserved[3] != 0
                ):
                    raise AttemptBudgetError(
                        "Campaign run exceeds reserved native config"
                    )
            elif native_receipt_kind == "campaign_panel" and (
                "--execute-runs" in args
                or reserved[0] != 0
                or reserved[1] != 0
                or reserved[2] < stage_limits[0]
                or reserved[3] < stage_limits[1]
            ):
                raise AttemptBudgetError("Campaign panel exceeds reserved declaration")
            elif native_receipt_kind == "campaign_evaluation" and (
                "--execute-runs" in args or any(reserved)
            ):
                raise AttemptBudgetError("Campaign evaluation must have zero counters")
            return
        if native_receipt_kind == "panel":
            raise AttemptBudgetError("direct panel execution has no native CLI binding")
        if native_receipt_kind != "train" or len(args) < 2 or args[0] != "train":
            raise AttemptBudgetError("native run receipt requires klab train")
        if args.count("--run-id") != 1 or args.count("--runs-dir") != 1:
            raise AttemptBudgetError("native train requires explicit run ID and store")
        run_id = args[args.index("--run-id") + 1 : args.index("--run-id") + 2]
        runs_dir = args[args.index("--runs-dir") + 1 : args.index("--runs-dir") + 2]
        if (
            not run_id
            or not runs_dir
            or native_receipt_path is None
            or (Path(runs_dir[0]).resolve() / run_id[0])
            != native_receipt_path.resolve()
        ):
            raise AttemptBudgetError("native train command differs from receipt path")
        from klab.config.loading import load_config
        from klab.training.manifest import sha256_file

        config_source = Path(args[1])
        if (
            config_source.is_symlink()
            or sha256_file(config_source) != content_identity_sha256
        ):
            raise AttemptBudgetError(
                "direct train config differs from contract identity"
            )
        config = load_config(config_source)
        if sha256_file(config_source) != content_identity_sha256:
            raise AttemptBudgetError("direct train config changed during preflight")
        if (
            reserved[0] < config.training.max_steps
            or reserved[1] < config.training.max_tokens
            or reserved[2] != 0
            or reserved[3] != 0
        ):
            raise AttemptBudgetError("native train config exceeds phase reservation")
        parent_flags = [
            name
            for name in ("--resume", "--extend-budget", "--promote")
            if name in args
        ]
        if len(parent_flags) > 1 or "--recover" in args:
            raise AttemptBudgetError("ambiguous native training continuation")
        if parent_flags:
            flag = parent_flags[0]
            if parent_checkpoint_path is None or args[
                args.index(flag) + 1 : args.index(flag) + 2
            ] != [str(parent_checkpoint_path)]:
                raise AttemptBudgetError("training parent differs from receipt binding")
        elif parent_checkpoint_path is not None:
            raise AttemptBudgetError("fresh native training cannot claim a parent")

    @staticmethod
    def _approved_counter_free_command(command: list[str]) -> bool:
        try:
            effect = classify_attempt_command(command).effect
        except ValueError:
            return False
        return effect in {"inspection", "preparation", "stage"}

    @staticmethod
    def _approved_zero_update_command(command: list[str]) -> bool:
        try:
            return classify_attempt_command(command).effect != "train"
        except ValueError:
            return False

    @staticmethod
    def _verify_owned_completion(path: Path, returncode: int) -> None:
        try:
            receipt = json.loads(path.read_text())
            if (
                receipt["format"] != "klab-owned-completion-v1"
                or receipt["returncode"] != returncode
                or receipt["living_descendants"] != 0
            ):
                raise ValueError("invalid owned-process completion")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise AttemptBudgetError("owned shutdown unverified") from error

    @staticmethod
    def _final_storage_receipt(
        *,
        completion: Path,
        workspace_root: Path,
        baseline: object,
        policy: object,
        label: str,
        native_status: str,
    ) -> dict[str, object]:
        """Count the outer owned receipt and this final receipt in the same root."""
        from klab.operational_monitor import (
            MonitorPolicy,
            WorkspaceBaseline,
            sample_workspace_tree,
        )

        if not isinstance(baseline, WorkspaceBaseline) or not isinstance(
            policy, MonitorPolicy
        ):
            raise TypeError("typed native baseline and monitor policy required")
        path = completion.with_name(completion.name + ".attempt.json")
        if path.exists():
            raise AttemptBudgetError("attempt final receipt already exists")
        record: dict[str, object] = {
            "format": "klab-attempt-phase-storage-v1",
            "label": label,
            "native_monitor_status": native_status,
            "baseline_sha256": baseline.sha256,
            "status": "WITHIN_CAP",
            "final_added_workspace_bytes": None,
            "final_added_workspace_inodes": None,
        }
        try:
            current_bytes, current_inodes = sample_workspace_tree(workspace_root)
            identity = workspace_root.stat()
            if (identity.st_dev, identity.st_ino) != (
                baseline.device,
                baseline.root_inode,
            ):
                raise ValueError("common-root identity changed before final receipt")
            for _ in range(16):
                pending = len(
                    (
                        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
                )
                added_bytes = max(0, current_bytes + pending - baseline.apparent_bytes)
                added_inodes = max(0, current_inodes + 1 - baseline.inodes)
                exceeded = (
                    policy.max_added_workspace_bytes is not None
                    and added_bytes > policy.max_added_workspace_bytes
                ) or (
                    policy.max_added_workspace_inodes is not None
                    and added_inodes > policy.max_added_workspace_inodes
                )
                updated = {
                    **record,
                    "status": "VIOLATED" if exceeded else "WITHIN_CAP",
                    "final_added_workspace_bytes": added_bytes,
                    "final_added_workspace_inodes": added_inodes,
                }
                if updated == record:
                    break
                record = updated
            else:
                raise RuntimeError("attempt receipt byte projection did not converge")
        except (OSError, RuntimeError, ValueError) as error:
            record["status"] = "UNAVAILABLE"
            record["reason"] = str(error)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        if record["status"] != "WITHIN_CAP":
            raise AttemptBudgetError(f"final common-root storage {record['status']}")
        return record


def _monitored_phase(
    command: list[str],
    *,
    policy: Path,
    policy_sha256: str,
    baseline: Path,
    baseline_sha256: str,
    workspace: Path,
    log_dir: Path,
) -> int:
    """Private child adapter: always route contract work through native monitor."""
    from klab.operational_monitor import (
        load_monitor_policy,
        load_workspace_baseline,
        monitor_command,
    )

    if hashlib.sha256(policy.read_bytes()).hexdigest() != policy_sha256:
        raise AttemptBudgetError("monitor policy changed before native launch")
    if load_workspace_baseline(baseline, workspace).sha256 != baseline_sha256:
        raise AttemptBudgetError("workspace baseline changed before native launch")

    result = monitor_command(
        command,
        load_monitor_policy(policy),
        workspace=workspace,
        log_dir=log_dir,
        policy_path=policy,
        baseline_path=baseline,
    )
    return 0 if result.status == "COMPLETE" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    private = commands.add_parser("_monitored_phase", help=argparse.SUPPRESS)
    private.add_argument("--policy", type=Path, required=True)
    private.add_argument("--policy-sha256", required=True)
    private.add_argument("--baseline", type=Path, required=True)
    private.add_argument("--baseline-sha256", required=True)
    private.add_argument("--workspace", type=Path, required=True)
    private.add_argument("--log-dir", type=Path, required=True)
    private.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.action == "_monitored_phase":
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            if not command:
                raise ValueError("monitored phase command is required")
            return _monitored_phase(
                command,
                policy=args.policy,
                policy_sha256=args.policy_sha256,
                baseline=args.baseline,
                baseline_sha256=args.baseline_sha256,
                workspace=args.workspace,
                log_dir=args.log_dir,
            )
    except (
        AttemptBudgetError,
        OSError,
        sqlite3.DatabaseError,
        ValueError,
    ) as error:
        print(f"attempt budget: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
