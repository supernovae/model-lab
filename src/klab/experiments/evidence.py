"""Read-only collection of lock-bound controller evidence and immutable snapshots.

An index records observations, not an effectiveness finding or promotion decision.
The controller's ingestion status is necessary but never sufficient: local bytes and
all three lock/spec/receipt/run identities are checked again at collection time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from klab.canonical import canonical_digest, canonical_json
from klab.data.verification import verify_file
from klab.evaluation.evidence import experiment_evidence
from klab.experiments.lock import ResolvedCell, ResolvedExperimentPlan
from klab.training.manifest import config_sha256, read_manifest
from klab.verification_proofs import (
    ProofStore,
    VerificationMode,
    file_binding,
    validate_mode,
)
from klab.workers.controller import Controller
from klab.workers.models import AttemptReceipt, ExperimentSpec

_FORMAT = "klab-experiment-evidence-index"


def _plan_metadata(lock: ResolvedExperimentPlan, cell: ResolvedCell) -> dict[str, Any]:
    return {
        "plan_id": lock.id,
        "plan_sha256": lock.plan_sha256,
        "scientific_sha256": lock.scientific_sha256,
        "cell_id": cell.id,
        "phase_id": cell.phase,
        "coordinate": cell.coordinate,
        "config_sha256": cell.config_sha256,
    }


def _compatible_execution_source(lock: ResolvedExperimentPlan, digest: str) -> bool:
    from klab.experiments.source_compatibility import source_identities_compatible

    return source_identities_compatible(str(lock.source_identity.get("sha256")), digest)


def _verify_attempt(
    lock: ResolvedExperimentPlan,
    cell: ResolvedCell,
    row: dict[str, Any],
    root: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    """Reject mismatched provenance even when the queue reports COMPLETE."""
    spec = ExperimentSpec.model_validate(row["spec"])
    plan = spec.plan
    expected = _plan_metadata(lock, cell)
    if plan is None or any(
        getattr(plan, key) != value for key, value in expected.items()
    ):
        raise ValueError("worker specification differs from locked plan/cell")
    if (
        spec.config.model_dump(mode="json") != cell.config.model_dump(mode="json")
        or spec.config_sha256 != cell.config_sha256
        or config_sha256(spec.config.model_dump(mode="json")) != cell.config_sha256
        or not _compatible_execution_source(lock, spec.source_identity_sha256)
        or spec.continuation.checkpoint_sha256 != plan.parent_checkpoint_sha256
    ):
        raise ValueError(
            "worker specification configuration/source/parent differs from lock"
        )
    if row["experiment_id"] != spec.experiment_id:
        raise ValueError("controller experiment differs from worker specification")
    result: dict[str, Any] = {
        "attempt_id": row["attempt_id"],
        "run_id": row["run_id"],
        "experiment_id": row["experiment_id"],
        "worker_id": row["worker_id"],
        "spec_sha256": spec.digest(),
        "status": "pending",
        "queue_status": row["status"],
        "ingestion_status": row["ingestion_status"],
        "parent_checkpoint_sha256": plan.parent_checkpoint_sha256,
        "execution_binding_sha256": plan.execution_binding_sha256,
        "runtime_binding_sha256": plan.runtime_binding_sha256,
    }
    receipt_value = row.get("terminal_receipt")
    if receipt_value is not None:
        receipt = AttemptReceipt.model_validate(receipt_value)
        if (
            any(
                getattr(receipt, name) != row[name]
                for name in ("attempt_id", "run_id", "experiment_id", "worker_id")
            )
            or receipt.spec_digest != spec.digest()
            or receipt.bundle_digest != spec.dispatch_bundle_digest
        ):
            raise ValueError(
                "terminal receipt differs from controller run/spec/dispatch"
            )
        result["receipt_sha256"] = canonical_digest(receipt.model_dump(mode="json"))
        result["receipt_state"] = receipt.state
        if row["status"] == "COMPLETE" and receipt.state != "COMPLETE":
            raise ValueError("controller completion disagrees with terminal receipt")
        if row["status"] in {
            "FAILED",
            "INTERRUPTED",
            "CANCELLED",
        } and receipt.state not in {"FAILED", "INTERRUPTED"}:
            raise ValueError("controller terminal state disagrees with receipt")
    elif row["status"] in {"COMPLETE", "FAILED", "INTERRUPTED", "CANCELLED"}:
        raise ValueError("terminal controller row has no terminal receipt")
    if row["ingestion_status"] == "ERROR":
        result["status"] = "ingestion_error"
        result["reason"] = row.get("ingestion_error") or "artifact ingestion failed"
        return result
    if row["status"] in {"FAILED", "INTERRUPTED", "CANCELLED"}:
        result["status"] = "failed" if row["status"] == "FAILED" else "censored"
        return result
    if row["status"] == "UNKNOWN":
        result["status"] = "unavailable"
        return result
    if row["status"] != "COMPLETE" or row["ingestion_status"] == "PENDING":
        return result
    if row["ingestion_status"] != "COMPLETE":
        raise ValueError("completed run has invalid ingestion state")
    if receipt_value is None:
        raise ValueError("completed run has no receipt")
    run = root / row["run_id"]
    if run.is_symlink() or not run.is_dir() or run.resolve().parent != root.resolve():
        raise ValueError("ingested run is absent or unsafe")
    manifest_path = run / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("ingested manifest is a symlink")
    manifest = read_manifest(manifest_path)
    dispatch = [
        value
        for value in manifest.get("resource_decisions", [])
        if isinstance(value, dict) and value.get("kind") == "worker_dispatch"
    ]
    if (
        any(
            manifest.get(name) != row[name]
            for name in ("run_id", "attempt_id", "experiment_id", "worker_id")
        )
        or manifest.get("requested_config") != spec.config.model_dump(mode="json")
        or manifest.get("requested_config_sha256") != cell.config_sha256
        or manifest.get("effective_config_sha256") != cell.config_sha256
        or manifest.get("source_identity", {}).get("sha256")
        != spec.source_identity_sha256
        or len(dispatch) != 1
        or dispatch[0].get("spec_digest") != spec.digest()
        or dispatch[0].get("bundle_digest") != spec.dispatch_bundle_digest
        or dispatch[0].get("plan") != plan.model_dump(mode="json")
        or manifest.get("checkpoint_sha256") != plan.parent_checkpoint_sha256
        or manifest.get("continuation_kind") != spec.continuation.kind
    ):
        raise ValueError("ingested manifest differs from lock/spec/receipt")
    resolved = run / "resolved_config.yaml"
    if (
        resolved.is_symlink()
        or not resolved.is_file()
        or json.loads(resolved.read_bytes()) != manifest["effective_config"]
    ):
        raise ValueError("ingested effective configuration differs from manifest")
    receipt = AttemptReceipt.model_validate(receipt_value)
    inventory: list[dict[str, Any]] = []
    for item in receipt.artifacts:
        if not item.relative_path.startswith("run/"):
            continue
        relative = item.relative_path.removeprefix("run/")
        path = run / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or run.resolve() not in path.resolve().parents
        ):
            raise ValueError(f"missing or unsafe ingested artifact {relative}")
        if path.stat().st_size != item.size_bytes:
            raise ValueError(
                f"ingested artifact digest differs from receipt: {relative}"
            )
        verify_file(
            path,
            expected_sha256=item.sha256,
            proof_store=proof_store,
            verification_mode=verification_mode,
            binding=file_binding(
                path,
                item.sha256,
                kind="ingested_run_artifact",
                identifier=relative,
                closure={
                    "receipt_sha256": result["receipt_sha256"],
                    "run_id": row["run_id"],
                    "artifacts": [
                        artifact.model_dump(mode="json")
                        for artifact in receipt.artifacts
                    ],
                },
            ),
        )
        inventory.append(item.model_dump(mode="json"))
    if not inventory:
        raise ValueError("completed run receipt has no ingested artifact inventory")
    observed = experiment_evidence(
        run, proof_store=proof_store, verification_mode=verification_mode
    )
    if not observed["verified_checkpoints"] or observed["run_id"] != row["run_id"]:
        raise ValueError("completed run lacks verified checkpoint evidence")
    result.update(
        status="complete",
        manifest_sha256=canonical_digest(manifest),
        artifacts=inventory,
        checkpoints=[
            {
                key: checkpoint[key]
                for key in (
                    "path",
                    "digest",
                    "step",
                    "tokens_seen",
                    "validation_loss",
                    "verification_scope",
                )
            }
            for checkpoint in observed["checkpoints"]
        ],
        quality_observations=observed["quality_observations"],
        rejected_reports=observed["rejected_reports"],
        missing_reports=observed["missing_reports"],
        evidence_level=observed["evidence_level"],
    )
    return result


def _build_index(
    lock: ResolvedExperimentPlan,
    rows: list[dict[str, Any]],
    controller_root: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    cells = {cell.id: cell for cell in lock.cells}
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in cells}
    for row in rows:
        plan = row.get("spec", {}).get("plan")
        if not isinstance(plan, dict) or plan.get("plan_sha256") != lock.plan_sha256:
            continue  # Other campaigns and old attempts never contaminate this lock.
        cell_id = plan.get("cell_id")
        if cell_id not in cells:
            raise ValueError(
                f"controller spec claims unknown cell {cell_id!r} of locked plan"
            )
        try:
            grouped[cell_id].append(
                _verify_attempt(
                    lock,
                    cells[cell_id],
                    row,
                    controller_root,
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            grouped[cell_id].append(
                {
                    "attempt_id": row.get("attempt_id"),
                    "run_id": row.get("run_id"),
                    "status": "invalid",
                    "reason": str(error),
                }
            )
    output: dict[str, dict[str, Any]] = {}
    for cell in lock.cells:
        attempts = sorted(
            grouped[cell.id],
            key=lambda item: (str(item["run_id"]), str(item["attempt_id"])),
        )
        states = {item["status"] for item in attempts}
        complete = [item for item in attempts if item["status"] == "complete"]
        if "invalid" in states or len(complete) > 1:
            status = "invalid"
        elif complete:
            status = "complete"
        else:
            status = next(
                (
                    state
                    for state in (
                        "ingestion_error",
                        "failed",
                        "censored",
                        "unavailable",
                        "pending",
                    )
                    if state in states
                ),
                "pending",
            )
        output[cell.id] = {
            "cell_id": cell.id,
            "phase_id": cell.phase,
            "coordinate": cell.coordinate,
            "config_sha256": cell.config_sha256,
            "status": status,
            "attempts": attempts,
            "selected_run_id": complete[0]["run_id"] if status == "complete" else None,
        }
    lookup = output
    comparisons = [
        {
            "id": comparison.id,
            "baseline": comparison.baseline,
            "variant": comparison.variant,
            "mode": comparison.mode,
            "interventions": list(comparison.interventions),
            "confounders": list(comparison.confounders),
            "differences": list(comparison.differences),
            "status": "available"
            if all(
                lookup[name]["status"] == "complete"
                for name in (comparison.baseline, comparison.variant)
            )
            else "unavailable",
            "baseline_run_id": lookup[comparison.baseline]["selected_run_id"],
            "variant_run_id": lookup[comparison.variant]["selected_run_id"],
        }
        for comparison in lock.comparisons
    ]
    return {
        "format": _FORMAT,
        "diagnostic_status": "not_collected",
        "diagnostic_evaluations": [],
        "plan_id": lock.id,
        "plan_sha256": lock.plan_sha256,
        "scientific_sha256": lock.scientific_sha256,
        "cells": output,
        "comparisons": comparisons,
        "complete": bool(output)
        and all(entry["status"] == "complete" for entry in output.values()),
        "interpretation": "Integrity-bound execution observations only; CPU smoke and validation metrics do not establish scientific effectiveness or authorize promotion.",
    }


def collect_evidence(
    lock: ResolvedExperimentPlan,
    workspace: Path,
    *,
    diagnostic_references: list[dict] | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    """Publish one immutable, content-addressed snapshot; identical calls reuse it.

    ``workspace`` is the plan workspace, with controller runs under ``controller/``.
    No controller tick, execution, download, or evaluation is performed here.
    """
    validate_mode(verification_mode)
    root = Path(workspace)
    controller = Controller(
        root / "controller",
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    payload = _build_index(
        lock,
        controller.list_experiments(),
        controller.root,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    if diagnostic_references is not None:
        from klab.experiments.diagnostic_evidence import bind_diagnostics

        payload = bind_diagnostics(
            lock,
            payload,
            root,
            diagnostic_references,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
    digest = canonical_digest(payload)
    directory = root / "evidence"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    encoded = canonical_json(payload) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != encoded:
            raise ValueError(
                "existing immutable evidence index differs from its digest"
            ) from None
    return {**payload, "index_sha256": digest, "index_path": str(path)}


def read_evidence(
    lock: ResolvedExperimentPlan,
    workspace: Path,
    index_path: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    """Re-verify a snapshot's identity and its referenced controller/run evidence."""
    validate_mode(verification_mode)
    path = Path(index_path)
    directory = Path(workspace) / "evidence"
    if path.is_symlink() or path.parent.resolve() != directory.resolve():
        raise ValueError("evidence index is outside plan workspace or is a symlink")
    raw = path.read_bytes()
    payload = json.loads(raw)
    digest = canonical_digest(payload)
    if path.name != f"{digest}.json" or raw != canonical_json(payload) + b"\n":
        raise ValueError("evidence index digest or canonical encoding differs")
    if (
        payload.get("format") != _FORMAT
        or "version" in payload
        or payload.get("diagnostic_status") not in {"not_collected", "collected"}
        or not isinstance(payload.get("diagnostic_evaluations"), list)
    ):
        raise ValueError(
            "obsolete or unsupported evidence contract; collect fresh evidence with the current lab"
        )
    if payload.get("plan_sha256") != lock.plan_sha256:
        raise ValueError("evidence index belongs to a different locked plan")
    controller = Controller(Path(workspace) / "controller", read_only=True)
    rows = controller.list_experiments()
    indexed = {
        attempt["attempt_id"]
        for cell in payload.get("cells", {}).values()
        for attempt in cell.get("attempts", [])
    }
    selected = [row for row in rows if row["attempt_id"] in indexed]
    rebuilt = _build_index(
        lock,
        selected,
        controller.root,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    if payload["diagnostic_status"] == "collected":
        from klab.experiments.diagnostic_evidence import bind_diagnostics

        rebuilt = bind_diagnostics(
            lock,
            rebuilt,
            Path(workspace),
            [item["reference"] for item in payload["diagnostic_evaluations"]],
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
    if rebuilt != payload:
        raise ValueError(
            "indexed controller/spec/run evidence changed or is unavailable"
        )
    return {**payload, "index_sha256": digest, "index_path": str(path)}
