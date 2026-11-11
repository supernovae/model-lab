from __future__ import annotations

import copy
from pathlib import Path

from klab.config import load_config
from klab.experiments.evidence import _build_index, collect_evidence, read_evidence
from klab.experiments.lock import (
    ResolvedCell,
    ResolvedExperimentPlan,
    ResolvedPhase,
    _identities,
)
from klab.training.manifest import config_sha256, source_identity
from klab.workers.models import ExperimentSpec, current_required_versions

BASE = Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"


def _lock() -> ResolvedExperimentPlan:
    config = load_config(BASE)
    digest = config_sha256(config.model_dump(mode="json"))
    payload = {
        "lock_version": 1,
        "id": "evidence-test",
        "source_identity": source_identity(),
        "cells": [
            ResolvedCell.model_validate(
                {
                    "id": "main:single",
                    "coordinate": {},
                    "phase": "main",
                    "config": config.model_dump(mode="json"),
                    "config_sha256": digest,
                    "requested_runtime": config.runtime.model_dump(mode="json"),
                    "effective": {},
                    "artifacts": {},
                }
            )
        ],
        "inputs": {},
        "artifacts": {},
        "comparisons": [],
        "phases": [ResolvedPhase(id="main", transition="fresh")],
        "evaluations": [],
        "execution": {},
        "retention": {},
        "availability": {},
    }
    for field in ("cells", "comparisons", "phases", "evaluations"):
        payload[field] = tuple(payload[field])
    normalized = ResolvedExperimentPlan.model_construct(
        **payload, scientific_sha256="", plan_sha256=""
    ).model_dump(mode="json")
    scientific, plan = _identities(normalized)
    return ResolvedExperimentPlan.model_validate(
        {
            **normalized,
            "scientific_sha256": scientific,
            "plan_sha256": plan,
        }
    )


def _row(lock: ResolvedExperimentPlan, *, state: str = "QUEUED") -> dict:
    cell = lock.cells[0]
    metadata = {
        "plan_id": lock.id,
        "plan_sha256": lock.plan_sha256,
        "scientific_sha256": lock.scientific_sha256,
        "cell_id": cell.id,
        "phase_id": cell.phase,
        "coordinate": cell.coordinate,
        "config_sha256": cell.config_sha256,
    }
    spec = ExperimentSpec(
        experiment_id="experiment-one",
        config=cell.config,
        config_sha256=cell.config_sha256,
        source_identity_sha256=lock.source_identity["sha256"],
        dispatch_bundle_digest="a" * 64,
        required_versions=current_required_versions(),
        plan=metadata,
    )
    return {
        "experiment_id": "experiment-one",
        "attempt_id": "attempt-one",
        "run_id": "run-one",
        "worker_id": "worker-one",
        "spec": spec.model_dump(mode="json"),
        "status": state,
        "ingestion_status": "PENDING",
        "terminal_receipt": None,
    }


def _terminal(row: dict, state: str) -> dict:
    spec = ExperimentSpec.model_validate(row["spec"])
    return {
        "attempt_id": row["attempt_id"],
        "run_id": row["run_id"],
        "experiment_id": row["experiment_id"],
        "worker_id": row["worker_id"],
        "spec_digest": spec.digest(),
        "bundle_digest": spec.dispatch_bundle_digest,
        "state": state,
        "phase": "training",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:01Z",
        "artifacts": [],
    }


def test_missing_pending_failed_and_ingestion_error_are_distinct(
    tmp_path: Path,
) -> None:
    lock = _lock()
    missing = _build_index(lock, [], tmp_path)
    assert missing["cells"]["main:single"]["status"] == "pending"
    assert not missing["complete"]
    row = _row(lock)
    pending = _build_index(lock, [row], tmp_path)
    assert pending["cells"]["main:single"]["status"] == "pending"
    failed = copy.deepcopy(row)
    failed.update(status="FAILED", terminal_receipt=_terminal(row, "FAILED"))
    result = _build_index(lock, [failed], tmp_path)
    assert result["cells"]["main:single"]["status"] == "failed"
    failed["ingestion_status"] = "ERROR"
    failed["ingestion_error"] = "transfer incomplete"
    assert (
        _build_index(lock, [failed], tmp_path)["cells"]["main:single"]["status"]
        == "ingestion_error"
    )
    failed.update(
        status="INTERRUPTED",
        ingestion_status="NOT_REQUIRED",
        terminal_receipt=_terminal(row, "INTERRUPTED"),
    )
    assert (
        _build_index(lock, [failed], tmp_path)["cells"]["main:single"]["status"]
        == "censored"
    )


def test_plan_spec_and_receipt_mismatch_never_becomes_complete(tmp_path: Path) -> None:
    lock = _lock()
    wrong_spec = _row(lock, state="COMPLETE")
    wrong_spec["spec"]["plan"]["config_sha256"] = "b" * 64
    wrong_spec["terminal_receipt"] = _terminal(wrong_spec, "COMPLETE")
    wrong_spec["ingestion_status"] = "COMPLETE"
    cell = _build_index(lock, [wrong_spec], tmp_path)["cells"]["main:single"]
    assert cell["status"] == "invalid"
    assert "locked plan" in cell["attempts"][0]["reason"]
    wrong_receipt = _row(lock, state="COMPLETE")
    wrong_receipt.update(
        ingestion_status="COMPLETE",
        terminal_receipt=_terminal(wrong_receipt, "COMPLETE"),
    )
    wrong_receipt["terminal_receipt"]["run_id"] = "other-run"
    assert (
        _build_index(lock, [wrong_receipt], tmp_path)["cells"]["main:single"]["status"]
        == "invalid"
    )
    unrelated = _row(lock, state="FAILED")
    unrelated["spec"]["plan"]["plan_sha256"] = "f" * 64
    assert (
        _build_index(lock, [unrelated], tmp_path)["cells"]["main:single"]["attempts"]
        == []
    )


def test_immutable_index_replay_and_tamper_rejection(tmp_path: Path) -> None:
    lock = _lock()
    first = collect_evidence(lock, tmp_path)
    again = collect_evidence(lock, tmp_path)
    assert first["index_path"] == again["index_path"]
    assert read_evidence(lock, tmp_path, Path(first["index_path"])) == first
    index = Path(first["index_path"])
    index.write_text("{}\n")
    try:
        read_evidence(lock, tmp_path, index)
    except ValueError as error:
        assert "digest" in str(error)
    else:
        raise AssertionError("modified published evidence must be rejected")
