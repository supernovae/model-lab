"""Bind existing diagnostic suite results to an execution evidence snapshot.

This module verifies recorded results; it never runs an evaluation or discovers
new results by globbing a mutable run directory.
"""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import Field, StrictStr

from klab.config.models import StrictModel


class DiagnosticReference(StrictModel):
    stage: StrictStr
    cell_id: StrictStr
    run_id: StrictStr
    checkpoint: StrictStr
    checkpoint_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    suite: Path
    suite_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    index: Path
    index_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


def bind_diagnostics(
    lock, payload: dict, workspace: Path, references: list[dict], **verification
) -> dict:
    """Require one explicitly selected diagnostic index per completed cell."""
    from klab.evaluation.suite import verify_evaluation_index

    refs = [DiagnosticReference.model_validate(value) for value in references]
    if (
        not refs
        or len({ref.stage for ref in refs}) != len(refs)
        or len({ref.cell_id for ref in refs}) != len(refs)
        or {ref.cell_id for ref in refs} != set(payload["cells"])
    ):
        raise ValueError("final evidence requires exactly one diagnostic per cell")
    observations = []
    for ref in sorted(refs, key=lambda ref: ref.stage):
        if (
            lock.evaluation_suite is None
            or ref.suite_sha256 != lock.evaluation_suite["sha256"]
            or ref.suite != Path(lock.availability["evaluation_suite"])
        ):
            raise ValueError("diagnostic suite differs from locked declaration")
        cell = payload["cells"][ref.cell_id]
        run = Path(workspace) / "controller" / ref.run_id
        if cell["status"] != "complete" or cell["selected_run_id"] != ref.run_id:
            raise ValueError("diagnostic run differs from completed evidence cell")
        if (
            Path(ref.run_id).name != ref.run_id
            or ref.run_id in {".", ".."}
            or not ref.index.is_absolute()
            or not ref.suite.is_absolute()
            or any(
                p.is_symlink()
                for p in (ref.index, *ref.index.parents, ref.suite, *ref.suite.parents)
            )
            or ref.index.parent.resolve() != (run / "evaluations").resolve()
        ):
            raise ValueError("diagnostic reference has unsafe or mismatched locations")
        attempt = next(a for a in cell["attempts"] if a["status"] == "complete")
        if not any(
            c["digest"] == ref.checkpoint_sha256
            and f"checkpoints/{c['path']}" == ref.checkpoint
            for c in attempt["checkpoints"]
        ):
            raise ValueError("diagnostic checkpoint is absent from execution evidence")
        index = verify_evaluation_index(ref.index, **verification)
        if (
            index["run_id"] != ref.run_id
            or Path(index["run"]).resolve() != run.resolve()
            or index["checkpoint"] != ref.checkpoint
            or index["checkpoint_sha256"] != ref.checkpoint_sha256
            or Path(index["suite"]).absolute() != ref.suite
            or index["suite_sha256"] != ref.suite_sha256
            or index["index_sha256"] != ref.index_sha256
        ):
            raise ValueError(
                "diagnostic index differs from declared run/checkpoint/suite/result identity"
            )
        rows = index["evaluations"]
        if (
            len(rows) != 1
            or rows[0]["kind"] != "heldout_lm"
            or rows[0]["role"] != "diagnostic"
        ):
            raise ValueError("final evidence supports one diagnostic heldout_lm result")
        row = rows[0]
        if row["status"] == "COMPLETED":
            metrics = row["result"]
            if not isinstance(metrics, dict):
                raise ValueError("diagnostic metrics must be an object")
            for key in ("loss", "perplexity"):
                value = metrics.get(key)
                if key == "perplexity" and value is None:
                    continue
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("diagnostic metrics must be finite numbers")
            for key in ("batches", "valid_targets"):
                if type(metrics.get(key)) is not int or metrics[key] <= 0:
                    raise ValueError("diagnostic counters must be positive integers")
        observations.append(
            {
                "reference": ref.model_dump(mode="json"),
                "record_sha256": index["record_sha256"],
                "evaluation_id": row["id"],
                "status": row["status"],
                "result_sha256": row["sha256"],
                "result": row["result"],
                "reason_present": row.get("reason") is not None,
            }
        )
    return {
        **payload,
        "diagnostic_status": "collected",
        "diagnostic_evaluations": observations,
    }


def report_diagnostics(evidence: dict) -> list[dict]:
    """Project verified evidence without exposing local paths or failure text."""
    result = []
    for item in evidence["diagnostic_evaluations"]:
        ref = item["reference"]
        metrics = item["result"] if item["status"] == "COMPLETED" else {}
        result.append(
            {
                **{
                    key: ref[key]
                    for key in (
                        "stage",
                        "cell_id",
                        "run_id",
                        "checkpoint_sha256",
                        "suite_sha256",
                        "index_sha256",
                    )
                },
                **{
                    key: item[key]
                    for key in (
                        "record_sha256",
                        "evaluation_id",
                        "status",
                        "result_sha256",
                        "reason_present",
                    )
                },
                "metrics": {
                    key: metrics[key]
                    for key in ("loss", "perplexity", "batches", "valid_targets")
                    if key in metrics
                },
                "interpretation": "diagnostic_only",
            }
        )
    return result
