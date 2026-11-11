"""Evidence-only publication drafts; no collection, execution or promotion."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Literal

from klab.experiments.lock import open_lock
from klab.experiments.retrospective import retrospective_views
from klab.result_publication import write_bytes_exclusive

FORMAT = "klab-publication-report"
_LIMITATIONS = (
    "This is a draft, not an external publication or promotion decision.",
    "Coverage refers to the explicitly selected snapshot, not a live or latest experiment state.",
    "Execution completeness and verified evidence integrity do not establish scientific effectiveness.",
    "A comparison marked available means both executions completed; it does not establish scientific comparability or a winner.",
    "Validation observations do not prove untouched held-out separation or general model quality. Single-seed noise is not across-seed uncertainty.",
    "Checkpoint verification_scope is preserved; weights verification does not establish full-state resume availability.",
    "Licensing, access rights, redistributability and distributed weights/resume availability are unknown in this evidence contract.",
    "Local paths, arbitrary source metadata, free-form diagnostics and comparison difference values are omitted. Original identity hashes are unchanged; this projection is not the original evidence.",
    "No training, evaluation, acquisition, controller mutation or missing-work execution is performed by rendering.",
)


def _fields(value: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    """Only named scalar fields cross the export boundary, never nested metadata."""
    return {
        name: value[name]
        for name in names
        if name in value
        and (value[name] is None or type(value[name]) in (str, int, float, bool))
    }


def _attempt(value: dict[str, Any]) -> dict[str, Any]:
    result = _fields(
        value,
        (
            "attempt_id",
            "run_id",
            "experiment_id",
            "spec_sha256",
            "status",
            "queue_status",
            "ingestion_status",
            "receipt_sha256",
            "receipt_state",
            "parent_checkpoint_sha256",
            "execution_binding_sha256",
            "runtime_binding_sha256",
            "manifest_sha256",
            "evidence_level",
        ),
    )
    result["checkpoints"] = [
        _fields(
            item,
            ("digest", "step", "tokens_seen", "validation_loss", "verification_scope"),
        )
        for item in value.get("checkpoints", [])
    ]
    result["quality_observations"] = [
        {
            **_fields(
                item,
                (
                    "sha256",
                    "kind",
                    "checkpoint_sha256",
                    "step",
                    "tokens_seen",
                    "loss",
                    "perplexity",
                    "valid_targets",
                    "batches",
                    "batch_size",
                    "max_batches",
                    "seq_len",
                ),
            ),
            "identities": _fields(
                item.get("identities", {}),
                (
                    "protocol",
                    "data/validation.npy",
                    "tokenizer.json",
                    "data/validation_supervision.npy",
                    "data/validation_byte_addresses.npy",
                    "source_identity_sha256",
                ),
            ),
        }
        for item in value.get("quality_observations", [])
    ]
    result["missing_report_count"] = len(value.get("missing_reports", []))
    result["rejected_report_count"] = len(value.get("rejected_reports", []))
    result["diagnostic_omitted"] = "reason" in value
    return result


def build_publication_report(
    lock_path: Path,
    *,
    workspace: Path | None = None,
    index_path: Path | None = None,
) -> dict[str, Any]:
    """Reopen a native lock and project explicitly selected, verified evidence.

    Omit both evidence arguments for a planned-only draft. Supplying either requires
    both. Mutable dictionaries and in-memory resolved plans are not evidence inputs.
    Verification errors propagate; they are never downgraded to missing evidence.
    """
    if (workspace is None) != (index_path is None):
        raise ValueError("workspace and index_path must be supplied together")
    lock = open_lock(Path(lock_path))
    retrospective = retrospective_views(
        lock, workspace=workspace, index_path=index_path
    )
    views = retrospective["views"]
    coverage = views["evidence_coverage"]["facts"]
    lineage = {item["cell_id"]: item for item in views["phased_lineage"]["facts"]}
    observations = {item["cell_id"]: item for item in coverage["cells"]}
    cells = []
    for cell in lock.cells:
        row = observations[cell.id]
        evidence = row["evidence"] or {}
        cells.append(
            {
                "cell_id": cell.id,
                "phase": cell.phase,
                "config_sha256": cell.config_sha256,
                "seed": cell.config.seed,
                "planned_budget": _fields(
                    cell.config.training.model_dump(mode="json"),
                    (
                        "max_steps",
                        "max_tokens",
                        "seq_len",
                        "micro_batch_size",
                        "gradient_accumulation",
                    ),
                ),
                "planned_evaluation": _fields(
                    cell.config.evaluation.model_dump(mode="json"), ("max_batches",)
                ),
                "inputs": [
                    _fields(identity, ("kind", "version", "sha256"))
                    for _, identity in sorted(cell.artifacts.items())
                ],
                "status": row["status"],
                "selected_run_id": evidence.get("selected_run_id"),
                "lineage": _fields(
                    lineage[cell.id],
                    (
                        "transition",
                        "parent_cell_id",
                        "selector",
                        "at_step",
                        "observed_parent_checkpoint_sha256",
                    ),
                ),
                "attempts": [
                    _attempt(attempt) for attempt in evidence.get("attempts", [])
                ],
            }
        )
    observed_comparisons = {item["id"]: item for item in coverage["comparisons"]}
    readiness = views["readiness_decision"]["facts"]
    report = {
        "format": FORMAT,
        "diagnostic_status": "not_collected",
        "diagnostic_evaluations": [],
        "provenance": {
            "plan_id": lock.id,
            "plan_sha256": lock.plan_sha256,
            "scientific_sha256": lock.scientific_sha256,
            "source_identity_sha256": lock.source_identity.get("sha256"),
            "evidence_index_sha256": coverage["evidence_index_sha256"],
            "identity_policy": "original_hashes_allowlisted_projection",
        },
        "scope": retrospective["scope"],
        "readiness": {
            "execution_coverage": readiness["execution_coverage"],
            "evidence_integrity": "verified_snapshot"
            if index_path is not None
            else "no_snapshot",
            "engineering_smoke": "not_assessed",
            "scientific_result": "not_assessed",
            "decision": readiness["decision"],
            "unassessed_gate_count": len(readiness["unassessed_preregistered_gates"]),
        },
        "cells": cells,
        "comparisons": [
            {
                "id": comparison.id,
                "baseline_cell_id": comparison.baseline,
                "variant_cell_id": comparison.variant,
                "planned_mode": comparison.mode,
                "recorded_status": observed_comparisons.get(comparison.id, {}).get(
                    "status", "unavailable"
                ),
                "scientific_comparability": "not_assessed",
                "winner": None,
                "baseline_run_id": observed_comparisons.get(comparison.id, {}).get(
                    "baseline_run_id"
                ),
                "variant_run_id": observed_comparisons.get(comparison.id, {}).get(
                    "variant_run_id"
                ),
                "declared_intervention_count": len(comparison.interventions),
                "declared_invariant_count": len(comparison.invariants),
                "declared_confounder_count": len(comparison.confounders),
            }
            for comparison in lock.comparisons
        ],
        "availability": {
            "dataset_license": "unknown",
            "dataset_access": "unknown",
            "redistributability": "unknown",
            "weights_license": "unknown",
            "weights_distribution": "unknown",
            "full_state_resume": "unknown",
        },
        "limitations": list(_LIMITATIONS),
    }

    if index_path is not None:
        from klab.experiments.diagnostic_evidence import report_diagnostics
        from klab.experiments.evidence import read_evidence

        evidence = read_evidence(lock, workspace, index_path)
        report["diagnostic_status"] = evidence["diagnostic_status"]
        report["diagnostic_evaluations"] = report_diagnostics(evidence)
    return report


def _json(report: dict[str, Any]) -> bytes:
    return (
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _markdown(report: dict[str, Any]) -> bytes:
    # Escape source-authored identifiers as inert text, including Markdown punctuation.
    def literal(value: Any) -> str:
        text = html.escape(str(value), quote=True)
        for character in "\\`*_{}[]()#+-.!|":
            text = text.replace(character, "\\" + character)
        return text.replace("\n", " ").replace("\r", " ")

    lines = [
        "# Experiment report / model-card draft",
        "",
        f"Format: {report['format']}",
        "",
        "## Provenance",
        "",
    ]
    lines.extend(
        f"- {key}: {literal(value)}" for key, value in report["provenance"].items()
    )
    lines.extend(["", "## Evidence and readiness", "", f"Scope: {report['scope']}", ""])
    lines.extend(
        f"- {key}: {literal(value)}" for key, value in report["readiness"].items()
    )
    lines.extend(
        ["", "| Cell | Recorded status | Selected run |", "| --- | --- | --- |"]
    )
    lines.extend(
        f"| {literal(cell['cell_id'])} | {literal(cell['status'])} | {literal(cell['selected_run_id'] or 'unavailable')} |"
        for cell in report["cells"]
    )
    for cell in report["cells"]:
        lines.extend(
            [
                "",
                f"### Cell {literal(cell['cell_id'])}",
                "",
                f"Recorded status: {literal(cell['status'])}",
                "",
            ]
        )
        # The fenced payload is the exact allowlisted record, with fences/HTML escaped
        # by JSON's ASCII encoding and explicit angle/backtick escaping below.
        payload = json.dumps(
            cell, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
        )
        payload = (
            payload.replace("`", "\\u0060")
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
        )
        lines.extend(["```json", payload, "```"])
    lines.extend(["", "## Comparisons", ""])
    for comparison in report["comparisons"]:
        lines.append(
            f"- {literal(comparison['id'])}: {literal(comparison['baseline_cell_id'])} vs {literal(comparison['variant_cell_id'])}; recorded status {literal(comparison['recorded_status'])}; planned mode {literal(comparison['planned_mode'])}; scientific comparability not assessed; no winner inferred."
        )
    if not report["comparisons"]:
        lines.append("No comparisons recorded.")
    lines.extend(["", "## Availability and rights", ""])
    lines.extend(f"- {key}: {value}" for key, value in report["availability"].items())
    lines.extend(["", "## Recorded diagnostic evaluations", ""])
    lines.append(f"Coverage: {literal(report['diagnostic_status'])}")
    for item in report["diagnostic_evaluations"]:
        lines.append(
            f"- {literal(item['cell_id'])}: {literal(item['status'])}; index {literal(item['index_sha256'])}; metrics {literal(item['metrics'])}"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_publication_report(
    lock_path: Path,
    output: Path,
    *,
    workspace: Path | None = None,
    index_path: Path | None = None,
    format: Literal["json", "markdown"] = "json",
) -> Path:
    """Verify then durably publish one draft; identical bytes replay, others refuse.

    Each file is independently atomic. Call twice for JSON and Markdown; if the
    second fails, retain the first and safely replay it after resolving the failure.
    """
    if format not in ("json", "markdown"):
        raise ValueError("publication format must be json or markdown")
    report = build_publication_report(
        lock_path, workspace=workspace, index_path=index_path
    )
    encoded = _json(report) if format == "json" else _markdown(report)
    destination = Path(output)
    try:
        write_bytes_exclusive(destination, encoded)
    except FileExistsError:
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != encoded
        ):
            raise ValueError(
                "publication destination already exists with different bytes or is unsafe"
            ) from None
    return destination
