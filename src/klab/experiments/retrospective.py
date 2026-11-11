"""Five identity-bound, non-promotional views over planned or collected experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from klab.experiments.lock import ResolvedExperimentPlan
from klab.verification_proofs import ProofStore, VerificationMode, validate_mode

_FORMAT = "klab-experiment-retrospective-v1"


def retrospective_views(
    lock: ResolvedExperimentPlan,
    *,
    workspace: Path | None = None,
    index_path: Path | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    """Describe a resolved plan; optionally read a *verified* collected evidence index.

    A caller must resolve or reopen the lock before calling this function. Collected
    outcomes are accepted only via ``read_evidence`` with an explicit workspace and
    content-addressed index path; an unverified dict cannot supply observations.
    """
    validate_mode(verification_mode)
    lock = ResolvedExperimentPlan.model_validate(lock.model_dump(mode="json"))
    if (workspace is None) != (index_path is None):
        raise ValueError(
            "collected retrospective requires workspace and index_path together"
        )
    evidence: dict[str, Any] | None = None
    if index_path is not None:
        from klab.experiments.evidence import read_evidence

        evidence = read_evidence(
            lock,
            Path(workspace),
            Path(index_path),
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        if (
            evidence.get("plan_sha256") != lock.plan_sha256
            or evidence.get("scientific_sha256") != lock.scientific_sha256
        ):
            raise ValueError("retrospective evidence plan/scientific identity mismatch")

    provenance = {
        "plan_id": lock.id,
        "plan_sha256": lock.plan_sha256,
        "scientific_sha256": lock.scientific_sha256,
    }
    identity = {
        "facts": {
            **provenance,
            "source_identity": lock.source_identity,
            "inputs": lock.inputs,
            "artifacts": lock.artifacts,
            "cells": [
                {
                    "id": cell.id,
                    "coordinate": cell.coordinate,
                    "phase": cell.phase,
                    "config_sha256": cell.config_sha256,
                    "requested_runtime": cell.requested_runtime,
                    "effective": cell.effective,
                    "artifacts": cell.artifacts,
                }
                for cell in lock.cells
            ],
        },
        "inferences": [],
    }
    contrast = {
        "facts": [
            {
                "id": comparison.id,
                "baseline": comparison.baseline,
                "variant": comparison.variant,
                "mode": comparison.mode,
                "interventions": list(comparison.interventions),
                "invariants": list(comparison.invariants),
                "confounders": list(comparison.confounders),
                "differences": list(comparison.differences),
            }
            for comparison in lock.comparisons
        ],
        "inferences": [
            "A planned controlled contrast is a design assertion, not observed causal effectiveness."
        ],
    }
    cells = {
        (cell.phase, tuple(sorted(cell.coordinate.items()))): cell.id
        for cell in lock.cells
    }
    observations = evidence.get("cells", {}) if evidence else {}
    if not isinstance(observations, dict):
        raise TypeError("verified evidence cells must be keyed by locked cell ID")
    if set(observations) - {cell.id for cell in lock.cells}:
        raise ValueError("verified evidence contains an unknown locked cell")
    phases = {phase.id: phase for phase in lock.phases}
    lineage = {
        "facts": [
            {
                "cell_id": cell.id,
                "phase": cell.phase,
                "transition": phase.transition,
                "parent_cell_id": cells.get(
                    (phase.parent, tuple(sorted(cell.coordinate.items())))
                )
                if phase.parent is not None
                else None,
                "external_checkpoint": phase.checkpoint,
                "selector": phase.selector,
                "at_step": phase.at_step,
                "parent_artifact": cell.artifacts.get("parent_checkpoint"),
                "observed_parent_checkpoint_sha256": next(
                    (
                        attempt.get("parent_checkpoint_sha256")
                        for attempt in observations.get(cell.id, {}).get("attempts", [])
                        if attempt.get("run_id")
                        == observations.get(cell.id, {}).get("selected_run_id")
                    ),
                    None,
                ),
                "selected_run_id": observations.get(cell.id, {}).get("selected_run_id"),
            }
            for cell in lock.cells
            for phase in (phases[cell.phase],)
        ],
        "inferences": [
            "A planned parent selector is not proof that a checkpoint was produced or bound."
        ],
    }

    coverage_cells = []
    for cell in lock.cells:
        observation = observations.get(cell.id)
        coverage_cells.append(
            {
                "cell_id": cell.id,
                "status": (
                    observation.get("status", "pending") if observation else "pending"
                ),
                "evidence": observation,
            }
        )
    coverage = {
        "facts": {
            "evidence_index_sha256": evidence.get("index_sha256") if evidence else None,
            "cells": coverage_cells,
            "comparisons": evidence.get("comparisons", []) if evidence else [],
            "evaluations": [
                {"declaration": item, "status": "unassessed"}
                for item in lock.evaluations
            ],
        },
        "inferences": [
            "Missing evidence is pending/unavailable, never a zero score or a negative result."
        ],
    }
    successful = bool(coverage_cells) and all(
        row["status"] == "complete" for row in coverage_cells
    )
    missing = [
        {"cell_id": row["cell_id"], "status": row["status"]}
        for row in coverage_cells
        if row["status"] != "complete"
    ]
    gates = [item for item in lock.evaluations if item.get("role") == "gate"]
    readiness = {
        "facts": {
            "execution_coverage": "complete" if successful else "incomplete",
            "missing_or_noncomplete_cells": missing,
            "unassessed_preregistered_gates": [item["id"] for item in gates],
            "retention": lock.retention,
            "decision": "no_automatic_promotion",
        },
        "inferences": [
            (
                "Verified execution alone does not establish scientific effectiveness; "
                "a CPU smoke is software evidence only."
            ),
            (
                "Promotion requires independent reviewed lifecycle evidence and a human decision; "
                "this view never promotes a baseline."
            ),
        ],
    }
    return {
        "format": _FORMAT,
        "plan_sha256": lock.plan_sha256,
        "scientific_sha256": lock.scientific_sha256,
        "scope": "collected" if evidence else "planned_only",
        "views": {
            "identity_provenance": identity,
            "contrast_interventions": contrast,
            "phased_lineage": lineage,
            "evidence_coverage": coverage,
            "readiness_decision": readiness,
        },
    }
