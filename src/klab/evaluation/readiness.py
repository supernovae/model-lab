"""Typed readiness decisions and explicit, single-reviewer human review receipts."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from klab.campaign.state import publish_immutable, read_canonical, utc_now
from klab.canonical import canonical_digest, canonical_json
from klab.config.models import StrictModel
from klab.documents import read_document
from klab.evaluation.suite import load_suite, verify_evaluation_index
from klab.training.manifest import sha256_file

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class LossCriterion(StrictModel):
    evaluation_id: str
    value: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("evaluation_id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("invalid loss evaluation ID")
        return value


class ReviewedScoreCriterion(StrictModel):
    """Predeclared item and formatting thresholds for independent panel review."""

    frozen_content_sha256: str
    expected_items: int = Field(gt=0)
    expected_axes: int = Field(gt=0)
    items_per_axis: int = Field(gt=0)
    min_correct: int = Field(ge=0)
    min_axis_correct: int = Field(ge=0)
    min_format: int = Field(ge=0)
    min_axis_format: int = Field(ge=0)

    @field_validator("frozen_content_sha256")
    @classmethod
    def valid_digest(cls, value: str) -> str:
        if not _SHA.fullmatch(value):
            raise ValueError("frozen item digest must be SHA-256")
        return value

    @model_validator(mode="after")
    def valid_denominators(self) -> ReviewedScoreCriterion:
        if (
            self.expected_axes * self.items_per_axis != self.expected_items
            or self.min_correct > self.expected_items
            or self.min_format > self.expected_items
            or self.min_axis_correct > self.items_per_axis
            or self.min_axis_format > self.items_per_axis
        ):
            raise ValueError("reviewed-score thresholds exceed denominators")
        return self


class ModelReadinessPolicy(StrictModel):
    readiness_version: Literal[1]
    id: str
    require_verified_checkpoint: Literal[True]
    required_gate_ids: tuple[str, ...]
    min_completed_evaluations: int = Field(gt=0)
    max_heldout_loss: LossCriterion | None = None
    require_human_review: bool
    reviewed_scores: ReviewedScoreCriterion | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("invalid readiness policy ID")
        return value

    @model_validator(mode="after")
    def valid_gate_ids(self) -> ModelReadinessPolicy:
        if len(self.required_gate_ids) != len(set(self.required_gate_ids)) or any(
            not _ID.fullmatch(value) for value in self.required_gate_ids
        ):
            raise ValueError("gate IDs must be unique safe identifiers")
        if (
            self.max_heldout_loss is not None
            and self.max_heldout_loss.evaluation_id not in self.required_gate_ids
        ):
            raise ValueError("loss threshold must target a required gate")
        return self


def load_policy(path: Path) -> ModelReadinessPolicy:
    return ModelReadinessPolicy.model_validate(read_document(path))


def _index_path(index: dict[str, Any] | Path) -> Path:
    if isinstance(index, Path):
        return index
    if not isinstance(index, dict) or "run" not in index or "index_sha256" not in index:
        raise ValueError("invalid evaluation index")
    return Path(index["run"]) / "evaluations" / f"suite-{index['index_sha256']}.json"


def inspect_observation_coverage(
    verified_index: Path, panel_result: Path | None = None
) -> dict[str, Any]:
    """Report execution coverage without granting model readiness.

    A completed suite or descriptive generation panel contains no independently
    reviewed item scores. This check deliberately has no promotion state; the
    reviewed-score criterion and named human review are separate evidence.
    """
    index = verify_evaluation_index(Path(verified_index))
    evaluations = index["evaluations"]
    completed = sum(row["status"] == "COMPLETED" for row in evaluations)
    observation: dict[str, Any] = {
        "index_sha256": index["index_sha256"],
        "checkpoint_sha256": index["checkpoint_sha256"],
        "suite_completed": completed,
        "suite_total": len(evaluations),
        "panel_completed": None,
        "panel_total": None,
        "reviewed_scores": "MISSING",
        "reader_eligibility": "UNESTABLISHED",
    }
    if panel_result is not None:
        from klab.evaluation.panel import verify_panel_result

        panel = verify_panel_result(Path(panel_result))
        if (
            panel["evaluation_index_sha256"] != index["index_sha256"]
            or panel["checkpoint_sha256"] != index["checkpoint_sha256"]
        ):
            raise ValueError(
                "panel and evaluation index identify different checkpoints"
            )
        observation["panel_completed"] = sum(
            row["status"] == "COMPLETED" for row in panel["rows"]
        )
        observation["panel_total"] = len(panel["rows"])
    return observation


def _review_identity(record: dict[str, Any]) -> str:
    science = {
        key: value
        for key, value in record.items()
        if key not in {"receipt_sha256", "index_path", "surface_bundle"}
    }
    return canonical_digest(science)


def _readiness_identity(record: dict[str, Any]) -> str:
    science = {
        key: value
        for key, value in record.items()
        if key not in {"result_sha256", "assessed_at_utc", "policy", "index", "review"}
    }
    return canonical_digest(science)


def verify_review_receipt(path: Path) -> dict[str, Any]:
    path = Path(path)
    receipt = read_canonical(path)
    if receipt.get("format") != "model-review-receipt-v1":
        raise ValueError("invalid model review receipt")
    expected = receipt.get("receipt_sha256")
    if (
        not isinstance(expected, str)
        or not _SHA.fullmatch(expected)
        or _review_identity(receipt) != expected
    ):
        raise ValueError("model review identity mismatch")
    if (
        not isinstance(receipt.get("reviewer"), str)
        or not _ID.fullmatch(receipt["reviewer"])
        or receipt.get("decision") not in {"approve", "reject"}
        or not isinstance(receipt.get("note"), str)
        or not receipt["note"].strip()
    ):
        raise ValueError("invalid named human review decision")
    index = verify_evaluation_index(Path(receipt["index_path"]))
    if (
        index["index_sha256"] != receipt["index_sha256"]
        or index["checkpoint_sha256"] != receipt["checkpoint_sha256"]
    ):
        raise ValueError("review checkpoint or evaluation index mismatch")
    if receipt["surface_review_digest"] is not None:
        from klab.evaluation.suite import _surface

        bundle = Path(receipt["surface_bundle"])
        reviewed_sha, _ = _surface(bundle, receipt["checkpoint_sha256"])
        if reviewed_sha != receipt["surface_review_digest"]:
            raise ValueError("Surface Review changed")
    elif receipt["surface_bundle"] is not None:
        raise ValueError("unverified Surface Review reference")
    return receipt


def issue_review(
    index: Path,
    reviewer: str,
    decision: Literal["approve", "reject"],
    note: str,
    output: Path,
    surface_bundle: Path | None = None,
) -> dict[str, Any]:
    verified = verify_evaluation_index(Path(index))
    if (
        not _ID.fullmatch(reviewer)
        or decision not in {"approve", "reject"}
        or not note.strip()
    ):
        raise ValueError("human review requires named reviewer, decision and note")
    surface_digest = None
    if surface_bundle is not None:
        from klab.evaluation.suite import _surface

        surface_digest, _ = _surface(
            Path(surface_bundle), verified["checkpoint_sha256"]
        )
    body = {
        "format": "model-review-receipt-v1",
        "index_sha256": verified["index_sha256"],
        "index_path": str(Path(index).resolve()),
        "checkpoint_sha256": verified["checkpoint_sha256"],
        "reviewer": reviewer,
        "decision": decision,
        "note": note,
        "surface_bundle": str(Path(surface_bundle).resolve())
        if surface_bundle
        else None,
        "surface_review_digest": surface_digest,
    }
    output = Path(output)
    if output.exists():
        old = verify_review_receipt(output)
        if {key: old[key] for key in body} != body:
            raise ValueError("conflicting human review receipt")
        return old
    body["issued_at_utc"] = utc_now()
    receipt = {**body, "receipt_sha256": _review_identity(body)}
    publish_immutable(output, receipt)
    return receipt


def _validate_policy(policy: ModelReadinessPolicy, index: dict[str, Any]) -> None:
    suite = load_suite(Path(index["suite"]))
    evaluations = {item.id: item for item in suite.evaluations}
    if policy.min_completed_evaluations > len(evaluations):
        raise ValueError("minimum evaluation coverage exceeds suite size")
    for gate in policy.required_gate_ids:
        if gate not in evaluations or evaluations[gate].role != "gate":
            raise ValueError(f"required gate is not a declared gate: {gate}")
    if policy.max_heldout_loss is not None:
        target = evaluations[policy.max_heldout_loss.evaluation_id]
        if target.kind != "heldout_lm":
            raise ValueError("loss threshold must target heldout_lm gate")


def _decision(
    policy: ModelReadinessPolicy,
    index: dict[str, Any],
    review: dict[str, Any] | None,
    scores: dict[str, Any] | None = None,
) -> tuple[str, list[str], int]:
    rows = {row["id"]: row for row in index["evaluations"]}
    missing = [
        gate for gate in policy.required_gate_ids if rows[gate]["status"] != "COMPLETED"
    ]
    completed = sum(row["status"] == "COMPLETED" for row in rows.values())
    breached = False
    threshold = policy.max_heldout_loss
    if threshold is not None and rows[threshold.evaluation_id]["status"] == "COMPLETED":
        metrics = rows[threshold.evaluation_id]["result"]
        loss = metrics.get("loss") if isinstance(metrics, dict) else None
        if (
            not isinstance(loss, (int, float))
            or isinstance(loss, bool)
            or not math.isfinite(loss)
            or loss < 0
        ):
            raise ValueError(
                "completed heldout evaluation lacks a finite nonnegative loss"
            )
        breached = loss > threshold.value
    if review is not None and review["decision"] == "reject":
        state = "DO_NOT_ADVANCE"
    elif missing or completed < policy.min_completed_evaluations:
        state = "INCONCLUSIVE"
    elif breached:
        state = "DO_NOT_ADVANCE"
    elif policy.reviewed_scores is not None and scores is None:
        state = "INCONCLUSIVE"
    elif scores is not None and not scores["passed"]:
        state = "DO_NOT_ADVANCE"
    elif policy.require_human_review and review is None:
        state = "NEEDS_REVIEW"
    else:
        state = "READY_FOR_NEXT_STAGE"
    return state, missing, completed


def assess_readiness(
    policy: Path | ModelReadinessPolicy,
    verified_index: Path | dict[str, Any],
    human_review_receipt: Path | None = None,
    reviewed_score_receipt: Path | None = None,
) -> Path:
    """Verify all bindings before publishing a content-addressed model decision."""
    policy_source = Path(policy).resolve() if isinstance(policy, (str, Path)) else None
    declaration = load_policy(policy_source) if policy_source else policy
    index_path = _index_path(verified_index).resolve()
    index = verify_evaluation_index(index_path)
    _validate_policy(declaration, index)
    review = (
        verify_review_receipt(human_review_receipt) if human_review_receipt else None
    )
    if review is not None and (
        review["index_sha256"] != index["index_sha256"]
        or review["checkpoint_sha256"] != index["checkpoint_sha256"]
    ):
        raise ValueError(
            "review decision bound to a different evaluation or checkpoint"
        )
    scores = _scores_for_readiness(declaration, index, reviewed_score_receipt)
    state, missing, completed = _decision(declaration, index, review, scores)
    body = {
        "format": "model-readiness-result-v1",
        "policy": str(policy_source) if policy_source else None,
        "policy_data": None if policy_source else declaration.model_dump(mode="json"),
        "policy_sha256": sha256_file(policy_source)
        if policy_source
        else hashlib.sha256(
            canonical_json(declaration.model_dump(mode="json"))
        ).hexdigest(),
        "index": str(index_path),
        "index_sha256": index["index_sha256"],
        "checkpoint_sha256": index["checkpoint_sha256"],
        "review": str(Path(human_review_receipt).resolve()) if review else None,
        "review_sha256": review["receipt_sha256"] if review else None,
        "state": state,
        "missing_gates": missing,
        "completed_evaluations": completed,
    }
    if declaration.reviewed_scores is not None:
        body["reviewed_scores"] = (
            str(Path(reviewed_score_receipt).resolve())
            if reviewed_score_receipt is not None
            else None
        )
        body["reviewed_scores_sha256"] = scores["receipt_sha256"] if scores else None
        body["reviewed_score_summary"] = (
            {key: scores[key] for key in ("correct", "format_ok", "axes", "passed")}
            if scores
            else None
        )
    identity = _readiness_identity(body)
    output = index_path.parent / f"readiness-{identity}.json"
    if output.exists():
        verify_readiness_result(output)
        return output
    publish_immutable(
        output, {**body, "result_sha256": identity, "assessed_at_utc": utc_now()}
    )
    return output


def verify_readiness_result(path: Path) -> dict[str, Any]:
    path = Path(path)
    result = read_canonical(path)
    if result.get("format") != "model-readiness-result-v1":
        raise ValueError("invalid model readiness result")
    identity = _readiness_identity(result)
    if (
        result.get("result_sha256") != identity
        or path.name != f"readiness-{identity}.json"
    ):
        raise ValueError("readiness identity mismatch")
    policy = Path(result["policy"]) if result["policy"] else None
    if policy is not None:
        if (
            result["policy_data"] is not None
            or sha256_file(policy) != result["policy_sha256"]
        ):
            raise ValueError("readiness policy changed")
        declaration = load_policy(policy)
    else:
        declaration = ModelReadinessPolicy.model_validate(result["policy_data"])
        if (
            hashlib.sha256(
                canonical_json(declaration.model_dump(mode="json"))
            ).hexdigest()
            != result["policy_sha256"]
        ):
            raise ValueError("readiness policy identity mismatch")
    index = verify_evaluation_index(Path(result["index"]))
    if (
        index["index_sha256"] != result["index_sha256"]
        or index["checkpoint_sha256"] != result["checkpoint_sha256"]
    ):
        raise ValueError("readiness checkpoint or index changed")
    receipt = (
        verify_review_receipt(Path(result["review"]))
        if result["review"] is not None
        else None
    )
    if receipt is not None and (
        receipt["receipt_sha256"] != result["review_sha256"]
        or receipt["index_sha256"] != result["index_sha256"]
        or receipt["checkpoint_sha256"] != result["checkpoint_sha256"]
    ):
        raise ValueError("readiness review changed")
    _validate_policy(declaration, index)
    scores = _scores_for_readiness(
        declaration,
        index,
        Path(result["reviewed_scores"])
        if result.get("reviewed_scores") is not None
        else None,
    )
    if declaration.reviewed_scores is not None and (
        result.get("reviewed_scores_sha256")
        != (scores["receipt_sha256"] if scores else None)
        or result.get("reviewed_score_summary")
        != (
            {key: scores[key] for key in ("correct", "format_ok", "axes", "passed")}
            if scores
            else None
        )
    ):
        raise ValueError("reviewed-score readiness evidence changed")
    if (
        result["state"],
        result["missing_gates"],
        result["completed_evaluations"],
    ) != _decision(declaration, index, receipt, scores):
        raise ValueError("readiness decision does not follow policy")
    return result


def _scores_for_readiness(
    policy: ModelReadinessPolicy,
    index: dict[str, Any],
    receipt: Path | None,
) -> dict[str, Any] | None:
    criterion = policy.reviewed_scores
    if criterion is None:
        if receipt is not None:
            raise ValueError("reviewed scores supplied without a policy criterion")
        return None
    if receipt is None:
        return None
    from klab.evaluation.reviewed_scores import verify_reviewed_scores

    return verify_reviewed_scores(
        Path(receipt),
        index_sha256=index["index_sha256"],
        checkpoint_sha256=index["checkpoint_sha256"],
        **criterion.model_dump(),
    )
