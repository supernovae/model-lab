"""Independent provenance, verification evidence, and training-shape ledger v1."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from klab.canonical import canonical_digest, canonical_json

SOURCE_ORIGIN = "primary_source"
HUMAN_ORIGIN = "human_authored"
DERIVED_ORIGIN = "source_transformed_synthetic"
MULTI_SOURCE_ORIGIN = "multi_source_synthetic"
DETERMINISTIC_ORIGIN = "deterministic_synthetic"
INFERENCE_ORIGIN = "free_generated_synthetic"
SELF_GENERATED_ORIGIN = "model_self_generated"
ORIGINS = frozenset(
    {
        SOURCE_ORIGIN,
        HUMAN_ORIGIN,
        DERIVED_ORIGIN,
        MULTI_SOURCE_ORIGIN,
        DETERMINISTIC_ORIGIN,
        INFERENCE_ORIGIN,
        SELF_GENERATED_ORIGIN,
    }
)
SHAPES = frozenset(
    {
        "raw_document",
        "lexical_inventory",
        "definition",
        "troubleshooting_scenario",
        "direct_qa",
        "tool_trace",
        "decision_record",
        "source_grounded_qa",
        "paraphrased_qa",
        "error_diagnosis",
        "stop_or_abstain",
        "explanation",
        "multi_turn_dialogue",
        "verification_episode",
    }
)
STATUSES = frozenset(
    {
        "source_entailed",
        "oracle_verified",
        "cross_source_verified",
        "human_reviewed",
        "schema_validated",
        "unverified",
        "rejected",
    }
)
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ATTRIBUTES = (
    "interaction",
    "grounding",
    "answer_style",
    "supervision",
    "reasoning_depth",
    "task_family",
    "source_domains",
)
_NORMALIZED = {
    "interaction": {"none", "single_turn", "multi_turn", "tool_use"},
    "grounding": {"none", "source", "synthetic_world"},
    "answer_style": {
        "not_applicable",
        "concise",
        "explanatory",
        "structured",
        "action",
    },
    "supervision": {"not_training", "all_tokens", "assistant_only", "selected_span"},
    "reasoning_depth": {"direct", "one_hop", "multi_step"},
}


def rendered_digest(payload: Any, *, text: bool = False) -> str:
    """Hash precisely the text or the canonical JSON payload consumed by a view."""
    return hashlib.sha256(
        payload.encode("utf-8") if text else canonical_json(payload)
    ).hexdigest()


def shape(
    identifier: str,
    *,
    domains: list[str],
    interaction: str,
    grounding: str,
    answer_style: str,
    supervision: str,
    reasoning_depth: str,
    task_family: str,
) -> dict[str, Any]:
    if identifier not in SHAPES:
        raise ValueError(f"unknown training shape: {identifier}")
    attributes = {
        "interaction": interaction,
        "grounding": grounding,
        "answer_style": answer_style,
        "supervision": supervision,
        "reasoning_depth": reasoning_depth,
        "task_family": task_family,
        "source_domains": sorted(set(domains)),
    }
    return {"schema_version": 1, "id": identifier, "attributes": attributes}


def shape_for_record(
    kind: str,
    identifier: str,
    *,
    domains: list[str],
    parent_document_ids: list[str],
    scenario_id: str | None = None,
) -> dict[str, Any]:
    """Bind normalized construction attributes to the record, not its verification."""
    return shape(
        identifier,
        domains=domains,
        interaction="tool_use"
        if kind == "tool_episode"
        else "single_turn"
        if kind in {"chat_sft", "generation", "scenario"}
        else "none",
        grounding="synthetic_world"
        if scenario_id
        else "source"
        if parent_document_ids
        else "none",
        answer_style="structured"
        if kind in {"tool_episode", "semantic_candidate", "lexical_candidate"}
        else "action"
        if identifier == "decision_record"
        else "concise"
        if kind in {"chat_sft", "generation", "scenario"}
        else "not_applicable",
        supervision="assistant_only"
        if kind in {"chat_sft", "tool_episode"}
        else "all_tokens"
        if kind == "document"
        else "not_training",
        reasoning_depth="direct",
        task_family=identifier,
    )


def verification(
    status: str,
    method: str,
    evidence: dict[str, Any],
    *,
    verifier_id: str = "klab.corpus.provenance",
    verifier_version: str = "1",
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "status": status,
        "method": method,
        "evidence": evidence,
        "verifier_id": verifier_id,
        "verifier_version": verifier_version,
    }
    validate_verification(result)
    return result


def validate_verification(
    value: dict[str, Any],
    *,
    documents: dict[str, dict[str, Any]] | None = None,
    parent_document_ids: list[str] | None = None,
) -> None:
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("status") not in STATUSES
    ):
        raise ValueError("unknown verification schema/status")
    if (
        not isinstance(value.get("method"), str)
        or not value["method"]
        or value.get("verifier_id") != "klab.corpus.provenance"
        or value.get("verifier_version") != "1"
    ):
        raise ValueError("unrecognized verifier identity/version/method")
    evidence = value.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("missing typed verification evidence")
    status = value["status"]
    if status in {"rejected", "unverified"}:
        if (
            not isinstance(evidence.get("reason"), str)
            or not evidence["reason"].strip()
        ):
            raise ValueError("rejection/unverified reason required")
    elif status == "schema_validated":
        if (
            not isinstance(evidence.get("schema_id"), str)
            or not evidence["schema_id"].strip()
        ):
            raise ValueError("schema identity required")
    elif status == "source_entailed":
        document_id, passage, span = (
            evidence.get(k) for k in ("document_id", "passage", "span")
        )
        if (
            not isinstance(document_id, str)
            or not isinstance(passage, str)
            or not passage
        ):
            raise ValueError("source entailment needs document and verbatim passage")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(n, int) and not isinstance(n, bool) for n in span)
            or span[0] < 0
            or span[1] <= span[0]
        ):
            raise ValueError("source entailment needs a valid span")
        if "answer" in evidence and (
            not isinstance(evidence["answer"], str)
            or not evidence["answer"]
            or evidence["answer"] not in passage
        ):
            raise ValueError("entailed answer is not copied from the cited passage")
        if parent_document_ids is not None and document_id not in parent_document_ids:
            raise ValueError("entailment document absent from parent lineage")
        if documents is not None:
            doc = documents.get(document_id)
            if doc is None or doc["text"][span[0] : span[1]] != passage:
                raise ValueError("source entailment passage mismatch")
            byte_span = evidence.get("byte_span")
            if byte_span is not None and (
                not isinstance(byte_span, list)
                or len(byte_span) != 2
                or not all(type(n) is int for n in byte_span)
                or byte_span[0] < 0
                or byte_span[1] <= byte_span[0]
                or doc["text"].encode("utf-8")[byte_span[0] : byte_span[1]]
                != passage.encode("utf-8")
            ):
                raise ValueError("normalized source byte span mismatch")
    elif status == "oracle_verified":
        if not all(
            isinstance(evidence.get(k), str) and evidence[k]
            for k in (
                "oracle_identity",
                "generator_world_id",
                "oracle_implementation",
                "oracle_version",
                "interpreter",
            )
        ):
            raise ValueError(
                "oracle evidence requires identity, world, implementation and interpreter"
            )
        if not _HEX.fullmatch(evidence["oracle_implementation"]):
            raise ValueError("oracle implementation SHA-256 required")
        world = evidence.get("world_state")
        if (
            not isinstance(world, dict)
            or not world
            or any(
                not isinstance(key, str)
                or not (
                    isinstance(value, str)
                    or (evidence["oracle_version"] == "2" and type(value) is bool)
                )
                for key, value in world.items()
            )
        ):
            raise ValueError("oracle world state must be a typed dictionary")
        answer = evidence.get("oracle_answer")
        if (
            not isinstance(answer, str)
            or evidence.get("actual_result") != answer
            or evidence.get("comparison_status") != "match"
        ):
            raise ValueError("oracle result is not verified against expected answer")
        expected = canonical_digest([world, answer])
        if evidence["oracle_identity"] != expected:
            raise ValueError("oracle identity does not match world and answer")
        receipt = evidence.get("receipt")
        if receipt is not None and (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != 1
            or receipt.get("generator_version") not in {"1", "2"}
            or receipt["generator_version"] != evidence["oracle_version"]
            or receipt.get("world_id") != evidence["generator_world_id"]
            or receipt.get("world_facts") != world
            or receipt.get("judgment") not in {"proceed", "inspect", "ask", "stop"}
            or not answer.startswith(receipt["judgment"] + ":")
            or any(
                not isinstance(receipt.get(name), str) or not receipt[name]
                for name in (
                    "generator_id",
                    "scenario_family_id",
                    "template_family_id",
                    "evidence",
                    "next_diagnostic",
                    "verification",
                    "rollback",
                )
            )
        ):
            raise ValueError("invalid versioned scenario oracle receipt")
    elif status == "human_reviewed":
        if not all(
            isinstance(evidence.get(k), str) and evidence[k]
            for k in (
                "reviewer_id",
                "review_id",
                "review_version",
                "review_receipt_sha256",
            )
        ):
            raise ValueError("human review receipt required")
        if not _HEX.fullmatch(evidence["review_receipt_sha256"]):
            raise ValueError("invalid human review receipt digest")
    elif status == "cross_source_verified":
        citations = evidence.get("citations")
        if not isinstance(citations, list) or len(citations) < 2:
            raise ValueError("cross-source verification requires two citations")
        source_ids = set()
        for citation in citations:
            validate_verification(
                verification("source_entailed", "verbatim_span", citation),
                documents=documents,
                parent_document_ids=parent_document_ids,
            )
            if documents is not None:
                source_ids.add(documents[citation["document_id"]]["source_id"])
            else:
                source_ids.add(citation.get("source_id"))
        if len(source_ids) < 2 or None in source_ids:
            raise ValueError("cross-source citations require distinct sources")


def validate_lineage(
    row: dict[str, Any], *, documents: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Validate a frozen row without trusting a status label or generator claim."""
    if row.get("origin_schema_version") != 1 or row.get("origin") not in ORIGINS:
        raise ValueError("missing/invalid provenance origin v1")
    parents = row.get("parent_document_ids")
    if not isinstance(parents, list) or any(not isinstance(p, str) for p in parents):
        raise ValueError("invalid parent document IDs")
    modalities = row.get("modalities")
    if modalities != ["text"]:
        raise ValueError("unsupported or missing source modality")
    if documents is not None and modalities != sorted(
        {documents[parent]["modality"] for parent in parents} or {"text"}
    ):
        raise ValueError("lineage modality disagrees with source documents")
    shp = row.get("shape")
    if (
        not isinstance(shp, dict)
        or shp.get("schema_version") != 1
        or shp.get("id") not in SHAPES
    ):
        raise ValueError("invalid shape v1")
    attrs = shp.get("attributes")
    if not isinstance(attrs, dict) or set(attrs) != set(_ATTRIBUTES):
        raise ValueError("invalid shape attributes")
    if any(
        not isinstance(attrs[key], str) or not attrs[key]
        for key in _ATTRIBUTES
        if key != "source_domains"
    ):
        raise ValueError("invalid shape dimension")
    if any(attrs[key] not in values for key, values in _NORMALIZED.items()):
        raise ValueError("non-normalized shape attribute")
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", attrs["task_family"]):
        raise ValueError("invalid shape task family")
    domains = attrs["source_domains"]
    if (
        not isinstance(domains, list)
        or domains != sorted(set(domains))
        or any(not isinstance(domain, str) or not domain for domain in domains)
    ):
        raise ValueError("invalid source domains")
    if not isinstance(row.get("rendered_sha256"), str) or not _HEX.fullmatch(
        row["rendered_sha256"]
    ):
        raise ValueError("missing rendered SHA-256")
    validate_verification(
        row.get("verification"), documents=documents, parent_document_ids=parents
    )
    if (
        row.get("record_kind") in {"generation", "chat_sft"}
        and row["verification"]["status"] == "source_entailed"
        and not row["verification"]["evidence"].get("answer")
    ):
        raise ValueError("source-entailed answer evidence required")
    if (
        row.get("verification", {}).get("status") == "oracle_verified"
        and row.get("oracle_identity") is not None
        and row["oracle_identity"] != row["verification"]["evidence"]["oracle_identity"]
    ):
        raise ValueError("oracle verification identity mismatch")
    if (
        row["origin"]
        in {
            DERIVED_ORIGIN,
            MULTI_SOURCE_ORIGIN,
            DETERMINISTIC_ORIGIN,
            INFERENCE_ORIGIN,
            SELF_GENERATED_ORIGIN,
        }
        and row.get("record_kind")
        in {"generation", "scenario", "chat_sft", "tool_episode"}
        and not (
            isinstance(row.get("generator_identity"), str)
            and _HEX.fullmatch(row["generator_identity"])
        )
    ):
        raise ValueError("synthetic generated record missing generator identity")
    if (
        row["origin"] in {SOURCE_ORIGIN, HUMAN_ORIGIN}
        and row.get("record_kind") != "document"
    ):
        raise ValueError("source origin reserved for source documents")
    if documents is not None and row.get("record_kind") == "document":
        doc = documents.get(row["record_id"])
        if (
            doc is None
            or rendered_digest(doc["text"], text=True) != row["rendered_sha256"]
        ):
            raise ValueError("document rendered hash mismatch")
    return row
