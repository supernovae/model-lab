"""Freeze reviewed Card 03 evaluation items against a verified corpus release.

This is an authoring validator, not a scorer. It cannot establish semantic
correctness or replace independent review of questions and gold answers.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from klab.canonical import canonical_digest, canonical_json, reject_duplicate_keys
from klab.corpus.release import verify_release
from klab.training.manifest import sha256_file

LANGUAGE_CATEGORIES = (
    "paraphrase",
    "reference_coreference",
    "negation",
    "quantifier_scope",
    "conditional_reasoning",
    "temporal_ordering",
    "compositional_inference",
    "answer_constraints",
    "ambiguity_clarification",
    "contradiction_insufficient",
)
EVIDENCE_CATEGORIES = (
    "direct_extraction",
    "paraphrased_support",
    "reference_resolution",
    "negation_scope",
    "conditions_exceptions",
    "temporal_version_precedence",
    "two_source_inference",
    "contradiction",
    "multipart_constraints",
    "missing_ambiguous_evidence",
)
_ITEM_FIELDS = {
    "id",
    "suite",
    "category",
    "parent_family",
    "parent_document_ids",
    "source_versions",
    "split",
    "question",
    "answerability",
    "required_claims",
    "prohibited_claims",
    "acceptable_paraphrases",
    "numeric_tolerance",
    "support_chunk_ids",
    "precedence_rule",
    "clarification_target",
    "answer_length_max_words",
    "reviewer",
    "review_status",
    "controls",
    "content_sha256",
}
_CHUNK_FIELDS = {"id", "document_id", "start", "end", "text_sha256"}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [
            json.loads(line, object_pairs_hook=reject_duplicate_keys)
            for line in stream
            if line.strip()
        ]


def _families(path: Path, docs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    family_splits: dict[str, str] = {}
    for row in _rows(path):
        if set(row) != {
            "document_id",
            "family_id",
            "split",
            "stratum",
            "content_sha256",
        } or any(not isinstance(value, str) or not value for value in row.values()):
            raise ValueError("invalid family inventory row")
        doc = docs.get(row["document_id"])
        if doc is None or row["document_id"] in rows:
            raise ValueError("unknown or duplicate family document")
        if (
            doc["content_sha256"] != row["content_sha256"]
            or doc["split"] != row["split"]
            or row["split"] not in {"train", "validation", "test"}
        ):
            raise ValueError("family inventory differs from verified release")
        family = row["family_id"]
        if family in family_splits and family_splits[family] != row["split"]:
            raise ValueError("family leaks across splits")
        family_splits[family] = row["split"]
        rows[row["document_id"]] = row
    if set(rows) != set(docs):
        raise ValueError("family inventory does not cover all kept documents")
    return rows


def _chunks(
    raw: Any, docs: dict[str, dict[str, Any]], families: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list):
        raise TypeError("chunks must be a list")
    chunks: dict[str, dict[str, Any]] = {}
    for chunk in raw:
        if not isinstance(chunk, dict) or set(chunk) != _CHUNK_FIELDS:
            raise ValueError("invalid chunk")
        doc_id = chunk["document_id"]
        if doc_id not in docs or families[doc_id]["split"] != "test":
            raise ValueError("evaluation chunk is not from a held-out test document")
        text = docs[doc_id]["text"]
        start, end = chunk["start"], chunk["end"]
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or end > len(text)
            or hashlib.sha256(text[start:end].encode("utf-8")).hexdigest()
            != chunk["text_sha256"]
        ):
            raise ValueError("chunk does not bind exact held-out source text")
        expected = canonical_digest({key: chunk[key] for key in _CHUNK_FIELDS - {"id"}})
        if chunk["id"] != expected or expected in chunks:
            raise ValueError("invalid or duplicate chunk identity")
        chunks[expected] = chunk
    return chunks


def _nonempty_strings(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, str) and bool(item.strip()) for item in value
    )


def _items(
    raw: Any,
    chunks: dict[str, dict[str, Any]],
    families: dict[str, dict[str, Any]],
    docs: dict[str, dict[str, Any]],
    spans: dict[str, dict[str, Any]],
    *,
    require_complete: bool,
) -> dict[str, int]:
    if not isinstance(raw, list) or len(raw) > 600:
        raise ValueError("invalid item list or count")
    counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    identifiers: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or set(item) != _ITEM_FIELDS:
            raise ValueError("invalid evaluation item fields")
        suite = item["suite"]
        categories = (
            LANGUAGE_CATEGORIES
            if suite == "closed_book"
            else EVIDENCE_CATEGORIES
            if suite == "open_book"
            else ()
        )
        if item["category"] not in categories:
            raise ValueError("invalid evaluation category")
        if item["split"] != "test":
            raise ValueError("evaluation item must declare test split")
        identifier = item["id"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
            or item["content_sha256"]
            != canonical_digest(
                {key: value for key, value in item.items() if key != "content_sha256"}
            )
        ):
            raise ValueError("duplicate item or content hash mismatch")
        identifiers.add(identifier)
        parents = item["parent_document_ids"]
        if (
            not _nonempty_strings(parents)
            or not parents
            or len(set(parents)) != len(parents)
        ):
            raise ValueError("evaluation item requires distinct parent documents")
        if any(
            doc_id not in families or families[doc_id]["split"] != "test"
            for doc_id in parents
        ):
            raise ValueError("evaluation item parent is not held-out")
        parent_families = {families[doc_id]["family_id"] for doc_id in parents}
        declared_family = item["parent_family"]
        if isinstance(declared_family, str):
            if not declared_family or parent_families != {declared_family}:
                raise ValueError("evaluation item parent is not one held-out family")
        elif suite == "open_book" and item["category"] == "two_source_inference":
            if (
                not isinstance(declared_family, list)
                or len(declared_family) < 2
                or declared_family != sorted(parent_families)
            ):
                raise ValueError("two-source parents need sorted held-out families")
        else:
            raise ValueError("evaluation item parent is not one held-out family")
        expected_versions = {
            doc_id: {
                "source_id": docs[doc_id]["source_id"],
                "source_revision": docs[doc_id]["source_revision"],
                "snapshot_sha256": spans[doc_id]["snapshot_sha256"],
            }
            for doc_id in parents
        }
        if item["source_versions"] != expected_versions:
            raise ValueError("item source versions differ from verified release")
        support = item["support_chunk_ids"]
        if not _nonempty_strings(support) or len(set(support)) != len(support):
            raise ValueError("invalid support chunk IDs")
        if any(
            chunk_id not in chunks or chunks[chunk_id]["document_id"] not in parents
            for chunk_id in support
        ):
            raise ValueError("support chunk is not a bound parent chunk")
        if (
            isinstance(declared_family, list)
            and item["answerability"] == "answerable"
            and {
                families[chunks[chunk_id]["document_id"]]["family_id"]
                for chunk_id in support
            }
            != parent_families
        ):
            raise ValueError("two-source support must cover both parent families")
        if (
            not isinstance(item["question"], str)
            or not item["question"].strip()
            or item["answerability"] not in {"answerable", "unanswerable", "clarify"}
            or not _nonempty_strings(item["required_claims"])
            or not _nonempty_strings(item["prohibited_claims"])
            or not _nonempty_strings(item["acceptable_paraphrases"])
            or type(item["answer_length_max_words"]) is not int
            or item["answer_length_max_words"] <= 0
            or not isinstance(item["reviewer"], str)
            or not item["reviewer"].strip()
            or item["review_status"] != "reviewed"
        ):
            raise ValueError("unreviewed or incomplete evaluation item")
        for optional in (
            "numeric_tolerance",
            "precedence_rule",
            "clarification_target",
        ):
            value = item[optional]
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"invalid {optional}")
        if item["answerability"] == "answerable" and (
            not item["required_claims"] or suite == "open_book" and not support
        ):
            raise ValueError("answerable item lacks claims or evidence support")
        if item["answerability"] != "answerable" and not item["clarification_target"]:
            raise ValueError("unanswerable item lacks abstention/clarification target")
        controls = item["controls"]
        if not isinstance(controls, dict) or set(controls) != {
            "no_evidence",
            "gold",
            "plausible_wrong",
            "shuffled_absent",
        }:
            raise ValueError("missing evidence-control condition")
        if suite == "closed_book":
            if any(value is not None for value in controls.values()) or support:
                raise ValueError("closed-book item cannot carry evidence controls")
        elif item["answerability"] == "answerable":
            if controls["no_evidence"] != [] or controls["gold"] != support:
                raise ValueError(
                    "gold and no-evidence conditions disagree with support"
                )
            for condition in ("plausible_wrong", "shuffled_absent"):
                selected = controls[condition]
                if (
                    not _nonempty_strings(selected)
                    or not selected
                    or set(selected) & set(support)
                    or any(chunk_id not in chunks for chunk_id in selected)
                ):
                    raise ValueError("invalid wrong/absent evidence condition")
        elif any(value is not None for value in controls.values()):
            raise ValueError("unanswerable evidence controls must be inapplicable")
        counts[f"{suite}/{item['category']}"] += 1
        if suite == "open_book" and item["category"] == "missing_ambiguous_evidence":
            missing_counts[item["answerability"]] += 1
    if any(
        count > (20 if key.startswith("closed_book/") else 40)
        for key, count in counts.items()
    ):
        raise ValueError("evaluation category exceeds declared denominator")
    if require_complete and (
        len(raw) != 600
        or any(counts[f"closed_book/{name}"] != 20 for name in LANGUAGE_CATEGORIES)
        or any(counts[f"open_book/{name}"] != 40 for name in EVIDENCE_CATEGORIES)
        or missing_counts["unanswerable"] == 0
        or missing_counts["clarify"] == 0
    ):
        raise ValueError(
            "Card 03 requires 200 closed-book and 400 open-book reviewed items "
            "with missing and clarification denominators"
        )
    return dict(sorted(counts.items()))


def freeze_card03_items(
    release: Path,
    family_inventory: Path,
    draft: Path,
    output: Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate draft lineage and publish a new content-addressed item manifest."""
    manifest = verify_release(release)
    docs = {
        row["document_id"]: row
        for row in _rows(release / "documents.jsonl")
        if row["drop_reason"] is None
    }
    spans = {row["record_id"]: row for row in _rows(release / "spans.jsonl")}
    families = _families(family_inventory, docs)
    candidate = json.loads(
        draft.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
    )
    if set(candidate) != {
        "schema_version",
        "protocol",
        "release_id",
        "family_inventory_sha256",
        "decoder",
        "evidence_token_budget",
        "chunks",
        "items",
    } or (
        candidate["schema_version"] != 1
        or candidate["protocol"] != "kml-card03-evaluation-v1"
        or candidate["release_id"] != manifest["release_id"]
        or candidate["family_inventory_sha256"] != sha256_file(family_inventory)
    ):
        raise ValueError(
            "draft does not bind the verified release and family inventory"
        )
    if (
        not isinstance(candidate["decoder"], dict)
        or not candidate["decoder"]
        or type(candidate["evidence_token_budget"]) is not int
        or candidate["evidence_token_budget"] <= 0
    ):
        raise ValueError("evaluation needs a shared decoder and evidence-token budget")
    chunks = _chunks(candidate["chunks"], docs, families)
    counts = _items(
        candidate["items"],
        chunks,
        families,
        docs,
        spans,
        require_complete=require_complete,
    )
    result = {
        "format": "kml-card03-evaluation-manifest-v1",
        "complete_denominators": require_complete,
        "protocol": candidate["protocol"],
        "release_id": manifest["release_id"],
        "family_inventory_sha256": candidate["family_inventory_sha256"],
        "draft_sha256": sha256_file(draft),
        "decoder": candidate["decoder"],
        "evidence_token_budget": candidate["evidence_token_budget"],
        "category_denominators": counts,
        "missing_evidence_denominators": {
            status: sum(
                item["category"] == "missing_ambiguous_evidence"
                and item["answerability"] == status
                for item in candidate["items"]
            )
            for status in ("unanswerable", "clarify")
        },
        "chunks": candidate["chunks"],
        "items": candidate["items"],
    }
    result["content_sha256"] = canonical_digest(result)
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".kml-eval-", dir=output.parent, delete=False
        ) as stream:
            name = stream.name
            stream.write(canonical_json(result) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, output)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)
    return result


def freeze_item_directory(
    release: Path, family_inventory: Path, draft: Path, output: Path
) -> dict[str, Any]:
    """Freeze a complete native manifest in an exclusively reserved directory."""
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".heldout-", dir=output.parent) as scratch:
        candidate = Path(scratch) / "items.json"
        result = freeze_card03_items(release, family_inventory, draft, candidate)
        # mkdir, unlike rename, never replaces a concurrent empty destination.
        output.mkdir(exist_ok=False)
        try:
            os.link(candidate, output / "items.json")
        except BaseException:
            output.rmdir()
            raise
    denominators = {
        key: result[key]
        for key in ("category_denominators", "missing_evidence_denominators")
    }
    return {
        "output": str(output / "items.json"),
        "content_sha256": result["content_sha256"],
        "manifest_sha256": sha256_file(output / "items.json"),
        "items_sha256": canonical_digest(result["items"]),
        "chunks_sha256": canonical_digest(result["chunks"]),
        "denominators_sha256": canonical_digest(denominators),
        "release_id": result["release_id"],
        "family_inventory_sha256": result["family_inventory_sha256"],
        "complete_denominators": result["complete_denominators"],
        **denominators,
    }
