"""Whole-document TRAIN byte quotas and disjoint TRAIN tokenizer audits.

Release verification belongs to the caller. Selection follows the immutable
release document order; no validation/test text is selected or scored.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple

from tokenizers import Tokenizer

from klab.canonical import canonical_json
from klab.config.models import TokenizerTrainConfig
from klab.corpus.mixture import _source_rights
from klab.corpus.mixture_index import MAX_ROW_BYTES
from klab.corpus.release_verify import range_records, read_at
from klab.corpus.shards import declare_stage, plan_ranges, run_sharded


def quotas(config: TokenizerTrainConfig) -> dict[str, int]:
    spec = config.sampling
    assert spec is not None
    total = sum(spec.byte_weights.values())
    result = {
        key: spec.max_fit_bytes * weight // total
        for key, weight in spec.byte_weights.items()
    }
    ordered = sorted(
        result,
        key=lambda key: (-(spec.max_fit_bytes * spec.byte_weights[key] % total), key),
    )
    for key in ordered[: spec.max_fit_bytes - sum(result.values())]:
        result[key] += 1
    return result


LINEAGE_STAGE = declare_stage("tokenizer sampling: TRAIN lineage")
SCAN_STAGE = declare_stage("tokenizer sampling: documents scan")
_ELIGIBLE = frozenset({"eligible", "eligible_with_obligations"})


class _Candidate(NamedTuple):
    """One retained TRAIN document, located but not carrying its text."""

    document_id: str
    content_sha256: str
    stratum: str
    size: int
    offset: int
    length: int
    problem: str | None


def _lineage_ids(path: str, start: int, end: int) -> list[str]:
    return [row["record_id"] for row, _, _ in range_records(Path(path), start, end)]


def _scan_range(
    path: str, start: int, end: int, source_strata: dict[str, str]
) -> list[tuple]:
    """Select one byte range's undropped TRAIN documents, in file order.

    A document that violates a rule is returned with its `problem`, so the merge
    raises it at the position a serial scan would have reached it.
    """
    found = []
    for row, offset, length in range_records(Path(path), start, end):
        if length > MAX_ROW_BYTES:
            raise ValueError("mixture record-byte cap exceeded")
        if row["drop_reason"] is not None or row["split"] != "train":
            continue
        problem = None
        stratum = source_strata.get(row["source_id"])
        if row.get("rights", {}).get("training_eligibility") not in _ELIGIBLE:
            problem = "ineligible"
        elif stratum is None:
            problem = "source"
        elif stratum not in row["domains"]:
            problem = "domains"
        found.append(
            (
                row["document_id"],
                row["content_sha256"],
                stratum or row["source_id"],
                len(row["text"].encode("utf-8")),
                offset,
                length,
                problem,
            )
        )
    return found


def _train_rows(config: TokenizerTrainConfig):
    """Yield the TRAIN documents of the frozen release in document order.

    Byte ranges of `documents.jsonl` are scanned by worker processes and merged in
    range order, so the stream, and everything selected from it, is exactly what
    a serial scan would produce.
    """
    spec = config.sampling
    assert spec is not None
    release = config.dataset.corpus_release_path
    assert release is not None
    _source_rights(release, spec.source_strata)
    lineage = release / "lm/train.lineage.jsonl"
    selected: set[str] = set()
    for ids in run_sharded(plan_ranges(LINEAGE_STAGE, lineage), _lineage_ids):
        selected.update(ids)
    documents_path = release / "documents.jsonl"
    seen = 0
    results = run_sharded(
        plan_ranges(SCAN_STAGE, documents_path, spec.source_strata),
        _scan_range,
        ordered=True,
    )
    try:
        for found in results:
            for fields in found:
                if fields[0] not in selected:
                    continue
                if seen >= config.max_documents:
                    return
                candidate = _Candidate(*fields)
                seen += 1
                if candidate.problem == "ineligible":
                    raise ValueError(
                        "tokenizer sampling encountered ineligible TRAIN document"
                    )
                if candidate.problem == "source":
                    raise KeyError(candidate.stratum)
                if candidate.problem == "domains":
                    raise ValueError(
                        "tokenizer stratum disagrees with authenticated document domains"
                    )
                yield candidate, candidate.stratum
    finally:
        results.close()


def _text(release: Path, candidate: _Candidate) -> str:
    return read_at(release / "documents.jsonl", candidate.offset, candidate.length)[
        "text"
    ]


def documents(
    config: TokenizerTrainConfig, stats: dict, selected_hashes: set[str] | None = None
):
    caps = quotas(config)
    used = dict.fromkeys(caps, 0)
    counts = dict.fromkeys(caps, 0)
    acquired = 0
    seen_content: set[str] = set()
    digest = hashlib.sha256()
    identities = hashlib.sha256()
    release = config.dataset.corpus_release_path
    assert release is not None
    for row, stratum in _train_rows(config):
        acquired += 1
        if row.content_sha256 in seen_content:
            continue
        size = row.size
        if not size or used[stratum] + size > caps[stratum]:
            continue
        seen_content.add(row.content_sha256)
        used[stratum] += size
        counts[stratum] += 1
        text = _text(release, row)
        digest.update(text.encode("utf-8") + b"\0")
        identities.update(
            canonical_json([row.document_id, stratum, row.content_sha256]) + b"\n"
        )
        if selected_hashes is not None:
            selected_hashes.add(row.content_sha256)
        yield text
    missing = [key for key, count in counts.items() if not count]
    if missing:
        raise ValueError(
            "tokenizer byte sampling has no fitting TRAIN document in strata: "
            + ", ".join(sorted(missing))
        )
    stats.update(
        acquired=acquired,
        selected=sum(counts.values()),
        bytes=sum(used.values()),
        digest=digest.hexdigest(),
        stop_reason="bounded_stratified_scan",
        sampling={
            "order": "frozen-release-document-order-v1",
            "split": "train",
            "requested_bytes": caps,
            "selected_bytes": used,
            "selected_documents": counts,
            "unfilled_bytes": {key: caps[key] - used[key] for key in caps},
            "selected_identity_sha256": identities.hexdigest(),
            "scan_document_limit": config.max_documents,
        },
    )


def audit(config: TokenizerTrainConfig, tokenizer) -> dict:
    # Audit ordinary text, including literal special-token spellings, without
    # mutating the fitted artifact or its production encoding behavior.
    tokenizer = Tokenizer.from_str(tokenizer.to_str())
    tokenizer.encode_special_tokens = True
    spec = config.sampling
    assert spec is not None
    selected: set[str] = set()
    fit_stats = {}
    for _ in documents(config, fit_stats, selected):
        pass
    groups = {
        key: {
            "documents": 0,
            "bytes": 0,
            "tokens": 0,
            "unknown_tokens": 0,
            "roundtrip_failures": 0,
        }
        for key in spec.byte_weights
    }
    total_bytes = 0
    digest = hashlib.sha256()
    unknown = tokenizer.token_to_id("<unk>")
    release = config.dataset.corpus_release_path
    assert release is not None
    for row, stratum in _train_rows(config):
        group = groups[stratum]
        if (
            row.content_sha256 in selected
            or group["documents"] >= spec.audit_documents_per_stratum
        ):
            continue
        size = row.size
        if not size or total_bytes + size > spec.max_audit_bytes:
            continue
        selected.add(row.content_sha256)
        text = _text(release, row)
        ids = tokenizer.encode(text).ids
        group["documents"] += 1
        group["bytes"] += size
        group["tokens"] += len(ids)
        group["unknown_tokens"] += ids.count(unknown)
        group["roundtrip_failures"] += int(
            tokenizer.decode(ids, skip_special_tokens=False) != text
        )
        total_bytes += size
        digest.update(canonical_json([row.document_id, row.content_sha256]) + b"\n")
    for group in groups.values():
        group["bytes_per_token"] = (
            group["bytes"] / group["tokens"] if group["tokens"] else None
        )
        group["document_shortfall"] = max(
            0, spec.audit_documents_per_stratum - group["documents"]
        )
    probes = {
        "identifier": "some_identifier42 camelCase HTTPServer2",
        "numeric": "-0.125 1e-9 1234567890",
        "multilingual": "café Ελληνικά 日本語 العربية",
        "code_whitespace": "def example():\n\treturn 42  \n",
    }
    probe_results = {}
    for label, text in probes.items():
        ids = tokenizer.encode(text).ids
        probe_results[label] = {
            "roundtrip": tokenizer.decode(ids, skip_special_tokens=False) == text,
            "unknown_tokens": ids.count(unknown),
            "tokens": len(ids),
        }
    return {
        "split": "train",
        "disjoint_from_fit": True,
        "special_tokens_as_text": True,
        "requested_documents_per_stratum": spec.audit_documents_per_stratum,
        "max_bytes": spec.max_audit_bytes,
        "selected_bytes": total_bytes,
        "selected_identity_sha256": digest.hexdigest(),
        "strata": groups,
        "fixed_probes": probe_results,
        "complete": all(not group["document_shortfall"] for group in groups.values()),
    }
