"""Read-only lexical screening of cross-split normalized document overlap.

Lowercase Unicode word tokens form sets of three-word shingles (short texts use
all their words). Sixty-four deterministic MinHashes in sixteen four-hash bands
propose pairs; exact shingle Jaccard scores every proposed pair. Identical text
hashes independently propose pairs. LSH can miss similar pairs: this is not a
proof of semantic independence, and neither paraphrases nor all lexical matches
are guaranteed to appear. Caps raise rather than return incomplete results.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_SPLITS = ("train", "validation", "test")
_PRIME = (1 << 61) - 1
_COEFFICIENTS = tuple(
    int.from_bytes(
        hashlib.sha256(f"near-duplicate-v1:{index}:{part}".encode()).digest()[:8],
        "big",
    )
    % (_PRIME - 1)
    + 1
    for index in range(64)
    for part in ("a", "b")
)


def _shingles(text: str, cap: int) -> frozenset[int]:
    words = re.findall(r"\w+", text.casefold())
    if not words:
        return frozenset()
    width = min(3, len(words))
    if len(words) - width + 1 > cap:
        raise ValueError(f"near-duplicate shingle cap exceeded ({cap})")
    return frozenset(
        int.from_bytes(
            hashlib.blake2b(
                "\x1f".join(words[index : index + width]).encode(), digest_size=8
            ).digest(),
            "big",
        )
        % _PRIME
        for index in range(len(words) - width + 1)
    )


def _bands(shingles: frozenset[int]) -> Iterable[tuple[int, tuple[int, ...]]]:
    if not shingles:
        return
    signature = [
        min(
            (_COEFFICIENTS[2 * index] * item + _COEFFICIENTS[2 * index + 1]) % _PRIME
            for item in shingles
        )
        for index in range(64)
    ]
    for band in range(16):
        yield band, tuple(signature[band * 4 : band * 4 + 4])


def candidates(
    documents: Iterable[dict[str, Any]],
    *,
    max_documents: int = 100_000,
    max_input_text_bytes: int = 128 * 1024 * 1024,
    max_total_shingles: int = 1_000_000,
    max_shingles_per_document: int = 50_000,
    max_bucket_documents: int = 1_000,
    max_comparisons: int = 200_000,
    max_candidates: int = 20_000,
    min_jaccard: float = 0.65,
) -> dict[str, Any]:
    """Screen all normalized documents, including excluded or duplicate records.

    Bounds cap input documents, shingles per document, LSH posting-list fanout,
    exact cross-split comparisons and output count. No partial report on overflow.
    The threshold deliberately retains ambiguous lexical overlaps for review.
    """
    if (
        min(
            max_documents,
            max_shingles_per_document,
            max_input_text_bytes,
            max_total_shingles,
            max_bucket_documents,
            max_comparisons,
            max_candidates,
        )
        < 1
        or not 0 < min_jaccard <= 1
    ):
        raise ValueError("near-duplicate caps must be positive and threshold in (0, 1]")
    rows = []
    text_bytes = 0
    for row in documents:
        if len(rows) >= max_documents:
            raise ValueError(f"near-duplicate document cap exceeded ({max_documents})")
        text_bytes += len(row["text"].encode("utf-8"))
        if text_bytes > max_input_text_bytes:
            raise ValueError(
                f"near-duplicate input text byte cap exceeded ({max_input_text_bytes})"
            )
        rows.append(row)
    rows.sort(key=lambda row: row["document_id"])
    if len({row["document_id"] for row in rows}) != len(rows):
        raise ValueError("near-duplicate duplicate document ID")
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    identical: dict[str, list[int]] = defaultdict(list)
    shingle_sets: list[frozenset[int]] = []
    pairs: set[tuple[int, int]] = set()
    total_shingles = 0
    for index, row in enumerate(rows):
        if row["split"] not in _SPLITS:
            raise ValueError("near-duplicate invalid document split")
        shingles = _shingles(row["text"], max_shingles_per_document)
        shingle_sets.append(shingles)
        total_shingles += len(shingles)
        if total_shingles > max_total_shingles:
            raise ValueError(
                f"near-duplicate total shingle cap exceeded ({max_total_shingles})"
            )
        keys = list(_bands(shingles))
        matches = set(identical[row["content_sha256"]])
        for key in keys:
            matches.update(buckets[key])
        for previous in sorted(matches):
            if rows[previous]["split"] != row["split"]:
                pairs.add((previous, index))
                if len(pairs) > max_comparisons:
                    raise ValueError(
                        f"near-duplicate comparison cap exceeded ({max_comparisons})"
                    )
        identical[row["content_sha256"]].append(index)
        if len(identical[row["content_sha256"]]) > max_bucket_documents:
            raise ValueError(
                f"near-duplicate identical-content bucket cap exceeded ({max_bucket_documents})"
            )
        for key in keys:
            buckets[key].append(index)
            if len(buckets[key]) > max_bucket_documents:
                raise ValueError(
                    f"near-duplicate bucket cap exceeded ({max_bucket_documents})"
                )
    found = []
    for left, right in sorted(pairs):
        a, b = shingle_sets[left], shingle_sets[right]
        exact = rows[left]["content_sha256"] == rows[right]["content_sha256"]
        intersection = len(a & b)
        union = len(a | b)
        similarity = intersection / union if union else float(exact)
        if exact or similarity >= min_jaccard:
            found.append(
                {
                    "left_document_id": rows[left]["document_id"],
                    "left_split": rows[left]["split"],
                    "left_source_id": rows[left].get("source_id"),
                    "left_source_location": rows[left].get("source_location"),
                    "left_content_sha256": rows[left]["content_sha256"],
                    "right_document_id": rows[right]["document_id"],
                    "right_split": rows[right]["split"],
                    "right_source_id": rows[right].get("source_id"),
                    "right_source_location": rows[right].get("source_location"),
                    "right_content_sha256": rows[right]["content_sha256"],
                    "jaccard": similarity,
                    "shared_shingles": intersection,
                    "union_shingles": union,
                    "identical_content_sha256": exact,
                }
            )
            if len(found) > max_candidates:
                raise ValueError(
                    f"near-duplicate candidate cap exceeded ({max_candidates})"
                )
    return {
        "algorithm": "word_shingle_minhash_lsh_v1",
        "limitations": (
            "Lexical candidates only; LSH is not exhaustive and does not establish "
            "semantic independence."
        ),
        "parameters": {
            "word_shingle_width": 3,
            "short_document_width": "min(3, word_count)",
            "hashes": 64,
            "bands": 16,
            "hashes_per_band": 4,
            "min_jaccard": min_jaccard,
            "max_input_text_bytes": max_input_text_bytes,
            "max_total_shingles": max_total_shingles,
            "max_documents": max_documents,
            "max_shingles_per_document": max_shingles_per_document,
            "max_bucket_documents": max_bucket_documents,
            "max_comparisons": max_comparisons,
            "max_candidates": max_candidates,
        },
        "document_count": len(rows),
        "cross_split_comparisons": len(pairs),
        "candidates": found,
    }


def audit_release(path: Path, **caps: int) -> dict[str, Any]:
    """Verify the immutable release, then screen its original normalized text."""
    from klab.corpus.release import verify_release

    manifest = verify_release(path)
    with (Path(path) / "documents.jsonl").open(encoding="utf-8") as handle:
        result = candidates(
            (json.loads(line) for line in handle if line.strip()), **caps
        )
    return {"release_id": manifest["release_id"], **result}
