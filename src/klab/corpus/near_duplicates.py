"""Read-only lexical screening of cross-split normalized document overlap.

Lowercase Unicode word tokens form sets of three-word shingles (short texts use
all their words). Sixty-four deterministic MinHashes in sixteen four-hash bands
propose pairs; exact shingle Jaccard scores every proposed pair. Identical text
hashes independently propose pairs. LSH can miss similar pairs: this is not a
proof of semantic independence, and neither paraphrases nor all lexical matches
are guaranteed to appear.

The screen indexes the held-out splits (validation and test) and streams the
training split past that index, so memory grows with the held-out side and the
longest single document, not with the corpus. Training documents are compared
with held-out ones and held-out splits with each other; training documents are
never compared with each other. Every document is screened: a held-out side
over its caps raises before anything is reported.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

_SPLITS = ("train", "validation", "test")
_HELD_OUT = ("validation", "test")
_HASHES = 64
_BANDS = 16
_ROWS = _HASHES // _BANDS
# Shingles whose MinHashes are computed at once; bounds the working matrix.
_BLOCK = 8192
_WORD_CACHE = 1 << 20


def _seeded(label: str, count: int) -> np.ndarray:
    return np.array(
        [
            int.from_bytes(
                hashlib.sha256(f"near-duplicate-v2:{label}:{i}".encode()).digest()[:8],
                "big",
            )
            | 1
            for i in range(count)
        ],
        dtype=np.uint64,
    )


_POSITION = _seeded("position", 3)
_MULTIPLY = _seeded("multiply", _HASHES)[:, None]
_ADD = _seeded("add", _HASHES)[:, None]


def _mix(values: np.ndarray) -> np.ndarray:
    """SplitMix64's finalizer: spreads every input bit over the output."""
    values = values ^ (values >> np.uint64(30))
    values = values * np.uint64(0xBF58476D1CE4E5B9)
    values = values ^ (values >> np.uint64(27))
    values = values * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


class _Shingler:
    """Turns text into its sorted, distinct three-word shingle hashes."""

    def __init__(self) -> None:
        self._words: dict[str, int] = {}

    def _word(self, word: str) -> int:
        value = self._words.get(word)
        if value is None:
            if len(self._words) >= _WORD_CACHE:
                self._words.clear()
            value = int.from_bytes(
                hashlib.blake2b(word.encode(), digest_size=8).digest(), "big"
            )
            self._words[word] = value
        return value

    def __call__(self, text: str) -> np.ndarray:
        words = re.findall(r"\w+", text.casefold())
        if not words:
            return np.empty(0, dtype=np.uint64)
        hashes = np.fromiter(map(self._word, words), dtype=np.uint64, count=len(words))
        width = min(3, len(words))
        count = len(words) - width + 1
        combined = np.zeros(count, dtype=np.uint64)
        for offset in range(width):
            combined += hashes[offset : offset + count] * _POSITION[offset]
        return np.unique(_mix(combined))


def _band_keys(shingles: np.ndarray) -> list[int]:
    if not len(shingles):
        return []
    signature = np.full(_HASHES, np.iinfo(np.uint64).max, dtype=np.uint64)
    for start in range(0, len(shingles), _BLOCK):
        block = shingles[start : start + _BLOCK][None, :]
        np.minimum(signature, (block * _MULTIPLY + _ADD).min(axis=1), out=signature)
    # One 64-bit key per band. Keys that collide only propose extra pairs, and
    # every proposed pair is scored exactly.
    rows = signature.reshape(_BANDS, _ROWS)
    combined = np.arange(_BANDS, dtype=np.uint64)
    for row in range(_ROWS):
        combined = _mix(combined ^ rows[:, row])
    return combined.tolist()


_FIELDS = ("document_id", "split", "source_id", "source_location", "content_sha256")


def _order(pair: dict[str, Any]) -> tuple[str, str]:
    return pair["left_document_id"], pair["right_document_id"]


def _coverage_key(row: dict[str, Any]) -> tuple[str, str]:
    return row["split"], str(row.get("source_id"))


def screen(
    read: Callable[[], Iterable[dict[str, Any]]],
    *,
    max_heldout_documents: int = 1_000_000,
    max_heldout_shingles: int = 200_000_000,
    max_candidates: int = 20_000,
    min_jaccard: float = 0.65,
) -> dict[str, Any]:
    """Screen every normalized document, including excluded or duplicate records.

    ``read`` returns a fresh pass over the documents; it is called twice. The
    first pass indexes the held-out splits and refuses past
    ``max_heldout_documents`` or ``max_heldout_shingles``: the index is
    about 8 bytes per held-out shingle plus about 4 KB per held-out document.
    The second pass streams the training split. Every candidate is counted;
    the first ``max_candidates`` in document-ID order are listed.
    The threshold deliberately retains ambiguous lexical overlaps for review.
    """
    if (
        min(max_heldout_documents, max_heldout_shingles, max_candidates) < 1
        or not 0 < min_jaccard <= 1
    ):
        raise ValueError("near-duplicate caps must be positive and threshold in (0, 1]")
    shingler = _Shingler()
    held: list[dict[str, Any]] = []
    held_shingles: list[np.ndarray] = []
    buckets: dict[int, list[int]] = defaultdict(list)
    identical: dict[str, list[int]] = defaultdict(list)
    documents: Counter[tuple[str, str]] = Counter()
    text_bytes: Counter[tuple[str, str]] = Counter()
    without_words: Counter[tuple[str, str]] = Counter()
    found: list[dict[str, Any]] = []
    candidate_count = 0
    identical_count = 0
    comparisons = 0
    total_shingles = 0
    seen_ids: set[str] = set()

    def proposals(digest: str, keys: list[int]) -> set[int]:
        matches = set(identical.get(digest, ()))
        for key in keys:
            matches.update(buckets.get(key, ()))
        return matches

    def keep_listed() -> None:
        found.sort(key=_order)
        del found[max_candidates:]

    def compare(row: dict[str, Any], shingles: np.ndarray, other: int) -> None:
        nonlocal comparisons, candidate_count, identical_count
        comparisons += 1
        left, right = held[other], row
        a, b = held_shingles[other], shingles
        exact = left["content_sha256"] == right["content_sha256"]
        intersection = len(np.intersect1d(a, b, assume_unique=True))
        union = len(a) + len(b) - intersection
        similarity = intersection / union if union else float(exact)
        if not (exact or similarity >= min_jaccard):
            return
        candidate_count += 1
        identical_count += exact
        if right["document_id"] < left["document_id"]:
            left, right = right, left
        found.append(
            {
                **{f"left_{field}": left.get(field) for field in _FIELDS},
                **{f"right_{field}": right.get(field) for field in _FIELDS},
                "jaccard": similarity,
                "shared_shingles": intersection,
                "union_shingles": union,
                "identical_content_sha256": exact,
            }
        )
        if len(found) >= 2 * max_candidates:
            keep_listed()

    def account(row: dict[str, Any], shingles: np.ndarray) -> None:
        key = _coverage_key(row)
        documents[key] += 1
        text_bytes[key] += len(row["text"].encode("utf-8"))
        if not len(shingles):
            without_words[key] += 1

    for row in read():
        if row["split"] not in _SPLITS:
            raise ValueError("near-duplicate invalid document split")
        if row["split"] == "train":
            continue
        if row["document_id"] in seen_ids:
            raise ValueError("near-duplicate duplicate document ID")
        seen_ids.add(row["document_id"])
        if len(held) >= max_heldout_documents:
            raise ValueError(
                f"near-duplicate held-out document cap exceeded ({max_heldout_documents})"
            )
        shingles = shingler(row["text"])
        total_shingles += len(shingles)
        if total_shingles > max_heldout_shingles:
            raise ValueError(
                f"near-duplicate held-out shingle cap exceeded ({max_heldout_shingles})"
            )
        keys = _band_keys(shingles)
        for other in sorted(proposals(row["content_sha256"], keys)):
            if held[other]["split"] != row["split"]:
                compare(row, shingles, other)
        index = len(held)
        held.append({field: row.get(field) for field in _FIELDS})
        held_shingles.append(shingles)
        identical[row["content_sha256"]].append(index)
        for key in keys:
            buckets[key].append(index)
        account(row, shingles)

    for row in read():
        if row["split"] != "train":
            continue
        if row["document_id"] in seen_ids:
            raise ValueError("near-duplicate duplicate document ID")
        shingles = shingler(row["text"])
        keys = _band_keys(shingles)
        for other in sorted(proposals(row["content_sha256"], keys)):
            compare(row, shingles, other)
        account(row, shingles)

    keep_listed()
    coverage = [
        {
            "split": split,
            "source_id": source,
            "documents": documents[split, source],
            "text_bytes": text_bytes[split, source],
            "documents_without_words": without_words[split, source],
        }
        for split, source in sorted(documents)
    ]
    return {
        "algorithm": "word_shingle_minhash_lsh_v2",
        "limitations": (
            "Lexical candidates only; LSH is not exhaustive and does not establish "
            "semantic independence. Documents without words are matched only by "
            "identical text."
        ),
        "parameters": {
            "word_shingle_width": 3,
            "short_document_width": "min(3, word_count)",
            "hashes": _HASHES,
            "bands": _BANDS,
            "hashes_per_band": _ROWS,
            "min_jaccard": min_jaccard,
            "max_heldout_documents": max_heldout_documents,
            "max_heldout_shingles": max_heldout_shingles,
            "max_candidates": max_candidates,
        },
        "document_count": sum(documents.values()),
        "heldout_shingles": total_shingles,
        "coverage": coverage,
        "cross_split_comparisons": comparisons,
        "candidate_count": candidate_count,
        "identical_text_count": identical_count,
        "candidates_truncated": candidate_count > max_candidates,
        "candidates": found,
    }


def audit_release(path: Path, **caps: int) -> dict[str, Any]:
    """Verify the immutable release, then screen its original normalized text."""
    from klab.corpus.release import verify_release

    manifest = verify_release(path)
    documents = Path(path) / "documents.jsonl"

    def read() -> Iterable[dict[str, Any]]:
        with documents.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    return {"release_id": manifest["release_id"], **screen(read, **caps)}
