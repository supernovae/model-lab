"""Source-local observations of repeated normalized lines in a partition."""

from __future__ import annotations

import hashlib
import heapq
import itertools
import re
import struct
import unicodedata
from collections import Counter
from functools import partial
from pathlib import Path

from klab.canonical import canonical_digest
from klab.corpus.normalized_annotations import NormalizedAnnotator, annotate_partition

_ROW = struct.Struct("<32sHH?")
_LINE = re.compile(r"[^\n]+")
_MAX_LINES = 128
_MAX_CHARS = 256
_MIN_WORDS = 3


def _line_keys(text):
    lines = _LINE.finditer(text)
    keys = []
    for line in itertools.islice(lines, _MAX_LINES):
        start, end = line.span()
        if end - start > _MAX_CHARS:
            continue
        words = line.group().casefold().split()
        if len(words) >= _MIN_WORDS:
            keys.append(
                hashlib.blake2b(
                    " ".join(words).encode("utf-8"), digest_size=16
                ).digest()
            )
    return keys, next(lines, None) is not None


def _prepare(documents, _source, *, sample_documents, min_support):
    sample = []
    for identifier, doc in documents:
        rank = -int.from_bytes(identifier, "big")
        if len(sample) < sample_documents:
            keys, _ = _line_keys(doc["text"])
            heapq.heappush(sample, (rank, keys))
        elif rank > sample[0][0]:
            keys, _ = _line_keys(doc["text"])
            heapq.heapreplace(sample, (rank, keys))
    frequency = Counter(key for _, keys in sample for key in set(keys))
    supported = {key for key, count in frequency.items() if count >= min_support}
    return {"sample_documents": len(sample), "supported": supported}


def _render(writer, documents, profile):
    supported = profile["supported"]
    examined = repeated = marked = truncated = 0
    for identifier, doc in documents:
        keys, limited = _line_keys(doc["text"])
        matches = sum(key in supported for key in keys)
        writer.write(_ROW.pack(identifier, len(keys), matches, limited))
        examined += len(keys)
        repeated += matches
        marked += matches > 0
        truncated += limited
    return {
        "sample_documents": profile["sample_documents"],
        "supported_lines": len(supported),
        "profile_sha256": canonical_digest(sorted(key.hex() for key in supported)),
        "totals": {
            "eligible_lines": examined,
            "template_lines": repeated,
            "marked_documents": marked,
            "line_limit_documents": truncated,
        },
    }


def _decode(values, _summary):
    identifier, eligible, repeated, limited = values
    if repeated > eligible or eligible > _MAX_LINES:
        raise ValueError("invalid template-lines counts")
    return {
        "document_id": identifier.hex(),
        "eligible_lines": eligible,
        "template_lines": repeated,
        "line_limit_reached": limited,
    }


def annotate_template_lines(
    work_root: Path,
    partition: str,
    *,
    offset=0,
    limit=0,
    sample_documents=256,
    min_support=4,
):
    """Sample one partition's documents and page exact repeated-line observations."""
    if (
        type(sample_documents) is not int
        or type(min_support) is not int
        or not 2 <= min_support <= sample_documents <= 2048
    ):
        raise ValueError(
            "template sample requires 2 <= min support <= sample documents <= 2048"
        )
    annotator = NormalizedAnnotator(
        "template-lines",
        "template_lines.py",
        _ROW,
        _render,
        _decode,
        options={
            "sample_documents": sample_documents,
            "min_support": min_support,
            "max_lines_per_document": _MAX_LINES,
            "max_line_chars": _MAX_CHARS,
            "min_line_words": _MIN_WORDS,
            "unicode_version": unicodedata.unidata_version,
        },
        prepare=partial(
            _prepare, sample_documents=sample_documents, min_support=min_support
        ),
    )
    return annotate_partition(
        work_root, partition, annotator, offset=offset, limit=limit
    )
