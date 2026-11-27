"""Bounded deterministic language observations for normalized prose."""

from __future__ import annotations

import re
import struct
import unicodedata
from functools import partial
from pathlib import Path

from klab.corpus.normalized_annotations import NormalizedAnnotator, annotate_partition

_MARKERS = {
    "de": "der die das und ist nicht für mit eine einer dem den sich auch bei aus zum über wird auf seine haben werden sie",
    "en": "the and of this that with from which were their you would what have when about could there some has they because into",
    "es": "los las una para del pero son mucho donde cuando como esta desde cada todo nos está han ellos además entre",
    "fr": "les des une avec dans cette mais pour sont que nous vous leur aux dont ainsi être été entre par où chez depuis",
    "pt": "uma dos das não que pelo pela são mais também como isso ainda sobre quando muito essa esta então aquele nossa ao aos fazer",
}
_WORDS = {name: frozenset(words.split()) for name, words in _MARKERS.items()}
_LABELS = (None, *sorted(_WORDS))
_STATES = ("insufficient", "ambiguous", "detected", "code")
_TOKEN = re.compile(r"[^\W\d_]+")
_MAX_CHARS = 8192
_MEMO_ENTRIES = 256
_MIN_MARKERS = 3
_MIN_MARGIN = 2
_ROW = struct.Struct("<32sBBBB")


def _detect(prefix):
    words = set(_TOKEN.findall(prefix.casefold()))
    counts = [len(words & _WORDS[label]) for label in _LABELS[1:]]
    label = max(range(len(counts)), key=counts.__getitem__)
    top = counts[label]
    counts[label] = 0
    margin = top - max(counts)
    if top < _MIN_MARKERS:
        return 0, 0, top, margin
    if margin < _MIN_MARGIN:
        return 0, 1, top, margin
    return label + 1, 2, top, margin


def _render(writer, documents, _source, *, memoize_content):
    totals = dict.fromkeys(_STATES, 0)
    memo = {}
    hits = 0
    for identifier, doc in documents:
        metadata = doc.get("metadata") or {}
        if (
            doc.get("document_kind") == "code"
            or metadata.get("qualification_profile") == "code"
        ):
            result = (0, 3, 0, 0)
        else:
            prefix = doc["text"][:_MAX_CHARS]
            key = doc["content_sha256"]
            if (
                memoize_content
                and (cached := memo.get(key)) is not None
                and cached[0] == prefix
            ):
                hits += 1
                result = cached[1]
            else:
                result = _detect(prefix)
                if memoize_content:
                    if len(memo) >= _MEMO_ENTRIES:
                        memo.pop(next(iter(memo)))
                    memo[key] = (prefix, result)
        totals[_STATES[result[1]]] += 1
        writer.write(_ROW.pack(identifier, *result))
    return {
        "totals": totals,
        "memoized_documents": hits,
        "supported_languages": list(_LABELS[1:]),
    }


def _decode(values, _summary):
    identifier, label, status, matches, margin = values
    if label >= len(_LABELS) or status >= len(_STATES) or bool(label) != (status == 2):
        raise ValueError("invalid detected-language observation")
    return {
        "document_id": identifier.hex(),
        "language": _LABELS[label],
        "status": _STATES[status],
        "marker_hits": matches,
        "marker_margin": margin,
    }


def annotate_detected_language(
    work_root: Path, partition: str, *, offset=0, limit=0, memoize_content=False
):
    """Publish language observations separately from upstream labels."""
    if type(memoize_content) is not bool:
        raise ValueError("language memoization option must be boolean")
    annotator = NormalizedAnnotator(
        "detected-language",
        "language_detection.py",
        _ROW,
        partial(_render, memoize_content=memoize_content),
        _decode,
        options={
            "max_chars": _MAX_CHARS,
            "min_markers": _MIN_MARKERS,
            "min_margin": _MIN_MARGIN,
            "unicode_version": unicodedata.unidata_version,
            "memoize_content": memoize_content,
            "memo_entries": _MEMO_ENTRIES if memoize_content else 0,
        },
    )
    return annotate_partition(
        work_root, partition, annotator, offset=offset, limit=limit
    )
