"""Upstream scores, token counts and provenance diagnostics from normalized rows."""

import math
import struct
from pathlib import Path

from klab.corpus.normalized_annotations import NormalizedAnnotator, annotate_partition

_FIELDS = ("score", "int_score", "token_count")
_VALUE_STATES = ("missing", "reported", "invalid")
_QUALIFICATION_STATES = ("not_recorded", "family_hint_recorded", "unresolved")
_ROW = struct.Struct("<32sdqQB")


def _score(value):
    if type(value) is float and math.isfinite(value):
        return 1, value
    if type(value) is int and abs(value) <= 2**53:
        return 1, value
    return (0 if value is None else 2), 0


def _integer(value, lower, upper):
    if type(value) is int and lower <= value < upper:
        return 1, value
    return (0 if value is None else 2), 0


def _render(writer, documents, _source):
    counts = [[0, 0, 0] for _ in _FIELDS]
    qualification_counts = [0, 0, 0]
    for doc_id, row in documents:
        metadata = row.get("metadata")
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            raise ValueError("invalid metadata in normalized partition")
        score_state, score = _score(metadata.get("score"))
        int_state, int_score = _integer(metadata.get("int_score"), -(2**63), 2**63)
        token_state, tokens = _integer(metadata.get("token_count"), 0, 2**64)
        if metadata.get("qualification_status") == "unresolved":
            qualification = 2
        else:
            hint = metadata.get("qualification_family_hint")
            qualification = int(isinstance(hint, str) and bool(hint))
        counts[0][score_state] += 1
        counts[1][int_state] += 1
        counts[2][token_state] += 1
        qualification_counts[qualification] += 1
        flags = (
            score_state | (int_state << 2) | (token_state << 4) | (qualification << 6)
        )
        writer.write(_ROW.pack(doc_id, score, int_score, tokens, flags))
    return {
        "totals": {
            name: dict(zip(_VALUE_STATES, field_counts, strict=True))
            for name, field_counts in zip(_FIELDS, counts, strict=True)
        }
        | {
            "qualification_evidence": dict(
                zip(_QUALIFICATION_STATES, qualification_counts, strict=True)
            )
        }
    }


def _decode(values, _summary):
    doc_id, score, int_score, tokens, flags = values
    qualification = flags >> 6
    if qualification == 3:
        raise ValueError("invalid source-quality qualification state")
    row = {
        "document_id": doc_id.hex(),
        "qualification_evidence": _QUALIFICATION_STATES[qualification],
        "invalid_fields": [],
    }
    for offset, (field, value) in enumerate(
        zip(_FIELDS, (score, int_score, tokens), strict=True)
    ):
        state = (flags >> (offset * 2)) & 3
        if state == 3:
            raise ValueError("invalid source-quality field state")
        row[field] = value if state == 1 else None
        if state == 2:
            row["invalid_fields"].append(field)
    return row


_ANNOTATOR = NormalizedAnnotator(
    "source-quality", "source_quality.py", _ROW, _render, _decode
)


def annotate_source_quality(
    work_root: Path, partition: str, *, offset=0, limit=0
) -> dict:
    """Publish or inspect source-supplied metadata without comparing score scales."""
    return annotate_partition(
        work_root, partition, _ANNOTATOR, offset=offset, limit=limit
    )
