"""Language evidence recorded by sources and retained by normalization."""

import math
import struct
from pathlib import Path

from klab.corpus.normalized_annotations import NormalizedAnnotator, annotate_partition

# A compact label dictionary, independent of rows and bounded per partition.
_MAX_LABELS = 4096
_MAX_LABEL_BYTES = 64
_MISSING = 0
_UNUSABLE = 2**32 - 1
_ROW = struct.Struct("<32sI?d")


def _usable_label(value):
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value.encode("utf-8")) <= _MAX_LABEL_BYTES
    )


def _render(writer, documents, source):
    declared = source.get("acquisition", {}).get("language")
    labels = {}
    totals = dict.fromkeys(
        ("reported", "missing", "unusable", "code_context", "with_score"), 0
    )
    for doc_id, row in documents:
        metadata = row.get("metadata")
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            raise ValueError("invalid metadata in normalized partition")
        label = metadata.get("language")
        if label is None:
            index = _MISSING
            totals["missing"] += 1
        elif not _usable_label(label) or (
            len(labels) == _MAX_LABELS and label not in labels
        ):
            index = _UNUSABLE
            totals["unusable"] += 1
        else:
            index = labels.setdefault(label, len(labels) + 1)
            totals["reported"] += 1
        code = (
            row.get("document_kind") == "code"
            or metadata.get("qualification_profile") == "code"
        )
        totals["code_context"] += code
        score = metadata.get("language_score")
        if (type(score) is float and math.isfinite(score)) or (
            type(score) is int and abs(score) <= 2**53
        ):
            totals["with_score"] += 1
        else:
            score = math.nan
        writer.write(_ROW.pack(doc_id, index, code, score))
    return {
        "totals": totals,
        "declared_language": declared if _usable_label(declared) else None,
        "row_labels": list(labels),
    }


def _decode(values, summary):
    doc_id, index, code, score = values
    reported = index not in (_MISSING, _UNUSABLE)
    return {
        "document_id": doc_id.hex(),
        "row_label": summary["row_labels"][index - 1] if reported else None,
        "row_label_status": "reported"
        if reported
        else "missing"
        if index == _MISSING
        else "unusable",
        "code_context": code,
        "row_score": score if math.isfinite(score) else None,
    }


_ANNOTATOR = NormalizedAnnotator(
    "source-language", "source_language.py", _ROW, _render, _decode
)


def annotate_source_language(
    work_root: Path, partition: str, *, offset=0, limit=0
) -> dict:
    """Produce or page existing language claims without classifying document text."""
    return annotate_partition(
        work_root, partition, _ANNOTATOR, offset=offset, limit=limit
    )
