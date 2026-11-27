"""Receipted text statistics over one published normalized partition."""

import struct
from pathlib import Path

from klab.corpus.normalized_annotations import NormalizedAnnotator, annotate_partition

_FIELDS = ("document_id", "utf8_bytes", "codepoints", "lines")
_ROW = struct.Struct("<32sIII")


def _render(writer, documents, _source):
    utf8 = points = lines = 0
    for doc_id, row in documents:
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError("invalid document in normalized partition")  # noqa: TRY004 - invalid serialized artifact
        size = len(text.encode("utf-8"))
        codepoints = len(text)
        line_count = text.count("\n") + bool(text and not text.endswith("\n"))
        writer.write(_ROW.pack(doc_id, size, codepoints, line_count))
        utf8 += size
        points += codepoints
        lines += line_count
    return {"totals": {"utf8_bytes": utf8, "codepoints": points, "lines": lines}}


def _decode(values, _summary):
    doc_id, *counts = values
    return dict(zip(_FIELDS, (doc_id.hex(), *counts), strict=True))


_ANNOTATOR = NormalizedAnnotator(
    "text-statistics", "text_statistics.py", _ROW, _render, _decode
)


def annotate_normalized(work_root: Path, partition: str, *, offset=0, limit=0) -> dict:
    """Produce or reuse normalized statistics and return an optional bounded page."""
    return annotate_partition(
        work_root, partition, _ANNOTATOR, offset=offset, limit=limit
    )
