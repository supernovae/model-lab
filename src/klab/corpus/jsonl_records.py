"""Read physical LF-delimited JSON records without Unicode line splitting.

Blank physical lines are ignored, as in the existing corpus JSONL readers.
The physical line number still includes them; source-row ordinals count only
nonblank records. Source bytes are never rewritten by this reader.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


@dataclass(frozen=True)
class JSONLRecord:
    physical_line: int
    value: Any
    raw: bytes
    byte_offset: int = 0


def iter_lf_lines(
    stream: BinaryIO, *, max_record_bytes: int | None = None
) -> Iterator[tuple[int, bytes]]:
    """Yield raw physical lines, including blank lines and a final unterminated line."""
    if max_record_bytes is None:
        yield from enumerate(stream, 1)
        return
    if type(max_record_bytes) is not int or max_record_bytes <= 0:
        raise ValueError("record-byte cap must be a positive integer")
    physical_line = 0
    while raw := stream.readline(max_record_bytes + 1):
        if len(raw) > max_record_bytes:
            raise ValueError("JSONL record-byte cap exceeded")
        physical_line += 1
        yield physical_line, raw


def iter_jsonl_records(
    stream: BinaryIO, *, source: str, max_record_bytes: int | None = None
) -> Iterator[JSONLRecord]:
    """Decode UTF-8 strictly and parse one JSON value per nonblank LF record."""
    offset = 0
    for physical_line, raw in iter_lf_lines(stream, max_record_bytes=max_record_bytes):
        start = offset
        offset += len(raw)
        if not raw.strip():
            continue
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"invalid UTF-8 in {source} physical line {physical_line}"
            ) from error
        try:
            value = json.loads(decoded)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"malformed JSON in {source} physical line {physical_line}: {error.msg}"
            ) from error
        yield JSONLRecord(physical_line, value, raw, start)


def records_from_bytes(raw: bytes, *, source: str) -> Iterator[JSONLRecord]:
    return iter_jsonl_records(io.BytesIO(raw), source=source)


class _HashingLines:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def readline(self, size=-1):
        raw = self.stream.readline(size)
        self.digest.update(raw)
        return raw

    def __iter__(self):
        return self

    def __next__(self):
        raw = self.readline()
        if not raw:
            raise StopIteration
        return raw


def records_from_path(
    path: Path,
    *,
    max_record_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> Iterator[JSONLRecord]:
    """Validate the whole-file hash in the parsing pass, including blank lines."""
    with path.open("rb") as stream:
        reader = _HashingLines(stream) if expected_sha256 is not None else stream
        yield from iter_jsonl_records(
            reader, source=str(path), max_record_bytes=max_record_bytes
        )
        if expected_sha256 is not None and reader.digest.hexdigest() != expected_sha256:
            raise ValueError("JSONL source SHA-256 mismatch")


class ForwardRows:
    """One-row replay cursor for callers ordered by nonblank source ordinal."""

    def __init__(self, rows: Iterator[JSONLRecord]):
        self.rows = rows
        self.ordinal = 0
        self.current = None

    def get(self, ordinal: int) -> JSONLRecord:
        if type(ordinal) is not int or ordinal < 1 or ordinal < self.ordinal:
            raise ValueError("source replay requires increasing positive row ordinals")
        while self.ordinal < ordinal:
            try:
                self.current = next(self.rows)
            except StopIteration as error:
                raise ValueError(
                    "source replay row exceeds source inventory"
                ) from error
            self.ordinal += 1
        assert self.current is not None
        return self.current


@contextmanager
def forward_rows(
    path: Path, *, max_record_bytes: int | None, expected_sha256: str
) -> Iterator[ForwardRows]:
    """Validate skipped/trailing rows and close the reader on every exit path."""
    with closing(records_from_path(path, max_record_bytes=max_record_bytes)) as rows:
        yield ForwardRows(rows)
        # Preserve complete JSON validation even for rows with no kept document.
        for _ in rows:
            pass
    from klab.training.manifest import sha256_file

    if sha256_file(path) != expected_sha256:
        raise ValueError("source replay file changed")
