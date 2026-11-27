"""Rejoin a release's narrow document decisions with retained normalized text."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from klab.canonical import canonical_digest
from klab.corpus.normalized_spans import SPAN_MEMBER, span_partitions
from klab.safe_io import read_file_range, symlinked

DOCUMENTS = "documents.jsonl"
DOCUMENT_REFERENCES = "documents.refs.json"
TEXT_REFERENCE = "normalized_text"
_NORMALIZED_DOCUMENTS = "docs.jsonl"
_MAX_OPEN = 64
_MAX_BATCH_ROWS = 2048
_MAX_BATCH_BYTES = 8 * 1024**2
_MAX_READ_BYTES = 256 * 1024


def overlay_dependencies(root, files):
    """Resolve immutable backing files through the release's existing span catalog."""
    if DOCUMENT_REFERENCES not in files:
        return []
    parts = span_partitions(root, files)
    if not parts:
        raise ValueError("document overlays require normalized span partitions")
    paths = [part.directory / _NORMALIZED_DOCUMENTS for part in parts]
    for path in paths:
        if symlinked(path) or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError(f"missing or unsafe normalized document input: {path}")
    return paths


def overlay_inventory(root, files):
    paths = overlay_dependencies(root, files)
    if not paths:
        return {}
    if json.loads((root / DOCUMENT_REFERENCES).read_bytes()) != files[SPAN_MEMBER]:
        raise ValueError("document overlay catalog differs from its normalized spans")
    result = {}
    for part, path in zip(span_partitions(root, files), paths, strict=True):
        raw = part.receipt.read_bytes()
        seal = json.loads(raw)
        if (
            hashlib.sha256(raw).hexdigest() != part.entry["receipt_sha256"]
            or seal["key"] != part.entry["partition_key"]
            or canonical_digest(seal["identity"]) != seal["key"]
        ):
            raise ValueError("normalized document partition receipt changed")
        result[path] = seal["files"][_NORMALIZED_DOCUMENTS]
    return result


@lru_cache(maxsize=8)
def _cached_paths(companion, generation):
    del generation
    return overlay_dependencies(
        companion.parent,
        {DOCUMENT_REFERENCES: {}, SPAN_MEMBER: json.loads(companion.read_bytes())},
    )


def _paths(path):
    from klab.corpus.shards import range_file_identity

    companion = path.with_name(DOCUMENT_REFERENCES)
    if path.name != DOCUMENTS or not companion.is_file():
        return []
    return _cached_paths(companion, tuple(range_file_identity(companion)))


def source_generations(path):
    from klab.corpus.shards import range_file_identity

    path = Path(path)
    paths = _paths(path)
    return (
        [
            range_file_identity(member)
            for member in (path.with_name(DOCUMENT_REFERENCES), *paths)
        ]
        if paths
        else []
    )


def document_chunks(path, *, max_bytes=None):
    """Keep worker input bounded by referenced bytes rather than overlay bytes."""
    from klab.corpus.release_verify import range_records
    from klab.corpus.shards import CHUNK_BYTES, chunk_ranges

    text = DocumentText(path)
    if not text.paths:
        return chunk_ranges(path)
    limit = max_bytes if max_bytes is not None else CHUNK_BYTES
    ranges = []
    start = end = used = 0
    for row, offset, length in range_records(path, 0, path.stat().st_size):
        size = text._locator(row)[2]
        if used and used + size > limit:
            ranges.append((start, offset))
            start, used = offset, 0
        used += size
        end = offset + length
    if end > start:
        ranges.append((start, end))
    return ranges


class DocumentText:
    """Hydrate only callers that need text, keeping metadata offsets physical."""

    def __init__(self, path: Path):
        self.paths = _paths(path)
        self.streams = OrderedDict()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for stream in self.streams.values():
            stream.close()
        self.streams.clear()

    def _stream(self, partition):
        if partition not in self.streams:
            if len(self.streams) == _MAX_OPEN:
                self.streams.popitem(last=False)[1].close()
            self.streams[partition] = self.paths[partition].open("rb")
        self.streams.move_to_end(partition)
        return self.streams[partition]

    def _locator(self, row):
        locator = row.get(TEXT_REFERENCE)
        if (
            "text" in row
            or not isinstance(locator, list)
            or len(locator) != 3
            or any(type(value) is not int for value in locator)
            or not 0 <= locator[0] < len(self.paths)
            or locator[1] < 0
            or locator[2] < 1
        ):
            raise ValueError("invalid normalized document text locator")
        return locator

    def _load(self, row, locator):
        partition, offset, size = locator
        stream = self._stream(partition)
        if size > 64 * 1024**2 and offset + size > os.fstat(stream.fileno()).st_size:
            raise ValueError("normalized document locator exceeds the partition")
        return self._resolve(row, read_file_range(stream, offset, size))

    @staticmethod
    def _resolve(row, raw):
        if not raw.endswith(b"\n"):
            raise ValueError(
                "normalized document locator misses a physical row boundary"
            )
        parent = json.loads(raw)
        decisions = {"split", "representative_id", "drop_reason"}
        core = {
            key: value
            for key, value in row.items()
            if key not in decisions | {TEXT_REFERENCE}
        }
        normalized = {
            key: value
            for key, value in parent.items()
            if key not in decisions | {"text"}
        }
        if normalized != core or not isinstance(parent.get("text"), str):
            raise ValueError(
                "document decisions do not align with their normalized row"
            )
        resolved = dict(row)
        del resolved[TEXT_REFERENCE]
        resolved["text"] = parent["text"]
        return resolved

    def hydrate(self, row):
        if self.paths:
            return self._load(row, self._locator(row))
        return row

    def records(self, rows, *, coalesce: bool = False) -> Iterator:
        """Read normalized ranges in source order inside bounded batches."""
        if not self.paths:
            yield from rows
            return
        read_batch = self._coalesced_batch if coalesce else self._batch
        pending, size = [], 0
        for row, location, length in rows:
            locator = self._locator(row)
            if pending and (
                len(pending) == _MAX_BATCH_ROWS or size + locator[2] > _MAX_BATCH_BYTES
            ):
                yield from read_batch(pending)
                pending, size = [], 0
            pending.append((row, location, length, locator))
            size += locator[2]
        yield from read_batch(pending)

    def _batch(self, rows):
        resolved = [None] * len(rows)
        for index in sorted(range(len(rows)), key=lambda index: rows[index][3][:2]):
            row, location, length, locator = rows[index]
            resolved[index] = self._load(row, locator), location, length
        yield from resolved

    def _coalesced_batch(self, rows):
        resolved = [None] * len(rows)
        ordered = sorted(range(len(rows)), key=lambda index: rows[index][3][:2])
        start = 0
        while start < len(ordered):
            index = ordered[start]
            row, location, length, locator = rows[index]
            partition, offset, size = locator
            stop = start + 1
            end = offset + size
            while stop < len(ordered):
                following, position, width = rows[ordered[stop]][3]
                if (
                    following != partition
                    or position != end
                    or position + width - offset > _MAX_READ_BYTES
                ):
                    break
                end += width
                stop += 1
            if stop == start + 1:
                resolved[index] = self._load(row, locator), location, length
            else:
                payload = read_file_range(self._stream(partition), offset, end - offset)
                for index in ordered[start:stop]:
                    row, location, length, (_, position, width) = rows[index]
                    begin = position - offset
                    resolved[index] = (
                        self._resolve(row, payload[begin : begin + width]),
                        location,
                        length,
                    )
            start = stop
        yield from resolved


def document_range(path, start, end):
    from klab.corpus.release_verify import range_records

    with DocumentText(Path(path)) as text:
        yield from text.records(range_records(Path(path), start, end))
