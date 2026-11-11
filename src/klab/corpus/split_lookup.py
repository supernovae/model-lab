"""Compact operation-owned split lookup; canonical JSONL remains authoritative."""

import mmap
import os
import struct
from collections.abc import Mapping
from uuid import uuid4

from klab.corpus.bounded_metadata import _fingerprint, _input_stat
from klab.corpus.jsonl_records import records_from_path

_SPLITS = ("train", "validation", "test")


def stamp(path):
    return _fingerprint(_input_stat(path))


class AssignmentLookup(Mapping):
    """Sorted UTF-8 keys, uint64 offsets and one split byte per document.

    No JSON payloads are duplicated in a relational index. The operation owns
    and closes the maps before removing scratch; this is not a durable artifact.
    """

    def __init__(self, directory, path, reference):
        self.maps = []
        self.source_stamp = stamp(path)
        self.count = 0
        prefix = directory / ("assignment-" + uuid4().hex)
        keys, offsets, splits = [
            prefix.with_suffix(suffix) for suffix in (".keys", ".offsets", ".splits")
        ]
        previous, position = None, 0
        with (
            keys.open("xb") as key_file,
            offsets.open("xb") as offset_file,
            splits.open("xb") as split_file,
        ):
            for record in records_from_path(
                path, max_record_bytes=8 * 1024**2, expected_sha256=reference["sha256"]
            ):
                row = record.value
                if (
                    not isinstance(row, dict)
                    or set(row) != {"document_id", "split"}
                    or not isinstance(row["document_id"], str)
                    or not row["document_id"]
                    or row["split"] not in _SPLITS
                    or previous is not None
                    and row["document_id"] <= previous
                ):
                    raise ValueError("invalid or unordered split assignment")
                previous = row["document_id"]
                encoded = previous.encode("utf-8")
                offset_file.write(struct.pack("<Q", position))
                key_file.write(encoded)
                split_file.write(bytes([_SPLITS.index(row["split"])]))
                position += len(encoded)
                self.count += 1
            offset_file.write(struct.pack("<Q", position))
            for stream in (key_file, offset_file, split_file):
                stream.flush()
                os.fsync(stream.fileno())
        if self.count != reference["rows"]:
            raise ValueError("split assignment count mismatch")
        if stamp(path) != self.source_stamp:
            raise ValueError("split assignments changed during lookup")
        for name in (keys, offsets, splits):
            with name.open("rb") as stream:
                self.maps.append(
                    mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
                    if name.stat().st_size
                    else b""
                )

    def _key(self, index):
        start = struct.unpack_from("<Q", self.maps[1], 8 * index)[0]
        end = struct.unpack_from("<Q", self.maps[1], 8 * (index + 1))[0]
        return self.maps[0][start:end]

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise KeyError(key)
        encoded = key.encode("utf-8")
        left, right = 0, self.count
        while left < right:
            middle = (left + right) // 2
            if self._key(middle) < encoded:
                left = middle + 1
            else:
                right = middle
        if left == self.count or self._key(left) != encoded:
            raise KeyError(key)
        return _SPLITS[self.maps[2][left]]

    def __iter__(self):
        for index in range(self.count):
            yield self._key(index).decode("utf-8")

    def __len__(self):
        return self.count

    def close(self):
        for mapping in self.maps:
            if isinstance(mapping, mmap.mmap):
                mapping.close()
        self.maps.clear()
