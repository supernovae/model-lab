"""Disk-backed duplicate reduction for completed preparation."""

from __future__ import annotations

import hashlib
import itertools
import mmap
import os
import struct
from array import array

from klab.canonical import canonical_json
from klab.corpus.completed_row_index import RowWriter
from klab.corpus.split_runs import rows


class _Parents:
    def __init__(self, path, count):
        self.count = count
        with path.open("xb") as stream:
            for start in range(0, count, 131072):
                values = array("Q", range(start, min(count, start + 131072)))
                if values.itemsize != 8:
                    raise ValueError("64-bit duplicate ordinals unavailable")
                if os.sys.byteorder != "little":
                    values.byteswap()
                stream.write(values.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        self.stream = path.open("r+b")
        self.mapping = mmap.mmap(self.stream.fileno(), 0) if count else None

    def get(self, index):
        return struct.unpack_from("<Q", self.mapping, index * 8)[0]

    def set(self, index, value):
        struct.pack_into("<Q", self.mapping, index * 8, value)

    def root(self, index):
        while (parent := self.get(index)) != index:
            self.set(index, self.get(parent))
            index = parent
        return index

    def join(self, first, second):
        left, right = self.root(first), self.root(second)
        self.set(max(left, right), min(left, right))

    def close(self):
        if self.mapping is not None:
            self.mapping.flush()
            self.mapping.close()
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


def deduplicate(directory, assigned, origins):
    parent_path = directory / "parents.bin"
    parents = _Parents(parent_path, assigned["documents"])
    output = RowWriter(directory / "groups.jsonl")
    groups = 0
    try:
        for (method, key), values in itertools.groupby(
            rows(origins["origins"]), key=lambda row: row[:2]
        ):
            members = iter(values)
            first, second = next(members), next(members, None)
            if second is None:
                continue
            splits, incompatible = set(), False
            output.write(
                b'{"method":'
                + canonical_json(("raw", "normalized")[method])
                + b',"origins":['
            )
            for index, row in enumerate(itertools.chain([first, second], members)):
                if index:
                    output.write(b",")
                output.write(canonical_json(row[3]))
                splits.add(row[4])
                incompatible |= bool(method and row[5] != first[5])
                parents.join(first[2], row[2])
            output.write(
                b'],"representative":'
                + canonical_json(first[3])
                + b',"sha256":'
                + canonical_json(key)
                + b',"splits":'
                + canonical_json(sorted(splits))
                + b"}\n"
            )
            if len(splits) > 1:
                raise ValueError("cross-split exact text overlap")
            if incompatible:
                raise ValueError(
                    "duplicate source text has incompatible rights evidence"
                )
            groups += 1
        dropped = sum(
            bool(row[8]) or parents.root(ordinal) != ordinal
            for ordinal, row in enumerate(rows(assigned["references"]))
        )
    finally:
        output.close()
        parents.close()
    # Only the compact parent array is read here, never source text.
    digest = hashlib.sha256()
    with parent_path.open("rb") as stream:
        while raw := stream.read(1024**2):
            digest.update(raw)
    parent_record = {
        "path": str(parent_path),
        "sha256": digest.hexdigest(),
        "bytes": parent_path.stat().st_size,
    }
    return {
        "files": [output.record(), parent_record],
        "parents": str(parent_path),
        "groups": str(output.path),
        "group_count": groups,
        "dropped": dropped,
    }
