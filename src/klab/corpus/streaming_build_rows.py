"""Compact ordered references for the existing v2 LM release writer.

Canonical documents and spans stay in completed immutable source shards. Only
IDs, offsets, split labels and duplicate keys enter the sorted working files.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import mmap
import os
import struct
from array import array
from contextlib import ExitStack
from pathlib import Path

from klab.canonical import canonical_json
from klab.corpus.jsonl_records import records_from_path
from klab.corpus.split_runs import _stamp, external_sort, rows, write_rows


class _Writer:
    def __init__(self, path):
        self.path = path
        self.stream = path.open("xb")
        self.digest = hashlib.sha256()

    def write(self, raw):
        self.stream.write(raw)
        self.digest.update(raw)

    def close(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()

    def record(self):
        return {
            "path": str(self.path),
            "sha256": self.digest.hexdigest(),
            "bytes": self.path.stat().st_size,
        }


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


def _references(directory, prepared, project):
    from klab.corpus.pipeline import _split

    policy = project.splits.model_dump(mode="json")
    external = policy.get("assignment_file") is not None

    def records():
        for shard_index, shard in enumerate(prepared):
            with (
                (shard / "docs.jsonl").open("rb") as docs,
                (shard / "spans.jsonl").open("rb") as spans,
            ):
                while True:
                    position = docs.tell()
                    raw = docs.readline()
                    span = spans.readline()
                    if not raw and not span:
                        break
                    if not raw or not span:
                        raise ValueError("prepared document/span identity mismatch")
                    doc = json.loads(raw)
                    if doc["document_id"] != json.loads(span)["record_id"]:
                        raise ValueError("prepared document/span identity mismatch")
                    identifier = doc["document_id"]
                    if len(identifier) != 64 or any(
                        c not in "0123456789abcdef" for c in identifier
                    ):
                        raise ValueError("streamed v2 document ID must be SHA-256")
                    rights = doc["rights"]
                    signature = hashlib.sha256(
                        canonical_json(
                            {
                                "license": doc["license"],
                                "redistribution": doc["redistribution"],
                                **{
                                    name: rights[name]
                                    for name in (
                                        "training_eligibility",
                                        "license_references",
                                        "notices",
                                        "training_restriction",
                                    )
                                },
                            }
                        )
                    ).hexdigest()
                    yield [
                        identifier,
                        shard_index,
                        position,
                        len(raw),
                        None if external else _split(doc, policy),
                        doc["raw_content_sha256"],
                        doc["content_sha256"],
                        signature,
                        (doc.get("structure") or {}).get("decision")
                        == "exclude_lm_metadata_only",
                        doc["source_family"],
                        doc.get("domains", []),
                    ]

    item = external_sort(records(), directory / "sorted", key=lambda row: row[0])
    return {"files": [item], "references": item["path"], "documents": item["rows"]}


def _assign(directory, references, project):
    reference = project.splits.model_dump(mode="json").get("assignment_file")
    assignments = None
    if reference is not None:
        from klab.corpus.project import project_path

        path = project_path(project.root, project.config.splits).parent / (
            "assignments-" + reference["sha256"] + ".jsonl"
        )
        _stamp(path)
        assignments = iter(
            records_from_path(
                path, max_record_bytes=8 * 1024**2, expected_sha256=reference["sha256"]
            )
        )
    current, seen_assignments, last_assignment = None, 0, None

    def next_assignment():
        nonlocal seen_assignments, last_assignment
        record = next(assignments, None)
        if record is None:
            return None
        value = record.value
        if (
            set(value) != {"document_id", "split"}
            or value["split"] not in {"train", "validation", "test"}
            or last_assignment is not None
            and value["document_id"] <= last_assignment
        ):
            raise ValueError("invalid or unordered split assignment")
        last_assignment = value["document_id"]
        seen_assignments += 1
        return value

    families = {}
    ids = _Writer(directory / "ids.bin")
    previous = None

    def records():
        nonlocal previous, current
        for row in rows(references["references"]):
            if row[0] == previous:
                raise ValueError("duplicate prepared document ID")
            previous = row[0]
            if assignments is not None:
                while current is None or current["document_id"] < row[0]:
                    current = next_assignment()
                    if current is None:
                        raise ValueError("missing/conflicting split assignment")
                if current["document_id"] != row[0]:
                    raise ValueError("missing/conflicting split assignment")
                row[4] = current["split"]
            families.setdefault(row[9], set()).add(row[4])
            ids.write(row[0].encode("ascii"))
            yield row
        if assignments is not None:
            while next_assignment() is not None:
                pass
            if seen_assignments != reference["rows"]:
                raise ValueError("split assignment count mismatch")

    try:
        item = write_rows(directory / "references.jsonl", records())
    finally:
        ids.close()
        if assignments is not None:
            assignments.close()
    return {
        "files": [item, ids.record()],
        "references": item["path"],
        "ids": str(ids.path),
        "documents": item["rows"],
        "family_splits": [
            [key, sorted(value)]
            for key, value in sorted(families.items())
            if len(value) > 1
        ],
    }


def _bind_families(directory, assigned, reference):
    """Join frozen families and check isolation without a source-body replay."""
    stream = iter(
        records_from_path(
            Path(reference["path"]),
            max_record_bytes=8 * 1024**2,
            expected_sha256=reference["sha256"],
        )
    )
    current = next(stream, None)
    family_rows = directory / "family-splits.jsonl"
    previous = None

    def records():
        nonlocal current, previous
        with family_rows.open("xb") as families:
            for row in rows(assigned["references"]):
                if current is None:
                    raise ValueError(
                        "frozen family inventory coverage differs from rendered documents"
                    )
                candidate = current.value
                if (
                    set(candidate)
                    != {
                        "document_id",
                        "family_id",
                        "split",
                        "stratum",
                        "content_sha256",
                    }
                    or candidate["document_id"] != row[0]
                    or previous is not None
                    and candidate["document_id"] <= previous
                    or candidate["content_sha256"] != row[6]
                    or candidate["split"] != row[4]
                    or not isinstance(candidate["family_id"], str)
                    or not candidate["family_id"]
                    or candidate["stratum"] not in row[10]
                ):
                    raise ValueError(
                        "frozen family binding differs from rendered document"
                    )
                previous = candidate["document_id"]
                families.write(
                    canonical_json([candidate["family_id"], candidate["split"]]) + b"\n"
                )
                yield [*row, candidate["family_id"], candidate["stratum"]]
                current = next(stream, None)
            if current is not None:
                raise ValueError(
                    "frozen family inventory coverage differs from rendered documents"
                )
            families.flush()
            os.fsync(families.fileno())

    try:
        output = write_rows(directory / "references.jsonl", records())
    finally:
        stream.close()
    families = external_sort(rows(family_rows), directory / "families")
    previous = None
    for family in rows(families["path"]):
        if (
            previous is not None
            and family[0] == previous[0]
            and family[1] != previous[1]
        ):
            raise ValueError("document family leaks across splits")
        previous = family
    return {
        **assigned,
        "references": output["path"],
        "files": [
            output,
            families,
            *[item for item in assigned["files"] if item["path"] == assigned["ids"]],
        ],
    }


def _origins(directory, assigned):
    def records():
        for ordinal, row in enumerate(rows(assigned["references"])):
            for method, key in enumerate(row[5:7]):
                yield [method, key, ordinal, row[0], row[4], row[7]]

    item = external_sort(records(), directory / "sorted")
    return {"files": [item], "origins": item["path"]}


def _deduplicate(directory, assigned, origins):
    parent_path = directory / "parents.bin"
    parents = _Parents(parent_path, assigned["documents"])
    output = _Writer(directory / "groups.jsonl")
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


class StreamBuildRows:
    """Ordered input protocol shared with the existing native release writer."""

    def __init__(self, prepared, assigned, deduplicated):
        self.prepared = prepared
        self.assigned = assigned
        self.deduplicated = deduplicated

    @classmethod
    def create(cls, stages, prepared, project, *, family_reference=None):
        if project.release.schema_version != 2:
            raise ValueError("completed-preparation build requires v2 deduplication")
        refs = stages.phase(
            "build-document-references", lambda d: _references(d, prepared, project)
        )
        assigned = stages.phase(
            "build-join-splits", lambda d: _assign(d, refs, project)
        )
        if family_reference is not None:
            unbound = assigned
            assigned = stages.phase(
                "build-frozen-family-bindings",
                lambda d: _bind_families(d, unbound, family_reference),
            )
        origins = stages.phase("build-duplicate-keys", lambda d: _origins(d, assigned))
        deduplicated = stages.phase(
            "build-deduplicate", lambda d: _deduplicate(d, assigned, origins)
        )
        return cls(prepared, assigned, deduplicated)

    def spans(self):
        for shard in self.prepared:
            with (shard / "spans.jsonl").open("rb") as stream:
                for raw in stream:
                    yield raw.rstrip(b"\n")

    def family_splits(self):
        yield from self.assigned["family_splits"]

    def documents(self):
        with ExitStack() as stack:
            docs = [
                stack.enter_context((path / "docs.jsonl").open("rb"))
                for path in self.prepared
            ]
            parent_stream = stack.enter_context(
                Path(self.deduplicated["parents"]).open("rb")
            )
            id_stream = stack.enter_context(Path(self.assigned["ids"]).open("rb"))
            count = self.assigned["documents"]
            parent_map = (
                stack.enter_context(
                    mmap.mmap(parent_stream.fileno(), 0, access=mmap.ACCESS_READ)
                )
                if count
                else None
            )
            ids = (
                stack.enter_context(
                    mmap.mmap(id_stream.fileno(), 0, access=mmap.ACCESS_READ)
                )
                if count
                else None
            )
            for ordinal, row in enumerate(rows(self.assigned["references"])):
                root = ordinal
                while (
                    parent := struct.unpack_from("<Q", parent_map, root * 8)[0]
                ) != root:
                    root = parent
                representative = ids[root * 64 : (root + 1) * 64].decode("ascii")
                doc = docs[row[1]]
                doc.seek(row[2])
                raw = doc.read(row[3])
                if len(raw) != row[3]:
                    raise ValueError("completed prepared shard changed")
                reason = (
                    "metadata_only_front_matter"
                    if row[8]
                    else "duplicate"
                    if root != ordinal
                    else None
                )
                yield raw.rstrip(b"\n"), row[4], representative, reason

    def family_inventory(self):
        count = self.assigned["documents"]
        with ExitStack() as stack:
            source = stack.enter_context(Path(self.deduplicated["parents"]).open("rb"))
            parents = (
                stack.enter_context(
                    mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
                )
                if count
                else None
            )
            for ordinal, row in enumerate(rows(self.assigned["references"])):
                if len(row) != 13:
                    raise ValueError("completed build has no reviewed family bindings")
                if (
                    row[8]
                    or struct.unpack_from("<Q", parents, ordinal * 8)[0] != ordinal
                ):
                    continue
                yield {
                    "document_id": row[0],
                    "family_id": row[11],
                    "split": row[4],
                    "stratum": row[12],
                    "content_sha256": row[6],
                }
