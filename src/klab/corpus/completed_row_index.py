"""Sorted reference, split, family and duplicate-key tables for completed builds."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from klab.canonical import canonical_json
from klab.corpus.duplicate_evidence import is_metadata_only, rights_signature
from klab.corpus.jsonl_records import records_from_path
from klab.corpus.split_runs import external_sort, rows, write_rows


class RowWriter:
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


def references(directory, prepared, policy):
    """Write offsets with inline split labels when a policy is supplied."""
    from klab.corpus.split_assignments import assign_split

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
                    yield [
                        identifier,
                        shard_index,
                        position,
                        len(raw),
                        assign_split(doc, policy) if policy is not None else None,
                        doc["raw_content_sha256"],
                        doc["content_sha256"],
                        rights_signature(doc).hex(),
                        is_metadata_only(doc),
                        doc["source_family"],
                        doc.get("domains", []),
                    ]

    item = external_sort(records(), directory / "sorted", key=lambda row: row[0])
    return {"files": [item], "references": item["path"], "documents": item["rows"]}


def assign(directory, references, reference, path):
    assignments = None
    if path is not None:
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
    ids = RowWriter(directory / "ids.bin")
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


def bind_families(directory, assigned, reference):
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


def origins(directory, assigned):
    def records():
        for ordinal, row in enumerate(rows(assigned["references"])):
            for method, key in enumerate(row[5:7]):
                yield [method, key, ordinal, row[0], row[4], row[7]]

    item = external_sort(records(), directory / "sorted")
    return {"files": [item], "origins": item["path"]}
