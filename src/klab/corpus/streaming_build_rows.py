"""Compact ordered references for the existing v2 LM release writer.

Canonical documents and spans stay in completed immutable source shards. Only
IDs, offsets, split labels and duplicate keys enter the sorted working files.
"""

from __future__ import annotations

import json
import mmap
import struct
from contextlib import ExitStack
from pathlib import Path

from klab.corpus import completed_row_dedup, completed_row_index
from klab.corpus.admission_inspection_cache import capture, unchanged
from klab.corpus.build_stages import implementation_digests
from klab.corpus.project import project_path
from klab.corpus.split_runs import SplitStages, rows, write_rows
from klab.safe_io import read_file_range

_PHASE_FILES = ("admission_inspection_cache.py", "build_stages.py", "split_runs.py")
_INDEX_FILES = (
    "completed_row_index.py",
    "duplicate_evidence.py",
    "jsonl_records.py",
    "split_assignments.py",
)


def _run_phase(build, name, paths, options, files, produce):
    inputs = capture(paths)
    stages = SplitStages(
        build.root.parents[1] / "build-row-stages" / name,
        {
            "producer": implementation_digests(*_PHASE_FILES, *files),
            "inputs": inputs,
            "options": options,
        },
        work_root=build.cache.root,
        producer_scoped=True,
    )
    result = stages.phase(name, produce)
    unchanged(inputs)
    if stages.reused:
        build.reused.extend(stages.reused)
    return result


def _bind_build_phase(stages, name, produce):
    """Record the shared output in the signed build chain used by retained mixtures."""

    def link(directory):
        result = produce()
        directory.rmdir()
        return dict(result)

    return stages.phase(name, link)


class StreamBuildRows:
    """Ordered input protocol shared with the existing native release writer."""

    def __init__(self, prepared, assigned, deduplicated):
        self.prepared = prepared
        self.assigned = assigned
        self.deduplicated = deduplicated

    @classmethod
    def create(cls, stages, prepared, project, *, family_reference=None):
        from klab.corpus.dedup_policy import is_strict_chain

        if not is_strict_chain(project.release.policy.duplicates):
            raise ValueError(
                "completed-preparation build supports only the strict raw/normalized "
                "chain duplicate policy"
            )
        policy = project.splits.model_dump(mode="json")
        assignment = policy.get("assignment_file")
        # With external assignments the sorted references carry no split label.
        reference_policy = None if assignment else policy
        refs = _run_phase(
            stages,
            "build-document-references",
            [
                shard / name
                for shard in prepared
                for name in ("docs.jsonl", "spans.jsonl")
            ],
            {"shards": [str(shard) for shard in prepared], "policy": reference_policy},
            _INDEX_FILES,
            lambda d: completed_row_index.references(d, prepared, reference_policy),
        )
        assignment_path = (
            project_path(project.root, project.config.splits).parent
            / ("assignments-" + assignment["sha256"] + ".jsonl")
            if assignment
            else None
        )
        assigned = _run_phase(
            stages,
            "build-join-splits",
            [refs["references"], *([assignment_path] if assignment_path else [])],
            {"assignment": assignment},
            _INDEX_FILES,
            lambda d: completed_row_index.assign(d, refs, assignment, assignment_path),
        )
        if family_reference is not None:
            unbound = assigned
            assigned = _bind_build_phase(
                stages,
                "build-frozen-family-bindings",
                lambda: _run_phase(
                    stages,
                    "build-frozen-family-bindings",
                    [unbound["references"], unbound["ids"], family_reference["path"]],
                    {"family": family_reference},
                    _INDEX_FILES,
                    lambda d: completed_row_index.bind_families(
                        d, unbound, family_reference
                    ),
                ),
            )
        origins = _bind_build_phase(
            stages,
            "build-duplicate-keys",
            lambda: _run_phase(
                stages,
                "build-duplicate-keys",
                [assigned["references"]],
                {},
                _INDEX_FILES,
                lambda d: completed_row_index.origins(d, assigned),
            ),
        )
        deduplicated = _bind_build_phase(
            stages,
            "build-deduplicate",
            lambda: _run_phase(
                stages,
                "build-deduplicate",
                [assigned["references"], origins["origins"]],
                {"documents": assigned["documents"]},
                ("completed_row_dedup.py", "completed_row_index.py", "dedup_policy.py"),
                lambda d: completed_row_dedup.deduplicate(d, assigned, origins),
            ),
        )
        return cls(prepared, assigned, deduplicated)

    def final_family_stage(self, stages):
        """Reuse the final kept families by verified references and decisions."""
        return _run_phase(
            stages,
            "build-final-family-inventory",
            [self.assigned["references"], self.deduplicated["parents"]],
            {"documents": self.assigned["documents"]},
            ("streaming_build_rows.py",),
            lambda d: {
                "files": [
                    write_rows(d / "family-inventory.jsonl", self.family_inventory())
                ]
            },
        )

    def span_rows(self, after):
        first, offset = after or (0, 0)
        for number in range(first, len(self.prepared)):
            with (self.prepared[number] / "spans.jsonl").open("rb") as stream:
                stream.seek(offset)
                for raw in stream:
                    offset += len(raw)
                    yield (number, offset), raw.removesuffix(b"\n")
            offset = 0

    def family_splits(self):
        yield from self.assigned["family_splits"]

    def documents(self, after):
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
            references = stack.enter_context(
                Path(self.assigned["references"]).open("rb")
            )
            first, offset = after or (0, 0)
            references.seek(offset)
            for ordinal, line in enumerate(references, start=first):
                offset += len(line)
                row = json.loads(line)
                root = ordinal
                while (
                    parent := struct.unpack_from("<Q", parent_map, root * 8)[0]
                ) != root:
                    root = parent
                representative = ids[root * 64 : (root + 1) * 64].decode("ascii")
                raw = read_file_range(docs[row[1]], row[2], row[3])
                reason = (
                    "metadata_only_front_matter"
                    if row[8]
                    else "duplicate"
                    if root != ordinal
                    else None
                )
                yield (
                    raw.rstrip(b"\n"),
                    row[4],
                    representative,
                    reason,
                    row[0],
                    (ordinal + 1, offset),
                )

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
