"""Bounded-memory, resumable Corpus Forge build for prospective LM source pools.

Selected source bytes remain in the verified acquisition snapshots. A prepared
file is immutable only after its receipt and all three output hashes validate;
incomplete staging directories are never inputs to a resumed build.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections import Counter, OrderedDict, defaultdict, deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack, closing, nullcontext
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from klab.canonical import canonical_digest, canonical_json
from klab.corpus import native_dedup, native_index
from klab.corpus.build_outputs import (
    Tally,
    license_metadata_entry,
    publish_staging,
    record_overlap,
    release_report,
    rights_file_entry,
    rights_report,
    source_declarations,
    source_ledger,
    stage_declared_inputs,
    write_json,
    write_release_files,
)
from klab.corpus.build_stages import StageStore, implementation_digests, run_stage
from klab.corpus.duplicate_evidence import source_rank
from klab.corpus.normalized_documents import DOCUMENT_REFERENCES, TEXT_REFERENCE
from klab.corpus.normalized_retention import (
    bind_normalized_view,
    clear_normalized_use,
    record_normalized_use,
)
from klab.corpus.normalized_spans import (
    NORMALIZED_SPANS,
    SPAN_MEMBER,
    capture_span_partitions,
    seal_span_projection,
)
from klab.corpus.prepared_shards import (
    SourceParseError,
    prepare_file,
    prepared_file_exists,
)
from klab.corpus.progress import BuildProgress, memory_bytes
from klab.corpus.project import (
    ROW_SOURCE_KINDS,
    SPLITS,
    declaration_payload,
)
from klab.corpus.provenance import (
    HUMAN_ORIGIN,
    SOURCE_ORIGIN,
    rendered_digest,
    shape_for_record,
    validate_lineage,
    verification,
)
from klab.corpus.rights import (
    resolve_file_rights,
)
from klab.corpus.shards import range_file_identity
from klab.corpus.view_checkpoints import ViewCheckpoints
from klab.safe_io import read_file_range
from klab.training.manifest import sha256_file

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore, VerificationMode


@dataclass(frozen=True)
class CanarySample:
    """A deterministic slice of the acquired data for a canary build.

    Per source, the first `files_per_source` rights-eligible files in path order
    until `bytes_per_source` bytes are covered (the first file is always taken),
    and the first `rows_per_file` rows of each JSONL file.
    """

    files_per_source: int = 3
    rows_per_file: int = 1000
    bytes_per_source: int = 64 * 1024**2

    def __post_init__(self) -> None:
        if min(self.files_per_source, self.rows_per_file, self.bytes_per_source) < 1:
            raise ValueError("canary sample sizes must be at least 1")


_PROCESS_SHARD_MIN_BYTES = 32 * 1024**2
_PROCESS_SHARD_MAX_LINE_BYTES = 1024**2


def _output(handle: Any, value: Any) -> None:
    handle.write(canonical_json(value) + b"\n")


class _ShardProgress:
    """Accumulate worker counters; the parent journals in input order."""

    def __init__(self) -> None:
        self.documents = 0
        self.input_bytes = 0
        self.input_bytes_reused = 0
        self.output_records = 0
        self.reused = False

    def update(
        self,
        *,
        documents: int = 0,
        input_bytes: int = 0,
        input_bytes_reused: int = 0,
        output_records: int = 0,
    ) -> None:
        self.documents += documents
        self.input_bytes += input_bytes
        self.input_bytes_reused += input_bytes_reused
        self.output_records += output_records

    def record(self, event: str) -> None:
        if event == "reused_verified_shard":
            self.reused = True


def _prepare_shard(**kwargs: Any) -> tuple[Path, _ShardProgress]:
    progress = _ShardProgress()
    try:
        shard = prepare_file(**kwargs, progress=progress)
    except SourceParseError as error:
        error.documents = progress.documents
        error.input_bytes = progress.input_bytes
        error.output_records = progress.output_records
        raise
    return shard, progress


def _process_shard_workers(count: int) -> int:
    """Use the shared planner with the measured 512 MiB per-process bound."""
    if count < 2:
        return 1
    import psutil

    from klab import host_capacity
    from klab.host_capacity import plan_host_workers

    try:
        total = host_capacity.measure_memory().total
        return plan_host_workers(
            "corpus_jsonl_shards",
            worker_memory_bytes=512 * 1024**2,
            reserve_bytes=max(1024**3, total // 10),
            operator_cap=min(2, count),
        ).workers
    except AttributeError, OSError, psutil.Error, TypeError, ValueError:
        # Unknown or insufficient capacity retains the existing serial parser.
        return 1


def _bounded_jsonl_shard(path: Path) -> bool:
    """Keep oversized JSON rows on the serial path rather than guessing RSS."""
    with path.open("rb") as stream:
        while line := stream.readline(_PROCESS_SHARD_MAX_LINE_BYTES + 1):
            if len(line) > _PROCESS_SHARD_MAX_LINE_BYTES:
                return False
    return True


# Rights resolution inspects only 30 LF-delimited lines. This is a header
# evidence limit, not a bound on fallback normalization or admission screening.
_RIGHTS_HEADER_MAX_BYTES = 64 * 1024


def _read_rights_header(path: Path, name: str) -> bytes:
    """Read complete rights evidence within 64 KiB, plus one overflow probe byte.

    Match resolve_file_rights' case-sensitive dataset suffixes and binary LF
    boundaries (including CRLF and a final unterminated line). Never classify a
    truncated header: fewer than 30 lines require EOF within the byte ceiling.
    """
    if name.endswith((".jsonl", ".json", ".parquet")):
        return b""
    header = bytearray()
    with path.open("rb") as stream:
        for _ in range(30):
            line = stream.readline(_RIGHTS_HEADER_MAX_BYTES - len(header) + 1)
            if len(header) + len(line) > _RIGHTS_HEADER_MAX_BYTES:
                raise ValueError(
                    f"rights header exceeds {_RIGHTS_HEADER_MAX_BYTES} bytes: {name}"
                )
            header.extend(line)
            if not line.endswith(b"\n"):
                break
    return bytes(header)


def native_file_rights(project, source, name, path, nested_metadata):
    """Resolve a native partition's file-level admission before cache lookup."""
    if source.rights is None:
        return None
    return resolve_file_rights(
        source.rights,
        name,
        _read_rights_header(path, name),
        nested_metadata=nested_metadata,
        prospective_private_research=(
            project.release.policy.rights.training_use
            == "allowed_unless_explicitly_prohibited"
        ),
    )


def _require_every_source_has_documents(
    connection: sqlite3.Connection,
    project: Any,
    source_events: list[Path | dict[str, Any]],
) -> None:
    """Fail when a declared source contributed no document at all.

    This build is language-model only, so every declared source feeds the
    corpus. A source that yields nothing has had every file set aside (for
    example by a rights rule), and a quiet gap would shrink the training mix.
    """
    present = {
        source_id
        for (source_id,) in connection.execute(
            "SELECT DISTINCT source_id FROM documents"
        )
    }
    empty = sorted(source.id for source in project.sources if source.id not in present)
    if not empty:
        return
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, str] = {}
    for event in source_events:
        if isinstance(event, dict) and event["source_id"] in empty:
            reasons[event["source_id"]][event["reason"][:120]] += 1
            examples.setdefault(event["source_id"], event["path"])
    details = "; ".join(
        f"{source} (e.g. file {examples.get(source, 'none seen')}: "
        + (
            ", ".join(f"{n} x {why}" for why, n in reasons[source].most_common(3))
            or "no file reached the normalizer"
        )
        + ")"
        for source in empty
    )
    raise ValueError(
        f"declared source(s) produced no documents: {details}. Every file was set "
        "aside (see rejected.jsonl and license-report.json); the options that "
        "decide this are release.policy.rights.training_use (rights evidence), "
        "release.policy.documents.empty_text and the source's own rights block"
    )


def _write_audit(
    target: Path, groups: Path, rejected: Path, *, dropped: int, leakage=(), warnings=()
) -> None:
    with target.open("wb") as output:
        output.write(b'{"drop_count":' + str(dropped).encode() + b',"duplicates":[')
        with groups.open("rb") as source:
            for index, row in enumerate(source):
                if index:
                    output.write(b",")
                output.write(row.rstrip(b"\n"))
        output.write(b'],"leakage":' + canonical_json(list(leakage)) + b',"rejected":[')
        with rejected.open("rb") as source:
            for index, row in enumerate(source):
                if index:
                    output.write(b",")
                output.write(row.rstrip(b"\n"))
        output.write(b'],"warnings":' + canonical_json(list(warnings)) + b"}\n")
        output.flush()
        os.fsync(output.fileno())


def _document_lineage(
    doc: dict[str, Any], source_origins: dict[str, str], normalizer: str
) -> dict[str, Any]:
    record_id = doc["document_id"]
    representative = doc["representative_id"]
    row = {
        "record_id": record_id,
        "record_kind": "document",
        "transform_id": doc.get("metadata", {}).get("normalizer", normalizer),
        "parent_document_ids": [record_id],
        "original_parent_document_ids": [record_id],
        "representative_id": representative,
        "drop_reason": doc["drop_reason"],
        "source_family_ids": [doc["source_family"]],
        "split": doc["split"],
        "representative_parent_document_ids": [representative],
        "origin": (
            HUMAN_ORIGIN
            if source_origins[doc["source_id"]] == HUMAN_ORIGIN
            else SOURCE_ORIGIN
        ),
        "modalities": [doc["modality"]],
        "origin_schema_version": 1,
        "domains": doc["domains"],
        "shape": shape_for_record(
            "document",
            "raw_document",
            domains=doc["domains"],
            parent_document_ids=[record_id],
        ),
        "verification": verification(
            "schema_validated",
            doc.get("metadata", {}).get("normalizer", normalizer),
            {"schema_id": "normalized_document_v1"},
        ),
        "validation_status": "schema_validated",
        "rendered_sha256": rendered_digest(doc["text"], text=True),
    }
    validate_lineage(row, documents={record_id: doc})
    return row


def _write_rejected(target: Path, source_events: list[Path | dict[str, Any]]) -> None:
    with target.open("wb") as output:
        output.writelines(native_dedup.rejected_rows(source_events))
        output.flush()
        os.fsync(output.fileno())


_DOCUMENT_READ_BYTES = 8 * 1024**2
_DOCUMENT_READ_ROWS = 2048
_DOCUMENT_OPEN_FILES = 64
_MAX_DOCUMENT_CACHE_BYTES = 1024**3


class _SQLBuildRows:
    """The ordinary builder's ordered inputs to the shared release writer."""

    def __init__(self, connection, prepared, annotations=None):
        self.connection = connection
        self.prepared = prepared
        self.annotations = annotations

    def span_rows(self, after: int | None):
        with ExitStack() as sources:
            current, position = None, 0
            for rowid, shard, offset, length in self.connection.execute(
                "SELECT rowid,shard,span_offset,span_length FROM documents "
                "WHERE rowid>? ORDER BY rowid",
                (after or 0,),
            ):
                if shard != current:
                    sources.close()
                    stream = sources.enter_context(
                        (self.prepared[shard] / "spans.jsonl").open("rb")
                    )
                    current, position = shard, 0
                if offset != position:
                    stream.seek(offset)
                raw = stream.read(length)
                if len(raw) != length:
                    raise ValueError("prepared span is truncated")
                position = offset + length
                yield rowid, raw.removesuffix(b"\n")

    def documents(self, after: str | None):
        ordinal = "d.rowid" if self.annotations is not None else "NULL"
        references = self.connection.execute(
            "SELECT d.shard,d.document_offset,d.document_length,d.split,"
            f"x.representative_id,x.drop_reason,d.document_id,{ordinal} "
            "FROM documents d CROSS JOIN dedup x ON x.document_number=d.rowid "
            "WHERE d.document_id>? ORDER BY d.document_id",
            (after or "",),
        )
        streams = OrderedDict()
        try:
            if not after and self._warm_document_cache():
                for (
                    shard,
                    offset,
                    length,
                    split,
                    representative,
                    reason,
                    doc_id,
                    rowid,
                ) in references:
                    stream = self._document_stream(streams, shard)
                    raw = read_file_range(stream, offset, length)
                    evidence = self.annotations[rowid] if rowid is not None else None
                    yield (
                        raw.removesuffix(b"\n"),
                        split,
                        representative,
                        reason,
                        doc_id,
                        doc_id,
                        evidence,
                        (shard, offset, length),
                    )
                return
            batch = []
            size = 0
            for reference in references:
                if batch and (
                    size + reference[2] > _DOCUMENT_READ_BYTES
                    or len(batch) >= _DOCUMENT_READ_ROWS
                ):
                    yield from self._read_documents(batch, streams)
                    batch, size = [], 0
                batch.append(reference)
                size += reference[2]
            yield from self._read_documents(batch, streams)
        finally:
            for stream in streams.values():
                stream.close()

    def _document_stream(self, streams, shard):
        if shard not in streams:
            if len(streams) == _DOCUMENT_OPEN_FILES:
                streams.popitem(last=False)[1].close()
            streams[shard] = (self.prepared[shard] / "docs.jsonl").open("rb")
        streams.move_to_end(shard)
        return streams[shard]

    def _warm_document_cache(self) -> bool:
        """Use sequential reads for a cold working set that fits a bounded cache budget."""
        nowait = getattr(os, "RWF_NOWAIT", None)
        if nowait is None or not hasattr(os, "preadv"):
            return False
        files = [shard / "docs.jsonl" for shard in self.prepared]
        sizes = [path.stat().st_size for path in files]
        _, _, available = memory_bytes()
        if not available or sum(sizes) > min(available // 8, _MAX_DOCUMENT_CACHE_BYTES):
            return False
        buffer = bytearray(1024**2)
        probe = memoryview(buffer)[:1]
        for path, size in zip(files, sizes, strict=True):
            if not size:
                continue
            with path.open("rb") as stream:
                try:
                    for offset in (0, size // 2, size - 1):
                        os.preadv(stream.fileno(), [probe], offset, nowait)
                except BlockingIOError:
                    while stream.readinto(buffer):
                        pass
                except OSError:
                    return False
        return True

    def _read_documents(self, batch, streams):
        # Improve locality within a bounded batch and emit the original ID order.
        payloads = [b""] * len(batch)
        order = sorted(range(len(batch)), key=lambda i: batch[i][:2])
        advise = getattr(os, "posix_fadvise", None)
        for shard, indices in groupby(order, key=lambda i: batch[i][0]):
            stream = self._document_stream(streams, shard)
            indices = list(indices)
            if advise is not None and len(indices) > 1:
                try:
                    descriptor = stream.fileno()
                    for index in indices:
                        _, offset, length, *_ = batch[index]
                        advise(descriptor, offset, length, os.POSIX_FADV_WILLNEED)
                except OSError:
                    advise = None
            for index in indices:
                _, offset, length, *_ = batch[index]
                payloads[index] = read_file_range(stream, offset, length)
        for raw, (
            shard,
            offset,
            length,
            split,
            representative,
            reason,
            doc_id,
            rowid,
        ) in zip(payloads, batch, strict=True):
            evidence = self.annotations[rowid] if rowid is not None else None
            yield (
                raw.removesuffix(b"\n"),
                split,
                representative,
                reason,
                doc_id,
                doc_id,
                evidence,
                (shard, offset, length),
            )

    def family_splits(self):
        for (family,) in self.connection.execute(
            "SELECT source_family FROM documents "
            "GROUP BY source_family HAVING COUNT(DISTINCT split)>1 ORDER BY source_family"
        ):
            yield (
                family,
                [
                    row[0]
                    for row in self.connection.execute(
                        "SELECT DISTINCT split FROM documents WHERE source_family=? ORDER BY split",
                        (family,),
                    )
                ],
            )


_VIEW_CHECKPOINT_ROWS = 25_000
_VIEW_CHECKPOINT_BYTES = 64 * 1024**2


def native_view_members(
    stage_id: str, *, reference_normalized_spans: bool = False
) -> tuple[str, ...]:
    """Logical outputs whose prefixes the native view journal tracks."""
    return (
        "documents.jsonl",
        *((SPAN_MEMBER,) if not reference_normalized_spans else ()),
        "lineage.jsonl",
        f"stages/{stage_id}.jsonl",
        *(f"lm/{split}.jsonl" for split in SPLITS),
        *(f"lm/{split}.lineage.jsonl" for split in SPLITS),
    )


def _emit_build(
    *,
    connection: sqlite3.Connection,
    staging: Path,
    project: Any,
    lock: dict[str, Any],
    identity: dict[str, Any],
    build_id: str,
    transform: Any,
    prepared: list[Path],
    rights_files: list[dict[str, Any]],
    source_events: list[Path | dict[str, Any]],
    groups_path: Path,
    group_count: int,
    dropped: int,
    progress: BuildProgress,
    checkpoints: ViewCheckpoints,
    ordered_rows=None,
    descriptive_census=True,
    diagnostics: Path | None = None,
    span_partitions: list[dict[str, Any]] | None = None,
    span_projection: dict[str, Any] | None = None,
    train_annotations=None,
    duplicate_evidence: dict[str, Any] | None = None,
) -> None:
    from klab.corpus.measurement import summarize_release
    from klab.corpus.train_selection import (
        BELOW_MINIMUM,
        CHECK_ROW_LABEL,
        minimum_train_bytes,
        train_row_selection,
    )

    ordered_rows = ordered_rows or _SQLBuildRows(
        connection, prepared, train_annotations
    )
    sources = source_declarations(project, lock)
    rights = rights_report(project, sources, rights_files)
    source_origins = {source.id: source.origin for source in project.sources}
    release_spec = declaration_payload(project.release)
    stage_spec = declaration_payload(transform)
    minimum = minimum_train_bytes([stage_spec])
    row_selection = train_row_selection([stage_spec])
    filtering = minimum is not None or row_selection is not None
    completed = identity.get("completed_preparation")
    if completed is None and filtering != (train_annotations is not None):
        raise ValueError("TRAIN selection requires verified sidecars")
    stage_id = stage_spec["id"]
    selected_sources = set(stage_spec["inputs"])
    tally = Tally()
    lm_counts: Counter[str] = Counter()
    train_bytes: Counter[str] = Counter()
    excluded_train = Counter()
    if minimum is not None:
        excluded_train[BELOW_MINIMUM] = 0
    reasons = row_selection.reasons() if row_selection is not None else {}
    excluded_train.update(dict.fromkeys(reasons.values(), 0))
    count = 0
    stage_count = 0
    stage_digest = hashlib.sha256()
    stage_digest.update(b"[" + canonical_json(stage_id) + b",[")

    document_overlay = bool(identity.get("reference_normalized_documents"))
    if document_overlay:
        if span_projection is None:
            raise ValueError("normalized document text requires a span projection")
        write_json(staging / DOCUMENT_REFERENCES, span_projection)

    with ExitStack() as stack:

        def writer(name: str, *, resumable: bool = True) -> Any:
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if resumable:
                if path.is_symlink():
                    raise ValueError(f"symlinked corpus view: {path}")
                stream = stack.enter_context(path.open("a+b"))
                return checkpoints.add(name, stream)
            return stack.enter_context(path.open("wb"))

        resumable = {
            name: writer(name)
            for name in native_view_members(
                stage_id, reference_normalized_spans=span_projection is not None
            )
        }
        documents = resumable["documents.jsonl"]
        lineage = resumable["lineage.jsonl"]
        stage = resumable[f"stages/{stage_id}.jsonl"]
        views = {split: resumable[f"lm/{split}.jsonl"] for split in SPLITS}
        links = {split: resumable[f"lm/{split}.lineage.jsonl"] for split in SPLITS}
        for path in (
            "lexical/candidates.jsonl",
            "semantic/candidates.jsonl",
            "scenarios.jsonl",
            "generations.jsonl",
            "tool_episodes.jsonl",
            "chat/records.jsonl",
        ):
            writer(path, resumable=False)
        for split in SPLITS:
            writer(f"chat/{split}.jsonl", resumable=False)
            writer(f"chat/{split}.lineage.jsonl", resumable=False)

        span_cursor = document_cursor = None
        pending_rows, pending_bytes = 0, 0
        state = checkpoints.restore()
        if checkpoints.damaged:
            progress.record("discarded_damaged_view_checkpoint")
        if state:
            tally.restore(state["tally"])
            lm_counts.update(state["lm_counts"])
            span_cursor, document_cursor = (
                state["span_cursor"],
                state["document_cursor"],
            )
            count, stage_count = state["count"], state["stage_count"]
            if filtering:
                excluded_train.update(state["excluded_train"])
                train_bytes.update(state["selected_train_bytes"])
            with (staging / f"stages/{stage_id}.jsonl").open("rb") as prefix:
                for index, line in enumerate(prefix):
                    if index:
                        stage_digest.update(b",")
                    stage_digest.update(line.removesuffix(b"\n"))
            progress.update(output_records=sum(lm_counts.values()))
            progress.record("reused_view_checkpoints")

        def boundary(size: int, *, finished: bool = False) -> None:
            nonlocal pending_rows, pending_bytes
            pending_rows += not finished
            pending_bytes += size
            if not (
                (finished and pending_rows)
                or pending_rows >= _VIEW_CHECKPOINT_ROWS
                or pending_bytes >= _VIEW_CHECKPOINT_BYTES
            ):
                return
            checkpoints.commit(
                {
                    "span_cursor": span_cursor,
                    "document_cursor": document_cursor,
                    "count": count,
                    "stage_count": stage_count,
                    "lm_counts": lm_counts,
                    "tally": tally.snapshot(),
                    **(
                        {
                            "excluded_train": excluded_train,
                            "selected_train_bytes": train_bytes,
                        }
                        if filtering
                        else {}
                    ),
                }
            )
            pending_bytes = pending_rows = 0

        # Source spans retain acquisition order; documents and views use ID order.
        if spans := resumable.get(SPAN_MEMBER):
            span_rows = ordered_rows.span_rows(span_cursor)
            for span_cursor, span in stack.enter_context(closing(span_rows)):
                spans.write(span + b"\n")
                boundary(len(span))
            boundary(0, finished=True)
        document_rows = ordered_rows.documents(document_cursor)
        for (
            raw_doc,
            split,
            representative,
            reason,
            document_id,
            document_cursor,
            annotation,
            *locator,
        ) in stack.enter_context(closing(document_rows)):
            doc = json.loads(raw_doc)
            if doc["document_id"] != document_id:
                raise ValueError("prepared document differs from the indexed identity")
            doc["split"] = split
            doc["representative_id"] = representative
            doc["drop_reason"] = reason
            if document_overlay:
                (reference,) = locator
                _output(
                    documents,
                    {
                        **{key: value for key, value in doc.items() if key != "text"},
                        TEXT_REFERENCE: reference,
                    },
                )
            else:
                _output(documents, doc)
            _output(
                lineage,
                _document_lineage(
                    doc, source_origins, project.release.normalizer.identity
                ),
            )
            count += 1
            record_id = doc["document_id"]
            origin = (
                HUMAN_ORIGIN
                if source_origins[doc["source_id"]] == HUMAN_ORIGIN
                else SOURCE_ORIGIN
            )
            if reason is not None:
                tally.add_dropped(doc)
                boundary(len(raw_doc))
                continue
            tally.add(doc)
            text = doc["text"]

            selected_for_stage = (
                not selected_sources or doc["source_id"] in selected_sources
            )
            if selected_for_stage and filtering and split == "train":
                size, row_reason = annotation if annotation is not None else (None, 0)
                if minimum is not None:
                    if size is None:
                        # Completed source renders were authenticated before emission.
                        size = len(text.encode("utf-8"))
                    if size < minimum:
                        excluded_train[BELOW_MINIMUM] += 1
                        boundary(len(raw_doc))
                        continue
                if row_selection is not None and (
                    completed is not None or row_reason == CHECK_ROW_LABEL
                ):
                    row_reason = row_selection.document_decision(doc)
                if row_reason:
                    excluded_train[reasons[row_reason]] += 1
                    boundary(len(raw_doc))
                    continue
            if selected_for_stage:
                output = {"record_id": record_id, "text": text, "split": split}
                encoded = canonical_json(output)
                stage.write(encoded + b"\n")
                if stage_count:
                    stage_digest.update(b",")
                stage_digest.update(encoded)
                stage_count += 1
            if (
                selected_for_stage
                and (
                    release_spec.get("include_shapes") is None
                    or "raw_document" in release_spec["include_shapes"]
                )
                and (
                    release_spec.get("include_origins") is None
                    or origin in release_spec["include_origins"]
                )
            ):
                _output(views[split], {"text": text})
                _output(links[split], {"record_id": record_id, "split": split})
                lm_counts[split] += 1
                if filtering and split == "train":
                    if size is None:
                        size = len(text.encode("utf-8"))
                    for domain in doc["domains"]:
                        train_bytes[domain] += size
                progress.update(output_records=1)
            boundary(len(raw_doc))
        boundary(0, finished=True)

        view_digests = checkpoints.file_digests()
        progress.observe_stage_output(
            "views",
            checkpoints.identity,
            sum(digest.size for digest in view_digests.values()),
        )

    stage_digest.update(b"]]")
    stages = [
        {
            "id": stage_id,
            "kind": stage_spec["kind"],
            "version": stage_spec["version"],
            "parameters_sha256": canonical_digest(stage_spec["parameters"]),
            "inputs": stage_spec["inputs"],
            "implementation_sha256": identity["implementation_sha256"],
            "output_count": stage_count,
            "output_id": stage_digest.hexdigest(),
        }
    ]
    _write_rejected(staging / "rejected.jsonl", source_events)
    warnings: list[str] = []
    leakage: list[dict[str, Any]] = []
    # LM-only lineages have no generated families. Source-family overlap is
    # descriptive unless that family is the declared split policy unit.
    for family, splits in ordered_rows.family_splits():
        record_overlap(
            leakage,
            warnings,
            unit=project.splits.unit,
            field="source_family_ids",
            family=family,
            splits=splits,
            diagnostics=diagnostics
            or staging.parent / "diagnostics" / f"{build_id}.json",
        )
    _write_audit(
        staging / "audit.json",
        groups_path,
        staging / "rejected.jsonl",
        dropped=dropped,
        leakage=leakage,
        warnings=warnings,
    )
    write_json(staging / "splits.json", project.splits.model_dump(mode="json"))
    write_json(staging / "sources.json", sources)
    write_json(staging / "license-report.json", rights)

    with (staging / "rejected.jsonl").open("rb") as rejected_rows:
        ledger = source_ledger(
            project, rights_files, (json.loads(line) for line in rejected_rows), tally
        )
    report = release_report(
        project,
        rights,
        tally,
        ledger,
        document_count=count,
        view_counts={
            "lm": {split: lm_counts[split] for split in SPLITS},
            "chat": dict.fromkeys(SPLITS, 0),
        },
        duplicate_groups=group_count,
        warnings=warnings,
    )
    if duplicate_evidence is not None:
        report["duplicate_evidence"] = duplicate_evidence
    if filtering:
        from klab.corpus.quota import estimate_supply

        selection_report = {"excluded_by_reason": dict(excluded_train)}
        if completed is not None:
            selection_report["source_render_ids"] = completed["source_stage_ids"]
        if minimum is not None:
            selection_report["minimum_utf8_bytes"] = minimum
        if row_selection is not None:
            selection_report.update(row_selection.report())
        if completed is None:
            selection_report.update(train_annotations.inventories)
        report["train_selection"] = selection_report
        report["mixture_supply"] = estimate_supply(
            project.release.policy.mixture, project.release.mixture, train_bytes
        )
    progress.update(phase="measuring")
    if not descriptive_census:
        report["measurement"] = {
            "status": "deferred",
            "reason": "Descriptive lineage census is outside the selected completed local preparation scope",
            "token_count_reason": "tokenizer_not_declared",
        }
    else:
        store = StageStore(checkpoints.root / "census")
        store.discard_incomplete("measurement")

        def measure(output: Path, result: dict[str, Any]) -> None:
            summary = summarize_release(
                staging,
                release_spec=release_spec,
                cache=checkpoints.root / "measurement-cache",
            )
            write_json(output / "measurement.json", summary)

        census_dir, _ = run_stage(
            store,
            "measurement",
            {
                "views": checkpoints.identity,
                "frontier": checkpoints.previous,
                "release": canonical_digest(release_spec),
                "implementation": implementation_digests(
                    "measurement.py", "provenance.py", "release_verify.py"
                ),
            },
            measure,
            on_reuse=lambda: progress.record("reused_build_census"),
        )
        report["measurement"] = json.loads(
            (census_dir / "measurement.json").read_text(encoding="utf-8")
        )
    progress.update(phase="writing_manifest")
    write_release_files(
        staging,
        project,
        report,
        build_id=build_id,
        identity=identity,
        stages=stages,
        lock=lock,
        view_digests=view_digests,
        span_partitions=span_partitions,
        span_projection=span_projection,
    )


def _verify_admitted_inventory(
    connection: sqlite3.Connection,
    admissions: dict,
    admission_files: tuple[tuple[str, str], ...],
) -> None:
    """Compare qualified and observed source identities without materialized sets."""
    from klab.corpus.admission_coverage import AdmissionCoverage

    with AdmissionCoverage(connection) as coverage:
        for source, path in admission_files:
            for index, choice in admissions[source, path]["decisions"].items():
                if choice["decision"] == "qualify":
                    coverage.expect(source, path, index)
        for number, index in connection.execute(
            "SELECT file_number, source_row_index FROM indexed.admitted_document_rows"
        ):
            source, path = admission_files[number]
            coverage.observe(source, path, index)
        coverage.verify()


# While a build runs, a newly published stage output evicts all but the newest
# earlier one, so a failed attempt and the one before it stay available.
_PRUNE_ON_PUBLISH = 2


def _open_read_only(database: Path):
    return closing(sqlite3.connect(f"file:{quote(str(database))}?mode=ro", uri=True))


def _open_decided(index_dir: Path, dedup_dir: Path):
    """Read-only view of the index joined to its deduplication decisions."""
    connection = sqlite3.connect(
        f"file:{quote(str(dedup_dir / 'dedup.sqlite'))}?mode=ro", uri=True
    )
    connection.execute(
        "ATTACH DATABASE ? AS indexed",
        (f"file:{quote(str(index_dir / 'index.sqlite'))}?mode=ro",),
    )
    return closing(connection)


def native_index_inputs(
    project: Any,
    prepared: list[Path],
    *,
    admission_files: tuple[tuple[str, str], ...] = (),
) -> dict[str, Any]:
    """Resolve the index identity from ordered prepared receipts and policy.

    The index follows the receipts of the prepared shards (which bind the
    source bytes, rights, normalizer and document policy), the split
    declaration, evidence methods, source ranks and the ordered files with
    row-level admission evidence.
    """
    index = {
        "shards": [
            {
                "source": shard.parent.name,
                "receipt_sha256": sha256_file(shard / "receipt.json"),
            }
            for shard in prepared
        ],
        "split": project.splits.model_dump(mode="json"),
        "duplicate_evidence_methods": native_index.evidence_methods(
            project.release.policy.duplicates
        ),
        "source_rank": {
            source.id: source_rank(source)
            for source in sorted(project.sources, key=lambda item: item.id)
        },
        "implementation": implementation_digests(
            *native_index.INDEX_IMPLEMENTATION_FILES
        ),
    }
    if admission_files:
        index["row_admission_files"] = admission_files
    return {"stage": "index", **index}


def native_dedup_inputs(project: Any, index_receipt_sha256: str) -> dict[str, Any]:
    """Resolve duplicate policy against the exact index receipt it reads."""
    return {
        "stage": "dedup",
        "duplicates": project.release.policy.duplicates.model_dump(mode="json"),
        "implementation": implementation_digests(
            *native_dedup.DEDUP_IMPLEMENTATION_FILES
        ),
        "index": index_receipt_sha256,
    }


def native_prepared_members(prepared: list[Path]) -> list[Path]:
    """Return the ordered payloads that bind native index and view generations."""
    return [
        shard / name for shard in prepared for name in ("docs.jsonl", "spans.jsonl")
    ]


def native_view_checkpoints(
    workspace: Path,
    build_id: str,
    *,
    index_receipt_sha256: str,
    dedup_receipt_sha256: str,
    generations: list[list[int]],
) -> ViewCheckpoints:
    """Resolve the journal for this build and the exact retained stage inputs."""
    key = canonical_digest(
        {
            "build": build_id,
            "index_receipt": index_receipt_sha256,
            "dedup_receipt": dedup_receipt_sha256,
            "prepared_generations": generations,
            "implementation": implementation_digests(
                "view_checkpoints.py", "build_outputs.py"
            ),
        }
    )
    return ViewCheckpoints(workspace / f".views-{key}", key)


def _sample_file(
    canary: CanarySample,
    taken: dict[str, tuple[int, int]],
    source_id: str,
    file: dict[str, Any],
) -> bool:
    """Whether the canary sample includes this eligible file; records it if so.

    `taken` maps a source to the number of files and bytes already sampled.
    """
    files, covered = taken.get(source_id, (0, 0))
    if files and (
        files >= canary.files_per_source or covered >= canary.bytes_per_source
    ):
        return False
    taken[source_id] = (files + 1, covered + min(file["size"], canary.bytes_per_source))
    return True


def build_large(
    project: Any,
    workspace: Path,
    lock: dict[str, Any],
    identity: dict[str, Any],
    build_id: str,
    target: Path,
    progress: BuildProgress,
    *,
    work_root: Path,
    prepared_root: Path,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    record_admission: dict | None = None,
    admission_path: Path | None = None,
    keep_stages: int = 0,
    canary: CanarySample | None = None,
) -> Path:
    """Build LM-only v2/v3 inputs using their declared admission/dedup policies."""
    record_admission = record_admission or {}
    if canary is not None and record_admission:
        raise ValueError(
            "a canary build cannot sample rows under release.record_admission: "
            "every reviewed row must be present; build without --canary"
        )
    from klab.corpus.acquisition import verify_snapshot

    root = workspace / "builds"
    root.mkdir(parents=True, exist_ok=True)
    if canary is not None:
        prepared_root = root / "prepared"
    prepared_root.mkdir(parents=True, exist_ok=True)
    sources = sorted(project.sources, key=lambda source: source.id)
    source_ids = [source.id for source in sources]
    if canary is None:
        record_normalized_use(workspace.resolve(), build_id, source_ids)
    transform = project.transforms[0]
    rights_files: list[dict[str, Any]] = []
    source_events: list[Path | dict[str, Any]] = []
    prepared: list[Path] = []
    prepared_raw_paths: list[str] = []
    inputs: list[tuple[Any, dict[str, Any], Path, dict[str, Any], dict[str, Any]]] = []
    total_bytes = 0
    progress.record("verifying_snapshots")
    for source in sources:
        lock_row = lock["sources"].get(source.id)
        if not lock_row or not lock_row.get("snapshot_path"):
            continue
        snapshot_path = Path(lock_row["snapshot_path"])
        snapshot = verify_snapshot(
            snapshot_path, proof_store=proof_store, verification_mode=verification_mode
        )
        nested_path = source.rights.nested_metadata_path if source.rights else None
        nested_metadata = (
            {
                nested_path: json.loads(
                    (snapshot_path / "files" / nested_path).read_text(encoding="utf-8")
                )
            }
            if nested_path
            else {}
        )
        for file in sorted(snapshot["files"], key=lambda item: item["path"]):
            inputs.append(
                (
                    source,
                    file,
                    snapshot_path / "files" / file["path"],
                    lock_row,
                    nested_metadata,
                )
            )
            if file["path"] != nested_path:
                total_bytes += file["size"]
    progress.expected_input_bytes = total_bytes
    progress.update(phase="normalizing")

    def cached_jsonl(source, file, lock_row, nested_metadata):
        rights = (
            resolve_file_rights(
                source.rights,
                file["path"],
                b"",
                nested_metadata=nested_metadata,
                prospective_private_research=(
                    project.release.policy.rights.training_use
                    == "allowed_unless_explicitly_prohibited"
                ),
            )
            if source.rights
            else None
        )
        return prepared_file_exists(
            prepared_root,
            source,
            lock_row["snapshot_sha256"],
            file,
            project.release.normalizer,
            project.release.policy.documents,
            rights,
            canary.rows_per_file if canary is not None else None,
        )

    eligible_jsonl = {
        path
        for source, file, path, lock_row, nested_metadata in inputs
        if source.kind in ROW_SOURCE_KINDS
        and file["path"].endswith(".jsonl")
        and file["size"] >= _PROCESS_SHARD_MIN_BYTES
        and file["path"] not in nested_metadata
        and not record_admission
        # Cached sources reach exact identity/rights checking in prepare_file.
        # Do not pre-scan their bodies merely to choose parallel workers.
        and not cached_jsonl(source, file, lock_row, nested_metadata)
    }
    workers = _process_shard_workers(len(eligible_jsonl))
    if workers > 1:
        eligible_jsonl = {path for path in eligible_jsonl if _bounded_jsonl_shard(path)}
        workers = _process_shard_workers(len(eligible_jsonl))
    pool = (
        ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
    )
    sampled: dict[str, tuple[int, int]] = {}
    pending: deque[tuple[Future[tuple[Path, _ShardProgress]], Any, dict[str, Any]]] = (
        deque()
    )

    def finish_one() -> None:
        future, source, file = pending.popleft()
        try:
            shard, counters = future.result()
        except SourceParseError as exc:
            source_events.append(
                {"source_id": source.id, "path": file["path"], "reason": str(exc)}
            )
            progress.update(
                documents=exc.documents,
                input_bytes=file["size"] + exc.input_bytes,
                output_records=exc.output_records,
            )
            progress.record("source_rejected")
            return
        progress.update(
            documents=counters.documents,
            input_bytes=counters.input_bytes,
            input_bytes_reused=counters.input_bytes_reused,
            output_records=counters.output_records,
        )
        if counters.reused:
            progress.record("reused_verified_shard")
        prepared.append(shard)
        prepared_raw_paths.append(file["path"])
        source_events.append(shard)

    with pool as executor:
        for source, file, path, lock_row, nested_metadata in inputs:
            name = file["path"]
            if name in nested_metadata:
                while pending:
                    finish_one()
                rights_files.append(
                    license_metadata_entry(
                        source, path=name, sha256=file["sha256"], size=file["size"]
                    )
                )
                continue
            decision = native_file_rights(project, source, name, path, nested_metadata)
            admitted = record_admission.get((source.id, name))
            if admitted is not None:
                decision = admitted["rights"]
            if (
                canary is not None
                and (
                    decision is None
                    or decision.training_eligibility
                    in {"eligible", "eligible_with_obligations"}
                )
                and not _sample_file(canary, sampled, source.id, file)
            ):
                continue
            if decision:
                rights_files.append(
                    rights_file_entry(
                        source,
                        path=name,
                        sha256=file["sha256"],
                        size=file["size"],
                        role="document",
                        rights=decision.model_dump(mode="json"),
                        admitted=admitted,
                    )
                )
                if decision.training_eligibility not in {
                    "eligible",
                    "eligible_with_obligations",
                }:
                    while pending:
                        finish_one()
                    source_events.append(
                        {
                            "source_id": source.id,
                            "path": name,
                            "reason": f"rights {decision.training_eligibility}: {decision.reason}",
                        }
                    )
                    progress.update(input_bytes=file["size"])
                    continue
            kwargs = {
                "source": source,
                "file": file,
                "source_path": path,
                "decision": decision,
                "snapshot_sha": lock_row["snapshot_sha256"],
                "prepared_root": prepared_root,
                "record_admission": admitted,
                "admission_sha256": project.release.record_admission.sha256.lower()
                if admitted is not None
                else None,
                "normalizer_options": project.release.normalizer,
                "document_policy": project.release.policy.documents,
                "row_limit": canary.rows_per_file if canary is not None else None,
            }
            if executor is not None and path in eligible_jsonl:
                pending.append(
                    (executor.submit(_prepare_shard, **kwargs), source, file)
                )
                if len(pending) >= workers:
                    finish_one()
                continue
            while pending:
                finish_one()
            try:
                shard = prepare_file(**kwargs, progress=progress)
            except SourceParseError as exc:
                source_events.append(
                    {"source_id": source.id, "path": name, "reason": str(exc)}
                )
                progress.update(input_bytes=file["size"])
                progress.record("source_rejected")
                continue
            prepared.append(shard)
            prepared_raw_paths.append(name)
            source_events.append(shard)
        while pending:
            finish_one()

    stages = StageStore(root / "stages")
    admission_files = tuple(
        sorted(
            key for key, admitted in record_admission.items() if "decisions" in admitted
        )
    )
    index_inputs = native_index_inputs(
        project, prepared, admission_files=admission_files
    )
    span_partitions = (
        capture_span_partitions(prepared, prepared_raw_paths, index_inputs["shards"])
        if canary is None and not record_admission
        else None
    )
    prepared_files = native_prepared_members(prepared)
    generations = [range_file_identity(path) for path in prepared_files]
    index_key = stages.key(index_inputs)
    groups_root = stages.root / f".index-{index_key}.groups"
    if groups_root.is_symlink():
        raise ValueError(f"symlinked index checkpoint: {groups_root}")
    if canary is None:
        record_normalized_use(
            workspace.resolve(),
            build_id,
            source_ids,
            partitions=prepared,
            checkpoints=(groups_root.name,),
        )
    group_store = StageStore(groups_root)
    progress.update(phase="indexing")

    def require_prepared_generation() -> None:
        if [range_file_identity(path) for path in prepared_files] != generations:
            raise ValueError("prepared payloads changed during native build")

    def index_stage(staging: Path, result: dict[str, Any]) -> None:
        with progress.track_stage_workspace("index", index_key, groups_root):
            result["documents"] = native_index.index_prepared(
                staging,
                prepared,
                project,
                progress,
                admission_files=admission_files,
                checkpoint_store=group_store,
                index_key=index_key,
                generations=generations,
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
            with _open_read_only(staging / "index.sqlite") as connection:
                result["duplicate_evidence"] = native_index.group_counts(
                    connection, index_inputs["duplicate_evidence_methods"]
                )
            require_prepared_generation()

    with (workspace / ".index.lock").open("a") as index_lock:
        fcntl.flock(index_lock, fcntl.LOCK_EX)
        stages.discard_incomplete("index")
        group_store.discard_incomplete("index-group")
        index_dir, index_result = run_stage(
            stages,
            "index",
            index_inputs,
            index_stage,
            on_reuse=lambda: progress.record("reused_index_stage"),
            retained=groups_root / "assembly",
        )
        progress.observe_stage_output(
            "index", index_key, (index_dir / "index.sqlite").stat().st_size
        )
        if groups_root.exists():
            shutil.rmtree(groups_root)
        stages.prune("index", keep=_PRUNE_ON_PUBLISH, protect=index_dir.name)
    with _open_read_only(index_dir / "index.sqlite") as connection:
        _require_every_source_has_documents(connection, project, source_events)
    if admission_files:
        progress.update(phase="checking_admissions")
        # The coverage check writes a table, so it runs in a scratch database
        # that reads the checkpointed index and never changes it.
        with tempfile.TemporaryDirectory(prefix=".coverage-", dir=root) as scratch:
            scratch_database = quote(str(Path(scratch) / "coverage.sqlite"))
            with closing(
                sqlite3.connect(f"file:{scratch_database}", uri=True)
            ) as connection:
                connection.execute(
                    "ATTACH DATABASE ? AS indexed",
                    (f"file:{quote(str(index_dir / 'index.sqlite'))}?mode=ro",),
                )
                _verify_admitted_inventory(
                    connection, record_admission, admission_files
                )
    progress.update(phase="deduplicating")
    index_receipt_sha256 = sha256_file(index_dir / "receipt.json")
    dedup_inputs = native_dedup_inputs(project, index_receipt_sha256)
    dedup_key = stages.key(dedup_inputs)
    dedup_groups_root = stages.root / f".dedup-{dedup_key}.groups"
    if dedup_groups_root.is_symlink():
        raise ValueError(f"symlinked dedup checkpoint: {dedup_groups_root}")
    if canary is None:
        record_normalized_use(
            workspace.resolve(),
            build_id,
            source_ids,
            partitions=prepared,
            checkpoints=(groups_root.name, dedup_groups_root.name),
        )

    def dedup_stage(staging: Path, result: dict[str, Any]) -> None:
        # The decisions are a narrow table beside the index, not a copy of it.
        with (
            progress.track_stage_workspace(
                "deduplicate", dedup_key, staging, dedup_groups_root
            ),
            closing(
                native_dedup.open_dedup(index_dir, staging / "dedup.sqlite")
            ) as connection,
        ):
            result["groups"], result["dropped"] = native_dedup.deduplicate(
                connection,
                staging / "groups.jsonl",
                diagnostics=root / "diagnostics" / f"{build_id}.json",
                progress=progress,
                policy=project.release.policy.duplicates,
                source_events=source_events,
                checkpoints=ViewCheckpoints(dedup_groups_root, dedup_key),
            )

    with (workspace / ".dedup.lock").open("a") as dedup_lock:
        fcntl.flock(dedup_lock, fcntl.LOCK_EX)
        stages.discard_incomplete("dedup")
        dedup_dir, dedup_result = run_stage(
            stages,
            "dedup",
            dedup_inputs,
            dedup_stage,
            on_reuse=lambda: progress.record("reused_dedup_stage"),
        )
        progress.observe_stage_output(
            "deduplicate",
            dedup_dir.name,
            sum(
                (dedup_dir / name).stat().st_size
                for name in ("dedup.sqlite", "groups.jsonl")
            ),
        )
        if dedup_groups_root.exists():
            shutil.rmtree(dedup_groups_root)
        stages.prune("dedup", keep=_PRUNE_ON_PUBLISH, protect=dedup_dir.name)
    progress.update(phase="writing_views")
    checkpoints = native_view_checkpoints(
        workspace,
        build_id,
        index_receipt_sha256=index_receipt_sha256,
        dedup_receipt_sha256=sha256_file(dedup_dir / "receipt.json"),
        generations=generations,
    )
    view_root = checkpoints.root
    with (
        _open_decided(index_dir, dedup_dir) as connection,
        (workspace / ".view-emission.lock").open("a") as lock_file,
        ExitStack() as selection,
    ):
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another Corpus build is writing views") from error
        staging = view_root / "build"
        if view_root.is_symlink() or staging.is_symlink():
            raise ValueError(f"symlinked build checkpoint: {view_root}")
        if canary is None:
            bind_normalized_view(view_root.resolve(), build_id)
        staging.mkdir(parents=True, exist_ok=True)
        # An interruption during publication may have left a manifest from the
        # previous attempt. The inventory must never include the manifest itself.
        (staging / "build.json").unlink(missing_ok=True)
        stage_declared_inputs(project, staging, admission_path, resume=True)
        span_projection = None
        span_receipt_generations = {}
        if identity.get("reference_normalized_spans"):
            if not span_partitions:
                raise ValueError("no normalized span partitions to reference")
            span_receipts = [path / "receipt.json" for path in prepared]
            span_receipt_generations = {
                path: range_file_identity(path) for path in span_receipts
            }
            progress.update(phase="sealing_spans")

            def seal(directory, _result):
                digest = seal_span_projection(
                    target,
                    span_partitions,
                    on_read=lambda size: progress.update(input_bytes=size),
                )
                write_json(directory / "sha256.json", digest)

            projection_store = StageStore(view_root / "span-projection")
            projection_store.discard_incomplete("seal")
            sealed, _ = run_stage(
                projection_store,
                "seal",
                {
                    "views": checkpoints.identity,
                    "catalog": canonical_digest(span_partitions),
                    "receipts": list(span_receipt_generations.values()),
                    "implementation": implementation_digests("normalized_spans.py"),
                },
                seal,
                on_reuse=lambda: progress.record("reused_span_projection"),
            )
            span_projection = {
                "storage": NORMALIZED_SPANS,
                "sha256": json.loads((sealed / "sha256.json").read_bytes()),
                "size": sum(part["size"] for part in span_partitions),
                "partitions": span_partitions,
            }
            progress.update(phase="writing_views")
        from klab.corpus.train_selection import (
            minimum_train_bytes,
            open_train_annotations,
            train_row_selection,
        )

        specs = [declaration_payload(transform)]
        minimum, row_selection = minimum_train_bytes(specs), train_row_selection(specs)
        annotations = None
        if minimum is not None or row_selection is not None:
            progress.update(
                phase="train_annotations" if row_selection else "text_statistics"
            )
            annotations = selection.enter_context(
                open_train_annotations(
                    work_root, prepared, connection, minimum, row_selection, progress
                )
            )
            progress.update(phase="writing_views")
        with progress.track_stage_workspace("views", checkpoints.identity, view_root):
            _emit_build(
                connection=connection,
                staging=staging,
                project=project,
                lock=lock,
                identity=identity,
                build_id=build_id,
                transform=transform,
                prepared=prepared,
                rights_files=rights_files,
                source_events=source_events,
                groups_path=dedup_dir / "groups.jsonl",
                group_count=dedup_result["groups"],
                dropped=dedup_result["dropped"],
                progress=progress,
                checkpoints=checkpoints,
                diagnostics=root / "diagnostics" / f"{build_id}.json",
                span_partitions=span_partitions,
                span_projection=span_projection,
                train_annotations=annotations,
                duplicate_evidence={
                    "index_key": index_key,
                    "groups_by_method": index_result["duplicate_evidence"],
                },
            )
        require_prepared_generation()
        if any(
            range_file_identity(path) != before
            for path, before in span_receipt_generations.items()
        ):
            raise ValueError("normalized span receipts changed during native build")
        progress.update(phase="publishing")
        publish_staging(staging, target)
        shutil.rmtree(view_root)
    with (workspace / ".dedup.lock").open("a") as dedup_lock:
        fcntl.flock(dedup_lock, fcntl.LOCK_EX)
        stages.prune("dedup", keep=keep_stages)
    with (workspace / ".index.lock").open("a") as index_lock:
        fcntl.flock(index_lock, fcntl.LOCK_EX)
        stages.prune("index", keep=keep_stages)
    if canary is None:
        clear_normalized_use(workspace.resolve(), build_id)
    progress.record("complete")
    return target
