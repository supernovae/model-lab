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
import re
import shutil
import sqlite3
import struct
import tempfile
from array import array
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack, closing, nullcontext
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from klab.canonical import canonical_digest, canonical_json
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
from klab.corpus.progress import BuildProgress, memory_bytes
from klab.corpus.project import (
    ROW_SOURCE_KINDS,
    SPLITS,
    NormalizerOptions,
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
    FILE_ADMISSION_KINDS,
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


_PREPARED_FILES = ("docs.jsonl", "spans.jsonl", "rejected.jsonl")
_PROCESS_SHARD_MIN_BYTES = 32 * 1024**2
_PROCESS_SHARD_MAX_LINE_BYTES = 1024**2
_ADMITTED_RECORD_MAX_BYTES = 8 * 1024**2


def _output(handle: Any, value: Any) -> None:
    handle.write(canonical_json(value) + b"\n")


def _receipt_identity(
    build_id: str, source: Any, snapshot_sha: str, file: dict[str, Any]
) -> dict[str, Any]:
    return {
        "stage_version": "normalized-source-v2",
        "implementation": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "large_build.py",
                "pipeline.py",
                "jsonl_records.py",
                "structure_cleaning.py",
                "rights.py",
                "provenance.py",
            )
        },
        "source": source.model_dump(mode="json"),
        "source_id": source.id,
        "revision": source.revision,
        "snapshot_sha256": snapshot_sha,
        "path": file["path"],
        "file_sha256": file["sha256"],
        "file_size": file["size"],
    }


def _verify_prepared(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("prepared source shard is not a directory")
    receipt = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
    if receipt["identity"] != identity or set(receipt["files"]) != set(_PREPARED_FILES):
        raise ValueError("prepared source shard provenance mismatch")
    for name, expected in receipt["files"].items():
        item = path / name
        if (
            item.is_symlink()
            or not item.is_file()
            or item.stat().st_size != expected["size"]
            or sha256_file(item) != expected["sha256"]
        ):
            raise ValueError(f"prepared source shard changed: {name}")
    return receipt


class _SourceParseError(ValueError):
    """Input file rejected by the normalizer, not a broken receipt or build."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.documents = 0
        self.input_bytes = 0
        self.output_records = 0


class _ShardProgress:
    """Accumulate worker counters; the parent journals in input order."""

    def __init__(self) -> None:
        self.documents = 0
        self.input_bytes = 0
        self.output_records = 0
        self.reused = False

    def update(
        self, *, documents: int = 0, input_bytes: int = 0, output_records: int = 0
    ) -> None:
        self.documents += documents
        self.input_bytes += input_bytes
        self.output_records += output_records

    def record(self, event: str) -> None:
        if event == "reused_verified_shard":
            self.reused = True


def _prepare_shard(**kwargs: Any) -> tuple[Path, int, int, int, bool]:
    progress = _ShardProgress()
    try:
        shard = _prepare_file(**kwargs, progress=progress)
    except _SourceParseError as error:
        error.documents = progress.documents
        error.input_bytes = progress.input_bytes
        error.output_records = progress.output_records
        raise
    return (
        shard,
        progress.documents,
        progress.input_bytes,
        progress.output_records,
        progress.reused,
    )


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


def _prepare_file(
    *,
    build_id: str,
    prepared_root: Path,
    source: Any,
    file: dict[str, Any],
    snapshot_sha: str,
    source_path: Path,
    decision: Any,
    progress: BuildProgress,
    normalizer_options: NormalizerOptions,
    document_policy: Any,
    record_admission: dict[str, Any] | None = None,
    admission_sha256: str | None = None,
    row_limit: int | None = None,
) -> Path:
    from klab.corpus.pipeline import EmptyDocumentError, _records_for_file
    from klab.safe_io import rename_noreplace

    # Internal primitive: callers must first use verify_record_admission against
    # the pinned original rows. The digest binds that complete reviewed artifact,
    # not a newly inferred policy or an approval supplied by this helper.
    if (record_admission is None) != (admission_sha256 is None):
        raise ValueError(
            "admitted shard requires verified decisions and manifest digest"
        )
    if record_admission is not None:
        if not isinstance(admission_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", admission_sha256
        ):
            raise ValueError("invalid admission manifest digest")
        if "decisions" in record_admission:
            if source.kind != "huggingface_dataset" or not file["path"].endswith(
                ".jsonl"
            ):
                raise ValueError("streamed record admission requires an HF JSONL shard")
        elif source.kind not in FILE_ADMISSION_KINDS:
            raise ValueError("file admission requires a file-admitted source")
        if decision != record_admission["rights"]:
            raise ValueError("admitted shard file rights differ from verified policy")
    identity = _receipt_identity(build_id, source, snapshot_sha, file)
    identity["normalizer"] = normalizer_options.model_dump(mode="json")
    identity["document_policy"] = document_policy.model_dump(mode="json")
    identity["rights"] = decision.model_dump(mode="json")
    if row_limit is not None:
        identity["row_limit"] = row_limit
    if record_admission is not None:
        # Bind only this verified file's decisions, not the admission plan for
        # unrelated sources. Stream disk-backed decisions without materializing.
        digest = hashlib.sha256()
        for key in sorted(record_admission):
            value = record_admission[key]
            digest.update(canonical_json(key))
            if key == "decisions":
                for row_index, row in value.items():
                    digest.update(canonical_json([row_index, row]))
            else:
                if hasattr(value, "model_dump"):
                    value = value.model_dump(mode="json")
                digest.update(canonical_json(value))
        identity["file_admission_sha256"] = digest.hexdigest()
    parent = prepared_root / source.id
    destination = parent / canonical_digest(identity)
    if destination.exists():
        receipt = _verify_prepared(destination, identity)
        progress.update(
            documents=receipt["counts"]["documents"],
            input_bytes=receipt["counts"]["input_bytes"],
            output_records=receipt["counts"]["documents"],
        )
        progress.record("reused_verified_shard")
        return destination
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".incomplete-", dir=parent) as temporary:
        staging = Path(temporary)
        counts = {"documents": 0, "rejected": 0, "input_bytes": 0}
        with ExitStack() as stack:
            docs = stack.enter_context((staging / "docs.jsonl").open("wb"))
            spans = stack.enter_context((staging / "spans.jsonl").open("wb"))
            rejected = stack.enter_context((staging / "rejected.jsonl").open("wb"))
            if source.kind in ROW_SOURCE_KINDS and file["path"].endswith(".jsonl"):
                with source_path.open("rb") as stream:
                    index = 0
                    from klab.corpus.jsonl_records import iter_lf_lines

                    input_digest = hashlib.sha256()
                    for _, raw in iter_lf_lines(
                        stream,
                        max_record_bytes=(
                            _ADMITTED_RECORD_MAX_BYTES
                            if record_admission is not None
                            else None
                        ),
                    ):
                        input_digest.update(raw)
                        counts["input_bytes"] += len(raw)
                        if not raw.strip():
                            progress.update(input_bytes=len(raw))
                            continue
                        index += 1
                        if row_limit is not None and index > row_limit:
                            break
                        if index > source.acquisition.max_rows:
                            progress.update(input_bytes=len(raw))
                            continue
                        dropped: list[dict[str, Any]] = []
                        try:
                            rows = _records_for_file(
                                raw,
                                file["path"],
                                source,
                                snapshot_sha,
                                file_rights=decision,
                                record_admission=record_admission,
                                rejected_records=dropped,
                                full_file_sha256=file["sha256"],
                                first_row_index=index,
                                normalizer_options=normalizer_options,
                                document_policy=document_policy,
                            )
                        except EmptyDocumentError:
                            raise
                        except (ValueError, UnicodeError, KeyError, TypeError) as exc:
                            raise _SourceParseError(str(exc)) from exc
                        for doc, span in rows:
                            _output(docs, doc)
                            _output(spans, span)
                            counts["documents"] += 1
                        for row in dropped:
                            _output(rejected, row)
                            counts["rejected"] += 1
                        progress.update(
                            documents=1, input_bytes=len(raw), output_records=len(rows)
                        )
                    if (
                        record_admission is not None
                        and input_digest.hexdigest() != file["sha256"]
                    ):
                        raise ValueError(
                            "admitted source shard bytes differ from acquisition"
                        )
            else:
                if record_admission is not None:
                    with source_path.open("rb") as source_stream:
                        raw = source_stream.read(_ADMITTED_RECORD_MAX_BYTES + 1)
                    if len(raw) > _ADMITTED_RECORD_MAX_BYTES:
                        raise ValueError("admitted file-byte cap exceeded")
                    if hashlib.sha256(raw).hexdigest() != file["sha256"]:
                        raise ValueError(
                            "admitted source bytes differ from acquisition"
                        )
                else:
                    raw = source_path.read_bytes()
                dropped = []
                try:
                    rows = _records_for_file(
                        raw,
                        file["path"],
                        source,
                        snapshot_sha,
                        file_rights=decision,
                        record_admission=record_admission,
                        rejected_records=dropped,
                        normalizer_options=normalizer_options,
                        document_policy=document_policy,
                    )
                except EmptyDocumentError:
                    raise
                except (ValueError, UnicodeError, KeyError, TypeError) as exc:
                    raise _SourceParseError(str(exc)) from exc
                for doc, span in rows:
                    _output(docs, doc)
                    _output(spans, span)
                for row in dropped:
                    _output(rejected, row)
                counts = {
                    "documents": len(rows),
                    "rejected": len(dropped),
                    "input_bytes": len(raw),
                }
                progress.update(
                    documents=len(rows), input_bytes=len(raw), output_records=len(rows)
                )
            for output in (docs, spans, rejected):
                output.flush()
                os.fsync(output.fileno())
        if row_limit is None and counts["input_bytes"] != file["size"]:
            raise ValueError(
                "prepared source shard input size differs from acquisition"
            )
        receipt = {
            "schema_version": 1,
            "identity": identity,
            "provenance": {"build_id": build_id, "admission_sha256": admission_sha256},
            "counts": counts,
            "files": {
                name: {
                    "size": (staging / name).stat().st_size,
                    "sha256": sha256_file(staging / name),
                }
                for name in _PREPARED_FILES
            },
        }
        write_json(staging / "receipt.json", receipt)
        if destination.exists():
            _verify_prepared(destination, identity)
        else:
            rename_noreplace(staging, destination)
    _verify_prepared(destination, identity)
    return destination


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


DOCUMENT_INDEX_SCHEMA = """
CREATE TABLE documents (
    document_id TEXT NOT NULL UNIQUE,
    shard INTEGER NOT NULL,
    document_offset INTEGER NOT NULL,
    document_length INTEGER NOT NULL,
    span_offset INTEGER NOT NULL,
    span_length INTEGER NOT NULL,
    split TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_family TEXT NOT NULL,
    document_kind TEXT NOT NULL,
    source_rank INTEGER NOT NULL
);
CREATE TABLE origins (
    method TEXT NOT NULL,
    origin TEXT NOT NULL,
    document_number INTEGER NOT NULL
);
"""
DEDUP_SCHEMA = """
CREATE TABLE dedup (
    document_number INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL UNIQUE,
    split TEXT NOT NULL,
    source_rank INTEGER NOT NULL,
    document_kind TEXT NOT NULL,
    metadata_only INTEGER NOT NULL,
    rights_signature BLOB NOT NULL,
    representative_id TEXT,
    drop_reason TEXT
);
"""
INDEX_SCHEMA = (
    DOCUMENT_INDEX_SCHEMA
    + DEDUP_SCHEMA
    + """
CREATE TABLE admitted_document_rows (
    file_number INTEGER NOT NULL,
    source_row_index INTEGER NOT NULL
);
"""
)
_INDEX_GROUP_BYTES = 64 * 1024**2


def insert_indexed(
    connection: sqlite3.Connection,
    doc: dict[str, Any],
    references: tuple[int, int, int, int, int],
    split: str,
    rank: int,
    origins: Iterable[tuple[str, str]],
) -> int:
    """Add one prepared record to the index; return its document number."""
    from klab.corpus.dedup_policy import is_metadata_only, rights_signature

    number = connection.execute(
        "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            doc["document_id"],
            *references,
            split,
            doc["source_id"],
            doc["source_family"],
            doc["document_kind"],
            rank,
        ),
    ).lastrowid
    connection.execute(
        "INSERT INTO dedup (document_number,document_id,split,source_rank,"
        "document_kind,metadata_only,rights_signature) VALUES (?,?,?,?,?,?,?)",
        (
            number,
            doc["document_id"],
            split,
            rank,
            doc["document_kind"],
            int(is_metadata_only(doc)),
            rights_signature(doc),
        ),
    )
    connection.executemany(
        "INSERT INTO origins(method,origin,document_number) VALUES (?,?,?)",
        ((method, key, number) for method, key in origins),
    )
    return number


def create_index(directory: Path) -> sqlite3.Connection:
    """Create a private index of document references and duplicate evidence."""
    connection = sqlite3.connect(directory / "index.sqlite")
    connection.execute("PRAGMA cache_size=-16384")
    connection.execute("PRAGMA temp_store=FILE")
    # The files are private to a staging directory that is discarded on failure
    # and published only whole, so they need no journal.
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA journal_mode=OFF")
    connection.executescript(INDEX_SCHEMA)
    return connection


def finish_index(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE INDEX origins_by_key ON origins(method,origin,document_number)"
    )
    connection.execute("CREATE INDEX documents_source ON documents(source_id)")
    connection.commit()


def _index_rows(
    connection: sqlite3.Connection,
    group: list[tuple[int, Path]],
    policy: dict,
    rank: dict[str, int],
    declared_keys: set[str],
    file_ids: dict[tuple[str, str], int],
    progress: BuildProgress,
) -> int:
    from klab.corpus.pipeline import _origin_keys
    from klab.corpus.split_assignments import assign_split

    total = 0
    for shard_number, shard in group:
        with (
            (shard / "docs.jsonl").open("rb") as docs,
            (shard / "spans.jsonl").open("rb") as spans,
        ):
            document_offset = span_offset = 0
            for raw_doc, raw_span in zip(docs, spans, strict=True):
                doc = json.loads(raw_doc)
                span = json.loads(raw_span)
                if doc["document_id"] != span["record_id"]:
                    raise ValueError("prepared document/span identity mismatch")
                document_size, span_size = len(raw_doc), len(raw_span)
                origins = (
                    ("raw", doc["raw_content_sha256"]),
                    ("normalized", doc["content_sha256"]),
                    *(("canonical_origin", value) for value in _origin_keys(doc)),
                )
                insert_indexed(
                    connection,
                    doc,
                    (
                        shard_number,
                        document_offset,
                        document_size,
                        span_offset,
                        span_size,
                    ),
                    assign_split(doc, policy),
                    rank[doc["source_id"]],
                    [
                        (method, key)
                        for method, key in origins
                        if method in declared_keys
                    ],
                )
                document_offset += document_size
                span_offset += span_size
                if file_ids:
                    path = doc["source_location"].split("#", 1)[0]
                    number = file_ids.get((doc["source_id"], path))
                    if number is not None:
                        connection.execute(
                            "INSERT INTO admitted_document_rows VALUES (?,?)",
                            (number, doc["metadata"]["source_row_index"]),
                        )
                total += 1
                progress.update(documents=1)
                if total % 10_000 == 0:
                    connection.commit()
    connection.commit()
    return total


def _index_groups(
    prepared: list[Path], generations: list[list[int]]
) -> list[list[tuple[int, Path]]]:
    """Group small prepared files into sequential, approximately 64 MiB inputs."""
    groups = []
    group = []
    size = 0
    for number, shard in enumerate(prepared):
        cost = generations[2 * number][2] + generations[2 * number + 1][2]
        if group and size + cost > _INDEX_GROUP_BYTES:
            groups.append(group)
            group, size = [], 0
        group.append((number, shard))
        size += cost
    if group:
        groups.append(group)
    return groups


def _index_prepared(
    directory: Path,
    prepared: list[Path],
    project: Any,
    progress: BuildProgress,
    *,
    admission_files: tuple[tuple[str, str], ...],
    checkpoint_store: StageStore,
    index_key: str,
    generations: list[list[int]],
) -> int:
    from klab.corpus.dedup_policy import source_rank
    from klab.corpus.split_assignments import split_policy

    rank = {source.id: source_rank(source) for source in project.sources}
    declared_keys = set(project.release.policy.duplicates.keys)
    file_ids = {key: ordinal for ordinal, key in enumerate(admission_files)}
    policy = None

    def index_group(connection, group):
        nonlocal policy
        if policy is None:
            policy = split_policy(project)
        return _index_rows(
            connection, group, policy, rank, declared_keys, file_ids, progress
        )

    groups = _index_groups(prepared, generations)
    total = 0
    with closing(create_index(directory)) as connection:
        if len(groups) == 1:
            total = index_group(connection, groups[0])
        elif groups:
            for group in groups:

                def produce(staging, result, group=group):
                    with closing(create_index(staging)) as part:
                        result["documents"] = index_group(part, group)

                part, result = run_stage(
                    checkpoint_store,
                    "index-group",
                    {
                        "index": index_key,
                        "shards": [number for number, _ in group],
                        "generations": [
                            generations[2 * number : 2 * number + 2]
                            for number, _ in group
                        ],
                    },
                    produce,
                    on_reuse=lambda: progress.record("reused_index_group"),
                )
                connection.execute(
                    "ATTACH DATABASE ? AS part",
                    (f"file:{quote(str(part / 'index.sqlite'))}?mode=ro",),
                )
                connection.execute(
                    "INSERT INTO documents SELECT * FROM part.documents ORDER BY rowid"
                )
                connection.execute(
                    "INSERT INTO origins SELECT method,origin,document_number+? FROM part.origins",
                    (total,),
                )
                connection.execute(
                    "INSERT INTO dedup SELECT document_number+?, document_id, split, source_rank, "
                    "document_kind, metadata_only, rights_signature, representative_id, drop_reason "
                    "FROM part.dedup",
                    (total,),
                )
                connection.execute(
                    "INSERT INTO admitted_document_rows SELECT * FROM part.admitted_document_rows"
                )
                connection.commit()
                connection.execute("DETACH DATABASE part")
                total += result["documents"]
        finish_index(connection)
    return total


def open_dedup(index: Path, database: Path) -> sqlite3.Connection:
    """A new decisions database over a read-only index, ready for `_deduplicate`.

    `dedup`, the narrow table the passes read and write, lives in `database`;
    `documents` and `origins` are read from the attached index, which is never
    changed.
    """
    connection = sqlite3.connect(f"file:{quote(str(database))}", uri=True)
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute(
        "ATTACH DATABASE ? AS indexed",
        (f"file:{quote(str(index / 'index.sqlite'))}?mode=ro",),
    )
    connection.executescript(
        DEDUP_SCHEMA
        + "INSERT INTO main.dedup (document_number,document_id,split,source_rank,"
        "document_kind,metadata_only,rights_signature) SELECT document_number,"
        "document_id,split,source_rank,document_kind,metadata_only,rights_signature "
        "FROM indexed.dedup "
        "ORDER BY document_number;"
    )
    connection.commit()
    return connection


def _duplicate_diagnostics(
    groups_path: Path,
    source_events: Iterable[Path | dict[str, Any]],
    diagnostics: Path,
    error: str,
) -> None:
    """Preserve the complete observed groups without loading them."""
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics.open("wb") as report:
        report.write(b'{"duplicates":[')
        with groups_path.open("rb") as rows:
            for index, row in enumerate(rows):
                if index:
                    report.write(b",")
                report.write(row.rstrip(b"\n"))
        report.write(b'],"error":' + canonical_json(error) + b',"rejected":[')
        for index, row in enumerate(_rejected_rows(source_events)):
            if index:
                report.write(b",")
            report.write(row.rstrip(b"\n"))
        report.write(b"]}\n")
        report.flush()
        os.fsync(report.fileno())


_DEDUP_WINDOW_KEYS = 100_000
_DEDUP_WINDOW_BYTES = 64 * 1024**2
_DEDUP_LINK_COUNTS = struct.Struct("<II")


def _duplicate_key_bounds(connection, method, after):
    """Seek a batch boundary in the covering index, including unique origins."""
    predicate, values = _duplicate_key_start(method, after)
    row = connection.execute(
        f"SELECT origin FROM origins WHERE {predicate} ORDER BY origin LIMIT 1 OFFSET ?",
        (*values, _DEDUP_WINDOW_KEYS - 1),
    ).fetchone()
    return row[0] if row else None


def _duplicate_key_start(method, after):
    return (
        ("method=?", (method,))
        if after is None
        else ("method=? AND origin>?", (method, after))
    )


def _duplicate_keys(connection, method, after, end):
    predicate, values = _duplicate_key_start(method, after)
    if end is not None:
        predicate += " AND origin<=?"
        values = (*values, end)
    return connection.execute(
        f"SELECT origin FROM origins WHERE {predicate} "
        "GROUP BY origin HAVING COUNT(*)>1 ORDER BY origin",
        values,
    )


def _deduplicate(
    connection: sqlite3.Connection,
    groups_path: Path,
    *,
    policy: Any,
    diagnostics: Path,
    progress: BuildProgress,
    source_events: Iterable[Path | dict[str, Any]] = (),
    checkpoints: ViewCheckpoints | None = None,
) -> tuple[int, int]:
    """Apply `release.policy.duplicates` with disk-backed joins.

    Every group is decided by `dedup_policy.resolve_group`, the same rules the
    in-memory builder uses; only the way groups are found and applied differs.
    """
    from klab.corpus.dedup_policy import (
        CLEANING_CONFLICT,
        PAGE_OVERLAP,
        Member,
        describe_failure,
        drop_reason,
        find_root,
        ranked_choice,
        resolve_group,
    )

    chain = policy.resolution == "chain_smallest_id"
    methods = [
        method
        for method in ("raw", "normalized", "canonical_origin")
        if method in policy.keys
    ]
    parents = None
    if not chain:
        total = connection.execute("SELECT COUNT(*) FROM dedup").fetchone()[0]
        parents = array("I", range(total + 1))
    progress.update(phase="dedup_groups")

    if chain:
        connection.execute("UPDATE dedup SET representative_id=document_id")

    members_query = (
        "SELECT x.document_number,x.document_id,x.split,x.source_rank,"
        "x.document_kind,x.rights_signature FROM origins o "
        "JOIN dedup x ON x.document_number=o.document_number "
        "WHERE o.method=? AND o.origin=? ORDER BY x.document_id"
    )
    groups = 0

    def apply(representative, linked, train):
        nonlocal groups
        if chain:
            if train:
                connection.executemany(
                    "UPDATE dedup SET representative_id=? WHERE document_number=?",
                    ((representative, number) for number in train),
                )
            connection.executemany(
                "UPDATE dedup SET "
                "representative_id=MIN(representative_id,?) WHERE document_number=?",
                ((representative, number) for number in linked),
            )
        else:
            assert parents is not None
            members = iter(linked)
            first = find_root(parents, next(members))
            for number in members:
                parents[find_root(parents, number)] = first
        groups += 1
        progress.update(documents=len(linked) + len(train))

    report_path = groups_path
    if checkpoints is not None:
        report_path = checkpoints.root / "build" / "groups.jsonl"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if any(
            path.is_symlink()
            for path in (
                report_path.parent,
                report_path,
                report_path.with_name("links.bin"),
            )
        ):
            raise ValueError(f"symlinked dedup output: {report_path.parent}")
    with ExitStack() as stack:
        output = stack.enter_context(report_path.open("a+b" if checkpoints else "wb"))
        links = None
        first_method, after = 0, None
        if checkpoints is not None:
            links_stream = stack.enter_context(
                report_path.with_name("links.bin").open("a+b")
            )
            output_view = checkpoints.add("groups.jsonl", output)
            links = checkpoints.add("links.bin", links_stream)
            state = checkpoints.restore()
            if checkpoints.damaged:
                progress.record("discarded_damaged_dedup_checkpoint")
            if state:
                first_method, after = state["method"], state["after"]
                progress.record("reused_dedup_group_windows")
                output.seek(0)
                links_stream.seek(0)
                for rendered in output:
                    linked, train = _DEDUP_LINK_COUNTS.unpack(
                        links_stream.read(_DEDUP_LINK_COUNTS.size)
                    )
                    numbers = struct.unpack(
                        f"<{linked + train}I", links_stream.read(4 * (linked + train))
                    )
                    representative = (
                        json.loads(rendered)["representative"] if chain else None
                    )
                    apply(representative, numbers[:linked], numbers[linked:])
                if links_stream.read(1):
                    raise ValueError("dedup links and group report disagree")
                output.seek(0, os.SEEK_END)
                links_stream.seek(0, os.SEEK_END)
            target = output_view
        else:
            target = output

        def scan(method, after, end):
            needs_rights = method == "normalized" and policy.rights_conflict == "error"
            written = 0
            with closing(_duplicate_keys(connection, method, after, end)) as keys:
                for (key,) in keys:
                    rows = connection.execute(members_query, (method, key)).fetchall()
                    group = resolve_group(
                        method,
                        (
                            Member(
                                identifier,
                                split,
                                rank,
                                kind,
                                signature if needs_rights else None,
                            )
                            for _, identifier, split, rank, kind, signature in rows
                        ),
                        policy,
                    )
                    ids = [row[1] for row in rows]
                    rendered = (
                        canonical_json(
                            {
                                "method": method,
                                "sha256": hashlib.sha256(key.encode()).hexdigest()
                                if method == "canonical_origin"
                                else key,
                                "origins": ids,
                                "splits": group.splits,
                                "representative": group.representative,
                            }
                        )
                        + b"\n"
                    )
                    target.write(rendered)
                    if group.error:
                        output.flush()
                        os.fsync(output.fileno())
                        _duplicate_diagnostics(
                            report_path, source_events, diagnostics, group.error
                        )
                        raise ValueError(
                            describe_failure(
                                group.error, method, key, ids, group.splits
                            )
                        )
                    linked, train = [], []
                    for number, _, split, *_ in rows:
                        if chain and group.contaminated and split == "train":
                            train.append(number)
                        else:
                            linked.append(number)
                    if links is not None:
                        encoded_links = _DEDUP_LINK_COUNTS.pack(
                            len(linked), len(train)
                        ) + struct.pack(
                            f"<{len(linked) + len(train)}I", *linked, *train
                        )
                        links.write(encoded_links)
                        written += len(rendered) + len(encoded_links)
                    apply(group.representative, linked, train)
                    if written >= _DEDUP_WINDOW_BYTES:
                        return key
            return end

        for index in range(first_method, len(methods)):
            method = methods[index]
            if checkpoints is None:
                scan(method, None, None)
                continue
            while True:
                end = _duplicate_key_bounds(connection, method, after)
                after = scan(method, after, end)
                checkpoints.commit(
                    {
                        "method": index if after is not None else index + 1,
                        "after": after,
                    }
                )
                if after is None:
                    break
        output.flush()
        os.fsync(output.fileno())

    if checkpoints is not None:
        # The stage can seal this immutable link without copying the report.
        os.link(report_path, groups_path)

    if chain:
        connection.commit()
        progress.update(phase="dedup_compress")
        # Every link decreases lexicographically (train copies link to a held-out
        # document, which never links back), so compression terminates without an
        # in-memory union table. Only documents that point elsewhere are touched.
        while True:
            changed = connection.execute(
                "UPDATE dedup SET representative_id=(SELECT p.representative_id "
                "FROM dedup p WHERE p.document_id=dedup.representative_id) "
                "WHERE document_id<>representative_id AND representative_id<>"
                "(SELECT p.representative_id FROM dedup p "
                "WHERE p.document_id=dedup.representative_id)"
            ).rowcount
            progress.update(documents=changed)
            if not changed:
                break
        connection.commit()
        progress.update(phase="dedup_reasons")
        # Only a metadata-only document or one that is not its own representative
        # can have a drop reason; every other row stays NULL and is not written.
        dropped = 0
        last = 0
        while rows := connection.execute(
            "SELECT x.document_number,x.metadata_only,x.split,x.document_id,"
            "x.representative_id,p.split FROM dedup x JOIN dedup p "
            "ON p.document_id=x.representative_id WHERE x.document_number>? "
            "AND (x.metadata_only=1 OR x.document_id<>x.representative_id) "
            "ORDER BY x.document_number LIMIT 100000",
            (last,),
        ).fetchall():
            updates = []
            for number, metadata_only, split, identifier, representative, kept in rows:
                reason = drop_reason(
                    metadata_only=bool(metadata_only),
                    split=split,
                    is_representative=identifier == representative,
                    representative_split=kept,
                )
                if reason is not None:
                    updates.append((reason, number))
            connection.executemany(
                "UPDATE dedup SET drop_reason=? WHERE document_number=?", updates
            )
            dropped += len(updates)
            last = rows[-1][0]
            progress.update(documents=len(rows))
        connection.commit()
        return groups, dropped

    # A connected component can link distinct raw, normalized and page origins.
    # Whole components select one deterministic representative, preserving the
    # validation/test isolation gate even for transitive overlaps.
    assert parents is not None
    winner: dict[int, tuple[Any, ...]] = {}
    heldout_mask: dict[int, int] = defaultdict(int)
    cleaning_mask: dict[int, int] = defaultdict(int)
    progress.update(phase="dedup_components")
    for number, split, metadata_only in connection.execute(
        "SELECT document_number,split,metadata_only FROM dedup ORDER BY document_number"
    ):
        component = find_root(parents, number)
        heldout_mask[component] |= (
            1 if split == "validation" else 2 if split == "test" else 0
        )
        cleaning_mask[component] |= 2 if metadata_only else 1
    for masks, ok, message in (
        (heldout_mask, lambda mask: mask != 3, PAGE_OVERLAP),
        (cleaning_mask, lambda mask: mask != 3, CLEANING_CONFLICT),
    ):
        bad = next((root for root, mask in masks.items() if not ok(mask)), None)
        if bad is not None:
            members = [
                (identifier, split)
                for number, identifier, split in connection.execute(
                    "SELECT document_number,document_id,split FROM dedup "
                    "ORDER BY document_number"
                )
                if find_root(parents, number) == bad
            ]
            text = describe_failure(
                message,
                "component",
                members[0][0],
                [item[0] for item in members],
                {item[1] for item in members},
            )
            write_json(diagnostics, {"error": text})
            raise ValueError(text)
    for number, identifier, split, rank, kind in connection.execute(
        "SELECT document_number,document_id,split,source_rank,document_kind "
        "FROM dedup ORDER BY document_number"
    ):
        component = find_root(parents, number)
        preferred_split = (
            "validation"
            if heldout_mask[component] == 1
            else "test"
            if heldout_mask[component] == 2
            else None
        )
        choice = ranked_choice(split, preferred_split, rank, kind, identifier)
        if component not in winner or choice < winner[component][:4]:
            winner[component] = (*choice, identifier, split)
    dropped = 0
    updates = []
    for number, identifier, split, metadata_only in connection.execute(
        "SELECT document_number,document_id,split,metadata_only FROM dedup "
        "ORDER BY document_number"
    ):
        selected, retained_split = winner[find_root(parents, number)][4:6]
        reason = drop_reason(
            metadata_only=bool(metadata_only),
            split=split,
            is_representative=selected == identifier,
            representative_split=retained_split,
        )
        dropped += reason is not None
        updates.append((selected, reason, number))
        if len(updates) >= 10_000:
            connection.executemany(
                "UPDATE dedup SET representative_id=?,drop_reason=? "
                "WHERE document_number=?",
                updates,
            )
            connection.commit()
            updates.clear()
    if updates:
        connection.executemany(
            "UPDATE dedup SET representative_id=?,drop_reason=? "
            "WHERE document_number=?",
            updates,
        )
        connection.commit()
    return groups, dropped


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


def _rejected_rows(source_events: Iterable[Path | dict[str, Any]]) -> Iterator[bytes]:
    for event in source_events:
        if isinstance(event, Path):
            with (event / "rejected.jsonl").open("rb") as source:
                yield from source
        else:
            yield canonical_json(event) + b"\n"


def _write_rejected(target: Path, source_events: list[Path | dict[str, Any]]) -> None:
    with target.open("wb") as output:
        output.writelines(_rejected_rows(source_events))
        output.flush()
        os.fsync(output.fileno())


_DOCUMENT_READ_BYTES = 8 * 1024**2
_DOCUMENT_READ_ROWS = 2048
_DOCUMENT_OPEN_FILES = 64
_MAX_DOCUMENT_CACHE_BYTES = 1024**3


class _SQLBuildRows:
    """The ordinary builder's ordered inputs to the shared release writer."""

    def __init__(self, connection, prepared):
        self.connection = connection
        self.prepared = prepared

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
        references = self.connection.execute(
            "SELECT d.shard,d.document_offset,d.document_length,d.split,"
            "x.representative_id,x.drop_reason,d.document_id "
            "FROM documents d CROSS JOIN dedup x ON x.document_number=d.rowid "
            "WHERE d.document_id>? ORDER BY d.document_id",
            (after or "",),
        )
        streams = OrderedDict()
        try:
            if not after and self._warm_document_cache():
                for shard, offset, length, *metadata in references:
                    stream = self._document_stream(streams, shard)
                    raw = read_file_range(stream, offset, length)
                    yield raw.removesuffix(b"\n"), *metadata, metadata[-1]
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
        for raw, (_, _, _, *metadata) in zip(payloads, batch, strict=True):
            yield raw.removesuffix(b"\n"), *metadata, metadata[-1]

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
) -> None:
    from klab.corpus.measurement import summarize_release

    ordered_rows = ordered_rows or _SQLBuildRows(connection, prepared)
    sources = source_declarations(project, lock)
    rights = rights_report(project, sources, rights_files)
    source_origins = {source.id: source.origin for source in project.sources}
    release_spec = declaration_payload(project.release)
    stage_spec = declaration_payload(transform)
    stage_id = stage_spec["id"]
    selected_sources = set(stage_spec["inputs"])
    tally = Tally()
    lm_counts: Counter[str] = Counter()
    count = 0
    stage_count = 0
    stage_digest = hashlib.sha256()
    stage_digest.update(b"[" + canonical_json(stage_id) + b",[")

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

        documents = writer("documents.jsonl")
        spans = writer("spans.jsonl")
        lineage = writer("lineage.jsonl")
        stage = writer(f"stages/{stage_id}.jsonl")
        views = {split: writer(f"lm/{split}.jsonl") for split in SPLITS}
        links = {split: writer(f"lm/{split}.lineage.jsonl") for split in SPLITS}
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
                }
            )
            pending_bytes = pending_rows = 0

        # Source spans retain acquisition order; documents and views use ID order.
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
        ) in stack.enter_context(closing(document_rows)):
            doc = json.loads(raw_doc)
            if doc["document_id"] != document_id:
                raise ValueError("prepared document differs from the indexed identity")
            doc["split"] = split
            doc["representative_id"] = representative
            doc["drop_reason"] = reason
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
                progress.update(output_records=1)
            boundary(len(raw_doc))
        boundary(0, finished=True)

        view_digests = checkpoints.file_digests()

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


def _stage_inputs(
    project: Any,
    prepared: list[Path],
    *,
    admission_files: tuple[tuple[str, str], ...] = (),
) -> dict[str, dict[str, Any]]:
    """Exactly what each checkpointed stage depends on.

    The index follows the receipts of the prepared shards (which bind the
    source bytes, rights, normalizer and document policy), the split
    declaration, duplicate keys, source ranks and the ordered files with
    row-level admission evidence. Deduplication adds the full duplicate policy
    to the index it reads.
    """
    from klab.corpus.dedup_policy import source_rank

    index = {
        "shards": [
            {
                "source": shard.parent.name,
                "receipt_sha256": sha256_file(shard / "receipt.json"),
            }
            for shard in prepared
        ],
        "split": project.splits.model_dump(mode="json"),
        "duplicate_keys": sorted(project.release.policy.duplicates.keys),
        "source_rank": {
            source.id: source_rank(source)
            for source in sorted(project.sources, key=lambda item: item.id)
        },
        "implementation": implementation_digests(
            "large_build.py", "pipeline.py", "split_assignments.py", "dedup_policy.py"
        ),
    }
    if admission_files:
        index["row_admission_files"] = admission_files
    return {
        "index": {"stage": "index", **index},
        "dedup": {
            "stage": "dedup",
            "duplicates": project.release.policy.duplicates.model_dump(mode="json"),
            "implementation": implementation_digests(
                "large_build.py", "dedup_policy.py", "view_checkpoints.py"
            ),
        },
    }


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
    prepared_root = root / "prepared" / "normalized-source-v2"
    prepared_root.mkdir(parents=True, exist_ok=True)
    sources = sorted(project.sources, key=lambda source: source.id)
    transform = project.transforms[0]
    rights_files: list[dict[str, Any]] = []
    source_events: list[Path | dict[str, Any]] = []
    prepared: list[Path] = []
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
    eligible_jsonl = {
        path
        for source, file, path, lock_row, nested_metadata in inputs
        if source.kind in ROW_SOURCE_KINDS
        and file["path"].endswith(".jsonl")
        and file["size"] >= _PROCESS_SHARD_MIN_BYTES
        and file["path"] not in nested_metadata
        # Cached sources reach exact identity/rights checking in _prepare_file.
        # Do not pre-scan their bodies merely to choose parallel workers.
        and not (prepared_root / source.id).exists()
    }
    if record_admission:
        # Operation-local SQLite decisions must stay in their owning process.
        eligible_jsonl.clear()
    workers = _process_shard_workers(len(eligible_jsonl))
    if workers > 1:
        eligible_jsonl = {path for path in eligible_jsonl if _bounded_jsonl_shard(path)}
        workers = _process_shard_workers(len(eligible_jsonl))
    pool = (
        ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
    )
    sampled: dict[str, tuple[int, int]] = {}
    pending: deque[
        tuple[Future[tuple[Path, int, int, int, bool]], Any, dict[str, Any]]
    ] = deque()

    def finish_one() -> None:
        future, source, file = pending.popleft()
        try:
            shard, documents, input_bytes, output_records, reused = future.result()
        except _SourceParseError as exc:
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
            documents=documents, input_bytes=input_bytes, output_records=output_records
        )
        if reused:
            progress.record("reused_verified_shard")
        prepared.append(shard)
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
            decision = (
                resolve_file_rights(
                    source.rights,
                    name,
                    _read_rights_header(path, name),
                    nested_metadata=nested_metadata,
                    prospective_private_research=(
                        project.release.policy.rights.training_use
                        == "allowed_unless_explicitly_prohibited"
                    ),
                )
                if source.rights
                else None
            )
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
                "build_id": build_id,
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
                shard = _prepare_file(**kwargs, progress=progress)
            except _SourceParseError as exc:
                source_events.append(
                    {"source_id": source.id, "path": name, "reason": str(exc)}
                )
                progress.update(input_bytes=file["size"])
                progress.record("source_rejected")
                continue
            prepared.append(shard)
            source_events.append(shard)
        while pending:
            finish_one()

    stages = StageStore(root / "stages")
    admission_files = tuple(
        sorted(
            key for key, admitted in record_admission.items() if "decisions" in admitted
        )
    )
    stage_inputs = _stage_inputs(project, prepared, admission_files=admission_files)
    prepared_files = [
        shard / name for shard in prepared for name in ("docs.jsonl", "spans.jsonl")
    ]
    generations = [range_file_identity(path) for path in prepared_files]
    index_key = stages.key(stage_inputs["index"])
    groups_root = stages.root / f".index-{index_key}.groups"
    group_store = StageStore(groups_root)
    progress.update(phase="indexing")

    def require_prepared_generation() -> None:
        if [range_file_identity(path) for path in prepared_files] != generations:
            raise ValueError("prepared payloads changed during native build")

    def index_stage(staging: Path, result: dict[str, Any]) -> None:
        result["documents"] = _index_prepared(
            staging,
            prepared,
            project,
            progress,
            admission_files=admission_files,
            checkpoint_store=group_store,
            index_key=index_key,
            generations=generations,
        )
        require_prepared_generation()

    with (workspace / ".index.lock").open("a") as index_lock:
        fcntl.flock(index_lock, fcntl.LOCK_EX)
        stages.discard_incomplete("index")
        group_store.discard_incomplete("index-group")
        index_dir, _ = run_stage(
            stages,
            "index",
            stage_inputs["index"],
            index_stage,
            on_reuse=lambda: progress.record("reused_index_stage"),
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
    dedup_inputs = {
        **stage_inputs["dedup"],
        "index": sha256_file(index_dir / "receipt.json"),
    }
    dedup_groups_root = stages.root / f".dedup-{stages.key(dedup_inputs)}.groups"
    if dedup_groups_root.is_symlink():
        raise ValueError(f"symlinked dedup checkpoint: {dedup_groups_root}")

    def dedup_stage(staging: Path, result: dict[str, Any]) -> None:
        # The decisions are a narrow table beside the index, not a copy of it.
        connection = open_dedup(index_dir, staging / "dedup.sqlite")
        try:
            result["groups"], result["dropped"] = _deduplicate(
                connection,
                staging / "groups.jsonl",
                diagnostics=root / "diagnostics" / f"{build_id}.json",
                progress=progress,
                policy=project.release.policy.duplicates,
                source_events=source_events,
                checkpoints=ViewCheckpoints(
                    dedup_groups_root, stages.key(dedup_inputs)
                ),
            )
        finally:
            connection.close()

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
        if dedup_groups_root.exists():
            shutil.rmtree(dedup_groups_root)
        stages.prune("dedup", keep=_PRUNE_ON_PUBLISH, protect=dedup_dir.name)
    progress.update(phase="writing_views")
    view_key = canonical_digest(
        {
            "build": build_id,
            "index_receipt": sha256_file(index_dir / "receipt.json"),
            "dedup_receipt": sha256_file(dedup_dir / "receipt.json"),
            "prepared_generations": generations,
            "implementation": implementation_digests(
                "view_checkpoints.py", "build_outputs.py"
            ),
        }
    )
    view_root = workspace / f".views-{view_key}"
    with (
        _open_decided(index_dir, dedup_dir) as connection,
        (workspace / ".view-emission.lock").open("a") as lock_file,
    ):
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another Corpus build is writing views") from error
        staging = view_root / "build"
        if view_root.is_symlink() or staging.is_symlink():
            raise ValueError(f"symlinked build checkpoint: {view_root}")
        staging.mkdir(parents=True, exist_ok=True)
        # An interruption during publication may have left a manifest from the
        # previous attempt. The inventory must never include the manifest itself.
        (staging / "build.json").unlink(missing_ok=True)
        stage_declared_inputs(project, staging, admission_path, resume=True)
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
            checkpoints=ViewCheckpoints(view_root, view_key),
            diagnostics=root / "diagnostics" / f"{build_id}.json",
        )
        require_prepared_generation()
        progress.update(phase="publishing")
        publish_staging(staging, target)
        shutil.rmtree(view_root)
    with (workspace / ".dedup.lock").open("a") as dedup_lock:
        fcntl.flock(dedup_lock, fcntl.LOCK_EX)
        stages.prune("dedup", keep=keep_stages)
    with (workspace / ".index.lock").open("a") as index_lock:
        fcntl.flock(index_lock, fcntl.LOCK_EX)
        stages.prune("index", keep=keep_stages)
    progress.record("complete")
    return target
