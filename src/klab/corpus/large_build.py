"""Bounded-memory, resumable Corpus Forge build for prospective LM source pools.

Selected source bytes remain in the verified acquisition snapshots. A prepared
file is immutable only after its receipt and all three output hashes validate;
incomplete staging directories are never inputs to a resumed build.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from array import array
from collections import Counter, defaultdict, deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    stage_declared_inputs,
    write_json,
    write_release_files,
)
from klab.corpus.progress import BuildProgress
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
from klab.training.manifest import sha256_file

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore, VerificationMode


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
) -> Path:
    from klab.corpus.pipeline import _records_for_file
    from klab.engram.packs import _rename_noreplace

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
        if counts["input_bytes"] != file["size"]:
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
            _rename_noreplace(staging, destination)
    _verify_prepared(destination, identity)
    return destination


def _index_prepared(
    database: Path, prepared: list[Path], project: Any, progress: BuildProgress
) -> sqlite3.Connection:
    from klab.corpus.pipeline import _origin_keys
    from klab.corpus.split_assignments import assign_split

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA cache_size=-16384")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE documents (
            document_id TEXT NOT NULL UNIQUE,
            payload BLOB NOT NULL,
            span BLOB NOT NULL,
            split TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_family TEXT NOT NULL,
            document_kind TEXT NOT NULL,
            source_rank INTEGER NOT NULL,
            representative_id TEXT,
            drop_reason TEXT
        );
        CREATE TABLE origins (
            method TEXT NOT NULL,
            origin TEXT NOT NULL,
            document_number INTEGER NOT NULL
        );
        """
    )
    from klab.corpus.split_assignments import split_policy

    policy = split_policy(project)
    from klab.corpus.dedup_policy import source_rank

    rank = {source.id: source_rank(source) for source in project.sources}
    declared_keys = set(project.release.policy.duplicates.keys)
    total = 0
    for shard in prepared:
        with (
            (shard / "docs.jsonl").open("rb") as docs,
            (shard / "spans.jsonl").open("rb") as spans,
        ):
            for raw_doc, raw_span in zip(docs, spans, strict=True):
                doc = json.loads(raw_doc)
                span = json.loads(raw_span)
                if doc["document_id"] != span["record_id"]:
                    raise ValueError("prepared document/span identity mismatch")
                split = assign_split(doc, policy)
                cursor = connection.execute(
                    "INSERT INTO documents (document_id,payload,span,split,source_id,source_family,document_kind,source_rank) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        doc["document_id"],
                        raw_doc.rstrip(b"\n"),
                        raw_span.rstrip(b"\n"),
                        split,
                        doc["source_id"],
                        doc["source_family"],
                        doc["document_kind"],
                        rank[doc["source_id"]],
                    ),
                )
                number = cursor.lastrowid
                origins = (
                    ("raw", doc["raw_content_sha256"]),
                    ("normalized", doc["content_sha256"]),
                    *(("canonical_origin", value) for value in _origin_keys(doc)),
                )
                connection.executemany(
                    "INSERT INTO origins(method,origin,document_number) VALUES (?,?,?)",
                    (
                        (method, key, number)
                        for method, key in origins
                        if method in declared_keys
                    ),
                )
                total += 1
                progress.update(documents=1)
                if total % 10_000 == 0:
                    connection.commit()
    connection.commit()
    connection.execute(
        "CREATE INDEX origins_by_key ON origins(method,origin,document_number)"
    )
    connection.execute("CREATE INDEX documents_source ON documents(source_id)")
    connection.commit()
    return connection


def _duplicate_diagnostics(
    groups_path: Path, rejected_path: Path | None, diagnostics: Path, error: str
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
        if rejected_path is not None:
            with rejected_path.open("rb") as rejected:
                for index, row in enumerate(rejected):
                    if index:
                        report.write(b",")
                    report.write(row.rstrip(b"\n"))
        report.write(b"]}\n")
        report.flush()
        os.fsync(report.fileno())


def _deduplicate(
    connection: sqlite3.Connection,
    groups_path: Path,
    *,
    policy: Any,
    diagnostics: Path,
    progress: BuildProgress,
    rejected_path: Path | None = None,
) -> tuple[int, int]:
    """Apply `release.policy.duplicates` with disk-backed joins.

    Every group is decided by `dedup_policy.resolve_group`, the same rules the
    in-memory builder uses; only the way groups are found and applied differs.
    """
    from klab.corpus.dedup_policy import (
        CLEANING_CONFLICT,
        PAGE_OVERLAP,
        Member,
        drop_reason,
        find_root,
        is_metadata_only,
        ranked_choice,
        resolve_group,
        rights_signature,
    )

    chain = policy.resolution == "chain_smallest_id"
    methods = [
        method
        for method in ("raw", "normalized", "canonical_origin")
        if method in policy.keys
    ]
    total = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    parents = array("I", range(total + 1))

    if chain:
        connection.execute("UPDATE documents SET representative_id=document_id")

    def members_query(with_payload: bool) -> str:
        return (
            "SELECT d.rowid,d.document_id,d.split,d.source_rank,d.document_kind,"
            + ("d.payload" if with_payload else "NULL")
            + " FROM origins o JOIN documents d ON d.rowid=o.document_number "
            "WHERE o.method=? AND o.origin=? ORDER BY d.document_id"
        )

    groups = 0
    with groups_path.open("wb") as output:
        for method in methods:
            keys = connection.execute(
                "SELECT origin FROM origins WHERE method=? GROUP BY origin "
                "HAVING COUNT(*)>1 ORDER BY origin",
                (method,),
            )
            needs_rights = method == "normalized" and policy.rights_conflict == "error"
            for (key,) in keys:
                members: list[Member] = []
                rows: list[tuple[int, str, str]] = []
                for (
                    number,
                    identifier,
                    split,
                    rank,
                    kind,
                    payload,
                ) in connection.execute(members_query(needs_rights), (method, key)):
                    members.append(
                        Member(
                            identifier,
                            split,
                            rank,
                            kind,
                            rights_signature(json.loads(payload))
                            if needs_rights
                            else None,
                        )
                    )
                    rows.append((number, identifier, split))
                group = resolve_group(method, members, policy)
                _output(
                    output,
                    {
                        "method": method,
                        "sha256": (
                            hashlib.sha256(key.encode()).hexdigest()
                            if method == "canonical_origin"
                            else key
                        ),
                        "origins": [row[1] for row in rows],
                        "splits": group.splits,
                        "representative": group.representative,
                    },
                )
                if group.error:
                    output.flush()
                    os.fsync(output.fileno())
                    _duplicate_diagnostics(
                        groups_path, rejected_path, diagnostics, group.error
                    )
                    raise ValueError(group.error)
                if chain:
                    train = [row[0] for row in rows if row[2] == "train"]
                    linked = [row[0] for row in rows]
                    if group.contaminated:
                        # Train copies follow the held-out document.
                        connection.executemany(
                            "UPDATE documents SET representative_id=? WHERE rowid=?",
                            ((group.representative, number) for number in train),
                        )
                        linked = [row[0] for row in rows if row[2] != "train"]
                    connection.executemany(
                        "UPDATE documents SET "
                        "representative_id=MIN(representative_id,?) WHERE rowid=?",
                        ((group.representative, number) for number in linked),
                    )
                else:
                    first = find_root(parents, rows[0][0])
                    for row in rows[1:]:
                        parents[find_root(parents, row[0])] = first
                groups += 1
                progress.update(documents=len(rows))
        output.flush()
        os.fsync(output.fileno())

    if chain:
        # Every link decreases lexicographically (train copies link to a held-out
        # document, which never links back), so compression terminates without an
        # in-memory union table.
        while True:
            changed = connection.execute(
                "UPDATE documents SET representative_id=(SELECT p.representative_id "
                "FROM documents p WHERE p.document_id=documents.representative_id) "
                "WHERE representative_id<>(SELECT p.representative_id FROM documents p "
                "WHERE p.document_id=documents.representative_id)"
            ).rowcount
            if not changed:
                break
        dropped = 0
        for (
            number,
            identifier,
            representative,
            payload,
            split,
            kept_split,
        ) in connection.execute(
            "SELECT d.rowid,d.document_id,d.representative_id,d.payload,d.split,"
            "p.split FROM documents d JOIN documents p "
            "ON p.document_id=d.representative_id ORDER BY d.rowid"
        ):
            reason = drop_reason(
                metadata_only=is_metadata_only(json.loads(payload)),
                split=split,
                is_representative=identifier == representative,
                representative_split=kept_split,
            )
            dropped += reason is not None
            connection.execute(
                "UPDATE documents SET drop_reason=? WHERE rowid=?", (reason, number)
            )
        connection.commit()
        return groups, dropped

    # A connected component can link distinct raw, normalized and page origins.
    # Whole components select one deterministic representative, preserving the
    # validation/test isolation gate even for transitive overlaps.
    winner: dict[int, tuple[Any, ...]] = {}
    heldout_mask: dict[int, int] = defaultdict(int)
    cleaning_mask: dict[int, int] = defaultdict(int)
    for number, split, payload in connection.execute(
        "SELECT rowid,split,payload FROM documents ORDER BY rowid"
    ):
        component = find_root(parents, number)
        heldout_mask[component] |= (
            1 if split == "validation" else 2 if split == "test" else 0
        )
        cleaning_mask[component] |= 2 if is_metadata_only(json.loads(payload)) else 1
    for masks, ok, message in (
        (heldout_mask, lambda mask: mask != 3, PAGE_OVERLAP),
        (cleaning_mask, lambda mask: mask != 3, CLEANING_CONFLICT),
    ):
        if not all(ok(mask) for mask in masks.values()):
            write_json(diagnostics, {"error": message})
            raise ValueError(message)
    for number, identifier, split, rank, kind in connection.execute(
        "SELECT rowid,document_id,split,source_rank,document_kind FROM documents "
        "ORDER BY rowid"
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
    for number, identifier, split, payload in connection.execute(
        "SELECT rowid,document_id,split,payload FROM documents ORDER BY rowid"
    ):
        selected, retained_split = winner[find_root(parents, number)][4:6]
        reason = drop_reason(
            metadata_only=is_metadata_only(json.loads(payload)),
            split=split,
            is_representative=selected == identifier,
            representative_split=retained_split,
        )
        dropped += reason is not None
        updates.append((selected, reason, number))
        if len(updates) >= 10_000:
            connection.executemany(
                "UPDATE documents SET representative_id=?,drop_reason=? WHERE rowid=?",
                updates,
            )
            connection.commit()
            updates.clear()
    if updates:
        connection.executemany(
            "UPDATE documents SET representative_id=?,drop_reason=? WHERE rowid=?",
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


def _write_rejected(target: Path, source_events: list[Path | dict[str, Any]]) -> None:
    with target.open("wb") as output:
        for event in source_events:
            if isinstance(event, Path):
                with (event / "rejected.jsonl").open("rb") as source:
                    for line in source:
                        output.write(line)
            else:
                _output(output, event)
        output.flush()
        os.fsync(output.fileno())


class _SQLBuildRows:
    """The ordinary builder's ordered inputs to the shared release writer."""

    def __init__(self, connection):
        self.connection = connection

    def spans(self):
        for (span,) in self.connection.execute(
            "SELECT span FROM documents ORDER BY rowid"
        ):
            yield span

    def documents(self):
        yield from self.connection.execute(
            "SELECT payload,split,representative_id,drop_reason FROM documents ORDER BY document_id"
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
    ordered_rows=None,
    descriptive_census=True,
) -> None:
    from klab.corpus.measurement import summarize_release

    ordered_rows = ordered_rows or _SQLBuildRows(connection)
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

        def writer(name: str) -> Any:
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
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
            writer(path)
        for split in SPLITS:
            writer(f"chat/{split}.jsonl")
            writer(f"chat/{split}.lineage.jsonl")

        # Source spans retain acquisition order; documents and views use ID order.
        for span in ordered_rows.spans():
            spans.write(span + b"\n")
        for raw_doc, split, representative, reason in ordered_rows.documents():
            doc = json.loads(raw_doc)
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
        for handle in (
            documents,
            spans,
            lineage,
            stage,
            *views.values(),
            *links.values(),
        ):
            handle.flush()
            os.fsync(handle.fileno())

    stage_digest.update(b"]]")
    stage_path = staging / "stages" / f"{stage_id}.jsonl"
    stages = [
        {
            "id": stage_id,
            "kind": stage_spec["kind"],
            "version": stage_spec["version"],
            "parameters_sha256": canonical_digest(stage_spec["parameters"]),
            "inputs": stage_spec["inputs"],
            "implementation_sha256": identity["implementation_sha256"],
            "output_sha256": sha256_file(stage_path),
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
            diagnostics=staging.parent / "diagnostics" / f"{build_id}.json",
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

    report = release_report(
        project,
        rights,
        tally,
        document_count=count,
        view_counts={
            "lm": {split: lm_counts[split] for split in SPLITS},
            "chat": dict.fromkeys(SPLITS, 0),
        },
        duplicate_groups=group_count,
        warnings=warnings,
    )
    progress.update(phase="measuring")
    report["measurement"] = (
        summarize_release(staging, release_spec=release_spec, tokenizer=None)
        if descriptive_census
        else {
            "status": "deferred",
            "reason": "Descriptive lineage census is outside the selected completed local preparation scope",
            "token_count_reason": "tokenizer_not_declared",
        }
    )
    write_release_files(
        staging,
        project,
        report,
        build_id=build_id,
        identity=identity,
        stages=stages,
        lock=lock,
    )


def _verify_admitted_inventory(
    connection: sqlite3.Connection, admissions: dict
) -> None:
    """Compare qualified and observed source identities without materialized sets."""
    from klab.corpus.admission_coverage import AdmissionCoverage

    if not admissions:
        return
    with AdmissionCoverage(connection) as coverage:
        for (source, path), admitted in admissions.items():
            for index, choice in admitted.get("decisions", {}).items():
                if choice["decision"] == "qualify":
                    coverage.expect(source, path, index)
        for source, raw in connection.execute(
            "SELECT source_id,payload FROM documents"
        ):
            doc = json.loads(raw)
            path = doc["source_location"].split("#", 1)[0]
            if "decisions" in admissions.get((source, path), {}):
                coverage.observe(source, path, doc["metadata"]["source_row_index"])
        coverage.verify()


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
) -> Path:
    """Build LM-only v2/v3 inputs using their declared admission/dedup policies."""
    record_admission = record_admission or {}
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

    progress.update(phase="indexing")
    index_root = root / "indices" / build_id
    index_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="index-", dir=index_root) as index_dir:
        connection = _index_prepared(
            Path(index_dir) / "documents.sqlite", prepared, project, progress
        )
        try:
            _verify_admitted_inventory(connection, record_admission)
            progress.update(phase="deduplicating")
            groups_path = Path(index_dir) / "groups.jsonl"
            rejected_path = Path(index_dir) / "rejected.jsonl"
            _write_rejected(rejected_path, source_events)
            group_count, dropped = _deduplicate(
                connection,
                groups_path,
                diagnostics=root / "diagnostics" / f"{build_id}.json",
                progress=progress,
                policy=project.release.policy.duplicates,
                rejected_path=rejected_path,
            )
            progress.update(phase="writing_views")
            with tempfile.TemporaryDirectory(prefix=".build-", dir=root) as temporary:
                staging = Path(temporary)
                stage_declared_inputs(project, staging, admission_path)
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
                    groups_path=groups_path,
                    group_count=group_count,
                    dropped=dropped,
                    progress=progress,
                )
                progress.update(phase="publishing")
                publish_staging(staging, target)
        finally:
            connection.close()
    progress.record("complete")
    return target
