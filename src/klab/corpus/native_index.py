"""Compact native index creation and transactional assembly of prepared shards."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn
from urllib.parse import quote

from klab.canonical import canonical_digest
from klab.corpus.build_stages import StageStore, run_stage
from klab.corpus.duplicate_evidence import (
    is_metadata_only,
    origin_keys,
    rights_signature,
    source_rank,
)
from klab.corpus.progress import BuildProgress
from klab.corpus.shards import range_file_identity
from klab.corpus.split_assignments import assign_split, split_policy
from klab.result_publication import sync_directory

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore, VerificationMode


INDEX_IMPLEMENTATION_FILES = (
    "duplicate_evidence.py",
    "native_index.py",
    "split_assignments.py",
)


_DOCUMENT_INDEX_SCHEMA = """
CREATE TABLE documents (
    document_id TEXT NOT NULL{unique},
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
    document_id TEXT NOT NULL{unique},
    split TEXT NOT NULL,
    source_rank INTEGER NOT NULL,
    document_kind TEXT NOT NULL,
    metadata_only INTEGER NOT NULL,
    rights_signature BLOB NOT NULL,
    representative_id TEXT,
    drop_reason TEXT
);
"""
_INDEX_TABLES = (
    _DOCUMENT_INDEX_SCHEMA
    + DEDUP_SCHEMA
    + """
CREATE TABLE admitted_document_rows (
    file_number INTEGER NOT NULL,
    source_row_index INTEGER NOT NULL
);
"""
)
INDEX_SCHEMA = _INDEX_TABLES.format(unique=" UNIQUE")
_INDEX_GROUP_BYTES = 64 * 1024**2
_INDEX_RANGE_BYTES = 64 * 1024**2


def insert_indexed(
    connection: sqlite3.Connection,
    doc: dict[str, Any],
    references: tuple[int, int, int, int, int],
    split: str,
    rank: int,
    origins: Iterable[tuple[str, str]],
) -> int:
    """Add one prepared record to the index; return its document number."""
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


def _index_connection(directory: Path, *, durable: bool) -> sqlite3.Connection:
    connection = sqlite3.connect(directory / "index.sqlite")
    try:
        connection.execute("PRAGMA cache_size=-16384")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute(
            "PRAGMA synchronous=FULL" if durable else "PRAGMA synchronous=OFF"
        )
        connection.execute(
            "PRAGMA journal_mode=DELETE" if durable else "PRAGMA journal_mode=OFF"
        )
    except BaseException:
        connection.close()
        raise
    return connection


def create_index(
    directory: Path, *, defer_unique: bool = False, durable: bool = False
) -> sqlite3.Connection:
    """Create a private index of document references and duplicate evidence."""
    connection = _index_connection(directory, durable=durable)
    connection.executescript(
        _INDEX_TABLES.format(unique="") if defer_unique else INDEX_SCHEMA
    )
    return connection


def finish_index(
    connection: sqlite3.Connection, *, deferred_unique: bool = False
) -> None:
    if deferred_unique:
        connection.execute(
            "CREATE UNIQUE INDEX documents_by_id ON documents(document_id)"
        )
        connection.execute("CREATE UNIQUE INDEX dedup_by_id ON dedup(document_id)")
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
    *,
    start: tuple[int, int] = (0, 0),
    max_bytes: int | None = None,
) -> int:
    total = 0
    end_at = sum(start) + max_bytes if max_bytes is not None else None
    for shard_number, shard in group:
        with (
            (shard / "docs.jsonl").open("rb") as docs,
            (shard / "spans.jsonl").open("rb") as spans,
        ):
            document_offset, span_offset = start
            if document_offset or span_offset:
                docs.seek(document_offset)
                spans.seek(span_offset)
            for raw_doc, raw_span in zip(docs, spans, strict=True):
                doc = json.loads(raw_doc)
                span = json.loads(raw_span)
                if doc["document_id"] != span["record_id"]:
                    raise ValueError("prepared document/span identity mismatch")
                document_size, span_size = len(raw_doc), len(raw_span)
                origins = (
                    ("raw", doc["raw_content_sha256"]),
                    ("normalized", doc["content_sha256"]),
                    *(("canonical_origin", value) for value in origin_keys(doc)),
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
                if end_at is not None and document_offset + span_offset >= end_at:
                    break
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


_ASSEMBLY_TABLES = ("documents", "origins", "dedup", "admitted_document_rows")
_ASSEMBLY_INDEXES = (
    "documents_by_id",
    "dedup_by_id",
    "origins_by_key",
    "documents_source",
)


class _InvalidIndexAssembly(ValueError):
    """Retained rows or their assembly frontier failed verification."""


def _assembly_read_error(error: sqlite3.DatabaseError, message: str) -> NoReturn:
    # An interrupted initialization may also leave a database missing tables.
    # Keep resumable work intact on transient filesystem and lock failures.
    if error.sqlite_errorcode & 0xFF in {
        sqlite3.SQLITE_CORRUPT,
        sqlite3.SQLITE_NOTADB,
        sqlite3.SQLITE_ERROR,
    }:
        raise _InvalidIndexAssembly(message) from error
    raise error


def _assembly_counts(connection: sqlite3.Connection) -> tuple[int, ...]:
    try:
        return tuple(
            connection.execute(
                f"SELECT COALESCE(MAX(rowid),0) FROM main.{table}"
            ).fetchone()[0]
            for table in _ASSEMBLY_TABLES
        )
    except sqlite3.DatabaseError as error:
        _assembly_read_error(error, "could not read retained index row counts")


def _index_part_queries(document_offset: int):
    return (
        ("SELECT * FROM part.documents ORDER BY rowid", ()),
        (
            "SELECT method,origin,document_number+? FROM part.origins ORDER BY rowid",
            (document_offset,),
        ),
        (
            (
                "SELECT document_number+?, document_id, split, source_rank, document_kind, "
                "metadata_only, rights_signature, representative_id, drop_reason "
                "FROM part.dedup ORDER BY rowid"
            ),
            (document_offset,),
        ),
        ("SELECT * FROM part.admitted_document_rows ORDER BY rowid", ()),
    )


def _append_index_part(
    connection: sqlite3.Connection,
    part: Path,
    offsets: tuple[int, ...],
    *,
    verify_only: bool,
    verify_rows: bool = True,
) -> tuple[tuple[int, ...], tuple[int, int]]:
    """Append a verified part or compare its exact ordered rows with a committed prefix."""
    connection.execute(
        "ATTACH DATABASE ? AS part",
        (f"file:{quote(str(part / 'index.sqlite'))}?mode=ro",),
    )
    frontier = connection.execute(
        "SELECT document_offset+document_length, span_offset+span_length "
        "FROM part.documents ORDER BY rowid DESC LIMIT 1"
    ).fetchone() or (0, 0)
    ends = []
    for table, offset, (query, args) in zip(
        _ASSEMBLY_TABLES, offsets, _index_part_queries(offsets[0]), strict=True
    ):
        count = connection.execute(
            f"SELECT COALESCE(MAX(rowid),0) FROM part.{table}"
        ).fetchone()[0]
        end = offset + count
        if verify_only and verify_rows:
            conditions = []
            values = [offset]
            for _, name, *_ in connection.execute(f"PRAGMA part.table_info({table})"):
                name = '"' + name.replace('"', '""') + '"'
                if name == '"document_number"':
                    conditions.append(f"m.{name} IS p.{name}+?")
                    values.append(offsets[0])
                else:
                    conditions.append(f"m.{name} IS p.{name}")
            matched = connection.execute(
                f"SELECT COUNT(*) FROM part.{table} p CROSS JOIN main.{table} m "
                "WHERE m.rowid=p.rowid+? AND " + " AND ".join(conditions),
                values,
            ).fetchone()[0]
            if matched != count:
                raise _InvalidIndexAssembly(f"retained index rows changed in {table}")
        elif not verify_only:
            connection.execute(f"INSERT INTO main.{table} {query}", args)
        ends.append(end)
    if not verify_only:
        connection.execute("UPDATE index_assembly SET parts=parts+1")
        connection.commit()
    connection.execute("DETACH DATABASE part")
    return tuple(ends), frontier


def _open_index_assembly(directory: Path, identity: str):
    database = directory / "index.sqlite"
    journal = directory / "index.sqlite-journal"
    if database.is_symlink() or journal.is_symlink():
        raise ValueError(f"symlinked index assembly: {directory}")
    if not database.exists():
        connection = create_index(directory, defer_unique=True, durable=True)
        connection.execute(
            "CREATE TABLE index_assembly(identity TEXT NOT NULL, parts INTEGER NOT NULL)"
        )
        connection.execute("INSERT INTO index_assembly VALUES (?,0)", (identity,))
        connection.commit()
        sync_directory(directory)
        return connection, 0

    try:
        connection = _index_connection(directory, durable=True)
    except sqlite3.DatabaseError as error:
        _assembly_read_error(error, "could not open retained index assembly")
    try:
        ((scope, count),) = connection.execute(
            "SELECT identity,parts FROM index_assembly"
        )
        expected = set(_ASSEMBLY_TABLES) | {"index_assembly"}
        objects = connection.execute("SELECT type,name FROM sqlite_schema").fetchall()
        if (
            scope != identity
            or type(count) is not int
            or count < 0
            or {name for kind, name in objects if kind == "table"} != expected
            or any(
                kind != "table" and (kind != "index" or name not in _ASSEMBLY_INDEXES)
                for kind, name in objects
            )
        ):
            raise _InvalidIndexAssembly(
                "retained index assembly has a different scope or schema"
            )
    except sqlite3.DatabaseError as error:
        connection.close()
        _assembly_read_error(error, "could not read retained index assembly")
    except ValueError as error:
        connection.close()
        raise _InvalidIndexAssembly("could not read retained index assembly") from error
    return connection, count


def index_prepared(
    directory: Path,
    prepared: list[Path],
    project: Any,
    progress: BuildProgress,
    *,
    admission_files: tuple[tuple[str, str], ...],
    checkpoint_store: StageStore,
    index_key: str,
    generations: list[list[int]],
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> int:
    rank = {source.id: source_rank(source) for source in project.sources}
    declared_keys = set(project.release.policy.duplicates.keys)
    file_ids = {key: ordinal for ordinal, key in enumerate(admission_files)}
    policy = None

    def index_group(connection, group, start=(0, 0), max_bytes=None):
        nonlocal policy
        if policy is None:
            policy = split_policy(project)
        return _index_rows(
            connection,
            group,
            policy,
            rank,
            declared_keys,
            file_ids,
            progress,
            start=start,
            max_bytes=max_bytes,
        )

    def range_sizes(group):
        if len(group) == 1:
            number = group[0][0]
            sizes = (generations[2 * number][2], generations[2 * number + 1][2])
            if sum(sizes) > _INDEX_RANGE_BYTES:
                return sizes
        return None

    groups = _index_groups(prepared, generations)
    if len(groups) == 1 and range_sizes(groups[0]) is None:
        (directory / "index.sqlite").unlink(missing_ok=True)
        with closing(create_index(directory)) as connection:
            total = index_group(connection, groups[0])
            finish_index(connection)
        return total

    def parts():
        for group in groups:
            sizes = range_sizes(group)
            budget = _INDEX_RANGE_BYTES if sizes is not None else None
            start = (0, 0)
            inputs = {
                "index": index_key,
                "shards": [number for number, _ in group],
                "generations": [
                    generations[2 * number : 2 * number + 2] for number, _ in group
                ],
            }
            while True:

                def produce(staging, _result, group=group, start=start, budget=budget):
                    # The final index enforces uniqueness across all parts.
                    with closing(create_index(staging, defer_unique=True)) as part:
                        index_group(part, group, start, budget)

                part, _ = run_stage(
                    checkpoint_store,
                    "index-group",
                    inputs
                    if sizes is None
                    else {**inputs, "start": start, "budget": budget},
                    produce,
                    on_reuse=lambda: progress.record("reused_index_group"),
                )
                end = yield part
                if sizes is None or end == sizes:
                    break
                if not all(
                    begin < cursor < size
                    for begin, cursor, size in zip(start, end, sizes, strict=True)
                ):
                    raise ValueError("prepared document/span range offsets disagree")
                start = end

    def assemble() -> int:
        scope = canonical_digest(
            [index_key, generations, _INDEX_GROUP_BYTES, _INDEX_RANGE_BYTES]
        )
        connection, committed = _open_index_assembly(directory, scope)
        with closing(connection), closing(parts()) as source_parts:
            database = directory / "index.sqlite"
            proof_path = (
                proof_store.directory / f"index-assembly-{scope}.json"
                if proof_store is not None and verification_mode == "verified_reuse"
                else None
            )

            def binding() -> dict[str, object]:
                return {
                    "kind": "native-index-assembly",
                    "scope": scope,
                    "database": str(database.absolute()),
                    "generation": range_file_identity(database),
                }

            trusted = bool(
                committed
                and proof_path is not None
                and proof_store.trusted(database)
                and proof_store.load(proof_path) == binding()
            )

            def attest_prefix() -> None:
                nonlocal proof_path
                if proof_path is None:
                    return
                payload = binding()
                try:
                    saved = proof_store.trusted(database) and proof_store.publish(
                        proof_path,
                        payload,
                        before_replace=lambda: (
                            proof_store.trusted(database) and binding() == payload
                        ),
                    )
                except OSError:
                    saved = False
                if not saved:
                    progress.record("index_assembly_proof_unavailable")
                    proof_path = None

            offsets = (0, 0, 0, 0)
            part = next(source_parts, None)
            position = 0
            while part is not None:
                if position == committed and offsets != _assembly_counts(connection):
                    raise _InvalidIndexAssembly(
                        "retained index assembly has uncommitted rows"
                    )
                verify_only = position < committed
                try:
                    offsets, end = _append_index_part(
                        connection,
                        part,
                        offsets,
                        verify_only=verify_only,
                        verify_rows=not trusted,
                    )
                except sqlite3.DatabaseError as error:
                    if not verify_only:
                        raise
                    _assembly_read_error(
                        error, "retained index part failed verification"
                    )
                if verify_only:
                    progress.record("reused_index_assembly_part")
                else:
                    attest_prefix()
                position += 1
                try:
                    part = source_parts.send(end)
                except StopIteration:
                    break
            if position < committed or offsets != _assembly_counts(connection):
                raise _InvalidIndexAssembly(
                    "retained index assembly frontier differs from its parts"
                )

            indexes = {
                name
                for (name,) in connection.execute(
                    "SELECT name FROM sqlite_schema WHERE type='index'"
                )
            }
            if not (
                trusted and position == committed and indexes == set(_ASSEMBLY_INDEXES)
            ):
                # Keep the committed row merge if an incomplete SQLite sort stops.
                connection.execute("BEGIN IMMEDIATE")
                for name in _ASSEMBLY_INDEXES:
                    connection.execute(f"DROP INDEX IF EXISTS {name}")
                finish_index(connection, deferred_unique=True)
                attest_prefix()
            return offsets[0]

    try:
        return assemble()
    except _InvalidIndexAssembly:
        progress.record("discarded_damaged_index_assembly")
        for name in ("index.sqlite", "index.sqlite-journal"):
            (directory / name).unlink(missing_ok=True)
        return assemble()
