"""Verified directed document links and bounded one-hop TRAIN neighborhoods."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import struct
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

from klab.canonical import canonical_digest
from klab.corpus.admission_inspection_cache import capture, unchanged
from klab.corpus.build_stages import implementation_digests
from klab.corpus.jsonl_records import records_from_path
from klab.corpus.mixture_pipeline import mixture_writer
from klab.corpus.release import open_release
from klab.corpus.split_runs import SplitStages, checkpoint_stamp
from klab.progress import progress_phase
from klab.result_publication import sync_directory
from klab.safe_io import rename_noreplace, symlinked
from klab.training.manifest import sha256_file
from klab.workdir import resolve_work_dir
from klab.workspace_preflight import check_storage, require_storage

_DIGEST = re.compile(r"[0-9a-f]{64}")
_EDGE_FIELDS = {"from_document_id", "to_document_id"}
_MAX_EDGE_BYTES = 4096
_MEMBERSHIP = struct.Struct("<32sB")
_INDEX = "links.sqlite"


def _document_id(value):
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError("link document IDs must be lowercase SHA-256 strings")
    return bytes.fromhex(value)


def _edges(path, digest):
    for record in records_from_path(
        path, max_record_bytes=_MAX_EDGE_BYTES, expected_sha256=digest
    ):
        row = record.value
        if not isinstance(row, dict) or row.keys() != _EDGE_FIELDS:
            raise ValueError(
                f"invalid link evidence at physical line {record.physical_line}"
            )
        yield _document_id(row["from_document_id"]), _document_id(row["to_document_id"])


def _read_graph(db, path, digest, inventory, strata):
    with progress_phase("corpus_link_evidence", unit="edges") as progress:
        input_rows = 0
        for input_rows, edge in enumerate(_edges(path, digest), 1):
            db.execute("INSERT OR IGNORE INTO edges VALUES (?, ?)", edge)
            if input_rows % 8192 == 0:
                progress.update(completed_work=input_rows, emit=False)
        progress.update(completed_work=input_rows)
    db.executescript("""
        INSERT OR IGNORE INTO endpoints(id) SELECT source FROM edges;
        INSERT OR IGNORE INTO endpoints(id) SELECT target FROM edges;
    """)
    with progress_phase("corpus_link_scope", unit="inventory rows") as progress:
        for offset, row in enumerate(inventory, 1):
            if row["split"] == "train" and row["stratum"] in strata:
                db.execute(
                    "UPDATE endpoints SET stratum=? WHERE id=?",
                    (
                        strata[row["stratum"]],
                        _document_id(row["document_id"]),
                    ),
                )
            if offset % 8192 == 0:
                progress.update(completed_work=offset, emit=False)
    if db.execute("SELECT 1 FROM endpoints WHERE stratum IS NULL LIMIT 1").fetchone():
        raise ValueError(
            "link endpoint falls outside the verified TRAIN family inventory"
        )
    return input_rows


def _assign_neighborhoods(db, max_neighborhoods):
    db.execute(
        """
        INSERT INTO anchors
        WITH degree AS (
            SELECT t.stratum, e.target, COUNT(*) AS inbound FROM edges e
            JOIN endpoints s ON s.id=e.source JOIN endpoints t ON t.id=e.target
            WHERE s.stratum=t.stratum AND e.source<>e.target
            GROUP BY t.stratum, e.target
        ), ranked AS (
            SELECT stratum, target, ROW_NUMBER() OVER (
                PARTITION BY stratum ORDER BY inbound DESC, target
            ) AS rank FROM degree
        ) SELECT target, stratum, rank FROM ranked WHERE rank<=?
    """,
        (max_neighborhoods,),
    )
    db.executescript("""
        INSERT INTO assignments
        SELECT e.source, MIN(a.rank) FROM edges e
        JOIN anchors a ON a.id=e.target JOIN endpoints s ON s.id=e.source
        WHERE s.stratum=a.stratum AND e.source<>e.target
        GROUP BY e.source;
        INSERT OR REPLACE INTO assignments SELECT id, rank FROM anchors;
    """)
    digest = hashlib.sha256()
    members = 0
    for document, rank in db.execute("SELECT id, rank FROM assignments ORDER BY id"):
        digest.update(_MEMBERSHIP.pack(document, rank))
        members += 1
    return digest.hexdigest(), members


def build_link_index(
    directory, partial, path, digest, inventory, *, strata, max_neighborhoods
):
    """Parse edge bytes once, then join only compact facts on bounded storage."""
    labels = sorted(strata)
    projected = 8 * 1024**2 + 4 * path.stat().st_size
    if symlinked(partial):
        raise ValueError("unsafe link index staging path")
    partial.unlink(missing_ok=True)
    require_storage([check_storage(partial.parent, projected_bytes=projected)])
    try:
        with closing(sqlite3.connect(partial)) as db:
            db.executescript(f"""
                PRAGMA journal_mode=OFF;
                PRAGMA page_size=4096;
                PRAGMA cache_size=-8192;
                PRAGMA temp_store=FILE;
                PRAGMA max_page_count={projected // 8192};
                CREATE TABLE edges(source BLOB, target BLOB,
                    PRIMARY KEY(source,target)) WITHOUT ROWID;
                CREATE TABLE endpoints(id BLOB PRIMARY KEY, stratum INTEGER) WITHOUT ROWID;
                CREATE TABLE anchors(id BLOB PRIMARY KEY, stratum INTEGER, rank INTEGER) WITHOUT ROWID;
                CREATE TABLE assignments(id BLOB PRIMARY KEY, rank INTEGER) WITHOUT ROWID;
            """)
            input_rows = _read_graph(
                db,
                path,
                digest,
                inventory,
                {name: ordinal for ordinal, name in enumerate(labels)},
            )
            with progress_phase(
                "corpus_link_neighborhoods", unit="documents"
            ) as progress:
                membership, members = _assign_neighborhoods(db, max_neighborhoods)
                progress.update(completed_work=members)
            anchors = {
                labels[ordinal]: count
                for ordinal, count in db.execute(
                    "SELECT stratum, COUNT(*) FROM anchors GROUP BY stratum"
                )
            }
            unique_edges = db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            db.commit()
        artifact = directory / _INDEX
        with partial.open("rb") as stream:
            # The sidecar stage publishes only a durable, closed database.
            os.fsync(stream.fileno())
        rename_noreplace(partial, artifact)
        sync_directory(directory)
        return {
            "edges": unique_edges,
            "input_rows": input_rows,
            "anchors_by_stratum": anchors,
            "assigned_documents": members,
            "membership_sha256": membership,
            "files": [
                {
                    "path": str(artifact),
                    "bytes": artifact.stat().st_size,
                    "sha256": sha256_file(artifact),
                }
            ],
        }
    finally:
        partial.unlink(missing_ok=True)


class LinkNeighborhoods:
    def __init__(self, db, result, identity, evidence):
        self.db = db
        self.max_neighborhoods = evidence.max_neighborhoods
        self.receipt = {
            "id": identity,
            "evidence_sha256": evidence.sha256,
            "max_neighborhoods": evidence.max_neighborhoods,
            **{key: value for key, value in result.items() if key != "files"},
        }

    def group(self, document_id):
        row = self.db.execute(
            "SELECT rank FROM assignments WHERE id=?", (_document_id(document_id),)
        ).fetchone()
        return row[0] if row else 0

    def outgoing(self, document, *, after=None, limit=20):
        if not 1 <= limit <= 1000:
            raise ValueError("link inspection limit must be between 1 and 1000")
        rows = self.db.execute(
            "SELECT target FROM edges WHERE source=? AND target>? ORDER BY target LIMIT ?",
            (
                _document_id(document),
                b"" if after is None else _document_id(after),
                limit + 1,
            ),
        ).fetchall()
        targets = [row[0].hex() for row in rows[:limit]]
        return {
            "document_id": document,
            "outgoing_document_ids": targets,
            "next_after": targets[-1] if len(rows) > limit else None,
        }


@contextmanager
def link_neighborhoods(spec, base, *, audit_directory=None):
    evidence = spec.sequence_order.links
    assert evidence is not None
    if not evidence.path.is_file():
        raise ValueError("link evidence must be an existing JSONL file")
    with ExitStack() as stack:
        releases = [
            stack.enter_context(open_release(path))
            for path in (
                spec.release_path,
                *(item.release_path for item in spec.additional_releases),
            )
        ]
        inventories = [(spec.family_inventory, base["family_inventory_sha256"])]
        inventories.extend(
            (
                release.member("family-inventory.jsonl"),
                release.manifest["files"]["family-inventory.jsonl"]["sha256"],
            )
            for release in releases[1:]
        )
        dependencies = capture([evidence.path, *(path for path, _ in inventories)])
        request = {
            "producer": implementation_digests(
                "jsonl_records.py", "link_neighborhoods.py"
            ),
            "scope": [
                [release.manifest["release_id"], digest]
                for release, (_, digest) in zip(releases, inventories, strict=True)
            ],
            "strata": sorted(spec.target_quotas),
            "evidence_sha256": evidence.sha256,
            "max_neighborhoods": evidence.max_neighborhoods,
        }
        identity = canonical_digest(request)
        work_root = resolve_work_dir()
        root = work_root / "cache" / "link-neighborhoods"

        def inventory():
            for path, digest in inventories:
                for record in records_from_path(path, expected_sha256=digest):
                    yield record.value

        def build(directory, partial):
            result = build_link_index(
                directory,
                partial,
                evidence.path,
                evidence.sha256,
                inventory(),
                strata=spec.target_quotas,
                max_neighborhoods=evidence.max_neighborhoods,
            )
            unchanged(dependencies)
            return result

        with mixture_writer(root / identity):
            if audit_directory is None:
                stages = SplitStages(
                    root,
                    {**request, "inputs": dependencies},
                    work_root=work_root,
                    producer_scoped=True,
                )
                result = stages.phase(
                    "train-links",
                    lambda directory: build(
                        directory, stages.root / ".building.sqlite"
                    ),
                )
            else:
                result = build(audit_directory, audit_directory / ".building.sqlite")
            path = Path(result["files"][0]["path"])
            generation = checkpoint_stamp(path)
            db = stack.enter_context(
                closing(
                    sqlite3.connect(
                        path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True
                    )
                )
            )
            db.execute("PRAGMA cache_size=-8192")
        yield LinkNeighborhoods(db, result, identity, evidence)
        if checkpoint_stamp(path) != generation:
            raise ValueError("link index changed during use")
