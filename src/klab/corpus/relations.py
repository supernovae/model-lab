"""Shared Corpus assertions, split-indexed edges, and source evidence."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.build_stages import (
    RECEIPT,
    StageStore,
    implementation_digests,
    run_stage,
)
from klab.corpus.jsonl_records import records_from_path
from klab.corpus.relation_sources import (
    MAX_ROW_BYTES,
    Extracted,
    RelationSpec,
    ScopedEntity,
    extract_relation_source,
)
from klab.corpus.release_members import shared_payload_lease
from klab.corpus.shards import ShardPlan, Single, declare_stage, run_sharded_files
from klab.corpus.stage_inspection import open_verified_stage
from klab.progress import progress_phase
from klab.workspace_preflight import check_storage, require_storage

EXTRACTION = declare_stage("relation source extraction")
INDEX = "relation-index"
declare_stage(
    "relation index reduction",
    Single(
        "a single writer builds the compact dictionaries and split-scoped adjacency"
    ),
)
SPLITS = ("train", "validation", "test")
CLAIM_CLASSES = ("explicit", "derived", "candidate")
_DATABASE = "relations.sqlite"

_SCHEMA = """
CREATE TABLE families (id TEXT PRIMARY KEY, split INTEGER NOT NULL) WITHOUT ROWID;
CREATE TABLE entities (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, family TEXT NOT NULL,
                       split INTEGER NOT NULL);
CREATE TABLE predicates (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE sources (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE assertions (
    id INTEGER PRIMARY KEY, digest BLOB NOT NULL UNIQUE, split INTEGER NOT NULL,
    subject INTEGER NOT NULL, predicate INTEGER NOT NULL, object INTEGER,
    literal TEXT, qualifiers TEXT NOT NULL, classes INTEGER NOT NULL,
    CHECK ((object IS NULL) != (literal IS NULL))
);
CREATE INDEX assertions_subject ON assertions(subject, id);
CREATE INDEX assertions_split ON assertions(split, id);
CREATE TABLE evidence (
    id INTEGER PRIMARY KEY, assertion INTEGER NOT NULL, source INTEGER NOT NULL,
    line INTEGER NOT NULL, location TEXT NOT NULL, claim_class TEXT NOT NULL,
    source_id TEXT, rank TEXT, refs TEXT NOT NULL
);
CREATE INDEX evidence_assertion ON evidence(assertion, id);
CREATE TABLE unresolved (id INTEGER PRIMARY KEY, source INTEGER NOT NULL,
                         line INTEGER NOT NULL, location TEXT NOT NULL, reason TEXT NOT NULL);
CREATE TABLE neighbors (subject INTEGER NOT NULL, slot INTEGER NOT NULL, assertion INTEGER NOT NULL,
                        PRIMARY KEY (subject, slot)) WITHOUT ROWID;
CREATE TABLE degrees (subject INTEGER PRIMARY KEY, degree INTEGER NOT NULL);
"""


def relation_store(root: Path) -> StageStore:
    return StageStore(root / "cache" / "corpus" / "relations")


def publish_relation_stage(store, stage, inputs, produce):
    """Reuse the common verification proof on warm opens."""
    key = store.key(inputs)
    destination = store.directory(stage, key)
    with shared_payload_lease(destination):
        if destination.exists():
            with open_verified_stage(store, stage, key) as (path, receipt, _):
                return _result(path, receipt, reused=True)
        # A killed writer leaves only this key's private checkpoint. StageStore
        # can adopt a fully receipted checkpoint or restart its unfinished work.
        retained = destination.with_name(f".incomplete-{key}")

        def restart(staging, result):
            for member in staging.iterdir():
                member.unlink()
            produce(staging, result)

        path, _ = run_stage(store, stage, inputs, restart, retained=retained)
        receipt = json.loads((path / RECEIPT).read_bytes())
        return _result(path, receipt, reused=False)


def _result(path, receipt, *, reused):
    return {
        "key": receipt["key"],
        "path": str(path),
        "files": receipt["files"],
        "reused": reused,
        **receipt["result"],
    }


def _inputs(spec):
    return {
        "implementation": implementation_digests(
            "relations.py", "relation_sources.py", "jsonl_records.py"
        ),
        "scope": spec.scope.model_dump(mode="json", exclude={"path"}),
        "sources": [
            source.model_dump(mode="json", exclude={"path"})
            for source in sorted(spec.sources, key=lambda item: item.id)
        ],
        "max_index_bytes": spec.max_index_bytes,
        "max_shard_output_bytes": spec.max_shard_output_bytes,
    }


def extract_relations(root: Path, declaration: Path, *, plan_only=False):
    spec = RelationSpec.model_validate_json(declaration.read_bytes())
    store = relation_store(root)
    inputs = _inputs(spec)
    if plan_only:
        key = store.key(inputs)
        return {
            "key": key,
            "path": str(store.directory(INDEX, key)),
            "inputs": inputs,
            "stages": [EXTRACTION, "relation index reduction"],
            "candidate": store.directory(INDEX, key).is_dir(),
        }
    base = declaration.resolve().parent

    def produce(staging, result):
        sources = sorted(spec.sources, key=lambda item: item.id)
        tasks = [
            (str((base / source.path).resolve()), metadata, spec.max_shard_output_bytes)
            for source, metadata in zip(sources, inputs["sources"], strict=True)
        ]
        # Each shard has a separately enforced output bound; index pages have their own cap.
        require_storage(
            [
                check_storage(
                    staging,
                    projected_bytes=spec.max_index_bytes
                    + len(tasks) * spec.max_shard_output_bytes,
                    projected_inodes=5 * len(tasks) + 16,
                )
            ]
        )
        plan = ShardPlan(
            EXTRACTION,
            tasks,
            sum(Path(task[0]).stat().st_size for task in tasks),
            inputs={
                "implementation": implementation_digests(
                    "relation_sources.py", "jsonl_records.py"
                )
            },
            store=store.root / "extraction",
            worker_memory_bytes=256 * 1024**2,
            min_parallel_bytes=8 * 1024**2,
        )
        shards = run_sharded_files(plan, extract_relation_source, ordered=True)
        with closing(sqlite3.connect(staging / _DATABASE)) as db, db, closing(shards):
            db.executescript(f"""
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA page_size=4096;
                PRAGMA cache_size=-8192;
                PRAGMA temp_store=FILE;
                PRAGMA max_page_count={spec.max_index_bytes // 4096};
            """)
            db.executescript(_SCHEMA)
            _load_scope(db, spec, base)
            result["sources"] = {}
            for source, (path, counts) in zip(sources, shards, strict=True):
                number = db.execute(
                    "INSERT INTO sources(name) VALUES (?)", (source.id,)
                ).lastrowid
                with progress_phase(
                    "corpus_relation_admission",
                    total_work=counts["claims"],
                    unit="assertions",
                    derived={"source": source.id},
                ) as progress:
                    admitted = _load_claims(db, number, path, progress)
                result["sources"][source.id] = {
                    **counts,
                    "admitted": admitted,
                    "unresolved": counts["unresolved"] + counts["claims"] - admitted,
                }
            with progress_phase(
                "corpus_relation_adjacency", total_work=1, unit="indexes"
            ) as progress:
                _build_adjacency(db)
                result["coverage"] = _coverage(db)
                progress.update(completed_work=1)
        result["index_bytes"] = (staging / _DATABASE).stat().st_size

    try:
        return publish_relation_stage(store, INDEX, inputs, produce)
    except sqlite3.Error as error:
        raise ValueError(f"cannot build relation index: {error}") from error


def _load_scope(db, spec, base):
    for row in spec.scope.rows(base):
        entity = ScopedEntity.model_validate(row.value)
        split = SPLITS.index(entity.split)
        family = db.execute(
            "INSERT INTO families VALUES (?, ?) ON CONFLICT(id) DO UPDATE "
            "SET split=excluded.split WHERE split=excluded.split RETURNING split",
            (entity.family, split),
        ).fetchone()
        if family is None:
            raise ValueError(f"semantic family {entity.family} spans multiple splits")
        db.execute(
            "INSERT INTO entities(name, family, split) VALUES (?, ?, ?)",
            (entity.id, entity.family, split),
        )


def _load_claims(db, source, directory, progress):
    def unresolved(line, location, reason):
        db.execute(
            "INSERT INTO unresolved(source, line, location, reason) VALUES (?, ?, ?, ?)",
            (source, line, location, reason),
        )

    @lru_cache(maxsize=4096)
    def entity(name):
        return db.execute(
            "SELECT id, split FROM entities WHERE name=?", (name,)
        ).fetchone()

    @lru_cache(maxsize=4096)
    def predicate(name):
        db.execute("INSERT OR IGNORE INTO predicates(name) VALUES (?)", (name,))
        return db.execute("SELECT id FROM predicates WHERE name=?", (name,)).fetchone()[
            0
        ]

    receipt = json.loads((directory / RECEIPT).read_bytes())
    for row in records_from_path(
        directory / "unresolved.jsonl",
        max_record_bytes=2 * MAX_ROW_BYTES,
        expected_sha256=receipt["files"]["unresolved.jsonl"]["sha256"],
    ):
        unresolved(**row.value)
    admitted = 0
    claims = records_from_path(
        directory / "claims.jsonl",
        max_record_bytes=2 * MAX_ROW_BYTES,
        expected_sha256=receipt["files"]["claims.jsonl"]["sha256"],
    )
    processed = 0
    for processed, row in enumerate(claims, 1):
        if processed % 8192 == 0:
            progress.update(completed_work=processed, emit=False)
        claim = Extracted.model_validate(row.value)
        assertion = claim.assertion
        binding = [entity(name) for name in assertion.scoped_ids()]
        if None in binding:
            unresolved(claim.line, claim.location, "entity_outside_scope")
            continue
        subject, split = binding[0]
        if any(item[1] != split for item in binding):
            unresolved(claim.line, claim.location, "cross_split_assertion")
            continue
        canonical = assertion.canonical()
        digest = bytes.fromhex(canonical_digest(canonical))
        target = binding[1][0] if canonical["value"]["kind"] == "entity" else None
        literal = (
            canonical_json(canonical["value"]).decode() if target is None else None
        )
        classification = 1 << CLAIM_CLASSES.index(claim.evidence.claim_class)
        (number,) = db.execute(
            """
            INSERT INTO assertions(digest, split, subject, predicate, object, literal, qualifiers, classes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(digest) DO UPDATE
            SET classes=assertions.classes | excluded.classes RETURNING id
            """,
            (
                digest,
                split,
                subject,
                predicate(assertion.predicate),
                target,
                literal,
                canonical_json(canonical["qualifiers"]).decode(),
                classification,
            ),
        ).fetchone()
        evidence = claim.evidence.model_dump(mode="json")
        db.execute(
            """
            INSERT INTO evidence(assertion, source, line, location, claim_class, source_id, rank, refs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                number,
                source,
                claim.line,
                claim.location,
                evidence["claim_class"],
                evidence["source_id"],
                evidence["rank"],
                canonical_json(evidence["references"]).decode(),
            ),
        )
        admitted += 1
    progress.update(completed_work=processed)
    return admitted


def _build_adjacency(db):
    # Scan the existing subject index rather than sorting the entire edge table.
    db.execute("""
        INSERT INTO neighbors
        SELECT subject, row_number() OVER (PARTITION BY subject ORDER BY id)-1, id
        FROM assertions WHERE object IS NOT NULL
        """)
    db.execute(
        "INSERT INTO degrees SELECT subject, count(*) FROM neighbors GROUP BY subject"
    )


def _coverage(db):
    return {
        split: {
            "families": db.execute(
                "SELECT count(*) FROM families WHERE split=?", (index,)
            ).fetchone()[0],
            "entities": db.execute(
                "SELECT count(*) FROM entities WHERE split=?", (index,)
            ).fetchone()[0],
            "assertions": assertions,
            "edges": edges,
            "qualified_assertions": qualified,
        }
        for index, split in enumerate(SPLITS)
        for assertions, edges, qualified in [
            db.execute(
                "SELECT count(*),coalesce(sum(object IS NOT NULL),0),coalesce(sum(qualifiers!='[]'),0) "
                "FROM assertions WHERE split=?",
                (index,),
            ).fetchone()
        ]
    }


@contextmanager
def open_relations(root, key):
    with open_verified_stage(relation_store(root), INDEX, key) as (
        path,
        receipt,
        digest,
    ):
        uri = (path / _DATABASE).as_uri()
        with closing(sqlite3.connect(f"{uri}?mode=ro&immutable=1", uri=True)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA cache_size=-8192")
            yield db, receipt, digest


ASSERTION_COLUMNS = """
    a.id AS cursor, lower(hex(a.digest)) AS assertion_id, s.name AS subject, p.name AS predicate,
    o.name AS object, a.literal, a.qualifiers, a.classes
"""
ASSERTION_JOINS = """
    JOIN entities s ON a.subject=s.id JOIN predicates p ON a.predicate=p.id
    LEFT JOIN entities o ON a.object=o.id
"""


def assertion_record(row):
    return {
        "subject": row["subject"],
        "predicate": row["predicate"],
        "value": (
            json.loads(row["literal"])
            if row["object"] is None
            else {"kind": "entity", "id": row["object"]}
        ),
        "qualifiers": json.loads(row["qualifiers"]),
    }


def claim_classes(row):
    return [
        label for bit, label in enumerate(CLAIM_CLASSES) if row["classes"] & (1 << bit)
    ]


def _evidence_record(row):
    record = dict(row)
    record["references"] = json.loads(record.pop("refs"))
    return record


def inspect_relations(
    root, key, *, entity=None, assertion=None, unresolved=False, after=0, limit=50
):
    if not 1 <= limit <= 1000 or after < 0:
        raise ValueError("relation limit is 1–1000 and cursor must be nonnegative")
    if sum((entity is not None, assertion is not None, unresolved)) > 1:
        raise ValueError("choose one relation inspection filter")
    with open_relations(root, key) as (db, receipt, _):
        result = {"key": key, **receipt["result"], "inputs": receipt["inputs"]}
        if entity is not None:
            rows = db.execute(
                f"""SELECT {ASSERTION_COLUMNS} FROM assertions a {ASSERTION_JOINS}
                WHERE a.subject=(SELECT id FROM entities WHERE name=?) AND a.id>?
                ORDER BY a.id LIMIT ?""",
                (entity, after, limit + 1),
            )
            records = (
                {
                    **assertion_record(row),
                    "claim_classes": claim_classes(row),
                    "id": row["assertion_id"],
                    "cursor": row["cursor"],
                }
                for row in rows
            )
        elif assertion is not None:
            digest = bytes.fromhex(assertion)
            if len(digest) != 32:
                raise ValueError("assertion inspection requires a full SHA-256 ID")
            records = map(
                _evidence_record,
                db.execute(
                    """SELECT e.id AS cursor,s.name AS source,e.line,e.location,e.claim_class,
                    e.source_id,e.rank,e.refs FROM evidence e JOIN sources s ON s.id=e.source
                    WHERE e.assertion=(SELECT id FROM assertions WHERE digest=?) AND e.id>?
                    ORDER BY e.id LIMIT ?""",
                    (digest, after, limit + 1),
                ),
            )
        elif unresolved:
            records = map(
                dict,
                db.execute(
                    """SELECT u.id AS cursor,s.name AS source,u.line,u.location,u.reason FROM unresolved u
                    JOIN sources s ON s.id=u.source WHERE u.id>? ORDER BY u.id LIMIT ?""",
                    (after, limit + 1),
                ),
            )
        else:
            return result
        page, total_bytes, next_cursor = [], 0, None
        for record in records:
            size = len(canonical_json(record))
            if len(page) >= limit or (page and total_bytes + size > 2 * MAX_ROW_BYTES):
                next_cursor = page[-1]["cursor"]
                break
            total_bytes += size
            page.append(record)
        return {**result, "records": page, "next": next_cursor}
