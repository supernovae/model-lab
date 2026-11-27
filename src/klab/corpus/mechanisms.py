"""Pinned procedural worlds, resumable execution, and bounded replay."""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from contextlib import closing, contextmanager
from itertools import groupby
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from klab.canonical import canonical_json
from klab.config.models import StrictModel
from klab.corpus.build_stages import RECEIPT, StageStore, implementation_digests
from klab.corpus.jsonl_records import records_from_path
from klab.corpus.mechanism_rules import ObservedWorld, World, simulate_world
from klab.corpus.relation_sources import Identifier, Nonempty, PinnedRows
from klab.corpus.relations import SPLITS, publish_relation_stage
from klab.corpus.shards import ShardPlan, Single, declare_stage, run_sharded_files
from klab.corpus.stage_inspection import open_verified_stage
from klab.workspace_preflight import check_storage, require_storage

WORLD_STAGE = "procedural-worlds"
SIMULATION = declare_stage("procedural world simulation")
CATALOG = declare_stage(
    "procedural scope validation",
    Single("one writer validates split families across shards"),
)
RECORD_BYTES = 1024**2
ByteLimit = Annotated[int, Field(strict=True, ge=1024**2, le=2**40)]
Seed = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
WORLD_CATALOG = "worlds.sqlite"
_CATALOG_SCHEMA = """
CREATE TABLE families (id TEXT PRIMARY KEY, split TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE prose (digest BLOB PRIMARY KEY, split TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE worlds (
    ordinal INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, family TEXT NOT NULL,
    split TEXT NOT NULL, shard INTEGER NOT NULL, offset INTEGER NOT NULL, bytes INTEGER NOT NULL
);
"""


class WorldSource(PinnedRows):
    id: Identifier
    uri: Nonempty
    license: Nonempty
    origin: Literal["human", "procedural", "structured", "teacher", "unknown", "mixed"]


class SimulationSpec(StrictModel):
    sources: Annotated[tuple[WorldSource, ...], Field(min_length=1, max_length=4096)]
    seed: Seed
    max_catalog_bytes: ByteLimit
    max_shard_bytes: ByteLimit

    @model_validator(mode="after")
    def unique_sources(self):
        if len({source.id for source in self.sources}) != len(self.sources):
            raise ValueError("world source IDs must be unique")
        return self


def mechanism_store(root: Path) -> StageStore:
    return StageStore(root / "cache" / "corpus" / "mechanisms")


def build_procedural_shard(staging, path, pin, seed, budget):
    """Return small worker results and keep observations and offsets on disk."""
    source = WorldSource.model_validate({**pin, "path": path})
    count = offset = sidecar_bytes = steps = refused = 0
    with (
        (staging / "worlds.jsonl").open("wb") as worlds,
        (staging / "scope.jsonl").open("wb") as scopes,
    ):
        for row in source.rows(Path(".")):
            world = World.model_validate(row.value)
            observed = simulate_world(world, seed, source.id, row.physical_line)
            raw = canonical_json(observed.model_dump(mode="json")) + b"\n"
            scope = {
                "id": world.id,
                "family": world.family,
                "split": world.split,
                "offset": offset,
                "bytes": len(raw),
                "prose": hashlib.sha256(world.prose.encode()).hexdigest()
                if world.prose is not None
                else None,
            }
            scope_raw = canonical_json(scope) + b"\n"
            offset += len(raw)
            sidecar_bytes += len(scope_raw)
            if offset + sidecar_bytes > budget:
                raise ValueError(f"procedural output budget exceeded for {source.id}")
            worlds.write(raw)
            scopes.write(scope_raw)
            count += 1
            steps += world.steps
            refused += sum(
                bool(item.actual.violations) for item in observed.observations
            )
    return {
        "worlds": count,
        "steps": steps,
        "failed_derivations": refused,
        "payload_bytes": offset + sidecar_bytes,
    }


def configure_private_database(db, limit):
    """Give a staged SQLite file a fixed page budget and cache."""
    db.executescript(f"""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA page_size=4096;
        PRAGMA cache_size=-8192;
        PRAGMA temp_store=FILE;
        PRAGMA max_page_count={limit // 4096};
    """)


def _validate_scope(db, directory, receipt, shard):
    for record in records_from_path(
        directory / "scope.jsonl",
        max_record_bytes=2048,
        expected_sha256=receipt["files"]["scope.jsonl"]["sha256"],
    ):
        row = record.value
        for table, key, value in (
            ("families", "id", row["family"]),
            (
                "prose",
                "digest",
                bytes.fromhex(row["prose"]) if row["prose"] is not None else None,
            ),
        ):
            if (
                value is not None
                and db.execute(
                    f"""INSERT INTO {table} VALUES (?,?)
                ON CONFLICT({key}) DO UPDATE SET split=excluded.split
                WHERE split=excluded.split RETURNING split""",
                    (value, row["split"]),
                ).fetchone()
                is None
            ):
                reason = "semantic family" if table == "families" else "exact prose"
                raise ValueError(
                    f"{reason} spans multiple splits for world {row['id']}"
                )
        db.execute(
            "INSERT INTO worlds(id,family,split,shard,offset,bytes) VALUES (?,?,?,?,?,?)",
            (
                row["id"],
                row["family"],
                row["split"],
                shard,
                row["offset"],
                row["bytes"],
            ),
        )


def simulate_mechanisms(root, declaration, *, plan_only=False):
    spec = SimulationSpec.model_validate_json(declaration.read_bytes())
    store = mechanism_store(root)
    sources = sorted(spec.sources, key=lambda source: source.id)
    inputs = {
        **spec.model_dump(mode="json", exclude={"sources"}),
        "implementation": implementation_digests(
            "mechanisms.py",
            "mechanism_rules.py",
            "relation_sources.py",
            "jsonl_records.py",
        ),
        "sources": [
            source.model_dump(mode="json", exclude={"path"}) for source in sources
        ],
    }
    key = store.key(inputs)
    if plan_only:
        return {
            "key": key,
            "inputs": inputs,
            "candidate": store.directory(WORLD_STAGE, key).is_dir(),
            "stages": [SIMULATION, CATALOG],
        }
    base = declaration.resolve().parent
    shard_root = store.root / "simulation"

    def produce(staging, result):
        require_storage(
            [
                check_storage(
                    staging,
                    projected_bytes=spec.max_catalog_bytes
                    + len(sources) * spec.max_shard_bytes,
                    projected_inodes=5 * len(sources) + 16,
                )
            ]
        )
        tasks = [
            (str((base / source.path).resolve()), pin, spec.seed, spec.max_shard_bytes)
            for source, pin in zip(sources, inputs["sources"], strict=True)
        ]
        plan = ShardPlan(
            SIMULATION,
            tasks,
            sum(Path(task[0]).stat().st_size for task in tasks),
            inputs={"implementation": inputs["implementation"]},
            store=shard_root,
            worker_memory_bytes=256 * 1024**2,
            min_parallel_bytes=8 * 1024**2,
        )
        references = []
        with (
            closing(sqlite3.connect(staging / WORLD_CATALOG)) as db,
            db,
            closing(
                run_sharded_files(plan, build_procedural_shard, ordered=True)
            ) as shards,
        ):
            configure_private_database(db, spec.max_catalog_bytes)
            db.executescript(_CATALOG_SCHEMA)
            for ordinal, (directory, counts) in enumerate(shards):
                receipt = json.loads((directory / RECEIPT).read_bytes())
                _validate_scope(db, directory, receipt, ordinal)
                references.append(
                    {
                        "key": receipt["key"],
                        "payload": receipt["files"]["worlds.jsonl"],
                        "source": sources[ordinal].id,
                        **counts,
                    }
                )
            coverage = dict(
                db.execute("SELECT split,count(*) FROM worlds GROUP BY split")
            )
            if not coverage:
                raise ValueError("simulation requires at least one world")
            result.update(
                shards=references,
                coverage={split: coverage.get(split, 0) for split in SPLITS},
                failed_derivations=sum(
                    shard["failed_derivations"] for shard in references
                ),
            )

    try:
        return publish_relation_stage(store, WORLD_STAGE, inputs, produce)
    except sqlite3.Error as error:
        raise ValueError(f"cannot publish procedural scope: {error}") from error


@contextmanager
def open_world_shard(store, shard):
    shards = StageStore(store.root / "simulation" / "files")
    with open_verified_stage(shards, "shard", shard["key"]) as (directory, receipt, _):
        if (
            receipt["stage"] != "shard"
            or receipt["inputs"]["stage"] != SIMULATION
            or receipt["files"]["worlds.jsonl"] != shard["payload"]
        ):
            raise ValueError("procedural shard differs from the canonical reference")
        yield directory / "worlds.jsonl"


def observed_worlds(store, receipt):
    for shard in receipt["result"]["shards"]:
        with open_world_shard(store, shard) as path:
            for record in records_from_path(path, max_record_bytes=RECORD_BYTES):
                yield ObservedWorld.model_validate(record.value)


def audit_mechanisms(root, key, *, samples, seed):
    if not 1 <= samples <= 100 or not 0 <= seed < 2**63:
        raise ValueError("audit requires 1–100 samples and a nonnegative 63-bit seed")
    store = mechanism_store(root)
    with open_verified_stage(store, WORLD_STAGE, key) as (directory, receipt, _):
        total = sum(receipt["result"]["coverage"].values())
        ordinals = sorted(
            random.Random(seed).sample(range(1, total + 1), min(samples, total))
        )
        with closing(
            sqlite3.connect(
                f"{(directory / WORLD_CATALOG).as_uri()}?mode=ro&immutable=1", uri=True
            )
        ) as db:
            rows = [
                db.execute(
                    "SELECT shard,id,offset,bytes FROM worlds WHERE ordinal=?",
                    (ordinal,),
                ).fetchone()
                for ordinal in ordinals
            ]
        checks = []
        for shard, group in groupby(rows, key=lambda row: row[0]):
            with (
                open_world_shard(store, receipt["result"]["shards"][shard]) as path,
                path.open("rb") as stream,
            ):
                for _, name, offset, size in group:
                    if not 1 <= size <= RECORD_BYTES:
                        raise ValueError("procedural audit row exceeds the record cap")
                    stream.seek(offset)
                    observed = ObservedWorld.model_validate_json(stream.read(size))
                    expected = simulate_world(
                        observed.world,
                        receipt["inputs"]["seed"],
                        observed.source,
                        observed.line,
                    )
                    if observed.world.id != name or observed != expected:
                        raise ValueError(f"procedural replay differs for {name}")
                    checks.append(
                        {
                            "world": name,
                            "split": observed.world.split,
                            "steps": len(observed.observations),
                        }
                    )
        return {"key": key, "audit_seed": seed, "worlds": total, "replayed": checks}
