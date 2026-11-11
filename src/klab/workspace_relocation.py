"""Build a consolidated projection of quiescent, standalone historical stores.

Run trees and original databases remain the caller's responsibility. This narrow
helper deliberately rejects controller queues: it must not invent ownership or
rewrite worker receipts. Preserve the originals as provenance when moving trees.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from klab.training.metrics import ExperimentStore, _strict_json_loads


def consolidate_standalone_stores(sources: list[Path], destination: Path) -> None:
    """Create one projection, preserving identities and rebasing current pointers.

    Sources must be stopped and exclusively owned by the caller for the entire
    operation. No source database or immutable record is modified. Destination
    must not exist; failed output is retained for inspection, never reused.
    """
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not sources:
        raise ValueError("at least one source store is required")
    snapshots: list[tuple[Path, dict[str, list[tuple]], list[str]]] = []
    known_runs: set[str] = set()
    known_origins: set[str] = set()
    tables = (
        "runs",
        "metrics",
        "events",
        "stage_history",
        "manifests",
        "checkpoints",
        "calibration",
    )
    for source in sources:
        source = source.resolve()
        with closing(
            sqlite3.connect(
                (source / "experiments.sqlite3").as_uri() + "?mode=ro", uri=True
            )
        ) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError(f"damaged store: {source}")
            for table in ("workers", "experiments", "attempts", "ingested_records"):
                if connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]:
                    raise ValueError(
                        f"requires standalone, unreplicated stores: {source}"
                    )
            snapshot = {
                table: connection.execute(f"SELECT * FROM {table}").fetchall()
                for table in tables
            }
            origin = connection.execute(
                "SELECT origin_id FROM store_metadata"
            ).fetchone()[0]
            if origin in known_origins:
                raise ValueError("duplicate origin")
            known_origins.add(origin)
            for row in snapshot["runs"]:
                if row[2] not in {"completed", "interrupted", "failed", "cancelled"}:
                    raise ValueError(f"run is not terminal: {row[0]}")
                if row[0] in known_runs:
                    raise ValueError(f"duplicate run ID: {row[0]}")
                known_runs.add(row[0])
                run = source / row[0]
                if (
                    run.parent != source
                    or run.resolve().parent != source
                    or run.is_symlink()
                    or not run.is_dir()
                ):
                    raise ValueError(f"missing or unsafe run tree: {run}")
                if row[8] is not None:
                    Path(row[8]).resolve().relative_to(run.resolve())
            records = [
                row[0]
                for row in connection.execute(
                    "SELECT record_json FROM outbox ORDER BY sequence"
                )
            ]
            snapshots.append((source, snapshot, records))
    store = ExperimentStore(destination)
    for source, snapshot, records in snapshots:
        store.import_records(records)
        for run_id, digest, encoded in snapshot["manifests"]:
            store.materialize_manifest(run_id, digest, _strict_json_loads(encoded))
        with store._connect() as connection:
            for row in snapshot["runs"]:
                pointer = (
                    str(destination / Path(row[8]).relative_to(source))
                    if row[8] is not None
                    else None
                )
                connection.execute(
                    "UPDATE runs SET created_at=?,updated_at=?,latest_checkpoint=? WHERE run_id=?",
                    (row[4], row[5], pointer, row[0]),
                )
            connection.commit()
            run_ids = {row[0] for row in snapshot["runs"]}
            for table in tables:
                actual = connection.execute(f"SELECT * FROM {table}").fetchall()
                if table == "events":
                    expected = [row[1:] for row in snapshot[table]]
                    actual = [row[1:] for row in actual if row[1] in run_ids]
                elif table == "runs":
                    expected = [row[:8] for row in snapshot[table]]
                    actual = [row[:8] for row in actual if row[0] in run_ids]
                else:
                    index = 1 if table == "calibration" else 0
                    expected = snapshot[table]
                    actual = [row for row in actual if row[index] in run_ids]
                if sorted(expected) != sorted(actual):
                    raise ValueError(
                        f"projection differs from source: {source}: {table}"
                    )
