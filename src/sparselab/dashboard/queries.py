"""Short-lived, read-only SQLite queries for the local dashboard.

This module deliberately never imports :class:`ExperimentStore`: opening that class can
migrate a database, while a dashboard must be safe to point at a live or legacy run
store.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from sparselab.training.metrics import SCHEMA_VERSION


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    name: str
    status: str
    parent_run_id: str | None
    updated_at: str
    config: dict[str, object]
    metadata: dict[str, object]
    purpose: str | None


@dataclass(frozen=True)
class DashboardSnapshot:
    """A consistent, read-only projection used by one dashboard render."""

    runs: tuple[RunRecord, ...]
    metrics: tuple[dict[str, object], ...]
    events: tuple[dict[str, object], ...]
    runtime_progress: tuple[dict[str, object], ...]
    stages: tuple[dict[str, object], ...]
    checkpoints: tuple[dict[str, object], ...]
    manifests: dict[str, dict[str, object]]


@contextmanager
def _connect(root: Path) -> Iterator[sqlite3.Connection]:
    database = root / "experiments.sqlite3"
    with closing(
        sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    ) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=1000")
        connection.execute("BEGIN")
        if connection.execute("PRAGMA user_version").fetchone()[0] > SCHEMA_VERSION:
            raise ValueError("unsupported newer experiment-store schema")
        yield connection


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, str):
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _run_rows(connection: sqlite3.Connection) -> tuple[RunRecord, ...]:
    if "runs" not in _tables(connection):
        return ()
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(runs)")}
    metadata_column = "metadata_json" if "metadata_json" in columns else "NULL"
    rows = connection.execute(
        "SELECT run_id,name,status,parent_run_id,updated_at,config_json,"
        f"{metadata_column} FROM runs ORDER BY updated_at DESC"
    ).fetchall()
    records: list[RunRecord] = []
    for row in rows:
        metadata = _object(row[6])
        purpose = metadata.get("purpose")
        records.append(
            RunRecord(
                run_id=str(row[0]),
                name=str(row[1]),
                status=str(row[2]),
                parent_run_id=str(row[3]) if row[3] is not None else None,
                updated_at=str(row[4]),
                config=_object(row[5]),
                metadata=metadata,
                purpose=str(purpose) if isinstance(purpose, str) else None,
            )
        )
    return tuple(records)


def _rows(
    connection: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    order: str,
    run_ids: tuple[str, ...] | None = None,
) -> tuple[dict[str, object], ...]:
    if table not in _tables(connection):
        return ()
    selected = ",".join(columns)
    where = ""
    parameters = ()
    if run_ids is not None:
        where = " WHERE run_id IN (" + ",".join("?" for _ in run_ids) + ")"
        parameters = run_ids
    values = connection.execute(
        f"SELECT {selected} FROM {table}{where} ORDER BY {order}", parameters
    ).fetchall()
    return tuple(dict(zip(columns, row, strict=True)) for row in values)


def snapshot(root: Path, run_ids: tuple[str, ...] | None = None) -> DashboardSnapshot:
    """Read the known projection tables without upgrading a legacy database."""
    with _connect(root) as connection:
        records = _run_rows(connection)
        metric_rows = _rows(
            connection,
            "metrics",
            ("run_id", "step", "tokens_seen", "wall_time", "name", "value"),
            "step, name",
            run_ids,
        )
        event_rows = _rows(
            connection,
            "events",
            ("run_id", "step", "tokens_seen", "wall_time", "kind", "payload_json"),
            "id",
            run_ids,
        )
        stage_rows = _rows(
            connection,
            "stage_history",
            (
                "run_id",
                "sequence",
                "stage",
                "status",
                "step",
                "tokens_seen",
                "started_at",
                "finished_at",
                "payload_json",
            ),
            "run_id, sequence",
            run_ids,
        )
        checkpoint_rows = _rows(
            connection,
            "checkpoints",
            (
                "run_id",
                "checkpoint_id",
                "relative_path",
                "digest",
                "step",
                "tokens_seen",
                "created_at",
                "size_bytes",
                "validation_loss",
                "verified_at",
                "verification_status",
                "resume_level",
                "backend",
            ),
            "run_id, step, created_at",
            run_ids,
        )
        runtime_progress_rows = _rows(
            connection,
            "runtime_progress_snapshots",
            ("run_id", "step", "tokens_seen", "wall_time", "payload_json"),
            "run_id",
            run_ids,
        )
        manifest_rows = _rows(
            connection, "manifests", ("run_id", "digest", "json"), "run_id", run_ids
        )
        assignments = {
            str(row["run_id"]): row["worker_id"]
            for row in _rows(
                connection, "attempts", ("run_id", "worker_id"), "run_id", run_ids
            )
        }
    manifests = {
        str(row["run_id"]): {"digest": row["digest"], **_object(row["json"])}
        for row in manifest_rows
    }
    for record in records:
        manifest = manifests.get(record.run_id, {})
        if "manifest_reference" in manifest:
            # Presentation only: never mistake unresolved lineage for a fresh run.
            pending = "pending artifact collection"
            manifest.update(
                collection_status=pending,
                worker_id=assignments.get(record.run_id) or pending,
                parent_run_id=record.parent_run_id or pending,
                continuation_kind=pending,
                checkpoint_sha256=pending,
            )
    records = tuple(
        replace(
            record,
            purpose=record.purpose
            or (
                str(manifests[record.run_id]["purpose"])
                if isinstance(manifests.get(record.run_id, {}).get("purpose"), str)
                else None
            ),
        )
        for record in records
    )
    return DashboardSnapshot(
        runs=records,
        metrics=metric_rows,
        events=event_rows,
        stages=stage_rows,
        runtime_progress=runtime_progress_rows,
        checkpoints=checkpoint_rows,
        manifests=manifests,
    )


def runs(root: Path) -> list[RunRecord]:
    """Read sidebar run metadata without materializing every metric series."""
    with _connect(root) as connection:
        return list(_run_rows(connection))


def metrics(root: Path, run_ids: list[str]) -> list[dict[str, object]]:
    wanted = set(run_ids)
    return [row for row in snapshot(root).metrics if row["run_id"] in wanted]


def events(root: Path, run_ids: list[str]) -> list[dict[str, object]]:
    wanted = set(run_ids)
    return [row for row in snapshot(root).events if row["run_id"] in wanted]


def stages(root: Path, run_ids: list[str]) -> list[dict[str, object]]:
    wanted = set(run_ids)
    return [row for row in snapshot(root).stages if row["run_id"] in wanted]
