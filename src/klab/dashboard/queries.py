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

from klab.training.metrics import SCHEMA_VERSION


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
    created_at: str = ""


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
        if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise ValueError("unsupported experiment-store schema")
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
    rows = connection.execute(
        "SELECT run_id,name,status,parent_run_id,updated_at,config_json,"
        "metadata_json,created_at FROM runs ORDER BY updated_at DESC"
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
                created_at=str(row[7] or ""),
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


# What happened to a run, as recorded: these event kinds decide its verdict.
OUTCOME_EVENTS = (
    "run_failed",
    "run_interrupted",
    "recovery",
    "resumable",
    "runtime_optimization_transition",
    "runtime_drift",
    "runtime_final_observation",
)
_LATEST_METRICS = ("train/loss", "validation/loss", "optimizer/learning_rate")


@dataclass(frozen=True)
class RunFacts:
    """Aggregates of one run's recorded telemetry, without its full metric series."""

    first_step: int | None = None
    last_step: int | None = None
    first_tokens: int | None = None
    last_tokens: int | None = None
    first_wall_time: float | None = None
    last_wall_time: float | None = None
    updates: int = 0
    latest: dict[str, float] | None = None
    best_validation_loss: float | None = None
    tokens_per_second: float | None = None
    events: tuple[tuple[str, int, dict[str, object]], ...] = ()
    verified_checkpoints: int = 0
    last_checkpoint_step: int | None = None
    heartbeat_at: str | None = None
    heartbeat_event: str | None = None


def run_facts(root: Path) -> dict[str, RunFacts]:
    """One aggregate per run for the runs hub: a few grouped queries, no series."""
    with _connect(root) as connection:
        tables = _tables(connection)
        facts: dict[str, dict[str, object]] = {}

        def entry(run_id: object) -> dict[str, object]:
            return facts.setdefault(str(run_id), {})

        if "metrics" in tables:
            for row in connection.execute(
                "SELECT run_id,MIN(step),MAX(step),MIN(tokens_seen),MAX(tokens_seen),"
                "MIN(wall_time),MAX(wall_time),COUNT(DISTINCT step) FROM metrics "
                "GROUP BY run_id"
            ):
                entry(row[0]).update(
                    first_step=row[1],
                    last_step=row[2],
                    first_tokens=row[3],
                    last_tokens=row[4],
                    first_wall_time=row[5],
                    last_wall_time=row[6],
                    updates=row[7],
                )
            marks = ",".join("?" for _ in _LATEST_METRICS)
            for run_id, name, value in connection.execute(
                "SELECT m.run_id,m.name,m.value FROM metrics m JOIN ("
                f"SELECT run_id,name,MAX(step) AS step FROM metrics WHERE name IN ({marks}) "
                "GROUP BY run_id,name) l ON m.run_id=l.run_id AND m.name=l.name "
                "AND m.step=l.step",
                _LATEST_METRICS,
            ):
                entry(run_id).setdefault("latest", {})[str(name)] = float(value)  # type: ignore[index]
            for run_id, value in connection.execute(
                "SELECT run_id,MIN(value) FROM metrics WHERE name='validation/loss' "
                "GROUP BY run_id"
            ):
                entry(run_id)["best_validation_loss"] = value
            # The first updates of a process compile the model; leave them out of the rate.
            for run_id, value in connection.execute(
                "SELECT m.run_id,AVG(m.value) FROM metrics m JOIN ("
                "SELECT run_id,MIN(step) AS first FROM metrics GROUP BY run_id) f "
                "ON m.run_id=f.run_id WHERE m.name='performance/tokens_per_second' "
                "AND m.step>f.first+3 GROUP BY m.run_id"
            ):
                entry(run_id)["tokens_per_second"] = value
        if "events" in tables:
            marks = ",".join("?" for _ in OUTCOME_EVENTS)
            for run_id, kind, step, payload in connection.execute(
                f"SELECT run_id,kind,step,payload_json FROM events WHERE kind IN ({marks}) "
                "ORDER BY id",
                OUTCOME_EVENTS,
            ):
                entry(run_id).setdefault("events", []).append(  # type: ignore[union-attr]
                    (str(kind), int(step), _object(payload))
                )
        if "checkpoints" in tables:
            for run_id, count, step in connection.execute(
                "SELECT run_id,COUNT(*),MAX(step) FROM checkpoints "
                "WHERE verification_status='verified' GROUP BY run_id"
            ):
                entry(run_id).update(
                    verified_checkpoints=count, last_checkpoint_step=step
                )
        if "runtime_progress_snapshots" in tables:
            for run_id, payload in connection.execute(
                "SELECT run_id,payload_json FROM runtime_progress_snapshots"
            ):
                progress = _object(payload)
                observed, event = progress.get("observed_at_utc"), progress.get("event")
                entry(run_id).update(
                    heartbeat_at=observed if isinstance(observed, str) else None,
                    heartbeat_event=event if isinstance(event, str) else None,
                )
    return {
        run_id: RunFacts(**{**values, "events": tuple(values.get("events", ()))})  # type: ignore[arg-type]
        for run_id, values in facts.items()
    }


def metric_series(
    root: Path, run_ids: tuple[str, ...], names: tuple[str, ...], max_points: int = 2000
) -> list[dict[str, object]]:
    """The named series of the given runs, thinned to about ``max_points`` per series.

    Thinning keeps every ``k``-th step of a run (and always its last), so a series of a
    multi-day run stays cheap to send to the browser without changing its shape.
    """
    if not run_ids or not names:
        return []
    rows: list[dict[str, object]] = []
    with _connect(root) as connection:
        if "metrics" not in _tables(connection):
            return []
        name_marks = ",".join("?" for _ in names)
        for run_id in run_ids:
            for name, count, last in connection.execute(
                f"SELECT name,COUNT(*),MAX(step) FROM metrics WHERE run_id=? AND name IN "
                f"({name_marks}) GROUP BY name",
                (run_id, *names),
            ).fetchall():
                stride = max(1, -(-int(count) // max_points))
                for step, tokens, wall_time, value in connection.execute(
                    "SELECT step,tokens_seen,wall_time,value FROM metrics WHERE run_id=? "
                    "AND name=? AND (step % ? = 0 OR step=?) ORDER BY step",
                    (run_id, name, stride, last),
                ):
                    rows.append(
                        {
                            "run_id": run_id,
                            "name": name,
                            "step": step,
                            "tokens_seen": tokens,
                            "wall_time": wall_time,
                            "value": value,
                        }
                    )
    return rows
