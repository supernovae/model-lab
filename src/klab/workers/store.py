"""Durable controller queue projection backed by the local ExperimentStore."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from klab.training.metrics import ExperimentStore, _canonical_json

QUEUE_STATES = frozenset(
    {
        "QUEUED",
        "ASSIGNED",
        "RUNNING",
        "CANCEL_REQUESTED",
        "CANCELLED",
        "COMPLETE",
        "FAILED",
        "INTERRUPTED",
        "UNKNOWN",
    }
)
ACTIVE_QUEUE_STATES = frozenset({"ASSIGNED", "RUNNING", "CANCEL_REQUESTED"})
RECONCILABLE_QUEUE_STATES = ACTIVE_QUEUE_STATES | frozenset({"UNKNOWN"})
TERMINAL_QUEUE_STATES = frozenset({"CANCELLED", "COMPLETE", "FAILED", "INTERRUPTED"})
TERMINAL_RECEIPT_STATES = frozenset({"COMPLETE", "FAILED", "INTERRUPTED", "UNKNOWN"})


class ControllerStore:
    """Queue authority co-located with the controller's experiment projection.

    Queue writes are intentionally not replication events. Remote workers retain
    their own outbox authority; this database only records controller decisions
    and imports their immutable records through :class:`ExperimentStore`.
    """

    def __init__(self, root: Path, *, read_only: bool = False) -> None:
        self.root = root
        self.metrics = ExperimentStore(root, read_only=read_only)
        self.path = self.metrics.path

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.metrics._connect() as con:
            try:
                con.execute("BEGIN IMMEDIATE")
                yield con
                con.commit()
            except BaseException:
                con.rollback()
                raise

    @staticmethod
    def _json(value: object) -> str:
        return _canonical_json(value)

    def save_worker(self, worker_id: str, record: Mapping[str, object]) -> None:
        with self.transaction() as con:
            con.execute(
                "INSERT INTO workers(worker_id,record_json) VALUES(?,?) "
                "ON CONFLICT(worker_id) DO UPDATE SET record_json=excluded.record_json",
                (worker_id, self._json(record)),
            )

    def worker_records(self) -> list[dict[str, object]]:
        with self.metrics._connect() as con:
            rows = con.execute(
                "SELECT record_json FROM workers ORDER BY worker_id"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def enqueue_many(self, entries: list[Mapping[str, object]]) -> None:
        """Atomically reserve every submission after all bundles are prepared."""
        with self.transaction() as con:
            for entry in entries:
                experiment_id = str(entry["experiment_id"])
                attempt_id = str(entry["attempt_id"])
                run_id = str(entry["run_id"])
                spec = entry["spec"]
                if not isinstance(spec, Mapping):
                    raise TypeError("experiment specification must be an object")
                con.execute(
                    "INSERT INTO experiments(experiment_id,specification_json,status,submitted_at) "
                    "VALUES(?,?,?,datetime('now'))",
                    (experiment_id, self._json(spec), "QUEUED"),
                )
                con.execute(
                    "INSERT INTO attempts("
                    "attempt_id,experiment_id,run_id,worker_id,status,receipt_json,"
                    "terminal_receipt_json,queued_reason,ingestion_status,ingestion_error"
                    ") VALUES(?,?,?,NULL,'QUEUED',NULL,NULL,NULL,'PENDING',NULL)",
                    (attempt_id, experiment_id, run_id),
                )

    def raw_attempt_specifications(self) -> list[dict[str, Any]]:
        """Read identity rows and original specification JSON in one snapshot.

        Recovery readers retain strict decoding at their boundary rather than
        losing duplicate keys through the ordinary queue projection.
        """
        with self.metrics._connect() as connection:
            records = connection.execute(
                "SELECT a.experiment_id,a.attempt_id,a.run_id,e.specification_json "
                "FROM attempts a JOIN experiments e ON e.experiment_id=a.experiment_id"
            ).fetchall()
        return [
            {
                "experiment_id": record[0],
                "attempt_id": record[1],
                "run_id": record[2],
                "specification_json": record[3],
            }
            for record in records
        ]

    def attempts(self, statuses: frozenset[str] | None = None) -> list[dict[str, Any]]:
        with self.metrics._connect() as con:
            columns = {row[1] for row in con.execute("PRAGMA table_info(attempts)")}
        recovery = (
            "a.recovery_observation_json"
            if "recovery_observation_json" in columns
            else "NULL"
        )
        query = (
            "SELECT a.attempt_id,a.experiment_id,a.run_id,a.worker_id,a.status,"
            "a.receipt_json,a.terminal_receipt_json,a.queued_reason,"
            f"a.ingestion_status,a.ingestion_error,e.specification_json,e.submitted_at,{recovery} "
            "FROM attempts a JOIN experiments e ON e.experiment_id=a.experiment_id"
        )
        params: tuple[object, ...] = ()
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" WHERE a.status IN ({placeholders})"
            params = tuple(sorted(statuses))
        query += " ORDER BY e.submitted_at,a.rowid"
        with self.metrics._connect() as con:
            rows = con.execute(query, params).fetchall()
        return [
            {
                "attempt_id": row[0],
                "experiment_id": row[1],
                "run_id": row[2],
                "worker_id": row[3],
                "status": row[4],
                "receipt": json.loads(row[5]) if row[5] else None,
                "terminal_receipt": json.loads(row[6]) if row[6] else None,
                "queued_reason": row[7],
                "ingestion_status": row[8],
                "ingestion_error": row[9],
                "spec": json.loads(row[10]),
                "submitted_at": row[11],
                "recovery_observation": json.loads(row[12]) if row[12] else None,
            }
            for row in rows
        ]

    def attempt_by_run(self, run_id: str) -> dict[str, Any] | None:
        return next((row for row in self.attempts() if row["run_id"] == run_id), None)

    def assign(self, attempt_id: str, worker_id: str) -> bool:
        with self.transaction() as con:
            updated = con.execute(
                "UPDATE attempts SET worker_id=?,status='ASSIGNED',queued_reason=NULL "
                "WHERE attempt_id=? AND status='QUEUED'",
                (worker_id, attempt_id),
            ).rowcount
            if updated:
                con.execute(
                    "UPDATE experiments SET status='ASSIGNED' WHERE experiment_id="
                    "(SELECT experiment_id FROM attempts WHERE attempt_id=?)",
                    (attempt_id,),
                )
            return updated == 1

    def set_queued_reason(self, attempt_id: str, reason: str | None) -> None:
        with self.transaction() as con:
            con.execute(
                "UPDATE attempts SET queued_reason=? WHERE attempt_id=? AND status='QUEUED'",
                (reason, attempt_id),
            )

    def set_receipt(
        self, attempt_id: str, receipt: Mapping[str, object], status: str
    ) -> None:
        """Persist worker evidence without allowing stale nonterminal regressions."""
        if status not in QUEUE_STATES:
            raise ValueError(f"invalid queue status: {status}")
        encoded = self._json(receipt)
        receipt_state = receipt.get("state")
        if receipt_state is not None and receipt_state not in {
            "PREPARED",
            "RUNNING",
            "COMPLETE",
            "FAILED",
            "INTERRUPTED",
            "UNKNOWN",
        }:
            raise ValueError("invalid attempt receipt state")
        terminal = receipt_state in TERMINAL_RECEIPT_STATES
        with self.transaction() as con:
            current = con.execute(
                "SELECT status,terminal_receipt_json,recovery_observation_json FROM attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if current is None:
                raise KeyError(f"unknown attempt: {attempt_id}")
            if current[2] is not None:
                if terminal:
                    self.retain_recovery_conflict(attempt_id, dict(receipt))
                    con.execute(
                        "UPDATE attempts SET ingestion_error=? WHERE attempt_id=?",
                        (
                            "CONFLICT: terminal evidence after confirmed recovery",
                            attempt_id,
                        ),
                    )
                return
            if current[1] is not None:
                # Worker terminal receipts are monotonic.  A delayed response
                # cannot revise the first durable terminal evidence.
                return
            if current[0] == "CANCEL_REQUESTED" and not terminal:
                status = "CANCEL_REQUESTED"
            con.execute(
                "UPDATE attempts SET receipt_json=?,terminal_receipt_json="
                "CASE WHEN ? THEN ? ELSE terminal_receipt_json END,"
                "status=?,ingestion_status=CASE WHEN ? THEN 'PENDING' "
                "ELSE ingestion_status END,ingestion_error=CASE WHEN ? THEN NULL "
                "ELSE ingestion_error END WHERE attempt_id=?",
                (encoded, terminal, encoded, status, terminal, terminal, attempt_id),
            )
            con.execute(
                "UPDATE experiments SET status=? WHERE experiment_id="
                "(SELECT experiment_id FROM attempts WHERE attempt_id=?)",
                (status, attempt_id),
            )

    def retain_recovery_conflict(
        self, attempt_id: str, evidence: dict[str, Any]
    ) -> None:
        """Retain contradictory evidence without revising recovered lineage."""
        import hashlib

        from .execution import _atomic_json
        from .models import _identifier

        _identifier(attempt_id, "attempt_id")
        encoded = self._json(evidence).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        path = self.root / ".relay" / attempt_id / "conflicts" / f"{digest}.json"
        if not path.exists():
            _atomic_json(path, evidence)

    def record_recovery(
        self,
        attempt_id: str,
        observation: dict[str, Any],
        *,
        expected_worker_id: str,
        expected_spec_digest: str,
        expected_bundle_digest: str,
    ) -> None:
        """Fence a confirmed lost worker without fabricating terminal evidence."""
        from datetime import datetime

        from .models import ExperimentSpec, _identifier, _sha256

        if (
            set(observation)
            != {"reason", "commit_sha256", "confirmed_at", "instance_id"}
            or observation["reason"] != "HOSTED_WORKER_LOST"
        ):
            raise ValueError("invalid recovery observation")
        _sha256(observation["commit_sha256"])
        _identifier(observation["instance_id"], "instance_id")
        confirmed = datetime.fromisoformat(observation["confirmed_at"])
        if confirmed.tzinfo is None:
            raise ValueError("recovery confirmation requires a timezone")
        encoded = self._json(observation)
        with self.transaction() as con:
            row = con.execute(
                "SELECT a.worker_id,a.terminal_receipt_json,a.recovery_observation_json,"
                "e.specification_json FROM attempts a JOIN experiments e "
                "ON e.experiment_id=a.experiment_id WHERE a.attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown attempt: {attempt_id}")
            spec = ExperimentSpec.model_validate(json.loads(row[3]))
            if (
                row[0] != expected_worker_id
                or spec.digest() != expected_spec_digest
                or spec.dispatch_bundle_digest != expected_bundle_digest
            ):
                raise ValueError("recovery assigned identity mismatch")
            if row[1] is not None:
                raise ValueError("terminal evidence conflicts with recovery")
            if row[2] is not None:
                if row[2] != encoded:
                    raise ValueError("conflicting recovery observation")
                return
            con.execute(
                "UPDATE attempts SET status='UNKNOWN',recovery_observation_json=?,"
                "ingestion_status='PENDING',ingestion_error=NULL WHERE attempt_id=?",
                (encoded, attempt_id),
            )
            con.execute(
                "UPDATE experiments SET status='UNKNOWN' WHERE experiment_id="
                "(SELECT experiment_id FROM attempts WHERE attempt_id=?)",
                (attempt_id,),
            )

    def mark_ingestion_complete(self, attempt_id: str) -> None:
        with self.transaction() as con:
            row = con.execute(
                "SELECT ingestion_error FROM attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if row is not None and "CONFLICT" in str(row[0] or ""):
                raise ValueError(
                    "conflicting evidence requires explicit reconciliation"
                )
            con.execute(
                "UPDATE attempts SET ingestion_status='COMPLETE',ingestion_error=NULL "
                "WHERE attempt_id=? AND (terminal_receipt_json IS NOT NULL "
                "OR recovery_observation_json IS NOT NULL)",
                (attempt_id,),
            )

    def mark_ingestion_not_required(self, attempt_id: str) -> None:
        with self.transaction() as con:
            con.execute(
                "UPDATE attempts SET ingestion_status='NOT_REQUIRED',ingestion_error=NULL "
                "WHERE attempt_id=? AND terminal_receipt_json IS NOT NULL",
                (attempt_id,),
            )

    def mark_ingestion_error(self, attempt_id: str, error: Exception) -> None:
        message = f"{type(error).__name__}: {error}"
        with self.transaction() as con:
            con.execute(
                "UPDATE attempts SET ingestion_status='ERROR',ingestion_error=? "
                "WHERE attempt_id=? AND (terminal_receipt_json IS NOT NULL "
                "OR recovery_observation_json IS NOT NULL)",
                (message, attempt_id),
            )

    def pending_ingestion(self) -> list[dict[str, Any]]:
        return [
            attempt
            for attempt in self.attempts()
            if (
                attempt["terminal_receipt"] is not None
                or attempt["recovery_observation"] is not None
            )
            and attempt["ingestion_status"] in {"PENDING", "ERROR"}
            and "CONFLICT" not in str(attempt.get("ingestion_error") or "")
        ]

    def mark_unknown_if_nonterminal(self, attempt_id: str, reason: str) -> None:
        """Record controller reachability without replacing worker receipt evidence."""
        with self.transaction() as con:
            row = con.execute(
                "SELECT terminal_receipt_json FROM attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown attempt: {attempt_id}")
            if row[0] is not None:
                return
            con.execute(
                "UPDATE attempts SET status=CASE WHEN status='CANCEL_REQUESTED' "
                "THEN status ELSE 'UNKNOWN' END,queued_reason=? WHERE attempt_id=?",
                (reason, attempt_id),
            )
            con.execute(
                "UPDATE experiments SET status=(SELECT status FROM attempts WHERE attempt_id=?) "
                "WHERE experiment_id=(SELECT experiment_id FROM attempts WHERE attempt_id=?)",
                (attempt_id, attempt_id),
            )

    def request_cancel(self, run_id: str) -> dict[str, Any]:
        with self.transaction() as con:
            row = con.execute(
                "SELECT attempt_id,status,terminal_receipt_json,recovery_observation_json "
                "FROM attempts WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown run: {run_id}")
            attempt_id, status, terminal_receipt, recovery_observation = row
            if status == "QUEUED":
                target = "CANCELLED"
                con.execute(
                    "UPDATE attempts SET ingestion_status='NOT_REQUIRED',ingestion_error=NULL "
                    "WHERE attempt_id=?",
                    (attempt_id,),
                )
            elif (
                status in TERMINAL_QUEUE_STATES
                or terminal_receipt is not None
                or recovery_observation is not None
            ):
                target = status
            else:
                target = "CANCEL_REQUESTED"
            con.execute(
                "UPDATE attempts SET status=? WHERE attempt_id=?", (target, attempt_id)
            )
            con.execute(
                "UPDATE experiments SET status=? WHERE experiment_id="
                "(SELECT experiment_id FROM attempts WHERE attempt_id=?)",
                (target, attempt_id),
            )
            return {"attempt_id": attempt_id, "status": target}

    def import_records(self, records: list[Mapping[str, object] | str | bytes]) -> None:
        self.metrics.import_records(records)

    def imported_sequence(self, origin_id: str) -> int:
        """Return the committed contiguous replication watermark for one origin."""
        with self.metrics._connect() as con:
            return int(
                con.execute(
                    "SELECT COALESCE(MAX(sequence),0) FROM ingested_records WHERE origin_id=?",
                    (origin_id,),
                ).fetchone()[0]
            )
