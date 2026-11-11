"""Versioned, stderr-only progress records for long-running operations."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime

PROGRESS_SCHEMA_VERSION = 1
_DEFAULT_INTERVAL_SECONDS = 30.0
_DEFAULT_STALLED_AFTER_SECONDS = 300.0


class ProgressReporter:
    """Emit one common machine-readable record shape for an operation phase."""

    def __init__(
        self,
        phase: str,
        *,
        operation_id: str | None = None,
        run_id: str | None = None,
        interval_seconds: float | None = _DEFAULT_INTERVAL_SECONDS,
        stalled_after_seconds: float = _DEFAULT_STALLED_AFTER_SECONDS,
        completed_work: float | None = None,
        total_work: float | None = None,
        unit: str | None = None,
        raw_counters: Mapping[str, object] | None = None,
        derived: Mapping[str, object] | None = None,
        on_record: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        if not phase:
            raise ValueError("progress phase must not be blank")
        if interval_seconds is not None and interval_seconds <= 0:
            raise ValueError("progress interval must be positive or None")
        if stalled_after_seconds < 0:
            raise ValueError("stalled_after_seconds must be non-negative")
        self.phase = phase
        self.operation_id = operation_id
        self.run_id = run_id
        self.interval_seconds = interval_seconds
        self.stalled_after_seconds = stalled_after_seconds
        self.started_at = time.monotonic()
        self.completed_work = completed_work
        self.total_work = total_work
        self.unit = unit
        self.raw_counters = dict(raw_counters or {})
        self.derived = dict(derived or {})
        self.on_record = on_record
        self._reported_state: str | None = None
        self.last_meaningful_progress_at: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._emit("started", self.started_at)
        if interval_seconds is not None:
            self._thread = threading.Thread(
                target=self._beat,
                name=f"progress-{phase}",
                daemon=True,
            )
            self._thread.start()

    def update(
        self,
        *,
        completed_work: float | None = None,
        total_work: float | None = None,
        unit: str | None = None,
        raw_counters: Mapping[str, object] | None = None,
        state: str | None = None,
        derived: Mapping[str, object] | None = None,
        emit: bool = True,
    ) -> None:
        """Record known work and raw counters without assigning missing values."""
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot update a closed progress phase")
            previous_completed = self.completed_work
            previous_counters = self.raw_counters
            if completed_work is not None:
                self.completed_work = completed_work
            if total_work is not None:
                self.total_work = total_work
            if unit is not None:
                self.unit = unit
            if raw_counters is not None:
                self.raw_counters = dict(raw_counters)
            if state is not None:
                self._reported_state = state
            if derived is not None:
                self.derived = dict(derived)
            now = time.monotonic()
            if self._advanced(previous_completed, previous_counters):
                self.last_meaningful_progress_at = now
            if emit:
                self._emit("progress", now, state=state)

    def heartbeat(self) -> None:
        """Emit liveness separately from meaningful counter advancement."""
        with self._lock:
            if not self._closed:
                self._emit("heartbeat", time.monotonic())

    def close(self, *, failed: bool = False, state: str | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            now = time.monotonic()
            self._emit("failed" if failed else "finished", now, state=state)

    def _advanced(
        self,
        previous_completed: float | None,
        previous_counters: Mapping[str, object],
    ) -> bool:
        if (
            self.completed_work is not None
            and previous_completed is not None
            and self.completed_work > previous_completed
        ):
            return True
        for name, value in self.raw_counters.items():
            previous = previous_counters.get(name)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isinstance(previous, (int, float))
                and not isinstance(previous, bool)
                and value > previous
            ):
                return True
        return False

    def _beat(self) -> None:
        assert self.interval_seconds is not None
        while not self._stop.wait(self.interval_seconds):
            self.heartbeat()

    def _emit(self, event: str, now: float, *, state: str | None = None) -> None:
        elapsed = max(0.0, now - self.started_at)
        last_progress_age = (
            None
            if self.last_meaningful_progress_at is None
            else max(0.0, now - self.last_meaningful_progress_at)
        )
        if state is None:
            stalled = elapsed >= self.stalled_after_seconds and (
                self.last_meaningful_progress_at is None
                or last_progress_age is not None
                and last_progress_age >= self.stalled_after_seconds
            )
            state = (
                "FAILED"
                if event == "failed"
                else "COMPLETING"
                if event == "finished"
                else "NO_PROGRESS"
                if stalled
                else self._reported_state or "RUNNING"
            )
        derived = dict(self.derived)
        if state == "NO_PROGRESS":
            live = derived.get("live")
            if isinstance(live, dict):
                live = dict(live)
                live.update(
                    state="NO_PROGRESS",
                    eta_low_seconds=None,
                    eta_high_seconds=None,
                    eta_status="suspended",
                    eta_basis=None,
                )
                optimizer_eta = live.get("optimizer_only_eta")
                if isinstance(optimizer_eta, dict):
                    optimizer_eta = dict(optimizer_eta)
                    optimizer_eta.update(
                        low_seconds=None,
                        high_seconds=None,
                        status="suspended",
                        basis=None,
                    )
                    live["optimizer_only_eta"] = optimizer_eta
                derived["live"] = live
            elif "eta_low_seconds" in derived or "eta_high_seconds" in derived:
                derived.update(
                    state="NO_PROGRESS",
                    eta_low_seconds=None,
                    eta_high_seconds=None,
                    eta_status="suspended",
                    eta_basis=None,
                )
        record: dict[str, object] = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "operation_id": self.operation_id,
            "run_id": self.run_id,
            "phase": self.phase,
            "event": event,
            "observed_at_utc": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "monotonic_timestamp_seconds": now,
            "elapsed_seconds": elapsed,
            "completed_work": self.completed_work,
            "total_work": self.total_work,
            "unit": self.unit,
            "last_meaningful_progress_at_monotonic": (self.last_meaningful_progress_at),
            "last_meaningful_progress_age_seconds": last_progress_age,
            "raw_counters": dict(self.raw_counters),
            "derived": derived,
            "state": state,
        }
        print(
            json.dumps(record, sort_keys=True, allow_nan=False),
            file=sys.stderr,
            flush=True,
        )
        if self.on_record is not None:
            self.on_record(record)


@contextmanager
def progress_phase(
    phase: str,
    *,
    operation_id: str | None = None,
    run_id: str | None = None,
    interval_seconds: float | None = _DEFAULT_INTERVAL_SECONDS,
    stalled_after_seconds: float = _DEFAULT_STALLED_AFTER_SECONDS,
    completed_work: float | None = None,
    total_work: float | None = None,
    unit: str | None = None,
    raw_counters: Mapping[str, object] | None = None,
    derived: Mapping[str, object] | None = None,
    on_record: Callable[[dict[str, object]], None] | None = None,
) -> Generator[ProgressReporter]:
    """Emit versioned JSON Lines to stderr around one phase."""
    reporter = ProgressReporter(
        phase,
        operation_id=operation_id,
        run_id=run_id,
        interval_seconds=interval_seconds,
        stalled_after_seconds=stalled_after_seconds,
        completed_work=completed_work,
        total_work=total_work,
        unit=unit,
        raw_counters=raw_counters,
        derived=derived,
        on_record=on_record,
    )
    try:
        yield reporter
    except BaseException:
        reporter.close(failed=True)
        raise
    else:
        reporter.close()
