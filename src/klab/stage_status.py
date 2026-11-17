"""A slow one-line heartbeat and a small status file for a long `klab stage` phase.

A watchdog or a person reads the line in the log, or the status file beside the
stage output, to tell a stuck phase from a slow one. A daemon thread wakes once
per interval; nothing runs per step, and the pilot supervisor only stores its
counters here.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

INTERVAL_SECONDS = 60.0


class StageStatus:
    def __init__(self, path: Path, *, interval: float = INTERVAL_SECONDS) -> None:
        self.path = path
        self.interval = interval
        self.phase_name = "starting"
        self.state = "running"
        self.step: int | None = None
        self.total: int | None = None
        self.detail: str | None = None
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @contextmanager
    def phase(self, name: str) -> Iterator[StageStatus]:
        self.phase_name, self.state = name, "running"
        self.step = self.total = self.detail = None
        self.started = time.monotonic()
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._beat, name="stage-status", daemon=True
            )
            self._thread.start()
        self.emit()
        try:
            yield self
        except BaseException:
            self.state = "failed"
            self.emit()
            raise
        self.state = "finished"
        self.emit()

    def pilot(self, supervision: Any) -> None:
        """Record the supervisor's trusted counters; called about once a second."""
        self.step = supervision.completed_steps
        self.total = supervision.expected_steps
        self.detail = supervision.phases[-1] if supervision.phases else None

    def emit(self) -> None:
        elapsed = time.monotonic() - self.started
        counted = "" if self.step is None else f" step={self.step}/{self.total or '?'}"
        where = "" if self.detail is None else f" in={self.detail}"
        print(
            f"stage {self.phase_name} {self.state} elapsed={elapsed:.0f}s{counted}{where}",
            file=sys.stderr,
            flush=True,
        )
        record = {
            "phase": self.phase_name,
            "state": self.state,
            "elapsed_seconds": round(elapsed),
            "step": self.step,
            "total_steps": self.total,
            "detail": self.detail,
            "updated_unix": round(time.time()),
        }
        try:
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
            os.replace(temporary, self.path)
        except OSError:
            pass  # status is advisory; never fail the stage for it

    def _beat(self) -> None:
        while not self._stop.wait(self.interval):
            if self.state == "running":
                self.emit()
