"""Durable progress counters for long, private corpus builds.

The journal lives outside immutable build artifacts; sampling and worker choices
must never change the scientific build identity or its output bytes.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path


def memory_bytes() -> tuple[int, int, int | None]:
    """Return current RSS, process peak RSS, and system available bytes."""
    try:
        rss_pages = int(Path("/proc/self/statm").read_text().split()[1])
        rss = rss_pages * os.sysconf("SC_PAGE_SIZE")
    except OSError, IndexError, ValueError:
        rss = 0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    available = None
    try:
        with Path("/proc/meminfo").open(encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except OSError, IndexError, ValueError:
        pass
    return rss, peak_bytes, available


class BuildProgress:
    """Append a bounded number of fsynced progress samples across interruptions."""

    def __init__(
        self,
        journal: Path,
        build_id: str,
        *,
        expected_input_bytes: int | None = None,
        interval_documents: int = 10_000,
        interval_input_bytes: int = 64 * 1024 * 1024,
        interval_seconds: float = 30.0,
    ) -> None:
        if min(interval_documents, interval_input_bytes) < 1 or interval_seconds <= 0:
            raise ValueError("progress intervals must be positive")
        self.journal = journal
        self.build_id = build_id
        self.expected_input_bytes = expected_input_bytes
        self.interval_documents = interval_documents
        self.interval_input_bytes = interval_input_bytes
        self.interval_seconds = interval_seconds
        self.started = time.monotonic()
        self.last_time = self.started
        self.last_documents = 0
        self.last_bytes = 0
        self.documents = 0
        self.input_bytes = 0
        self.output_records = 0
        self.phase = "starting"
        self.attempt = f"{os.getpid()}-{time.time_ns()}"
        self.record("start")

    def update(
        self,
        *,
        documents: int = 0,
        input_bytes: int = 0,
        output_records: int = 0,
        phase: str | None = None,
    ) -> None:
        if min(documents, input_bytes, output_records) < 0:
            raise ValueError("progress counters must be nonnegative")
        self.documents += documents
        self.input_bytes += input_bytes
        self.output_records += output_records
        if phase is not None and phase != self.phase:
            self.phase = phase
            self.record("phase")
        now = time.monotonic()
        if (
            self.documents - self.last_documents >= self.interval_documents
            or self.input_bytes - self.last_bytes >= self.interval_input_bytes
            or now - self.last_time >= self.interval_seconds
        ):
            self.record("interval")

    def record(self, event: str = "interval") -> None:
        now = time.monotonic()
        elapsed = max(now - self.started, 1e-9)
        rss, peak, available = memory_bytes()
        remaining = (
            max(self.expected_input_bytes - self.input_bytes, 0)
            if self.expected_input_bytes is not None
            else None
        )
        estimate = (
            remaining * elapsed / self.input_bytes
            if remaining is not None and self.input_bytes
            else None
        )
        row = {
            "attempt": self.attempt,
            "build_id": self.build_id,
            "event": event,
            "phase": self.phase,
            "documents_processed": self.documents,
            "input_bytes_processed": self.input_bytes,
            "output_records_written": self.output_records,
            "elapsed_seconds": round(elapsed, 3),
            "documents_per_second": round(self.documents / elapsed, 3),
            "input_bytes_per_second": round(self.input_bytes / elapsed, 3),
            "estimated_remaining_seconds": round(estimate, 3)
            if estimate is not None
            else None,
            "rss_bytes": rss,
            "peak_rss_bytes": peak,
            "system_available_bytes": available,
        }
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("ab") as output:
            output.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            )
            output.flush()
            os.fsync(output.fileno())
        self.last_time = now
        self.last_documents = self.documents
        self.last_bytes = self.input_bytes

    def close(self) -> None:
        self.record("complete")
