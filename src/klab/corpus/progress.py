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
from typing import TextIO


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
        echo: TextIO | None = None,
    ) -> None:
        if min(interval_documents, interval_input_bytes) < 1 or interval_seconds <= 0:
            raise ValueError("progress intervals must be positive")
        self.journal = journal
        self.build_id = build_id
        self.expected_input_bytes = expected_input_bytes
        self.interval_documents = interval_documents
        self.interval_input_bytes = interval_input_bytes
        self.interval_seconds = interval_seconds
        self.echo = echo
        self.started = time.monotonic()
        self.phase_started = self.started
        self.phase_documents = 0
        self.phase_bytes = 0
        self.last_time = self.started
        self.last_documents = 0
        self.last_bytes = 0
        self.documents = 0
        self.input_bytes = 0
        self.output_records = 0
        self.phase = "starting"
        self.attempt = f"{os.getpid()}-{time.time_ns()}"
        self.record("start")

    def start_phase(
        self, phase: str, *, expected_input_bytes: int | None = None
    ) -> None:
        """Begin a phase whose rates and ETA are measured from its own start."""
        self.phase = phase
        self.expected_input_bytes = expected_input_bytes
        self.phase_started = time.monotonic()
        self.phase_documents = self.documents
        self.phase_bytes = self.input_bytes
        self.record("phase")

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
        phase_elapsed = max(now - self.phase_started, 1e-9)
        phase_documents = self.documents - self.phase_documents
        phase_bytes = self.input_bytes - self.phase_bytes
        rss, peak, available = memory_bytes()
        remaining = (
            max(self.expected_input_bytes - phase_bytes, 0)
            if self.expected_input_bytes is not None
            else None
        )
        estimate = (
            remaining * phase_elapsed / phase_bytes
            if remaining is not None and phase_bytes
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
            "documents_per_second": round(phase_documents / phase_elapsed, 3),
            "input_bytes_per_second": round(phase_bytes / phase_elapsed, 3),
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
        if self.echo is not None:
            eta = row["estimated_remaining_seconds"]
            print(
                f"[{row['phase']}] {row['documents_processed']} documents, "
                f"{row['input_bytes_processed'] / 1e9:.2f} GB, "
                f"{row['documents_per_second']:.0f} documents/s, "
                f"{row['input_bytes_per_second'] / 1e6:.0f} MB/s, "
                f"ETA {'unknown' if eta is None else f'{eta:.0f} s'}",
                file=self.echo,
                flush=True,
            )
        self.last_time = now
        self.last_documents = self.documents
        self.last_bytes = self.input_bytes

    def close(self) -> None:
        self.record("complete")
