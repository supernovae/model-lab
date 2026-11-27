"""Durable progress counters for long, private corpus builds.

The journal lives outside immutable build artifacts; sampling and worker choices
must never change the scientific build identity or its output bytes.
"""

from __future__ import annotations

import json
import os
import resource
import stat
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from klab.safe_io import symlinked
from klab.verification_proofs import writable_by_others

_STAGE_SAMPLE_SECONDS = 30
_STAGE_SAMPLE_ENTRIES = 8192


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


def _scratch_bytes(directory: Path) -> tuple[int, int, bool]:
    """Sample named regular files in one operation's workspace."""
    if directory.is_symlink():
        return 0, 0, False
    logical = allocated = 0
    complete = True

    def unreadable(error: OSError) -> None:
        nonlocal complete
        complete = False

    for root, directories, files in os.walk(
        directory, followlinks=False, onerror=unreadable
    ):
        parent = Path(root)
        for name in files:
            try:
                info = (parent / name).lstat()
            except OSError:
                complete = False
                continue
            if stat.S_ISREG(info.st_mode):
                logical += info.st_size
                allocated += info.st_blocks * 512
            else:
                complete = False
        if any((parent / name).is_symlink() for name in directories):
            complete = False
    return logical, allocated, complete


def _stage_workspace_bytes(roots: tuple[Path, ...]) -> tuple[int, int, bool]:
    """Sample named files with a shared entry budget across stage roots."""
    pending = list(roots)
    seen = set()
    logical = allocated = visited = 0
    complete = True
    while pending:
        directory = pending.pop()
        try:
            info = directory.lstat()
        except FileNotFoundError:
            if directory not in roots:
                complete = False
            continue
        except OSError:
            complete = False
            continue
        if not stat.S_ISDIR(info.st_mode):
            complete = False
            continue
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if visited == _STAGE_SAMPLE_ENTRIES:
                        return logical, allocated, False
                    visited += 1
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError:
                        complete = False
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(Path(entry.path))
                    elif stat.S_ISREG(info.st_mode):
                        inode = info.st_dev, info.st_ino
                        if inode not in seen:
                            seen.add(inode)
                            logical += info.st_size
                            allocated += info.st_blocks * 512
                    else:
                        complete = False
        except OSError:
            complete = False
    return logical, allocated, complete


def recent_progress(path: Path) -> Iterator[dict[str, Any]]:
    """Yield complete journal rows, newest first, from the last 64 KiB."""
    if symlinked(path):
        return
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or writable_by_others(info)
            ):
                return
            end = info.st_size
            offset = max(0, end - 64 * 1024)
            stream.seek(offset)
            rows = stream.read(end - offset).split(b"\n")
    except OSError:
        return
    for raw in reversed(rows[bool(offset) : -1]):
        try:
            row = json.loads(raw)
        except UnicodeError, ValueError:
            continue
        if isinstance(row, dict):
            yield row


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
        verification_mode: str | None = None,
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
        self.verification_mode = verification_mode
        self._scratch_directory: Path | None = None
        self._scratch: dict[str, int] | None = None
        self._stage_outputs: dict[str, dict[str, str | int]] = {}
        self._stage_workspaces: dict[str, dict[str, Any]] = {}
        self._stage_sample_scope: tuple[str, tuple[Path, ...]] | None = None
        self._last_stage_sample = 0.0
        self.started = time.monotonic()
        self.phase_started = self.started
        self.phase_documents = 0
        self.phase_bytes = 0
        self.phase_reused_bytes = 0
        self.last_time = self.started
        self.last_documents = 0
        self.last_bytes = 0
        self.documents = 0
        self.input_bytes = 0
        self.input_bytes_reused = 0
        self.output_records = 0
        self.phase = "starting"
        self.attempt = f"{os.getpid()}-{time.time_ns()}"
        self.record("start")

    @contextmanager
    def track_scratch(self, directory: Path) -> Iterator[None]:
        """Sample an owned workspace at journal boundaries and before cleanup."""
        previous = self._scratch_directory
        self._scratch_directory = directory
        if self._scratch is None:
            self._scratch = {
                "samples": 0,
                "incomplete_samples": 0,
                "observed_peak_logical_bytes": 0,
                "observed_peak_allocated_bytes": 0,
            }
        try:
            yield
            # Preserve the pre-cleanup footprint in the completion record,
            # which can be written after disposable space is released.
            self._sample_scratch()
        finally:
            self._scratch_directory = previous

    def _sample_scratch(self) -> tuple[int, int]:
        if self._scratch_directory is None or self._scratch is None:
            return 0, 0
        logical, allocated, complete = _scratch_bytes(self._scratch_directory)
        self._scratch["samples"] += 1
        self._scratch["incomplete_samples"] += not complete
        self._scratch["observed_peak_logical_bytes"] = max(
            self._scratch["observed_peak_logical_bytes"], logical
        )
        self._scratch["observed_peak_allocated_bytes"] = max(
            self._scratch["observed_peak_allocated_bytes"], allocated
        )
        return logical, allocated

    def start_phase(
        self, phase: str, *, expected_input_bytes: int | None = None
    ) -> None:
        """Begin a phase whose rates and ETA are measured from its own start."""
        self.phase = phase
        self.expected_input_bytes = expected_input_bytes
        self.phase_started = time.monotonic()
        self.phase_documents = self.documents
        self.phase_bytes = self.input_bytes
        self.phase_reused_bytes = self.input_bytes_reused
        self.record("phase")

    def observe_stage_output(self, stage: str, key: str, payload_bytes: int) -> None:
        """Carry a sealed stage's logical output footprint in later samples."""
        self._stage_outputs[stage] = {
            "computation_key": key,
            "payload_bytes": payload_bytes,
        }

    @contextmanager
    def track_stage_workspace(
        self, stage: str, key: str, *directories: Path
    ) -> Iterator[None]:
        """Record named workspace samples while a native stage writes files."""
        self._stage_workspaces[stage] = {
            "computation_key": key,
            "completed": False,
            "samples": 0,
            "incomplete_samples": 0,
            "initial_logical_bytes": 0,
            "initial_allocated_bytes": 0,
            "observed_peak_logical_bytes": 0,
            "observed_peak_allocated_bytes": 0,
        }
        self._stage_sample_scope = stage, directories
        try:
            self._sample_stage_workspace(time.monotonic())
            yield
            self._sample_stage_workspace(time.monotonic())
            self._stage_workspaces[stage]["completed"] = True
        finally:
            self._stage_sample_scope = None

    def _sample_stage_workspace(self, now: float) -> None:
        stage, roots = self._stage_sample_scope
        logical, allocated, complete = _stage_workspace_bytes(roots)
        state = self._stage_workspaces[stage]
        if not state["samples"]:
            state["initial_logical_bytes"] = logical
            state["initial_allocated_bytes"] = allocated
        state["samples"] += 1
        state["incomplete_samples"] += not complete
        state["observed_peak_logical_bytes"] = max(
            state["observed_peak_logical_bytes"], logical
        )
        state["observed_peak_allocated_bytes"] = max(
            state["observed_peak_allocated_bytes"], allocated
        )
        self._last_stage_sample = now

    def update(
        self,
        *,
        documents: int = 0,
        input_bytes: int = 0,
        input_bytes_reused: int = 0,
        output_records: int = 0,
        phase: str | None = None,
    ) -> None:
        if min(documents, input_bytes, input_bytes_reused, output_records) < 0:
            raise ValueError("progress counters must be nonnegative")
        self.documents += documents
        self.input_bytes += input_bytes
        self.input_bytes_reused += input_bytes_reused
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
        if (
            self._stage_sample_scope is not None
            and now - self._last_stage_sample >= _STAGE_SAMPLE_SECONDS
        ):
            self._sample_stage_workspace(now)
        elapsed = max(now - self.started, 1e-9)
        phase_elapsed = max(now - self.phase_started, 1e-9)
        phase_documents = self.documents - self.phase_documents
        phase_bytes = self.input_bytes - self.phase_bytes
        phase_reused_bytes = self.input_bytes_reused - self.phase_reused_bytes
        rss, peak, available = memory_bytes()
        remaining = (
            max(self.expected_input_bytes - phase_bytes - phase_reused_bytes, 0)
            if self.expected_input_bytes is not None
            else None
        )
        estimate = (
            remaining * phase_elapsed / phase_bytes
            if remaining is not None and phase_bytes
            else 0
            if remaining == 0
            else None
        )
        row: dict[str, object] = {
            "attempt": self.attempt,
            "build_id": self.build_id,
            "event": event,
            "phase": self.phase,
            "documents_processed": self.documents,
            "input_bytes_processed": self.input_bytes,
            "input_bytes_reused": self.input_bytes_reused,
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
        if self.verification_mode is not None:
            row["verification_mode"] = self.verification_mode
        if self._stage_outputs:
            row["stage_outputs"] = self._stage_outputs
        if self._stage_workspaces:
            row["stage_workspaces"] = self._stage_workspaces
        if self._scratch is not None:
            logical, allocated = self._sample_scratch()
            row["scratch"] = {
                **self._scratch,
                "logical_bytes": logical,
                "allocated_bytes": allocated,
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
