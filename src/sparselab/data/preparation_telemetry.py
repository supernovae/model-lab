"""Operational, per-preparation measurements; never part of cache identity."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from sparselab import host_capacity

if TYPE_CHECKING:
    from sparselab.bottleneck_observations import BottleneckObserver

_DURATIONS = (
    "source_iteration_seconds",
    "tokenizer_encoding_seconds",
    "python_bookkeeping_seconds",
    "spool_write_seconds",
    "finalize_fsync_seconds",
    "deep_verification_hash_seconds",
)


class PreparationTelemetry:
    def __init__(
        self, workspace: Path, *, observer: BottleneckObserver | None = None
    ) -> None:
        self.workspace = workspace
        self.observer = observer
        self.started = time.monotonic()
        self.shards: dict[str, object] = {}
        self.shard_workers = 1
        self.workers: dict[str, object] = {}
        self.records = 0
        self.source_bytes = 0
        self.output_tokens = 0
        self.logical_input_bytes = 0
        self.logical_output_bytes = 0
        self.peak_rss_bytes: int | None = None
        self.tokenizer_rayon_threads: int | None = None
        self.tokenizer_host_work_plan: dict[str, object] | None = None
        self.durations = dict.fromkeys(_DURATIONS, 0.0)
        try:
            self._process = psutil.Process()
        except OSError, psutil.Error:
            self._process = None

    def add(self, name: str, elapsed: float) -> None:
        self.durations[name] += elapsed

    def snapshot(self) -> dict[str, object]:
        elapsed = max(time.monotonic() - self.started, 0.0)
        try:
            rss: int | None = (
                self._process.memory_info().rss if self._process is not None else None
            )
        except OSError, psutil.Error:
            rss = None
        if rss is not None:
            self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)
        try:
            available_ram: int | None = host_capacity.measure_memory().available
        except OSError, psutil.Error:
            available_ram = None
        try:
            disk = os.statvfs(self.workspace)
            disk_free: int | None = disk.f_bavail * disk.f_frsize
            free_inodes: int | None = disk.f_favail if disk.f_files else None
        except OSError:
            disk_free = free_inodes = None
        result = {
            "shard_workers": self.shard_workers,
            "worker_pool": self.workers,
            "shards": self.shards,
            "records": self.records,
            "source_bytes": self.source_bytes,
            "output_tokens": self.output_tokens,
            "elapsed_seconds": elapsed,
            "source_mb_per_second": self.source_bytes / 1_000_000 / elapsed
            if elapsed
            else 0.0,
            "output_tokens_per_second": self.output_tokens / elapsed
            if elapsed
            else 0.0,
            "current_rss_bytes": rss,
            "peak_rss_bytes": self.peak_rss_bytes,
            "tokenizer_rayon_threads": self.tokenizer_rayon_threads,
            "tokenizer_host_work_plan": self.tokenizer_host_work_plan,
            "host_available_ram_bytes": available_ram,
            "logical_input_bytes": self.logical_input_bytes,
            "logical_output_bytes": self.logical_output_bytes,
            "disk_free_bytes": disk_free,
            "disk_free_inodes": free_inodes,
            **self.durations,
        }
        if self.observer is not None:
            result["bottleneck_observations"] = list(self.observer.records)
        return result
