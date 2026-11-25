"""Capacity and backpressure guarantees of the shared Corpus runner."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from contextlib import closing, suppress
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from klab import host_capacity
from klab.corpus import shards
from klab.corpus.release_verify import verification_settings


def test_warm_results_are_read_only_within_the_worker_window(tmp_path, monkeypatch):
    plan = shards.ShardPlan(
        "release verification", [(i,) for i in range(9)], 0, store=tmp_path
    )
    assert list(shards.run_sharded(plan, int, ordered=True)) == list(range(9))

    reads = []
    read_text = Path.read_text

    def count_reads(path, *args, **kwargs):
        if path.parent == tmp_path:
            reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", count_reads)
    monkeypatch.setattr(shards, "PARALLEL_MIN_BYTES", 0)
    monkeypatch.setattr(shards, "worker_count", lambda: 3)
    monkeypatch.setattr(host_capacity.psutil, "cpu_count", lambda *, logical: 3)
    monkeypatch.setattr(
        host_capacity.psutil,
        "Process",
        lambda: SimpleNamespace(cpu_affinity=lambda: [0, 1]),
    )
    with (
        verification_settings(workers=3),
        closing(shards.run_sharded(plan, int, ordered=True)) as results,
    ):
        assert next(results) == 0
        assert len(reads) == 2
        assert list(results) == list(range(1, 9))
    assert len(set(reads)) == len(reads) == 9


def test_runner_refuses_parallel_work_when_memory_cannot_fit_a_worker(monkeypatch):
    monkeypatch.setattr(shards, "PARALLEL_MIN_BYTES", 0)
    monkeypatch.setattr(shards, "worker_count", lambda: 3)
    monkeypatch.setattr(
        host_capacity,
        "measure_memory",
        lambda: SimpleNamespace(total=4 * 1024**3, available=1024**3),
    )
    plan = shards.ShardPlan("release verification", [(0,), (1,)], 0)
    with pytest.raises(ValueError, match="inadequate measured RAM"):
        list(shards.run_sharded(plan, int))


def test_spawned_workers_exit_when_coordinator_is_killed(tmp_path):
    observed = tmp_path / "workers.json"
    script = """
import json
import os
import signal
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import psutil

from klab.corpus import shards

shards.worker_count = lambda: 2
shards.plan_host_workers = lambda *args, **kwargs: SimpleNamespace(workers=2)
plan = shards.ShardPlan("release verification", [(0,), (1,)], 0, min_parallel_bytes=0)
with closing(shards.run_sharded(plan, int, ordered=True)) as results:
    next(results)
    children = psutil.Process().children()
    Path(sys.argv[1]).write_text(json.dumps([(p.pid, p.create_time()) for p in children]))
    os.kill(os.getpid(), signal.SIGKILL)
"""
    stopped = subprocess.run(
        [sys.executable, "-c", script, str(observed)], timeout=20, check=False
    )
    assert stopped.returncode == -signal.SIGKILL
    children = json.loads(observed.read_text())
    assert len(children) >= 2
    workers = []
    for pid, created in children:
        try:
            worker = psutil.Process(pid)
            if worker.create_time() == created:
                workers.append(worker)
        except psutil.NoSuchProcess:
            continue

    def active_workers():
        for worker in workers:
            with suppress(psutil.NoSuchProcess):
                if worker.is_running() and worker.status() != psutil.STATUS_ZOMBIE:
                    yield worker

    deadline = time.monotonic() + 5
    try:
        while list(active_workers()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not list(active_workers())
    finally:
        for worker in active_workers():
            with suppress(psutil.NoSuchProcess):
                worker.terminate()
