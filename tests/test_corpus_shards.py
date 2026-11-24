"""Capacity and backpressure guarantees of the shared Corpus runner."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

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
    with (
        verification_settings(workers=2),
        closing(shards.run_sharded(plan, int, ordered=True)) as results,
    ):
        assert next(results) == 0
        assert 1 <= len(reads) <= 2
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
