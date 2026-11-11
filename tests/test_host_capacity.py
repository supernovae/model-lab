from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from klab import host_capacity as host
from klab.resource_envelope import ResourceEnvelope, check_envelope

# Bound at import, before tests/conftest.py's per-test pin replaces it.
_REAL_MEASURE_MEMORY = host.measure_memory


@pytest.fixture
def capacity(monkeypatch):
    monkeypatch.setattr(
        host.psutil, "cpu_count", lambda *, logical: 24 if logical else 12
    )
    monkeypatch.setattr(
        host.psutil,
        "Process",
        lambda: SimpleNamespace(cpu_affinity=lambda: list(range(6))),
    )
    memory = SimpleNamespace(available=1000, total=2000)
    monkeypatch.setattr(host.psutil, "virtual_memory", lambda: memory)
    return memory


def test_capacity_uses_all_bounds(capacity):
    plan = host.plan_host_workers(
        "sha256", worker_memory_bytes=200, reserve_bytes=200, operator_cap=3
    )
    assert (
        plan.logical_cpus,
        plan.physical_cpus,
        plan.affinity_cpus,
        plan.workers,
    ) == (24, 12, 6, 3)
    assert (
        host.plan_host_workers(
            "sha256", worker_memory_bytes=200, reserve_bytes=200
        ).workers
        == 4
    )
    capacity.available = 200
    with pytest.raises(ValueError, match="inadequate"):
        host.plan_host_workers("sha256", worker_memory_bytes=200, reserve_bytes=200)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"worker_memory_bytes": 0, "reserve_bytes": 1},
        {"worker_memory_bytes": 1, "reserve_bytes": -1},
        {"worker_memory_bytes": True, "reserve_bytes": 1},
        {"worker_memory_bytes": 1, "reserve_bytes": 1, "operator_cap": 0},
    ],
)
def test_nonpositive_bounds_rejected(kwargs):
    with pytest.raises(ValueError):
        host.plan_host_workers("sha256", **kwargs)


def test_unknown_memory_is_sha_only(capacity, monkeypatch):
    def unavailable():
        raise OSError

    monkeypatch.setattr(host.psutil, "virtual_memory", unavailable)
    plan = host.plan_host_workers(
        "sha256", worker_memory_bytes=host.SHA_WORKER_BYTES, reserve_bytes=1
    )
    assert plan.available_bytes is None and plan.workers == 1
    for operation, size in [("encoding", 1), ("sha256", host.SHA_WORKER_BYTES + 1)]:
        with pytest.raises(ValueError, match="unavailable"):
            host.plan_host_workers(operation, worker_memory_bytes=size, reserve_bytes=1)


def test_unknown_cpu_and_affinity_fallback(capacity, monkeypatch):
    monkeypatch.setattr(host.psutil, "cpu_count", lambda **_: None)
    monkeypatch.setattr(host.psutil, "Process", lambda: object())
    assert (
        host.plan_host_workers("sha256", worker_memory_bytes=1, reserve_bytes=1).workers
        == 1
    )


def test_ordered_window_and_cleanup(capacity):
    plan = host.plan_host_workers(
        "sha256", worker_memory_bytes=1, reserve_bytes=1, operator_cap=2
    )
    second = threading.Event()
    consumed = []

    def items():
        for number in range(5):
            consumed.append(number)
            yield number

    def worker(number):
        if number == 0:
            assert second.wait(5)
        elif number == 1:
            second.set()
        return number * 2

    iterator = host.run_ordered(items(), worker, plan=plan)
    assert next(iterator) == 0
    assert consumed == [0, 1]
    assert list(iterator) == [2, 4, 6, 8]
    completed = []

    def failed(number):
        if number == 0:
            raise ValueError("failure")
        completed.append(number)

    with pytest.raises(ValueError, match="failure"):
        list(host.run_ordered(range(10), failed, plan=plan))
    assert set(completed) <= {1}
    assert not [t for t in threading.enumerate() if t.name.startswith("sha256_")]


def test_native_sha_and_canonical_bytes_match(capacity, tmp_path):
    paths = []
    for n in range(8):
        path = tmp_path / f"{n}.bin"
        path.write_bytes(bytes([n]) * 4096)
        paths.append(path)
    plan = host.plan_host_workers("sha256", worker_memory_bytes=1, reserve_bytes=1)

    def worker(path):
        return (path.name, hashlib.sha256(path.read_bytes()).hexdigest())

    from klab.canonical import canonical_json

    reference = canonical_json(
        list(host.run_ordered(paths, worker, plan=replace(plan, workers=1)))
    )
    for workers in (2, plan.workers):
        assert (
            canonical_json(
                list(
                    host.run_ordered(paths, worker, plan=replace(plan, workers=workers))
                )
            )
            == reference
        )


def test_isa_parsing_and_fallback(monkeypatch):
    assert (
        host._parse_isa("x86_64", "flags: avx2 sha_ni aes\nflags: avx2 aes")["sha_ni"]
        is False
    )
    assert host._parse_isa("x86_64", "flags: sse4_1 sha_ni")["sha_ni"] is True
    assert (
        host._parse_isa("aarch64", "Features: asimd sha2 aes crc32")["asimd/neon"]
        is True
    )
    assert set(host._parse_isa("arm64", None).values()) == {"unknown"}
    assert host._parse_isa("riscv64", None) == {"capabilities": "unknown"}
    monkeypatch.setattr(host.platform, "system", lambda: "unsupported")
    assert host.hardware_observation()["probe"] == "unknown"


def test_suite_ram_fixture_is_consistent_on_a_physical_host_above_480_gib(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reserves from ``total`` and checks on ``available`` share one observation.

    A 512 GiB physical host once gave callers a 51.2 GiB reserve (10% of the
    real total) against the fixture's 48 GiB available, so planning raised
    "inadequate measured RAM after reserve". The physical reading is faked
    below psutil's public API, so the suite's deterministic fixture stays active.
    """
    import psutil
    from tokenizers import Tokenizer, models

    from klab.corpus import large_build
    from klab.data import encoding

    gib = 1024**3
    physical = psutil._psplatform.virtual_memory
    monkeypatch.setattr(
        psutil._psplatform,
        "virtual_memory",
        lambda: physical()._replace(total=512 * gib, available=500 * gib),
    )
    assert psutil.virtual_memory().total == 512 * gib
    observed = host.measure_memory()
    assert (observed.total, observed.available) == (64 * gib, 48 * gib)

    # Corpus shard planning: 10% of total as reserve, two workers fit.
    assert large_build._process_shard_workers(4) == 2

    # Tokenizer preparation: same reserve rule; stop before the child spawns.
    plans = []
    real_plan = encoding.plan_host_workers

    def recording(*args, **kwargs):
        plans.append((kwargs["reserve_bytes"], real_plan(*args, **kwargs)))
        return plans[-1][1]

    class Spawned(Exception):
        pass

    def no_spawn(*_args, **_kwargs):
        raise Spawned

    monkeypatch.setattr(encoding, "plan_host_workers", recording)
    monkeypatch.setattr(encoding.subprocess, "Popen", no_spawn)
    with pytest.raises(Spawned):
        encoding.PreparationEncoder(
            Tokenizer(models.BPE()), tmp_path / "workspace", max_workers=None
        )
    ((reserve, plan),) = plans
    assert reserve == 64 * gib // 10
    assert plan.workers >= 1


def _child_memory(environment: dict[str, str]) -> list[str]:
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import psutil; from klab.host_capacity import measure_memory;"
                "print(measure_memory().total, psutil.virtual_memory().total)"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    return child.stdout.split()


def test_child_processes_inherit_the_suite_ram_fixture() -> None:
    """Fresh interpreters launched by tests plan against the same fixed host."""
    measured, _ = _child_memory(os.environ.copy())
    assert measured == str(64 * 1024**3)


def test_ram_env_cannot_change_production_measurement() -> None:
    """Outside the tests-only bootstrap the variable is inert in a real process."""
    environment = os.environ.copy()
    bootstrap = str(Path(__file__).with_name("child_bootstrap"))
    environment["PYTHONPATH"] = os.pathsep.join(
        entry
        for entry in environment.get("PYTHONPATH", "").split(os.pathsep)
        if entry and entry != bootstrap
    )
    environment["KLAB_TEST_HOST_RAM"] = f"{1 << 50}:{1 << 50}"
    measured, real = _child_memory(environment)
    assert measured == real != str(1 << 50)


@pytest.mark.parametrize(
    ("limit", "rss"),
    [
        ({"min_available_ram_bytes": 8 * 1024**3}, None),
        ({"max_host_memory_fraction": 0.5}, 3 * 1024**3 // 2),
    ],
    ids=["min-available", "host-fraction"],
)
def test_ram_env_cannot_bypass_low_memory_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: dict, rss: int | None
) -> None:
    """A 1 GiB-available / 2 GiB host stays low-memory whatever the env says."""
    monkeypatch.setenv("KLAB_TEST_HOST_RAM", f"{64 * 1024**3}:{48 * 1024**3}")
    monkeypatch.setattr(host, "measure_memory", _REAL_MEASURE_MEMORY)
    small = SimpleNamespace(total=2 * 1024**3, available=1024**3)
    monkeypatch.setattr(host.psutil, "virtual_memory", lambda: small)
    assert host.measure_memory() is small
    envelope = ResourceEnvelope.model_validate(
        {"resource_envelope_version": 1, **limit}
    )
    with pytest.raises(ValueError, match=next(iter(limit))):
        check_envelope(envelope, workspace=tmp_path, rss_bytes=rss)


def test_corpus_encoder_budget_keeps_preparation_pool_cap(capacity, tmp_path):
    from tokenizers import Tokenizer, models

    from klab.data.encoding import tokenizer_work_plan
    from klab.data.preparation_workers import PreparationWorkers

    capacity.total = 64 * 1024**3
    capacity.available = 48 * 1024**3
    # Six available CPUs: corpus assembly can use its explicit encoder budget.
    assert tokenizer_work_plan(6).workers == 6
    assert tokenizer_work_plan(None).workers == 4
    tokenizer = Tokenizer(models.BPE())
    for envelope in (
        None,
        ResourceEnvelope(resource_envelope_version=1),
        ResourceEnvelope(
            resource_envelope_version=1, max_workers=6, preparation_shard_workers=6
        ),
    ):
        pool = PreparationWorkers(
            tokenizer, tmp_path, envelope=envelope, parallel_supported=True
        )
        assert pool.plan.workers == 4
        assert sum(pool.threads) == 4
        assert pool.count <= 4
