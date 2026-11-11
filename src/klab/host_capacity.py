"""Capacity-bounded independent native work; observations are never identity inputs."""

from __future__ import annotations

import hashlib
import platform
import ssl
import subprocess
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import psutil

T = TypeVar("T")
R = TypeVar("R")
_OBSERVATION_ERRORS = (psutil.Error, OSError, AttributeError, ValueError, TypeError)
SHA_WORKER_BYTES = (
    8 * 1024 * 1024
)  # 1 MiB native hasher buffer plus bounded thread overhead


@dataclass(frozen=True)
class HostWorkPlan:
    operation: str
    logical_cpus: int | None
    physical_cpus: int | None
    affinity_cpus: int | None
    available_bytes: int | None
    total_bytes: int | None
    reserve_bytes: int
    worker_memory_bytes: int
    operator_cap: int | None
    workers: int


def measure_memory() -> Any:
    """The single host-RAM observation (``psutil.virtual_memory()``).

    Every Model Lab reader of host total/available RAM goes through here, so
    reserves derived from ``total`` and capacity checks against ``available``
    always come from one consistent observation. The test suite pins it in
    tests/conftest.py.
    """
    return psutil.virtual_memory()


def _positive(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def plan_host_workers(
    operation: str,
    *,
    worker_memory_bytes: int,
    reserve_bytes: int,
    operator_cap: int | None = None,
) -> HostWorkPlan:
    """Reject unsafe bounds before submitting anything; unknown SHA capacity is serial."""
    if not isinstance(operation, str) or not operation:
        raise ValueError("operation must be nonempty")
    if _positive(worker_memory_bytes) is None or _positive(reserve_bytes) is None:
        raise ValueError("worker memory and reserve must be positive integer bounds")
    if operator_cap is not None and _positive(operator_cap) is None:
        raise ValueError("operator cap must be a positive integer")
    try:
        logical = _positive(psutil.cpu_count(logical=True))
        physical = _positive(psutil.cpu_count(logical=False))
    except _OBSERVATION_ERRORS:
        logical = physical = None
    try:
        affinity = _positive(len(psutil.Process().cpu_affinity()))
    except _OBSERVATION_ERRORS:
        affinity = None
    try:
        memory = measure_memory()
        available, total = _positive(memory.available), _positive(memory.total)
        if memory.available == 0:
            available = 0
    except _OBSERVATION_ERRORS:
        available = total = None
    known_cpus = [value for value in (affinity, physical or logical) if value]
    cpu_bound = min(known_cpus) if known_cpus else 1
    if available is None:
        if operation != "sha256" or worker_memory_bytes > SHA_WORKER_BYTES:
            raise ValueError("measured RAM unavailable for memory-heavy host work")
        workers = 1
    else:
        memory_bound = (available - reserve_bytes) // worker_memory_bytes
        if memory_bound < 1:
            raise ValueError("inadequate measured RAM after reserve")
        workers = min(cpu_bound, memory_bound, operator_cap or cpu_bound)
    return HostWorkPlan(
        operation,
        logical,
        physical,
        affinity,
        available,
        total,
        reserve_bytes,
        worker_memory_bytes,
        operator_cap,
        workers,
    )


def sha_work_plan(*, operator_cap: int = 1) -> HostWorkPlan:
    """Serial default: medium end-to-end cold SHA did not demonstrate parallel ROI."""
    try:
        total = _positive(measure_memory().total) or 0
    except _OBSERVATION_ERRORS:
        total = 0
    return plan_host_workers(
        "sha256",
        worker_memory_bytes=SHA_WORKER_BYTES,
        reserve_bytes=max(1024**3, total // 10),
        operator_cap=operator_cap,
    )


def run_ordered[T, R](
    items: Iterable[T], worker: Callable[[T], R], *, plan: HostWorkPlan
) -> Iterator[R]:
    """At most workers outstanding; errors cancel queued work and join running work.

    Input order must already be canonical. Closing the iterator also joins workers.
    No Python parsing, GPU execution, or non-independent mutable tasks belong here.
    """
    if _positive(plan.workers) is None:
        raise ValueError("worker count must be positive")
    if plan.workers == 1:
        for item in items:
            yield worker(item)
        return
    iterator = iter(items)
    executor = ThreadPoolExecutor(
        max_workers=plan.workers, thread_name_prefix=plan.operation
    )
    pending = deque()
    try:
        for _ in range(plan.workers):
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append(executor.submit(worker, item))
        while pending:
            result = pending.popleft().result()
            yield result
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(worker, item))
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _parse_isa(machine: str, raw: str | None) -> dict[str, bool | str]:
    machine = machine.lower()
    if machine in {"x86_64", "amd64", "i386", "i686"}:
        names = {
            name: {name}
            for name in ("sse4_1", "sse4_2", "avx", "avx2", "avx512f", "sha_ni", "aes")
        }
        names["sha_ni"] |= {"sha", "sha256"}
    elif machine in {"arm64", "aarch64"} or machine.startswith("arm"):
        names = {
            "asimd/neon": {"asimd", "neon"},
            "sha2": {"sha2", "sha256"},
            "aes": {"aes"},
            "crc32": {"crc32"},
        }
    else:
        return {"capabilities": "unknown"}
    sets = []
    if raw is not None:
        for line in raw.lower().splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {
                "flags",
                "features",
                "machdep.cpu.features",
                "machdep.cpu.leaf7_features",
            }:
                sets.append(set(value.replace(".", "_").split()))
    flags = set.intersection(*sets) if sets else None
    return {
        name: bool(flags & aliases) if flags is not None else "unknown"
        for name, aliases in names.items()
    }


def hardware_observation() -> dict[str, object]:
    """Bounded safe capability probes; presence alone is not measured acceleration."""
    machine = platform.machine()
    raw = None
    probe = "unknown"
    try:
        if platform.system() == "Linux":
            with Path("/proc/cpuinfo").open() as stream:
                raw = stream.read(64 * 1024)
            probe = "bounded /proc/cpuinfo"
        elif platform.system() == "Darwin":
            keys = (
                ["machdep.cpu.features", "machdep.cpu.leaf7_features"]
                if machine.lower() in {"x86_64", "amd64"}
                else [
                    "hw.optional.neon",
                    "hw.optional.arm.FEAT_SHA256",
                    "hw.optional.arm.FEAT_AES",
                    "hw.optional.armv8_crc32",
                ]
            )
            result = subprocess.run(
                ["sysctl", *keys],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and len(result.stdout) <= 64 * 1024:
                raw = result.stdout
                if machine.lower() in {"arm64", "aarch64"}:
                    aliases = dict(
                        zip(keys, ["neon", "sha2", "aes", "crc32"], strict=True)
                    )
                    raw = "Features: " + " ".join(
                        aliases[key] for key in keys if f"{key}: 1" in raw
                    )
                probe = "bounded sysctl"
    except OSError, subprocess.SubprocessError:
        pass
    return {
        "machine": machine,
        "isa": _parse_isa(machine, raw),
        "probe": probe,
        "openssl": ssl.OPENSSL_VERSION,
        "hashlib_sha256_module": hashlib.sha256.__module__,
        "hashlib_algorithms_guaranteed": sorted(hashlib.algorithms_guaranteed),
    }
