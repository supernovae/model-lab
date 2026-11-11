"""Optional, non-gating phase observations; no scientific or artifact identity inputs."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal

import psutil

from klab import host_capacity

HostKind = Literal["verification_bound", "copy_bound", "serialization_bound"]
CacheEvent = Literal["hit", "miss"]

_LOGGER = logging.getLogger(__name__)


def _counter(value: object) -> int | None:
    return (
        int(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
        else None
    )


def _snapshot() -> dict[str, object]:
    """Endpoint sample, not a lifetime peak; exited children cannot be recovered."""
    try:
        parent = psutil.Process()
        processes = [parent, *parent.children(recursive=True)]
    except OSError, psutil.Error:
        processes = []
    cpu: dict[tuple[int, float], tuple[float, float]] = {}
    io: dict[tuple[int, float], tuple[int, int]] = {}
    rss = swap = 0
    rss_complete = swap_complete = bool(processes)
    for process in processes:
        try:
            identity = (process.pid, process.create_time())
            times = process.cpu_times()
            cpu[identity] = (float(times.user), float(times.system))
        except OSError, psutil.Error, AttributeError:
            rss_complete = swap_complete = False
            continue
        try:
            info = process.memory_info()
            rss += int(info.rss)
        except OSError, psutil.Error, AttributeError:
            rss_complete = False
        try:
            value = _counter(getattr(process.memory_full_info(), "swap", None))
            if value is None:
                swap_complete = False
            else:
                swap += value
        except OSError, psutil.Error, AttributeError:
            swap_complete = False
        try:
            counters = process.io_counters()
            read_bytes = _counter(getattr(counters, "read_bytes", None))
            write_bytes = _counter(getattr(counters, "write_bytes", None))
            if read_bytes is not None and write_bytes is not None:
                io[identity] = (read_bytes, write_bytes)
        except OSError, psutil.Error, AttributeError, NotImplementedError:
            pass
    try:
        memory = host_capacity.measure_memory()
        available, total = int(memory.available), int(memory.total)
    except OSError, psutil.Error, AttributeError:
        available = total = None
    return {
        "cpu": cpu,
        "io": io,
        "rss": rss if rss_complete else None,
        "swap": swap if swap_complete else None,
        "available": available,
        "total": total,
    }


def _safe_snapshot() -> dict[str, object]:
    try:
        return _snapshot()
    except Exception:
        _LOGGER.exception("Optional bottleneck process counters unavailable")
        return {
            "cpu": {},
            "io": {},
            "rss": None,
            "swap": None,
            "available": None,
            "total": None,
        }


def _delta(
    before: dict[tuple[int, float], tuple[int | float, int | float]],
    after: dict[tuple[int, float], tuple[int | float, int | float]],
) -> tuple[int | float | None, int | float | None, int]:
    common = before.keys() & after.keys()
    if not common:
        return None, None, 0
    first = sum(after[key][0] - before[key][0] for key in common)
    second = sum(after[key][1] - before[key][1] for key in common)
    if first < 0 or second < 0:
        return None, None, len(common)
    return first, second, len(common)


def _classify(
    *,
    seconds: float,
    cpu_seconds: float | None,
    io_bytes: float | None,
    swap_bytes: int | None,
    available_bytes: int | None,
    total_bytes: int | None,
    accelerator_utilization_percent: float | None,
    host_kind: HostKind | None,
    cache_event: CacheEvent | None,
) -> tuple[str, str]:
    activity = (cpu_seconds is not None and cpu_seconds > 0) or (
        io_bytes is not None and io_bytes > 0
    )
    if (swap_bytes is not None and swap_bytes > 0) or (
        available_bytes is not None
        and total_bytes is not None
        and total_bytes > 0
        and available_bytes / total_bytes < 0.05
    ):
        bottleneck = "memory_pressure"
    elif (
        accelerator_utilization_percent is not None
        and accelerator_utilization_percent >= 80
    ):
        bottleneck = "accelerator"
    elif (
        accelerator_utilization_percent is not None
        and accelerator_utilization_percent <= 20
        and activity
    ):
        bottleneck = "input_host"
    else:
        bottleneck = "unknown"
    if seconds <= 0:
        host = "unknown"
    elif cache_event is not None and cpu_seconds is not None:
        # A cache event is supplied by the caller only after observing that branch.
        host = f"cache_{cache_event}"
    elif host_kind is not None and activity:
        host = host_kind
    elif cpu_seconds is not None and cpu_seconds / seconds >= 0.8:
        host = "cpu_bound"
    elif (
        cpu_seconds is not None
        and io_bytes is not None
        and io_bytes > 0
        and cpu_seconds / seconds < 0.5
    ):
        host = "io_bound"
    else:
        host = "unknown"
    return bottleneck, host


class BottleneckObserver:
    """Collect endpoint deltas and sampled process-tree memory for opted-in phases.

    CPU/I/O deltas include only processes alive at both endpoints, so brief child
    work can be missed. RSS/swap maxima are sampled at phase boundaries, not
    guaranteed lifetime peaks. Missing process I/O and accelerator probes are null.
    """

    def __init__(
        self, *, accelerator_probe: Callable[[], float | None] | None = None
    ) -> None:
        self.records: list[dict[str, object]] = []
        self._accelerator_probe = accelerator_probe

    def _accelerator(self) -> float | None:
        if self._accelerator_probe is None:
            return None
        try:
            value = self._accelerator_probe()
        except Exception:
            _LOGGER.exception("Optional accelerator utilization probe unavailable")
            return None
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 <= value <= 100
        ):
            return float(value)
        return None

    @contextmanager
    def phase(
        self,
        name: str,
        *,
        host_kind: HostKind | None = None,
        cache_event: CacheEvent | None = None,
    ) -> Iterator[None]:
        if host_kind not in (
            None,
            "verification_bound",
            "copy_bound",
            "serialization_bound",
        ):
            raise ValueError("unrecognized host phase provenance")
        if cache_event not in (None, "hit", "miss"):
            raise ValueError("unrecognized cache event provenance")
        try:
            before = _safe_snapshot()
            accelerator_before = self._accelerator()
            started = time.monotonic()
        except Exception:
            _LOGGER.warning(
                "Optional phase observation setup unavailable", exc_info=True
            )
            yield
            return
        try:
            yield
        finally:
            try:
                seconds = max(time.monotonic() - started, 0.0)
                after = _safe_snapshot()
                accelerator_after = self._accelerator()
                user, system, cpu_count = _delta(before["cpu"], after["cpu"])
                read, written, io_count = _delta(before["io"], after["io"])
                cpu_seconds = (
                    user + system if user is not None and system is not None else None
                )
                io_bytes = (
                    read + written if read is not None and written is not None else None
                )
                rss_samples = [
                    value
                    for value in (before["rss"], after["rss"])
                    if value is not None
                ]
                swap_samples = [
                    value
                    for value in (before["swap"], after["swap"])
                    if value is not None
                ]
                utilization = (
                    max(accelerator_before, accelerator_after)
                    if accelerator_before is not None and accelerator_after is not None
                    else None
                )
                available = after["available"]
                total = after["total"]
                bottleneck, host = _classify(
                    seconds=seconds,
                    cpu_seconds=cpu_seconds,
                    io_bytes=io_bytes,
                    swap_bytes=max(swap_samples) if swap_samples else None,
                    available_bytes=available,
                    total_bytes=total,
                    accelerator_utilization_percent=utilization,
                    host_kind=host_kind,
                    cache_event=cache_event,
                )
                self.records.append(
                    {
                        "phase": name,
                        "elapsed_seconds": seconds,
                        "cpu_user_seconds": user,
                        "cpu_system_seconds": system,
                        "cpu_matched_processes": cpu_count,
                        "process_read_bytes": int(read) if read is not None else None,
                        "process_write_bytes": int(written)
                        if written is not None
                        else None,
                        "io_matched_processes": io_count,
                        "observed_tree_rss_max_bytes": max(rss_samples)
                        if rss_samples
                        else None,
                        "observed_tree_swap_max_bytes": max(swap_samples)
                        if swap_samples
                        else None,
                        "host_available_ram_bytes": available,
                        "accelerator_utilization_percent": utilization,
                        "accelerator_bottleneck": bottleneck,
                        "host_bottleneck": host,
                        "host_kind_provenance": host_kind,
                        "cache_event_provenance": cache_event,
                        "memory_sampling": "phase_endpoints_not_lifetime_peak",
                        "cpu_io_coverage": "surviving_processes_at_both_endpoints_only",
                    }
                )
            except Exception:
                _LOGGER.warning(
                    "Optional phase observations unavailable", exc_info=True
                )
