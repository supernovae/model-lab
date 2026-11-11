"""Host-wide worker and physical-device leases.

Accelerators with a reported physical identity hold a shared host gate plus an
exclusive per-device lock.  An unknown identity holds the host gate
exclusively, so it cannot race a known identity that happens to be the same
physical device.  CPU workers only consume their own logical worker slot.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import psutil

_SAFE_LOCK = re.compile(r"[^A-Za-z0-9_.-]+")
_GLOBAL_ACCELERATOR_GATE = "accelerator-host-wide"


def boot_identity() -> str:
    """A stable identity for this boot, sufficient to reject recycled PIDs."""
    return f"{socket.gethostname()}:{int(psutil.boot_time())}"


def process_start(pid: int | None = None) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error, TypeError:
        return None


def process_matches(
    pid: int | None, expected_start: str | float | None, expected_boot: str | None
) -> bool:
    if not isinstance(pid, int) or pid <= 0 or expected_boot != boot_identity():
        return False
    try:
        expected = float(expected_start) if expected_start is not None else None
    except TypeError, ValueError:
        return False
    actual = process_start(pid)
    return actual is not None and expected is not None and abs(actual - expected) < 0.01


def _lock_root(state_dir: Path | None = None) -> Path:
    raw = state_dir or os.environ.get("KLAB_WORKER_LEASE_DIR")
    root = (
        Path(raw).expanduser()
        if raw
        else Path.home() / ".cache" / "klab" / "worker-leases"
    )
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("worker lease namespace must be a real directory")
    return root.resolve()


def lease_key(*, worker_id: str, backend: str, physical_device_id: str | None) -> str:
    """Return the exclusive slot key; unknown accelerators use the global gate."""
    if backend == "cpu":
        raw = f"worker-{worker_id}"
    elif physical_device_id:
        raw = (
            "physical-apple-metal"
            if backend in {"mps", "metal"}
            else f"physical-{physical_device_id}"
        )
    else:
        raw = _GLOBAL_ACCELERATOR_GATE
    return _SAFE_LOCK.sub("_", raw)


def _open_lock(root: Path, key: str) -> int:
    # The lock inode is deliberately stable: all writers update it in place.
    return os.open(root / f"{key}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)


@dataclass
class DeviceLease:
    key: str
    path: Path
    fd: int
    metadata: dict[str, Any]
    _gate_fd: int | None = None
    _slot_fd: int | None = None

    def heartbeat(self) -> None:
        self.metadata["heartbeat_at"] = time.time()
        encoded = json.dumps(
            self.metadata, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        os.lseek(self.fd, 0, os.SEEK_SET)
        os.ftruncate(self.fd, 0)  # Never replace the inode which carries flock.
        os.write(self.fd, encoded)
        os.fsync(self.fd)

    @property
    def inherited_fds(self) -> tuple[int, ...]:
        """Keep resource ownership alive in staging children after wrapper death."""
        return tuple(
            descriptor
            for descriptor in (self.fd, self._gate_fd, self._slot_fd)
            if descriptor is not None and descriptor >= 0
        )

    def close(self) -> None:
        # Closing releases ownership only after every inherited reference closes.
        # LOCK_UN would also unlock a still-running pilot's inherited description.
        for descriptor in self.inherited_fds:
            os.close(descriptor)
        self.fd = -1
        self._gate_fd = None
        self._slot_fd = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def acquire_lease(
    *,
    worker_id: str,
    backend: str,
    physical_device_id: str | None,
    state_dir: Path | None = None,
) -> DeviceLease | None:
    root = _lock_root(state_dir)
    key = lease_key(
        worker_id=worker_id, backend=backend, physical_device_id=physical_device_id
    )
    held: list[int] = []

    def take(lock_key: str, mode: int) -> int:
        descriptor = _open_lock(root, lock_key)
        try:
            fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
        except BaseException:
            os.close(descriptor)
            raise
        held.append(descriptor)
        return descriptor

    try:
        slot = take(
            lease_key(worker_id=worker_id, backend="cpu", physical_device_id=None),
            fcntl.LOCK_EX,
        )
        gate: int | None = None
        if backend == "cpu":
            descriptor, extra_slot = slot, None
        else:
            host_gate = take(
                _GLOBAL_ACCELERATOR_GATE,
                fcntl.LOCK_SH if physical_device_id else fcntl.LOCK_EX,
            )
            if physical_device_id:
                descriptor = take(key, fcntl.LOCK_EX)
                gate = host_gate
            else:
                # The exclusive gate already owns this inode; do not flock it twice.
                descriptor = host_gate
            extra_slot = slot
        metadata: dict[str, Any] = {
            "key": key,
            "worker_id": worker_id,
            "pid": os.getpid(),
            "process_start": process_start(),
            "boot_id": boot_identity(),
            "acquired_at": time.time(),
            "heartbeat_at": time.time(),
        }
        lease = DeviceLease(
            key, root / f"{key}.lock", descriptor, metadata, gate, extra_slot
        )
        lease.heartbeat()
    except BaseException as error:
        for descriptor in reversed(held):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        if isinstance(error, BlockingIOError):
            return None
        raise
    return lease


@dataclass
class MultiDeviceLease:
    worker_id: str
    backend: str
    leases: tuple[DeviceLease, ...]

    def heartbeat(self) -> None:
        for lease in self.leases:
            lease.heartbeat()

    @property
    def inherited_fds(self) -> tuple[int, ...]:
        return tuple(
            descriptor for lease in self.leases for descriptor in lease.inherited_fds
        )

    @property
    def metadata(self) -> list[dict[str, Any]]:
        return [dict(lease.metadata) for lease in self.leases]

    def close(self) -> None:
        for lease in reversed(self.leases):
            lease.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def acquire_multi_device_leases(
    *,
    worker_id: str,
    backend: str,
    physical_device_ids: Sequence[str],
    state_dir: Path | None = None,
) -> MultiDeviceLease | None:
    if isinstance(physical_device_ids, (str, bytes)) or not physical_device_ids:
        raise ValueError(
            "physical_device_ids must be a nonempty sequence of non-blank strings"
        )
    cleaned: list[str] = []
    for raw in physical_device_ids:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("physical_device_ids must contain only non-blank strings")
        cleaned.append(raw.strip())
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("physical_device_ids must not contain duplicates")

    acquired_by_index: dict[int, DeviceLease] = {}
    try:
        for orig_idx, phys_id in sorted(enumerate(cleaned), key=lambda item: item[1]):
            slot_worker_id = (
                worker_id if orig_idx == 0 else f"{worker_id}-dev{orig_idx}"
            )
            lease = acquire_lease(
                worker_id=slot_worker_id,
                backend=backend,
                physical_device_id=phys_id,
                state_dir=state_dir,
            )
            if lease is None:
                for held in reversed(list(acquired_by_index.values())):
                    held.close()
                return None
            acquired_by_index[orig_idx] = lease
    except BaseException:
        for held in reversed(list(acquired_by_index.values())):
            held.close()
        raise

    leases_in_caller_order = tuple(
        acquired_by_index[idx] for idx in range(len(cleaned))
    )
    return MultiDeviceLease(
        worker_id=worker_id,
        backend=backend,
        leases=leases_in_caller_order,
    )
