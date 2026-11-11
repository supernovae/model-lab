"""Operation-scoped Linux process monitoring; no scientific identity or retry semantics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import psutil
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from klab import host_capacity


def _private_output(path: str, flags: int) -> int:
    """Create private evidence without changing the process-wide umask."""
    return os.open(path, flags, 0o600)


def _private_parents(path: Path) -> None:
    """Create missing ancestors privately; never repair existing directories."""
    try:
        path.mkdir(mode=0o700)
    except FileNotFoundError:
        _private_parents(path.parent)
        path.mkdir(mode=0o700)
    except FileExistsError:
        if not path.is_dir():
            raise


_NEW_POLICY_FIELDS = (
    "max_device_memory_bytes",
    "expected_device_uuid",
    "max_added_workspace_bytes",
    "max_added_workspace_inodes",
    "max_wall_seconds",
)


def _omit_absent(data: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    for field in fields:
        if data.get(field) is None:
            data.pop(field, None)
    return data


class MonitorPolicy(BaseModel):
    """Independent operational guard policy (not a ResourceEnvelope or run config)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    monitor_policy_version: Literal[1]
    max_tree_rss_bytes: int | None = Field(default=None, strict=True, gt=0)
    max_tree_swap_bytes: int | None = Field(default=None, strict=True, ge=0)
    min_host_available_ram_bytes: int | None = Field(default=None, strict=True, ge=0)
    min_host_free_swap_bytes: int | None = Field(default=None, strict=True, ge=0)
    min_disk_free_bytes: int | None = Field(default=None, strict=True, ge=0)
    min_disk_free_inodes: int | None = Field(default=None, strict=True, ge=0)
    min_projected_disk_free_bytes: int | None = Field(default=None, strict=True, ge=0)
    min_projected_disk_free_inodes: int | None = Field(default=None, strict=True, ge=0)
    interval_seconds: float = Field(default=1.0, gt=0, le=2)
    termination_grace_seconds: float = Field(default=5.0, ge=0)
    max_device_memory_bytes: int | None = Field(default=None, strict=True, gt=0)
    expected_device_uuid: str | None = None
    max_added_workspace_bytes: int | None = Field(default=None, strict=True, ge=0)
    max_added_workspace_inodes: int | None = Field(default=None, strict=True, ge=0)
    max_wall_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_device_identity(self) -> MonitorPolicy:
        if not math.isfinite(self.termination_grace_seconds):
            raise ValueError("monitor termination grace must be finite")
        if (self.max_device_memory_bytes is None) != (
            self.expected_device_uuid is None
        ):
            raise ValueError(
                "device memory limit and expected UUID must be supplied together"
            )
        if (
            self.expected_device_uuid is not None
            and not self.expected_device_uuid.strip()
        ):
            raise ValueError("expected device UUID must be nonempty")
        if self.max_wall_seconds is not None and not math.isfinite(
            self.max_wall_seconds
        ):
            raise ValueError("monitor wall limit must be finite")
        return self

    @model_serializer(mode="wrap")
    def serialize_compatibly(self, handler: Any) -> dict[str, Any]:
        return _omit_absent(handler(self), _NEW_POLICY_FIELDS)

    @field_validator("monitor_policy_version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("monitor policy version must be integer 1")
        return value


class ProcessIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    pid: int
    create_time: float


class MonitorSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    monitor_record_version: Literal[1] = 1
    kind: Literal["sample"] = "sample"
    elapsed_seconds: float
    processes: list[ProcessIdentity]
    tree_rss_bytes: int | None
    tree_swap_bytes: int | None
    host_available_ram_bytes: int | None
    host_free_swap_bytes: int | None
    host_used_swap_bytes: int | None
    disk_free_bytes: int | None
    disk_free_inodes: int | None
    projected_disk_free_bytes: int | None
    projected_disk_free_inodes: int | None
    violations: list[str]
    device_memory_bytes: int | None = None
    added_workspace_bytes: int | None = None
    added_workspace_inodes: int | None = None
    requested_device_memory: bool = False
    requested_added_workspace: bool = False

    @model_validator(mode="before")
    @classmethod
    def retain_explicit_missing_metrics(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = value.copy()
            if (
                "requested_device_memory" not in value
                and "device_memory_bytes" in value
            ):
                value["requested_device_memory"] = True
            if "requested_added_workspace" not in value and (
                "added_workspace_bytes" in value or "added_workspace_inodes" in value
            ):
                value["requested_added_workspace"] = True
        return value

    @model_serializer(mode="wrap")
    def serialize_compatibly(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        data.pop("requested_device_memory")
        data.pop("requested_added_workspace")
        if not self.requested_device_memory:
            _omit_absent(data, ("device_memory_bytes",))
        if not self.requested_added_workspace:
            _omit_absent(data, ("added_workspace_bytes", "added_workspace_inodes"))
        return data


class MonitorLaunch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    monitor_record_version: Literal[1] = 1
    kind: Literal["launch"] = "launch"
    command: list[str]
    root: ProcessIdentity
    policy: MonitorPolicy
    workspace: str
    cwd: str
    reserved_bytes: int
    reserved_inodes: int
    source_sha256: str
    policy_sha256: str
    source_path: str
    policy_path: str
    stdout_path: str
    stderr_path: str
    events_path: str
    completion_path: str
    baseline_sha256: str | None = None
    owned_source_sha256: str | None = None

    @model_serializer(mode="wrap")
    def serialize_compatibly(self, handler: Any) -> dict[str, Any]:
        return _omit_absent(handler(self), ("baseline_sha256", "owned_source_sha256"))


class MonitorCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    monitor_record_version: Literal[1] = 1
    kind: Literal["completion"] = "completion"
    launch: MonitorLaunch
    returncode: int | None
    status: Literal["COMPLETE", "FAILED", "VIOLATED", "MONITOR_ERROR"]
    violations: list[str]
    peak_tree_rss_bytes: int | None
    peak_tree_swap_bytes: int | None
    min_host_available_ram_bytes: int | None
    min_host_free_swap_bytes: int | None
    min_disk_free_bytes: int | None
    min_disk_free_inodes: int | None
    min_projected_disk_free_bytes: int | None
    min_projected_disk_free_inodes: int | None
    elapsed_seconds: float
    sample_count: int
    peak_device_memory_bytes: int | None = None
    peak_added_workspace_bytes: int | None = None
    peak_added_workspace_inodes: int | None = None
    final_added_workspace_bytes: int | None = None
    final_added_workspace_inodes: int | None = None

    @model_serializer(mode="wrap")
    def serialize_compatibly(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if self.launch.policy.max_device_memory_bytes is None:
            _omit_absent(data, ("peak_device_memory_bytes",))
        if (
            self.launch.policy.max_added_workspace_bytes is None
            and self.launch.policy.max_added_workspace_inodes is None
            and self.launch.baseline_sha256 is None
        ):
            _omit_absent(
                data,
                (
                    "peak_added_workspace_bytes",
                    "peak_added_workspace_inodes",
                    "final_added_workspace_bytes",
                    "final_added_workspace_inodes",
                ),
            )
        return data


class WorkspaceBaseline(BaseModel):
    """Immutable, content-addressed common-root baseline shared by all phases."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    baseline_version: Literal[1] = 1
    root: str
    device: int
    root_inode: int
    apparent_bytes: int = Field(ge=0)
    inodes: int = Field(ge=1)
    free_bytes: int = Field(ge=0)
    free_inodes: int = Field(ge=0)
    sha256: str

    @model_validator(mode="after")
    def check_digest(self) -> WorkspaceBaseline:
        payload = self.model_dump(exclude={"sha256"})
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.sha256 != expected:
            raise ValueError("workspace baseline digest mismatch")
        return self


def _sample_tree_once(root: Path, deadline: float) -> tuple[int, int]:
    root_stat = os.stat(root, follow_symlinks=False)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("workspace root is not a directory")
    device = root_stat.st_dev
    stack = [root]
    seen_links: set[tuple[int, int]] = set()
    apparent_bytes = entries = 0
    while stack:
        if time.monotonic() >= deadline:
            raise TimeoutError("workspace sampling deadline exhausted")
        path = stack.pop()
        info = os.stat(path, follow_symlinks=False)
        # Mount boundaries are directories. Overlayfs without xino reports
        # files under a different st_dev than their directories, so comparing
        # every entry's device would silently drop all regular files.
        if stat.S_ISDIR(info.st_mode) and info.st_dev != device:
            continue
        entries += 1
        identity = (info.st_dev, info.st_ino)
        if not stat.S_ISDIR(info.st_mode) and (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink <= 1
            or identity not in seen_links
        ):
            apparent_bytes += info.st_size
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            seen_links.add(identity)
        if stat.S_ISDIR(info.st_mode):
            with os.scandir(path) as listing:
                stack.extend(Path(item.path) for item in listing)
    return apparent_bytes, entries


def sample_workspace_tree(root: Path, *, seconds: float = 4.0) -> tuple[int, int]:
    """Retry only vanished entries; never return an earlier partial or stale count."""
    if not 0 < seconds <= 5:
        raise ValueError("sampling deadline must be in (0, 5] seconds")
    deadline = time.monotonic() + seconds
    while True:
        try:
            return _sample_tree_once(root, deadline)
        except FileNotFoundError as error:
            if time.monotonic() >= deadline:
                raise TimeoutError("workspace sampling deadline exhausted") from error
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


def capture_workspace_baseline(
    root: Path, path: Path, *, seconds: float = 4.0
) -> WorkspaceBaseline:
    """Exclusively persist one baseline before any monitored work begins."""
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("workspace root is not a directory")
    root_identity = resolved.stat()
    used_bytes, used_inodes = sample_workspace_tree(resolved, seconds=seconds)
    free_bytes, free_inodes = _storage(resolved)
    if (resolved.stat().st_dev, resolved.stat().st_ino) != (
        root_identity.st_dev,
        root_identity.st_ino,
    ):
        raise RuntimeError("workspace root changed during baseline capture")
    payload = {
        "baseline_version": 1,
        "root": str(resolved),
        "device": root_identity.st_dev,
        "root_inode": root_identity.st_ino,
        "apparent_bytes": used_bytes,
        "inodes": used_inodes,
        "free_bytes": free_bytes,
        "free_inodes": free_inodes,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    baseline = WorkspaceBaseline.model_validate({**payload, "sha256": digest})
    with open(path, "x", encoding="utf-8", opener=_private_output) as stream:
        stream.write(baseline.model_dump_json() + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return baseline


def load_workspace_baseline(path: Path, root: Path) -> WorkspaceBaseline:
    baseline = WorkspaceBaseline.model_validate_json(path.read_bytes())
    resolved = root.resolve(strict=True)
    root_identity = resolved.stat()
    if (
        baseline.root != str(resolved)
        or baseline.device != root_identity.st_dev
        or baseline.root_inode != root_identity.st_ino
    ):
        raise ValueError("workspace baseline root or filesystem differs")
    return baseline


def _read_device_memory_direct(expected_uuid: str) -> int:
    """Direct AMD SMI read, executed only in a bounded child process."""
    import amdsmi

    amdsmi.amdsmi_init()
    try:
        handles = amdsmi.amdsmi_get_processor_handles()
        if len(handles) != 1:
            raise RuntimeError(f"expected one GPU, found {len(handles)}")
        handle = handles[0]
        if amdsmi.amdsmi_get_gpu_device_uuid(handle) != expected_uuid:
            raise RuntimeError("GPU UUID mismatch")
        used = amdsmi.amdsmi_get_gpu_memory_usage(handle, amdsmi.AmdSmiMemoryType.VRAM)
        total = amdsmi.amdsmi_get_gpu_memory_total(handle, amdsmi.AmdSmiMemoryType.VRAM)
        if (
            type(used) is not int
            or type(total) is not int
            or total <= 0
            or not 0 <= used <= total
        ):
            raise RuntimeError("invalid VRAM reading")
        return used
    finally:
        amdsmi.amdsmi_shut_down()


def read_device_memory_bytes(expected_uuid: str) -> int:
    """Read UUID-bound whole-device VRAM with a strict subprocess timeout."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "klab.operational_monitor",
                "--device-memory-probe",
                expected_uuid,
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=True,
        )
        value = result.stdout.strip()
        if not value.isascii() or not value.isdecimal():
            raise ValueError("invalid device sensor output")
        return int(value)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
        ValueError,
    ) as error:
        raise RuntimeError(f"device memory sensor unavailable: {error}") from error


def _parse_policy(raw: bytes, path: Path) -> MonitorPolicy:
    try:
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise TypeError("policy must be a YAML mapping")
        return MonitorPolicy.model_validate(data)
    except (yaml.YAMLError, ValidationError, TypeError) as error:
        raise ValueError(f"invalid monitor policy {path}: {error}") from error


def load_monitor_policy(path: Path) -> MonitorPolicy:
    try:
        return _parse_policy(path.read_bytes(), path)
    except OSError as error:
        raise ValueError(f"cannot read monitor policy {path}: {error}") from error


def _identity(process: psutil.Process) -> ProcessIdentity:
    return ProcessIdentity(pid=process.pid, create_time=process.create_time())


def _owned(
    root: ProcessIdentity, known: dict[int, ProcessIdentity]
) -> list[psutil.Process]:
    """Discover living descendants by parent linkage; retain observed orphans."""
    snapshot: dict[int, tuple[psutil.Process, ProcessIdentity, int]] = {}
    for process in psutil.process_iter(["ppid"]):
        try:
            identity = _identity(process)
            if process.pid != root.pid and process.status() == psutil.STATUS_ZOMBIE:
                continue
            snapshot[process.pid] = (process, identity, process.ppid())
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as error:
            if process.pid in known:
                raise RuntimeError(
                    f"owned process attribution unavailable: {process.pid}"
                ) from error
            continue
    root_entry = snapshot.get(root.pid)
    if root_entry is not None and root_entry[1] != root:
        raise RuntimeError("root PID was reused")
    reused = {
        pid
        for pid, entry in snapshot.items()
        if pid in known and known[pid] != entry[1]
    }
    discovered = {pid for pid, entry in snapshot.items() if known.get(pid) == entry[1]}
    if root_entry is not None:
        discovered.add(root.pid)
    changed = True
    while changed:
        before = len(discovered)
        for pid, (_, identity, ppid) in snapshot.items():
            if pid not in reused and ppid in discovered and pid not in discovered:
                discovered.add(pid)
                known[pid] = identity
        changed = len(discovered) != before
    for pid in reused:
        # Never adopt a recycled PID, even if it is now another descendant.
        del known[pid]
    return [snapshot[pid][0] for pid in sorted(discovered)]


def _swap_bytes(process: psutil.Process) -> int:
    try:
        swap = process.memory_full_info().swap
        if swap is not None:
            return int(swap)
    except psutil.NoSuchProcess:
        raise
    except AttributeError, psutil.Error, OSError:
        pass
    try:
        for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
            if line.startswith("VmSwap:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError) as error:
        raise RuntimeError(
            f"swap attribution unavailable for PID {process.pid}"
        ) from error
    raise RuntimeError(f"swap attribution unavailable for PID {process.pid}")


def _storage(workspace: Path) -> tuple[int, int]:
    current = workspace.absolute()
    while not current.exists():
        if current.parent == current:
            raise RuntimeError(f"workspace ancestor unavailable: {workspace}")
        current = current.parent
    if not current.is_dir():
        raise RuntimeError(f"workspace ancestor not a directory: {current}")
    stats = os.statvfs(current)
    return stats.f_bavail * stats.f_frsize, stats.f_favail


def sample(
    root: ProcessIdentity,
    known: dict[int, ProcessIdentity],
    policy: MonitorPolicy,
    workspace: Path,
    *,
    reserved_bytes: int = 0,
    reserved_inodes: int = 0,
    elapsed_seconds: float = 0,
    baseline: WorkspaceBaseline | None = None,
) -> MonitorSample:
    """Read one snapshot; missing requested metrics violate rather than pass."""
    if reserved_bytes < 0 or reserved_inodes < 0:
        raise ValueError("reservations must be nonnegative")
    processes = _owned(root, known)
    identities = []
    violations: list[str] = []
    rss = 0
    swap = 0
    rss_valid = swap_valid = True
    for process in processes:
        try:
            expected = known[process.pid]
            if _identity(process) != expected:
                raise RuntimeError(f"PID reused during sample: {process.pid}")
        except psutil.NoSuchProcess:
            continue
        except psutil.Error, OSError, RuntimeError:
            rss_valid = swap_valid = False
            continue
        current_rss = current_swap = None
        try:
            current_rss = int(process.memory_info().rss)
            if current_rss < 0:
                raise RuntimeError("negative owned process RSS")
        except psutil.NoSuchProcess:
            continue
        except psutil.Error, OSError, RuntimeError:
            current_rss = None
        try:
            current_swap = _swap_bytes(process)
            if current_swap < 0:
                raise RuntimeError("negative owned process swap")
        except psutil.NoSuchProcess:
            continue
        except psutil.Error, OSError, RuntimeError:
            current_swap = None
        try:
            if _identity(process) != expected:
                raise RuntimeError(f"PID reused during sample: {process.pid}")
        except psutil.NoSuchProcess:
            continue
        except psutil.Error, OSError, RuntimeError:
            rss_valid = swap_valid = False
            continue
        identities.append(expected)
        if current_rss is None:
            rss_valid = False
        else:
            rss += current_rss
        if current_swap is None:
            swap_valid = False
        else:
            swap += current_swap
    try:
        available = int(host_capacity.measure_memory().available)
    except psutil.Error, OSError, AttributeError, ValueError:
        available = None
    try:
        swap_info = psutil.swap_memory()
        free_swap, used_swap = int(swap_info.free), int(swap_info.used)
    except psutil.Error, OSError, AttributeError, ValueError:
        free_swap = used_swap = None
    try:
        disk, inodes = _storage(workspace)
    except OSError, RuntimeError:
        disk = inodes = None
    projected_disk = disk - reserved_bytes if disk is not None else None
    projected_inodes = inodes - reserved_inodes if inodes is not None else None
    device_memory = None
    if policy.max_device_memory_bytes is not None:
        try:
            device_memory = read_device_memory_bytes(policy.expected_device_uuid or "")
        except ImportError, OSError, RuntimeError, ValueError, AttributeError:
            device_memory = None
    added_bytes = added_inodes = None
    if baseline is not None:
        try:
            current_identity = workspace.stat()
            if (
                baseline.root != str(workspace.resolve(strict=True))
                or baseline.device != current_identity.st_dev
                or baseline.root_inode != current_identity.st_ino
            ):
                raise ValueError("workspace baseline identity differs")
            remaining = (
                policy.max_wall_seconds - elapsed_seconds
                if policy.max_wall_seconds is not None
                else 4.0
            )
            if remaining <= 0:
                raise TimeoutError("monitor deadline exhausted before workspace sample")
            current_bytes, current_inodes = sample_workspace_tree(
                workspace, seconds=min(4.0, remaining)
            )
            after_identity = workspace.stat()
            if (after_identity.st_dev, after_identity.st_ino) != (
                baseline.device,
                baseline.root_inode,
            ):
                raise ValueError("workspace root changed during sample")
            added_bytes = max(0, current_bytes - baseline.apparent_bytes)
            added_inodes = max(0, current_inodes - baseline.inodes)
        except OSError, RuntimeError, ValueError, TimeoutError:
            added_bytes = added_inodes = None
            violations.append("workspace_sampling")
    for name, actual, limit, direction in (
        (
            "max_tree_rss_bytes",
            rss if rss_valid else None,
            policy.max_tree_rss_bytes,
            "max",
        ),
        (
            "max_tree_swap_bytes",
            swap if swap_valid else None,
            policy.max_tree_swap_bytes,
            "max",
        ),
        (
            "min_host_available_ram_bytes",
            available,
            policy.min_host_available_ram_bytes,
            "min",
        ),
        ("min_host_free_swap_bytes", free_swap, policy.min_host_free_swap_bytes, "min"),
        ("min_disk_free_bytes", disk, policy.min_disk_free_bytes, "min"),
        ("min_disk_free_inodes", inodes, policy.min_disk_free_inodes, "min"),
        (
            "min_projected_disk_free_bytes",
            projected_disk,
            policy.min_projected_disk_free_bytes,
            "min",
        ),
        (
            "min_projected_disk_free_inodes",
            projected_inodes,
            policy.min_projected_disk_free_inodes,
            "min",
        ),
        (
            "max_device_memory_bytes",
            device_memory,
            policy.max_device_memory_bytes,
            "max",
        ),
        (
            "max_added_workspace_bytes",
            added_bytes,
            policy.max_added_workspace_bytes,
            "max",
        ),
        (
            "max_added_workspace_inodes",
            added_inodes,
            policy.max_added_workspace_inodes,
            "max",
        ),
    ):
        if limit is not None and (
            actual is None or (actual > limit if direction == "max" else actual < limit)
        ):
            violations.append(name)
    return MonitorSample(
        elapsed_seconds=elapsed_seconds,
        processes=identities,
        tree_rss_bytes=rss if rss_valid else None,
        tree_swap_bytes=swap if swap_valid else None,
        host_available_ram_bytes=available,
        host_free_swap_bytes=free_swap,
        host_used_swap_bytes=used_swap,
        disk_free_bytes=disk,
        disk_free_inodes=inodes,
        projected_disk_free_bytes=projected_disk,
        projected_disk_free_inodes=projected_inodes,
        violations=violations,
        device_memory_bytes=device_memory,
        added_workspace_bytes=added_bytes,
        added_workspace_inodes=added_inodes,
        requested_device_memory=policy.max_device_memory_bytes is not None,
        requested_added_workspace=baseline is not None,
    )


def _signal_owned(known: dict[int, ProcessIdentity], signum: int) -> None:
    """Signal attributed processes, using race-free pidfds where available.

    On other hosts psutil checks process creation time again before signalling;
    those hosts do not provide Linux's atomic pidfd signalling guarantee.
    """
    for identity in list(known.values()):
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            try:
                process = psutil.Process(identity.pid)
                if _identity(process) == identity:
                    process.send_signal(signum)
            except psutil.NoSuchProcess:
                pass
            continue
        try:
            fd = os.pidfd_open(identity.pid)
        except ProcessLookupError:
            continue
        try:
            try:
                if _identity(psutil.Process(identity.pid)) == identity:
                    signal.pidfd_send_signal(fd, signum)
            except psutil.NoSuchProcess, ProcessLookupError:
                pass
        finally:
            os.close(fd)


def _record(stream: object, record: BaseModel) -> None:
    stream.write(record.model_dump_json() + "\n")  # type: ignore[attr-defined]
    stream.flush()  # type: ignore[attr-defined]
    os.fsync(stream.fileno())  # type: ignore[attr-defined]


@contextmanager
def _startup_signals() -> Iterator[None]:
    """Defer handled cancellation until the helper can accept owned shutdown.

    Keep the Popen handle/identity handoff atomic with respect to Python signal
    handlers. Unlike a signal mask, these temporary handlers are reset by exec:
    the helper must be able to receive termination after publishing readiness.
    Non-main-thread callers cannot receive Python signals or install handlers.
    """
    previous = {}
    pending = []

    def defer(signum: int, frame: Any) -> None:
        if not pending:
            pending.append((signum, frame))

    try:
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                handler = signal.getsignal(sig)
                if callable(handler):
                    previous[sig] = handler
                    signal.signal(sig, defer)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if pending:
            signum, frame = pending[0]
            previous[signum](signum, frame)


def _stop_enhanced_supervisor(
    process: subprocess.Popen[bytes],
    root: ProcessIdentity,
    completion: Path,
    grace: float,
) -> None:
    """Request owned cleanup and refuse to certify an unverifiable shutdown."""
    if process.poll() is None:
        _signal_owned({root.pid: root}, signal.SIGTERM)
    try:
        rc = process.wait(timeout=max(2.0, grace + 4.0))
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("owned supervisor cleanup deadline exhausted") from error
    try:
        receipt = json.loads(completion.read_text())
        if (
            receipt["format"] != "klab-owned-completion-v1"
            or receipt["root"] != root.model_dump()
            or receipt["living_descendants"] != 0
            or receipt["returncode"] != rc
        ):
            raise ValueError("invalid owned shutdown receipt")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("owned shutdown could not be verified") from error


def _final_storage_projection(
    result: MonitorCompletion,
    baseline: WorkspaceBaseline,
    workspace: Path,
    policy: MonitorPolicy,
    start: float,
) -> MonitorCompletion:
    """Count pending final receipt bytes/inode before any COMPLETE is published."""
    try:
        remaining = (
            policy.max_wall_seconds - result.elapsed_seconds
            if policy.max_wall_seconds is not None
            else 4.0
        )
        current_bytes, current_inodes = sample_workspace_tree(
            workspace, seconds=min(4.0, max(0.01, remaining))
        )
        identity = workspace.stat()
        if (identity.st_dev, identity.st_ino) != (
            baseline.device,
            baseline.root_inode,
        ):
            raise ValueError("workspace root changed before final receipt")
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        return result.model_copy(
            update={
                "status": "MONITOR_ERROR",
                "violations": [
                    *result.violations,
                    f"final workspace sampling unavailable: {error}",
                ],
            }
        )
    elapsed = time.monotonic() - start
    base_violations = list(result.violations)
    if (
        policy.max_wall_seconds is not None
        and elapsed >= policy.max_wall_seconds
        and "max_wall_seconds" not in base_violations
    ):
        base_violations.append("max_wall_seconds")
    base_status = (
        "MONITOR_ERROR"
        if result.status == "MONITOR_ERROR"
        else "VIOLATED"
        if len(base_violations) > len(result.violations)
        else result.status
    )
    candidate = result.model_copy(
        update={
            "elapsed_seconds": elapsed,
            "violations": base_violations,
            "status": base_status,
        }
    )
    for _ in range(16):
        pending_bytes = len((candidate.model_dump_json() + "\n").encode("utf-8"))
        pending_bytes += len(
            (candidate.model_dump_json(indent=2) + "\n").encode("utf-8")
        )
        final_bytes = max(0, current_bytes + pending_bytes - baseline.apparent_bytes)
        final_inodes = max(0, current_inodes + 1 - baseline.inodes)
        new_violations = list(base_violations)
        if (
            policy.max_added_workspace_bytes is not None
            and final_bytes > policy.max_added_workspace_bytes
        ):
            new_violations.append("max_added_workspace_bytes")
        if (
            policy.max_added_workspace_inodes is not None
            and final_inodes > policy.max_added_workspace_inodes
        ):
            new_violations.append("max_added_workspace_inodes")
        new_status = (
            "MONITOR_ERROR"
            if base_status == "MONITOR_ERROR"
            else "VIOLATED"
            if len(new_violations) > len(base_violations)
            else base_status
        )
        updated = candidate.model_copy(
            update={
                "status": new_status,
                "violations": new_violations,
                "final_added_workspace_bytes": final_bytes,
                "final_added_workspace_inodes": final_inodes,
                "peak_added_workspace_bytes": max(
                    final_bytes, result.peak_added_workspace_bytes or 0
                ),
                "peak_added_workspace_inodes": max(
                    final_inodes, result.peak_added_workspace_inodes or 0
                ),
            }
        )
        if updated == candidate:
            return updated
        candidate = updated
    raise RuntimeError("final receipt byte projection did not converge")


def monitor_command(
    command: list[str],
    policy: MonitorPolicy,
    *,
    workspace: Path,
    log_dir: Path,
    policy_path: Path,
    reserved_bytes: int = 0,
    reserved_inodes: int = 0,
    cwd: Path | None = None,
    baseline_path: Path | None = None,
    owned_supervisor: bool = False,
) -> MonitorCompletion:
    """Launch once; exclusively create a log directory and persist each observation."""
    if not command or reserved_bytes < 0 or reserved_inodes < 0:
        raise ValueError("command required and reservations must be nonnegative")
    if not workspace.exists() or not workspace.is_dir():
        raise ValueError("workspace must be an existing directory")
    working_directory = Path.cwd() if cwd is None else cwd
    if not working_directory.is_dir():
        raise ValueError("monitor command cwd must be an existing directory")
    policy_bytes = policy_path.read_bytes()
    if _parse_policy(policy_bytes, policy_path) != policy:
        raise ValueError("monitor policy differs from policy_path bytes")
    baseline = None
    if (
        policy.max_added_workspace_bytes is not None
        or policy.max_added_workspace_inodes is not None
    ) and baseline_path is None:
        raise ValueError("added-workspace limits require a persisted baseline")
    if baseline_path is not None:
        baseline = load_workspace_baseline(baseline_path, workspace)
    if policy.max_device_memory_bytes is not None:
        # Probe the UUID-bound sensor before a command or receipt directory exists.
        device_memory = read_device_memory_bytes(policy.expected_device_uuid or "")
        if device_memory > policy.max_device_memory_bytes:
            raise ValueError("max_device_memory_bytes exceeded before launch")
    free_bytes, free_inodes = _storage(workspace)
    current_added_bytes = current_added_inodes = 0
    if baseline is not None:
        preflight = sample_workspace_tree(workspace)
        current_added_bytes = max(0, preflight[0] - baseline.apparent_bytes)
        current_added_inodes = max(0, preflight[1] - baseline.inodes)
        if (
            policy.max_added_workspace_bytes is not None
            and current_added_bytes > policy.max_added_workspace_bytes
        ) or (
            policy.max_added_workspace_inodes is not None
            and current_added_inodes > policy.max_added_workspace_inodes
        ):
            raise ValueError("added workspace cap exceeded before launch")
    required_bytes = max(
        reserved_bytes, (policy.max_added_workspace_bytes or 0) - current_added_bytes
    )
    required_inodes = max(
        reserved_inodes, (policy.max_added_workspace_inodes or 0) - current_added_inodes
    )
    if (
        free_bytes < required_bytes
        or free_inodes < required_inodes
        or (
            policy.min_disk_free_bytes is not None
            and free_bytes < policy.min_disk_free_bytes
        )
        or (
            policy.min_disk_free_inodes is not None
            and free_inodes < policy.min_disk_free_inodes
        )
        or (
            policy.min_projected_disk_free_bytes is not None
            and free_bytes - required_bytes < policy.min_projected_disk_free_bytes
        )
        or (
            policy.min_projected_disk_free_inodes is not None
            and free_inodes - required_inodes < policy.min_projected_disk_free_inodes
        )
    ):
        raise ValueError("insufficient free workspace margin before launch")
    _private_parents(log_dir.parent)
    log_dir.mkdir(mode=0o700, exist_ok=False)
    events = log_dir / "events.jsonl"
    completion_path = log_dir / "completion.json"
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    stdout_path = log_dir / "command.stdout.log"
    stderr_path = log_dir / "command.stderr.log"
    start = time.monotonic()
    with (
        open(events, "x", encoding="utf-8", opener=_private_output) as stream,
        open(stdout_path, "xb", opener=_private_output) as command_stdout,
        open(stderr_path, "xb", opener=_private_output) as command_stderr,
    ):
        enhanced = (
            owned_supervisor
            or baseline is not None
            or policy.max_device_memory_bytes is not None
            or policy.max_wall_seconds is not None
        )
        launch_command = command
        if enhanced:
            deadline = (
                start + policy.max_wall_seconds
                if policy.max_wall_seconds is not None
                else 0.0
            )
            launch_command = [
                sys.executable,
                "-m",
                "klab.owned_process",
                "--deadline",
                repr(deadline),
                "--completion",
                str(log_dir / "owned-completion.json"),
                "--grace",
                str(policy.termination_grace_seconds),
                "--ready",
                str(log_dir / "owned-ready.json"),
                "--",
                *command,
            ]
        process = None
        root = None
        known: dict[int, ProcessIdentity] = {}
        try:
            with _startup_signals():
                process = subprocess.Popen(
                    launch_command,
                    cwd=working_directory,
                    stdout=command_stdout,
                    stderr=command_stderr,
                )
                root = _identity(psutil.Process(process.pid))
                known[root.pid] = root
                if enhanced:
                    ready_path = log_dir / "owned-ready.json"
                    ready_deadline = start + min(
                        4.0,
                        policy.max_wall_seconds
                        if policy.max_wall_seconds is not None
                        else 4.0,
                    )
                    while (
                        not ready_path.exists()
                        and process.poll() is None
                        and time.monotonic() < ready_deadline
                    ):
                        time.sleep(0.01)
                    if not ready_path.exists():
                        # The helper publishes readiness before it can launch work.
                        # A failed handshake cannot be treated as a monitored command.
                        raise RuntimeError(
                            "owned supervisor did not become ready before deadline"
                        )
                    ready_record = json.loads(ready_path.read_text())
                    if ready_record != {
                        "pid": root.pid,
                        "create_time": root.create_time,
                    }:
                        raise RuntimeError(
                            "owned supervisor readiness identity mismatch"
                        )
            launch = MonitorLaunch(
                command=command,
                root=root,
                policy=policy,
                workspace=str(workspace.resolve()),
                cwd=str(working_directory.resolve()),
                reserved_bytes=reserved_bytes,
                reserved_inodes=reserved_inodes,
                source_sha256=source_sha,
                policy_sha256=policy_sha,
                source_path=str(Path(__file__).resolve()),
                policy_path=str(policy_path.resolve()),
                stdout_path=str(stdout_path.resolve()),
                stderr_path=str(stderr_path.resolve()),
                events_path=str(events.resolve()),
                completion_path=str(completion_path.resolve()),
                baseline_sha256=baseline.sha256 if baseline is not None else None,
                owned_source_sha256=(
                    hashlib.sha256(
                        (Path(__file__).parent / "owned_process.py").read_bytes()
                    ).hexdigest()
                    if enhanced
                    else None
                ),
            )
            _record(stream, launch)
            minima: dict[str, int | None] = dict.fromkeys(
                (
                    "host_available_ram_bytes",
                    "host_free_swap_bytes",
                    "disk_free_bytes",
                    "disk_free_inodes",
                    "projected_disk_free_bytes",
                    "projected_disk_free_inodes",
                )
            )
            peak_rss: int | None = None
            peak_swap: int | None = None
            peak_device: int | None = None
            peak_added_bytes: int | None = None
            peak_added_inodes: int | None = None
            sample_count = 0
            violations: list[str] = []
            monitor_error = False
            terminating_at: float | None = None
            while True:
                try:
                    observation = sample(
                        root,
                        known,
                        policy,
                        workspace,
                        reserved_bytes=reserved_bytes,
                        reserved_inodes=reserved_inodes,
                        elapsed_seconds=time.monotonic() - start,
                        baseline=baseline,
                    )
                    sample_count += 1
                    if observation.tree_rss_bytes is not None:
                        peak_rss = (
                            observation.tree_rss_bytes
                            if peak_rss is None
                            else max(peak_rss, observation.tree_rss_bytes)
                        )
                    if observation.tree_swap_bytes is not None:
                        peak_swap = (
                            observation.tree_swap_bytes
                            if peak_swap is None
                            else max(peak_swap, observation.tree_swap_bytes)
                        )
                    if observation.device_memory_bytes is not None:
                        peak_device = (
                            observation.device_memory_bytes
                            if peak_device is None
                            else max(peak_device, observation.device_memory_bytes)
                        )
                    if observation.added_workspace_bytes is not None:
                        peak_added_bytes = (
                            observation.added_workspace_bytes
                            if peak_added_bytes is None
                            else max(
                                peak_added_bytes, observation.added_workspace_bytes
                            )
                        )
                    if observation.added_workspace_inodes is not None:
                        peak_added_inodes = (
                            observation.added_workspace_inodes
                            if peak_added_inodes is None
                            else max(
                                peak_added_inodes, observation.added_workspace_inodes
                            )
                        )
                    for field, prior in minima.items():
                        value = getattr(observation, field)
                        if value is not None:
                            minima[field] = (
                                value if prior is None else min(prior, value)
                            )
                    _record(stream, observation)
                    violations.extend(
                        v for v in observation.violations if v not in violations
                    )
                except (RuntimeError, OSError, psutil.Error) as error:
                    monitor_error = True
                    violations.append(f"monitor attribution: {error}")
                if (
                    policy.max_wall_seconds is not None
                    and time.monotonic() - start >= policy.max_wall_seconds
                    and "max_wall_seconds" not in violations
                ):
                    violations.append("max_wall_seconds")
                if violations and terminating_at is None:
                    terminating_at = time.monotonic()
                    _signal_owned(
                        {root.pid: root} if enhanced else known, signal.SIGTERM
                    )
                if (
                    terminating_at is not None
                    and time.monotonic() - terminating_at
                    >= policy.termination_grace_seconds
                    and not enhanced
                ):
                    _signal_owned(known, signal.SIGKILL)
                if (
                    enhanced
                    and terminating_at is not None
                    and time.monotonic() - terminating_at
                    >= policy.termination_grace_seconds + 4.0
                ):
                    raise RuntimeError("owned cleanup deadline exhausted")
                # Continue until root AND all identified children exit; no group signals.
                if process.poll() is not None:
                    if enhanced and terminating_at is None:
                        terminating_at = time.monotonic()
                    try:
                        if not _owned(root, known):
                            break
                    except RuntimeError as error:
                        monitor_error = True
                        violations.append(f"monitor attribution: {error}")
                        # A reused root can no longer own newly discovered children.
                        # Keep only already identified, living children for shutdown.
                        known.pop(root.pid, None)
                        if not any(
                            psutil.pid_exists(pid)
                            and _identity(psutil.Process(pid)) == identity
                            for pid, identity in known.items()
                        ):
                            break
                time.sleep(policy.interval_seconds)
        except BaseException:
            if process is None:
                raise  # Spawn failed; there is no owned child to stop.
            if root is None:
                # A returned, unreaped direct child still belongs to this Popen.
                root = _identity(psutil.Process(process.pid))
                known[root.pid] = root
            if enhanced:
                _stop_enhanced_supervisor(
                    process,
                    root,
                    log_dir / "owned-completion.json",
                    policy.termination_grace_seconds,
                )
            else:
                _signal_owned(known, signal.SIGKILL)
            raise
        rc = process.wait()
        if (
            policy.max_wall_seconds is not None
            and time.monotonic() - start >= policy.max_wall_seconds
            and "max_wall_seconds" not in violations
        ):
            violations.append("max_wall_seconds")
        if baseline_path is not None:
            try:
                if load_workspace_baseline(baseline_path, workspace) != baseline:
                    raise ValueError("workspace baseline changed during command")
            except (OSError, ValueError, ValidationError) as error:
                monitor_error = True
                violations.append(f"workspace baseline unverified: {error}")
        if enhanced:
            try:
                owned_receipt = json.loads(
                    (log_dir / "owned-completion.json").read_text()
                )
                if (
                    owned_receipt["format"] != "klab-owned-completion-v1"
                    or owned_receipt["root"] != root.model_dump()
                    or owned_receipt["living_descendants"] != 0
                    or owned_receipt["returncode"] != rc
                ):
                    raise ValueError("invalid owned shutdown receipt")
            except (OSError, ValueError, KeyError, TypeError) as error:
                monitor_error = True
                violations.append(f"owned shutdown unverified: {error}")
        result = MonitorCompletion(
            launch=launch,
            returncode=rc,
            status=(
                "MONITOR_ERROR"
                if monitor_error
                else "VIOLATED"
                if violations
                else "COMPLETE"
                if rc == 0
                else "FAILED"
            ),
            violations=violations,
            peak_tree_rss_bytes=peak_rss,
            peak_tree_swap_bytes=peak_swap,
            min_host_available_ram_bytes=minima["host_available_ram_bytes"],
            min_host_free_swap_bytes=minima["host_free_swap_bytes"],
            min_disk_free_bytes=minima["disk_free_bytes"],
            min_disk_free_inodes=minima["disk_free_inodes"],
            min_projected_disk_free_bytes=minima["projected_disk_free_bytes"],
            min_projected_disk_free_inodes=minima["projected_disk_free_inodes"],
            elapsed_seconds=time.monotonic() - start,
            sample_count=sample_count,
            peak_device_memory_bytes=peak_device,
            peak_added_workspace_bytes=peak_added_bytes,
            peak_added_workspace_inodes=peak_added_inodes,
        )
        for output in (command_stdout, command_stderr):
            output.flush()
            os.fsync(output.fileno())
        if baseline is not None:
            result = _final_storage_projection(
                result, baseline, workspace, policy, start
            )
        _record(stream, result)
        fd = os.open(completion_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(result.model_dump_json(indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        return result


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--device-memory-probe":
        raise SystemExit(2)
    print(_read_device_memory_direct(sys.argv[2]))
