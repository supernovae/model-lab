"""One-shot, bounded staging pilot supervisor (operational evidence only)."""

from __future__ import annotations

import hashlib
import math
import os
import secrets
import selectors
import signal
import subprocess
import time
from pathlib import Path
from typing import Literal

import psutil
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from klab.canonical import canonical_digest, canonical_json
from klab.operational_monitor import (
    MonitorPolicy,
    ProcessIdentity,
    _identity,
    _owned,
    _signal_owned,
    sample,
)
from klab.result_publication import sync_directory
from klab.training.pilot_progress import validate_event

_MAX_FRAME = 2048


def _numeric_timeout(value: object) -> object:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("pilot deadline must be a finite numeric duration")
    return value


class PurposeDeadlines(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    initialization_timeout_seconds: float = Field(gt=0)
    no_progress_timeout_seconds: float = Field(gt=0)
    absolute_timeout_seconds: float = Field(gt=0)

    @field_validator(
        "initialization_timeout_seconds",
        "no_progress_timeout_seconds",
        "absolute_timeout_seconds",
        mode="before",
    )
    @classmethod
    def numeric_timeout(cls, value: object) -> object:
        return _numeric_timeout(value)


class PurposeDeadlineOverride(BaseModel):
    """Purpose-specific overrides; absent limits inherit policy-wide defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    initialization_timeout_seconds: float | None = Field(default=None, gt=0)
    no_progress_timeout_seconds: float | None = Field(default=None, gt=0)
    absolute_timeout_seconds: float | None = Field(default=None, gt=0)

    @field_validator(
        "initialization_timeout_seconds",
        "no_progress_timeout_seconds",
        "absolute_timeout_seconds",
        mode="before",
    )
    @classmethod
    def numeric_timeout(cls, value: object) -> object:
        return None if value is None else _numeric_timeout(value)


class PilotDeadlinePolicy(BaseModel):
    """Operational initialization/idle allowances with a mandatory absolute cap.

    The 30-minute initialization and 20-minute idle windows are operational
    allowances, not measured maximums. The two-hour absolute cap stays finite
    even while trusted work counters continue advancing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    pilot_deadline_version: Literal[1] = 1
    initialization_timeout_seconds: float = Field(default=1800, gt=0)
    no_progress_timeout_seconds: float = Field(default=1200, gt=0)
    absolute_timeout_seconds: float = Field(default=7200, gt=0)
    termination_grace_seconds: float = Field(default=5, ge=0)
    smoke: PurposeDeadlineOverride | None = None
    warmup: PurposeDeadlineOverride | None = None

    @field_validator("pilot_deadline_version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("pilot deadline version must be integer 1")
        return value

    @field_validator(
        "initialization_timeout_seconds",
        "no_progress_timeout_seconds",
        "absolute_timeout_seconds",
        "termination_grace_seconds",
        mode="before",
    )
    @classmethod
    def numeric_timeout(cls, value: object) -> object:
        return _numeric_timeout(value)

    @model_validator(mode="after")
    def check_windows(self) -> PilotDeadlinePolicy:
        for windows in (self.for_purpose("smoke"), self.for_purpose("warmup")):
            if (
                windows.no_progress_timeout_seconds > windows.absolute_timeout_seconds
                or windows.initialization_timeout_seconds
                > windows.absolute_timeout_seconds
            ):
                raise ValueError("pilot deadline window exceeds absolute limit")
        return self

    def for_purpose(self, purpose: str) -> PurposeDeadlines:
        if purpose not in {"smoke", "warmup"}:
            raise ValueError("pilot purpose must be smoke or warmup")
        override = getattr(self, purpose)
        return PurposeDeadlines(
            initialization_timeout_seconds=(
                override.initialization_timeout_seconds
                if override is not None
                and override.initialization_timeout_seconds is not None
                else self.initialization_timeout_seconds
            ),
            no_progress_timeout_seconds=(
                override.no_progress_timeout_seconds
                if override is not None
                and override.no_progress_timeout_seconds is not None
                else self.no_progress_timeout_seconds
            ),
            absolute_timeout_seconds=(
                override.absolute_timeout_seconds
                if override is not None
                and override.absolute_timeout_seconds is not None
                else self.absolute_timeout_seconds
            ),
        )


def load_pilot_deadline_policy(path: Path) -> PilotDeadlinePolicy:
    for component in (path.absolute(), *path.absolute().parents):
        if component.is_symlink():
            raise ValueError("pilot deadline policy path must not contain symlinks")
    value = yaml.safe_load(path.read_bytes())
    if (
        not isinstance(value, dict)
        or value.get("pilot_deadline_version") != 1
        or type(value.get("pilot_deadline_version")) is not int
    ):
        raise ValueError("pilot deadline policy requires explicit integer version 1")
    return PilotDeadlinePolicy.model_validate(value)


class PilotDeadlineState:
    """Pure injected-clock deadline machine; untrusted events never reach observe."""

    def __init__(
        self,
        policy: PurposeDeadlines,
        *,
        clock: object = time.monotonic,
        expected_steps: int | None = None,
    ) -> None:
        if expected_steps is not None and (
            type(expected_steps) is not int or expected_steps < 1
        ):
            raise ValueError("expected_steps must be positive")
        self.clock = clock
        self.policy = policy
        self.started = self.last_progress = clock()  # type: ignore[operator]
        self.last_progress_timestamp: float | None = None
        self.training_started = False
        self.phases: list[str] = []
        self.seen_phases: set[str] = set()
        self.completed_phases: set[str] = set()
        self.last_complete_phase: str | None = None
        self.current_step = self.completed_steps = self.completed_targets = 0
        self.counters: dict[tuple[str, str, str | None], tuple[int, int]] = {}
        self.sequence = 0
        self.expected_steps = expected_steps
        self.finished = False

    def observe(self, event: dict[str, object]) -> bool:
        """Return whether the event advanced a trusted unit or phase transition."""
        sequence = event["sequence"]
        if type(sequence) is not int or sequence != self.sequence + 1:
            raise ValueError("invalid progress sequence")
        phase, kind = event["phase"], event["kind"]
        if not isinstance(phase, str) or kind not in {"start", "complete", "progress"}:
            raise ValueError("invalid progress transition")
        if self.finished:
            raise ValueError("progress event after pilot completion")
        # Validate the whole transition before changing attributable state.
        if kind == "start":
            if phase in self.phases or len(self.phases) >= 8:
                raise ValueError("recursive or excessive pilot phases")
            if phase == "pilot_complete" and (self.phases or not self.training_started):
                raise ValueError("pilot completion without finished training")
        elif kind == "complete":
            if not self.phases or self.phases[-1] != phase:
                raise ValueError("phase completion without matching start")
        elif not self.phases or phase != self.phases[-1]:
            raise ValueError("counter outside active phase")
        for name in ("current_step", "completed_steps", "completed_targets"):
            number = event[name]
            if number is not None:
                if type(number) is not int or number < getattr(self, name):
                    raise ValueError("regressing pilot step or target counter")
                if (
                    name == "completed_steps"
                    and self.expected_steps is not None
                    and number > self.expected_steps
                ):
                    raise ValueError("pilot steps exceed expected bound")
        current = (
            self.current_step
            if event["current_step"] is None
            else event["current_step"]
        )
        completed = (
            self.completed_steps
            if event["completed_steps"] is None
            else event["completed_steps"]
        )
        if completed > current and current != 0:
            raise ValueError("completed steps exceed current step")
        if kind == "progress":
            counter, value, total = event["counter"], event["value"], event["total"]
            key = (phase, counter, event["subject"])
            previous_value, previous_total = self.counters.get(key, (0, total))
            if total != previous_total or value < previous_value:
                raise ValueError("regressing or changed pilot counter total")
            if len(self.counters) >= 4096 and key not in self.counters:
                raise ValueError("too many pilot counter subjects")
        advanced = False
        if kind == "start":
            self.phases.append(phase)
            if phase == "training":
                self.training_started = True
            if phase not in self.seen_phases:
                self.seen_phases.add(phase)
                advanced = True
        elif kind == "complete":
            self.phases.pop()
            self.last_complete_phase = phase
            if phase not in self.completed_phases:
                self.completed_phases.add(phase)
                advanced = True
            if phase == "pilot_complete":
                self.finished = True
        for name in ("current_step", "completed_steps", "completed_targets"):
            number = event[name]
            if number is not None and number > getattr(self, name):
                advanced = True
                setattr(self, name, number)
        if kind == "progress" and value > previous_value:
            advanced = True
            self.counters[key] = (value, total)
        self.sequence = sequence
        if advanced:
            self.last_progress = self.clock()  # type: ignore[operator]
            timestamp = event.get("timestamp")
            if isinstance(timestamp, (int, float)) and math.isfinite(timestamp):
                self.last_progress_timestamp = float(timestamp)
        return advanced

    @property
    def current_phase(self) -> str | None:
        return self.phases[-1] if self.phases else None

    def deadline(self) -> str | None:
        now = self.clock()  # type: ignore[operator]
        # Absolute wins even if both thresholds are reached simultaneously.
        if now - self.started >= self.policy.absolute_timeout_seconds:
            return "absolute"
        if (
            not self.training_started
            and now - self.started >= self.policy.initialization_timeout_seconds
        ):
            return "initialization"
        if now - self.last_progress >= self.policy.no_progress_timeout_seconds:
            return "no_progress"
        return None


class PilotSupervisionRecord(BaseModel):
    """Sealed operational outcome; never an optimizer or scientific state record."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    format_version: Literal[1] = 1
    pilot_supervisor_version: Literal[1] = 1
    purpose: Literal["smoke", "warmup"]
    status: Literal["COMPLETE", "FAILED"]
    returncode: int
    reason: str | None
    timeout_kind: Literal["initialization", "no_progress", "absolute"] | None
    root: ProcessIdentity
    policy: PilotDeadlinePolicy
    effective_deadlines: PurposeDeadlines
    policy_sha256: str
    source_sha256: str
    monitor_source_sha256: str
    monitor_policy: MonitorPolicy
    monitor_policy_sha256: str
    last_complete_phase: str | None
    current_phase: str | None
    last_progress_timestamp: float | None
    last_progress_elapsed_seconds: float
    elapsed_seconds: float
    elapsed_since_progress_seconds: float
    sequence: int
    current_step: int
    completed_steps: int
    completed_targets: int
    counters: list[dict[str, object]]
    peak_tree_rss_bytes: int | None
    peak_tree_swap_bytes: int | None
    min_host_available_ram_bytes: int | None
    min_host_free_swap_bytes: int | None
    sample_count: int
    events_ref: str
    events_sha256: str
    execution_log_ref: str
    execution_log_bytes: int
    execution_log_omitted_bytes: int


class PilotSupervisorError(RuntimeError):
    def __init__(self, message: str, evidence: dict[str, object]) -> None:
        super().__init__(message)
        self.evidence = evidence


class PilotTimeout(PilotSupervisorError):
    pass


class PilotCancelled(PilotSupervisorError):
    pass


class PilotResourceFailure(PilotSupervisorError):
    pass


class PilotProcessFailure(PilotSupervisorError):
    pass


def _record(stream: object, payload: dict[str, object]) -> None:
    stream.write(canonical_json(payload) + b"\n")  # type: ignore[attr-defined]
    stream.flush()  # type: ignore[attr-defined]
    os.fsync(stream.fileno())  # type: ignore[attr-defined]


def _seal(path: Path, payload: dict[str, object]) -> dict[str, object]:
    sealed = {**payload, "sha256": canonical_digest(payload)}
    with path.open("xb") as stream:
        _record(stream, sealed)
    sync_directory(path.parent)
    return sealed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _drain_pipe(fd: int) -> tuple[bytes, bool]:
    """Drain available bytes before another process scan, with a fairness cap.

    Reading only one small chunk per process scan can mistake buffered output
    from an exited child for an orphan holding its pipe open. Keep each turn
    bounded so a noisy writer cannot starve progress or deadline checks.
    """
    chunks = bytearray()
    while len(chunks) < 1_048_576:
        try:
            chunk = os.read(fd, min(65_536, 1_048_576 - len(chunks)))
        except BlockingIOError:
            break
        if not chunk:
            return bytes(chunks), True
        chunks.extend(chunk)
    return bytes(chunks), False


def supervise_pilot(
    argv: list[str],
    *,
    purpose: str,
    directory: Path,
    policy: PilotDeadlinePolicy | None = None,
    cancel_path: Path | None = None,
    inherited_fds: tuple[int, ...] = (),
    expected_steps: int | None = None,
    monitor_policy: MonitorPolicy | None = None,
) -> subprocess.CompletedProcess[str]:
    """Launch once; safely terminate only owned PID/create-time descendants.

    The combined execution log is capped at 1 MiB and fsynced; discarded bytes
    are counted. Free-text output is never authority for deadline refresh.
    Structured progress and launch/failure evidence are independently fsynced.
    """
    if not argv or purpose not in {"smoke", "warmup"}:
        raise ValueError("pilot argv and supported purpose required")
    policy = policy or PilotDeadlinePolicy()
    windows = policy.for_purpose(purpose)
    if cancel_path is not None and cancel_path.exists():
        raise PilotCancelled("pilot cancelled before launch", {"reason": "cancelled"})
    directory.mkdir(parents=True, exist_ok=True)
    events_path = directory / "supervisor-events.jsonl"
    log_path = directory / "execution.log"
    if any(path.exists() or path.is_symlink() for path in (events_path, log_path)):
        raise FileExistsError("pilot supervisor evidence already exists")
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    secret = secrets.token_bytes(32)
    env = os.environ.copy()
    env["KLAB_PILOT_PROGRESS_FD"] = str(write_fd)
    env["KLAB_PILOT_PROGRESS_SECRET"] = secret.hex()
    state = PilotDeadlineState(windows, expected_steps=expected_steps)
    monitor = monitor_policy or MonitorPolicy(monitor_policy_version=1)
    monitor_dump = monitor.model_dump(mode="json")
    monitor_digest = canonical_digest(monitor_dump)
    known: dict[int, ProcessIdentity] = {}
    root: ProcessIdentity | None = None
    process: subprocess.Popen[bytes] | None = None
    reason: str | None = None
    terminating_at: float | None = None
    peak_rss: int | None = None
    peak_swap: int | None = None
    min_ram: int | None = None
    min_swap: int | None = None
    sample_count = 0
    buffer = bytearray()
    selector = selectors.DefaultSelector()
    output_bytes = omitted_bytes = 0
    output_tail = bytearray()
    orphaned_at: float | None = None
    try:
        with events_path.open("xb") as stream, log_path.open("xb") as output:
            try:
                process = subprocess.Popen(
                    argv,
                    env=env,
                    pass_fds=(*inherited_fds, write_fd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                root = _identity(psutil.Process(process.pid))
                known[root.pid] = root
            finally:
                os.close(write_fd)
            _record(
                stream,
                {
                    "kind": "launch",
                    "purpose": purpose,
                    "root": root.model_dump(),
                    "policy": policy.model_dump(),
                    "policy_sha256": hashlib.sha256(
                        canonical_json(policy.model_dump())
                    ).hexdigest(),
                    "source_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "monitor_source_sha256": hashlib.sha256(
                        Path(sample.__code__.co_filename).read_bytes()
                    ).hexdigest(),
                    "monitor_policy": monitor_dump,
                    "monitor_policy_sha256": monitor_digest,
                    "expected_steps": expected_steps,
                },
            )
            sync_directory(directory)
            selector.register(read_fd, selectors.EVENT_READ)
            assert process.stdout is not None
            os.set_blocking(process.stdout.fileno(), False)
            selector.register(process.stdout, selectors.EVENT_READ)
            last_sample = float("-inf")
            while True:
                now = time.monotonic()
                if now - last_sample >= 1:
                    last_sample = now
                    output.flush()
                    os.fsync(output.fileno())
                    try:
                        observation = sample(
                            root,
                            known,
                            monitor,
                            directory,
                            elapsed_seconds=now - state.started,
                        )
                        if len(observation.processes) > 256:
                            raise RuntimeError("too many observed pilot descendants")
                        _record(
                            stream,
                            {"kind": "sample", "observation": observation.model_dump()},
                        )
                        sample_count += 1
                        if observation.tree_rss_bytes is not None:
                            peak_rss = max(peak_rss or 0, observation.tree_rss_bytes)
                        if observation.tree_swap_bytes is not None:
                            peak_swap = max(peak_swap or 0, observation.tree_swap_bytes)
                        if observation.host_available_ram_bytes is not None:
                            min_ram = (
                                min(min_ram, observation.host_available_ram_bytes)
                                if min_ram is not None
                                else observation.host_available_ram_bytes
                            )
                        if observation.host_free_swap_bytes is not None:
                            min_swap = (
                                min(min_swap, observation.host_free_swap_bytes)
                                if min_swap is not None
                                else observation.host_free_swap_bytes
                            )
                        if observation.violations and reason is None:
                            reason = "resource"
                    except (RuntimeError, OSError, psutil.Error) as error:
                        if reason is None:
                            reason = "resource"
                        _record(
                            stream,
                            {
                                "kind": "monitor_error",
                                "error_type": type(error).__name__,
                            },
                        )
                if reason is None:
                    if cancel_path is not None and cancel_path.exists():
                        reason = "cancelled"
                    else:
                        reason = state.deadline()
                try:
                    _owned(root, known)
                except RuntimeError, OSError, psutil.Error:
                    reason = reason or "resource"
                if reason is not None and terminating_at is None:
                    terminating_at = now
                    _record(
                        stream,
                        {"kind": "stop", "reason": reason, "sequence": state.sequence},
                    )
                    _signal_owned(known, signal.SIGTERM)
                if (
                    terminating_at is not None
                    and now - terminating_at >= policy.termination_grace_seconds
                ):
                    _signal_owned(known, signal.SIGKILL)
                for ready, _ in selector.select(timeout=0.1):
                    fd = ready.fileobj
                    chunk, eof = _drain_pipe(fd if isinstance(fd, int) else fd.fileno())
                    if eof:
                        selector.unregister(fd)
                    if not chunk:
                        continue
                    if fd != read_fd:
                        output_tail.extend(chunk)
                        if len(output_tail) > 131_072:
                            del output_tail[:-131_072]
                        allowed = min(len(chunk), max(0, 786_432 - output_bytes))
                        if allowed:
                            output.write(chunk[:allowed])
                            output_bytes += allowed
                        omitted_bytes += len(chunk) - allowed
                        continue
                    buffer.extend(chunk)
                    if len(buffer) > 8192 and b"\n" not in buffer:
                        reason = reason or "invalid_progress"
                        buffer.clear()
                    while b"\n" in buffer:
                        line, _, remainder = buffer.partition(b"\n")
                        buffer = bytearray(remainder)
                        try:
                            event = validate_event(
                                line + b"\n",
                                secret,
                                purpose=purpose,
                                pid=root.pid,
                                create_time=root.create_time,
                                last_sequence=state.sequence,
                            )
                            if state.sequence >= 50000:
                                raise ValueError("pilot event budget exceeded")
                            state.observe(event)
                            _record(stream, {"kind": "progress", "event": event})
                        except ValueError:
                            reason = reason or "invalid_progress"
                    if len(buffer) > _MAX_FRAME:
                        reason = reason or "invalid_progress"
                        buffer.clear()
                if process.poll() is not None:
                    try:
                        alive = _owned(root, known)
                    except RuntimeError:
                        reason = reason or "resource"
                        alive = [
                            psutil.Process(pid)
                            for pid, identity in known.items()
                            if psutil.pid_exists(pid)
                            and _identity(psutil.Process(pid)) == identity
                        ]
                    if not alive:
                        if not selector.get_map():
                            break
                        if orphaned_at is None:
                            orphaned_at = now
                        elif now - orphaned_at >= 0.5:
                            # An unobserved pipe holder cannot keep a dead pilot alive.
                            reason = reason or "unclosed_pilot_pipe"
                            break
            if buffer:
                reason = reason or "invalid_progress"
            if omitted_bytes:
                suffix = (
                    b"\n[earlier output truncated; final output follows]\n"
                    + output_tail
                )
                output.write(suffix)
                output.flush()
                os.fsync(output.fileno())
                output_bytes += len(suffix)
            rc = process.wait()
            now = time.monotonic()
            if (
                reason is None
                and rc == 0
                and expected_steps is not None
                and (not state.finished or state.completed_steps != expected_steps)
            ):
                reason = "incomplete_pilot_progress"
            result = {
                "purpose": purpose,
                "returncode": rc,
                "reason": reason,
                "status": "COMPLETE" if reason is None and rc == 0 else "FAILED",
                "timeout_kind": reason
                if reason in {"initialization", "no_progress", "absolute"}
                else None,
                "policy": policy.model_dump(mode="json"),
                "effective_deadlines": windows.model_dump(mode="json"),
                "monitor_source_sha256": hashlib.sha256(
                    Path(sample.__code__.co_filename).read_bytes()
                ).hexdigest(),
                "monitor_policy": monitor_dump,
                "monitor_policy_sha256": monitor_digest,
                "root": root.model_dump(),
                "last_complete_phase": state.last_complete_phase,
                "current_phase": state.current_phase,
                "last_progress_timestamp": state.last_progress_timestamp,
                "last_progress_elapsed_seconds": state.last_progress - state.started,
                "elapsed_seconds": now - state.started,
                "elapsed_since_progress_seconds": now - state.last_progress,
                "sequence": state.sequence,
                "current_step": state.current_step,
                "completed_steps": state.completed_steps,
                "completed_targets": state.completed_targets,
                "counters": [
                    {
                        "phase": phase,
                        "counter": counter,
                        "subject": subject,
                        "value": value,
                        "total": total,
                    }
                    for (phase, counter, subject), (
                        value,
                        total,
                    ) in state.counters.items()
                ],
                "peak_tree_rss_bytes": peak_rss,
                "peak_tree_swap_bytes": peak_swap,
                "min_host_available_ram_bytes": min_ram,
                "min_host_free_swap_bytes": min_swap,
                "sample_count": sample_count,
                "events_ref": events_path.name,
                "events_sha256": _file_sha256(events_path),
                "policy_sha256": hashlib.sha256(
                    canonical_json(policy.model_dump())
                ).hexdigest(),
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "execution_log_ref": "execution.log",
                "execution_log_bytes": output_bytes,
                "execution_log_omitted_bytes": omitted_bytes,
            }
            result = PilotSupervisionRecord.model_validate(result).model_dump(
                mode="json"
            )
            if reason is not None or rc != 0:
                evidence = _seal(directory / "supervisor-failure.json", result)
                if reason in {"absolute", "initialization", "no_progress"}:
                    raise PilotTimeout(f"pilot {reason} deadline expired", evidence)
                if reason == "cancelled":
                    raise PilotCancelled("pilot cancelled", evidence)
                if reason == "resource":
                    raise PilotResourceFailure(
                        "pilot resource monitoring failed", evidence
                    )
                raise PilotProcessFailure(f"pilot failed: {reason or rc}", evidence)
            _seal(directory / "supervisor-completion.json", result)
            return subprocess.CompletedProcess(argv, rc, "", "")
    except BaseException:
        if process is not None:
            _signal_owned(known, signal.SIGKILL)
            process.wait()
        raise
    finally:
        selector.close()
        os.close(read_fd)
