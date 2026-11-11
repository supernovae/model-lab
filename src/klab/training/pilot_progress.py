"""Bounded, authenticated operational progress for one explicitly activated pilot.

This channel never reads or changes training state. It is not scientific evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import psutil

PHASES = frozenset(
    {
        "pilot_start",
        "stage_bundle_verification",
        "run_input_materialization",
        "runtime_initialization",
        "model_initialization",
        "data_open",
        "training",
        "optimizer_update",
        "checkpoint",
        "checkpoint_verification",
        "model_reload",
        "finite_forward",
        "pilot_complete",
        "input_validation",
        "optimizer_initialization",
        "validation",
        "engine_initialization",
    }
)
KINDS = frozenset({"start", "complete", "progress"})
COUNTERS = frozenset({"files", "bytes", "items", "steps", "targets", "chunks"})
_SAFE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_MAX_EVENT = 2048
_ENV_FD = "KLAB_PILOT_PROGRESS_FD"
_ENV_SECRET = "KLAB_PILOT_PROGRESS_SECRET"
_ACTIVE: ContextVar[PilotProgress | None] = ContextVar("pilot_progress", default=None)


def _canonical(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0 or value > 10**15:
        raise ValueError(f"invalid pilot progress {name}")
    return value


def validate_event(
    raw: bytes,
    secret: bytes,
    *,
    purpose: str,
    pid: int,
    create_time: float,
    last_sequence: int,
) -> dict[str, object]:
    """Authenticate one complete frame and strictly validate its bounded schema."""
    if len(raw) > _MAX_EVENT or not raw.endswith(b"\n"):
        raise ValueError("oversized or incomplete pilot progress event")
    try:
        wire = json.loads(raw)
        if not isinstance(wire, dict) or set(wire) != {"payload", "mac"}:
            raise ValueError("invalid event envelope")
        payload = wire["payload"]
        if not isinstance(payload, dict) or not isinstance(wire["mac"], str):
            raise TypeError("invalid event payload")
        mac = hmac.new(secret, _canonical(payload), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(wire["mac"], mac):
            raise ValueError("unauthenticated pilot progress")
        if set(payload) != {
            "version",
            "purpose",
            "pid",
            "create_time",
            "sequence",
            "kind",
            "phase",
            "timestamp",
            "current_step",
            "completed_steps",
            "completed_targets",
            "counter",
            "value",
            "elapsed_phase_seconds",
            "total",
            "subject",
        }:
            raise ValueError("invalid progress fields")
        if type(payload["version"]) is not int or payload["version"] != 1:
            raise ValueError("invalid progress version")
        if (
            payload["purpose"] != purpose
            or type(payload["pid"]) is not int
            or payload["pid"] != pid
        ):
            raise ValueError("forged progress owner")
        if (
            type(payload["create_time"]) not in (float, int)
            or payload["create_time"] != create_time
        ):
            raise ValueError("forged progress process creation time")
        if (
            type(payload["sequence"]) is not int
            or payload["sequence"] != last_sequence + 1
        ):
            raise ValueError("out of order progress sequence")
        if payload["kind"] not in KINDS or payload["phase"] not in PHASES:
            raise ValueError("invalid progress kind or phase")
        if (
            type(payload["timestamp"]) not in (float, int)
            or not math.isfinite(payload["timestamp"])
            or payload["timestamp"] < 0
        ):
            raise ValueError("invalid progress timestamp")
        elapsed_phase = payload["elapsed_phase_seconds"]
        if (
            type(elapsed_phase) not in (float, int)
            or not math.isfinite(elapsed_phase)
            or elapsed_phase < 0
        ):
            raise ValueError("invalid elapsed pilot phase time")
        for key in (
            "current_step",
            "completed_steps",
            "completed_targets",
            "value",
            "total",
        ):
            _integer(payload[key], key)
        if payload["counter"] is not None and payload["counter"] not in COUNTERS:
            raise ValueError("invalid progress counter")
        if payload["subject"] is not None and (
            not isinstance(payload["subject"], str)
            or not _SAFE.fullmatch(payload["subject"])
        ):
            raise ValueError("invalid progress subject")
        if payload["kind"] == "progress":
            if (
                payload["counter"] is None
                or payload["value"] is None
                or payload["total"] is None
            ):
                raise ValueError("progress requires bounded counter")
            if payload["value"] > payload["total"] or payload["total"] == 0:
                raise ValueError("progress exceeds total")
        elif any(payload[key] is not None for key in ("counter", "value", "total")):
            raise ValueError("phase event cannot include a counter")
        return payload
    except (
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        OverflowError,
    ) as error:
        raise ValueError("malformed pilot progress") from error


class PilotProgress:
    """One process-local writer; a pipe write is atomic at the 2048-byte limit."""

    def __init__(self, fd: int, secret: bytes, purpose: str, journal: Path) -> None:
        self.fd = fd
        self.secret = secret
        self.purpose = purpose
        self.pid = os.getpid()
        self.create_time = psutil.Process(self.pid).create_time()
        self.sequence = 0
        self.phases: list[tuple[str, float]] = []
        self.journal = journal.open("xb")
        parent_fd = os.open(journal.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    @property
    def current_phase(self) -> str | None:
        return self.phases[-1][0] if self.phases else None

    def emit(
        self,
        kind: str,
        phase: str,
        *,
        current_step: int | None = None,
        completed_steps: int | None = None,
        completed_targets: int | None = None,
        counter: str | None = None,
        value: int | None = None,
        total: int | None = None,
        subject: str | None = None,
    ) -> None:
        if os.getpid() != self.pid:
            raise RuntimeError("pilot progress writer cannot be reused after fork")
        now = time.monotonic()
        if kind == "start":
            self.phases.append((phase, now))
        elapsed_phase = now - self.phases[-1][1] if self.phases else 0.0
        payload: dict[str, object] = {
            "version": 1,
            "purpose": self.purpose,
            "pid": self.pid,
            "create_time": self.create_time,
            "sequence": self.sequence + 1,
            "kind": kind,
            "phase": phase,
            "timestamp": time.time(),
            "current_step": current_step,
            "completed_steps": completed_steps,
            "completed_targets": completed_targets,
            "counter": counter,
            "elapsed_phase_seconds": elapsed_phase,
            "value": value,
            "total": total,
            "subject": subject,
        }
        mac = hmac.new(self.secret, _canonical(payload), hashlib.sha256).hexdigest()
        wire = _canonical({"payload": payload, "mac": mac}) + b"\n"
        validate_event(
            wire,
            self.secret,
            purpose=self.purpose,
            pid=self.pid,
            create_time=self.create_time,
            last_sequence=self.sequence,
        )
        # Persist the same bounded metadata, not paths or data, before notifying parent.
        self.journal.write(_canonical(payload) + b"\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())
        os.write(self.fd, wire)
        self.sequence += 1
        if kind == "complete" and self.phases and self.phases[-1][0] == phase:
            self.phases.pop()


def current_pilot_progress() -> PilotProgress | None:
    return _ACTIVE.get()


@contextmanager
def activate_pilot_progress(
    *, purpose: str, directory: Path
) -> Iterator[PilotProgress | None]:
    """Activate only when the supervisor supplied an inherited private FD and secret."""
    fd = os.environ.pop(_ENV_FD, None)
    key = os.environ.pop(_ENV_SECRET, None)
    if fd is None and key is None:
        yield None
        return
    if fd is None or key is None or purpose not in {"smoke", "warmup"}:
        raise ValueError("incomplete pilot progress launch")
    secret = bytes.fromhex(key)
    if len(secret) != 32:
        raise ValueError("invalid pilot progress launch secret")
    os.set_inheritable(int(fd), False)
    emitter = PilotProgress(int(fd), secret, purpose, directory / "progress.jsonl")
    token = _ACTIVE.set(emitter)
    try:
        yield emitter
    finally:
        _ACTIVE.reset(token)
        emitter.journal.close()
        os.close(emitter.fd)


def emit_pilot_progress(
    kind: str,
    phase: str,
    *,
    current_step: int | None = None,
    completed_steps: int | None = None,
    completed_targets: int | None = None,
    counter: str | None = None,
    value: int | None = None,
    total: int | None = None,
    subject: str | None = None,
) -> None:
    emitter = current_pilot_progress()
    if emitter is not None:
        emitter.emit(
            kind,
            phase,
            current_step=current_step,
            completed_steps=completed_steps,
            completed_targets=completed_targets,
            counter=counter,
            value=value,
            total=total,
            subject=subject,
        )


@contextmanager
def pilot_phase(
    name: str,
    *,
    current_step: int | None = None,
    completed_steps: int | None = None,
    completed_targets: int | None = None,
) -> Iterator[None]:
    emit_pilot_progress(
        "start",
        name,
        current_step=current_step,
        completed_steps=completed_steps,
        completed_targets=completed_targets,
    )
    yield
    emit_pilot_progress(
        "complete",
        name,
        current_step=current_step,
        completed_steps=completed_steps,
        completed_targets=completed_targets,
    )
