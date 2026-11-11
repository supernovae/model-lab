"""One cancellation and resource context for a whole lab operation.

A :class:`LabContext` travels through every phase of ``klab try`` (train,
score, probe) and through a standalone ``klab probe``. It owns the cancel
sentinel, the optional :class:`~klab.resource_envelope.ResourceEnvelope`
and the SIGINT/SIGTERM handlers, and knows the current arm and phase so a stop
is recorded where it happened.
"""

from __future__ import annotations

import signal
import threading
from pathlib import Path
from typing import Any


class LabCancelled(Exception):
    """The operation stopped at a safe point (cancel sentinel or resources)."""

    kind = "cancelled"

    def __init__(
        self, arm: str | None, reason: str, row: dict[str, Any] | None, phase: str
    ) -> None:
        super().__init__(f"{arm} arm stopped during {phase}: {reason}")
        self.arm = arm
        self.reason = reason
        self.row = row
        self.phase = phase


class LabResourceExceeded(LabCancelled):
    """A requested resource-envelope limit was violated at a safe point."""

    kind = "resources"


class LabSignal(BaseException):
    """Raised by the lab signal handler outside the trainer's own handler."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


def is_out_of_memory(error: BaseException) -> bool:
    """Host MemoryError or an accelerator OOM from torch."""
    if isinstance(error, MemoryError):
        return True
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a core dependency
        return False
    oom = getattr(torch, "OutOfMemoryError", None) or getattr(
        torch.cuda, "OutOfMemoryError", None
    )
    return oom is not None and isinstance(error, oom)


def release_memory() -> None:
    """Return freed tensors to the allocator between arms and after an OOM."""
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache"):
        try:
            mps.empty_cache()
        except RuntimeError:
            pass


class LabContext:
    """Track arm/phase, honor the cancel sentinel and the resource envelope.

    The trainer installs its own handlers while it trains (stopping at a
    checkpointed step) and restores these afterwards, so evaluation, scoring,
    probing and record writing are covered too. After the first signal, or once
    the record is being finalized, further signals are only noted and never
    interrupt the record write.
    """

    def __init__(
        self,
        cancel_path: Path,
        *,
        resource_envelope: Any = None,
        workspace: Path | None = None,
    ) -> None:
        self.cancel_path = cancel_path
        self.resource_envelope = resource_envelope
        self.workspace = workspace or cancel_path.parent
        self.arm: str | None = None
        self.phase = "setup"
        self.row: dict[str, Any] | None = None
        self.signals: list[str] = []
        self.finalizing = False
        self._previous: dict[int, Any] = {}

    def _handle(self, signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        self.signals.append(name)
        if not self.finalizing and len(self.signals) == 1:
            raise LabSignal(name)

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.signal(signum, self._handle)

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()

    def enter(self, arm: str | None, phase: str) -> None:
        self.arm, self.phase = arm, phase

    def check_cancel(self) -> None:
        if self.cancel_path.exists():
            raise LabCancelled(self.arm, "cancelled", self.row, self.phase)

    def measure(self) -> dict[str, Any] | None:
        """Envelope reading now; raises LabResourceExceeded on a violation."""
        if self.resource_envelope is None:
            return None
        from klab.resource_envelope import (
            check_envelope,
            current_process_rss_bytes,
        )

        try:
            return check_envelope(
                self.resource_envelope,
                workspace=self.workspace,
                rss_bytes=current_process_rss_bytes(),
            )
        except ValueError as error:
            raise LabResourceExceeded(
                self.arm, str(error), self.row, self.phase
            ) from None

    def checkpoint(self) -> None:
        """Honor the cancel sentinel and the resource envelope at a safe point."""
        self.check_cancel()
        self.measure()
