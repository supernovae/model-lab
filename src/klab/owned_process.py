"""Isolated Linux subreaper for a single native monitored command.

The monitor runs this helper as its child.  Unlike a process-group kill, the
subreaper retains attribution when a parent exits or a child starts a session.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from klab.operational_monitor import (
    _identity,
    _owned,
    _private_output,
    _signal_owned,
)


def supervise(
    command: list[str],
    *,
    completion: Path,
    deadline: float = 0.0,
    grace: float = 1.0,
    ready: Path | None = None,
) -> int:
    if (
        sys.platform != "linux"
        or not command
        or grace < 0
        or not math.isfinite(grace)
        or not math.isfinite(deadline)
    ):
        raise ValueError("a nonempty Linux command and nonnegative grace are required")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot establish command subreaper")
    root = _identity(psutil.Process())
    known = {root.pid: root}
    received = 0

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal received
        received = signum

    previous = {
        sig: signal.signal(sig, request_stop)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    process: subprocess.Popen[bytes] | None = None
    reason = ""
    rc = 1
    living = []
    try:
        if ready is not None:
            pending = ready.with_name(ready.name + ".pending")
            with open(pending, "x", encoding="utf-8", opener=_private_output) as stream:
                stream.write(
                    json.dumps({"pid": root.pid, "create_time": root.create_time})
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, ready)
        if not received and (not deadline or time.monotonic() < deadline):
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, start_new_session=True
            )
            known[process.pid] = _identity(psutil.Process(process.pid))
        while True:
            if process is None:
                reason = "stopped before command launch"
                break
            leader_rc = process.poll()
            living = [p for p in _owned(root, known) if p.pid != root.pid]
            if received:
                reason = f"supervisor received signal {received}"
                break
            if deadline and time.monotonic() >= deadline:
                reason = "command deadline reached"
                break
            if leader_rc is not None:
                if living:
                    reason = "command exited with live descendants"
                    break
                rc = leader_rc
                reason = "completed" if rc == 0 else f"command exited {rc}"
                break
            time.sleep(0.02)
    finally:
        if process is not None:
            grace_end = time.monotonic() + grace
            cleanup_end = grace_end + 2.0
            term_sent = False
            while True:
                process.poll()  # Reap the direct child before discovering adopted descendants.
                living = [p for p in _owned(root, known) if p.pid != root.pid]
                if not living:
                    break
                targets = {p.pid: known[p.pid] for p in living}
                if not term_sent:
                    _signal_owned(targets, signal.SIGTERM)
                    term_sent = True
                elif time.monotonic() >= grace_end:
                    _signal_owned(targets, signal.SIGKILL)
                if time.monotonic() >= cleanup_end:
                    reason = "owned cleanup deadline exhausted"
                    rc = 1
                    break
                time.sleep(0.02)
            while time.monotonic() < cleanup_end:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    break
            # Record a fresh living count after draining adopted zombies.
            living = [p for p in _owned(root, known) if p.pid != root.pid]
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        record = {
            "format": "klab-owned-completion-v1",
            "reason": reason,
            "returncode": rc,
            "observed_descendants": sorted(pid for pid in known if pid != root.pid),
            "root": root.model_dump(),
            "living_descendants": len(living),
        }
        fd = os.open(completion, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--grace", type=float, default=1.0)
    parser.add_argument("--ready", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return supervise(
        command,
        completion=args.completion,
        deadline=args.deadline,
        grace=args.grace,
        ready=args.ready,
    )


if __name__ == "__main__":
    raise SystemExit(main())
