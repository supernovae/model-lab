"""Run one untrusted Python program with no network, an empty directory and limits.

Each program runs in a fresh Bubblewrap (``bwrap``) sandbox: new user, network,
PID, IPC and UTS namespaces, so it has no network interface but loopback, sees
only its own processes and dies with its parent. The filesystem is read-only
system libraries plus the interpreter; the only writable places are an empty
per-program directory (``/work``) and a private ``/tmp``. The environment is
empty. Inside, before the program starts, hard resource limits are set: CPU
seconds, address space, file size, open files and processes (counted per user
namespace, so only the sandbox's own processes count). A wall-clock timeout
kills the whole process group; the namespace's init dies with it.

When ``bwrap`` cannot create the namespaces, nothing runs: :func:`require_sandbox`
raises :class:`SandboxUnavailable`. Running without isolation needs the
caller's explicit ``allow_unsandboxed`` (``klab bench
--allow-unsandboxed-execution``); the record then names ``unsandboxed``.
"""

from __future__ import annotations

import functools
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from klab.runtime_env_subprocess import run_bounded

SANDBOX_PROTOCOL = "klab-code-sandbox-v1"
MECHANISMS = ("bwrap", "unsandboxed")


class SandboxUnavailable(RuntimeError):
    """No isolating sandbox works on this host."""


class SandboxError(RuntimeError):
    """The sandbox itself failed (not the program): never scored as a failure."""


@dataclass(frozen=True)
class SandboxLimits:
    """Per-program limits. They change what passes, so they join the group."""

    timeout_seconds: float = 10.0
    cpu_seconds: int = 10
    memory_bytes: int = 2 * 1024**3
    file_bytes: int = 16 * 1024**2
    processes: int = 16
    open_files: int = 64
    output_bytes: int = 64 * 1024


# Runs inside the sandbox: lower soft and hard limits, then replace itself
# with the program, so the untrusted code never runs with raisable limits.
_LAUNCHER = """\
import json, os, resource, sys
limits = json.loads(sys.argv[1])
for name, value in limits.items():
    # CPU: SIGXCPU at the soft limit (reported as a timeout), SIGKILL 1 s later.
    hard = value + 1 if name == "RLIMIT_CPU" else value
    resource.setrlimit(getattr(resource, name), (value, hard))
os.execv(sys.executable, [sys.executable, "-I", "-S", "-B", "program.py"])
"""


def _interpreter() -> tuple[Path, Path]:
    """The real interpreter (venv links resolved) and its installation prefix."""
    return Path(sys.executable).resolve(), Path(sys.base_prefix).resolve()


def _rlimits(limits: SandboxLimits, *, processes: bool) -> dict[str, int]:
    values = {
        "RLIMIT_CPU": limits.cpu_seconds,
        "RLIMIT_AS": limits.memory_bytes,
        "RLIMIT_FSIZE": limits.file_bytes,
        "RLIMIT_NOFILE": limits.open_files,
        "RLIMIT_CORE": 0,
    }
    if processes:
        # Without a user namespace RLIMIT_NPROC counts every process of the
        # real user, so it is only meaningful inside the sandbox.
        values["RLIMIT_NPROC"] = limits.processes
    return values


def _bwrap_command(work: Path, python: Path, prefix: Path) -> list[str]:
    command = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "HOME",
        "/work",
        "--ro-bind",
        "/usr",
        "/usr",
    ]
    for top in ("/bin", "/lib", "/lib64", "/sbin"):
        path = Path(top)
        if path.is_symlink():
            command += ["--symlink", str(path.readlink()), top]
        elif path.is_dir():
            command += ["--ro-bind", top, top]
    if not prefix.is_relative_to("/usr"):
        command += ["--ro-bind", str(prefix), str(prefix)]
    command += [
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--bind",
        str(work),
        "/work",
        "--chdir",
        "/work",
        str(python),
    ]
    return command


@functools.cache
def _bwrap_works() -> str | None:
    """None when bwrap isolates on this host, else why not."""
    if sys.platform != "linux":
        return "the sandbox needs Linux namespaces"
    if shutil.which("bwrap") is None:
        return "bubblewrap (`bwrap`) is not installed"
    python, prefix = _interpreter()
    with tempfile.TemporaryDirectory(prefix="klab-sandbox-check-") as folder:
        try:
            done = run_bounded(
                [*_bwrap_command(Path(folder), python, prefix), "-I", "-S", "-c", ""],
                timeout=30,
                env={},
            )
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            return f"bwrap failed: {error}"
    if done.returncode != 0:
        return (
            "bwrap cannot create namespaces here: "
            + done.stderr.decode(errors="replace").strip()[-300:]
        )
    return None


def require_sandbox(*, allow_unsandboxed: bool = False) -> str:
    """The mechanism to use; refuses when isolation is unavailable and not waived."""
    reason = _bwrap_works()
    if reason is None:
        return "bwrap"
    if allow_unsandboxed:
        return "unsandboxed"
    raise SandboxUnavailable(
        f"{reason}; generated code is never run without isolation unless you pass "
        "--allow-unsandboxed-execution (it then runs with your user's network "
        "and file access, limited only by rlimits and a timeout)"
    )


def describe(mechanism: str, limits: SandboxLimits) -> dict[str, Any]:
    python, _ = _interpreter()
    return {
        "protocol": SANDBOX_PROTOCOL,
        "mechanism": mechanism,
        "network": "none" if mechanism == "bwrap" else "host",
        "filesystem": "read-only system and interpreter; writable empty /work "
        "and private /tmp"
        if mechanism == "bwrap"
        else "host filesystem; cwd is an empty temporary directory",
        "environment": "empty",
        "python": {
            "executable": str(python),
            "version": ".".join(map(str, sys.version_info[:3])),
            "flags": "-I -S -B",
        },
        "limits": asdict(limits),
    }


def run_program(
    source: str, limits: SandboxLimits, *, mechanism: str = "bwrap"
) -> dict[str, Any]:
    """Run SOURCE once; status is passed, failed, timeout or error (output limit).

    A sandbox that fails to start raises :class:`SandboxError`: that is missing
    evidence, never a model's failed problem.
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown sandbox mechanism: {mechanism}")
    python, prefix = _interpreter()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="klab-code-") as folder:
        work = Path(folder)
        (work / "program.py").write_text(source, encoding="utf-8")
        isolated = mechanism == "bwrap"
        prefix_command = (
            _bwrap_command(work, python, prefix) if isolated else [str(python)]
        )
        command = [
            *prefix_command,
            "-I",
            "-S",
            "-c",
            _LAUNCHER,
            json.dumps(_rlimits(limits, processes=isolated)),
        ]
        try:
            done = run_bounded(
                command,
                timeout=limits.timeout_seconds,
                output_limit=limits.output_bytes,
                env={},
                cwd=work,
            )
        except subprocess.TimeoutExpired:
            return _outcome("timeout", None, b"", started)
        except ValueError:
            return _outcome("error", None, b"output limit exceeded", started)
    if isolated and done.returncode != 0 and done.stderr.startswith(b"bwrap:"):
        raise SandboxError(done.stderr.decode(errors="replace").strip()[-300:])
    status = "passed" if done.returncode == 0 else "failed"
    if done.returncode in {-signal.SIGXCPU, 128 + signal.SIGXCPU}:
        status = "timeout"  # CPU seconds exhausted
    return _outcome(status, done.returncode, done.stderr, started)


def _outcome(
    status: str, returncode: int | None, stderr: bytes, started: float
) -> dict[str, Any]:
    return {
        "status": status,
        "returncode": returncode,
        "stderr_tail": stderr.decode(errors="replace")[-400:],
        "seconds": round(time.monotonic() - started, 3),
    }
