"""Deadline and byte-bounded child output without limiting unrelated child files."""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from pathlib import Path


def run_bounded(
    command: list[str],
    *,
    timeout: float,
    output_limit: int = 32768,
    input: bytes | None = None,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stdout_limit: int | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    if stdout_path is not None and (type(stdout_limit) is not int or stdout_limit < 0):
        raise ValueError("file stdout requires an exact nonnegative byte limit")
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=env,
        cwd=cwd,
    )
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout
    stdout_file = None
    stdout_bytes = 0
    completed = False
    try:
        if stdout_path is not None:
            descriptor = os.open(
                stdout_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            stdout_file = os.fdopen(descriptor, "wb")
        pending = memoryview(input or b"")
        input_offset = 0
        if process.stdin is not None:
            os.set_blocking(process.stdin.fileno(), False)
            if not pending:
                process.stdin.close()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            if pending:
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                for key, _ in selector.select(remaining):
                    if key.data == "stdin":
                        try:
                            input_offset += os.write(
                                key.fileobj.fileno(),
                                pending[input_offset : input_offset + 8192],
                            )
                        except BlockingIOError:
                            continue
                        except BrokenPipeError:
                            input_offset = len(pending)
                        if input_offset == len(pending):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        continue
                    data = os.read(key.fileobj.fileno(), 8192)
                    if not data:
                        selector.unregister(key.fileobj)
                    else:
                        if key.data == "stdout" and stdout_file is not None:
                            stdout_bytes += len(data)
                            if stdout_bytes > stdout_limit:
                                raise ValueError("subprocess output exceeds byte limit")
                            stdout_file.write(data)
                        else:
                            streams[key.data].extend(data)
                            if len(streams[key.data]) > output_limit:
                                raise ValueError("subprocess output exceeds byte limit")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout)
        code = process.wait(timeout=remaining)
        completed = True
        return subprocess.CompletedProcess(
            command, code, bytes(streams["stdout"]), bytes(streams["stderr"])
        )
    finally:
        if not completed:
            # The leader can exit while descendants still hold our pipes open.
            # Kill the owned group even then; detached sessions are outside it.
            # Do not poll/wait first: the unreaped leader reserves its PID/PGID
            # through the signal, so cleanup cannot target a recycled group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        process.stdout.close()
        process.stderr.close()
        if process.stdin is not None:
            process.stdin.close()
        if stdout_file is not None:
            stdout_file.close()
            if not completed:
                stdout_path.unlink(missing_ok=True)
