"""Operation-scoped tokenizer subprocess with bounded synchronous batches."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Self

import psutil
from tokenizers import Tokenizer

from sparselab import host_capacity
from sparselab.host_capacity import plan_host_workers

TOKENIZER_BATCH_DOCUMENTS = 256
TOKENIZER_BATCH_SOURCE_BYTES = 1_048_576
_MAX_REPLY_BYTES = 128 * 1024 * 1024
_REPLY_TIMEOUT_SECONDS = 180


def validate_tokenizer_batch_limits(
    documents: int, source_bytes: int
) -> tuple[int, int]:
    if type(documents) is not int or not 1 <= documents <= 256:
        raise ValueError("tokenizer_batch_documents must be an integer from 1 to 256")
    if type(source_bytes) is not int or not 1 <= source_bytes <= 8_388_608:
        raise ValueError(
            "tokenizer_batch_source_bytes must be an integer from 1 to 8388608"
        )
    return documents, source_bytes


def tokenizer_work_plan(max_workers: int | None):
    """One shared capacity budget for native tokenization across output shards."""
    try:
        total = host_capacity.measure_memory().total
    except OSError, psutil.Error, AttributeError:
        total = 0
    return plan_host_workers(
        "tokenizer_rayon",
        worker_memory_bytes=512 * 1024**2,
        reserve_bytes=max(1024**3, total // 10),
        operator_cap=4 if max_workers is None else min(4, max_workers),
    )


class PreparationEncoder:
    """One fresh Rayon pool per preparation, shared by train and validation."""

    def __init__(
        self, tokenizer: Tokenizer, workspace: Path, *, max_workers: int | None
    ):
        self._path: Path | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._pending = bytearray()
        self.batch_count = 0
        self.request_bytes = 0
        self.reply_bytes = 0
        self.wait_seconds = 0.0
        self.child_cpu_seconds = 0.0
        try:
            self.host_work_plan = tokenizer_work_plan(max_workers)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=".preparation-tokenizer-",
                suffix=".json",
                dir=workspace.parent,
                delete=False,
            ) as scratch:
                self._path = Path(scratch.name)
                scratch.write(tokenizer.to_str())
                scratch.flush()
            workers = self.host_work_plan.workers
            self.rayon_threads = workers
            env = os.environ.copy()
            env["RAYON_NUM_THREADS"] = str(workers)
            env["TOKENIZERS_PARALLELISM"] = "true"
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "sparselab.preparation_encoder",
                    str(self._path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            assert self._process.stdout is not None
            deadline = time.monotonic() + 15
            with selectors.DefaultSelector() as selector:
                selector.register(self._process.stdout, selectors.EVENT_READ)
                while len(self._pending) < 6:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise RuntimeError(
                            "preparation tokenizer child did not load tokenizer"
                        )
                    chunk = os.read(
                        self._process.stdout.fileno(), 6 - len(self._pending)
                    )
                    if not chunk:
                        raise RuntimeError(
                            "preparation tokenizer child exited loading tokenizer"
                        )
                    self._pending.extend(chunk)
            if self._pending != b"READY\n":
                raise RuntimeError("preparation tokenizer child sent invalid readiness")
            self._pending.clear()
            self._path.unlink()
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def sample_cpu(self) -> float:
        if self._process is not None:
            try:
                cpu = psutil.Process(self._process.pid).cpu_times()
                self.child_cpu_seconds = cpu.user + cpu.system
            except psutil.Error:
                pass
        return self.child_cpu_seconds

    def close(self) -> None:
        self.sample_cpu()
        process = self._process
        self._process = None
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if process.stdout is not None:
                process.stdout.close()
        if self._path is not None:
            self._path.unlink(missing_ok=True)

    def encode(self, texts: list[str]) -> list[list[int]]:
        if not texts:
            return []
        process = self._process
        assert (
            process is not None
            and process.stdin is not None
            and process.stdout is not None
        )
        payload = (
            json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
        started = time.monotonic()
        self.request_bytes += len(payload)
        try:
            process.stdin.write(payload)
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(
                "preparation tokenizer child exited before accepting a batch"
            ) from error
        deadline = time.monotonic() + _REPLY_TIMEOUT_SECONDS
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError(
                        "preparation tokenizer child exceeded batch timeout"
                    )
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError(
                        "preparation tokenizer child exited without a batch result"
                    )
                self._pending.extend(chunk)
                if len(self._pending) > _MAX_REPLY_BYTES:
                    raise ValueError(
                        "preparation tokenizer child batch reply exceeds 128 MiB"
                    )
        self.wait_seconds += time.monotonic() - started
        self.batch_count += 1
        line, _, rest = self._pending.partition(b"\n")
        self.reply_bytes += len(line) + 1
        self._pending = bytearray(rest)
        try:
            response: Any = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("invalid preparation tokenizer child reply") from error
        if isinstance(response, dict) and "error" in response:
            raise ValueError(f"preparation tokenizer child: {response['error']}")
        if not isinstance(response, list) or len(response) != len(texts):
            raise RuntimeError(
                "preparation tokenizer child returned incorrect batch length"
            )
        return response
