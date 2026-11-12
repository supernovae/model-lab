"""The one tokenizer pool behind every preparation path.

``EncoderPool`` runs single-threaded ``klab.token_encode_worker`` processes and
moves text in and token IDs out as binary frames. ``PreparationEncoder`` is the
bounded, host-planned view of it that encodes one batch of texts at a time.
"""

from __future__ import annotations

import pickle
import queue
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Self

import numpy as np
import psutil
from tokenizers import Tokenizer

from klab import host_capacity, token_encode_worker
from klab.host_capacity import plan_host_workers

TOKENIZER_BATCH_DOCUMENTS = 256
TOKENIZER_BATCH_SOURCE_BYTES = 1_048_576


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


def tokenizer_work_plan(max_workers: int | None, *, default_cap: int | None = 4):
    """One shared capacity budget for native tokenization across output shards.

    An explicit ``max_workers`` is honored up to measured CPU and RAM capacity.
    Without one, a single child keeps ``default_cap`` threads; ``None`` means the
    measured host capacity.
    """
    try:
        total = host_capacity.measure_memory().total
    except OSError, psutil.Error, AttributeError:
        total = 0
    return plan_host_workers(
        "tokenizer_rayon",
        worker_memory_bytes=512 * 1024**2,
        reserve_bytes=max(1024**3, total // 10),
        operator_cap=default_cap if max_workers is None else max_workers,
    )


class EncoderPool:
    """Tokenizer worker processes; each request runs on whichever is free."""

    def __init__(self, workers: int, tokenizer_json: str, eos: int = 0) -> None:
        self.workers = workers
        self._processes: list[subprocess.Popen[bytes]] = []
        self._idle: queue.SimpleQueue[subprocess.Popen[bytes]] = queue.SimpleQueue()
        self._threads = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="token-encoder"
        )
        self._cpu_seconds = 0.0
        setup = pickle.dumps((tokenizer_json, eos))
        try:
            for _ in range(workers):
                process = subprocess.Popen(
                    [sys.executable, "-m", "klab.token_encode_worker"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                self._processes.append(process)
                assert process.stdin is not None
                token_encode_worker.write_frame(process.stdin, setup)
                process.stdin.flush()
            # Workers load the tokenizer concurrently; collect readiness after.
            for process in self._processes:
                assert process.stdout is not None
                if token_encode_worker.read_frame(process.stdout) != b"ready":
                    raise RuntimeError("tokenizer worker did not load the tokenizer")
                self._idle.put(process)
        except BaseException:
            self.close()
            raise

    def _round_trip(self, request: bytes) -> tuple[bytes, ...]:
        process = self._idle.get()
        try:
            assert process.stdin is not None and process.stdout is not None
            try:
                token_encode_worker.write_frame(process.stdin, request)
                process.stdin.flush()
                status = token_encode_worker.read_frame(process.stdout)
                if status is not None and status[:2] == b"ok":
                    frames = tuple(
                        token_encode_worker.read_frame(process.stdout)
                        for _ in range(status[2])
                    )
                    if any(frame is None for frame in frames):
                        status = None
                    else:
                        return frames
                detail = (
                    token_encode_worker.read_frame(process.stdout)
                    if status in (b"line", b"error")
                    else None
                )
            except (BrokenPipeError, EOFError, OSError) as error:
                raise RuntimeError("tokenizer worker exited during a batch") from error
            if status == b"line" and detail is not None:
                raise token_encode_worker.LineError(*pickle.loads(detail))
            if status == b"error" and detail is not None:
                raise ValueError(
                    f"tokenizer worker: {detail.decode('utf-8', 'replace')}"
                )
            raise RuntimeError("tokenizer worker exited without a reply")
        finally:
            self._idle.put(process)

    def submit(self, request: bytes) -> Future[tuple[bytes, ...]]:
        """Queue one worker request (operation byte first); frames come back."""
        return self._threads.submit(self._round_trip, request)

    def cpu_seconds(self) -> float:
        total = 0.0
        for process in self._processes:
            try:
                cpu = psutil.Process(process.pid).cpu_times()
                total += cpu.user + cpu.system
            except psutil.Error:
                return self._cpu_seconds
        self._cpu_seconds = max(self._cpu_seconds, total)
        return self._cpu_seconds

    def close(self) -> None:
        self.cpu_seconds()
        self._threads.shutdown(wait=True, cancel_futures=True)
        for process in self._processes:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if process.stdout is not None:
                process.stdout.close()
        self._processes = []


class PreparationEncoder:
    """One host-planned encoder pool per preparation, shared by its splits.

    ``encode`` spreads a batch over the workers and returns results in input
    order. ``rayon_threads`` is the number of workers; each is single-threaded.
    """

    def __init__(self, tokenizer: Tokenizer, *, max_workers: int | None):
        self.batch_count = 0
        self.request_bytes = 0
        self.reply_bytes = 0
        self.wait_seconds = 0.0
        self.child_cpu_seconds = 0.0
        self.host_work_plan = tokenizer_work_plan(max_workers)
        self.rayon_threads = self.host_work_plan.workers
        self._pool: EncoderPool | None = EncoderPool(
            self.rayon_threads, tokenizer.to_str()
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def sample_cpu(self) -> float:
        if self._pool is not None:
            self.child_cpu_seconds = self._pool.cpu_seconds()
        return self.child_cpu_seconds

    def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()
            self.child_cpu_seconds = pool.cpu_seconds()

    def encode(self, texts: list[str]) -> list[list[int]]:
        return [ids.tolist() for ids in self.encode_arrays(texts)]

    def encode_arrays(self, texts: list[str]) -> list[np.ndarray]:
        """Token IDs of each text as an int32 array, never as Python integers."""
        if not texts:
            return []
        assert self._pool is not None
        # Contiguous groups of similar size, one per worker.
        sizes = [len(text) for text in texts]
        share = max(1, -(-sum(sizes) // min(self.rayon_threads, len(texts))))
        groups: list[list[str]] = [[]]
        filled = 0
        for text, size in zip(texts, sizes, strict=True):
            if groups[-1] and filled + size > share:
                groups.append([])
                filled = 0
            groups[-1].append(text)
            filled += size
        started = time.monotonic()
        pending = []
        for group in groups:
            request = b"T" + token_encode_worker.pack_texts(group)
            self.request_bytes += len(request)
            pending.append(self._pool.submit(request))
        result: list[np.ndarray] = []
        for group, future in zip(groups, pending, strict=True):
            raw_ids, raw_counts = future.result()
            self.reply_bytes += len(raw_ids) + len(raw_counts)
            counts = np.frombuffer(raw_counts, dtype=np.uint32).tolist()
            if len(counts) != len(group):
                raise RuntimeError("tokenizer worker returned incorrect batch length")
            ids = np.frombuffer(raw_ids, dtype=np.int32)
            position = 0
            for count in counts:
                result.append(ids[position : position + count])
                position += count
        self.wait_seconds += time.monotonic() - started
        self.batch_count += 1
        return result
