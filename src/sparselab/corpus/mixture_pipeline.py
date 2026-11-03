"""Ordered bounded source reads and authenticated assembly checkpoints.

Checkpoints are operational evidence. They never replace release verification.
Only the coordinator mutates the token cache, schedule, or disposable SQL index.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing, contextmanager
from pathlib import Path
from threading import Lock

from sparselab.canonical import canonical_digest, canonical_json
from sparselab.corpus.admission_inspection_cache import capture
from sparselab.corpus.mixture_index import MAX_ROW_BYTES, rows
from sparselab.corpus.split_runs import write_rows
from sparselab.result_publication import sync_directory
from sparselab.training.manifest import sha256_file

PREFETCH_BYTES = 128 * 1024**2
PREFETCH_DOCUMENTS = 256  # One tokenizer batch can read ahead while encoding runs.


def ordered_fetch(
    items,
    fetch,
    *,
    workers,
    size=lambda row: MAX_ROW_BYTES,
    metrics=None,
    cached=lambda item: None,
):
    """Reserve raw bytes before submission, including out-of-order results.

    JSON object overhead is additional to the raw-byte budget; worker planning
    reserves room for it. Workers own independent handles and cannot reorder
    admission decisions. Exceptions propagate at the original candidate slot.
    """
    metrics = metrics if metrics is not None else {}
    metrics.setdefault("read_active", 0)
    metrics.setdefault("read_active_peak", 0)
    active_lock = Lock()

    def timed(item):
        started = time.monotonic()
        with active_lock:
            metrics["read_active"] += 1
            metrics["read_active_peak"] = max(
                metrics["read_active_peak"], metrics["read_active"]
            )
        try:
            value = fetch(item)
            return value, time.monotonic() - started
        finally:
            with active_lock:
                metrics["read_active"] -= 1

    if workers == 0:
        # A one-worker operation reads inline, so encoding never overlaps a
        # second worker. Larger plans reserve dedicated prefetch workers.
        for item in items:
            charge = size(item)
            if type(charge) is not int or not 0 < charge <= MAX_ROW_BYTES:
                raise ValueError("invalid mixture prefetch byte reservation")
            value = cached(item)
            if value is None:
                value, busy = timed(item)
                metrics["read_worker_seconds"] = (
                    metrics.get("read_worker_seconds", 0) + busy
                )
            metrics["read_queue_bytes_peak"] = max(
                metrics.get("read_queue_bytes_peak", 0), charge
            )
            metrics["read_queue_records_peak"] = 1
            yield item, value
        return

    iterator = iter(items)
    queued, reserved = deque(), 0
    current = next(iterator, None)
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="mixture-read"
    ) as pool:
        try:
            while current is not None or queued:
                while current is not None and len(queued) < max(
                    PREFETCH_DOCUMENTS, 2 * workers
                ):
                    charge = size(current)
                    if type(charge) is not int or not 0 < charge <= MAX_ROW_BYTES:
                        raise ValueError("invalid mixture prefetch byte reservation")
                    if reserved + charge > PREFETCH_BYTES:
                        break
                    reused = cached(current)
                    if reused is None:
                        future = pool.submit(timed, current)
                    else:
                        future = Future()
                        future.set_result((reused, 0.0))
                    queued.append((current, charge, future))
                    reserved += charge
                    metrics["read_queue_bytes_peak"] = max(
                        metrics.get("read_queue_bytes_peak", 0), reserved
                    )
                    metrics["read_queue_records_peak"] = max(
                        metrics.get("read_queue_records_peak", 0), len(queued)
                    )
                    current = next(iterator, None)
                item, charge, future = queued.popleft()
                waiting = time.monotonic()
                value, busy = future.result()
                metrics["read_wait_seconds"] = (
                    metrics.get("read_wait_seconds", 0) + time.monotonic() - waiting
                )
                metrics["read_worker_seconds"] = (
                    metrics.get("read_worker_seconds", 0) + busy
                )
                yield item, value
                reserved -= charge
        finally:
            for _, _, future in queued:
                future.cancel()
            close = getattr(iterator, "close", None)
            if close is not None:
                close()


class MixtureFrontier:
    """Immutable segments and native proof-backed committed schedule frontier.

    Restoring SQL acceleration structures replays index deltas only: it does
    not fetch sources, tokenize, make dedup decisions or re-emit schedule rows.
    Uncommitted segment directories are ignored and retained for diagnosis.
    """

    def __init__(self, output, cache, request):
        self.root = output.parent / ("." + output.name + ".assembly")
        self.root.mkdir(exist_ok=True)
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ValueError("unsafe mixture checkpoint root")
        self.cache = cache
        if cache.proofs is None or not cache.writable:
            raise ValueError("mixture checkpoint proof storage unavailable")
        self.identity = canonical_digest(request)
        binding = self.root / "request.json"
        if binding.exists():
            if json.loads(binding.read_bytes()) != request:
                raise ValueError(
                    "mixture checkpoint identity mismatch; use a new output"
                )
        else:
            write_rows(binding, [request])
        self.sequence = 0
        self.state = None
        while (marker := self.root / f"commit-{self.sequence:012d}.json").exists():
            entry = json.loads(marker.read_bytes())
            expected = {"identity": self.identity, "sequence": self.sequence}
            found = cache.load("mixture-frontier", expected, limit=4 * 1024**2)
            if found is None or found[0] != entry:
                raise ValueError(
                    "mixture checkpoint authentication or corruption failure"
                )
            if entry["previous"] != (
                canonical_digest(self.state) if self.state else None
            ):
                raise ValueError("mixture checkpoint frontier chain mismatch")
            self.state = entry
            self.sequence += 1
        if sum(1 for _ in self.root.glob("commit-*.json")) != self.sequence:
            raise ValueError("mixture checkpoint sequence gap")
        self.output_offset = 0
        self.excluded_offset = 0
        self.event_file = None

    def restore(self, index, stream, digest, excluded):
        for sequence in range(self.sequence):
            segment = json.loads(
                (self.root / f"commit-{sequence:012d}.json").read_bytes()
            )
            directory = Path(segment["directory"])
            for event in rows(directory / "index.jsonl"):
                index.restore_event(event)
            with (directory / "output.jsonl").open("rb") as source:
                while chunk := source.read(1024**2):
                    stream.write(chunk)
                    digest.update(chunk)
            excluded.extend(rows(directory / "excluded.jsonl"))
        self.output_offset = stream.tell()
        self.excluded_offset = len(excluded)

    def begin(self, index):
        from uuid import uuid4

        self.directory = self.root / ("segment-" + uuid4().hex)
        self.directory.mkdir()
        self.event_file = (self.directory / "index.jsonl").open("xb")
        index.event_sink = lambda event: self.event_file.write(
            canonical_json(event) + b"\n"
        )

    def commit(self, state, stream, excluded, index):
        stream.flush()
        end = stream.tell()
        with (self.directory / "output.jsonl").open("xb") as target:
            stream.seek(self.output_offset)
            remaining = end - self.output_offset
            while remaining:
                raw = stream.read(min(1024**2, remaining))
                if not raw:
                    raise ValueError("mixture checkpoint output truncated")
                target.write(raw)
                remaining -= len(raw)
            target.flush()
            os.fsync(target.fileno())
        stream.seek(end)
        write_rows(self.directory / "excluded.jsonl", excluded[self.excluded_offset :])
        self.event_file.flush()
        os.fsync(self.event_file.fileno())
        self.event_file.close()
        sync_directory(self.directory)
        entry = {
            "previous": canonical_digest(self.state) if self.state else None,
            "directory": str(self.directory),
            "state": state,
        }
        request = {"identity": self.identity, "sequence": self.sequence}
        self.cache.publish(
            "mixture-frontier",
            request,
            entry,
            capture([self.directory]),
            limit=4 * 1024**2,
        )
        # Publication refusal is explicit; an unsigned frontier is never committed.
        found = self.cache.load("mixture-frontier", request, limit=4 * 1024**2)
        if found is None or found[0] != entry:
            raise ValueError("mixture checkpoint could not authenticate commit")
        write_rows(self.root / f"commit-{self.sequence:012d}.json", [entry])
        self.state = entry
        self.sequence += 1
        self.output_offset = end
        self.excluded_offset = len(excluded)
        self.begin(index)

    def close(self, index):
        index.event_sink = None
        if self.event_file is not None:
            self.event_file.close()


def pipeline_request(request, index):
    producer = {
        name: sha256_file(Path(__file__).with_name(name))
        for name in (
            "mixture.py",
            "mixture_index.py",
            "mixture_pipeline.py",
            "retained_mixture.py",
            "token_cache.py",
        )
    }
    producer.update(
        {
            "data/" + name: sha256_file(Path(__file__).parents[1] / "data" / name)
            for name in ("encoding.py", "tokenizer.py")
        }
    )
    producer["preparation_encoder.py"] = sha256_file(
        Path(__file__).parents[1] / "preparation_encoder.py"
    )
    return {
        **request,
        "producer": producer,
        "inputs": getattr(index, "dependencies", None)
        or {str(path): digest for path, digest in index.input_digests.items()},
    }


@contextmanager
def mixture_writer(output):
    """One coordinator per output; do not remove the inode carrying the lock."""
    import fcntl

    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("unsafe mixture output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / ("." + output.name + ".assembly.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("mixture output already has an active writer") from error
        yield
    finally:
        os.close(descriptor)


class MixturePreflight:
    """Reuse a verified ordinary input index; mutable schedule SQL is never proof."""

    def __init__(self, cache, request, release, inventory):
        self.cache = cache
        self.inputs = {
            path: sha256_file(path)
            for path in (
                release / "documents.jsonl",
                inventory,
                release / "lm/train.lineage.jsonl",
            )
        }
        self.request = {
            **request,
            "inputs": {str(p): digest for p, digest in self.inputs.items()},
        }
        prior = cache.load("mixture-preflight", self.request, limit=4 * 1024**2)
        self.snapshot = prior[0]["snapshot"] if prior is not None else None

    def publish(self, index):
        import sqlite3
        from uuid import uuid4

        directory = self.cache.root / "cache" / "mixture-preflight"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (uuid4().hex + ".sqlite")
        index.verify_inputs()
        index.db.commit()
        with closing(sqlite3.connect(path)) as target:
            index.db.backup(target)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        sync_directory(directory)
        self.cache.publish(
            "mixture-preflight",
            self.request,
            {"snapshot": str(path)},
            capture([path]),
            limit=4 * 1024**2,
        )


def process_tree_sample():
    """Sample diagnostics without making a disappearing child fail the work."""
    import psutil

    from sparselab.corpus.progress import memory_bytes

    rss, peak, _available = memory_bytes()
    sample = {
        "rss_bytes": rss,
        "process_peak_rss_bytes": peak,
        "child_cpu_seconds": 0.0,
        "child_rss_bytes": 0,
        "child_process_threads": 0,
        "child_sample_complete": True,
        "process_observation_errors": [],
    }
    errors = set()
    try:
        children = psutil.Process().children()
    except psutil.Error as error:
        children = []
        errors.add(type(error).__name__)
    for child in children:
        try:
            cpu = child.cpu_times()
            memory = child.memory_info()
            threads = child.num_threads()
        except psutil.Error as error:
            errors.add(type(error).__name__)
            continue
        sample["child_cpu_seconds"] += cpu.user + cpu.system
        sample["child_rss_bytes"] += memory.rss
        sample["child_process_threads"] += threads
    sample["child_sample_complete"] = not errors
    sample["process_observation_errors"] = sorted(errors)
    return sample
