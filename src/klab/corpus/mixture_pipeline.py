"""Ordered bounded source reads and authenticated assembly checkpoints.

Checkpoints are operational evidence. They never replace release verification.
Only the coordinator mutates the token cache, schedule, or disposable SQL index.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

import numpy as np

from klab import token_encode_worker
from klab.canonical import canonical_digest, canonical_json
from klab.corpus.admission_inspection_cache import capture
from klab.corpus.mixture_index import MAX_ROW_BYTES, rows
from klab.corpus.split_runs import write_rows
from klab.result_publication import sync_directory
from klab.safe_io import symlinked
from klab.training.manifest import sha256_file
from klab.verification_proofs import artifact_fingerprint

PREFETCH_BYTES = 128 * 1024**2
PREFETCH_DOCUMENTS = 256  # One tokenizer batch can read ahead while encoding runs.
# Source bytes the encoder may hold ahead of the ordered consumer, and batches
# in flight per encoder worker.
ENCODE_AHEAD_BYTES = 128 * 1024**2
ENCODE_AHEAD_BATCHES = 3


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


class _Done:
    """A finished request, for batches encoded in this process."""

    def __init__(self, frames):
        self.frames = frames

    def result(self):
        return self.frames

    def cancel(self):
        return False


class _Batch:
    def __init__(self):
        self.items = []
        self.bytes = 0
        self.future = None


def encode_ahead(
    source,
    *,
    position,
    encoder,
    tokenizer,
    tokens,
    documents,
    source_bytes,
    metrics,
):
    """Yield ``(doc, family, cursor, ids)`` in source order, encoding ahead.

    ``source`` yields ``(doc, family)``. A document carries ``cached_payload``
    (IDs already known), ``text`` (read by the caller) or ``path`` and
    ``location`` (a byte range the encoder reads itself, so the text never
    reaches this process). Uncached documents are grouped into batches of at
    most ``documents`` rows and ``source_bytes``; each batch is one request to
    whichever encoder worker is free, and up to ``ENCODE_AHEAD_BATCHES`` batches
    per worker are in flight while the consumer applies earlier ones in order.
    ``cursor`` is ``position()`` as the document left ``source``, so a consumer
    that stops after a document resumes after it however far ahead the source
    ran. ``encoder`` is a pool or ``None`` to encode in this process. Fresh IDs
    are stored in ``tokens`` under the SHA-256 of the text the encoder read.
    """
    limit = ENCODE_AHEAD_BATCHES * (encoder.rayon_threads if encoder else 1)
    window, flying, held = deque(), 0, 0
    open_batch = _Batch()
    exhausted = False
    iterator = iter(source)
    metrics.setdefault("encode_wait_seconds", 0.0)

    def submit():
        nonlocal open_batch, flying
        batch, open_batch = open_batch, _Batch()
        first = batch.items[0][0]
        if "location" in first:
            locations = [item[0]["location"] for item in batch.items]
            request = b"D" + token_encode_worker.pack_documents(
                first["path"], locations
            )
        else:
            request = b"T" + token_encode_worker.pack_texts(
                [item[0]["text"] for item in batch.items]
            )
        if encoder is not None:
            metrics["batches"] = metrics.get("batches", 0) + 1
            batch.future = encoder.submit(request)
        else:
            batch.future = _Done(_encode_here(tokenizer, request))
        for item in batch.items:
            item[4] = batch
        flying += 1

    def finish(batch):
        nonlocal flying
        waiting = time.monotonic()
        frames = batch.future.result()
        metrics["encode_wait_seconds"] += time.monotonic() - waiting
        flying -= 1
        ids = np.frombuffer(frames[0], dtype=np.int32)
        counts = np.frombuffer(frames[1], dtype=np.uint32).tolist()
        if len(counts) != len(batch.items):
            raise ValueError("tokenizer returned incorrect batch length")
        started, at = time.monotonic(), 0
        for index, (item, count) in enumerate(zip(batch.items, counts, strict=True)):
            item[3] = ids[at : at + count]
            at += count
            if tokens is not None:
                digest = (
                    frames[2][32 * index : 32 * index + 32]
                    if len(frames) > 2
                    else hashlib.sha256(item[0]["text"].encode("utf-8")).digest()
                )
                tokens.store(digest, item[3])
        if tokens is not None:
            tokens.write_seconds += time.monotonic() - started
        batch.items = None

    try:
        while True:
            while (
                not exhausted
                and held < ENCODE_AHEAD_BYTES
                and flying + bool(open_batch.items) <= limit
            ):
                found = next(iterator, None)
                if found is None:
                    exhausted = True
                    break
                doc, family = found
                item = [doc, family, position(), None, None, 0]
                if "cached_payload" in doc:
                    item[3] = tokens.decode_ids(doc.pop("cached_payload"))
                    item[5] = 4 * len(item[3])
                else:
                    if "text" in doc and tokens is not None:
                        digest = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
                        item[3] = tokens.lookup(digest, count_miss=False)
                    if item[3] is None:
                        item[5] = (
                            doc["location"][1]
                            if "location" in doc
                            else len(doc["text"].encode("utf-8"))
                        )
                        if (
                            open_batch.items
                            and open_batch.bytes + item[5] > source_bytes
                        ):
                            submit()
                        open_batch.items.append(item)
                        open_batch.bytes += item[5]
                        if (
                            len(open_batch.items) >= documents
                            or open_batch.bytes >= source_bytes
                        ):
                            submit()
                held += item[5]
                window.append(item)
            if not window:
                return
            item = window.popleft()
            if item[3] is None:
                if item[4] is None:
                    submit()
                if item[4].items is not None:
                    finish(item[4])
            held -= item[5]
            metrics["window_peak"] = max(metrics.get("window_peak", 0), len(window) + 1)
            yield item[0], item[1], item[2], item[3]
    finally:
        for item in window:
            if item[4] is not None and item[4].future is not None:
                item[4].future.cancel()


def _encode_here(tokenizer, request):
    """Answer an encoder request in this process (a padded or stand-in tokenizer)."""
    if request[:1] == b"D":
        texts = token_encode_worker.read_texts(
            *token_encode_worker.unpack_documents(request[1:])
        )
    else:
        texts = token_encode_worker.unpack_texts(request[1:])
    encodings = [tokenizer.encode(text, add_special_tokens=False).ids for text in texts]
    counts = np.fromiter(map(len, encodings), dtype=np.uint32, count=len(encodings))
    ids = np.fromiter(
        (token for encoding in encodings for token in encoding),
        dtype=np.int32,
        count=int(counts.sum()),
    )
    frames = (ids.tobytes(), counts.tobytes())
    if request[:1] == b"D":
        frames += (token_encode_worker.text_digests(texts).tobytes(),)
    return frames


class MixtureFrontier:
    """Authenticated append prefixes and immutable index-event segments.

    Restoring SQL acceleration structures replays index deltas only: it does
    not fetch sources, tokenize, make dedup decisions or re-emit schedule rows.
    Uncommitted segment directories are ignored and retained for diagnosis.
    """

    def __init__(self, root, cache, request):
        self.root = root
        self.root.mkdir(exist_ok=True)
        if symlinked(self.root):
            raise ValueError("unsafe mixture checkpoint root")
        self.cache = cache
        if cache.proofs is None or not cache.writable:
            raise ValueError("mixture checkpoint proof storage unavailable")
        self.identity = canonical_digest(request)
        binding = self.root / "request.json"
        if binding.exists():
            if json.loads(binding.read_bytes()) != request:
                raise ValueError(
                    f"mixture checkpoint {self.root} was written for another "
                    "request; delete that directory or use a new output"
                )
        else:
            write_rows(binding, [request])
        self.sequence = 0
        self.state = None
        self.excluded_offset = 0
        self.event_file = None
        self.output_digests = {}

    def restore(self, index, outputs, excluded):
        """Validate the committed files in place and replay narrow index events."""
        generations = {
            name: artifact_fingerprint(output.stream.fileno())
            for name, output in outputs.items()
        }
        for marker in sorted(self.root.glob("commit-*.json")):
            if marker.name != f"commit-{self.sequence:012d}.json":
                raise ValueError("mixture checkpoint sequence gap")
            segment = json.loads(marker.read_bytes())
            expected = {"identity": self.identity, "sequence": self.sequence}
            found = self.cache.load("mixture-frontier", expected, limit=4 * 1024**2)
            if found is None or found[0] != segment:
                raise ValueError(
                    "mixture checkpoint authentication or corruption failure"
                )
            if segment["previous"] != (
                canonical_digest(self.state) if self.state else None
            ):
                raise ValueError("mixture checkpoint frontier chain mismatch")
            parts = segment["outputs"]
            if parts.keys() != outputs.keys():
                raise ValueError("mixture checkpoint output inventory mismatch")
            for name, output in outputs.items():
                digest = output.validated_prefix(parts[name])
                if digest is None:
                    raise ValueError(
                        f"mixture checkpoint output prefix changed: {name}"
                    )
                output.whole = digest
                output.offset = parts[name]["end"]
            directory = Path(segment["directory"])
            for event in rows(directory / "index.jsonl"):
                index.restore_event(event)
            excluded.extend(rows(directory / "excluded.jsonl"))
            self.state = segment
            self.sequence += 1
        for name, output in outputs.items():
            if artifact_fingerprint(output.stream.fileno()) != generations[name]:
                raise ValueError(f"mixture checkpoint changed during recovery: {name}")
            output.stream.seek(output.offset)
            if os.fstat(output.stream.fileno()).st_size != output.offset:
                output.stream.truncate()
                os.fsync(output.stream.fileno())
        self.excluded_offset = len(excluded)

    def pending_bytes(self, outputs):
        return sum(output.stream.tell() - output.offset for output in outputs.values())

    def begin(self, index):
        from uuid import uuid4

        self.directory = self.root / ("segment-" + uuid4().hex)
        self.directory.mkdir()
        self.event_file = (self.directory / "index.jsonl").open("xb")
        index.event_sink = lambda event: self.event_file.write(
            canonical_json(event) + b"\n"
        )

    def commit(self, state, outputs, excluded, index):
        parts = {name: output.seal() for name, output in outputs.items()}
        if self.sequence == 0:
            for directory in {
                Path(output.stream.name).parent for output in outputs.values()
            }:
                sync_directory(directory)
            sync_directory(self.root.parent)
        write_rows(self.directory / "excluded.jsonl", excluded[self.excluded_offset :])
        self.event_file.flush()
        os.fsync(self.event_file.fileno())
        self.event_file.close()
        sync_directory(self.directory)
        entry = {
            "previous": canonical_digest(self.state) if self.state else None,
            "directory": str(self.directory),
            "state": state,
            "outputs": parts,
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
        for name, output in outputs.items():
            output.offset = parts[name]["end"]
        self.excluded_offset = len(excluded)
        self.begin(index)

    def close(self, index):
        index.event_sink = None
        if self.event_file is not None:
            self.event_file.close()


def producer_digests():
    """Digests of the source that produces a mixture; a changed producer is a new checkpoint."""
    producer = {
        name: sha256_file(Path(__file__).with_name(name))
        for name in (
            "mixture.py",
            "mixture_index.py",
            "mixture_pipeline.py",
            "retained_mixture.py",
            "view_checkpoints.py",
        )
    }
    producer.update(
        {
            "data/" + name: sha256_file(Path(__file__).parents[1] / "data" / name)
            for name in (
                "encoding.py",
                "token_segments.py",
                "token_store.py",
                "tokenizer.py",
            )
        }
    )
    producer["token_encode_worker.py"] = sha256_file(
        Path(__file__).parents[1] / "token_encode_worker.py"
    )
    return producer


def claim_assembly(output, request):
    """The checkpoint directory for exactly this request, checked before any work.

    The directory is named by the request and the producer's source, so another
    release, declaration, inventory, tokenizer or code revision never meets an
    unrelated checkpoint. Assemblies of other requests beside the same output are
    reported so the operator can delete the ones they will not resume.
    """
    output = Path(output).absolute()
    request = {**request, "producer": producer_digests()}
    prefix = "." + output.name + ".assembly"
    root = output.parent / f"{prefix}-{canonical_digest(request)[:16]}"
    if symlinked(root):
        raise ValueError(f"unsafe mixture checkpoint root {root}")
    binding = root / "request.json"
    if binding.exists():
        stored = json.loads(binding.read_bytes())
        if any(stored.get(key) != value for key, value in request.items()):
            raise ValueError(
                f"mixture checkpoint {root} was written for another request; "
                "delete that directory or use a new output"
            )
    others = sorted(
        path.name
        for path in output.parent.glob(prefix + "*")
        if path.is_dir() and path != root
    )
    if others:
        import sys

        print(
            f"mixture: {len(others)} checkpoint(s) beside {output.name} belong to "
            f"other requests and will not be resumed ({', '.join(others)}); "
            "delete the ones you no longer need",
            file=sys.stderr,
            flush=True,
        )
    return root


def pipeline_request(request, index):
    producer = producer_digests()
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
    if symlinked(output):
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
    """Reuse a verified selection index of one release, inventory, seed and order policy.

    The key is what the index is computed from, not the mixture declaration:
    changing a quota, the tokenizer or the output keeps the index. Mutable
    schedule state is never part of it.
    """

    def __init__(self, cache, digests, *, seed, order_policy, strata):
        """`digests` maps the documents, inventory and TRAIN lineage files to SHA-256."""
        self.cache = cache
        self.request = {
            "inputs": {str(p): digest for p, digest in digests.items()},
            "seed": seed,
            "order_policy": order_policy,
            "source_strata": strata,
        }
        prior = cache.load("mixture-preflight", self.request, limit=4 * 1024**2)
        self.snapshot = prior[0]["snapshot"] if prior is not None else None

    def publish(self, index):
        import shutil
        from uuid import uuid4

        directory = self.cache.root / "cache" / "mixture-preflight"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / uuid4().hex
        index.verify_inputs()
        shutil.copytree(index.selection.directory, path)
        for member in path.iterdir():
            with member.open("rb") as stream:
                os.fsync(stream.fileno())
        sync_directory(path)
        sync_directory(directory)
        self.cache.publish(
            "mixture-preflight",
            self.request,
            {"snapshot": str(path)},
            capture([path]),
            limit=4 * 1024**2,
        )


def process_tree_sample(pids):
    """Sample the given worker processes without scanning every process on the
    host; a disappearing worker does not fail the work."""
    import psutil

    from klab.corpus.progress import memory_bytes

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
    for pid in pids:
        try:
            child = psutil.Process(pid)
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
