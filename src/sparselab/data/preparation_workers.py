"""Bounded native tokenization workers with exclusive chunks and ordered adoption."""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from threading import Event

from tokenizers import Tokenizer

from sparselab.data.encoding import PreparationEncoder, tokenizer_work_plan
from sparselab.data.preparation_telemetry import PreparationTelemetry
from sparselab.progress import progress_phase
from sparselab.resource_envelope import check_envelope, current_process_rss_bytes


class PreparationWorkers:
    def __init__(self, tokenizer, root, *, envelope, parallel_supported):
        self.plan = tokenizer_work_plan(
            None if envelope is None else envelope.max_workers
        )
        self.requested = 1 if envelope is None else envelope.preparation_shard_workers
        queue_cap = None if envelope is None else envelope.max_queue_depth
        self.count = min(self.requested, self.plan.workers, queue_cap or self.requested)
        self.fallback = (
            "worker_or_queue_capacity" if self.count < self.requested else None
        )
        if not parallel_supported:
            self.count = 1
            self.fallback = "tokenizer_not_parallel_eligible"
        self.threads = [
            self.plan.workers // self.count + (i < self.plan.workers % self.count)
            for i in range(self.count)
        ]
        self.root = root
        self.envelope = envelope
        self.tokenizer_json = tokenizer.to_str()
        self.cancelled = Event()
        self.stack = ExitStack()
        self.encoders = []
        self.models = []
        self.busy = [0.0] * self.count
        self.started = time.monotonic()

    def __enter__(self):
        try:
            for threads in self.threads:
                model = Tokenizer.from_str(self.tokenizer_json)
                self.models.append(model)
                self.encoders.append(
                    self.stack.enter_context(
                        PreparationEncoder(model, self.root, max_workers=threads)
                    )
                )
            self.executor = self.stack.enter_context(
                ThreadPoolExecutor(
                    max_workers=self.count, thread_name_prefix="prepare-chunk"
                )
            )
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exc):
        self.cancelled.set()
        self.stack.__exit__(*exc)

    def snapshot(self):
        elapsed = max(time.monotonic() - self.started, 1e-9)
        return {
            "requested_workers": self.requested,
            "effective_workers": self.count,
            "fallback": self.fallback,
            "host_work_plan": asdict(self.plan),
            "elapsed_seconds": elapsed,
            "workers": [
                {
                    "worker": i,
                    "rayon_threads": encoder.rayon_threads,
                    "batches": encoder.batch_count,
                    "request_bytes": encoder.request_bytes,
                    "reply_bytes": encoder.reply_bytes,
                    "ipc_wait_seconds": encoder.wait_seconds,
                    "child_cpu_seconds": encoder.sample_cpu(),
                    "child_cpu_utilization": encoder.child_cpu_seconds
                    / (elapsed * encoder.rayon_threads),
                    "busy_seconds": self.busy[i] if self.count > 1 else None,
                }
                for i, encoder in enumerate(self.encoders)
            ],
        }

    def write_work(
        self,
        slot,
        chunks,
        sequence,
        records,
        transform,
        batch_documents,
        batch_source_bytes,
    ):
        started = time.monotonic()
        measured = PreparationTelemetry(self.root)
        try:
            if self.cancelled.is_set():
                raise RuntimeError("preparation workers cancelled")
            with chunks.owner.work_chunks(
                chunks.split, sequence, telemetry=measured
            ) as owner:
                work = owner.open_split(chunks.split)
                pending = []
                pending_bytes = 0

                def encode_pending():
                    nonlocal pending_bytes
                    if not pending:
                        return
                    texts = [record.text for _, record in pending if record.text]
                    encoded_at = time.monotonic()
                    encodings = iter(self.encoders[slot].encode(texts))
                    measured.add(
                        "tokenizer_encoding_seconds", time.monotonic() - encoded_at
                    )
                    for ordinal, record in pending:
                        if self.cancelled.is_set():
                            raise RuntimeError("preparation workers cancelled")
                        bookkeeping_at = time.monotonic()
                        values = (
                            transform(self.models[slot], next(encodings), record.text)
                            if record.text
                            else None
                        )
                        work.append_record(ordinal, values)
                        measured.add(
                            "python_bookkeeping_seconds",
                            time.monotonic() - bookkeeping_at,
                        )
                    pending.clear()
                    pending_bytes = 0

                for ordinal, record in enumerate(records, 1):
                    size = len(record.text.encode("utf-8"))
                    if pending and (
                        len(pending) == batch_documents
                        or pending_bytes + size > batch_source_bytes
                    ):
                        encode_pending()
                    pending.append((ordinal, record))
                    pending_bytes += size
                encode_pending()
                work.seal()
            return work, measured
        except BaseException:
            self.cancelled.set()
            raise
        finally:
            self.busy[slot] += time.monotonic() - started


def collect_parallel_stream(
    documents,
    chunks,
    workers,
    *,
    selected_documents,
    max_tokens,
    batch_documents,
    batch_source_bytes,
    transform,
    telemetry,
):
    """Select in one coordinator; reserve conservative token bounds before reading ahead.

    Only committed receipts advance the recovery frontier. At most one bounded work item per
    worker is outstanding, including completed work waiting for an earlier batch.
    """
    state = chunks.state
    telemetry.output_tokens += state["output_tokens"]
    acquired = state["acquired_documents"]
    selected = state["retained_documents"]
    source = iter(documents)
    for ordinal in range(acquired):
        read_at = time.monotonic()
        try:
            record = next(source)
        except StopIteration as error:
            raise ValueError(
                f"{chunks.split} source ended before recovered record {ordinal + 1}"
            ) from error
        telemetry.add("source_iteration_seconds", time.monotonic() - read_at)
        size = len(record.text.encode("utf-8"))
        telemetry.records += 1
        telemetry.source_bytes += size
        telemetry.logical_input_bytes += size
    # Amortize receipts/fsync over native chunks, not individual encoder calls.
    # One bounded coordinator buffer may coexist with one work item per worker.
    work_documents = chunks.owner.record_limit
    work_source_bytes = min(
        8_388_608,
        max(
            batch_source_bytes,
            chunks.owner.raw_byte_limit
            // sum(dtype.itemsize for dtype in chunks.owner.arrays.values()),
        ),
    )
    pending = deque()
    free = deque(range(workers.count))
    batch = []
    batch_bytes = 0
    batch_upper = 0
    reserved = 0
    sequence = 0
    queue_peak = 0
    queue_wait = 0.0

    def drain():
        nonlocal reserved, queue_wait
        future, slot, upper, first = pending.popleft()
        waited = time.monotonic()
        work, measured = future.result()
        if work.state["output_tokens"] > upper:
            raise ValueError("tokenizer violated the UTF-8 output reservation")
        queue_wait += time.monotonic() - waited
        if chunks.state["output_tokens"] + work.state["output_tokens"] > max_tokens:
            raise ValueError(f"{chunks.split} token cap would truncate selected story")
        committed_at = time.monotonic()
        chunks.adopt(work, first_acquired_index=first)
        telemetry.add("finalize_fsync_seconds", time.monotonic() - committed_at)
        for name, elapsed in measured.durations.items():
            telemetry.add(name, elapsed)
        telemetry.output_tokens += work.state["output_tokens"]
        reserved -= upper
        free.append(slot)

    def submit():
        nonlocal batch, batch_bytes, batch_upper, reserved, sequence, queue_peak
        if not batch:
            return
        if not free:
            drain()
        if workers.envelope is not None:
            check_envelope(
                workers.envelope,
                workspace=chunks.root,
                rss_bytes=current_process_rss_bytes(),
                pending_workers=len(pending) + 1,
                queue_depth=len(pending) + 1,
            )
        slot = free.popleft()
        first = acquired - len(batch) + 1
        future = workers.executor.submit(
            workers.write_work,
            slot,
            chunks,
            sequence,
            batch,
            transform,
            batch_documents,
            batch_source_bytes,
        )
        pending.append((future, slot, batch_upper, first))
        reserved += batch_upper
        queue_peak = max(queue_peak, len(pending))
        sequence += 1
        batch, batch_bytes, batch_upper = [], 0, 0

    with progress_phase(
        f"data_{chunks.split}_document_iteration_and_collection",
        completed_work=selected,
        total_work=selected_documents,
        unit="documents",
        raw_counters=telemetry.snapshot(),
    ) as progress:
        try:
            while selected < selected_documents:
                if workers.cancelled.is_set():
                    # Surface the original worker error instead of a cancellation wrapper.
                    for future, *_ in pending:
                        if future.done():
                            future.result()
                    raise RuntimeError("preparation workers cancelled")
                read_at = time.monotonic()
                try:
                    record = next(source)
                except StopIteration:
                    break
                finally:
                    telemetry.add(
                        "source_iteration_seconds", time.monotonic() - read_at
                    )
                size = len(record.text.encode("utf-8"))
                if size > batch_source_bytes:
                    raise ValueError(
                        f"{chunks.split} source document exceeds tokenizer_batch_source_bytes"
                    )
                upper = size + 1 if record.text else 0
                # Submit before incrementing acquired: that batch ends at the
                # previous record. A byte bound can be conservative, so an
                # oversized single reservation is resolved before reading further.
                if batch and (
                    len(batch) == work_documents
                    or batch_bytes + size > work_source_bytes
                    or chunks.state["output_tokens"] + reserved + batch_upper + upper
                    > max_tokens
                ):
                    submit()
                while (
                    pending
                    and chunks.state["output_tokens"] + reserved + batch_upper + upper
                    > max_tokens
                ):
                    drain()
                acquired += 1
                telemetry.records += 1
                telemetry.source_bytes += size
                telemetry.logical_input_bytes += size
                selected += bool(record.text)
                batch.append(record)
                batch_bytes += size
                batch_upper += upper
                if (
                    len(batch) == work_documents
                    or batch_bytes == work_source_bytes
                    or chunks.state["output_tokens"] + reserved + batch_upper
                    >= max_tokens
                ):
                    submit()
                    if chunks.state["output_tokens"] + reserved >= max_tokens:
                        while pending:
                            drain()
                if not batch:
                    progress.update(
                        completed_work=chunks.state["retained_documents"],
                        total_work=selected_documents,
                        unit="documents",
                        raw_counters={
                            **telemetry.snapshot(),
                            "pending_batches": len(pending),
                            "queue_peak_work_items": queue_peak,
                            "queue_wait_seconds": queue_wait,
                        },
                    )
            submit()
            while pending:
                drain()
            if chunks.state["retained_documents"] != selected_documents:
                raise ValueError(
                    f"{chunks.split} source has fewer selected distinct stories than required"
                )
            proofs, stats = chunks.finish()
            progress.update(
                completed_work=selected_documents,
                total_work=selected_documents,
                unit="documents",
                raw_counters={
                    **telemetry.snapshot(),
                    **stats,
                    "queue_peak_work_items": queue_peak,
                    "queue_wait_seconds": queue_wait,
                },
            )
            return (
                stats,
                proofs,
                {
                    "queue_peak_work_items": queue_peak,
                    "queue_wait_seconds": queue_wait,
                    "work_item_record_limit": work_documents,
                    "work_item_source_byte_limit": work_source_bytes,
                },
            )
        except BaseException:
            workers.cancelled.set()
            for future, *_ in pending:
                future.cancel()
            progress.update(
                completed_work=chunks.state["retained_documents"],
                total_work=selected_documents,
                unit="documents",
                raw_counters={
                    **telemetry.snapshot(),
                    "worker_pool": workers.snapshot(),
                    "queue_peak_work_items": queue_peak,
                    "queue_wait_seconds": queue_wait,
                },
            )
            raise
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()
