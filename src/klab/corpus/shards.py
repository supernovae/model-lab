"""The one way a corpus stage runs in parallel: plan shards, run them, merge in order.

A stage cuts its input into shards (line-aligned byte ranges of a JSONL file),
declares itself `shardable` or `Single(reason)`, and runs a worker function over
the shards in a bounded process pool. Results come back in shard order when a
merge needs the serial order, so the output does not depend on the worker count.
With a `store` each shard result is written under a key derived from the stage
inputs and the shard, so a rerun reuses finished shards and a killed run resumes.
The operator's `--workers` choice lives in `release_verify.verification_settings`.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing
import os
import resource
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from threading import Thread
from typing import Any

from klab.corpus.build_stages import StageStore, run_stage
from klab.corpus.release_verify import current_settings, worker_count
from klab.host_capacity import plan_host_workers

CHUNK_BYTES = 64 * 1024**2
# Below this much row data a worker pool costs more than it saves.
PARALLEL_MIN_BYTES = 256 * 1024**2
_COORDINATOR_RESERVE = 1024**3
_FILE_STAGE = "shard"


@dataclass(frozen=True)
class Single:
    """A stage that cannot be sharded, with the reason it cannot."""

    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("a single-process stage must say why it cannot be sharded")


STAGES: dict[str, str | Single] = {}


def declare_stage(name: str, parallelism: str | Single = "shardable") -> str:
    """Register a stage; the stage table in docs/corpus-preparation.md must list it."""
    if parallelism != "shardable" and not isinstance(parallelism, Single):
        raise ValueError("parallelism is 'shardable' or Single(reason)")
    STAGES[name] = parallelism
    return name


@dataclass(frozen=True)
class ShardPlan:
    """One stage's shards, in the order a serial pass would visit them."""

    stage: str
    tasks: list[tuple]
    total_bytes: int
    inputs: dict[str, Any] = field(default_factory=dict)
    store: Path | None = None
    worker_memory_bytes: int = 512 * 1024**2
    threaded: bool = False
    operator_cap: int | None = None
    min_parallel_bytes: int | None = None


def chunk_ranges(path: Path) -> list[tuple[int, int]]:
    """Split a file into byte ranges that each start and end on a line boundary."""
    size = path.stat().st_size
    ranges, start = [], 0
    with path.open("rb") as stream:
        while start < size:
            stream.seek(min(start + CHUNK_BYTES, size))
            stream.readline()
            end = stream.tell()
            ranges.append((start, end))
            start = end
    return ranges


def range_file_identity(path: Path) -> list[int]:
    """Bind range offsets to a specific local generation of the input file."""
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def plan_ranges(
    stage: str,
    path: Path,
    *extra: Any,
    inputs: dict[str, Any] | None = None,
) -> ShardPlan:
    """One shard per line-aligned byte range of `path`, called `(path, start, end, *extra)`.

    Results are kept under the operator's scratch directory, if they chose one,
    keyed by the stage inputs and a fingerprint of the file.
    """
    identity = range_file_identity(path)
    scratch = current_settings().scratch_dir
    return ShardPlan(
        stage=stage,
        tasks=[(str(path), start, end, *extra) for start, end in chunk_ranges(path)],
        total_bytes=identity[2],
        inputs={**(inputs or {}), "file": identity},
        store=None if scratch is None else scratch / "shards",
    )


def _key(plan: ShardPlan, task: tuple) -> str:
    material = json.dumps(
        [plan.stage, plan.inputs, task], sort_keys=True, default=str
    ).encode()
    return hashlib.sha256(material).hexdigest()


def _receipt(path: Path) -> Any | None:
    """A stored shard result, or None when it is absent or does not verify."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        body = json.dumps(record["result"], separators=(",", ":"))
        if hashlib.sha256(body.encode()).hexdigest() == record["sha256"]:
            return record["result"]
    except OSError, ValueError, KeyError, TypeError:
        pass
    return None


def _commit(path: Path, result: Any) -> None:
    body = json.dumps(result, separators=(",", ":"))
    record = {"sha256": hashlib.sha256(body.encode()).hexdigest(), "result": result}
    partial = path.with_name(f".incomplete-{os.getpid()}-{path.name}")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


@contextmanager
def _workers_without_main():
    """Start workers without re-running the caller's `__main__` script.

    Spawned workers only need this package's importable functions. Re-running an
    unguarded script (or a test runner's bootstrap file) in each worker would
    repeat its work or fail, so the main module is hidden while the pool lives.
    """
    main = sys.modules.get("__main__")
    file, spec = getattr(main, "__file__", None), getattr(main, "__spec__", None)
    main.__spec__ = None
    if file is not None:
        del main.__file__
    try:
        yield
    finally:
        main.__spec__ = spec
        if file is not None:
            main.__file__ = file


def _cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def _limit_native_threads() -> None:
    # Spawned workers initialize these native libraries after this runs.
    os.environ.update(
        dict.fromkeys(
            (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "RAYON_NUM_THREADS",
            ),
            "1",
        )
    )

    parent = multiprocessing.parent_process()
    if parent is not None:

        def exit_with_coordinator() -> None:
            parent.join()
            os._exit(1)

        # A hard stop cannot shut down the pool. Waiting on the coordinator's
        # process sentinel also wakes workers blocked on an empty work queue.
        Thread(target=exit_with_coordinator, daemon=True).start()


@contextmanager
def _worker_pool(workers: int, *, threaded: bool):
    if threaded:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield pool
        return
    with (
        _workers_without_main(),
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_limit_native_threads,
        ) as pool,
    ):
        yield pool


def _pool_results(
    function: Callable[..., Any],
    jobs: Iterator[tuple[tuple, Any]],
    workers: int,
    ordered: bool,
    *,
    threaded: bool,
    on_discarded: Callable[[Any], None] | None,
) -> Iterator[tuple[tuple, Any, bool]]:
    """Keep at most one running, completed or cached result per slot."""
    pending: dict[Future, tuple[tuple, bool]] = {}
    try:
        with _worker_pool(workers, threaded=threaded) as pool:
            try:
                while True:
                    for task, cached in jobs:
                        if cached is None:
                            future = pool.submit(function, *task)
                        else:
                            future = Future()
                            future.set_result(cached)
                        pending[future] = (task, cached is not None)
                        del cached
                        if len(pending) == workers:
                            break
                    if not pending:
                        return
                    if len(pending) == 1 or ordered:
                        ready = (next(iter(pending)),)
                    else:
                        ready = wait(pending, return_when=FIRST_COMPLETED).done
                    for future in ready:
                        task, reused = pending.pop(future)
                        yield task, future.result(), reused
                    # Futures retain their result even after they leave the window.
                    del ready, future
            finally:
                for future in pending:
                    future.cancel()
    finally:
        if on_discarded is not None:
            for future in pending:
                if (
                    future.done()
                    and not future.cancelled()
                    and future.exception() is None
                ):
                    on_discarded(future.result())


def _run_sharded(
    plan: ShardPlan,
    function: Callable[..., Any],
    load: Callable[[tuple], Any],
    save: Callable[[tuple, Any], None] | None,
    *,
    ordered: bool,
    on_discarded: Callable[[Any], None] | None = None,
) -> Iterator[Any]:
    if plan.stage not in STAGES:
        raise ValueError(f"stage {plan.stage!r} is not declared with declare_stage")
    declared = STAGES[plan.stage]
    workers = min(worker_count(), len(plan.tasks))
    if plan.operator_cap is not None:
        workers = min(workers, plan.operator_cap)
    threshold = (
        PARALLEL_MIN_BYTES
        if plan.min_parallel_bytes is None
        else plan.min_parallel_bytes
    )
    if (
        workers < 2
        or isinstance(declared, Single)
        or (not plan.threaded and plan.total_bytes < threshold)
    ):
        workers = 1
    else:
        workers = plan_host_workers(
            plan.stage,
            worker_memory_bytes=plan.worker_memory_bytes,
            reserve_bytes=_COORDINATOR_RESERVE,
            operator_cap=workers,
        ).workers
    echo = current_settings().echo
    if echo is not None:
        note = f" ({declared.reason})" if isinstance(declared, Single) else ""
        print(
            f"{plan.stage}: {len(plan.tasks)} shards, up to {workers} workers{note}",
            file=echo,
            flush=True,
        )
    started, cpu = time.monotonic(), _cpu_seconds()
    reused = executed = 0
    fault = os.environ.get("KLAB_FAULT", "")
    fault_stage, _, fault_after = fault.rpartition(":")
    jobs = ((task, load(task)) for task in plan.tasks)
    results = (
        (
            (task, function(*task), False) if cached is None else (task, cached, True)
            for task, cached in jobs
        )
        if workers == 1
        else _pool_results(
            function,
            jobs,
            workers,
            ordered,
            threaded=plan.threaded,
            on_discarded=on_discarded,
        )
    )
    try:
        for task, result, from_cache in results:
            if from_cache:
                reused += 1
            elif plan.store is not None:
                if save is not None:
                    save(task, result)
                executed += 1
                if fault_stage == plan.stage and executed == int(fault_after):
                    raise RuntimeError(f"KLAB_FAULT after shard {executed}")
            yield result
            del result
    finally:
        results.close()
        if echo is not None:
            wall = max(time.monotonic() - started, 1e-9)
            used = _cpu_seconds() - cpu
            reuse = (
                f", reused {reused}/{len(plan.tasks)}" if plan.store is not None else ""
            )
            print(
                f"{plan.stage}: {wall:.1f}s wall, {used:.1f}s cpu, "
                f"{used / wall:.1f} cores used{reuse}",
                file=echo,
                flush=True,
            )


def run_sharded(
    plan: ShardPlan,
    function: Callable[..., Any],
    *,
    ordered: bool = False,
    on_discarded: Callable[[Any], None] | None = None,
) -> Iterator[Any]:
    """Yield bounded results and account for completed lookahead on close.

    The optional callback receives successful results the consumer did not use.
    They are reported after the pool joins, without saving them to the store.
    """
    store = plan.store
    if store is not None:
        store.mkdir(parents=True, exist_ok=True)

    def load(task: tuple) -> Any:
        return None if store is None else _receipt(store / f"{_key(plan, task)}.json")

    def save(task: tuple, result: Any) -> None:
        assert store is not None
        _commit(store / f"{_key(plan, task)}.json", result)

    yield from _run_sharded(
        plan, function, load, save, ordered=ordered, on_discarded=on_discarded
    )


def _file_inputs(stage: str, inputs: dict[str, Any], task: tuple) -> dict[str, Any]:
    return {"stage": stage, "inputs": inputs, "task": task}


def _produce_files(
    store: StageStore,
    stage: str,
    inputs: dict[str, Any],
    function: Callable[..., dict[str, Any]],
    *task: Any,
) -> tuple[Path, dict[str, Any]]:
    shard_inputs = _file_inputs(stage, inputs, task)
    locks = store.root / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / store.key(shard_inputs)).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        def produce(staging: Path, result: dict[str, Any]) -> None:
            result.update(function(staging, *task))

        return run_stage(store, _FILE_STAGE, shard_inputs, produce)


def run_sharded_files(
    plan: ShardPlan, function: Callable[..., dict[str, Any]], *, ordered: bool = False
) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Run a worker with a private output directory and yield verified file references.

    The plan must bind the producer identity in its inputs. Workers return small
    JSON metadata. File bytes stay in the store and are verified before reuse.
    """
    if plan.store is None:
        raise ValueError("file-backed shards need a store")
    store = StageStore(plan.store / "files")

    def load(task: tuple) -> tuple[Path, dict[str, Any]] | None:
        inputs = _file_inputs(plan.stage, plan.inputs, task)
        receipt = store.verified(_FILE_STAGE, inputs)
        if receipt is None:
            return None
        return store.directory(_FILE_STAGE, receipt["key"]), receipt["result"]

    worker = partial(_produce_files, store, plan.stage, plan.inputs, function)
    yield from _run_sharded(plan, worker, load, None, ordered=ordered)


declare_stage("release verification")


def run_tasks(
    function: Callable[..., Any],
    tasks: Any,
    consume: Callable[[Any], None],
    *,
    total_bytes: int,
) -> None:
    """Run release-verification tasks in the pool, consuming results as they land."""
    plan = ShardPlan("release verification", list(tasks), total_bytes)
    for result in run_sharded(plan, function):
        consume(result)
