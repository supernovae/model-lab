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

import hashlib
import json
import multiprocessing
import os
import resource
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from klab.corpus.release_verify import current_settings, worker_count

CHUNK_BYTES = 64 * 1024**2
# Below this much row data a worker pool costs more than it saves.
PARALLEL_MIN_BYTES = 256 * 1024**2


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


def plan_ranges(
    stage: str,
    path: Path,
    *extra: Any,
    inputs: dict[str, Any] | None = None,
) -> ShardPlan:
    """One shard per line-aligned byte range of `path`, called `(path, start, end, *extra)`.

    Results are kept under the operator's scratch directory, if they chose one,
    keyed by the stage inputs and the file's size and modification time.
    """
    stat = path.stat()
    scratch = current_settings().scratch_dir
    return ShardPlan(
        stage=stage,
        tasks=[(str(path), start, end, *extra) for start, end in chunk_ranges(path)],
        total_bytes=stat.st_size,
        inputs={
            **(inputs or {}),
            "file": [path.name, stat.st_size, stat.st_mtime_ns],
        },
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


def _pool_results(
    function: Callable[..., Any], tasks: list[tuple], workers: int, ordered: bool
) -> Iterator[tuple[int, Any]]:
    """Yield `(task index, result)` from a bounded pool, two tasks per worker in flight."""
    pending: deque = deque()
    queue = iter(enumerate(tasks))
    context = multiprocessing.get_context("spawn")
    with (
        _workers_without_main(),
        ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool,
    ):
        try:
            while True:
                while len(pending) < workers * 2:
                    item = next(queue, None)
                    if item is None:
                        break
                    pending.append((item[0], pool.submit(function, *item[1])))
                if not pending:
                    return
                if ordered:
                    index, future = pending.popleft()
                    yield index, future.result()
                    continue
                done, _ = wait([f for _, f in pending], return_when=FIRST_COMPLETED)
                for index, future in [p for p in pending if p[1] in done]:
                    pending.remove((index, future))
                    yield index, future.result()
        finally:
            for _, future in pending:
                future.cancel()


def run_sharded(
    plan: ShardPlan, function: Callable[..., Any], *, ordered: bool = False
) -> Iterator[Any]:
    """Run a declared stage's shards and yield each result.

    With `ordered`, results are yielded in shard order so a caller can merge them
    exactly as a serial loop would; otherwise as they finish. Runs inline for one
    worker, a `Single` stage or a small input. Stops early, cancelling the rest,
    if the caller stops iterating. The operator sees the workers used at the
    start and, at the end, wall time, CPU time, cores used and reused shards.
    `KLAB_FAULT=<stage>:<n>` (tests) raises after the n-th stored shard.
    """
    if plan.stage not in STAGES:
        raise ValueError(f"stage {plan.stage!r} is not declared with declare_stage")
    declared = STAGES[plan.stage]
    workers = 1 if isinstance(declared, Single) else worker_count()
    if plan.total_bytes < PARALLEL_MIN_BYTES:
        workers = 1
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
    keys = [None if plan.store is None else _key(plan, task) for task in plan.tasks]
    if plan.store is not None:
        plan.store.mkdir(parents=True, exist_ok=True)
    fault = os.environ.get("KLAB_FAULT", "")
    fault_stage, _, fault_after = fault.rpartition(":")
    stored: dict[int, Any] = {}
    todo: list[int] = []
    for index, key in enumerate(keys):
        found = None if key is None else _receipt(plan.store / f"{key}.json")
        if found is None:
            todo.append(index)
        else:
            stored[index] = found
            reused += 1

    def computed() -> Iterator[tuple[int, Any]]:
        tasks = [plan.tasks[i] for i in todo]
        if workers == 1:
            for position, task in enumerate(tasks):
                yield todo[position], function(*task)
            return
        for position, result in _pool_results(function, tasks, workers, ordered):
            yield todo[position], result

    def finish() -> None:
        if echo is None:
            return
        wall = max(time.monotonic() - started, 1e-9)
        used = _cpu_seconds() - cpu
        print(
            f"{plan.stage}: {wall:.1f}s wall, {used:.1f}s cpu, "
            f"{used / wall:.1f} cores used, reused {reused}/{len(plan.tasks)}",
            file=echo,
            flush=True,
        )

    results = computed()
    fresh: dict[int, Any] = {}
    next_index = 0
    try:
        for index, result in results:
            executed += 1
            if keys[index] is not None:
                _commit(plan.store / f"{keys[index]}.json", result)
                if fault_stage == plan.stage and executed == int(fault_after):
                    raise RuntimeError(f"KLAB_FAULT after shard {executed}")
            if not ordered:
                yield result
                continue
            fresh[index] = result
            while True:
                while next_index in stored:
                    yield stored.pop(next_index)
                    next_index += 1
                if next_index not in fresh:
                    break
                yield fresh.pop(next_index)
                next_index += 1
        if ordered:
            while next_index in stored:
                yield stored.pop(next_index)
                next_index += 1
        else:
            for result in stored.values():
                yield result
    finally:
        results.close()
        finish()


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
