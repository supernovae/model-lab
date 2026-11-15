"""Bounded, parallel and observable machinery for verifying immutable releases.

Verification never changes what is checked, only how: files are hashed
concurrently, large JSONL files are split into line-aligned byte ranges that
worker processes read independently, and the one disposable index lives under
an operator-chosen scratch directory. Nothing here is evidence.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from klab.corpus.progress import BuildProgress

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore, VerificationMode

CHUNK_BYTES = 64 * 1024**2
# Below this much row data a worker pool costs more than it saves.
PARALLEL_MIN_BYTES = 256 * 1024**2
DEFAULT_WORKERS = 8


@dataclass(frozen=True)
class VerificationSettings:
    """Operator choices that change speed and placement, never the verdict."""

    scratch_dir: Path | None = None
    workers: int = DEFAULT_WORKERS
    echo: TextIO | None = None
    proof_store: ProofStore | None = None
    verification_mode: VerificationMode = "cold"


_settings: ContextVar[VerificationSettings | None] = ContextVar(
    "release_verification_settings", default=None
)


def current_settings() -> VerificationSettings:
    return _settings.get() or VerificationSettings()


@contextmanager
def verification_settings(
    *,
    scratch_dir: Path | None = None,
    workers: int | None = None,
    echo: TextIO | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode | None = None,
):
    """Select scratch, parallelism, live progress and receipt reuse for nested verifiers.

    `proof_store` and `verification_mode` take the keywords of
    `verification_options`; a `verify_release` call that passes neither uses them.
    """
    if workers is not None and workers < 1:
        raise ValueError("verification workers must be at least 1")
    current = current_settings()
    token = _settings.set(
        VerificationSettings(
            scratch_dir=Path(scratch_dir).resolve()
            if scratch_dir is not None
            else current.scratch_dir,
            workers=workers if workers is not None else current.workers,
            echo=echo if echo is not None else current.echo,
            proof_store=proof_store if proof_store is not None else current.proof_store,
            verification_mode=verification_mode or current.verification_mode,
        )
    )
    try:
        yield
    finally:
        _settings.reset(token)


def scratch_root() -> Path:
    """Directory for disposable verification state, created when it is first used."""
    chosen = current_settings().scratch_dir
    root = chosen if chosen is not None else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return root


def worker_count() -> int:
    return max(1, min(current_settings().workers, os.cpu_count() or 1))


def open_progress(release_id: str) -> BuildProgress:
    """Journal verification progress next to the scratch it uses, and echo it live."""
    settings = current_settings()
    return BuildProgress(
        scratch_root() / f"verify-{release_id[:16]}.progress.jsonl",
        release_id,
        interval_seconds=10.0,
        echo=settings.echo,
    )


def hash_inventory(
    root: Path,
    inventory: dict[str, Any],
    resolve: Callable[[Path, str], Path],
    progress: BuildProgress | None = None,
    skip: frozenset[str] = frozenset(),
) -> None:
    """Check every inventoried file's size and SHA-256, concurrently.

    Files in `skip` were hashed by the caller while writing them, so only their
    presence and size are rechecked.
    """

    lock = threading.Lock()

    def advance(size: int) -> None:
        if progress is not None:
            with lock:
                progress.update(input_bytes=size)

    def check(name: str) -> None:
        path = resolve(root, name)
        entry = inventory[name]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"tampered artifact: {name}")
        if name in skip:
            advance(entry["size"])
            return
        # sha256_file has no byte callback; a multi-gigabyte file must not go
        # minutes without a progress line.
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024**2):
                digest.update(block)
                advance(len(block))
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"tampered artifact: {name}")

    names = sorted(inventory, key=lambda name: -inventory[name]["size"])
    with ThreadPoolExecutor(max_workers=worker_count()) as pool:
        list(pool.map(check, names))


def chunk_ranges(path: Path) -> list[tuple[int, int]]:
    """Split a file into byte ranges that each start and end on a line boundary."""
    target_bytes = CHUNK_BYTES
    size = path.stat().st_size
    ranges, start = [], 0
    with path.open("rb") as stream:
        while start < size:
            stream.seek(min(start + target_bytes, size))
            stream.readline()
            end = stream.tell()
            ranges.append((start, end))
            start = end
    return ranges


def range_records(path: Path, start: int, end: int) -> Iterator[tuple[Any, int, int]]:
    """Yield (value, byte offset, byte length) for each nonblank JSON line in a range."""
    with path.open("rb") as stream:
        stream.seek(start)
        offset = start
        for raw in stream:
            if offset >= end:
                return
            position, offset = offset, offset + len(raw)
            if not raw.strip():
                continue
            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"invalid UTF-8 in {path} at byte {position}"
                ) from error
            try:
                yield json.loads(decoded), position, len(raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"malformed JSON in {path} at byte {position}: {error.msg}"
                ) from error


def read_at(path: Path, offset: int, length: int) -> Any:
    """Parse one JSON record from a known byte range."""
    with path.open("rb") as stream:
        stream.seek(offset)
        return json.loads(stream.read(length))


def line_boundaries(path: Path, every: int) -> tuple[list[int], int]:
    """Byte offsets of every `every`-th line start, and the file's line count."""
    starts, lines, position, target = [0], 0, 0, every
    last = b"\n"
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024**2):
            newlines = block.count(b"\n")
            index, consumed = -1, 0
            while target - lines <= newlines:
                need = target - lines - consumed
                for _ in range(need):
                    index = block.find(b"\n", index + 1)
                consumed += need
                starts.append(position + index + 1)
                target += every
            lines += newlines
            position += len(block)
            last = block[-1:]
    if last != b"\n":
        lines += 1
    if starts[-1] >= position:
        starts.pop()
    return starts, lines


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


def run_tasks(
    function: Callable[..., Any],
    tasks: Iterable[tuple],
    consume: Callable[[Any], None],
    *,
    total_bytes: int,
) -> None:
    """Run tasks inline or in a bounded process pool, consuming results as they land.

    The pool holds at most two tasks per worker in flight so result memory stays
    bounded, and any failure cancels the rest.
    """
    workers = worker_count()
    if workers == 1 or total_bytes < PARALLEL_MIN_BYTES:
        for task in tasks:
            consume(function(*task))
        return
    pending: set = set()
    iterator = iter(tasks)
    context = multiprocessing.get_context("spawn")
    with (
        _workers_without_main(),
        ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool,
    ):
        try:
            while True:
                while len(pending) < workers * 2:
                    task = next(iterator, None)
                    if task is None:
                        break
                    pending.add(pool.submit(function, *task))
                if not pending:
                    return
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    consume(future.result())
        except BaseException:
            for future in pending:
                future.cancel()
            raise
