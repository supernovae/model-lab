"""Bounded, parallel and observable machinery for verifying immutable releases.

Verification never changes what is checked, only how: files are hashed
concurrently, large JSONL files are split into line-aligned byte ranges that
worker processes read independently, and the one disposable index lives under
an operator-chosen scratch directory. Nothing here is evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from klab.corpus.progress import BuildProgress

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore, VerificationMode


def default_workers() -> int:
    """Half the cores: corpus stages share the host with whatever else it runs."""
    return max(1, (os.cpu_count() or 1) // 2)


@dataclass(frozen=True)
class VerificationSettings:
    """Operator choices that change speed and placement, never the verdict."""

    scratch_dir: Path | None = None
    workers: int = field(default_factory=default_workers)
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


def add_workers_flag(command: Any) -> None:
    """Give a command the one `--workers` option every parallel corpus pass honours."""
    command.add_argument(
        "--workers",
        type=int,
        help="Worker processes for parallel passes over corpus files (default: "
        f"half the CPU count, {default_workers()} here; 1 runs in one process)",
    )


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
