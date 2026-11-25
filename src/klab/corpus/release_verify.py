"""Bounded, parallel and observable machinery for verifying immutable releases.

Verification never changes what is checked, only how: files are hashed
concurrently, large JSONL files are split into line-aligned byte ranges that
worker processes read independently, and the one disposable index lives under
an operator-chosen scratch directory. Nothing here is evidence.
"""

from __future__ import annotations

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
from klab.training import manifest as file_manifest

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore, VerificationMode

INVENTORY_RECEIPT_MIN_BYTES = 64 * 1024**2
_HASH_CHUNK_BYTES = 8 * 1024**2


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
        help="Worker limit for parallel corpus passes (default: "
        f"half the CPU count, {default_workers()} here; 1 runs serially)",
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
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> None:
    """Check every size and SHA-256, reusing signed evidence for completed hashes."""

    lock = threading.Lock()

    def advance(chunk: bytes) -> None:
        if progress is not None:
            with lock:
                progress.update(input_bytes=len(chunk))

    def check(name: str) -> int | None:
        path = resolve(root, name)
        entry = inventory[name]
        size, expected = entry["size"], entry["sha256"]
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"tampered artifact: {name}")
        if (
            proof_store is not None
            and verification_mode == "verified_reuse"
            and size >= INVENTORY_RECEIPT_MIN_BYTES
        ):
            from klab.data.verification import verify_file

            try:
                proof = verify_file(
                    path,
                    expected_sha256=expected,
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                    chunk_size=_HASH_CHUNK_BYTES,
                    hash_chunk=advance if progress is not None else None,
                )
            except ValueError as error:
                raise ValueError(f"tampered artifact: {name}: {error}") from error
            if proof.size_bytes != size or proof.sha256 != expected:
                raise ValueError(f"tampered artifact: {name}")
            if not proof.cold_verified:
                if progress is not None:
                    with lock:
                        progress.update(input_bytes_reused=size)
                return size
        elif (
            file_manifest.sha256_file(
                path,
                chunk_size=_HASH_CHUNK_BYTES,
                hash_chunk=advance if progress is not None else None,
            )
            != expected
        ):
            raise ValueError(f"tampered artifact: {name}")
        return None

    names = sorted(inventory, key=lambda name: -inventory[name]["size"])
    workers = min(worker_count(), len(names))
    if workers < 2:
        reused = [size for name in names if (size := check(name)) is not None]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            reused = [
                size
                for size in pool.map(check, names, buffersize=workers)
                if size is not None
            ]
    settings = current_settings()
    if reused and settings.echo is not None:
        print(
            f"inventory hashes: reused {len(reused)}/{len(names)} files, {sum(reused)} bytes",
            file=settings.echo,
            flush=True,
        )


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
