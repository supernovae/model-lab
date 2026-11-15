"""Copy a verified build into a release while hashing it, concurrently."""

from __future__ import annotations

import fcntl
import hashlib
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from klab.corpus.progress import BuildProgress
from klab.corpus.release_verify import worker_count
from klab.training.manifest import sha256_file

# linux/fs.h FICLONE: share the source's blocks (btrfs, XFS with reflink).
_FICLONE = 0x40049409
_BUFFER = 8 * 1024**2


def copy_hashed(
    source: Path, target: Path, on_bytes: Callable[[int], None] | None = None
) -> tuple[str, int]:
    """Copy a file durably and return the SHA-256 and size of the bytes written.

    A copy-on-write clone is used where the filesystem offers one, and its bytes
    are then hashed; otherwise the bytes are hashed as they stream through. A
    hard link is never used: it would let a later write to the build change the
    published release, and receipts refuse multiply linked files.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        try:
            fcntl.ioctl(writer.fileno(), _FICLONE, reader.fileno())
        except OSError:
            digest, size = hashlib.sha256(), 0
            buffer = bytearray(_BUFFER)
            view = memoryview(buffer)
            while count := reader.readinto(buffer):
                digest.update(view[:count])
                writer.write(view[:count])
                size += count
                if on_bytes is not None:
                    on_bytes(count)
            writer.flush()
            os.fsync(writer.fileno())
            return digest.hexdigest(), size
        os.fsync(writer.fileno())
    return sha256_file(target), target.stat().st_size


def copy_inventory(
    source_root: Path,
    target_root: Path,
    inventory: dict[str, Any],
    resolve: Callable[[Path, str], Path],
    progress: BuildProgress | None = None,
) -> None:
    """Copy every inventoried file, refusing any whose bytes differ from the inventory."""

    lock = threading.Lock()

    def advance(size: int) -> None:
        if progress is not None:
            with lock:
                progress.update(input_bytes=size)

    def copy(name: str) -> None:
        digest, size = copy_hashed(
            resolve(source_root, name), resolve(target_root, name), advance
        )
        entry = inventory[name]
        if digest != entry["sha256"] or size != entry["size"]:
            raise ValueError(f"tampered artifact: {name}")

    names = sorted(inventory, key=lambda name: -inventory[name]["size"])
    # Targets are created here so concurrent workers never race on a directory.
    for name in names:
        resolve(target_root, name).parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=worker_count()) as pool:
        list(pool.map(copy, names))
