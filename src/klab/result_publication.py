"""No-clobber publication of already encoded immutable evidence."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes_exclusive(
    path: Path,
    encoded: bytes,
    *,
    sync_directory: Callable[[Path], None] = sync_directory,
) -> None:
    """Publish complete, file-synced bytes without replacing a destination.

    FileExistsError follows a directory sync, leaving collision/replay policy
    to the caller. A failure after linking leaves complete evidence with uncertain
    directory durability;
    never remove that surviving destination. Only our temporary file is removed.
    """
    missing_parents = []
    parent = path.parent
    while not parent.exists():
        missing_parents.append(parent)
        parent = parent.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    # Persist newly created directory entries before publishing evidence inside them.
    for parent in reversed(missing_parents):
        sync_directory(parent.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            if handle.write(encoded) != len(encoded):
                raise OSError("short immutable publication write")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            # A matching caller may return success on replay, even when the
            # original publisher failed (or has not yet finished) this sync.
            sync_directory(path.parent)
            raise
        try:
            sync_directory(path.parent)
        except OSError as error:
            raise OSError(
                f"complete evidence published at {path}, but directory fsync failed; "
                "its directory entry may not be durable (file retained)"
            ) from error
    finally:
        os.unlink(temporary)
