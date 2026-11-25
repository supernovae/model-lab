"""Checkpointed publication for ordered split and family inventories."""

from __future__ import annotations

import fcntl
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from klab.canonical import canonical_digest
from klab.corpus.view_checkpoints import ViewCheckpoints
from klab.result_publication import sync_directory
from klab.verification_proofs import writable_by_others

INVENTORY_WINDOW_BYTES = 64 * 1024**2


@contextmanager
def inventory_writer(output: Path, identity: str):
    """Commit an ordered inventory in place and link its completed payload."""
    if output.exists() or output.is_symlink():
        raise ValueError(f"inventory output already exists: {output}")
    missing_parents = []
    parent = output.parent
    while not parent.exists():
        missing_parents.append(parent)
        parent = parent.parent
    output.parent.mkdir(parents=True, exist_ok=True)
    for parent in reversed(missing_parents):
        sync_directory(parent.parent)
    directory = output.parent / (".inventory-" + canonical_digest(output.name)[:24])
    directory.mkdir(mode=0o700, exist_ok=True)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if info.st_uid != os.getuid() or writable_by_others(info):
            raise ValueError(f"unsafe inventory staging directory: {directory}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if output.exists() or output.is_symlink():
            raise ValueError(f"inventory output already exists: {output}")
        if directory.stat().st_ino != info.st_ino:
            raise ValueError(f"inventory staging directory changed: {directory}")
        payload = directory / "build" / "inventory.jsonl"
        if payload.is_symlink() or payload.parent.is_symlink():
            raise ValueError(f"symlinked inventory payload: {payload}")
        payload.parent.mkdir(exist_ok=True)
        with payload.open("a+b") as stream:
            checkpoints = ViewCheckpoints(directory, identity)
            view = checkpoints.add(payload.name, stream)
            state = checkpoints.restore()
            yield checkpoints, view, state
            view.file_digest()
        os.link(payload, output)
        sync_directory(output.parent)
        shutil.rmtree(directory)
        sync_directory(output.parent)
    finally:
        os.close(descriptor)
