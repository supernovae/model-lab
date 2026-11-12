"""No-clobber publication of already encoded immutable evidence."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from klab.canonical import canonical_json, reject_duplicate_keys


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


def read_canonical(path: Path) -> dict[str, Any]:
    """Read a canonical JSON record, refusing links, duplicate keys and reformatting."""
    if path.is_symlink():
        raise ValueError(f"symlinked record: {path}")
    raw = path.read_bytes()

    def nonfinite(value: str) -> None:
        raise ValueError(f"nonfinite record value: {value}")

    try:
        data = json.loads(
            raw, object_pairs_hook=reject_duplicate_keys, parse_constant=nonfinite
        )
        if raw != canonical_json(data) + b"\n":
            raise ValueError(f"noncanonical record: {path}")
    except (UnicodeError, json.JSONDecodeError, TypeError, OverflowError) as exc:
        raise ValueError(f"invalid record: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"record is not an object: {path}")  # noqa: TRY004 - serialized schema
    return data


def publish_immutable(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Publish once; replaying the same payload returns the existing record."""
    data = canonical_json(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, path)
            sync_directory(path.parent)
        except FileExistsError:
            previous = read_canonical(path)
            if previous != payload:
                raise ValueError(f"conflicting immutable record: {path}") from None
            return previous
    finally:
        os.unlink(name)
    return payload


def utc_now() -> str:
    """The current UTC time as an ISO 8601 string ending in Z."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
