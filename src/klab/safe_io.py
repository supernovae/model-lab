"""Small filesystem and database primitives shared by corpus and data code."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from klab.canonical import canonical_json


def symlinked(path: Path) -> bool:
    """Whether the path or any ancestor is a symbolic link."""
    return any(part.is_symlink() for part in (path, *path.parents))


@contextmanager
def database_byte_cap(message: str) -> Iterator[None]:
    """Report a full SQLite database (``max_page_count`` reached) as a ValueError."""
    try:
        yield
    except sqlite3.OperationalError as error:
        if error.sqlite_errorcode == sqlite3.SQLITE_FULL:
            raise ValueError(message) from error
        raise


def write_canonical_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    """Write canonical JSON plus a newline and fsync it; ``exclusive`` refuses overwrite."""
    with path.open("xb" if exclusive else "wb") as handle:
        handle.write(canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
