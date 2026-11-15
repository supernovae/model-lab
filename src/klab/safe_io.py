"""Small filesystem and database primitives shared by corpus and data code."""

from __future__ import annotations

import ctypes
import errno
import os
import sqlite3
import sys
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


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename, failing with ``FileExistsError`` if the destination exists."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = getattr(library, "renamex_np", None)
        if function is None:
            raise OSError(errno.ENOTSUP, "renamex_np is unavailable")
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(os.fsencode(source), os.fsencode(destination), 0x00000004)
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:
            raise OSError(errno.ENOTSUP, "renameat2 is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise OSError(
            errno.ENOTSUP, f"atomic no-replace rename unsupported on {sys.platform}"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))
