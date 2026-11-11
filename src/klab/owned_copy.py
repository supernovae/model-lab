"""Private, no-replace file copies with byte authentication before publication."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import stat
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from klab.data.verification import (
    VerifiedFile,
    _fingerprint,
    _owned_file,
    _written_file,
)

_CHUNK = 1024 * 1024
_FICLONE = 0x40049409


@dataclass(frozen=True)
class CopyResult:
    proof: VerifiedFile
    mechanism: str
    logical_bytes: int


def _safe_parent(path: Path, *, create: bool = False) -> None:
    """Reject symlink traversal and optionally create owned parent directories."""
    parent = path.parent.absolute()
    for current in (*reversed(parent.parents), parent):
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create:
                raise
            current.mkdir(mode=0o700)
            info = current.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"unsafe copy parent: {current}")


def _hash_fd(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while block := os.read(fd, _CHUNK):
        digest.update(block)
    return digest.hexdigest()


def _clonefile(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    clone = library.clonefile
    clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    clone.restype = ctypes.c_int
    if clone(os.fsencode(source), os.fsencode(destination), 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _transfer(
    incoming: int,
    outgoing: int,
    length: int,
    progress: Callable[[int], None] | None,
    reported: int,
) -> tuple[str, int]:
    def advance(position: int) -> None:
        nonlocal reported
        if progress is not None and position > reported:
            reported = position
            progress(reported)

    if hasattr(os, "copy_file_range"):
        try:
            copied = 0
            while copied < length:
                count = os.copy_file_range(
                    incoming, outgoing, min(_CHUNK, length - copied)
                )
                if count <= 0:
                    raise OSError("short native copy")
                copied += count
                advance(copied)
            return "copy_file_range", reported
        except OSError:
            # A native transfer may have written an arbitrary prefix. Reset *both*
            # offsets and truncate it before starting the buffered fallback.
            os.ftruncate(outgoing, 0)
            os.lseek(outgoing, 0, os.SEEK_SET)
            os.lseek(incoming, 0, os.SEEK_SET)

    copied = 0
    while copied < length:
        block = os.read(incoming, min(_CHUNK, length - copied))
        if not block:
            raise ValueError("source shortened during copy")
        view = memoryview(block)
        while view:
            count = os.write(outgoing, view)
            if count <= 0:
                raise OSError("short buffered write")
            view = view[count:]
            copied += count
            advance(copied)
    return "buffered", reported


def owned_copy(
    source: Path,
    destination: Path,
    *,
    proof: VerifiedFile | None = None,
    expected_sha256: str | None = None,
    progress: Callable[[int], None] | None = None,
) -> CopyResult:
    """Authenticate a private copy, then publish exclusively without linking the source.

    The optional seal must be a current-process, fingerprint-matched VerifiedFile;
    without it the source is cold-hashed. The destination is always cold-hashed.
    Neither the destination nor its parent may be a symlink, and an existing
    destination is never replaced (including a broken symlink).
    """
    source = Path(source).absolute()
    destination = Path(destination).absolute()
    _safe_parent(source)
    _safe_parent(destination, create=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    before = _fingerprint(source)
    incoming = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    temporary: Path | None = None
    try:
        opened = os.fstat(incoming)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != before:
            raise ValueError("source changed before copying")
        if proof is not None:
            if (
                not _owned_file(proof)
                or proof.path != source.resolve(strict=True)
                or proof.fingerprint != before
            ):
                raise ValueError("source proof is not a current file seal")
            authenticated = proof.sha256
        else:
            authenticated = _hash_fd(incoming)
            if _fingerprint(source) != before:
                raise ValueError("source changed during authentication")
        if expected_sha256 is not None and authenticated != expected_sha256:
            raise ValueError("source digest mismatch")
        os.lseek(incoming, 0, os.SEEK_SET)
        fd, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        temporary = Path(name)
        reported = 0
        try:
            os.fchmod(fd, 0o600)
            mechanism: str | None = None
            if sys.platform.startswith("linux"):
                try:
                    fcntl.ioctl(fd, _FICLONE, incoming)
                    mechanism = "reflink"
                except OSError:
                    os.ftruncate(fd, 0)
                    os.lseek(fd, 0, os.SEEK_SET)
                    os.lseek(incoming, 0, os.SEEK_SET)
            elif sys.platform == "darwin":
                os.close(fd)
                fd = -1
                temporary.unlink()
                try:
                    _clonefile(source, temporary)
                    fd = os.open(temporary, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
                    os.fchmod(fd, 0o600)
                    mechanism = "clone"
                except OSError, AttributeError:
                    temporary.unlink(missing_ok=True)
                    fd = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
                    os.lseek(incoming, 0, os.SEEK_SET)
            if mechanism is None:
                mechanism, reported = _transfer(
                    incoming, fd, before[3], progress, reported
                )
            elif progress is not None and before[3]:
                progress(before[3])
            if os.fstat(fd).st_size != before[3]:
                raise ValueError("copied file size mismatch")
            if _hash_fd(fd) != authenticated:
                raise ValueError("copied file digest mismatch")
            if _fingerprint(source) != before:
                raise ValueError("source changed during copy")
            os.fsync(fd)
            if _fingerprint(source) != before:
                raise ValueError("source changed before publication")
            temporary_fp = _fingerprint(temporary)
            # Link only the private copy. Remove the temporary name before
            # sealing: both link and unlink change inode ctime.
            os.link(temporary, destination, follow_symlinks=False)
            try:
                temporary.unlink()
                temporary = None
                published = _fingerprint(destination)
                opened = os.fstat(fd)
                fd_fp = (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_mode,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                )
                if (
                    published[:5] != temporary_fp[:5]
                    or published != fd_fp
                    or _fingerprint(source) != before
                ):
                    raise ValueError("file changed during publication")
                proof_result = _written_file(destination, authenticated, before[3])
                if proof_result.fingerprint != published:
                    raise ValueError("destination changed during publication")
                parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
                if _fingerprint(destination) != proof_result.fingerprint:
                    raise ValueError("destination changed during publication")
            except BaseException:
                destination.unlink()
                raise
            return CopyResult(proof_result, mechanism, before[3])
        finally:
            if fd >= 0:
                os.close(fd)
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    finally:
        os.close(incoming)
