"""Private staging and checkpoints for pinned, bounded acquisition."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.project import HFBoundedShard, PdfAcquisition, PdfDocument
from klab.result_publication import (
    read_canonical,
    sync_directory,
    write_bytes_exclusive,
)
from klab.safe_io import rename_noreplace, symlinked
from klab.training.manifest import sha256_file
from klab.verification_proofs import ArtifactFingerprint, artifact_fingerprint


def _checkpoint_fingerprint(path: Path) -> ArtifactFingerprint | None:
    try:
        return artifact_fingerprint(path)
    except OSError, ValueError:
        return None


@dataclass(frozen=True)
class CheckedCheckpoint[T]:
    """Cold verification evidence kept only within one acquisition call."""

    path: Path
    value: T
    fingerprint: ArtifactFingerprint | None

    def matches(self, path: Path, *, renamed: bool = False) -> bool:
        if not renamed and self.path != path:
            return False
        return self._matches_fingerprint(_checkpoint_fingerprint(path), renamed=renamed)

    def _matches_fingerprint(
        self, current: ArtifactFingerprint | None, *, renamed: bool
    ) -> bool:
        if self.fingerprint is None or current is None:
            return False
        if renamed:
            # The atomic directory rename can change the root's ctime.
            # Require the same inode and all other metadata, including member ctimes.
            return (
                current[0][:-1] == self.fingerprint[0][:-1]
                and current[1:] == self.fingerprint[1:]
            )
        return current == self.fingerprint

    def after_rename(self, path: Path) -> CheckedCheckpoint[T]:
        """Bind checked bytes to the same artifact after an atomic rename."""
        current = _checkpoint_fingerprint(path)
        if not self._matches_fingerprint(current, renamed=True):
            raise ValueError(f"checkpoint changed during promotion: {path}")
        return CheckedCheckpoint(path, self.value, current)


def check_checkpoint[T](
    path: Path, verify: Callable[[Path], T | None]
) -> CheckedCheckpoint[T] | None:
    before = _checkpoint_fingerprint(path)
    value = verify(path)
    if value is None:
        return None
    if before is not None and before != _checkpoint_fingerprint(path):
        raise ValueError(f"checkpoint changed during verification: {path}")
    return CheckedCheckpoint(path, value, before)


class AcquisitionParts:
    """Keep completed parts in a private staging tree until snapshot publication."""

    def __init__(self, parent: Path, identity: str, *, adapter: str):
        self.adapter = adapter
        self.root = parent / f".{adapter}-parts-{identity}"
        self.staging = self.root / "snapshot"
        self.identity = identity

    def _check_root(self) -> None:
        if symlinked(self.staging):
            raise ValueError(f"unsafe acquisition checkpoint root: {self.root}")
        for path in (self.root, self.staging):
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700
            ):
                raise ValueError(
                    f"acquisition checkpoint must be privately owned: {path}"
                )

    def staged_file_size(self, path: Path) -> int | None:
        if symlinked(path):
            raise ValueError(f"unsafe acquisition checkpoint file: {path}")
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
        ):
            raise ValueError(f"unsafe acquisition checkpoint file: {path}")
        return info.st_size

    def verified_size(self, path: Path, expected: str, limit: int) -> int | None:
        self._check_root()
        size = self.staged_file_size(path)
        return (
            size
            if size is not None and size <= limit and sha256_file(path) == expected
            else None
        )

    def sync_path(self, path: Path) -> None:
        parent = path.parent
        while parent != self.root:
            sync_directory(parent)
            parent = parent.parent


@dataclass(frozen=True)
class PinnedBody:
    sha256: str
    size: int
    sha1: str | None = None


class PinnedBodyParts(AcquisitionParts):
    """Retain one pinned input independently of its derived snapshot."""

    def __init__(
        self,
        parent: Path,
        identity: str,
        *,
        adapter: str,
        name: str,
        sha256: str | None,
        max_bytes: int,
        sha1: str | None = None,
    ):
        super().__init__(parent, identity, adapter=adapter)
        self.name = name
        self.sha256 = sha256
        self.sha1 = sha1
        self.max_bytes = max_bytes
        self.body = self.root / "input" / name
        self.manifest = self.staging / "manifest.json"

    @property
    def download(self) -> Path:
        return self.body.with_name(".download")

    @property
    def cache_download(self) -> Path:
        return self.body.with_name(".cache-download")

    def checked_body(
        self, previous: CheckedCheckpoint[PinnedBody] | None = None
    ) -> CheckedCheckpoint[PinnedBody] | None:
        self._check_root()
        if previous is not None and previous.matches(self.body):
            return previous
        return check_checkpoint(self.body, self._verify_body)

    def promote_body(
        self, checked: CheckedCheckpoint[PinnedBody]
    ) -> CheckedCheckpoint[PinnedBody]:
        if not checked.matches(checked.path):
            raise ValueError(f"checkpoint changed before promotion: {checked.path}")
        rename_noreplace(checked.path, self.body)
        sync_directory(self.body.parent)
        return checked.after_rename(self.body)

    def recover_download(self) -> CheckedCheckpoint[PinnedBody] | None:
        """Recheck the official-download candidate before discarding it."""
        self._check_root()
        self.staged_file_size(self.cache_download)
        self.cache_download.unlink(missing_ok=True)
        checked = check_checkpoint(self.download, self._verify_body)
        if checked is None:
            self.download.unlink(missing_ok=True)
            return None
        with self.download.open("rb") as body:
            os.fsync(body.fileno())
        return self.promote_body(checked)

    def _verify_body(self, path: Path) -> PinnedBody | None:
        if self.sha1 is None:
            assert self.sha256 is not None
            size = self.verified_size(path, self.sha256, self.max_bytes)
            return PinnedBody(self.sha256, size) if size is not None else None
        size = self.staged_file_size(path)
        if size is None or size > self.max_bytes:
            return None
        sha1 = hashlib.sha1()
        sha256 = sha256_file(path, hash_chunk=sha1.update)
        if sha1.hexdigest() != self.sha1 or (self.sha256 and sha256 != self.sha256):
            return None
        return PinnedBody(sha256, size, self.sha1)

    def manifest_size(self) -> int | None:
        self._check_root()
        return self.staged_file_size(self.manifest)

    def reset_derivation(self) -> None:
        self.manifest.unlink(missing_ok=True)
        files = self.staging / "files"
        if files.exists() or files.is_symlink():
            shutil.rmtree(files)

    def discard_body(self) -> None:
        # The directory removal on success also handles unsafe input entries
        # without following links.
        if symlinked(self.body.parent):
            return
        try:
            self.body.unlink()
        except FileNotFoundError:
            return
        sync_directory(self.body.parent)

    def reclaim_published(self) -> None:
        """A moved snapshot leaves no usable derivation in this request root."""
        if not self.root.exists() or self.staging.exists():
            return
        with _parts_lock(self) as locked:
            if not locked:
                print(
                    f"existing acquisition staging retained at {self.root}",
                    file=sys.stderr,
                )
                return
            if self.root.exists() and not self.staging.exists():
                shutil.rmtree(self.root)
                sync_directory(self.root.parent)


class ReceiptedParts(AcquisitionParts):
    """Bind derived output receipts to their request and logical part."""

    def __init__(
        self,
        parent: Path,
        identity: str,
        *,
        adapter: str,
        receipt_limit: int | None,
    ):
        super().__init__(parent, identity, adapter=adapter)
        self.receipt_limit = receipt_limit

    def _marker(self, name: str) -> Path:
        return self.root / ("part-" + canonical_digest(name) + ".json")

    def _read_result(self, name: str) -> dict[str, Any] | None:
        self._check_root()
        marker = self._marker(name)
        try:
            if (
                self.receipt_limit is not None
                and marker.stat().st_size > self.receipt_limit
            ):
                return None
            entry = read_canonical(marker)
            body = entry["body"]
            if (
                canonical_digest(body) != entry["sha256"]
                or body["identity"] != self.identity
                or body["part"] != name
                or not isinstance(body["result"], dict)
            ):
                return None
            return body["result"]
        except FileNotFoundError, ValueError, TypeError, KeyError:
            return None

    def _discard_marker(self, name: str) -> bool:
        try:
            self._marker(name).unlink()
        except FileNotFoundError:
            return False
        return True

    def receipt_bytes(self) -> int:
        self._check_root()
        return sum(path.lstat().st_size for path in self.root.glob("part-*.json"))

    def _commit(self, name: str, output: Path, result: dict[str, Any]) -> None:
        self.sync_path(output)
        body = {"identity": self.identity, "part": name, "result": result}
        encoded = (
            canonical_json({"body": body, "sha256": canonical_digest(body)}) + b"\n"
        )
        if (
            self.receipt_limit is not None
            and self.receipt_bytes() + len(encoded) > self.receipt_limit
        ):
            raise ValueError("acquisition step checkpoint metadata cap exceeded")
        write_bytes_exclusive(self._marker(name), encoded)


class HFParts(ReceiptedParts):
    """Keep receipts for selected rows and verify the raw bodies of unfinished shards."""

    def __init__(
        self, parent: Path, identity: str, *, receipt_limit: int | None = None
    ):
        super().__init__(parent, identity, adapter="hf", receipt_limit=receipt_limit)

    def body(self, shard: HFBoundedShard) -> Path:
        return self.staging / "hf-input" / shard.path

    def output(self, shard: HFBoundedShard) -> Path:
        return self.staging / "files" / (shard.path + ".sample.jsonl")

    def verified_body(self, shard: HFBoundedShard) -> int | None:
        return self.verified_size(
            self.body(shard), shard.expected_sha256.lower(), shard.max_shard_bytes
        )

    def verified_part(
        self,
        shard: HFBoundedShard,
        max_bytes: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]] | None:
        result = self._read_result(shard.path)
        if result is None:
            return None
        try:
            inventory, receipt = result["inventory"], result["receipt"]
            if (
                receipt["source_shard_path"] != shard.path
                or receipt["source_shard_sha256"] != shard.expected_sha256.lower()
                or receipt["output_path"] != (inventory["path"] if inventory else None)
                or not isinstance(receipt["selected_rows"], list)
                or bool(receipt["selected_rows"]) != (inventory is not None)
            ):
                return None
            if inventory is not None:
                path = self.output(shard)
                size = self.verified_size(path, inventory["sha256"], max_bytes)
                if (
                    size is None
                    or size != inventory["size"]
                    or inventory["path"] != shard.path + ".sample.jsonl"
                ):
                    return None
            elif self.output(shard).exists() or self.output(shard).is_symlink():
                return None
            return inventory, receipt
        except FileNotFoundError, ValueError, TypeError, KeyError:
            return None

    def discard_marker(self, shard: HFBoundedShard) -> bool:
        return self._discard_marker(shard.path)

    def commit(
        self,
        shard: HFBoundedShard,
        inventory: dict[str, Any] | None,
        receipt: dict[str, Any],
    ) -> None:
        self._commit(
            shard.path, self.output(shard), {"inventory": inventory, "receipt": receipt}
        )


def pinned_pdf_path(name: str) -> str:
    """Relative path of the original PDF in a snapshot's pinned inventory."""
    return name.removesuffix(".txt") + ".pdf"


class PDFParts(ReceiptedParts):
    """Checkpoints for pinned PDF bodies and completed text extraction."""

    def __init__(
        self, parent: Path, identity: str, *, receipt_limit: int | None = None
    ):
        super().__init__(parent, identity, adapter="pdf", receipt_limit=receipt_limit)

    def body(self, document: PdfDocument) -> Path:
        return self.staging / "pinned" / pinned_pdf_path(document.name)

    def output(self, document: PdfDocument) -> Path:
        return self.staging / "files" / document.name

    def verified_document(
        self, document: PdfDocument, spec: PdfAcquisition
    ) -> tuple[int | None, tuple[str, int, int] | None]:
        """Return verified original bytes and, when valid, the extracted text receipt."""
        body_size = self.verified_size(
            self.body(document), document.expected_sha256, document.max_bytes
        )
        if body_size is None:
            return None, None
        return body_size, self._verified_text(document, spec)

    def _verified_text(
        self, document: PdfDocument, spec: PdfAcquisition
    ) -> tuple[str, int, int] | None:
        result = self._read_result(document.name)
        if result is None:
            return None
        try:
            pages, digest = result["pages"], result["text_sha256"]
            if type(pages) is not int or not 0 < pages <= spec.max_pages:
                return None
            size = self.verified_size(
                self.output(document), digest, spec.max_text_bytes
            )
            if size is None or size != result["text_bytes"]:
                return None
            return digest, size, pages
        except TypeError, KeyError:
            return None

    def discard_marker(self, document: PdfDocument) -> bool:
        return self._discard_marker(document.name)

    def commit(self, document: PdfDocument, digest: str, size: int, pages: int) -> None:
        self._commit(
            document.name,
            self.output(document),
            {"pages": pages, "text_sha256": digest, "text_bytes": size},
        )


@contextmanager
def _parts_lock(parts: AcquisitionParts) -> Iterator[bool]:
    parts._check_root()
    descriptor = os.open(
        parts.root.parent / f".{parts.adapter}-parts.lock",
        os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            parts._check_root()
            yield True


@contextmanager
def acquisition_staging(parent: Path, parts: AcquisitionParts | None) -> Iterator[Path]:
    for sibling in sorted(parent.glob(".acquire-*")):
        if not sibling.is_symlink() and sibling.is_dir():
            print(
                f"existing acquisition staging retained at {sibling}",
                file=sys.stderr,
            )
    if parts is None:
        with tempfile.TemporaryDirectory(prefix=".acquire-", dir=parent) as temporary:
            yield Path(temporary)
        return
    with _parts_lock(parts) as locked:
        if not locked:
            raise ValueError("another acquisition is writing this source")
        for sibling in sorted(parent.glob(f".{parts.adapter}-parts-*")):
            if sibling != parts.root and not sibling.is_symlink() and sibling.is_dir():
                print(
                    f"acquisition checkpoint for another request retained at {sibling}",
                    file=sys.stderr,
                )
        parts.root.mkdir(mode=0o700, exist_ok=True)
        sync_directory(parent)
        parts.staging.mkdir(mode=0o700, exist_ok=True)
        sync_directory(parts.root)
        yield parts.staging
        shutil.rmtree(parts.root)
