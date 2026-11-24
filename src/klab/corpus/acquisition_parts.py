"""Private checkpoint storage for a pinned, bounded HF acquisition."""

from __future__ import annotations

import fcntl
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.project import HFBoundedShard
from klab.result_publication import (
    read_canonical,
    sync_directory,
    write_bytes_exclusive,
)
from klab.safe_io import symlinked
from klab.training.manifest import sha256_file


class HFParts:
    """Selected outputs stay in staging and move into the snapshot by rename."""

    def __init__(
        self, parent: Path, identity: str, *, receipt_limit: int | None = None
    ):
        self.root = parent / f".hf-parts-{identity}"
        self.staging = self.root / "snapshot"
        self.identity = identity
        self.receipt_limit = receipt_limit

    def _check_root(self) -> None:
        if symlinked(self.staging):
            raise ValueError(f"unsafe HF checkpoint root: {self.root}")
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
                raise ValueError(f"HF checkpoint must be privately owned: {path}")

    def body(self, shard: HFBoundedShard) -> Path:
        return self.staging / "hf-input" / shard.path

    def output(self, shard: HFBoundedShard) -> Path:
        return self.staging / "files" / (shard.path + ".sample.jsonl")

    def _marker(self, shard: HFBoundedShard) -> Path:
        return self.root / ("part-" + canonical_digest(shard.path) + ".json")

    def _size(self, path: Path, expected: str, limit: int) -> int | None:
        if symlinked(path):
            raise ValueError(f"unsafe HF checkpoint file: {path}")
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
        ):
            raise ValueError(f"unsafe HF checkpoint file: {path}")
        return (
            info.st_size
            if info.st_size <= limit and sha256_file(path) == expected
            else None
        )

    def verified_body(self, shard: HFBoundedShard) -> int | None:
        self._check_root()
        return self._size(
            self.body(shard), shard.expected_sha256.lower(), shard.max_shard_bytes
        )

    def verified_part(
        self,
        shard: HFBoundedShard,
        max_bytes: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]] | None:
        self._check_root()
        marker = self._marker(shard)
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
                or body["shard"] != shard.path
            ):
                return None
            inventory, receipt = body["inventory"], body["receipt"]
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
                size = self._size(path, inventory["sha256"], max_bytes)
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
        try:
            self._marker(shard).unlink()
        except FileNotFoundError:
            return False
        return True

    def receipt_bytes(self) -> int:
        self._check_root()
        return sum(path.lstat().st_size for path in self.root.glob("part-*.json"))

    def sync_path(self, path: Path) -> None:
        parent = path.parent
        while parent != self.root:
            sync_directory(parent)
            parent = parent.parent

    def commit(
        self,
        shard: HFBoundedShard,
        inventory: dict[str, Any] | None,
        receipt: dict[str, Any],
    ) -> None:
        self.sync_path(self.output(shard))
        body = {
            "identity": self.identity,
            "shard": shard.path,
            "inventory": inventory,
            "receipt": receipt,
        }
        encoded = (
            canonical_json({"body": body, "sha256": canonical_digest(body)}) + b"\n"
        )
        if (
            self.receipt_limit is not None
            and self.receipt_bytes() + len(encoded) > self.receipt_limit
        ):
            raise ValueError("acquisition step HF checkpoint metadata cap exceeded")
        write_bytes_exclusive(self._marker(shard), encoded)


@contextmanager
def acquisition_staging(parent: Path, parts: HFParts | None) -> Iterator[Path]:
    if parts is None:
        with tempfile.TemporaryDirectory(prefix=".acquire-", dir=parent) as temporary:
            yield Path(temporary)
        return
    parts._check_root()
    descriptor = os.open(
        parent / ".hf-parts.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another acquisition is writing this HF source") from error
        parts._check_root()
        parts.root.mkdir(mode=0o700, exist_ok=True)
        sync_directory(parent)
        parts.staging.mkdir(mode=0o700, exist_ok=True)
        sync_directory(parts.root)
        yield parts.staging
        shutil.rmtree(parts.root)
