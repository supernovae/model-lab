"""Immutable segments of the encoder's content token cache."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import sqlite3
import stat
import time
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tokenizers

from klab.canonical import canonical_digest
from klab.data.token_store import CONTENT_INDEX_DTYPE, SHARD_TOKENS, ContentTokens
from klab.result_publication import read_canonical, sync_directory
from klab.safe_io import (
    read_file_range,
    rename_noreplace,
    symlinked,
    write_canonical_json,
)
from klab.training.manifest import sha256_file
from klab.verification_proofs import artifact_fingerprint, writable_by_others

SEGMENT_DOCUMENTS = 65_536
_OPEN_SEGMENTS = 16
_MANIFEST = "segment.json"
_FILES = {"tokens.ids", "index.bin"}
_SHA256 = re.compile("[0-9a-f]{64}")


def _digest(value) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _owned_file(path: Path) -> os.stat_result:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or writable_by_others(info)
    ):
        raise ValueError(f"unsafe token segment member: {path}")
    return info


class SealedContentTokens:
    """Use the existing append writer for the unfinished shard.

    A segment commits by directory rename. The SQLite locator is derived from
    the narrow, sealed indexes and has no authority over token contents.
    """

    decode_ids = staticmethod(ContentTokens.decode_ids)

    def __init__(self, root: Path, tokenizer_sha: str):
        if not _digest(tokenizer_sha):
            raise ValueError("invalid token segment tokenizer digest")
        self.tokenizer_sha = tokenizer_sha
        self.encoding = {
            "tokenizer_sha256": tokenizer_sha,
            "tokenizers_version": tokenizers.__version__,
            "input": "exact-utf8",
            "add_special_tokens": False,
            "eos": None,
            "dtype": "int32-le",
        }
        self.path = root / "sealed" / canonical_digest(self.encoding)
        self.segments = self.path / "segments"
        self.active_root = self.path / "active"
        self.active_path = ContentTokens.cache_path(self.active_root, tokenizer_sha)
        self.active: ContentTokens | None = None
        self.hits = self.misses = self.sealed_hits = self.published_segments = 0
        self.lookup_seconds = self.write_seconds = 0.0
        self.handles = OrderedDict()
        self.verified_indexes = {}
        if symlinked(self.path):
            raise ValueError(f"unsafe sealed content token store: {self.path}")
        # Persist new ancestors before any sealed slots can name their contents.
        for directory in reversed([self.path, *self.path.parents]):
            if not directory.exists():
                directory.mkdir(mode=0o700, exist_ok=True)
                sync_directory(directory.parent)
        info = self.path.lstat()
        if info.st_uid != os.getuid() or writable_by_others(info):
            raise ValueError(f"unsafe sealed content token store: {self.path}")
        with ExitStack() as opening:
            lock = os.open(
                self.path / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
            )
            opening.callback(os.close, lock)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(
                    f"sealed content token store already has a writer: {self.path}"
                ) from error
            for directory in (self.segments, self.active_root, self.active_path):
                if directory.is_symlink():
                    raise ValueError(f"unsafe token segment directory: {directory}")
            for directory in (self.segments, self.active_root):
                if not directory.exists():
                    directory.mkdir(mode=0o700)
                    sync_directory(self.path)
            catalog = self.path / "locations.sqlite"
            if catalog.exists() or catalog.is_symlink():
                _owned_file(catalog)
            self.db = sqlite3.connect(catalog)
            opening.callback(self.db.close)
            self.db.executescript("""
                PRAGMA cache_size=-8192;
                CREATE TABLE IF NOT EXISTS segment (
                    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL
                );
                CREATE TABLE IF NOT EXISTS document (
                    content BLOB PRIMARY KEY, segment INTEGER NOT NULL,
                    row INTEGER NOT NULL
                ) WITHOUT ROWID;
            """)
            pending = self.active_path / _MANIFEST
            if pending.exists() or pending.is_symlink():
                manifest = self._read_manifest(self.active_path)
                verified = {}
                for name, entry in manifest["files"].items():
                    path = self.active_path / name
                    stamp = artifact_fingerprint(path)
                    if (
                        sha256_file(path) != entry["sha256"]
                        or artifact_fingerprint(path) != stamp
                    ):
                        raise ValueError(f"changed pending token segment: {name}")
                    verified[name] = stamp
                name = self._publish(manifest)
                self.verified_indexes[self.segments / name / "index.bin"] = verified[
                    "index.bin"
                ]
            self.names = dict(self.db.execute("SELECT id, name FROM segment"))
            self._register_segments()
            self._open_active()
            self.close_resources = opening.pop_all()

    def _read_manifest(self, directory: Path) -> dict:
        if directory.is_symlink() or _owned_file(directory / _MANIFEST).st_size > 8192:
            raise ValueError(f"unsafe token segment manifest: {directory}")
        manifest = read_canonical(directory / _MANIFEST)
        if (
            manifest.keys()
            != {"schema_version", "encoding", "documents", "tokens", "files"}
            or manifest["schema_version"] != 1
            or manifest["encoding"] != self.encoding
            or type(manifest["documents"]) is not int
            or not 1 <= manifest["documents"] <= SEGMENT_DOCUMENTS
            or type(manifest["tokens"]) is not int
            or manifest["tokens"] < 0
            or (manifest["tokens"] > SHARD_TOKENS and manifest["documents"] != 1)
            or not isinstance(manifest["files"], dict)
            or manifest["files"].keys() != _FILES
        ):
            raise ValueError(f"invalid token segment manifest: {directory}")
        sizes = {
            "index.bin": manifest["documents"] * CONTENT_INDEX_DTYPE.itemsize,
            "tokens.ids": manifest["tokens"] * 4,
        }
        for name, size in sizes.items():
            entry = manifest["files"][name]
            if (
                not isinstance(entry, dict)
                or entry.keys() != {"sha256", "size"}
                or not _digest(entry["sha256"])
                or entry["size"] != size
                or _owned_file(directory / name).st_size != size
            ):
                raise ValueError(f"invalid token segment inventory: {directory / name}")
        return manifest

    def _verified_index(self, directory: Path, manifest: dict, *, rows: bool = False):
        path = directory / "index.bin"
        stamp = artifact_fingerprint(path)
        if not rows and self.verified_indexes.get(path) == stamp:
            return None
        entry = manifest["files"]["index.bin"]
        with path.open("rb") as stream:
            raw = stream.read(entry["size"] + 1)
        if artifact_fingerprint(path) != stamp or (
            self.verified_indexes.get(path) != stamp
            and hashlib.sha256(raw).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"token segment index digest mismatch: {path}")
        self.verified_indexes[path] = stamp
        return raw

    def _register_segments(self):
        known = set(self.names.values())
        present = {path.name for path in self.segments.iterdir()}
        if known - present:
            raise ValueError("sealed token segment is missing from the store")
        for name in sorted(present - known):
            self._register(name)

    def _register(self, name: str):
        directory = self.segments / name
        manifest = self._read_manifest(directory)
        if canonical_digest(manifest) != name:
            raise ValueError(f"token segment identity mismatch: {directory}")
        records = np.frombuffer(
            self._verified_index(directory, manifest, rows=True),
            dtype=CONTENT_INDEX_DTYPE,
        )
        ends = records["offset"] + records["count"]
        if (
            len(records) != manifest["documents"]
            or records["offset"][0] != 0
            or np.any(records["offset"][1:] != ends[:-1])
            or ends[-1] != manifest["tokens"]
        ):
            raise ValueError(f"invalid token segment offsets: {directory}")
        with self.db:
            segment = self.db.execute(
                "INSERT INTO segment(name) VALUES (?)", (name,)
            ).lastrowid
            self.db.executemany(
                "INSERT INTO document VALUES (?, ?, ?)",
                (
                    (bytes(content), segment, row)
                    for row, content in enumerate(records["content"])
                ),
            )
        self.names[segment] = name

    def _open_active(self):
        for name in (*_FILES, "lock"):
            path = self.active_path / name
            if path.exists() or path.is_symlink():
                info = _owned_file(path)
                if (
                    name == "index.bin"
                    and info.st_size
                    >= (SEGMENT_DOCUMENTS + 1) * CONTENT_INDEX_DTYPE.itemsize
                ):
                    raise ValueError(
                        f"active token index exceeds segment bound: {path}"
                    )
        self.active = ContentTokens(
            self.active_root, self.tokenizer_sha, track_digest=True
        )

    def _publish(self, manifest: dict) -> str:
        # The outer writer lock stays in place when the active directory moves.
        (self.active_path / "lock").unlink(missing_ok=True)
        sync_directory(self.active_path)
        name = canonical_digest(manifest)
        rename_noreplace(self.active_path, self.segments / name)
        sync_directory(self.segments)
        sync_directory(self.active_root)
        return name

    def _seal(self):
        active = self.active
        if active is None or not len(active):
            return
        manifest = {
            "schema_version": 1,
            "encoding": self.encoding,
            "documents": len(active),
            "tokens": active.token_count,
            "files": active.seal_files(),
        }
        temporary = self.active_path / ".segment.tmp"
        temporary.unlink(missing_ok=True)
        write_canonical_json(temporary, manifest, exclusive=True)
        rename_noreplace(temporary, self.active_path / _MANIFEST)
        self.active = None
        active.__exit__(None, None, None)
        stamp = artifact_fingerprint(self.active_path / "index.bin")
        name = self._publish(manifest)
        self.verified_indexes[self.segments / name / "index.bin"] = stamp
        self._register(name)
        self.published_segments += 1

    def _location(self, key: bytes):
        if not self.names:
            return None
        return self.db.execute(
            "SELECT segment, row FROM document WHERE content=?",
            (key,),
        ).fetchone()

    def _read(self, key: bytes, location) -> bytes:
        segment, row = location
        name = self.names.get(segment)
        if not _digest(name):
            raise ValueError("invalid token segment locator")
        streams = self.handles.get(name)
        if streams is None:
            directory = self.segments / name
            manifest = self._read_manifest(directory)
            if canonical_digest(manifest) != name:
                raise ValueError(f"token segment identity mismatch: {directory}")
            self._verified_index(directory, manifest)
            if len(self.handles) == _OPEN_SEGMENTS:
                old_index, _, handle = self.handles.popitem(last=False)[1]
                with handle:
                    self._check_index(old_index)
            with ExitStack() as opening:
                index = opening.enter_context((directory / "index.bin").open("rb"))
                ids = opening.enter_context((directory / "tokens.ids").open("rb"))
                self._check_index(index)
                streams = index, ids, opening.pop_all()
            self.handles[name] = streams
        self.handles.move_to_end(name)
        index, ids, _ = streams
        record_size = CONTENT_INDEX_DTYPE.itemsize
        if type(row) is not int or not 0 <= row < SEGMENT_DOCUMENTS:
            raise ValueError("invalid token segment row locator")
        record = np.frombuffer(
            read_file_range(index, row * record_size, record_size),
            dtype=CONTENT_INDEX_DTYPE,
        )[0]
        if bytes(record["content"]) != key:
            raise ValueError("token segment locator points to different content")
        payload = read_file_range(
            ids, 4 * int(record["offset"]), 4 * int(record["count"])
        )
        if hashlib.sha256(payload).digest() != bytes(record["digest"]):
            raise ValueError("sealed token segment payload digest mismatch")
        return payload

    def _check_index(self, index):
        if self.verified_indexes[Path(index.name)] != artifact_fingerprint(
            index.fileno()
        ):
            raise ValueError(f"token segment index changed while open: {index.name}")

    def lookup(self, content: str, *, count_miss: bool = True, decode: bool = True):
        started = time.monotonic()
        assert self.active is not None
        payload = (
            self.active.lookup(content, count_miss=False, decode=False)
            if len(self.active)
            else None
        )
        if payload is None and _digest(content):
            key = bytes.fromhex(content)
            if (location := self._location(key)) is not None:
                payload = self._read(key, location)
                self.sealed_hits += 1
        self.lookup_seconds += time.monotonic() - started
        if payload is None:
            self.misses += count_miss
            return None
        self.hits += 1
        return self.decode_ids(payload) if decode else payload

    def store(self, content: bytes, ids: np.ndarray) -> None:
        active = self.active
        assert active is not None
        if content in active or self._location(content) is not None:
            return
        if len(active) and (
            len(active) >= SEGMENT_DOCUMENTS
            or active.token_count + len(ids) > SHARD_TOKENS
        ):
            self._seal()
            self._open_active()
        assert self.active is not None
        self.active.store(content, ids)
        self.misses += 1

    def finish(self):
        self._seal()
        for index, _, _ in self.handles.values():
            self._check_index(index)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            if self.active is not None:
                self.active.__exit__(*exc)
        finally:
            for _, _, handle in self.handles.values():
                handle.close()
            self.close_resources.close()
