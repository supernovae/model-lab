"""Reusable binary token shards for JSONL text sources.

Tokenizing a source file with a tokenizer is a pure function of both, so its
result is stored once and shared by every model, token budget and code revision
that uses the same pair. A store is a directory of append-only shards:

    <root>/<key>/store.json            what the store holds, written last
    <root>/<key>/shard-000000.ids      int32 token IDs, each document then EOS
    <root>/<key>/shard-000000.ends.npy cumulative token count after each document

Documents are kept in source order, so the first N documents of the source are
the first N documents of the store, and a larger budget extends a store instead
of rebuilding it. Shards are sealed by rename and listed in ``store.json`` only
after they are complete; an interrupted build leaves at most ``*.tmp`` files,
which the next build removes, and resumes from the last sealed shard.

No database is involved: accounting is one small JSON document per store.

``ContentTokens`` holds the same int32 IDs for callers that select documents
by content instead of by position in a file, such as corpus mixtures.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import struct
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from klab import token_encode_worker
from klab.canonical import canonical_json
from klab.data.datasets import local_text_error
from klab.data.encoding import EncoderPool
from klab.token_encode_worker import LineError

STORE_VERSION = 1
SHARD_TOKENS = 1 << 26  # 256 MiB of int32 per shard
_BATCH_BYTES = 2 * 1024 * 1024
_BATCH_DOCUMENTS = 2048
_PARALLEL_BYTES = 32 * 1024 * 1024  # source bytes that repay one worker process
_COPY_BYTES = 4 * 1024 * 1024


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical_json(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class TokenStore:
    """Token shards for one (source file, tokenizer) pair."""

    def __init__(self, root: Path, identity: dict[str, object]) -> None:
        self.root = root
        self.identity = identity
        self.shards: list[dict[str, object]] = []
        self.complete = False
        document = root / "store.json"
        if document.is_file():
            payload = json.loads(document.read_text(encoding="utf-8"))
            if payload.get("identity") != identity:
                raise ValueError(f"token store identity mismatch: {root}")
            self.shards = payload["shards"]
            self.complete = payload["complete"]

    @classmethod
    def open(
        cls, base: Path, *, source_sha256: str, tokenizer_sha256: str, eos: int
    ) -> TokenStore:
        identity: dict[str, object] = {
            "version": STORE_VERSION,
            "source": "local_text",
            "source_sha256": source_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "eos": eos,
            "dtype": "int32",
        }
        key = hashlib.sha256(canonical_json(identity)).hexdigest()[:32]
        return cls(base / key, identity)

    @property
    def documents(self) -> int:
        return sum(int(shard["documents"]) for shard in self.shards)

    @property
    def tokens(self) -> int:
        return sum(int(shard["tokens"]) for shard in self.shards)

    @property
    def size_bytes(self) -> int:
        return 4 * self.tokens

    def ends(self) -> np.ndarray:
        """Cumulative token count after each stored document."""
        parts, base = [], 0
        for shard in self.shards:
            local = np.load(self.root / f"{shard['name']}.ends.npy", allow_pickle=False)
            parts.append(local + np.uint64(base))
            base += int(shard["tokens"])
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64)

    def max_document_bytes(self, documents: int) -> int:
        """Largest source document among the shards holding the first ones."""
        largest, held = 0, 0
        for shard in self.shards:
            if held >= documents:
                break
            largest = max(largest, int(shard["max_document_bytes"]))
            held += int(shard["documents"])
        return largest

    def _save(self) -> None:
        _write_json(
            self.root / "store.json",
            {
                "identity": self.identity,
                "shards": self.shards,
                "complete": self.complete,
            },
        )

    def _discard_unfinished(self) -> None:
        listed = {str(shard["name"]) for shard in self.shards}
        for entry in self.root.iterdir():
            name = entry.name
            if name == "store.json":
                continue
            if name.endswith(".tmp") or name.split(".", 1)[0] not in listed:
                entry.unlink()

    def extend(
        self,
        source: Path,
        tokenizer: Tokenizer,
        *,
        documents: int,
        max_tokens: int,
        source_bytes: int | None,
        workers: int,
        split: str = "train",
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Tokenize further documents until the store holds enough.

        Stops once ``documents`` are stored, once the stored tokens exceed
        ``max_tokens`` (the caller refuses such a selection anyway), or at the
        end of the source. Each sealed shard is durable before it is listed.
        """
        if self.complete or self.documents >= documents or self.tokens > max_tokens:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self._discard_unfinished()
        offset = int(self.shards[-1]["source_end"]) if self.shards else 0
        number = self.documents + 1
        remaining = source.stat().st_size - offset
        workers = max(1, min(workers, math.ceil(remaining / _PARALLEL_BYTES)))
        eos = int(self.identity["eos"])  # type: ignore[call-overload]
        writer = _ShardWriter(self, offset)
        stored, tokens = self.documents, self.tokens
        try:
            for ids, lengths, sizes, end in _encoded_batches(
                source, offset, tokenizer, eos, workers, source_bytes, split, number
            ):
                # Keep only what the selection can use, at document boundaries.
                keep = min(len(lengths), documents - stored)
                if keep < len(lengths):
                    ids = ids[: int(lengths[:keep].sum())]
                    lengths, sizes, end = lengths[:keep], sizes[:keep], end[:keep]
                writer.append(ids, lengths, sizes, end)
                stored += keep
                tokens += len(ids)
                if progress is not None:
                    progress(stored, tokens)
                if stored >= documents or tokens > max_tokens:
                    break
            else:
                self.complete = True
            writer.seal()
            self._save()
        finally:
            writer.close()

    def copy_tokens(self, tokens: int, write: Callable[[bytes], object]) -> None:
        """Stream the first ``tokens`` IDs to ``write``, checking every shard read.

        Each shard touched is read to its end and compared with the digest
        recorded when it was sealed, so a damaged store is never copied silently.
        """
        remaining = 4 * tokens
        for shard in self.shards:
            if remaining <= 0:
                break
            digest = hashlib.sha256()
            path = self.root / f"{shard['name']}.ids"
            with path.open("rb") as handle:
                while block := handle.read(_COPY_BYTES):
                    digest.update(block)
                    if remaining > 0:
                        write(block[:remaining])
                        remaining -= min(remaining, len(block))
            if digest.hexdigest() != shard["sha256"]:
                raise ValueError(f"token shard changed after it was sealed: {path}")
        if remaining:
            raise ValueError("token store holds fewer tokens than requested")


class _ShardWriter:
    """Append encoded batches to the current shard and seal it when full."""

    def __init__(self, store: TokenStore, source_end: int) -> None:
        self.store = store
        self.source_end = source_end
        self.handle = None
        self._reset()

    def _reset(self) -> None:
        self.digest = hashlib.sha256()
        self.tokens = 0
        self.ends: list[np.ndarray] = []
        self.largest = 0

    def append(
        self, ids: np.ndarray, lengths: np.ndarray, sizes: np.ndarray, end: np.ndarray
    ) -> None:
        if not len(lengths):
            return
        if self.handle is None:
            name = f"shard-{len(self.store.shards):06d}"
            self.name = name
            self.handle = (self.store.root / f"{name}.ids.tmp").open("wb")
        data = ids.tobytes()
        self.handle.write(data)
        self.digest.update(data)
        self.ends.append(np.cumsum(lengths, dtype=np.uint64) + np.uint64(self.tokens))
        self.tokens += len(ids)
        self.largest = max(self.largest, int(sizes.max()))
        self.source_end = int(end[-1])
        if self.tokens >= SHARD_TOKENS:
            self.seal()
            self.store._save()

    def seal(self) -> None:
        if self.handle is None:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.handle = None
        root, name = self.store.root, self.name
        ends = np.concatenate(self.ends)
        with (root / f"{name}.ends.npy.tmp").open("wb") as handle:
            np.save(handle, ends, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        (root / f"{name}.ends.npy.tmp").replace(root / f"{name}.ends.npy")
        (root / f"{name}.ids.tmp").replace(root / f"{name}.ids")
        self.store.shards.append(
            {
                "name": name,
                "documents": len(ends),
                "tokens": self.tokens,
                "source_end": self.source_end,
                "max_document_bytes": self.largest,
                "sha256": self.digest.hexdigest(),
            }
        )
        self._reset()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.store.root.is_dir():
            self.store._discard_unfinished()


def _encoded_batches(
    source: Path,
    offset: int,
    tokenizer: Tokenizer,
    eos: int,
    workers: int,
    source_bytes: int | None,
    split: str,
    first_number: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(ids, token_counts, text_bytes, source_ends)`` in source order.

    The parent only cuts the file into runs of whole lines; workers parse,
    validate and tokenize them.
    """
    # Escaped JSON may need six raw bytes per decoded UTF-8 byte.
    line_limit = 6 * source_bytes + 1024 if source_bytes is not None else -1
    header = struct.pack(
        "<qq", line_limit, -1 if source_bytes is None else source_bytes
    )

    def chunks() -> Iterator[bytes]:
        with source.open("rb") as handle:
            handle.seek(offset)
            while data := handle.read(_BATCH_BYTES):
                if not data.endswith(b"\n"):
                    data += handle.readline()
                yield data

    number, position = first_number, offset

    def located(
        reply: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nonlocal number, position
        ids, counts, text_bytes, line_bytes = reply
        ends = np.cumsum(line_bytes, dtype=np.uint64) + np.uint64(position)
        number += len(counts)
        position = int(ends[-1]) if len(ends) else position
        return ids, counts, text_bytes, ends

    def refused(error: LineError) -> ValueError:
        return local_text_error(source, number + error.index, error, split)

    if workers <= 1:
        token_encode_worker.initialize(tokenizer.to_str(), eos)
        for chunk in chunks():
            try:
                reply = token_encode_worker.encode_lines(
                    chunk, line_limit, source_bytes
                )
            except LineError as error:
                raise refused(error) from error
            yield located(reply)
        return
    pool = EncoderPool(workers, tokenizer.to_str(), eos)
    try:
        pending: deque[Future] = deque()

        def finished() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            try:
                ids, counts, text_bytes, line_bytes = pending.popleft().result()
                return located(
                    (
                        np.frombuffer(ids, dtype=np.int32),
                        np.frombuffer(counts, dtype=np.uint32),
                        np.frombuffer(text_bytes, dtype=np.uint32),
                        np.frombuffer(line_bytes, dtype=np.uint32),
                    )
                )
            except LineError as error:
                raise refused(error) from error

        for chunk in chunks():
            pending.append(pool.submit(b"L" + header + chunk))
            # Bounded look-ahead keeps memory flat however large the source.
            while len(pending) >= 3 * workers:
                yield finished()
        while pending:
            yield finished()
    finally:
        pool.close()


_CONTENT_RECORD = np.dtype(
    [("content", "V32"), ("offset", "<u8"), ("count", "<u4"), ("digest", "V32")]
)
_CONTENT_FLUSH_DOCUMENTS = 4096


def _content_key(content: str) -> bytes:
    """32 bytes for a content identity: a SHA-256 hex digest as it is."""
    try:
        key = bytes.fromhex(content)
    except ValueError:
        key = b""
    return key if len(key) == 32 else hashlib.sha256(content.encode("utf-8")).digest()


class ContentTokens:
    """Token IDs of documents, found by the SHA-256 of their text.

    Two append-only files per tokenizer: ``tokens.ids`` (raw int32, no EOS) and
    ``index.bin`` (one fixed-width record per document: content digest, token
    offset, token count and the digest of its IDs). The index is read once into
    memory, so a lookup is one positioned read and never a database query.

    This is a cache and favors throughput over durability: nothing is synced.
    IDs are written before their index record, so after a crash the index is
    cut back to the records whose IDs are present and the rest are encoded
    again. Every read checks the stored digest of the IDs.
    """

    def __init__(self, root: Path, tokenizer_sha: str):
        name = hashlib.sha256(
            canonical_json(["content-tokens-v1", tokenizer_sha])
        ).hexdigest()
        self.path = root / name
        self.path.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("content token store must not be a symlink")
        self._lock = (self.path / "lock").open("ab")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._lock.close()
            raise ValueError(
                f"content token store is in use by another build: {self.path}"
            ) from error
        ids_path, index_path = self.path / "tokens.ids", self.path / "index.bin"
        records = (
            np.fromfile(index_path, dtype=_CONTENT_RECORD)
            if index_path.exists()
            else np.empty(0, dtype=_CONTENT_RECORD)
        )
        stored = ids_path.stat().st_size // 4 if ids_path.exists() else 0
        ends = records["offset"] + records["count"]
        complete = int(np.searchsorted(ends, stored, side="right"))
        records = records[:complete]
        self._tokens = int(ends[complete - 1]) if complete else 0
        self._ids = ids_path.open("ab+")
        self._ids.truncate(4 * self._tokens)
        self._index = index_path.open("ab")
        self._index.truncate(complete * _CONTENT_RECORD.itemsize)
        self._flushed = self._tokens
        contents = records["content"].tobytes()
        self._rows = {
            contents[position : position + 32]: row
            for row, position in enumerate(range(0, len(contents), 32))
        }
        self._offsets = records["offset"].tolist()
        self._counts = records["count"].tolist()
        self._digests = records["digest"].tobytes()
        self._new_digests: list[bytes] = []
        self._loaded = complete
        self.pending = 0
        self.hits = self.misses = 0
        self.lookup_seconds = self.write_seconds = self.encode_seconds = 0.0

    def __len__(self) -> int:
        return len(self._rows)

    def _digest(self, row: int) -> bytes:
        if row < self._loaded:
            return self._digests[32 * row : 32 * row + 32]
        return self._new_digests[row - self._loaded]

    def lookup(self, content: str, *, count_miss: bool = True, decode: bool = True):
        """The document's IDs (or their raw int32 bytes), or None if absent."""
        started = time.monotonic()
        row = self._rows.get(_content_key(content))
        if row is None:
            self.lookup_seconds += time.monotonic() - started
            if count_miss:
                self.misses += 1
            return None
        offset, count = self._offsets[row], self._counts[row]
        if offset + count > self._flushed:
            self._ids.flush()
            self._flushed = self._tokens
        payload = os.pread(self._ids.fileno(), 4 * count, 4 * offset)
        if hashlib.sha256(payload).digest() != self._digest(row):
            raise ValueError("cached token payload digest mismatch")
        self.lookup_seconds += time.monotonic() - started
        self.hits += 1
        return self.decode_ids(payload) if decode else payload

    @staticmethod
    def decode_ids(payload: bytes) -> list[int]:
        if len(payload) % 4:
            raise ValueError("invalid cached token IDs")
        ids = np.frombuffer(payload, dtype="<i4")
        if len(ids) and int(ids.min()) < 0:
            raise ValueError("invalid cached token IDs")
        return ids.tolist()

    def payload_bytes(self, content: str) -> int:
        row = self._rows.get(_content_key(content))
        return 0 if row is None else 4 * self._counts[row]

    def _append(self, content: bytes, ids: list[int]) -> None:
        payload = np.asarray(ids, dtype="<i4").tobytes()
        digest = hashlib.sha256(payload).digest()
        record = np.empty(1, dtype=_CONTENT_RECORD)
        record["content"], record["digest"] = np.void(content), np.void(digest)
        record["offset"], record["count"] = self._tokens, len(ids)
        self._ids.write(payload)
        self._index.write(record.tobytes())
        self._rows[content] = len(self._offsets)
        self._offsets.append(self._tokens)
        self._counts.append(len(ids))
        self._new_digests.append(digest)
        self._tokens += len(ids)
        self.pending += 1
        if self.pending >= _CONTENT_FLUSH_DOCUMENTS:
            self.flush()

    def flush(self) -> None:
        self._ids.flush()
        self._index.flush()
        self._flushed = self._tokens
        self.pending = 0

    def encode_many(self, texts, encode_missing):
        """Return encodings in input order; only absent content is encoded."""
        keys, known, missing = [], {}, {}
        for text in texts:
            content = hashlib.sha256(text.encode("utf-8")).hexdigest()
            keys.append(content)
            if content in known or content in missing:
                continue
            ids = self.lookup(content)
            if ids is None:
                missing[content] = text
            else:
                known[content] = ids
        if missing:
            started = time.monotonic()
            results = encode_missing(list(missing.values()))
            self.encode_seconds += time.monotonic() - started
            if len(results) != len(missing):
                raise ValueError("tokenizer returned incorrect batch length")
            started = time.monotonic()
            for content, ids in zip(missing, results, strict=True):
                self._append(_content_key(content), ids)
                known[content] = ids
            self.write_seconds += time.monotonic() - started
        return [known[content] for content in keys]

    def encode(self, tokenizer, text):
        return self.encode_many(
            [text],
            lambda texts: [
                tokenizer.encode(value, add_special_tokens=False).ids for value in texts
            ],
        )[0]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Valid encodings survive quota failure or cooperative interruption.
        try:
            self.flush()
        finally:
            self._ids.close()
            self._index.close()
            self._lock.close()


def list_stores(base: Path) -> list[dict[str, object]]:
    """Describe every store under ``base`` from its ``store.json`` alone."""
    described = []
    for document in sorted(base.glob("*/store.json")) if base.is_dir() else []:
        payload = json.loads(document.read_text(encoding="utf-8"))
        shards = payload["shards"]
        described.append(
            {
                "path": str(document.parent),
                "source_sha256": payload["identity"]["source_sha256"],
                "tokenizer_sha256": payload["identity"]["tokenizer_sha256"],
                "documents": sum(shard["documents"] for shard in shards),
                "tokens": sum(shard["tokens"] for shard in shards),
                "size_bytes": 4 * sum(shard["tokens"] for shard in shards),
                "shards": len(shards),
                "complete": payload["complete"],
            }
        )
    return described
