"""Bounded readers for immutable JSON manifests with disk-backed row arrays.

SQLite is disposable operation-local storage, never authoritative evidence.
The JSON bytes and existing native verifiers remain the source of truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from klab.canonical import canonical_json

_MAX_INPUT_BYTES = 256 * 1024**2
_STRING_SPECIAL = re.compile(rb'["\\]')
_REUSED_INPUTS = 128
_REUSED_HEADER_BYTES = 8 * 1024**2  # Eviction budget, not a workload/RSS limit.


def _fingerprint(info):
    return tuple(
        getattr(info, name)
        for name in (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_nlink",
        )
    )


def _input_stat(path):
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise ValueError("unsafe metadata input")
    return path.stat()


class _HashedInput:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size):
        raw = self.stream.read(size)
        self.digest.update(raw)
        return raw


def _content_digest(stream, limit):
    digest = hashlib.sha256()
    consumed = 0
    while raw := stream.read(min(256 * 1024, limit - consumed + 1)):
        consumed += len(raw)
        if consumed > limit:
            raise ValueError("metadata input-byte cap exceeded")
        digest.update(raw)
    return digest.hexdigest()


def _metadata_view(value):
    """Fresh small headers, shared immutable row backing; never expand rows."""
    if isinstance(value, DiskRows):
        result = DiskRows(value.store, value.identifier)
        result._length = value._length
        result._shared = True
        return result
    if isinstance(value, dict):
        return {key: _metadata_view(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_metadata_view(item) for item in value]
    return value


def _materialization_present(value):
    if isinstance(value, DiskRows):
        if not value._length:
            return True
        # SQLite can roll back earlier imports after a capacity/write error.
        # Whole collections are private and otherwise changed only by append
        # detachment or explicit scoped release; check their indexed endpoints.
        return all(
            value.store.connection.execute(
                "SELECT 1 FROM rows WHERE collection=? AND ordinal=?",
                (value.identifier, ordinal),
            ).fetchone()
            is not None
            for ordinal in {0, value._length - 1}
        )
    if isinstance(value, dict):
        return all(_materialization_present(item) for item in value.values())
    if isinstance(value, list):
        return all(_materialization_present(item) for item in value)
    return True


class DiskRows(list):
    """Read-only list interface for ordered JSON rows, valid inside its store."""

    def __init__(self, store, identifier):
        self.store = store
        self.identifier = identifier
        self._length = 0
        self._shared = False

    def __len__(self):
        return self._length

    def __repr__(self):
        return f"DiskRows(length={len(self)})"

    def __iter__(self):
        cursor = self.store.connection.execute(
            "SELECT payload FROM rows WHERE collection=? ORDER BY ordinal",
            (self.identifier,),
        )
        try:
            for (payload,) in cursor:
                yield json.loads(payload)
        finally:
            cursor.close()

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("disk-backed rows require an integer index")
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        row = self.store.connection.execute(
            "SELECT payload FROM rows WHERE collection=? AND ordinal=?",
            (self.identifier, index),
        ).fetchone()
        return json.loads(row[0])

    def __eq__(self, other):
        if not isinstance(other, (list, tuple)) or len(self) != len(other):
            return False
        return all(left == right for left, right in zip(self, other, strict=True))

    def __ne__(self, other):
        return not self == other

    def __contains__(self, value):
        return any(row == value for row in self)

    def __reversed__(self):
        for index in range(len(self) - 1, -1, -1):
            yield self[index]

    def count(self, value):
        return sum(row == value for row in self)

    def index(self, value, start=0, stop=None):
        start, stop, _ = slice(start, stop).indices(len(self))
        for index in range(start, stop):
            if self[index] == value:
                return index
        raise ValueError("value is not in disk-backed rows")

    def _materialization(self, *args, **kwargs):
        raise TypeError(
            "explicitly iterate disk-backed rows; materialization is unsupported"
        )

    copy = __add__ = __radd__ = __mul__ = __rmul__ = _materialization

    def _readonly(self, *args, **kwargs):
        raise TypeError("disk-backed rows are read-only; use the owning store")

    append = extend = insert = clear = pop = remove = reverse = sort = _readonly
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _readonly


class MetadataStore:
    """Owned disk scratch with an optional explicit quota; cache size is not RSS enforcement."""

    def __init__(self, directory: Path, *, max_database_bytes: int | None):
        if max_database_bytes is not None and (
            type(max_database_bytes) is not int or max_database_bytes < 16384
        ):
            raise ValueError("metadata database cap must be at least 16384 bytes")
        if any(path.is_symlink() for path in (directory, *directory.parents)):
            raise ValueError("metadata scratch directory must not be a symlink")
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix=".metadata-", dir=directory)
        self.connection = sqlite3.connect(Path(self.temporary.name) / "rows.sqlite")
        self.next_identifier = 0
        self.max_database_bytes = max_database_bytes
        self._loaded = OrderedDict()
        self._loaded_header_bytes = 0
        self._assignment_lookups = {}
        try:
            self.connection.execute("PRAGMA page_size=4096")
            if max_database_bytes is not None:
                self.connection.execute(
                    f"PRAGMA max_page_count={max_database_bytes // 4096}"
                )
            self.connection.execute("PRAGMA cache_size=-4096")
            self.connection.execute("PRAGMA temp_store=FILE")
            # No rollback log is needed for disposable, unpublished derived rows.
            self.connection.execute("PRAGMA journal_mode=OFF")
            self.connection.execute(
                "CREATE TABLE rows(collection INTEGER, ordinal INTEGER, payload BLOB, "
                "PRIMARY KEY(collection, ordinal)) WITHOUT ROWID"
            )
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        for lookup in self._assignment_lookups.values():
            lookup.close()
        self._assignment_lookups.clear()
        self._loaded.clear()
        self.connection.close()
        self.temporary.cleanup()

    def new_rows(self):
        self.next_identifier += 1
        return DiskRows(self, self.next_identifier)

    def release_after(self, identifier):
        """Release scoped scratch and invalidate only materializations using it."""
        for key, cached in tuple(self._loaded.items()):
            if cached[4] is not None and cached[4] > identifier:
                self._loaded_header_bytes -= cached[3]
                del self._loaded[key]
        self.connection.execute("DELETE FROM rows WHERE collection > ?", (identifier,))

    def append(self, rows, value):
        if rows.store is not self:
            raise ValueError("row collection belongs to another metadata store")
        try:
            if rows._shared:
                # Preserve the owning-store append API without changing another
                # reader or the cached input. Read-only use never copies rows.
                detached = self.new_rows()
                self.connection.execute(
                    "INSERT INTO rows SELECT ?, ordinal, payload FROM rows "
                    "WHERE collection=? AND ordinal<?",
                    (detached.identifier, rows.identifier, rows._length),
                )
                rows.identifier = detached.identifier
                rows._shared = False
            self.connection.execute(
                "INSERT INTO rows VALUES(?,?,?)",
                (
                    rows.identifier,
                    rows._length,
                    canonical_json(value),
                ),
            )
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode == sqlite3.SQLITE_FULL:
                raise ValueError("metadata database byte cap exceeded") from error
            raise
        rows._length += 1

    def load(
        self,
        path: Path,
        *,
        max_input_bytes: int,
        max_header_bytes=4 * 1024**2,
        max_record_bytes=8 * 1024**2,
    ):
        """Parse JSON strictly; spill allowlisted inventory arrays without rewriting input."""
        for value in (max_input_bytes, max_header_bytes, max_record_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("metadata limits must be positive integers")
        info = _input_stat(path)
        if info.st_size > max_input_bytes:
            raise ValueError("metadata input-byte cap exceeded")
        key = (
            str(path.absolute()),
            max_input_bytes,
            max_header_bytes,
            max_record_bytes,
        )
        stamp = _fingerprint(info)

        def stable(stream):
            opened = os.fstat(stream.fileno())
            current = _input_stat(path)
            if max(opened.st_size, current.st_size) > max_input_bytes:
                raise ValueError("metadata input-byte cap exceeded")
            if _fingerprint(opened) != stamp or _fingerprint(current) != stamp:
                raise ValueError("metadata input changed while reading")

        with path.open("rb") as stream:
            stable(stream)
            cached = self._loaded.get(key)
            if cached is not None and cached[0] == stamp:
                # Hash-only validation still checks exact bytes on filesystems
                # with coarse timestamps; it does not reparse/reinsert rows.
                digest = _content_digest(stream, max_input_bytes)
                stable(stream)
                if digest == cached[1] and _materialization_present(cached[2]):
                    self._loaded.move_to_end(key)
                    return _metadata_view(cached[2])
                stream.seek(0)
            if cached is not None:
                self._loaded_header_bytes -= self._loaded.pop(key)[3]
            first_identifier = self.next_identifier
            hashed = _HashedInput(stream)
            reader = _Reader(
                hashed, self, max_input_bytes, max_header_bytes, max_record_bytes
            )
            value = reader.read()
            stable(stream)
            if reader.header <= _REUSED_HEADER_BYTES:
                while self._loaded and (
                    len(self._loaded) >= _REUSED_INPUTS
                    or self._loaded_header_bytes + reader.header > _REUSED_HEADER_BYTES
                ):
                    _, previous = self._loaded.popitem(last=False)
                    self._loaded_header_bytes -= previous[3]
                self._loaded[key] = (
                    stamp,
                    hashed.digest.hexdigest(),
                    value,
                    reader.header,
                    self.next_identifier
                    if self.next_identifier > first_identifier
                    else None,
                )
                self._loaded_header_bytes += reader.header
                return _metadata_view(value)
            return value


def _spill_path(path):
    """Only inventory positions spill; schema declarations stay header-bounded."""
    if path in {(key,) for key in ("files", "records", "selected_rows")}:
        return True
    if len(path) == 3 and path[:2] == ("sources", None):
        return path[2] in {"files", "records"}
    if path == ("retrieval", "shards", None, "selected_rows"):
        return True
    return (
        len(path) == 7
        and path[0] == "sources"
        and isinstance(path[1], str)
        and path[2:] == ("receipt", "retrieval", "shards", None, "selected_rows")
    )


class _Reader:
    def __init__(self, stream, store, input_limit, header_limit, row_limit):
        self.stream, self.store = stream, store
        self.input_limit, self.header_limit, self.row_limit = (
            input_limit,
            header_limit,
            row_limit,
        )
        self.buffer = b""
        self.offset = self.total = self.header = 0
        self.row_bytes = None

    def peek(self):
        if self.offset == len(self.buffer):
            self.buffer = self.stream.read(min(4096, self.input_limit - self.total + 1))
            self.offset = 0
        return self.buffer[self.offset : self.offset + 1]

    def take(self):
        value = self.peek()
        if not value:
            raise ValueError("truncated metadata JSON")
        self.offset += 1
        self.total += 1
        if self.total > self.input_limit:
            raise ValueError("metadata input-byte cap exceeded")
        if self.row_bytes is None:
            self.header += 1
            if self.header > self.header_limit:
                raise ValueError("metadata header-byte cap exceeded")
        else:
            self.row_bytes += 1
            if self.row_bytes > self.row_limit:
                raise ValueError("metadata record-byte cap exceeded")
        return value

    def take_run(self, size):
        # String scans already have a buffered span. Charge it once, retaining
        # the first byte limit that the single-byte path would encounter.
        value = self.buffer[self.offset : self.offset + size]
        if not value:
            raise ValueError("truncated metadata JSON")
        size = len(value)
        input_left = self.input_limit - self.total
        field_left = (
            self.header_limit - self.header
            if self.row_bytes is None
            else self.row_limit - self.row_bytes
        )
        if size > min(input_left, field_left):
            if input_left <= field_left:
                raise ValueError("metadata input-byte cap exceeded")
            field = "header" if self.row_bytes is None else "record"
            raise ValueError(f"metadata {field}-byte cap exceeded")
        self.offset += size
        self.total += size
        if self.row_bytes is None:
            self.header += size
        else:
            self.row_bytes += size
        return value

    def whitespace(self):
        while self.peek() and self.peek() in b" \t\r\n":
            self.take()

    def expect(self, value):
        self.whitespace()
        if self.take() != value:
            raise ValueError("malformed metadata JSON")

    def string(self):
        self.expect(b'"')
        raw = bytearray(b'"')
        escaped = False
        while True:
            if escaped:
                raw.extend(self.take())
                escaped = False
                continue
            self.peek()
            special = _STRING_SPECIAL.search(self.buffer, self.offset)
            end = special.end() if special is not None else len(self.buffer)
            raw.extend(self.take_run(end - self.offset))
            if special is not None:
                if special[0] == b'"':
                    return json.loads(raw)
                escaped = True

    @contextmanager
    def record(self):
        self.row_bytes = 0
        try:
            yield
        finally:
            self.row_bytes = None

    def value(self, depth=0, *, path=(), spill=False):
        if depth > 64:
            raise ValueError("metadata nesting cap exceeded")
        self.whitespace()
        char = self.peek()
        if char == b'"':
            return self.string()
        if char == b"{":
            self.take()
            result = {}
            self.whitespace()
            if self.peek() == b"}":
                self.take()
                return result
            while True:
                key = self.string()
                if key in result:
                    raise ValueError("duplicate metadata object key")
                self.expect(b":")
                result[key] = self.value(
                    depth + 1,
                    path=(*path, key),
                    spill=self.row_bytes is None and _spill_path((*path, key)),
                )
                self.whitespace()
                end = self.take()
                if end == b"}":
                    return result
                if end != b",":
                    raise ValueError("malformed metadata object")
        if char == b"[":
            self.take()
            result = self.store.new_rows() if spill else []
            self.whitespace()
            if self.peek() == b"]":
                self.take()
                return result
            while True:
                if spill:
                    with self.record():
                        item = self.value(depth + 1, path=(*path, None))
                        self.whitespace()
                        end = self.take()
                    self.store.append(result, item)
                else:
                    result.append(self.value(depth + 1, path=(*path, None)))
                    self.whitespace()
                    end = self.take()
                if end == b"]":
                    return result
                if end != b",":
                    raise ValueError("malformed metadata array")
        raw = bytearray()
        while self.peek() and self.peek() not in b" \t\r\n,]}":
            raw.extend(self.take())
        if not raw:
            raise ValueError("missing metadata value")
        result = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("nonfinite metadata number")
            ),
        )

        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("nonfinite metadata number")
        return result

    def read(self):
        value = self.value()
        self.whitespace()
        if self.peek():
            raise ValueError("trailing metadata JSON")
        return value


_ACTIVE_STORE: ContextVar[MetadataStore | None] = ContextVar(
    "corpus_metadata_store", default=None
)


@contextmanager
def metadata_operation(directory: Path, *, max_database_bytes=1024**3):
    """Share disposable storage through nested native verification calls."""
    existing = _ACTIVE_STORE.get()
    if existing is not None:
        yield existing
        return
    with MetadataStore(directory, max_database_bytes=max_database_bytes) as store:
        token = _ACTIVE_STORE.set(store)
        try:
            yield store
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode == sqlite3.SQLITE_FULL:
                raise ValueError("metadata database byte cap exceeded") from error
            raise
        finally:
            _ACTIVE_STORE.reset(token)


def read_metadata(path: Path):
    store = _ACTIVE_STORE.get()
    if store is None:
        return json.loads(path.read_text(encoding="utf-8"))
    limit = (
        _MAX_INPUT_BYTES
        if store.max_database_bytes is not None
        else max(1, path.stat().st_size)
    )
    return store.load(path, max_input_bytes=limit)


class DiskDecisions(dict):
    """Integer-keyed decision lookup sharing the operation's bounded database."""

    def __init__(self, store):
        self.store = store
        self.rows = store.new_rows()

    def __len__(self):
        return self.rows._length

    def __iter__(self):
        cursor = self.store.connection.execute(
            "SELECT ordinal FROM rows WHERE collection=? ORDER BY ordinal",
            (self.rows.identifier,),
        )
        try:
            for (key,) in cursor:
                yield key
        finally:
            cursor.close()

    def __contains__(self, key):
        if type(key) is not int or not 0 <= key <= 2**63 - 1:
            return False
        return (
            self.store.connection.execute(
                "SELECT 1 FROM rows WHERE collection=? AND ordinal=?",
                (self.rows.identifier, key),
            ).fetchone()
            is not None
        )

    def __getitem__(self, key):
        if type(key) is not int or not 0 <= key <= 2**63 - 1:
            raise KeyError(key)
        row = self.store.connection.execute(
            "SELECT payload FROM rows WHERE collection=? AND ordinal=?",
            (self.rows.identifier, key),
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        if type(key) is not int or not 0 <= key <= 2**63 - 1:
            raise ValueError(
                "metadata decision index must be a nonnegative signed 64-bit integer"
            )
        if key in self:
            raise ValueError("duplicate metadata decision index")
        try:
            self.store.connection.execute(
                "INSERT INTO rows VALUES(?,?,?)",
                (
                    self.rows.identifier,
                    key,
                    canonical_json(value),
                ),
            )
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode == sqlite3.SQLITE_FULL:
                raise ValueError("metadata database byte cap exceeded") from error
            raise
        self.rows._length += 1

    def __eq__(self, other):
        from collections.abc import Mapping

        if not isinstance(other, Mapping) or len(self) != len(other):
            return False
        return all(key in other and value == other[key] for key, value in self.items())

    def __ne__(self, other):
        return not self == other

    def copy(self):
        raise TypeError(
            "explicitly iterate disk-backed decisions; materialization is unsupported"
        )

    def keys(self):
        return iter(self)

    def items(self):
        for key in self:
            yield key, self[key]

    def values(self):
        yield from self.rows


def decision_index():
    store = _ACTIVE_STORE.get()
    return DiskDecisions(store) if store is not None else {}


def current_metadata_store():
    store = _ACTIVE_STORE.get()
    if store is None:
        raise RuntimeError("bounded evidence requires an active metadata operation")
    return store
