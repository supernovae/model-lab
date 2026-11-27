"""Private operation-owned row indexes; public schemas always contain plain data."""

import json
import sqlite3
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from uuid import uuid4

from klab.canonical import canonical_json
from klab.corpus.jsonl_records import iter_jsonl_records
from klab.safe_io import database_byte_cap, read_file_range, symlinked
from klab.verification_proofs import artifact_fingerprint


class _KeyIndex(Mapping):
    @staticmethod
    def encode(key):
        return canonical_json(key).decode("utf-8")

    def __contains__(self, key):
        return (
            self.connection.execute(
                f"SELECT 1 FROM {self.table} WHERE key=?", (self.encode(key),)
            ).fetchone()
            is not None
        )

    def __iter__(self):
        for row in self.connection.execute(
            f"SELECT key FROM {self.table} ORDER BY ordinal"
        ):
            key = json.loads(row[0])
            yield tuple(key) if isinstance(key, list) else key

    def __len__(self):
        return self.connection.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[
            0
        ]

    def same_keys(self, other):
        for left, right in ((self, other), (other, self)):
            if (
                self.connection.execute(
                    f"SELECT key FROM {left.table} EXCEPT SELECT key FROM {right.table} LIMIT 1"
                ).fetchone()
                is not None
            ):
                return False
        return True


class RowIndex(_KeyIndex):
    """Point lookup and repeatable streaming iteration in the owned operation DB."""

    def __init__(
        self,
        connection,
        rows,
        key,
        *,
        duplicate="duplicate evidence identity",
        max_record_bytes=None,
    ):
        self.connection = connection
        self.table = "evidence_" + uuid4().hex
        connection.execute(
            f"CREATE TABLE {self.table}(key TEXT PRIMARY KEY, ordinal INTEGER UNIQUE, payload BLOB) WITHOUT ROWID"
        )
        try:
            with database_byte_cap("evidence database byte cap exceeded"):
                for ordinal, row in enumerate(rows):
                    payload = canonical_json(row)
                    if max_record_bytes is not None and len(payload) > max_record_bytes:
                        raise ValueError("evidence record-byte cap exceeded")
                    connection.execute(
                        f"INSERT INTO {self.table} VALUES(?,?,?)",
                        (self.encode(key(row)), ordinal, payload),
                    )
        except sqlite3.IntegrityError as error:
            raise ValueError(duplicate) from error

    def __getitem__(self, key):
        row = self.connection.execute(
            f"SELECT payload FROM {self.table} WHERE key=?", (self.encode(key),)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def values(self):
        for row in self.connection.execute(
            f"SELECT payload FROM {self.table} ORDER BY ordinal"
        ):
            yield json.loads(row[0])

    def ordered_values(self):
        for row in self.connection.execute(
            f"SELECT payload FROM {self.table} ORDER BY json_extract(key,'$')"
        ):
            yield json.loads(row[0])


class EvidenceFiles:
    """Operation-owned generations and a bounded pool for indexed JSONL reads."""

    def __init__(self):
        self.generations = {}
        self.streams = OrderedDict()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_exc):
        try:
            if exc_type is None:
                for path, generation in self.generations.items():
                    if self._generation(path) != generation:
                        raise ValueError(f"indexed evidence changed: {path}")
        finally:
            for stream in self.streams.values():
                stream.close()
            self.streams.clear()

    @staticmethod
    def _generation(path):
        if symlinked(path) or not path.is_file():
            raise ValueError(f"missing or unsafe evidence input: {path}")
        return artifact_fingerprint(path)

    def records(self, path):
        self.bind(path)
        with path.open("rb") as stream:
            if artifact_fingerprint(stream.fileno()) != self.generations[path]:
                raise ValueError(f"indexed evidence changed: {path}")
            yield from iter_jsonl_records(stream, source=str(path))

    def bind(self, path):
        """Track an input generation even when a retained index supplies its ranges."""
        if path not in self.generations:
            self.generations[path] = self._generation(path)

    def row(self, path, offset, size):
        if path not in self.streams:
            if len(self.streams) == 32:
                self.streams.popitem(last=False)[1].close()
            stream = path.open("rb")
            if artifact_fingerprint(stream.fileno()) != self.generations[path]:
                stream.close()
                raise ValueError(f"indexed evidence changed: {path}")
            self.streams[path] = stream
        self.streams.move_to_end(path)
        raw = read_file_range(self.streams[path], offset, size)
        return json.loads(raw.decode("utf-8"))


class FileRowIndex(_KeyIndex):
    """Keys and byte ranges into generation-bound files, with optional metadata."""

    def __init__(
        self,
        connection,
        files,
        paths,
        key,
        *,
        duplicate="duplicate evidence identity",
        order=None,
        projection=None,
        table=None,
        progress=None,
    ):
        self.connection = connection
        self.files = files
        self.paths = tuple(paths.values())
        self.table = table or "evidence_" + uuid4().hex
        if not self.table.isidentifier():
            raise ValueError("invalid row index table name")
        self.last = None
        connection.execute(
            f"CREATE TABLE {self.table}(key TEXT PRIMARY KEY, ordinal INTEGER UNIQUE, "
            "member INTEGER, byte_offset INTEGER, byte_length INTEGER, "
            "order_name TEXT, order_position INTEGER, metadata BLOB) WITHOUT ROWID"
        )

        def references():
            ordinal = 0
            pending_rows = pending_bytes = 0
            for member, (label, path) in enumerate(paths.items()):
                for record in files.records(path):
                    if progress is not None:
                        pending_rows += 1
                        pending_bytes += len(record.raw)
                        if pending_rows >= 4096 or pending_bytes >= 64 * 1024**2:
                            progress.update(
                                documents=pending_rows, input_bytes=pending_bytes
                            )
                            pending_rows = pending_bytes = 0
                    row = record.value
                    row_key = key(label, row)
                    if row_key is None:
                        continue
                    # Preserve the payload store's refusal of values that cannot
                    # be represented by canonical, finite UTF-8 JSON.
                    canonical_json(row)
                    name, position = order(row) if order is not None else (None, None)
                    yield (
                        self.encode(row_key),
                        ordinal,
                        member,
                        record.byte_offset,
                        len(record.raw),
                        name,
                        position,
                        canonical_json(projection(row))
                        if projection is not None
                        else None,
                    )
                    ordinal += 1
            if progress is not None:
                progress.update(documents=pending_rows, input_bytes=pending_bytes)

        try:
            with database_byte_cap("evidence database byte cap exceeded"):
                connection.executemany(
                    f"INSERT INTO {self.table} VALUES (?,?,?,?,?,?,?,?)", references()
                )
        except sqlite3.IntegrityError as error:
            raise ValueError(duplicate) from error

    @classmethod
    def attach(cls, connection, files, path, database, alias):
        """Open a verified, single-member retained range index without copying it."""
        if not alias.isidentifier():
            raise ValueError("invalid row index database name")
        files.bind(path)
        connection.execute(
            f"ATTACH DATABASE ? AS {alias}", (database.resolve().as_uri() + "?mode=ro",)
        )
        index = cls.__new__(cls)
        index.connection = connection
        index.files = files
        index.paths = (path,)
        index.table = f"{alias}.records"
        index.last = None
        return index

    def _row(self, member, offset, size):
        return self.files.row(self.paths[member], offset, size)

    def __getitem__(self, key):
        encoded = self.encode(key)
        if self.last is not None and self.last[0] == encoded:
            return self.last[1]
        reference = self.connection.execute(
            f"SELECT member,byte_offset,byte_length FROM {self.table} WHERE key=?",
            (encoded,),
        ).fetchone()
        if reference is None:
            raise KeyError(key)
        row = self._row(*reference)
        # Consecutive lineage checks reuse one record without retaining wide rows.
        self.last = (encoded, row) if reference[-1] <= 1024**2 else None
        return row

    def metadata(self, key):
        row = self.connection.execute(
            f"SELECT metadata FROM {self.table} WHERE key=?", (self.encode(key),)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def metadata_rows(self):
        for (metadata,) in self.connection.execute(
            f"SELECT metadata FROM {self.table} ORDER BY ordinal"
        ):
            yield json.loads(metadata)

    def source_order(self, spans):
        """Sort only locators and join the matching input ranges once."""
        for row in self.connection.execute(
            f"SELECT d.member,d.byte_offset,d.byte_length,s.member,s.byte_offset,s.byte_length "
            f"FROM {self.table} d JOIN {spans.table} s ON d.key=s.key "
            "ORDER BY d.order_name,s.order_name,s.order_position,d.ordinal"
        ):
            yield self._row(*row[:3]), spans._row(*row[3:])


class ValueMap(MutableMapping):
    """Mutable scalar/tuple joins for split construction, private to one operation."""

    def __init__(self, connection):
        self.connection = connection
        self.table = "values_" + uuid4().hex
        connection.execute(
            f"CREATE TABLE {self.table}(key TEXT PRIMARY KEY, value BLOB) WITHOUT ROWID"
        )
        connection.execute(f"CREATE INDEX {self.table}_value ON {self.table}(value)")

    def __getitem__(self, key):
        row = self.connection.execute(
            f"SELECT value FROM {self.table} WHERE key=?", (RowIndex.encode(key),)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        value = json.loads(row[0])
        return tuple(value) if isinstance(value, list) else value

    def __setitem__(self, key, value):
        payload = canonical_json(value)
        if len(payload) > 8 * 1024**2:
            raise ValueError("evidence record-byte cap exceeded")
        with database_byte_cap("evidence database byte cap exceeded"):
            self.connection.execute(
                f"INSERT INTO {self.table} VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (RowIndex.encode(key), payload),
            )

    def __delitem__(self, key):
        if (
            self.connection.execute(
                f"DELETE FROM {self.table} WHERE key=?", (RowIndex.encode(key),)
            ).rowcount
            != 1
        ):
            raise KeyError(key)

    def __iter__(self):
        for row in self.connection.execute(
            f"SELECT key FROM {self.table} ORDER BY json_extract(key,'$')"
        ):
            key = json.loads(row[0])
            yield tuple(key) if isinstance(key, list) else key

    def __len__(self):
        return self.connection.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[
            0
        ]

    def keys_for_value(self, value):
        for row in self.connection.execute(
            f"SELECT key FROM {self.table} WHERE value=? ORDER BY key",
            (canonical_json(value),),
        ):
            yield json.loads(row[0])

    def same_keys(self, other):
        return _KeyIndex.same_keys(self, other)
