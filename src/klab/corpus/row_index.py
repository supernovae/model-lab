"""Private operation-owned row indexes; public schemas always contain plain data."""

import json
import sqlite3
from collections.abc import Mapping, MutableMapping
from uuid import uuid4

from klab.canonical import canonical_json


class RowIndex(Mapping):
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
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode == sqlite3.SQLITE_FULL:
                raise ValueError("evidence database byte cap exceeded") from error
            raise

    @staticmethod
    def encode(key):
        return canonical_json(key).decode("utf-8")

    def __getitem__(self, key):
        row = self.connection.execute(
            f"SELECT payload FROM {self.table} WHERE key=?", (self.encode(key),)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

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

    def source_order(self, spans):
        for row in self.connection.execute(
            f"SELECT d.payload FROM {self.table} d JOIN {spans.table} s ON d.key=s.key ORDER BY json_extract(d.payload,'$.source_id'),json_extract(s.payload,'$.raw_path'),json_extract(s.payload,'$.line_start'),d.ordinal"
        ):
            yield json.loads(row[0])


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
        try:
            self.connection.execute(
                f"INSERT INTO {self.table} VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (RowIndex.encode(key), payload),
            )
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode == sqlite3.SQLITE_FULL:
                raise ValueError("evidence database byte cap exceeded") from error
            raise

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
        return RowIndex.same_keys(self, other)
