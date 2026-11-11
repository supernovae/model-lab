"""Operation-owned SQL joins for exact qualified source-row coverage."""

from __future__ import annotations

import sqlite3
from uuid import uuid4


class AdmissionCoverage:
    """Keep expected/observed identities in the caller's bounded database."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.table = "admission_coverage_" + uuid4().hex

    def __enter__(self):
        self.connection.execute(
            f"CREATE TABLE {self.table}(expected INTEGER,source TEXT,path TEXT,row_index INTEGER,"
            "PRIMARY KEY(expected,source,path,row_index)) WITHOUT ROWID"
        )
        return self

    def __exit__(self, *exc):
        self.connection.execute(f"DROP TABLE IF EXISTS {self.table}")

    def _add(self, expected, source, path, index, *, unique):
        command = "INSERT" if unique else "INSERT OR IGNORE"
        try:
            self.connection.execute(
                f"{command} INTO {self.table} VALUES(?,?,?,?)",
                (expected, source, path, index),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError("duplicate admitted source row") from error
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode == sqlite3.SQLITE_FULL:
                raise ValueError(
                    "admission coverage database byte cap exceeded"
                ) from error
            raise

    def expect(self, source, path, index):
        self._add(1, source, path, index, unique=False)

    def observe(self, source, path, index, *, unique=False):
        self._add(0, source, path, index, unique=unique)

    def verify(self):
        for left, right in ((1, 0), (0, 1)):
            if (
                self.connection.execute(
                    f"SELECT source,path,row_index FROM {self.table} WHERE expected=? "
                    f"EXCEPT SELECT source,path,row_index FROM {self.table} WHERE expected=? LIMIT 1",
                    (left, right),
                ).fetchone()
                is not None
            ):
                raise ValueError("admitted source row inventory differs from documents")
