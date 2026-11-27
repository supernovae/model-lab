"""Typed JSONL interchange for explicitly authorized Corpus commands."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from itertools import zip_longest
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, field_validator

from klab.config.models import StrictModel
from klab.corpus.jsonl_records import iter_lf_lines
from klab.corpus.mechanisms import configure_private_database
from klab.corpus.relation_sources import Digest, ExplicitRecord, ScopedEntity
from klab.safe_io import database_byte_cap

Kind = Literal[
    "local_text",
    "document_ids",
    "text_statistics",
    "relation_scope",
    "relation_records",
]
Count = Annotated[int, Field(strict=True, ge=0, lt=2**32)]
DOCUMENT_CONTRACTS = frozenset({"document_ids", "text_statistics"})


class Text(StrictModel):
    text: str

    @field_validator("text")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("document text must contain non-whitespace")
        return value


class DocumentID(StrictModel):
    document_id: Digest


class TextStatistics(DocumentID):
    utf8_bytes: Count
    codepoints: Count
    lines: Count


_CONTRACTS = {
    "local_text": TypeAdapter(Text),
    "document_ids": TypeAdapter(DocumentID),
    "text_statistics": TypeAdapter(TextStatistics),
    "relation_scope": TypeAdapter(ScopedEntity),
    "relation_records": TypeAdapter(ExplicitRecord),
}


def validated_rows(
    path: Path, contract: Kind, *, max_rows: int, max_record_bytes: int
) -> Iterator:
    """Validate every physical record, including refusal of blank lines."""
    adapter = _CONTRACTS[contract]
    with path.open("rb") as stream:
        for line, raw in iter_lf_lines(stream, max_record_bytes=max_record_bytes):
            if line > max_rows:
                raise ValueError(f"{path.name} exceeds its declared row limit")
            try:
                yield adapter.validate_json(raw)
            except ValueError:
                raise ValueError(
                    f"{path.name} line {line} violates its {contract} contract"
                ) from None


def aligned_rows(rows, source):
    """Require ordered one-to-one alignment and, for sidecars, exact document IDs."""
    absent = object()
    for row, parent in zip_longest(rows, source, fillvalue=absent):
        if row is absent or parent is absent or row.document_id != parent.document_id:
            raise ValueError("command output does not align with its declared input")
        yield row


def scoped_rows(rows, entities, scratch, max_bytes):
    """Check canonical entity references and split constraints with bounded disk state."""
    with (
        closing(sqlite3.connect(scratch)) as db,
        db,
        database_byte_cap("command scope index exceeds its scratch budget"),
    ):
        configure_private_database(db, max_bytes)
        db.executescript("""
            CREATE TABLE family (id TEXT PRIMARY KEY, split TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE entity (id TEXT PRIMARY KEY, split TEXT NOT NULL) WITHOUT ROWID;
        """)
        for entity in entities:
            admitted = db.execute(
                """INSERT INTO family VALUES (?,?) ON CONFLICT(id) DO UPDATE
                SET split=excluded.split WHERE split=excluded.split RETURNING split""",
                (entity.family, entity.split),
            ).fetchone()
            if admitted is None:
                raise ValueError(
                    "command relation scope places a family in multiple splits"
                )
            try:
                db.execute("INSERT INTO entity VALUES (?,?)", (entity.id, entity.split))
            except sqlite3.IntegrityError:
                raise ValueError(
                    "command relation scope repeats an entity ID"
                ) from None
        for row in rows:
            splits = {
                db.execute("SELECT split FROM entity WHERE id=?", (entity,)).fetchone()
                for entity in row.scoped_ids()
            }
            if None in splits or len(splits) != 1:
                raise ValueError(
                    "command relation output names an unscoped or cross-split entity"
                )
            yield row
