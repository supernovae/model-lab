"""Stream inventory coverage against authenticated Corpus release documents."""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import closing
from itertools import chain
from pathlib import Path

from klab.corpus.jsonl_records import records_from_path
from klab.corpus.release import ReleaseReader
from klab.corpus.release_verify import scratch_root
from klab.corpus.row_index import RowIndex

_FIELDS = {"document_id", "family_id", "split", "stratum", "content_sha256"}
_MISMATCH = "family inventory disagrees with verified release"


def _inventory_rows(
    path: Path, *, max_record_bytes: int | None, expected_sha256: str | None
) -> Iterator[dict[str, str]]:
    for parsed in records_from_path(
        path, max_record_bytes=max_record_bytes, expected_sha256=expected_sha256
    ):
        row = parsed.value
        if (
            not isinstance(row, dict)
            or set(row) != _FIELDS
            or not all(isinstance(value, str) for value in row.values())
        ):
            raise ValueError(_MISMATCH)
        yield row


def _check_row(doc: dict, row: dict[str, str] | None) -> dict[str, str]:
    if (
        row is None
        or row["split"] != doc["split"]
        or row["content_sha256"] != doc["content_sha256"]
        or row["stratum"] not in doc["domains"]
    ):
        raise ValueError(_MISMATCH)
    return row


def _indexed_coverage(
    docs: Iterator[dict], rows: Iterator[dict[str, str]]
) -> Iterator[tuple[dict, dict[str, str]]]:
    with (
        tempfile.TemporaryDirectory(
            prefix="family-inventory-", dir=scratch_root()
        ) as scratch,
        closing(sqlite3.connect(Path(scratch) / "families.sqlite")) as db,
    ):
        db.execute("PRAGMA cache_size=-4096")
        # The index is private and disposable. No transaction needs recovery.
        db.execute("PRAGMA journal_mode=OFF")
        index = RowIndex(
            db, rows, key=lambda row: row["document_id"], duplicate=_MISMATCH
        )
        matched = 0
        for doc in docs:
            yield doc, _check_row(doc, index.get(doc["document_id"]))
            matched += 1
        # Any extra row, including a repeat of a matched prefix ID, remains unmatched.
        if matched != len(index):
            raise ValueError("family inventory does not cover kept release documents")


def covered_family_rows(
    release: ReleaseReader,
    path: Path,
    *,
    max_record_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> Iterator[dict[str, str]]:
    with (
        closing(release.rows("documents.jsonl")) as documents,
        closing(
            join_family_documents(
                documents,
                path,
                max_record_bytes=max_record_bytes,
                expected_sha256=expected_sha256,
            )
        ) as pairs,
    ):
        for _, row in pairs:
            yield row


def join_family_documents(
    documents: Iterator[dict],
    path: Path,
    *,
    max_record_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> Iterator[tuple[dict, dict[str, str]]]:
    """Stream kept documents with their inventory rows, indexing a reordered tail."""
    with closing(
        _inventory_rows(
            path, max_record_bytes=max_record_bytes, expected_sha256=expected_sha256
        )
    ) as rows:
        docs = (doc for doc in documents if doc["drop_reason"] is None)
        for doc in docs:
            row = next(rows, None)
            if row is None:
                raise ValueError(
                    "family inventory does not cover kept release documents"
                )
            if row["document_id"] != doc["document_id"]:
                yield from _indexed_coverage(chain((doc,), docs), chain((row,), rows))
                return
            yield doc, _check_row(doc, row)
        if next(rows, None) is not None:
            raise ValueError("family inventory does not cover kept release documents")
