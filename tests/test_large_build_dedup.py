from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from klab.corpus.large_build import _deduplicate_v2


class _Progress:
    def update(self, **_: object) -> None:
        pass


def _database(documents: list[tuple[str, str, str, str]]) -> sqlite3.Connection:
    """(document_id, split, raw_sha, normalized_sha) rows."""
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE documents (document_id TEXT NOT NULL UNIQUE, payload BLOB NOT NULL,
            span BLOB NOT NULL, split TEXT NOT NULL, source_id TEXT NOT NULL,
            source_family TEXT NOT NULL, document_kind TEXT NOT NULL,
            source_rank INTEGER NOT NULL, representative_id TEXT, drop_reason TEXT);
        CREATE TABLE origins (method TEXT NOT NULL, origin TEXT NOT NULL,
            document_number INTEGER NOT NULL);
        """
    )
    rights = {
        "training_eligibility": "eligible",
        "license_references": [],
        "notices": [],
        "training_restriction": None,
    }
    for identifier, split, raw, normalized in documents:
        payload = json.dumps(
            {"license": "MIT", "redistribution": "open", "rights": rights}
        ).encode()
        number = connection.execute(
            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,NULL,NULL)",
            (identifier, payload, b"{}", split, "s", "f", "prose", 2),
        ).lastrowid
        connection.executemany(
            "INSERT INTO origins VALUES (?,?,?)",
            [("raw", raw, number), ("normalized", normalized, number)],
        )
    return connection


def _run(connection: sqlite3.Connection, tmp_path: Path):
    return _deduplicate_v2(
        connection,
        tmp_path / "groups.jsonl",
        diagnostics=tmp_path / "diagnostics.json",
        progress=_Progress(),
    )


def _outcome(connection: sqlite3.Connection) -> dict[str, tuple[str, str | None]]:
    return {
        identifier: (representative, reason)
        for identifier, representative, reason in connection.execute(
            "SELECT document_id,representative_id,drop_reason FROM documents"
        )
    }


def test_train_copies_of_held_out_text_are_dropped_not_the_held_out_document(
    tmp_path: Path,
) -> None:
    # The train id sorts before the held-out id: it must still be the one dropped.
    connection = _database(
        [
            ("a-train", "train", "r1", "n1"),
            ("b-valid", "validation", "r1", "n1"),
            ("c-train", "train", "r2", "n2"),
            ("d-train", "train", "r2", "n2"),
        ]
    )
    groups, dropped = _run(connection, tmp_path)
    assert _outcome(connection) == {
        "a-train": ("b-valid", "contaminated_heldout"),
        "b-valid": ("b-valid", None),
        "c-train": ("c-train", None),
        "d-train": ("c-train", "duplicate"),
    }
    assert (groups, dropped) == (4, 2)
    record = json.loads((tmp_path / "groups.jsonl").read_text().splitlines()[0])
    assert record["splits"] == ["train", "validation"]
    assert record["representative"] == "b-valid"


def test_text_shared_by_two_held_out_splits_still_fails(tmp_path: Path) -> None:
    connection = _database(
        [
            ("a-train", "train", "r1", "n1"),
            ("b-valid", "validation", "r1", "n1"),
            ("c-test", "test", "r1", "n1"),
        ]
    )
    with pytest.raises(ValueError, match="cross-split exact text overlap"):
        _run(connection, tmp_path)
