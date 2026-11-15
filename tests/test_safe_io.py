import sqlite3

import pytest

from klab.canonical import canonical_json
from klab.safe_io import database_byte_cap, symlinked, write_canonical_json


def test_symlinked_detects_link_in_any_ancestor(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    assert not symlinked(tmp_path / "real" / "file")
    assert symlinked(tmp_path / "link" / "file")


def test_exclusive_canonical_write_refuses_overwrite(tmp_path):
    target = tmp_path / "value.json"
    write_canonical_json(target, {"b": 1, "a": 2}, exclusive=True)
    assert target.read_bytes() == canonical_json({"a": 2, "b": 1}) + b"\n"
    with pytest.raises(FileExistsError):
        write_canonical_json(target, {}, exclusive=True)
    write_canonical_json(target, {})
    assert target.read_bytes() == b"{}\n"


def test_full_database_reports_the_named_cap(tmp_path):
    db = sqlite3.connect(tmp_path / "capped.sqlite")
    db.execute("PRAGMA max_page_count=8")
    db.execute("CREATE TABLE rows(payload BLOB)")
    with (
        pytest.raises(ValueError, match="cap exceeded"),
        database_byte_cap("cap exceeded"),
    ):
        for _ in range(100):
            db.execute("INSERT INTO rows VALUES(?)", (b"x" * 4096,))
