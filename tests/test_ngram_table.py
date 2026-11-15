"""The corpus n-gram table: build, verify, draft."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from klab.evaluation.ngram_table import (
    CorpusNgramSpeculation,
    build_ngram_table,
    load_ngram_table,
)

SOURCE = {"tokenizer_sha256": "a" * 64, "run_id": "r"}
EOS = 2


def _build(root: Path, stream: list[int], **options: object) -> dict:
    settings = {
        "order": 2,
        "table_size": 1 << 12,
        "vocab_size": 260,
        "eos_id": EOS,
        "source": SOURCE,
    }
    return build_ngram_table(
        np.array(stream, dtype=np.int32), root, **{**settings, **options}
    )


def test_the_table_drafts_the_most_frequent_continuation_and_stops_at_eos(
    tmp_path: Path,
) -> None:
    # (10, 11) is followed by 12 twice and by 15 once, then 12 runs on to EOS.
    stream = [10, 11, 12, 13, 14, EOS, 10, 11, 12, 13, 14, EOS, 10, 11, 15]
    _build(tmp_path / "table", stream)
    table = load_ngram_table(tmp_path / "table", tokenizer_sha256="a" * 64)
    assert table.lookup([10, 11]) == (12, 2)
    drafter = CorpusNgramSpeculation(table, max_draft_tokens=8)
    assert drafter.draft([9, 10, 11], 8) == [12, 13, 14, EOS]
    assert drafter.draft([9, 10, 11], 2) == [12, 13]
    assert drafter.draft([11], 8) == []  # shorter than the order
    assert drafter.draft([99, 98], 8) == []  # never seen
    # A context holding EOS spans two documents and is not counted.
    assert table.lookup([EOS, 10]) is None
    assert CorpusNgramSpeculation(table, min_count=3).draft([10, 11], 8) == []


def test_a_table_is_refused_when_it_changed_or_was_built_for_another_tokenizer(
    tmp_path: Path,
) -> None:
    manifest = _build(tmp_path / "table", [10, 11, 12] * 20)
    with pytest.raises(ValueError, match="different tokenizer"):
        load_ngram_table(tmp_path / "table", tokenizer_sha256="b" * 64)
    with pytest.raises(FileExistsError):
        _build(tmp_path / "table", [1, 2, 3])
    assert manifest["filled_slots"] == 3
    counts = tmp_path / "table" / "counts.npy"
    data = bytearray(counts.read_bytes())
    data[-1] ^= 1
    counts.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_ngram_table(tmp_path / "table", tokenizer_sha256="a" * 64)
    (tmp_path / "table" / "extra").write_text("x")
    with pytest.raises(ValueError, match="inventory"):
        load_ngram_table(tmp_path / "table", tokenizer_sha256="a" * 64)


def test_the_table_is_bounded_and_deterministic(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    stream = rng.integers(3, 260, size=5000)
    first = _build(tmp_path / "a", list(stream), table_size=64)
    second = _build(tmp_path / "b", list(stream), table_size=64)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["filled_slots"] <= 64
    assert (tmp_path / "a" / "tokens.npy").stat().st_size < 2000
