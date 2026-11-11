from __future__ import annotations

import pytest

from klab.data.byte_hash import hash_bytes, hash_text, table_address


def test_equal_utf8_bytes_share_address_across_segment_boundaries() -> None:
    source = "café\n"
    reconstructed = "caf" + "é\n"
    assert source.encode("utf-8") == reconstructed.encode("utf-8")
    assert hash_text(source) == hash_text(reconstructed)
    assert table_address(source.encode("utf-8"), 257) == table_address(
        reconstructed.encode("utf-8"), 257
    )


def test_raw_bytes_do_not_apply_unicode_normalization() -> None:
    composed = "é"
    decomposed = "e\u0301"
    assert composed != decomposed
    assert composed.encode("utf-8") != decomposed.encode("utf-8")
    assert hash_text(composed) != hash_text(decomposed)


def test_terminal_separator_distinguishes_leading_zero_sequences() -> None:
    assert hash_bytes(b"\x00") != hash_bytes(b"\x00\x00")
    with pytest.raises(ValueError, match="positive"):
        table_address(b"x", 0)
