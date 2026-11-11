"""Memory vocabulary equivalences never replace the model's lossless tokenizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models

from klab.canonical import canonical_digest
from klab.memory_tokenizer import (
    build_memory_token_map,
    load_memory_token_map,
    save_memory_token_map,
    validate_memory_token_map,
)


@pytest.fixture
def tokenizer_path(tmp_path: Path) -> Path:
    vocabulary = [
        "<unk>",
        "Value",
        "value",
        " VALUE ",
        "Válue",
        "Ｖａｌｕｅ",
        " ",
        "\t\n",
        "value  x",
        "value\tx",
        "value\\tx",
        "<SPECIAL>",
        "<special>",
        "different",
    ]
    tokenizer = Tokenizer(
        models.WordLevel(
            dict(zip(vocabulary, range(len(vocabulary)), strict=True)),
            unk_token="<unk>",
        )
    )
    tokenizer.add_special_tokens(
        [AddedToken("<unk>", special=True), AddedToken("<SPECIAL>", special=True)]
    )
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return path


def test_identity_preserves_every_token_and_original_tokenizer(
    tokenizer_path: Path, tmp_path: Path
) -> None:
    original = tokenizer_path.read_bytes()
    mapping = build_memory_token_map(tokenizer_path)
    assert mapping.token_ids == tuple(range(mapping.vocab_size))
    assert mapping.memory_vocab_size == mapping.vocab_size
    target = tmp_path / "memory-map.json"
    save_memory_token_map(mapping, target)
    assert (
        load_memory_token_map(
            target,
            tokenizer_path=tokenizer_path,
            expected_sha256=mapping.mapping_sha256,
        )
        == mapping
    )
    assert tokenizer_path.read_bytes() == original
    assert build_memory_token_map(tokenizer_path).to_dict() == mapping.to_dict()


@pytest.mark.parametrize("mode", ["deepseek_v1", "deepseek_safe_v1"])
def test_paper_normalization_is_explicitly_lossy_for_code(
    tokenizer_path: Path, mode: str
) -> None:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    original_tokens = [
        tokenizer.decode([i], skip_special_tokens=False)
        for i in range(tokenizer.get_vocab_size())
    ]
    mapping = build_memory_token_map(tokenizer_path, mode)
    # Case, accents, compatibility characters and boundary whitespace collapse.
    assert len(set(mapping.token_ids[1:6])) == 1
    # Whitespace-only tokens remain a space, not the empty-string class.
    assert mapping.token_ids[6] == mapping.token_ids[7]
    assert mapping.token_ids[6] != mapping.token_ids[1]
    assert mapping.token_ids[8] == mapping.token_ids[9]
    # Literal backslash-t is not an actual tab.
    assert mapping.token_ids[10] != mapping.token_ids[9]
    assert mapping.token_ids[13] != mapping.token_ids[1]
    assert mapping.memory_vocab_size < mapping.vocab_size
    assert original_tokens == [
        tokenizer.decode([i], skip_special_tokens=False)
        for i in range(tokenizer.get_vocab_size())
    ]
    assert tokenizer.token_to_id("Value") != tokenizer.token_to_id("value")


def test_safe_mode_keeps_special_tokens_distinct(tokenizer_path: Path) -> None:
    demo = build_memory_token_map(tokenizer_path, "deepseek_v1")
    safe = build_memory_token_map(tokenizer_path, "deepseek_safe_v1")
    assert demo.token_ids[11] == demo.token_ids[12]
    assert safe.token_ids[11] != safe.token_ids[12]
    assert demo.mapping_sha256 != safe.mapping_sha256


@pytest.mark.parametrize("mode", ["deepseek_v1", "deepseek_safe_v1"])
def test_byte_fallback_fragments_do_not_merge_on_replacement_character(
    tmp_path: Path, mode: str
) -> None:
    vocabulary = {"<unk>": 0, "<0x80>": 1, "<0x81>": 2, "�": 3, "<0xE9>": 4}
    tokenizer = Tokenizer(
        models.BPE(vocab=vocabulary, merges=[], unk_token="<unk>", byte_fallback=True)
    )
    tokenizer.decoder = decoders.ByteFallback()
    tokenizer.add_special_tokens(["<unk>"])
    path = tmp_path / "bytes.json"
    tokenizer.save(str(path))
    assert tokenizer.decode([1]) == tokenizer.decode([2]) == "�"
    mapping = build_memory_token_map(path, mode)
    assert len(set(mapping.token_ids)) == len(vocabulary)
    assert mapping.token_ids[1] != mapping.token_ids[2] != mapping.token_ids[3]


def test_map_rejects_tampering_wrong_tokenizer_and_unexpected_fields(
    tokenizer_path: Path, tmp_path: Path
) -> None:
    mapping = build_memory_token_map(tokenizer_path, "deepseek_safe_v1")
    changed = mapping.to_dict()
    changed["token_ids"][1] = mapping.token_ids[13]
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_memory_token_map(changed)
    with pytest.raises(ValueError, match="expected digest mismatch"):
        validate_memory_token_map(mapping.to_dict(), expected_sha256="0" * 64)
    wrong = tmp_path / "other-tokenizer.json"
    wrong.write_bytes(tokenizer_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="tokenizer digest mismatch"):
        validate_memory_token_map(mapping.to_dict(), tokenizer_path=wrong)
    with pytest.raises(ValueError, match="exactly"):
        validate_memory_token_map({**mapping.to_dict(), "ignored": True})
    target = tmp_path / "duplicate.json"
    target.write_text(json.dumps(mapping.to_dict())[:-1] + ', "vocab_size": 14}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_memory_token_map(target)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"algorithm": "unknown"}, "unsupported"),
        ({"vocab_size": True}, "positive integer"),
        ({"memory_vocab_size": 0}, "memory_vocab_size"),
        ({"token_ids": [0]}, "cover every"),
        ({"token_ids": [False] * 14}, "integers"),
        ({"token_ids": [0] * 14}, "surjective"),
        ({"tokenizer_sha256": "bad"}, "tokenizer_sha256"),
    ],
)
def test_malformed_maps_are_refused_before_allocation(
    tokenizer_path: Path, changes: dict[str, object], message: str
) -> None:
    payload = build_memory_token_map(tokenizer_path).to_dict()
    payload.update(changes)
    with pytest.raises(ValueError, match=message):
        validate_memory_token_map(payload)


def test_identity_cannot_be_relabelled_compression(tokenizer_path: Path) -> None:
    payload = build_memory_token_map(tokenizer_path, "deepseek_v1").to_dict()
    payload["algorithm"] = "identity"
    payload["mapping_sha256"] = canonical_digest(
        {key: value for key, value in payload.items() if key != "mapping_sha256"}
    )
    with pytest.raises(ValueError, match="preserve all"):
        validate_memory_token_map(payload)


def test_tokenizer_bound_validation_refuses_resigned_wrong_algorithm_map(
    tokenizer_path: Path,
) -> None:
    payload = build_memory_token_map(tokenizer_path, "deepseek_safe_v1").to_dict()
    ids = payload["token_ids"]
    # Relabel all memory IDs: still complete, surjective, and self-consistently
    # hashed, but not the declared first-seen-ID assignment for this tokenizer.
    payload["token_ids"] = [payload["memory_vocab_size"] - 1 - item for item in ids]
    payload["mapping_sha256"] = canonical_digest(
        {key: value for key, value in payload.items() if key != "mapping_sha256"}
    )
    validate_memory_token_map(payload)  # Filesystem-free structural verification.
    with pytest.raises(ValueError, match="declared algorithm"):
        validate_memory_token_map(payload, tokenizer_path=tokenizer_path)
