"""Regression: memory-table addresses must not degenerate at any table size.

With the legacy base-257 recurrence, ``memory_table_size: 257`` turned hash
head 0 of every token n-gram table into a single-token table (the address was
the oldest token in the window modulo 257) and sent every raw-byte address to
row 0. Table sizes coprime with the multiplier must keep their legacy
addresses bit-for-bit, so existing checkpoints and packs stay valid.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
import torch
from test_training import config as training_config

from klab.address_hash import (
    BYTE_SCHEME_V1,
    BYTE_SCHEME_V2,
    ENGRAM_SCHEME_V1,
    TOKEN_SCHEME_V1,
    TOKEN_SCHEME_V2,
    byte_scheme,
    engram_address_finish,
    engram_address_step,
    engram_multipliers,
    engram_ngram_address,
    engram_table_sizes,
    token_addressing_marker,
    token_ngram_address,
    token_scheme,
    validate_engram_table_sizes,
)
from klab.cli.main import main
from klab.config.models import RunConfig
from klab.data.byte_hash import hash_bytes, table_address
from klab.model import portable_engram
from klab.model.memory import TokenNgramMemory
from klab.model.portable_engram import (
    PortableEngramManifest,
    export_portable_engram,
    load_portable_engram,
)
from klab.training import manifest as manifest_module
from klab.training.manifest import architecture_sha256, trained_memory_addressing
from klab.training.trainer import train


def read_manifest_unverified(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


TABLE_SIZES = (257, 514, 251, 1021)
VOCAB = 512


def _legacy_token_address(key: tuple[int, ...], table_size: int, head: int) -> int:
    address = head + 1
    for token in key:
        address = (address * (257 + 2 * head) + token) % table_size
    return address


def _expected_distinct(draws: int, table_size: int) -> float:
    return table_size * (1 - (1 - 1 / table_size) ** draws)


def _ids(seed: int = 0, batch: int = 8, length: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, length), generator=generator)


@pytest.mark.parametrize("table_size", TABLE_SIZES)
@pytest.mark.parametrize("order", (2, 3))
@pytest.mark.parametrize("head", (0, 1))
def test_token_ngram_addresses_use_many_rows_and_whole_window(
    table_size: int, order: int, head: int
) -> None:
    memory = TokenNgramMemory(16, table_size, 3, 8, (2, 3), 2)
    ids = _ids()
    addresses = memory.addresses(ids, order, head)[:, order - 1 :]
    assert int(addresses.min()) >= 0 and int(addresses.max()) < table_size
    draws = addresses.numel()
    assert addresses.unique().numel() >= 0.8 * _expected_distinct(draws, table_size)

    # Not a single-token table: the address is not the oldest token mod size...
    oldest = ids[:, : ids.shape[1] - (order - 1)] % table_size
    assert (addresses == oldest).float().mean() < 0.05
    # ...and changing any single token in the window moves the address.
    for position in range(order - 1, ids.shape[1], 7):
        window = ids[0, position - order + 1 : position + 1].clone()
        for slot in range(order):
            altered = ids[:1].clone()
            moved_rows = 0
            for delta in range(1, 6):
                altered[0, position - order + 1 + slot] = (window[slot] + delta) % VOCAB
                row = memory.addresses(altered, order, head)[0, position]
                moved_rows += int(row != addresses[0, position - order + 1])
            assert moved_rows >= 4, (position, slot)


@pytest.mark.parametrize("table_size", (*TABLE_SIZES, 17, 127, 259, 8192))
def test_tensor_and_python_token_hashes_agree(table_size: int) -> None:
    memory = TokenNgramMemory(16, table_size, 3, 8, (2, 3), 3)
    ids = _ids(seed=1, batch=2, length=24)
    for order in (2, 3):
        for head in range(3):
            addresses = memory.addresses(ids, order, head)
            for batch in range(ids.shape[0]):
                for position in range(order - 1, ids.shape[1]):
                    key = tuple(int(ids[batch, position - k]) for k in range(order))
                    expected = token_ngram_address(key, table_size, head)
                    assert int(addresses[batch, position]) == expected


@pytest.mark.parametrize("table_size", (17, 127, 251, 1021, 8192, 65521))
def test_coprime_table_sizes_keep_legacy_token_addresses(table_size: int) -> None:
    rng = random.Random(table_size)
    for head in range(3):
        assert token_scheme(table_size, head) == TOKEN_SCHEME_V1
        for _ in range(200):
            key = tuple(rng.randrange(VOCAB) for _ in range(rng.choice((2, 3))))
            assert token_ngram_address(key, table_size, head) == (
                _legacy_token_address(key, table_size, head)
            )


def test_degenerate_sizes_switch_scheme_per_head() -> None:
    assert token_scheme(257, 0) == TOKEN_SCHEME_V2
    assert token_scheme(257, 1) == TOKEN_SCHEME_V1
    assert token_scheme(514, 0) == TOKEN_SCHEME_V2
    assert token_scheme(259, 1) == TOKEN_SCHEME_V2  # 259 = 257 + 2
    assert token_scheme(7, 1) == TOKEN_SCHEME_V2  # 7 divides 259


@pytest.mark.parametrize("table_size", TABLE_SIZES)
def test_byte_addresses_are_not_constant(table_size: int) -> None:
    rng = random.Random(table_size)
    values = [bytes(rng.randrange(256) for _ in range(3)) for _ in range(2000)]
    rows = [table_address(value, table_size) for value in values]
    assert all(0 <= row < table_size for row in rows)
    assert len(set(rows)) >= 0.8 * _expected_distinct(len(set(values)), table_size)
    # Every byte of the suffix matters, not only the first or last.
    for value, row in zip(values[:50], rows[:50], strict=True):
        for slot in range(3):
            altered = bytearray(value)
            moved = 0
            for delta in range(1, 6):
                altered[slot] = (value[slot] + delta) % 256
                moved += int(table_address(bytes(altered), table_size) != row)
            assert moved >= 4


@pytest.mark.parametrize("table_size", (127, 251, 1021, 8192, 65521))
def test_coprime_table_sizes_keep_legacy_byte_addresses(table_size: int) -> None:
    assert byte_scheme(table_size) == BYTE_SCHEME_V1
    rng = random.Random(table_size)
    for _ in range(200):
        value = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 33)))
        assert table_address(value, table_size) == hash_bytes(value) % table_size


def test_byte_scheme_marks_multiples_of_257() -> None:
    assert byte_scheme(257) == BYTE_SCHEME_V2
    assert byte_scheme(514) == BYTE_SCHEME_V2
    assert byte_scheme(256) == BYTE_SCHEME_V1


def test_portable_pack_label_is_explicit_and_must_match_the_table_size(
    tmp_path: Path,
) -> None:
    table = torch.zeros((257, 4), dtype=torch.float32)
    # The default label is the legacy scheme; it is never inferred from rows,
    # so a 257-row table cannot be packaged without an explicit v2 label...
    with pytest.raises(ValueError, match="cannot be exported"):
        export_portable_engram(table, tmp_path / "default.enbyte", ngram_size=3)
    with pytest.raises(ValueError, match="cannot be exported"):
        export_portable_engram(
            table, tmp_path / "old.enbyte", ngram_size=3, hashing=BYTE_SCHEME_V1
        )
    assert not list(tmp_path.iterdir())
    # ...and a v2 label is refused where current code addresses with v1.
    with pytest.raises(ValueError, match="cannot be exported"):
        export_portable_engram(
            torch.zeros((256, 4)), tmp_path / "x.enbyte", ngram_size=3,
            hashing=BYTE_SCHEME_V2,
        )  # fmt: skip

    manifest = export_portable_engram(
        table, tmp_path / "new.enbyte", ngram_size=3, hashing=BYTE_SCHEME_V2
    )
    assert manifest.hashing == BYTE_SCHEME_V2
    load_portable_engram(
        tmp_path / "new.enbyte", expected_shape=(257, 4), expected_ngram_size=3
    )
    legacy = torch.zeros((256, 4), dtype=torch.float32)
    assert (
        export_portable_engram(legacy, tmp_path / "legacy.enbyte", ngram_size=3).hashing
        == BYTE_SCHEME_V1
    )


def test_pre_fix_packs_at_affected_sizes_are_refused_by_the_consumer(
    tmp_path: Path,
) -> None:
    """A package minted by pre-fix code (v1 label, 257 rows) cannot be loaded."""
    package = tmp_path / "prefix.enbyte"
    table = torch.zeros((257, 4), dtype=torch.float32)
    manifest = PortableEngramManifest(
        1, "raw-utf8-v1", BYTE_SCHEME_V1, 3, 257, 4, portable_engram._digest(table)
    )
    torch.save({"manifest": manifest.as_dict(), "table": table}, package)
    with pytest.raises(ValueError, match="addressing algorithm is unsupported"):
        load_portable_engram(package, expected_shape=(257, 4), expected_ngram_size=3)


def _model(**overrides: object) -> dict[str, object]:
    return {
        "memory": "ngram", "memory_table_size": 257, "memory_hash_heads": 2,
        "memory_ngram_size": 3, **overrides,
    }  # fmt: skip


def test_architecture_identity_binds_changed_memory_addressing() -> None:
    affected = {"model": _model(), "attention": {}}
    assert architecture_sha256(affected) != architecture_sha256(
        affected, memory_addressing=False
    )
    assert trained_memory_addressing(affected, architecture_sha256(affected)) == {
        "token": [TOKEN_SCHEME_V2, TOKEN_SCHEME_V1]
    }
    legacy_digest = architecture_sha256(affected, memory_addressing=False)
    assert trained_memory_addressing(affected, legacy_digest) == {
        "token": [TOKEN_SCHEME_V1, TOKEN_SCHEME_V1]
    }
    with pytest.raises(ValueError, match="no known memory addressing"):
        trained_memory_addressing(affected, "0" * 64)
    byte = {"model": _model(memory="byte", memory_hash_heads=1), "attention": {}}
    assert trained_memory_addressing(byte, architecture_sha256(byte)) == {
        "byte": BYTE_SCHEME_V2
    }
    # Unaffected configs keep their pre-fix architecture digest exactly.
    for model in (
        _model(memory_table_size=1021),
        _model(memory="byte", memory_table_size=1021, memory_hash_heads=1),
        {"memory": "none", "memory_table_size": 0},
    ):
        unaffected = {"model": model, "attention": {}}
        assert architecture_sha256(unaffected) == architecture_sha256(
            unaffected, memory_addressing=False
        )


def _byte_run_config(root: Path, table_size: int) -> RunConfig:
    base = training_config(root)
    return base.model_copy(
        update={
            "model": base.model.model_copy(
                update={
                    "memory": "byte",
                    "memory_table_size": table_size,
                    "memory_ngram_size": 3,
                    "memory_dim": 4,
                }
            ),
            "training": base.training.model_copy(update={"max_steps": 2}),
            "optimizer": base.optimizer.model_copy(update={"warmup_steps": 0}),
        }
    )


def _export(
    monkeypatch: pytest.MonkeyPatch, runs: Path, run_id: str, output: Path
) -> None:
    argv = ["klab", "engram", "export", run_id, "--output", str(output)]
    monkeypatch.setattr(sys, "argv", [*argv, "--runs-dir", str(runs)])
    main()


def test_pre_fix_257_checkpoint_cannot_mint_a_v2_pack_by_re_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _byte_run_config(tmp_path, 257)
    runs = config.logging.root_dir
    # Train exactly as pre-fix code recorded it: no addressing in the identity.
    with monkeypatch.context() as patch:
        patch.setattr(manifest_module, "memory_addressing_identity", lambda _: None)
        train(config, run_id="prefix")
    recorded = read_manifest_unverified(runs / "prefix" / "manifest.json")
    assert trained_memory_addressing(
        recorded["effective_config"], recorded["architecture_sha256"]
    ) == {"byte": BYTE_SCHEME_V1}

    output = tmp_path / "prefix.enbyte"
    with pytest.raises(ValueError, match="pre-fix degenerate memory addressing"):
        _export(monkeypatch, runs, "prefix", output)
    assert not output.exists()

    # A table trained (and bound) under current code exports as v2 and loads.
    train(config, run_id="fixed")
    fixed = tmp_path / "fixed.enbyte"
    _export(monkeypatch, runs, "fixed", fixed)
    package = load_portable_engram(
        fixed, expected_shape=(257, 4), expected_ngram_size=3
    )
    assert package.manifest.hashing == BYTE_SCHEME_V2


def test_token_addressing_identity_names_changed_heads_only() -> None:
    assert token_addressing_marker(8192, 1) == {}
    assert token_addressing_marker(1021, 2) == {}
    assert token_addressing_marker(257, 2) == {
        "hashing": [TOKEN_SCHEME_V2, TOKEN_SCHEME_V1]
    }


# Fixed signed-int64 vectors pin both coefficient expansion and overflow semantics.
# The scalar oracle below is separate from the tensor primitive and NumPy demo.
_PRIME_XOR_MULTIPLIERS = (22818641653045131, 27993534731251561, 17422327423556949)
_PRIME_XOR_TABLES = (31, 37, 41, 43)
_PRIME_XOR_VECTORS = (
    ((0, 0, 0), (0, 0, 0, 0)),
    ((1, 0, 0), (2, 16, 6, 40)),
    ((259, 258, 257), (21, 33, 19, 37)),
    ((2**63 - 1, 1, 0), (2, 15, 21, 18)),
    ((-(2**63), -1, 2), (11, 28, 1, 29)),
)


def test_prime_xor_coefficients_are_frozen_bounded_and_rng_independent() -> None:
    assert ENGRAM_SCHEME_V1 == "token-ngram-prime-xor-splitmix64-v1"
    random.seed(123)
    before = random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    assert engram_multipliers(0, 3, 260) == _PRIME_XOR_MULTIPLIERS
    assert engram_multipliers(17, 3, 260, layer_id=1) == (
        23414436029034979,
        32322975159504861,
        9307579752262803,
    )
    assert engram_multipliers(0, 2, 260) == _PRIME_XOR_MULTIPLIERS[:2]
    assert random.getstate() == before
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    for vocab_size in (1, 2, 260, 32768, 2**31 - 1):
        for value in engram_multipliers(2**64 - 1, 8, vocab_size, layer_id=15):
            assert value % 2 == 1
            assert 1 <= value * vocab_size <= 2**63 - 1


@pytest.mark.parametrize("key,rows", _PRIME_XOR_VECTORS)
def test_prime_xor_frozen_vectors_match_scalar_tensor_and_numpy_demo(
    key: tuple[int, ...], rows: tuple[int, ...]
) -> None:
    import numpy as np

    # This is the demo's multiply/XOR/remainder operation with our explicitly
    # frozen constants, not a claim of matching NumPy's seed expansion.
    with np.errstate(over="ignore"):
        demo_mix = np.int64(key[0]) * np.int64(_PRIME_XOR_MULTIPLIERS[0])
        for token, multiplier in zip(key[1:], _PRIME_XOR_MULTIPLIERS[1:], strict=True):
            demo_mix = np.bitwise_xor(demo_mix, np.int64(token) * np.int64(multiplier))
    tensor_mix = torch.zeros((), dtype=torch.long)
    for token, multiplier in zip(key, _PRIME_XOR_MULTIPLIERS, strict=True):
        tensor_mix = engram_address_step(tensor_mix, torch.tensor(token), multiplier)
    for prime, row in zip(_PRIME_XOR_TABLES, rows, strict=True):
        assert engram_ngram_address(key, prime, _PRIME_XOR_MULTIPLIERS) == row
        assert int(engram_address_finish(tensor_mix, prime)) == row
        assert int(demo_mix % prime) == row


def test_prime_xor_suffix_padding_eos_and_whole_window_conformance() -> None:
    ids = torch.tensor([[4, 17, 2, 17, 4], [17, 4, 2, 4, 17]], dtype=torch.long)
    padding = 2  # Explicit canonical pad token, deliberately nonzero.
    coefficients = engram_multipliers(41, 3, 260)
    for order in (2, 3):
        tensor_mix = torch.zeros_like(ids)
        for offset in range(order):
            shifted = torch.full_like(ids, padding)
            shifted[:, offset:] = ids[:, : ids.shape[1] - offset]
            tensor_mix = engram_address_step(tensor_mix, shifted, coefficients[offset])
        for prime in engram_table_sizes(1021, 2):
            addresses = engram_address_finish(tensor_mix, prime)
            for batch in range(ids.shape[0]):
                for position in range(ids.shape[1]):
                    key = tuple(
                        int(ids[batch, position - offset])
                        if position >= offset
                        else padding
                        for offset in range(order)
                    )
                    assert addresses[batch, position] == engram_ngram_address(
                        key, prime, coefficients
                    )
            # Every suffix component can affect the lookup. Repeated keys keep
            # their exact row, including the EOS token (2) in the middle.
            key = (4, 17, 2)[:order]
            row = engram_ngram_address(key, prime, coefficients)
            for slot in range(order):
                changed = [*key]
                changes = 0
                for token in range(32):
                    changed[slot] = token
                    changes += engram_ngram_address(changed, prime, coefficients) != row
                assert changes >= 28


def test_prime_xor_distinct_prime_table_expansion_and_validation() -> None:
    assert engram_table_sizes(31, 4) == _PRIME_XOR_TABLES
    assert engram_table_sizes(32, 4) == (37, 41, 43, 47)
    assert engram_table_sizes(2, 4) == (2, 3, 5, 7)
    assert engram_table_sizes(59, 4) == (59, 61, 67, 71)
    assert validate_engram_table_sizes(_PRIME_XOR_TABLES, 4) == _PRIME_XOR_TABLES
    assert validate_engram_table_sizes((2**31 - 1,), 1) == (2**31 - 1,)
    with pytest.raises(ValueError, match="exceeds"):
        engram_table_sizes(2**31 - 1, 2)
    for invalid in ((31, 31), (31, 35), (True, 31), (1, 31), (2**31 + 1, 31)):
        with pytest.raises(ValueError, match="primes"):
            validate_engram_table_sizes(invalid, 2)
    with pytest.raises(ValueError, match="stream count"):
        validate_engram_table_sizes((31,), 2)
    # Carmichael / strong pseudoprime examples must not pass as table primes.
    for composite in (341, 561, 1105, 1729, 2047, 1373653, 25326001):
        with pytest.raises(ValueError, match="primes"):
            validate_engram_table_sizes((composite,), 1)


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((-1, 3, 260), {}),
        ((2**64, 3, 260), {}),
        ((True, 3, 260), {}),
        ((0, 0, 260), {}),
        ((0, 3, 0), {}),
        ((0, 3, 260), {"layer_id": -1}),
    ],
)
def test_prime_xor_invalid_coefficient_domains_are_rejected(args, kwargs) -> None:
    with pytest.raises(ValueError):
        engram_multipliers(*args, **kwargs)


def test_prime_xor_invalid_keys_and_coefficients_are_rejected() -> None:
    for key, coefficients in (
        ((), (1,)),
        ((1, 2), (1,)),
        ((True,), (1,)),
        ((2**63,), (1,)),
        ((1,), (2,)),
        ((1,), (0,)),
    ):
        with pytest.raises(ValueError):
            engram_ngram_address(key, 31, coefficients)
