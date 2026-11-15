"""Memory-table addressing: raw-byte hashing and the token n-gram prime/XOR scheme."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from klab.address_hash import (
    BYTE_HASHING,
    ENGRAM_SCHEME,
    engram_address_finish,
    engram_address_step,
    engram_multipliers,
    engram_ngram_address,
    engram_table_sizes,
    validate_byte_table_size,
    validate_engram_table_sizes,
)
from klab.data.byte_hash import hash_bytes, table_address
from klab.model import portable_engram
from klab.model.portable_engram import (
    PortableEngramManifest,
    export_portable_engram,
    load_portable_engram,
)


def _expected_distinct(draws: int, table_size: int) -> float:
    return table_size * (1 - (1 - 1 / table_size) ** draws)


@pytest.mark.parametrize("table_size", (127, 1021, 8192))
def test_byte_addresses_spread_over_rows_and_use_every_byte(table_size: int) -> None:
    rng = random.Random(table_size)
    values = [bytes(rng.randrange(256) for _ in range(3)) for _ in range(2000)]
    rows = [table_address(value, table_size) for value in values]
    assert rows == [hash_bytes(value) % table_size for value in values]
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


def test_byte_table_sizes_sharing_a_factor_with_257_are_refused(
    tmp_path: Path,
) -> None:
    """Such a table would send every byte address to row 0."""
    for size in (257, 514):
        with pytest.raises(ValueError, match="shares a factor"):
            validate_byte_table_size(size)
        with pytest.raises(ValueError, match="shares a factor"):
            table_address(b"abc", size)
        with pytest.raises(ValueError, match="shares a factor"):
            export_portable_engram(
                torch.zeros((size, 4)), tmp_path / "x.enbyte", ngram_size=3
            )
    assert validate_byte_table_size(256) == 256
    assert not list(tmp_path.iterdir())


def test_portable_pack_names_its_hashing_and_refuses_another(tmp_path: Path) -> None:
    table = torch.zeros((256, 4), dtype=torch.float32)
    package = tmp_path / "ok.enbyte"
    assert export_portable_engram(table, package, ngram_size=3).hashing == BYTE_HASHING
    load_portable_engram(package, expected_shape=(256, 4), expected_ngram_size=3)
    other = tmp_path / "other.enbyte"
    manifest = PortableEngramManifest(
        1, "raw-utf8-v1", "other-hash", 3, 256, 4, portable_engram._digest(table)
    )
    torch.save({"manifest": manifest.as_dict(), "table": table}, other)
    with pytest.raises(ValueError, match="addressing algorithm is unsupported"):
        load_portable_engram(other, expected_shape=(256, 4), expected_ngram_size=3)


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
    assert ENGRAM_SCHEME == "token-ngram-prime-xor-splitmix64"
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


def test_prime_xor_frozen_vectors_match_scalar_tensor_and_numpy_demo() -> None:
    import numpy as np

    for key, rows in _PRIME_XOR_VECTORS:
        # This is the demo's multiply/XOR/remainder operation with our explicitly
        # frozen constants, not a claim of matching NumPy's seed expansion.
        with np.errstate(over="ignore"):
            demo_mix = np.int64(key[0]) * np.int64(_PRIME_XOR_MULTIPLIERS[0])
            for token, multiplier in zip(
                key[1:], _PRIME_XOR_MULTIPLIERS[1:], strict=True
            ):
                demo_mix = np.bitwise_xor(
                    demo_mix, np.int64(token) * np.int64(multiplier)
                )
        tensor_mix = torch.zeros((), dtype=torch.long)
        for token, multiplier in zip(key, _PRIME_XOR_MULTIPLIERS, strict=True):
            tensor_mix = engram_address_step(
                tensor_mix, torch.tensor(token), multiplier
            )
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


def test_prime_xor_invalid_coefficient_domains_are_rejected() -> None:
    for args, kwargs in (
        ((-1, 3, 260), {}),
        ((2**64, 3, 260), {}),
        ((True, 3, 260), {}),
        ((0, 0, 260), {}),
        ((0, 3, 0), {}),
        ((0, 3, 260), {"layer_id": -1}),
    ):
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


def test_omitted_ngram_table_sizes_expand_to_distinct_primes_per_stream_and_site() -> (
    None
):
    from klab.config.models import ModelConfig

    base = {
        "vocab_size": 260, "hidden_dim": 16, "num_layers": 3, "num_heads": 2,
        "ffn_dim": 32, "max_seq_len": 16, "memory": "ngram", "memory_table_size": 32,
        "memory_ngram_size": 3, "memory_dim": 4, "memory_ngram_orders": (2, 3),
        "memory_hash_heads": 2, "memory_injection": "after_block",
        "memory_block_indices": (0, 2),
    }  # fmt: skip
    expanded = ModelConfig(**base)
    assert expanded.memory_table_sizes == engram_table_sizes(32, 8)
    assert expanded.memory_table_size == expanded.memory_table_sizes[0] == 37
    explicit = ModelConfig(
        **base
        | {"memory_table_size": 37, "memory_table_sizes": engram_table_sizes(37, 8)}
    )
    assert explicit == expanded
    with pytest.raises(ValueError, match="table sizes, seed"):
        ModelConfig(**base | {"memory": "none", "memory_table_size": 0,
                              "memory_ngram_size": 0, "memory_dim": 0,
                              "memory_ngram_orders": (), "memory_hash_heads": 1,
                              "memory_injection": "final", "memory_block_indices": (),
                              "memory_hash_seed": 3})  # fmt: skip
    with pytest.raises(ValueError, match="shares a factor"):
        ModelConfig(**base | {"memory": "byte", "memory_table_size": 514,
                              "memory_ngram_orders": (), "memory_hash_heads": 1,
                              "memory_injection": "final", "memory_block_indices": ()})  # fmt: skip
