"""Memory-table address hashing.

* **Token n-gram memory** (Engram): each n-gram is XOR-mixed from per-position odd
  multipliers expanded from a SplitMix64 seed and reduced modulo a distinct
  prime row count per (order, hash head) stream.
* **Raw-byte memory** hashes the bytes as a base-257 polynomial with a terminal
  separator step, reduced modulo the table size. A table size that shares a
  factor with 257 sends every input to row 0 and is refused.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from math import gcd
from typing import Final

BYTE_BASE: Final = 257
BYTE_HASHING: Final = "poly257-terminal"
ENGRAM_SCHEME: Final = "token-ngram-prime-xor-splitmix64"
_INT64_MAX: Final = 2**63 - 1
_UINT64_MASK: Final = 2**64 - 1
_MAX_PRIME: Final = 2**31 - 1


def validate_byte_table_size(table_size: int) -> int:
    """Return a raw-byte table size whose hash reaches every row."""
    if type(table_size) is not int or table_size <= 0:
        raise ValueError("table_size must be positive")
    if gcd(BYTE_BASE, table_size) != 1:
        raise ValueError(
            f"raw-byte table size {table_size} shares a factor with {BYTE_BASE}"
        )
    return table_size


def engram_identity(model: Mapping[str, object]) -> dict | None:
    """The token-memory addressing bound into architecture identity.

    None when the model has no n-gram memory.
    """
    if model.get("memory") != "ngram":
        return None
    orders = model.get("memory_ngram_orders") or (model["memory_ngram_size"],)
    mapping = model.get("memory_token_mapping")
    vocabulary = model["vocab_size"]
    sites = model.get("memory_block_indices") or (model.get("memory_block_index") or 0,)
    streams = len(orders) * model.get("memory_hash_heads", 1)
    return {
        "token": ENGRAM_SCHEME,
        "ports": [
            {
                "block_index": site,
                "table_sizes": list(model["memory_table_sizes"])[
                    index * streams : (index + 1) * streams
                ],
                "multipliers": list(
                    engram_multipliers(
                        model.get("memory_hash_seed", 0),
                        max(orders),
                        vocabulary,
                        layer_id=site,
                    )
                ),
            }
            for index, site in enumerate(sites)
        ],
        "padding_token_id": model.get("memory_padding_token_id", 0),
        "mapping_sha256": mapping["mapping_sha256"] if mapping else None,
        "multiplier_vocab_size": vocabulary,
    }


@lru_cache(maxsize=4096)
def _is_prime(value: int) -> bool:
    """Deterministic Miller–Rabin over the supported positive int32 range."""
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11):
        if value % prime == 0:
            return value == prime
    odd = value - 1
    shifts = 0
    while odd % 2 == 0:
        shifts += 1
        odd //= 2
    # Bases 2, 7, 61 are deterministic below 4,759,123,141.
    for base in (2, 7, 61):
        if base % value == 0:
            continue
        witness = pow(base, odd, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = witness * witness % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def validate_engram_table_sizes(sizes: Iterable[int], streams: int) -> tuple[int, ...]:
    """Validate one distinct prime row count per ordered (order, head) stream."""
    if type(streams) is not int or streams <= 0:
        raise ValueError("Engram stream count must be positive")
    result = tuple(sizes)
    if len(result) != streams:
        raise ValueError("Engram table sizes must match the stream count")
    if any(
        type(size) is not int or not 2 <= size <= _MAX_PRIME or not _is_prime(size)
        for size in result
    ):
        raise ValueError("Engram table sizes must be primes in [2, 2**31 - 1]")
    if len(set(result)) != streams:
        raise ValueError("Engram table sizes must be distinct primes")
    return result


def engram_table_sizes(minimum: int, streams: int) -> tuple[int, ...]:
    """Return distinct ascending primes >= minimum, in stream order.

    Explicit sizes remain part of the architecture declaration; this helper is
    only the deterministic convenience expansion, not an implicit address change.
    """
    if type(minimum) is not int or not 2 <= minimum <= _MAX_PRIME:
        raise ValueError("Engram minimum table size must be in [2, 2**31 - 1]")
    if type(streams) is not int or not 1 <= streams <= 4096:
        raise ValueError("Engram stream count must be in [1, 4096]")
    result: list[int] = []
    candidate = minimum
    while len(result) < streams:
        if candidate > _MAX_PRIME:
            raise ValueError("Engram prime table expansion exceeds 2**31 - 1")
        if _is_prime(candidate):
            result.append(candidate)
        candidate += 1 if candidate <= 2 else 2 if candidate % 2 else 1
    return tuple(result)


def engram_multipliers(
    seed: int, max_order: int, vocab_size: int, *, layer_id: int = 0
) -> tuple[int, ...]:
    """Expand fixed SplitMix64 seed bits into bounded odd position multipliers.

    Engram's public demo shares multipliers across heads/orders at one layer and
    bounds them by INT64_MAX / canonical vocabulary size. This uses that same
    construction but pins SplitMix64 + rejection sampling instead of the demo's
    NumPy default_rng expansion. The algorithm name is therefore explicit and
    does not claim identical coefficients for the same demo seed. No global RNG
    is consulted. layer_id is the explicitly declared attachment-site number.
    """
    if type(seed) is not int or not 0 <= seed <= _UINT64_MASK:
        raise ValueError("Engram hash seed must be an unsigned 64-bit integer")
    if type(max_order) is not int or not 1 <= max_order <= 4096:
        raise ValueError("Engram maximum order must be in [1, 4096]")
    if type(vocab_size) is not int or not 1 <= vocab_size <= _MAX_PRIME:
        raise ValueError("Engram vocabulary size must be in [1, 2**31 - 1]")
    if type(layer_id) is not int or not 0 <= layer_id <= _MAX_PRIME:
        raise ValueError("Engram layer ID must be in [0, 2**31 - 1]")
    half_bound = max(1, (_INT64_MAX // vocab_size) // 2)
    # Reject the high incomplete interval to avoid modulo-biased coefficients.
    limit = 2**64 - (2**64 % half_bound)
    state = (seed + 10007 * layer_id) & _UINT64_MASK
    result: list[int] = []
    while len(result) < max_order:
        state = (state + 0x9E3779B97F4A7C15) & _UINT64_MASK
        mixed = state
        mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
        mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
        mixed ^= mixed >> 31
        if mixed < limit:
            result.append(2 * (mixed % half_bound) + 1)
    return tuple(result)


def engram_address_step[Value](address: Value, token: Value, multiplier: int) -> Value:
    """XOR a newest-first token product using signed int64 two's complement.

    Int64 tensors naturally wrap multiplication; Python ints need the explicit
    conversion below. For canonical IDs in the declared vocabulary and generated
    multipliers, products stay nonnegative and never overflow. The wrap rule is
    still fixed so boundary tests and independent implementations agree.
    """
    product = token * multiplier  # type: ignore[operator]
    result = address ^ product  # type: ignore[operator]
    if isinstance(result, int):
        return ((result + 2**63) & _UINT64_MASK) - 2**63  # type: ignore[return-value]
    return result


def engram_address_finish[Value](address: Value, table_size: int) -> Value:
    """Reduce a signed int64 mix into [0, prime), using mathematical remainder."""
    return address % table_size  # type: ignore[operator]


def engram_ngram_address(
    key: Iterable[int], table_size: int, multipliers: Iterable[int]
) -> int:
    """Scalar prime/XOR address; key is complete and newest-first, padding included.

    Padding is supplied by the caller after memory-side canonicalization. There
    is no historical token recurrence, final mixing step or head-dependent salt.
    Heads differ by their declared prime modulus, as in the Engram demo.
    """
    validate_engram_table_sizes((table_size,), 1)
    values = tuple(key)
    coefficients = tuple(multipliers)
    if not values or len(coefficients) < len(values):
        raise ValueError("Engram key needs one multiplier per token")
    if any(
        type(token) is not int or not -(2**63) <= token <= _INT64_MAX
        for token in values
    ):
        raise ValueError("Engram key tokens must be signed int64 integers")
    if any(
        type(value) is not int or not 1 <= value <= _INT64_MAX or value % 2 != 1
        for value in coefficients
    ):
        raise ValueError("Engram multipliers must be positive odd int64 integers")
    address = 0
    for token, multiplier in zip(values, coefficients, strict=False):
        address = engram_address_step(address, token, multiplier)
    return engram_address_finish(address, table_size)
