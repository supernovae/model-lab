"""Shared memory-table address hashing for token n-gram and raw-byte tables.

Two schemes exist per table size:

* **v1 (legacy)** reduces a base-257 polynomial modulo ``table_size`` at every
  step. When ``gcd(multiplier, table_size) == 1`` the step is a bijection on the
  previous address, so v1 is a real hash and every address it produces is kept
  bit-for-bit: existing checkpoints, prepared byte addresses and portable packs
  at those sizes are unaffected.
* **v2 (mixed)** is used only when the multiplier shares a factor with
  ``table_size`` (for example ``table_size == 257`` or ``514``). There v1
  degenerates: with ``gcd == table_size`` every step discards the previous
  address, so an "n-gram" address is just the oldest token modulo the table
  size, and the byte hash's trailing ``* 257`` sends every input to row 0. v2
  mixes in the prime field ``2**31 - 1`` with a large odd multiplier and reduces
  modulo ``table_size`` only once at the end, so no table size collapses it.

Legacy-scheme arithmetic is plain ``*``, ``+`` and ``%`` on non-negative values below
``2**62``, so the same helpers work on Python ints and int64 torch tensors.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from math import gcd
from typing import Final

LEGACY_BASE: Final = 257
MIX_MODULUS: Final = 2**31 - 1
MIX_MULTIPLIER: Final = 1_103_515_245

TOKEN_SCHEME_V1: Final = "token-ngram-recurrence-v1"
TOKEN_SCHEME_V2: Final = "token-ngram-recurrence-v2"
BYTE_SCHEME_V1: Final = "poly257-terminal-v1"
BYTE_SCHEME_V2: Final = "mix31-terminal-v2"
ENGRAM_SCHEME_V1: Final = "token-ngram-prime-xor-splitmix64-v1"
_INT64_MAX: Final = 2**63 - 1
_UINT64_MASK: Final = 2**64 - 1
_MAX_PRIME: Final = 2**31 - 1


def _positive(table_size: int) -> int:
    if type(table_size) is not int or table_size <= 0:
        raise ValueError("table_size must be positive")
    return table_size


def token_multiplier(head: int) -> int:
    """The legacy per-hash-head multiplier ``257 + 2 * head``."""
    if type(head) is not int or head < 0:
        raise ValueError("hash head must be a non-negative integer")
    return LEGACY_BASE + 2 * head


def token_uses_legacy(table_size: int, head: int) -> bool:
    """True when v1 is a proper hash for this table size and hash head."""
    return gcd(token_multiplier(head), _positive(table_size)) == 1


def token_scheme(table_size: int, head: int) -> str:
    return TOKEN_SCHEME_V1 if token_uses_legacy(table_size, head) else TOKEN_SCHEME_V2


def token_address_init(head: int) -> int:
    token_multiplier(head)
    return head + 1


def token_address_step[Value](
    address: Value, token: Value, table_size: int, head: int
) -> Value:
    """Fold one token (newest first) into a running address."""
    if token_uses_legacy(table_size, head):
        return (address * token_multiplier(head) + token) % table_size  # type: ignore[operator]
    multiplier = MIX_MULTIPLIER + 2 * head
    return (address * multiplier + token + 1) % MIX_MODULUS  # type: ignore[operator]


def token_address_finish[Value](address: Value, table_size: int, head: int) -> Value:
    """Reduce a running address to a table row."""
    if token_uses_legacy(table_size, head):
        return address
    return address % table_size  # type: ignore[operator]


def token_ngram_address(key: Iterable[int], table_size: int, head: int) -> int:
    """Address of one zero-padded token n-gram key, newest token first."""
    address = token_address_init(head)
    for token in key:
        address = token_address_step(address, token, table_size, head)
    return token_address_finish(address, table_size, head)


def byte_uses_legacy(table_size: int) -> bool:
    return gcd(LEGACY_BASE, _positive(table_size)) == 1


def byte_scheme(table_size: int) -> str:
    """The raw-byte hashing identifier recorded for a table of this size."""
    return BYTE_SCHEME_V1 if byte_uses_legacy(table_size) else BYTE_SCHEME_V2


def mixed_byte_address(value: bytes, table_size: int) -> int:
    """v2 raw-byte address with a terminal separator step."""
    _positive(table_size)
    address = 1
    for byte in value:
        address = (address * MIX_MULTIPLIER + byte + 1) % MIX_MODULUS
    address = (address * MIX_MULTIPLIER) % MIX_MODULUS
    return address % table_size


def _model_int(model: Mapping[str, object], key: str, default: int) -> int:
    value = model.get(key, default)
    return value if type(value) is int else default


def memory_schemes(model: Mapping[str, object], *, legacy: bool = False) -> dict:
    """Address schemes a model config's memory tables use under current code.

    ``legacy=True`` gives what pre-fix code used for the same config (v1
    everywhere). Byte and portable memories read raw-byte addresses; n-gram
    memory hashes one table per (order, hash head), so it lists every head.
    """
    kind = model.get("memory", "none")
    table_size = _model_int(model, "memory_table_size", 0)
    if kind == "none" or table_size <= 0:
        return {}
    if kind in {"byte", "portable"}:
        return {"byte": BYTE_SCHEME_V1 if legacy else byte_scheme(table_size)}
    if (
        kind == "ngram"
        and model.get("memory_addressing") == "prime_xor_v1"
        and not legacy
    ):
        return {"token": [ENGRAM_SCHEME_V1] * len(model["memory_table_sizes"])}
    if kind == "ngram":
        heads = max(_model_int(model, "memory_hash_heads", 1), 1)
        return {
            "token": [
                TOKEN_SCHEME_V1 if legacy else token_scheme(table_size, head)
                for head in range(heads)
            ]
        }
    return {}


def memory_addressing_identity(model: Mapping[str, object]) -> dict | None:
    """The marker bound into architecture identity, or None when unchanged.

    Only configs whose addresses differ from pre-fix code get a marker, so
    every unaffected run keeps its architecture digest. A checkpoint trained
    by pre-fix code at an affected size carries the digest without it and is
    therefore recognisably legacy rather than silently reinterpreted.
    """
    if model.get("memory_addressing") == "prime_xor_v1":
        orders = model.get("memory_ngram_orders") or (model["memory_ngram_size"],)
        mapping = model.get("memory_token_mapping")
        vocabulary = model["vocab_size"]
        sites = model.get("memory_block_indices") or (
            model.get("memory_block_index") or 0,
        )
        streams = len(orders) * model.get("memory_hash_heads", 1)
        return {
            "token": ENGRAM_SCHEME_V1,
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
    current = memory_schemes(model)
    return current if current != memory_schemes(model, legacy=True) else None


def token_addressing_marker(table_size: int, hash_heads: int) -> dict[str, list[str]]:
    """Extra addressing-identity field for token tables whose scheme changed.

    Empty for unaffected sizes, so every existing token addressing identity is
    unchanged; affected sizes must name the v2 heads they were built with.
    """
    schemes = [token_scheme(table_size, head) for head in range(max(hash_heads, 1))]
    if all(scheme == TOKEN_SCHEME_V1 for scheme in schemes):
        return {}
    return {"hashing": schemes}


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
