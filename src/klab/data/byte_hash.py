"""Tokenizer-independent raw UTF-8 byte address primitives."""

from typing import TYPE_CHECKING

from klab.address_hash import byte_uses_legacy, mixed_byte_address

if TYPE_CHECKING:
    from tokenizers import Tokenizer


def _byte_to_unicode() -> dict[str, int]:
    """Return Tokenizers' reversible GPT-2/ByteLevel BPE alphabet."""
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    extra = [value for value in range(256) if value not in visible]
    return {
        chr(codepoint): value
        for value, codepoint in zip(
            visible + extra, visible + list(range(256, 256 + len(extra)))
        )
    }


_BYTE_LEVEL_BYTES = _byte_to_unicode()


def token_bytes(tokenizer: Tokenizer, token_id: int) -> bytes:
    """Return one ByteLevel BPE token's raw bytes without lossy UTF-8 decoding."""
    token = tokenizer.id_to_token(token_id)
    if token is None:
        raise ValueError(f"tokenizer has no token for ID {token_id}")
    try:
        return bytes(_BYTE_LEVEL_BYTES[character] for character in token)
    except KeyError as error:
        raise ValueError("raw byte addresses require a ByteLevel tokenizer") from error


BASE = 257


def hash_bytes(value: bytes) -> int:
    """Hash raw bytes with a terminal separator; no normalization is performed."""
    result = 0
    for byte in value:
        result = result * BASE + byte + 1
    return result * BASE


def hash_text(value: str) -> int:
    """Hash the exact UTF-8 representation of text."""
    return hash_bytes(value.encode("utf-8"))


def table_address(value: bytes, table_size: int) -> int:
    """Row for raw bytes; see ``klab.address_hash`` for the scheme.

    ``hash_bytes`` ends with ``* BASE``, so reducing it modulo a multiple of 257
    would send every input to row 0. Those table sizes use the mixed v2 scheme;
    every other size keeps the legacy ``poly257-terminal-v1`` address.
    """
    if type(table_size) is not int or table_size <= 0:
        raise ValueError("table_size must be positive")
    if not byte_uses_legacy(table_size):
        return mixed_byte_address(value, table_size)
    return hash_bytes(value) % table_size
