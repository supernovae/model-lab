"""Tokenizer-independent raw UTF-8 byte address primitives."""

from typing import TYPE_CHECKING

from klab.address_hash import BYTE_BASE, validate_byte_table_size

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


def hash_bytes(value: bytes) -> int:
    """Hash raw bytes with a terminal separator; no normalization is performed."""
    result = 0
    for byte in value:
        result = result * BYTE_BASE + byte + 1
    return result * BYTE_BASE


def hash_text(value: str) -> int:
    """Hash the exact UTF-8 representation of text."""
    return hash_bytes(value.encode("utf-8"))


def table_address(value: bytes, table_size: int) -> int:
    """Row for raw bytes; see ``klab.address_hash`` for the scheme."""
    return hash_bytes(value) % validate_byte_table_size(table_size)
