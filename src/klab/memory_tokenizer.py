"""Versioned, tokenizer-bound vocabulary projections for n-gram addresses only.

``deepseek_v1`` follows the normalization/key construction in DeepSeek's
``engram_demo_v1.py`` (CompressedTokenizer), revision
``a31cd4d08cd293612d7803a2ec247af924a38fac``. ``deepseek_safe_v1`` uses the same
text normalization but isolates special tokens and undecodable byte fragments.
Neither lossy mode is code-preserving: case, accents and whitespace may merge.
The input/output tokenizer and labels must never be replaced by this projection.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from klab.canonical import canonical_digest, canonical_json, reject_duplicate_keys
from klab.training.manifest import sha256_file

MemoryTokenAlgorithm = Literal["identity", "deepseek_v1", "deepseek_safe_v1"]
ALGORITHMS = frozenset({"identity", "deepseek_v1", "deepseek_safe_v1"})
_FIELDS = frozenset(
    {
        "algorithm",
        "token_ids",
        "vocab_size",
        "memory_vocab_size",
        "tokenizer_sha256",
        "mapping_sha256",
    }
)


@dataclass(frozen=True)
class MemoryTokenMap:
    """A complete surjective vocabulary projection with an immutable identity."""

    algorithm: MemoryTokenAlgorithm
    token_ids: tuple[int, ...]
    vocab_size: int
    memory_vocab_size: int
    tokenizer_sha256: str
    mapping_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "token_ids": list(self.token_ids),
            "vocab_size": self.vocab_size,
            "memory_vocab_size": self.memory_vocab_size,
            "tokenizer_sha256": self.tokenizer_sha256,
            "mapping_sha256": self.mapping_sha256,
        }


def validate_memory_token_map(
    payload: Mapping[str, object],
    *,
    tokenizer_path: Path | None = None,
    expected_sha256: str | None = None,
) -> MemoryTokenMap:
    """Refuse incomplete, malformed or tampered maps before model construction.

    ``expected_sha256`` is the canonical mapping identity, not the file digest.
    Binding to tokenizer bytes is optional here so checkpoint/config validation
    remains filesystem-independent; data/generation entrypoints must supply it.
    """
    if not isinstance(payload, Mapping) or set(payload) != _FIELDS:
        raise ValueError("memory token map requires exactly the six identity fields")
    algorithm = payload["algorithm"]
    if not isinstance(algorithm, str) or algorithm not in ALGORITHMS:
        raise ValueError("unsupported memory token map algorithm")
    vocab_size = payload["vocab_size"]
    memory_vocab_size = payload["memory_vocab_size"]
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError("memory token map vocab_size must be a positive integer")
    if type(memory_vocab_size) is not int or not 1 <= memory_vocab_size <= vocab_size:
        raise ValueError("invalid memory token map memory_vocab_size")
    token_ids = payload["token_ids"]
    if not isinstance(token_ids, (list, tuple)) or len(token_ids) != vocab_size:
        raise ValueError("memory token map must cover every tokenizer ID")
    if any(
        type(item) is not int or not 0 <= item < memory_vocab_size for item in token_ids
    ):
        raise ValueError("memory token map IDs must be integers in the declared range")
    if len(set(token_ids)) != memory_vocab_size:
        raise ValueError("memory token map must be surjective")
    if algorithm == "identity" and any(i != item for i, item in enumerate(token_ids)):
        raise ValueError("identity memory token map must preserve all tokenizer IDs")
    for field in ("tokenizer_sha256", "mapping_sha256"):
        if not isinstance(payload[field], str) or not re.fullmatch(
            r"[0-9a-f]{64}", payload[field]
        ):
            raise ValueError(f"invalid memory token map {field}")
    identity = {key: payload[key] for key in _FIELDS if key != "mapping_sha256"}
    identity["token_ids"] = list(token_ids)
    if canonical_digest(identity) != payload["mapping_sha256"]:
        raise ValueError("memory token map digest mismatch")
    if expected_sha256 is not None and payload["mapping_sha256"] != expected_sha256:
        raise ValueError("memory token map expected digest mismatch")
    if (
        tokenizer_path is not None
        and sha256_file(tokenizer_path) != payload["tokenizer_sha256"]
    ):
        raise ValueError("memory token map tokenizer digest mismatch")
    result = MemoryTokenMap(
        algorithm=cast(MemoryTokenAlgorithm, algorithm),
        token_ids=tuple(token_ids),
        vocab_size=vocab_size,
        memory_vocab_size=memory_vocab_size,
        tokenizer_sha256=cast(str, payload["tokenizer_sha256"]),
        mapping_sha256=cast(str, payload["mapping_sha256"]),
    )
    if tokenizer_path is not None:
        expected = build_memory_token_map(tokenizer_path, result.algorithm)
        if result != expected:
            raise ValueError("memory token map does not match its declared algorithm")
    return result


def build_memory_token_map(
    tokenizer_path: Path, mode: MemoryTokenAlgorithm = "identity"
) -> MemoryTokenMap:
    """Build from a local tokenizer only, without fitting or reading any corpus.

    IDs are assigned in original token-ID order, never vocabulary iteration order.
    The exact resulting map and original tokenizer bytes are authenticated. The
    ``deepseek_v1`` key behavior includes the demo's replacement-character/raw
    token fallback and private-use sentinel handling; the safer variant avoids
    conflating special/undecodable IDs with ordinary textual keys.
    """
    from tokenizers import Regex, Tokenizer, normalizers

    if mode not in ALGORITHMS:
        raise ValueError("unsupported memory token map algorithm")
    # Parse exactly the bytes whose digest we retain. A concurrent replacement
    # is rejected rather than binding one tokenizer's map to another's digest.
    tokenizer_digest = sha256_file(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if sha256_file(tokenizer_path) != tokenizer_digest:
        raise ValueError("tokenizer changed while building memory token map")
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if vocab_size < 1 or any(
        tokenizer.id_to_token(i) is None for i in range(vocab_size)
    ):
        raise ValueError(
            "memory token mapping requires a nonempty contiguous tokenizer vocabulary"
        )
    if mode == "identity":
        mapping = list(range(vocab_size))
    else:
        sentinel = "\ue000"
        normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), sentinel),
                normalizers.Strip(),
                normalizers.Replace(sentinel, " "),
            ]
        )
        special_ids = {
            token_id
            for token_id, added in tokenizer.get_added_tokens_decoder().items()
            if added.special
        }
        keys: dict[object, int] = {}
        mapping = []
        for token_id in range(vocab_size):
            text = tokenizer.decode([token_id], skip_special_tokens=False)
            if mode == "deepseek_safe_v1" and token_id in special_ids:
                key: object = ("special", token_id)
            elif "\ufffd" in text:
                key = (
                    ("undecodable", token_id)
                    if mode == "deepseek_safe_v1"
                    else tokenizer.id_to_token(token_id)
                )
            else:
                normalized = normalizer.normalize_str(text)
                key = normalized if normalized else text
                if mode == "deepseek_safe_v1":
                    key = ("text", key)
            if key not in keys:
                keys[key] = len(keys)
            mapping.append(keys[key])
    identity: dict[str, object] = {
        "algorithm": mode,
        "token_ids": mapping,
        "vocab_size": vocab_size,
        "memory_vocab_size": len(set(mapping)),
        "tokenizer_sha256": tokenizer_digest,
    }
    return validate_memory_token_map(
        {**identity, "mapping_sha256": canonical_digest(identity)}
    )


def save_memory_token_map(mapping: MemoryTokenMap, path: Path) -> None:
    """Serialize a validated map as canonical JSON; never mutate the tokenizer."""
    payload = validate_memory_token_map(mapping.to_dict()).to_dict()
    path.write_bytes(canonical_json(payload) + b"\n")


def load_memory_token_map(
    path: Path,
    *,
    tokenizer_path: Path | None = None,
    expected_sha256: str | None = None,
) -> MemoryTokenMap:
    """Read and authenticate a map asset, rejecting repeated JSON keys."""
    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
    )
    return validate_memory_token_map(
        payload, tokenizer_path=tokenizer_path, expected_sha256=expected_sha256
    )
