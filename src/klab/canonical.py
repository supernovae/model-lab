"""Canonical JSON encoding and digests shared by every Lab identity."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: object) -> bytes:
    """Encode only JSON-native data in a canonical, finite representation."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: object) -> str:
    """Return the SHA-256 hex digest of ``canonical_json(value)``."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``json.loads`` ``object_pairs_hook`` that refuses repeated object keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
