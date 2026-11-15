"""A bounded next-token table built from prepared training arrays.

The table maps a hashed context of ``order`` tokens to the continuation that
followed it most often in the training split, with its count. It is a drafter's
guess source only: the target model verifies every drafted token, so a stale,
colliding or wrong entry costs speed and never changes the output.

The table is stored beside the run it was built from as a directory with its own
manifest, so it has its own identity and is verified on load like other prepared
artifacts: signed manifest, exact file inventory, file digests, array shapes and
token ranges, and the tokenizer it was built with.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from klab.canonical import canonical_digest
from klab.training.manifest import sha256_file
from klab.verification_proofs import writable_by_others

TABLE_FORMAT = "corpus-ngram-table-v1"
HASH_SCHEME = "token-window-multiply-add-murmur64-v1"
_MANIFEST = "manifest.json"
_ARRAYS = {"tokens.npy": np.dtype("int32"), "counts.npy": np.dtype("uint32")}
_MULTIPLIER = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1
_CHUNK_TOKENS = 1 << 22
_MAX_ORDER = 16
_MAX_TABLE_SIZE = 1 << 28


def _positive(name: str, value: object, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= maximum
    ):
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return value


def _slot(context: Sequence[int], table_size: int) -> int:
    """Hash one context; ``_slots`` computes the same value for many at once."""
    value = 0
    for token in context:
        value = (value * _MULTIPLIER + token + 1) & _MASK
    value ^= value >> 33
    value = (value * 0xFF51AFD7ED558CCD) & _MASK
    value ^= value >> 33
    return value % table_size


def _slots(windows: np.ndarray, table_size: int) -> np.ndarray:
    """Hash each row of an (n, order) int64 array of contexts."""
    with np.errstate(over="ignore"):
        value = np.zeros(windows.shape[0], dtype=np.uint64)
        for column in range(windows.shape[1]):
            value = value * np.uint64(_MULTIPLIER) + (
                windows[:, column].astype(np.uint64) + np.uint64(1)
            )
        value ^= value >> np.uint64(33)
        value = value * np.uint64(0xFF51AFD7ED558CCD)
        value ^= value >> np.uint64(33)
        return (value % np.uint64(table_size)).astype(np.int64)


def count_continuations(
    stream: np.ndarray,
    *,
    order: int,
    table_size: int,
    vocab_size: int,
    eos_id: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the most frequent continuation of each hashed context.

    ``stream`` is the flat token sequence. A context that contains ``eos_id``
    spans two documents and is skipped; ``eos_id`` itself is a valid
    continuation. Ties go to the smaller token id. Returns ``tokens`` (-1 for an
    empty slot) and ``counts``.
    """
    keys = np.empty(0, dtype=np.int64)
    totals = np.empty(0, dtype=np.int64)
    for start in range(0, max(len(stream) - order, 0), _CHUNK_TOKENS):
        segment = np.asarray(stream[start : start + _CHUNK_TOKENS + order]).astype(
            np.int64
        )
        rows = len(segment) - order
        if rows <= 0:
            continue
        windows = np.stack(
            [segment[column : column + rows] for column in range(order)], 1
        )
        following = segment[order : order + rows]
        valid = (following >= 0) & (following < vocab_size)
        if eos_id is not None:
            valid &= ~(windows == eos_id).any(axis=1)
        pairs = _slots(windows[valid], table_size) * vocab_size + following[valid]
        found, found_counts = np.unique(pairs, return_counts=True)
        merged, inverse = np.unique(np.concatenate([keys, found]), return_inverse=True)
        totals = np.bincount(
            inverse, weights=np.concatenate([totals, found_counts]).astype(np.float64)
        ).astype(np.int64)
        keys = merged
    slots, tokens = keys // vocab_size, keys % vocab_size
    best = np.lexsort((tokens, -totals, slots))
    first = np.unique(slots[best], return_index=True)[1]
    chosen = best[first]
    table_tokens = np.full(table_size, -1, dtype=np.int32)
    table_counts = np.zeros(table_size, dtype=np.uint32)
    table_tokens[slots[chosen]] = tokens[chosen]
    table_counts[slots[chosen]] = np.minimum(totals[chosen], np.iinfo(np.uint32).max)
    return table_tokens, table_counts


def _write_array(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.lib.format.write_array(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def build_ngram_table(
    stream: np.ndarray,
    output: Path,
    *,
    order: int,
    table_size: int,
    vocab_size: int,
    eos_id: int | None,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Write a table directory at ``output`` (which must not exist); return its manifest.

    ``source`` records what the arrays came from (digests of the prepared
    manifest, the split and the tokenizer); it is part of the table's identity.
    ``source["tokenizer_sha256"]`` is required because a table is only meaningful
    for the tokenizer whose token ids it holds.
    """
    order = _positive("order", order, maximum=_MAX_ORDER)
    table_size = _positive("table_size", table_size, maximum=_MAX_TABLE_SIZE)
    vocab_size = _positive("vocab_size", vocab_size, maximum=1 << 31)
    if not isinstance(source.get("tokenizer_sha256"), str):
        raise TypeError("source must name tokenizer_sha256")
    if table_size * vocab_size >= 1 << 62:
        raise ValueError("table_size times vocab_size is too large")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"ngram table already exists: {output}")
    tokens, counts = count_continuations(
        stream,
        order=order,
        table_size=table_size,
        vocab_size=vocab_size,
        eos_id=eos_id,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_array(staged / "tokens.npy", tokens)
        _write_array(staged / "counts.npy", counts)
        body: dict[str, Any] = {
            "format": TABLE_FORMAT,
            "hash_scheme": HASH_SCHEME,
            "order": order,
            "table_size": table_size,
            "vocab_size": vocab_size,
            "eos_id": eos_id,
            "source": source,
            "stream_tokens": len(stream),
            "filled_slots": int((tokens >= 0).sum()),
            "files": {
                name: {
                    "sha256": sha256_file(staged / name),
                    "size_bytes": (staged / name).stat().st_size,
                }
                for name in _ARRAYS
            },
        }
        manifest = {**body, "manifest_sha256": canonical_digest(body)}
        (staged / _MANIFEST).write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staged.chmod(0o755)
        staged.rename(output)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    return manifest


@dataclass(frozen=True)
class NgramTable:
    """A verified table: ``tokens[slot]`` continues the context hashing to ``slot``."""

    root: Path
    manifest: dict[str, Any]
    tokens: np.ndarray
    counts: np.ndarray

    @property
    def order(self) -> int:
        return int(self.manifest["order"])

    @property
    def eos_id(self) -> int | None:
        return self.manifest["eos_id"]

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "manifest_sha256": self.manifest["manifest_sha256"],
            "order": self.order,
            "table_size": self.manifest["table_size"],
            "source": self.manifest["source"],
        }

    def lookup(self, context: Sequence[int]) -> tuple[int, int] | None:
        """Return the stored (token, count) for the last ``order`` tokens, if any."""
        if len(context) < self.order:
            return None
        slot = _slot(context[-self.order :], int(self.manifest["table_size"]))
        token = int(self.tokens[slot])
        return None if token < 0 else (token, int(self.counts[slot]))


def load_ngram_table(root: Path, *, tokenizer_sha256: str) -> NgramTable:
    """Verify a table directory against its manifest and ``tokenizer_sha256``."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"ngram table is not a directory: {root}")
    if writable_by_others(root.lstat()):
        raise ValueError(f"ngram table directory is writable by others: {root}")
    manifest_path = root / _MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("ngram table has no manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != TABLE_FORMAT:
        raise ValueError(f"not a {TABLE_FORMAT} manifest; build the table again")
    body = dict(manifest)
    digest = body.pop("manifest_sha256", None)
    if canonical_digest(body) != digest:
        raise ValueError("ngram table manifest signature mismatch")
    if manifest.get("hash_scheme") != HASH_SCHEME:
        raise ValueError("ngram table uses a different hash scheme; build it again")
    if manifest["source"].get("tokenizer_sha256") != tokenizer_sha256:
        raise ValueError("the ngram table was built with a different tokenizer")
    if {entry.name for entry in root.iterdir()} != {_MANIFEST, *_ARRAYS}:
        raise ValueError("unexpected ngram table file inventory")
    arrays: dict[str, np.ndarray] = {}
    for name, dtype in _ARRAYS.items():
        path = root / name
        info = path.lstat()
        recorded = manifest["files"][name]
        if not stat.S_ISREG(info.st_mode) or info.st_size != recorded["size_bytes"]:
            raise ValueError(f"ngram table file changed: {name}")
        if sha256_file(path) != recorded["sha256"]:
            raise ValueError(f"ngram table file digest mismatch: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != dtype or array.shape != (manifest["table_size"],):
            raise ValueError(f"ngram table array has the wrong shape or type: {name}")
        arrays[name] = array
    tokens = arrays["tokens.npy"]
    if int(tokens.min()) < -1 or int(tokens.max()) >= manifest["vocab_size"]:
        raise ValueError("ngram table holds token ids outside its vocabulary")
    return NgramTable(root, manifest, tokens, arrays["counts.npy"])


class CorpusNgramSpeculation:
    """Draft the corpus's most frequent continuation of the last ``order`` tokens.

    Each drafted token extends the context for the next lookup. Drafting stops
    at an empty slot, a count below ``min_count`` or after an end-of-sequence
    token. Counters accumulate like ``PromptLookupSpeculation``'s.
    """

    def __init__(
        self, table: NgramTable, *, max_draft_tokens: int = 8, min_count: int = 1
    ) -> None:
        self.table = table
        self.max_draft_tokens = _positive(
            "max_draft_tokens", max_draft_tokens, maximum=1 << 16
        )
        self.min_count = _positive("min_count", min_count, maximum=1 << 31)
        self.model_calls = 0
        self.drafted_tokens = 0
        self.accepted_tokens = 0

    def begin(self) -> None:
        """Nothing is remembered between requests."""

    def draft(self, ids: Sequence[int], limit: int) -> list[int]:
        context = list(ids[-self.table.order :])
        proposed: list[int] = []
        for _ in range(min(limit, self.max_draft_tokens)):
            found = self.table.lookup(context)
            if found is None or found[1] < self.min_count:
                break
            proposed.append(found[0])
            if found[0] == self.table.eos_id:
                break
            context = [*context[1:], found[0]]
        return proposed

    @property
    def acceptance_rate(self) -> float:
        return (
            self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0
        )
