"""Held-out leakage: probe items that must be held out, found in training data.

Selecting ideas on probe items that a run trained on inflates every result
silently, so ``klab try`` screens each arm's prepared training tokens (the
``train.npy`` the trainer read, verified by :func:`load_prepared_data`) for the
held-out and final probe items (:func:`klab.probes.suite.leakage_screen`).

Matching is the usual n-gram contamination screen in the run's own tokens: an
item is tokenized with the run's tokenizer (as written, and after a space, as
it would appear inside a document) and is contaminated when any window of
``n = min(13, item tokens)`` consecutive tokens that includes an answer token
occurs verbatim in the training tokens. Items without an answer (prompts)
count any window.

The scan hashes every 8-token window of the training array (fewer for very
short items) by doubling, keeps the positions whose hash falls in a bit table
of the items' leading windows, and verifies every candidate token by token, so
a hash collision never reports contamination. Results are cached under the lab
work root keyed by the prepared data, the tokenizer, the probe suite and ``n``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from klab.data.withheld_facts import split_facts
from klab.probes.suite import Window, leakage_screen, suite_identity
from klab.training.manifest import canonical_digest
from klab.verification_proofs import writable_by_others

LEAKAGE_FORMAT = "klab-leakage-v1"
WINDOW = 13
CLEAN, CONTAMINATED, ERROR = "CLEAN", "CONTAMINATED", "ERROR"
_PREFILTER = 8
_TABLE_BITS = 24
_SHIFT = np.uint64(64 - _TABLE_BITS)
_CHUNK = 1 << 24
_VERIFY_BATCH = 1 << 16
_MIX = np.uint64(0x9E3779B97F4A7C15)
_BASE = 0xC2B2AE3D27D4EB4F
_MASK = (1 << 64) - 1


def _hash_windows(tokens: np.ndarray, width: int) -> np.ndarray:
    """Hash every WIDTH-token window (WIDTH a power of two) by doubling."""
    hashed = (tokens.astype(np.uint64) + np.uint64(1)) * _MIX
    step, power = 1, _BASE
    while step < width:
        hashed = hashed[:-step] * np.uint64(power) + hashed[step:]
        power = power * power & _MASK
        step *= 2
    return hashed


def find_windows(train: np.ndarray, windows: set[Window]) -> set[Window]:
    """The WINDOWS that occur verbatim (exact token sequence) in TRAIN."""
    total = len(train)
    wanted = [window for window in windows if 0 < len(window) <= total]
    if not wanted:
        return set()
    width = 1 << (min(min(map(len, wanted)), _PREFILTER).bit_length() - 1)
    groups: dict[int, list[Window]] = {}
    for window in wanted:
        lead = _hash_windows(np.asarray(window[:width], dtype=np.int64), width)
        groups.setdefault(int(lead[0]), []).append(window)
    table = np.zeros(1 << _TABLE_BITS, dtype=bool)
    table[np.fromiter(groups, dtype=np.uint64, count=len(groups)) >> _SHIFT] = True
    found: set[Window] = set()
    for start in range(0, total - width + 1, _CHUNK):
        chunk = np.asarray(train[start : min(start + _CHUNK + width - 1, total)])
        hashed = _hash_windows(chunk, width)
        candidates = np.flatnonzero(table[hashed >> _SHIFT])
        if candidates.size == 0:
            continue
        keys = hashed[candidates]
        order = np.argsort(keys, kind="stable")
        keys, positions = keys[order], candidates[order] + start
        for key, group in groups.items():
            low = np.searchsorted(keys, np.uint64(key), side="left")
            high = np.searchsorted(keys, np.uint64(key), side="right")
            for window in group:
                if low == high or window in found:
                    continue
                at = positions[low:high]
                at = at[at + len(window) <= total]
                if _verify(train, at, window):
                    found.add(window)
    return found


def _verify(train: np.ndarray, positions: np.ndarray, window: Window) -> bool:
    """Whether WINDOW starts at any of POSITIONS, compared token by token."""
    expected = np.asarray(window)
    offsets = np.arange(len(window))
    for begin in range(0, positions.size, _VERIFY_BATCH):
        at = positions[begin : begin + _VERIFY_BATCH]
        if (train[at[:, None] + offsets] == expected).all(axis=1).any():
            return True
    return False


def item_windows(tokenizer: Any, text: str, answer: str | None, n: int) -> list[Window]:
    """The token windows that would leak one item (see the module docstring)."""
    whole = text if answer is None else f"{text} {answer}"
    windows: list[Window] = []
    for lead in ("", " "):
        encoding = tokenizer.encode(lead + whole, add_special_tokens=False)
        ids = list(encoding.ids)
        if not ids:
            continue
        width = min(n, len(ids))
        first = 0
        if answer is not None:
            boundary = len(lead) + len(text)
            first = next(
                (i for i, (_, end) in enumerate(encoding.offsets) if end > boundary),
                len(ids) - 1,
            )
        for begin in range(max(0, first - width + 1), len(ids) - width + 1):
            windows.append(tuple(ids[begin : begin + width]))
    return windows


def _parametric_seed(manifest: dict[str, Any]) -> int | None:
    """The withheld-facts seed the prepared data was built with, if any."""
    dataset = (manifest.get("cache_identity") or {}).get("dataset") or {}
    if (dataset.get("source"), dataset.get("synthetic_task")) != (
        "synthetic",
        "withheld_facts",
    ):
        return None
    seed = dataset.get("synthetic_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("withheld_facts prepared data records no synthetic_seed")
    return seed


def _read_cache(path: Path, key: str) -> dict[str, Any] | None:
    """A cached result, or None when absent, foreign, writable by others or edited."""
    try:
        if path.is_symlink() or writable_by_others(path.stat()):
            return None
        stored = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if not isinstance(stored, dict) or stored.get("key") != key:
        return None
    result = stored.get("result")
    if stored.get("sha256") != canonical_digest({"key": key, "result": result}):
        return None
    return result if isinstance(result, dict) else None


def _write_cache(path: Path, key: str, result: dict[str, Any]) -> dict[str, Any]:
    """Publish RESULT atomically; say why when the cache cannot be written."""
    from klab.lab_records import write_json_atomic

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            path,
            {
                "key": key,
                "result": result,
                "sha256": canonical_digest({"key": key, "result": result}),
            },
        )
    except OSError as error:
        reason = f"{type(error).__name__}: {error}"
        print(f"leakage cache not written ({path}): {reason}", file=sys.stderr)
        return {"status": "not_written", "path": str(path), "reason": reason}
    return {"status": "written", "path": str(path)}


def check_prepared(
    prepared_root: Path,
    tokenizer_path: Path,
    *,
    cache_dir: Path | None,
    n: int = WINDOW,
) -> dict[str, Any]:
    """Screen one prepared training split for held-out and final probe items.

    PREPARED_ROOT is a prepared bundle (a run's ``data/`` or a cache entry);
    it is verified before it is read, and TOKENIZER_PATH must be the
    tokenizer it was prepared with. Returns ``status`` (``CLEAN`` or
    ``CONTAMINATED``), ``checked``/``contaminated`` counts per split, the
    contaminated held-out item ids (never anything about a final item),
    the cache outcome and the seconds spent.
    """
    from klab.data.packing import canonical_tokenizer_sha256, load_prepared_data
    from klab.data.tokenizer import load_tokenizer

    started = time.monotonic()
    root = Path(prepared_root)
    header = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    data = load_prepared_data(
        root, byte_enabled=header.get("byte_addressing") is not None
    )
    tokenizer = load_tokenizer(Path(tokenizer_path))
    tokenizer_sha256 = canonical_tokenizer_sha256(tokenizer)
    if data.manifest.get("tokenizer_sha256") != tokenizer_sha256:
        raise ValueError("tokenizer differs from the one the data was prepared with")
    seed = _parametric_seed(data.manifest)
    identity = {
        "format": LEAKAGE_FORMAT,
        "prepared_manifest_sha256": data.receipt.manifest_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "suite_sha256": suite_identity()["sha256"],
        "n": n,
    }
    key = canonical_digest(identity)
    path = None if cache_dir is None else Path(cache_dir) / f"{key}.json"
    cached = None if path is None else _read_cache(path, key)
    if cached is not None:
        return {
            **cached,
            "cache": {"status": "hit", "path": str(path)},
            "seconds": round(time.monotonic() - started, 3),
        }
    scan_started = time.monotonic()
    screen = leakage_screen(
        lambda text, answer: item_windows(tokenizer, text, answer, n),
        lambda windows: find_windows(data.train, windows),
        parametric_seed=seed,
    )
    contaminated = {split: row["contaminated"] for split, row in screen.items()}
    result: dict[str, Any] = {
        **identity,
        "key": key,
        "status": CONTAMINATED if any(contaminated.values()) else CLEAN,
        "train_tokens": len(data.train),
        "checked": {split: row["checked"] for split, row in screen.items()},
        "contaminated": contaminated,
        "items": screen["heldout"]["items"],
        # Trained facts of a withheld_facts run are in its data by design.
        "exempt": {
            "heldout_trained_facts": 0 if seed is None else len(split_facts(seed)[0])
        },
        "scan_seconds": round(time.monotonic() - scan_started, 3),
    }
    cache = {"status": "disabled"} if path is None else _write_cache(path, key, result)
    return {
        **result,
        "cache": cache,
        "seconds": round(time.monotonic() - started, 3),
    }


def leakage_cache_dir(lab_root: Path) -> Path:
    """Where leakage results are cached: beside other operational lab caches."""
    return Path(lab_root) / "cache" / "leakage-v1"


__all__ = [
    "CLEAN",
    "CONTAMINATED",
    "ERROR",
    "LEAKAGE_FORMAT",
    "WINDOW",
    "check_prepared",
    "find_windows",
    "item_windows",
    "leakage_cache_dir",
]
