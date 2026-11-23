"""Assign test files to CI shards from recorded durations.

The fast suite is split by file across CI jobs. ``shard_durations.json`` holds the
seconds each file took in a full run; files in it are placed longest first onto the
currently lightest shard, so the shards finish together. A file missing from the table
(a new one) goes by a hash of its path, so no job keeps a list of files and a new test
file still lands in a shard on its own.

The assignment depends only on the table and the shard count, never on what a job
collected, so every shard job computes the same one and each file runs exactly once.
Refresh the table with ``KLAB_RECORD_DURATIONS=tests/shard_durations.json pytest``
(see docs/test-speed.md).
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

DURATIONS_FILE = Path(__file__).with_name("shard_durations.json")


def load_durations(path: Path = DURATIONS_FILE) -> dict[str, float]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {str(module): float(seconds) for module, seconds in raw.items()}


def assign(durations: dict[str, float], count: int) -> dict[str, int]:
    """Map each recorded file to a 1-based shard, longest first onto the lightest."""
    loads = [0.0] * count
    placed: dict[str, int] = {}
    for module, seconds in sorted(durations.items(), key=lambda kv: (-kv[1], kv[0])):
        lightest = min(range(count), key=lambda index: (loads[index], index))
        loads[lightest] += seconds
        placed[module] = lightest + 1
    return placed


def shard_of(module: str, count: int, placed: dict[str, int]) -> int:
    return placed.get(module) or zlib.crc32(module.encode()) % count + 1
