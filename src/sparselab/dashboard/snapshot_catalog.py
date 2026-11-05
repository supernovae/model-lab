"""Read-only generation catalog and serialized, single-model CPU ownership."""

from __future__ import annotations

import gc
import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

ARCHIVE_MAP = Path(
    "/tank/sparselab/projects/devmind/archive/checkpoint-copy-20261105-v1/original-to-archive.json"
)


@dataclass(frozen=True)
class SavedSnapshot:
    run: str
    checkpoint: str
    digest: str
    step: int
    created_at: str
    size_bytes: int

    @property
    def key(self):
        return (self.run, self.checkpoint, self.digest)

    @property
    def label(self):
        return f"{self.run} · {self.checkpoint} · {self.created_at}"


def catalog(root: Path) -> list[SavedSnapshot]:
    """Published verified rows are candidates; native cold verification is mandatory on load."""
    with sqlite3.connect(
        (root / "experiments.sqlite3").resolve().as_uri() + "?mode=ro",
        uri=True,
        timeout=1,
    ) as connection:
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute(
            "SELECT run_id, relative_path, digest, step, created_at, size_bytes FROM checkpoints WHERE verification_status='verified' ORDER BY created_at DESC, step DESC"
        ).fetchall()
    result = []
    for run, relative, digest, step, created, size in rows:
        name = relative.removeprefix("checkpoints/")
        if Path(run).name != run or run in (".", ".."):
            continue
        if not re.fullmatch(r"step_\d+_gen_\d+", name):
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            continue
        result.append(
            SavedSnapshot(run, "checkpoints/" + name, digest, step, created, size)
        )
    return result


def resolve_candidate(
    root: Path, selected: SavedSnapshot, archive_map: Path = ARCHIVE_MAP
) -> Path:
    """Consult relocation mapping before any generation read; never fall back to an old copy."""
    original = root / selected.run / selected.checkpoint
    if archive_map.is_file():
        mapping = json.loads(archive_map.read_text())
        for item in mapping["checkpoints"]:
            if item["source"] == str(original):
                raise ValueError(
                    "Archived checkpoint at "
                    + item["archive"]
                    + "; the native inference loader does not support relocated generations. No fallback copy was loaded."
                )
    if (
        original.resolve().parent != (root / selected.run / "checkpoints").resolve()
        or original.is_symlink()
    ):
        raise ValueError("Unsafe checkpoint location")
    metadata = json.loads((original / "manifest.json").read_text())
    if (
        metadata.get("sha256") != selected.digest
        or metadata.get("step") != selected.step
    ):
        raise ValueError("Published checkpoint identity differs from its manifest")
    if selected.size_bytes <= 0 or selected.size_bytes > 5 * 1024**3:
        raise ValueError("Checkpoint exceeds the 5 GiB local preview inventory limit")
    return original


class SnapshotOwner:
    """No caller receives the model. The lock covers verification, eviction, loading and use."""

    def __init__(self):
        self.lock = Lock()
        self.loaded = None
        self.key = None
        self.status = ("idle", "No model loaded")

    def execute(self, selected: SavedSnapshot, loader: Callable, action: Callable):
        if not self.lock.acquire(blocking=False):
            raise ValueError(
                "Sampler busy loading or generating for another request. Try again after it finishes."
            )
        try:
            if self.key != selected.key:
                self.status = ("loading", selected.label)
                self.loaded = None
                self.key = None
                gc.collect()
                try:
                    loaded = loader(selected)
                    if (
                        loaded.identity["run_id"] != selected.run
                        or loaded.identity["checkpoint_sha256"] != selected.digest
                    ):
                        raise ValueError(
                            "Loaded identity differs from selected checkpoint"
                        )
                    self.loaded = loaded
                    self.key = selected.key
                except BaseException:
                    self.status = ("failed", selected.label)
                    raise
            self.status = ("generating", selected.label)
            try:
                return action(self.loaded)
            finally:
                self.status = ("ready", selected.label)
        finally:
            self.lock.release()
