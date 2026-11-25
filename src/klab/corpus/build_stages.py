"""Content-addressed, receipted stage outputs for the disk corpus builder.

A stage output is a directory `builds/stages/<stage>/<key>/` holding its files
and a `receipt.json`. The key is the digest of exactly the inputs the stage
depends on, so a different input is a different directory rather than a stale
hit. A stage is reused only when its receipt verifies: the recorded key is the
digest of the recorded inputs, the directory holds exactly the receipted files,
and every size and SHA-256 matches. Names and modification times are never
trusted. An output that does not verify is rebuilt, and an incomplete staging
directory is never an input: outputs appear only by one atomic rename.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from klab.canonical import canonical_digest, canonical_json
from klab.safe_io import rename_noreplace
from klab.training.manifest import sha256_file

RECEIPT = "receipt.json"
_INCOMPLETE = ".incomplete-"


def implementation_digests(*names: str) -> dict[str, str]:
    """SHA-256 of sibling source files of this package, for a stage's key."""
    directory = Path(__file__).parent
    return {name: sha256_file(directory / name) for name in sorted(names)}


class StageStore:
    """The stage outputs of one build workspace."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def directory(self, stage: str, key: str) -> Path:
        return self.root / stage / key

    @staticmethod
    def key(inputs: dict[str, Any]) -> str:
        return canonical_digest(inputs)

    def verified(
        self,
        stage: str,
        inputs: dict[str, Any],
        *,
        hash_chunks: Mapping[str, Callable[[bytes], None]] | None = None,
    ) -> dict[str, Any] | None:
        """The receipt of a reusable output for exactly these inputs, else None."""
        key = self.key(inputs)
        path = self.directory(stage, key)
        if not path.is_dir():
            return None
        try:
            return verify_stage(path, stage, key, inputs, hash_chunks=hash_chunks)
        except OSError, ValueError, KeyError:
            return None

    @contextmanager
    def build(
        self, stage: str, inputs: dict[str, Any], *, retained: Path | None = None
    ):
        """Yield a private staging directory; publish it atomically on success.

        The caller writes the stage files into the directory and assigns the
        stage's result values to the dictionary it also receives. Leaving the
        block with an exception removes transient staging. A retained directory
        stays private for a producer that validates its own checkpoints.
        """
        key = self.key(inputs)
        parent = self.root / stage
        parent.mkdir(parents=True, exist_ok=True)
        if retained is not None:
            if retained.is_symlink():
                raise ValueError(f"symlinked stage checkpoint: {retained}")
            retained.mkdir(parents=True, exist_ok=True)
        temporary_directory = (
            nullcontext(retained)
            if retained is not None
            else tempfile.TemporaryDirectory(prefix=_INCOMPLETE, dir=parent)
        )
        with temporary_directory as temporary:
            staging = Path(temporary)
            if retained is not None:
                (staging / RECEIPT).unlink(missing_ok=True)
            result: dict[str, Any] = {}
            yield staging, result
            files = {
                item.name: {"size": item.stat().st_size, "sha256": sha256_file(item)}
                for item in sorted(staging.iterdir())
            }
            if RECEIPT in files or any(
                not item.is_file() for item in staging.iterdir()
            ):
                raise ValueError(f"stage {stage} wrote an unexpected entry")
            receipt = {
                "schema_version": 1,
                "stage": stage,
                "key": key,
                "inputs": inputs,
                "files": files,
                "result": result,
            }
            with (staging / RECEIPT).open("wb") as handle:
                handle.write(canonical_json(receipt) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            _sync_stage_files(staging)
            self._publish(staging, stage, key, inputs)

    def _publish(
        self, staging: Path, stage: str, key: str, inputs: dict[str, Any]
    ) -> None:
        destination = self.directory(stage, key)
        if destination.exists():
            try:
                verify_stage(destination, stage, key, inputs)
                return
            except OSError, ValueError, KeyError:
                shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rename_noreplace(staging, destination)
        _sync(destination.parent)
        if staging.parent != destination.parent:
            _sync(staging.parent)

    def discard_incomplete(self, stage: str) -> None:
        """Clear abandoned staging directories while the caller owns the writer lock."""
        for directory in (self.root / stage).glob(f"{_INCOMPLETE}*"):
            shutil.rmtree(directory, ignore_errors=True)

    def prune(self, stage: str, *, keep: int, protect: str | None = None) -> list[str]:
        """Remove incomplete directories and all but the newest `keep` outputs.

        Age orders deletions only; it never makes an output trusted. The output
        named by `protect` is always kept.
        """
        parent = self.root / stage
        if not parent.is_dir():
            return []
        self.discard_incomplete(stage)
        removed = []
        outputs = []
        for item in parent.iterdir():
            if (
                item.is_dir()
                and not item.name.startswith(_INCOMPLETE)
                and item.name != protect
            ):
                outputs.append(item)
        outputs.sort(key=lambda item: item.stat().st_mtime_ns, reverse=True)
        for item in outputs[max(keep - (protect is not None), 0) :]:
            shutil.rmtree(item)
            removed.append(item.name)
        return removed


def _sync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_stage_files(staging: Path) -> None:
    for item in staging.iterdir():
        with item.open("rb") as handle:
            os.fsync(handle.fileno())
    _sync(staging)


def verify_stage(
    path: Path,
    stage: str,
    key: str,
    inputs: dict[str, Any],
    *,
    hash_chunks: Mapping[str, Callable[[bytes], None]] | None = None,
) -> dict[str, Any]:
    """Verify the stage, optionally observing bytes as selected files are hashed.

    Observed bytes are provisional until the full receipt verifies.
    """
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"stage {stage} output is not a directory")
    receipt_path = path / RECEIPT
    if receipt_path.is_symlink():
        raise ValueError(f"stage {stage} receipt is symlinked")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt["stage"] != stage
        or receipt["key"] != key
        or receipt["inputs"] != json.loads(canonical_json(inputs))
        or canonical_digest(receipt["inputs"]) != key
    ):
        raise ValueError(f"stage {stage} receipt does not match its inputs")
    if {item.name for item in path.iterdir()} != {RECEIPT, *receipt["files"]}:
        raise ValueError(f"stage {stage} output has missing or extra files")
    for name, expected in receipt["files"].items():
        item = path / name
        if (
            item.is_symlink()
            or not item.is_file()
            or item.stat().st_size != expected["size"]
        ):
            raise ValueError(f"stage {stage} output changed: {name}")
        observe = hash_chunks.get(name) if hash_chunks is not None else None
        digest = sha256_file(item, hash_chunk=observe) if observe else sha256_file(item)
        if digest != expected["sha256"]:
            raise ValueError(f"stage {stage} output changed: {name}")
    return receipt


def run_stage(
    store: StageStore,
    stage: str,
    inputs: dict[str, Any],
    produce: Callable[[Path, dict[str, Any]], None],
    *,
    on_reuse: Callable[[], None] | None = None,
    retained: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Reuse a verified output for `inputs`, or produce and publish one."""
    receipt = store.verified(stage, inputs)
    if receipt is not None:
        if on_reuse is not None:
            on_reuse()
        return store.directory(stage, receipt["key"]), receipt["result"]
    if retained is not None and (retained / RECEIPT).is_file():
        key = store.key(inputs)
        try:
            receipt = verify_stage(retained, stage, key, inputs)
        except OSError, ValueError, KeyError:
            pass
        else:
            _sync_stage_files(retained)
            store._publish(retained, stage, key, inputs)
            if on_reuse is not None:
                on_reuse()
            return store.directory(stage, key), receipt["result"]
    with store.build(stage, inputs, retained=retained) as (staging, result):
        produce(staging, result)
    path = store.directory(stage, store.key(inputs))
    return path, json.loads((path / RECEIPT).read_text(encoding="utf-8"))["result"]
