"""Physical placement of content-addressed transform outputs in the Corpus work area."""

from __future__ import annotations

import fcntl
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from klab.corpus.shards import range_file_identity
from klab.result_publication import sync_directory
from klab.safe_io import rename_noreplace, symlinked

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore, VerificationMode

SHARED_STAGE = "shared_stage"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_STAGE_FILE = re.compile(r"stages/[a-zA-Z0-9_-]+\.jsonl")


def shared_stage_path(root: Path, name: str, entry: dict[str, Any]) -> Path | None:
    """Resolve a shared member from its digest without taking paths from the manifest."""
    if "storage" not in entry:
        return None
    digest = entry.get("sha256")
    if (
        entry["storage"] != SHARED_STAGE
        or not isinstance(digest, str)
        or not _DIGEST.fullmatch(digest)
        or not _STAGE_FILE.fullmatch(name)
        or set(entry) != {"storage", "sha256", "size"}
        or type(entry["size"]) is not int
        or entry["size"] < 0
        or len(root.parents) < 4
        or root.parent.name != "releases"
        or root.parents[2].name != "corpora"
    ):
        raise ValueError(f"invalid shared stage member: {name}")
    path = (
        root.parents[3]
        / "artifacts"
        / "corpus"
        / "release-stages"
        / digest[:2]
        / digest
    )
    if symlinked(path):
        raise ValueError(f"symlinked shared stage member: {name}")
    return path


def shared_stage_members(
    root: Path, files: dict[str, dict[str, Any]]
) -> dict[str, Path]:
    """Return the shared part of a release's logical membership."""
    return {
        name: path
        for name, entry in files.items()
        if (path := shared_stage_path(root, name, entry)) is not None
    }


def publish_shared_stages(
    staging: Path,
    files: dict[str, dict[str, Any]],
    *,
    verified: bool,
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
) -> dict[Path, list[int]]:
    """Move verified files once, or consume staging copies after authenticating reuse."""
    generations = {}
    for name, destination in shared_stage_members(staging, files).items():
        source = staging / name
        entry = files[name]
        locks = destination.parents[1] / "locks"
        lock_path = locks / destination.name
        if symlinked(lock_path):
            raise ValueError("symlinked shared stage lock")
        locks.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if symlinked(destination) or symlinked(source):
                raise ValueError(f"symlinked shared stage input: {name}")
            existing = destination.exists()
            input_path = destination if existing else source
            if not input_path.is_file():
                raise ValueError(f"missing shared stage payload: {name}")
            if input_path.stat().st_nlink != 1:
                raise ValueError(f"hard-linked shared stage payload: {name}")
            stamp = range_file_identity(input_path)
            if stamp[2] != entry["size"]:
                raise ValueError(f"shared stage payload size differs: {name}")
            # A resumed release replays references whose source was already moved.
            if (existing and (verified or source.exists())) or (
                not existing and not verified
            ):
                from klab.data.verification import verify_file

                verify_file(
                    input_path,
                    expected_sha256=entry["sha256"],
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
            if range_file_identity(input_path) != stamp:
                raise ValueError(f"shared stage payload changed: {name}")
            if not existing:
                destination.parent.mkdir(parents=True, exist_ok=True)
                rename_noreplace(source, destination)
            if not existing or not verified:
                with destination.open("rb") as stream:
                    os.fsync(stream.fileno())
                for parent in destination.parents[:5]:
                    sync_directory(parent)
            if existing and source.exists():
                source.unlink()
            if source.parent.exists():
                sync_directory(source.parent)
            generations[destination] = range_file_identity(destination)
    return generations
