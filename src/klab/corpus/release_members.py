"""Physical placement of content-addressed release payloads in the Corpus work area."""

from __future__ import annotations

import fcntl
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from klab.corpus.shards import range_file_identity
from klab.result_publication import sync_directory
from klab.safe_io import rename_noreplace, symlinked

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore, VerificationMode

SHARED_STAGE = "shared_stage"
SHARED_DOCUMENT = "shared_document"
DOCUMENT_MEMBERS = ("documents.jsonl", "spans.jsonl")
SHARED_STORES = {
    SHARED_STAGE: "release-stages",
    SHARED_DOCUMENT: "release-documents",
}
_DIGEST = re.compile(r"[0-9a-f]{64}")
_STAGE_FILE = re.compile(r"stages/[a-zA-Z0-9_-]+\.jsonl")


@dataclass(frozen=True)
class ReleasePlacement:
    """Opt-in storage choices for ordinary release publication."""

    stages: bool = False
    documents: bool = False

    @property
    def shared(self) -> bool:
        return self.stages or self.documents

    @classmethod
    def from_inventory(cls, files: dict[str, dict[str, Any]]) -> ReleasePlacement:
        stores = {entry.get("storage") for entry in files.values()}
        return cls(stages=SHARED_STAGE in stores, documents=SHARED_DOCUMENT in stores)


def shared_store(work_root: Path, storage: str) -> Path:
    """Return the retained store for a known manifest storage option."""
    return work_root / "artifacts" / "corpus" / SHARED_STORES[storage]


def shared_member_path(root: Path, name: str, entry: dict[str, Any]) -> Path | None:
    """Resolve a shared member from its digest without taking paths from the manifest."""
    if "storage" not in entry:
        return None
    storage = entry["storage"]
    digest = entry.get("sha256")
    if (
        not (
            (storage == SHARED_STAGE and _STAGE_FILE.fullmatch(name))
            or (storage == SHARED_DOCUMENT and name in DOCUMENT_MEMBERS)
        )
        or not isinstance(digest, str)
        or not _DIGEST.fullmatch(digest)
        or set(entry) != {"storage", "sha256", "size"}
        or type(entry["size"]) is not int
        or entry["size"] < 0
        or len(root.parents) < 4
        or root.parent.name != "releases"
        or root.parents[2].name != "corpora"
    ):
        raise ValueError(f"invalid shared release member: {name}")
    path = shared_store(root.parents[3], storage) / digest[:2] / digest
    if symlinked(path):
        raise ValueError(f"symlinked shared release member: {name}")
    return path


def shared_members(root: Path, files: dict[str, dict[str, Any]]) -> dict[str, Path]:
    """Return the shared part of a release's logical membership."""
    return {
        name: path
        for name, entry in files.items()
        if (path := shared_member_path(root, name, entry)) is not None
    }


@dataclass(frozen=True)
class SharedPublication:
    """Shared generations and the subset authenticated during this publication."""

    generations: dict[Path, list[int]]
    checked: dict[Path, list[int]]


def publish_shared_members(
    staging: Path,
    files: dict[str, dict[str, Any]],
    *,
    verified: bool,
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
) -> SharedPublication:
    """Move verified files once, or consume staging copies after authenticating reuse."""
    generations = {}
    checked = {}
    for name, destination in shared_members(staging, files).items():
        source = staging / name
        entry = files[name]
        locks = destination.parents[1] / "locks"
        lock_path = locks / destination.name
        if symlinked(lock_path):
            raise ValueError("symlinked shared release lock")
        locks.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if symlinked(destination) or symlinked(source):
                raise ValueError(f"symlinked shared release input: {name}")
            existing = destination.exists()
            input_path = destination if existing else source
            if not input_path.is_file():
                raise ValueError(f"missing shared release payload: {name}")
            if input_path.stat().st_nlink != 1:
                raise ValueError(f"hard-linked shared release payload: {name}")
            stamp = range_file_identity(input_path)
            if stamp[2] != entry["size"]:
                raise ValueError(f"shared release payload size differs: {name}")
            # A resumed release replays references whose source was already moved.
            authenticate = (existing and (verified or source.exists())) or (
                not existing and not verified
            )
            if authenticate:
                from klab.data.verification import verify_file

                verify_file(
                    input_path,
                    expected_sha256=entry["sha256"],
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
            if range_file_identity(input_path) != stamp:
                raise ValueError(f"shared release payload changed: {name}")
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
            generation = range_file_identity(destination)
            # Rename may change ctime. The inode, size and mtime still bind
            # the authenticated input to the new directory entry.
            if (existing and generation != stamp) or (
                not existing and generation[:-1] != stamp[:-1]
            ):
                raise ValueError(f"shared release payload changed while moving: {name}")
            generations[destination] = generation
            if authenticate or verified:
                checked[destination] = generation
    return SharedPublication(generations, checked)
