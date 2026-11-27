"""Physical placement of content-addressed release payloads in the Corpus work area."""

from __future__ import annotations

import fcntl
import os
import re
import stat
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from klab.corpus.normalized_spans import NORMALIZED_SPANS, SPAN_MEMBER
from klab.corpus.release_verify import INVENTORY_RECEIPT_MIN_BYTES
from klab.corpus.shards import range_file_identity
from klab.corpus.typed_artifacts import typed_publisher_work_root
from klab.result_publication import sync_directory
from klab.safe_io import rename_noreplace, symlinked

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore, VerificationMode

SHARED_STAGE = "shared_stage"
SHARED_DOCUMENT = "shared_document"
SHARED_LM_VIEW = "shared_lm_view"
DOCUMENT_MEMBERS = ("documents.jsonl", "spans.jsonl")
LM_VIEW_MEMBERS = ("lm/train.jsonl", "lm/validation.jsonl", "lm/test.jsonl")
SHARED_STORES = {
    SHARED_STAGE: "release-stages",
    SHARED_DOCUMENT: "release-documents",
    SHARED_LM_VIEW: "release-lm-views",
}
_DIGEST = re.compile(r"[0-9a-f]{64}")
_STAGE_FILE = re.compile(r"stages/[a-zA-Z0-9_-]+\.jsonl")


@dataclass(frozen=True)
class ReleasePlacement:
    """Opt-in storage choices for ordinary release publication."""

    stages: bool = False
    documents: bool = False
    normalized_spans: bool = False
    lm_views: bool = False

    @property
    def shared(self) -> bool:
        return self.stages or self.documents or self.normalized_spans or self.lm_views

    @classmethod
    def from_inventory(cls, files: dict[str, dict[str, Any]]) -> ReleasePlacement:
        stores = {entry.get("storage") for entry in files.values()}
        return cls(
            stages=SHARED_STAGE in stores,
            documents=SHARED_DOCUMENT in stores,
            normalized_spans=NORMALIZED_SPANS in stores,
            lm_views=SHARED_LM_VIEW in stores,
        )


def shared_store(work_root: Path, storage: str) -> Path:
    """Return the retained store for a known manifest storage option."""
    return work_root / "artifacts" / "corpus" / SHARED_STORES[storage]


def shared_member_path(root: Path, name: str, entry: dict[str, Any]) -> Path | None:
    """Resolve a shared member from its digest without taking paths from the manifest."""
    if "storage" not in entry:
        return None
    storage = entry["storage"]
    if storage == NORMALIZED_SPANS and name == SPAN_MEMBER:
        return None
    digest = entry.get("sha256")
    if (
        not (
            (storage == SHARED_STAGE and _STAGE_FILE.fullmatch(name))
            or (storage == SHARED_DOCUMENT and name in DOCUMENT_MEMBERS)
            or (storage == SHARED_LM_VIEW and name in LM_VIEW_MEMBERS)
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


@contextmanager
def _release_lock(path: Path, operation: int) -> Iterator[None]:
    if symlinked(path):
        raise ValueError(f"symlinked shared release lock: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as lock:
        info = os.fstat(lock.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"unsafe shared release lock: {path}")
        fcntl.flock(lock, operation)
        yield


def release_retention_lease(
    work_root: Path, *, prune: bool = False
) -> AbstractContextManager[None]:
    """Serialize an explicit prune with the shared manifest and payload commit."""
    path = work_root / "artifacts" / "corpus" / ".release-retention.lock"
    return _release_lock(
        path, fcntl.LOCK_EX | fcntl.LOCK_NB if prune else fcntl.LOCK_SH
    )


def normalized_retention_lease(
    work_root: Path, *, prune: bool = False
) -> AbstractContextManager[None]:
    """Keep normalized inputs available through native build and placement."""
    path = work_root / "artifacts" / "corpus" / ".normalized-retention.lock"
    return _release_lock(
        path, fcntl.LOCK_EX | fcntl.LOCK_NB if prune else fcntl.LOCK_SH
    )


def normalized_partition_lease(partition: Path) -> AbstractContextManager[None]:
    """Take the normalizer's existing per-key writer lock without waiting."""
    return _release_lock(
        partition.parent / ".locks" / partition.name, fcntl.LOCK_EX | fcntl.LOCK_NB
    )


@contextmanager
def shared_payload_lease(payload: Path, *, wait: bool = True) -> Iterator[None]:
    """Lease a digest and protect dependencies for a typed publication."""
    path = payload.parents[1] / "locks" / payload.name
    root = typed_publisher_work_root(payload)
    with (
        release_retention_lease(root) if root is not None else nullcontext(),
        _release_lock(path, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB),
    ):
        yield


@dataclass(frozen=True)
class SharedPublication:
    """Shared generations and the subset authenticated during this publication."""

    generations: dict[Path, list[int]]
    checked: dict[Path, list[int]]


def shared_payload_receipt_hit(
    path: Path,
    entry: dict[str, Any],
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
) -> bool:
    """Inspect the signed generation of an already retained large payload."""
    if (
        proof_store is None
        or verification_mode != "verified_reuse"
        or entry["size"] < INVENTORY_RECEIPT_MIN_BYTES
    ):
        return False
    from klab.corpus import release_receipt

    return release_receipt.lookup(
        proof_store,
        "shared_payload",
        path,
        entry["sha256"],
        [],
        skip_index_on_missing=True,
    ).hit


def publish_shared_members(
    staging: Path,
    files: dict[str, dict[str, Any]],
    *,
    verified: bool,
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
    verified_sources: Mapping[str, list[int]] | None = None,
) -> SharedPublication:
    """Move verified files once, or consume staging copies after authenticating reuse."""
    from klab.corpus import release_receipt

    generations = {}
    checked = {}
    for name, destination in shared_members(staging, files).items():
        source = staging / name
        entry = files[name]
        with shared_payload_lease(destination):
            if symlinked(destination) or symlinked(source):
                raise ValueError(f"symlinked shared release input: {name}")
            if verified_sources is not None and (
                not source.is_file()
                or range_file_identity(source) != verified_sources[name]
            ):
                raise ValueError(
                    f"shared staging payload changed after verification: {name}"
                )
            existing = destination.exists()
            input_path = destination if existing else source
            if not input_path.is_file():
                raise ValueError(f"missing shared release payload: {name}")
            if input_path.stat().st_nlink != 1:
                raise ValueError(f"hard-linked shared release payload: {name}")
            stamp = range_file_identity(input_path)
            if stamp[2] != entry["size"]:
                raise ValueError(f"shared release payload size differs: {name}")
            receiptable = (
                proof_store is not None and entry["size"] >= INVENTORY_RECEIPT_MIN_BYTES
            )
            reused = checked.get(destination) == stamp or (
                existing
                and shared_payload_receipt_hit(
                    destination, entry, proof_store, verification_mode
                )
            )
            # A resumed release replays references whose source was already moved.
            authenticate = not reused and (
                (existing and (verified or source.exists()))
                or (not existing and not verified)
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
            if authenticate or verified or reused:
                checked[destination] = generation
            if (
                receiptable
                and not existing
                and (authenticate or verified_sources is not None)
            ):
                release_receipt.record(
                    proof_store,
                    "shared_payload",
                    destination,
                    entry["sha256"],
                    [],
                    shared_generations={destination: generation},
                )
    return SharedPublication(generations, checked)
