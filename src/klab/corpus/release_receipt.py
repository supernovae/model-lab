"""Verify a release or build once; later commands trust a signed receipt until its files change.

A receipt is stored in the work root's proof store, signed with the host-local key,
and names the content-derived id, the verifier implementation that accepted it, and
a digest of a stat index (device, inode, mode, size, mtime and ctime) of every file
in the release and in the snapshots it cites, like a Git index. A command that
finds a matching receipt does a few tens of thousands of `lstat` calls and no
reads. A mismatch, another verifier implementation, a forged receipt, an untrusted
store or `--cold` verifies from the bytes.

Trust model: this believes the local filesystem's metadata, as every other receipt
does. It catches any edit, truncation, replacement or restore of a file, including
one that resets mtime, because ctime cannot be set from user space. It does not
defend against an account that can write the proof store and its key, or a root
user who changes the clock; `--cold` re-hashes every byte. The key never leaves the
host, so a receipt copied elsewhere is rejected. Moving, copying or restoring a
release changes its inodes and verifies it again.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.release_verify import current_settings
from klab.verification_proofs import writable_by_others

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore, VerificationMode

SCHEMA = 1


class Untrustable(ValueError):
    """A file that cannot back a receipt, with the reason the operator needs."""


@cache
def verifier_digest(kind: str) -> str:
    """Digest of the source that decides whether a release, build or export is valid.

    This is the import closure of the verifier inside the corpus package plus
    a few shared modules, not of the whole program: editing the near-duplicate
    screen, the trainer or the command line does not make a release unverified,
    and editing anything that verification runs does.
    """
    if kind in {"stage_output", "raw_replay", "staged_release"}:
        return verifier_digest("release")
    from klab.verifier_authority import CHAT_RENDERER_MODULES, SHARED, scoped_modules

    exported = kind == "export"
    return canonical_digest(
        scoped_modules(
            corpus=("corpus.release", *(("corpus.export",) if exported else ())),
            leaves=(
                *SHARED,
                "data.verification",
                *CHAT_RENDERER_MODULES,
                *(("config.loading",) if exported else ()),
            ),
        )
    )


def stat_index(roots: list[Path], *, strict: bool = True) -> tuple[str, int]:
    """Digest and entry count of the stat index of every file under `roots`.

    Raises Untrustable for anything that is neither a file nor a directory and,
    when `strict`, for anything receipts do not stand behind: a file with another
    hard link, an entry not owned by this user or one that others may write.
    """
    digest, count, owner = hashlib.sha256(), 0, os.getuid()
    for root in roots:
        digest.update(canonical_json(str(root)))
        pending: list[tuple[Path, str]] = [(root, ".")]
        while pending:
            path, label = pending.pop()
            info = path.lstat()
            directory = stat.S_ISDIR(info.st_mode)
            if not (directory or stat.S_ISREG(info.st_mode)):
                raise Untrustable(f"{path} is not a regular file or directory")
            if strict and (info.st_uid != owner or writable_by_others(info)):
                raise Untrustable(f"{path} is not private to this user")
            if strict and not directory and info.st_nlink != 1:
                raise Untrustable(f"{path} has another hard link")
            digest.update(
                canonical_json(
                    [
                        label,
                        info.st_dev,
                        info.st_ino,
                        info.st_mode,
                        info.st_size,
                        info.st_mtime_ns,
                        info.st_ctime_ns,
                    ]
                )
            )
            count += 1
            if directory:
                with os.scandir(path) as children:
                    names = sorted(child.name for child in children)
                pending.extend(
                    (path / name, name if label == "." else f"{label}/{name}")
                    for name in reversed(names)
                )
    return digest.hexdigest(), count


def _receipt_path(store: ProofStore, kind: str, root: Path, identity: str) -> Path:
    return store.directory / f"{kind}-{canonical_digest([identity, str(root)])}.json"


def _payload(
    kind: str, root: Path, identity: str, dependencies: list[Path], index: str
) -> dict[str, Any]:
    return {
        "receipt": f"corpus_{kind}",
        "schema": SCHEMA,
        "id": identity,
        "path": str(root),
        "dependencies": [str(path) for path in dependencies],
        "verifier": verifier_digest(kind),
        "index_sha256": index,
    }


def _say(message: str) -> None:
    echo = current_settings().echo
    if echo is not None:
        print(message, file=echo, flush=True)


class Lookup(NamedTuple):
    hit: bool
    # The stat index read on the way, so a miss can hand it to `record`.
    index: tuple[str, int] | None


def lookup(
    store: ProofStore,
    kind: str,
    root: Path,
    identity: str,
    dependencies: list[Path],
) -> Lookup:
    """Whether a signed receipt vouches for `root` as it is on disk now."""
    started = time.monotonic()
    receipt = _receipt_path(store, kind, root, identity)
    try:
        index = stat_index([root, *dependencies])
    except (OSError, Untrustable) as error:
        _say(f"{kind} {identity[:12]}: files not receipt-eligible ({error})")
        return Lookup(False, None)
    proof = store.load(receipt)
    if proof is None:
        _say(f"{kind} {identity[:12]}: no usable receipt, verifying from the bytes")
        return Lookup(False, index)
    if proof != _payload(kind, root, identity, dependencies, index[0]):
        why = (
            "verifier changed"
            if proof.get("verifier") != verifier_digest(kind)
            else "files changed"
        )
        _say(
            f"{kind} {identity[:12]}: {why} since the receipt, verifying from the bytes"
        )
        return Lookup(False, index)
    _say(
        f"{kind} {identity[:12]}: verified by receipt "
        f"({index[1]} entries, {time.monotonic() - started:.1f}s)"
    )
    return Lookup(True, index)


def record(
    store: ProofStore,
    kind: str,
    root: Path,
    identity: str,
    dependencies: list[Path],
    *,
    index: tuple[str, int] | None = None,
    shared_generations: dict[Path, list[int]] | None = None,
) -> bool:
    """Sign a receipt for files that were just verified; say why when that is impossible.

    `index` is the stat index taken before verification began: the receipt is
    published only if the files still match it, so a change during verification
    is never blessed.
    """
    try:
        current = stat_index([root, *dependencies])
        if index is not None and index != current:
            raise Untrustable("files changed during verification")
        if shared_generations:
            from klab.corpus.shards import range_file_identity

            if any(
                range_file_identity(path) != stamp
                for path, stamp in shared_generations.items()
            ):
                raise Untrustable("shared files changed before receipt publication")
        published = store.publish(
            _receipt_path(store, kind, root, identity),
            _payload(kind, root, identity, dependencies, current[0]),
        )
    except (OSError, Untrustable) as error:
        _say(f"{kind} {identity[:12]}: no receipt recorded ({error})")
        return False
    if not published:
        _say(f"{kind} {identity[:12]}: no receipt recorded (proof store unavailable)")
    return published


def fingerprint(root: Path, manifest_name: str = "manifest.json") -> str:
    """Digest of the stat index of a release and the snapshots it cites.

    It changes when any of those files is rewritten, replaced, added or removed.
    Within one command it answers "did this release change since I verified it".
    """
    from klab.experiments.artifacts import snapshot_dependencies

    return stat_index(
        [root, *snapshot_dependencies(root, manifest_name)], strict=False
    )[0]


def receipt_first(
    kind: str,
    root: Path,
    identity: str,
    dependencies: list[Path],
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
    verify: Callable[[], Any],
    reused: Callable[[], Any],
) -> tuple[Any, str]:
    """Return `reused()` and the files' fingerprint, running `verify` only without a receipt.

    Reuse a completed Corpus check through a signed receipt for this id
    and verifier whose stat index still
    matches the files. Without one, `verify` runs the complete check and, when a
    proof store is available, a receipt is recorded for it, so `--cold` refreshes
    the receipt a later command reuses. The fingerprint is taken before verifying.
    """
    reuse = proof_store is not None and verification_mode == "verified_reuse"
    found = (
        lookup(proof_store, kind, root, identity, dependencies)
        if reuse
        else Lookup(False, None)
    )
    if found.hit:
        assert found.index is not None
        return reused(), found.index[0]
    before = found.index or stat_index([root, *dependencies], strict=False)
    started = time.monotonic()
    verified = verify()
    if proof_store is not None:
        record(proof_store, kind, root, identity, dependencies, index=before)
    _say(
        f"{kind} {identity[:12]}: verified from the bytes in "
        f"{time.monotonic() - started:.0f}s"
    )
    return verified, before[0]
