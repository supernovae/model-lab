"""Explicit, Git-authenticated operational source compatibility; never source spoofing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import StrictModel
from klab.documents import read_document
from klab.training.manifest import source_identity

_ENV = (
    "KLAB_SOURCE_COMPATIBILITY",
    "KLAB_SOURCE_COMPATIBILITY_SHA256",
    "KLAB_SOURCE_COMPATIBILITY_COMMIT",
)
_HEX = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
# These integration points may change provenance/operational checks, not model math.
# Each record also pins the exact before/after bytes and requires operator acceptance.
_OPERATIONAL_PATHS = frozenset(
    {
        "operational_monitor.py",
        "operational_monitor_cli.py",
        "workspace_preflight.py",
        "cli/main.py",
        "experiments/cli.py",
        "experiments/lock.py",
        "experiments/artifacts.py",
        "experiments/storage.py",
        "experiments/evidence.py",
        "experiments/source_compatibility.py",
        "experiments/binding.py",
        "data/packing.py",
        "data/verification.py",
        "engines/pytorch.py",
        "staging.py",
        "training/pilot_deadline.py",
        "training/pilot_progress.py",
        "training/pilot.py",
        "training/trainer.py",
        "training/manifest.py",
    }
)


class OperationalSourceCompatibility(StrictModel):
    """Operator-pinned reviewed mapping; an unsigned document grants no authority."""

    compatibility_version: Literal[1]
    baseline_commit: str
    execution_commit: str
    baseline_source_identity: dict[str, Any]
    execution_source_identity: dict[str, Any]
    operational_changes: list[dict[str, str | None]]
    authorization: str = Field(min_length=1)

    @field_validator("compatibility_version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("compatibility version must be integer 1")
        return value

    @field_validator("baseline_commit", "execution_commit")
    @classmethod
    def commit_sha(cls, value: str) -> str:
        if not _COMMIT.fullmatch(value):
            raise ValueError("compatibility requires full Git commit IDs")
        return value


def _git(repo: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, check=False
    )
    if result.returncode:
        raise ValueError("source compatibility Git evidence unavailable")
    return result.stdout


@lru_cache(maxsize=8)
def _committed_identity_bytes(repo: Path, commit: str) -> bytes:
    inventory = []
    process = subprocess.Popen(
        ["git", "-C", str(repo), "archive", commit, "src/klab"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                name = Path(member.name)
                if not member.isfile() or name.suffix not in {
                    ".py",
                    ".json",
                    ".yaml",
                    ".yml",
                    ".md",
                }:
                    continue
                relative = name.relative_to("src/klab").as_posix()
                if (
                    relative == "research/resources/lifecycle.json"
                    or "__pycache__" in name.parts
                ):
                    continue
                stream = archive.extractfile(member)
                assert stream is not None
                with stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                inventory.append((relative, digest))
        # Tar EOF precedes the producer's trailing block padding. Keep the pipe
        # open until Git finishes writing, or a complete inventory can SIGPIPE.
        while process.stdout.read(64 * 1024):
            pass
    finally:
        process.stdout.close()
        returncode = process.wait()
    if returncode or not inventory:
        raise ValueError("source compatibility package inventory unavailable")
    inventory.sort()
    return canonical_json(
        {
            "algorithm": "package-path-sha256-v2",
            "files": inventory,
            "sha256": canonical_digest(inventory),
        }
    )


def _committed_identity(repo: Path, commit: str) -> dict[str, Any]:
    return json.loads(_committed_identity_bytes(repo, commit))


def _changes(
    baseline: dict[str, Any], execution: dict[str, Any]
) -> list[dict[str, str | None]]:
    before, after = dict(baseline["files"]), dict(execution["files"])
    return [
        {
            "path": path,
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
        }
        for path in sorted(before.keys() | after.keys())
        if before.get(path) != after.get(path)
    ]


@lru_cache(maxsize=8)
def _authenticate(
    path: Path, expected_sha256: str, evidence_commit: str, current_sha256: str
) -> bytes:
    """Cache authenticated canonical bytes, not caller-constructible trust flags."""
    if not _HEX.fullmatch(expected_sha256) or not _COMMIT.fullmatch(evidence_commit):
        raise ValueError("source compatibility needs exact digest and evidence commit")
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError("symlinked source compatibility evidence")
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError("source compatibility evidence digest changed")
    repo = Path(_git(path.parent, "rev-parse", "--show-toplevel").decode().strip())
    relative = path.relative_to(repo).as_posix()
    if _git(repo, "cat-file", "blob", f"{evidence_commit}:{relative}") != body:
        raise ValueError("source compatibility differs from trusted committed evidence")
    record = OperationalSourceCompatibility.model_validate(read_document(path))
    baseline = _committed_identity(repo, record.baseline_commit)
    execution = _committed_identity(repo, record.execution_commit)
    if (
        canonical_json(record.baseline_source_identity) != canonical_json(baseline)
        or canonical_json(record.execution_source_identity) != canonical_json(execution)
        or execution["sha256"] != current_sha256
    ):
        raise ValueError("source compatibility implementation inventory changed")
    changes = _changes(baseline, execution)
    if record.operational_changes != changes:
        raise ValueError("source compatibility change inventory differs")
    if any(change["path"] not in _OPERATIONAL_PATHS for change in changes):
        raise ValueError(
            "source compatibility includes unsupported scientific code changes"
        )
    return canonical_json(record.model_dump(mode="json"))


def active_source_compatibility() -> dict[str, Any] | None:
    """Authenticate an explicitly pinned operational environment in each process."""
    values = tuple(os.environ.get(key) for key in _ENV)
    if not any(value is not None for value in values):
        return None
    if not all(values):
        raise ValueError("source compatibility requires path, SHA and trusted commit")
    path, digest, commit = values
    assert path is not None and digest is not None and commit is not None
    target = Path(path)
    if not target.is_absolute() or ".." in target.parts:
        raise ValueError(
            "source compatibility requires an absolute non-traversing path"
        )
    for component in (target, *target.parents):
        if component.is_symlink():
            raise ValueError("symlinked source compatibility evidence")
    # Recheck bytes even on a cache hit; current package identity is independently
    # measured, so a stale environment or source mutation cannot extend authority.
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise ValueError("source compatibility evidence digest changed")
    current = source_identity()
    return json.loads(_authenticate(target, digest, commit, str(current["sha256"])))


def locked_source_identity() -> dict[str, Any]:
    """Return authenticated historical lock provenance, not execution provenance."""
    record = active_source_compatibility()
    return source_identity() if record is None else record["baseline_source_identity"]


def source_identities_compatible(recorded_sha256: str, current_sha256: str) -> bool:
    """Permit only equality or the explicitly authenticated historical→current pair."""
    record = active_source_compatibility()
    if recorded_sha256 == current_sha256:
        return True
    return record is not None and (
        recorded_sha256 == record["baseline_source_identity"]["sha256"]
        and current_sha256 == record["execution_source_identity"]["sha256"]
    )
