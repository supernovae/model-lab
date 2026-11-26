"""Metadata-only inventory of shared Corpus artifacts and their release roots."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from heapq import nsmallest
from operator import itemgetter
from pathlib import Path
from typing import Any

from klab.canonical import canonical_digest
from klab.corpus.prepared_shards import normalized_store_root
from klab.corpus.release_members import shared_stage_members
from klab.safe_io import symlinked

_SOURCE = re.compile(r"[a-z][a-z0-9_-]*")
_KEY = re.compile(r"[0-9a-f]{64}")
_INCOMPLETE = ".incomplete-"
_FREEZING = ".freezing-"
_PREFIX = re.compile(r"[0-9a-f]{2}")


@contextmanager
def _directory(path: str | Path, *, parent: int | None = None) -> Iterator[int]:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
    )
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def _published(root: int, source_id: str | None, totals: dict[str, int]):
    with os.scandir(root) as sources:
        for source in sources:
            if not _SOURCE.fullmatch(source.name) or (
                source_id is not None and source.name != source_id
            ):
                continue
            with (
                _directory(source.name, parent=root) as namespace,
                os.scandir(namespace) as entries,
            ):
                for entry in entries:
                    key = entry.name.removeprefix(_INCOMPLETE)
                    if not _KEY.fullmatch(key):
                        continue
                    if entry.name.startswith(_INCOMPLETE):
                        if entry.is_dir(follow_symlinks=False):
                            totals["staging_directories"] += 1
                        continue
                    try:
                        with (
                            _directory(key, parent=namespace) as partition,
                            os.scandir(partition) as members,
                        ):
                            size = 0
                            for member in members:
                                info = member.stat(follow_symlinks=False)
                                if not stat.S_ISREG(info.st_mode):
                                    raise ValueError(
                                        f"invalid normalized member: {source.name}/{key}/{member.name}"
                                    )
                                size += info.st_size
                    except FileNotFoundError:
                        totals["disappeared_partitions"] += 1
                        continue
                    totals["published_partitions"] += 1
                    totals["retained_bytes"] += size
                    yield {
                        "artifact_id": f"{source.name}/{key}",
                        "source_id": source.name,
                        "computation_key": key,
                        "retained_bytes": size,
                    }


def normalized_inventory(
    work_root: Path, *, source_id: str | None, limit: int, after: str | None
) -> dict[str, Any]:
    """Count visible published bytes and return one bounded page of partition keys."""
    if source_id is not None and not _SOURCE.fullmatch(source_id):
        raise ValueError("invalid source ID")
    if not 1 <= limit <= 1000:
        raise ValueError("artifact limit must be between 1 and 1000")
    if after is not None:
        source, _, key = after.partition("/")
        if (
            not _SOURCE.fullmatch(source)
            or not _KEY.fullmatch(key)
            or (source_id is not None and source_id != source)
        ):
            raise ValueError("artifact cursor must be a source ID and computation key")
    root = normalized_store_root(work_root)
    if symlinked(root):
        raise ValueError("symlinked normalized store")
    totals = {
        "published_partitions": 0,
        "retained_bytes": 0,
        "staging_directories": 0,
        "disappeared_partitions": 0,
    }

    page = []
    if root.exists():
        with _directory(root) as directory:
            page = nsmallest(
                limit + 1,
                (
                    row
                    for row in _published(directory, source_id, totals)
                    if after is None or row["artifact_id"] > after
                ),
                key=itemgetter("artifact_id"),
            )
    return {
        "artifact_type": "normalized_source_partition",
        "verification": "advisory",
        "source_id": source_id,
        **totals,
        "partitions": page[:limit],
        "next_after": page[limit - 1]["artifact_id"] if len(page) > limit else None,
    }


def _stamp(info: os.stat_result | None) -> tuple[int, ...] | None:
    if info is None:
        return None
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class _RetentionScan:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.observed: dict[Path, tuple[int, ...] | None] = {}
        self.problems: list[dict[str, str]] = []
        self.problem_count = 0
        self.published_manifests = self.staging_manifests = 0

    def issue(self, path: Path, reason: str) -> None:
        self.problem_count += 1
        if len(self.problems) < 8:
            self.problems.append(
                {"path": path.relative_to(self.root).as_posix(), "reason": reason}
            )

    def watch(self, path: Path) -> os.stat_result | None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            info = None
        self.observed[path] = _stamp(info)
        return info

    def directory(self, path: Path) -> bool:
        info = self.watch(path)
        if info is None:
            return False
        if not stat.S_ISDIR(info.st_mode):
            self.issue(path, "unsafe_directory")
            return False
        return True

    def check_generations(self) -> None:
        for path, before in self.observed.items():
            try:
                after = _stamp(path.lstat())
            except FileNotFoundError:
                after = None
            except OSError:
                self.issue(path, "unreadable_generation")
                continue
            if after != before:
                self.issue(path, "changed_during_scan")


@dataclass(slots=True)
class _ReleaseReferences:
    size: int
    published: int = 0
    staging: int = 0
    example: str = ""


def _stage_references(scan: _RetentionScan) -> dict[str, _ReleaseReferences]:
    references: dict[str, _ReleaseReferences] = {}
    corpora = scan.root / "corpora"
    if not scan.directory(corpora):
        return references
    for project in corpora.iterdir():
        if not _SOURCE.fullmatch(project.name) or not scan.directory(project):
            continue
        releases = project / "releases"
        if not scan.directory(releases):
            continue
        for release in releases.iterdir():
            staging = release.name.startswith(_FREEZING)
            key = release.name.removeprefix(_FREEZING)
            if not _KEY.fullmatch(key):
                continue
            if not scan.directory(release):
                scan.issue(release, "missing_release_directory")
                continue
            path = release / "manifest.json"
            try:
                with _directory(release) as directory:
                    descriptor = os.open(
                        path.name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=directory,
                    )
                    with os.fdopen(descriptor, "rb") as stream:
                        info = os.fstat(stream.fileno())
                        if not stat.S_ISREG(info.st_mode):
                            raise ValueError("unsafe release manifest")
                        scan.observed[path] = _stamp(info)
                        manifest = json.load(stream)
                release_id = manifest.pop("release_id")
                if (
                    manifest["schema_version"] != 1
                    or manifest["corpus_id"] != project.name
                    or release_id != canonical_digest(manifest)
                    or key != (manifest["build_id"] if staging else release_id)
                ):
                    raise ValueError("release identity mismatch")
                files = manifest["files"]
                members = shared_stage_members(release, files)
                if members:
                    outputs = {
                        f"stages/{stage['id']}.jsonl": stage["output_sha256"]
                        for stage in manifest["stages"]
                    }
                    if any(files[name]["sha256"] != outputs[name] for name in members):
                        raise ValueError("stage digest mismatch")
            except OSError:
                scan.issue(path, "missing_or_unreadable_manifest")
                continue
            except ValueError, KeyError, TypeError, AttributeError:
                scan.issue(path, "invalid_manifest")
                continue
            if staging:
                scan.staging_manifests += 1
            else:
                scan.published_manifests += 1
            seen: set[str] = set()
            example = release.relative_to(scan.root).as_posix()
            for name, member in members.items():
                digest, size = member.name, files[name]["size"]
                reference = references.setdefault(digest, _ReleaseReferences(size))
                if size != reference.size:
                    scan.issue(path, "inconsistent_reference_size")
                if digest in seen:
                    continue
                seen.add(digest)
                reference.staging += staging
                reference.published += not staging
                reference.example = (
                    min(reference.example, example) if reference.example else example
                )
    return references


def _stage_payloads(scan: _RetentionScan, root: Path) -> Iterator[tuple[str, int]]:
    if symlinked(root):
        scan.issue(root, "unsafe_store")
        return
    if not scan.directory(root):
        return
    with _directory(root) as directory, os.scandir(directory) as entries:
        for prefix in entries:
            if not _PREFIX.fullmatch(prefix.name) or not scan.directory(
                root / prefix.name
            ):
                continue
            with (
                _directory(prefix.name, parent=directory) as shard,
                os.scandir(shard) as payloads,
            ):
                for entry in payloads:
                    path = root / prefix.name / entry.name
                    if not _KEY.fullmatch(entry.name) or not entry.name.startswith(
                        prefix.name
                    ):
                        scan.issue(path, "invalid_payload_name")
                        continue
                    info = entry.stat(follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        scan.issue(path, "unsafe_payload")
                        continue
                    yield entry.name, info.st_size


def shared_stage_inventory(
    work_root: Path, *, limit: int, after: str | None, unreferenced: bool
) -> dict[str, Any]:
    """Preview shared-stage retention from published and in-flight release manifests."""
    if not 1 <= limit <= 1000:
        raise ValueError("artifact limit must be between 1 and 1000")
    if after is not None and not _KEY.fullmatch(after):
        raise ValueError("shared stage cursor must be a full SHA-256 digest")
    root = work_root / "artifacts" / "corpus" / "release-stages"
    scan = _RetentionScan(work_root)
    references: dict[str, _ReleaseReferences] = {}
    totals = dict.fromkeys(
        (
            "shared_payloads",
            "retained_bytes",
            "unreferenced_payloads",
            "unreferenced_bytes",
        ),
        0,
    )

    def rows():
        for digest, size in _stage_payloads(scan, root):
            reference = references.pop(digest, None)
            totals["shared_payloads"] += 1
            totals["retained_bytes"] += size
            if reference is None:
                totals["unreferenced_payloads"] += 1
                totals["unreferenced_bytes"] += size
            elif size != reference.size:
                scan.issue(root / digest[:2] / digest, "payload_size_mismatch")
            if (after is None or digest > after) and (
                not unreferenced or reference is None
            ):
                yield {
                    "artifact_id": digest,
                    "payload_sha256": digest,
                    "retained_bytes": size,
                    "published_releases": reference.published if reference else 0,
                    "staging_releases": reference.staging if reference else 0,
                    "example_reference": reference.example if reference else None,
                }

    page = []
    try:
        references = _stage_references(scan)
        page = nsmallest(limit + 1, rows(), key=itemgetter("artifact_id"))
        for digest, reference in references.items():
            if reference.published:
                scan.issue(root / digest[:2] / digest, "missing_published_payload")
    except OSError as error:
        path = Path(error.filename) if error.filename else work_root
        scan.issue(
            path if path.is_relative_to(work_root) else work_root, "scan_interrupted"
        )
    scan.check_generations()
    if scan.problem_count and unreferenced:
        page = []
    return {
        "artifact_type": "shared_release_stage",
        "verification": "advisory",
        "scan_status": "incomplete" if scan.problem_count else "complete",
        "problem_count": scan.problem_count,
        "problems": scan.problems,
        "published_manifests": scan.published_manifests,
        "staging_manifests": scan.staging_manifests,
        **totals,
        "unreferenced_bytes": None
        if scan.problem_count
        else totals["unreferenced_bytes"],
        "unreferenced_payloads": None
        if scan.problem_count
        else totals["unreferenced_payloads"],
        "payloads": page[:limit],
        "next_after": page[limit - 1]["artifact_id"] if len(page) > limit else None,
    }
