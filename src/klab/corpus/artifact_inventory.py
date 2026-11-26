"""Metadata-only inventory of shared Corpus artifacts and their release roots."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from heapq import nsmallest
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Any

from klab.canonical import canonical_digest
from klab.corpus.normalized_spans import (
    NORMALIZED_SPANS,
    has_span_projection,
    span_partitions,
)
from klab.corpus.prepared_shards import normalized_store_root
from klab.corpus.release_members import (
    SHARED_STAGE,
    release_retention_lease,
    shared_members,
    shared_payload_lease,
    shared_store,
)
from klab.safe_io import symlinked
from klab.verification_proofs import ProofStore, writable_by_others

_SOURCE = re.compile(r"[a-z][a-z0-9_-]*")
_KEY = re.compile(r"[0-9a-f]{64}")
_INCOMPLETE = ".incomplete-"
_FREEZING = ".freezing-"
_VIEW = ".views-"
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
    work_root: Path,
    *,
    source_id: str | None,
    limit: int,
    after: str | None,
    release_references: bool = False,
    build_references: bool = False,
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
    rows = page[:limit]
    result = {
        "artifact_type": "normalized_source_partition",
        "verification": "advisory",
        "source_id": source_id,
        **totals,
        "partitions": rows,
        "next_after": page[limit - 1]["artifact_id"] if len(page) > limit else None,
    }
    if release_references or build_references:
        result.update(
            _normalized_reference_report(
                work_root, rows, include_builds=build_references
            )
        )
    return result


def _stamp(info: os.stat_result | None) -> tuple[int, ...] | None:
    if info is None:
        return None
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class _RetentionScan:
    def __init__(self, root: Path, *, owned: bool = False) -> None:
        self.root = root
        self.owned = owned
        self.observed: dict[Path, tuple[int, ...] | None] = {}
        self.problems: list[dict[str, str]] = []
        self.problem_count = 0
        self.published_manifests = self.staging_manifests = 0
        self.build_manifests = self.staging_build_manifests = self.consumed_builds = 0

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
        self.observe(path, info)
        return info

    def observe(self, path: Path, info: os.stat_result | None) -> None:
        generation = _stamp(info)
        if self.observed.setdefault(path, generation) != generation:
            self.issue(path, "changed_during_scan")
        if (
            self.owned
            and info is not None
            and (
                info.st_uid != os.getuid()
                or writable_by_others(info)
                or (stat.S_ISREG(info.st_mode) and info.st_nlink != 1)
            )
        ):
            self.issue(path, "unowned_or_writable_by_others")

    def directory(self, path: Path) -> bool:
        info = self.watch(path)
        if info is None:
            return False
        if not stat.S_ISDIR(info.st_mode):
            self.issue(path, "unsafe_directory")
            return False
        return True

    def control(self, root: Path, name: str, *, optional: bool = False) -> dict | None:
        path = root / name
        with _directory(root) as directory:
            try:
                descriptor = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
                )
            except FileNotFoundError:
                if not optional:
                    raise
                self.observe(path, None)
                return None
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("unsafe control file")
                previous = self.problem_count
                self.observe(path, info)
                if self.problem_count != previous:
                    return None
                result = json.load(stream)
                if not isinstance(result, dict):
                    raise TypeError("invalid control file")
                return result

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

    def interrupted(self, error: OSError) -> None:
        path = Path(error.filename) if error.filename else self.root
        self.issue(
            path if path.is_relative_to(self.root) else self.root, "scan_interrupted"
        )

    def summary(self) -> dict[str, Any]:
        return {
            "scan_status": "incomplete" if self.problem_count else "complete",
            "problem_count": self.problem_count,
            "problems": self.problems,
            "published_manifests": self.published_manifests,
            "staging_manifests": self.staging_manifests,
        }


@dataclass(slots=True)
class _ArtifactReferences:
    size: int
    published: int = 0
    staging: int = 0
    example: str = ""


def _project_roots(scan: _RetentionScan) -> Iterator[Path]:
    corpora = scan.root / "corpora"
    if not scan.directory(corpora):
        if scan.owned:
            scan.issue(corpora, "missing_corpus_roots")
        return
    for project in corpora.iterdir():
        if not _SOURCE.fullmatch(project.name):
            if scan.owned and (project / "releases").exists():
                scan.issue(project, "unrecognized_project_root")
            continue
        if scan.directory(project):
            yield project


def _span_references(root: Path, files: dict) -> Iterator[tuple[str, int]]:
    return (
        (f"{part.entry['source_id']}/{part.entry['partition_key']}", part.entry["size"])
        for part in span_partitions(root, files)
    )


def _count_references(
    scan: _RetentionScan,
    path: Path,
    retained: Iterator[tuple[str, int]],
    reference_counts: dict[str, _ArtifactReferences],
    *,
    staging: bool,
    artifact_ids: set[str] | None,
) -> None:
    seen: set[str] = set()
    example = path.parent.relative_to(scan.root).as_posix()
    for artifact_id, size in retained:
        if artifact_ids is not None and artifact_id not in artifact_ids:
            continue
        reference = reference_counts.setdefault(artifact_id, _ArtifactReferences(size))
        if size != reference.size:
            scan.issue(path, "inconsistent_reference_size")
        if artifact_id in seen:
            continue
        seen.add(artifact_id)
        reference.staging += staging
        reference.published += not staging
        reference.example = (
            min(reference.example, example) if reference.example else example
        )


def _release_references(
    scan: _RetentionScan,
    storage: str,
    *,
    artifact_ids: set[str] | None = None,
    release_builds: dict[tuple[str, str], str] | None = None,
) -> dict[str, _ArtifactReferences]:
    references: dict[str, _ArtifactReferences] = {}
    for project in _project_roots(scan):
        releases = project / "releases"
        if not scan.directory(releases):
            continue
        for release in releases.iterdir():
            staging = release.name.startswith(_FREEZING)
            key = release.name.removeprefix(_FREEZING)
            if not _KEY.fullmatch(key):
                if scan.owned and (
                    (release / "manifest.json").exists() or release.is_symlink()
                ):
                    scan.issue(release, "unrecognized_release_root")
                continue
            if not scan.directory(release):
                scan.issue(release, "missing_release_directory")
                continue
            path = release / "manifest.json"
            try:
                manifest = scan.control(release, path.name)
                if manifest is None:
                    continue
                release_id = manifest.pop("release_id")
                if (
                    manifest["schema_version"] != 1
                    or manifest["corpus_id"] != project.name
                    or release_id != canonical_digest(manifest)
                    or key != (manifest["build_id"] if staging else release_id)
                ):
                    raise ValueError("release identity mismatch")
                files = manifest["files"]
                members = shared_members(release, files)
                stages = [
                    name for name in members if files[name]["storage"] == SHARED_STAGE
                ]
                if stages:
                    outputs = {
                        f"stages/{stage['id']}.jsonl": stage["output_sha256"]
                        for stage in manifest["stages"]
                    }
                    if any(files[name]["sha256"] != outputs[name] for name in stages):
                        raise ValueError("stage digest mismatch")
                if storage == NORMALIZED_SPANS:
                    retained = _span_references(release, files)
                else:
                    retained = (
                        (member.name, files[name]["size"])
                        for name, member in members.items()
                        if files[name]["storage"] == storage
                    )
                if release_builds is not None:
                    release_builds[project.name, release_id] = manifest["build_id"]
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
            _count_references(
                scan,
                path,
                retained,
                references,
                staging=staging,
                artifact_ids=artifact_ids,
            )
    return references


def _build_roots(scan: _RetentionScan, project: Path) -> Iterator[tuple[Path, bool]]:
    builds = project / "builds"
    if scan.directory(builds):
        for build in builds.iterdir():
            if _KEY.fullmatch(build.name):
                yield build, False
    for view in project.iterdir():
        if (
            view.name.startswith(_VIEW)
            and _KEY.fullmatch(view.name.removeprefix(_VIEW))
            and scan.directory(view)
        ):
            yield view / "build", True


def _build_references(
    scan: _RetentionScan,
    artifact_ids: set[str],
    release_builds: dict[tuple[str, str], str],
) -> dict[str, _ArtifactReferences]:
    references: dict[str, _ArtifactReferences] = {}
    for project in _project_roots(scan):
        for root, staging in _build_roots(scan, project):
            if not scan.directory(root):
                scan.issue(root, "missing_build_directory")
                continue
            path = root / "consumed.json"
            try:
                if (
                    not staging
                    and (marker := scan.control(root, path.name, optional=True))
                    is not None
                ):
                    if set(marker) != {"release_id"} or not _KEY.fullmatch(
                        marker["release_id"]
                    ):
                        raise ValueError("invalid consumed build marker")
                    if (
                        release_builds.get((project.name, marker["release_id"]))
                        == root.name
                    ):
                        scan.consumed_builds += 1
                    else:
                        scan.issue(path, "missing_consumed_release")
                    continue
                path = root / "build.json"
                manifest = scan.control(root, path.name, optional=staging)
                if manifest is None:
                    scan.issue(path, "unsealed_build")
                    continue
                identity = manifest["identity"]
                build_id = manifest["build_id"]
                files = manifest["files"]
                if (
                    manifest["schema_version"] != 1
                    or identity["project_id"] != project.name
                    or build_id != canonical_digest(identity)
                    or (not staging and build_id != root.name)
                    or has_span_projection(files)
                    != bool(identity.get("reference_normalized_spans"))
                ):
                    raise ValueError("build identity mismatch")
                retained = _span_references(project / "builds" / build_id, files)
            except OSError:
                scan.issue(path, "missing_or_unreadable_build_control")
                continue
            except ValueError, KeyError, TypeError, AttributeError:
                scan.issue(path, "invalid_build_control")
                continue
            scan.staging_build_manifests += staging
            scan.build_manifests += not staging
            _count_references(
                scan,
                path,
                retained,
                references,
                staging=staging,
                artifact_ids=artifact_ids,
            )
    return references


def _normalized_reference_report(
    work_root: Path, rows: list[dict[str, Any]], *, include_builds: bool
) -> dict[str, Any]:
    scan = _RetentionScan(work_root)
    artifact_ids = {row["artifact_id"] for row in rows}
    releases: dict[str, _ArtifactReferences] = {}
    builds: dict[str, _ArtifactReferences] = {}
    release_builds: dict[tuple[str, str], str] | None = {} if include_builds else None
    try:
        releases = _release_references(
            scan,
            NORMALIZED_SPANS,
            artifact_ids=artifact_ids,
            release_builds=release_builds,
        )
        if release_builds is not None:
            builds = _build_references(scan, artifact_ids, release_builds)
    except OSError as error:
        scan.interrupted(error)
    scan.check_generations()
    if scan.problem_count:
        releases.clear()
        builds.clear()
    absent_count = None if scan.problem_count else 0
    for row in rows:
        reference = releases.get(row["artifact_id"])
        row.update(
            published_releases=reference.published if reference else absent_count,
            staging_releases=reference.staging if reference else absent_count,
            example_reference=reference.example if reference else None,
        )
        if include_builds:
            reference = builds.get(row["artifact_id"])
            row.update(
                published_builds=reference.published if reference else absent_count,
                staging_builds=reference.staging if reference else absent_count,
                example_build_reference=reference.example if reference else None,
            )
    report = scan.summary()
    if include_builds:
        report.update(
            build_manifests=scan.build_manifests,
            staging_build_manifests=scan.staging_build_manifests,
            consumed_builds=scan.consumed_builds,
        )
    return report


def _shared_payloads(scan: _RetentionScan, root: Path) -> Iterator[tuple[str, int]]:
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


def shared_release_inventory(
    work_root: Path, *, storage: str, limit: int, after: str | None, unreferenced: bool
) -> dict[str, Any]:
    """Preview shared retention from published and in-flight release manifests."""
    if not 1 <= limit <= 1000:
        raise ValueError("artifact limit must be between 1 and 1000")
    if after is not None and not _KEY.fullmatch(after):
        raise ValueError("shared release cursor must be a full SHA-256 digest")
    root = shared_store(work_root, storage)
    scan = _RetentionScan(work_root)
    references: dict[str, _ArtifactReferences] = {}
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
        for digest, size in _shared_payloads(scan, root):
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
        references = _release_references(scan, storage)
        page = nsmallest(limit + 1, rows(), key=itemgetter("artifact_id"))
        for digest, reference in references.items():
            if reference.published:
                scan.issue(root / digest[:2] / digest, "missing_published_payload")
    except OSError as error:
        scan.interrupted(error)
    scan.check_generations()
    if scan.problem_count and unreferenced:
        page = []
    return {
        "artifact_type": f"shared_release_{storage.removeprefix('shared_')}",
        "verification": "advisory",
        **scan.summary(),
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


def prune_shared_release_payloads(
    work_root: Path, *, storage: str, digests: list[str], apply: bool = False
) -> dict[str, Any]:
    """Prune explicitly named digest files after checking the retained release roots."""
    if not digests or any(not _KEY.fullmatch(digest) for digest in digests):
        raise ValueError("prune requires full SHA-256 digests")
    work_root = work_root.absolute()
    store = shared_store(work_root, storage)
    paths = {digest: store / digest[:2] / digest for digest in sorted(set(digests))}
    rows: list[dict[str, Any]] = []
    removed: list[str] = []
    pruned_bytes = 0
    result = {
        "store": str(store),
        "payloads": rows,
        "pruned": removed,
    }
    try:
        trust = ProofStore(work_root, read_only=True)
        if not trust.trusted(work_root):
            raise ValueError("shared release prune requires an owned work root")
        if symlinked(store) or (store.exists() and not trust.trusted(store)):
            raise ValueError("unsafe shared release store")
        with ExitStack() as locks:
            leased = set()
            if apply:
                locks.enter_context(release_retention_lease(work_root, prune=True))
                for digest, path in paths.items():
                    if path.exists():
                        locks.enter_context(shared_payload_lease(path, wait=False))
                        leased.add(digest)
            scan = _RetentionScan(work_root, owned=True)
            references = _release_references(scan, storage, artifact_ids=set(paths))
            for digest, path in paths.items():
                if symlinked(path):
                    raise ValueError(f"unsafe shared payload: {path}")
                info = scan.watch(path)
                if info is not None and (
                    not stat.S_ISREG(info.st_mode) or not trust.trusted(path)
                ):
                    raise ValueError(f"unsafe shared payload: {path}")
                if apply and info is not None and digest not in leased:
                    raise ValueError(f"payload appeared during prune: {digest}")
                reference = references.get(digest)
                rows.append(
                    {
                        "artifact_id": digest,
                        "action": "protected"
                        if reference
                        else ("prune" if info else "missing"),
                        "bytes": info.st_size if info else 0,
                        "published_releases": reference.published if reference else 0,
                        "staging_releases": reference.staging if reference else 0,
                        "example_reference": reference.example if reference else None,
                    }
                )
            scan.check_generations()
            if scan.problem_count:
                return {
                    **result,
                    "status": "blocked",
                    "payloads": [],
                    "reason": "release root scan was incomplete",
                    "problems": scan.problems,
                    "pruned_bytes": 0,
                }
            if not apply:
                return {**result, "status": "planned", "pruned_bytes": 0}
            if any(row["action"] == "protected" for row in rows):
                raise ValueError(
                    "requested payload is referenced by a retained or staged release"
                )
            candidates = [row for row in rows if row["action"] == "prune"]
            if candidates:
                with _directory(store) as namespace:
                    for prefix, batch in groupby(
                        candidates, key=lambda row: row["artifact_id"][:2]
                    ):
                        with _directory(prefix, parent=namespace) as shard:
                            before = len(removed)
                            try:
                                for row in batch:
                                    digest = row["artifact_id"]
                                    info = os.stat(
                                        digest, dir_fd=shard, follow_symlinks=False
                                    )
                                    if _stamp(info) != scan.observed[paths[digest]]:
                                        raise ValueError(
                                            f"payload changed during prune: {digest}"
                                        )
                                    os.unlink(digest, dir_fd=shard)
                                    removed.append(digest)
                                    pruned_bytes += row["bytes"]
                            finally:
                                if len(removed) != before:
                                    os.fsync(shard)
            return {**result, "status": "pruned", "pruned_bytes": pruned_bytes}
    except BlockingIOError:
        reason = "shared release publisher or prune holds a required lock"
    except (OSError, ValueError) as error:
        reason = str(error)
    return {
        **result,
        "status": "partial" if removed else "blocked",
        "reason": reason,
        "pruned_bytes": pruned_bytes,
    }
