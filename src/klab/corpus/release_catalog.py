"""Bounded browsing of published Corpus release metadata."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from heapq import nsmallest
from pathlib import Path

from klab.canonical import canonical_digest
from klab.corpus.admission_inspection_cache import capture, unchanged
from klab.corpus.preparation_inputs import local_scope
from klab.safe_io import symlinked

_CORPUS = re.compile(r"[a-z][a-z0-9_-]*")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_METADATA_BYTES = 16 * 1024**2
RELEASE_SECTIONS = ("sources", "annotations", "duplicate-decisions")
_SIDECARS = {
    "sidecars": "text-statistics",
    "language_sidecars": "source-language",
    "quality_sidecars": "source-quality",
    "privacy_sidecars": "privacy-candidates",
}


def release_locator(work_root: Path, reference: str) -> tuple[str, str, str]:
    """Resolve a release ID or an exact published location within the work root."""
    path = Path(reference)
    if "@" in reference and len(path.parts) == 1:
        corpus, key = reference.split("@", 1)
        area = "releases"
    else:
        if path.is_absolute():
            if not path.is_relative_to(work_root):
                raise ValueError("Corpus release is outside the selected work root")
            path = path.relative_to(work_root)
        if len(path.parts) != 4 or path.parts[0] != "corpora":
            raise ValueError(
                "use CORPUS@RELEASE or a release location in this work root"
            )
        _, corpus, area, key = path.parts
    if (
        not _CORPUS.fullmatch(corpus)
        or area not in {"releases", "builds"}
        or not _DIGEST.fullmatch(key)
    ):
        raise ValueError("invalid published Corpus release location")
    return corpus, area, key


def _published_locations(root: Path, corpus_id: str | None):
    corpora = root / "corpora"
    if symlinked(corpora):
        raise ValueError("symlinked Corpus store")
    if not corpora.exists():
        return
    projects = [corpora / corpus_id] if corpus_id is not None else corpora.iterdir()
    for project in projects:
        if not _CORPUS.fullmatch(project.name) or not project.is_dir(
            follow_symlinks=False
        ):
            continue
        for area in ("releases", "builds"):
            directory = project / area
            if not directory.is_dir(follow_symlinks=False):
                continue
            with os.scandir(directory) as candidates:
                for candidate in candidates:
                    if not _DIGEST.fullmatch(candidate.name) or not candidate.is_dir(
                        follow_symlinks=False
                    ):
                        continue
                    manifest = Path(candidate.path) / "manifest.json"
                    if manifest.is_file(follow_symlinks=False):
                        yield (project.name, area, candidate.name)


def release_catalog(work_root: Path, *, corpus_id=None, after=None, limit=20):
    """Discover one page of candidate locations without reading release members."""
    if not 1 <= limit <= 100 or (
        corpus_id is not None and not _CORPUS.fullmatch(corpus_id)
    ):
        raise ValueError(
            "release catalog requires a valid corpus ID and a limit of 1–100"
        )
    cursor = release_locator(work_root, after) if after is not None else None
    page = nsmallest(
        limit + 1,
        (
            item
            for item in _published_locations(work_root, corpus_id)
            if cursor is None or item > cursor
        ),
    )
    rows = [
        {
            "corpus_id": corpus,
            "placement": area,
            "release_id": key if area == "releases" else None,
            "location": f"corpora/{corpus}/{area}/{key}",
        }
        for corpus, area, key in page[:limit]
    ]
    return {
        "releases": rows,
        "next_after": rows[-1]["location"] if len(page) > limit else None,
    }


def _read_metadata(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > _METADATA_BYTES:
            raise ValueError(
                f"{path.name} exceeds the release metadata limit or is not a regular file"
            )
        raw = stream.read(info.st_size + 1)
        if len(raw) != info.st_size:
            raise ValueError(f"{path.name} changed size during release inspection")
        return raw


def _control_member(directory: Path, files: dict, name: str):
    entry = files[name]
    if "storage" in entry:
        raise ValueError(f"release control member must be inline: {name}")
    raw = _read_metadata(directory / name)
    if len(raw) != entry["size"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"release control member changed: {name}")
    return json.loads(raw)


def release_profile(
    work_root: Path, reference: str, *, section="sources", offset=0, limit=20
):
    """Authenticate the named control files and page their recorded evidence."""
    if section not in RELEASE_SECTIONS or offset < 0 or not 1 <= limit <= 100:
        raise ValueError(
            "release detail requires a known section, a nonnegative offset and a limit of 1–100"
        )
    corpus, area, key = release_locator(work_root, reference)
    directory = work_root / "corpora" / corpus / area / key
    if symlinked(directory):
        raise ValueError("symlinked Corpus release location")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file(follow_symlinks=False):
        raise ValueError("published release metadata is missing or unsafe")
    generations = capture([manifest_path])
    raw_manifest = _read_metadata(manifest_path)
    manifest = json.loads(raw_manifest)
    release_id = manifest.pop("release_id")
    build_id, identity, files = (
        manifest["build_id"],
        manifest["build_identity"],
        manifest["files"],
    )
    if (
        manifest["schema_version"] != 1
        or manifest["corpus_id"] != corpus
        or release_id != canonical_digest(manifest)
        or build_id != canonical_digest(identity)
        or (area == "releases" and key != release_id)
        or (area == "builds" and (key != build_id or local_scope(manifest) is None))
    ):
        raise ValueError(
            "Corpus release metadata does not match its publication identity"
        )
    section_member = {
        "sources": "sources.json",
        "duplicate-decisions": "audit.json",
    }.get(section)
    control_names = ["report.json"]
    if section_member is not None and section_member in files:
        control_names.append(section_member)
    control_paths = [directory / name for name in control_names]
    if any(not path.is_file(follow_symlinks=False) for path in control_paths):
        raise ValueError("published release control file is missing or unsafe")
    generations.update(capture(control_paths))
    report = _control_member(directory, files, "report.json")
    selection = report.get("train_selection", {})
    inventories = {kind: selection.get(field, []) for field, kind in _SIDECARS.items()}
    annotation_counts = {
        kind: len(inventory) for kind, inventory in inventories.items()
    }
    if section == "sources":
        values = (
            _control_member(directory, files, "sources.json")
            if "sources.json" in files
            else []
        )
        count = len(values)
        rows = values[offset : offset + limit]
    elif section == "annotations":
        count = sum(annotation_counts.values())
        rows = []
        skipped = offset
        for kind, inventory in inventories.items():
            rows.extend(
                {"kind": kind, **row}
                for row in inventory[skipped : skipped + limit - len(rows)]
            )
            skipped = max(0, skipped - len(inventory))
    else:
        groups = (
            _control_member(directory, files, "audit.json")["duplicates"]
            if "audit.json" in files
            else []
        )
        count = len(groups)
        rows = [
            {
                **{
                    name: group[name]
                    for name in ("method", "sha256", "splits", "representative")
                    if name in group
                },
                "members": len(group["origins"]),
                "sample_origins": group["origins"][:5],
            }
            for group in groups[offset : offset + limit]
        ]
    unchanged(generations)
    return {
        "verification": "manifest_and_control_members",
        "verified_members": ["manifest.json", *control_names],
        "corpus_id": corpus,
        "release_id": release_id,
        "build_id": build_id,
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "declared_bytes": sum(entry["size"] for entry in files.values()),
        "snapshots": len(manifest["snapshots"]),
        "documents": report.get("document_count"),
        "kept_documents": report.get("kept_document_count"),
        "view_counts": report.get("view_counts", {}),
        "dedupe": report.get("dedupe", {}),
        "selection": {
            "excluded_by_reason": selection.get("excluded_by_reason", {}),
            "annotation_counts": annotation_counts,
        },
        "duplicate_evidence": report.get("duplicate_evidence"),
        "duplicate_policy": identity.get("release", {})
        .get("policy", {})
        .get("duplicates"),
        "native_stage_implementation": identity.get("native_stage_implementation", {}),
        "section": section,
        "rows": rows,
        "row_count": count,
        "next_offset": offset + len(rows) if offset + len(rows) < count else None,
    }
