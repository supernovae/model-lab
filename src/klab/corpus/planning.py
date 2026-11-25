"""Read-only work estimates for native Corpus build partitions."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from klab.corpus.acquisition import declaration_sha256
from klab.corpus.bounded_metadata import read_inventory_metadata
from klab.corpus.large_build import native_file_rights
from klab.corpus.pipeline import uses_disk_lm_builder
from klab.corpus.prepared_shards import normalized_store_root, prepared_file_exists
from klab.corpus.project import (
    Project,
    SourceDeclaration,
    acquisition_identity_payload,
    safe_name,
)
from klab.safe_io import symlinked

_ELIGIBLE = {"eligible", "eligible_with_obligations"}
_RIGHTS_METADATA_BYTES = 4 * 1024**2
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _acquisition_inventory(
    project: Project, work_root: Path, lock_path: Path
) -> Iterator[tuple[SourceDeclaration, Path, str, list[dict[str, Any]]]]:
    lock = read_inventory_metadata(lock_path)
    if (
        lock["schema_version"] != 1
        or lock["project_id"] != project.config.id
        or set(lock["sources"]) != {source.id for source in project.sources}
    ):
        raise ValueError("acquisition source inventory does not match project")
    effects = {
        effect.source_id: effect for effect in project.config.source_effects or ()
    }
    for source in sorted(project.sources, key=lambda item: item.id):
        entry = lock["sources"][source.id]
        pin = entry["snapshot_sha256"]
        if not isinstance(pin, str) or _DIGEST.fullmatch(pin) is None:
            raise ValueError(f"invalid snapshot pin: {source.id}")
        expected = lock_path.parent / "snapshots" / source.id / pin
        snapshot = expected.resolve()
        effect = effects.get(source.id)
        if (
            entry["snapshot_path"] != str(snapshot)
            or snapshot.name != pin
            or not snapshot.is_relative_to(work_root / "corpora")
            or (effect is not None and symlinked(expected))
        ):
            raise ValueError(f"invalid snapshot path: {source.id}")
        origin = entry.get("reuse_origin")
        if effect is not None and effect.effect == "reuse_only":
            expected_origin = (
                work_root
                / "corpora"
                / effect.origin_project_id
                / "snapshots"
                / source.id
                / pin
            )
            if pin != effect.snapshot_sha256 or origin != {
                "project_id": effect.origin_project_id,
                "snapshot_sha256": pin,
                "declaration_sha256": effect.declaration_sha256,
                "path": str(expected_origin),
            }:
                raise ValueError(f"retained origin binding differs: {source.id}")
        elif origin is not None:
            raise ValueError(f"undeclared retained origin: {source.id}")
        manifest = read_inventory_metadata(snapshot / "manifest.json")
        if (
            manifest["schema_version"] != 1
            or manifest["source_id"] != source.id
            or manifest["snapshot_sha256"] != pin
            or manifest["adapter"]["id"] != source.kind
            or manifest["declaration_sha256"] != entry["declaration_sha256"]
            or entry["declaration_sha256"] != declaration_sha256(source)
            or manifest["declaration"] != acquisition_identity_payload(source)
        ):
            raise ValueError(f"snapshot metadata mismatch: {source.id}")
        yield source, snapshot, pin, manifest["files"]


def _source_plan(
    project: Project,
    source: SourceDeclaration,
    snapshot: Path,
    pin: str,
    inventory: list[dict[str, Any]],
    normalized_root: Path,
) -> dict[str, Any]:
    files_root = snapshot / "files"
    nested_path = source.rights.nested_metadata_path if source.rights else None
    nested_metadata = {}
    if nested_path is not None:
        metadata = files_root / safe_name(nested_path)
        if symlinked(metadata) or not metadata.is_file():
            raise ValueError(f"unsafe rights metadata: {source.id}/{nested_path}")
        with metadata.open("rb") as stream:
            raw = stream.read(_RIGHTS_METADATA_BYTES + 1)
        if len(raw) > _RIGHTS_METADATA_BYTES:
            raise ValueError(f"rights metadata header cap exceeded: {source.id}")
        nested_metadata[nested_path] = json.loads(raw)

    if symlinked(normalized_root / source.id):
        raise ValueError(f"symlinked normalized namespace: {source.id}")
    partitions = []
    seen = set()
    for file in sorted(inventory, key=lambda item: item["path"]):
        name = safe_name(file["path"])
        size = file["size"]
        digest = file["sha256"]
        if (
            name in seen
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or _DIGEST.fullmatch(digest) is None
        ):
            raise ValueError(f"invalid snapshot inventory row: {source.id}/{name}")
        seen.add(name)
        path = files_root / name
        if symlinked(path) or not path.is_file() or path.stat().st_size != size:
            raise ValueError(
                f"snapshot file missing or changed size: {source.id}/{name}"
            )
        if name in nested_metadata:
            action, reason = "skip", "rights metadata"
        else:
            decision = native_file_rights(project, source, name, path, nested_metadata)
            if decision is not None and decision.training_eligibility not in _ELIGIBLE:
                action, reason = "skip", f"rights {decision.training_eligibility}"
            elif prepared_file_exists(
                normalized_root,
                source,
                pin,
                file,
                project.release.normalizer,
                project.release.policy.documents,
                decision,
                None,
            ):
                action, reason = (
                    "verify_candidate",
                    "matching normalized cache location",
                )
            else:
                action, reason = "normalize", "no matching normalized cache location"
        partitions.append(
            {"path": name, "source_bytes": size, "action": action, "reason": reason}
        )
    return {"source_id": source.id, "snapshot_sha256": pin, "partitions": partitions}


def plan_build(project: Project, work_root: Path) -> dict[str, Any]:
    """Plan full native normalization using acquisition inventory and cache hints."""
    lock = work_root / "corpora" / project.config.id / "acquisition.json"
    normalized_root = normalized_store_root(work_root)
    plan: dict[str, Any] = {
        "project_id": project.config.id,
        "verification": "advisory",
        "stages": [
            {"id": "acquire", "inputs": []},
            {"id": "normalize", "inputs": ["acquire"]},
            {"id": "index", "inputs": ["normalize"]},
            {"id": "deduplicate", "inputs": ["index"]},
            {"id": "views", "inputs": ["index", "deduplicate"]},
            {"id": "freeze", "inputs": ["views"]},
        ],
        "sources": [],
        "normalization_source_bytes": None,
        "verification_candidate_source_bytes": None,
    }
    if not uses_disk_lm_builder(project, list(project.transforms)):
        return {
            **plan,
            "status": "unsupported",
            "reason": "build uses the in-memory builder",
        }
    if project.release.record_admission is not None:
        return {
            **plan,
            "status": "unsupported",
            "reason": "build requires reviewed row admission",
        }
    if not lock.exists() and not symlinked(lock):
        return {
            **plan,
            "status": "acquisition_required",
            "reason": "acquisition lock missing",
        }
    try:
        sources = [
            _source_plan(project, source, snapshot, pin, files, normalized_root)
            for source, snapshot, pin, files in _acquisition_inventory(
                project, work_root, lock
            )
        ]
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {**plan, "status": "blocked", "reason": str(error)}
    source_bytes = {"normalize": 0, "verify_candidate": 0, "skip": 0}
    for source_plan in sources:
        for partition in source_plan["partitions"]:
            source_bytes[partition["action"]] += partition["source_bytes"]
    return {
        **plan,
        "status": "planned",
        "sources": sources,
        "normalization_source_bytes": source_bytes["normalize"],
        "verification_candidate_source_bytes": source_bytes["verify_candidate"],
    }
