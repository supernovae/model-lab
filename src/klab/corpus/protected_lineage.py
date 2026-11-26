"""Read-only lineage check for frozen evaluation documents across corpus releases."""

from __future__ import annotations

import hashlib
import json
from itertools import chain
from pathlib import Path
from typing import Any

from klab.corpus.family_inventory_rows import covered_family_rows
from klab.corpus.release import ReleaseReader, open_release
from klab.corpus.shards import range_file_identity
from klab.result_publication import publish_immutable
from klab.training.manifest import sha256_file


def audit_protected_lineage(
    prior_release: Path,
    candidate_release: Path,
    prior_inventory: Path,
    candidate_inventory: Path,
    profile_path: Path,
    suite_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Bind prior protected IDs/content to candidate train exclusions and assignments."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"protected-lineage output already exists: {output}")
    with (
        open_release(prior_release) as prior,
        open_release(candidate_release) as candidate,
    ):
        receipt = _audit(
            prior,
            candidate,
            prior_inventory,
            candidate_inventory,
            profile_path,
            suite_path,
        )
    publish_immutable(output, receipt)
    return receipt


def _audit(
    prior: ReleaseReader,
    candidate: ReleaseReader,
    prior_inventory: Path,
    candidate_inventory: Path,
    profile_path: Path,
    suite_path: Path,
) -> dict[str, Any]:
    profile_bytes = profile_path.read_bytes()
    suite_bytes = suite_path.read_bytes()
    profile = json.loads(profile_bytes)
    suite = json.loads(suite_bytes)
    prior_manifest = prior.manifest
    candidate_manifest = candidate.manifest
    prior_manifest_sha256 = sha256_file(prior.root / "manifest.json")
    prior_generation = range_file_identity(prior_inventory)
    candidate_generation = range_file_identity(candidate_inventory)
    prior_inventory_sha256 = sha256_file(prior_inventory)
    if (
        profile.get("release_id") != prior_manifest["release_id"]
        or profile.get("release_manifest_sha256") != prior_manifest_sha256
        or profile.get("documents_sha256")
        != prior_manifest["files"]["documents.jsonl"]["sha256"]
        or suite.get("release_id") != prior_manifest["release_id"]
        or suite.get("family_inventory_sha256") != prior_inventory_sha256
    ):
        raise ValueError("protected profile or suite release binding mismatch")
    protected_ids = {
        row["document_id"] for row in chain(profile["loss_slices"], suite["chunks"])
    }
    for item in suite["items"]:
        protected_ids.update(item["parent_document_ids"] or ())
    protected = {
        row["document_id"]: row
        for row in covered_family_rows(prior, prior_inventory)
        if row["document_id"] in protected_ids
    }
    _check_protected_documents(protected, profile, suite)
    protected_families = {row["family_id"] for row in protected.values()}
    protected_content = {row["content_sha256"] for row in protected.values()}
    family_collisions: set[str] = set()
    content_collisions: set[str] = set()
    assignments: dict[str, tuple[str, str, str]] = {}
    multiple_assignments: set[str] = set()
    for row in covered_family_rows(candidate, candidate_inventory):
        family, content, split = row["family_id"], row["content_sha256"], row["split"]
        if split == "train" and family in protected_families:
            family_collisions.add(family)
        if content in protected_content:
            if split == "train":
                content_collisions.add(content)
            assignment = (family, split, row["stratum"])
            if assignments.setdefault(content, assignment) != assignment:
                multiple_assignments.add(content)
    changed = sorted(
        key
        for key, row in protected.items()
        if (content := row["content_sha256"]) in assignments
        and (
            content in multiple_assignments
            or assignments[content] != (row["family_id"], row["split"], row["stratum"])
        )
    )
    candidate_inventory_sha256 = sha256_file(candidate_inventory)
    if (
        range_file_identity(prior_inventory) != prior_generation
        or range_file_identity(candidate_inventory) != candidate_generation
    ):
        raise ValueError("family inventory changed during protected-lineage audit")
    return {
        "format": "klab-protected-lineage-v1",
        "prior_release_id": prior_manifest["release_id"],
        "candidate_release_id": candidate_manifest["release_id"],
        "prior_manifest_sha256": prior_manifest_sha256,
        "candidate_manifest_sha256": sha256_file(candidate.root / "manifest.json"),
        "prior_inventory_sha256": prior_inventory_sha256,
        "candidate_inventory_sha256": candidate_inventory_sha256,
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "suite_file_sha256": hashlib.sha256(suite_bytes).hexdigest(),
        "suite_content_sha256": suite.get("content_sha256"),
        "protected_document_ids": len(protected),
        "protected_families": len(protected_families),
        "protected_content_hashes": len(protected_content),
        "retained_protected_content_hashes": len(assignments),
        "new_train_family_collisions": sorted(family_collisions),
        "new_train_content_collisions": sorted(content_collisions),
        "changed_retained_protected_assignments": changed,
        "status": "BLOCKED"
        if family_collisions or content_collisions or changed
        else "PASS",
    }


def _check_protected_documents(
    prior: dict[str, dict[str, str]], profile: dict[str, Any], suite: dict[str, Any]
) -> None:
    slices, items, chunks = profile["loss_slices"], suite["items"], suite["chunks"]
    if not slices or not items or not chunks:
        raise ValueError("protected profile or suite is empty")
    for row in slices:
        doc = prior.get(row["document_id"])
        if (
            doc is None
            or row["split"] not in {"validation", "test"}
            or any(
                row[key] != doc[key] for key in ("content_sha256", "family_id", "split")
            )
        ):
            raise ValueError("fixed profile differs from prior family inventory")
    for item in items:
        if item["split"] != "test":
            raise ValueError("frozen suite item is not held out")
        parents = item["parent_document_ids"]
        if not parents or any(doc_id not in prior for doc_id in parents):
            raise ValueError("frozen suite has unknown parent document")
        families = {prior[doc_id]["family_id"] for doc_id in parents}
        declared = item["parent_family"]
        if set(declared if isinstance(declared, list) else [declared]) != families:
            raise ValueError("frozen suite parent family mismatch")
    for chunk in chunks:
        if chunk["document_id"] not in prior:
            raise ValueError("frozen suite chunk has unknown document")
