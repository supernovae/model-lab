"""Read-only lineage check for frozen evaluation documents across corpus releases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from klab.campaign.state import publish_immutable
from klab.corpus.jsonl_records import records_from_path
from klab.corpus.release import verify_release
from klab.training.manifest import sha256_file


def _inventory(path: Path, release: Path) -> dict[str, dict[str, str]]:
    docs = {
        row.value["document_id"]: row.value
        for row in records_from_path(release / "documents.jsonl")
        if row.value["drop_reason"] is None
    }
    result: dict[str, dict[str, str]] = {}
    for parsed in records_from_path(path):
        row = parsed.value
        document_id = row["document_id"]
        doc = docs.get(document_id)
        if (
            document_id in result
            or doc is None
            or set(row)
            != {"document_id", "family_id", "split", "stratum", "content_sha256"}
            or row["split"] != doc["split"]
            or row["content_sha256"] != doc["content_sha256"]
            or row["stratum"] not in doc["domains"]
        ):
            raise ValueError("family inventory disagrees with verified release")
        result[document_id] = row
    if set(result) != set(docs):
        raise ValueError("family inventory does not cover kept release documents")
    return result


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
    prior_manifest = verify_release(prior_release)
    candidate_manifest = verify_release(candidate_release)
    prior = _inventory(prior_inventory, prior_release)
    candidate = _inventory(candidate_inventory, candidate_release)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if (
        profile.get("release_id") != prior_manifest["release_id"]
        or profile.get("release_manifest_sha256")
        != sha256_file(prior_release / "manifest.json")
        or profile.get("documents_sha256")
        != prior_manifest["files"]["documents.jsonl"]["sha256"]
        or suite.get("release_id") != prior_manifest["release_id"]
        or suite.get("family_inventory_sha256") != sha256_file(prior_inventory)
    ):
        raise ValueError("protected profile or suite release binding mismatch")
    slices = profile["loss_slices"]
    items = suite["items"]
    chunks = suite["chunks"]
    if not slices or not items or not chunks:
        raise ValueError("protected profile or suite is empty")
    protected: set[str] = set()
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
        protected.add(row["document_id"])
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
        protected.update(parents)
    for chunk in chunks:
        if chunk["document_id"] not in prior:
            raise ValueError("frozen suite chunk has unknown document")
        protected.add(chunk["document_id"])
    protected_families = {prior[key]["family_id"] for key in protected}
    protected_content = {prior[key]["content_sha256"] for key in protected}
    train = [row for row in candidate.values() if row["split"] == "train"]
    family_collisions = sorted(protected_families & {row["family_id"] for row in train})
    content_collisions = sorted(
        protected_content & {row["content_sha256"] for row in train}
    )
    candidate_by_content: dict[str, set[tuple[str, str, str]]] = {}
    for row in candidate.values():
        candidate_by_content.setdefault(row["content_sha256"], set()).add(
            (row["family_id"], row["split"], row["stratum"])
        )
    changed = sorted(
        key
        for key in protected
        if prior[key]["content_sha256"] in candidate_by_content
        and candidate_by_content[prior[key]["content_sha256"]]
        != {(prior[key]["family_id"], prior[key]["split"], prior[key]["stratum"])}
    )
    receipt = {
        "format": "klab-protected-lineage-v1",
        "prior_release_id": prior_manifest["release_id"],
        "candidate_release_id": candidate_manifest["release_id"],
        "prior_manifest_sha256": sha256_file(prior_release / "manifest.json"),
        "candidate_manifest_sha256": sha256_file(candidate_release / "manifest.json"),
        "prior_inventory_sha256": sha256_file(prior_inventory),
        "candidate_inventory_sha256": sha256_file(candidate_inventory),
        "profile_sha256": sha256_file(profile_path),
        "suite_file_sha256": sha256_file(suite_path),
        "suite_content_sha256": suite.get("content_sha256"),
        "protected_document_ids": len(protected),
        "protected_families": len(protected_families),
        "protected_content_hashes": len(protected_content),
        "retained_protected_content_hashes": len(
            protected_content & candidate_by_content.keys()
        ),
        "new_train_family_collisions": family_collisions,
        "new_train_content_collisions": content_collisions,
        "changed_retained_protected_assignments": changed,
        "status": "BLOCKED"
        if family_collisions or content_collisions or changed
        else "PASS",
    }
    publish_immutable(output, receipt)
    return receipt
