"""Reconstruction-only metadata, never an implicit redistribution of source text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from klab.corpus.release import verify_release
from klab.training.manifest import sha256_file


def publication_manifest(release: Path) -> dict[str, Any]:
    """Verify a prospective release and describe its policy without copying text.

    Training exports remain private local artifacts. This manifest publishes no
    snapshots, normalized passages, derived examples, tokenizer or model weights.
    Those artifacts require their own explicit review before publication.
    """
    release = Path(release)
    manifest = verify_release(release)
    rights = json.loads((release / "license-report.json").read_text(encoding="utf-8"))
    if rights.get("schema_version") not in (2, 3):
        raise ValueError("historical release has no prospective publication policy")
    sources = []
    for source in rights["sources"]:
        snapshot_id = source["snapshot_sha256"]
        declaration = (
            json.loads(
                (
                    release.parent.parent
                    / "snapshots"
                    / source["id"]
                    / snapshot_id
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )["declaration"]
            if snapshot_id
            else None
        )
        if declaration is not None:
            acquisition = declaration["acquisition"]
            if source["kind"] == "local":
                # A local absolute path may reveal private machine state and is
                # not a portable reconstruction instruction.
                acquisition = {
                    "file_names": [item["name"] for item in acquisition["files"]]
                }
        else:
            acquisition = None
        files = [
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "role": item["role"],
                **({"rights": item["rights"]} if "rights" in item else {}),
            }
            for item in rights["files"]
            if item["source_id"] == source["id"]
        ]
        sources.append(
            {
                "id": source["id"],
                "canonical_uri": source["canonical_uri"],
                "revision": source["revision"],
                "license": source["license"],
                "license_url": source["license_url"],
                "rights_policy": source["rights_policy"],
                **(
                    {
                        "explicit_training_restriction": source[
                            "explicit_training_restriction"
                        ]
                    }
                    if rights["schema_version"] == 3
                    else {}
                ),
                "snapshot_sha256": snapshot_id,
                "acquisition": acquisition,
                "files": files,
            }
        )
    return {
        "schema_version": 1,
        "release_id": manifest["release_id"],
        "release_manifest_sha256": sha256_file(release / "manifest.json"),
        "rights_report_sha256": sha256_file(release / "license-report.json"),
        "publication_mode": rights["publication_mode"],
        **(
            {"training_use_policy": rights["training_use_policy"]}
            if rights["schema_version"] == 3
            else {}
        ),
        "weight_license_status": rights["weight_license_status"],
        "sources": sources,
        "project": manifest["build_identity"]["project"],
        "transforms": [
            {"id": item["id"], "kind": item["kind"], "version": item["version"]}
            for item in manifest["build_identity"]["transforms"]
        ],
        "contents": "metadata_only; reconstruct source-backed content from pinned upstream",
    }
