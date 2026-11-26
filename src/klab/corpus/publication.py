"""Reconstruction-only metadata, never an implicit redistribution of source text."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from klab.corpus.project import SourceDeclaration, acquisition_identity_payload
from klab.corpus.release import ReleaseReader, open_release
from klab.training.manifest import sha256_file


def publication_manifest(release: Path) -> dict[str, Any]:
    """Verify a prospective release and describe its policy without copying text.

    Training exports remain private local artifacts. This manifest publishes no
    snapshots, normalized passages, derived examples, tokenizer or model weights.
    Those artifacts require their own explicit review before publication.
    """
    with open_release(release) as verified:
        return _publication_manifest(verified)


def _publication_manifest(release: ReleaseReader) -> dict[str, Any]:
    manifest = release.manifest
    identity = manifest["build_identity"]
    rights = json.loads(
        release.member("license-report.json").read_text(encoding="utf-8")
    )
    if rights.get("schema_version") != 1 or "release_policy" not in rights:
        raise ValueError("release has no recorded publication policy")
    files = defaultdict(list)
    for item in rights["files"]:
        files[item["source_id"]].append(
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "role": item["role"],
                **({"rights": item["rights"]} if "rights" in item else {}),
            }
        )
    sources = []
    for source in rights["sources"]:
        snapshot_id = source["snapshot_sha256"]
        acquisition = None
        if snapshot_id:
            # Release verification binds the snapshot acquisition to this declaration.
            declaration = SourceDeclaration.model_validate(
                identity["sources"][source["id"]]
            )
            acquisition = acquisition_identity_payload(declaration)["acquisition"]
            if source["kind"] == "local":
                # A local absolute path may reveal private machine state and is
                # not a portable reconstruction instruction.
                acquisition = {
                    "file_names": [item["name"] for item in acquisition["files"]]
                }
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
                    if "explicit_training_restriction" in source
                    else {}
                ),
                "snapshot_sha256": snapshot_id,
                "acquisition": acquisition,
                "files": files.get(source["id"], []),
            }
        )
    return {
        "schema_version": 1,
        "release_id": manifest["release_id"],
        "release_manifest_sha256": sha256_file(release.root / "manifest.json"),
        "rights_report_sha256": manifest["files"]["license-report.json"]["sha256"],
        "publication_mode": rights["publication_mode"],
        "release_policy": rights["release_policy"],
        "weight_license_status": rights["weight_license_status"],
        "sources": sources,
        "project": identity["project"],
        "transforms": [
            {"id": item["id"], "kind": item["kind"], "version": item["version"]}
            for item in identity["transforms"]
        ],
        "contents": "metadata_only; reconstruct source-backed content from pinned upstream",
    }
