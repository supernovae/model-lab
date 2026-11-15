"""Derive a mixture family inventory from a release built with declared splits.

A release split by `source_document_family` or `source_repository` assigns whole
source families to splits in the project's own declaration; there is no reviewed
family freeze to export. The split unit is then the only family the release
knows, so the inventory names exactly that unit: a document's family is its
`source_family`, and the mixture's family-isolation check re-verifies that no
family spans splits. The mixture ranks documents, not families, so a family as
coarse as a source does not change which documents a stratum quota takes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import yaml

from klab.canonical import canonical_json
from klab.corpus.jsonl_records import iter_rows
from klab.corpus.mixture import MixtureDeclaration
from klab.corpus.release import verify_release
from klab.corpus.split_freeze import _publish_new
from klab.training.manifest import sha256_file

DECLARED_UNITS = ("source_document_family", "source_repository")


def derive_family_inventory(declaration: Path) -> dict[str, Any]:
    """Write the mixture's `family_inventory` from its verified release."""
    spec = MixtureDeclaration.model_validate(
        yaml.safe_load(declaration.read_text(encoding="utf-8"))
    )
    release = spec.release_path.resolve()
    output = spec.family_inventory.resolve()
    manifest = verify_release(release)
    split = manifest["build_identity"]["split"]
    if split["unit"] not in DECLARED_UNITS or split.get("assignment_file"):
        raise ValueError(
            "release was not built with declared source-family splits; use "
            "finalize-family-inventory with its frozen splits"
        )
    if output.exists() or output.is_symlink():
        raise ValueError("family inventory already exists")
    strata = spec.source_strata
    output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".family-") as scratch:
        payload = Path(scratch) / "family-inventory.jsonl"
        with payload.open("xb") as stream:
            for doc in iter_rows(release / "documents.jsonl"):
                if doc["drop_reason"] is not None:
                    continue
                stratum = strata.get(doc["source_id"])
                if stratum is None:
                    raise ValueError(
                        f"source {doc['source_id']} has no stratum in the mixture"
                    )
                if stratum not in doc["domains"]:
                    raise ValueError(
                        f"source {doc['source_id']} declares domains {doc['domains']}, "
                        f"which do not include its mixture stratum {stratum}"
                    )
                stream.write(
                    canonical_json(
                        {
                            "document_id": doc["document_id"],
                            "family_id": doc["source_family"],
                            "split": doc["split"],
                            "stratum": stratum,
                            "content_sha256": doc["content_sha256"],
                        }
                    )
                    + b"\n"
                )
                kept += 1
        _publish_new(output, payload)
    return {
        "schema_version": 1,
        "release_id": manifest["release_id"],
        "output": str(output),
        "sha256": sha256_file(output),
        "kept_documents": kept,
        "family_unit": split["unit"],
    }
