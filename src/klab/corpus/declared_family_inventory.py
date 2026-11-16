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
from klab.corpus.mixture import MixtureDeclaration
from klab.corpus.release import verify_release
from klab.corpus.release_verify import range_records
from klab.corpus.shards import declare_stage, plan_ranges, run_sharded
from klab.corpus.split_freeze import _publish_new
from klab.training.manifest import sha256_file

DECLARED_UNITS = ("source_document_family", "source_repository")


INVENTORY_STAGE = declare_stage("family inventory")


def _inventory_range(
    path: str, start: int, end: int, strata: dict[str, str]
) -> tuple[bytes, int]:
    """The inventory lines of one byte range of documents, in file order."""
    lines = []
    for doc, _, _ in range_records(Path(path), start, end):
        if doc["drop_reason"] is not None:
            continue
        stratum = strata.get(doc["source_id"])
        if stratum is None:
            raise ValueError(f"source {doc['source_id']} has no stratum in the mixture")
        if stratum not in doc["domains"]:
            raise ValueError(
                f"source {doc['source_id']} declares domains {doc['domains']}, "
                f"which do not include its mixture stratum {stratum}"
            )
        lines.append(
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
    return b"".join(lines), len(lines)


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
        plan = plan_ranges(INVENTORY_STAGE, release / "documents.jsonl", strata)
        with payload.open("xb") as stream:
            for blob, count in run_sharded(plan, _inventory_range, ordered=True):
                stream.write(blob)
                kept += count
        _publish_new(output, payload)
    return {
        "schema_version": 1,
        "release_id": manifest["release_id"],
        "output": str(output),
        "sha256": sha256_file(output),
        "kept_documents": kept,
        "family_unit": split["unit"],
    }
