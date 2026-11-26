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

import shutil
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.build_stages import implementation_digests
from klab.corpus.inventory_writer import inventory_writer
from klab.corpus.mixture import MixtureDeclaration
from klab.corpus.release import ReleaseReader, open_release
from klab.corpus.release_verify import current_settings, range_records
from klab.corpus.shards import (
    declare_stage,
    plan_ranges,
    range_file_identity,
    run_sharded_files,
)

DECLARED_UNITS = ("source_document_family", "source_repository")


INVENTORY_STAGE = declare_stage("family inventory")


def _inventory_range(
    staging: Path, path: str, start: int, end: int, strata: dict[str, str]
) -> dict[str, Any]:
    """Write one inventory segment in document order, without buffering its rows."""
    kept = 0
    with (staging / "inventory.jsonl").open("xb") as stream:
        for doc, _, _ in range_records(Path(path), start, end):
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
    return {"kept_documents": kept}


def derive_family_inventory(declaration: Path) -> dict[str, Any]:
    """Write the mixture's `family_inventory` from its verified release."""
    spec = MixtureDeclaration.model_validate(
        yaml.safe_load(declaration.read_text(encoding="utf-8"))
    )
    with open_release(spec.release_path) as release:
        return _write_inventory(spec, release)


def _write_inventory(
    spec: MixtureDeclaration, release: ReleaseReader
) -> dict[str, Any]:
    documents = release.member("documents.jsonl")
    verified_input = range_file_identity(documents)
    output = spec.family_inventory.resolve()
    manifest = release.manifest
    split = manifest["build_identity"]["split"]
    if split["unit"] not in DECLARED_UNITS or split.get("assignment_file"):
        raise ValueError(
            "release was not built with declared source-family splits; use "
            "finalize-family-inventory with its frozen splits"
        )
    if output.exists() or output.is_symlink():
        raise ValueError("family inventory already exists")
    strata = spec.source_strata
    plan = plan_ranges(
        INVENTORY_STAGE,
        documents,
        strata,
        inputs={
            "release": manifest["release_id"],
            "producer": implementation_digests(
                "declared_family_inventory.py", "release_verify.py"
            ),
        },
    )
    if plan.inputs["file"] != verified_input:
        raise ValueError("release documents changed during family inventory")
    identity = canonical_digest(
        {
            "inputs": plan.inputs,
            "assembler": implementation_digests(
                "inventory_writer.py", "view_checkpoints.py"
            ),
            "strata": strata,
            "ranges": [(start, end) for _, start, end, _ in plan.tasks],
        }
    )
    with inventory_writer(output, identity) as (checkpoints, view, state):
        completed = state.get("ranges", 0)
        kept = state.get("kept", 0)
        echo = current_settings().echo
        if completed and echo is not None:
            print(
                f"family inventory: reused {completed}/{len(plan.tasks)} assembled ranges",
                file=echo,
            )
        tasks = plan.tasks[completed:]
        disposable = plan.store is None
        tail = replace(
            plan,
            tasks=tasks,
            total_bytes=sum(end - start for _, start, end, _ in tasks),
            store=checkpoints.root / "shards" if disposable else plan.store,
        )
        with closing(
            run_sharded_files(tail, _inventory_range, ordered=True)
        ) as segments:
            for number, (directory, result) in enumerate(segments, start=completed + 1):
                with (directory / "inventory.jsonl").open("rb") as segment:
                    while block := segment.read(1024**2):
                        view.write(block)
                kept += result["kept_documents"]
                checkpoints.commit({"ranges": number, "kept": kept})
                if disposable:
                    shutil.rmtree(directory)
        if not plan.tasks:
            checkpoints.commit({"ranges": 0, "kept": 0})
        if range_file_identity(documents) != verified_input:
            raise ValueError("release documents changed during family inventory")
        digest = view.file_digest().sha256
        release.require_unchanged()
    return {
        "schema_version": 1,
        "release_id": manifest["release_id"],
        "output": str(output),
        "sha256": digest,
        "kept_documents": kept,
        "family_unit": split["unit"],
    }
