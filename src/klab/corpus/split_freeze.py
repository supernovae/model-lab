"""Freeze reviewed, admission-aware document-family splits from native inventory."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import yaml

from klab.canonical import canonical_json
from klab.corpus.acquisition import verify_acquisition
from klab.corpus.bounded_metadata import (
    current_metadata_store,
    metadata_operation,
    read_metadata,
)
from klab.corpus.jsonl_records import records_from_path
from klab.corpus.project import Project, project_path, source_declaration_payload
from klab.corpus.release import verify_release
from klab.corpus.rights import verify_record_admission
from klab.corpus.row_index import RowIndex, ValueMap
from klab.corpus.split_inventory import write_split_inventory
from klab.corpus.split_inventory_cache import InventoryCache, verified_inventory
from klab.result_publication import sync_directory as _sync_directory
from klab.training.manifest import sha256_file
from klab.verification_proofs import verification_options

_SPLITS = ("train", "validation", "test")
_WEIGHTS = (8, 1, 1)
_FREEZE_MARKER = b"# klab frozen split bundle; requires matching receipt\n"


def _sync_parents(path: Path, root: Path) -> None:
    # Recovery cannot know which directories a failed publisher created. Re-sync
    # their entries through the existing work root, never unrelated ancestors.
    path = path.resolve()
    root = root.resolve()
    path.relative_to(root)
    while True:
        _sync_directory(path)
        if path == root:
            break
        path = path.parent


def verify_split_bundle(
    output: Path, *, required: bool = True
) -> dict[str, Any] | None:
    """Read-only commit check; legacy complete schema-1 receipts remain valid."""
    candidate = output.with_name(output.stem + ".family-candidates.jsonl")
    receipt_path = output.with_name(output.stem + ".receipt.json")
    try:
        with output.open("rb") as stream:
            payload = stream.read(4 * 1024**2 + 1)
        if len(payload) > 4 * 1024**2:
            raise ValueError("split declaration header byte cap exceeded")
        if not required and not (
            payload.startswith(_FREEZE_MARKER)
            or candidate.exists()
            or candidate.is_symlink()
            or receipt_path.exists()
            or receipt_path.is_symlink()
        ):
            return None
        if any(
            path.is_symlink() or not path.is_file()
            for path in (output, candidate, receipt_path)
        ):
            raise ValueError("missing or nonregular bundle member")
        with receipt_path.open("rb") as stream:
            receipt_bytes = stream.read(4 * 1024**2 + 1)
        if len(receipt_bytes) > 4 * 1024**2:
            raise ValueError("split receipt header byte cap exceeded")
        receipt = json.loads(receipt_bytes)
        if (
            not isinstance(receipt, dict)
            or not {
                "schema_version",
                "split_sha256",
                "family_candidate_sha256",
                "inventory_sha256",
                "admission_sha256",
                "cluster_sha256",
                "family_counts",
            }.issubset(receipt)
            or receipt.get("schema_version") != 1
            or receipt.get("split_sha256") != hashlib.sha256(payload).hexdigest()
            or receipt.get("family_candidate_sha256") != sha256_file(candidate)
        ):
            raise ValueError("receipt digest mismatch")
    except (OSError, ValueError) as exc:
        raise ValueError("incomplete or mismatched frozen split bundle") from exc
    from klab.corpus.project import SplitDeclaration
    from klab.corpus.split_assignments import assignment_path

    declaration = SplitDeclaration.model_validate(yaml.safe_load(payload))
    if declaration.assignment_file is not None:
        assignment_path(
            output.parent, declaration.assignment_file.model_dump(mode="json")
        )
    return receipt


def _split_weights(value: object) -> tuple[int, int, int]:
    if (
        not isinstance(value, dict)
        or set(value) != set(_SPLITS)
        or any(type(value[key]) is not int or value[key] <= 0 for key in _SPLITS)
        or sum(value.values()) > 10000
    ):
        raise ValueError(
            "split_weights requires positive integer train/validation/test weights totaling at most 10000"
        )
    return tuple(value[key] for key in _SPLITS)


def _counts(
    total: int, weights: tuple[int, int, int] = _WEIGHTS
) -> tuple[int, int, int]:
    denominator = sum(weights)
    floors = [total * weight // denominator for weight in weights]
    remainders = [total * weight % denominator for weight in weights]
    for index in sorted(range(3), key=lambda i: (-remainders[i], i))[
        : total - sum(floors)
    ]:
        floors[index] += 1
    return floors[0], floors[1], floors[2]


def _publish_new(
    path: Path, data: bytes | Path, *, work_root: Path | None = None
) -> None:
    sync_root = work_root
    if sync_root is None:
        # Standalone family-inventory publication: persist every directory we
        # create, stopping at the first already-existing ancestor.
        sync_root = path.parent
        while not sync_root.exists():
            sync_root = sync_root.parent
    path.parent.resolve().relative_to(sync_root.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=".freeze-splits-", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            if isinstance(data, Path):
                with data.open("rb") as source:
                    shutil.copyfileobj(source, stream)
            else:
                stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
            _sync_parents(path.parent, sync_root)
        finally:
            temporary.unlink(missing_ok=True)


def _prior_family_splits(
    binding: object, root: Path
) -> tuple[dict[str, str], dict[str, str], dict[str, tuple[str, str]]]:
    """Authenticate a previous release's kept family assignments before extending it."""
    if not isinstance(binding, dict) or set(binding) != {
        "path",
        "sha256",
        "release_path",
    }:
        raise ValueError("invalid prior family-inventory binding")
    prior_path = Path(binding["path"]).resolve()
    release = Path(binding["release_path"]).resolve()
    if (
        not prior_path.is_relative_to(root)
        or not release.is_relative_to(root)
        or prior_path.is_symlink()
        or release.is_symlink()
        or sha256_file(prior_path) != binding["sha256"]
    ):
        raise ValueError("prior family inventory path or digest mismatch")
    verify_release(release, **verification_options(root))
    connection = current_metadata_store().connection
    kept = RowIndex(
        connection,
        (
            record.value
            for record in records_from_path(release / "documents.jsonl")
            if record.value["drop_reason"] is None
        ),
        lambda row: row["document_id"],
    )
    families = ValueMap(connection)
    strata = ValueMap(connection)
    content = ValueMap(connection)
    seen = ValueMap(connection)
    for record in records_from_path(prior_path, max_record_bytes=8 * 1024**2):
        row = record.value
        if (
            not isinstance(row, dict)
            or set(row)
            != {"document_id", "family_id", "split", "stratum", "content_sha256"}
            or row["document_id"] in seen
            or row["split"] not in _SPLITS
            or not isinstance(row["family_id"], str)
            or not row["family_id"]
        ):
            raise ValueError("invalid prior family inventory row")
        seen[row["document_id"]] = True
        doc = kept.get(row["document_id"])
        if (
            doc is None
            or doc["split"] != row["split"]
            or doc["content_sha256"] != row["content_sha256"]
            or row["stratum"] not in doc["domains"]
        ):
            raise ValueError("prior family inventory differs from release")
        family = row["family_id"]
        if family in families and (
            families[family] != row["split"] or strata[family] != row["stratum"]
        ):
            raise ValueError("prior family spans split or stratum")
        families[family] = row["split"]
        strata[family] = row["stratum"]
        previous = content.setdefault(row["content_sha256"], (family, row["split"]))
        if previous != (family, row["split"]):
            raise ValueError("prior content spans families or splits")
    if not seen.same_keys(kept):
        raise ValueError("prior family inventory does not cover kept release")
    return families, strata, content


def _split_inputs(
    project: Project,
    work_root: Path,
    inventory_path: Path,
    clusters_path: Path,
    output: Path,
    *,
    recover: bool = False,
    load_prior: bool = True,
) -> dict[str, Any]:
    """Assign eligible document families; excluded rows get held-out placeholders."""
    root = work_root.resolve()
    if any(
        part.is_symlink()
        for path in (inventory_path, clusters_path, output)
        for part in (path, *path.parents)
    ):
        raise ValueError("symlink in split input or output")
    paths = [inventory_path.resolve(), clusters_path.resolve(), output.resolve()]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("split inputs and output must reside in the work root")
    if not recover and (output.exists() or output.is_symlink()):
        raise ValueError("frozen split output already exists")
    from klab.corpus.admission_inspection_cache import capture, unchanged

    configuration = capture([clusters_path])
    clusters_sha256 = sha256_file(clusters_path)
    raw_clusters = read_metadata(clusters_path)
    unchanged(configuration)
    if (
        not isinstance(raw_clusters, dict)
        or not {
            "schema_version",
            "inventory_sha256",
            "seed",
            "source_strata",
            "merges",
            "reviewer",
            "reviewed_on",
        }.issubset(raw_clusters)
        or set(raw_clusters)
        - {
            "schema_version",
            "inventory_sha256",
            "seed",
            "source_strata",
            "merges",
            "reviewer",
            "reviewed_on",
            "prior_family_inventory",
            "split_weights",
            "inventory_completion",
            "merge_provenance",
        }
        or raw_clusters["schema_version"] != 1
        or not isinstance(raw_clusters["seed"], str)
        or not raw_clusters["seed"].strip()
        or not isinstance(raw_clusters["reviewer"], str)
        or not raw_clusters["reviewer"].strip()
        or not isinstance(raw_clusters["reviewed_on"], str)
        or not raw_clusters["reviewed_on"].strip()
        or not isinstance(raw_clusters["inventory_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", raw_clusters["inventory_sha256"]) is None
        or not isinstance(raw_clusters["source_strata"], dict)
        or set(raw_clusters["source_strata"])
        != {source.id for source in project.sources}
        or any(
            not isinstance(stratum, str)
            or re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", stratum) is None
            for stratum in raw_clusters["source_strata"].values()
        )
        or not isinstance(raw_clusters["merges"], list)
    ):
        raise ValueError("invalid reviewed family-cluster declaration")
    weights = (
        _split_weights(raw_clusters["split_weights"])
        if "split_weights" in raw_clusters
        else _WEIGHTS
    )
    if "merge_provenance" in raw_clusters:
        from klab.corpus.split_merge_provenance import (
            check_reviewed_merge_provenance,
        )

        configuration.update(
            check_reviewed_merge_provenance(
                project,
                root,
                inventory_path,
                raw_clusters,
                raw_clusters["merge_provenance"],
            )
        )
    prior_families, prior_strata, prior_content = (
        _prior_family_splits(raw_clusters["prior_family_inventory"], root)
        if load_prior and "prior_family_inventory" in raw_clusters
        else ({}, {}, {})
    )
    adopted = None
    if "inventory_completion" in raw_clusters:
        from klab.corpus.split_inventory_adoption import adopt_completed_inventory

        adopted = adopt_completed_inventory(
            project, root, inventory_path, raw_clusters["inventory_completion"]
        )
    return (
        root,
        configuration,
        raw_clusters,
        weights,
        clusters_sha256,
        prior_families,
        prior_strata,
        prior_content,
        adopted,
    )


def _freeze_splits(
    project: Project,
    work_root: Path,
    inventory_path: Path,
    clusters_path: Path,
    output: Path,
    *,
    recover: bool = False,
) -> dict[str, Any]:
    """Assign eligible document families; excluded rows get held-out placeholders."""
    from klab.corpus.admission_inspection_cache import unchanged

    (
        root,
        configuration,
        raw_clusters,
        weights,
        clusters_sha256,
        prior_families,
        prior_strata,
        prior_content,
        adopted,
    ) = _split_inputs(
        project, work_root, inventory_path, clusters_path, output, recover=recover
    )
    inventory_work = {}
    report, inventory_reused = verified_inventory(
        project,
        root,
        inventory_path,
        raw_clusters["inventory_sha256"],
        write_split_inventory,
        adopted=adopted,
        work=inventory_work,
    )
    adopted = inventory_work["adopted"]
    lock = verify_acquisition(project, root, **verification_options(root))
    reference = project.release.record_admission
    if reference is None:
        raise ValueError("frozen splits require a reviewed record admission")
    admission_path = project_path(project.root, reference.path)
    if sha256_file(admission_path) != reference.sha256:
        raise ValueError("record admission SHA-256 mismatch")
    manifest = read_metadata(admission_path)
    if manifest.get("schema_version") != 2:
        raise ValueError("frozen splits require complete v2 admission")
    sources = {
        source.id: source_declaration_payload(source) for source in project.sources
    }
    snapshots = {
        source_id: read_metadata(Path(entry["snapshot_path"]) / "manifest.json")
        for source_id, entry in lock["sources"].items()
    }
    admitted = verify_record_admission(
        manifest,
        sources,
        snapshots,
        root / "corpora" / project.config.id / "snapshots",
    )
    connection = current_metadata_store().connection
    inventory = RowIndex(
        connection,
        (
            record.value
            for record in records_from_path(
                inventory_path, max_record_bytes=8 * 1024**2
            )
        ),
        lambda row: row["document_id"],
        duplicate="split inventory has duplicate document IDs",
    )
    hints = ValueMap(connection)
    for row in inventory.values():
        hints[row["family_hint"]] = True
    merged = ValueMap(connection)
    for item in raw_clusters["merges"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"family_id", "member_hints"}
            or not isinstance(item["family_id"], str)
            or not item["family_id"].strip()
            or not isinstance(item["member_hints"], list)
            or len(item["member_hints"]) < 2
            or len(set(item["member_hints"])) != len(item["member_hints"])
            or any(hint not in hints or hint in merged for hint in item["member_hints"])
        ):
            raise ValueError("invalid reviewed family merge")
        for hint in item["member_hints"]:
            merged[hint] = item["family_id"]
    families = ValueMap(connection)
    assignments = ValueMap(connection)
    qualified_family = ValueMap(connection)
    eligible = 0
    for row in inventory.values():
        source_id = row["source_id"]
        path = row["source_location"].split("#", 1)[0]
        decision = admitted.get((source_id, path))
        if decision is None:
            raise ValueError("split inventory lacks file admission decision")
        if "decisions" in decision:
            index = row["source_row_index"]
            choice = decision["decisions"].get(index)
            if choice is None:
                raise ValueError("split inventory lacks HF row admission decision")
            state = choice["decision"]
        else:
            state = decision["decision"]
        if state != "qualify":
            assignments[row["document_id"]] = "test"
            continue
        hint = row["family_hint"]
        if not isinstance(hint, str) or hint.startswith("unresolved-family:"):
            raise ValueError("eligible document has unresolved family hint")
        stratum = raw_clusters["source_strata"][source_id]
        family = merged.get(hint, hint)
        previous = prior_content.get(row["content_sha256"])
        if previous is not None and previous[0] != family:
            raise ValueError("prior content changed reviewed family")
        families[stratum, family] = hashlib.sha256(
            canonical_json([raw_clusters["seed"], stratum, family])
        ).hexdigest()
        qualified_family[row["document_id"]] = (stratum, family)
        eligible += 1
    # A family may not straddle source strata. It must be merged/reviewed first.
    family_strata = ValueMap(connection)
    for stratum, family in families:
        if family_strata.setdefault(family, stratum) != stratum:
            raise ValueError("reviewed family spans multiple token strata")
    realized: dict[str, dict[str, Any]] = {}
    for stratum in sorted(set(raw_clusters["source_strata"].values())):

        def names(stratum=stratum):
            for row in connection.execute(
                f"SELECT key FROM {families.table} WHERE json_extract(key,'$[0]')=? ORDER BY json_extract(value,'$'),json_extract(key,'$[1]')",
                (stratum,),
            ):
                yield json.loads(row[0])[1]

        name_count = connection.execute(
            f"SELECT COUNT(*) FROM {families.table} WHERE json_extract(key,'$[0]')=?",
            (stratum,),
        ).fetchone()[0]
        if name_count < sum(weights):
            raise ValueError(f"too few eligible families for {stratum} split")
        train, validation, test = _counts(name_count, weights)
        realized[stratum] = {
            "families": name_count,
            "train": train,
            "validation": validation,
            "test": test,
            "fractions": {
                "train": f"{train}/{name_count}",
                "validation": f"{validation}/{name_count}",
                "test": f"{test}/{name_count}",
            },
        }
        planned = dict(zip(_SPLITS, (train, validation, test), strict=True))
        pinned = ValueMap(connection)
        for family in names():
            if family in prior_families:
                pinned[family] = prior_families[family]
                if prior_strata[family] != stratum:
                    raise ValueError("prior family changed token stratum")
        remaining = {
            split: planned[split] - sum(value == split for value in pinned.values())
            for split in _SPLITS
        }
        if any(count < 0 for count in remaining.values()):
            raise ValueError("prior family assignments exceed new split quota")
        if sum(remaining.values()) != name_count - len(pinned):
            raise ValueError("prior family split accounting mismatch")
        index = 0
        for family in names():
            if family in pinned:
                split = pinned[family]
            else:
                split = (
                    "train"
                    if index < remaining["train"]
                    else "validation"
                    if index < remaining["train"] + remaining["validation"]
                    else "test"
                )
                index += 1
            for document_id in qualified_family.keys_for_value((stratum, family)):
                assignments[document_id] = split
    if not assignments.same_keys(inventory):
        raise ValueError("split assignments do not cover inventory")
    by_content = ValueMap(connection)
    for row in inventory.values():
        if row["document_id"] not in assignments:
            raise ValueError("missing document assignment")
        if row["document_id"] in qualified_family:
            if (
                by_content.setdefault(
                    row["content_sha256"], assignments[row["document_id"]]
                )
                != assignments[row["document_id"]]
            ):
                raise ValueError("cross-split exact content duplicate")
            previous = prior_content.get(row["content_sha256"])
            if previous is not None and assignments[row["document_id"]] != previous[1]:
                raise ValueError("prior held-out content changed split")
    temporary = Path(current_metadata_store().temporary.name)
    if adopted is not None:
        adopted.check(
            InventoryCache(root),
            project,
            inventory_path,
            raw_clusters["inventory_sha256"],
        )
    assignment_payload = temporary / ("assignments-" + uuid4().hex)
    with assignment_payload.open("xb") as stream:
        for document_id in assignments:
            stream.write(
                canonical_json(
                    {"document_id": document_id, "split": assignments[document_id]}
                )
                + b"\n"
            )
    assignment_digest = sha256_file(assignment_payload)
    assignment_output = output.parent / ("assignments-" + assignment_digest + ".jsonl")
    declaration = {
        "schema_version": 1,
        "unit": "document",
        "family_key": None,
        "assignments": {},
        "assignment_file": {"sha256": assignment_digest, "rows": len(assignments)},
    }
    overrides = getattr(getattr(project, "splits", None), "family_fields_by_source", {})
    if overrides:
        declaration["family_fields_by_source"] = {
            source_id: list(fields) for source_id, fields in sorted(overrides.items())
        }
    payload = _FREEZE_MARKER + yaml.safe_dump(declaration, sort_keys=False).encode(
        "utf-8"
    )
    family_payload = temporary / ("families-" + uuid4().hex)
    with family_payload.open("xb") as stream:
        for row in inventory.ordered_values():
            document_id = row["document_id"]
            if document_id in qualified_family:
                stratum, family = qualified_family[document_id]
                stream.write(
                    canonical_json(
                        {
                            "document_id": document_id,
                            "family_id": family,
                            "split": assignments[document_id],
                            "stratum": stratum,
                            "content_sha256": row["content_sha256"],
                        }
                    )
                    + b"\n"
                )
    candidate_path = output.with_name(output.stem + ".family-candidates.jsonl")
    receipt_path = output.with_name(output.stem + ".receipt.json")
    if not recover and any(
        path.exists() or path.is_symlink() for path in (candidate_path, receipt_path)
    ):
        raise ValueError("frozen split companion output already exists")
    receipt = {
        "schema_version": 1,
        "split_sha256": hashlib.sha256(payload).hexdigest(),
        "family_candidate_sha256": sha256_file(family_payload),
        "inventory_sha256": report["sha256"],
        "admission_sha256": reference.sha256,
        "cluster_sha256": clusters_sha256,
        "family_counts": realized,
        **(
            {"split_weights": raw_clusters["split_weights"]}
            if "split_weights" in raw_clusters
            else {}
        ),
        **(
            {
                "prior_family_inventory_sha256": raw_clusters["prior_family_inventory"][
                    "sha256"
                ],
                "prior_release_id": Path(
                    raw_clusters["prior_family_inventory"]["release_path"]
                ).name,
            }
            if prior_families
            else {}
        ),
    }
    members = (
        (candidate_path, family_payload),
        (assignment_output, assignment_payload),
        (output, payload),
        (receipt_path, canonical_json(receipt) + b"\n"),
    )
    # Validate every existing member before publishing anything, including receipt.
    for path, expected in members:
        if path.is_symlink() or (
            path.exists()
            and (
                not path.is_file()
                or (
                    sha256_file(path) != sha256_file(expected)
                    if isinstance(expected, Path)
                    else path.read_bytes() != expected
                )
            )
        ):
            raise ValueError(f"conflicting frozen split evidence: {path}")
    unchanged(configuration)
    if adopted is not None:
        adopted.check(
            InventoryCache(root),
            project,
            inventory_path,
            raw_clusters["inventory_sha256"],
        )
    for path, expected in members:
        if (recover or path == assignment_output) and path.exists():
            # Re-establish durability after a previous uncertain publication.
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            _sync_parents(path.parent, root)
            continue
        _publish_new(path, expected, work_root=root)
    verify_split_bundle(output)
    return {
        "schema_version": 1,
        "output": str(output),
        "sha256": sha256_file(output),
        "inventory_sha256": report["sha256"],
        "admission_sha256": reference.sha256,
        "cluster_sha256": clusters_sha256,
        "family_candidates": str(candidate_path),
        "family_candidate_sha256": receipt["family_candidate_sha256"],
        "receipt": str(receipt_path),
        "eligible_documents": eligible,
        "inventory_reused": inventory_reused,
        "inventory_completion": inventory_work["completion_mode"],
        "placeholder_documents": len(inventory) - eligible,
        "family_counts": realized,
        **(
            {"split_weights": raw_clusters["split_weights"]}
            if "split_weights" in raw_clusters
            else {}
        ),
    }


def _finalize_family_inventory(
    release: Path, frozen_splits: Path, output: Path
) -> dict[str, Any]:
    """Filter the authenticated family candidates to exactly kept release docs."""
    manifest = verify_release(release)
    candidate_path = frozen_splits.with_name(
        frozen_splits.stem + ".family-candidates.jsonl"
    )
    receipt = cast(dict[str, Any], verify_split_bundle(frozen_splits))
    if (
        receipt.get("schema_version") != 1
        or receipt.get("split_sha256") != sha256_file(frozen_splits)
        or receipt.get("family_candidate_sha256") != sha256_file(candidate_path)
        or manifest["build_identity"]["split"]
        != yaml.safe_load(frozen_splits.read_text(encoding="utf-8"))
    ):
        raise ValueError("family candidates differ from frozen release split")
    connection = current_metadata_store().connection
    candidates = RowIndex(
        connection,
        (
            record.value
            for record in records_from_path(
                candidate_path, max_record_bytes=8 * 1024**2
            )
        ),
        lambda row: row["document_id"],
        duplicate="duplicate family candidate document",
    )
    docs = RowIndex(
        connection,
        (record.value for record in records_from_path(release / "documents.jsonl")),
        lambda row: row["document_id"],
    )
    if not candidates.same_keys(docs):
        raise ValueError("family candidates differ from release documents")
    kept = 0
    payload = Path(current_metadata_store().temporary.name) / (
        "final-family-" + uuid4().hex
    )
    with payload.open("xb") as stream:
        for doc in docs.ordered_values():
            document_id = doc["document_id"]
            row = candidates[document_id]
            if (
                set(row)
                != {"document_id", "family_id", "split", "stratum", "content_sha256"}
                or row["split"] != doc["split"]
                or row["content_sha256"] != doc["content_sha256"]
                or row["stratum"] not in doc["domains"]
            ):
                raise ValueError("family candidate disagrees with verified release")
            if doc["drop_reason"] is None:
                stream.write(canonical_json(row) + b"\n")
                kept += 1
    if output.exists() or output.is_symlink():
        raise ValueError("final family inventory already exists")
    _publish_new(output, payload)
    return {
        "schema_version": 1,
        "release_id": manifest["release_id"],
        "output": str(output),
        "sha256": sha256_file(output),
        "kept_documents": kept,
        "duplicate_documents_removed": len(docs) - kept,
        "split_sha256": receipt["split_sha256"],
        "family_candidate_sha256": receipt["family_candidate_sha256"],
    }


def freeze_splits(
    project: Project,
    work_root: Path,
    inventory_path: Path,
    clusters_path: Path,
    output: Path,
    *,
    recover: bool = False,
    reuse_completed_preparation: bool = False,
) -> dict[str, Any]:
    if reuse_completed_preparation:
        from klab.corpus.streaming_split_freeze import freeze_completed_preparation

        return freeze_completed_preparation(
            project, work_root, inventory_path, clusters_path, output, recover=recover
        )
    # Keep verified row decisions alive through assignment and recovery checks.
    # All inventory/family/assignment joins share the owned operation database.
    with metadata_operation(
        Path(work_root) / "corpora" / project.config.id / "metadata-scratch",
        max_database_bytes=None,
    ):
        return _freeze_splits(
            project, work_root, inventory_path, clusters_path, output, recover=recover
        )


def finalize_family_inventory(
    release: Path, frozen_splits: Path, output: Path
) -> dict[str, Any]:
    from klab.corpus.completed_preparation import local_scope

    manifest = verify_release(release)
    scope = local_scope(manifest)
    if scope is not None:
        from klab.corpus.project import SplitDeclaration

        # The executed local build already joined the frozen candidates to each
        # rendered document, checked family isolation, and filtered native dedup.
        # Export that authenticated result instead of reconstructing a RowIndex.
        frozen_splits = frozen_splits.resolve()
        receipt_path = frozen_splits.with_name(frozen_splits.stem + ".receipt.json")
        if any(
            str(p) not in scope["request"]["inputs"]
            for p in (frozen_splits, receipt_path)
        ):
            raise ValueError("family export is not bound to the completed build split")
        receipt = json.loads(receipt_path.read_bytes())
        split = SplitDeclaration.model_validate(
            yaml.safe_load(frozen_splits.read_text(encoding="utf-8"))
        )
        if manifest["build_identity"]["split"] != split.model_dump(mode="json"):
            raise ValueError("family export split differs from completed build")
        if output.exists() or output.is_symlink():
            raise ValueError("final family inventory already exists")
        _publish_new(output, release / "family-inventory.jsonl")
        report = json.loads((release / "report.json").read_bytes())
        return {
            "schema_version": 1,
            "release_id": manifest["release_id"],
            "output": str(output),
            "sha256": manifest["files"]["family-inventory.jsonl"]["sha256"],
            "kept_documents": report["kept_document_count"],
            "duplicate_documents_removed": report["document_count"]
            - report["kept_document_count"],
            "split_sha256": receipt["split_sha256"],
            "family_candidate_sha256": receipt["family_candidate_sha256"],
        }
    with metadata_operation(output.parent, max_database_bytes=None):
        return _finalize_family_inventory(release, frozen_splits, output)
