"""Explicit, streamed recovery archives; thin metadata is not recovered model data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Literal

from klab.campaign.state import read_canonical
from klab.canonical import canonical_digest, canonical_json, reject_duplicate_keys
from klab.recovery.manifest import load_manifest
from klab.training.manifest import sha256_file
from klab.workdir import resolve_work_dir

_CHUNK = 1024 * 1024
_SMALL_LIMIT = 16 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 16384
_MAX_CAPTURED_METADATA = 128 * 1024 * 1024


def _safe(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or "\x00" in name
        or any(p in {"", ".", ".."} for p in name.split("/"))
        or path.as_posix() != name
    ):
        raise ValueError(f"unsafe archive member: {name!r}")
    return name


def _files(path: Path) -> list[Path]:
    if path.is_symlink():
        raise ValueError(f"symlinked archive input: {path}")
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"missing archive input: {path}")
    result: list[Path] = []
    for member in sorted(path.rglob("*")):
        if member.is_symlink() or (not member.is_file() and not member.is_dir()):
            raise ValueError(f"unsafe archive input: {member}")
        if member.is_file():
            result.append(member)
    return result


def _add(
    inventory: dict[str, tuple[Path, str]], name: str, file: Path, role: str
) -> None:
    _safe(name)
    if file.is_symlink() or not file.is_file():
        raise ValueError(f"missing or unsafe archive file: {file}")
    if name in inventory and inventory[name][0] != file:
        raise ValueError(f"duplicate archive destination: {name}")
    inventory[name] = (file, role)


def _declarations(source: Path) -> list[Path]:
    from klab.family.manifest import load_family
    from klab.recovery.provenance import (
        declaration_paths,
        declaration_reference,
        repository_root,
    )

    paths = set(declaration_paths(source, "recovery"))
    manifest = load_manifest(source)
    for reference in manifest.evidence:
        evidence = declaration_reference(source, reference)
        paths.add(evidence)
        if evidence.name.endswith(".json"):
            try:
                record = read_canonical(evidence)
            except OSError, ValueError:
                raise ValueError(f"invalid declared evidence: {evidence}") from None
            if record.get("lock_version") == 1:
                paths.add(evidence.with_name(evidence.stem + ".availability.json"))
    if manifest.family:
        family_path = declaration_reference(source, manifest.family)
        repository = repository_root(source) or source.parent
        for node in load_family(family_path).nodes:
            for reference in node.lifecycle_receipts:
                receipt = declaration_reference(family_path, reference)
                if not receipt.is_relative_to(repository):
                    raise ValueError("lifecycle receipt declaration outside repository")
                paths.add(receipt)
    return sorted(paths)


def _corpus_inputs(source: Path, manifest: Any, mode: str) -> set[Path]:
    """Identify local corpus payloads separately from small authored declarations."""
    from klab.campaign.plan import safe_path
    from klab.corpus.project import load_project
    from klab.experiments.plan import load_plan
    from klab.recovery.provenance import declaration_reference

    project_paths: set[Path] = set()
    for step in manifest.steps:
        if getattr(step, "project", None):
            project_paths.add(declaration_reference(source, step.project))
        if getattr(step, "plan", None):
            plan_path = declaration_reference(source, step.plan)
            for variant in load_plan(plan_path).corpus_variants:
                project_paths.add(declaration_reference(plan_path, variant.project))
    payloads: set[Path] = set()
    for project_path in sorted(project_paths):
        project = load_project(project_path)
        if mode == "portable" and not (
            project.release.publication_mode == "redistributable_under_source_terms"
            or (
                project.release.schema_version == 1
                and all(
                    item.redistribution == "redistributable" for item in project.sources
                )
            )
        ):
            raise ValueError("portable archive prohibited by corpus publication rights")
        for item in project.sources:
            if item.kind != "local":
                continue
            if mode == "portable" and not (
                item.redistribution == "redistributable"
                or (
                    item.rights is not None
                    and item.rights.redistribution_mode
                    == "redistributable_under_source_terms"
                )
            ):
                raise ValueError(
                    "portable archive prohibited by local source redistribution rights"
                )
            for entry in item.acquisition.files:
                payloads.add(safe_path(project.root, entry.path))
    return payloads


def _index_dependencies(
    inventory: dict[str, tuple[Path, str]],
    prefix: str,
    record: dict[str, Any],
    checkpoint: Path,
    ancestors: set[str],
) -> None:
    """Inventory only files consumed by a verified checkpoint-bound index."""
    from klab.evaluation.suite import _source, load_suite, verify_evaluation_index

    identity = record["index_sha256"]
    if identity in ancestors:
        raise ValueError("cyclic portable evaluation index dependency")
    # A shared index needs its full closure under every branch-local prefix.
    # Only ancestors of this invocation indicate a cycle, not completed siblings.
    ancestors = ancestors | {identity}
    run = Path(record["run"])
    if (run / record["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("portable index checkpoint differs from declared family node")
    _add(
        inventory, f"{prefix}/run-manifest.json", run / "manifest.json", "run_manifest"
    )
    suite_path = Path(record["suite"])
    _add(inventory, f"{prefix}/suite/{suite_path.name}", suite_path, "evaluation_suite")
    suite = load_suite(suite_path)
    for row, item in zip(record["evaluations"], suite.evaluations, strict=True):
        source = _source(suite_path, item.source) if item.source is not None else None
        base = f"{prefix}/evaluations/{item.id}"
        if source is not None and not (
            item.kind == "surface_review" and row["status"] == "SKIPPED_REVIEW"
        ):
            if source.is_dir():
                for file in _files(source):
                    _add(
                        inventory,
                        f"{base}/source/{file.relative_to(source).as_posix()}",
                        file,
                        "surface_bundle",
                    )
            else:
                _add(
                    inventory,
                    f"{base}/source/{source.name}",
                    source,
                    "evaluation_source",
                )
        if row["status"] != "COMPLETED" or item.kind == "surface_review":
            continue
        result_path = Path(row["path"])
        if item.kind == "evidence_reference":
            supplied = verify_evaluation_index(result_path)
            nested = f"{base}/supplied"
            _add(
                inventory,
                f"{nested}/index/{result_path.name}",
                result_path,
                "evaluation_index_dependency",
            )
            _index_dependencies(inventory, nested, supplied, checkpoint, ancestors)
        else:
            _add(
                inventory,
                f"{base}/result/{result_path.name}",
                result_path,
                "evaluation_result",
            )


def _review_dependencies(
    inventory: dict[str, tuple[Path, str]],
    prefix: str,
    review: Path,
) -> None:
    from klab.evaluation.readiness import verify_review_receipt

    verified = verify_review_receipt(review)
    _add(inventory, f"{prefix}/review/{review.name}", review, "model_review")
    if verified["surface_bundle"] is not None:
        bundle = Path(verified["surface_bundle"])
        for file in _files(bundle):
            _add(
                inventory,
                f"{prefix}/review/surface/{file.relative_to(bundle).as_posix()}",
                file,
                "surface_bundle",
            )


def _readiness_dependencies(
    inventory: dict[str, tuple[Path, str]],
    prefix: str,
    record: dict[str, Any],
) -> None:
    if record["policy"] is not None:
        policy = Path(record["policy"])
        _add(inventory, f"{prefix}/policy/{policy.name}", policy, "readiness_policy")
    if record["review"] is not None:
        _review_dependencies(inventory, prefix, Path(record["review"]))


def _inventory(
    source: Path, mode: Literal["thin", "portable"], root: Path
) -> tuple[dict[str, tuple[Path, str]], list[dict[str, Any]], str, dict[str, Any]]:
    from klab.family.cli import show
    from klab.family.manifest import load_family
    from klab.recovery.engine import inspect_manifest
    from klab.recovery.provenance import declaration_reference, repository_root

    manifest = load_manifest(source)
    rows = inspect_manifest(source, root)["steps"]
    inventory: dict[str, tuple[Path, str]] = {}
    external: list[dict[str, Any]] = []
    corpus_identities: dict[str, Any] = {}
    declaration_root = repository_root(source) or source.parent
    local_payloads = _corpus_inputs(source, manifest, mode)
    for declaration in _declarations(source):
        relative = declaration.relative_to(declaration_root).as_posix()
        if declaration in local_payloads and mode == "thin":
            external.append(
                {
                    "kind": "local_source",
                    "id": relative,
                    "sha256": sha256_file(declaration),
                    "location": str(declaration),
                    "state": "PRESENT_EXTERNAL",
                }
            )
            continue
        _add(inventory, f"declarations/{relative}", declaration, "declaration")
    for reference in manifest.evidence:
        evidence = declaration_reference(source, reference)
        record = read_canonical(evidence)
        if record.get("format") == "scientific-evidence-reference-v1":
            if (
                record.get("kind")
                not in {
                    "corpus_release",
                    "tokenizer_selection",
                    "prepared_data",
                    "runtime_probe",
                    "experiment_lock",
                    "checkpoint",
                    "evaluation_index",
                }
                or not isinstance(record.get("sha256"), str)
                or len(record["sha256"]) != 64
            ):
                raise ValueError(f"invalid compact scientific evidence: {evidence}")
            location = record.get("external_location")
            included = any(
                row["classification"] == "PRESENT"
                and row["actual_sha256"] == record["sha256"]
                for row in rows
            )
            if mode == "portable" and not included:
                raise ValueError(
                    f"portable archive lacks verified compact evidence payload: {reference}"
                )
            if mode == "thin":
                external.append(
                    {
                        "kind": record["kind"],
                        "id": record["identifier"],
                        "sha256": record["sha256"],
                        "location": location,
                        "state": "PRESENT_EXTERNAL" if included else "REFERENCE_ONLY",
                    }
                )
        elif record.get("format") == "model-lifecycle-receipt-v1":
            from klab.family.receipts import (
                verify_lifecycle_metadata,
                verify_lifecycle_receipt,
            )

            record = verify_lifecycle_metadata(evidence)
            try:
                verify_lifecycle_receipt(
                    evidence,
                    family_source=declaration_reference(source, manifest.family)
                    if manifest.family
                    else None,
                )
            except OSError:
                if mode == "portable":
                    raise
                external.append(
                    {
                        "kind": "lifecycle_binding",
                        "id": record["node"],
                        "sha256": record["receipt_sha256"],
                        "location": reference,
                        "state": "MISSING_EXTERNAL",
                    }
                )
        elif record.get("format") == "klab-recovery-receipt-v1":
            from klab.recovery.engine import verify_recovery_receipt

            record = verify_recovery_receipt(evidence)
            pinned = {row["id"]: row for row in rows}
            for outcome in record["outcomes"]:
                actual = outcome.get("actual_sha256")
                if actual is None:
                    continue
                current = pinned.get(outcome["id"])
                present = (
                    current is not None
                    and current["classification"] == "PRESENT"
                    and current["actual_sha256"] == actual
                )
                if mode == "portable" and not present:
                    raise ValueError(
                        f"portable recovery receipt output not verified: {outcome['id']}"
                    )
                if mode == "thin":
                    external.append(
                        {
                            "kind": "recovery_output",
                            "id": outcome["id"],
                            "sha256": actual,
                            "location": outcome.get("path"),
                            "state": "PRESENT_EXTERNAL"
                            if present
                            else "REFERENCE_ONLY",
                        }
                    )
        elif record.get("lock_version") == 1:
            from klab.campaign.state import digest
            from klab.experiments.lock import ResolvedExperimentPlan, open_lock

            sidecar = evidence.with_name(evidence.stem + ".availability.json")
            binding = read_canonical(sidecar)
            plan_sha = record.get("plan_sha256")
            if (
                evidence.name != f"{plan_sha}.json"
                or binding.get("plan_sha256") != plan_sha
                or binding.get("sha256")
                != digest(
                    "klab-experiment-availability-v1",
                    {
                        "plan_sha256": plan_sha,
                        "availability": binding.get("availability"),
                    },
                )
                or ResolvedExperimentPlan.model_validate(
                    {
                        **record,
                        "availability": binding["availability"],
                    }
                ).plan_sha256
                != plan_sha
            ):
                raise ValueError(f"invalid declared experiment lock: {evidence}")
            if mode == "portable":
                open_lock(evidence)
            _add(
                inventory,
                f"evidence/declared-lock/{plan_sha}.json",
                evidence,
                "experiment_lock",
            )
            _add(
                inventory,
                f"evidence/declared-lock/{plan_sha}.availability.json",
                sidecar,
                "lock_availability",
            )
        elif record.get("format") in {
            "evaluation-index-reference-v1",
            "model-readiness-reference-v1",
        }:
            digest_key = (
                "index_sha256"
                if record["format"] == "evaluation-index-reference-v1"
                else "result_sha256"
            )
            identity = record.get(digest_key)
            if not isinstance(identity, str) or len(identity) != 64:
                raise ValueError(
                    f"invalid compact evaluation/readiness evidence: {evidence}"
                )
            linked = False
            if manifest.family:
                family_path = declaration_reference(source, manifest.family)
                pinned = load_family(family_path)
                observed = show(family_path, work_root=root)["nodes"]
                for node, availability in zip(pinned.nodes, observed, strict=True):
                    binding = (
                        node.evaluation_index
                        if digest_key == "index_sha256"
                        else node.readiness_result
                    )
                    field = (
                        "evaluation_index"
                        if digest_key == "index_sha256"
                        else "readiness_result"
                    )
                    linked |= (
                        binding is not None
                        and binding.sha256 == identity
                        and availability["availability"][field] == "PRESENT"
                    )
            if mode == "portable" and not linked:
                raise ValueError(
                    f"portable archive needs domain-bound evaluation/readiness payload: {reference}"
                )
            if mode == "thin":
                external.append(
                    {
                        "kind": record["format"],
                        "id": identity,
                        "sha256": identity,
                        "location": record.get("index", record.get("result")),
                        "state": "PRESENT_EXTERNAL" if linked else "REFERENCE_ONLY",
                    }
                )
        else:
            raise ValueError(f"unsupported compact evidence format: {evidence}")
        _add(
            inventory,
            f"evidence/declared/{evidence.relative_to(declaration_root).as_posix()}",
            evidence,
            "compact_evidence",
        )
    family_state = "NOT_DECLARED"
    if manifest.family:
        family_path = declaration_reference(source, manifest.family)
        family = load_family(family_path)
        family_state = "DECLARED"
        _add(inventory, f"family/{family_path.name}", family_path, "family")
        statuses = show(family_path, work_root=root)
        for node, status in zip(family.nodes, statuses["nodes"], strict=True):
            for kind, ref in (
                ("checkpoint", node.checkpoint),
                ("evaluation_index", node.evaluation_index),
                ("readiness_result", node.readiness_result),
            ):
                if ref is None or status["availability"][kind] == "PRESENT":
                    continue
                if mode == "portable":
                    raise ValueError(
                        f"portable archive requires verified model {node.id}.{kind}"
                    )
                external.append(
                    {
                        "kind": kind,
                        "id": node.id,
                        "sha256": ref.sha256,
                        "location": ref.path,
                        "state": status["availability"][kind],
                    }
                )
            if (
                node.checkpoint
                and status["availability"]["checkpoint"] == "PRESENT"
                and mode == "thin"
            ):
                external.append(
                    {
                        "kind": "checkpoint",
                        "id": node.id,
                        "sha256": node.checkpoint.sha256,
                        "location": node.checkpoint.path,
                        "state": "PRESENT_EXTERNAL",
                    }
                )
            for kind, ref in (
                ("evaluation_index", node.evaluation_index),
                ("readiness_result", node.readiness_result),
            ):
                if ref and status["availability"][kind] == "PRESENT":
                    file = family_path.parent / ref.path
                    _add(
                        inventory, f"evidence/{node.id}/{kind}/{file.name}", file, kind
                    )
                    if mode == "portable":
                        if kind == "evaluation_index":
                            from klab.evaluation.suite import (
                                verify_evaluation_index,
                            )

                            _index_dependencies(
                                inventory,
                                f"evidence/{node.id}/{kind}",
                                verify_evaluation_index(file),
                                family_path.parent / node.checkpoint.path,
                                set(),
                            )
                        else:
                            from klab.evaluation.readiness import (
                                verify_readiness_result,
                            )

                            _readiness_dependencies(
                                inventory,
                                f"evidence/{node.id}/{kind}",
                                verify_readiness_result(file),
                            )
            if (
                node.checkpoint
                and status["availability"]["checkpoint"] == "PRESENT"
                and mode == "portable"
            ):
                path = family_path.parent / node.checkpoint.path
                _add(
                    inventory,
                    f"artifacts/run/{node.id}/manifest.json",
                    path.parent.parent / "manifest.json",
                    "run_manifest",
                )
                for file in _files(path):
                    _add(
                        inventory,
                        f"artifacts/checkpoint/{node.id}/{file.relative_to(path).as_posix()}",
                        file,
                        "checkpoint",
                    )
        # The family declares compact lifecycle receipts explicitly; never infer
        # an absent historical promotion from today's mutable state root.
        from klab.family.manifest import binding_path
        from klab.family.receipts import (
            verify_lifecycle_metadata,
            verify_lifecycle_receipt,
        )

        for node in family.nodes:
            location = root / "family" / family.id / family.identities()[node.id]
            observed = (
                sorted(location.glob("decision-*.json")) if location.is_dir() else []
            )
            declared: set[str] = set()
            for reference in node.lifecycle_receipts:
                decision = binding_path(family_path, reference)
                receipt = verify_lifecycle_metadata(decision)
                if (
                    receipt.get("family") != family.id
                    or receipt.get("node") != node.id
                    or receipt.get("node_sha256") != family.identities()[node.id]
                ):
                    raise ValueError("declared lifecycle receipt identity mismatch")
                try:
                    verify_lifecycle_receipt(decision, family_source=family_path)
                except OSError:
                    if mode == "portable":
                        raise
                    external.append(
                        {
                            "kind": "lifecycle_binding",
                            "id": node.id,
                            "sha256": receipt["receipt_sha256"],
                            "location": reference,
                            "state": "MISSING_EXTERNAL",
                        }
                    )
                declared.add(receipt["receipt_sha256"])
                _add(
                    inventory,
                    f"evidence/{node.id}/lifecycle/{decision.name}",
                    decision,
                    "lifecycle_receipt",
                )
                if mode == "portable":
                    _review_dependencies(
                        inventory,
                        f"evidence/{node.id}/lifecycle/{receipt['receipt_sha256']}",
                        Path(receipt["approval_path"]),
                    )
            for path in observed:
                record = verify_lifecycle_metadata(path)
                if record.get("receipt_sha256") not in declared:
                    raise ValueError(
                        f"unpublished lifecycle receipt for model node: {node.id}"
                    )
    for step, row in zip(manifest.steps, rows, strict=True):
        if step.kind in {"optional_cache"}:
            continue
        if mode == "portable" and row["classification"] != "PRESENT":
            raise ValueError(
                f"portable archive requires verified {step.id}: {row['classification']}"
            )
        if step.kind == "external_required":
            if mode == "portable":
                raise ValueError(
                    "portable archive cannot bundle unresolved external requirement"
                )
            external.append(
                {
                    "kind": step.kind,
                    "id": step.id,
                    "sha256": row["expected_sha256"],
                    "location": row["path"],
                    "state": row["classification"],
                }
            )
            continue
        if row["classification"] != "PRESENT":
            external.append(
                {
                    "kind": step.kind,
                    "id": step.id,
                    "sha256": row["expected_sha256"],
                    "location": row["path"],
                    "state": row["classification"],
                }
            )
            continue
        path = Path(row["path"])
        if step.kind == "corpus_release":
            from klab.corpus.identity import explain_corpus_identity
            from klab.corpus.project import load_project

            corpus_identities[step.id] = explain_corpus_identity(
                path, load_project(declaration_reference(source, step.project))
            )
        if mode == "portable":
            members = (
                [path / name for name in ("export.json", "run.yaml", "tokenizer.yaml")]
                if step.kind == "corpus_export"
                else _files(path)
            )
            for file in members:
                name = file.relative_to(path).as_posix() if path.is_dir() else path.name
                _add(inventory, f"artifacts/{step.id}/{name}", file, step.kind)
            if step.kind == "tokenizer_train":
                provenance = path.with_name("tokenizer_manifest.json")
                _add(
                    inventory,
                    f"artifacts/{step.id}/{provenance.name}",
                    provenance,
                    "tokenizer_manifest",
                )
            if step.kind == "corpus_release":
                from klab.corpus.release import verify_release

                release = verify_release(path)
                for snapshot in release["snapshots"]:
                    folder = (
                        path.parent.parent
                        / "snapshots"
                        / snapshot["source_id"]
                        / snapshot["sha256"]
                    )
                    from klab.corpus.acquisition import verify_snapshot

                    manifest = verify_snapshot(folder)
                    members = [
                        folder / "manifest.json",
                        *(
                            folder / "files" / entry["path"]
                            for entry in manifest["files"]
                        ),
                    ]
                    if set(_files(folder)) != set(members):
                        raise ValueError(f"unexpected source snapshot files: {folder}")
                    for file in members:
                        _add(
                            inventory,
                            f"artifacts/{step.id}/snapshots/{snapshot['source_id']}/{snapshot['sha256']}/{file.relative_to(folder).as_posix()}",
                            file,
                            "source_snapshot",
                        )
            if step.kind == "prepared_data" and path.is_file():
                from klab.config.loading import load_config
                from klab.corpus.export import verify_release_export
                from klab.data.packing import load_prepared_data

                record = read_canonical(path)
                variant = next(
                    (
                        entry
                        for entry in record["variants"]
                        if entry["id"] == step.variant_id
                    ),
                    None,
                )
                if variant is None:
                    raise ValueError(
                        f"prepared variant missing in archive inventory: {step.id}"
                    )
                exported = Path(variant["export_path"])
                config = load_config(exported / "run.yaml")
                verify_release_export(config.dataset)
                packed = Path(variant["prepared_path"])
                prepared = load_prepared_data(
                    packed, byte_enabled=config.model.memory in {"byte", "portable"}
                )
                if (
                    prepared.manifest_sha256 != row["actual_sha256"]
                    or sha256_file(packed / "manifest.json")
                    != variant["prepared_manifest_sha256"]
                    or sha256_file(Path(variant["tokenizer_path"]))
                    != variant["tokenizer_sha256"]
                ):
                    raise ValueError("variant prepared/tokenizer identity mismatch")
                for role, folder in (
                    ("corpus_export", exported),
                    ("prepared_data", packed),
                ):
                    members = (
                        [
                            folder / name
                            for name in ("export.json", "run.yaml", "tokenizer.yaml")
                        ]
                        if role == "corpus_export"
                        else _files(folder)
                    )
                    for file in members:
                        _add(
                            inventory,
                            f"artifacts/{step.id}/{role}/{file.relative_to(folder).as_posix()}",
                            file,
                            role,
                        )
                tokenizer = Path(variant["tokenizer_path"])
                _add(
                    inventory,
                    f"artifacts/{step.id}/tokenizer/{tokenizer.name}",
                    tokenizer,
                    "tokenizer_train",
                )
                provenance = tokenizer.with_name("tokenizer_manifest.json")
                _add(
                    inventory,
                    f"artifacts/{step.id}/tokenizer/{provenance.name}",
                    provenance,
                    "tokenizer_manifest",
                )
        else:
            for name in ("manifest.json", "lock.json", "preparation.json"):
                file = path / name if path.is_dir() else path
                if file.is_file() and file.stat().st_size <= _SMALL_LIMIT:
                    _add(inventory, f"evidence/{step.id}/{file.name}", file, step.kind)
                    if path.is_file():
                        break
            external.append(
                {
                    "kind": step.kind,
                    "id": step.id,
                    "sha256": row["actual_sha256"],
                    "location": str(path),
                    "state": "PRESENT_EXTERNAL",
                }
            )
        if step.kind == "experiment_lock":
            sidecar = path.with_name(path.stem + ".availability.json")
            location = (
                "artifacts" if mode == "portable" else "evidence"
            ) + f"/{step.id}/{sidecar.name}"
            _add(inventory, location, sidecar, "lock_availability")
    return inventory, external, family_state, corpus_identities


class _ArchiveReader:
    """Hash exactly the bytes handed to TAR, without buffering a whole member."""

    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int) -> bytes:
        data = self.handle.read(size)
        self.digest.update(data)
        self.size += len(data)
        return data


def create_archive(
    source: Path,
    mode: Literal["thin", "portable"],
    output: Path,
    *,
    work_root: Path | None = None,
) -> dict[str, Any]:
    """Exclusively publish a TAR whose sole payload comes from verified explicit references."""
    if mode not in {"thin", "portable"}:
        raise ValueError("archive mode must be thin or portable")
    source, output = Path(source).absolute(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    from klab.recovery.provenance import git_provenance

    provenance = git_provenance(tuple(_declarations(source)))
    if provenance["status"] != "CLEAN_AND_COMMITTED":
        raise ValueError(
            f"archive declaration closure must be checked in: {provenance['status']}"
        )
    inventory, external, family, corpus_identities = _inventory(
        source, mode, work_root or resolve_work_dir(None)
    )
    entries = [
        {
            "path": name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "role": role,
        }
        for name, (path, role) in sorted(inventory.items())
    ]
    index = {
        "format": "archive-index-v2",
        "corpus_identities": corpus_identities,
        "mode": mode,
        "family_lineage": family,
        "entries": entries,
        "external": external,
    }
    _validate_archive_index(index)
    index_bytes = canonical_json(index) + b"\n"
    if len(index_bytes) > _SMALL_LIMIT:
        raise ValueError("archive index exceeds bounded metadata size")
    required_bytes = (
        sum(entry["size"] for entry in entries) + len(index_bytes) + len(entries) * 1024
    )
    if mode == "portable":
        disk = shutil.disk_usage(output.parent)
        if disk.free < required_bytes:
            raise ValueError("insufficient archive destination bytes")
        stat = os.statvfs(output.parent)
        if stat.f_favail and stat.f_favail < len(entries) + 2:
            raise ValueError("insufficient archive destination inodes")
    # A sibling stage keeps publication on one filesystem. Only the stage is
    # ever cleaned up: a successful link (even followed by a durability error)
    # and a competing destination both belong to the caller.
    # Exclusive creation preserves the original output's umask/default-ACL
    # semantics without reading or changing the process-wide umask.
    staged = output.with_name(f".{output.name}.{secrets.token_hex(16)}.tmp")
    with staged.open("xb") as stream:
        try:
            with tarfile.open(fileobj=stream, mode="w|") as tar:
                info = tarfile.TarInfo("archive-index.json")
                info.size, info.mode, info.mtime = len(index_bytes), 0o644, 0
                import io

                tar.addfile(info, io.BytesIO(index_bytes))
                for entry in entries:
                    file = inventory[entry["path"]][0]
                    if (
                        file.is_symlink()
                        or not file.is_file()
                        or file.stat().st_size != entry["size"]
                    ):
                        raise ValueError(f"archive input changed: {file}")
                    info = tarfile.TarInfo(entry["path"])
                    info.size, info.mode, info.mtime = entry["size"], 0o644, 0
                    with file.open("rb") as handle:
                        reader = _ArchiveReader(handle)
                        tar.addfile(info, reader)
                        if (
                            reader.size != entry["size"]
                            or reader.digest.hexdigest() != entry["sha256"]
                            or handle.read(1)
                        ):
                            raise ValueError(f"archive input changed: {file}")
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(staged, output)
            except FileExistsError as exc:
                raise FileExistsError(f"refusing to overwrite {output}") from exc
            staged.unlink()
            try:
                directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except OSError as exc:
                raise OSError(
                    f"complete archive published at {output}, but its directory "
                    f"entry may not be durable: {exc}"
                ) from exc
        finally:
            staged.unlink(missing_ok=True)
    return index


def _verify_domain_inventory(
    index: dict[str, Any], manifests: dict[str, bytes]
) -> None:
    """Check archived release, checkpoint and prepared manifests against byte inventory."""
    present = {entry["path"]: entry for entry in index["entries"]}
    for name, raw in manifests.items():
        record = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("nonfinite domain manifest")
            ),
        )
        prefix = name.removesuffix(
            "tokenizer_manifest.json"
            if name.endswith("tokenizer_manifest.json")
            else "manifest.json"
        )
        if present[name]["role"] == "checkpoint":
            identity = record.get("sha256")
            body = {key: value for key, value in record.items() if key != "sha256"}
            if canonical_digest(body) != identity:
                raise ValueError("archived checkpoint manifest digest mismatch")
            listed = {
                entry["name"]: (entry["bytes"], entry["sha256"])
                for entry in record["files"]
            }
            if set(listed) != {
                path.removeprefix(prefix)
                for path in present
                if path.startswith(prefix) and path != name
            }:
                raise ValueError(
                    "archived checkpoint files differ from domain inventory"
                )
        elif any(
            entry["path"] == name and entry["role"] == "prepared_data"
            for entry in index["entries"]
        ):
            from klab.data.verification import required_arrays

            identity = record.get("manifest_sha256")
            body = {
                key: value for key, value in record.items() if key != "manifest_sha256"
            }
            if canonical_digest(body) != identity:
                raise ValueError("archived prepared-data manifest digest mismatch")
            required = required_arrays(record)
            available = {
                path.removeprefix(prefix)
                for path in present
                if path.startswith(prefix) and path != name
            }
            if not set(required) <= available or available - (
                set(required) | {".klab-cache-owner.json"}
            ):
                raise ValueError("archived prepared-data inventory mismatch")
            listed = {
                member: (
                    metadata.get("size_bytes", present[prefix + member]["size"]),
                    metadata["sha256"],
                )
                for member, (metadata, _, _) in required.items()
            }
        elif any(
            entry["path"] == name and entry["role"] == "source_snapshot"
            for entry in index["entries"]
        ):
            identity_fields = {
                key: record[key] for key in ("declaration_sha256", "adapter", "files")
            }
            if record.get("declaration", {}).get("kind") == "wikimedia_dump":
                identity_fields["retrieval"] = record["retrieval"]
            if hashlib.sha256(
                canonical_json(identity_fields)
            ).hexdigest() != record.get("snapshot_sha256"):
                raise ValueError("archived source snapshot identity mismatch")
            listed = {
                f"files/{entry['path']}": (entry["size"], entry["sha256"])
                for entry in record["files"]
            }
            available = {
                path.removeprefix(prefix)
                for path in present
                if path.startswith(prefix) and path != name
            }
            if not set(listed) <= available:
                raise ValueError("archived snapshot missing pinned source files")
        elif name.endswith("tokenizer_manifest.json"):
            token = present.get(prefix + "tokenizer.json")
            if token is None or record.get("sha256") != token["sha256"]:
                raise ValueError(
                    "archived tokenizer provenance differs from tokenizer bytes"
                )
            listed = {"tokenizer.json": (token["size"], record["sha256"])}
        else:
            identity = record.get("release_id")
            body = {key: value for key, value in record.items() if key != "release_id"}
            if canonical_digest(body) != identity:
                raise ValueError("archived corpus release manifest digest mismatch")
            listed = {
                file: (entry["size"], entry["sha256"])
                for file, entry in record["files"].items()
            }
            if set(listed) != {
                path.removeprefix(prefix)
                for path in present
                if path.startswith(prefix)
                and path != name
                and not path.startswith(prefix + "snapshots/")
            }:
                raise ValueError("archived corpus files differ from domain inventory")
            expected_snapshots: set[str] = set()
            for snapshot in record["snapshots"]:
                source_id, snapshot_sha = snapshot["source_id"], snapshot["sha256"]
                if not isinstance(source_id, str) or not re.fullmatch(
                    r"[A-Za-z0-9_-]+", source_id
                ):
                    raise ValueError("unsafe archived source snapshot ID")
                if not isinstance(snapshot_sha, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", snapshot_sha
                ):
                    raise ValueError("unsafe archived source snapshot SHA")
                member = f"{prefix}snapshots/{source_id}/{snapshot_sha}/manifest.json"
                expected_snapshots.add(member)
                if member not in manifests or member not in present:
                    raise ValueError("portable archive missing release-pinned snapshot")
                pinned = json.loads(
                    manifests[member], object_pairs_hook=reject_duplicate_keys
                )
                if (
                    pinned.get("source_id") != source_id
                    or pinned.get("snapshot_sha256") != snapshot_sha
                ):
                    raise ValueError(
                        "portable source snapshot identity differs from release"
                    )
            available_snapshots = {
                member
                for member in manifests
                if member.startswith(prefix + "snapshots/")
            }
            if expected_snapshots != available_snapshots:
                raise ValueError("portable snapshot inventory differs from release")
        for member, (size, sha) in listed.items():
            entry = present[prefix + member]
            if entry["size"] != size or entry["sha256"] != sha:
                raise ValueError(f"archived domain member identity mismatch: {member}")


def _verify_lock_bindings(index: dict[str, Any], records: dict[str, bytes]) -> None:
    from klab.campaign.state import digest

    entries = {entry["path"]: entry for entry in index["entries"]}
    locks = [entry for entry in index["entries"] if entry["role"] == "experiment_lock"]
    for entry in locks:
        path = entry["path"]
        if not path.endswith(".json") or path not in records:
            raise ValueError("archive lock record unavailable")
        sidecar = path.removesuffix(".json") + ".availability.json"
        if (
            sidecar not in entries
            or entries[sidecar]["role"] != "lock_availability"
            or sidecar not in records
        ):
            raise ValueError("archive lacks authenticated lock availability sidecar")
        lock = json.loads(records[path])
        binding = json.loads(records[sidecar])
        if (
            records[path] != canonical_json(lock) + b"\n"
            or records[sidecar] != canonical_json(binding) + b"\n"
        ):
            raise ValueError("noncanonical archive lock or availability")
        plan_sha = path.rsplit("/", 1)[-1].removesuffix(".json")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", plan_sha)
            or lock.get("plan_sha256") != plan_sha
            or binding.get("plan_sha256") != plan_sha
            or binding.get("sha256")
            != digest(
                "klab-experiment-availability-v1",
                {"plan_sha256": plan_sha, "availability": binding.get("availability")},
            )
        ):
            raise ValueError("archive experiment lock availability mismatch")
        from klab.experiments.lock import ResolvedExperimentPlan

        ResolvedExperimentPlan.model_validate(
            {
                **lock,
                "availability": binding["availability"],
            }
        )
    if sum(entry["role"] == "lock_availability" for entry in index["entries"]) != len(
        locks
    ):
        raise ValueError("orphan archive lock availability sidecar")


def _verify_surface_bundles(index: dict[str, Any], records: dict[str, bytes]) -> None:
    """Authenticate each bundled blinded/private seal without extraction."""
    entries = {entry["path"]: entry for entry in index["entries"]}
    for entry in index["entries"]:
        name = entry["path"]
        if entry["role"] != "surface_bundle" or not name.endswith("/manifest.json"):
            continue
        prefix = name.removesuffix("manifest.json")
        if any(
            prefix + member not in records
            for member in ("manifest.json", "blind.json", "provenance.json")
        ):
            raise ValueError("portable Surface Review seal incomplete")
        seal = json.loads(records[name])
        if (
            seal.get("format") != "klab_surface_manifest_v1"
            or seal.get("version") != 1
            or seal.get("blind_sha256") != entries[prefix + "blind.json"]["sha256"]
            or seal.get("provenance_sha256")
            != entries[prefix + "provenance.json"]["sha256"]
        ):
            raise ValueError("portable Surface Review seal mismatch")


def _verify_portable_evaluations(
    index: dict[str, Any], records: dict[str, bytes]
) -> None:
    """Bind included evaluations and reviews to their exact prerequisites."""
    import yaml

    from klab.evaluation.readiness import _readiness_identity, _review_identity
    from klab.evaluation.suite import _index_binding
    from klab.family.manifest import FamilyManifest

    entries = {entry["path"]: entry for entry in index["entries"]}

    def required(path: str, sha: str | None = None) -> bytes:
        if path not in records or path not in entries:
            raise ValueError(f"missing portable evaluation dependency: {path}")
        if sha is not None and entries[path]["sha256"] != sha:
            raise ValueError(f"portable evaluation dependency digest mismatch: {path}")
        return records[path]

    def document(path: str) -> dict[str, Any]:
        data = required(path)
        value = json.loads(data)
        if data != canonical_json(value) + b"\n":
            raise ValueError(f"noncanonical portable evaluation record: {path}")
        return value

    families = [
        entry["path"] for entry in index["entries"] if entry["role"] == "family"
    ]
    if (index["family_lineage"] == "DECLARED") != (len(families) == 1):
        raise ValueError("portable archive lacks its declared family graph")
    family = (
        FamilyManifest.model_validate(yaml.safe_load(required(families[0])))
        if families
        else None
    )
    nodes = {node.id: node for node in family.nodes} if family else {}

    for entry in index["entries"]:
        if entry["role"] != "checkpoint" or not re.fullmatch(
            r"artifacts/checkpoint/[^/]+/manifest\.json", entry["path"]
        ):
            continue
        node = entry["path"].split("/")[2]
        checkpoint = document(entry["path"])
        run_manifest = document(f"artifacts/run/{node}/manifest.json")
        if (
            checkpoint.get("manifest_sha256")
            != hashlib.sha256(
                canonical_json(
                    {
                        key: value
                        for key, value in run_manifest.items()
                        if key != "sha256"
                    }
                )
            ).hexdigest()
        ):
            raise ValueError("portable checkpoint lacks its bound run manifest")
        declared = nodes.get(node)
        effective = run_manifest["effective_config"]
        if (
            declared is None
            or declared.checkpoint is None
            or checkpoint["sha256"] != declared.checkpoint.sha256
            or checkpoint.get("parent_checkpoint_sha256")
            != declared.parent_checkpoint_sha256
            or run_manifest.get("checkpoint_sha256")
            != declared.parent_checkpoint_sha256
            or run_manifest["architecture_sha256"] != declared.architecture_sha256
            or checkpoint["architecture_sha256"] != declared.architecture_sha256
            or effective["training"]["max_steps"] != declared.budget.max_steps
            or effective["training"]["max_tokens"] != declared.budget.max_tokens
            or declared.objective != "next_token"
        ):
            raise ValueError("portable checkpoint differs from declared family node")

        from klab.family.cli import validate_run_bindings

        validate_run_bindings(run_manifest, declared, local_release=False)
    for entry in index["entries"]:
        name, role = entry["path"], entry["role"]
        if role in {"evaluation_index", "evaluation_index_dependency"}:
            record = document(name)
            prefix = (
                name.rsplit("/index/", 1)[0]
                if role == "evaluation_index_dependency"
                else name.rsplit("/", 1)[0]
            )
            node = name.split("/")[1]
            checkpoint = document(f"artifacts/checkpoint/{node}/manifest.json")
            run_manifest = document(f"{prefix}/run-manifest.json")
            suite = f"{prefix}/suite/{Path(record['suite']).name}"
            required(suite, record["suite_sha256"])
            if (
                checkpoint["sha256"] != record["checkpoint_sha256"]
                or checkpoint["manifest_sha256"]
                != hashlib.sha256(
                    canonical_json(
                        {
                            key: value
                            for key, value in run_manifest.items()
                            if key != "sha256"
                        }
                    )
                ).hexdigest()
            ):
                raise ValueError("portable index run/checkpoint mismatch")
            identity = hashlib.sha256(
                canonical_json(
                    _index_binding(
                        record["suite_sha256"],
                        record["checkpoint_sha256"],
                        record["evaluations"],
                        record["evaluation_runtime"],
                    )
                )
            ).hexdigest()
            if (
                identity != record["index_sha256"]
                or record.get("record_sha256")
                != hashlib.sha256(
                    canonical_json(
                        {
                            key: value
                            for key, value in record.items()
                            if key != "record_sha256"
                        }
                    )
                ).hexdigest()
            ):
                raise ValueError("portable index identity mismatch")
            if role == "evaluation_index" and (
                nodes[node].evaluation_index is None
                or record["index_sha256"] != nodes[node].evaluation_index.sha256
            ):
                raise ValueError(
                    "portable evaluation index differs from declared family node"
                )
            for row in record["evaluations"]:
                base = f"{prefix}/evaluations/{row['id']}"
                source = row.get("source_sha256")
                if source is not None:
                    members = [
                        item
                        for item in index["entries"]
                        if item["path"].startswith(base + "/source/")
                    ]
                    if not members:
                        raise ValueError(
                            "portable index omits declared evaluation source"
                        )
                    if row["kind"] == "surface_review":
                        required(base + "/source/review.json", row["sha256"])
                    elif len(members) != 1 or members[0]["sha256"] != source:
                        raise ValueError("portable evaluation source mismatch")
                if row["status"] == "COMPLETED" and row["kind"] != "surface_review":
                    result = Path(row["path"]).name
                    if row["kind"] == "evidence_reference":
                        supplied = document(f"{base}/supplied/index/{result}")
                        if supplied.get("index_sha256") != row["sha256"]:
                            raise ValueError(
                                "portable supplied evaluation index mismatch"
                            )
                    else:
                        result_record = document(f"{base}/result/{result}")
                        required(f"{base}/result/{result}", row["sha256"])
                        if (
                            result_record.get("identity", {}).get("checkpoint_sha256")
                            != record["checkpoint_sha256"]
                        ):
                            raise ValueError(
                                "portable evaluation result checkpoint mismatch"
                            )
        elif role == "readiness_result":
            record = document(name)
            prefix = name.rsplit("/", 1)[0]
            if record.get("result_sha256") != _readiness_identity(record):
                raise ValueError("portable readiness identity mismatch")
            node = nodes[name.split("/")[1]]
            if (
                node.readiness_result is None
                or record["result_sha256"] != node.readiness_result.sha256
                or record["index_sha256"] != node.evaluation_index.sha256
                or record["checkpoint_sha256"] != node.checkpoint.sha256
            ):
                raise ValueError("portable readiness differs from declared family node")
            if record["policy"] is not None:
                required(
                    f"{prefix}/policy/{Path(record['policy']).name}",
                    record["policy_sha256"],
                )
            if record["review"] is not None:
                review = document(f"{prefix}/review/{Path(record['review']).name}")
                if review.get("receipt_sha256") != record["review_sha256"]:
                    raise ValueError("portable readiness review mismatch")
        elif role == "model_review":
            record = document(name)
            if record.get("receipt_sha256") != _review_identity(record):
                raise ValueError("portable human review identity mismatch")
            if record["surface_bundle"] is not None:
                prefix = name.rsplit("/review/", 1)[0]
                required(
                    f"{prefix}/review/surface/review.json",
                    record["surface_review_digest"],
                )
        elif role == "lifecycle_receipt":
            from klab.campaign.state import digest

            record = document(name)
            body = {
                key: value
                for key, value in record.items()
                if key
                not in {
                    "approval_path",
                    "readiness_path",
                    "evaluation_path",
                    "family_path",
                    "issued_at_utc",
                    "receipt_sha256",
                    "record_sha256",
                }
            }
            if (
                record.get("receipt_sha256")
                != digest("klab-model-lifecycle-receipt-v1", body)
                or record.get("record_sha256")
                != hashlib.sha256(
                    canonical_json(
                        {
                            key: value
                            for key, value in record.items()
                            if key != "record_sha256"
                        }
                    )
                ).hexdigest()
            ):
                raise ValueError("portable lifecycle receipt identity mismatch")
            prefix = name.rsplit("/", 1)[0] + "/" + record["receipt_sha256"]
            reviewed = document(f"{prefix}/review/{Path(record['approval_path']).name}")
            if reviewed.get("receipt_sha256") != record["approval_sha256"]:
                raise ValueError("portable lifecycle approval mismatch")
            from klab.family.receipts import _ancestors, _validate_action

            node = nodes.get(record["node"])
            if (
                family is None
                or node is None
                or family.id != record["family"]
                or family.identities()[node.id] != record["node_sha256"]
            ):
                raise ValueError("portable lifecycle family node mismatch")
            readiness = document(
                f"evidence/{node.id}/readiness_result/{Path(record['readiness_path']).name}"
            )
            if (
                readiness["result_sha256"] != record["readiness_sha256"]
                or readiness["index_sha256"] != record["evaluation_sha256"]
                or readiness["checkpoint_sha256"] != record["checkpoint_sha256"]
                or reviewed["index_sha256"] != record["evaluation_sha256"]
                or reviewed["checkpoint_sha256"] != record["checkpoint_sha256"]
            ):
                raise ValueError("portable lifecycle scientific evidence mismatch")
            _validate_action(
                record["action"],
                record["successor"],
                readiness["state"],
                reviewed["decision"],
            )
            successor = record["successor"]
            if successor is not None and (
                successor not in nodes or node.id not in _ancestors(nodes, successor)
            ):
                raise ValueError("portable lifecycle successor is not a descendant")


def _verify_portable_closure(
    index: dict[str, Any],
    declarations: dict[str, bytes],
    records: dict[str, bytes],
) -> None:
    """Reopen bundled recovery intent, never trusting rewritten outer membership."""
    import yaml

    from klab.documents import _UniqueLoader
    from klab.recovery.manifest import RecoveryManifest

    if index["external"]:
        raise ValueError("portable archive cannot have unresolved external references")
    entries = {entry["path"]: entry for entry in index["entries"]}
    candidates = []
    for name, raw in declarations.items():
        try:
            declaration = yaml.load(raw.decode("utf-8"), Loader=_UniqueLoader)
        except UnicodeError, yaml.YAMLError, ValueError, TypeError:
            continue
        if isinstance(declaration, dict) and declaration.get("recovery_version") == 1:
            candidates.append((name, declaration))
    if len(candidates) != 1:
        raise ValueError("portable archive needs exactly one bundled RecoveryManifest")
    source, authored = candidates[0]
    manifest = RecoveryManifest.model_validate(authored)

    def declared(reference: str, document: str = source) -> str:
        local = (PurePosixPath(document).parent / reference).as_posix()
        repository = f"declarations/{reference}"
        path = local if local in entries else repository
        if path not in entries or entries[path]["role"] != "declaration":
            raise ValueError(f"portable archive lacks declared prerequisite: {path}")
        return path

    def required(path: str, role: str) -> dict[str, Any]:
        entry = entries.get(path)
        if entry is None or entry["role"] != role:
            raise ValueError(f"portable archive lacks declared {role}: {path}")
        return entry

    if manifest.family is None:
        if index["family_lineage"] != "NOT_DECLARED":
            raise ValueError("portable archive invents model family lineage")
    else:
        from klab.family.manifest import FamilyManifest

        path = declared(manifest.family)
        family_path = f"family/{PurePosixPath(path).name}"
        if required(family_path, "family")["sha256"] != entries[path]["sha256"]:
            raise ValueError("portable family bytes differ from declaration")
        if index["family_lineage"] != "DECLARED":
            raise ValueError("portable archive omits declared model family lineage")
        family = FamilyManifest.model_validate(
            yaml.load(records[family_path].decode("utf-8"), Loader=_UniqueLoader)
        )
        for node in family.nodes:
            if node.recovery_manifest is not None:
                declared(node.recovery_manifest, path)
            for receipt in node.lifecycle_receipts:
                declaration = declared(receipt, path)
                archived = required(
                    f"evidence/{node.id}/lifecycle/{Path(receipt).name}",
                    "lifecycle_receipt",
                )
                if archived["sha256"] != entries[declaration]["sha256"]:
                    raise ValueError(
                        "portable lifecycle receipt differs from declaration"
                    )
        for node in family.nodes:
            if node.checkpoint is not None:
                pinned = required(
                    f"artifacts/checkpoint/{node.id}/manifest.json", "checkpoint"
                )
                checkpoint = json.loads(records[pinned["path"]])
                if checkpoint.get("sha256") != node.checkpoint.sha256:
                    raise ValueError("portable family checkpoint identity mismatch")
                required(f"artifacts/run/{node.id}/manifest.json", "run_manifest")
            for kind, binding in (
                ("evaluation_index", node.evaluation_index),
                ("readiness_result", node.readiness_result),
            ):
                if binding is not None:
                    required(
                        f"evidence/{node.id}/{kind}/{Path(binding.path).name}", kind
                    )
            for receipt in node.lifecycle_receipts:
                required(
                    f"evidence/{node.id}/lifecycle/{Path(receipt).name}",
                    "lifecycle_receipt",
                )
    for reference in (manifest.evaluation_suite, manifest.readiness_policy):
        if reference is not None:
            declared(reference)
    for reference in manifest.evidence:
        path = declared(reference)
        archived = required(
            f"evidence/declared/{path.removeprefix('declarations/')}",
            "compact_evidence",
        )
        if archived["sha256"] != entries[path]["sha256"]:
            raise ValueError("portable compact evidence differs from declaration")
    releases: dict[str, str] = {}
    for step in manifest.steps:
        if step.kind == "optional_cache":
            continue
        if step.kind == "external_required":
            raise ValueError(
                "portable archive cannot bundle unresolved external requirement"
            )
        for field in ("project", "base_run", "config", "plan"):
            reference = getattr(step, field, None)
            if reference is not None:
                declared(reference)
        prefix = f"artifacts/{step.id}/"
        if step.kind == "corpus_release":
            member = required(prefix + "manifest.json", "corpus_release")
            release = json.loads(records[member["path"]])
            if (
                step.expected_release_sha256 is not None
                and release.get("release_id") != step.expected_release_sha256
            ):
                raise ValueError("portable corpus release differs from pinned identity")
            releases[step.id] = release["release_id"]
        elif step.kind == "corpus_export":
            member = required(prefix + "export.json", "corpus_export")
            export = json.loads(records[member["path"]])
            run_config = required(prefix + "run.yaml", "corpus_export")
            tokenizer_config = required(prefix + "tokenizer.yaml", "corpus_export")
            actual = hashlib.sha256(
                canonical_json(
                    {
                        key: export[key]
                        for key in (
                            "release_id",
                            "view",
                            "base_config_sha256",
                            "vocab_size",
                        )
                    }
                )
            ).hexdigest()
            if (
                step.expected_export_sha256 is not None
                and actual != step.expected_export_sha256
            ):
                raise ValueError("portable corpus export differs from pinned identity")
            if export["view"] != step.view or export["vocab_size"] != step.vocab_size:
                raise ValueError("portable corpus export declaration mismatch")
            if (
                export["release_id"] != releases[step.corpus]
                or export["base_config_sha256"]
                != entries[declared(step.base_run)]["sha256"]
                or export["run_config_sha256"] != run_config["sha256"]
                or export["tokenizer_config_sha256"] != tokenizer_config["sha256"]
            ):
                raise ValueError("portable corpus export prerequisite binding mismatch")
        elif step.kind == "tokenizer_train":
            token = required(prefix + "tokenizer.json", "tokenizer_train")
            required(prefix + "tokenizer_manifest.json", "tokenizer_manifest")
            if (
                step.expected_tokenizer_sha256 is not None
                and token["sha256"] != step.expected_tokenizer_sha256
            ):
                raise ValueError("portable tokenizer differs from pinned identity")
        elif step.kind == "prepared_data":
            prepared = prefix + "manifest.json"
            if prepared not in entries:
                required(prefix + "preparation.json", "prepared_data")
                prepared = prefix + "prepared_data/manifest.json"
            required(prepared, "prepared_data")
            payload = json.loads(records[prepared])
            if (
                step.expected_manifest_sha256 is not None
                and payload.get("manifest_sha256") != step.expected_manifest_sha256
            ):
                raise ValueError("portable prepared data differs from pinned identity")
        elif step.kind == "experiment_lock":
            locks = [
                entry
                for path, entry in entries.items()
                if path.startswith(prefix)
                and path.endswith(".json")
                and entry["role"] == "experiment_lock"
            ]
            if len(locks) != 1:
                raise ValueError(
                    f"portable archive lacks declared experiment lock: {step.id}"
                )
            actual = json.loads(records[locks[0]["path"]])["plan_sha256"]
            if (
                step.expected_plan_sha256 is not None
                and actual != step.expected_plan_sha256
            ):
                raise ValueError(
                    "portable experiment lock differs from pinned identity"
                )
        elif step.kind == "checkpoint":
            member = required(prefix + "manifest.json", "checkpoint")
            checkpoint = json.loads(records[member["path"]])
            if (
                step.expected_sha256 is not None
                and checkpoint.get("sha256") != step.expected_sha256
            ):
                raise ValueError("portable checkpoint differs from pinned identity")


def _validate_archive_index(index: Any) -> None:
    fields = {"format", "mode", "family_lineage", "entries", "external"}
    if isinstance(index, dict) and index.get("format") == "archive-index-v2":
        fields.add("corpus_identities")
    if (
        not isinstance(index, dict)
        or set(index) != fields
        or index["format"] not in {"archive-index-v1", "archive-index-v2"}
        or index["mode"] not in {"thin", "portable"}
        or index["family_lineage"] not in {"DECLARED", "NOT_DECLARED"}
        or not isinstance(index["entries"], list)
        or not isinstance(index["external"], list)
    ):
        raise ValueError("malformed archive index")
    if index["format"] == "archive-index-v2":
        from klab.corpus.identity import verify_corpus_identity

        if not isinstance(index["corpus_identities"], dict):
            raise ValueError("malformed archive corpus identities")
        for step_id, explanation in index["corpus_identities"].items():
            _safe(step_id)
            verify_corpus_identity(explanation)
    if len(index["entries"]) > _MAX_ARCHIVE_MEMBERS:
        raise ValueError("archive inventory exceeds bounded member count")
    seen: set[str] = set()
    for entry in index["entries"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "size", "sha256", "role"}
            or not isinstance(entry["path"], str)
            or type(entry["size"]) is not int
            or entry["size"] < 0
            or not isinstance(entry["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            or not isinstance(entry["role"], str)
            or not entry["role"]
        ):
            raise ValueError("malformed archive inventory entry")
        name = _safe(entry["path"])
        if name == "archive-index.json" or name in seen:
            raise ValueError(f"duplicate archive inventory entry: {name}")
        if name.startswith("artifacts/checkpoint/") and entry["role"] != "checkpoint":
            raise ValueError("checkpoint artifact cannot be relabeled in archive index")
        if name.startswith("artifacts/run/") and entry["role"] != "run_manifest":
            raise ValueError("run manifest cannot be relabeled in archive index")
        seen.add(name)
    for reference in index["external"]:
        if (
            not isinstance(reference, dict)
            or set(reference) != {"kind", "id", "sha256", "location", "state"}
            or not isinstance(reference["kind"], str)
            or not reference["kind"]
            or not isinstance(reference["id"], str)
            or not reference["id"]
            or (
                reference["sha256"] is not None
                and (
                    not isinstance(reference["sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])
                )
            )
            or (
                reference["location"] is not None
                and not isinstance(reference["location"], str)
            )
            or not isinstance(reference["state"], str)
            or not reference["state"]
        ):
            raise ValueError("malformed archive external reference")


def verify_archive(path: Path) -> dict[str, Any]:
    """Stream and authenticate every TAR member; never extract or follow links."""
    names: set[str] = set()
    found: list[dict[str, Any]] = []
    index: dict[str, Any] | None = None
    manifests: dict[str, bytes] = {}
    lock_records: dict[str, bytes] = {}
    evaluation_records: dict[str, bytes] = {}
    declarations: dict[str, bytes] = {}
    captured_bytes = 0
    roles: dict[str, str] = {}
    with tarfile.open(path, mode="r|*") as tar:
        for member in tar:
            name = _safe(member.name)
            if name in names or not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"duplicate or unsafe archive member: {name}")
            names.add(name)
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable archive member: {name}")
            checksum = hashlib.sha256()
            if name == "archive-index.json":
                if index is not None or found or member.size > _SMALL_LIMIT:
                    raise ValueError("archive index must be first and bounded")
                raw = stream.read(_SMALL_LIMIT + 1)
                if len(raw) != member.size:
                    raise ValueError("archive index size mismatch")

                index = json.loads(
                    raw,
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError("nonfinite archive index")
                    ),
                )
                if raw != canonical_json(index) + b"\n":
                    raise ValueError("noncanonical archive index")
                _validate_archive_index(index)
                roles = {entry["path"]: entry["role"] for entry in index["entries"]}
                continue
            if index is None:
                raise ValueError("missing archive index")
            if name not in roles or len(found) >= len(index["entries"]):
                raise ValueError(f"unlisted archive member: {name}")
            remaining = member.size
            declaration_capture = (
                index["mode"] == "portable"
                and name.startswith("declarations/")
                and name.endswith((".yaml", ".yml", ".json"))
                and roles.get(name) == "declaration"
                and member.size <= _SMALL_LIMIT
            )
            step_capture = (
                index["mode"] == "portable"
                and name.startswith("artifacts/")
                and (
                    (
                        name.endswith("/export.json")
                        and roles.get(name) == "corpus_export"
                    )
                    or (
                        name.endswith("/preparation.json")
                        and roles.get(name) == "prepared_data"
                    )
                )
            )
            lock_capture = roles.get(name) in {"experiment_lock", "lock_availability"}
            evaluation_capture = index["mode"] == "portable" and (
                (
                    name.startswith("evidence/")
                    and roles.get(name)
                    in {
                        "evaluation_index",
                        "evaluation_index_dependency",
                        "evaluation_result",
                        "evaluation_source",
                        "evaluation_suite",
                        "run_manifest",
                        "readiness_result",
                        "readiness_policy",
                        "model_review",
                        "surface_bundle",
                        "lifecycle_receipt",
                    }
                )
                or (
                    name.startswith("artifacts/run/")
                    and roles.get(name) == "run_manifest"
                )
                or (name.startswith("family/") and roles.get(name) == "family")
            )
            if (
                lock_capture or evaluation_capture or step_capture
            ) and member.size > _SMALL_LIMIT:
                raise ValueError(f"oversized portable evidence metadata: {name}")
            corpus_capture = (
                name.endswith("/manifest.json") and roles.get(name) == "corpus_release"
            )
            if corpus_capture and member.size > _SMALL_LIMIT:
                raise ValueError("oversized corpus identity metadata")
            domain_capture = (
                index["mode"] == "portable"
                and name.endswith(("/manifest.json", "/tokenizer_manifest.json"))
                and roles.get(name)
                in {
                    "corpus_release",
                    "checkpoint",
                    "prepared_data",
                    "source_snapshot",
                    "tokenizer_manifest",
                }
            )
            capture = member.size <= _SMALL_LIMIT and (
                corpus_capture
                or lock_capture
                or evaluation_capture
                or domain_capture
                or declaration_capture
                or step_capture
            )
            if capture:
                captured_bytes += member.size
                if captured_bytes > _MAX_CAPTURED_METADATA:
                    raise ValueError("archive captured metadata exceeds bounded memory")
            pieces: list[bytes] = []
            while remaining:
                part = stream.read(min(_CHUNK, remaining))
                if not part:
                    raise ValueError(f"truncated archive member: {name}")
                checksum.update(part)
                if capture:
                    pieces.append(part)
                remaining -= len(part)
            found.append(
                {"path": name, "size": member.size, "sha256": checksum.hexdigest()}
            )
            if capture:
                if lock_capture:
                    lock_records[name] = b"".join(pieces)
                elif evaluation_capture or step_capture:
                    evaluation_records[name] = b"".join(pieces)
                elif declaration_capture:
                    declarations[name] = b"".join(pieces)
                else:
                    manifests[name] = b"".join(pieces)
    if (
        index is None
        or not isinstance(index.get("entries"), list)
        or not isinstance(index.get("external"), list)
    ):
        raise ValueError("archive index missing entries")
    expected = index["entries"]
    if found != [
        {key: entry[key] for key in ("path", "size", "sha256")} for entry in expected
    ]:
        raise ValueError("archive member inventory differs from index")
    if len(names) != len(expected) + 1 or any(
        not isinstance(entry.get("role"), str) for entry in expected
    ):
        raise ValueError("archive inventory malformed")
    _verify_lock_bindings(index, lock_records)
    if index["format"] == "archive-index-v2":
        prefix = "artifacts" if index["mode"] == "portable" else "evidence"
        expected_corpora = {
            entry["path"]
            for entry in index["entries"]
            if entry["role"] == "corpus_release"
            and entry["path"].endswith("/manifest.json")
        }
        explained = {
            f"{prefix}/{step_id}/manifest.json"
            for step_id in index["corpus_identities"]
        }
        if explained != expected_corpora:
            raise ValueError("archive corpus_release explanation inventory mismatch")
        for step_id, explanation in index["corpus_identities"].items():
            member = f"{prefix}/{step_id}/manifest.json"
            if canonical_json(explanation["release"]) + b"\n" != manifests[member]:
                raise ValueError("archive corpus explanation release binding mismatch")
    if index["mode"] == "portable":
        _verify_portable_closure(
            index, declarations, {**lock_records, **evaluation_records, **manifests}
        )
        _verify_domain_inventory(index, manifests)
        _verify_portable_evaluations(index, {**evaluation_records, **manifests})
        _verify_surface_bundles(index, evaluation_records)
    return {
        **index,
        "verified_members": len(expected),
        "unresolved_references": index["external"],
    }


def _handle(args: argparse.Namespace) -> None:
    try:
        result = (
            create_archive(
                Path(args.source),
                args.mode,
                Path(args.output),
                work_root=resolve_work_dir(args.work_dir),
            )
            if args.archive_command == "create"
            else verify_archive(Path(args.source))
        )
        print(
            json.dumps(result, sort_keys=True, allow_nan=False)
            if args.json
            else json.dumps(result, indent=2, sort_keys=True)
        )
    except (OSError, ValueError, TypeError, KeyError, tarfile.TarError) as error:
        print(
            json.dumps(
                {"format": "archive-index-v2", "state": "FAILED", "reason": str(error)}
            )
        )
        raise SystemExit(1) from error


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "archive", help="Create or verify explicit recovery archives"
    )
    verbs = parser.add_subparsers(dest="archive_command", required=True)
    create = verbs.add_parser("create")
    create.add_argument("source")
    create.add_argument("--mode", choices=("thin", "portable"), required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--json", action="store_true")
    create.set_defaults(handler=_handle)
    verify = verbs.add_parser("verify")
    verify.add_argument("source")
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(handler=_handle)
