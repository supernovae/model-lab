"""Read-only availability and explicit replay of deterministic scientific prerequisites."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

from klab.campaign.state import digest, publish_immutable, read_canonical, utc_now
from klab.canonical import canonical_digest, canonical_json
from klab.config.loading import load_config, load_tokenizer_config
from klab.corpus.project import load_project
from klab.corpus.release import verify_release
from klab.data.packing import load_prepared_data
from klab.data.tokenizer import load_tokenizer, verify_tokenizer_artifact
from klab.experiments.lock import open_lock
from klab.experiments.plan import load_plan
from klab.recovery.manifest import RecoveryManifest, load_manifest
from klab.training.manifest import sha256_file


def _reference(source: Path, reference: str) -> Path:
    from klab.recovery.provenance import declaration_reference

    return declaration_reference(source, reference)


def _manifest(source: Path | RecoveryManifest) -> tuple[RecoveryManifest, Path]:
    if isinstance(source, RecoveryManifest):
        raise TypeError(
            "recovery inspection needs manifest source for relative references"
        )
    path = Path(source).absolute()
    return load_manifest(path), path


def _snapshot_inheritance(step: Any, source: Path, root: Path) -> dict[str, Any]:
    """Cold-check declared ancestry before planning or any replay mutation."""
    inheritance = step.snapshot_inheritance
    if inheritance is None:
        return {}
    from klab.recovery.implementation_replay import (
        _ANCESTRY_FORMAT,
        _ancestry_changed_declarations,
        _ancestry_parent_verified,
        verify_replay_receipt,
    )

    parent_path = root / inheritance.parent_receipt
    if parent_path.parent != root / "replay" / "receipts" or any(
        path.is_symlink() for path in (parent_path, *parent_path.parents)
    ):
        raise ValueError(
            "INVALID_PARENT_RECEIPT: parent must be task-owned and unlinked"
        )
    if sha256_file(parent_path) != inheritance.parent_receipt_sha256:
        raise ValueError("INVALID_PARENT_RECEIPT: declared parent digest mismatch")
    parent = verify_replay_receipt(parent_path)
    if parent.get("status") != "MATCH" or parent.get("format") != _ANCESTRY_FORMAT:
        raise ValueError("INVALID_PARENT_RECEIPT: successful parent required")
    project = load_project(_reference(source, step.project))
    if project.config.source_effects is not None:
        raise ValueError(
            "INVALID_ANCESTRY_REQUEST: effect-bound imports require native acquisition"
        )
    for item in project.sources:
        if item.id in inheritance.snapshots and item.kind not in {
            "git",
            "huggingface_dataset",
            "wikimedia_dump",
        }:
            raise ValueError(f"INHERITED_REUSE_UNAVAILABLE: {item.id}")
    _, inherited, _ = _ancestry_parent_verified(
        parent, tuple(inheritance.snapshots), project
    )
    if {
        key: item["snapshot_sha256"] for key, item in inherited.items()
    } != inheritance.snapshots:
        raise ValueError("INVALID_PARENT_RECEIPT: inherited snapshot identity mismatch")
    prior = parent["scientific_identity"]["snapshots"]
    if prior != {**inheritance.snapshots, **inheritance.changed_snapshots}:
        raise ValueError(
            "INVALID_PARENT_RECEIPT: expected maps must cover entire parent"
        )
    if inheritance.changed_snapshots:
        _ancestry_changed_declarations(parent, project, inheritance.changed_snapshots)
    return {
        "parent_receipt": parent_path,
        "inherited_source_ids": tuple(inheritance.snapshots),
        "expected_unchanged_snapshots": inheritance.snapshots,
        "expected_changed_snapshots": inheritance.changed_snapshots,
        "use_historical_project": True,
    }


def _pinned_inputs(manifest: RecoveryManifest, source: Path) -> list[dict[str, str]]:
    """Verify the source commit against the existing scientific declaration closure."""
    from klab.recovery.provenance import repository_root, verify_source_commit

    verified = verify_source_commit(source, manifest.source_commit)
    root = repository_root(source)
    assert root is not None
    return [
        {"path": relative, "sha256": sha256_file(root / relative)}
        for relative in verified["verified_paths"]
    ]


def _receipt_binding(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_sha256": record["manifest_sha256"],
        "source_commit": record["source_commit"],
        "declaration_hashes": record["declaration_hashes"],
        "declaration_status": record["provenance"]["status"],
        "allow_uncommitted_declaration": record["allow_uncommitted_declaration"],
        "outcomes": [
            {
                key: value
                for key, value in row.items()
                if key
                in (
                    "id",
                    "status",
                    "expected_sha256",
                    "actual_sha256",
                    "reason",
                )
            }
            for row in record["outcomes"]
        ],
    }


def verify_recovery_receipt(path: Path) -> dict[str, Any]:
    """Authenticate scientific identity and the complete observed operational record."""
    record = read_canonical(path)
    if (
        record.get("format") != "klab-recovery-receipt-v1"
        or record.get("receipt_sha256")
        != digest("klab-recovery-receipt-v1", _receipt_binding(record))
        or record.get("record_sha256")
        != hashlib.sha256(
            canonical_json(
                {key: value for key, value in record.items() if key != "record_sha256"}
            )
        ).hexdigest()
    ):
        raise ValueError(f"invalid recovery receipt: {path}")
    return record


def _export_path(root: Path, release: Path, step: Any, source: Path) -> Path:
    release_manifest = json.loads(
        (release / "manifest.json").read_text(encoding="utf-8")
    )
    release_id = release_manifest["release_id"]
    request = {
        "release_id": release_id,
        "view": step.view,
        "base_config_sha256": sha256_file(_reference(source, step.base_run)),
        "vocab_size": step.vocab_size,
    }
    identity = canonical_digest(request)
    return (
        root
        / "corpora"
        / release_manifest["corpus_id"]
        / "exports"
        / release_id
        / step.view
        / identity
    )


def _location(
    step: Any, root: Path, source: Path, paths: dict[str, Path], replay_commit: str
) -> Path | None:
    if step.kind == "corpus_release":
        project = load_project(_reference(source, step.project))
        if not step.expected_release_sha256:
            return None
        relative = (
            Path("corpora")
            / project.config.id
            / "releases"
            / step.expected_release_sha256
        )
        ordinary = root / relative
        historical = root / "replay" / "work" / replay_commit / relative
        return ordinary if ordinary.exists() or not historical.exists() else historical
    if step.kind == "corpus_export":
        if step.corpus not in paths:
            return None
        return _export_path(root, paths[step.corpus], step, source)
    if step.kind == "tokenizer_train":
        if step.export:
            if step.export not in paths:
                return None
            config = paths[step.export] / "tokenizer.yaml"
            return (
                load_tokenizer_config(config).output_dir
                if config.is_file()
                else paths[step.export] / "tokenizer"
            ) / "tokenizer.json"
        return (
            load_tokenizer_config(_reference(source, step.config)).output_dir
            / "tokenizer.json"
        )
    if step.kind == "prepared_data":
        if step.plan:
            return (
                root
                / "experiments"
                / load_plan(_reference(source, step.plan)).id
                / "preparation.json"
            )
        if step.export and step.export not in paths:
            return None
        config_path = (
            paths[step.export] / "run.yaml"
            if step.export
            else _reference(source, step.config)
        )
        if not config_path.is_file():
            return None
        cache = load_config(config_path).dataset.cache_dir
        if step.expected_manifest_sha256 and cache.is_dir():
            for entry in sorted(cache.iterdir()):
                if entry.is_symlink():
                    raise ValueError(f"unsafe symlinked prepared cache: {entry}")
                if entry.is_dir() and (entry / "manifest.json").is_file():
                    manifest = json.loads((entry / "manifest.json").read_text())
                    if manifest.get("manifest_sha256") == step.expected_manifest_sha256:
                        return entry
        return None
    if step.kind == "experiment_lock":
        plan = load_plan(_reference(source, step.plan))
        return (
            root
            / "experiments"
            / plan.id
            / "locks"
            / f"{step.expected_plan_sha256}.json"
            if step.expected_plan_sha256
            else None
        )
    if step.kind == "checkpoint":
        return root / "runs" / step.run / "checkpoints" / step.cell
    return None


def _expected(step: Any) -> str | None:
    for key in (
        "expected_release_sha256",
        "expected_export_sha256",
        "expected_tokenizer_sha256",
        "expected_manifest_sha256",
        "expected_plan_sha256",
        "expected_sha256",
    ):
        if getattr(step, key, None):
            return getattr(step, key)
    return None


def _verify(step: Any, path: Path, source: Path, paths: dict[str, Path]) -> str:
    if step.kind == "corpus_release":
        release = verify_release(path)
        if (
            step.expected_build_sha256
            and release["build_id"] != step.expected_build_sha256
        ):
            raise ValueError(
                f"EXPECTED_DIGEST_MISMATCH: build: expected "
                f"{step.expected_build_sha256}, actual {release['build_id']}"
            )
        return str(release["release_id"])
    if step.kind == "corpus_export":
        from klab.corpus.export import verify_release_export

        verify_release_export(load_config(path / "run.yaml").dataset)
        return path.name
    if step.kind == "tokenizer_train":
        config_path = (
            paths[step.export] / "tokenizer.yaml"
            if step.export
            else _reference(source, step.config)
        )
        config = load_tokenizer_config(config_path)
        verify_tokenizer_artifact(
            path,
            source=config.dataset.source,
            revision=config.dataset.revision,
            vocab_size=config.vocab_size,
            dataset=config.dataset,
        )
        return sha256_file(path)
    if step.kind == "prepared_data":
        if path.is_file():
            record = read_canonical(path)
            if (
                record.get("format") != "experiment-preparation-v1"
                or record.get("id") != load_plan(_reference(source, step.plan)).id
            ):
                raise ValueError("prepared variant record has incompatible plan")
            variant = next(
                (
                    entry
                    for entry in record["variants"]
                    if entry["id"] == step.variant_id
                ),
                None,
            )
            if variant is None:
                raise ValueError(f"prepared variant missing: {step.variant_id}")
            root = Path(variant["prepared_path"])
            if (
                sha256_file(root / "manifest.json")
                != variant["prepared_manifest_sha256"]
            ):
                raise ValueError(
                    "prepared variant record differs from prepared manifest"
                )
            config = load_config(Path(variant["export_path"]) / "run.yaml")
            load_prepared_data(
                root, byte_enabled=config.model.memory in {"byte", "portable"}
            )
            return str(
                json.loads((root / "manifest.json").read_text())["manifest_sha256"]
            )
        config_path = (
            paths[step.export] / "run.yaml"
            if step.export
            else _reference(source, step.config)
        )
        config = load_config(config_path)
        prepared = load_prepared_data(
            path, byte_enabled=config.model.memory in {"byte", "portable"}
        )
        return str(
            json.loads((prepared.root / "manifest.json").read_text())["manifest_sha256"]
        )
    if step.kind == "experiment_lock":
        return open_lock(path).plan_sha256
    if step.kind == "checkpoint":
        from klab.training.checkpoints import CheckpointManager

        report = CheckpointManager(path.parent.parent).verify(path)
        if not report.valid:
            raise ValueError(f"invalid pinned checkpoint: {report.errors}")
        from klab.training.mlx_checkpoints import strict_json

        return str(strict_json(path / "manifest.json")["sha256"])
    raise ValueError(f"step {step.kind} is not a verifiable artifact")


def inspect_manifest(source: Path, work_root: Path) -> dict[str, Any]:
    """Refresh availability without creating the selected work root or aliases."""
    manifest, source = _manifest(source)
    from klab.recovery.implementation_replay import implementation_preflight
    from klab.recovery.provenance import verify_source_commit

    implementations = {
        step.id: implementation_preflight(
            source, manifest.source_commit, _reference(source, step.project)
        )
        for step in manifest.steps
        if step.kind == "corpus_release"
    }

    missing_local: set[Path] = set()
    for step in manifest.steps:
        if step.kind == "corpus_release":
            project = load_project(_reference(source, step.project))
            for item in project.sources:
                if item.kind == "local":
                    missing_local.update(
                        project.root / entry.path
                        for entry in item.acquisition.files
                        if not (project.root / entry.path).is_file()
                    )
    if not any(item["repository"] is None for item in implementations.values()):
        verify_source_commit(
            source, manifest.source_commit, missing_local=frozenset(missing_local)
        )
    root = Path(work_root).expanduser().resolve()
    paths: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for step in manifest.steps:
        expected = _expected(step)
        if step.kind == "corpus_release" and step.snapshot_inheritance is not None:
            _snapshot_inheritance(step, source, root)
        path = _location(step, root, source, paths, manifest.source_commit)
        if path is not None:
            paths[step.id] = path
        actual = None
        if step.kind == "optional_cache":
            classification, reason, action = (
                "PRESENT",
                "disposable cache is optional",
                None,
            )
        elif step.kind == "external_required":
            classification, reason, action = "MISSING_EXTERNAL", step.reason, None
        elif path is None and step.kind != "corpus_release":
            classification, reason, action = (
                ("NOT_CREATED", "no expected output identity published", step.kind)
                if expected is None
                else (
                    "MISSING_RECONSTRUCTABLE",
                    "prior deterministic output is required",
                    step.kind,
                )
            )
        elif (
            step.kind == "corpus_release"
            and implementations[step.id]["repository"] is None
        ):
            classification, reason, action = (
                "MISSING_IMPLEMENTATION",
                "historical Corpus Forge implementation is unavailable",
                None,
            )
        elif path is not None and path.exists():
            actual = _verify(step, path, source, paths)
            if expected and expected != actual:
                raise ValueError(
                    f"EXPECTED_DIGEST_MISMATCH: {step.id}: expected {expected}, actual {actual} at {path}"
                )
            classification, reason, action = "PRESENT", "verified immutable bytes", None
        elif step.kind == "corpus_release" and (
            implementations[step.id]["status"] != "MATCH"
            or step.snapshot_inheritance is not None
        ):
            classification = (
                "PINNED_IMPLEMENTATION_REPLAY_REQUIRED"
                if step.snapshot_inheritance is not None
                else implementations[step.id]["status"]
            )
            reason = (
                "historical Corpus Forge implementation is unavailable"
                if classification == "MISSING_IMPLEMENTATION"
                else "declared snapshot inheritance requires explicit verified replay"
                if step.snapshot_inheritance is not None
                else "historical identity-producing implementation differs; explicit replay required"
            )
            action = None
        elif step.kind == "checkpoint":
            classification, reason, action = (
                (
                    "MISSING_NONRECONSTRUCTABLE",
                    "pinned checkpoint bytes are absent",
                    None,
                )
                if expected
                else ("NOT_CREATED", "no checkpoint identity published", None)
            )
        elif step.kind == "corpus_release":
            project = load_project(_reference(source, step.project))
            acquisition = root / "corpora" / project.config.id / "acquisition.json"
            if not acquisition.is_file() and any(
                item.kind not in {"local", "deterministic_generator"}
                for item in project.sources
                if item.redistribution != "rejected"
            ):
                classification, reason, action = (
                    "MISSING_EXTERNAL",
                    "pinned external corpus acquisition is unavailable",
                    None,
                )
            elif any(
                not (project.root / entry.path).is_file()
                for item in project.sources
                if item.kind == "local"
                for entry in item.acquisition.files
            ):
                classification, reason, action = (
                    "MISSING_EXTERNAL",
                    "checked-in local corpus source is unavailable",
                    None,
                )
            elif expected is None:
                classification, reason, action = (
                    "NOT_CREATED",
                    "no expected output identity published",
                    step.kind,
                )
            else:
                classification, reason, action = (
                    "MISSING_RECONSTRUCTABLE",
                    "pinned deterministic output absent",
                    step.kind,
                )
        elif expected is None:
            classification, reason, action = (
                "NOT_CREATED",
                "no expected output identity published",
                step.kind,
            )
        else:
            classification, reason, action = (
                "MISSING_RECONSTRUCTABLE",
                "pinned deterministic output absent",
                step.kind,
            )
        rows.append(
            {
                "id": step.id,
                "kind": step.kind,
                "classification": classification,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "action": action,
                "reason": reason,
                "path": str(path) if path else None,
                **(
                    {"implementation": implementations[step.id]}
                    if step.id in implementations
                    else {}
                ),
            }
        )
    from klab.workdir import storage_checks

    return {
        "format": "klab-recovery-v1",
        "id": manifest.id,
        "source_commit": manifest.source_commit,
        "work_root": str(root),
        "storage_warnings": storage_checks(root),
        "steps": rows,
    }


def plan_manifest(source: Path, work_root: Path) -> dict[str, Any]:
    """Render exact prerequisite commands, stopping at non-deterministic decisions."""
    result = inspect_manifest(source, work_root)
    manifest, source = _manifest(source)
    prefix = f"klab --work-dir {shlex.quote(result['work_root'])}"
    commands: list[dict[str, Any]] = []
    paths = {row["id"]: row["path"] for row in result["steps"]}
    for step, row in zip(manifest.steps, result["steps"], strict=True):
        if row["classification"] == "PINNED_IMPLEMENTATION_REPLAY_REQUIRED":
            commands.append(
                {
                    "step": step.id,
                    "command": (
                        f"{prefix} recovery reconstruct {shlex.quote(str(source))} "
                        "--replay-pinned-implementation"
                    ),
                    "output": row["path"],
                    "requires_explicit_review": True,
                    "requirements": {
                        "network_permission": "add --allow-network only after source-rights review",
                        "implementation": row["implementation"],
                    },
                }
            )
            break
        if row["classification"] == "MISSING_IMPLEMENTATION":
            break
        if row["classification"] in {"MISSING_EXTERNAL", "MISSING_NONRECONSTRUCTABLE"}:
            break
        if row["action"] is None:
            continue
        requirements: dict[str, Any] = {
            "storage_estimate_bytes": None,
            "network_max_bytes": 0,
            "note": "Storage depends on preceding verified artifact sizes and preparation settings",
        }
        if step.kind == "corpus_release":
            project = shlex.quote(str(_reference(source, step.project)))
            declarations = load_project(_reference(source, step.project))
            source_bound = sum(
                sum(
                    (declarations.root / entry.path).stat().st_size
                    for entry in item.acquisition.files
                )
                if item.kind == "local"
                else getattr(
                    item.acquisition,
                    "max_decompressed_bytes",
                    getattr(item.acquisition, "max_bytes", 0),
                )
                for item in declarations.sources
                if item.redistribution != "rejected"
            )
            network_bound = sum(
                getattr(
                    item.acquisition,
                    "max_bytes",
                    getattr(item.acquisition, "max_compressed_bytes", 0),
                )
                for item in declarations.sources
                if item.kind not in {"local", "deterministic_generator"}
                and item.redistribution != "rejected"
            )
            requirements = {
                "storage_estimate_bytes": source_bound * 3 or None,
                "network_max_bytes": network_bound,
                "note": "Planning estimate: 3x local input sizes or declared byte caps for snapshot/build/release; transforms may expand; null means no byte bound is available",
            }
            command = (
                f"{prefix} corpus acquire {project} && {prefix} corpus freeze "
                f'"$({prefix} corpus build {project})"'
            )
        elif step.kind == "corpus_export":
            corpus = paths[step.corpus]
            corpus_id = Path(corpus).parent.parent.name
            command = (
                f"{prefix} corpus export {corpus_id}@{Path(corpus).name} --view {step.view} "
                f"--base-run-config {shlex.quote(str(_reference(source, step.base_run)))} --vocab-size {step.vocab_size}"
            )
        elif step.kind == "tokenizer_train":
            config = (
                Path(paths[step.export]) / "tokenizer.yaml"
                if step.export
                else _reference(source, step.config)
            )
            command = f"{prefix} tokenizer train {shlex.quote(str(config))}"
        elif step.kind == "prepared_data":
            if step.plan:
                command = f"{prefix} experiment prepare {shlex.quote(str(_reference(source, step.plan)))}"
            else:
                config = (
                    Path(paths[step.export]) / "run.yaml"
                    if step.export
                    else _reference(source, step.config)
                )
                command = f"{prefix} data prepare {shlex.quote(str(config))}"
        elif step.kind == "experiment_lock":
            command = f"{prefix} experiment lock {shlex.quote(str(_reference(source, step.plan)))}"
        else:
            continue
        commands.append(
            {
                "step": step.id,
                "command": command,
                "output": row["path"],
                "requires": [
                    getattr(step, name)
                    for name in ("corpus", "export", "tokenizer", "prepared")
                    if getattr(step, name, None)
                ],
                "requirements": requirements,
            }
        )
        if _expected(step) is None:
            break  # Publish the actual new identity before planning dependent commands.
    return {**result, "commands": commands}


def reconstruct_manifest(
    source: Path,
    work_root: Path,
    *,
    allow_network: bool = False,
    allow_uncommitted_declaration: bool = False,
    evidence_output: Path | None = None,
    replay_pinned_implementation: bool = False,
) -> dict[str, Any]:
    """Replay only deterministic steps; never launch training or fabricate identities."""
    from klab.recovery.provenance import declaration_preflight

    manifest, source = _manifest(source)
    provenance = declaration_preflight(
        source, "recovery", allow_uncommitted=allow_uncommitted_declaration
    )
    root = Path(work_root).expanduser().resolve()
    before = inspect_manifest(source, root)
    first = next(
        (
            row
            for row in before["steps"]
            if row["kind"] != "optional_cache" and row["classification"] != "PRESENT"
        ),
        before["steps"][0],
    )
    for row in before["steps"]:
        if row["classification"] == "MISSING_IMPLEMENTATION":
            raise ValueError(f"MISSING_IMPLEMENTATION: {row['id']}: {row['reason']}")
        if (
            row["classification"] == "PINNED_IMPLEMENTATION_REPLAY_REQUIRED"
            and not replay_pinned_implementation
        ):
            raise ValueError(
                f"PINNED_IMPLEMENTATION_REPLAY_REQUIRED: {row['id']}: "
                "review preflight and pass --replay-pinned-implementation explicitly"
            )
    if first["classification"] == "MISSING_EXTERNAL" and (
        not allow_network
        or first["kind"] != "corpus_release"
        or first["reason"] != "pinned external corpus acquisition is unavailable"
    ):
        raise ValueError(f"MISSING_EXTERNAL: {first['reason']}")
    pinned = _pinned_inputs(manifest, source)
    from klab.workdir import ensure_work_dir, storage_checks, warn_storage_checks

    warnings = storage_checks(root)
    for step in manifest.steps:
        if step.kind == "tokenizer_train" and step.config:
            destination = load_tokenizer_config(
                _reference(source, step.config)
            ).output_dir
            warnings.extend(storage_checks(destination, kind="tokenizer_output"))
        elif step.kind == "prepared_data" and step.config:
            destination = load_config(_reference(source, step.config)).dataset.cache_dir
            warnings.extend(storage_checks(destination, kind="dataset_cache"))
    if first["classification"] == "MISSING_NONRECONSTRUCTABLE":
        raise ValueError(f"MISSING_NONRECONSTRUCTABLE: {first['reason']}")
    warn_storage_checks(warnings)
    ensure_work_dir(root)
    paths = {row["id"]: Path(row["path"]) for row in before["steps"] if row["path"]}
    outcomes: list[dict[str, Any]] = []
    replay_receipts: list[dict[str, Any]] = []
    for step, row in zip(manifest.steps, before["steps"], strict=True):
        if row["classification"] == "PRESENT":
            status = (
                "UNSEALED_RESULT"
                if _expected(step) is None
                and step.kind not in {"optional_cache", "checkpoint"}
                else "PRESENT"
            )
            outcomes.append(
                {
                    "id": step.id,
                    "status": status,
                    "expected_sha256": row["expected_sha256"],
                    "actual_sha256": row["actual_sha256"],
                    "path": row["path"],
                }
            )
            if status == "UNSEALED_RESULT":
                break
            continue
        if row["classification"] in {
            "MISSING_EXTERNAL",
            "MISSING_NONRECONSTRUCTABLE",
        } and not (
            allow_network
            and step.kind == "corpus_release"
            and row["reason"] == "pinned external corpus acquisition is unavailable"
        ):
            outcomes.append(
                {
                    "id": step.id,
                    "status": row["classification"],
                    "reason": row["reason"],
                }
            )
            break
        if step.kind == "corpus_release":
            if replay_pinned_implementation:
                from klab.recovery.implementation_replay import replay_corpus

                replay = replay_corpus(
                    source,
                    manifest.source_commit,
                    _reference(source, step.project),
                    root,
                    allow_network=allow_network,
                    expected_release_sha256=step.expected_release_sha256,
                    expected_build_sha256=step.expected_build_sha256,
                    **_snapshot_inheritance(step, source, root),
                )
                path = Path(replay["path"])
                replay_receipts.append(
                    {
                        "step": step.id,
                        "path": replay["receipt_path"],
                        "receipt": replay["receipt"],
                    }
                )
                actual = _verify(step, path, source, paths)
                if _expected(step) and actual != _expected(step):
                    raise ValueError(
                        f"EXPECTED_DIGEST_MISMATCH: {step.id}: expected "
                        f"{_expected(step)}, actual {actual}"
                    )
                paths[step.id] = path
                outcomes.append(
                    {
                        "id": step.id,
                        "status": "PRESENT" if _expected(step) else "UNSEALED_RESULT",
                        "expected_sha256": _expected(step),
                        "actual_sha256": actual,
                        "path": str(path),
                    }
                )
                if not _expected(step):
                    break
                continue
            from klab.corpus.acquisition import acquire
            from klab.corpus.pipeline import build
            from klab.corpus.release import freeze

            project = load_project(_reference(source, step.project))
            remote = any(
                item.kind not in {"local", "deterministic_generator"}
                for item in project.sources
                if item.redistribution != "rejected"
            )
            if (
                remote
                and not allow_network
                and not (
                    root / "corpora" / project.config.id / "acquisition.json"
                ).is_file()
            ):
                outcomes.append(
                    {
                        "id": step.id,
                        "status": "MISSING_EXTERNAL",
                        "reason": "NETWORK_PERMISSION_REQUIRED",
                    }
                )
                break
            acquire(project, root, offline=remote and not allow_network)
            build_path = build(project, root, offline=True)
            if (
                step.expected_build_sha256
                and build_path.name != step.expected_build_sha256
            ):
                raise ValueError(
                    f"EXPECTED_DIGEST_MISMATCH: build: expected "
                    f"{step.expected_build_sha256}, actual {build_path.name} at {build_path}"
                )
            path = freeze(build_path, root)
        elif step.kind == "corpus_export":
            from klab.corpus.export import export_release

            path = export_release(
                paths[step.corpus],
                step.view,
                _reference(source, step.base_run),
                step.vocab_size,
                root,
            )
        elif step.kind == "tokenizer_train":
            from klab.data.tokenizer import train_tokenizer

            config = (
                paths[step.export] / "tokenizer.yaml"
                if step.export
                else _reference(source, step.config)
            )
            path = train_tokenizer(load_tokenizer_config(config))
        elif step.kind == "prepared_data":
            from klab.data.packing import prepare_data
            from klab.experiments.prepare import prepare_plan

            if step.plan:
                plan_path = _reference(source, step.plan)
                plan = load_plan(plan_path)
                try:
                    record = prepare_plan(
                        plan, plan_path, root / "experiments" / plan.id
                    )
                finally:
                    ensure_work_dir(root)
                variant = next(
                    (
                        entry
                        for entry in record["variants"]
                        if entry["id"] == step.variant_id
                    ),
                    None,
                )
                if variant is None:
                    raise ValueError(f"unknown prepared variant: {step.variant_id}")
                path = (
                    root / "experiments" / load_plan(plan_path).id / "preparation.json"
                )
                if path.exists():
                    previous = read_canonical(path)
                    if any(
                        previous.get(key) != record[key]
                        for key in ("format", "id", "variants")
                    ):
                        raise ValueError(f"conflicting preparation record: {path}")
                else:
                    publish_immutable(path, record)
                actual = _verify(step, path, source, paths)
            else:
                config = (
                    paths[step.export] / "run.yaml"
                    if step.export
                    else _reference(source, step.config)
                )
                path = prepare_data(
                    load_config(config), load_tokenizer(paths[step.tokenizer])
                ).root
                actual = _verify(step, path, source, paths)
        elif step.kind == "experiment_lock":
            from klab.experiments.lock import publish_lock, resolve_plan

            plan_path = _reference(source, step.plan)
            plan = load_plan(plan_path)
            prepared = None
            if (
                step.prepared
                and next(s for s in manifest.steps if s.id == step.prepared).plan
            ):
                prepared = read_canonical(paths[step.prepared])
            path = publish_lock(
                resolve_plan(plan, plan_path, prepared=prepared),
                root / "experiments" / plan.id,
            )
        else:
            break
        paths[step.id] = path
        actual = (
            actual
            if step.kind == "prepared_data"
            else _verify(step, path, source, paths)
        )
        expected = _expected(step)
        status = (
            "EXPECTED_DIGEST_MISMATCH"
            if expected and actual != expected
            else "UNSEALED_RESULT"
            if expected is None
            else "PRESENT"
        )
        outcomes.append(
            {
                "id": step.id,
                "status": status,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "path": str(path),
            }
        )
        if status != "PRESENT":
            break
    receipt_body = {
        "format": "klab-recovery-receipt-v1",
        "manifest_sha256": sha256_file(source),
        "source_commit": manifest.source_commit,
        "declaration_hashes": pinned,
        "provenance": provenance,
        "allow_uncommitted_declaration": allow_uncommitted_declaration,
        "work_root": str(root),
        "storage_warnings": warnings,
        "outcomes": outcomes,
        "implementation_replays": replay_receipts,
    }
    scientific_binding = _receipt_binding(receipt_body)
    receipt_id = digest("klab-recovery-receipt-v1", scientific_binding)
    receipt_path = root / "recovery" / manifest.id / f"{receipt_id}.json"
    if receipt_path.exists():
        receipt = verify_recovery_receipt(receipt_path)
        if (
            receipt["receipt_sha256"] != receipt_id
            or _receipt_binding(receipt) != scientific_binding
            or receipt["work_root"] != str(root)
            or receipt["outcomes"] != outcomes
        ):
            raise ValueError(f"conflicting recovery receipt: {receipt_path}")
    else:
        receipt = {
            **receipt_body,
            "receipt_sha256": receipt_id,
            "issued_at_utc": utc_now(),
        }
        receipt["record_sha256"] = canonical_digest(receipt)
        publish_immutable(receipt_path, receipt)
    if evidence_output is not None:
        publish_immutable(Path(evidence_output), receipt)
    return {
        **inspect_manifest(source, root),
        "outcomes": outcomes,
        "receipt": receipt,
        "storage_warnings": warnings,
        "provenance": provenance,
    }
