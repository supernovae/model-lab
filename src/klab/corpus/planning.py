"""Read-only work estimates for native Corpus builds and their retained stages."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from klab.canonical import canonical_digest
from klab.corpus.acquisition import declaration_sha256
from klab.corpus.bounded_metadata import read_inventory_metadata
from klab.corpus.build_stages import RECEIPT, StageStore
from klab.corpus.large_build import (
    native_dedup_inputs,
    native_file_rights,
    native_index_inputs,
    native_prepared_members,
    native_view_checkpoints,
    native_view_members,
)
from klab.corpus.pipeline import build_identity, uses_disk_lm_builder
from klab.corpus.prepared_shards import normalized_store_root, prepared_file_directory
from klab.corpus.project import (
    Project,
    SourceDeclaration,
    acquisition_identity_payload,
    safe_name,
)
from klab.corpus.release import consumed_release
from klab.corpus.shards import range_file_identity
from klab.safe_io import symlinked

_ELIGIBLE = {"eligible", "eligible_with_obligations"}
_RIGHTS_METADATA_BYTES = 4 * 1024**2
_STAGE_RECEIPT_BYTES = 4 * 1024**2
_DIGEST = re.compile(r"[0-9a-f]{64}")
_VIEW_COMMIT = re.compile(r"commit-([0-9]{12})\.json")


def _acquisition_inventory(
    project: Project, work_root: Path, lock_path: Path, entries: dict[str, Any]
) -> Iterator[tuple[SourceDeclaration, Path, str, list[dict[str, Any]]]]:
    effects = {
        effect.source_id: effect for effect in project.config.source_effects or ()
    }
    for source in sorted(project.sources, key=lambda item: item.id):
        entry = entries[source.id]
        pin = entry["snapshot_sha256"]
        if not isinstance(pin, str) or _DIGEST.fullmatch(pin) is None:
            raise ValueError(f"invalid snapshot pin: {source.id}")
        expected = lock_path.parent / "snapshots" / source.id / pin
        snapshot = expected.resolve()
        effect = effects.get(source.id)
        if (
            entry["snapshot_path"] != str(snapshot)
            or snapshot.name != pin
            or not snapshot.is_relative_to(work_root / "corpora")
            or (effect is not None and symlinked(expected))
        ):
            raise ValueError(f"invalid snapshot path: {source.id}")
        origin = entry.get("reuse_origin")
        if effect is not None and effect.effect == "reuse_only":
            expected_origin = (
                work_root
                / "corpora"
                / effect.origin_project_id
                / "snapshots"
                / source.id
                / pin
            )
            if pin != effect.snapshot_sha256 or origin != {
                "project_id": effect.origin_project_id,
                "snapshot_sha256": pin,
                "declaration_sha256": effect.declaration_sha256,
                "path": str(expected_origin),
            }:
                raise ValueError(f"retained origin binding differs: {source.id}")
        elif origin is not None:
            raise ValueError(f"undeclared retained origin: {source.id}")
        manifest = read_inventory_metadata(snapshot / "manifest.json")
        if (
            manifest["schema_version"] != 1
            or manifest["source_id"] != source.id
            or manifest["snapshot_sha256"] != pin
            or manifest["adapter"]["id"] != source.kind
            or manifest["declaration_sha256"] != entry["declaration_sha256"]
            or entry["declaration_sha256"] != declaration_sha256(source)
            or manifest["declaration"] != acquisition_identity_payload(source)
        ):
            raise ValueError(f"snapshot metadata mismatch: {source.id}")
        yield source, snapshot, pin, manifest["files"]


def _source_plan(
    project: Project,
    source: SourceDeclaration,
    snapshot: Path,
    pin: str,
    inventory: list[dict[str, Any]],
    normalized_root: Path,
    prepared: list[Path],
    normalization_implementation: dict[str, str],
) -> dict[str, Any]:
    files_root = snapshot / "files"
    nested_path = source.rights.nested_metadata_path if source.rights else None
    nested_metadata = {}
    if nested_path is not None:
        metadata = files_root / safe_name(nested_path)
        if symlinked(metadata) or not metadata.is_file():
            raise ValueError(f"unsafe rights metadata: {source.id}/{nested_path}")
        with metadata.open("rb") as stream:
            raw = stream.read(_RIGHTS_METADATA_BYTES + 1)
        if len(raw) > _RIGHTS_METADATA_BYTES:
            raise ValueError(f"rights metadata header cap exceeded: {source.id}")
        nested_metadata[nested_path] = json.loads(raw)

    if symlinked(normalized_root / source.id):
        raise ValueError(f"symlinked normalized namespace: {source.id}")
    partitions = []
    seen = set()
    for file in sorted(inventory, key=lambda item: item["path"]):
        name = safe_name(file["path"])
        size = file["size"]
        digest = file["sha256"]
        if (
            name in seen
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or _DIGEST.fullmatch(digest) is None
        ):
            raise ValueError(f"invalid snapshot inventory row: {source.id}/{name}")
        seen.add(name)
        path = files_root / name
        if symlinked(path) or not path.is_file() or path.stat().st_size != size:
            raise ValueError(
                f"snapshot file missing or changed size: {source.id}/{name}"
            )
        key = None
        if name in nested_metadata:
            action, reason = "skip", "rights metadata"
        else:
            decision = native_file_rights(project, source, name, path, nested_metadata)
            if decision is not None and decision.training_eligibility not in _ELIGIBLE:
                action, reason = "skip", f"rights {decision.training_eligibility}"
            else:
                directory = prepared_file_directory(
                    normalized_root,
                    source,
                    pin,
                    file,
                    project.release.normalizer,
                    project.release.policy.documents,
                    decision,
                    None,
                    implementation=normalization_implementation,
                )
                key = directory.name
                staging = directory.with_name(f".incomplete-{key}")
                if directory.exists() or directory.is_symlink():
                    receipt = directory / RECEIPT
                    if symlinked(receipt) or not receipt.is_file():
                        raise ValueError(
                            f"missing or unsafe normalized receipt: {receipt}"
                        )
                    prepared.append(directory)
                    action, reason = "verify_candidate", "published normalized receipt"
                elif (staging / RECEIPT).is_file():
                    action, reason = "resume_candidate", "staged normalized receipt"
                else:
                    action, reason = "normalize", "no matching normalized receipt"
        partitions.append(
            {
                "path": name,
                "source_bytes": size,
                "computation_key": key,
                "action": action,
                "reason": reason,
            }
        )
    return {"source_id": source.id, "snapshot_sha256": pin, "partitions": partitions}


def plan_stage_candidate(
    store: StageStore, name: str, inputs: dict[str, Any]
) -> tuple[dict[str, Any], str | None]:
    """Inspect a matching stage's metadata and return its candidate receipt digest."""
    key = store.key(inputs)
    target = store.directory(name, key)
    checkpoint = store.root / f".{name}-{key}.groups"
    assembly = checkpoint / "assembly"
    candidate = (
        assembly
        if name == "index" and not target.exists() and (assembly / RECEIPT).exists()
        else target
    )
    estimate: dict[str, Any] = {
        "computation_key": key,
        "verification_candidate_bytes": None,
    }
    receipt = candidate / RECEIPT
    try:
        if symlinked(receipt) or symlinked(checkpoint):
            raise ValueError("symlinked stage metadata")
        if not receipt.is_file():
            if candidate.exists():
                raise ValueError("missing stage receipt")
            if checkpoint.is_dir():
                return {**estimate, "action": "resume_candidate"}, None
            return {**estimate, "action": "run"}, None
        with receipt.open("rb") as stream:
            raw = stream.read(_STAGE_RECEIPT_BYTES + 1)
        if len(raw) > _STAGE_RECEIPT_BYTES:
            return {
                **estimate,
                "action": "resolve",
                "reason": "stage receipt exceeds planning limit",
            }, None
        metadata = json.loads(raw)
        files = metadata["files"]
        if (
            metadata["stage"] != name
            or metadata["key"] != key
            or canonical_digest(metadata["inputs"]) != key
            or {item.name for item in candidate.iterdir()} != {RECEIPT, *files}
        ):
            raise ValueError("stage receipt inventory differs")
        size = 0
        for filename, entry in files.items():
            path = candidate / filename
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != entry["size"]
            ):
                raise ValueError("stage member missing or changed size")
            size += entry["size"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {**estimate, "action": "resolve", "reason": str(error)}, None
    return {
        **estimate,
        "action": "verify_candidate" if candidate == target else "resume_candidate",
        "verification_candidate_bytes": size,
    }, hashlib.sha256(raw).hexdigest()


def _view_candidate(
    workspace: Path,
    build_id: str,
    stage_id: str,
    prepared: list[Path],
    *,
    index_receipt_sha256: str,
    dedup_receipt_sha256: str,
) -> dict[str, Any]:
    estimate: dict[str, Any] = {"verification_candidate_bytes": None}
    try:
        generations = []
        for path in native_prepared_members(prepared):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"missing or unsafe normalized member: {path}")
            generations.append(range_file_identity(path))
        checkpoint = native_view_checkpoints(
            workspace,
            build_id,
            index_receipt_sha256=index_receipt_sha256,
            dedup_receipt_sha256=dedup_receipt_sha256,
            generations=generations,
        )
        estimate["computation_key"] = checkpoint.identity
        root = checkpoint.root
        if symlinked(root):
            raise ValueError("symlinked view checkpoint")
        if not root.exists():
            return {**estimate, "action": "run"}

        before = range_file_identity(root)
        count = 0
        latest = -1
        with os.scandir(root) as entries:
            for entry in entries:
                if not (
                    entry.name.startswith("commit-") and entry.name.endswith(".json")
                ):
                    continue
                match = _VIEW_COMMIT.fullmatch(entry.name)
                if match is None or not entry.is_file(follow_symlinks=False):
                    raise ValueError("unsafe view checkpoint record")
                count += 1
                latest = max(latest, int(match[1]))
        if count != latest + 1:
            raise ValueError("view checkpoint sequence has gaps")
        if not count:
            return {**estimate, "action": "run", "reason": "no committed view prefix"}

        # Opening only the last record bounds metadata reads independently
        # of output size. Resume still authenticates every record and range.
        entry = read_inventory_metadata(root / f"commit-{latest:012d}.json")
        body = entry["body"]
        outputs = body["outputs"]
        previous = body["previous"]
        rows = body["state"]["count"]
        if (
            canonical_digest(body) != entry["sha256"]
            or body["identity"] != checkpoint.identity
            or outputs.keys() != set(native_view_members(stage_id))
            or (count == 1 and previous is not None)
            or (
                count > 1
                and not (isinstance(previous, str) and _DIGEST.fullmatch(previous))
            )
            or type(rows) is not int
            or rows < 0
        ):
            raise ValueError("invalid view checkpoint frontier")

        prefix = tail = 0
        for name, member in outputs.items():
            end = member["end"]
            path = root / "build" / name
            digest = member["prefix_sha256"]
            if (
                type(end) is not int
                or end < 0
                or not isinstance(digest, str)
                or not _DIGEST.fullmatch(digest)
                or symlinked(path)
                or not path.is_file()
            ):
                raise ValueError(f"unsafe or missing view checkpoint member: {name}")
            size = path.stat().st_size
            if size < end:
                raise ValueError(
                    f"view checkpoint member is shorter than its prefix: {name}"
                )
            prefix += end
            tail += size - end
        if range_file_identity(root) != before:
            raise ValueError("view checkpoint changed during planning")
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
        return {**estimate, "action": "resolve", "reason": str(error)}
    return {
        **estimate,
        "action": "resume_candidate",
        "checkpoint_records": count,
        "committed_document_rows": rows,
        "verification_candidate_bytes": prefix,
        "stage_replay_candidate_bytes": outputs[f"stages/{stage_id}.jsonl"]["end"],
        "uncommitted_tail_bytes": tail,
    }


def plan_build(project: Project, work_root: Path) -> dict[str, Any]:
    """Plan a full native build using acquisition inventory and retained metadata."""
    lock = work_root / "corpora" / project.config.id / "acquisition.json"
    normalized_root = normalized_store_root(work_root)
    plan: dict[str, Any] = {
        "project_id": project.config.id,
        "verification": "advisory",
        "stages": [
            {"id": "acquire", "inputs": []},
            {"id": "normalize", "inputs": ["acquire"]},
            {"id": "index", "inputs": ["normalize"]},
            {"id": "deduplicate", "inputs": ["index"]},
            {"id": "views", "inputs": ["index", "deduplicate"]},
            {"id": "freeze", "inputs": ["views"]},
        ],
        "sources": [],
        "build_id": None,
        "build_state": None,
        "normalization_source_bytes": None,
        "verification_candidate_source_bytes": None,
        "build_scratch_bytes": None,
    }
    if not uses_disk_lm_builder(project, list(project.transforms)):
        return {
            **plan,
            "status": "unsupported",
            "reason": "build uses the in-memory builder",
        }
    if project.release.record_admission is not None:
        return {
            **plan,
            "status": "unsupported",
            "reason": "build requires reviewed row admission",
        }
    if not lock.exists() and not symlinked(lock):
        return {
            **plan,
            "status": "acquisition_required",
            "reason": "acquisition lock missing",
        }
    try:
        acquisition = read_inventory_metadata(lock)
        entries = acquisition["sources"]
        if (
            acquisition["schema_version"] != 1
            or acquisition["project_id"] != project.config.id
            or set(entries) != {source.id for source in project.sources}
        ):
            raise ValueError("acquisition source inventory does not match project")
        identity, _, _ = build_identity(project, entries)
        build_id = canonical_digest(identity)
        target = lock.parent / "builds" / build_id
        stage_by_id = {stage["id"]: stage for stage in plan["stages"]}
        stage_by_id["acquire"]["action"] = "verify_pinned"
        if symlinked(target):
            raise ValueError("symlinked build path")
        if target.exists():
            manifest_path = target / "build.json"
            marker_path = target / "consumed.json"
            if symlinked(manifest_path) or symlinked(marker_path):
                raise ValueError("symlinked build metadata")
            manifest = read_inventory_metadata(manifest_path)
            if (
                manifest["schema_version"] != 1
                or manifest["build_id"] != build_id
                or canonical_digest(manifest["identity"]) != build_id
            ):
                raise ValueError("build manifest identity mismatch")
            for _ in _acquisition_inventory(project, work_root, lock, entries):
                pass
            release_id = consumed_release(target)
            if release_id is not None and (
                not isinstance(release_id, str) or not _DIGEST.fullmatch(release_id)
            ):
                raise ValueError("invalid consumed release identity")
            for name in ("normalize", "index", "deduplicate", "views"):
                stage_by_id[name]["action"] = "skip_existing_build"
            plan["stages"][-1] = {
                "id": "freeze",
                "inputs": ["acquire"],
                "action": "verify_release" if release_id else "verify_build",
            }
            return {
                **plan,
                "status": "planned",
                "build_id": build_id,
                "build_state": "consumed" if release_id else "published",
                "release_id": release_id,
                "normalization_source_bytes": 0,
                "verification_candidate_source_bytes": 0,
            }

        prepared: list[Path] = []
        sources = [
            _source_plan(
                project,
                source,
                snapshot,
                pin,
                files,
                normalized_root,
                prepared,
                identity["normalization_implementation"],
            )
            for source, snapshot, pin, files in _acquisition_inventory(
                project, work_root, lock, entries
            )
        ]
        source_bytes = dict.fromkeys(
            ("normalize", "verify_candidate", "resume_candidate", "skip"), 0
        )
        for source_plan in sources:
            for partition in source_plan["partitions"]:
                source_bytes[partition["action"]] += partition["source_bytes"]
        pending = any(
            row["action"] in {"normalize", "resume_candidate"}
            for source in sources
            for row in source["partitions"]
        )
        stage_by_id["normalize"]["action"] = "resolve_partitions"
        stage_by_id["index"]["action"] = "await_normalization"
        stage_by_id["deduplicate"]["action"] = "await_index"
        stage_by_id["views"]["action"] = "await_deduplication"
        stage_by_id["freeze"]["action"] = "plan_after_build"
        if not pending:
            store = StageStore(target.parent / "stages")
            index, index_sha = plan_stage_candidate(
                store, "index", native_index_inputs(project, prepared)
            )
            stage_by_id["index"].update(index)
            if index_sha is not None:
                dedup, dedup_sha = plan_stage_candidate(
                    store, "dedup", native_dedup_inputs(project, index_sha)
                )
                stage_by_id["deduplicate"].update(dedup)
                if dedup_sha is not None:
                    stage_by_id["views"].update(
                        _view_candidate(
                            lock.parent,
                            build_id,
                            project.transforms[0].id,
                            prepared,
                            index_receipt_sha256=index_sha,
                            dedup_receipt_sha256=dedup_sha,
                        )
                    )
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {**plan, "status": "blocked", "reason": str(error)}
    return {
        **plan,
        "status": "planned",
        "build_id": build_id,
        "build_state": "new",
        "sources": sources,
        "normalization_source_bytes": source_bytes["normalize"],
        "verification_candidate_source_bytes": source_bytes["verify_candidate"]
        + source_bytes["resume_candidate"],
    }
