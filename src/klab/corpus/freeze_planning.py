"""Metadata-only plan for ordinary Corpus freeze and shared payload placement."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
from pathlib import Path
from typing import Any

from klab.canonical import canonical_digest
from klab.corpus.normalized_spans import (
    NORMALIZED_SPANS,
    SPAN_MEMBER,
    has_span_projection,
    span_dependencies,
    span_partitions,
)
from klab.corpus.preparation_inputs import local_scope
from klab.corpus.project import NormalizerOptions
from klab.corpus.release import (
    ReleaseMembers,
    consumed_release,
    release_index_workspace,
    release_manifest_payload,
    release_row_index_inputs,
    streaming_verification,
)
from klab.corpus.release_members import (
    DOCUMENT_MEMBERS,
    SHARED_STAGE,
    ReleasePlacement,
    shared_members,
)
from klab.corpus.release_receipt import lookup
from klab.corpus.release_verify import (
    current_settings,
    scratch_location,
    verification_scratch_observation,
)
from klab.experiments.artifacts import snapshot_dependencies
from klab.safe_io import symlinked
from klab.verification_proofs import verification_options

_DIGEST = re.compile(r"[0-9a-f]{64}")


def _metadata(path: Path) -> dict[str, Any]:
    if symlinked(path) or not path.is_file():
        raise ValueError(f"missing or unsafe metadata file: {path}")
    return json.loads(path.read_bytes())


def _payload_stat(
    path: Path, size: int, *, shared: bool = False
) -> os.stat_result | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if (
        not stat.S_ISREG(info.st_mode)
        or symlinked(path)
        or info.st_size != size
        or (shared and info.st_nlink != 1)
    ):
        raise ValueError(f"unsafe payload or changed size: {path}")
    return info


def _device(path: Path) -> int:
    while not path.exists():
        path = path.parent
    return path.stat().st_dev


def _shared_placement(
    root: Path,
    destination: Path,
    payload: dict[str, Any],
    *,
    state: str,
    projected_spans: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    files = payload["files"]
    members = shared_members(destination, files)
    totals = dict.fromkeys(
        ("new_shared_bytes", "shared_candidate_bytes", "duplicate_staging_bytes"), 0
    )
    scheduled: set[Path] = set()
    actions = []
    inline = ReleaseMembers(root, payload)
    parts = span_partitions(destination, files)
    external = set(members)
    if parts:
        external.add(SPAN_MEMBER)
        span_dependencies(parts)
        entry = files[SPAN_MEMBER]
        for part in parts:
            if _payload_stat(part.spans, part.entry["size"], shared=True) is None:
                raise ValueError(f"missing normalized spans: {part.spans}")
        size = entry["size"]
        original = _payload_stat(root / SPAN_MEMBER, size)
        if state == "build" and original is None and not projected_spans:
            raise ValueError("missing build span payload")
        if (state == "published" or projected_spans) and original is not None:
            raise ValueError("referenced spans have an unexpected flat payload")
        if original is not None:
            totals["duplicate_staging_bytes"] += size
        actions.append(
            {
                "member": SPAN_MEMBER,
                "sha256": entry["sha256"],
                "storage": NORMALIZED_SPANS,
                "bytes": size,
                "partitions": len(parts),
                "action": "verify_published"
                if state == "published"
                else "verify_build_reference"
                if state == "build" and projected_spans
                else (
                    "reference_normalized_spans"
                    if original is not None
                    else "verify_staged_reference"
                ),
            }
        )
    for name in files.keys() - external:
        if _payload_stat(inline.member(name), files[name]["size"]) is None:
            raise ValueError(f"missing inline payload: {name}")
    for name, target in members.items():
        size = files[name]["size"]
        source = root / name
        original = _payload_stat(source, size, shared=True)
        stored = _payload_stat(target, size, shared=True)
        if state == "build" and original is None:
            raise ValueError(f"missing build payload: {name}")
        if state == "published":
            if stored is None or original is not None:
                raise ValueError(
                    f"published shared member has different placement: {name}"
                )
            action = "verify_published"
        elif stored is not None:
            action = (
                "verify_candidate"
                if original is not None
                else "verify_staged_reference"
            )
            if target not in scheduled:
                totals["shared_candidate_bytes"] += size
        elif original is not None:
            if original.st_dev != _device(target.parent):
                raise ValueError(
                    f"shared member move requires the same filesystem: {name}"
                )
            if target in scheduled:
                action = "reuse_after_move"
            else:
                action = "move_to_shared_store"
                totals["new_shared_bytes"] += size
        else:
            raise ValueError(f"missing staged and shared copies: {name}")
        if original is not None and action in {"verify_candidate", "reuse_after_move"}:
            totals["duplicate_staging_bytes"] += size
        scheduled.add(target)
        actions.append(
            {
                "member": name,
                "sha256": target.name,
                "storage": files[name]["storage"],
                "bytes": size,
                "action": action,
            }
        )
    return actions, totals


def _retained_row_indexes(
    root: Path, scratch: Path, manifest: dict[str, Any], identity: dict[str, Any]
) -> list[dict[str, Any]]:
    from klab.corpus.build_stages import StageStore
    from klab.corpus.planning import plan_stage_candidate

    workspace = release_index_workspace(root, scratch)
    if not workspace.exists() and not symlinked(workspace):
        return []
    trust = verification_options(scratch, read_only=True)["proof_store"]
    if trust is None:
        # The verifier uses disposable scratch when the selected root cannot
        # retain owned checkpoints.
        return []
    if not trust.trusted(workspace.parent) or not trust.trusted(workspace):
        raise ValueError(f"unsafe release index workspace: {workspace}")
    members = ReleaseMembers(root, manifest)
    keys = release_row_index_inputs(
        {name: members.member_paths(name) for name in DOCUMENT_MEMBERS},
        manifest["files"],
        identity,
        NormalizerOptions.model_validate(identity["release"]["normalizer"]),
        checkpointed=True,
    )
    store = StageStore(workspace)
    return [
        {
            "id": stage,
            "when": "if_document_index_rebuilt" if stage == "spans" else "first",
            **plan_stage_candidate(store, stage, inputs)[0],
        }
        for stage, inputs in zip(("spans", "documents"), keys, strict=True)
    ]


def _verification_step(
    step: str,
    root: Path,
    manifest: dict[str, Any],
    scratch: Path,
    *,
    placement_pending: bool,
) -> dict[str, Any]:
    settings = current_settings()
    kind = step.removeprefix("verify_")
    build = kind == "build"
    identity = manifest["identity" if build else "build_identity"]
    artifact_id = manifest["build_id" if build else "release_id"]
    streaming = streaming_verification(identity)
    result = {
        "id": step,
        "row_verifier": "streaming" if streaming else "row_store",
        "scratch_bytes": None,
        "row_indexes": [],
    }
    if observed := verification_scratch_observation(scratch, artifact_id):
        result["prior_scratch_observation"] = observed
    if placement_pending:
        return {**result, "method": "after_placement"}

    if (
        settings.proof_store is not None
        and settings.verification_mode == "verified_reuse"
    ):
        filename = "build.json" if build else "manifest.json"

        def dependency_paths():
            return list(snapshot_dependencies(root, filename))

        if lookup(
            settings.proof_store,
            kind,
            root,
            artifact_id,
            dependency_paths,
            skip_index_on_missing=True,
        ).hit:
            return {**result, "method": "receipt", "scratch_bytes": 0}
    if (
        streaming
        and settings.scratch_dir is not None
        and settings.verification_mode == "verified_reuse"
    ):
        result["row_indexes"] = _retained_row_indexes(root, scratch, manifest, identity)
    return {**result, "method": "verify_bytes"}


def _inspect_verification(
    steps: list[dict[str, Any]],
    build: Path,
    root: Path,
    manifest: dict[str, Any],
    release: dict[str, Any],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    scratch = scratch_location().absolute()
    anchor = scratch
    while not anchor.exists() and not anchor.is_symlink():
        anchor = anchor.parent
    if not anchor.is_dir():
        raise ValueError(f"scratch location is not a directory: {anchor}")
    forecasts = []
    for step in (item["id"] for item in steps if item["id"].startswith("verify_")):
        placement_pending = step == "verify_staged_release" and (
            not (root / "manifest.json").is_file()
            or (root / "build.json").exists()
            or any(row["action"] != "verify_staged_reference" for row in actions)
        )
        verify_build = step == "verify_build"
        forecasts.append(
            _verification_step(
                step,
                build if verify_build else root,
                manifest if verify_build else release,
                scratch,
                placement_pending=placement_pending,
            )
        )
    return {
        "mode": current_settings().verification_mode,
        "scratch_path": str(scratch),
        "scratch_available_bytes": shutil.disk_usage(anchor).free,
        "steps": forecasts,
    }


def plan_freeze(
    build: Path,
    work_root: Path,
    *,
    placement: ReleasePlacement,
    inspect_verification: bool = False,
) -> dict[str, Any]:
    """Show release identity, placement and resume work before reading payload bytes."""
    base: dict[str, Any] = {"operation": "freeze", "verification": "advisory"}
    try:
        build = build.resolve()
        work_root = work_root.resolve()
        if (
            build.parent.name != "builds"
            or build.parent.parent.parent != work_root / "corpora"
            or not _DIGEST.fullmatch(build.name)
        ):
            raise ValueError(
                "freeze planning requires a build path in the selected work root"
            )
        releases = build.parent.parent / "releases"
        staging = releases / f".freezing-{build.name}"
        if symlinked(staging):
            raise ValueError("symlinked release staging path")
        root = staging if staging.exists() else build
        reference = root / "build.json"
        manifest = _metadata(reference if reference.exists() else build / "build.json")
        if (
            manifest["schema_version"] != 1
            or manifest["build_id"] != build.name
            or canonical_digest(manifest["identity"]) != build.name
            or manifest["identity"]["project_id"] != build.parent.parent.name
        ):
            raise ValueError("build manifest identity mismatch")
        if local_scope(manifest) is not None:
            return {
                **base,
                "status": "unsupported",
                "reason": "completed local preparation",
            }

        payload = release_manifest_payload(manifest, placement=placement)
        release_id = canonical_digest(payload)
        marker = build / "consumed.json"
        if symlinked(marker):
            raise ValueError("symlinked consumed release marker")
        consumed = consumed_release(build)
        state = "staging" if root == staging else "build"
        if state == "staging":
            if consumed is not None and consumed != release_id:
                raise ValueError(
                    "retry the staged release with the same freeze options"
                )
            sealed = staging / "manifest.json"
            if sealed.exists() and _metadata(sealed) != {
                **payload,
                "release_id": release_id,
            }:
                raise ValueError(
                    "staged release manifest differs from planned placement"
                )
        else:
            if consumed is not None:
                if not isinstance(consumed, str) or not _DIGEST.fullmatch(consumed):
                    raise ValueError("invalid consumed release identity")
                if placement.shared and consumed != release_id:
                    raise ValueError(
                        "build was already frozen with different shared placement"
                    )
                release_id = consumed
            existing = releases / release_id
            if consumed is not None or existing.exists():
                sealed = _metadata(existing / "manifest.json")
                payload = release_manifest_payload(
                    manifest, placement=ReleasePlacement.from_inventory(sealed["files"])
                )
                if (
                    sealed != {**payload, "release_id": release_id}
                    or canonical_digest(payload) != release_id
                ):
                    raise ValueError(
                        "published release manifest differs from its build"
                    )
                root, state = existing, "published"

        if state == "build" and root.stat().st_dev != _device(releases):
            raise ValueError("release publication requires the same filesystem")
        actions, totals = _shared_placement(
            root,
            releases / release_id,
            payload,
            state=state,
            projected_spans=has_span_projection(manifest["files"]),
        )
        verify = {
            "build": "verify_build",
            "staging": "verify_staged_release",
            "published": "verify_release",
        }[state]
        steps = [{"id": verify, "inputs": []}]
        if state == "published" and consumed is None:
            steps = [
                {"id": "verify_build", "inputs": []},
                {"id": verify, "inputs": ["verify_build"]},
            ]
        previous = verify
        if state != "published":
            if actions:
                share = (
                    "place_shared_stages"
                    if all(row["storage"] == SHARED_STAGE for row in actions)
                    else "place_shared_members"
                )
                if state == "staging":
                    steps = [
                        {"id": share, "inputs": []},
                        {"id": verify, "inputs": [share]},
                    ]
                else:
                    steps.append({"id": share, "inputs": [verify]})
                    previous = share
            steps.append({"id": "publish_release", "inputs": [previous]})
        verification = {}
        scratch_bytes = None
        if inspect_verification:
            inspection = _inspect_verification(
                steps,
                build,
                root,
                manifest,
                {**payload, "release_id": release_id},
                actions,
            )
            verification["verification_inspection"] = inspection
            if all(step["method"] == "receipt" for step in inspection["steps"]):
                scratch_bytes = 0
        return {
            **base,
            "status": "planned",
            "state": state,
            "build_id": build.name,
            "release_id": release_id,
            "stages": steps,
            "shared_members": actions,
            **totals,
            "publication_payload_copy_bytes": 0,
            "verification_scratch_bytes": scratch_bytes,
            **verification,
        }
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {**base, "status": "blocked", "reason": str(error)}
