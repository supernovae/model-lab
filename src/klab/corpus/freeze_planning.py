"""Metadata-only plan for ordinary Corpus freeze and shared payload placement."""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from klab.canonical import canonical_digest
from klab.corpus.preparation_inputs import local_scope
from klab.corpus.release import (
    ReleaseMembers,
    consumed_release,
    release_manifest_payload,
)
from klab.corpus.release_members import SHARED_STAGE, ReleasePlacement, shared_members
from klab.safe_io import symlinked

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
    root: Path, destination: Path, payload: dict[str, Any], *, state: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    files = payload["files"]
    members = shared_members(destination, files)
    totals = dict.fromkeys(
        ("new_shared_bytes", "shared_candidate_bytes", "duplicate_staging_bytes"), 0
    )
    scheduled: set[Path] = set()
    actions = []
    inline = ReleaseMembers(root, payload)
    for name in files.keys() - members.keys():
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


def plan_freeze(
    build: Path, work_root: Path, *, placement: ReleasePlacement
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
            root, releases / release_id, payload, state=state
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
            "verification_scratch_bytes": None,
        }
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {**base, "status": "blocked", "reason": str(error)}
