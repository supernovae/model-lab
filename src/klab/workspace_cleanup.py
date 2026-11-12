"""Reviewable, workspace-scoped cleanup of explicitly owned mutable artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

_MARKER = ".klab-cache-owner.json"
_CACHE_MEMBERS = frozenset(
    {
        _MARKER,
        "manifest.json",
        "train.npy",
        "validation.npy",
        "train_supervision.npy",
        "validation_supervision.npy",
        "train_byte_addresses.npy",
        "validation_byte_addresses.npy",
        "train_owner_ids.npy",
        "validation_owner_ids.npy",
        "train_semantic_queries.npy",
        "validation_semantic_queries.npy",
        "train_semantic_mask.npy",
        "validation_semantic_mask.npy",
    }
)
_TERMINAL_RUNS = frozenset({"completed", "failed", "interrupted", "cancelled"})
_TERMINAL_ATTEMPTS = frozenset({"COMPLETE", "FAILED", "INTERRUPTED", "CANCELLED"})


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


@contextmanager
def workspace_lock(workspace: Path) -> Iterator[None]:
    """Serialize workspace cache publication with explicit cleanup."""
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(workspace / ".cleanup.lock", flags, 0o600)
    with os.fdopen(descriptor, "a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _LOGGER.warning(
                "Waiting for workspace lock %s; preserve the lock and retained "
                "state. Owner identity is unknown; normal blocking wait continues.",
                workspace / ".cleanup.lock",
            )
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def mark_prepared_cache(workspace: Path, cache: Path) -> None:
    """Mark only a newly published cache while its workspace lock is held."""
    workspace, cache = workspace.resolve(), cache.resolve()
    if not _inside(cache, workspace) or cache.is_symlink():
        raise ValueError("cache is outside its owning workspace")
    marker = cache / _MARKER
    with marker.open("xb") as handle:
        handle.write(
            _canonical(
                {
                    "format_version": 1,
                    "workspace": str(workspace),
                    "manifest_sha256": hashlib.sha256(
                        (cache / "manifest.json").read_bytes()
                    ).hexdigest(),
                }
            )
        )
        handle.flush()
        os.fsync(handle.fileno())


def _inventory(path: Path) -> tuple[str, int]:
    """Record exact filesystem state, rejecting links and special files."""
    rows: list[tuple[str, int, int, int, int, int]] = []
    bytes_total = 0
    for member in (path, *sorted(path.rglob("*"))):
        info = member.lstat()
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ValueError(f"unsafe cleanup member: {member}")
        if stat.S_ISREG(info.st_mode):
            bytes_total += info.st_size
        rows.append(
            (
                member.relative_to(path).as_posix(),
                info.st_mode,
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
            )
        )
    return hashlib.sha256(_canonical(rows)).hexdigest(), bytes_total


def _store_rows(
    workspace: Path,
) -> tuple[
    list[tuple[str, str, str]], set[tuple[str, str]], set[str], dict[str, str], set[str]
]:
    if (workspace / "runs").is_symlink():
        raise ValueError("run store must be a real directory")
    database = workspace / "runs" / "experiments.sqlite3"
    if not database.is_file() or database.is_symlink():
        raise ValueError("experiment workspace needs an existing run database")
    with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as con:
        con.execute("PRAGMA query_only=ON")
        runs = [
            (str(run_id), str(status), str(config))
            for run_id, status, config in con.execute(
                "SELECT run_id,status,config_json FROM runs"
            )
        ]
        registered = {
            (str(run_id), str(relative_path))
            for run_id, relative_path in con.execute(
                "SELECT run_id,relative_path FROM checkpoints"
            )
        }
        active_attempts = {
            str(run_id)
            for run_id, status in con.execute("SELECT run_id,status FROM attempts")
            if status not in _TERMINAL_ATTEMPTS
        }
        manifests: dict[str, str] = {}
        bound_checkpoints: set[str] = set()
        for run_id, digest, raw in con.execute(
            "SELECT run_id,digest,json FROM manifests"
        ):
            manifests[str(run_id)] = str(digest)
            value = json.loads(raw)
            if isinstance(value, dict) and "manifest_reference" in value:
                raise ValueError(
                    f"unresolved manifest reference for {run_id}; "
                    "collect verified run artifacts first"
                )
            if isinstance(value, dict) and isinstance(
                value.get("checkpoint_sha256"), str
            ):
                bound_checkpoints.add(value["checkpoint_sha256"])
    return runs, registered, active_attempts, manifests, bound_checkpoints


def _candidate(
    workspace: Path, path: Path, kind: str, reason: str
) -> dict[str, object]:
    fingerprint, size = _inventory(path)
    return {
        "path": path.relative_to(workspace).as_posix(),
        "kind": kind,
        "reason": reason,
        "size_bytes": size,
        "fingerprint": fingerprint,
    }


def _checkpoint_candidates(
    workspace: Path,
    runs: list[tuple[str, str, str]],
    registered: set[tuple[str, str]],
    active_attempts: set[str],
    manifests: dict[str, str],
    bound_checkpoints: set[str],
    max_extra_periodic: int,
) -> list[dict[str, object]]:
    from klab.training.checkpoints import CheckpointManager, strict_json

    candidates: list[dict[str, object]] = []
    for run_id, status, _ in runs:
        if status not in _TERMINAL_RUNS or run_id in active_attempts:
            continue
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            continue
        run = workspace / "runs" / run_id
        checkpoints = run / "checkpoints"
        if run.is_symlink() or checkpoints.is_symlink() or not checkpoints.is_dir():
            continue
        if run_id not in manifests:
            continue
        manager = CheckpointManager(run, manifest_sha256=manifests[run_id])
        records, rejected = manager._verified_records()
        if rejected or len(records) <= 2:
            continue
        protected = {record.relative_path for record in records[:2]}
        for pointer_name in ("latest.json", "best.json"):
            pointer = checkpoints / pointer_name
            if pointer.is_symlink():
                protected = {record.relative_path for record in records}
                break
            if pointer.exists():
                try:
                    selected = strict_json(pointer)["relative_path"]
                except OSError, KeyError, TypeError, ValueError:
                    protected = {record.relative_path for record in records}
                    break
                if not isinstance(selected, str):
                    protected = {record.relative_path for record in records}
                    break
                protected.add(selected)
        eligible = [
            record
            for record in records
            if record.relative_path not in protected
            and (run_id, record.relative_path) not in registered
            and record.manifest_sha256 not in bound_checkpoints
        ]
        for record in eligible[max_extra_periodic:]:
            candidates.append(
                _candidate(
                    workspace,
                    checkpoints / record.relative_path,
                    "checkpoint",
                    "exceeds_max_extra_periodic",
                )
            )
    return candidates


def _cache_candidates(
    workspace: Path,
    runs: list[tuple[str, str, str]],
    active_attempts: set[str],
    max_cache_entries: int,
) -> list[dict[str, object]]:
    from klab.data.packing import supervision_requires_mask
    from klab.training.checkpoints import strict_json

    if active_attempts or any(status not in _TERMINAL_RUNS for _, status, _ in runs):
        return []
    cache_dirs: set[Path] = set()
    for _, _, raw in runs:
        try:
            path = Path(json.loads(raw)["dataset"]["cache_dir"])
        except KeyError, TypeError, ValueError:
            continue
        if (
            path.is_absolute()
            and _inside(path, workspace)
            and path.resolve() == path
            and not path.is_symlink()
        ):
            cache_dirs.add(path)
    entries: list[Path] = []
    for base in sorted(cache_dirs):
        if not base.is_dir():
            continue
        for path in base.iterdir():
            if not path.is_dir() or path.is_symlink() or len(path.name) != 16:
                continue
            try:
                int(path.name, 16)
                marker = strict_json(path / _MARKER)
                manifest = strict_json(path / "manifest.json")
                needs_mask = supervision_requires_mask(manifest)
                if any(
                    (path / f"{split}_supervision.npy").is_file() != needs_mask
                    for split in ("train", "validation")
                ):
                    continue
            except OSError, ValueError:
                continue
            if (
                marker
                != {
                    "format_version": 1,
                    "workspace": str(workspace),
                    "manifest_sha256": hashlib.sha256(
                        (path / "manifest.json").read_bytes()
                    ).hexdigest(),
                }
                or not isinstance(manifest, dict)
                or not str(manifest.get("settings_sha256", "")).startswith(path.name)
                or any(child.name not in _CACHE_MEMBERS for child in path.iterdir())
            ):
                continue
            entries.append(path)
    entries.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)
    return [
        _candidate(workspace, path, "prepared_data_cache", "exceeds_max_cache_entries")
        for path in entries[max_cache_entries:]
    ]


def plan_cleanup(
    workspace: Path, *, max_extra_periodic: int = 2, max_cache_entries: int = 2
) -> dict[str, object]:
    if type(max_extra_periodic) is not int or max_extra_periodic < 0:
        raise ValueError("max_extra_periodic must be nonnegative")
    if type(max_cache_entries) is not int or max_cache_entries < 0:
        raise ValueError("max_cache_entries must be nonnegative")
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError("workspace must be an existing real directory")
    workspace = workspace.resolve()
    runs, registered, active_attempts, manifests, bound_checkpoints = _store_rows(
        workspace
    )
    candidates = [
        *_checkpoint_candidates(
            workspace,
            runs,
            registered,
            active_attempts,
            manifests,
            bound_checkpoints,
            max_extra_periodic,
        ),
        *_cache_candidates(workspace, runs, active_attempts, max_cache_entries),
    ]
    candidates.sort(key=lambda item: str(item["path"]))
    body: dict[str, object] = {
        "format_version": 1,
        "workspace": str(workspace),
        "max_extra_periodic": max_extra_periodic,
        "max_cache_entries": max_cache_entries,
        "candidates": candidates,
        "reclaimable_bytes": sum(int(item["size_bytes"]) for item in candidates),
    }
    return {**body, "plan_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def write_plan(plan: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write("\n")


def apply_cleanup(plan_path: Path) -> dict[str, int]:
    from klab.training.checkpoints import CheckpointManager, strict_json

    saved = strict_json(plan_path)
    if not isinstance(saved, dict) or saved.get("format_version") != 1:
        raise ValueError("invalid cleanup plan")
    workspace = Path(str(saved["workspace"]))
    if not workspace.is_absolute() or workspace.is_symlink() or not workspace.is_dir():
        raise ValueError("unsafe cleanup workspace")
    body = {key: value for key, value in saved.items() if key != "plan_sha256"}
    if hashlib.sha256(_canonical(body)).hexdigest() != saved.get("plan_sha256"):
        raise ValueError("cleanup plan digest mismatch")
    with workspace_lock(workspace), ExitStack() as locks:
        first = plan_cleanup(
            workspace,
            max_extra_periodic=saved["max_extra_periodic"],
            max_cache_entries=saved["max_cache_entries"],
        )
        if first != saved:
            raise ValueError("cleanup plan is stale; create a new plan")
        locked_runs: set[str] = set()
        for item in saved["candidates"]:
            if item["kind"] == "checkpoint":
                relative = Path(item["path"])
                if len(relative.parts) != 4 or relative.parts[0] != "runs":
                    raise ValueError("invalid checkpoint candidate path")
                if relative.parts[1] in locked_runs:
                    continue
                locked_runs.add(relative.parts[1])
                locks.enter_context(
                    CheckpointManager(
                        workspace / "runs" / relative.parts[1]
                    ).writer_lease()
                )
        current = plan_cleanup(
            workspace,
            max_extra_periodic=saved["max_extra_periodic"],
            max_cache_entries=saved["max_cache_entries"],
        )
        if current != saved:
            raise ValueError("cleanup plan is stale; create a new plan")
        for item in saved["candidates"]:
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe cleanup candidate path")
            path = workspace / relative
            if _candidate(workspace, path, item["kind"], item["reason"]) != item:
                raise ValueError("cleanup candidate changed")
        for item in saved["candidates"]:
            path = workspace / Path(item["path"])
            shutil.rmtree(path)
    return {
        "removed": len(saved["candidates"]),
        "reclaimed_bytes": int(saved["reclaimable_bytes"]),
    }
