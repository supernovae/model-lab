"""Durable normalized input declarations for interrupted native builds."""

from __future__ import annotations

import tempfile
from pathlib import Path

from klab.canonical import canonical_digest
from klab.result_publication import publish_immutable, sync_directory
from klab.safe_io import symlinked, write_canonical_json

VIEW_INPUTS = "normalized-inputs.json"


def normalized_uses(workspace: Path) -> Path:
    return workspace / "normalized-uses"


def record_normalized_use(
    workspace: Path,
    build_id: str,
    source_ids: list[str],
    *,
    partitions: list[Path] | None = None,
    checkpoints: tuple[str, ...] = (),
) -> None:
    """Record sources before use, then narrow the retained set to exact keys."""
    root = normalized_uses(workspace)
    path = root / f"{build_id}.json"
    if symlinked(path):
        raise ValueError(f"unsafe normalized input declaration: {path}")
    if not root.exists():
        root.mkdir(exist_ok=True)
        sync_directory(workspace)
    body = {
        "schema_version": 1,
        "build_id": build_id,
        "sources": sorted(source_ids),
        "checkpoints": list(checkpoints),
        "partitions": None
        if partitions is None
        else sorted(f"{path.parent.name}/{path.name}" for path in partitions),
    }
    with tempfile.TemporaryDirectory(prefix=".inputs-", dir=root) as temporary:
        staged = Path(temporary) / path.name
        write_canonical_json(staged, {"body": body, "sha256": canonical_digest(body)})
        staged.replace(path)
        sync_directory(root)


def bind_normalized_view(view_root: Path, build_id: str) -> None:
    publish_immutable(view_root / VIEW_INPUTS, {"build_id": build_id})
    sync_directory(view_root.parent)


def clear_normalized_use(workspace: Path, build_id: str) -> None:
    path = normalized_uses(workspace) / f"{build_id}.json"
    if symlinked(path):
        raise ValueError(f"unsafe normalized input declaration: {path}")
    if path.exists():
        path.unlink()
        sync_directory(path.parent)
