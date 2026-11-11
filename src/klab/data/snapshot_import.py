"""Authenticated, write-new imports of bounded historical story snapshots.

The imported identity is distinct from both the historical identity and a fresh
Hub acquisition. Retained legacy bytes remain verifiable with the original
verifier; no upstream source is fetched and no exhaustion claim is inferred.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import DatasetConfig
from klab.data import local_stories
from klab.engram.packs import _rename_noreplace
from klab.result_publication import sync_directory
from klab.training.manifest import sha256_file

FORMAT = "klab-dataset-import-v1"


def _seal(value: dict, field: str) -> dict:
    return {**value, field: canonical_digest(value)}


def _regular(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"snapshot import requires a regular nonsymlink file: {path}")
    return path


def _relative(root: Path, name: object) -> Path:
    if not isinstance(name, str):
        raise TypeError("snapshot inventory path must be a string")
    relative = PurePosixPath(name)
    if (
        not name
        or relative.is_absolute()
        or "\\" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError("unsafe snapshot inventory path")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink in snapshot inventory path")
    return _regular(path)


def _read(path: Path) -> dict:
    try:
        value = json.loads(_regular(path).read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid snapshot import manifest") from error
    if not isinstance(value, dict):
        raise TypeError("snapshot import manifest must be a mapping")
    return value


def _legacy(
    path: Path, supplied: DatasetConfig | None = None
) -> tuple[dict, dict[str, Path]]:
    """Validate paths before invoking the unchanged historical verifier."""
    raw = _read(path)
    try:
        names = {
            split: raw["splits"][split]["path"] for split in ("train", "validation")
        }
        names["excluded"] = raw["excluded"]["path"]
        paths = {key: _relative(path.parent, name) for key, name in names.items()}
        if len({path.resolve(), *(p.resolve() for p in paths.values())}) != 4:
            raise ValueError("legacy snapshot inventory paths must be distinct")
        for split in ("train", "validation"):
            count = raw["splits"][split]["count"]
            if type(count) is not int or count <= 0:
                raise ValueError("legacy snapshot counts must be positive integers")
        verification_config = DatasetConfig(
            source="local_stories",
            revision=raw["revision"],
            license=raw["license"],
            cache_dir=path.parent,
            train_path=paths["train"],
            validation_path=paths["validation"],
            source_manifest_path=path,
            train_max_documents=raw["splits"]["train"]["count"],
            validation_max_documents=raw["splits"]["validation"]["count"],
            # These are verifier-only values, never emitted as experiment budgets.
            train_max_tokens=1,
            validation_max_tokens=1,
        )
        verified = local_stories.verify_snapshot(supplied or verification_config)
        if canonical_json(verified) != canonical_json(raw):
            raise ValueError("legacy snapshot manifest changed during verification")
    except (KeyError, TypeError, OSError) as error:
        raise ValueError("incomplete legacy snapshot inventory") from error
    paths["manifest"] = path
    return verified, paths


def _inventory(paths: dict[str, Path], root: Path) -> list[dict]:
    return sorted(
        (
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in paths.values()
        ),
        key=lambda row: row["path"],
    )


def _manifest(root: Path, legacy_path: Path, old: dict, paths: dict[str, Path]) -> dict:
    old_digest = sha256_file(legacy_path)
    source = {
        "schema_version": 1,
        "kind": "legacy_snapshot",
        "repo_id": old["source"],
        "revision": old["revision"],
        "license": old["license"],
        "attribution": old["source"],
        "text_field": "text",
        "splits": {"train": "train", "validation": "validation"},
        "selection": {
            "mode": "bounded",
            "documents": {
                split: old["splits"][split]["count"]
                for split in ("train", "validation")
            },
        },
        "legacy_manifest_sha256": old_digest,
    }
    lock = _seal(
        {
            "format": "klab-dataset-import-lock-v1",
            "source": source,
            "provenance": {
                "kind": "legacy_snapshot",
                "manifest_path": legacy_path.relative_to(root).as_posix(),
                "manifest_sha256": old_digest,
                "files": _inventory(paths, root),
            },
        },
        "lock_sha256",
    )
    splits = {}
    for split in ("train", "validation"):
        details = old["splits"][split]
        splits[split] = {
            "path": f"{split}.jsonl",
            "sha256": details["sha256"],
            "count": details["count"],
            "text_bytes": details["text_bytes"],
            "content_sha256": details["content_sha256"],
            "source_records_consumed": details["source_records_consumed"],
            "duplicate": details["duplicates"],
            "overlap": details["overlap"],
            "empty": details["empty"],
            "null": 0,
            "text_bytes_limit": 0,
            "stop_reason": "legacy_bounded_import",
            "source_exhausted": False,
        }
    return _seal(
        {
            "format": FORMAT,
            "lock": lock,
            "snapshot_source_sha256": canonical_digest(source),
            "splits": splits,
            "excluded": {**old["excluded"], "path": "excluded.jsonl"},
            "source_exhausted": False,
        },
        "manifest_sha256",
    )


def verify_import(config_or_manifest: DatasetConfig | str | Path) -> dict:
    """Cold authenticate retained old evidence and the exact derived new identity."""
    supplied = (
        config_or_manifest if isinstance(config_or_manifest, DatasetConfig) else None
    )
    if supplied is not None:
        if supplied.source != "snapshot" or supplied.source_manifest_path is None:
            raise ValueError(
                "import verification requires snapshot source and manifest"
            )
        path = supplied.source_manifest_path
    else:
        path = Path(config_or_manifest)
    manifest = _read(path)
    if manifest.get("format") != FORMAT:
        raise ValueError("unsupported snapshot import format")
    try:
        name = manifest["lock"]["provenance"]["manifest_path"]
        if not isinstance(name, str) or not name.startswith("legacy/"):
            raise ValueError("import must retain its historical manifest under legacy/")
        legacy_path = _relative(path.parent, name)
        old, paths = _legacy(legacy_path)
        expected = _manifest(path.parent, legacy_path, old, paths)
        if canonical_json(manifest) != canonical_json(expected):
            raise ValueError(
                "snapshot import provenance, identity or coverage mismatch"
            )
        inventory = {path.relative_to(path.parent).as_posix()}
        inventory.update(row["path"] for row in expected["lock"]["provenance"]["files"])
        for key in ("train", "validation", "excluded"):
            row = expected["excluded"] if key == "excluded" else expected["splits"][key]
            exported = _relative(path.parent, row["path"])
            if sha256_file(exported) != row["sha256"]:
                raise ValueError("imported snapshot content digest mismatch")
            inventory.add(row["path"])
        directories = {"legacy"}
        for name in inventory:
            directories.update(
                str(p) for p in PurePosixPath(name).parents if str(p) != "."
            )
        actual = set()
        for child in path.parent.rglob("*"):
            relative = child.relative_to(path.parent).as_posix()
            if child.is_symlink():
                raise ValueError("symlink in imported snapshot")
            if child.is_dir():
                if relative not in directories:
                    raise ValueError("unexpected directory in imported snapshot")
            else:
                _regular(child)
                actual.add(relative)
        if actual != inventory:
            raise ValueError("snapshot import inventory mismatch")
        if supplied is not None:
            if (supplied.revision, supplied.license) != (
                old["revision"],
                old["license"],
            ):
                raise ValueError("snapshot import config provenance mismatch")
            for split in ("train", "validation"):
                bound = getattr(supplied, f"{split}_path")
                if (
                    bound is None
                    or bound.resolve() != (path.parent / f"{split}.jsonl").resolve()
                ):
                    raise ValueError("snapshot import config path mismatch")
    except (KeyError, TypeError, OSError) as error:
        raise ValueError("incomplete snapshot import") from error
    return manifest


def import_legacy_snapshot(
    config_or_manifest: DatasetConfig | str | Path, output: str | Path
) -> Path:
    """Copy authenticated legacy bytes into an exclusively published new snapshot."""
    supplied = (
        config_or_manifest if isinstance(config_or_manifest, DatasetConfig) else None
    )
    if supplied is not None:
        if supplied.source != "local_stories" or supplied.source_manifest_path is None:
            raise ValueError("import requires a legacy local_stories snapshot")
        path = supplied.source_manifest_path
    else:
        path = Path(config_or_manifest)
    _regular(path)
    path = path.absolute()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    if output.resolve().is_relative_to(path.parent.resolve()):
        raise ValueError(
            "import output must be outside the historical snapshot directory"
        )
    _, paths = _legacy(path, supplied)
    before = _inventory(paths, path.parent)
    required = 2 * sum(row["size_bytes"] for row in before) + 1024 * 1024
    if shutil.disk_usage(output.parent).free < required:
        raise ValueError("insufficient free space for immutable snapshot import copies")
    if hasattr(os, "statvfs"):
        stats = os.statvfs(output.parent)
        # Some filesystems report zero total inodes when the limit is not meaningful.
        if stats.f_files and stats.f_favail < 16 + sum(
            len(Path(row["path"]).parts) for row in before
        ):
            raise ValueError("insufficient free inodes for snapshot import")
    staging = Path(tempfile.mkdtemp(prefix=".snapshot-import-", dir=output.parent))
    try:
        for old_path in paths.values():
            copied = staging / "legacy" / old_path.relative_to(path.parent)
            copied.parent.mkdir(parents=True, exist_ok=True)
            with old_path.open("rb") as reader, copied.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
                writer.flush()
                os.fsync(writer.fileno())
        for key in ("train", "validation", "excluded"):
            shutil.copyfile(
                staging / "legacy" / paths[key].relative_to(path.parent),
                staging / f"{key}.jsonl",
            )
        copied_manifest = staging / "legacy" / path.name
        copied_old, copied_paths = _legacy(copied_manifest)
        if _inventory(copied_paths, copied_manifest.parent) != before:
            raise ValueError("legacy snapshot changed during import")
        if _inventory(paths, path.parent) != before:
            raise ValueError("legacy snapshot changed during import")
        manifest = _manifest(staging, copied_manifest, copied_old, copied_paths)
        with (staging / "manifest.json").open("xb") as handle:
            handle.write(canonical_json(manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        verify_import(staging / "manifest.json")
        for key in ("train", "validation", "excluded"):
            with (staging / f"{key}.jsonl").open("rb") as handle:
                os.fsync(handle.fileno())
        directories = [p for p in staging.rglob("*") if p.is_dir()]
        for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
            sync_directory(directory)
        sync_directory(staging)
        _rename_noreplace(staging, output)
        sync_directory(output.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output / "manifest.json"
