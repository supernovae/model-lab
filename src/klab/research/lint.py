"""Offline checks of tracked research declarations, never their external payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from klab.documents import read_document
from klab.recovery.evidence import KINDS
from klab.recovery.provenance import declaration_paths, declaration_reference

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_RESEARCH = Path("experiments/research")
_HISTORY = _RESEARCH / "history"
_HISTORY_MAP = _HISTORY / "path-map.json"
_HISTORY_METADATA = {
    _HISTORY_MAP.as_posix(),
    (_HISTORY / "README.md").as_posix(),
    (_HISTORY / "LESSONS_LEARNED.md").as_posix(),
}
_FORBIDDEN_DIRS = {
    "checkpoints",
    "checkpoint",
    "cache",
    "caches",
    ".cache",
    "__pycache__",
    "datasets",
    "dataset",
    "logs",
    "runs",
    "run",
    "outputs",
    "output",
    "scratch",
    "tmp",
    "prepared",
    "downloads",
    "wandb",
    "tensorboard",
}
_FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".npy",
    ".npz",
    ".arrow",
    ".parquet",
    ".csv",
    ".tsv",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".log",
    ".pyc",
    ".pkl",
    ".pickle",
    ".zip",
    ".gz",
    ".tar",
    ".zst",
}
_RECORD_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".json", ".jsonl"}
_MUTABLE_NAMES = {"metrics.jsonl", "events.jsonl", "stdout", "stderr", "run.db"}


def _sha(value: Any) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("expected a lowercase SHA-256 digest")
    return value


def _history_inventory(root: Path, tracked: set[str]) -> dict[str, tuple[str, int]]:
    """Authenticate relocated bytes, without interpreting past executable schemas."""
    source = _safe_file(root, root / _HISTORY_MAP, tracked)
    if source.stat().st_size > 1024 * 1024:
        raise ValueError("history path map exceeds the 1 MiB durable-record limit")
    raw = read_document(source)
    if raw.get("format") != "research-history-path-map-v1":
        raise ValueError("unsupported history path map format")
    if not isinstance(raw.get("source_commit"), str) or not _COMMIT.fullmatch(
        raw["source_commit"]
    ):
        raise ValueError("history source_commit must be a full Git object ID")
    if not isinstance(raw.get("entries"), list):
        raise TypeError("history entries must be a list")
    inventory: dict[str, tuple[str, int]] = {}
    originals: set[str] = set()
    for entry in raw["entries"]:
        if not isinstance(entry, dict):
            raise TypeError("history entry must be an object")
        original, archived = entry.get("original_path"), entry.get("archive_path")
        for name in (original,) + (() if archived is None else (archived,)):
            if (
                not isinstance(name, str)
                or not name
                or Path(name).is_absolute()
                or "\\" in name
                or any(part in {"", ".", ".."} for part in name.split("/"))
            ):
                raise ValueError(f"unsafe history path: {name!r}")
        if not isinstance(original, str) or original in originals:
            raise ValueError("missing or duplicate original history path")
        originals.add(original)
        sha = _sha(entry.get("sha256"))
        size = entry.get("bytes")
        if type(size) is not int or size < 0:
            raise ValueError("history byte count must be a nonnegative integer")
        if size > 1024 * 1024:
            raise ValueError("archived record exceeds the 1 MiB durable-record limit")
        if entry.get("action") == "deleted_obsolete_executable":
            if archived is not None:
                raise ValueError(
                    "deleted historical executable cannot have archive path"
                )
            continue
        if (
            entry.get("action") != "relocated_unchanged"
            or not isinstance(archived, str)
            or not Path(archived).is_relative_to(_HISTORY)
            or archived in _HISTORY_METADATA
            or archived in inventory
        ):
            raise ValueError("invalid or duplicate archived history path")
        target = _safe_file(root, root / archived, tracked)
        if (
            target.stat().st_size != size
            or hashlib.sha256(target.read_bytes()).hexdigest() != sha
        ):
            raise ValueError(
                f"archived research record digest or size mismatch: {archived}"
            )
        inventory[archived] = (sha, size)
    return inventory


def _safe_file(root: Path, path: Path, tracked: set[str]) -> Path:
    """Reject links before reading bytes, including links in an ancestor."""
    if not path.is_relative_to(root):
        raise ValueError(f"reference outside repository: {path}")
    relative = path.relative_to(root)
    for component in (path, *path.parents):
        if component == root:
            break
        if component.is_symlink():
            raise ValueError(f"symlinked research reference: {relative}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"reference outside repository: {relative}")
    if relative.as_posix() not in tracked or not path.is_file():
        raise ValueError(f"missing tracked reference: {relative}")
    return path


def _reference(root: Path, source: Path, name: Any, tracked: set[str]) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or Path(name).is_absolute()
        or "\\" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError(f"unsafe repository reference: {name!r}")
    return _safe_file(root, declaration_reference(source, name), tracked)


def _binding(root: Path, source: Path, name: Any, sha: Any, tracked: set[str]) -> None:
    expected = _sha(sha)
    target = _reference(root, source, name, tracked)
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise ValueError(f"reference digest mismatch: {name}")


def _declaration_template(source: Path) -> bool:
    """Check explicit data templates; native consumers validate rendered values."""
    if not source.name.endswith((".template.yaml", ".template.json")):
        return False
    from klab.corpus.declaration_render import _SLOT

    raw = source.read_bytes()
    if b"${" not in raw:
        return False  # A concrete declaration still receives full schema checks.
    if b"${" in _SLOT.sub(b"", raw):
        raise ValueError("invalid declaration template slot syntax")
    # Reuse the strict data reader: mapping root, duplicate keys, invalid YAML/JSON
    # and nonfinite values remain errors. Numeric slots are strings until rendering.
    read_document(source)
    return True


def _evidence(root: Path, source: Path, raw: dict, tracked: set[str]) -> None:
    # The export envelope describes a past verification. Do not open external_location.
    if raw.get("kind") not in KINDS:
        raise ValueError("unknown scientific evidence kind")
    _sha(raw.get("sha256"))
    commit = raw.get("source_commit")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError("evidence source_commit must be a full Git object ID")
    for key in ("identifier", "external_location", "verified_at"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"evidence requires nonempty {key}")
    expected_scope = (
        "observation" if raw["kind"] == "runtime_probe" else "verified_artifact"
    )
    if raw.get("verification_scope") != expected_scope:
        raise ValueError("evidence verification_scope disagrees with kind")
    declarations = raw.get("declaration_hashes")
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("evidence requires declaration_hashes")
    seen: set[str] = set()
    for item in declarations:
        if not isinstance(item, dict):
            raise TypeError("declaration_hashes entries must be objects")
        name = item.get("path")
        _reference(root, source, name, tracked)
        if name in seen:
            raise ValueError(f"duplicate declaration binding: {name}")
        seen.add(name)
        _binding(root, source, name, item.get("sha256"), tracked)
        if (
            item.get("head_sha256") is not None
            and item["head_sha256"] != item["sha256"]
        ):
            raise ValueError(f"declaration committed digest mismatch: {name}")


def _document(
    root: Path, source: Path, raw: dict, tracked: set[str], visited: set[Path]
) -> None:
    if source in visited:
        return
    visited.add(source)
    for version, kind in (
        ("plan_version", "experiment"),
        ("campaign_version", "campaign"),
        ("recovery_version", "recovery"),
    ):
        if version in raw:
            for target in declaration_paths(source, kind, include_local_inputs=False):
                _safe_file(root, target, tracked)
                if target != source and target.suffix.lower() in {
                    ".yaml",
                    ".yml",
                    ".json",
                }:
                    _document(root, target, read_document(target), tracked, visited)
            if kind == "experiment":
                from klab.experiments.plan import base_run_config, load_plan

                base_run_config(load_plan(source), source)
    if "producer_record_version" in raw:
        if (
            type(raw["producer_record_version"]) is not int
            or raw["producer_record_version"] != 1
        ):
            raise ValueError("unsupported producer record version")
        _reference(root, source, raw.get("protocol"), tracked)
        project = _reference(root, source, raw.get("corpus_project"), tracked)
        for target in declaration_paths(project, "corpus", include_local_inputs=False):
            _safe_file(root, target, tracked)
            _document(root, target, read_document(target), tracked, visited)
    if "recovery_version" in raw:
        from klab.config.loading import load_config, load_tokenizer_config
        from klab.recovery.manifest import load_manifest

        for step in load_manifest(source).steps:
            for field in ("config", "base_run"):
                name = getattr(step, field, None)
                if name:
                    target = _reference(root, source, name, tracked)
                    loader = (
                        load_tokenizer_config
                        if step.kind == "tokenizer_train"
                        else load_config
                    )
                    loader(target)
    if "readiness_version" in raw:
        from klab.evaluation.readiness import load_policy

        load_policy(source)
    if {"model", "dataset", "training"}.issubset(raw):
        from klab.config.models import RunConfig

        RunConfig.model_validate(raw)
    if "family_version" in raw:
        from klab.family.manifest import load_family

        family = load_family(source)
        for node in family.nodes:
            for name in (*node.lifecycle_receipts, node.recovery_manifest):
                if name:
                    _reference(root, source, name, tracked)
    if "evaluation_suite_version" in raw:
        from klab.evaluation.suite import load_suite

        suite = load_suite(source)
        for item in suite.evaluations:
            # surface_review sources are external run outputs, not declarations.
            if item.source and item.kind != "surface_review":
                _reference(root, source, item.source, tracked)
    for field in ("frozen_sha256", "inputs_sha256"):
        if field not in raw:
            continue
        frozen = raw[field]
        if not isinstance(frozen, dict) or not frozen:
            raise ValueError(f"{field} must be a nonempty path/digest mapping")
        for name, sha in frozen.items():
            _binding(root, source, name, sha, tracked)
    if (
        str(raw.get("format", "")).startswith("scientific-evidence-reference-")
        and raw["format"] != "scientific-evidence-reference-v1"
    ):
        raise ValueError("unsupported scientific evidence reference version")
    if (
        "declaration_hashes" in raw
        and raw.get("format") != "scientific-evidence-reference-v1"
    ):
        raise ValueError(
            "declaration_hashes requires a scientific evidence reference envelope"
        )
    if raw.get("format") == "scientific-evidence-reference-v1":
        _evidence(root, source, raw, tracked)
    # Explicit scientific bindings are current immutable declarations; unrelated
    # nested implementation/archive inventories remain historical metadata.
    if "bindings" in raw:
        if not isinstance(raw["bindings"], dict):
            raise ValueError("bindings must be a mapping")
        for binding in raw["bindings"].values():
            if not isinstance(binding, dict):
                raise TypeError("bindings entries must be path/digest objects")
            _binding(root, source, binding.get("path"), binding.get("sha256"), tracked)
    for key in ("protocol", "config", "base_run"):
        binding = raw.get(key)
        if isinstance(binding, dict) and "path" in binding:
            _binding(root, source, binding["path"], binding.get("sha256"), tracked)
    if "reference_checkpoint_receipt" in raw:
        _reference(root, source, raw["reference_checkpoint_receipt"], tracked)


def lint_research(root: Path) -> dict[str, Any]:
    """Lint tracked working-tree bytes (including staged additions), without writes."""
    root = Path(root).resolve()
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    tracked = set(result.stdout.decode().rstrip("\0").split("\0"))
    paths = sorted(name for name in tracked if Path(name).is_relative_to(_RESEARCH))
    errors: list[dict[str, str]] = []
    visited: set[Path] = set()
    history: dict[str, tuple[str, int]] = {}
    if any(Path(name).is_relative_to(_HISTORY) for name in paths):
        try:
            history = _history_inventory(root, tracked)
        except (ValueError, TypeError, KeyError, OSError) as error:
            errors.append({"path": _HISTORY_MAP.as_posix(), "message": str(error)})
    for name in paths:
        source = root / name
        relative = Path(name).relative_to(_RESEARCH)
        try:
            _safe_file(root, source, tracked)
            if (
                {part.lower() for part in relative.parts[:-1]} & _FORBIDDEN_DIRS
                or source.suffix.lower() in _FORBIDDEN_SUFFIXES
                or source.name.lower() in _MUTABLE_NAMES
                or source.name.startswith("events.out.tfevents.")
            ):
                raise ValueError(
                    "forbidden mutable output or payload in research records"
                )
            if source.stat().st_size > 1024 * 1024:
                raise ValueError(
                    "research record exceeds the 1 MiB durable-record limit"
                )
            if Path(name).is_relative_to(_HISTORY):
                if name not in history and name not in _HISTORY_METADATA:
                    raise ValueError("unindexed archived research record")
                continue
            if source.suffix.lower() not in _RECORD_SUFFIXES:
                raise ValueError(
                    "unsupported research record format; payloads belong outside the checkout"
                )
            if _declaration_template(source):
                continue
            if source.suffix.lower() in {".json", ".yaml", ".yml"}:
                _document(root, source, read_document(source), tracked, visited)
            elif source.suffix.lower() == ".jsonl":
                raise ValueError(
                    "JSONL payloads belong outside active research records"
                )
        except (ValueError, TypeError, KeyError, OSError) as error:
            errors.append({"path": name, "message": str(error)})
    return {
        "valid": not errors,
        "checked_files": len(paths),
        "errors": errors,
        "scope": "tracked declarations and references; external payloads not verified",
    }


def _handle(args: argparse.Namespace) -> None:
    report = lint_research(Path(args.repository))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"Research records: {report['checked_files']} checked, {len(report['errors'])} errors"
        )
        for error in report["errors"]:
            print(f"{error['path']}: {error['message']}")
        print(report["scope"])
    if not report["valid"]:
        raise SystemExit(1)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "lint", help="Validate tracked research declarations offline."
    )
    parser.add_argument("--repository", default=".")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(handler=_handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--json", action="store_true")
    _handle(parser.parse_args())
