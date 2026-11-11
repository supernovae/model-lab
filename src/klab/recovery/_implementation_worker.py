"""Bounded subprocess entry point; import producers only from the materialized Git tree."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _snapshot_map(lock: dict) -> dict[str, str]:
    return {
        key: entry["snapshot_sha256"]
        for key, entry in lock["sources"].items()
        if entry["snapshot_sha256"] is not None
    }


def _prepare_inheritance(
    acquisition, corpus, work: Path, request: dict
) -> dict[str, str]:
    inherited = request["inherited_snapshots"]
    unchanged = request["expected_unchanged_snapshots"]
    changed = request["expected_changed_snapshots"]
    sources = {
        item.id: item for item in corpus.sources if item.redistribution != "rejected"
    }
    if (
        set(inherited) != set(unchanged)
        or set(inherited) & set(changed)
        or not (set(inherited) | set(changed)) <= set(sources)
        or any(
            inherited[key]["snapshot_sha256"] != sha for key, sha in unchanged.items()
        )
    ):
        raise ValueError(
            "INHERITED_IDENTITY_MISMATCH: inconsistent inherited source sets"
        )
    base = work / "corpora" / corpus.config.id
    pre = {}
    active = base / "acquisition.json"
    if active.exists():
        if active.is_symlink():
            raise ValueError("unsafe active acquisition lock")
        prior = json.loads(active.read_text(encoding="utf-8"))
        pre = _snapshot_map(prior)
    if changed and (
        set(pre) != set(inherited) | set(changed)
        or any(pre[key] != sha for key, sha in changed.items())
    ):
        raise ValueError("CHANGED_IDENTITY_MISMATCH: prior snapshot set differs")
    if any(
        key in pre and pre[key] != value["snapshot_sha256"]
        for key, value in inherited.items()
    ):
        raise ValueError("INHERITED_IDENTITY_MISMATCH: prior lock differs")
    for source_id, identity in inherited.items():
        item = sources[source_id]
        if item.kind not in {"git", "huggingface_dataset", "wikimedia_dump"}:
            raise ValueError(f"INHERITED_REUSE_UNAVAILABLE: {source_id}")
        path = base / "snapshots" / source_id / identity["snapshot_sha256"]
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError(f"unsafe inherited snapshot: {source_id}")
        manifest = acquisition.verify_snapshot(path)
        manifest_path = path / "manifest.json"
        if (
            manifest_path.is_symlink()
            or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            != identity["manifest_sha256"]
            or manifest["snapshot_sha256"] != identity["snapshot_sha256"]
            or manifest["adapter"]["module_sha256"] != identity["adapter_module_sha256"]
            or manifest["declaration"] != acquisition.source_declaration_payload(item)
            or manifest["declaration_sha256"] != acquisition.declaration_sha256(item)
            or not acquisition._reusable_immutable_adapter(item, manifest)
        ):
            raise ValueError(f"INHERITED_IDENTITY_MISMATCH: {source_id}")
    if changed:
        for source_id, previous in changed.items():
            old_path = base / "snapshots" / source_id / previous
            if old_path.is_symlink() or any(
                parent.is_symlink() for parent in old_path.parents
            ):
                raise ValueError(f"unsafe changed snapshot: {source_id}")
            old = acquisition.verify_snapshot(old_path)
            if old["snapshot_sha256"] != previous or (
                old["declaration"]
                == acquisition.source_declaration_payload(sources[source_id])
            ):
                raise ValueError(f"CHANGED_DECLARATION_MISMATCH: {source_id}")
    # The producer still decides reuse; prevent a producer fallback from retrieving
    # any inherited ID even if the adapter's reuse algorithm changes.
    for name in (
        "_acquire_git",
        "_acquire_hf",
        "_acquire_http",
        "_acquire_local",
        "_acquire_wikimedia",
    ):
        original = getattr(acquisition, name, None)
        if original is None:
            continue

        def guard(item, *args, _original=original, **kwargs):
            if item.id in inherited:
                raise ValueError(f"INHERITED_REACQUISITION_FORBIDDEN: {item.id}")
            return _original(item, *args, **kwargs)

        setattr(acquisition, name, guard)
    return pre


def _archive_lock(corpus, work: Path, closure: Path) -> dict[str, str]:
    active = work / "corpora" / corpus.config.id / "acquisition.json"
    if active.is_symlink() or not active.is_file():
        raise ValueError("unsafe active acquisition lock")
    data = active.read_bytes()
    closure.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        closure, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if active.read_bytes() != data:
        raise ValueError("acquisition lock changed while archiving")
    return {"path": str(closure), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("worker requires request and response paths")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    common = {
        "version",
        "source_root",
        "project",
        "work_root",
        "allow_network",
        "expected_build_sha256",
        "expected_release_sha256",
        "components",
    }
    extra = {
        "phase",
        "expected_project_sha256",
        "inherited_snapshots",
        "expected_unchanged_snapshots",
        "expected_changed_snapshots",
        "acquisition_closure_path",
    }
    if (
        type(request) is not dict
        or type(request.get("version")) is not int
        or request["version"] not in (1, 2)
        or set(request) != common | (extra if request["version"] == 2 else set())
    ):
        raise ValueError("invalid historical worker request")
    if (
        type(request["allow_network"]) is not bool
        or type(request["components"]) is not dict
    ):
        raise ValueError("invalid historical worker request types")
    for key in ("expected_release_sha256", "expected_build_sha256"):
        value = request[key]
        if value is not None and (
            type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)
        ):
            raise ValueError(f"invalid historical worker {key}")
    if request["version"] == 2:
        if (
            type(request["phase"]) is not str
            or request["phase"] not in ("build", "release")
            or (
                request["expected_project_sha256"] is not None
                and (
                    type(request["expected_project_sha256"]) is not str
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", request["expected_project_sha256"]
                    )
                )
            )
            or (
                request["phase"] == "build"
                and request["expected_release_sha256"] is not None
            )
            or (
                request["phase"] == "release"
                and request["expected_release_sha256"] is None
            )
            or type(request["expected_build_sha256"]) is not str
        ):
            raise ValueError("invalid historical worker v2 digest or phase")
        for key in ("expected_unchanged_snapshots", "expected_changed_snapshots"):
            mapping = request[key]
            if type(mapping) is not dict or any(
                type(source_id) is not str
                or type(digest) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                for source_id, digest in mapping.items()
            ):
                raise ValueError(f"invalid historical worker {key}")
        inherited = request["inherited_snapshots"]
        if type(inherited) is not dict or any(
            type(source_id) is not str
            or type(identity) is not dict
            or set(identity)
            != {"snapshot_sha256", "manifest_sha256", "adapter_module_sha256"}
            or any(
                type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in identity.values()
            )
            for source_id, identity in inherited.items()
        ):
            raise ValueError("invalid historical worker inherited_snapshots")
        if (
            type(request["acquisition_closure_path"]) is not str
            or not Path(request["acquisition_closure_path"]).is_absolute()
        ):
            raise ValueError("invalid historical worker acquisition_closure_path")
    for key in ("source_root", "project", "work_root"):
        if type(request[key]) is not str or not Path(request[key]).is_absolute():
            raise ValueError(f"invalid historical worker {key}")
    source = Path(request["source_root"]).resolve(strict=True)
    project_path = Path(request["project"])
    work_path = Path(request["work_root"])
    if (
        project_path.is_symlink()
        or not project_path.is_file()
        or not project_path.resolve(strict=True).is_relative_to(source)
    ):
        raise ValueError("invalid historical worker project outside source tree")
    if work_path.resolve(strict=True).is_relative_to(source) or source.is_relative_to(
        work_path.resolve(strict=True)
    ):
        raise ValueError("invalid historical worker root overlap")
    if request["version"] == 2:
        if any(part.is_symlink() for part in (work_path, *work_path.parents)):
            raise ValueError("unsafe historical work root")
        closure = Path(request["acquisition_closure_path"])
        if (
            closure.suffix != ".json"
            or closure.parent != source.parent.parent / "acquisition-closures"
            or closure.exists()
            or any(part.is_symlink() for part in (closure, *closure.parents))
            or closure.is_relative_to(work_path)
            or closure.is_relative_to(source)
        ):
            raise ValueError("unsafe historical acquisition closure path")
    if set(request["components"]) != {
        "acquisition",
        "pipeline",
        "project",
        "provenance",
        "rights",
        "large_build",
        "release",
        "export",
        "training_manifest",
        "project_config",
        "pyproject",
        "uv_lock",
    }:
        raise ValueError("invalid historical worker component set")
    sys.path.insert(0, str(source / "src"))
    phase = "producer_preflight"
    artifacts = {}
    try:
        modules = {}
        for name, entry in request["components"].items():
            if (
                type(entry) is not dict
                or not {
                    "role",
                    "affected_stage",
                    "source_commit",
                    "path",
                    "pinned_sha256",
                    "current_sha256",
                    "direct_identity_hash",
                    "status",
                }.issubset(entry)
                or entry["role"] != name
                or entry["source_commit"] != source.name
                or type(entry["affected_stage"]) is not str
                or type(entry["path"]) is not str
                or type(entry["pinned_sha256"]) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", entry["pinned_sha256"])
            ):
                raise ValueError(f"invalid historical worker component: {name}")
            if (
                not (source / entry["path"]).absolute().is_relative_to(source)
                or ".." in Path(entry["path"]).parts
            ):
                raise ValueError(f"invalid historical worker component path: {name}")
            expected = source / entry["path"]
            if not expected.is_file() or expected.is_symlink():
                raise ValueError(f"MISSING_IMPLEMENTATION: missing historical {name}")
            if (
                hashlib.sha256(expected.read_bytes()).hexdigest()
                != entry["pinned_sha256"]
            ):
                raise ValueError(
                    f"SOURCE_TAMPERED: historical {name} differs from pinned blob"
                )
            if name in {"pyproject", "uv_lock"}:
                continue
            module_name = (
                "klab.training.manifest"
                if name == "training_manifest"
                else "klab.config.models"
                if name == "project_config"
                else f"klab.corpus.{name}"
            )
            module = importlib.import_module(module_name)
            if Path(module.__file__).resolve(strict=True) != expected:
                raise ValueError(
                    f"MISSING_IMPLEMENTATION: historical {name} imported from {module.__file__}"
                )
            modules[name] = module
        acquisition = modules["acquisition"]
        pipeline = modules["pipeline"]
        project = modules["project"]
        release = modules["release"]
        phase = "project"
        corpus = project.load_project(Path(request["project"]))
        work = Path(request["work_root"])
        if request["version"] == 2:
            actual_project = acquisition._project_sha(corpus)
            artifacts["project_sha256"] = actual_project
            if (
                request["expected_project_sha256"] is not None
                and actual_project != request["expected_project_sha256"]
            ):
                raise ValueError(
                    f"EXPECTED_PROJECT_MISMATCH: actual={actual_project} expected={request['expected_project_sha256']}"
                )
            phase = "inheritance_preflight"
            artifacts["pre_snapshots"] = _prepare_inheritance(
                acquisition, corpus, work, request
            )
            artifacts["reuse_decisions"] = {
                key: "verified_immutable_reuse"
                for key in request["inherited_snapshots"]
            }
        remote = any(
            item.kind not in {"local", "deterministic_generator"}
            for item in corpus.sources
            if item.redistribution != "rejected"
        )
        if (
            remote
            and not request["allow_network"]
            and not (work / "corpora" / corpus.config.id / "acquisition.json").is_file()
        ):
            raise ValueError(
                "NETWORK_PERMISSION_REQUIRED: historical acquisition requires network"
            )
        phase = "acquisition"
        lock = acquisition.acquire(
            corpus, work, offline=remote and not request["allow_network"]
        )
        if request["version"] == 2:
            artifacts["acquisition_closure"] = _archive_lock(
                corpus, work, Path(request["acquisition_closure_path"])
            )
        acquisition.verify_acquisition(corpus, work)
        artifacts["snapshots"] = [
            {"source_id": key, "sha256": value["snapshot_sha256"]}
            for key, value in lock["sources"].items()
            if value["snapshot_sha256"]
        ]
        if request["version"] == 2:
            observed = _snapshot_map(lock)
            artifacts["post_snapshots"] = observed
            unchanged = request["expected_unchanged_snapshots"]
            changed = request["expected_changed_snapshots"]
            if any(observed.get(key) != sha for key, sha in unchanged.items()):
                raise ValueError(
                    "INHERITED_IDENTITY_MISMATCH: acquired immutable ID changed"
                )
            if any(
                observed.get(key) is None or observed[key] == sha
                for key, sha in changed.items()
            ):
                raise ValueError(
                    "CHANGED_IDENTITY_MISMATCH: changed source not reacquired"
                )
            if changed and set(observed) != set(unchanged) | set(changed):
                raise ValueError(
                    "CHANGED_IDENTITY_MISMATCH: unexpected final source set"
                )
            for item in corpus.sources:
                if item.redistribution == "rejected" or item.id in unchanged:
                    continue
                entry = lock["sources"][item.id]
                snap = acquisition.verify_snapshot(
                    work
                    / "corpora"
                    / corpus.config.id
                    / "snapshots"
                    / item.id
                    / entry["snapshot_sha256"]
                )
                if (
                    snap["adapter"]["module_sha256"]
                    != request["components"]["acquisition"]["pinned_sha256"]
                ):
                    raise ValueError(f"NEW_SNAPSHOT_PRODUCER_MISMATCH: {item.id}")
            if (
                request["expected_project_sha256"] is not None
                and acquisition._project_sha(corpus)
                != request["expected_project_sha256"]
            ):
                raise ValueError(
                    "EXPECTED_PROJECT_MISMATCH: historical project changed before build"
                )
        phase = "build"
        built = pipeline.build(corpus, work, offline=True)
        build = release.verify_build(built)
        artifacts["build_sha256"] = build["build_id"]
        artifacts["build_path"] = str(built)
        if (
            request["expected_build_sha256"] is not None
            and build["build_id"] != request["expected_build_sha256"]
        ):
            raise ValueError(
                f"EXPECTED_BUILD_MISMATCH: actual={build['build_id']} expected={request['expected_build_sha256']}"
            )
        if request["version"] == 2 and request["phase"] == "build":
            payload = {
                "version": 2,
                "status": "VERIFIED_BUILD",
                "phase": "build",
                "build_path": str(built),
                "build_sha256": build["build_id"],
                "snapshots": artifacts["snapshots"],
                "modules": {
                    name: str(Path(module.__file__).resolve())
                    for name, module in modules.items()
                },
                "project_sha256": artifacts["project_sha256"],
                "acquisition_closure": artifacts["acquisition_closure"],
                "reuse_decisions": artifacts["reuse_decisions"],
                "pre_snapshots": artifacts["pre_snapshots"],
                "post_snapshots": artifacts["post_snapshots"],
            }
        else:
            phase = "freeze"
            frozen = release.freeze(built, work)
            result = release.verify_release(frozen)
            artifacts["release_sha256"] = result["release_id"]
            artifacts["path"] = str(frozen)
            phase = "release_verification"
            if (
                request["expected_release_sha256"] is not None
                and result["release_id"] != request["expected_release_sha256"]
            ):
                raise ValueError(
                    f"EXPECTED_DIGEST_MISMATCH: actual={result['release_id']} expected={request['expected_release_sha256']}"
                )
            payload = {
                "version": request["version"],
                "status": "MATCH",
                "path": str(frozen),
                "build_path": str(built),
                "build_sha256": build["build_id"],
                "release_sha256": result["release_id"],
                "snapshots": artifacts["snapshots"],
                "modules": {
                    name: str(Path(module.__file__).resolve())
                    for name, module in modules.items()
                },
            }
            if request["version"] == 2:
                payload.update(
                    {
                        "phase": "release",
                        "project_sha256": artifacts["project_sha256"],
                        "acquisition_closure": artifacts["acquisition_closure"],
                        "reuse_decisions": artifacts["reuse_decisions"],
                        "pre_snapshots": artifacts["pre_snapshots"],
                        "post_snapshots": artifacts["post_snapshots"],
                    }
                )
    except Exception as error:
        logger.exception("Historical Corpus Forge replay failed in %s", phase)
        payload = {
            "version": request["version"],
            "status": "FAILED",
            "phase": phase,
            "artifacts": artifacts,
            "error": str(error),
            "error_type": type(error).__name__,
        }
    Path(sys.argv[2]).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return 0 if payload["status"] in {"MATCH", "VERIFIED_BUILD"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
