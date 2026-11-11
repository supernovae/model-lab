"""Explicit, bounded handoff of a completed executor-owned local build.

This is local execution evidence, not a newly minted verification signature.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from klab.canonical import canonical_digest
from klab.corpus.admission_inspection_cache import unchanged
from klab.corpus.project import ProjectConfig, project_path
from klab.corpus.split_inventory_adoption import (
    execution_reference,
    require_successful_execution,
)
from klab.training.manifest import sha256_file

_PRODUCER_FILES = frozenset(
    (
        "completed_preparation.py",
        "preparation_inputs.py",
        "streaming_build_rows.py",
        "split_runs.py",
        "large_build.py",
        "pipeline.py",
        "rights.py",
        "project.py",
        "jsonl_records.py",
    )
)

_ROLES = (
    "job_spec",
    "job_report",
    "session_spec",
    "session_report",
    "container_report",
    "stdout",
    "stderr",
    "events",
)


def adopt_completed_build_execution(root, path, build, scope, reference):
    """Bind a completed build to its actual closed executor and current inputs."""
    try:
        return _adopt(root, path, build, scope, reference)
    except (
        OSError,
        KeyError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError("missing or malformed completed build evidence") from error


def _adopt(root, path, build, scope, reference):
    root, path = Path(root).resolve(), Path(path).resolve()
    declaration, dependencies = execution_reference(root, reference)
    manifest = json.loads(declaration.read_bytes())
    if (
        set(manifest) != {"kind", "producer", "project", "build", "evidence"}
        or manifest["kind"] != "completed_local_build_execution"
        or not isinstance(manifest["producer"], dict)
        or set(manifest["producer"]) != {"revision", "runtime_sha256", "image"}
        or re.fullmatch(r"[0-9a-f]{40}", manifest["producer"].get("revision", ""))
        is None
        or re.fullmatch(r"[0-9a-f]{64}", manifest["producer"].get("runtime_sha256", ""))
        is None
        or re.fullmatch(
            r"[^\s]+@sha256:[0-9a-f]{64}", manifest["producer"].get("image", "")
        )
        is None
    ):
        raise ValueError("invalid completed build declaration")
    refs = manifest["evidence"]
    if not isinstance(refs, dict) or set(refs) != set(_ROLES):
        raise ValueError("completed build lacks closed execution evidence")
    paths = {}
    for role in _ROLES:
        paths[role], observed = execution_reference(root, refs[role])
        dependencies.update(observed)
    records = {
        role: json.loads(paths[role].read_bytes())
        for role in (
            "job_spec",
            "job_report",
            "session_spec",
            "session_report",
            "container_report",
        )
    }
    spec, job = records["job_spec"], records["job_report"]
    session_spec, session, outer = (
        records["session_spec"],
        records["session_report"],
        records["container_report"],
    )
    require_successful_execution(job, session, outer)
    producer = manifest["producer"]
    source = spec.get("source", {})
    source_path = Path(source.get("path", ""))
    runtime, identity = spec.get("runtime", {}), job.get("runtime_identity", {})
    if (
        job.get("spec_sha256") != refs["job_spec"]["sha256"]
        or outer.get("session_spec_sha256") != refs["session_spec"]["sha256"]
        or spec.get("source") != job.get("source")
        or source.get("commit") != producer["revision"]
        or not source_path.is_absolute()
        or any(p.is_symlink() for p in (source_path, *source_path.parents))
        or spec.get("cwd") != str(source_path / "src")
        or spec.get("wrap") is not None
        or job.get("runtime") != runtime
        or runtime.get("executable_sha256") != producer["runtime_sha256"]
        or identity.get("executable_sha256") != producer["runtime_sha256"]
        or outer.get("image") != producer["image"]
        or session_spec.get("execution_policy", {}).get("image") != producer["image"]
        or session_spec.get("runtime_sha256") != producer["runtime_sha256"]
    ):
        raise ValueError("completed build producer or runtime differs")
    project_file, observed = execution_reference(
        root, manifest["project"], control=False
    )
    dependencies.update(observed)
    config = ProjectConfig.model_validate(yaml.safe_load(project_file.read_bytes()))
    if config.model_dump(mode="json") != scope["request"]["project"]["config"]:
        raise ValueError("completed build project differs")
    if scope["request"]["project"]["root"] != str(project_file.parent):
        raise ValueError("completed build project root differs")
    argv = [
        identity.get("invocation"),
        "-B",
        "-m",
        "klab",
        "--work-dir",
        str(root),
        "corpus",
        "build",
        str(project_file),
        "--offline",
        "--reuse-completed-preparation",
    ]
    if (
        not argv[0]
        or spec.get("argv") != argv
        or paths["stdout"].read_text().strip() != str(path)
    ):
        raise ValueError("completed build command or output differs")
    progress = session.get("progress", {})
    jobs = [
        item
        for item in progress.get("jobs", [])
        if item.get("job_id") == job.get("job_id")
    ]
    if len(jobs) != 1 or spec.get("job_id") != job.get("job_id"):
        raise ValueError("completed build job is absent or ambiguous")
    item = jobs[0]
    for field, role in (
        ("spec_sha256", "job_spec"),
        ("report_sha256", "job_report"),
        ("stdout_sha256", "stdout"),
        ("stderr_sha256", "stderr"),
    ):
        if item.get(field) != refs[role]["sha256"]:
            raise ValueError("completed build progress identity differs")
    slots = [
        slot
        for slot in session_spec.get("slots", [])
        if slot.get("id") == item.get("slot") and slot.get("root") == item.get("root")
    ]
    if (
        len(slots) != 1
        or slots[0].get("argv") != argv
        or session_spec.get("roots", {}).get(item.get("root")) != str(root)
        or session.get("session_id") != session_spec.get("session_id")
        or item.get("launched") is not True
        or type(item.get("returncode")) is not int
        or item["returncode"] != 0
        or progress.get("events_sha256") != refs["events"]["sha256"]
    ):
        raise ValueError("completed build session identity differs")
    expected = manifest["build"]
    if (
        not isinstance(expected, dict)
        or set(expected) != {"path", "build_id", "manifest_sha256"}
        or expected["path"] != str(path)
        or expected["build_id"] != path.name
        or expected["manifest_sha256"] != sha256_file(path / "build.json")
        or build.get("build_id") != path.name
        or canonical_digest(build["identity"]) != path.name
    ):
        raise ValueError("completed build output identity differs")
    required = spec.get("required_files", {})
    scientific = [
        project_file,
        project_path(project_file.parent, config.release),
        project_path(project_file.parent, config.splits),
        *(
            project_path(project_file.parent, name)
            for name in (*config.sources, *config.transforms)
        ),
    ]
    for file in scientific:
        digest = required.get(str(file))
        if digest is None:
            raise ValueError("completed build did not bind a scientific input")
        _, observed = execution_reference(
            root, {"path": str(file), "sha256": digest}, control=False
        )
        dependencies.update(observed)
    if required.get(str(project_file)) != manifest["project"]["sha256"]:
        raise ValueError("completed build project identity differs")
    producers = scope["request"]["producer"]
    if not isinstance(producers, dict) or set(producers) != _PRODUCER_FILES:
        raise ValueError("completed build producer files missing")
    for name, digest in producers.items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("invalid completed build producer file")
        file = source_path / "src/klab/corpus" / name
        if (
            not file.is_file()
            or any(part.is_symlink() for part in (file, *file.parents))
            or sha256_file(file) != digest
        ):
            raise ValueError("completed build producer file changed")
    unchanged(scope["request"]["inputs"])
    unchanged(scope["source_bindings"])
    unchanged(dependencies)
    return {
        "kind": "adopted_execution",
        "method": "in_place_adopted_executor_build",
        "build_id": path.name,
        "execution_declaration": {
            "path": str(declaration),
            "sha256": reference["sha256"],
        },
        "declared_revision": producer["revision"],
        "job_report_sha256": refs["job_report"]["sha256"],
        "session_report_sha256": refs["session_report"]["sha256"],
    }
