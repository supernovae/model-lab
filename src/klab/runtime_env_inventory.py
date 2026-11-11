"""Bounded passive discovery of machine-local execution environments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from heapq import nsmallest
from pathlib import Path

from klab.runtime_env_subprocess import run_bounded
from klab.runtime_environments import (
    RuntimeRegistry,
    read_registry,
    resolve_runtime_dir,
)
from klab.runtime_identity_probe import source_identity

LIMIT = 32
OUTPUT_LIMIT = 32768
PROBE_TIMEOUT = 20


def canonical_python(path: Path) -> Path:
    directory = path.parent.resolve()
    selected = directory / path.name
    standard = directory / "python"
    # Normalize aliases inside one venv, never across venvs sharing base Python.
    if (
        (directory.parent / "pyvenv.cfg").is_file()
        and selected.is_file()
        and standard.is_file()
        and selected.resolve() == standard.resolve()
    ):
        return standard
    return selected


def inspect_python(python: Path) -> dict:
    python = canonical_python(python)
    base = {
        "python": str(python),
        "prefix": None,
        "python_version": None,
        "klab_import": {"success": False, "error": None},
        "source_sha256": None,
        "torch": {
            "installed": False,
            "version": None,
            "path": None,
            "hip": None,
            "cuda": None,
        },
        "mlx": {"installed": False, "version": None, "available": False},
        "devices": {},
        "backends": [],
        "status": "NOT_PROVISIONED",
        "reason": None,
    }
    if not python.is_file():
        return {**base, "reason": "interpreter does not exist"}
    if not os.access(python, os.X_OK):
        return {**base, "status": "ERROR", "reason": "interpreter is not executable"}
    script = Path(__file__).with_name("runtime_env_probe.py").resolve()
    try:
        result = run_bounded(
            [str(python), "-I", str(script)],
            timeout=PROBE_TIMEOUT,
            output_limit=OUTPUT_LIMIT,
        )
        data, error = result.stdout, result.stderr
        if result.returncode or len(data) > OUTPUT_LIMIT or len(error) > OUTPUT_LIMIT:
            raise ValueError(
                f"candidate probe status {result.returncode}; output capped={len(data) > OUTPUT_LIMIT or len(error) > OUTPUT_LIMIT}; {error[:1024].decode('utf-8', 'replace')}"
            )
        observation = json.loads(data)
        if not isinstance(observation, dict) or not isinstance(
            observation.get("backends"), list
        ):
            raise TypeError("invalid candidate probe response")
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError) as error:
        return {**base, "status": "ERROR", "reason": str(error)}
    merged = {**base, **observation, "python": str(python)}
    local = source_identity()
    if not observation.get("klab_import", {}).get("success"):
        status, reason = (
            "UNAVAILABLE",
            "Model Lab not importable in selected interpreter",
        )
    elif observation.get("source_sha256") != local["source_sha256"]:
        status, reason = (
            "SOURCE_MISMATCH",
            "installed Model Lab source differs from this checkout",
        )
    elif not observation["backends"]:
        status, reason = "UNAVAILABLE", "no executable framework backend available"
    else:
        status, reason = "READY", None
    return {**merged, "status": status, "reason": reason}


def discover_candidates(
    *,
    registry: RuntimeRegistry | None = None,
    runtime_dir: Path | None = None,
    project_dir: Path | None = None,
) -> tuple[list[dict], dict]:
    registry = read_registry() if registry is None else registry
    root = resolve_runtime_dir(runtime_dir)
    project = (
        Path(__file__).resolve().parents[2] if project_dir is None else project_dir
    )
    paths: dict[Path, dict] = {}

    def add(path: Path, origin: str, identifier: str | None = None) -> None:
        canonical = canonical_python(path)
        record = paths.setdefault(
            canonical, {"python": canonical, "ids": [], "origins": []}
        )
        if origin not in record["origins"]:
            record["origins"].append(origin)
        if identifier is not None and identifier not in record["ids"]:
            record["ids"].append(identifier)

    add(Path(sys.executable).absolute(), "active")
    add(project / ".venv/bin/python", "project")
    # Deterministic lexical selection with bounded memory and no recursion.
    children = []
    if root.is_dir():
        with os.scandir(root) as entries:
            children = nsmallest(
                LIMIT + 1,
                (
                    child.name
                    for child in entries
                    if child.name not in {".cache", ".scratch"}
                    and child.is_dir(follow_symlinks=False)
                ),
            )
    capped_root = len(children) > LIMIT
    children = children[:LIMIT]
    for name in children:
        add(root / name / "bin/python", "runtime_root")
    registered_paths: set[Path] = set()
    capped_registered = 0
    for identifier, entry in sorted(registry.runtimes.items()):
        path = canonical_python(entry.python)
        if path not in registered_paths and len(registered_paths) == LIMIT:
            capped_registered += 1
            continue
        registered_paths.add(path)
        add(entry.python, "registered", identifier)
    return list(paths.values()), {
        "truncated": capped_root or bool(capped_registered),
        "runtime_root_truncated": capped_root,
        "runtime_root_candidates": len(children),
        "registered_candidates": len(registered_paths),
        "registered_omitted_count": capped_registered,
        "candidate_limit": LIMIT * 2 + 2,
    }


def inventory(
    *,
    registry: RuntimeRegistry | None = None,
    runtime_dir: Path | None = None,
    project_dir: Path | None = None,
) -> dict:
    from klab.runtime_env_hardware import hardware_inventory

    candidates, limits = discover_candidates(
        registry=registry, runtime_dir=runtime_dir, project_dir=project_dir
    )
    environments = []
    for candidate in candidates:
        environments.append(
            {
                **inspect_python(candidate["python"]),
                "ids": candidate["ids"],
                "origins": candidate["origins"],
            }
        )
    return {
        "runtime_env_inventory_version": 1,
        "hardware": hardware_inventory(),
        "environments": environments,
        **limits,
    }


def observed_entry(identifier: str, entry, environments: list[dict]) -> dict:
    path = str(canonical_python(entry.python))
    environment = next((item for item in environments if item["python"] == path), None)
    status = environment["status"] if environment else "UNOBSERVED"
    reason = (
        environment.get("reason")
        if environment
        else "candidate omitted by bounded discovery"
    )
    if environment is None and not entry.python.is_file():
        status, reason = "NOT_PROVISIONED", "interpreter does not exist"
    if environment and status == "READY":
        device = environment.get("devices", {}).get(entry.backend, {})
        if entry.backend not in environment[
            "backends"
        ] or entry.device_index >= device.get(
            "count", 1 if entry.backend == "cpu" else 0
        ):
            status, reason = "UNAVAILABLE", "declared backend/device is not executable"
    return {
        "id": identifier,
        **entry.model_dump(mode="json"),
        "status": status,
        "reason": reason,
        "environment": environment,
    }
