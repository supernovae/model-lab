"""One-shot, versioned vendor provisioning outside the development environment."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from sparselab.runtime_env_doctor import doctor
from sparselab.runtime_env_inventory import inspect_python
from sparselab.runtime_env_recipes import get_recipe, preflight_runtime_root
from sparselab.runtime_env_subprocess import run_bounded
from sparselab.runtime_environments import (
    RuntimeEntry,
    read_registry,
    register_runtime,
    resolve_registry_path,
    resolve_runtime_dir,
    validate_id,
)
from sparselab.runtime_identity_probe import source_identity
from sparselab.runtime_profile import RuntimeProfile, probe_runtime_profile
from sparselab.training.manifest import sha256_file


def _validate_python(python: Path) -> dict:
    if (
        not python.is_absolute()
        or not python.is_file()
        or not os.access(python, os.X_OK)
    ):
        raise ValueError("provision python must be an existing absolute executable")
    result = run_bounded(
        [
            str(python),
            "-I",
            "-c",
            "import json,sys; print(json.dumps({'major':sys.version_info.major,'minor':sys.version_info.minor,'version':sys.version,'base_executable':sys._base_executable}))",
        ],
        timeout=20,
    )
    if result.returncode:
        raise ValueError("cannot inspect provisioning Python")
    value = json.loads(result.stdout)
    if (value.get("major"), value.get("minor")) != (3, 14):
        raise ValueError("selected recipe requires an existing Python 3.14 executable")
    return value


def _environment(root: Path) -> dict[str, str]:
    # A vendor environment must never inherit CPU indexes, extras, interpreter
    # targets, project synchronization or a cache on an unrelated filesystem.
    value = {
        key: item
        for key, item in os.environ.items()
        if not key.startswith("UV_")
        and key not in {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"}
    }
    value.update(
        {
            "UV_CACHE_DIR": str(root / ".cache"),
            "UV_PYTHON_DOWNLOADS": "never",
            "TMPDIR": str(root / ".scratch"),
            "TEMP": str(root / ".scratch"),
            "TMP": str(root / ".scratch"),
        }
    )
    return value


def _run_uv(
    command: list[str], *, environment: dict, checkout: Path, log: Path
) -> None:
    with log.open("xb") as stream:
        result = subprocess.run(
            command,
            env=environment,
            cwd=checkout,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
        )
        stream.flush()
        os.fsync(stream.fileno())
    if result.returncode:
        raise ValueError(
            f"provision command failed with status {result.returncode}; retained log: {log}"
        )


def _record(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _source_commit(checkout: Path) -> str:
    result = run_bounded(["git", "-C", str(checkout), "rev-parse", "HEAD"], timeout=5)
    commit = result.stdout.decode().strip()
    if result.returncode or not re.fullmatch(r"[a-f0-9]{40,64}", commit):
        raise ValueError(
            "provisioning requires a source checkout with a commit identity"
        )
    return commit


def _publish_ready(path: Path, receipt: dict) -> None:
    # A partial JSON write must never appear at the canonical READY path.
    # Keep staged evidence on failure; never overwrite an existing receipt.
    staged = path.with_name("provision-ready-staged.json")
    _record(staged, receipt)
    os.link(staged, path)
    try:
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as publication_error:
        # Withdraw only the link we just created, retaining staged evidence.
        # A process death here may leave a truthful READY (registration already
        # returned); the intent still identifies the durability recovery window.
        try:
            path.unlink()
        except OSError as withdrawal_error:
            raise OSError(
                f"READY publication failed: {publication_error}; "
                f"receipt withdrawal failed: {withdrawal_error}"
            ) from withdrawal_error
        raise


def _check_common(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if (
        re.search(r"(?im)^\s*torch(?:\[|\s|[=<>!~@])", text)
        or "pytorch-cpu" in text.lower()
        or "download.pytorch.org/whl/cpu" in text.lower()
    ):
        raise ValueError(
            "common export contains Torch or the CPU index; refusing vendor install"
        )


def provision(identifier: str, *, recipe: str, python: Path | None = None) -> dict:
    validate_id(identifier)
    selected = get_recipe(recipe)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError(f"{selected.id} requires Linux x86_64")
    if identifier in read_registry().runtimes:
        raise ValueError(f"runtime id already registered: {identifier}")
    root = resolve_runtime_dir()
    target = root / identifier
    if target.exists() or target.is_symlink():
        raise ValueError(
            f"provision target already exists; choose a fresh ID: {target}"
        )
    interpreter = Path(sys._base_executable).absolute() if python is None else python
    python_identity = _validate_python(interpreter)
    capacity = preflight_runtime_root(root)
    for name in (".cache", ".scratch"):
        if (root / name).is_symlink():
            raise ValueError(f"provision {name} must remain on the runtime filesystem")
        if (root / name).exists():
            preflight_runtime_root(root / name)
    checkout = selected.requirements_file.resolve().parent.parent
    source_commit = _source_commit(checkout)
    vendor_locks = (
        {
            name: checkout / "container/backends/rocm" / name
            for name in ("requirements.lock", "source-requirements.lock")
        }
        if selected.backend == "rocm"
        else {}
    )
    lock_hashes = {str(path): sha256_file(path) for path in vendor_locks.values()}
    requirements_bytes = selected.requirements_file.read_bytes()
    recipe_sha = hashlib.sha256(requirements_bytes).hexdigest()
    expected_torch = re.search(
        r"(?m)^torch(?:\[[^]]+\])?==([^;\s]+)", requirements_bytes.decode()
    )
    if expected_torch is None:
        raise ValueError("versioned recipe must pin Torch explicitly")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.mkdir(mode=0o700)  # Exclusive reservation: never reuse, repurpose or clean.
    failure = {
        "runtime_provision_version": 1,
        "id": identifier,
        "recipe": selected.id,
        "recipe_version": selected.version,
        "recipe_sha256": recipe_sha,
        "dependency_lock_sha256": lock_hashes,
        "environment_path": str(target),
        "status": "ERROR",
        "reason": None,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        environment = _environment(root)
        if selected.backend == "rocm":
            environment["ROCM_SDK_TARGET_FAMILY"] = "gfx1100"
        for name in (".cache", ".scratch"):
            (root / name).mkdir(mode=0o700, exist_ok=True)
        logs = root / ".scratch" / f"{identifier}-provision"
        logs.mkdir(mode=0o700)
        commands: list[list[str]] = []

        def uv(*args: str) -> None:
            command = ["uv", *args]
            commands.append(command)
            _run_uv(
                command,
                environment=environment,
                checkout=checkout,
                log=logs / f"{len(commands):02}.log",
            )

        uv(
            "venv",
            "--no-project",
            "--no-config",
            "--no-python-downloads",
            "--python",
            str(interpreter),
            str(target),
        )
        target.chmod(0o700)
        selected_python = target / "bin/python"
        common = target / "common-requirements.txt"
        uv(
            "export",
            "--python",
            str(interpreter),
            "--locked",
            "--no-default-groups",
            "--no-dev",
            "--no-emit-project",
            "--format",
            "requirements.txt",
            "--output-file",
            str(common),
        )
        _check_common(common)
        uv(
            "pip",
            "install",
            "--no-config",
            "--python",
            str(selected_python),
            "-r",
            str(common),
        )
        if selected.backend == "rocm":
            # Reuse the reviewed image locks, including pinned setuptools before
            # building the sole allowed source metapackage without isolation.
            for name, build_flag in (
                ("requirements.lock", "--only-binary=:all:"),
                ("source-requirements.lock", "--no-build-isolation"),
            ):
                uv(
                    "pip",
                    "install",
                    "--no-config",
                    "--python",
                    str(selected_python),
                    "--no-deps",
                    "--require-hashes",
                    build_flag,
                    "-r",
                    str(vendor_locks[name]),
                )
            uv("pip", "check", "--no-config", "--python", str(selected_python))
        else:
            uv(
                "pip",
                "install",
                "--no-config",
                "--python",
                str(selected_python),
                "-r",
                str(selected.requirements_file),
            )
        before = inspect_python(selected_python)["torch"]
        if (
            not before.get("installed")
            or before.get("version") != expected_torch.group(1)
            or bool(before.get("hip")) != bool(selected.requirements.get("torch_hip"))
            or (selected.backend == "cuda" and not before.get("cuda"))
        ):
            raise ValueError(f"recipe Torch build mismatch: {before}")
        uv(
            "pip",
            "install",
            "--no-config",
            "--python",
            str(selected_python),
            "--no-deps",
            "--editable",
            str(checkout),
        )
        observation = inspect_python(selected_python)
        if observation["torch"] != before:
            raise ValueError(
                "source-only editable install changed the vendor Torch identity"
            )
        if (
            observation["status"] != "READY"
            or selected.backend not in observation["backends"]
        ):
            raise ValueError(
                f"provisioned interpreter is not executable {selected.backend}: "
                f"{observation['status']}; {observation.get('reason')}"
            )
        imported = run_bounded(
            [
                str(selected_python),
                "-I",
                "-c",
                "from sparselab.cli.main import main; print('SparseLab CLI import complete')",
            ],
            timeout=30,
            env=environment,
        )
        if imported.returncode:
            raise ValueError(
                f"full SparseLab import failed: {imported.stderr.decode('utf-8', 'replace')[:1024]}"
            )
        profile = RuntimeProfile.model_validate(
            {
                "runtime_profile_version": 1,
                "id": identifier,
                "python": selected_python,
                "engine": "pytorch",
                "backend": selected.backend,
                "device_index": 0,
                "requirements": dict(selected.requirements),
            }
        )
        probe = probe_runtime_profile(profile)
        checked = doctor(profile, precision=selected.test_precision)
        if checked["status"] != "READY":
            raise ValueError(
                f"runtime doctor refused provision: {checked['status']}; {checked['reason']}"
            )
        tested = checked.get("tested_runtime")
        if not isinstance(tested, dict) or (
            tested.get("backend") != selected.backend
            or tested.get("device_index") != 0
            or selected.test_precision not in tested.get("tested_precisions", [])
            or "forward_backward_optimizer" not in tested.get("tested_features", [])
        ):
            raise ValueError(
                f"provision doctor did not verify the {selected.test_precision} optimizer update"
            )
        packages = run_bounded(
            [
                "uv",
                "pip",
                "list",
                "--no-config",
                "--python",
                str(selected_python),
                "--format",
                "json",
            ],
            timeout=30,
            env=environment,
        )
        if packages.returncode:
            raise ValueError("cannot record provisioned package inventory")
        package_inventory = json.loads(packages.stdout)
        if not isinstance(package_inventory, list):
            raise TypeError("invalid provisioned package inventory")
        receipt = {
            **failure,
            "status": "READY",
            "reason": None,
            "python": python_identity,
            "recipe_file": str(selected.requirements_file),
            "source_commit": source_commit,
            **source_identity(),
            "torch": observation["torch"],
            "indexes": list(selected.indexes),
            "package_inventory": package_inventory,
            "profile": profile.model_dump(mode="json"),
            "probe": probe,
            "tested_runtime": checked["tested_runtime"],
            "commands": commands,
            "capacity": capacity,
            "logs": str(logs),
        }
        receipt_path = target / "provision-receipt.json"
        registry_path = resolve_registry_path()
        entry = RuntimeEntry.model_validate(
            profile.model_dump(exclude={"id", "runtime_profile_version"})
        )
        receipt["registry_path"] = str(registry_path)
        intent_path = target / "provision-registration.json"
        recovery = {
            "registration_record": str(intent_path),
            "registry_path": str(registry_path),
            "receipt_path": str(receipt_path),
            "registration_state": "unknown",
            "recovery": (
                "Inspect the recorded registry path and compare the exact entry with "
                "the retained profile. An absent or different entry is not success; "
                "a matching entry does not prove directory durability. Preserve all "
                "evidence and resolve durability before using the runtime. Do not "
                "rerun provision, overwrite receipts, or unregister automatically."
            ),
        }
        _record(intent_path, {**receipt, **recovery, "status": "REGISTRATION_PENDING"})
        failure.update(recovery)
        try:
            register_runtime(identifier, entry, registry_path=registry_path)
        except ValueError:
            # Validation, registry reads and duplicate-ID refusal precede
            # publication. OSError may follow replacement and stays unknown.
            failure["registration_state"] = "refused"
            failure["recovery"] = (
                "Registration was refused before publication; this call did not "
                "modify the registry. Preserve the evidence and any conflicting "
                "entry. Do not overwrite or unregister it, or rerun provision "
                "into the reserved target."
            )
            raise
        failure["registration_state"] = "registered"
        _publish_ready(receipt_path, receipt)
        return {
            **checked,
            "runtime_provision_version": 1,
            "recipe": selected.id,
            "environment_path": str(target),
            "receipt_path": str(receipt_path),
        }
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as error:
        failure["reason"] = f"{type(error).__name__}: {error}"
        failure_path = target / "provision-failure.json"
        try:
            _record(failure_path, failure)
        except OSError as recording_error:
            raise OSError(
                f"{failure['reason']}; failure recording at {failure_path} "
                f"failed: {recording_error}"
            ) from recording_error
        return {**failure, "failure_record": str(failure_path)}
