"""Durable, single-attempt worker execution.

This module has no controller dependency.  A controller can disappear after launch;
the worker's receipt, run store, checkpoints, and outbox remain authoritative.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from sparselab.canonical import canonical_digest, canonical_json, reject_duplicate_keys
from sparselab.config.models import RunConfig
from sparselab.evaluation.post_train_triage import read_triage, triage_summary
from sparselab.resource_envelope import check_envelope, current_process_rss_bytes
from sparselab.runtime import discover_runtimes, validate_runtime
from sparselab.runtime_profile import authorize_worker, require_authorization
from sparselab.training.manifest import source_identity
from sparselab.verification_proofs import verification_options

from .leases import (
    DeviceLease,
    MultiDeviceLease,
    acquire_lease,
    acquire_multi_device_leases,
    boot_identity,
    process_matches,
    process_start,
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_TERMINAL = {"COMPLETE", "FAILED", "INTERRUPTED", "UNKNOWN"}
_LOGGER = logging.getLogger(__name__)


def _utc() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _check_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"invalid {field}")
    return value


def _definition_value(definition: Any, name: str) -> Any:
    return getattr(definition, name) if hasattr(definition, name) else definition[name]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = canonical_json(dict(payload))
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_json(path: Path) -> dict[str, Any]:

    payload = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise TypeError("receipt must be an object")
    canonical_json(payload)
    return payload


def _attempt_dir(definition: Any, attempt_id: str) -> Path:
    return (
        Path(_definition_value(definition, "root"))
        / "attempts"
        / _check_id(attempt_id, "attempt_id")
    )


def _receipt_path(definition: Any, attempt_id: str) -> Path:
    return _attempt_dir(definition, attempt_id) / "receipt.json"


def _load_receipt(definition: Any, attempt_id: str) -> dict[str, Any]:
    from .models import AttemptReceipt

    path = _receipt_path(definition, attempt_id)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"attempt receipt not found: {attempt_id}")
    return AttemptReceipt.model_validate(_strict_json(path)).model_dump(mode="json")


@contextmanager
def _receipt_lock(definition: Any, attempt_id: str) -> Any:
    """Serialize receipt read-modify-write without replacing its flock inode."""
    import fcntl

    directory = _attempt_dir(definition, attempt_id)
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / "receipt.lock").open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _write_receipt(definition: Any, receipt: Mapping[str, Any]) -> dict[str, Any]:
    from .models import AttemptReceipt

    payload = dict(receipt)
    payload["updated_at"] = _utc()
    payload = AttemptReceipt.model_validate(payload).model_dump(mode="json")
    _atomic_json(_receipt_path(definition, str(payload["attempt_id"])), payload)
    return payload


def _mutate_receipt(
    definition: Any,
    attempt_id: str,
    mutation: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> dict[str, Any]:
    """Apply a serialized mutation; terminal states cannot be regressed."""
    with _receipt_lock(definition, attempt_id):
        before = _load_receipt(definition, attempt_id)
        after = mutation(dict(before)) or before
        if before["state"] in _TERMINAL and after["state"] not in _TERMINAL:
            return before
        if after == before:
            return before
        return _write_receipt(definition, after)


@contextmanager
def _admission_lock(definition: Any) -> Any:
    """Serialize directory/spec/receipt creation across concurrent RPC processes."""
    import fcntl

    attempts = Path(_definition_value(definition, "root")) / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    handle = (attempts / ".admission.lock").open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _receipt_payload(definition: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    attempt_id = _check_id(payload.get("attempt_id"), "attempt_id")
    run_id = _check_id(payload.get("run_id"), "run_id")
    experiment_id = _check_id(payload.get("experiment_id"), "experiment_id")
    for name in ("spec_digest", "bundle_digest"):
        value = payload.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"invalid {name}")
    now = _utc()
    return {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "spec_digest": payload["spec_digest"],
        "bundle_digest": payload["bundle_digest"],
        "worker_id": _definition_value(definition, "worker_id"),
        "state": "PREPARED",
        "phase": "prepared",
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "boot_id": None,
        "pid": None,
        "process_start": None,
        "start_token": None,
        "heartbeat_at": None,
        "cancellation_requested": False,
        "cancellation_acknowledged": False,
        "error": None,
        "artifacts": [],
    }


def _final_artifacts(
    definition: Any, receipt: Mapping[str, Any]
) -> list[dict[str, object]]:
    """Seal only immutable, complete files into the receipt-authorized inventory."""
    from sparselab.training.manifest import sha256_file

    root = Path(_definition_value(definition, "root"))
    run = root / "runs" / str(receipt["run_id"])
    result: list[dict[str, object]] = []
    declared = set()
    manifest_path = run / "manifest.json"
    if manifest_path.is_file():
        from sparselab.training.manifest import read_manifest

        declared = {
            item["relative_path"]
            for item in read_manifest(manifest_path).get("artifacts", [])
        }
    if run.is_dir():
        for path in sorted(run.rglob("*")):
            if path.is_symlink():
                raise ValueError("run contains symlinked artifact")
            relative = path.relative_to(run).as_posix()
            if (
                path.is_file()
                and (
                    relative in declared
                    or not any(
                        part.startswith(".") for part in path.relative_to(run).parts
                    )
                )
                and not path.name.endswith(".tmp")
            ):
                result.append(
                    {
                        "relative_path": f"run/{path.relative_to(run).as_posix()}",
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
    sys.stdout.flush()
    sys.stderr.flush()
    attempt = _attempt_dir(definition, str(receipt["attempt_id"]))
    for name in ("stdout.log", "stderr.log"):
        path = attempt / name
        live = attempt / f"{Path(name).stem}.live.log"
        if live.is_file():
            shutil.copyfile(live, path)
            os.chmod(path, 0o600)
        if path.is_file():
            result.append(
                {
                    "relative_path": f"logs/{name}",
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return result


def _run_terminal_status(definition: Any, run_id: str) -> str | None:
    progress = (
        Path(_definition_value(definition, "root")) / "runs" / run_id / "progress.json"
    )
    if not progress.is_file() or progress.is_symlink():
        return None
    status = _strict_json(progress).get("status")
    return status if status in {"completed", "interrupted", "failed"} else None


def _check_worker_interpreter(definition: Any) -> None:
    python = Path(_definition_value(definition, "python"))
    if not python.is_absolute() or python.resolve(strict=True) != Path(
        sys.executable
    ).resolve(strict=True):
        raise ValueError(
            "registered worker Python differs from the running interpreter"
        )


def _runtime_for_definition(definition: Any) -> Any:
    engine, backend, index = (
        _definition_value(definition, key)
        for key in ("engine", "backend", "device_index")
    )
    infos = discover_runtimes()
    for runtime in infos:
        if (
            runtime.engine == engine
            and runtime.backend == backend
            and runtime.device_index == index
        ):
            return runtime
    raise ValueError("configured runtime is absent from passive discovery")


def _acquire_config_lease(
    definition: Any,
    passive: Any,
    config: RunConfig | None = None,
    *,
    state_dir: Path | None = None,
) -> DeviceLease | MultiDeviceLease | None:
    worker_id = _definition_value(definition, "worker_id")
    if (
        config is None
        or not config.runtime.secondary_device_indices
        or passive.backend == "cpu"
    ):
        return acquire_lease(
            worker_id=worker_id,
            backend=passive.backend,
            physical_device_id=passive.physical_device_id,
            state_dir=state_dir,
        )
    if not passive.physical_device_id or not str(passive.physical_device_id).strip():
        return acquire_lease(
            worker_id=worker_id,
            backend=passive.backend,
            physical_device_id=None,
            state_dir=state_dir,
        )
    engine = _definition_value(definition, "engine")
    infos_by_index = {
        info.device_index: info
        for info in discover_runtimes()
        if info.engine == engine and info.backend == passive.backend
    }
    physical_ids = [str(passive.physical_device_id).strip()]
    for sec_idx in config.runtime.secondary_device_indices:
        sec_info = infos_by_index.get(sec_idx)
        if sec_info is None:
            raise ValueError(
                f"secondary_device_indices index {sec_idx} is absent from passive discovery"
            )
        if (
            not sec_info.physical_device_id
            or not str(sec_info.physical_device_id).strip()
        ):
            return acquire_lease(
                worker_id=worker_id,
                backend=passive.backend,
                physical_device_id=None,
                state_dir=state_dir,
            )
        physical_ids.append(str(sec_info.physical_device_id).strip())
    return acquire_multi_device_leases(
        worker_id=worker_id,
        backend=passive.backend,
        physical_device_ids=physical_ids,
        state_dir=state_dir,
    )


def initialize_worker(definition: Any) -> None:
    """Persist a private immutable registration; conflicting endpoints fail closed."""
    root = Path(_definition_value(definition, "root"))
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or root.stat().st_mode & 0o077:
        raise PermissionError("worker root must be a private real directory")
    path = root / "worker.json"
    payload = (
        definition.model_dump(mode="json")
        if hasattr(definition, "model_dump")
        else dict(definition)
    )
    if path.exists():
        if path.is_symlink() or _strict_json(path) != payload:
            raise ValueError("worker root is bound to a different definition")
        return
    _atomic_json(path, payload)


def _caps(
    definition: Any,
    *,
    validation_status: str = "unverified",
    runtime: Any | None = None,
) -> dict[str, Any]:
    from .models import current_required_versions

    versions = current_required_versions()
    runtime = runtime or _runtime_for_definition(definition)
    from .hosted_status import instance_identity

    instance = instance_identity(Path(_definition_value(definition, "root")))
    return {
        "schema_version": 1,
        **({"instance_id": instance} if instance is not None else {}),
        **{
            key: (
                str(_definition_value(definition, key))
                if key in {"python", "root"}
                else _definition_value(definition, key)
            )
            for key in (
                "worker_id",
                "name",
                "transport",
                "host",
                "python",
                "root",
                "engine",
                "backend",
                "device_index",
            )
        },
        "runtime": runtime.as_dict(),
        "supported_precisions": list(runtime.tested_precisions),
        "supported_features": list(runtime.tested_features),
        "max_concurrent_runs": 1,
        "validation_status": validation_status,
        "validated_at": runtime.validated_at,
        "status": "idle",
        "last_seen_at": _utc(),
        "protocol_versions": versions["protocol"],
        "sparselab_version": _package_version(),
        "source_identity_sha256": source_identity()["sha256"],
        "architecture_versions": versions["architecture"],
        "config_versions": versions["config"],
        "manifest_versions": versions["manifest"],
        "checkpoint_versions": versions["checkpoint"],
        "bundle_versions": versions["bundle"],
        "state_codecs": versions["state_codecs"],
    }


def _package_version() -> str:
    try:
        return version("tiny-sparse-lab")
    except PackageNotFoundError:
        return "unknown"


def discover_worker(definition: Any) -> dict[str, Any]:
    """Fresh passive inventory; saved validation is never runtime authority."""
    _check_worker_interpreter(definition)
    fresh = _caps(definition)
    fresh["last_seen_at"] = _utc()
    return fresh


def _effective_config(definition: Any, config: RunConfig) -> RunConfig:
    payload = config.model_dump(mode="json")
    runtime = payload["runtime"]
    if runtime["engine"] != _definition_value(definition, "engine"):
        raise ValueError("experiment engine does not match worker")
    requested = runtime["backend"]
    actual = _definition_value(definition, "backend")
    if requested not in {"auto", actual}:
        raise ValueError("experiment backend does not match worker")
    if runtime["device_index"] != _definition_value(definition, "device_index"):
        raise ValueError("experiment device index does not match worker")
    runtime["backend"] = actual
    return RunConfig.model_validate(payload)


def validate_worker(
    definition: Any, config: RunConfig, bundle_digest: str | None = None
) -> dict[str, Any]:
    effective = _effective_config(definition, config)
    authorization = authorize_worker(definition, effective)
    passive = _runtime_for_definition(definition)
    try:
        lease = _acquire_config_lease(definition, passive, effective)
    except ValueError as error:
        failed = _caps(definition, validation_status="failed", runtime=passive)
        _atomic_json(
            Path(_definition_value(definition, "root")) / "capabilities.json", failed
        )
        return {
            "ok": False,
            "code": "VALIDATION_FAILED",
            "message": str(error),
            "runtime": passive.as_dict(),
        }
    if lease is None:
        return {
            "ok": False,
            "code": "BUSY",
            "message": "physical device is leased",
            "runtime": passive.as_dict(),
        }
    try:
        tested = validate_runtime(effective, authorization=authorization)
    except (OSError, ValueError, RuntimeError) as error:
        failed = _caps(definition, validation_status="failed", runtime=passive)
        _atomic_json(
            Path(_definition_value(definition, "root")) / "capabilities.json", failed
        )
        return {
            "ok": False,
            "code": "VALIDATION_FAILED",
            "message": str(error),
            "runtime": passive.as_dict(),
        }
    finally:
        lease.close()
    report = {
        "ok": True,
        "bundle_digest": bundle_digest,
        "runtime": tested.as_dict(),
        "runtime_authorization": authorization.as_dict(),
        "capabilities": _caps(definition, validation_status="passed", runtime=tested),
    }
    _atomic_json(
        Path(_definition_value(definition, "root")) / "capabilities.json",
        report["capabilities"],
    )
    return report


def _spec_from_path(path: Path) -> dict[str, Any]:
    spec = _strict_json(path)
    digest = canonical_digest(spec)
    return {"payload": spec, "digest": digest}


def _attempt_cold_verify(directory: Path) -> bool:
    """Operational launch policy; missing external/legacy policy stays independent."""
    path = directory / "verification.json"
    if path.is_symlink():
        raise ValueError("symlinked attempt verification policy")
    if not path.exists():
        return True
    policy = _strict_json(path)
    if (
        set(policy) != {"verification_version", "cold_verify"}
        or type(policy["verification_version"]) is not int
        or policy["verification_version"] != 1
        or type(policy["cold_verify"]) is not bool
    ):
        raise ValueError("invalid attempt verification policy")
    return policy["cold_verify"]


def install_relay_key(definition: Any, attempt_id: str, source: Path) -> None:
    """Retain a sensitive control attachment; never add it to artifact inventories."""
    if source.is_symlink() or source.stat().st_size != 32:
        raise ValueError("relay key must contain exactly 32 bytes")
    key = source.read_bytes()
    with _admission_lock(definition):
        directory = _attempt_dir(definition, attempt_id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory.is_symlink():
            raise ValueError("symlinked attempt directory")
        path = directory / "relay-key.bin"
        if path.exists() or path.is_symlink():
            if (
                path.is_symlink()
                or path.stat().st_mode & 0o077
                or path.read_bytes() != key
            ):
                raise ValueError("conflicting or unsafe replay relay key")
            return
        with path.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(key)
            stream.flush()
            os.fsync(stream.fileno())


def launch_attempt(
    definition: Any, payload: Mapping[str, Any], spec_path: Path
) -> dict[str, Any]:
    """Durably prepare then detach exactly one possible executor for an attempt."""
    return _prepare_attempt(definition, payload, spec_path, spawn=True)


def prepare_attempt(
    definition: Any, payload: Mapping[str, Any], spec_path: Path
) -> dict[str, Any]:
    """Commit the identical admission checks without starting an executor."""
    return _prepare_attempt(definition, payload, spec_path, spawn=False)


def _prepare_attempt(
    definition: Any, payload: Mapping[str, Any], spec_path: Path, *, spawn: bool
) -> dict[str, Any]:
    cold_verify = payload.get("cold_verify", True)
    if type(cold_verify) is not bool:
        raise ValueError("cold_verify must be a boolean")
    expected = _receipt_payload(definition, payload)
    spec = _spec_from_path(spec_path)
    if spec["digest"] != expected["spec_digest"]:
        raise ValueError("spec attachment digest mismatch")
    from .models import (
        ExperimentSpec,
        current_required_versions,
        validate_required_versions,
    )

    model = ExperimentSpec.model_validate(spec["payload"])
    if model.bound_worker not in {None, _definition_value(definition, "name")}:
        raise ValueError("experiment is bound to a different worker")
    body = model.model_dump(mode="json")
    if (
        model.experiment_id != expected["experiment_id"]
        or model.dispatch_bundle_digest != expected["bundle_digest"]
    ):
        raise ValueError("launch identities do not match spec")
    validate_required_versions(model.required_versions)
    if model.source_identity_sha256 != source_identity()["sha256"] and not (
        model.continuation.kind == "RESUMED" and model.continuation.allow_runtime_drift
    ):
        raise ValueError("experiment source identity does not match worker")
    if model.required_versions != current_required_versions():
        raise ValueError("experiment required versions do not match worker")
    with _admission_lock(definition):
        directory = _attempt_dir(definition, expected["attempt_id"])
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = directory / "spec.json"
        if target.exists():
            stored = _spec_from_path(target)
            if stored["digest"] != expected["spec_digest"]:
                raise ValueError("conflicting replay for attempt ID")
        else:
            _atomic_json(target, body)
        receipt_path = _receipt_path(definition, expected["attempt_id"])
        if receipt_path.exists():
            receipt = _load_receipt(definition, expected["attempt_id"])
            for key in (
                "run_id",
                "experiment_id",
                "spec_digest",
                "bundle_digest",
                "worker_id",
            ):
                if receipt.get(key) != expected[key]:
                    raise ValueError("conflicting replay for attempt ID")
            if receipt["state"] in {"RUNNING", *_TERMINAL}:
                return receipt
        else:
            receipt = _write_receipt(definition, expected)
        policy_path = directory / "verification.json"
        if policy_path.exists() or policy_path.is_symlink():
            previous_cold = _attempt_cold_verify(directory)
            cold_verify = cold_verify or previous_cold
        _atomic_json(
            policy_path, {"verification_version": 1, "cold_verify": cold_verify}
        )
        from .hosted_status import process_cancel_intent, write_hosted_status

        process_cancel_intent(definition, expected["attempt_id"])
        write_hosted_status(definition, receipt)
        from .relay_execution import publish_boundary, worker_binding

        if (
            worker_binding(Path(_definition_value(definition, "root"))) is not None
            and not (directory / "relay-commit.json").exists()
        ):
            publish_boundary(definition, expected["attempt_id"])
        if spawn:
            _spawn_executor(definition, expected["attempt_id"])
        return _load_receipt(definition, expected["attempt_id"])


def _spawn_executor(definition: Any, attempt_id: str) -> None:
    from sparselab.workdir import ensure_work_dir

    ensure_work_dir()
    directory = _attempt_dir(definition, attempt_id)
    stdout = (directory / "stdout.live.log").open("ab")
    stderr = (directory / "stderr.live.log").open("ab")
    args = [
        str(_definition_value(definition, "python")),
        "-m",
        "sparselab.workers.agent",
        "execute",
    ]
    for flag, key in (
        ("--root", "root"),
        ("--worker-id", "worker_id"),
        ("--name", "name"),
        ("--engine", "engine"),
        ("--backend", "backend"),
    ):
        args.extend((flag, str(_definition_value(definition, key))))
    args.extend(
        (
            "--device-index",
            str(_definition_value(definition, "device_index")),
            "--attempt-id",
            attempt_id,
        )
    )
    try:
        subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        stdout.close()
        stderr.close()


def _continuation_run(materialized: Path, kind: str) -> Path | None:
    """Reconstruct the minimal immutable parent-run layout expected by continuation."""
    if kind == "FRESH":
        return None
    if kind not in {"RESUMED", "PROMOTED"}:
        raise ValueError("unsupported continuation kind")
    source = materialized / "continuation"
    run = source / "_run"
    if run.exists():
        return run
    manifest = source / "parent_manifest.json"
    checkpoints = source / "checkpoints"
    assets = source / "run_assets"
    if not manifest.is_file() or not checkpoints.is_dir() or not assets.is_dir():
        raise ValueError("continuation bundle is incomplete")
    for member in source.rglob("*"):
        if member.is_symlink():
            raise ValueError("symlink in materialized continuation")
    run.mkdir()
    shutil.copy2(manifest, run / "manifest.json")
    shutil.copytree(checkpoints, run / "checkpoints")
    for member in assets.iterdir():
        destination = run / member.name
        if member.is_dir():
            shutil.copytree(member, destination)
        else:
            shutil.copy2(member, destination)
    for name in ("resolved_config.yaml", "tokenizer.json", "tokenizer_manifest.json"):
        candidate = source / name
        if candidate.is_file() and not (run / name).exists():
            shutil.copy2(candidate, run / name)
    return run


def _continuation_checkpoint(run: Path | None) -> Path:
    if run is None:
        raise ValueError("fresh execution has no continuation checkpoint")
    generations = sorted(
        path for path in (run / "checkpoints").iterdir() if path.is_dir()
    )
    if len(generations) != 1:
        raise ValueError("continuation bundle must contain exactly one generation")
    return generations[0]


def _claim_file(directory: Path) -> Any:
    import fcntl

    lock = (directory / "claim.lock").open("a+b")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    return lock


def _cancelled(directory: Path) -> bool:
    marker = directory / "cancel.json"
    return marker.is_file() and not marker.is_symlink()


def _heartbeat(definition: Any, attempt_id: str, stop: threading.Event) -> None:
    while not stop.wait(5):
        try:

            def touch(receipt: dict[str, Any]) -> dict[str, Any]:
                if receipt["state"] == "RUNNING":
                    receipt["heartbeat_at"] = _utc()
                return receipt

            from .hosted_status import process_cancel_intent, write_hosted_status

            process_cancel_intent(definition, attempt_id)
            receipt = _mutate_receipt(definition, attempt_id, touch)
            write_hosted_status(definition, receipt)
            if receipt["state"] != "RUNNING":
                return
        except OSError, ValueError, TypeError:
            _LOGGER.exception("Worker receipt heartbeat failed")
            return


def _failure(error: Exception) -> dict[str, str]:
    # Errors travel over a protocol boundary. Do not expose tracebacks, roots, or env.
    return {"code": type(error).__name__.upper(), "message": str(error)[:1024]}


def _terminal_receipt(
    definition: Any, attempt_id: str, *, state: str, phase: str, **fields: Any
) -> dict[str, Any]:
    """Publish a terminal result atomically; an existing terminal is immutable."""

    def finish(receipt: dict[str, Any]) -> dict[str, Any]:
        if receipt["state"] in _TERMINAL:
            return receipt
        receipt.update(state=state, phase=phase, finished_at=_utc(), **fields)
        return receipt

    result = _mutate_receipt(definition, attempt_id, finish)
    from .hosted_status import write_hosted_status

    write_hosted_status(definition, result)
    from .relay_execution import publish_boundary

    try:
        publish_boundary(definition, attempt_id, terminal=True)
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        _atomic_json(
            _attempt_dir(definition, attempt_id) / "relay-error.json",
            {
                "error": _failure(error),
                "observed_at": _utc(),
            },
        )
        _LOGGER.error("Terminal relay publication failed: %s", error)
    return result


def execute_attempt(definition: Any, attempt_id: str) -> dict[str, Any]:
    """Claim and execute a prepared attempt; duplicate wrappers never train."""
    directory = _attempt_dir(definition, attempt_id)
    receipt = _load_receipt(definition, attempt_id)
    claim = _claim_file(directory)
    if claim is None:
        return receipt
    try:
        receipt = _load_receipt(definition, attempt_id)
        if receipt["state"] != "PREPARED":
            return receipt
        if _cancelled(directory):
            return _terminal_receipt(
                definition,
                attempt_id,
                state="INTERRUPTED",
                phase="cancelled_before_initialization",
                cancellation_requested=True,
                cancellation_acknowledged=True,
            )
        spec = _spec_from_path(directory / "spec.json")["payload"]
        from .models import (
            ExperimentSpec,
            current_required_versions,
            validate_required_versions,
        )

        try:
            typed_spec = ExperimentSpec.model_validate(spec)
            validate_required_versions(typed_spec.required_versions)
            if typed_spec.required_versions != current_required_versions():
                raise ValueError("stored experiment schema versions are unsupported")
            if typed_spec.bound_worker not in {
                None,
                _definition_value(definition, "name"),
            }:
                raise ValueError("stored experiment is bound to a different worker")
            if typed_spec.experiment_id != receipt["experiment_id"]:
                raise ValueError("stored experiment identity does not match receipt")
            config = typed_spec.config
            _check_worker_interpreter(definition)
            effective = _effective_config(definition, config)
            envelope = typed_spec.resource_envelope
            if envelope is not None:
                check_envelope(
                    envelope,
                    workspace=Path(_definition_value(definition, "root")),
                    rss_bytes=current_process_rss_bytes(),
                )
            if typed_spec.source_identity_sha256 != source_identity()[
                "sha256"
            ] and not (
                typed_spec.continuation.kind == "RESUMED"
                and typed_spec.continuation.allow_runtime_drift
            ):
                raise ValueError(
                    "stored experiment source identity does not match worker"
                )
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            return _terminal_receipt(
                definition,
                attempt_id,
                state="FAILED",
                phase="rejected_before_initialization",
                error=_failure(error),
            )
        passive = _runtime_for_definition(definition)
        try:
            lease = _acquire_config_lease(definition, passive, effective)
        except ValueError as error:
            return _terminal_receipt(
                definition,
                attempt_id,
                state="FAILED",
                phase="rejected_before_initialization",
                error=_failure(error),
            )
        if lease is None:
            # PREPARED may be reconciled/spawned later; it has not begun execution.
            return receipt
        try:

            def begin(current: dict[str, Any]) -> dict[str, Any]:
                if current["state"] == "PREPARED":
                    current.update(
                        state="RUNNING",
                        phase="initializing",
                        started_at=_utc(),
                        boot_id=boot_identity(),
                        pid=os.getpid(),
                        process_start=str(process_start()),
                        start_token=uuid.uuid4().hex,
                        heartbeat_at=_utc(),
                    )
                return current

            receipt = _mutate_receipt(definition, attempt_id, begin)
            if receipt["state"] != "RUNNING":
                return receipt
            stop = threading.Event()
            beat = threading.Thread(
                target=_heartbeat, args=(definition, attempt_id, stop), daemon=True
            )
            beat.start()
            try:
                from .hosted_status import process_cancel_intent, write_hosted_status
                from .relay_execution import publish_boundary

                process_cancel_intent(definition, attempt_id)
                write_hosted_status(definition, receipt)
                publish_boundary(definition, attempt_id)
                if _cancelled(directory):
                    return _terminal_receipt(
                        definition,
                        attempt_id,
                        state="INTERRUPTED",
                        phase="cancelled_before_validation",
                        cancellation_requested=True,
                        cancellation_acknowledged=True,
                    )
                # Materialization authenticates cached bytes and the bundle closure.
                from .bundles import materialize_dispatch_bundle

                verification = verification_options(
                    Path(_definition_value(definition, "root")),
                    cold=_attempt_cold_verify(directory),
                )
                materialized = directory / "bundle"
                manifest = materialize_dispatch_bundle(
                    Path(_definition_value(definition, "root")),
                    receipt["bundle_digest"],
                    materialized,
                    **verification,
                )
                if manifest.digest() != receipt["bundle_digest"]:
                    raise ValueError("installed bundle digest changed")
                if (
                    spec.get("config_sha256") != manifest.config_sha256
                    or config.model_dump(mode="json")
                    != manifest.config.model_dump(mode="json")
                    or spec.get("source_identity_sha256")
                    != manifest.source_identity_sha256
                    or spec.get("continuation")
                    != manifest.continuation.model_dump(mode="json")
                ):
                    raise ValueError(
                        "experiment spec does not match verified dispatch bundle"
                    )
                asset_root = materialized / "assets"
                effective = effective.model_copy(
                    update={
                        "tokenizer": effective.tokenizer.model_copy(
                            update={"path": asset_root / "tokenizer.json"}
                        ),
                        "model": effective.model.model_copy(
                            update={
                                "memory_package_path": asset_root / "portable_package"
                                if effective.model.memory_package_path is not None
                                else None
                            }
                        ),
                    }
                )
                continuation_spec = spec.get("continuation", {})
                allow_bundle_drift = bool(
                    continuation_spec.get("kind") == "RESUMED"
                    and continuation_spec.get("allow_runtime_drift", False)
                )
                authorization = authorize_worker(definition, effective)
                require_authorization(effective, authorization)
                tested = validate_runtime(effective, authorization=authorization)
                _atomic_json(
                    Path(_definition_value(definition, "root")) / "capabilities.json",
                    _caps(definition, validation_status="passed", runtime=tested),
                )
                if _cancelled(directory):
                    return _terminal_receipt(
                        definition,
                        attempt_id,
                        state="INTERRUPTED",
                        phase="cancelled_before_training",
                        cancellation_requested=True,
                        cancellation_acknowledged=True,
                    )
                from sparselab.staging import stage

                stage_dir = directory / "stage"
                stage(
                    effective,
                    stage_dir,
                    through="warmup",
                    prepared_inputs=materialized,
                    allow_runtime_drift=allow_bundle_drift,
                    inherited_fds=lease.inherited_fds,
                    cancel_path=directory / "cancel.json",
                    authorization=authorization,
                    resource_envelope=envelope,
                    tokenizer_batch_documents=typed_spec.tokenizer_batch_documents,
                    tokenizer_batch_source_bytes=typed_spec.tokenizer_batch_source_bytes,
                    **verification,
                )
                if _cancelled(directory):
                    return _terminal_receipt(
                        definition,
                        attempt_id,
                        state="INTERRUPTED",
                        phase="cancelled_before_training",
                        cancellation_requested=True,
                        cancellation_acknowledged=True,
                    )
                from sparselab.training.trainer import train

                train_payload = effective.model_dump(mode="json")
                allocation_path = effective.dataset.allocation_manifest_path
                if allocation_path is not None:
                    train_payload["dataset"]["allocation_manifest_path"] = str(
                        stage_dir / "assets" / "allocation" / allocation_path.name
                    )
                train_payload["logging"]["root_dir"] = str(
                    Path(_definition_value(definition, "root")) / "runs"
                )
                effective = RunConfig.model_validate(train_payload)

                def training(current: dict[str, Any]) -> dict[str, Any]:
                    if current["state"] == "RUNNING":
                        current.update(phase="training", training_started_at=_utc())
                    return current

                receipt = _mutate_receipt(definition, attempt_id, training)
                if receipt["state"] != "RUNNING":
                    return receipt
                continuation = spec.get("continuation", {})
                kind = continuation.get("kind", "FRESH")
                bundle_run = _continuation_run(materialized, kind)
                kwargs: dict[str, Any] = {
                    "run_id": receipt["run_id"],
                    "authorization": authorization,
                    "worker_id": receipt["worker_id"],
                    "resource_envelope": envelope,
                    "tokenizer_batch_documents": typed_spec.tokenizer_batch_documents,
                    "tokenizer_batch_source_bytes": typed_spec.tokenizer_batch_source_bytes,
                    **verification,
                    "cancel_path": directory / "cancel.json",
                    "stage_bundle": stage_dir,
                    "experiment_id": receipt["experiment_id"],
                    "attempt_id": receipt["attempt_id"],
                    "dispatch_metadata": {
                        "spec_digest": receipt["spec_digest"],
                        "bundle_digest": receipt["bundle_digest"],
                        "matrix": spec.get("matrix"),
                        **(
                            {"plan": spec["plan"]}
                            if spec.get("plan") is not None
                            else {}
                        ),
                    },
                    "requested_config_override": config.model_dump(mode="json"),
                    "allow_runtime_drift": bool(
                        continuation.get("allow_runtime_drift", False)
                    ),
                }
                from .relay_execution import publish_boundary, worker_binding

                if (
                    worker_binding(Path(_definition_value(definition, "root")))
                    is not None
                ):
                    kwargs["checkpoint_committed"] = lambda run, record: (
                        publish_boundary(definition, attempt_id, checkpoint=record)
                    )
                selected = (
                    _continuation_checkpoint(bundle_run) if kind != "FRESH" else None
                )
                if kind == "RESUMED":
                    kwargs[
                        "extend_budget"
                        if continuation.get("budget_extension")
                        else "resume"
                    ] = selected
                elif kind == "PROMOTED":
                    kwargs["promote"] = selected
                elif kind != "FRESH":
                    raise ValueError("unsupported continuation kind")
                train(effective, **kwargs)
                terminal = _run_terminal_status(definition, receipt["run_id"])
                if terminal == "completed":
                    try:
                        report = read_triage(
                            receipt["run_id"],
                            Path(_definition_value(definition, "root")) / "runs",
                        )
                    except (OSError, ValueError) as error:
                        print(f"POST-TRAIN TRIAGE UNKNOWN: {error}", flush=True)
                    else:
                        if report is None:
                            print(
                                "POST-TRAIN TRIAGE UNKNOWN: no verified report",
                                flush=True,
                            )
                        else:
                            print(
                                f"POST-TRAIN TRIAGE: {triage_summary(report)}",
                                flush=True,
                            )
                artifacts = _final_artifacts(definition, receipt)
                if terminal == "interrupted":
                    # A signal interruption is not a user cancellation.  The marker
                    # is the only acknowledgement authority.
                    cancelled = _cancelled(directory)
                    return _terminal_receipt(
                        definition,
                        attempt_id,
                        state="INTERRUPTED",
                        phase="cancelled" if cancelled else "interrupted",
                        artifacts=artifacts,
                        cancellation_requested=cancelled,
                        cancellation_acknowledged=cancelled,
                    )
                if terminal == "completed":
                    # Natural completion wins a racing cancellation request.
                    return _terminal_receipt(
                        definition,
                        attempt_id,
                        state="COMPLETE",
                        phase="complete",
                        artifacts=artifacts,
                    )
                raise RuntimeError(
                    "trainer returned without a terminal progress record"
                )
            except InterruptedError as error:
                if not _cancelled(directory):
                    _LOGGER.exception("Worker training attempt failed")
                    # A stop-shaped exception is not cancellation authority. Retain
                    # failure evidence before the heartbeat and claim are released.
                    current = _load_receipt(definition, attempt_id)
                    _terminal_receipt(
                        definition,
                        attempt_id,
                        state="FAILED",
                        phase="failed",
                        artifacts=_final_artifacts(definition, current),
                        error=_failure(error),
                    )
                    raise
                return _terminal_receipt(
                    definition,
                    attempt_id,
                    state="INTERRUPTED",
                    phase="cancelled_before_training",
                    cancellation_requested=True,
                    cancellation_acknowledged=True,
                )
            except Exception as error:
                _LOGGER.exception("Worker training attempt failed")
                current = _load_receipt(definition, attempt_id)
                if current.get("state") == "RUNNING":
                    _terminal_receipt(
                        definition,
                        attempt_id,
                        state="FAILED",
                        phase="failed",
                        artifacts=_final_artifacts(definition, current),
                        error=_failure(error),
                    )
                raise
            finally:
                stop.set()
                beat.join(timeout=1)
        finally:
            lease.close()
    finally:
        claim.close()


def status_worker(definition: Any, attempt_id: str | None = None) -> dict[str, Any]:
    attempts = Path(_definition_value(definition, "root")) / "attempts"
    all_paths = sorted(attempts.glob("*/receipt.json"))
    selected = (
        [_receipt_path(definition, attempt_id)] if attempt_id else all_paths[-32:]
    )
    observed: dict[str, dict[str, Any]] = {}
    passive = _runtime_for_definition(definition)
    for path in all_paths if attempt_id is None else selected:
        if not path.is_file() or path.is_symlink():
            continue
        identifier = path.parent.name
        receipt = _load_receipt(definition, identifier)
        if receipt.get("state") == "RUNNING" and not process_matches(
            receipt.get("pid"), receipt.get("process_start"), receipt.get("boot_id")
        ):
            recovery_lease = acquire_lease(
                worker_id=_definition_value(definition, "worker_id"),
                backend=passive.backend,
                physical_device_id=passive.physical_device_id,
            )
            if recovery_lease is not None:
                claim = None
                try:
                    claim = _claim_file(path.parent)
                    if claim is not None:
                        receipt = _terminal_receipt(
                            definition,
                            identifier,
                            state="UNKNOWN",
                            phase="executor_not_live",
                            artifacts=_final_artifacts(definition, receipt),
                            error={
                                "code": "EXECUTOR_UNKNOWN",
                                "message": "executor and inherited resource holders are no longer live",
                            },
                        )
                finally:
                    if claim is not None:
                        claim.close()
                    recovery_lease.close()
        observed[identifier] = receipt
    busy = any(
        item["state"] == "RUNNING"
        and process_matches(
            item.get("pid"), item.get("process_start"), item.get("boot_id")
        )
        for item in observed.values()
    )
    # A validation probe has no receipt; consult the durable lease namespace too.
    lease = acquire_lease(
        worker_id=_definition_value(definition, "worker_id"),
        backend=passive.backend,
        physical_device_id=passive.physical_device_id,
    )
    if lease is None:
        busy = True
    else:
        lease.close()
    receipts = [
        observed[path.parent.name] for path in selected if path.parent.name in observed
    ]
    capabilities = discover_worker(definition)
    capabilities["status"] = "busy" if busy else "idle"
    return {
        "capabilities": capabilities,
        "receipt": receipts[0] if attempt_id and receipts else None,
        "receipts": receipts if not attempt_id else [],
    }


def cancel_attempt(definition: Any, attempt_id: str) -> dict[str, Any] | None:
    """Persist cancellation even when delivery precedes attempt preparation."""
    directory = _attempt_dir(definition, attempt_id)
    with _receipt_lock(definition, attempt_id):
        path = _receipt_path(definition, attempt_id)
        receipt = _load_receipt(definition, attempt_id) if path.exists() else None
        if receipt is not None and receipt["state"] in _TERMINAL:
            return receipt
        marker = directory / "cancel.json"
        if not marker.exists():
            _atomic_json(
                marker,
                {
                    "schema_version": 1,
                    "attempt_id": attempt_id,
                    "reason": "user",
                    "requested_at": _utc(),
                },
            )
        if receipt is None:
            return None
        receipt["cancellation_requested"] = True
        if receipt["state"] == "PREPARED":
            claim = _claim_file(directory)
            if claim is not None:
                try:
                    receipt.update(
                        state="INTERRUPTED",
                        phase="cancelled_before_initialization",
                        cancellation_acknowledged=True,
                        finished_at=_utc(),
                    )
                finally:
                    claim.close()
        return _write_receipt(definition, receipt)
