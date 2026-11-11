"""Bounded foreground adapter for user-supplied Google Colab sessions.

The adapter deliberately has no scheduler.  It maps one idle control RPC to the
existing finite stdio protocol and maps a prepared launch to exactly one official
``colab exec`` foreground process.  While that process is outstanding, status
and cancellation use the independent Contents upload/download API only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from klab.canonical import canonical_json
from klab.hosted.transport import ColabFileTransport
from klab.training.manifest import sha256_file
from klab.workdir import ensure_scratch_dir
from klab.workers.models import (
    PROTOCOL_VERSION,
    WorkerDefinition,
    validate_operation,
)
from klab.workers.transport import (
    ProtocolError,
    ProtocolReply,
    RemoteProtocolError,
    read_frame,
    validate_operation_result,
    write_frame,
)

_CONTROL_DIRECTORY = ".colab-control"
_STATUS_NAME = "hosted-status.json"
_KERNEL_STATUS_NAME = "kernel-status.json"
_CANCEL_NAME = "colab-cancel.json"
_DELIVERY_STATES = frozenset(
    {"PREPARED", "ISSUED", "OBSERVED_RUNNING", "OBSERVED_TERMINAL"}
)
_STATUS_VERSION = 1


class ColabTransportError(ProtocolError):
    """An unavailable or malformed official Colab CLI interaction."""


def _deadline(timeout: float) -> float:
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    return time.monotonic() + float(timeout)


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Colab operation deadline exhausted")
    return remaining


def _endpoint(worker: WorkerDefinition) -> Any:
    if worker.transport != "colab":
        raise ValueError("call_colab requires a Colab worker")
    endpoint = getattr(worker, "colab", None)
    if endpoint is None:
        raise ValueError("Colab worker is missing endpoint configuration")
    for name in ("session", "instance_id", "job_timeout_seconds"):
        if not getattr(endpoint, name, None):
            raise ValueError(f"Colab endpoint missing {name}")
    return endpoint


def _file_transport(worker: WorkerDefinition, *, timeout: float) -> Any:
    """Construct the hosted transfer boundary lazily to keep base installs light."""
    endpoint = _endpoint(worker)
    try:
        from klab.hosted.transport import ColabFileTransport
    except ImportError as error:  # pragma: no cover - integration error, not fallback
        raise ColabTransportError("Colab file transport is unavailable") from error
    return ColabFileTransport(
        endpoint.session,
        config_path=getattr(endpoint, "config_path", None),
        auth=getattr(endpoint, "auth", None) or "oauth2",
        timeout=timeout,
    )


def _control_path(worker: WorkerDefinition, *parts: str) -> str:
    root = Path(worker.root)
    if not root.is_absolute():
        raise ValueError("Colab worker root must be absolute")
    return str(root.joinpath(_CONTROL_DIRECTORY, *parts))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_json(dict(value)))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > 1024 * 1024:
            raise ColabTransportError("downloaded Colab JSON exceeds metadata limit")
    except OSError as error:
        raise ColabTransportError("unable to stat downloaded Colab JSON") from error

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ColabTransportError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ColabTransportError("invalid downloaded Colab JSON") from error
    if not isinstance(value, dict):
        raise ColabTransportError("downloaded Colab JSON must be an object")
    return value


def _cli_argv(worker: WorkerDefinition, *, file: Path, timeout: float) -> list[str]:
    endpoint = _endpoint(worker)
    argv = ["colab"]
    config_path = getattr(endpoint, "config_path", None)
    if config_path is not None:
        argv.extend(["--config", str(config_path)])
    argv.extend(
        [
            "--auth",
            getattr(endpoint, "auth", None) or "oauth2",
            "exec",
            "-s",
            endpoint.session,
        ]
    )
    argv.extend(["-f", str(file), "--timeout", str(timeout)])
    return argv


def _agent_argv(
    worker: WorkerDefinition, command: str, *, attempt_id: str | None = None
) -> list[str]:
    argv = [
        str(worker.python),
        "-m",
        "klab.workers.agent",
        command,
        "--root",
        str(worker.root),
        "--worker-id",
        worker.worker_id,
        "--name",
        worker.name,
        "--engine",
        worker.engine,
        "--backend",
        worker.backend,
        "--device-index",
        str(worker.device_index),
    ]
    if attempt_id is not None:
        argv.extend(["--attempt-id", attempt_id])
    return argv


def _cell_program(body: str, *, request: str, assembly: str, timeout: float) -> str:
    from klab import runtime_env_subprocess

    bounded = Path(runtime_env_subprocess.__file__).read_text(encoding="utf-8")
    # Function-local data never leaves request bytes (including keys) in notebook globals.
    return (
        "def _klab_rpc_cell():\n import json, pathlib, time\n"
        + f" native = {{'__name__': 'klab_rpc'}}\n exec(compile({bounded!r}, '<klab-bounded>', 'exec'), native)\n"
        + f" deadline = time.monotonic() + {timeout!r}\n request = pathlib.Path({request!r})\n assembled = False\n"
        + " try:\n"
        + f"  exec(compile({assembly!r}, '<klab-assembly>', 'exec'), {{'_klab_deadline': deadline}})\n  assembled = True\n"
        + "".join("  " + line + "\n" for line in body.splitlines())
        + " finally:\n  if assembled: request.unlink(missing_ok=True)\n"
        + "_klab_rpc_cell()\ndel _klab_rpc_cell\n"
    )


def _control_script(
    worker: WorkerDefinition,
    request: str,
    response: str,
    *,
    assembly: str,
    timeout: float = 30,
) -> str:
    argv = _agent_argv(worker, "serve-stdio")
    body = (
        "if request.is_symlink() or request.stat().st_size > 16 * 1024**2: raise RuntimeError('oversized control request')\n"
        + f"result = native['run_bounded']({argv!r}, input=request.read_bytes(), timeout=deadline-time.monotonic(), stdout_path=pathlib.Path({response!r}), stdout_limit=16 * 1024**2)\n"
        + "if result.returncode: raise RuntimeError(result.stderr.decode('utf-8', 'replace')[:2048])"
    )
    return _cell_program(body, request=request, assembly=assembly, timeout=timeout)


def _write_script(root: Path, name: str, content: str) -> Path:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / name
    path.write_text(content, encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def _bulk_control_script(
    worker: WorkerDefinition,
    descriptor: str,
    response: str,
    *,
    assembly: str,
    timeout: float = 30,
) -> str:
    definition = worker.model_dump(
        mode="json", exclude={"colab", "relay", "instance_id"}
    )
    definition.update(transport="local", host=None)
    definition_path = descriptor + ".definition.json"
    argv = [
        str(worker.python),
        "-m",
        "klab.workers.colab_bulk",
        "--definition",
        definition_path,
        "--descriptor",
        descriptor,
        "--response",
        response,
        "--timeout",
        str(timeout),
    ]
    body = (
        f"definition = pathlib.Path({definition_path!r})\n"
        + f"definition.write_bytes({canonical_json(definition)!r})\ndefinition.chmod(0o600)\n"
        + "try:\n"
        + f" result = native['run_bounded']({argv!r}, timeout=deadline-time.monotonic())\n"
        + " if result.returncode: raise RuntimeError(result.stderr.decode('utf-8', 'replace')[:2048])\n"
        + f" if pathlib.Path({response!r}).stat().st_size > 16 * 1024**2: raise RuntimeError('oversized control response')\n"
        + "finally:\n definition.unlink(missing_ok=True)"
    )
    return _cell_program(body, request=descriptor, assembly=assembly, timeout=timeout)


def call_colab(
    worker: WorkerDefinition,
    op: str,
    payload: dict[str, Any],
    *,
    attachments: Mapping[str, Path] | None = None,
    receive_dir: Path | None = None,
    timeout: float = 30,
) -> ProtocolReply:
    """Run one idle Colab RPC through authenticated bounded file transfers.

    It intentionally refuses to run when the caller has recorded a foreground
    launch.  The controller must use :func:`download_hosted_status` instead.
    """
    deadline = _deadline(timeout)
    _endpoint(worker)
    validate_operation(op, payload)
    if receive_dir is None and op in {"records", "artifact"}:
        raise ValueError("attachment responses require receive_dir")
    receive_dir = (
        Path(receive_dir)
        if receive_dir is not None
        else ensure_scratch_dir() / "colab-replies"
    )
    receive_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    request_id = secrets.token_hex(16)
    request_local = receive_dir / f"{request_id}.request"
    response_local = receive_dir / f"{request_id}.response"
    request_remote = _control_path(worker, "rpc", f"{request_id}.request")
    response_remote = _control_path(worker, "rpc", f"{request_id}.response")
    script: Path | None = None
    bulk = op == "install_bundle" and bool(attachments)
    transfer = None
    manifest = None
    try:
        if bulk and attachments is not None:
            from klab.workers.relay import RelayStore
            from klab.workers.relay_models import RelayArtifactIdentity

            if worker.relay is None:
                raise ColabTransportError("Colab bundle installation requires a relay")
            if len(attachments) > 4096:
                raise ColabTransportError("too many Colab bulk attachments")
            total = 0
            for source in attachments.values():
                size = Path(source).stat().st_size
                if size > 256 * 1024**3:
                    raise ColabTransportError("Colab bulk attachment exceeds limit")
                total += size
            if total > 1024**4:
                raise ColabTransportError(
                    "Colab bulk attachments exceed aggregate limit"
                )
            listed = []
            store = RelayStore(
                worker.relay, role="controller", scratch_root=receive_dir
            )
            for name, source in attachments.items():
                source = Path(source)
                identity = RelayArtifactIdentity(
                    relative_path=name,
                    sha256=sha256_file(source),
                    size_bytes=source.stat().st_size,
                )
                store.put_verified(source, identity)
                listed.append(
                    {
                        "name": name,
                        "sha256": identity.sha256,
                        "length": identity.size_bytes,
                    }
                )
            request_local.write_bytes(
                canonical_json(
                    {
                        "request_id": request_id,
                        "op": op,
                        "payload": payload,
                        "attachments": listed,
                    }
                )
            )
        else:
            with request_local.open("xb") as stream:
                write_frame(
                    stream,
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "request_id": request_id,
                        "op": op,
                        "payload": payload,
                    },
                    attachments or {},
                    deadline=deadline,
                )
        request_local.chmod(0o600)
        transfer = _file_transport(worker, timeout=_remaining(deadline))
        manifest = transfer.upload(request_local, request_remote)
        require_colab_idle(
            worker,
            receive_dir / f"{request_id}.hosted-status.json",
            timeout=_remaining(deadline),
        )

        from klab.hosted.transport import assembly_python

        script = _write_script(
            receive_dir,
            f"{request_id}.py",
            (
                _bulk_control_script(
                    worker,
                    request_remote,
                    response_remote,
                    assembly=assembly_python(manifest),
                    timeout=_remaining(deadline),
                )
                if bulk
                else _control_script(
                    worker,
                    request_remote,
                    response_remote,
                    assembly=assembly_python(manifest),
                    timeout=_remaining(deadline),
                )
            ),
        )
        from klab.runtime_env_subprocess import run_bounded

        process = run_bounded(
            _cli_argv(worker, file=script, timeout=_remaining(deadline)),
            timeout=_remaining(deadline),
            output_limit=32768,
        )
        if process.returncode:
            raise ColabTransportError(
                process.stderr.decode("utf-8", "replace")[:2048]
                or "Colab exec failed while serving control request"
            )
        transfer.download(response_remote, response_local)
        with response_local.open("rb") as stream:
            frame = read_frame(
                stream,
                response=True,
                destination=receive_dir / request_id,
                expected_op=op,
            )
        if frame.header["request_id"] != request_id:
            raise ColabTransportError("Colab response request ID mismatch")
        if not frame.header["ok"]:
            error = frame.header["error"]
            raise RemoteProtocolError(
                error["code"], error["message"], error["retryable"], error["details"]
            )
        return ProtocolReply(
            result=validate_operation_result(op, frame.header["result"]),
            attachments=frame.attachments,
            request_id=request_id,
        )
    finally:
        if transfer is not None:
            owned = [
                response_remote,
                request_remote,
                request_remote + ".definition.json",
            ]
            if manifest is not None:
                owned.extend(
                    f"{request_remote}.part-{index:08d}"
                    for index in range(manifest.parts)
                )
            for remote in owned:
                if time.monotonic() >= deadline:
                    break
                try:
                    transfer.remove(remote)
                except OSError, ValueError, TimeoutError, subprocess.SubprocessError:
                    pass  # Missing files are already clean; only this UUID is touched.
        request_local.unlink(missing_ok=True)
        response_local.unlink(missing_ok=True)
        if script is not None:
            script.unlink(missing_ok=True)


def delivery_path(controller_root: Path, attempt_id: str) -> Path:
    """Return the controller-owned durable launch-delivery record path."""
    if not attempt_id or "/" in attempt_id or "\\" in attempt_id:
        raise ValueError("invalid attempt ID")
    return Path(controller_root) / ".colab" / attempt_id / "delivery.json"


def read_delivery(controller_root: Path, attempt_id: str) -> dict[str, Any] | None:
    path = delivery_path(controller_root, attempt_id)
    if not path.exists():
        return None
    value = _strict_json(path)
    required = {
        "delivery_version",
        "state",
        "attempt_id",
        "instance_id",
        "spec_digest",
        "issued_at",
    }
    if (
        set(value) != required
        or type(value["delivery_version"]) is not int
        or value["delivery_version"] != 1
        or value["state"] not in _DELIVERY_STATES
    ):
        raise ColabTransportError("invalid Colab launch-delivery record")
    if value["attempt_id"] != attempt_id:
        raise ColabTransportError("launch-delivery attempt mismatch")
    return value


def _write_delivery(controller_root: Path, value: Mapping[str, Any]) -> None:
    state = value.get("state")
    if state not in _DELIVERY_STATES:
        raise ValueError("invalid launch-delivery state")
    _atomic_json(delivery_path(controller_root, str(value["attempt_id"])), value)


def _assert_no_other_delivery(
    controller_root: Path, attempt_id: str, instance_id: str
) -> None:
    """Keep one local foreground exec owner until the observer proves it idle."""
    root = Path(controller_root) / ".colab"
    if not root.exists():
        return
    for candidate in root.glob("*/delivery.json"):
        other = _strict_json(candidate)
        if (
            other.get("attempt_id") == attempt_id
            or other.get("instance_id") != instance_id
        ):
            continue
        if other.get("state") in {"PREPARED", "ISSUED", "OBSERVED_RUNNING"}:
            raise ColabTransportError(
                "another Colab foreground delivery requires kernel reconciliation"
            )


def record_prepared_delivery(
    worker: WorkerDefinition,
    *,
    attempt_id: str,
    spec_digest: str,
    controller_root: Path,
) -> dict[str, Any]:
    """Record a successful native prepare before any foreground CLI execution."""
    endpoint = _endpoint(worker)
    _assert_no_other_delivery(controller_root, attempt_id, endpoint.instance_id)
    existing = read_delivery(controller_root, attempt_id)
    if existing is not None:
        if (
            existing["instance_id"] != endpoint.instance_id
            or existing["spec_digest"] != spec_digest
        ):
            raise ColabTransportError("conflicting Colab launch delivery")
        return existing
    record = {
        "delivery_version": 1,
        "state": "PREPARED",
        "attempt_id": attempt_id,
        "instance_id": endpoint.instance_id,
        "spec_digest": spec_digest,
        "issued_at": time.time(),
    }
    _write_delivery(controller_root, record)
    return record


def _issue_foreground_launch(
    worker: WorkerDefinition,
    *,
    attempt_id: str,
    spec_digest: str,
    controller_root: Path,
    logs_root: Path | None = None,
) -> dict[str, Any]:
    """Durably issue one foreground ``colab exec`` after the native prepare RPC.

    A crash after the ISSUED write is purposely ambiguous.  Callers must never
    silently retry it; the worker's claim and relay/status reconciliation decide
    whether an execution was actually delivered.
    """
    endpoint = _endpoint(worker)
    if len(spec_digest) != 64 or any(c not in "0123456789abcdef" for c in spec_digest):
        raise ValueError("invalid spec digest")
    _assert_no_other_delivery(controller_root, attempt_id, endpoint.instance_id)
    existing = read_delivery(controller_root, attempt_id)
    if existing is None:
        raise ColabTransportError("Colab launch must be recorded PREPARED first")
    if (
        existing["instance_id"] != endpoint.instance_id
        or existing["spec_digest"] != spec_digest
    ):
        raise ColabTransportError("conflicting Colab launch delivery")
    if existing["state"] != "PREPARED":
        raise ColabTransportError("Colab launch already issued; reconcile it instead")
    logs = Path(logs_root or (Path(controller_root) / ".colab" / attempt_id))
    logs.mkdir(parents=True, exist_ok=True, mode=0o700)
    require_colab_idle(worker, logs / "status-before-launch.json")
    prepared = download_hosted_status(worker, logs / "prepared-before-launch.json")[
        "receipt"
    ]
    if (
        prepared is None
        or prepared.get("attempt_id") != attempt_id
        or prepared.get("spec_digest") != spec_digest
        or prepared.get("state") != "PREPARED"
    ):
        raise ColabTransportError(
            "foreground launch lacks its exact native PREPARED receipt"
        )
    record = {**existing, "state": "ISSUED", "issued_at": time.time()}
    _write_delivery(controller_root, record)
    script = _write_script(
        logs,
        "foreground.py",
        "import subprocess\n"
        + f"subprocess.run({_agent_argv(worker, 'execute', attempt_id=attempt_id)!r}, check=True)\n",
    )
    stdout = (logs / "colab-exec.stdout.log").open("ab")
    stderr = (logs / "colab-exec.stderr.log").open("ab")
    (logs / "colab-exec.stdout.log").chmod(0o600)
    (logs / "colab-exec.stderr.log").chmod(0o600)
    try:
        # Deliberately do not retain this handle in the ordinary RPC cleanup path.
        subprocess.Popen(
            _cli_argv(worker, file=script, timeout=float(endpoint.job_timeout_seconds)),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    except Exception:
        stdout.close()
        stderr.close()
        raise
    stdout.close()
    stderr.close()
    return record


def issue_foreground_launch(
    worker: WorkerDefinition,
    *,
    attempt_id: str,
    spec_digest: str,
    controller_root: Path,
    logs_root: Path | None = None,
) -> dict[str, Any]:
    from .execution import _admission_lock

    # Reuse native cross-process admission; no second scheduling state machine.
    slot = Path(controller_root) / ".colab" / "slots" / _endpoint(worker).instance_id
    with _admission_lock({"root": slot}):
        return _issue_foreground_launch(
            worker,
            attempt_id=attempt_id,
            spec_digest=spec_digest,
            controller_root=controller_root,
            logs_root=logs_root,
        )


def _process_start(stat: Path) -> int:
    """Read Linux ``/proc/<pid>/stat`` field 22 without trusting ``comm``."""
    try:
        raw = stat.read_text(encoding="utf-8")
        close = raw.rfind(")")
        if close <= 0:
            raise ValueError("missing proc comm terminator")
        fields = raw[close + 1 :].split()
        value = int(fields[19])
    except (OSError, UnicodeDecodeError, IndexError, ValueError) as error:
        raise ColabTransportError("invalid remote process stat") from error
    if value <= 0:
        raise ColabTransportError("invalid remote process start")
    return value


def download_kernel_status(
    worker: WorkerDefinition, destination: Path, *, timeout: float = 30
) -> dict[str, Any]:
    """Read the observer and independently prove its IDLE process still exists.

    Contents downloads are the only transport used here: this function never
    executes notebook code and does not turn an observer into a heartbeat.
    """
    return verify_kernel_observer(
        _file_transport(worker, timeout=timeout),
        str(worker.root),
        destination,
        expected_instance_id=_endpoint(worker).instance_id,
    )


def verify_kernel_observer(
    transport: ColabFileTransport,
    worker_root: str,
    destination: Path,
    *,
    expected_instance_id: str | None = None,
) -> dict[str, Any]:
    """Prove occupancy through Contents only, including before first setup."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    transport.download(str(Path(worker_root) / _KERNEL_STATUS_NAME), destination)
    value = _strict_json(destination)
    required = {
        "kernel_status_version",
        "instance_id",
        "boot_id",
        "pid",
        "process_start",
        "state",
        "observed_at",
    }
    if (
        set(value) != required
        or type(value["kernel_status_version"]) is not int
        or value["kernel_status_version"] != 1
    ):
        raise ColabTransportError("invalid kernel status fields")
    if (
        expected_instance_id is not None
        and value["instance_id"] != expected_instance_id
    ):
        raise ColabTransportError("Colab instance nonce changed")
    if (
        not isinstance(value["boot_id"], str)
        or not value["boot_id"]
        or type(value["pid"]) is not int
        or value["pid"] <= 0
        or type(value["process_start"]) is not int
        or value["process_start"] <= 0
        or not isinstance(value["instance_id"], str)
        or not value["instance_id"]
        or value["state"] not in {"BUSY", "IDLE"}
        or type(value["observed_at"]) not in {int, float}
        or not math.isfinite(float(value["observed_at"]))
    ):
        raise ColabTransportError("invalid kernel status values")
    observed_at = float(value["observed_at"])
    if observed_at - time.time() > 30:
        raise ColabTransportError("kernel status clock is implausibly ahead")
    if value["state"] == "BUSY" and time.time() - observed_at > 30:
        raise ColabTransportError("busy kernel status is stale")
    stat = destination.with_name(f"{destination.name}.proc-stat")
    boot = destination.with_name(f"{destination.name}.boot-id")
    try:
        transport.download(f"/proc/{value['pid']}/stat", stat)
        transport.download("/proc/sys/kernel/random/boot_id", boot)
        if _process_start(stat) != value["process_start"]:
            raise ColabTransportError("kernel process identity changed")
        if boot.read_text(encoding="utf-8").strip() != value["boot_id"]:
            raise ColabTransportError("kernel boot identity changed")
    finally:
        stat.unlink(missing_ok=True)
        boot.unlink(missing_ok=True)
    return value


def download_hosted_status(
    worker: WorkerDefinition, destination: Path, *, timeout: float = 30
) -> dict[str, Any]:
    """Fetch native worker heartbeat via Contents without invoking the kernel."""
    deadline = _deadline(timeout)
    endpoint = _endpoint(worker)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _file_transport(worker, timeout=_remaining(deadline)).download(
        str(Path(worker.root) / _STATUS_NAME), destination
    )
    value = _strict_json(destination)
    required = {
        "hosted_status_version",
        "instance_id",
        "observed_at",
        "capability",
        "receipt",
        "origin_id",
        "newest_commit",
    }
    if (
        set(value) != required
        or type(value["hosted_status_version"]) is not int
        or value["hosted_status_version"] != _STATUS_VERSION
    ):
        raise ColabTransportError("invalid hosted status fields")
    if value["instance_id"] != endpoint.instance_id:
        raise ColabTransportError("Colab instance nonce changed")
    if type(value["observed_at"]) not in {int, float} or not math.isfinite(
        float(value["observed_at"])
    ):
        raise ColabTransportError("invalid hosted status observation time")
    if abs(time.time() - float(value["observed_at"])) > 30:
        raise ColabTransportError("hosted status is stale or clock-skewed")
    if (
        not isinstance(value["capability"], dict)
        or not isinstance(value["origin_id"], str)
        or not value["origin_id"]
    ):
        raise ColabTransportError("invalid hosted status values")
    if value["newest_commit"] is not None:
        from .relay_models import RelayCommitDescriptor

        try:
            reported = RelayCommitDescriptor.model_validate(value["newest_commit"])
        except ValueError as error:
            raise ColabTransportError("invalid reported relay descriptor") from error
        if (
            reported.instance_id != endpoint.instance_id
            or reported.worker_id != worker.worker_id
        ):
            raise ColabTransportError("reported relay descriptor identity changed")
    receipt = value["receipt"]
    if receipt is not None:
        if not isinstance(receipt, dict):
            raise ColabTransportError("invalid hosted status receipt")
        terminal = receipt.get("state") in (
            "COMPLETE",
            "FAILED",
            "INTERRUPTED",
            "UNKNOWN",
        )
        stamp = (
            receipt.get("updated_at") or receipt.get("finished_at")
            if terminal
            else receipt.get("heartbeat_at") or receipt.get("updated_at")
        )
        if not isinstance(stamp, str):
            raise ColabTransportError("hosted status lacks receipt timestamp")
        try:
            from datetime import datetime

            parsed = datetime.fromisoformat(stamp)
            if parsed.tzinfo is None:
                raise ValueError("receipt timestamp must include a timezone")
            receipt_time = parsed.timestamp()
        except (ValueError, OSError, OverflowError) as error:
            raise ColabTransportError("invalid hosted receipt timestamp") from error
        age = float(value["observed_at"]) - receipt_time
        # Terminal receipts are immutable and may precede a long final upload.
        # The fresh status observation still bounds their transport freshness.
        if age < -30 or (not terminal and age > 30):
            raise ColabTransportError(
                "hosted receipt heartbeat is stale or clock-skewed"
            )
    return value


def require_colab_idle(
    worker: WorkerDefinition, destination: Path, *, timeout: float = 30
) -> None:
    """Refuse an exec unless a fresh observer proves this kernel is IDLE."""
    status = download_kernel_status(worker, destination, timeout=timeout)
    if status["state"] != "IDLE":
        raise ColabTransportError("Colab kernel occupancy is unknown or busy")


def upload_cancel_intent(
    worker: WorkerDefinition,
    *,
    attempt_id: str,
    spec_digest: str,
    attempt_key: bytes,
    local_root: Path,
    timeout: float = 30,
) -> None:
    """Upload the one authenticated cancellation intent without a kernel RPC."""
    endpoint = _endpoint(worker)
    if len(spec_digest) != 64 or len(attempt_key) != 32:
        raise ValueError("invalid cancellation identity")
    unsigned = {
        "instance_id": endpoint.instance_id,
        "attempt_id": attempt_id,
        "spec_digest": spec_digest,
    }
    payload = unsigned | {
        "hmac_sha256": hmac.new(
            attempt_key, canonical_json(unsigned), hashlib.sha256
        ).hexdigest()
    }
    local = Path(local_root) / ".colab" / attempt_id / _CANCEL_NAME
    _atomic_json(local, payload)
    _file_transport(worker, timeout=timeout).upload_exact(
        local, str(Path(worker.root) / "attempts" / attempt_id / _CANCEL_NAME)
    )


def mark_delivery_observed(
    controller_root: Path, attempt_id: str, *, terminal: bool
) -> None:
    """Advance only a matching issued record from a validated passive status."""
    record = read_delivery(controller_root, attempt_id)
    if record is None:
        raise ColabTransportError("missing Colab launch-delivery record")
    if record["state"] == "OBSERVED_TERMINAL" and not terminal:
        return
    record["state"] = "OBSERVED_TERMINAL" if terminal else "OBSERVED_RUNNING"
    _write_delivery(controller_root, record)
