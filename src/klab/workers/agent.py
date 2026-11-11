"""Finite stdio agent endpoint and detached executor entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

from klab.canonical import canonical_json
from klab.config.models import RunConfig
from klab.training.metrics import ExperimentStore
from klab.verification_proofs import verification_options
from klab.workdir import ensure_work_dir

from .execution import (
    cancel_attempt,
    discover_worker,
    execute_attempt,
    initialize_worker,
    install_relay_key,
    launch_attempt,
    prepare_attempt,
    status_worker,
    validate_worker,
)
from .models import PROTOCOL_VERSION, WorkerDefinition, validate_operation
from .transport import ProtocolError, WorkerBusyError, read_frame, write_frame

_LOGGER = logging.getLogger(__name__)


def _definition(args: argparse.Namespace) -> WorkerDefinition:
    return WorkerDefinition(
        worker_id=args.worker_id,
        name=args.name,
        transport="local",
        host=None,
        python=args.python or sys.executable,
        root=args.root,
        engine=args.engine,
        backend=args.backend,
        device_index=args.device_index,
    )


def _error(error: Exception) -> dict[str, Any]:
    if isinstance(error, WorkerBusyError):
        code, retryable = "BUSY", True
    elif isinstance(error, ProtocolError):
        code, retryable = "PROTOCOL_ERROR", False
    elif isinstance(error, FileNotFoundError):
        code, retryable = "NOT_FOUND", False
    elif isinstance(error, PermissionError):
        code, retryable = "FORBIDDEN", False
    elif isinstance(error, ValueError):
        code, retryable = "INVALID_REQUEST", False
    else:
        code, retryable = "WORKER_ERROR", True
    return {
        "code": code,
        "message": str(error)[:1024],
        "retryable": retryable,
        "details": None,
    }


def _store_origin(root: Path) -> str:
    return str(ExperimentStore(root / "runs").export_records(0, 1, 65_536)["origin_id"])


def _dispatch(
    definition: WorkerDefinition,
    op: str,
    payload: dict[str, Any],
    attachments: dict[str, Path],
    temporary: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    payload = validate_operation(op, payload)
    if op == "attention_probe":
        from klab.runtime_attention import attention_probe
        from klab.runtime_profile import authorize_worker

        from .execution import (
            _acquire_config_lease,
            _effective_config,
            _runtime_for_definition,
        )

        config = _effective_config(
            definition, RunConfig.model_validate(payload["config"])
        )
        authorization = authorize_worker(definition, config)
        runtime = _runtime_for_definition(definition)
        lease = _acquire_config_lease(definition, runtime, config)
        if lease is None:
            raise WorkerBusyError("physical device is leased")
        try:
            return attention_probe(config, authorization=authorization), {}
        finally:
            lease.close()
    if op == "configure_relay":
        from .relay_execution import configure_relay

        return configure_relay(definition, payload), {}
    if op == "relay_flush":
        from .relay_execution import flush_terminal

        return flush_terminal(definition, payload["attempt_id"]), {}
    if op == "discover":
        return discover_worker(definition), {}
    if op == "validate":
        report = validate_worker(
            definition,
            RunConfig.model_validate(payload["config"]),
            payload["bundle_digest"],
        )
        if report.get("code") == "BUSY":
            raise WorkerBusyError("physical device is leased")
        return {
            "runtime": report["runtime"],
            "validation_status": "passed" if report["ok"] else "failed",
            "reason": None if report["ok"] else report["message"],
            "runtime_authorization": report.get("runtime_authorization"),
        }, {}
    if op == "install_bundle":
        from .bundles import _read_bundle_manifest, install_dispatch_bundle

        manifest_path = attachments.get("bundle.json")
        if manifest_path is None:
            raise ValueError("install_bundle requires bundle.json attachment")
        manifest = _read_bundle_manifest(manifest_path)
        if manifest.digest() != payload["manifest_digest"]:
            raise ValueError("bundle manifest digest mismatch")
        return install_dispatch_bundle(
            definition.root,
            manifest,
            attachments,
            check_only=payload["mode"] == "check",
            **verification_options(
                definition.root, cold=payload.get("cold_verify", True)
            ),
        ), {}
    if op in {"launch", "prepare"}:
        spec_path = attachments.get("spec.json")
        if spec_path is None:
            raise ValueError(f"{op} requires spec.json attachment")
        key_path = attachments.get("relay-key.bin")
        if key_path is not None:
            install_relay_key(definition, payload["attempt_id"], key_path)
        operation = prepare_attempt if op == "prepare" else launch_attempt
        return operation(definition, payload, spec_path), {}
    if op == "status":
        status = status_worker(definition, payload["attempt_id"])
        return {
            "capabilities": status["capabilities"],
            "receipt": status["receipt"],
            "receipts": status["receipts"],
            "origin_id": _store_origin(definition.root),
        }, {}
    if op == "cancel":
        receipt = cancel_attempt(definition, payload["attempt_id"])
        return {
            "receipt": receipt,
            "acknowledged": bool(receipt and receipt.get("cancellation_acknowledged")),
        }, {}
    if op == "records":
        store = ExperimentStore(definition.root / "runs")
        exported = store.export_records(
            payload["after_sequence"], payload["limit"], payload["max_bytes"]
        )
        if exported["origin_id"] != payload["origin_id"]:
            raise ValueError("records origin does not match worker store")
        output = temporary / "records.json"
        output.write_bytes(canonical_json(exported["records"]))
        return {
            key: exported[key] for key in ("origin_id", "next_sequence", "has_more")
        } | {"count": len(exported["records"])}, {"records.json": output}
    if op == "artifact":
        from .artifacts import read_artifact_chunk
        from .models import AttemptReceipt

        receipt_payload = status_worker(definition, payload["attempt_id"])["receipt"]
        if receipt_payload is None:
            raise FileNotFoundError("attempt does not exist")
        receipt = AttemptReceipt.model_validate(receipt_payload)
        result, content = read_artifact_chunk(
            definition.root,
            receipt,
            payload["relative_path"],
            payload["offset"],
            payload["max_bytes"],
            temporary / "artifact.bin",
        )
        return result, {"content": content}
    raise ValueError("unsupported operation")


def serve_stdio(definition: WorkerDefinition) -> int:
    """Handle exactly one binary frame; stdout contains only the response frame."""
    root = Path(definition.root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or root.stat().st_mode & 0o077:
        raise PermissionError("worker root must be a private real directory")
    with tempfile.TemporaryDirectory(prefix="protocol-", dir=root) as directory:
        temporary = Path(directory)
        request_id = "invalid"
        try:
            frame = read_frame(
                sys.stdin.buffer, response=False, destination=temporary / "incoming"
            )
            request_id = frame.header["request_id"]
            result, outgoing = _dispatch(
                definition,
                frame.header["op"],
                frame.header["payload"],
                frame.attachments,
                temporary,
            )
            header = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": True,
                "result": result,
                "error": None,
                "attachments": [],
            }
            write_frame(
                sys.stdout.buffer, header, outgoing, expected_op=frame.header["op"]
            )
        except Exception as error:
            _LOGGER.exception("Worker request failed")
            header = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": False,
                "result": None,
                "error": _error(error),
                "attachments": [],
            }
            try:
                write_frame(sys.stdout.buffer, header, {})
            except OSError, ValueError, TypeError:
                return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m klab.workers.agent")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", required=True, type=Path)
    common.add_argument("--worker-id", required=True)
    common.add_argument("--name", required=True)
    common.add_argument("--engine", required=True, choices=("pytorch", "mlx"))
    common.add_argument(
        "--backend",
        required=True,
        choices=("cpu", "mps", "cuda", "rocm", "xpu", "metal"),
    )
    common.add_argument("--device-index", required=True, type=int)
    common.add_argument("--python")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve-stdio", parents=[common])
    execute = sub.add_parser("execute", parents=[common])
    execute.add_argument("--attempt-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_work_dir()
    definition = _definition(args)
    initialize_worker(definition)
    if args.command == "serve-stdio":
        return serve_stdio(definition)
    try:
        execute_attempt(definition, args.attempt_id)
    except Exception:
        # Detached executor diagnostics belong only to its stderr log.
        _LOGGER.exception("Detached worker execution failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
