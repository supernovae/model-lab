"""Fixed relay materializer; sensitive prepare/cancel messages are never bulk."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

from klab.canonical import canonical_json
from klab.workers.agent import _dispatch, _error
from klab.workers.models import (
    PROTOCOL_VERSION,
    AttachmentHeader,
    WorkerDefinition,
    validate_operation,
    validate_operation_attachments,
)
from klab.workers.relay import RelayStore
from klab.workers.relay_execution import worker_binding
from klab.workers.relay_models import RelayArtifactIdentity
from klab.workers.transport import (
    MAX_ATTACHMENTS,
    MAX_HEADER_BYTES,
    MAX_TOTAL_ATTACHMENT_BYTES,
    write_frame,
)


def _json(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > MAX_HEADER_BYTES
    ):
        raise ValueError("unsafe or oversized Colab bulk descriptor")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise ValueError("Colab bulk descriptor is not canonical JSON")
    return value


def serve(
    definition: WorkerDefinition,
    descriptor_path: Path,
    response_path: Path,
    *,
    timeout_seconds: float = 60,
) -> None:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("bulk timeout must be finite and positive")
    deadline = time.monotonic() + timeout_seconds
    descriptor = _json(descriptor_path)
    request_id = descriptor.get("request_id")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise ValueError("invalid Colab bulk request ID")
    if response_path.exists() or response_path.is_symlink():
        raise ValueError("Colab bulk response path must be fresh")
    created_response = False
    response_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        if (
            set(descriptor) != {"request_id", "op", "payload", "attachments"}
            or descriptor["op"] != "install_bundle"
        ):
            raise ValueError("Colab bulk transport supports install_bundle only")
        op, payload, listed = (
            descriptor["op"],
            descriptor["payload"],
            descriptor["attachments"],
        )
        if not isinstance(listed, list) or len(listed) > MAX_ATTACHMENTS:
            raise ValueError("invalid Colab bulk attachment count")
        payload = validate_operation(op, payload)
        headers = [AttachmentHeader.model_validate(item) for item in listed]
        validate_operation_attachments(op, payload, headers)
        if (
            len({header.name for header in headers}) != len(headers)
            or sum(header.length for header in headers) > MAX_TOTAL_ATTACHMENT_BYTES
        ):
            raise ValueError("duplicate or oversized Colab bulk attachments")
        binding = worker_binding(Path(definition.root))
        if binding is None:
            raise ValueError("worker relay is not configured")
        scratch = Path(definition.root) / ".colab-control"
        store = RelayStore(
            binding.worker_binding(), role="worker", scratch_root=scratch
        )
        with tempfile.TemporaryDirectory(prefix="bulk-", dir=scratch) as name:
            temporary = Path(name)
            incoming = temporary / "incoming"
            incoming.mkdir(mode=0o700)
            attachments: dict[str, Path] = {}
            for item in headers:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Colab bulk transfer deadline exhausted")
                store.timeout_seconds = min(store.timeout_seconds, remaining)
                identity = RelayArtifactIdentity(
                    relative_path=item.name, sha256=item.sha256, size_bytes=item.length
                )
                destination = incoming / item.name
                store.get_verified(identity, destination)
                attachments[item.name] = destination
            result, outgoing = _dispatch(
                definition, op, payload, attachments, temporary
            )
            header = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": True,
                "result": result,
                "error": None,
                "attachments": [],
            }
            with response_path.open("xb") as stream:
                created_response = True
                write_frame(stream, header, outgoing, expected_op=op)
    except Exception as error:
        header = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "ok": False,
            "result": None,
            "error": _error(error),
            "attachments": [],
        }
        if created_response:
            response_path.unlink()
        elif response_path.exists() or response_path.is_symlink():
            raise
        with response_path.open("xb") as stream:
            write_frame(stream, header, {})
    response_path.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--definition", required=True, type=Path)
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    serve(
        WorkerDefinition.model_validate(_json(args.definition)),
        args.descriptor,
        args.response,
        timeout_seconds=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
