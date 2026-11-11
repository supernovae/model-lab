"""Strict finite stdio transport shared by local and SSH workers."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import secrets
import shlex
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from klab.canonical import canonical_json
from klab.workdir import ensure_scratch_dir, ensure_work_dir
from klab.workers.models import (
    PROTOCOL_VERSION,
    REQUEST_OPERATIONS,
    AttachmentHeader,
    WorkerDefinition,
    validate_operation,
    validate_operation_attachments,
    validate_operation_result,
    validate_relative_path,
)

MAX_HEADER_BYTES = 1024 * 1024
MAX_ATTACHMENTS = 4096
MAX_ATTACHMENT_BYTES = 256 * 1024**3
MAX_TOTAL_ATTACHMENT_BYTES = 1024**4
MAX_CHUNK_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 32 * 1024 * 1024
_LOGGER = logging.getLogger(__name__)


class ProtocolError(ValueError):
    """Malformed, unsafe, or unsupported wire data."""


class WorkerBusyError(ProtocolError):
    """A local busy lease condition mapped to a retryable BUSY wire error."""


class RemoteProtocolError(ProtocolError):
    """A typed remote failure that intentionally excludes remote tracebacks."""

    def __init__(
        self,
        code: str,
        message: str,
        retryable: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details


@dataclass(frozen=True)
class Frame:
    header: dict[str, Any]
    attachments: dict[str, Path]


@dataclass(frozen=True)
class ProtocolReply:
    result: dict[str, Any]
    attachments: dict[str, Path]
    request_id: str


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicates)
        canonical_json(value)  # rejects NaN/Infinity and non-JSON-native values
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        if isinstance(error, ProtocolError):
            raise
        raise ProtocolError("invalid canonical JSON header") from error
    if not isinstance(value, dict):
        raise ProtocolError("frame header must be a JSON object")
    return value


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    pieces = bytearray()
    while len(pieces) < count:
        chunk = stream.read(min(MAX_CHUNK_BYTES, count - len(pieces)))
        if not chunk:
            raise ProtocolError("truncated frame")
        pieces.extend(chunk)
    return bytes(pieces)


def _safe_destination(root: Path, name: str) -> Path:
    validate_relative_path(name)
    current = root
    for part in Path(name).parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ProtocolError("attachment destination contains symlink")
        current.mkdir(exist_ok=True)
    target = root / name
    if target.exists() and target.is_symlink():
        raise ProtocolError("attachment destination is symlink")
    return target


def _validate_attachments(raw: object) -> list[AttachmentHeader]:
    if not isinstance(raw, list) or len(raw) > MAX_ATTACHMENTS:
        raise ProtocolError("invalid attachment list")
    try:
        attachments = [AttachmentHeader.model_validate(item) for item in raw]
    except Exception as error:
        raise ProtocolError("invalid attachment header") from error
    names = [item.name for item in attachments]
    if len(names) != len(set(names)):
        raise ProtocolError("duplicate attachment name")
    total = sum(item.length for item in attachments)
    if (
        any(item.length > MAX_ATTACHMENT_BYTES for item in attachments)
        or total > MAX_TOTAL_ATTACHMENT_BYTES
    ):
        raise ProtocolError("attachment limits exceeded")
    return attachments


def _validate_header(
    header: dict[str, Any], *, response: bool, expected_op: str | None = None
) -> list[AttachmentHeader]:
    expected = {"protocol_version", "request_id", "attachments"}
    if response:
        expected |= {"ok", "result", "error"}
    else:
        expected |= {"op", "payload"}
    if set(header) != expected:
        raise ProtocolError("invalid frame header fields")
    if (
        type(header["protocol_version"]) is not int
        or header["protocol_version"] != PROTOCOL_VERSION
    ):
        raise ProtocolError("unsupported protocol version")
    request_id = header["request_id"]
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise ProtocolError("invalid request ID")
    attachments = _validate_attachments(header["attachments"])
    try:
        if response:
            if expected_op is None:
                if attachments:
                    raise ValueError("response attachment context is required")
            elif expected_op not in REQUEST_OPERATIONS:
                raise ValueError("unsupported response operation context")
            if (
                type(header["ok"]) is not bool
                or (
                    header["ok"]
                    and (
                        not isinstance(header["result"], dict)
                        or header["error"] is not None
                    )
                )
                or (
                    not header["ok"]
                    and (
                        header["result"] is not None
                        or not isinstance(header["error"], dict)
                    )
                )
            ):
                raise ValueError("invalid response success/error shape")
            if not header["ok"]:
                if attachments:
                    raise ValueError("error responses cannot include attachments")
                error = header["error"]
                if (
                    set(error) != {"code", "message", "retryable", "details"}
                    or not isinstance(error["code"], str)
                    or not isinstance(error["message"], str)
                    or type(error["retryable"]) is not bool
                    or (
                        error["details"] is not None
                        and not isinstance(error["details"], dict)
                    )
                ):
                    raise ValueError("invalid remote error")
            elif expected_op is not None:
                validate_operation_result(expected_op, header["result"])
                validate_operation_attachments(
                    expected_op, None, attachments, response=True
                )
        else:
            op = header["op"]
            payload = validate_operation(op, header["payload"])
            validate_operation_attachments(op, payload, attachments)
    except (TypeError, ValueError) as error:
        raise ProtocolError(
            "invalid request operation, result, or attachments"
        ) from error
    return attachments


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def write_frame(
    stream: BinaryIO,
    header: Mapping[str, Any],
    attachments: Mapping[str, Path],
    *,
    expected_op: str | None = None,
    deadline: float | None = None,
) -> None:
    """Write a validated canonical frame without loading attachment contents into RAM."""

    def check_deadline() -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("worker request hashing or upload exceeded its deadline")

    check_deadline()
    frame = dict(header)
    supplied = dict(attachments)
    raw_attachments: list[dict[str, Any]] = []
    total = 0
    if len(supplied) > MAX_ATTACHMENTS:
        raise ProtocolError("too many attachments")
    for name, path in supplied.items():
        validate_relative_path(name)
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ProtocolError("attachment source must be a regular non-symlink file")
        size = source.stat().st_size
        if size > MAX_ATTACHMENT_BYTES:
            raise ProtocolError("attachment exceeds per-file limit")
        total += size
        if total > MAX_TOTAL_ATTACHMENT_BYTES:
            raise ProtocolError("attachment total exceeds protocol limit")
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(MAX_CHUNK_BYTES), b""):
                check_deadline()
                digest.update(chunk)
        raw_attachments.append(
            {"name": name, "length": size, "sha256": digest.hexdigest()}
        )
    frame["attachments"] = raw_attachments
    planned = _validate_header(frame, response="ok" in frame, expected_op=expected_op)
    if {item.name for item in planned} != set(supplied):
        raise ProtocolError("attachment header/source mismatch")
    raw_header = canonical_json(frame)
    if len(raw_header) > MAX_HEADER_BYTES:
        raise ProtocolError("header exceeds protocol limit")
    stream.write(struct.pack(">I", len(raw_header)))
    stream.write(raw_header)
    for item in planned:
        source = supplied[item.name]
        stream.write(struct.pack(">Q", item.length))
        digest = hashlib.sha256()
        remaining = item.length
        with source.open("rb") as handle:
            while remaining:
                check_deadline()
                chunk = handle.read(min(MAX_CHUNK_BYTES, remaining))
                if not chunk:
                    raise ProtocolError("attachment changed while transmitting")
                digest.update(chunk)
                stream.write(chunk)
                remaining -= len(chunk)
        if digest.hexdigest() != item.sha256:
            raise ProtocolError("attachment changed while transmitting")
    stream.flush()


def read_frame(
    stream: BinaryIO,
    *,
    response: bool,
    destination: Path,
    expected_op: str | None = None,
) -> Frame:
    """Read and verify one frame, publishing attachments only after complete success."""
    raw_length = _read_exact(stream, 4)
    header_length = struct.unpack(">I", raw_length)[0]
    if header_length > MAX_HEADER_BYTES:
        raise ProtocolError("header exceeds protocol limit")
    header = _strict_json(_read_exact(stream, header_length))
    attachments = _validate_header(
        header, response=response, expected_op=expected_op
    )  # before reading body
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ProtocolError("attachment destination cannot be symlink")
    temporary = destination / f".protocol-{secrets.token_hex(16)}.tmp"
    temporary.mkdir(mode=0o700)
    published: dict[str, Path] = {}
    if _free_bytes(destination) < sum(item.length for item in attachments):
        shutil.rmtree(temporary, ignore_errors=True)
        raise ProtocolError("insufficient destination disk space")
    # Detect conflicts and symlink components before any verified payload is published.
    for item in attachments:
        if _safe_destination(destination, item.name).exists():
            shutil.rmtree(temporary, ignore_errors=True)
            raise ProtocolError("refusing to overwrite attachment destination")
    try:
        for item in attachments:
            length = struct.unpack(">Q", _read_exact(stream, 8))[0]
            if length != item.length:
                raise ProtocolError("attachment length does not match header")
            if _free_bytes(destination) < length:
                raise ProtocolError("insufficient destination disk space")
            target = _safe_destination(temporary, item.name)
            digest = hashlib.sha256()
            remaining = length
            with target.open("xb") as handle:
                while remaining:
                    chunk = _read_exact(stream, min(MAX_CHUNK_BYTES, remaining))
                    digest.update(chunk)
                    handle.write(chunk)
                    remaining -= len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if digest.hexdigest() != item.sha256:
                raise ProtocolError("attachment digest mismatch")
        # Each final path is written only after all source attachments verify.
        for item in attachments:
            source = temporary / item.name
            target = _safe_destination(destination, item.name)
            if target.exists():
                raise ProtocolError("refusing to overwrite attachment destination")
            os.replace(source, target)
            published[item.name] = target
        return Frame(header=header, attachments=published)
    except Exception:
        for target in published.values():
            target.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _endpoint_argv(worker: WorkerDefinition) -> list[str]:
    fixed = [
        str(worker.python),
        "-m",
        "klab.workers.agent",
        "serve-stdio",
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
    if worker.transport == "local":
        return fixed
    if worker.host is None:
        raise ProtocolError("SSH worker missing host")
    remote_command = " ".join(shlex.quote(part) for part in fixed)
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        worker.host,
        remote_command,
    ]


def call_worker(
    worker: WorkerDefinition,
    op: str,
    payload: dict[str, Any],
    *,
    attachments: Mapping[str, Path] | None = None,
    receive_dir: Path | None = None,
    timeout: float = 30,
) -> ProtocolReply:
    """Invoke one finite endpoint under one deadline, including upload hashing."""
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if worker.transport == "colab":
        from .colab import call_colab

        return call_colab(
            worker,
            op,
            payload,
            attachments=attachments,
            receive_dir=receive_dir,
            timeout=timeout,
        )
    ensure_work_dir()
    deadline = time.monotonic() + float(timeout)
    validate_operation(op, payload)
    if receive_dir is None and op in {"records", "artifact"}:
        raise ValueError("attachment responses require an explicit receive_dir")
    owns_receive_dir = receive_dir is None
    receive_dir = (
        Path(tempfile.mkdtemp(prefix="klab-worker-reply-", dir=ensure_scratch_dir()))
        if receive_dir is None
        else Path(receive_dir)
    )
    receive_dir.mkdir(parents=True, exist_ok=True)
    request_id = secrets.token_hex(16)
    header = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "op": op,
        "payload": payload,
    }
    try:
        process = subprocess.Popen(
            _endpoint_argv(worker),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        if owns_receive_dir:
            shutil.rmtree(receive_dir, ignore_errors=True)
        raise
    assert process.stdin is not None and process.stdout is not None
    frame: Frame | None = None
    read_error: BaseException | None = None
    write_error: BaseException | None = None

    def receive() -> None:
        nonlocal frame, read_error
        try:
            frame = read_frame(
                process.stdout,
                response=True,
                destination=receive_dir / request_id,
                expected_op=op,
            )
        except BaseException as error:
            _LOGGER.exception("Worker RPC response reader failed")
            read_error = error
        finally:
            process.stdout.close()

    def send() -> None:
        nonlocal write_error
        try:
            write_frame(process.stdin, header, attachments or {}, deadline=deadline)
        except BaseException as error:
            _LOGGER.exception("Worker RPC request writer failed")
            write_error = error
        finally:
            try:
                process.stdin.close()
            except OSError as error:
                if write_error is None:
                    write_error = error

    reader = threading.Thread(target=receive, daemon=True)
    writer = threading.Thread(target=send, daemon=True)
    reader.start()
    writer.start()

    def remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    try:
        writer.join(remaining())
        if writer.is_alive():
            raise TimeoutError("worker transport timed out while uploading request")
        if write_error is not None:
            raise write_error
        reader.join(remaining())
        if reader.is_alive():
            raise TimeoutError("worker transport timed out awaiting response")
        if read_error is not None:
            raise read_error
        if frame is None:
            raise ProtocolError("worker returned no response frame")
        if frame.header["request_id"] != request_id:
            raise ProtocolError("response request ID mismatch")
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
        if process.poll() is None:
            process.kill()
        writer.join(remaining())
        reader.join(remaining())
        try:
            process.wait(timeout=remaining())
        except subprocess.TimeoutExpired:
            # SIGKILL has already been sent. Popen's nonblocking child reaper
            # owns the remaining wait; cleanup must not extend the caller's deadline.
            _LOGGER.warning("Worker endpoint reaping deferred after RPC deadline")
        if owns_receive_dir:
            shutil.rmtree(receive_dir, ignore_errors=True)
