from __future__ import annotations

import hashlib
import io
import json
import struct
from pathlib import Path

import pytest

from klab.canonical import canonical_json
from klab.workers.models import (
    AttemptReceipt,
    BundleManifest,
    ExperimentSpec,
    WorkerCapabilities,
    WorkerDefinition,
    current_required_versions,
    validate_operation_result,
    validate_required_versions,
)
from klab.workers.transport import (
    MAX_HEADER_BYTES,
    ProtocolError,
    _endpoint_argv,
    read_frame,
    write_frame,
)


def _header(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "protocol_version": 1,
        "request_id": "golden",
        "op": "install_bundle",
        "payload": {"manifest_digest": "0" * 64, "mode": "install"},
        "attachments": [],
    }
    value.update(overrides)
    return value


def _encoded(header: dict[str, object], body: bytes = b"") -> io.BytesIO:
    raw = canonical_json(header)
    return io.BytesIO(struct.pack(">I", len(raw)) + raw + body)


def test_exact_golden_request_frame(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.write_bytes(b"payload")
    stream = io.BytesIO()
    write_frame(stream, _header(), {"bundle.json": payload})
    expected = bytes.fromhex(
        (Path(__file__).parent / "fixtures/contracts/worker_protocol_v1_frame.hex")
        .read_text()
        .strip()
    )
    assert stream.getvalue() == expected


def test_exact_golden_error_response_frame() -> None:
    header = {
        "protocol_version": 1,
        "request_id": "golden",
        "ok": False,
        "result": None,
        "error": {
            "code": "BAD_REQUEST",
            "message": "invalid",
            "retryable": False,
            "details": None,
        },
    }
    stream = io.BytesIO()
    write_frame(stream, header, {})
    expected = bytes.fromhex(
        (
            Path(__file__).parent
            / "fixtures/contracts/worker_protocol_v1_error_frame.hex"
        )
        .read_text()
        .strip()
    )
    assert stream.getvalue() == expected


def test_nested_artifact_size_rejects_boolean_before_coercion() -> None:
    path = Path(__file__).parent / "fixtures/contracts/attempt_receipt_v1.json"
    receipt = json.loads(path.read_text())
    receipt["artifacts"] = [
        {"relative_path": "run/data/train.npy", "sha256": "a" * 64, "size_bytes": True}
    ]
    with pytest.raises(ValueError):
        AttemptReceipt.model_validate(receipt)


def test_canonical_worker_public_contract_fixtures() -> None:
    fixtures = Path(__file__).parent / "fixtures/contracts"
    capabilities = WorkerCapabilities.model_validate_json(
        (fixtures / "worker_capabilities_v1.json").read_bytes()
    )
    receipt = AttemptReceipt.model_validate_json(
        (fixtures / "attempt_receipt_v1.json").read_bytes()
    )
    assert receipt.state == "COMPLETE"
    validate_required_versions(current_required_versions(), capabilities)
    incompatible = json.loads(json.dumps(current_required_versions()))
    incompatible["state_codecs"]["mlx_native"] = [3]
    with pytest.raises(ValueError):
        validate_required_versions(incompatible, capabilities)
    status = json.loads((fixtures / "worker_status_v1.json").read_bytes())
    assert validate_operation_result("status", status)["receipts"] == []
    assert (
        ExperimentSpec.model_validate_json(
            (fixtures / "experiment_spec_v1.json").read_bytes()
        ).matrix
        is not None
    )
    assert (
        BundleManifest.model_validate_json(
            (fixtures / "bundle_manifest_v1.json").read_bytes()
        )
        .files[0]
        .relative_path
        == "tokenizer.json"
    )


def test_exact_golden_status_response_frame() -> None:
    fixtures = Path(__file__).parent / "fixtures/contracts"
    header = {
        "protocol_version": 1,
        "request_id": "golden",
        "ok": True,
        "result": json.loads((fixtures / "worker_status_v1.json").read_bytes()),
        "error": None,
    }
    stream = io.BytesIO()
    write_frame(stream, header, {}, expected_op="status")
    assert stream.getvalue() == bytes.fromhex(
        (fixtures / "worker_protocol_v1_status_frame.hex").read_text().strip()
    )


def test_attachment_contracts_reject_metadata_and_unexpected_bodies_before_read(
    tmp_path: Path,
) -> None:
    digest = hashlib.sha256(b"").hexdigest()
    oversized = _header(
        attachments=[
            {"name": "bundle.json", "length": 32 * 1024 * 1024 + 1, "sha256": digest}
        ]
    )
    with pytest.raises(ProtocolError):
        read_frame(_encoded(oversized), response=False, destination=tmp_path)
    source = tmp_path / "body"
    source.write_bytes(b"")
    unexpected = _header(op="discover", payload={}, attachments=[])
    with pytest.raises(ProtocolError):
        write_frame(io.BytesIO(), unexpected, {"bundle.json": source})


def test_unknown_version_and_operation_reject_before_attachment_read(
    tmp_path: Path,
) -> None:
    header = _header(
        protocol_version=2,
        attachments=[{"name": "bundle.json", "length": 1, "sha256": "0" * 64}],
    )
    with pytest.raises(ProtocolError):
        read_frame(_encoded(header), response=False, destination=tmp_path)
    header = _header(
        op="unknown",
        attachments=[{"name": "bundle.json", "length": 1, "sha256": "0" * 64}],
    )
    with pytest.raises(ProtocolError):
        read_frame(_encoded(header), response=False, destination=tmp_path)


@pytest.mark.parametrize(
    "name",
    ["/absolute", "../escape", "dir/../../escape", "newline\nname", "bidi\u202ename"],
)
def test_attachment_paths_are_rejected(tmp_path: Path, name: str) -> None:
    header = _header(
        attachments=[
            {"name": name, "length": 0, "sha256": hashlib.sha256(b"").hexdigest()}
        ]
    )
    with pytest.raises(ProtocolError):
        read_frame(
            _encoded(header, struct.pack(">Q", 0)), response=False, destination=tmp_path
        )


def test_duplicate_names_bad_hash_and_truncation_leave_no_publication(
    tmp_path: Path,
) -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    duplicate = _header(
        attachments=[
            {"name": "bundle.json", "length": 1, "sha256": digest},
            {"name": "bundle.json", "length": 1, "sha256": digest},
        ]
    )
    with pytest.raises(ProtocolError):
        read_frame(_encoded(duplicate), response=False, destination=tmp_path)
    bad = _header(
        attachments=[{"name": "bundle.json", "length": 1, "sha256": "0" * 64}]
    )
    with pytest.raises(ProtocolError):
        read_frame(
            _encoded(bad, struct.pack(">Q", 1) + b"x"),
            response=False,
            destination=tmp_path,
        )
    truncated = _header(
        attachments=[{"name": "bundle.json", "length": 2, "sha256": digest}]
    )
    with pytest.raises(ProtocolError):
        read_frame(
            _encoded(truncated, struct.pack(">Q", 2) + b"x"),
            response=False,
            destination=tmp_path,
        )
    assert not (tmp_path / "bundle.json").exists()


def test_oversized_header_and_symlink_component_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ProtocolError):
        read_frame(
            io.BytesIO(struct.pack(">I", MAX_HEADER_BYTES + 1)),
            response=False,
            destination=tmp_path,
        )
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "assets").symlink_to(outside, target_is_directory=True)
    metadata = b"{}"
    asset_digest = hashlib.sha256(b"").hexdigest()
    header = _header(
        attachments=[
            {
                "name": "bundle.json",
                "length": len(metadata),
                "sha256": hashlib.sha256(metadata).hexdigest(),
            },
            {"name": f"assets/{asset_digest}", "length": 0, "sha256": asset_digest},
        ]
    )
    with pytest.raises(ProtocolError):
        read_frame(
            _encoded(
                header,
                struct.pack(">Q", len(metadata)) + metadata + struct.pack(">Q", 0),
            ),
            response=False,
            destination=tmp_path,
        )
    assert not (outside / asset_digest).exists()
    assert not (tmp_path / "bundle.json").exists()


def test_ssh_argv_is_fixed_and_option_injection_is_rejected() -> None:
    worker = WorkerDefinition(
        worker_id="worker-1",
        name="worker-1",
        transport="ssh",
        host="gpu.example",
        python=Path("/opt/venv/bin/python"),
        root=Path("/var/lib/klab"),
        engine="pytorch",
        backend="cuda",
        device_index=0,
    )
    argv = _endpoint_argv(worker)
    assert argv[:5] == ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes"]
    assert argv[5] == "gpu.example"
    assert (
        argv[6]
        == "/opt/venv/bin/python -m klab.workers.agent serve-stdio --root /var/lib/klab --worker-id worker-1 --name worker-1 --engine pytorch --backend cuda --device-index 0"
    )
    with pytest.raises(ValueError):
        WorkerDefinition(
            worker_id="worker",
            name="worker",
            transport="ssh",
            host="-oProxyCommand=x",
            python=Path("/bin/python"),
            root=Path("/tmp/worker"),
            engine="pytorch",
            backend="cpu",
            device_index=0,
        )


def test_error_attachments_are_rejected_before_body_consumption(tmp_path: Path) -> None:
    header = {
        "protocol_version": 1,
        "request_id": "error-with-body",
        "ok": False,
        "result": None,
        "error": {
            "code": "REJECTED",
            "message": "closed",
            "retryable": False,
            "details": None,
        },
        "attachments": [
            {"name": "chunk", "length": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
        ],
    }
    stream = _encoded(header, struct.pack(">Q", 1) + b"x")
    with pytest.raises(ProtocolError):
        read_frame(stream, response=True, expected_op="artifact", destination=tmp_path)
    assert stream.tell() == 4 + len(canonical_json(header))
    assert not (tmp_path / "chunk").exists()


@pytest.mark.parametrize(
    ("field", "value"), [("pid", "123"), ("cancellation_requested", "false")]
)
def test_wire_scalars_do_not_coerce_type(field: str, value: object) -> None:
    path = Path(__file__).parent / "fixtures/contracts/attempt_receipt_v1.json"
    payload = json.loads(path.read_bytes())
    payload[field] = value
    with pytest.raises(ValueError):
        AttemptReceipt.model_validate(payload)


@pytest.mark.parametrize("invalid", [0, 1, "true", None, []])
def test_operational_cold_mode_rejects_nonboolean_protocol_values(invalid) -> None:
    from klab.workers.models import validate_operation

    for op, payload in (
        ("install_bundle", {"manifest_digest": "0" * 64, "mode": "install"}),
        (
            "launch",
            {
                "attempt_id": "attempt",
                "run_id": "run",
                "experiment_id": "experiment",
                "spec_digest": "0" * 64,
                "bundle_digest": "1" * 64,
            },
        ),
    ):
        with pytest.raises(ValueError, match="cold_verify must be a boolean"):
            validate_operation(op, {**payload, "cold_verify": invalid})
