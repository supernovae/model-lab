"""VM-local operational snapshots and authenticated Colab cancellation intent."""

from __future__ import annotations

import hashlib
import hmac
import time
from pathlib import Path
from typing import Any

from klab.canonical import canonical_json


def instance_identity(root: Path) -> str | None:
    from .execution import _strict_json
    from .leases import boot_identity

    path = root / "hosted-instance.json"
    if not path.exists():
        return None
    if path.is_symlink():
        raise ValueError("symlinked hosted instance identity")
    value = _strict_json(path)
    if set(value) != {"instance_id", "boot_id"} or value["boot_id"] != boot_identity():
        raise ValueError("hosted instance identity differs from VM boot")
    from .models import _identifier

    return _identifier(value["instance_id"], "instance_id")


def write_hosted_status(definition: Any, receipt: dict[str, Any] | None) -> None:
    from klab.training.metrics import ExperimentStore

    from .execution import _atomic_json, _caps, _definition_value, _strict_json

    root = Path(_definition_value(definition, "root"))
    instance = instance_identity(root)
    if instance is None:
        return
    capability = _caps(definition)
    capability["status"] = (
        "busy" if receipt and receipt["state"] == "RUNNING" else "idle"
    )
    origin = ExperimentStore(root / "runs").export_records(0, 1, 65_536)["origin_id"]
    newest = None
    if receipt:
        commit_path = root / "attempts" / receipt["attempt_id"] / "relay-commit.json"
        if commit_path.exists():
            if commit_path.is_symlink():
                raise ValueError("symlinked relay commit observation")
            newest = _strict_json(commit_path)
    _atomic_json(
        root / "hosted-status.json",
        {
            "hosted_status_version": 1,
            "instance_id": instance,
            "observed_at": time.time(),
            "capability": capability,
            "receipt": receipt,
            "origin_id": origin,
            "newest_commit": newest,
        },
    )


def process_cancel_intent(definition: Any, attempt_id: str) -> None:
    from .execution import (
        _atomic_json,
        _attempt_dir,
        _definition_value,
        _load_receipt,
        _strict_json,
        cancel_attempt,
    )

    root = Path(_definition_value(definition, "root"))
    instance = instance_identity(root)
    if instance is None:
        return
    directory = _attempt_dir(definition, attempt_id)
    intent = directory / "colab-cancel.json"
    if not intent.exists():
        return
    key_path = directory / "relay-key.bin"
    if key_path.is_symlink() or key_path.stat().st_mode & 0o077:
        raise PermissionError("relay key must be private")
    key = key_path.read_bytes()
    if len(key) != 32:
        raise ValueError("relay key must contain exactly 32 bytes")
    try:
        if intent.is_symlink() or intent.stat().st_size > 4096:
            raise ValueError("invalid Colab cancellation file")
        value = _strict_json(intent)
        if set(value) != {"instance_id", "attempt_id", "spec_digest", "hmac_sha256"}:
            raise ValueError("invalid Colab cancellation fields")
        receipt = _load_receipt(definition, attempt_id)
        if (value["instance_id"], value["attempt_id"], value["spec_digest"]) != (
            instance,
            attempt_id,
            receipt["spec_digest"],
        ):
            raise ValueError("Colab cancellation identity mismatch")
        signature = value.pop("hmac_sha256")
        expected = hmac.new(key, canonical_json(value), hashlib.sha256).hexdigest()
        if not isinstance(signature, str) or not hmac.compare_digest(
            signature, expected
        ):
            raise ValueError("Colab cancellation authentication failed")
    except (OSError, ValueError, TypeError) as error:
        _atomic_json(
            directory / "colab-cancel-rejected.json",
            {
                "code": "INVALID_CANCEL_INTENT",
                "message": str(error)[:1024],
                "observed_at": time.time(),
            },
        )
        return
    cancel_attempt(definition, attempt_id)
