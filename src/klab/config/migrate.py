"""Deterministic, validated v1-to-v2 run-configuration migration."""

from __future__ import annotations

import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from klab.config.loading import _PATH_KEYS, _resolve_paths
from klab.config.models import RunConfig


def migrate_v1(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a v1 run mapping without changing scientific settings."""
    if raw.get("schema_version") != 1:
        raise ValueError("only schema_version: 1 run configs can be migrated")
    result = deepcopy(raw)
    result["schema_version"] = 2
    device = result.pop("device", "auto")
    result["runtime"] = {
        "engine": "pytorch",
        "backend": device,
        "device_index": 0,
        "precision": "fp32",
        "memory": {
            "policy": "balanced",
            "max_device_memory_fraction": 0.90,
            "budget_bytes": None,
            "activation_checkpointing": {
                "enabled": False,
                "strategy": "transformer_block",
            },
            "activation_offload": {"enabled": False},
            "allowed_sequence_lengths": [],
            "allowed_optimizers": [],
        },
    }
    training = result["training"]
    training["micro_batch_size"] = training.pop("batch_size")
    training["gradient_accumulation"] = 1
    optimizer = result.get("optimizer", {})
    optimizer["name"] = "adamw"
    optimizer["peak"] = optimizer.pop("learning_rate", 3e-4)
    optimizer["floor"] = optimizer.pop("min_learning_rate", 3e-5)
    optimizer.setdefault("warmup_steps", 10)
    optimizer.setdefault("weight_decay", 0.1)
    optimizer.setdefault("betas", [0.9, 0.95])
    optimizer.setdefault("eps", 1e-8)
    optimizer["state_offload"] = False
    result["optimizer"] = optimizer
    logging = result["logging"]
    cadence = logging.pop("checkpoint_every_steps", 50)
    result["checkpoint"] = {
        "every_steps": cadence,
        "every_tokens": None,
        "every_minutes": None,
        "keep_periodic": True,
    }
    result["staging"] = {"smoke_steps": 2, "warmup_steps": 5}
    return result


def _rebase_paths(value: Any, output_base: Path, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _rebase_paths(item_value, output_base, item_key)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_rebase_paths(item, output_base) for item in value]
    # Rebase exactly the fields anchored by loading, including optional manifests.
    if key in _PATH_KEYS and isinstance(value, Path):
        return os.path.relpath(value, output_base)
    return value


def migrate_file(source: Path, output: Path) -> list[str]:
    """Migrate atomically, preserving every relative path's resolved target."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(
            f"cannot read config {source}: {error.strerror or error}"
        ) from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {source}: {error}") from error
    if not isinstance(raw, dict):
        raise TypeError("config must be a YAML mapping")

    migrated = migrate_v1(raw)
    output_base = output.parent.resolve()
    resolved = _resolve_paths(migrated, source.parent.resolve())
    published = _rebase_paths(resolved, output_base)
    try:
        RunConfig.model_validate(_resolve_paths(published, output_base))
    except ValidationError as error:
        raise ValueError(f"migrated config is invalid: {error}") from error

    encoded = yaml.safe_dump(published, sort_keys=False, allow_unicode=True).encode(
        "utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite {output}") from error
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return [
        "schema_version: 1 -> 2",
        "device -> runtime.backend",
        "training.batch_size -> training.micro_batch_size",
        "logging.checkpoint_every_steps -> checkpoint.every_steps",
    ]
