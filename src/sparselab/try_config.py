"""Validated try policy with small named declarations for the common journeys."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, field_validator

from sparselab.config.models import RunSeed, StrictModel


class TryContinuation(StrictModel):
    """Native terminal full-state budget extension, shared by every try arm."""

    checkpoint: Path
    append_data: bool = Field(default=False, strict=True)
    allow_runtime_optimization: bool = Field(default=False, strict=True)
    allow_runtime_drift: bool = Field(default=False, strict=True)


class TryConfig(StrictModel):
    try_config_version: Literal[1] = 1
    seed: RunSeed | None = Field(default=None, strict=True)
    probe_tier: Literal["none", "fast", "standard", "full"] = "fast"
    fresh_baseline: bool = Field(default=False, strict=True)
    repeat_baseline: bool = Field(default=False, strict=True)
    continuation: TryContinuation | None = None
    lab_dir: Path | None = None
    runtime_profile: Path | None = None
    resource_envelope: Path | None = None

    @field_validator("try_config_version", mode="before")
    @classmethod
    def exact_version(cls, value: int) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("try_config_version must be integer 1")
        return value


PRESETS = {
    "fast": {},
    "standard": {"probe_tier": "standard"},
    "full": {"probe_tier": "full"},
    "no-probes": {"probe_tier": "none"},
}


def load_try_config(selection: str) -> TryConfig:
    """Named declarations and YAML files use the same validated policy."""
    if selection in PRESETS:
        return TryConfig.model_validate(PRESETS[selection])
    path = Path(selection).expanduser()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        if "/" not in selection and "\\" not in selection and not path.suffix:
            raise ValueError(
                f"unknown try config {selection!r}; choose {', '.join(PRESETS)} "
                "or a YAML path (use ./NAME for an extensionless file)"
            ) from error
        raise ValueError(
            f"cannot read try config {path}: {error.strerror or error}"
        ) from error
    except OSError as error:
        raise ValueError(
            f"cannot read try config {path}: {error.strerror or error}"
        ) from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in try config {path}: {error}") from error
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"try config {path} must contain a YAML mapping")  # noqa: TRY004 - serialized schema
    try:
        config = TryConfig.model_validate(raw)
    except ValidationError as error:
        raise ValueError(f"invalid try config {path}: {error}") from error
    base = path.resolve().parent
    if config.continuation is not None:
        config = config.model_copy(
            update={
                "continuation": config.continuation.model_copy(
                    update={
                        "checkpoint": (
                            base / config.continuation.checkpoint.expanduser()
                        ).resolve()
                    }
                )
            }
        )
    return config.model_copy(
        update={
            key: (base / value.expanduser()).resolve()
            for key in ("lab_dir", "runtime_profile", "resource_envelope")
            if (value := getattr(config, key)) is not None
        }
    )


def prepare_try_config(args: argparse.Namespace) -> None:
    """Freeze one validated request before resource or runtime preparation."""
    if getattr(args, "try_config_loaded", None) is not None:
        return
    config = load_try_config(args.config)
    if config.runtime_profile is not None and args.runtime is not None:
        raise ValueError(
            "--runtime and declaration runtime_profile are mutually exclusive"
        )
    args.try_config_loaded = config
    for key in (
        "seed",
        "probe_tier",
        "fresh_baseline",
        "repeat_baseline",
        "continuation",
        "lab_dir",
        "runtime_profile",
        "resource_envelope",
    ):
        setattr(args, key, getattr(config, key))
