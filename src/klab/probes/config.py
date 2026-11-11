"""Validated probe policy with small named declarations for the common journeys."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from klab.config.models import StrictModel
from klab.probes.benchmark_protocol import BenchmarkProtocol


class ProbeConfig(StrictModel):
    probe_config_version: Literal[1] = 1
    tier: Literal["fast", "standard", "full"] = "fast"
    final: bool = Field(default=False, strict=True)
    fast_fail: bool = Field(default=True, strict=True)
    lm_eval_protocol: BenchmarkProtocol | None = None
    lm_eval_tasks: list[str] | None = None
    lm_eval_limit: int | None = Field(default=None, strict=True, ge=1)
    lab_dir: Path | None = None
    runs_dir: Path | None = None
    runtime_profile: Path | None = None
    resource_envelope: Path | None = None

    @field_validator("probe_config_version", mode="before")
    @classmethod
    def exact_version(cls, value: int) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("probe_config_version must be integer 1")
        return value

    @field_validator("lm_eval_tasks")
    @classmethod
    def task_names(cls, values: list[str] | None) -> list[str] | None:
        if values is not None and (
            not values
            or len(set(values)) != len(values)
            or any(not value or value.strip() != value for value in values)
        ):
            raise ValueError("lm_eval_tasks must contain distinct, nonempty task names")
        return values

    @model_validator(mode="after")
    def coherent_policy(self) -> ProbeConfig:
        if self.final and self.tier == "fast":
            raise ValueError(
                "final evaluation requires tier: standard or full; use --config final"
            )
        if self.tier != "full" and (
            self.lm_eval_tasks is not None
            or self.lm_eval_limit is not None
            or self.lm_eval_protocol is not None
        ):
            raise ValueError("lm_eval_tasks and lm_eval_limit require tier: full")
        if self.lm_eval_protocol is not None and (
            self.lm_eval_tasks is not None or self.lm_eval_limit is not None
        ):
            raise ValueError(
                "controlled protocol cannot be combined with smoke task/limit overrides"
            )
        return self


PRESETS = {
    "fast": {},
    "standard": {"tier": "standard"},
    "full": {"tier": "full"},
    "final": {"tier": "standard", "final": True},
    "final-full": {"tier": "full", "final": True},
}


def load_probe_config(selection: str) -> ProbeConfig:
    """Named declarations and YAML files use the same validated policy."""
    if selection in PRESETS:
        return ProbeConfig.model_validate(PRESETS[selection])
    path = Path(selection).expanduser()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        if "/" not in selection and "\\" not in selection and not path.suffix:
            raise ValueError(
                f"unknown probe config {selection!r}; choose {', '.join(PRESETS)} "
                "or a YAML path (use ./NAME for an extensionless file)"
            ) from error
        raise ValueError(
            f"cannot read probe config {path}: {error.strerror or error}"
        ) from error
    except OSError as error:
        raise ValueError(
            f"cannot read probe config {path}: {error.strerror or error}"
        ) from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in probe config {path}: {error}") from error
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"probe config {path} must contain a YAML mapping")  # noqa: TRY004 - serialized schema
    try:
        config = ProbeConfig.model_validate(raw)
    except ValidationError as error:
        raise ValueError(f"invalid probe config {path}: {error}") from error
    base = path.resolve().parent
    return config.model_copy(
        update={
            key: (base / value.expanduser()).resolve()
            for key in ("lab_dir", "runs_dir", "runtime_profile", "resource_envelope")
            if (value := getattr(config, key)) is not None
        }
    )


def prepare_probe_config(args: argparse.Namespace) -> None:
    """Freeze one validated request before resource or runtime preparation."""
    if getattr(args, "probe_config_loaded", None) is not None:
        return
    config = load_probe_config(args.config)
    if config.runtime_profile is not None and args.runtime is not None:
        raise ValueError(
            "--runtime and declaration runtime_profile are mutually exclusive"
        )
    args.probe_config_loaded = config
    for key in (
        "tier",
        "final",
        "lm_eval_tasks",
        "lm_eval_limit",
        "lm_eval_protocol",
        "lab_dir",
        "runs_dir",
        "runtime_profile",
        "resource_envelope",
    ):
        setattr(args, key, getattr(config, key))
    args.no_fast_fail = not config.fast_fail
