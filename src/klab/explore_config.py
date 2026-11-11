"""Validated descriptive explorer requests; runtime stays in native profiles."""

import argparse
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, field_validator

from klab.config.models import StrictModel
from klab.explorer import DEFAULT_TEXT


class ExploreConfig(StrictModel):
    explore_config_version: Literal[1] = 1
    text: str = Field(default=DEFAULT_TEXT, strict=True, min_length=1)
    refresh: bool = Field(default=False, strict=True)
    lab_dir: Path | None = None
    runs_dir: Path | None = None

    @field_validator("text")
    @classmethod
    def meaningful_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        # Whitespace affects tokenization; retain the declared prompt exactly.
        return value

    @field_validator("explore_config_version", mode="before")
    @classmethod
    def exact_version(cls, value: int) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("unsupported explorer config version")
        return value


def load_explore_config(path: Path) -> ExploreConfig:
    """Resolve lookup locations relative to the declaration, never to cwd."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(
            f"cannot read config {path}: {error.strerror or error}"
        ) from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {path}: {error}") from error
    if raw is None:
        raw = {}
    try:
        config = ExploreConfig.model_validate(raw)
    except ValidationError as error:
        raise ValueError(f"invalid config {path}: {error}") from error
    base = path.resolve().parent
    return config.model_copy(
        update={
            key: (base / value.expanduser()).resolve()
            for key in ("lab_dir", "runs_dir")
            if (value := getattr(config, key)) is not None
        }
    )


def prepare_explore_config(args: argparse.Namespace) -> None:
    """Validate before runtime preparation, storage checks or model loading."""
    config = load_explore_config(args.config) if args.config else ExploreConfig()
    for key in ("text", "refresh", "lab_dir", "runs_dir"):
        setattr(args, key, getattr(config, key))
