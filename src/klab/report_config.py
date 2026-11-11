"""Strict read-only selectors for advanced result reports."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from klab.config.models import StrictModel


class ReportConfig(StrictModel):
    """Locations and diagnostic scope, with no execution or publication controls."""

    report_config_version: Literal[1]
    surface_dir: Path | None = None
    lifecycle: Path | None = None
    evidence_root: Path = Path(".")
    study: Path | None = None
    protocol_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("report_config_version", mode="before")
    @classmethod
    def exact_version(cls, value: int) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("report_config_version must be integer 1")
        return value

    @model_validator(mode="after")
    def validate_scope(self) -> ReportConfig:
        if self.study is not None and self.protocol_sha256 is not None:
            raise ValueError("select study or protocol_sha256, not both")
        return self


def load_report_config(path: Path) -> ReportConfig:
    """Read YAML without creating locations; resolve paths beside the declaration."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid report YAML: {error}") from error
    config = ReportConfig.model_validate(raw)
    base = path.absolute().parent
    return config.model_copy(
        update={
            field: base / value
            for field in ("surface_dir", "lifecycle", "evidence_root", "study")
            if field in config.model_fields_set
            and (value := getattr(config, field)) is not None
            and not value.is_absolute()
        }
    )
