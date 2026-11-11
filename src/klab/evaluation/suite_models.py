"""Data-only evaluation declarations, shared by metadata and runtime readers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from klab.config.models import StrictModel
from klab.declaration_context import safe_path

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


class SuiteEvaluation(StrictModel):
    id: str
    role: Literal["gate", "diagnostic", "descriptive", "exploratory", "blinded_surface"]
    kind: Literal[
        "heldout_lm", "capability_card", "evidence_reference", "surface_review"
    ]
    source: str | None = None

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("evaluation ID must be safe and nonempty")
        return value

    @model_validator(mode="after")
    def valid_pair(self) -> SuiteEvaluation:
        if (self.role == "blinded_surface") != (self.kind == "surface_review"):
            raise ValueError("blinded_surface must pair only with surface_review")
        if self.role == "gate" and self.kind not in {"heldout_lm", "capability_card"}:
            raise ValueError("gate requires checkpoint-bound numeric evaluation")
        if (self.kind == "heldout_lm") != (self.source is None):
            raise ValueError("heldout_lm omits source; other kinds require source")
        if self.source is not None:
            safe_path(Path("."), self.source)
        return self


class EvaluationSuite(StrictModel):
    evaluation_suite_version: Literal[1]
    id: str
    evaluations: tuple[SuiteEvaluation, ...] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("suite ID must be safe and nonempty")
        return value

    @model_validator(mode="after")
    def unique_evaluations(self) -> EvaluationSuite:
        ids = [item.id for item in self.evaluations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate evaluation ID")
        return self
