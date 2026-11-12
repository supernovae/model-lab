"""Declared corpus readiness thresholds over verified, unique source evidence."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Annotated, Any, Literal

from pydantic import Field, WithJsonSchema, field_validator, model_validator

from klab.config.models import StrictModel

NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
PositiveWeight = Annotated[
    Decimal,
    Field(gt=0, allow_inf_nan=False),
    WithJsonSchema({"type": "number", "exclusiveMinimum": 0}),
]


class SourcePassRequirement(StrictModel):
    """Maximum projected per-domain reuse of unique source material.

    Overlapping source documents across domains are counted once *per domain*;
    these bounds do not describe train order or actual ingestion passes.
    """

    basis: Literal["bytes", "tokens"]
    requested_total: PositiveInt
    mixture: dict[str, PositiveWeight]
    max_required: PositiveInt

    @field_validator("mixture", mode="before")
    @classmethod
    def require_numeric_weights(cls, values: Any) -> Any:
        if isinstance(values, dict) and any(
            type(weight) not in (float, int, Decimal) for weight in values.values()
        ):
            raise ValueError("source pass mixture weights must be numeric")
        return values

    @model_validator(mode="after")
    def validate_mixture(self) -> SourcePassRequirement:
        if not self.mixture or any(not domain.strip() for domain in self.mixture):
            raise ValueError("source passes require named mixture domains")
        total = sum(
            (Fraction(weight) for weight in self.mixture.values()),
            Fraction(0),
        )
        if total != 1:
            raise ValueError("source pass mixture weights must sum to 1")
        return self


class CorpusReadinessPolicy(StrictModel):
    """Authored minima, independent of the frozen release's mixture."""

    min_unique_train_bytes_by_domain: dict[str, NonnegativeInt] = Field(
        default_factory=dict
    )
    min_unique_train_tokens_by_domain: dict[str, NonnegativeInt] = Field(
        default_factory=dict
    )
    required_nonzero_languages: tuple[str, ...] = ()
    required_nonzero_shapes: tuple[str, ...] = ()
    min_heldout_families: NonnegativeInt = 0
    passes: SourcePassRequirement | None = None

    @model_validator(mode="after")
    def validate_requirements(self) -> CorpusReadinessPolicy:
        if not any(
            (
                self.min_unique_train_bytes_by_domain,
                self.min_unique_train_tokens_by_domain,
                self.required_nonzero_languages,
                self.required_nonzero_shapes,
                self.min_heldout_families,
                self.passes is not None,
            )
        ):
            raise ValueError("corpus readiness needs at least one requirement")
        for labels in (
            self.min_unique_train_bytes_by_domain,
            self.min_unique_train_tokens_by_domain,
            self.required_nonzero_languages,
            self.required_nonzero_shapes,
        ):
            if any(not label.strip() for label in labels):
                raise ValueError("readiness labels must not be empty")
        if len(set(self.required_nonzero_languages)) != len(
            self.required_nonzero_languages
        ):
            raise ValueError("duplicate required language")
        if len(set(self.required_nonzero_shapes)) != len(self.required_nonzero_shapes):
            raise ValueError("duplicate required shape")
        return self
