"""Data-only experiment declarations. A declaration is not an executable lock."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from klab.config.models import RunConfig, StrictModel
from klab.declaration_context import RecipeContext
from klab.documents import read_document

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_HEX = re.compile(r"^[0-9a-f]{64}$")


class Choice(StrictModel):
    label: str
    set: dict[str, Any]

    @field_validator("label")
    @classmethod
    def valid_label(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("choice label must be a safe nonempty identifier")
        return value


class Axis(StrictModel):
    name: str
    choices: tuple[Choice, ...]

    @model_validator(mode="after")
    def valid_axis(self) -> Axis:
        if not _ID.fullmatch(self.name) or not self.choices:
            raise ValueError("axis requires a safe name and at least one choice")
        labels = [choice.label for choice in self.choices]
        if len(labels) != len(set(labels)):
            raise ValueError(f"axis {self.name} has duplicate choice labels")
        return self


class Artifact(StrictModel):
    """An external immutable identity, or a named future producer (not both)."""

    kind: Literal[
        "source_snapshot",
        "dataset_snapshot",
        "corpus_build",
        "corpus_release",
        "corpus_export",
        "tokenizer",
        "prepared_data",
        "stage_bundle",
        "checkpoint",
        "capability_card",
        "prompt_set",
        "outcome",
    ]
    version: int = Field(ge=1)
    producer: str
    identifier: str
    sha256: str | None = None
    path: str | None = None
    from_phase: str | None = None
    selector: Literal["terminal", "best_validation"] | None = None
    state: Literal["full", "weights"] | None = None

    @model_validator(mode="after")
    def valid_reference(self) -> Artifact:
        if not _ID.fullmatch(self.identifier) or not self.producer:
            raise ValueError("artifact requires safe identifier and producer")
        if self.from_phase is None:
            if (
                self.path is None
                or self.sha256 is None
                or not _HEX.fullmatch(self.sha256)
            ):
                raise ValueError(
                    f"external {self.kind} artifact requires path and full SHA-256"
                )
            if self.selector is not None:
                raise ValueError("external artifact cannot select a future generation")
        elif self.path is not None or self.sha256 is not None:
            raise ValueError("planned output cannot claim an external path or digest")
        elif self.kind != "checkpoint" or self.selector is None:
            raise ValueError("planned output requires a checkpoint generation selector")
        return self


class Comparison(StrictModel):
    id: str
    baseline: dict[str, str]
    variant: dict[str, str]
    interventions: tuple[str, ...]
    invariants: tuple[str, ...] = ()
    mode: Literal["controlled", "multi_factor", "descriptive", "none"] = "controlled"
    confounders: tuple[str, ...] = ()
    phases: tuple[str, ...] = Field(default=(), exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def valid_comparison(self) -> Comparison:
        if len(self.phases) != len(set(self.phases)) or any(
            not _ID.fullmatch(phase) for phase in self.phases
        ):
            raise ValueError("comparison phases must be unique safe identifiers")
        if not _ID.fullmatch(self.id) or self.baseline == self.variant:
            raise ValueError("comparison requires a safe ID and distinct selectors")
        if set(self.baseline) != set(self.variant):
            raise ValueError(f"comparison {self.id} selectors must use the same axes")
        if self.mode == "multi_factor" and not self.confounders:
            raise ValueError(f"comparison {self.id} multi_factor requires confounders")
        if self.mode == "none" and self.interventions:
            raise ValueError(f"comparison {self.id} none cannot declare interventions")
        for label, fields in (
            ("interventions", self.interventions),
            ("invariants", self.invariants),
        ):
            if len(fields) != len(set(fields)) or any(
                not field or field.startswith(".") or field.endswith(".")
                for field in fields
            ):
                raise ValueError(
                    f"comparison {self.id} {label} must be unique dotted paths"
                )
        return self


class Phase(StrictModel):
    id: str
    transition: Literal["fresh", "resume", "extend_budget", "promote"] = "fresh"
    parent: str | None = None
    checkpoint: str | None = None
    selector: Literal["terminal", "best_validation"] | None = None
    at_step: int | None = Field(default=None, gt=0)
    set: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_phase(self) -> Phase:
        if not _ID.fullmatch(self.id):
            raise ValueError("phase ID must be a safe nonempty identifier")
        if self.transition == "fresh" and (self.parent or self.checkpoint):
            raise ValueError(f"fresh phase {self.id} cannot have a parent")
        if self.transition != "fresh" and bool(self.parent) == bool(self.checkpoint):
            raise ValueError(
                f"phase {self.id} needs exactly one parent phase or checkpoint"
            )
        if self.parent and self.selector is None:
            raise ValueError(f"phase {self.id} requires deterministic parent selector")
        if self.selector == "terminal" and self.parent and self.at_step is None:
            raise ValueError(f"phase {self.id} terminal parent requires at_step")
        return self


class CorpusVariant(StrictModel):
    """One release recipe derived from a pinned Corpus Forge project."""

    id: str
    project: str
    release_set: dict[str, Any] = Field(default_factory=dict)
    view: Literal["lm", "chat"] = "lm"
    vocab_size: int | None = Field(default=None, ge=260)
    tokenizer_artifact: str | None = None
    fraction_tokenizer: str | None = None

    @model_validator(mode="after")
    def valid_variant(self) -> CorpusVariant:
        if not _ID.fullmatch(self.id) or not self.project:
            raise ValueError("corpus variant requires safe ID and project path")
        if (self.vocab_size is None) == (self.tokenizer_artifact is None):
            raise ValueError("corpus variant needs exactly one tokenizer strategy")
        if self.tokenizer_artifact is not None and not _ID.fullmatch(
            self.tokenizer_artifact
        ):
            raise ValueError("tokenizer_artifact must be a safe artifact name")
        if set(self.release_set) - {
            "include_shapes",
            "include_origins",
            "mixture",
            "fraction",
            "accepted_generation_statuses",
        }:
            raise ValueError(
                "corpus variant changes outside allowlisted release fields"
            )
        return self


class Evaluation(StrictModel):
    id: str
    checkpoint: str
    data: str
    metric: str
    denominator: str
    endpoint: int = Field(gt=0)
    threshold: float | None = None
    role: Literal["gate", "descriptive", "diagnostic", "exploratory", "blinded_surface"]
    cases: str | None = None


class AttemptContractReference(StrictModel):
    """Source-relative, content-pinned operational contract; not a runtime grant."""

    path: str
    sha256: str

    @model_validator(mode="after")
    def valid_reference(self) -> AttemptContractReference:
        from klab.declaration_context import safe_path

        safe_path(Path("."), self.path)
        if not _HEX.fullmatch(self.sha256):
            raise ValueError("attempt contract requires a full SHA-256")
        return self


class Execution(StrictModel):
    worker: str | None = None
    backend: Literal["cpu", "cuda", "rocm", "mps", "metal", "xpu"] | None = None
    workspace: str | None = None
    min_free_bytes: int = Field(default=0, ge=0)
    min_free_inodes: int = Field(default=0, ge=0)
    attempt_contract: AttemptContractReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class Retention(StrictModel):
    keep_periodic: bool = True
    keep_best: bool = True
    keep_previous: bool = True
    keep_terminal: bool = True


class ExperimentPlan(StrictModel):
    plan_version: Literal[1]
    id: str
    base_run: str | RunConfig
    artifacts: dict[str, Artifact] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict)
    corpus_variants: tuple[CorpusVariant, ...] = ()
    axes: tuple[Axis, ...] = ()
    comparisons: tuple[Comparison, ...] = ()
    phases: tuple[Phase, ...] = ()
    evaluations: tuple[Evaluation, ...] = ()
    evaluation_suite: str | None = None
    execution: Execution = Execution()
    retention: Retention = Retention()

    @model_validator(mode="after")
    def valid_plan(self) -> ExperimentPlan:
        if not _ID.fullmatch(self.id):
            raise ValueError("plan id must be a safe nonempty identifier")
        if self.evaluation_suite is not None:
            from klab.declaration_context import safe_path

            safe_path(Path.cwd(), self.evaluation_suite)
        for label, values in (
            ("axes", self.axes),
            ("comparisons", self.comparisons),
            ("phases", self.phases),
            ("evaluations", self.evaluations),
        ):
            names = [value.name if label == "axes" else value.id for value in values]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate {label} identifiers")
        if len(self.axes) > 20:
            raise ValueError("too many experiment axes")
        phase_ids = {phase.id for phase in self.phases} if self.phases else {"main"}
        for comparison in self.comparisons:
            if set(comparison.phases) - phase_ids:
                raise ValueError(
                    f"comparison {comparison.id} references unknown phases"
                )
        variant_ids = [variant.id for variant in self.corpus_variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("duplicate corpus variant identifiers")
        for variant in self.corpus_variants:
            if variant.tokenizer_artifact is None:
                continue
            for reference in (variant.tokenizer_artifact, variant.fraction_tokenizer):
                if reference is None:
                    continue
                artifact = self.artifacts.get(reference)
                if (
                    artifact is None
                    or artifact.kind != "tokenizer"
                    or artifact.from_phase
                ):
                    raise ValueError(
                        f"corpus variant {variant.id} requires an external tokenizer artifact {reference}"
                    )
        for name, reference in self.inputs.items():
            if reference not in self.artifacts:
                raise ValueError(
                    f"input {name} references unknown artifact {reference}"
                )
        return self


def load_plan(path: Path) -> ExperimentPlan:
    """Parse one authored document; relative file references resolve at consumption."""
    from pydantic import ValidationError

    try:
        return ExperimentPlan.model_validate(read_document(path))
    except ValidationError as error:
        raise ValueError(f"invalid experiment plan {path}: {error}") from error


def base_run_config(plan: ExperimentPlan, source: Path) -> RunConfig:
    """Resolve the existing RunConfig schema with paths anchored to its own file."""
    from klab.config.loading import _resolve_paths

    if isinstance(plan.base_run, RunConfig):
        return plan.base_run
    from klab.recovery.provenance import declaration_reference

    candidate = declaration_reference(source, plan.base_run)
    try:
        return RunConfig.model_validate(
            _resolve_paths(read_document(candidate), candidate.parent.resolve())
        )
    except ValueError as error:
        raise ValueError(f"invalid base_run {candidate}: {error}") from error


def recipe_run_config(
    plan: ExperimentPlan, source: Path, context: RecipeContext
) -> RunConfig:
    """Read the native config through an explicit authenticated recipe context."""
    from klab.config.recipe_paths import _bind_config

    if not isinstance(plan.base_run, str):
        raise TypeError("recipe context requires an inventoried base_run file")
    candidate = context.reference(source, plan.base_run)
    config = RunConfig.model_validate(context.read_document(candidate))
    return _bind_config(
        config,
        candidate,
        context,
        bindings=context.config_paths.get(context.document_name(candidate)),
    )


def schema() -> dict[str, Any]:
    """Versioned JSON Schema for the authored language."""
    return ExperimentPlan.model_json_schema()
