"""Strict, inert packaged research recipes and mechanism lessons."""

from __future__ import annotations

import json
import math
import re
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from klab.canonical import canonical_json, reject_duplicate_keys
from klab.config.models import StrictModel
from klab.training.manifest import sha256_file

_RESOURCE_ROOT = Path(str(files("klab.research").joinpath("resources"))).resolve()
_MAX_JSON_BYTES = 2 * 1024 * 1024
_RUNNER_ONLY_ENTRY_ID = "learned-engram-portability-v1"
_RUNNER_ONLY_GUIDANCE = (
    "learned-engram-portability-v1 uses research portability build "
    "--experiment learned-engram-portability-v1, not research scaffold"
)
_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_SCALE_IDS = ("smoke", "nano", "micro", "tiny")


class _Versioned(StrictModel):
    """Marker base for independently strict versioned resource models."""

    @model_validator(mode="before")
    @classmethod
    def strict_version_number(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        version = value.get("version")
        if type(version) is not int or version != 1:
            raise ValueError("version must be integer 1")
        return value


class Metric(StrictModel):
    name: StrictStr
    unit: StrictStr
    kind: Literal["architectural", "measured", "capability"]
    direction: Literal["lower", "higher", "descriptive"]


class CardReference(StrictModel):
    reference: StrictStr
    role: Literal["development", "stress", "final"]
    category: StrictStr
    applicability: dict[StrictStr, StrictStr]


class PaperReference(StrictModel):
    title: StrictStr
    url: StrictStr
    claim_boundary: StrictStr


class ImplementationLink(StrictModel):
    path: StrictStr
    symbol: StrictStr


class EvaluationPolicy(StrictModel):
    periodic_validation_steps: dict[StrictStr, StrictInt]
    endpoint_cards: list[StrictStr]
    endpoint_policy: Literal["first_configured_limit"]
    milestones_supported: StrictBool
    primary_thresholds: list[dict[str, object]]


class ResearchEntry(_Versioned):
    format: Literal["klab-research-entry"]
    version: Literal[1]
    classification: Literal["recipe_study", "learning_reference"] = "recipe_study"
    id: StrictStr
    title: StrictStr
    question: StrictStr
    mechanisms: list[StrictStr]
    hypothesis: StrictStr
    failure_interpretation: StrictStr
    controls: list[StrictStr]
    independent_variables: list[StrictStr]
    dependent_metrics: list[Metric]
    confounders: list[StrictStr]
    minimum_useful_scale: StrictStr
    recommended_scale: StrictStr
    hardware_notes: StrictStr
    runtime_class: StrictStr
    evaluation_policy: EvaluationPolicy
    cards: list[CardReference]
    papers: list[PaperReference]
    implementation_links: list[ImplementationLink]
    docs: list[StrictStr]
    can_establish: list[StrictStr]
    cannot_establish: list[StrictStr]
    prerequisites: list[StrictStr]
    recipe: StrictStr | None

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("id must be lowercase kebab case")
        return value

    @field_validator("minimum_useful_scale", "recommended_scale")
    @classmethod
    def valid_scale(cls, value: str) -> str:
        if value not in (
            *_SCALE_IDS,
            "reference-small",
            "medium-research",
            "large-local",
        ):
            raise ValueError(f"unsupported scale class: {value}")
        return value

    @field_validator("recipe")
    @classmethod
    def safe_recipe(cls, value: str | None) -> str | None:
        if value is not None:
            _relative_resource(value, "recipe")
        return value

    @model_validator(mode="after")
    def validate_record(self) -> ResearchEntry:
        required_text = (
            self.title,
            self.question,
            self.hypothesis,
            self.failure_interpretation,
            self.hardware_notes,
            self.runtime_class,
        )
        if any(not item.strip() for item in required_text):
            raise ValueError(
                "research title, question, hypothesis, interpretation, hardware, and runtime text must be nonempty"
            )
        if self.classification == "learning_reference":
            if self.recipe is not None:
                raise ValueError("learning reference entries must not declare a recipe")
            if (
                self.evaluation_policy.milestones_supported
                or self.evaluation_policy.primary_thresholds
            ):
                raise ValueError(
                    "learning reference entries do not declare recipe thresholds"
                )
        elif self.id == _RUNNER_ONLY_ENTRY_ID:
            if self.recipe is not None:
                raise ValueError(
                    "runner-only learned portability entry must not declare a recipe"
                )
            if (
                not self.evaluation_policy.milestones_supported
                or self.evaluation_policy.primary_thresholds
                != [{"metric": "held_out_accuracy", "value": 0.75}]
            ):
                raise ValueError(
                    "learned portability entry must declare its milestone policy and held-out threshold"
                )
        elif self.recipe is None:
            raise ValueError(
                "only runner or learning reference entries may omit a recipe"
            )
        elif (
            self.evaluation_policy.milestones_supported
            or self.evaluation_policy.primary_thresholds
        ):
            raise ValueError(
                "Phase A recipes do not support milestones or primary thresholds"
            )
        if any(card.role == "final" for card in self.cards):
            raise ValueError("public catalog cards cannot be marked final")
        if len({card.reference for card in self.cards}) != len(self.cards):
            raise ValueError("card references must be unique")
        if len({metric.name for metric in self.dependent_metrics}) != len(
            self.dependent_metrics
        ):
            raise ValueError("dependent metric names must be unique")
        for path in self.docs:
            _relative_resource(path, "documentation path")
        for link in self.implementation_links:
            _relative_resource(link.path, "implementation path")
        return self


class MatrixOption(StrictModel):
    label: StrictStr
    set: dict[StrictStr, object]


class FactorialFactor(StrictModel):
    axis: StrictStr
    control: StrictStr
    treatment: StrictStr

    @model_validator(mode="after")
    def validate_levels(self) -> FactorialFactor:
        if not self.axis or not self.control or not self.treatment:
            raise ValueError("factorial axis and levels must be nonempty")
        if self.control == self.treatment:
            raise ValueError("factorial control and treatment must differ")
        return self


class FactorialDesign(_Versioned):
    format: Literal["klab-factorial-design"]
    version: Literal[1]
    id: StrictStr
    factors: tuple[FactorialFactor, FactorialFactor]

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("factorial design id must be lowercase kebab case")
        return value

    @model_validator(mode="after")
    def validate_factors(self) -> FactorialDesign:
        if self.factors[0].axis == self.factors[1].axis:
            raise ValueError("factorial factors must use distinct axes")
        return self


class ScaleRecipe(StrictModel):
    base_set: dict[StrictStr, object]
    axes: dict[StrictStr, list[MatrixOption]]
    comparisons: list[dict[str, object]]
    factorial_designs: list[FactorialDesign] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_factorial_designs(self) -> ScaleRecipe:
        identifiers = [design.id for design in self.factorial_designs]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("factorial design ids must be unique")
        for design in self.factorial_designs:
            for factor in design.factors:
                options = self.axes.get(factor.axis)
                if options is None:
                    raise ValueError(
                        f"factorial {design.id} uses unknown axis {factor.axis!r}"
                    )
                labels = [option.label for option in options]
                if len(set(labels)) != len(labels):
                    raise ValueError(
                        f"factorial axis {factor.axis!r} has duplicate labels"
                    )
                if factor.control not in labels or factor.treatment not in labels:
                    raise ValueError(
                        f"factorial {design.id} levels must exist on {factor.axis!r}"
                    )
        return self


class ResearchRecipe(_Versioned):
    format: Literal["klab-research-recipe"]
    version: Literal[1]
    id: StrictStr
    designs: dict[StrictStr, dict[StrictStr, ScaleRecipe]]

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("id must be lowercase kebab case")
        return value

    @model_validator(mode="after")
    def validate_designs(self) -> ResearchRecipe:
        if "default" not in self.designs:
            raise ValueError("research recipe must define a default design")
        for name, scales in self.designs.items():
            if not _ID.fullmatch(name):
                raise ValueError(f"invalid design name: {name!r}")
            if set(scales) != set(_SCALE_IDS):
                raise ValueError(f"design {name} must define smoke, nano, micro, tiny")
            for scale_name, recipe in scales.items():
                if not recipe.axes or not recipe.comparisons:
                    raise ValueError(f"{name}/{scale_name} needs axes and comparisons")
        return self


class LessonStep(StrictModel):
    explanation: StrictStr
    shapes: list[StrictStr]
    observe: list[StrictStr]
    source_path: StrictStr
    source_symbol: StrictStr


class MechanismLesson(_Versioned):
    format: Literal["klab-mechanism-lesson"]
    version: Literal[1]
    id: StrictStr
    title: StrictStr
    mechanism_kind: Literal["model", "artifact"]
    summary: StrictStr
    benefit_scope: list[Literal["training", "inference", "capacity"]]
    steps: list[LessonStep]
    patches_by_scale: dict[StrictStr, dict[StrictStr, object]]
    prerequisites: list[StrictStr]
    try_changes: list[StrictStr]
    limits: list[StrictStr]
    papers: list[PaperReference]
    docs: list[StrictStr]

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("id must be lowercase kebab case")
        return value

    @field_validator("patches_by_scale")
    @classmethod
    def valid_patch_scales(
        cls, value: dict[str, dict[str, object]]
    ) -> dict[str, dict[str, object]]:
        if set(value) - set(_SCALE_IDS):
            raise ValueError("patches_by_scale may contain only runnable scales")
        return value

    @model_validator(mode="after")
    def validate_kind(self) -> MechanismLesson:
        if not self.title.strip() or not self.summary.strip():
            raise ValueError("lesson title and summary must be nonempty")
        if self.mechanism_kind == "model" and set(self.patches_by_scale) != set(
            _SCALE_IDS
        ):
            raise ValueError("model lessons must define every runnable scale")
        if self.mechanism_kind == "artifact" and self.patches_by_scale:
            raise ValueError("artifact-only lessons cannot define model patches")
        return self


class ScaleProfile(StrictModel):
    hidden_dim: StrictInt
    num_layers: StrictInt
    num_heads: StrictInt
    ffn_dim: StrictInt
    seq_len: StrictInt
    max_seq_len: StrictInt
    micro_batch_size: StrictInt
    gradient_accumulation: StrictInt
    max_steps: StrictInt
    max_tokens: StrictInt
    validation_steps: StrictInt
    memory_table_size: StrictInt
    memory_dim: StrictInt

    @model_validator(mode="after")
    def positive_dimensions(self) -> ScaleProfile:
        if any(getattr(self, name) <= 0 for name in type(self).model_fields):
            raise ValueError("scale profile dimensions and budgets must be positive")
        if self.seq_len > self.max_seq_len or self.hidden_dim % self.num_heads:
            raise ValueError(
                "scale profile sequence or head dimensions are inconsistent"
            )
        return self


class ProfilesFile(_Versioned):
    format: Literal["klab-research-profiles"]
    version: Literal[1]
    scales: dict[StrictStr, ScaleProfile]


class DatasetProfile(StrictModel):
    source: Literal["chat_recall", "tinystories", "fineweb_edu", "snapshot"]
    source_declaration: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    revision: StrictStr | None
    dataset_config: StrictStr | None = None
    dataset_seed: StrictInt
    train_max_documents: StrictInt
    validation_max_documents: StrictInt
    train_max_tokens: StrictInt
    validation_max_tokens: StrictInt
    vocab_size: StrictInt
    min_frequency: StrictInt
    tokenizer_max_documents: StrictInt
    tokenizer_train_max_tokens: StrictInt
    tokenizer_validation_max_documents: StrictInt
    tokenizer_validation_max_tokens: StrictInt
    license: StrictStr
    source_url: StrictStr
    notes: list[StrictStr]
    card_applicability: dict[StrictStr, StrictStr]

    @field_validator("source_declaration")
    @classmethod
    def safe_source_declaration(cls, value: str | None) -> str | None:
        if value is not None:
            _relative_resource(value, "source declaration")
        return value

    @model_validator(mode="after")
    def positive_limits(self) -> DatasetProfile:
        if any(
            getattr(self, name) <= 0
            for name in (
                "train_max_documents",
                "validation_max_documents",
                "train_max_tokens",
                "validation_max_tokens",
                "vocab_size",
                "min_frequency",
                "tokenizer_max_documents",
                "tokenizer_train_max_tokens",
                "tokenizer_validation_max_documents",
                "tokenizer_validation_max_tokens",
            )
        ):
            raise ValueError("dataset and tokenizer limits must be positive")
        if self.source == "fineweb_edu" and not self.dataset_config:
            raise ValueError("fineweb_edu requires dataset_config")
        if (self.source == "snapshot") != (self.source_declaration is not None):
            raise ValueError("snapshot profiles require exactly one source_declaration")
        return self


class DatasetsFile(_Versioned):
    format: Literal["klab-research-datasets"]
    version: Literal[1]
    datasets: dict[StrictStr, DatasetProfile]


def _relative_resource(value: str, field: str) -> Path:
    path = Path(value)
    if (
        path.is_absolute()
        or not value
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"{field} must be a safe relative resource path")
    return path


def _read_json(path: Path) -> tuple[object, bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"metadata must be a regular nonsymlink file: {path}")
    data = path.read_bytes()
    if len(data) > _MAX_JSON_BYTES:
        raise ValueError(f"metadata exceeds {_MAX_JSON_BYTES} bytes: {path}")
    try:
        value = json.loads(
            data,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON number: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {path}: {error}") from error
    _reject_nonfinite(value, path)
    return value, data


def _reject_nonfinite(value: object, path: Path) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite JSON number in {path}")
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite(item, path)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item, path)


def resource_root(root: Path | None = None) -> Path:
    """Select one resource closure, refusing symlinked local roots."""
    candidate = _RESOURCE_ROOT if root is None else Path(root).expanduser().absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise ValueError("resource root must not traverse a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("resource root must be a directory")
    return resolved


def resource_path(relative: str, *, root: Path | None = None) -> Path:
    path = _validate_contained(resource_root(root), relative, "resource")
    if not path.is_file():
        raise ValueError(f"resource must be a regular file: {relative}")
    return path


def _read_packaged(
    relative: str, *, root: Path | None = None
) -> tuple[object, bytes, Path]:
    path = resource_path(relative, root=root)
    value, data = _read_json(path)
    return value, data, path


def _validate_contained(root: Path, relative: str, field: str) -> Path:
    relative_path = _relative_resource(relative, field)
    candidate = root.joinpath(relative_path)
    # Reject symlinks at every component, including a symlinked parent directory.
    cursor = root
    for component in relative_path.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"{field} path traverses a symlink: {relative}")
    resolved = candidate.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{field} escapes declared resource directory: {relative}")
    return resolved


def _validate_https(value: str, field: str) -> str:
    from urllib.parse import urlsplit

    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise ValueError(f"{field} must be an inert HTTPS URL")
    return value


def _validate_links(entry: ResearchEntry | MechanismLesson) -> None:
    for paper in entry.papers:
        _validate_https(paper.url, "paper URL")


def _validate_recipe_file(path: Path, entry_id: str) -> ResearchRecipe:
    raw, _ = _read_json(path)
    recipe = ResearchRecipe.model_validate(raw)
    if recipe.id != entry_id:
        raise ValueError(f"recipe id {recipe.id!r} does not match entry {entry_id!r}")
    return recipe


def _available_ids(folder: str, *, root: Path | None = None) -> list[str]:
    return sorted(path.stem for path in (resource_root(root) / folder).glob("*.json"))


def load_research(reference: str | Path, *, root: Path | None = None) -> ResearchEntry:
    """Load a packaged entry id or an explicit local JSON fork."""
    candidate = Path(reference).expanduser()
    local_file = candidate.suffix.lower() == ".json"
    if local_file:
        if not candidate.exists():
            raise ValueError(
                f"research entry file does not exist: {candidate}; packaged IDs: "
                f"{', '.join(_available_ids('catalog', root=root))}"
            )
        if candidate.is_symlink():
            raise ValueError("local research entry must not be a symlink")
        source = candidate.resolve(strict=True)
        if root is not None:
            selected_root = resource_root(root)
            try:
                relative = candidate.absolute().relative_to(selected_root)
            except ValueError as error:
                raise ValueError(
                    "local research entry must be inside resource root"
                ) from error
            source = resource_path(relative.as_posix(), root=selected_root)
        raw, _ = _read_json(source)
        entry = ResearchEntry.model_validate(raw)
        recipe_path = (
            None
            if entry.recipe is None
            else _validate_contained(source.parent, entry.recipe, "recipe")
        )
    else:
        identifier = str(reference)
        if not _ID.fullmatch(identifier) or identifier not in _available_ids(
            "catalog", root=root
        ):
            raise ValueError(
                f"unknown research entry {identifier!r}; available IDs: "
                f"{', '.join(_available_ids('catalog', root=root))}"
            )
        raw, _, _ = _read_packaged(f"catalog/{identifier}.json", root=root)
        entry = ResearchEntry.model_validate(raw)
        recipe_path = (
            None
            if entry.recipe is None
            else _validate_contained(resource_root(root), entry.recipe, "recipe")
        )
    if entry.id != (candidate.stem if local_file else str(reference)):
        raise ValueError("research entry id does not match requested identifier")
    _validate_links(entry)
    if recipe_path is not None:
        _validate_recipe_file(recipe_path, entry.id)
    return entry


def load_recipe(
    entry: ResearchEntry, *, local_root: Path | None = None
) -> ResearchRecipe:
    if entry.id == _RUNNER_ONLY_ENTRY_ID:
        raise ValueError(_RUNNER_ONLY_GUIDANCE)
    if entry.recipe is None:
        raise ValueError(f"research entry {entry.id!r} has no runnable recipe")
    root = local_root.resolve(strict=True) if local_root else _RESOURCE_ROOT
    path = _validate_contained(root, entry.recipe, "recipe")
    return _validate_recipe_file(path, entry.id)


def list_research(*, root: Path | None = None) -> tuple[ResearchEntry, ...]:
    identifiers = sorted(
        path.stem for path in (resource_root(root) / "catalog").glob("*.json")
    )
    return tuple(load_research(identifier, root=root) for identifier in identifiers)


def load_lesson(identifier: str, *, root: Path | None = None) -> MechanismLesson:
    if not _ID.fullmatch(identifier) or identifier not in _available_ids(
        "lessons", root=root
    ):
        raise ValueError(
            f"unknown lesson {identifier!r}; available IDs: "
            f"{', '.join(_available_ids('lessons', root=root))}"
        )
    raw, _, _ = _read_packaged(f"lessons/{identifier}.json", root=root)
    lesson = MechanismLesson.model_validate(raw)
    if lesson.id != identifier:
        raise ValueError("lesson id does not match requested identifier")
    _validate_links(lesson)
    return lesson


def list_lessons(*, root: Path | None = None) -> tuple[MechanismLesson, ...]:
    identifiers = sorted(
        path.stem for path in (resource_root(root) / "lessons").glob("*.json")
    )
    return tuple(load_lesson(identifier, root=root) for identifier in identifiers)


def load_profiles(*, root: Path | None = None) -> ProfilesFile:
    raw, _, _ = _read_packaged("profiles.json", root=root)
    profiles = ProfilesFile.model_validate(raw)
    if set(profiles.scales) != set(_SCALE_IDS):
        raise ValueError("profiles must define exactly smoke, nano, micro, tiny")
    return profiles


def load_datasets(*, root: Path | None = None) -> DatasetsFile:
    raw, _, _ = _read_packaged("datasets.json", root=root)
    datasets = DatasetsFile.model_validate(raw)
    for dataset in datasets.datasets.values():
        _validate_https(dataset.source_url, "dataset source URL")
        if dataset.source_declaration is not None:
            load_dataset_source(dataset, root=root)
    return datasets


def load_dataset_source(
    profile: DatasetProfile, *, root: Path | None = None
) -> dict[str, object]:
    """Read an inert packaged declaration without resolving/acquiring Hub data."""
    from klab.data.sources import DatasetSource

    if profile.source_declaration is None:
        raise ValueError("profile has no source declaration")
    raw, _, _ = _read_packaged(profile.source_declaration, root=root)
    source = DatasetSource.model_validate(raw)
    if source.kind != "huggingface" or set(source.splits) != {"train", "validation"}:
        raise ValueError(
            "packaged snapshot profiles require Hub train/validation declarations"
        )
    if source.revision != profile.revision:
        raise ValueError("profile and source declaration revisions differ")
    if source.selection.mode == "bounded" and (
        source.selection.documents["train"]
        < max(profile.train_max_documents, profile.tokenizer_max_documents)
        or source.selection.documents["validation"]
        < max(
            profile.validation_max_documents, profile.tokenizer_validation_max_documents
        )
    ):
        raise ValueError("source selection cannot satisfy profile document limits")
    return source.model_dump(mode="json")


def resource_sha256(relative: str, *, root: Path | None = None) -> str:
    _, _, path = _read_packaged(relative, root=root)
    return sha256_file(path)


def canonical_payload(value: object) -> bytes:
    return canonical_json(value)
