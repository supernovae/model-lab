"""Authored, ordered deterministic recovery declarations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from klab.config.models import StrictModel
from klab.documents import read_document
from klab.recovery.provenance import declaration_reference

_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def _path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or "\\" in value
    ):
        raise ValueError("reference must be a safe declaration-relative path")
    return value


def _identifier(value: str) -> str:
    if not _ID.fullmatch(value):
        raise ValueError("reference must be a safe nonempty identifier")
    return value


def _reason(value: str) -> str:
    if not value.strip():
        raise ValueError("reason must be nonempty")
    return value


class Step(StrictModel):
    id: str

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("step id must be a safe identifier")
        return value


class SnapshotInheritance(StrictModel):
    """Pinned parent evidence in the work root and exact inherited identities."""

    parent_receipt: str
    parent_receipt_sha256: str
    snapshots: dict[str, str] = Field(min_length=1)
    changed_snapshots: dict[str, str] = Field(default_factory=dict)

    _parent = field_validator("parent_receipt")(_path)

    @model_validator(mode="after")
    def identities(self) -> SnapshotInheritance:
        if not _SHA.fullmatch(self.parent_receipt_sha256):
            raise ValueError("parent receipt requires a full SHA-256")
        if set(self.snapshots) & set(self.changed_snapshots):
            raise ValueError("inherited and changed source IDs overlap")
        for source_id, sha in {**self.snapshots, **self.changed_snapshots}.items():
            _identifier(source_id)
            if not _SHA.fullmatch(sha):
                raise ValueError("snapshot requires a full SHA-256")
        return self


class CorpusRelease(Step):
    kind: Literal["corpus_release"]
    project: str
    expected_release_sha256: str | None = None
    expected_build_sha256: str | None = None
    snapshot_inheritance: SnapshotInheritance | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    _project = field_validator("project")(_path)

    @model_validator(mode="after")
    def inheritance_requires_identity(self) -> CorpusRelease:
        if self.snapshot_inheritance is not None and (
            self.expected_build_sha256 is None or self.expected_release_sha256 is None
        ):
            raise ValueError(
                "snapshot inheritance requires expected build and release identities"
            )
        return self


class CorpusExport(Step):
    kind: Literal["corpus_export"]
    corpus: str
    base_run: str
    view: Literal["lm", "chat"]
    vocab_size: int = Field(ge=260)
    expected_export_sha256: str | None = None

    _base_run = field_validator("base_run")(_path)
    _corpus = field_validator("corpus")(_identifier)


class TokenizerTrain(Step):
    kind: Literal["tokenizer_train"]
    export: str | None = None
    config: str | None = None
    expected_tokenizer_sha256: str | None = None

    _config = field_validator("config")(
        lambda value: _path(value) if value is not None else value
    )
    _export = field_validator("export")(
        lambda value: _identifier(value) if value is not None else value
    )

    @model_validator(mode="after")
    def one_input(self) -> TokenizerTrain:
        if (self.export is None) == (self.config is None):
            raise ValueError("tokenizer_train requires exactly one of export or config")
        return self


class PreparedData(Step):
    kind: Literal["prepared_data"]
    export: str | None = None
    tokenizer: str | None = None
    config: str | None = None
    plan: str | None = None
    variant_id: str | None = None
    expected_manifest_sha256: str | None = None

    @field_validator("config", "plan")
    @classmethod
    def valid_path(cls, value: str | None) -> str | None:
        return _path(value) if value is not None else value

    @field_validator("export", "tokenizer", "variant_id")
    @classmethod
    def valid_link(cls, value: str | None) -> str | None:
        return _identifier(value) if value is not None else value

    @model_validator(mode="after")
    def route(self) -> PreparedData:
        if self.plan is not None:
            if not self.variant_id or any((self.config, self.export, self.tokenizer)):
                raise ValueError("variant preparation needs plan and variant_id only")
        elif (
            self.variant_id
            or (self.config is None and self.export is None)
            or (self.config is not None and self.export is not None)
            or self.tokenizer is None
        ):
            raise ValueError("prepared_data requires export or config and tokenizer")
        return self


class ExperimentLock(Step):
    kind: Literal["experiment_lock"]
    plan: str
    tokenizer: str | None = None
    prepared: str | None = None
    expected_plan_sha256: str | None = None

    _plan = field_validator("plan")(_path)

    @field_validator("tokenizer", "prepared")
    @classmethod
    def valid_link(cls, value: str | None) -> str | None:
        return _identifier(value) if value is not None else value


class ExternalRequired(Step):
    kind: Literal["external_required"]
    role: str
    reason: str
    expected_sha256: str | None = None
    _role = field_validator("role")(_identifier)
    _reason = field_validator("reason")(_reason)


class OptionalCache(Step):
    kind: Literal["optional_cache"]
    role: str
    reason: str
    _role = field_validator("role")(_identifier)
    _reason = field_validator("reason")(_reason)


class Checkpoint(Step):
    kind: Literal["checkpoint"]
    run: str
    cell: str
    expected_sha256: str | None = None
    _run = field_validator("run")(_identifier)
    _cell = field_validator("cell")(_identifier)


RecoveryStep = Annotated[
    CorpusRelease
    | CorpusExport
    | TokenizerTrain
    | PreparedData
    | ExperimentLock
    | ExternalRequired
    | OptionalCache
    | Checkpoint,
    Field(discriminator="kind"),
]


class RuntimeRequirement(StrictModel):
    engine: str
    backend: str
    device_index: int = Field(ge=0)
    requirements: dict[str, str]
    profile: str | None = None

    @field_validator("profile")
    @classmethod
    def valid_profile(cls, value: str | None) -> str | None:
        return _path(value) if value is not None else value


class RecoveryManifest(StrictModel):
    recovery_version: Literal[1]
    id: str
    source_commit: str
    steps: tuple[RecoveryStep, ...] = Field(min_length=1)
    runtime_requirement: RuntimeRequirement | None = None
    evaluation_suite: str | None = None
    readiness_policy: str | None = None
    family: str | None = None
    evidence: tuple[str, ...] = ()

    @field_validator("evidence")
    @classmethod
    def valid_evidence(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("duplicate recovery evidence references")
        for value in values:
            _path(value)
        return values

    @field_validator("evaluation_suite", "readiness_policy", "family")
    @classmethod
    def valid_reference(cls, value: str | None) -> str | None:
        return _path(value) if value is not None else value

    @model_validator(mode="after")
    def graph(self) -> RecoveryManifest:
        if not _ID.fullmatch(self.id) or not _COMMIT.fullmatch(self.source_commit):
            raise ValueError("invalid recovery id or source commit")
        previous: dict[str, str] = {}
        links = {
            "corpus_export": {"corpus": "corpus_release"},
            "tokenizer_train": {"export": "corpus_export"},
            "prepared_data": {
                "export": "corpus_export",
                "tokenizer": "tokenizer_train",
            },
            "experiment_lock": {
                "tokenizer": "tokenizer_train",
                "prepared": "prepared_data",
            },
        }
        for step in self.steps:
            if step.id in previous:
                raise ValueError(f"duplicate recovery step: {step.id}")
            for name, kind in links.get(step.kind, {}).items():
                reference = getattr(step, name, None)
                if reference is not None and previous.get(reference) != kind:
                    raise ValueError(
                        f"{step.id}.{name} must refer to an earlier {kind}"
                    )
            if (
                isinstance(step, PreparedData)
                and step.plan is not None
                and not _ID.fullmatch(step.variant_id or "")
            ):
                raise ValueError("variant_id must be a safe identifier")
            for name, value in step.model_dump().items():
                if (
                    name.startswith("expected_")
                    and name.endswith("sha256")
                    and value is not None
                    and not _SHA.fullmatch(value)
                ):
                    raise ValueError(f"invalid {step.id}.{name} SHA-256")
            previous[step.id] = step.kind
        if not any(isinstance(s, ExternalRequired) for s in self.steps) and (
            self.runtime_requirement is None
            or self.evaluation_suite is None
            or self.readiness_policy is None
            or self.family is None
        ):
            raise ValueError(
                "complete recovery needs runtime, suite, readiness and family"
            )
        return self


def load_manifest(source: Path) -> RecoveryManifest:
    source = Path(source).absolute()
    if source.is_symlink():
        raise ValueError(f"symlinked recovery manifest: {source}")
    manifest = RecoveryManifest.model_validate(read_document(source))
    for step in manifest.steps:
        for name in ("project", "base_run", "config", "plan"):
            reference = getattr(step, name, None)
            if reference is not None:
                declaration_reference(source, reference)
    for reference in (
        manifest.evaluation_suite,
        manifest.readiness_policy,
        manifest.family,
        manifest.runtime_requirement.profile if manifest.runtime_requirement else None,
        *manifest.evidence,
    ):
        if reference is not None:
            declaration_reference(source, reference)
    return manifest
