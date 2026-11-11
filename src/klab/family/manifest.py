"""Strict, ordered scientific model lineage; locations are availability bindings only."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from klab.campaign.plan import safe_path
from klab.campaign.state import digest
from klab.config.models import StrictModel
from klab.documents import read_document

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class Identity(StrictModel):
    id: str
    sha256: str

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("unsafe family identifier")
        return value

    @field_validator("sha256")
    @classmethod
    def valid_sha(cls, value: str) -> str:
        if not _SHA.fullmatch(value):
            raise ValueError("invalid SHA-256")
        return value


class LocatedDigest(StrictModel):
    sha256: str
    path: str

    @field_validator("sha256")
    @classmethod
    def valid_sha(cls, value: str) -> str:
        if not _SHA.fullmatch(value):
            raise ValueError("invalid SHA-256")
        return value

    @field_validator("path")
    @classmethod
    def valid_path(cls, value: str) -> str:
        if Path(value).is_absolute():
            if "\\" in value or "\x00" in value or ".." in Path(value).parts:
                raise ValueError("unsafe absolute availability path")
        else:
            safe_path(Path("/"), value)
        return value


class Budget(StrictModel):
    max_steps: int = Field(gt=0)
    max_tokens: int = Field(gt=0)


class FamilyNode(StrictModel):
    id: str
    parent: str | None
    parent_checkpoint_sha256: str | None
    corpus: Identity
    tokenizer: Identity
    plan: Identity
    architecture_sha256: str
    objective: str = Field(min_length=1)
    budget: Budget
    checkpoint: LocatedDigest | None
    evaluation_index: LocatedDigest | None
    readiness_result: LocatedDigest | None
    recovery_manifest: str | None = None
    lifecycle_receipts: tuple[str, ...] = ()

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        return Identity.valid_id(value)

    @field_validator("architecture_sha256", "parent_checkpoint_sha256")
    @classmethod
    def valid_sha(cls, value: str | None) -> str | None:
        if value is not None and not _SHA.fullmatch(value):
            raise ValueError("invalid SHA-256")
        return value

    @field_validator("recovery_manifest")
    @classmethod
    def valid_recovery(cls, value: str | None) -> str | None:
        if value is not None:
            safe_path(Path("/"), value)
        return value

    @field_validator("lifecycle_receipts")
    @classmethod
    def valid_receipts(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("duplicate lifecycle receipt reference")
        for value in values:
            safe_path(Path("/"), value)
        return values

    @model_validator(mode="after")
    def consistent_node(self) -> FamilyNode:
        if (self.parent is None) != (self.parent_checkpoint_sha256 is None):
            raise ValueError("parent and parent checkpoint must occur together")
        if self.parent is not None and not _ID.fullmatch(self.parent):
            raise ValueError("unsafe parent ID")
        if self.checkpoint is None and (self.evaluation_index or self.readiness_result):
            raise ValueError("uncreated checkpoint cannot have evaluation/readiness")
        if self.evaluation_index is None and self.readiness_result is not None:
            raise ValueError("readiness requires evaluation")
        return self


class FamilyManifest(StrictModel):
    family_version: Literal[1]
    id: str
    nodes: tuple[FamilyNode, ...] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        return Identity.valid_id(value)

    @model_validator(mode="after")
    def ordered_lineage(self) -> FamilyManifest:
        prior: dict[str, FamilyNode] = {}
        for node in self.nodes:
            if node.id in prior:
                raise ValueError(f"duplicate family node: {node.id}")
            if node.parent is not None:
                parent = prior.get(node.parent)
                if parent is None:
                    raise ValueError(f"missing or forward family parent: {node.parent}")
                if (
                    parent.checkpoint is None
                    or parent.checkpoint.sha256 != node.parent_checkpoint_sha256
                ):
                    raise ValueError(f"parent checkpoint identity mismatch: {node.id}")
            prior[node.id] = node
        return self

    def identities(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for node in self.nodes:
            result[node.id] = digest(
                "klab-model-family-node-v1",
                {
                    "id": node.id,
                    "parent_node_sha256": result.get(node.parent)
                    if node.parent
                    else None,
                    "parent_checkpoint_sha256": node.parent_checkpoint_sha256,
                    "corpus": node.corpus.model_dump(),
                    "tokenizer": node.tokenizer.model_dump(),
                    "plan": node.plan.model_dump(),
                    "architecture_sha256": node.architecture_sha256,
                    "objective": node.objective,
                    "budget": node.budget.model_dump(),
                    "checkpoint_sha256": node.checkpoint.sha256
                    if node.checkpoint
                    else None,
                    "evaluation_index_sha256": node.evaluation_index.sha256
                    if node.evaluation_index
                    else None,
                    "readiness_result_sha256": node.readiness_result.sha256
                    if node.readiness_result
                    else None,
                },
            )
        return result


def binding_path(source: Path, reference: str) -> Path:
    """Operational artifact location; absolute external-root bindings stay absolute."""
    target = Path(reference)
    if not target.is_absolute():
        return safe_path(source.parent, reference)
    for component in (target, *target.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked family availability path: {component}")
    return target


def load_family(path: Path) -> FamilyManifest:
    family = FamilyManifest.model_validate(read_document(path))
    for node in family.nodes:
        for binding in (node.checkpoint, node.evaluation_index, node.readiness_result):
            if binding is not None:
                binding_path(path, binding.path)
        if node.recovery_manifest is not None:
            safe_path(path.parent, node.recovery_manifest)
        for reference in node.lifecycle_receipts:
            safe_path(path.parent, reference)
    return family


def family_digest(path: Path) -> str:
    """Scientific family identity excludes local locations and later receipt references."""
    family = load_family(path)
    identities = family.identities()
    return digest(
        "klab-model-family-v1",
        {
            "id": family.id,
            "nodes": [identities[node.id] for node in family.nodes],
        },
    )


def schema() -> dict[str, Any]:
    return FamilyManifest.model_json_schema()
