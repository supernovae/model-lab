"""Inventory-backed preparation planning; no acquisition, admission or executor."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from klab.campaign.plan import CampaignPlan
from klab.campaign.preparation_plan import CompiledPreparation, compile_preparation
from klab.canonical import canonical_json
from klab.config.models import RunConfig, StrictModel, TokenizerTrainConfig
from klab.config.recipe_paths import _INPUT_FIELDS, _OUTPUT_FIELDS, _paths
from klab.declaration_context import RecipeContext, canonical_reference
from klab.documents import parse_document
from klab.experiments.recipe import (
    MAX_DOCUMENTS,
    _metadata_bytes,
    snapshot_inventory,
)
from klab.experiments.workflow_recipe import _location
from klab.training.manifest import source_identity


class PreparationFile(StrictModel):
    path: StrictStr
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal[
        "campaign", "run_config", "tokenizer_config", "metadata", "source_lock"
    ]

    @field_validator("path")
    @classmethod
    def valid_path(cls, value):
        return canonical_reference(value)


class PreparationRecipe(StrictModel):
    portable_preparation_version: StrictInt = Field(ge=1, le=1)
    id: StrictStr = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    entrypoint: StrictStr
    contracts: dict[StrictStr, StrictInt]
    files: tuple[PreparationFile, ...] = Field(min_length=1, max_length=MAX_DOCUMENTS)

    @model_validator(mode="after")
    def valid_inventory(self):
        canonical_reference(self.entrypoint)
        names = [item.path for item in self.files]
        if len(names) != len(set(names)):
            raise ValueError("duplicate preparation file path")
        if [m.path for m in self.files if m.role == "campaign"] != [self.entrypoint]:
            raise ValueError("entrypoint must name the sole declared campaign")
        if self.contracts != {"campaign": 1, "run_config": 2, "tokenizer_config": 1}:
            raise ValueError("unsupported preparation contracts")
        return self


class AcquisitionPaths(StrictModel):
    output: StrictStr
    cache_dir: StrictStr


class PreparationBindings(StrictModel):
    artifacts: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    config_paths: dict[StrictStr, dict[StrictStr, StrictStr]]
    acquisitions: dict[StrictStr, AcquisitionPaths] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def current_contract(cls, value):
        if isinstance(value, dict) and "preparation_bindings_version" in value:
            raise ValueError(
                "obsolete preparation bindings; regenerate locations for the current authored recipe"
            )
        return value


@dataclass(frozen=True)
class LoadedPreparation:
    descriptor: PreparationRecipe
    context: RecipeContext
    plan: CampaignPlan
    compiled: CompiledPreparation
    configs: Mapping[str, RunConfig | TokenizerTrainConfig]
    artifact_locations: Mapping[str, Path]
    recipe_sha256: str
    binding_sha256: str
    acquisition_paths: Mapping[str, Mapping[str, Path]]


def _config(
    context: RecipeContext,
    source: Path,
    reference: str,
    role: str,
    roles: Mapping[str, str],
) -> RunConfig | TokenizerTrainConfig:
    target = context.reference(source, reference)
    if roles.get(reference) != role:
        raise ValueError(f"preparation config must reference a {role} member")
    bindings = context.config_paths.get(reference)
    if bindings is None:
        raise ValueError(f"missing preparation config bindings: {reference}")
    model = RunConfig if role == "run_config" else TokenizerTrainConfig
    document = context.read_document(target)
    config = model.model_validate(document)
    paths = _paths(config)
    if set(bindings) != set(paths):
        raise ValueError("config_paths must bind exactly the declared native paths")
    allowed = _INPUT_FIELDS | _OUTPUT_FIELDS | {"output_dir"}
    raw = config.model_dump(mode="python")
    for field, value in paths.items():
        if field not in allowed:
            raise ValueError(f"unsupported preparation path field: {field}")
        parts = field.split(".")
        authored = document
        for part in parts:
            authored = authored.get(part) if isinstance(authored, dict) else None
        canonical_reference(authored if authored is not None else value.as_posix())
        container = raw
        for part in parts[:-1]:
            container = container[part]
        container[parts[-1]] = bindings[field]
    return model.model_validate(raw)


def load_preparation_recipe(path: Path, bindings: Path) -> LoadedPreparation:
    path, bindings = Path(path).absolute(), Path(bindings).absolute()
    raw = _metadata_bytes(path)
    descriptor = PreparationRecipe.model_validate(
        parse_document(raw.decode(), source=path)
    )
    context = snapshot_inventory(path, raw, descriptor.files)
    if bindings == path or bindings in [
        context.root / name for name in context.documents
    ]:
        raise ValueError(
            "local bindings must be separate from inventoried declarations"
        )
    local_bytes = _metadata_bytes(bindings)
    local_document = parse_document(local_bytes.decode(), source=bindings)
    local = PreparationBindings.model_validate(local_document)
    source = context.reference(
        context.root / descriptor.entrypoint, descriptor.entrypoint
    )
    authored = context.read_document(source)
    # Refuse unsupported stages before their native validators can import computation.
    if authored.get("recovery") is not None:
        raise ValueError("preparation recovery context is unavailable")
    stages = authored.get("stages")
    if not isinstance(stages, list):
        raise TypeError("preparation requires a native stages list")
    for stage in stages:
        if not isinstance(stage, dict) or stage.get("kind") not in {
            "tokenizer_train",
            "data_prepare",
            "artifact_reference",
            "dataset_snapshot",
        }:
            raise ValueError("unsupported preparation operation")
        if stage["kind"] == "artifact_reference":
            artifact = stage.get("artifact")
            if not isinstance(artifact, dict) or artifact.get("kind") not in {
                "tokenizer",
                "dataset_snapshot",
            }:
                raise ValueError(
                    "preparation supports only external tokenizer and dataset_snapshot references"
                )
            canonical_reference(artifact.get("path", ""))
    roles = {member.path: member.role for member in descriptor.files}
    if any(
        stage["kind"] == "dataset_snapshot" and not isinstance(stage.get("id"), str)
        for stage in stages
    ):
        raise ValueError("acquisition stage requires a string id")
    acquisitions = {
        stage["id"]: stage for stage in stages if stage["kind"] == "dataset_snapshot"
    }
    bound_acquisitions = local.acquisitions
    if set(bound_acquisitions) != set(acquisitions):
        raise ValueError("acquisition bindings must name exactly the snapshot stages")
    acquisition_paths = {}
    for name, stage in acquisitions.items():
        for field in ("lock", "output", "cache_dir"):
            canonical_reference(stage.get(field, ""))
        if roles.get(stage["lock"]) != "source_lock":
            raise ValueError(
                "acquisition lock must be an inventoried source_lock member"
            )
        if stage.get("resume", False):
            raise ValueError(
                "portable acquisition uses explicit Campaign resume, not stage.resume"
            )
        acquisition_paths[name] = {
            "lock": context.reference(source, stage["lock"]),
            "output": _location(bindings, bound_acquisitions[name].output),
            "cache_dir": _location(bindings, bound_acquisitions[name].cache_dir),
        }
    plan = CampaignPlan.model_validate(authored)
    compiled = compile_preparation(plan)
    if not compiled.operations:
        raise ValueError("preparation requires at least one preparation operation")
    references = {
        stage.id: stage for stage in plan.stages if stage.kind == "artifact_reference"
    }
    if set(local.artifacts) != set(references):
        raise ValueError(
            "artifact bindings must name exactly the external producer stages"
        )
    locations = {
        name: _location(bindings, value) for name, value in local.artifacts.items()
    }
    locations.update(
        {
            name: paths["output"] / "manifest.json"
            for name, paths in acquisition_paths.items()
        }
    )
    if set(local.config_paths) != {op.config_reference for op in compiled.operations}:
        raise ValueError(
            "config bindings must name exactly the compiled config references"
        )
    context = RecipeContext(
        context.root,
        context.documents,
        {
            name: {field: _location(bindings, value) for field, value in fields.items()}
            for name, fields in local.config_paths.items()
        },
    )
    roles = {member.path: member.role for member in descriptor.files}
    configs = {}
    for op in compiled.operations:
        config = _config(
            context,
            source,
            op.config_reference,
            "run_config" if op.kind == "data_prepare" else "tokenizer_config",
            roles,
        )
        configs[op.stage_id] = config
        snapshots = [item for item in op.inputs if item.port == "snapshot"]
        if (config.dataset.source == "snapshot") != bool(snapshots):
            raise ValueError("snapshot config requires a compiled snapshot input")
        if snapshots:
            expected = locations[snapshots[0].producer]
            if (
                config.dataset.source_manifest_path != expected
                or config.dataset.train_path != expected.parent / "train.jsonl"
                or config.dataset.validation_path
                != expected.parent / "validation.jsonl"
            ):
                raise ValueError(
                    "configured dataset differs from compiled snapshot location"
                )
    for op in compiled.operations:
        if op.kind != "data_prepare":
            continue
        producer = next(item.producer for item in op.inputs if item.port == "tokenizer")
        expected = (
            locations[producer]
            if producer in locations
            else configs[producer].output_dir / "tokenizer.json"
        )
        if configs[op.stage_id].tokenizer.path != expected:
            raise ValueError(
                "configured tokenizer differs from compiled producer location"
            )
    canonical = descriptor.model_dump(mode="json")
    canonical["files"] = sorted(canonical["files"], key=lambda member: member["path"])
    return LoadedPreparation(
        descriptor,
        context,
        plan,
        compiled,
        MappingProxyType(configs),
        MappingProxyType(locations),
        hashlib.sha256(
            b"klab-portable-preparation-v1\0" + canonical_json(canonical)
        ).hexdigest(),
        hashlib.sha256(local_bytes).hexdigest(),
        MappingProxyType(
            {name: MappingProxyType(paths) for name, paths in acquisition_paths.items()}
        ),
    )


def inspect_preparation_recipe(path: Path, bindings: Path) -> dict:
    loaded = load_preparation_recipe(path, bindings)
    return {
        "format": "klab-preparation-inspection-v1",
        "id": loaded.descriptor.id,
        "recipe_sha256": loaded.recipe_sha256,
        "binding_sha256": loaded.binding_sha256,
        "installed_lab": source_identity(),
        "declaration_integrity": "verified_inventory",
        "committed_provenance": "unverified",
        "runtime_capability": "unknown",
        "executable": False,
        "missing_gates": [{"code": "WORKFLOW_EXECUTION_UNAVAILABLE"}],
        "plan": loaded.compiled.model_dump(mode="json"),
        "configs": {
            op.stage_id: {
                "reference": op.config_reference,
                "sha256": hashlib.sha256(
                    loaded.context.documents[op.config_reference]
                ).hexdigest(),
                "bound_paths": {
                    name: str(value)
                    for name, value in _paths(loaded.configs[op.stage_id]).items()
                },
            }
            for op in loaded.compiled.operations
        },
        "acquisitions": {
            name: {field: str(value) for field, value in paths.items()}
            for name, paths in loaded.acquisition_paths.items()
        },
        "artifacts": {
            stage.id: {
                "identity": stage.artifact.model_dump(mode="json", exclude={"path"}),
                "location": str(loaded.artifact_locations[stage.id]),
                "status": "unverified",
            }
            for stage in loaded.plan.stages
            if stage.kind == "artifact_reference"
        },
    }
