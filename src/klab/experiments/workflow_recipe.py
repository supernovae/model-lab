"""External prepared-input workflow profile over native experiment locks.

Metadata inspection, cold resolution and publication are separate operations.
This first profile cannot authorize or submit execution.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Literal

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from klab.canonical import canonical_json
from klab.config.models import RunConfig, StrictModel
from klab.config.recipe_paths import _INPUT_FIELDS, _OUTPUT_FIELDS, _paths
from klab.declaration_context import RecipeContext, canonical_reference
from klab.documents import parse_document
from klab.experiments.compiler import expand_axes
from klab.experiments.plan import Artifact, ExperimentPlan, recipe_run_config
from klab.experiments.recipe import (
    MAX_DOCUMENTS,
    RecipeFile,
    _metadata_bytes,
    snapshot_inventory,
)


class ArtifactIdentity(StrictModel):
    kind: Literal["tokenizer", "prepared_data"]
    version: StrictInt = Field(ge=1, le=1)
    producer: StrictStr = Field(min_length=1)
    identifier: StrictStr = Field(min_length=1)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class WorkflowRecipe(StrictModel):
    portable_workflow_version: Literal[1]
    id: StrictStr = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    entrypoint: StrictStr
    campaign: StrictStr | None = None
    contracts: dict[StrictStr, StrictInt]
    files: tuple[RecipeFile, ...] = Field(min_length=1, max_length=MAX_DOCUMENTS)
    artifacts: dict[StrictStr, ArtifactIdentity]

    @field_validator("portable_workflow_version", mode="before")
    @classmethod
    def exact_version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("portable_workflow_version must be integer 1")
        return value

    @model_validator(mode="after")
    def valid_inventory(self):
        canonical_reference(self.entrypoint)
        if self.campaign is not None:
            canonical_reference(self.campaign)
        names = [member.path for member in self.files]
        if len(names) != len(set(names)):
            raise ValueError("duplicate workflow file path")
        if [m.path for m in self.files if m.role == "experiment_plan"] != [
            self.entrypoint
        ]:
            raise ValueError("entrypoint must name the sole declared experiment_plan")
        if self.contracts != {"experiment_plan": 1, "run_config": 2}:
            raise ValueError("unsupported workflow contracts")
        if sorted(item.kind for item in self.artifacts.values()) != [
            "prepared_data",
            "tokenizer",
        ]:
            raise ValueError(
                "workflow requires exactly one tokenizer and prepared_data identity"
            )
        return self


class LocalBindings(StrictModel):
    workflow_bindings_version: Literal[1]
    artifacts: dict[StrictStr, StrictStr]
    logging_root: StrictStr
    config_paths: dict[StrictStr, StrictStr] = Field(default_factory=dict)

    @field_validator("workflow_bindings_version", mode="before")
    @classmethod
    def exact_version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("workflow_bindings_version must be integer 1")
        return value


def _location(source: Path, value: str) -> Path:
    if not value or "\\" in value or "\x00" in value or PureWindowsPath(value).drive:
        raise ValueError("unsafe local workflow binding")
    declared = Path(value)
    if declared.is_absolute():
        if (
            value.startswith("//")
            or ".." in declared.parts
            or declared.as_posix() != value
        ):
            raise ValueError("ambiguous absolute workflow binding")
        target = declared
    else:
        canonical_reference(value)
        target = source.parent / declared
    for component in (target, *target.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked workflow binding: {component}")
    return target.absolute()


@dataclass(frozen=True)
class LoadedWorkflow:
    descriptor: WorkflowRecipe
    context: RecipeContext
    plan: ExperimentPlan
    source: Path
    descriptor_path: Path
    binding_path: Path
    recipe_sha256: str
    descriptor_sha256: str
    binding_sha256: str


def load_workflow(path: Path, bindings: Path) -> LoadedWorkflow:
    """Snapshot declared metadata and bind explicit locations; never read payloads."""
    path, bindings = Path(path).absolute(), Path(bindings).absolute()
    raw = _metadata_bytes(path)
    descriptor = WorkflowRecipe.model_validate(
        parse_document(raw.decode(), source=path)
    )
    context = snapshot_inventory(path, raw, descriptor.files)
    if bindings == path or bindings in [
        context.root / name for name in context.documents
    ]:
        raise ValueError(
            "local bindings must be separate from inventoried portable declarations"
        )
    binding_bytes = _metadata_bytes(bindings)
    local = LocalBindings.model_validate(
        parse_document(binding_bytes.decode(), source=bindings)
    )
    if set(local.artifacts) != set(descriptor.artifacts):
        raise ValueError("local artifact bindings differ from workflow identity names")
    source = context.reference(
        context.root / descriptor.entrypoint, descriptor.entrypoint
    )
    authored = context.read_document(source)
    for key in (
        "artifacts",
        "corpus_variants",
        "phases",
        "evaluations",
    ):
        if authored.get(key) not in (None, [], {}):
            raise ValueError(f"unavailable workflow profile feature: {key}")
    if descriptor.campaign is None and authored.get("evaluation_suite") not in (
        None,
        [],
        {},
    ):
        raise ValueError("unavailable workflow profile feature: evaluation_suite")
    if descriptor.campaign is not None:
        context.reference(source, descriptor.campaign)
        suite_reference = authored.get("evaluation_suite")
        if not isinstance(suite_reference, str):
            raise ValueError(
                "executable workflow requires an inventoried diagnostic suite"
            )
        suite_path = context.reference(source, suite_reference)
        from klab.evaluation.suite_models import EvaluationSuite

        suite = EvaluationSuite.model_validate(context.read_document(suite_path))
        if len(suite.evaluations) != 1 or any(
            item.kind != "heldout_lm" or item.role != "diagnostic"
            for item in suite.evaluations
        ):
            raise ValueError("workflow supports one diagnostic heldout_lm evaluation")
    execution = authored.get("execution")
    if isinstance(execution, dict) and execution.get("attempt_contract") is not None:
        raise ValueError(
            "unavailable workflow profile feature: execution.attempt_contract"
        )
    identities = descriptor.artifacts
    locations = {
        name: _location(bindings, value) for name, value in local.artifacts.items()
    }
    native_artifacts = {
        name: Artifact.model_validate(
            {**identity.model_dump(), "path": str(locations[name])}
        )
        for name, identity in identities.items()
    }
    plan = ExperimentPlan.model_validate({**authored, "artifacts": native_artifacts})
    if not isinstance(plan.base_run, str):
        raise TypeError("workflow requires an inventoried base_run file")
    if set(plan.inputs) != {"tokenizer", "prepared_data"} or any(
        identities[reference].kind != port for port, reference in plan.inputs.items()
    ):
        raise ValueError("workflow inputs must bind tokenizer and prepared_data ports")
    if plan.execution.workspace is not None or plan.execution.worker is not None:
        raise ValueError(
            "workflow execution locations belong to later native admission"
        )
    config_source = context.reference(source, plan.base_run)
    if (
        next((m.role for m in descriptor.files if m.path == plan.base_run), None)
        != "run_config"
    ):
        raise ValueError("workflow base_run must reference a run_config member")
    document = context.read_document(config_source)
    config = RunConfig.model_validate(document)
    # Refuse path-changing axes and opaque plugin fields; bind only native typed paths.
    paths = _paths(config)
    for field in paths:
        section, name = field.split(".", 1)
        value = document.get(section, {}).get(name)
        if value is not None:
            canonical_reference(value)
    path_fields = (
        _INPUT_FIELDS
        | _OUTPUT_FIELDS
        | {"dataset.mixture_declaration_path", "dataset.mixture_output_path"}
    )
    for axis in plan.axes:
        for choice in axis.choices:
            if any(
                key == field or field.startswith(key + ".")
                for key in choice.set
                for field in path_fields
            ):
                raise ValueError("workflow axes cannot change local path bindings")
    bound = {
        "tokenizer.path": locations[plan.inputs["tokenizer"]],
        "dataset.cache_dir": locations[plan.inputs["prepared_data"]].parent,
        "logging.root_dir": _location(bindings, local.logging_root),
    }
    if set(local.config_paths) != set(paths) - set(bound):
        raise ValueError(
            "config_paths must bind exactly the remaining native input paths"
        )
    bound.update(
        {
            field: _location(bindings, value)
            for field, value in local.config_paths.items()
        }
    )
    context = RecipeContext(context.root, context.documents, {plan.base_run: bound})
    # Exercise native validation with the explicit context before any artifact access.
    recipe_run_config(plan, source, context=context)
    canonical = descriptor.model_dump(mode="json")
    if descriptor.campaign is None:
        canonical.pop("campaign")
    canonical["files"] = sorted(canonical["files"], key=lambda item: item["path"])
    fingerprint = hashlib.sha256(
        b"klab-portable-workflow-v1\0" + canonical_json(canonical)
    ).hexdigest()
    return LoadedWorkflow(
        descriptor,
        context,
        plan,
        source,
        path,
        bindings,
        fingerprint,
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(binding_bytes).hexdigest(),
    )


def inspect_workflow(path: Path, bindings: Path, *, max_runs: int = 1000) -> dict:
    loaded = load_workflow(path, bindings)
    config = recipe_run_config(loaded.plan, loaded.source, context=loaded.context)
    cells = expand_axes(
        config, loaded.plan.axes, max_runs=max_runs, base_dir=loaded.context.root
    )
    return {
        "format": "klab-workflow-inspection-v1",
        "recipe_sha256": loaded.recipe_sha256,
        "binding_sha256": loaded.binding_sha256,
        "recipe_root": str(loaded.context.root),
        "cells": len(cells),
        "artifacts": {
            name: {**identity.model_dump(), "status": "unverified"}
            for name, identity in loaded.descriptor.artifacts.items()
        },
        "runtime_capability": "unknown",
        "executable": False,
        "next": "explicit resolve authenticates existing inputs; publish writes a native lock",
    }


def locked_workflow_context(lock, source: Path) -> LoadedWorkflow:
    """Reconstruct inventoried intent for read-only inspection, never admission.

    The caller must first authenticate the lock and availability sidecar with
    open_lock. Inventory integrity is not trusted committed-source provenance.
    """
    profile = lock.availability["workflow_profile"]
    if (
        not isinstance(profile, dict)
        or type(profile.get("version")) is not int
        or profile["version"] != 1
    ):
        raise ValueError("unsupported authenticated workflow profile")
    loaded = load_workflow(
        Path(profile["descriptor_path"]), Path(profile["binding_path"])
    )
    expected = {
        "recipe_sha256": loaded.recipe_sha256,
        "descriptor_sha256": loaded.descriptor_sha256,
        "binding_sha256": loaded.binding_sha256,
    }
    hashes = {
        name: hashlib.sha256(raw).hexdigest()
        for name, raw in loaded.context.documents.items()
    }
    if (
        any(profile.get(key) != value for key, value in expected.items())
        or source.absolute() != loaded.source
        or lock.availability.get("declaration_hashes") != hashes
        or lock.availability.get("declaration_root") != str(loaded.context.root)
    ):
        raise ValueError("workflow declaration or binding identity changed")
    return loaded


def resolve_workflow(path: Path, bindings: Path, *, max_runs: int = 1000):
    """Cold native verification and resolution; no publication or runtime probing."""
    from klab.data.legacy import require_current_dataset
    from klab.experiments.lock import ResolvedExperimentPlan, resolve_plan

    loaded = load_workflow(path, bindings)
    require_current_dataset(
        recipe_run_config(loaded.plan, loaded.source, context=loaded.context).dataset
    )
    lock = resolve_plan(
        loaded.plan, loaded.source, context=loaded.context, max_runs=max_runs
    )
    availability = {
        **lock.availability,
        "workflow_profile": {
            "version": 1,
            "recipe_sha256": loaded.recipe_sha256,
            "descriptor_sha256": loaded.descriptor_sha256,
            "descriptor_path": str(loaded.descriptor_path),
            "binding_sha256": loaded.binding_sha256,
            "binding_path": str(loaded.binding_path),
            "execution": "unavailable",
        },
    }
    # This new availability record is not the resolver-sealed object. The native
    # publisher intentionally reauthenticates artifacts rather than borrowing proof.
    return ResolvedExperimentPlan.model_validate(
        {**lock.model_dump(mode="python"), "availability": availability}
    )


def publish_workflow(
    path: Path, bindings: Path, workspace: Path, *, max_runs: int = 1000
) -> Path:
    from klab.experiments.lock import publish_lock

    return publish_lock(resolve_workflow(path, bindings, max_runs=max_runs), workspace)
