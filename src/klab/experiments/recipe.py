"""Inspect-only portable recipe contracts over the existing native compiler.

This is a bounded matrix preview, not a lock, capability proof or launch route.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from klab.canonical import canonical_json
from klab.config.models import RunConfig, StrictModel
from klab.config.recipe_paths import (
    _INPUT_FIELDS,
    _OUTPUT_FIELDS,
    _bind_config,
    _paths,
)
from klab.declaration_context import RecipeContext, canonical_reference, safe_path
from klab.documents import parse_document
from klab.experiments.artifact_paths import artifact_path
from klab.experiments.compiler import (
    apply_patch,
    compare_configs,
    expand_axes,
    flatten_config,
)
from klab.experiments.phase_graph import validate_phase_graph
from klab.experiments.plan import ExperimentPlan
from klab.recovery.provenance import declaration_reference
from klab.training.manifest import config_sha256, source_identity

MAX_DOCUMENT_BYTES = 1024 * 1024
MAX_RECIPE_BYTES = 8 * MAX_DOCUMENT_BYTES
MAX_DOCUMENTS = 64
_CONTRACTS = {"experiment_plan": 1, "run_config": 2}


class RecipeFile(StrictModel):
    path: StrictStr
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal["experiment_plan", "run_config", "metadata"]

    @field_validator("path")
    @classmethod
    def canonical_path(cls, value: str) -> str:
        return canonical_reference(value)


class PortableRecipe(StrictModel):
    portable_recipe_version: Literal[1]
    id: StrictStr = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    entrypoint: StrictStr
    contracts: dict[StrictStr, StrictInt]
    files: tuple[RecipeFile, ...] = Field(min_length=1, max_length=MAX_DOCUMENTS)
    # Expected identities only: inspection never reads or hashes these payloads.
    artifacts: dict[StrictStr, StrictStr] = Field(default_factory=dict)

    @field_validator("portable_recipe_version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("portable_recipe_version must be integer 1")
        return value

    @model_validator(mode="after")
    def valid_inventory(self) -> PortableRecipe:
        canonical_reference(self.entrypoint)
        names = [member.path for member in self.files]
        if len(set(names)) != len(names):
            raise ValueError("duplicate recipe file path")
        plans = [
            member.path for member in self.files if member.role == "experiment_plan"
        ]
        if plans != [self.entrypoint]:
            raise ValueError("entrypoint must name the sole declared experiment_plan")
        if self.contracts != _CONTRACTS:
            raise ValueError(
                f"unavailable recipe contracts: supported contracts are {_CONTRACTS}"
            )
        for field, digest in self.artifacts.items():
            if field not in _INPUT_FIELDS:
                raise ValueError(f"unsupported artifact field: {field}")
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError(f"artifact {field} requires lowercase SHA-256")
        return self


@dataclass(frozen=True)
class LoadedRecipe:
    descriptor: PortableRecipe
    context: RecipeContext
    descriptor_sha256: str
    fingerprint: str


def _metadata_bytes(path: Path) -> bytes:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked recipe document: {component}")
    if not path.is_file():
        raise ValueError(f"unavailable recipe document: {path}")
    with path.open("rb") as stream:
        data = stream.read(MAX_DOCUMENT_BYTES + 1)
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"recipe document exceeds {MAX_DOCUMENT_BYTES} bytes: {path}")
    return data


def snapshot_inventory(
    path: Path, raw: bytes, files: tuple[RecipeFile, ...]
) -> RecipeContext:
    snapshots: dict[str, bytes] = {}
    total = len(raw)
    for member in files:
        target = safe_path(path.parent, member.path)
        if target == path:
            raise ValueError("recipe descriptor cannot inventory itself")
        data = _metadata_bytes(target)
        total += len(data)
        if total > MAX_RECIPE_BYTES:
            raise ValueError(f"recipe metadata exceeds {MAX_RECIPE_BYTES} bytes")
        if hashlib.sha256(data).hexdigest() != member.sha256:
            raise ValueError(f"recipe document digest mismatch: {target}")
        # Every inventoried member is metadata, never a model/corpus payload.
        parse_document(data.decode("utf-8"), source=target)
        snapshots[member.path] = data
    return RecipeContext(path.parent, snapshots)


def load_recipe(path: Path) -> LoadedRecipe:
    """Authenticate only explicitly inventoried bounded declaration bytes."""
    path = Path(path).absolute()
    raw = _metadata_bytes(path)
    try:
        descriptor = PortableRecipe.model_validate(
            parse_document(raw.decode("utf-8"), source=path)
        )
        context = snapshot_inventory(path, raw, descriptor.files)
        canonical = descriptor.model_dump(mode="json")
        canonical["files"] = sorted(canonical["files"], key=lambda item: item["path"])
        fingerprint = hashlib.sha256(
            b"klab-portable-recipe-v1\0" + canonical_json(canonical)
        ).hexdigest()
        return LoadedRecipe(
            descriptor, context, hashlib.sha256(raw).hexdigest(), fingerprint
        )
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid portable recipe {path}: {error}") from error


def _unavailable(field: str) -> ValueError:
    return ValueError(
        f"unavailable recipe preview feature: {field}; native declarations can use "
        "`klab experiment inspect PLAN --json` (native path rules; recipe bindings are not inferred)"
    )


def inspect_recipe(path: Path, *, max_runs: int = 1000) -> dict[str, Any]:
    """Return a native matrix preview without execution or capability claims."""
    loaded = load_recipe(path)
    descriptor, context = loaded.descriptor, loaded.context
    source = context.root / descriptor.entrypoint
    raw = context.read_document(source)
    # These features require native artifact/phase/evaluation resolution. Do not
    # silently drop them or import their execution stack during a partial preview.
    for field in (
        "inputs",
        "corpus_variants",
        "evaluations",
    ):
        if raw.get(field):
            raise _unavailable(field)
    if raw.get("evaluation_suite") is not None:
        raise _unavailable("evaluation_suite")
    # Unsupported references are refused before native schema validation,
    # including malformed falsey values; schema checks themselves stay lightweight.
    execution = raw.get("execution")
    if isinstance(execution, dict) and execution.get("attempt_contract") is not None:
        raise _unavailable("execution.attempt_contract")
    plan = ExperimentPlan.model_validate(raw)
    if any(artifact.kind != "checkpoint" for artifact in plan.artifacts.values()):
        raise _unavailable("artifacts (only checkpoint declarations supported)")
    phases = validate_phase_graph(plan)
    if plan.phases and plan.comparisons:
        raise _unavailable("phase comparisons")
    if plan.execution.attempt_contract is not None:
        raise _unavailable("execution.attempt_contract")
    if not isinstance(plan.base_run, str):
        raise TypeError("portable recipe requires an inventoried base_run file")
    base_path = declaration_reference(source, plan.base_run, context=context)
    roles = {member.path: member.role for member in descriptor.files}
    if roles[context.document_name(base_path)] != "run_config":
        raise ValueError("base_run must reference a declared run_config")
    config_document = context.read_document(base_path)
    base = RunConfig.model_validate(config_document)
    # Validate original spellings before pathlib can collapse ./ or repeated
    # separators. Only native typed Path fields participate; metadata is opaque.
    for field in _paths(base):
        section, name = field.split(".")
        spelling = config_document.get(section, {}).get(name)
        if spelling is not None:
            canonical_reference(spelling)
    # Path changes across axes need typed artifact selections, not generic
    # last-key-name rewriting. Keep that future composition explicit.
    path_fields = (
        _INPUT_FIELDS
        | _OUTPUT_FIELDS
        | {"dataset.mixture_declaration_path", "dataset.mixture_output_path"}
    )
    for axis in plan.axes:
        for choice in axis.choices:
            for field in choice.set:
                if any(
                    field == item or item.startswith(field + ".")
                    for item in path_fields
                ):
                    raise ValueError(f"unavailable recipe path intervention: {field}")
    for phase in phases:
        for field in phase.set:
            if any(
                field == item or item.startswith(field + ".") for item in path_fields
            ):
                raise ValueError(f"unavailable recipe path intervention: {field}")
    base = _bind_config(base, base_path, context)
    cells = expand_axes(base, plan.axes, max_runs=max_runs, base_dir=context.root)
    phase_previews = []
    for phase in phases:
        # These are requested per-phase patches only. Checkpoint compatibility,
        # parent inheritance and resolved budgets require the native lock verifier.
        patched = [
            RunConfig.model_validate(apply_patch(cell, phase.set)) for cell, _ in cells
        ]
        phase_previews.append(
            {
                **phase.model_dump(mode="json"),
                "consumption": "supplied_checkpoint"
                if phase.checkpoint
                else ("produced_checkpoint" if phase.parent else "fresh_training"),
                "requested_configs": [cell.model_dump(mode="json") for cell in patched],
                "validation": "declaration_connectivity_only",
            }
        )
    comparisons = []
    for comparison in plan.comparisons:
        selected = []
        for selector in (comparison.baseline, comparison.variant):
            matches = [
                cell
                for cell, coordinate in cells
                if all(coordinate.get(key) == value for key, value in selector.items())
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"comparison {comparison.id} must select exactly one cell per arm"
                )
            selected.append(matches[0])
        differences = compare_configs(
            *selected,
            expected=comparison.interventions,
            invariants=comparison.invariants,
            include_derived=True,
        )
        comparisons.append({"id": comparison.id, "differences": differences})
    inputs = {
        field: value for field, value in _paths(base).items() if field in _INPUT_FIELDS
    }
    if set(inputs) != set(descriptor.artifacts):
        raise ValueError(
            "recipe artifacts must declare exactly the native input path fields"
        )
    requirements = [
        {
            "kind": "artifact",
            "field": field,
            "path": str(value),
            "expected_sha256": descriptor.artifacts[field],
            "status": "unverified" if value.exists() else "unavailable",
        }
        for field, value in sorted(inputs.items())
    ]
    for name, artifact in sorted(plan.artifacts.items()):
        if artifact.from_phase is not None:
            if artifact.from_phase not in {phase.id for phase in phases}:
                raise ValueError(f"checkpoint {name} references unknown producer phase")
            requirements.append(
                {
                    "kind": "checkpoint",
                    "name": name,
                    "status": "planned",
                    "from_phase": artifact.from_phase,
                    "selector": artifact.selector,
                    "expected_sha256": None,
                }
            )
        else:
            # Match native plan-relative resolution; portable previews additionally
            # require canonical relative paths and refuse symlinked ancestors.
            canonical_reference(artifact.path)
            path = artifact_path(artifact.path, source)
            safe_path(context.root, path.relative_to(context.root).as_posix())
            requirements.append(
                {
                    "kind": "checkpoint",
                    "name": name,
                    "path": str(path),
                    "expected_sha256": artifact.sha256,
                    "state": artifact.state,
                    "producer": artifact.producer,
                    "identifier": artifact.identifier,
                    "status": "unverified" if path.exists() else "unavailable",
                }
            )
    requirements.append(
        {
            "kind": "runtime",
            "status": "unknown",
            "reason": "capability evidence is not inspected; execution requires native admission",
        }
    )
    lab = source_identity()
    return {
        "format": "klab-recipe-inspection-v1",
        "id": descriptor.id,
        "recipe_sha256": loaded.fingerprint,
        "descriptor_sha256": loaded.descriptor_sha256,
        "installed_lab": {"algorithm": lab["algorithm"], "sha256": lab["sha256"]},
        "binding": {
            "recipe_root": str(context.root),
            "entrypoint": str(source),
            "paths": {
                field: str(value) for field, value in sorted(_paths(base).items())
            },
        },
        "plan_id": plan.id,
        "preview_kind": "native_config_matrix",
        "cells": [
            {
                "coordinate": coordinate,
                "config": cell.model_dump(mode="json"),
                "requested_config_sha256": config_sha256(cell.model_dump(mode="json")),
                "derived": {
                    key: value
                    for key, value in flatten_config(cell, include_derived=True).items()
                    if key.startswith("effective.")
                },
            }
            for cell, coordinate in cells
        ],
        "comparisons": comparisons,
        "phases": phase_previews,
        "execution_request": plan.execution.model_dump(mode="json"),
        "requirements": requirements,
        "status": "incomplete",
        "executable": False,
        "limitations": [
            "not a resolved lock or complete workflow plan",
            "no artifact content verification or runtime validation",
            "recipe fingerprint is not scientific equivalence",
            "phase previews do not resolve parent inheritance, checkpoint compatibility or evaluation",
        ],
    }
