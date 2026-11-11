"""Strict deterministic Cartesian expansion for concrete run configurations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from klab.canonical import canonical_json
from klab.config.loading import load_config
from klab.config.models import RunConfig
from klab.experiments.compiler import (
    apply_patch as _apply_patch,
)
from klab.experiments.compiler import (
    axis_options,
    expand_axes,
)
from klab.training.manifest import config_sha256
from klab.workers.models import SchedulingRequirements


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader which makes duplicate mapping keys an input error."""


def _construct_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


@dataclass(frozen=True)
class ExpandedExperiment:
    """A concrete, validated configuration selected by one matrix coordinate."""

    config: RunConfig
    coordinate: dict[str, str]
    matrix_sha256: str
    preferred_worker: str | None
    requirements: SchedulingRequirements


def _load_matrix(path: Path) -> dict[str, object]:
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except OSError as error:
        raise ValueError(
            f"cannot read matrix {path}: {error.strerror or error}"
        ) from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in matrix {path}: {error}") from error
    if not isinstance(raw, dict):
        raise TypeError("matrix must contain a YAML mapping")
    if type(raw.get("matrix_version")) is not int or raw["matrix_version"] != 1:
        raise ValueError("matrix_version must be exactly 1")
    return raw


def _requirements(value: object) -> SchedulingRequirements:
    if value is None:
        return SchedulingRequirements()
    if not isinstance(value, dict):
        raise TypeError("matrix scheduling.requirements must be a mapping")
    return SchedulingRequirements.model_validate(value)


def _scheduling(
    raw: Mapping[str, object],
) -> tuple[str | None, SchedulingRequirements]:
    scheduling = raw.get("scheduling", {})
    if not isinstance(scheduling, dict):
        raise TypeError("matrix scheduling must be a mapping")
    allowed = {"preferred_worker", "requirements"}
    unexpected = set(scheduling) - allowed
    if unexpected:
        raise ValueError(f"unknown matrix scheduling fields: {sorted(unexpected)!r}")
    preferred = scheduling.get("preferred_worker")
    if preferred is not None and (
        not isinstance(preferred, str) or not preferred.strip()
    ):
        raise ValueError("matrix scheduling.preferred_worker must be a nonempty string")
    return preferred, _requirements(scheduling.get("requirements"))


def _axis_options(
    raw: Mapping[str, object], base: RunConfig, *, max_runs: int
) -> tuple[list[str], list[list[dict[str, object]]]]:
    axes = raw.get("axes")
    if not isinstance(axes, dict) or not axes:
        raise ValueError("matrix axes must be a nonempty mapping")
    names: list[str] = []
    options: list[list[dict[str, object]]] = []
    for axis, entries in axes.items():
        if not isinstance(axis, str) or not axis.strip():
            raise ValueError("matrix axis names must be nonempty strings")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"axis {axis} must be nonempty")
        labels: set[str] = set()
        typed: list[dict[str, object]] = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"label", "set"}:
                raise ValueError(f"axis {axis} entries require exactly label and set")
            label = entry["label"]
            settings = entry["set"]
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"axis {axis} labels must be nonempty strings")
            if label in labels:
                raise ValueError(f"axis {axis} has duplicate label: {label}")
            labels.add(label)
            if not isinstance(settings, dict):
                raise TypeError(f"axis {axis} set must be a mapping")
            typed.append(entry)
        names.append(axis)
        options.append(typed)
    axis_options(
        base,
        [
            {"name": name, "choices": entries}
            for name, entries in zip(names, options, strict=True)
        ],
        max_runs=max_runs,
        source="matrix",
    )
    return names, options


def expand(path: Path, max_runs: int = 1000) -> list[ExpandedExperiment]:
    """Expand a strict v1 matrix into ordered, concrete, validated configurations."""
    if isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs <= 0:
        raise ValueError("max_runs must be a positive integer")
    raw = _load_matrix(path)
    if set(raw) - {"matrix_version", "base_config", "axes", "scheduling"}:
        raise ValueError(
            f"unknown matrix fields: {sorted(set(raw) - {'matrix_version', 'base_config', 'axes', 'scheduling'})!r}"
        )
    base_config = raw.get("base_config")
    if not isinstance(base_config, str) or not base_config:
        raise ValueError("matrix base_config must be a nonempty path string")
    base_path = Path(base_config)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    resolved_base = load_config(base_path)
    base = resolved_base.model_dump(mode="python")
    names, options = _axis_options(raw, resolved_base, max_runs=max_runs)
    preferred_worker, requirements = _scheduling(raw)
    matrix_sha256 = hashlib.sha256(
        canonical_json(
            {
                "matrix_version": 1,
                "base_config_sha256": config_sha256(base),
                "axes": [
                    {"name": name, "options": entries}
                    for name, entries in zip(names, options, strict=True)
                ],
                "scheduling": {
                    "preferred_worker": preferred_worker,
                    "requirements": requirements.model_dump(mode="json"),
                },
            }
        )
    ).hexdigest()
    matrix_dir = path.parent.resolve()
    expanded: list[ExpandedExperiment] = []
    axes = [
        {"name": name, "choices": entries}
        for name, entries in zip(names, options, strict=True)
    ]
    for concrete, coordinate in expand_axes(
        resolved_base, axes, max_runs=max_runs, base_dir=matrix_dir
    ):
        expanded.append(
            ExpandedExperiment(
                config=concrete,
                coordinate=coordinate,
                matrix_sha256=matrix_sha256,
                preferred_worker=preferred_worker,
                requirements=requirements,
            )
        )
    return expanded


def apply_patch(
    config: Mapping[str, object], patch: Mapping[str, object]
) -> dict[str, object]:
    """Legacy entry point for canonical checked dotted config updates."""
    return _apply_patch(config, patch, source="matrix")
