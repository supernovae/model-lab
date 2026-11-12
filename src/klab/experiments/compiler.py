"""Pure expansion and comparison of validated concrete run configurations.

This module does not depend on the authored plan language or the matrix format;
both adapt their declarations to this authority.
"""

from __future__ import annotations

import copy
from collections.abc import Collection, Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from klab.config.loading import _PATH_KEYS
from klab.config.models import AdamWConfig, RunConfig


def patchable_config(value: object) -> object:
    """Project Pydantic defaults to mutable Python mappings without JSON path loss."""
    if isinstance(value, BaseModel):
        return {
            name: patchable_config(getattr(value, name))
            for name in type(value).model_fields
        }
    if isinstance(value, Mapping):
        return {key: patchable_config(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(patchable_config(item) for item in value)
    if isinstance(value, list):
        return [patchable_config(item) for item in value]
    return value


def is_dotted_path(value: object) -> bool:
    """Recognize a nonempty, nondegenerate dotted field name."""
    return isinstance(value, str) and bool(value) and all(value.split("."))


def validate_path(
    config: Mapping[str, object], dotted: str, *, source: str = "config"
) -> None:
    """Require a dotted path to address an existing concrete configuration field."""
    if not isinstance(dotted, str):
        raise TypeError(f"{source} patch paths must be strings")
    if not is_dotted_path(dotted):
        raise ValueError(f"unknown {source} path: {dotted!r}")
    cursor: object = config
    for component in dotted.split("."):
        if not isinstance(cursor, Mapping) or component not in cursor:
            raise ValueError(f"unknown {source} path: {dotted}")
        cursor = cursor[component]


def paths_overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(right + ".") or right.startswith(left + ".")


def validate_patch(
    config: Mapping[str, object], patch: Mapping[str, object], *, source: str = "config"
) -> None:
    previous: list[str] = []
    for dotted in patch:
        validate_path(config, dotted, source=source)
        for other in previous:
            if paths_overlap(dotted, other):
                raise ValueError(f"overlapping {source} patches: {other} and {dotted}")
        previous.append(dotted)


def apply_patch(
    config: Mapping[str, object] | RunConfig,
    patch: Mapping[str, object],
    *,
    source: str = "config",
) -> dict[str, object]:
    """Apply checked, nonoverlapping dotted updates to a detached config copy."""
    projected = patchable_config(config)
    if not isinstance(projected, dict):
        raise TypeError("config must project to a mapping")
    validate_patch(projected, patch, source=source)
    result = copy.deepcopy(projected)
    for dotted, value in patch.items():
        cursor = result
        components = dotted.split(".")
        for component in components[:-1]:
            cursor = cursor[component]
        cursor[components[-1]] = value
    return result


def _member(value: object, name: str) -> object:
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def axis_options(
    base: RunConfig,
    axes: Sequence[object],
    *,
    max_runs: int,
    source: str = "plan",
) -> tuple[tuple[str, ...], tuple[tuple[tuple[str, Mapping[str, object]], ...], ...]]:
    """Validate axis declaration and bound the product before allocating cells."""
    if isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs <= 0:
        raise ValueError("max_runs must be a positive integer")
    projected = patchable_config(base)
    assert isinstance(projected, dict)
    names: list[str] = []
    options: list[tuple[tuple[str, Mapping[str, object]], ...]] = []
    paths_by_axis: dict[str, set[str]] = {}
    count = 1
    for axis in axes:
        name = _member(axis, "name")
        if not isinstance(name, str) or not name.strip() or name in paths_by_axis:
            raise ValueError(
                f"{source} axis names must be unique nonempty strings: {name!r}"
            )
        choices = _member(axis, "choices")
        if not isinstance(choices, (tuple, list)) or not choices:
            raise ValueError(f"axis {name} must be nonempty")
        count *= len(choices)
        if count > max_runs:
            raise ValueError(f"{source} expansion exceeds max_runs ({max_runs})")
        labels: set[str] = set()
        paths: set[str] = set()
        selected: list[tuple[str, Mapping[str, object]]] = []
        for choice in choices:
            label, settings = _member(choice, "label"), _member(choice, "set")
            if not isinstance(label, str) or not label.strip() or label in labels:
                raise ValueError(
                    f"axis {name} has invalid or duplicate label: {label!r}"
                )
            labels.add(label)
            if not isinstance(settings, Mapping):
                raise TypeError(f"axis {name} set must be a mapping")
            validate_patch(projected, settings, source=source)
            paths.update(settings)
            selected.append((label, settings))
        for other_name, other_paths in paths_by_axis.items():
            for dotted in sorted(paths):
                for other in sorted(other_paths):
                    if paths_overlap(dotted, other):
                        raise ValueError(
                            f"overlapping {source} patches: {name}.{dotted} "
                            f"conflicts with {other_name}.{other}"
                        )
        paths_by_axis[name] = paths
        names.append(name)
        options.append(tuple(selected))
    return tuple(names), tuple(options)


def expand_axes(
    base: RunConfig,
    axes: Sequence[object],
    *,
    max_runs: int,
    base_dir: Path,
) -> tuple[tuple[RunConfig, dict[str, str]], ...]:
    """Yield ordered concrete cells; no axes means the direct one-cell run."""
    names, options = axis_options(base, axes, max_runs=max_runs)
    if not options:
        return ((base, {}),)
    projected = patchable_config(base)
    assert isinstance(projected, dict)
    expanded: list[tuple[RunConfig, dict[str, str]]] = []
    for selected in product(*options):
        coordinate = dict(zip(names, (label for label, _ in selected), strict=True))
        patch: dict[str, object] = {}
        for _, settings in selected:
            for dotted, value in settings.items():
                if isinstance(value, str) and dotted.rsplit(".", 1)[-1] in _PATH_KEYS:
                    path = Path(value)
                    value = path if path.is_absolute() else (base_dir / path).resolve()
                patch[dotted] = value
        try:
            concrete = RunConfig.model_validate(apply_patch(projected, patch))
        except ValueError as error:
            raise ValueError(
                f"invalid config at coordinate {coordinate}: {error}"
            ) from error
        expanded.append((concrete, coordinate))
    return tuple(expanded)


def flatten_config(
    config: RunConfig | Mapping[str, object], *, include_derived: bool = False
) -> dict[str, object]:
    """Flatten all resolved fields, including defaults, without scrubbing identities."""
    value: Mapping[str, object] = (
        config.model_dump(mode="json") if isinstance(config, RunConfig) else config
    )
    flattened: dict[str, object] = {}

    def visit(item: object, prefix: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"non-string configuration field under {prefix!r}")
                visit(nested, f"{prefix}.{key}" if prefix else key)
        else:
            flattened[prefix] = item

    visit(value, "")
    if include_derived and isinstance(config, RunConfig):
        training = config.training
        flattened["effective.tokens_per_update"] = (
            training.seq_len
            * training.micro_batch_size
            * training.gradient_accumulation
        )
        flattened["effective.target_token_exposures"] = (
            flattened["effective.tokens_per_update"] * training.max_steps
        )
        if isinstance(config.optimizer, AdamWConfig):
            flattened["effective.optimizer_schedule_kind"] = (
                "warmup_cosine_floor_v1"
                if config.optimizer.decay_steps is not None
                else "warmup_cosine_v1"
            )
            flattened["effective.optimizer_decay_horizon"] = (
                config.optimizer.decay_steps or training.max_steps
            )
    return flattened


def _owner(path: str) -> str:
    root = path.split(".", 1)[0]
    if root == "artifacts":
        kind = path.split(".", 2)[1] if "." in path else ""
        if "tokenizer" in kind:
            return "Tokenizer"
        if "evaluation" in kind or "card" in kind or "prompt" in kind:
            return "Evaluation"
        if "source" in kind or "implementation" in kind:
            return "Implementation"
        return "Data"
    if root == "effective":
        return (
            "Optimization"
            if "optimizer" in path or "token" in path
            else "Runtime/numerical"
        )
    return {
        "model": "Architecture",
        "attention": "Architecture",
        "dataset": "Data",
        "tokenizer": "Tokenizer",
        "training": "Optimization",
        "optimizer": "Optimization",
        "evaluation": "Evaluation",
        "runtime": "Runtime/numerical",
        "seed": "Runtime/numerical",
        "checkpoint": "Operations/retention",
        "logging": "Operations/retention",
        "staging": "Operations/retention",
    }.get(root, "Implementation")


def compare_configs(
    base: RunConfig | Mapping[str, object],
    variant: RunConfig | Mapping[str, object],
    *,
    expected: Collection[str],
    invariants: Collection[str] = (),
    artifacts: Mapping[str, object] | None = None,
    require_all: bool = True,
    include_derived: bool = False,
) -> dict[str, dict[str, object]]:
    """Assert exact declared interventions and invariant equality on actual leaf values.

    ``artifacts`` may contain ``base``/``variant`` mappings of typed identities.
    Legacy category comparisons use ``require_all=False`` to permit subsets of a
    registered category while still rejecting every out-of-category change.
    """
    left = flatten_config(base, include_derived=include_derived)
    right = flatten_config(variant, include_derived=include_derived)
    if artifacts is not None:
        for side, target in (("base", left), ("variant", right)):
            inventory = artifacts.get(side, {})
            if not isinstance(inventory, Mapping):
                raise TypeError(f"{side} artifacts must be a mapping")
            for key, value in flatten_config({"artifacts": inventory}).items():
                target[key] = value
    paths = set(left) | set(right)
    expected_set, invariant_set = set(expected), set(invariants)
    for declaration, kind in (
        (expected_set, "intervention"),
        (invariant_set, "invariant"),
    ):
        for dotted in declaration:
            if not isinstance(dotted, str) or not dotted or dotted not in paths:
                raise ValueError(f"unknown {kind} field: {dotted!r}")
    for dotted in expected_set:
        for invariant in invariant_set:
            if paths_overlap(dotted, invariant):
                raise ValueError(
                    f"intervention {dotted} conflicts with invariant {invariant}"
                )
    differences = {
        key: {"base": left.get(key), "variant": right.get(key)}
        for key in sorted(paths)
        if left.get(key) != right.get(key)
    }
    violations = sorted(invariant_set & differences.keys())
    if violations:
        raise ValueError(f"invariant changed: {', '.join(violations)}")
    hidden = sorted(set(differences) - expected_set)
    if hidden:
        raise ValueError(f"undeclared changed fields: {', '.join(hidden)}")
    missing = sorted(expected_set - set(differences)) if require_all else []
    if missing:
        raise ValueError(f"declared interventions unchanged: {', '.join(missing)}")
    return differences


def describe_differences(
    differences: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, Any], ...]:
    """Attach stable field ownership without changing legacy difference payloads."""
    return tuple(
        {"path": path, "owner": _owner(path), **values}
        for path, values in differences.items()
    )
