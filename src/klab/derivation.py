"""Typed, exclusive authoring handoffs; receipts are operational provenance only."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from klab.canonical import canonical_json, reject_duplicate_keys
from klab.config.loading import _resolve_paths, load_config
from klab.config.models import RunConfig
from klab.documents import _reject_constant, read_document
from klab.experiments.compiler import (
    apply_patch,
    flatten_config,
    is_dotted_path,
    patchable_config,
)
from klab.experiments.lock import _exclusive_bytes
from klab.training.manifest import config_sha256, sha256_file


def parse_assignments(values: Sequence[str]) -> dict[str, Any]:
    """Decode duplicate-free dotted assignments using strict finite JSON values."""
    result: dict[str, Any] = {}
    for assignment in values:
        field, separator, text = assignment.partition("=")
        if not separator or not is_dotted_path(field):
            raise ValueError(
                f"expected FIELD=JSON with a valid dotted path: {assignment!r}"
            )
        if field in result:
            raise ValueError(f"duplicate assignment: {field}")
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
        canonical_json(value)
        result[field] = value
    _settings(result)
    return result


def _settings(settings: Mapping[str, Any]) -> None:
    if not isinstance(settings, Mapping) or not settings:
        raise ValueError("at least one assignment is required")
    for field in settings:
        if not is_dotted_path(field):
            raise ValueError(f"invalid assignment path: {field!r}")
    canonical_json(dict(settings))


def _json_projection(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _json_projection(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_projection(item) for item in value]
    return value


def _anchor_paths(value: Any, base: Path) -> Any:
    """Anchor typed Path defaults too, without dropping projected model defaults."""
    if isinstance(value, Path):
        return value if value.is_absolute() else (base / value).resolve()
    if isinstance(value, Mapping):
        return {key: _anchor_paths(item, base) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_anchor_paths(item, base) for item in value]
    return value


def _config(value: Any, base: Path) -> RunConfig:
    validated = RunConfig.model_validate(_resolve_paths(_json_projection(value), base))
    return RunConfig.model_validate(_anchor_paths(patchable_config(validated), base))


def _delta(base: Any, variant: Any) -> dict[str, Any]:
    before = flatten_config(_json_projection(patchable_config(base)))
    after = flatten_config(_json_projection(patchable_config(variant)))
    return {
        field: {"base": before.get(field), "variant": after.get(field)}
        for field in sorted(before.keys() | after.keys())
        if before.get(field) != after.get(field)
    }


def _destination(output: Path) -> Path:
    output = Path(output).absolute()
    if output.suffix not in {".yaml", ".yml"}:
        raise ValueError("derivation output must have a .yaml or .yml suffix")
    for component in (output, *output.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked output path: {component}")
    if not output.parent.is_dir():
        raise ValueError(
            f"output parent must be an existing directory: {output.parent}"
        )
    for path in (output, receipt_path(output)):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"derivation destination already exists: {path}")
    return output


def receipt_path(output: Path) -> Path:
    return output.with_name(output.name + ".derivation.json")


def _snapshot(path: Path) -> dict[str, str]:
    path = Path(path).absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked source path: {component}")
    return {"path": str(path), "file_sha256": sha256_file(path)}


def _unchanged(records: Sequence[dict[str, str]]) -> None:
    for record in records:
        if _snapshot(Path(record["path"])) != record:
            raise ValueError(f"source changed during derivation: {record['path']}")


def _body(value: Any) -> bytes:
    return yaml.safe_dump(value.model_dump(mode="json"), sort_keys=False).encode(
        "utf-8"
    )


def _receipt(
    kind: str,
    origin: dict[str, str],
    output: Path,
    body: bytes,
    settings: Mapping[str, Any],
    base: Any,
    candidate: Any,
    rebindings: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format": "klab-derivation-v1",
        "kind": kind,
        "origin": origin,
        "output": {
            "path": str(output),
            "file_sha256": hashlib.sha256(body).hexdigest(),
        },
        "declared_delta": dict(settings),
        "delta": _delta(base, candidate),
        "path_rebindings": sorted(rebindings, key=lambda item: item["path"]),
        "validation": {
            "scope": "typed_config" if kind == "run_config" else "resolved_experiment",
            "executable": False,
            "runtime_verified": False,
        },
    }


def _publish(
    output: Path,
    body: bytes,
    receipt: dict[str, Any],
    origins: Sequence[dict[str, str]],
) -> None:
    binding = canonical_json(receipt) + b"\n"
    _unchanged(origins)
    _destination(output)
    sidecar = receipt_path(output)
    _exclusive_bytes(sidecar, binding)
    try:
        _exclusive_bytes(output, body)
    except BaseException:
        sidecar.unlink(missing_ok=True)
        raise


def derive_config(
    source: Path, output: Path, settings: Mapping[str, Any]
) -> dict[str, Any]:
    """Author a new schema-verified config, not a verified checkpoint continuation."""
    _settings(settings)
    output = _destination(output)
    source = Path(source).absolute()
    origin = _snapshot(source)
    authored = read_document(source)
    loaded = load_config(source)
    base = _config(patchable_config(loaded), source.parent)
    candidate = _config(apply_patch(base, settings), source.parent)
    body = _body(candidate)
    # Validate serializer/default behavior at the actual destination anchor.
    reopened = _config(yaml.safe_load(body), output.parent)
    if patchable_config(reopened) != patchable_config(candidate):
        raise ValueError("serialized config changes when reloaded at output")
    requested = apply_patch(patchable_config(loaded), settings)
    original = flatten_config(_json_projection(requested))
    for field, value in flatten_config(authored).items():
        if not any(
            field == patch
            or field.startswith(patch + ".")
            or patch.startswith(field + ".")
            for patch in settings
        ):
            original[field] = value
    normalized = flatten_config(patchable_config(candidate))
    rebindings = [
        {
            "path": field,
            "source_value": original.get(field),
            "output_value": str(value),
            "resolved_target": str(value),
        }
        for field, value in normalized.items()
        if isinstance(value, Path) and original.get(field) != str(value)
    ]
    receipt = _receipt(
        "run_config", origin, output, body, settings, base, candidate, rebindings
    )
    receipt["config_sha256"] = config_sha256(candidate.model_dump(mode="json"))
    _publish(output, body, receipt, [origin])
    return receipt


def _materialized_base(value: Any, source: Path) -> RunConfig:
    from klab.recovery.provenance import declaration_reference

    if isinstance(value, str):
        target = declaration_reference(source, value)
        read_document(target)
        return _config(patchable_config(load_config(target)), target.parent)
    return _config(patchable_config(value), source.parent)


def _reference(
    value: str, source: Path, output: Path, field: str, *, artifact: bool = False
) -> tuple[str, Path]:
    from klab.experiments.artifacts import _safe_path
    from klab.recovery.provenance import declaration_reference, repository_root

    target = declaration_reference(source, value)
    _snapshot(target)
    if artifact and _safe_path(value, source) != target:
        raise ValueError(f"{field}: artifact and declaration targets disagree")
    candidates: list[str] = []
    if target.is_relative_to(output.parent):
        candidates.append(target.relative_to(output.parent).as_posix())
    root = repository_root(output)
    if root is not None and target.is_relative_to(root):
        candidates.append(target.relative_to(root).as_posix())
    for reference in candidates:
        if declaration_reference(output, reference) != target:
            continue
        if artifact:
            try:
                if _safe_path(reference, output) != target:
                    continue
            except ValueError:
                continue
        return reference, target
    raise ValueError(
        f"{field}: cannot relocate declaration reference preserving {target}"
    )


def _normalize_plan(
    view: Mapping[str, Any],
    source: Path,
    output: Path,
    rebindings: list[dict[str, Any]],
) -> Any:
    from klab.experiments.artifacts import _safe_path
    from klab.experiments.plan import ExperimentPlan

    value = _json_projection(patchable_config(view))
    base_before = value["base_run"]
    materialized = _materialized_base(base_before, source)
    value["base_run"] = _json_projection(patchable_config(materialized))
    value = _json_projection(patchable_config(ExperimentPlan.model_validate(value)))

    def remember(field: str, old: Any, new: Any, target: Path) -> None:
        if old != new:
            rebindings.append(
                {
                    "path": field,
                    "source_value": old,
                    "output_value": new,
                    "resolved_target": str(target),
                }
            )

    if isinstance(base_before, Mapping):
        before = flatten_config(base_before)
        for field, target in flatten_config(patchable_config(materialized)).items():
            if isinstance(target, Path):
                remember(f"base_run.{field}", before.get(field), str(target), target)

    def reference(
        container: dict[str, Any], key: str, field: str, *, artifact: bool = False
    ) -> None:
        old = container.get(key)
        if old is not None:
            new, target = _reference(old, source, output, field, artifact=artifact)
            container[key] = new
            remember(field, old, new, target)

    reference(value, "evaluation_suite", "evaluation_suite")
    for index, variant in enumerate(value["corpus_variants"]):
        reference(variant, "project", f"corpus_variants.{index}.project")
    for index, evaluation in enumerate(value["evaluations"]):
        reference(evaluation, "cases", f"evaluations.{index}.cases")
    for name, artifact in value["artifacts"].items():
        if artifact["from_phase"] is not None:
            continue
        field = f"artifacts.{name}.path"
        if artifact["kind"] in {"capability_card", "prompt_set"}:
            reference(artifact, "path", field, artifact=True)
        else:
            old = artifact["path"]
            target = _safe_path(old, source)
            artifact["path"] = str(target)
            remember(field, old, str(target), target)

    def normalize_set(patch: dict[str, Any], prefix: str) -> None:
        for field, old in list(patch.items()):
            if field.startswith("inputs."):
                continue
            new = _json_projection(
                _resolve_paths(old, source.parent, field.split(".")[-1])
            )
            patch[field] = new
            before = flatten_config({field: old})
            after = flatten_config({field: new})
            for leaf, target in after.items():
                if before.get(leaf) != target:
                    remember(f"{prefix}.{leaf}", before.get(leaf), target, Path(target))

    for index, phase in enumerate(value["phases"]):
        normalize_set(phase["set"], f"phases.{index}.set")
    for index, axis in enumerate(value["axes"]):
        for choice_index, choice in enumerate(axis["choices"]):
            normalize_set(choice["set"], f"axes.{index}.choices.{choice_index}.set")
    return ExperimentPlan.model_validate(value)


def derive_experiment(
    source: Path,
    output: Path,
    settings: Mapping[str, Any],
    *,
    prepared: Path | None = None,
    max_runs: int = 1000,
) -> dict[str, Any]:
    """Cold-resolve a bound authored plan without locking or executing it."""
    from klab.experiments.lock import resolve_plan
    from klab.experiments.plan import load_plan
    from klab.recovery.provenance import declaration_paths, declaration_reference

    _settings(settings)
    output = _destination(output)
    source = Path(source).absolute()
    origin = _snapshot(source)
    declarations = [_snapshot(path) for path in declaration_paths(source, "experiment")]
    plan = load_plan(source)
    view = patchable_config(plan)
    original_base = view["base_run"]
    view["base_run"] = patchable_config(_materialized_base(original_base, source))
    patched = apply_patch(view, settings, source="experiment")
    rebindings: list[dict[str, Any]] = []
    baseline = _normalize_plan(view, source, output, [])
    candidate = _normalize_plan(patched, source, output, rebindings)
    if isinstance(original_base, Mapping):
        before = flatten_config(_json_projection(original_base))
        normalized = flatten_config(view["base_run"])
        after = flatten_config(_json_projection(patchable_config(candidate.base_run)))
        recorded = {row["path"] for row in rebindings}
        for field, target in normalized.items():
            name = f"base_run.{field}"
            if (
                isinstance(target, Path)
                and before.get(field) != str(target)
                and after.get(field) == str(target)
                and name not in recorded
            ):
                rebindings.append(
                    {
                        "path": name,
                        "source_value": before.get(field),
                        "output_value": str(target),
                        "resolved_target": str(target),
                    }
                )
    if isinstance(original_base, str):
        target = declaration_reference(source, original_base)
        rebindings.append(
            {
                "path": "base_run",
                "source_value": original_base,
                "output_value": _json_projection(patchable_config(candidate.base_run)),
                "resolved_target": str(target),
            }
        )
    preparation = None
    preparation_origin = None
    if prepared is not None:
        preparation_origin = _snapshot(Path(prepared))
        preparation = read_document(Path(prepared))
        if preparation.get("format") != "experiment-preparation-v1":
            raise ValueError("expected experiment-preparation-v1")
    body = _body(candidate)
    staging = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=".derivation-validation-",
            suffix=".yaml",
            delete=False,
        ) as stream:
            staging = Path(stream.name)
            stream.write(body)
        resolve_plan(
            load_plan(staging), staging, prepared=preparation, max_runs=max_runs
        )
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)
    receipt = _receipt(
        "experiment_plan",
        origin,
        output,
        body,
        settings,
        baseline,
        candidate,
        rebindings,
    )
    receipt["origin_declarations"] = declarations
    origins = [origin, *declarations]
    if preparation_origin is not None:
        receipt["preparation"] = preparation_origin
        origins.append(preparation_origin)
    _publish(output, body, receipt, origins)
    return receipt
