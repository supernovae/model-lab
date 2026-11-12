"""Typed, exclusive authoring handoffs; receipts are operational provenance only."""

from __future__ import annotations

import hashlib
import json
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
from klab.result_publication import write_bytes_exclusive
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
    write_bytes_exclusive(sidecar, binding)
    try:
        write_bytes_exclusive(output, body)
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


def _merged(base: Any, overlay: Any) -> Any:
    """Overlay one mapping on another; mappings merge, any other value replaces."""
    if isinstance(base, Mapping) and isinstance(overlay, Mapping):
        merged = dict(base)
        for key, value in overlay.items():
            merged[key] = _merged(base[key], value) if key in base else value
        return merged
    return overlay


def parse_parts(values: Sequence[str]) -> list[tuple[str | None, Path]]:
    """Read ``PATH`` or ``FIELD=PATH`` arguments for :func:`compose_config`."""
    parts: list[tuple[str | None, Path]] = []
    for value in values:
        field, separator, path = value.partition("=")
        if not separator:
            parts.append((None, Path(value)))
        elif not is_dotted_path(field) or not path:
            raise ValueError(f"expected PATH or FIELD=PATH: {value!r}")
        else:
            parts.append((field, Path(path)))
    return parts


def compose_config(
    parts: Sequence[tuple[str | None, Path]], output: Path
) -> dict[str, Any]:
    """Author one run config from several declaration files.

    Each part is a YAML/JSON mapping, merged in order with later parts winning.
    A part given with a field is placed at that dotted field, so a model
    section, an optimizer section and a file of local paths compose without a
    script. Relative paths resolve from the part that declares them. The result
    must be a complete, valid run config; nothing is read or executed.
    """
    if not parts:
        raise ValueError("at least one part is required")
    output = _destination(output)
    origins: list[dict[str, str]] = []
    document: Any = {}
    for field, path in parts:
        path = Path(path).absolute()
        origins.append({**_snapshot(path), **({"field": field} if field else {})})
        part: Any = read_document(path)
        for name in reversed(field.split(".")) if field else ():
            part = {name: part}
        document = _merged(document, _resolve_paths(part, path.parent))
    candidate = _config(document, output.parent)
    body = _body(candidate)
    receipt = {
        "format": "klab-composition-v1",
        "parts": origins,
        "output": {
            "path": str(output),
            "file_sha256": hashlib.sha256(body).hexdigest(),
        },
        "validation": {
            "scope": "typed_config",
            "executable": False,
            "runtime_verified": False,
        },
    }
    snapshots = [
        {key: value for key, value in origin.items() if key != "field"}
        for origin in origins
    ]
    _publish(output, body, receipt, snapshots)
    return receipt
