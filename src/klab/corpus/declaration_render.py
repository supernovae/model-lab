"""Bounded, data-only rendering of reviewed corpus declaration templates."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

from klab.canonical import canonical_digest, canonical_json
from klab.training.manifest import sha256_file

_SLOT = re.compile(rb"\$\{([A-Z][A-Z0-9_]*)\}")
_JSON_ARRAY_SLOT = re.compile(rb'"\$\{([A-Z][A-Z0-9_]*_JSON)\}"')
_MAX_BYTES = 64 * 1024 * 1024
_BASELINE_IDENTITY_SLOT = "VERIFIED_WORKSPACE_BASELINE_IDENTITY_SHA256"


def render_declaration(
    template: Path,
    values_json: str,
    output: Path,
    work_root: Path,
    *,
    workspace_baseline: Path | None = None,
) -> dict[str, object]:
    """Fill exact named slots and publish one immutable JSON/YAML data file.

    This operation executes no input, cannot overwrite evidence, and deliberately
    leaves scientific validation to the native consumer of the rendered file.
    """
    root = work_root.resolve(strict=True)
    destination = output.absolute()
    if (
        template.is_symlink()
        or not template.is_file()
        or output.is_symlink()
        or destination.exists()
        or not destination.is_relative_to(root)
        or not destination.parent.resolve().is_relative_to(root)
        or any(parent.is_symlink() for parent in destination.parents)
        or template.suffix not in {".json", ".yaml"}
        or destination.suffix != template.suffix
    ):
        raise ValueError("unsafe declaration template or output")
    raw = template.read_bytes()
    if len(raw) > _MAX_BYTES:
        raise ValueError("declaration template exceeds 64 MiB")
    values = json.loads(values_json)
    if not isinstance(values, dict) or any(
        not isinstance(key, str)
        or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key)
        or not isinstance(value, str)
        or "${" in value
        for key, value in values.items()
    ):
        raise ValueError("invalid declaration bindings")
    slots = {name.decode("ascii") for name in _SLOT.findall(raw)}
    if workspace_baseline is not None:
        if (
            _BASELINE_IDENTITY_SLOT not in slots
            or _BASELINE_IDENTITY_SLOT in values
            or not workspace_baseline.is_absolute()
            or workspace_baseline.is_symlink()
        ):
            raise ValueError("invalid verified workspace baseline binding")
        from klab.operational_monitor import load_workspace_baseline

        values[_BASELINE_IDENTITY_SLOT] = load_workspace_baseline(
            workspace_baseline, root
        ).sha256
    elif _BASELINE_IDENTITY_SLOT in slots:
        raise ValueError("verified workspace baseline receipt is required")
    if set(values) != slots:
        raise ValueError("declaration bindings do not exactly cover template slots")

    def json_array(match: re.Match[bytes]) -> bytes:
        value = json.loads(values[match[1].decode("ascii")])
        if not isinstance(value, list):
            raise TypeError("reviewed JSON fragment must be an array")
        return canonical_json(value)

    rendered = _JSON_ARRAY_SLOT.sub(json_array, raw)
    rendered = _SLOT.sub(
        lambda match: values[match[1].decode("ascii")].encode(), rendered
    )
    if len(rendered) > _MAX_BYTES or b"${" in rendered:
        raise ValueError("rendered declaration exceeds limit or retains slots")
    parsed = (
        json.loads(rendered)
        if destination.suffix == ".json"
        else yaml.safe_load(rendered)
    )
    if not isinstance(parsed, dict):
        raise TypeError("rendered declaration must be a mapping")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.parent.resolve().is_relative_to(root) or any(
        parent.is_symlink() for parent in destination.parents
    ):
        raise ValueError("declaration output parent changed")
    with destination.open("xb") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "format": "klab-declaration-render-v1",
        "template_sha256": sha256_file(template),
        "bindings_sha256": canonical_digest(values),
        "path": str(destination),
        "sha256": sha256_file(destination),
        "bytes": len(rendered),
    }
