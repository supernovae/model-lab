"""Data-only YAML/JSON decoding, independent of schemas, paths and execution.

Extracted from the experiment reader; its diagnostic prefix is retained for
existing callers. This module never binds path fields or validates a schema.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml


class _UniqueLoader(yaml.SafeLoader):
    pass


class _ExactUniqueLoader(_UniqueLoader):
    """Use the same strict YAML reader with exact authored floating-point values."""


def _exact_yaml_float(loader: _ExactUniqueLoader, node: yaml.ScalarNode) -> Decimal:
    text = loader.construct_scalar(node).replace("_", "")
    if text.lower().lstrip("+-") in {".nan", ".inf"}:
        raise ValueError("non-finite YAML number")
    try:
        if ":" in text:
            sign = -1 if text.startswith("-") else 1
            parts = text.lstrip("+-").split(":")
            result = Decimal(0)
            for part in parts:
                result = result * 60 + Decimal(part)
            return sign * result
        return Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"invalid YAML number: {text}") from error


_ExactUniqueLoader.add_constructor("tag:yaml.org,2002:float", _exact_yaml_float)


def _unique_mapping(
    loader: _UniqueLoader, node: yaml.MappingNode
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        mark = key_node.start_mark
        location = f"line {mark.line + 1}, column {mark.column + 1}"
        if not isinstance(key, str):
            raise TypeError(f"experiment mapping keys must be strings at {location}")
        if key in result:
            raise ValueError(f"duplicate YAML key: {key!r} at {location}")
        result[key] = loader.construct_object(value_node)
    return result


for _loader in (_UniqueLoader, _ExactUniqueLoader):
    _loader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
    )


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def read_document(path: Path, *, exact_decimals: bool = False) -> dict[str, Any]:
    """Read strict duplicate-free YAML/JSON, without schema or path binding."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"invalid experiment document {path}: {error}") from error
    return parse_document(text, source=path, exact_decimals=exact_decimals)


def parse_document(
    text: str, *, source: Path, exact_decimals: bool = False
) -> dict[str, Any]:
    """Read strict duplicate-free YAML/JSON; optionally retain decimal number precision."""
    path = source
    try:
        raw = (
            json.loads(
                text,
                object_pairs_hook=_unique_json_pairs,
                parse_constant=_reject_constant,
                **({"parse_float": Decimal} if exact_decimals else {}),
            )
            if path.suffix.lower() == ".json"
            else yaml.load(
                text, Loader=_ExactUniqueLoader if exact_decimals else _UniqueLoader
            )
        )
    except TypeError as error:
        raise TypeError(f"invalid experiment document {path}: {error}") from error
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"invalid experiment document {path}: {error}") from error
    if not isinstance(raw, dict):
        raise TypeError(f"experiment document {path} must be a mapping")

    active: set[int] = set()
    checked: set[int] = set()

    def finite(value: object, field: str = "$") -> None:
        if (isinstance(value, float) and not math.isfinite(value)) or (
            isinstance(value, Decimal) and not value.is_finite()
        ):
            raise ValueError(
                f"non-finite value in experiment document {path} at {field}"
            )
        if isinstance(value, (dict, list)):
            identity = id(value)
            if identity in active:
                raise ValueError(
                    f"cyclic value in experiment document {path} at {field}"
                )
            if identity in checked:
                return
            active.add(identity)
            if isinstance(value, dict):
                for key, nested in value.items():
                    finite(nested, f"{field}.{key}")
            else:
                for index, nested in enumerate(value):
                    finite(nested, f"{field}[{index}]")
            active.remove(identity)
            checked.add(identity)

    finite(raw)
    return raw
