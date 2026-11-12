"""The ``mlx_native/v1`` state codec used by the common checkpoint manager.

This module deliberately has no MLX import.  It validates and restores the portable
native representation (NumPy arrays plus typed JSON); an MLX engine may convert the
loaded arrays to MLX arrays at its boundary.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from klab.canonical import canonical_json, reject_duplicate_keys
from klab.training.manifest import sha256_file

CODEC = "mlx_native"
CODEC_VERSION = 1
_STATE_FILE = "training_state.json"
_OPTIMIZER_FILE = "optimizer.safetensors"

_STATE_FIELDS = frozenset(
    {
        "format_version",
        "engine",
        "backend",
        "step",
        "tokens_seen",
        "cursor",
        "config",
        "run_id",
        "schedule",
        "cadence",
        "config_sha256",
        "architecture_sha256",
        "source_identity_sha256",
        "manifest_sha256",
        "parent_checkpoint_sha256",
        "cumulative_wall_seconds",
        "cumulative_update_seconds",
        "lineage_best",
        "state_codec",
        "state_codec_version",
        "optimizer",
        "rng",
        "scaler",
        "optimizer_parameter_names",
    }
)


def _no_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def strict_json(path: Path) -> dict[str, Any]:
    """Read a JSON object while rejecting duplicate keys and non-finite values."""

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=_no_constant,
    )
    if not isinstance(value, dict):
        raise TypeError("JSON root must be an object")
    return value


def _finite_scalar(value: object) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _flatten(value: object, arrays: dict[str, object], path: str) -> object:
    if isinstance(value, np.ndarray) or (
        value.__class__.__module__.startswith("mlx.")
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
    ):
        if isinstance(value, np.ndarray) and (
            not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value))
        ):
            raise ValueError(f"non-finite or non-numeric native array: {path}")
        key = f"a{len(arrays):08d}"
        arrays[key] = (
            value
            if isinstance(value, np.ndarray) and value.ndim == 0
            else np.ascontiguousarray(value)
            if isinstance(value, np.ndarray)
            else value
        )
        return {
            "type": "array",
            "key": key,
            "dtype": str(value.dtype).removeprefix("mlx.core."),
            "shape": list(value.shape),
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"native mapping keys must be strings: {path}")
        return {
            "type": "dict",
            "items": {
                key: _flatten(item, arrays, f"{path}.{key}")
                for key, item in sorted(value.items())
            },
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [
                _flatten(item, arrays, f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [
                _flatten(item, arrays, f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if (
        value is None
        or isinstance(value, str)
        or type(value) is bool
        or _finite_scalar(value)
    ):
        return {"type": "scalar", "value": value}
    raise TypeError(f"unsupported native value at {path}: {type(value).__name__}")


def _descriptor_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        type(dimension) is not int or dimension < 0 for dimension in value
    ):
        raise ValueError("invalid native array dimensions")
    return tuple(value)


def _unflatten(node: object, arrays: Mapping[str, np.ndarray]) -> object:
    if not isinstance(node, Mapping) or not isinstance(node.get("type"), str):
        raise TypeError("invalid typed native tree node")
    kind = node["type"]
    if kind == "scalar":
        if set(node) != {"type", "value"} or not (
            node["value"] is None
            or isinstance(node["value"], str)
            or type(node["value"]) is bool
            or _finite_scalar(node["value"])
        ):
            raise ValueError("invalid native scalar")
        return node["value"]
    if kind == "array":
        if (
            set(node) != {"type", "key", "dtype", "shape"}
            or not isinstance(node.get("key"), str)
            or not node["key"]
            or not isinstance(node.get("dtype"), str)
        ):
            raise ValueError("invalid native array descriptor")
        shape = _descriptor_shape(node.get("shape"))
        array = arrays.get(node["key"])
        if (
            array is None
            or str(array.dtype) != node["dtype"]
            or array.shape != shape
            or not np.issubdtype(array.dtype, np.number)
            or not np.all(np.isfinite(array))
        ):
            raise ValueError("native array descriptor does not match safetensors")
        return array
    if kind in {"dict", "tuple", "list"}:
        if set(node) != {"type", "items"}:
            raise ValueError("invalid native collection descriptor")
        items = node["items"]
        if kind == "dict":
            if not isinstance(items, Mapping) or not all(
                isinstance(key, str) for key in items
            ):
                raise ValueError("invalid native dictionary")
            return {key: _unflatten(value, arrays) for key, value in items.items()}
        if not isinstance(items, list):
            raise ValueError("invalid native sequence")
        values = [_unflatten(value, arrays) for value in items]
        return tuple(values) if kind == "tuple" else values
    raise ValueError("unknown native tree node type")


def _array_keys(node: object, seen: set[str] | None = None) -> set[str]:
    """Return tree references while rejecting aliasing within serialized state."""
    if seen is None:
        seen = set()
    if not isinstance(node, Mapping):
        raise TypeError("invalid typed native tree node")
    kind = node.get("type")
    if kind == "array":
        if set(node) != {"type", "key", "dtype", "shape"}:
            raise ValueError("invalid native array descriptor")
        key = node.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("invalid native array key")
        _descriptor_shape(node.get("shape"))
        if key in seen:
            raise ValueError("native tree references an array more than once")
        seen.add(key)
        return seen
    if kind == "scalar":
        if set(node) != {"type", "value"}:
            raise ValueError("invalid native scalar")
        return seen
    items = node.get("items")
    if kind == "dict":
        if set(node) != {"type", "items"} or not isinstance(items, Mapping):
            raise ValueError("invalid native dictionary")
        for value in items.values():
            _array_keys(value, seen)
        return seen
    if kind in {"list", "tuple"}:
        if set(node) != {"type", "items"} or not isinstance(items, list):
            raise ValueError("invalid native sequence")
        for value in items:
            _array_keys(value, seen)
        return seen
    raise ValueError("unknown native tree node type")


def validate_rng(rng: object) -> None:
    """Validate the public MLX RNG envelope without touching process-global RNG."""
    fields = {
        "python",
        "numpy_kind",
        "numpy_keys",
        "numpy_pos",
        "numpy_has_gauss",
        "numpy_cached_gaussian",
        "mlx",
    }
    if not isinstance(rng, Mapping) or set(rng) != fields:
        raise ValueError("MLX RNG envelope has invalid fields")
    import random

    python_state = rng["python"]
    keys, mlx = rng["numpy_keys"], rng["mlx"]
    if (
        not isinstance(rng["numpy_kind"], str)
        or not isinstance(keys, np.ndarray)
        or keys.dtype != np.uint32
        or keys.shape != (624,)
        or type(rng["numpy_pos"]) is not int
        or not 0 <= rng["numpy_pos"] <= 624
        or type(rng["numpy_has_gauss"]) is not int
        or rng["numpy_has_gauss"] not in {0, 1}
        or not _finite_scalar(rng["numpy_cached_gaussian"])
        or not isinstance(mlx, np.ndarray)
        or mlx.dtype != np.uint32
        or mlx.shape != (2,)
    ):
        raise ValueError("MLX RNG envelope has invalid values")
    try:
        random.Random(0).setstate(python_state)
        np.random.RandomState(0).set_state(
            (
                rng["numpy_kind"],
                keys.copy(),
                rng["numpy_pos"],
                rng["numpy_has_gauss"],
                rng["numpy_cached_gaussian"],
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("MLX RNG envelope has invalid generator state") from error


def _json_safe_python_state(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_safe_python_state(item) for item in value]
    if (
        type(value) is int
        or type(value) is bool
        or value is None
        or isinstance(value, str)
    ):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("MLX Python RNG state is not JSON-safe")


def _python_state_from_json(value: object) -> object:
    if isinstance(value, list):
        return tuple(_python_state_from_json(item) for item in value)
    if (
        type(value) is int
        or type(value) is bool
        or value is None
        or isinstance(value, str)
    ):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("MLX Python RNG JSON state is invalid")


def _uint32_words(value: object, length: int, field: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(type(word) is not int or not 0 <= word < 2**32 for word in value)
    ):
        raise ValueError(f"MLX RNG {field} must be {length} uint32 words")
    return value


def _encode_rng(rng: Mapping[str, object]) -> dict[str, object]:
    validate_rng(rng)
    return {
        "python": _json_safe_python_state(rng["python"]),
        "numpy_kind": rng["numpy_kind"],
        "numpy_keys": [int(word) for word in np.asarray(rng["numpy_keys"])],
        "numpy_pos": rng["numpy_pos"],
        "numpy_has_gauss": rng["numpy_has_gauss"],
        "numpy_cached_gaussian": rng["numpy_cached_gaussian"],
        "mlx": [int(word) for word in np.asarray(rng["mlx"])],
    }


def _decode_rng(value: object) -> dict[str, object]:
    fields = {
        "python",
        "numpy_kind",
        "numpy_keys",
        "numpy_pos",
        "numpy_has_gauss",
        "numpy_cached_gaussian",
        "mlx",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("MLX RNG JSON envelope has invalid fields")
    result: dict[str, object] = {
        "python": _python_state_from_json(value["python"]),
        "numpy_kind": value["numpy_kind"],
        "numpy_keys": np.asarray(
            _uint32_words(value["numpy_keys"], 624, "numpy_keys"), dtype=np.uint32
        ),
        "numpy_pos": value["numpy_pos"],
        "numpy_has_gauss": value["numpy_has_gauss"],
        "numpy_cached_gaussian": value["numpy_cached_gaussian"],
        "mlx": np.asarray(_uint32_words(value["mlx"], 2, "mlx"), dtype=np.uint32),
    }
    validate_rng(result)
    return result


def validate_optimizer_state(
    optimizer: object,
    groups: object,
    shapes: Mapping[str, tuple[int, ...]],
    *,
    step: int | None = None,
    learning_rate: float | None = None,
) -> int:
    """Validate the exact two-group MLX AdamW state and return its step."""
    if (
        not isinstance(optimizer, Mapping)
        or set(optimizer) != {"state"}
        or not isinstance(optimizer["state"], Mapping)
    ):
        raise ValueError("MLX optimizer state must contain a flat state mapping")
    if not isinstance(shapes, Mapping):
        raise TypeError("MLX optimizer parameter shapes are invalid")
    if (
        not isinstance(groups, list)
        or len(groups) != 2
        or not all(isinstance(group, list) for group in groups)
        or not all(isinstance(name, str) and name for group in groups for name in group)
    ):
        raise ValueError("MLX optimizer parameter groups are invalid")
    names = [name for group in groups for name in group]
    if len(names) != len(set(names)) or set(names) != set(shapes):
        raise ValueError("MLX optimizer groups do not match trainable parameters")
    for name, shape in shapes.items():
        if (
            not isinstance(name, str)
            or not isinstance(shape, tuple)
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            raise ValueError("MLX optimizer parameter shapes are invalid")
    for group_index, group in enumerate(groups):
        if any((len(shapes[name]) >= 2) != (group_index == 0) for name in group):
            raise ValueError("MLX optimizer parameter is in the wrong decay group")

    state = optimizer["state"]
    if not all(
        isinstance(name, str) and name and isinstance(value, np.ndarray)
        for name, value in state.items()
    ):
        raise ValueError("MLX optimizer state must be a flat array mapping")
    metadata = {
        f"states.{group_index}.{field}"
        for group_index in range(2)
        for field in ("step", "learning_rate")
    }
    moments = {
        f"states.{group_index}.{name}.{moment}"
        for group_index, group in enumerate(groups)
        for name in group
        for moment in ("m", "v")
    }
    actual = set(state)
    if not metadata <= actual or actual not in (metadata, metadata | moments):
        raise ValueError("MLX optimizer state has incomplete or unknown inventory")
    for group_index in range(2):
        state_step = state[f"states.{group_index}.step"]
        state_rate = state[f"states.{group_index}.learning_rate"]
        if (
            state_step.dtype != np.uint64
            or state_step.shape != ()
            or state_rate.dtype != np.float32
            or state_rate.shape != ()
            or not np.isfinite(state_rate)
            or float(state_rate) < 0
        ):
            raise ValueError("MLX optimizer group metadata is invalid")
    completed_steps = [int(state[f"states.{index}.step"]) for index in range(2)]
    if completed_steps[0] != completed_steps[1]:
        raise ValueError("MLX optimizer groups have different completed steps")
    completed = completed_steps[0]
    if step is not None and (type(step) is not int or step < 0 or completed != step):
        raise ValueError("MLX optimizer completed step differs from checkpoint")
    rates = [float(state[f"states.{index}.learning_rate"]) for index in range(2)]
    if rates[0] != rates[1]:
        raise ValueError("MLX optimizer groups have different learning rates")
    if learning_rate is not None:
        if not isinstance(learning_rate, (int, float)) or not math.isfinite(
            learning_rate
        ):
            raise ValueError("checkpoint learning rate is invalid")
        expected_rate = float(np.float32(learning_rate))
        if rates[0] != expected_rate:
            raise ValueError("MLX optimizer learning rate differs from checkpoint")
    if actual == metadata:
        if completed != 0:
            raise ValueError("updated MLX optimizer state lacks AdamW moments")
        return completed
    for group_index, group in enumerate(groups):
        for name in group:
            for moment in ("m", "v"):
                value = state[f"states.{group_index}.{name}.{moment}"]
                if (
                    value.dtype != np.float32
                    or value.shape != shapes[name]
                    or not np.all(np.isfinite(value))
                    or (completed == 0 and np.any(value != 0))
                ):
                    raise ValueError("MLX AdamW moment state is invalid")
    return completed


def write_native_state(
    snapshot: object, directory: Path, common: Mapping[str, object]
) -> list[dict[str, object]]:
    """Write native optimizer state lazily, retaining MLX arrays until serialization."""
    optimizer, rng = snapshot.optimizer, snapshot.rng
    if not isinstance(optimizer, Mapping) or not isinstance(rng, Mapping):
        raise TypeError("MLX native state requires optimizer and RNG mappings")
    validate_rng(rng)
    arrays: dict[str, object] = {}
    scaler = _flatten(snapshot.scaler, {}, "scaler")
    names = _flatten(
        snapshot.optimizer_parameter_names, {}, "optimizer_parameter_names"
    )
    if _array_keys(scaler) or _array_keys(names):
        raise ValueError(
            "MLX scaler and optimizer parameter names cannot contain arrays"
        )
    state = {
        **dict(common),
        "state_codec": CODEC,
        "state_codec_version": CODEC_VERSION,
        "optimizer": _flatten(dict(optimizer), arrays, "optimizer"),
        "rng": _encode_rng(rng),
        "scaler": scaler,
        "optimizer_parameter_names": names,
    }
    if any(not isinstance(value, np.ndarray) for value in arrays.values()):
        import mlx.core as mx

        serializable = {
            key: value if not isinstance(value, np.ndarray) else mx.array(value)
            for key, value in arrays.items()
        }
        mx.save_safetensors(str(directory / _OPTIMIZER_FILE), serializable)
        mx.eval(serializable)
    else:
        save_file(arrays, directory / _OPTIMIZER_FILE)
    (directory / _STATE_FILE).write_bytes(canonical_json(state) + b"\n")
    for name in (_STATE_FILE, _OPTIMIZER_FILE):
        with (directory / name).open("rb") as handle:
            os.fsync(handle.fileno())
    return [
        {
            "name": name,
            "sha256": sha256_file(directory / name),
            "bytes": (directory / name).stat().st_size,
        }
        for name in (_STATE_FILE, _OPTIMIZER_FILE)
    ]


def load_native_state(directory: Path) -> dict[str, object]:
    """Load verified NumPy native state; conversion to MLX belongs to the engine."""
    state = strict_json(directory / _STATE_FILE)
    arrays = load_file(directory / _OPTIMIZER_FILE)
    if (
        set(state) != _STATE_FIELDS
        or state.get("state_codec") != CODEC
        or type(state.get("state_codec_version")) is not int
        or state.get("state_codec_version") != CODEC_VERSION
    ):
        raise ValueError("unsupported MLX native codec")
    references = _array_keys(state["optimizer"])
    if _array_keys(state["scaler"]) or _array_keys(state["optimizer_parameter_names"]):
        raise ValueError("only MLX optimizer arrays may be stored in safetensors")
    if set(arrays) != references:
        raise ValueError("native optimizer safetensors inventory differs from tree")
    result = {
        "optimizer": _unflatten(state["optimizer"], arrays),
        "rng": _decode_rng(state["rng"]),
        "scaler": _unflatten(state["scaler"], arrays),
        "optimizer_parameter_names": _unflatten(
            state["optimizer_parameter_names"], arrays
        ),
    }
    validate_rng(result["rng"])
    return result
