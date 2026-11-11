"""Version 1 bitwise training-state fingerprints (not a runtime authorization).

Only run identity, file/source provenance, config paths, cumulative timing and
derived exposure report metadata are omitted. The trainer always saves a ``minutes`` checkpoint watermark even when
``checkpoint.every_minutes`` is disabled; in that case alone it is operational
timing and omitted. Enabled time-based cadence has no common clock policy and
fails closed. Input snapshots must be captured at the same update boundary.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import numpy as np
import torch

from klab.training.checkpoints import LineageBest, TrainingSnapshot

VERSION = 1
_OBSERVATION_FIELDS = frozenset(
    {
        "step",
        "tokens_seen",
        "learning_rate",
        "loss",
        "gradient_norm",
        "overflow",
        "retry_state",
    }
)
_OPERATIONAL_SNAPSHOT_FIELDS = frozenset(
    {
        "run_id",
        "config_sha256",
        "architecture_sha256",
        "source_identity_sha256",
        "manifest_sha256",
        "parent_checkpoint_sha256",
        "checkpoint_sha256",
        "cumulative_wall_seconds",
        "cumulative_update_seconds",
        "weight_source",
        # Exposure repeats committed counters and adds provenance/timing; it
        # cannot affect the next update. The operative counters remain hashed.
        "exposure",
    }
)
_STATE_FIELDS = frozenset(
    {
        "model",
        "optimizer",
        "schedule",
        "step",
        "tokens_seen",
        "cursor",
        "config",
        "rng",
        "scaler",
        "validation_loss",
        "cadence",
        "engine",
        "backend",
        "optimizer_parameter_names",
        "tensor_trainability",
        "lineage_best",
    }
)
# Explicit locations of operational paths in RunConfig; all other config values
# (including scheduling and training limits) remain bitwise semantic state.
_CONFIG_PATHS = frozenset(
    {
        ("model", "memory_package_path"),
        ("tokenizer", "path"),
        ("dataset", "cache_dir"),
        ("dataset", "train_path"),
        ("dataset", "validation_path"),
        ("dataset", "source_manifest_path"),
        ("dataset", "allocation_manifest_path"),
        ("dataset", "corpus_release_path"),
        ("dataset", "corpus_export_path"),
        ("training", "portability_manifest_path"),
        ("logging", "root_dir"),
    }
)


def _part(hasher: Any, data: bytes | memoryview) -> None:
    hasher.update(len(data).to_bytes(8, "big"))
    hasher.update(data)


def _tensor(hasher: Any, value: torch.Tensor | np.ndarray) -> None:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise TypeError("only dense strided tensors have a canonical byte codec")
        raw = value.detach()
        if raw.device.type != "cpu":
            raw = raw.cpu()
        if not raw.is_contiguous():
            raw = raw.contiguous()
        # uint8 view also supports bfloat16 and torch dtypes without NumPy codecs.
        array = (
            raw.reshape(1).view(torch.uint8).numpy()
            if raw.ndim == 0
            else raw.view(torch.uint8).numpy()
        )
        dtype = str(raw.dtype)
        shape = tuple(raw.shape)
    else:
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise TypeError("object and structured arrays have no canonical byte codec")
        if not value.dtype.isnative:
            raise TypeError(
                "non-native-endian arrays have no torch-compatible byte codec"
            )
        array = value if value.flags.c_contiguous else np.ascontiguousarray(value)
        try:
            dtype = str(torch.from_numpy(array).dtype)
        except (TypeError, ValueError, RuntimeError) as error:
            raise TypeError("unsupported NumPy training tensor dtype") from error
        shape = value.shape
    _part(hasher, b"tensor")
    _part(hasher, dtype.encode("ascii"))
    _encode(hasher, shape)
    _part(hasher, memoryview(array).cast("B"))


def _encode(hasher: Any, value: object) -> None:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        _tensor(hasher, value)
    elif value is None:
        _part(hasher, b"none")
    elif type(value) is bool:
        _part(hasher, b"bool:1" if value else b"bool:0")
    elif type(value) is int:
        _part(hasher, b"int")
        _part(hasher, str(value).encode("ascii"))
    elif type(value) is float:
        _part(hasher, b"float64")
        _part(hasher, struct.pack(">d", value))
    elif type(value) is str:
        _part(hasher, b"str")
        _part(hasher, value.encode("utf-8"))
    elif type(value) is bytes:
        _part(hasher, b"bytes")
        _part(hasher, value)
    elif type(value) in (list, tuple):
        _part(hasher, b"list" if type(value) is list else b"tuple")
        _part(hasher, len(value).to_bytes(8, "big"))
        for item in value:
            _encode(hasher, item)
    elif type(value) is dict:
        _part(hasher, b"dict")
        keys = []
        for key in value:
            if type(key) not in (str, int, bool, float, bytes, tuple):
                raise TypeError(f"unsupported mapping key type: {type(key).__name__}")
            digest = hashlib.sha256()
            _encode(digest, key)
            keys.append((digest.digest(), key))
        keys.sort(key=lambda pair: pair[0])
        _part(hasher, len(keys).to_bytes(8, "big"))
        for _, key in keys:
            _encode(hasher, key)
            _encode(hasher, value[key])
    else:
        raise TypeError(f"unsupported training state type: {type(value).__name__}")


def _digest(value: object) -> str:
    digest = hashlib.sha256()
    _encode(digest, value)
    return digest.hexdigest()


def _semantic_config(config: object) -> dict[str, object]:
    if type(config) is not dict:
        raise TypeError("snapshot.config must be a dictionary")

    def clean(value: object, path: tuple[str, ...]) -> object:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise TypeError("config keys must be strings")
            return {
                key: clean(item, (*path, key))
                for key, item in value.items()
                if (*path, key) not in _CONFIG_PATHS
            }
        if type(value) in (list, tuple):
            return type(value)(clean(item, path) for item in value)
        return value

    return clean(config, ())  # type: ignore[return-value]


def _model(snapshot: TrainingSnapshot) -> dict[str, object]:
    names = snapshot.optimizer_parameter_names
    if type(names) is dict:
        if any(
            type(key) not in (int, str) or type(value) is not str
            for key, value in names.items()
        ):
            raise TypeError("invalid optimizer parameter identity")
        trainable = set(names.values())
    elif type(names) is list:
        if any(
            type(group) is not list or any(type(name) is not str for name in group)
            for group in names
        ):
            raise TypeError("invalid optimizer parameter identity")
        trainable = {name for group in names for name in group}
    elif names is None:
        trainable = set()
    else:
        raise TypeError("unsupported optimizer parameter identity")
    if snapshot.weight_source is not None and snapshot.model:
        raise ValueError("model and weight_source are mutually exclusive")
    if snapshot.weight_source is None:
        if not snapshot.model:
            raise ValueError("training snapshot has no model weights")
        tensors = snapshot.model
        explicit = snapshot.tensor_trainability
        if explicit is not None and set(explicit) != set(tensors):
            raise ValueError("tensor_trainability must cover exactly the model tensors")
        result: dict[str, object] = {}
        for name, tensor in tensors.items():
            if type(name) is not str or not isinstance(
                tensor, (torch.Tensor, np.ndarray)
            ):
                raise TypeError("model requires named CPU torch/NumPy tensors")
            # Native resume tensors have requires_grad=False, so no trainability may
            # be inferred from the loaded tensor. Optimizer membership is stable.
            is_trainable = explicit[name] if explicit is not None else name in trainable
            if type(is_trainable) is not bool:
                raise TypeError("model trainability flags must be bool")
            result[name] = (_digest(tensor), is_trainable)
        return result
    source = snapshot.weight_source
    canonical: dict[str, object] = {}
    for item in source.tensors():
        if (
            type(item.name) is not str
            or not item.name
            or item.name in canonical
            or not isinstance(item.array, np.ndarray)
            or not item.array.flags.c_contiguous
            or type(item.trainable) is not bool
        ):
            raise ValueError("invalid canonical weight source tensor")
        canonical[item.name] = (_digest(item.array), item.trainable)
    aliases = source.aliases
    if not isinstance(aliases, Mapping):
        raise TypeError("weight source aliases must be a mapping")
    if set(aliases) & set(canonical) or any(
        type(alias) is not str or type(target) is not str or target not in canonical
        for alias, target in aliases.items()
    ):
        raise ValueError("invalid canonical weight aliases")
    canonical.update({alias: canonical[target] for alias, target in aliases.items()})
    if not canonical:
        raise ValueError("training snapshot has no model weights")
    if snapshot.tensor_trainability is not None:
        if set(snapshot.tensor_trainability) != set(canonical):
            raise ValueError("tensor_trainability must cover exactly the model tensors")
        canonical = {
            name: (entry[0], snapshot.tensor_trainability[name])
            for name, entry in canonical.items()
        }
    return canonical


def canonical_training_state(snapshot: TrainingSnapshot, *, observation: dict) -> dict:
    """Return compact JSON-safe v1 component digests; reject unknown/unsafe state.

    ``observation`` is the actual committed update boundary: step, tokens_seen,
    learning_rate, loss, gradient_norm, overflow and retry_state. Optional loss/
    gradient_norm may be None when unavailable, but keys must still be supplied;
    callers must not fabricate values. Retry state is a typed nested mapping.
    """
    if type(snapshot) is not TrainingSnapshot:
        raise TypeError("expected a native TrainingSnapshot")
    if {
        field.name for field in fields(snapshot)
    } != _STATE_FIELDS | _OPERATIONAL_SNAPSHOT_FIELDS or set(
        vars(snapshot)
    ) != _STATE_FIELDS | _OPERATIONAL_SNAPSHOT_FIELDS:
        raise ValueError("unknown native snapshot fields: update the digest schema")
    if type(observation) is not dict or set(observation) != _OBSERVATION_FIELDS:
        raise ValueError("observation requires exactly the v1 update fields")
    if (
        observation["step"] != snapshot.step
        or observation["tokens_seen"] != snapshot.tokens_seen
    ):
        raise ValueError("observation and snapshot update counters differ")
    if (
        type(observation["step"]) is not int
        or type(observation["tokens_seen"]) is not int
    ):
        raise TypeError("observation counters must be integers")
    if any(type(observation[key]) not in (int, float) for key in ("learning_rate",)):
        raise TypeError("learning_rate must be numeric")
    if (
        type(observation["overflow"]) is not bool
        or type(observation["retry_state"]) is not dict
    ):
        raise TypeError("overflow and retry_state must be bool and dict")
    for key in ("loss", "gradient_norm"):
        if observation[key] is not None and type(observation[key]) not in (float, int):
            raise TypeError(f"{key} must be numeric or null")

    def finite(value: object) -> None:
        if type(value) is float and not math.isfinite(value):
            raise ValueError("nonfinite semantic observation")
        if type(value) is dict:
            for key, item in value.items():
                finite(key)
                finite(item)
        elif type(value) in (list, tuple):
            for item in value:
                finite(item)
        elif type(value) not in (str, bytes, int, float, bool, type(None)):
            raise TypeError("unsupported semantic observation type")

    finite(observation)
    cadence = snapshot.cadence
    if cadence is not None and (
        type(cadence) is not dict or set(cadence) - {"step", "tokens", "minutes"}
    ):
        raise ValueError("unknown checkpoint cadence fields")
    if cadence is not None and any(
        type(value) not in (float, int) or not math.isfinite(value) or value < 0
        for value in cadence.values()
    ):
        raise ValueError("invalid checkpoint cadence watermark")
    checkpoint = (
        snapshot.config.get("checkpoint") if type(snapshot.config) is dict else None
    )
    if type(checkpoint) is dict and checkpoint.get("every_minutes") is not None:
        raise ValueError(
            "time-based checkpoint cadence lacks a common equivalence policy"
        )
    if cadence is not None and "minutes" in cadence:
        if type(checkpoint) is not dict or "every_minutes" not in checkpoint:
            raise ValueError(
                "time-based checkpoint cadence lacks a common equivalence policy"
            )
        cadence = {key: value for key, value in cadence.items() if key != "minutes"}
    if (
        snapshot.lineage_best is not None
        and type(snapshot.lineage_best) is not LineageBest
    ):
        raise TypeError("lineage_best must be a LineageBest")
    if snapshot.validation_loss is not None and (
        type(snapshot.validation_loss) not in (int, float)
        or not math.isfinite(snapshot.validation_loss)
    ):
        raise ValueError("invalid validation loss")
    if snapshot.lineage_best is not None and not math.isfinite(
        snapshot.lineage_best.loss
    ):
        raise ValueError("invalid lineage best loss")
    lineage = snapshot.lineage_best
    model = _model(snapshot)
    components: dict[str, object] = {
        "model": model,
        "optimizer": snapshot.optimizer,
        "schedule": snapshot.schedule,
        "step": snapshot.step,
        "tokens_seen": snapshot.tokens_seen,
        "cursor": snapshot.cursor,
        "config": _semantic_config(snapshot.config),
        "rng": snapshot.rng,
        "scaler": snapshot.scaler,
        "validation_loss": snapshot.validation_loss,
        "cadence": cadence,
        "engine": snapshot.engine,
        "backend": snapshot.backend,
        "optimizer_parameter_names": snapshot.optimizer_parameter_names,
        "tensor_trainability": {name: entry[1] for name, entry in model.items()},
        # The best loss/step affect future acceptance; run/checkpoint identities do not.
        "lineage_best": None if lineage is None else (lineage.step, lineage.loss),
        "observation": observation,
    }
    hashes = {key: _digest(value) for key, value in components.items()}
    return {
        "format": "klab-training-state",
        "version": VERSION,
        "components": hashes,
        "sha256": _digest(hashes),
    }


def compare_training_states(left: dict, right: dict) -> dict:
    """Compare two v1 manifests without numerical tolerance or missing components."""
    expected = _STATE_FIELDS | {"observation"}
    for state in (left, right):
        if type(state) is not dict or set(state) != {
            "format",
            "version",
            "components",
            "sha256",
        }:
            raise ValueError("invalid training state digest manifest")
        if (
            state["format"] != "klab-training-state"
            or type(state["version"]) is not int
            or state["version"] != VERSION
        ):
            raise ValueError("unsupported training state digest version")
        hashes = state["components"]
        if (
            type(hashes) is not dict
            or set(hashes) != expected
            or any(
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in hashes.values()
            )
            or state["sha256"] != _digest(hashes)
        ):
            raise ValueError("invalid training state digest components")
    differences = sorted(
        key for key in expected if left["components"][key] != right["components"][key]
    )
    return {
        "result": "DIFFERENT" if differences else "BITWISE_EQUIVALENT",
        "differences": differences,
    }
