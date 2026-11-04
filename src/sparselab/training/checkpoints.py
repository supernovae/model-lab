"""Validated immutable checkpoint generations and legacy v1 readers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import pickle
import random
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from sparselab.canonical import canonical_json
from sparselab.config.models import AdamWConfig, CheckpointRetention, RunConfig
from sparselab.data.verification import verify_file
from sparselab.engines.base import CanonicalTensor, WeightSource
from sparselab.model.inspection import TensorSpec, named_tensor_inventory
from sparselab.training import retention as retention_policy
from sparselab.training.manifest import architecture_sha256, config_sha256, sha256_file
from sparselab.training.mlx_checkpoints import (
    CODEC as MLX_CODEC,
)
from sparselab.training.mlx_checkpoints import (
    CODEC_VERSION as MLX_CODEC_VERSION,
)
from sparselab.training.mlx_checkpoints import (
    load_native_state as load_mlx_native_state,
)
from sparselab.training.mlx_checkpoints import (
    strict_json,
)
from sparselab.training.mlx_checkpoints import (
    write_native_state as write_mlx_native_state,
)
from sparselab.training.optimizer import (
    is_ngram_table_parameter,
    learning_rate_for_step,
    schedule_payload,
)
from sparselab.verification_proofs import (
    ProofStore,
    VerificationMode,
    file_binding,
    validate_mode,
)

FORMAT_VERSION = 2
SHARD_BYTES = 256 * 1024 * 1024


def _canonical(value: object) -> bytes:
    return canonical_json(value)


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _safe_member(directory: Path, name: object) -> Path | None:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        return None
    relative = Path(name)
    if ".." in relative.parts or relative.name != name.split("/")[-1]:
        return None
    member = directory / relative
    try:
        resolved_directory = directory.resolve(strict=True)
        resolved_member = member.resolve(strict=True)
    except OSError:
        return None
    if resolved_directory not in (resolved_member, *resolved_member.parents):
        return None
    current = directory
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return None
    return member


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(_canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


@dataclass(frozen=True)
class CheckpointRecord:
    generation_id: int
    relative_path: str
    manifest_sha256: str
    step: int
    tokens_seen: int
    created_at: str
    bytes: int
    validation_loss: float | None
    engine: str
    backend: str
    verification_status: str = "verified"
    resume_level: str = "full"


@dataclass(frozen=True)
class LineageBest:
    """Best evaluated checkpoint in this run's ancestry, possibly not local."""

    parent_run_id: str
    checkpoint_digest: str
    step: int
    loss: float

    def as_dict(self) -> dict[str, object]:
        return {
            "parent_run_id": self.parent_run_id,
            "checkpoint_digest": self.checkpoint_digest,
            "step": self.step,
            "loss": self.loss,
        }


def _lineage_best(value: object) -> LineageBest | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("lineage_best must be a mapping or null")
    required = {"parent_run_id", "checkpoint_digest", "step", "loss"}
    if set(value) != required:
        raise ValueError("lineage_best has invalid fields")
    parent_run_id = value["parent_run_id"]
    checkpoint_digest = value["checkpoint_digest"]
    step = value["step"]
    loss = value["loss"]
    if (
        not isinstance(parent_run_id, str)
        or not parent_run_id
        or not isinstance(checkpoint_digest, str)
        or len(checkpoint_digest) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_digest)
        or type(step) is not int
        or step < 0
        or not isinstance(loss, (int, float))
        or not math.isfinite(loss)
    ):
        raise ValueError("lineage_best has invalid values")
    return LineageBest(parent_run_id, checkpoint_digest, step, float(loss))


def choose_lineage_best(
    current: LineageBest | None, candidate: LineageBest | None
) -> LineageBest | None:
    """Keep the earliest lineage entry unless a candidate strictly improves loss."""
    if current is None or (candidate is not None and candidate.loss < current.loss):
        return candidate
    return current


@dataclass(frozen=True)
class VerificationReport:
    valid: bool
    errors: tuple[dict[str, str], ...]
    verified_files: tuple[dict[str, str], ...]
    resume_level: str


@dataclass(frozen=True)
class RecoveryResult:
    record: CheckpointRecord | None
    rejected: tuple[VerificationReport, ...]


@dataclass
class TrainingSnapshot:
    model: dict[str, torch.Tensor | np.ndarray]
    optimizer: dict[str, object]
    schedule: dict[str, object]
    step: int
    tokens_seen: int
    cursor: tuple[int, int]
    config: dict[str, object]
    run_id: str
    rng: dict[str, object] | None = None
    scaler: dict[str, object] | None = None
    validation_loss: float | None = None
    cadence: dict[str, object] | None = None
    engine: str = "pytorch"
    backend: str = "cpu"
    optimizer_parameter_names: dict[int | str, str] | list[list[str]] | None = None
    config_sha256: str | None = None
    architecture_sha256: str | None = None
    source_identity_sha256: str | None = None
    manifest_sha256: str | None = None
    parent_checkpoint_sha256: str | None = None
    cumulative_wall_seconds: float | None = None
    cumulative_update_seconds: float | None = None
    tensor_trainability: dict[str, bool] | None = None
    lineage_best: LineageBest | None = None
    checkpoint_sha256: str | None = None
    weight_source: WeightSource | None = None
    exposure: dict[str, Any] | None = None


def _check_rng(rng: dict[str, Any], backend: str, device_index: int) -> None:
    """Validate host RNGs without changing the process's random streams."""
    random.Random().setstate(rng["python"])
    keys = rng["numpy_keys"]
    if (
        not isinstance(keys, torch.Tensor)
        or keys.dtype != torch.uint32
        or keys.shape != (624,)
    ):
        raise ValueError("invalid NumPy RNG keys")
    np.random.RandomState().set_state(
        (
            rng["numpy_kind"],
            keys.numpy(),
            rng["numpy_pos"],
            rng["numpy_has_gauss"],
            rng["numpy_cached_gaussian"],
        )
    )
    if not 0 <= rng["numpy_pos"] <= 624 or rng["numpy_has_gauss"] not in {0, 1}:
        raise ValueError("invalid NumPy RNG cursor")
    torch.Generator(device="cpu").set_state(rng["torch"])
    if rng["device_type"] != backend or rng["device_index"] != device_index:
        raise ValueError("RNG backend or device index mismatch")
    device_rng = rng["device_rng"]
    if backend == "cpu":
        if device_rng is not None:
            raise ValueError("CPU RNG envelope contains accelerator state")
    elif (
        not isinstance(device_rng, torch.Tensor)
        or device_rng.dtype != torch.uint8
        or device_rng.ndim != 1
        or device_rng.numel() == 0
    ):
        raise ValueError("invalid accelerator RNG state")


def _check_tensor_inventory(
    config: RunConfig, tensors: dict[str, Any], aliases: dict[str, str]
) -> Mapping[str, TensorSpec]:
    expected = named_tensor_inventory(
        config.model,
        config.attention,
        trainable_parameters=config.training.trainable_parameters,
    )
    if set(expected) != set(tensors) | set(aliases):
        raise ValueError("configured tensor inventory mismatch")
    for name, spec in expected.items():
        if spec.alias_of is not None:
            if aliases.get(name) != spec.alias_of:
                raise ValueError(f"configured tensor alias mismatch: {name}")
        else:
            declared = tensors[name]
            if (
                declared.get("shape") != list(spec.shape)
                or declared.get("dtype") != f"torch.{spec.dtype}"
                or declared.get("trainable") is not spec.trainable
            ):
                raise ValueError(f"configured tensor metadata mismatch: {name}")
    return expected


def _check_common_native_state(
    native: Mapping[str, Any],
    raw: Mapping[str, Any],
    tensors: Mapping[str, Any],
    aliases: Mapping[str, str],
) -> tuple[RunConfig, dict[str, Any], float]:
    if (
        type(native.get("format_version")) is not int
        or native["format_version"] != FORMAT_VERSION
    ):
        raise ValueError("unsupported native checkpoint format")
    for field in ("format_version", "engine", "backend", "step", "tokens_seen"):
        if native.get(field) != raw.get(field) or type(native.get(field)) is not type(
            raw.get(field)
        ):
            raise ValueError(f"{field} differs from checkpoint manifest")
    raw_config = native["config"]
    if not isinstance(raw_config, dict):
        raise TypeError("native config must be a mapping")
    config = RunConfig.model_validate(raw_config)
    for field, digest in (
        ("config_sha256", config_sha256(raw_config)),
        ("architecture_sha256", architecture_sha256(raw_config)),
    ):
        if native.get(field) != digest or raw.get(field) != digest:
            raise ValueError(f"{field} does not match the raw saved configuration")
    for field in (
        "manifest_sha256",
        "source_identity_sha256",
        "parent_checkpoint_sha256",
        "lineage_best",
    ):
        if native.get(field) != raw.get(field):
            raise ValueError(f"{field} differs from checkpoint manifest")
    if (
        config.runtime.engine != native["engine"]
        or config.runtime.backend != native["backend"]
        or not isinstance(native["run_id"], str)
        or not native["run_id"]
    ):
        raise ValueError("native run or runtime identity mismatch")
    expected = _check_tensor_inventory(config, tensors, aliases)
    step, tokens = native["step"], native["tokens_seen"]
    if (
        type(step) is not int
        or type(tokens) is not int
        or not 0 <= step <= config.training.max_steps
        or not 0 <= tokens <= config.training.max_tokens
        or (step == 0) != (tokens == 0)
    ):
        raise ValueError("native update or target counter is out of bounds")
    cursor = native["cursor"]
    if (
        not isinstance(cursor, (tuple, list))
        or len(cursor) != 2
        or any(type(value) is not int or value < 0 for value in cursor)
    ):
        raise ValueError("invalid native data cursor")
    schedule = schedule_payload(config, step)
    if native["schedule"] != schedule:
        raise ValueError("schedule does not match configuration and completed updates")
    learning_rate = (
        learning_rate_for_step(
            step,
            config.training.max_steps,
            config.optimizer.warmup_steps,
            config.optimizer.peak,
            config.optimizer.floor,
            stable_steps=config.training.stable_steps,
            decay_steps=(
                config.optimizer.decay_steps
                if isinstance(config.optimizer, AdamWConfig)
                else None
            ),
        )
        if step
        else config.optimizer.peak
    )
    cadence = native.get("cadence")
    if cadence is not None:
        if not isinstance(cadence, dict) or set(cadence) - {
            "step",
            "tokens",
            "minutes",
        }:
            raise ValueError("invalid checkpoint cadence watermarks")
        for field, value in cadence.items():
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"invalid checkpoint cadence watermark: {field}")
    _lineage_best(native.get("lineage_best"))
    for field in ("cumulative_wall_seconds", "cumulative_update_seconds"):
        value = native.get(field)
        if value is not None and (
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"invalid cumulative timing: {field}")
    return config, expected, learning_rate


def _learned_audit_contract(config: RunConfig) -> dict[str, object] | None:
    """Return the one v2 local-table audit contract allowed in optimizer state."""
    manifest_path = config.training.portability_manifest_path
    if manifest_path is None:
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("learned audit portability manifest is unreadable") from error
    coordinate = manifest.get("coordinate") if isinstance(manifest, dict) else None
    if not (
        manifest.get("version") == 2
        and isinstance(coordinate, dict)
        and coordinate.get("role") in {"source", "native"}
        and coordinate.get("condition") in {"source-real", "native"}
    ):
        return None
    descriptor = manifest.get("world_manifest")
    training_ids = manifest.get("training_fact_ids")
    if (
        not isinstance(descriptor, dict)
        or not isinstance(descriptor.get("path"), str)
        or not isinstance(descriptor.get("sha256"), str)
        or type(manifest.get("seed")) is not int
        or not isinstance(training_ids, list)
        or not all(isinstance(item, str) for item in training_ids)
    ):
        raise ValueError("learned audit portability manifest is malformed")
    data_path = manifest_path.parent / descriptor["path"]
    try:
        data = json.loads(data_path.read_text(encoding="utf-8"))
        facts_path = data_path.parent / data["facts"]["path"]
        facts = [
            json.loads(line)
            for line in facts_path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("learned audit fact inventory is unreadable") from error
    from sparselab.data.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(config.tokenizer.path)
    targets: list[tuple[int, int]] = []
    wanted = set(training_ids)
    for fact in facts:
        if not isinstance(fact, dict) or fact.get("fact_id") not in wanted:
            continue
        row, symbol = fact.get("target_row"), fact.get("assigned_symbol")
        token = tokenizer.token_to_id(symbol) if isinstance(symbol, str) else None
        if type(row) is not int or token is None:
            raise ValueError("learned audit fact target is malformed")
        targets.append((row, token))
    if len(targets) != len(wanted) or len({row for row, _ in targets}) != len(targets):
        raise ValueError("learned audit fact inventory differs from run manifest")
    targets.sort()
    coordinate_with_seed = {**coordinate, "seed": manifest["seed"]}
    return {
        "coordinate_sha256": hashlib.sha256(
            canonical_json(coordinate_with_seed)
        ).hexdigest(),
        "data_manifest_sha256": descriptor["sha256"],
        "max_steps": config.training.max_steps,
        "rows": [row for row, _ in targets],
        "tokens": [token for _, token in targets],
    }


def _check_learned_audit_slot(
    value: object, contract: dict[str, object], table_shape: tuple[int, ...]
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "coordinate_sha256",
        "data_manifest_sha256",
        "max_steps",
        "rows",
        "tokens",
        "initial_rows",
        "exposures",
        "gradient_seen",
        "applied_steps",
        "invalid_targets",
    }:
        raise ValueError("learned audit optimizer slot is malformed")
    if any(
        value[key] != contract[key]
        for key in ("coordinate_sha256", "data_manifest_sha256", "max_steps")
    ):
        raise ValueError("learned audit optimizer slot is bound to another run")
    rows, tokens = value["rows"], value["tokens"]
    count = len(contract["rows"])
    if (
        not isinstance(rows, torch.Tensor)
        or rows.dtype != torch.int64
        or rows.ndim != 1
        or rows.cpu().tolist() != contract["rows"]
        or not isinstance(tokens, torch.Tensor)
        or tokens.dtype != torch.int64
        or tokens.ndim != 1
        or tokens.cpu().tolist() != contract["tokens"]
    ):
        raise ValueError("learned audit optimizer factual rows differ")
    expected_bytes = int(np.prod(table_shape[1:])) * 4
    shapes = {
        "initial_rows": ((count, expected_bytes), torch.uint8),
        "exposures": ((count,), torch.int64),
        "gradient_seen": ((count,), torch.bool),
        "applied_steps": ((int(contract["max_steps"]),), torch.bool),
        "invalid_targets": ((), torch.int64),
    }
    for name, (shape, dtype) in shapes.items():
        tensor = value[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
        ):
            raise ValueError(f"learned audit optimizer slot has invalid {name}")


def _check_native_state(
    native: dict[str, Any],
    raw: dict[str, Any],
    tensors: dict[str, Any],
    aliases: dict[str, str],
) -> None:
    config, expected, learning_rate = _check_common_native_state(
        native, raw, tensors, aliases
    )
    learned_audit = _learned_audit_contract(config)
    step = native["step"]
    optimizer = native["optimizer"]
    names = native["optimizer_parameter_names"]
    if not isinstance(optimizer, dict):
        raise TypeError("invalid optimizer state")
    groups, state = optimizer["param_groups"], optimizer["state"]
    if (
        not isinstance(names, dict)
        or not isinstance(groups, list)
        or not isinstance(state, dict)
    ):
        raise TypeError("invalid named optimizer state")
    ids = [parameter_id for group in groups for parameter_id in group["params"]]
    trainable = {
        name
        for name, spec in expected.items()
        if spec.trainable and spec.alias_of is None
    }
    if (
        any(type(parameter_id) is not int for parameter_id in ids)
        or len(ids) != len(set(ids))
        or set(ids) != set(names)
        or len(names.values()) != len(set(names.values()))
        or set(names.values()) != trainable
        or not set(state) <= set(ids)
    ):
        raise ValueError(
            "optimizer parameter IDs do not cover canonical trainable tensors"
        )
    ordinary_state: dict[int, dict[str, object]] = {}
    for parameter_id, values in state.items():
        if not isinstance(values, dict):
            raise TypeError("optimizer parameter state is malformed")
        audit = values.get("sparselab_learned_audit_v1")
        if audit is not None:
            if learned_audit is None or names[parameter_id] != "memory.table.weight":
                raise ValueError("unexpected learned audit optimizer slot")
            _check_learned_audit_slot(
                audit, learned_audit, expected[names[parameter_id]].shape
            )
        ordinary = {
            key: value
            for key, value in values.items()
            if key != "sparselab_learned_audit_v1"
        }
        if ordinary:
            ordinary_state[parameter_id] = ordinary
    updated = native["optimizer_updated_parameter_names"]
    if (
        not isinstance(updated, list)
        or len(updated) != len(set(updated))
        or set(updated) != {names[parameter_id] for parameter_id in ordinary_state}
        or (step == 0 and ordinary_state)
    ):
        raise ValueError("optimizer updated-parameter inventory mismatch")
    for group in groups:
        if config.optimizer.name == "adamw" and (
            tuple(group["betas"]) != config.optimizer.betas
            or group["eps"] != config.optimizer.eps
        ):
            raise ValueError("AdamW hyperparameters differ from configuration")
        for parameter_id in group["params"]:
            shape = expected[names[parameter_id]].shape
            decay = (
                config.optimizer.weight_decay
                if config.optimizer.name != "adamw" or len(shape) >= 2
                else 0.0
            )
            multiplier = 1.0
            if (
                isinstance(config.optimizer, AdamWConfig)
                and config.model.memory == "ngram"
                and is_ngram_table_parameter(names[parameter_id])
            ):
                multiplier = config.optimizer.memory_table_lr_multiplier
                if config.optimizer.memory_table_weight_decay is not None:
                    decay = config.optimizer.memory_table_weight_decay
            if (
                group.get("lr_multiplier", 1.0) != multiplier
                or group["lr"] != learning_rate * multiplier
            ):
                raise ValueError(
                    "optimizer learning rate differs from saved schedule and table policy"
                )
            if group["weight_decay"] != decay:
                raise ValueError("optimizer weight decay differs from configuration")
    for parameter_id, values in ordinary_state.items():
        shape = expected[names[parameter_id]].shape
        count = values["step"]
        if isinstance(count, torch.Tensor) and count.numel() == 1:
            count = count.item()
        if (
            not isinstance(count, (int, float))
            or not math.isfinite(count)
            or not 1 <= count <= step
            or int(count) != count
        ):
            raise ValueError("optimizer update counter is out of bounds")
        if config.optimizer.name == "adamw":
            moments = {"exp_avg": shape, "exp_avg_sq": shape}
        elif len(shape) >= 2:
            moments = {
                "row_var": (*shape[:-1], 1),
                "col_var": (*shape[:-2], 1, shape[-1]),
            }
        else:
            moments = {"variance": shape}
        if set(values) != {"step", *moments}:
            raise ValueError("optimizer moments are missing or unexpected")
        for name, expected_shape in moments.items():
            moment = values[name]
            if (
                not isinstance(moment, torch.Tensor)
                or tuple(moment.shape) != expected_shape
                or moment.dtype != torch.float32
            ):
                raise ValueError(f"invalid optimizer moment: {name}")
    _check_rng(native["rng"], native["backend"], config.runtime.device_index)
    scaler = native.get("scaler")
    if scaler is not None:
        if config.runtime.precision != "fp16" or native["engine"] != "pytorch":
            raise ValueError("a gradient scaler is only valid for PyTorch fp16")
        if not isinstance(scaler, dict):
            raise TypeError("invalid gradient scaler state")
        required_scaler = {
            "scale",
            "growth_factor",
            "backoff_factor",
            "growth_interval",
            "_growth_tracker",
        }
        if not required_scaler <= set(scaler):
            raise ValueError("gradient scaler state is incomplete")
        for field in ("scale", "growth_factor", "backoff_factor"):
            value = scaler[field]
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"invalid gradient scaler {field}")
        if type(scaler["growth_interval"]) is not int or scaler["growth_interval"] <= 0:
            raise ValueError("invalid gradient scaler growth interval")
        tracker = scaler["_growth_tracker"]
        if isinstance(tracker, torch.Tensor):
            if tracker.numel() != 1 or tracker.dtype not in {torch.int32, torch.int64}:
                raise ValueError("invalid gradient scaler growth tracker")
        elif type(tracker) is not int or tracker < 0:
            raise ValueError("invalid gradient scaler growth tracker")


def _check_mlx_native_semantics(
    native: Mapping[str, object],
    raw: Mapping[str, object],
    tensors: Mapping[str, object],
    aliases: Mapping[str, str],
    decoded: Mapping[str, object],
) -> None:
    """Check the engine-neutral continuation fields and MLX's two AdamW groups."""
    from sparselab.training.mlx_checkpoints import validate_optimizer_state

    config, expected, learning_rate = _check_common_native_state(
        native, raw, tensors, aliases
    )
    if (
        config.runtime.engine != "mlx"
        or config.runtime.backend != "metal"
        or config.optimizer.name != "adamw"
        or config.runtime.precision not in {"auto", "fp32"}
        or decoded.get("scaler") is not None
    ):
        raise ValueError("MLX native runtime, optimizer, or scaler is invalid")
    shapes = {
        name: spec.shape
        for name, spec in expected.items()
        if spec.trainable and spec.alias_of is None
    }
    validate_optimizer_state(
        decoded.get("optimizer"),
        decoded.get("optimizer_parameter_names"),
        shapes,
        step=native["step"],
        learning_rate=learning_rate,
    )


class CheckpointManager:
    """Owns immutable v2 generations rooted at a single run directory."""

    def __init__(
        self,
        run_dir: Path,
        *,
        manifest_sha256: str | None = None,
        keep_periodic: bool = True,
        retention: CheckpointRetention | None = None,
        inherited_milestones: list[dict] | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.root = run_dir / "checkpoints"
        self.manifest_sha256 = manifest_sha256
        self.keep_periodic = keep_periodic
        self.retention = retention
        self._inherited_milestones = inherited_milestones
        self._retention_admission: dict | None = None
        # Test-only crash seams: production callers leave this unset.
        self._fault_injector = fault_injector
        self._lease_handle: object | None = None
        self._lease_depth = 0
        self._write_records: list[CheckpointRecord] | None = None

    @contextmanager
    def writer_lease(self) -> Iterator[None]:
        """Take the per-run nonblocking advisory lock, reentrantly per manager."""
        if self._lease_depth:
            self._lease_depth += 1
            try:
                yield
            finally:
                self._lease_depth -= 1
            return
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".writer.lock"
        handle = lock_path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise RuntimeError(f"checkpoint writer lease is held: {self.run_dir}")
        self._lease_handle = handle
        self._lease_depth = 1
        self._write_records = None
        try:
            self._bind_retention()
            yield
        finally:
            self._lease_depth -= 1
            if not self._lease_depth:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
                self._lease_handle = None
                self._write_records = None

    def _publish_retention(self, path: Path, payload: dict) -> None:
        _atomic_json(path, payload)
        # Unlike legacy pointer repair, preservation metadata must report a
        # directory durability failure before any generation can be pruned.
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _bind_retention(self) -> None:
        path = self.root / "retention-policy.json"
        if path.is_symlink():
            raise ValueError("unsafe retention policy")
        if path.exists():
            saved = CheckpointRetention.model_validate(strict_json(path))
            if self.retention is not None and saved != self.retention:
                raise ValueError(
                    "checkpoint retention policy cannot change within a run"
                )
            self.retention = saved
        elif self.retention is not None:
            if any(self.root.glob("step_*_gen_*")):
                raise ValueError(
                    "milestone retention must be declared before the first checkpoint"
                )
            self._publish_retention(path, self.retention.model_dump(mode="json"))

        if self._inherited_milestones is not None:
            inherited_path = self.root / "retention-inherited.json"
            payload = {"references": self._inherited_milestones}
            if inherited_path.is_symlink():
                raise ValueError("unsafe inherited retention references")
            if inherited_path.exists():
                if strict_json(inherited_path) != payload:
                    raise ValueError("inherited milestone selections cannot change")
            else:
                self._publish_retention(inherited_path, payload)

    def _retention_check(self, operation: Callable[[], dict]) -> dict:
        try:
            return operation()
        except (OSError, ValueError, RuntimeError) as error:
            _atomic_json(
                self.root / "retention-failure.json",
                {
                    "status": "refused",
                    "reason": str(error),
                    "policy": self.retention.model_dump(mode="json")
                    if self.retention
                    else None,
                },
            )
            raise

    def _next_generation(self) -> int:
        generations: list[int] = []
        for candidate in self.root.glob("step_*_gen_*"):
            if candidate.is_dir():
                suffix = candidate.name.rsplit("_", 1)[-1]
                if suffix.isdigit():
                    generations.append(int(suffix))
        return max(generations, default=0) + 1

    def _fault(self, point: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(point)

    def _save_weights(
        self,
        destination: Path,
        tensors: Mapping[str, torch.Tensor | np.ndarray],
        trainability: Mapping[str, bool] | None = None,
        source: WeightSource | None = None,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        """Write canonical shards without treating native optimizer arrays as weights."""
        aliases: dict[str, str] = dict(source.aliases) if source is not None else {}
        canonical: (
            Iterator[tuple[str, torch.Tensor, bool]]
            | list[tuple[str, torch.Tensor, bool]]
        )
        if source is not None:
            if tensors:
                raise ValueError("snapshot model and weight_source are ambiguous")

            def source_tensors() -> Iterator[tuple[str, torch.Tensor, bool]]:
                names: set[str] = set()
                for item in source.tensors():
                    if not isinstance(item, CanonicalTensor) or item.name in names:
                        raise ValueError(
                            "weight source yielded an invalid or duplicate tensor"
                        )
                    names.add(item.name)
                    if (
                        not isinstance(item.array, np.ndarray)
                        or not item.array.flags.c_contiguous
                    ):
                        raise ValueError(
                            "weight source tensor must be a contiguous NumPy array"
                        )
                    yield item.name, torch.from_numpy(item.array), item.trainable
                    del item

            canonical = source_tensors()
        else:
            canonical = []
            seen: dict[tuple[int, int, tuple[int, ...], str], str] = {}
            for name, value in sorted(tensors.items()):
                tensor = (
                    value
                    if isinstance(value, torch.Tensor)
                    else torch.from_numpy(np.ascontiguousarray(value))
                )
                key = (
                    tensor.untyped_storage().data_ptr(),
                    tensor.storage_offset(),
                    tuple(tensor.shape),
                    str(tensor.dtype),
                )
                if key in seen:
                    aliases[name] = seen[key]
                else:
                    seen[key] = name
                    canonical.append(
                        (
                            name,
                            tensor,
                            trainability[name]
                            if trainability is not None and name in trainability
                            else bool(tensor.requires_grad),
                        )
                    )
        shards: list[dict[str, object]] = []
        current: dict[str, torch.Tensor] = {}
        size, shard_number = 0, 1

        def flush() -> None:
            nonlocal current, size, shard_number
            if not current:
                return
            filename = f"weights-{shard_number:05d}.safetensors"
            path = destination / filename
            save_file(current, path)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            shards.append(
                {
                    "name": filename,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                    "tensors": sorted(current),
                }
            )
            current, size, shard_number = {}, 0, shard_number + 1

        inventory: dict[str, object] = {}
        for name, tensor, is_trainable in canonical:
            tensor_bytes = tensor.numel() * tensor.element_size()
            if current and size + tensor_bytes > SHARD_BYTES:
                flush()
            current[name] = tensor.detach().to(device="cpu").contiguous()
            size += tensor_bytes
            inventory[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "trainable": is_trainable,
            }
            if size >= SHARD_BYTES:
                flush()
            del tensor
        flush()
        if set(aliases).intersection(inventory) or any(
            target not in inventory for target in aliases.values()
        ):
            raise ValueError(
                "weight source aliases must reference unique canonical tensors"
            )
        return {"tensors": inventory, "aliases": aliases}, shards

    def save(
        self, snapshot: TrainingSnapshot, validation_loss: float | None = None
    ) -> CheckpointRecord:
        if not self._lease_depth:
            with self.writer_lease():
                return self.save(snapshot, validation_loss)
        if (
            snapshot.manifest_sha256 is not None
            and self.manifest_sha256 is not None
            and snapshot.manifest_sha256 != self.manifest_sha256
        ):
            raise ValueError(
                "snapshot manifest digest does not match checkpoint manager"
            )
        if not isinstance(snapshot.config, dict):
            raise TypeError("snapshot config must be a mapping")
        resolved = RunConfig.model_validate(snapshot.config)
        if self.retention != resolved.checkpoint.retention:
            raise ValueError("snapshot and manager retention policies differ")
        if self._write_records is None:
            self.reconcile()
        generation_id = self._next_generation()
        name = f"step_{snapshot.step:08d}_gen_{generation_id:06d}"
        final = self.root / name
        if final.exists():
            raise FileExistsError(f"checkpoint generation already exists: {final}")
        if self.retention is not None:
            if self.keep_periodic:
                raise ValueError("milestone saves require keep_periodic=false")
            self._retention_admission = self._retention_check(
                lambda: retention_policy.admit_write(self.root, self.retention)
            )
        # Checkpoint identities bind the exact serialized config, not defaults
        # introduced by a later schema reader.
        snapshot.config_sha256 = snapshot.config_sha256 or config_sha256(
            snapshot.config
        )
        snapshot.architecture_sha256 = (
            snapshot.architecture_sha256 or architecture_sha256(snapshot.config)
        )
        if validation_loss is not None and (
            not isinstance(validation_loss, (int, float))
            or not math.isfinite(validation_loss)
        ):
            raise ValueError("checkpoint validation loss must be finite or null")
        temporary = Path(
            tempfile.mkdtemp(prefix=f".step_{snapshot.step}.", dir=self.root)
        )
        trainability = snapshot.tensor_trainability
        if trainability is None:
            trainability = {
                name: spec.trainable
                for name, spec in named_tensor_inventory(
                    resolved.model,
                    resolved.attention,
                    trainable_parameters=resolved.training.trainable_parameters,
                ).items()
                if spec.alias_of is None
            }
        try:
            weight_index, shards = self._save_weights(
                temporary, snapshot.model, trainability, snapshot.weight_source
            )
            common_native = {
                "format_version": FORMAT_VERSION,
                "schedule": snapshot.schedule,
                "step": snapshot.step,
                "tokens_seen": snapshot.tokens_seen,
                "cursor": snapshot.cursor,
                "config": snapshot.config,
                "run_id": snapshot.run_id,
                "cadence": snapshot.cadence,
                "engine": snapshot.engine,
                "backend": snapshot.backend,
                "config_sha256": snapshot.config_sha256,
                "architecture_sha256": snapshot.architecture_sha256,
                "source_identity_sha256": snapshot.source_identity_sha256,
                "manifest_sha256": snapshot.manifest_sha256 or self.manifest_sha256,
                "parent_checkpoint_sha256": snapshot.parent_checkpoint_sha256,
                "cumulative_wall_seconds": snapshot.cumulative_wall_seconds,
                "cumulative_update_seconds": snapshot.cumulative_update_seconds,
                "lineage_best": (
                    snapshot.lineage_best.as_dict()
                    if snapshot.lineage_best is not None
                    else None
                ),
            }
            if snapshot.engine == "mlx":
                codec, codec_version = MLX_CODEC, MLX_CODEC_VERSION
                native_files = write_mlx_native_state(
                    snapshot, temporary, common_native
                )
            elif snapshot.engine == "pytorch":
                codec, codec_version = "pytorch_native", 2
                native = {
                    **common_native,
                    "optimizer": snapshot.optimizer,
                    "rng": snapshot.rng,
                    "scaler": snapshot.scaler,
                    "state_codec": codec,
                    "state_codec_version": codec_version,
                    "optimizer_parameter_names": snapshot.optimizer_parameter_names,
                    "optimizer_updated_parameter_names": [
                        snapshot.optimizer_parameter_names[parameter_id]
                        for parameter_id in snapshot.optimizer["state"]
                    ],
                }
                native_path = temporary / "training_state.pt"
                torch.save(native, native_path)
                with native_path.open("rb") as handle:
                    os.fsync(handle.fileno())
                native_files = [
                    {
                        "name": "training_state.pt",
                        "sha256": _sha256(native_path),
                        "bytes": native_path.stat().st_size,
                    }
                ]
            else:
                raise ValueError(f"unsupported checkpoint engine: {snapshot.engine}")
            files = [*native_files, *shards]
            content = {
                "format_version": FORMAT_VERSION,
                "generation_id": generation_id,
                "step": snapshot.step,
                "tokens_seen": snapshot.tokens_seen,
                "created_at": datetime.now(UTC).isoformat(),
                "validation_loss": validation_loss,
                "manifest_sha256": self.manifest_sha256,
                "engine": snapshot.engine,
                "backend": snapshot.backend,
                "resume_level": "full",
                "state_codec": codec,
                "state_codec_version": codec_version,
                "files": files,
                "weights": weight_index,
                "config_sha256": snapshot.config_sha256,
                "architecture_sha256": snapshot.architecture_sha256,
                "source_identity_sha256": snapshot.source_identity_sha256,
                "parent_checkpoint_sha256": snapshot.parent_checkpoint_sha256,
                "lineage_best": (
                    snapshot.lineage_best.as_dict()
                    if snapshot.lineage_best is not None
                    else None
                ),
            }
            if snapshot.exposure is not None:
                if (
                    snapshot.exposure["committed_supervised_targets"]
                    != snapshot.tokens_seen
                ):
                    raise ValueError(
                        "exposure differs from committed checkpoint targets"
                    )
                content["exposure"] = snapshot.exposure
            digest = hashlib.sha256(_canonical(content)).hexdigest()
            manifest = {**content, "sha256": digest}
            _atomic_json(temporary / "manifest.json", manifest)
            report = self.verify(temporary)
            if not report.valid:
                raise ValueError(f"checkpoint verification failed: {report.errors}")
            if self.retention is not None:

                def check_written() -> dict:
                    actual = sum(
                        member.stat().st_size for member in temporary.iterdir()
                    )
                    if actual > self.retention.max_generation_bytes:
                        raise ValueError(
                            "checkpoint exceeds declared per-generation bytes"
                        )
                    if (
                        retention_policy.footprint(self.root)
                        + sum(
                            row["full_directory_bytes"]
                            for row in retention_policy.inherited_rows(self.root)
                        )
                        > self.retention.max_peak_bytes
                    ):
                        raise ValueError("staged checkpoint exceeds peak-byte plan")
                    return {"actual_generation_bytes": actual}

                self._retention_check(check_written)
            self._fault("before_finalized_rename")
            temporary.replace(final)
            _fsync_directory(self.root)
            self._fault("after_finalized_rename")
            record = CheckpointRecord(
                generation_id,
                name,
                digest,
                snapshot.step,
                snapshot.tokens_seen,
                str(content["created_at"]),
                sum(item["bytes"] for item in files),
                validation_loss,
                snapshot.engine,
                snapshot.backend,
            )
            assert self._write_records is not None
            self._write_records.append(record)
            self._fault("before_projection_publication")
            self._project_records(self._write_records)
            return record
        except BaseException:
            if temporary.exists():
                for item in temporary.iterdir():
                    item.unlink()
                temporary.rmdir()
            raise

    def _resolve(self, path: Path) -> Path:
        if path.name not in {"latest.json", "best.json"}:
            return path
        if path.is_symlink():
            raise ValueError("unsafe checkpoint pointer")
        record = strict_json(path)
        version = record.get("format_version")
        if version is not None and (
            type(version) is not int or version != FORMAT_VERSION
        ):
            raise ValueError("unsupported checkpoint pointer version")
        relative = record.get("relative_path")
        digest = record.get("manifest_sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or (
                version is not None
                and (not isinstance(digest, str) or len(digest) != 64)
            )
        ):
            raise ValueError("invalid checkpoint pointer")
        directory = path.parent / relative
        if directory.parent != path.parent or directory.is_symlink():
            raise ValueError("unsafe checkpoint pointer")
        return directory

    @staticmethod
    def _record_from_manifest(
        directory: Path, raw: Mapping[str, object]
    ) -> CheckpointRecord:
        files = raw.get("files", [])
        return CheckpointRecord(
            int(raw["generation_id"]),
            directory.name,
            str(raw["sha256"]),
            int(raw["step"]),
            int(raw["tokens_seen"]),
            str(raw["created_at"]),
            sum(int(item["bytes"]) for item in files if isinstance(item, dict)),
            raw.get("validation_loss")
            if isinstance(raw.get("validation_loss"), (int, float))
            else None,
            str(raw["engine"]),
            str(raw["backend"]),
        )

    @staticmethod
    def _generation_order(path: Path) -> tuple[int, int]:
        try:
            _, step, _, generation = path.name.split("_")
            return int(step), int(generation)
        except TypeError, ValueError:
            return -1, -1

    def _verified_records(
        self,
        *,
        expected_manifest: str | None = None,
        require_training_state: bool = True,
    ) -> tuple[list[CheckpointRecord], list[VerificationReport]]:
        records: list[CheckpointRecord] = []
        rejected: list[VerificationReport] = []
        for candidate in sorted(
            self.root.glob("step_*_gen_*"),
            key=self._generation_order,
            reverse=True,
        ):
            if not candidate.is_dir() or candidate.is_symlink():
                continue
            report = self.verify(
                candidate,
                expected_manifest,
                require_training_state=require_training_state,
            )
            if not report.valid:
                rejected.append(
                    VerificationReport(
                        False,
                        (
                            {"field": "candidate", "reason": candidate.name},
                            *report.errors,
                        ),
                        report.verified_files,
                        report.resume_level,
                    )
                )
                continue
            try:
                raw = strict_json(candidate / "manifest.json")
                records.append(self._record_from_manifest(candidate, raw))
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
            ) as error:
                rejected.append(
                    VerificationReport(
                        False,
                        (
                            {"field": "candidate", "reason": candidate.name},
                            {"field": "manifest", "reason": str(error)},
                        ),
                        (),
                        "full",
                    )
                )
        return records, rejected

    def recovery_report(self) -> RecoveryResult:
        """Read verified candidates without taking a lease or rewriting pointers."""
        records, rejected = self._verified_records(
            expected_manifest=self.manifest_sha256
        )
        latest = max(
            records, key=lambda item: (item.step, item.generation_id), default=None
        )
        return RecoveryResult(latest, tuple(rejected))

    def lineage_best_for_child(
        self, parent_run_id: str, selected_checkpoint_digest: str
    ) -> LineageBest | None:
        """Derive ancestry only through the explicitly selected parent generation."""
        records, _ = self._verified_records(
            expected_manifest=self.manifest_sha256, require_training_state=False
        )
        selected = next(
            (
                record
                for record in records
                if record.manifest_sha256 == selected_checkpoint_digest
            ),
            None,
        )
        if selected is None:
            raise ValueError("selected parent is not a verified generation of this run")
        records = [
            record
            for record in records
            if record.generation_id <= selected.generation_id
            and record.step <= selected.step
        ]
        local_record = min(
            (
                record
                for record in records
                if record.validation_loss is not None
                and math.isfinite(record.validation_loss)
            ),
            key=lambda record: (float(record.validation_loss), record.generation_id),
            default=None,
        )
        local = (
            LineageBest(
                parent_run_id,
                local_record.manifest_sha256,
                local_record.step,
                float(local_record.validation_loss),
            )
            if local_record is not None
            else None
        )
        inherited: LineageBest | None = None
        for record in records:
            try:
                raw = strict_json(self.root / record.relative_path / "manifest.json")
                inherited = choose_lineage_best(
                    inherited, _lineage_best(raw.get("lineage_best"))
                )
            except OSError, TypeError, ValueError, json.JSONDecodeError:
                continue
        return choose_lineage_best(inherited, local)

    def reconcile(self) -> RecoveryResult:
        """Repair lookup pointers from verified immutable generations under the lease."""
        if not self._lease_depth:
            with self.writer_lease():
                return self.reconcile()
        self._retention_admission = None
        records, rejected = self._verified_records(
            expected_manifest=self.manifest_sha256
        )
        if self.retention is not None and rejected:

            def invalid() -> dict:
                raise ValueError("retention refuses unverified checkpoint generations")

            self._retention_check(invalid)
        self._write_records = records
        latest = self._project_records(records)
        return RecoveryResult(latest, tuple(rejected))

    def _project_records(
        self, records: list[CheckpointRecord]
    ) -> CheckpointRecord | None:
        """Publish pointers only after generation verification and durable rename."""
        records.sort(key=lambda item: (item.step, item.generation_id))
        retention = None
        if self.retention is not None:
            retention = self._retention_check(
                lambda: retention_policy.plan(self.root, records, self.retention)
            )
        if not records:
            for pointer in ("latest.json", "best.json"):
                (self.root / pointer).unlink(missing_ok=True)
            _fsync_directory(self.root)
            return None
        latest = records[-1]
        finite = [
            item
            for item in records
            if item.validation_loss is not None and math.isfinite(item.validation_loss)
        ]
        best = (
            min(finite, key=lambda item: (item.validation_loss, item.generation_id))
            if finite
            else None
        )
        _atomic_json(
            self.root / "latest.json",
            {"format_version": FORMAT_VERSION, **asdict(latest)},
        )
        self._fault("after_latest_projection_before_best_projection")
        if best is not None:
            _atomic_json(
                self.root / "best.json",
                {"format_version": FORMAT_VERSION, **asdict(best)},
            )
        else:
            (self.root / "best.json").unlink(missing_ok=True)
            _fsync_directory(self.root)
        if not self.keep_periodic:
            protected = {item.relative_path for item in records[-2:]}
            if best is not None:
                protected.add(best.relative_path)
            if retention is not None:
                protected = {
                    row["relative_path"]
                    for row in retention["retained"]
                    if row.get("local", True)
                }
                retention["write_admission"] = self._retention_admission
                # Durable keep-set precedes deletion; a crash can only leave extra files.
                self._publish_retention(self.root / "retention.json", retention)
                self._fault("after_retention_publication_before_prune")
            for record in records:
                if record.relative_path not in protected:
                    directory = self.root / record.relative_path
                    for member in directory.iterdir():
                        member.unlink()
                    directory.rmdir()
            records[:] = [
                record for record in records if record.relative_path in protected
            ]
            _fsync_directory(self.root)
        return latest

    def verify(
        self,
        path: Path,
        expected_manifest: str | None = None,
        require_training_state: bool = True,
        *,
        expected_config: RunConfig | None = None,
        proof_store: ProofStore | None = None,
        verification_mode: VerificationMode = "cold",
    ) -> VerificationReport:
        errors: list[dict[str, str]] = []
        files: list[dict[str, str]] = []
        validate_mode(verification_mode)
        cold_members: list[tuple[dict[str, object], object]] = []
        try:
            pointer: dict[str, object] | None = None
            if path.name in {"latest.json", "best.json"}:
                pointer = strict_json(path)
            directory = self._resolve(path)
            if not directory.is_dir() or directory.is_symlink():
                raise ValueError("unsafe checkpoint directory")
            manifest_path = _safe_member(directory, "manifest.json")
            if manifest_path is None:
                raise ValueError("unsafe checkpoint manifest path")
            raw = strict_json(manifest_path)
            digest = raw.pop("sha256", None)
            if (
                not isinstance(digest, str)
                or hashlib.sha256(_canonical(raw)).hexdigest() != digest
            ):
                errors.append({"field": "manifest", "reason": "hash mismatch"})
            if pointer is not None and pointer.get("manifest_sha256") != digest:
                errors.append(
                    {"field": "pointer", "reason": "manifest digest mismatch"}
                )
            if (
                type(raw.get("format_version")) is not int
                or raw["format_version"] != FORMAT_VERSION
            ):
                errors.append({"field": "format_version", "reason": "unsupported"})
            if (
                expected_manifest is not None
                and raw.get("manifest_sha256") != expected_manifest
            ):
                errors.append({"field": "manifest_sha256", "reason": "mismatch"})
            listed = raw.get("files")
            if not isinstance(listed, list):
                errors.append({"field": "files", "reason": "invalid"})
                listed = []
            names: set[str] = set()
            safe_members: dict[str, Path] = {}
            for entry in listed:
                name = entry.get("name") if isinstance(entry, dict) else None
                member = _safe_member(directory, name)
                if (
                    member is None
                    or not isinstance(entry, dict)
                    or name in names
                    or not isinstance(entry.get("sha256"), str)
                    or type(entry.get("bytes")) is not int
                    or entry["bytes"] < 0
                ):
                    errors.append(
                        {"field": "files", "reason": "unsafe or invalid member"}
                    )
                    continue
                names.add(name)
                safe_members[name] = member
                if not member.is_file() or member.stat().st_size != entry["bytes"]:
                    errors.append({"field": name, "reason": "hash or size mismatch"})
                    continue
                if proof_store is None or verification_mode == "cold":
                    if _sha256(member) != entry["sha256"]:
                        errors.append(
                            {"field": name, "reason": "hash or size mismatch"}
                        )
                    else:
                        files.append({"name": name, "sha256": entry["sha256"]})
                    continue
                binding = file_binding(
                    member,
                    entry["sha256"],
                    kind="checkpoint_member",
                    identifier=f"{directory.name}/{name}",
                    closure={
                        "checkpoint_sha256": digest,
                        "run_manifest_sha256": raw.get("manifest_sha256"),
                    },
                )
                try:
                    proof = verify_file(
                        member,
                        expected_sha256=entry["sha256"],
                        proof_store=proof_store
                        if proof_store.lookup(binding)
                        else None,
                        verification_mode=verification_mode,
                        binding=binding,
                    )
                except ValueError:
                    errors.append({"field": name, "reason": "hash or size mismatch"})
                else:
                    if proof.cold_verified:
                        cold_members.append((binding, proof))
                    files.append({"name": name, "sha256": entry["sha256"]})
            codec, codec_version = (
                raw.get("state_codec"),
                raw.get("state_codec_version"),
            )
            supported_codec = (
                codec == "pytorch_native"
                and type(codec_version) is int
                and codec_version == 2
            ) or (
                codec == MLX_CODEC
                and type(codec_version) is int
                and codec_version == MLX_CODEC_VERSION
            )
            weights_only_codec = (
                not require_training_state
                and raw.get("resume_level") == "weights_only"
                and "state_codec" in raw
                and "state_codec_version" in raw
                and codec is None
                and codec_version is None
            )
            if not supported_codec and not weights_only_codec:
                errors.append(
                    {"field": "state_codec", "reason": "unsupported or unknown codec"}
                )
            native_files = (
                {"training_state.pt"}
                if codec == "pytorch_native"
                else {"training_state.json", "optimizer.safetensors"}
                if codec == MLX_CODEC
                else set()
            )
            if require_training_state and not native_files <= names:
                errors.append(
                    {"field": "training_state", "reason": "missing native state"}
                )
            weights = raw.get("weights", {})
            tensors = weights.get("tensors", {}) if isinstance(weights, dict) else {}
            if not isinstance(tensors, dict):
                errors.append({"field": "weights", "reason": "invalid inventory"})
                tensors = {}
            found: dict[str, torch.Tensor] = {}
            for name, member in safe_members.items():
                if name.startswith("weights-") and name.endswith(".safetensors"):
                    try:
                        shard_tensors = load_file(member, device="cpu")
                        overlap = set(found).intersection(shard_tensors)
                        if overlap:
                            errors.append(
                                {
                                    "field": name,
                                    "reason": "duplicate tensor across shards",
                                }
                            )
                        found.update(shard_tensors)
                    except (
                        SafetensorError,
                        OSError,
                        ValueError,
                        RuntimeError,
                    ) as error:
                        errors.append(
                            {"field": name, "reason": f"invalid safetensors: {error}"}
                        )
            if set(tensors) != set(found):
                errors.append(
                    {"field": "weights", "reason": "tensor inventory mismatch"}
                )
            for name, tensor in found.items():
                declared = tensors.get(name)
                if (
                    not isinstance(declared, dict)
                    or declared.get("shape") != list(tensor.shape)
                    or declared.get("dtype") != str(tensor.dtype)
                ):
                    errors.append(
                        {
                            "field": f"weights.{name}",
                            "reason": "shape or dtype mismatch",
                        }
                    )
            aliases = weights.get("aliases", {}) if isinstance(weights, dict) else {}
            if not isinstance(aliases, dict):
                errors.append({"field": "aliases", "reason": "invalid"})
                aliases = {}
            for alias, target in aliases.items():
                if (
                    not isinstance(alias, str)
                    or not isinstance(target, str)
                    or alias in tensors
                    or target not in tensors
                ):
                    errors.append({"field": "aliases", "reason": "invalid alias"})
            if expected_config is not None:
                if raw.get("architecture_sha256") != architecture_sha256(
                    expected_config.model_dump(mode="json")
                ):
                    raise ValueError(
                        "requested configuration differs in architecture semantics"
                    )
                _check_tensor_inventory(expected_config, tensors, aliases)
            validated_native: Mapping[str, Any] | None = None
            if (
                require_training_state
                and codec == "pytorch_native"
                and "training_state.pt" in safe_members
            ):
                try:
                    native = torch.load(
                        safe_members["training_state.pt"],
                        map_location="cpu",
                        weights_only=True,
                        mmap=True,
                    )
                    if not isinstance(native, dict):
                        raise TypeError("native state is not a mapping")
                    if (
                        native.get("state_codec") != "pytorch_native"
                        or type(native.get("state_codec_version")) is not int
                        or native["state_codec_version"] != 2
                    ):
                        raise ValueError("unsupported PyTorch native codec")
                    _check_native_state(native, raw, tensors, aliases)
                    validated_native = native
                except (
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    KeyError,
                    pickle.UnpicklingError,
                ) as error:
                    errors.append(
                        {
                            "field": "training_state.pt",
                            "reason": f"unsafe or invalid native state: {error}",
                        }
                    )
            elif require_training_state and codec == MLX_CODEC:
                try:
                    if (
                        not {
                            "training_state.json",
                            "optimizer.safetensors",
                        }
                        <= safe_members.keys()
                    ):
                        raise ValueError("MLX native file inventory is incomplete")
                    native_json = strict_json(safe_members["training_state.json"])
                    decoded = load_mlx_native_state(directory)
                    _check_mlx_native_semantics(
                        native_json, raw, tensors, aliases, decoded
                    )
                    validated_native = native_json
                except (
                    OSError,
                    TypeError,
                    ValueError,
                    KeyError,
                    SafetensorError,
                ) as error:
                    errors.append(
                        {
                            "field": "training_state.json",
                            "reason": f"unsafe or invalid native state: {error}",
                        }
                    )
            if (
                require_training_state
                and expected_config is not None
                and validated_native is not None
            ):
                from sparselab.training.continuation import _resume_settings

                saved = RunConfig.model_validate(validated_native["config"])
                requested = expected_config.model_dump(mode="json")
                if requested["runtime"]["backend"] == "auto":
                    requested["runtime"]["backend"] = saved.runtime.backend
                if requested["runtime"]["precision"] == "auto":
                    requested["runtime"]["precision"] = "fp32"
                if _resume_settings(saved) != _resume_settings(
                    RunConfig.model_validate(requested)
                ):
                    errors.append(
                        {"field": "config", "reason": "full-resume settings mismatch"}
                    )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as error:
            errors.append({"field": "checkpoint", "reason": str(error)})
        if (
            not errors
            and proof_store is not None
            and verification_mode == "verified_reuse"
        ):
            for binding, proof in cold_members:
                proof_store.record(binding, proof)
        return VerificationReport(
            not errors,
            tuple(errors),
            tuple(files),
            "full" if require_training_state else "weights_only",
        )

    def load(
        self,
        path: Path,
        mode: Literal["resume", "promote"] = "resume",
        *,
        proof_store: ProofStore | None = None,
        verification_mode: VerificationMode = "cold",
    ) -> TrainingSnapshot:
        report = self.verify(
            path,
            self.manifest_sha256 if mode == "resume" else None,
            require_training_state=mode == "resume",
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        if not report.valid:
            raise ValueError(f"invalid checkpoint: {report.errors}")
        directory = self._resolve(path)
        raw = strict_json(directory / "manifest.json")
        weights: dict[str, torch.Tensor] = {}
        for entry in raw["files"]:
            if entry["name"].startswith("weights-") and entry["name"].endswith(
                ".safetensors"
            ):
                weights.update(load_file(directory / entry["name"], device="cpu"))
        for alias, target in raw["weights"]["aliases"].items():
            weights[alias] = weights[target]
        if mode == "promote":
            return TrainingSnapshot(
                model=weights,
                optimizer={},
                schedule={},
                step=0,
                tokens_seen=0,
                cursor=(0, 0),
                config={},
                run_id="",
                engine=raw["engine"],
                backend=raw["backend"],
                lineage_best=_lineage_best(raw.get("lineage_best")),
                checkpoint_sha256=raw["sha256"],
            )
        if raw["state_codec"] == MLX_CODEC:
            native = strict_json(directory / "training_state.json")
            native.update(load_mlx_native_state(directory))
        else:
            native = torch.load(
                directory / "training_state.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        return TrainingSnapshot(
            model=weights,
            optimizer=native["optimizer"],
            schedule=native["schedule"],
            step=native["step"],
            tokens_seen=native["tokens_seen"],
            cursor=tuple(native["cursor"]),
            config=native["config"],
            run_id=native["run_id"],
            rng=native.get("rng"),
            scaler=native.get("scaler"),
            validation_loss=raw.get("validation_loss"),
            cadence=native.get("cadence"),
            engine=native.get("engine", "pytorch"),
            backend=native.get("backend", "cpu"),
            optimizer_parameter_names=native.get("optimizer_parameter_names"),
            config_sha256=native.get("config_sha256"),
            architecture_sha256=native.get("architecture_sha256"),
            source_identity_sha256=native.get("source_identity_sha256"),
            manifest_sha256=native.get("manifest_sha256"),
            parent_checkpoint_sha256=native.get("parent_checkpoint_sha256"),
            cumulative_wall_seconds=native.get("cumulative_wall_seconds"),
            cumulative_update_seconds=native.get("cumulative_update_seconds"),
            lineage_best=_lineage_best(native.get("lineage_best")),
            checkpoint_sha256=raw["sha256"],
            exposure=raw.get("exposure"),
        )

    def latest_valid(self) -> RecoveryResult:
        return self.reconcile()


def existing_run_recovery_report(
    run_dir: Path, *, manifest_sha256: str | None = None
) -> RecoveryResult:
    """Read-only report for a pre-existing run; never creates or repairs files."""
    return CheckpointManager(run_dir, manifest_sha256=manifest_sha256).recovery_report()


# Legacy v1 is promotion-only. These readers deliberately never create or publish
# legacy files; artifact import code can validate an already-existing snapshot.
def verify_legacy_checkpoint(path: Path) -> VerificationReport:
    errors: list[dict[str, str]] = []
    files: list[dict[str, str]] = []
    try:
        manifest = path.with_suffix(".json")
        if not manifest.is_file():
            raise ValueError(f"checkpoint manifest missing: {manifest}")
        record = strict_json(manifest)
        if not isinstance(record, dict) or record.get("format_version") != 1:
            raise ValueError("invalid legacy checkpoint manifest")
        if _sha256(path) != record.get("sha256"):
            raise ValueError("checkpoint hash mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        required = {"model", "optimizer", "step", "tokens_seen", "cursor", "config"}
        if not isinstance(state, dict) or not required <= state.keys():
            raise ValueError("invalid legacy checkpoint schema")
        files.append({"name": path.name, "sha256": str(record["sha256"])})
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        pickle.UnpicklingError,
    ) as error:
        errors.append({"field": "legacy_checkpoint", "reason": str(error)})
    return VerificationReport(not errors, tuple(errors), tuple(files), "weights_only")


def load_legacy_checkpoint(path: Path) -> dict[str, object]:
    report = verify_legacy_checkpoint(path)
    if not report.valid:
        raise ValueError(f"invalid legacy checkpoint: {report.errors}")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert isinstance(state, dict)
    return state


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer, state: dict[str, object]
) -> None:
    """Restore state onto parameter devices while keeping noncapturable steps on CPU."""
    optimizer.load_state_dict(state)
    for group in optimizer.param_groups:
        if group.get("capturable", False):
            continue
        for parameter in group["params"]:
            values = optimizer.state.get(parameter)
            if not values:
                continue
            step = values.get("step")
            if isinstance(step, torch.Tensor) and step.device.type != "cpu":
                values["step"] = step.cpu()
