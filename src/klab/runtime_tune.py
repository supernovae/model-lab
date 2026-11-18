"""Short measured trials of performance levers on the device a config runs on.

``tune`` trains a few real updates on synthetic tokens of the config's shape
under each variant of a small lever grid, one device, one process, and writes
a device performance profile: what was measured, on what, with which software.
It changes nothing in the config; the profile is evidence to read and keep.
"""

from __future__ import annotations

import copy
import gc
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from klab.config.models import RunConfig
from klab.device_hints import hints_for_device
from klab.engines.base import EngineOutOfMemory, Microbatch
from klab.engines.pytorch import PyTorchEngine
from klab.memory import DeviceMemorySpill
from klab.runtime import synchronize, torch_device_for, validate_runtime
from klab.runtime_profile import RuntimeAuthorization

PROFILE_VERSION = 1

_RECOMPUTE_OFF = {
    "runtime": {"memory": {"activation_checkpointing": {"enabled": False}}}
}
_FLEX = {"attention": {"implementation": "flex"}}
_FUSED = {"optimizer": {"fused": True}}
_COMPILED_HEAD = {
    "runtime": {"compile_output_head_loss": True, "output_head_chunk_tokens": None}
}

# Each variant is a named change to the config it is given. The grid is small
# on purpose: levers measured to matter on the first device families.
LEVER_GRID: tuple[tuple[str, tuple[dict[str, Any], ...]], ...] = (
    ("config", ()),
    ("fused_adamw", (_FUSED,)),
    ("recompute_off", (_RECOMPUTE_OFF,)),
    ("recompute_off+flex", (_RECOMPUTE_OFF, _FLEX)),
    ("recompute_off+flex+fused_adamw", (_RECOMPUTE_OFF, _FLEX, _FUSED)),
    (
        "recompute_off+flex+fused_adamw+compiled_head",
        (_RECOMPUTE_OFF, _FLEX, _FUSED, _COMPILED_HEAD),
    ),
)


def _merge(base: dict[str, Any], change: dict[str, Any]) -> None:
    for key, value in change.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def variant_config(config: RunConfig, changes: tuple[dict[str, Any], ...]) -> RunConfig:
    """``config`` on one device with the lever changes applied, validated."""
    values = copy.deepcopy(config.model_dump(mode="json"))
    runtime = values["runtime"]
    runtime.pop("secondary_device_indices", None)
    runtime.pop("parallel", None)
    for change in changes:
        _merge(values, change)
    return RunConfig.model_validate(values)


def _trial(
    config: RunConfig,
    authorization: RuntimeAuthorization | None,
    updates: int,
    warmup: int,
    microbatches: int,
) -> dict[str, Any]:
    """Time ``updates`` updates after ``warmup`` compile-and-warm-up ones."""
    engine = PyTorchEngine()
    engine.validate(config, authorization=authorization)
    device = torch_device_for(config.runtime.backend, config.runtime.device_index)
    api = getattr(torch, device.type, None)
    reset = getattr(api, "reset_peak_memory_stats", None)
    if reset is not None:
        torch.empty(1, device=device)  # the device runtime starts lazily
        reset(device)
    rng = np.random.default_rng(config.seed)
    rows, length = config.training.micro_batch_size, config.training.seq_len
    vocab = config.model.vocab_size

    def batches() -> list[Microbatch]:
        return [
            Microbatch(
                rng.integers(0, vocab, (rows, length), dtype=np.int64),
                rng.integers(0, vocab, (rows, length), dtype=np.int64),
            )
            for _ in range(microbatches)
        ]

    try:
        engine.initialize(config)
        seconds: list[float] = []
        for index in range(warmup + updates):
            batch = batches()
            synchronize(device)
            started = time.perf_counter()
            engine.train_update(batch, index, sum(b.targets.size for b in batch))
            synchronize(device)
            if index >= warmup:
                seconds.append(time.perf_counter() - started)
        median = float(np.median(seconds))
        peak = getattr(api, "max_memory_allocated", lambda _: None)(device)
        return {
            "status": "ok",
            "update_seconds": median,
            "tokens_per_second": microbatches * rows * length / median,
            "peak_allocated_bytes": peak,
        }
    finally:
        del engine
        gc.collect()
        empty = getattr(api, "empty_cache", None)
        if empty is not None:
            empty()


def tune(
    config: RunConfig,
    *,
    authorization: RuntimeAuthorization | None,
    updates: int = 4,
    warmup: int = 2,
    microbatches: int = 4,
    grid: tuple[tuple[str, tuple[dict[str, Any], ...]], ...] = LEVER_GRID,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """Run every variant of ``grid`` on the configured device; return the profile."""
    runtime = validate_runtime(config, authorization=authorization)
    ceiling = (
        None
        if runtime.device_total_bytes is None
        else int(
            runtime.device_total_bytes
            * config.runtime.memory.max_device_memory_fraction
        )
    )
    trials: list[dict[str, Any]] = []
    for name, changes in grid:
        row: dict[str, Any] = {"name": name, "changes": copy.deepcopy(list(changes))}
        try:
            candidate = variant_config(config, changes)
        except ValueError as error:
            row |= {"status": "not_applicable", "reason": str(error).splitlines()[-1]}
        else:
            try:
                row |= _trial(candidate, authorization, updates, warmup, microbatches)
            except (
                EngineOutOfMemory,
                DeviceMemorySpill,
                torch.OutOfMemoryError,
            ) as error:
                row |= {"status": "out_of_memory", "reason": str(error)[:200]}
            except Exception as error:  # noqa: BLE001 - one variant failing must not hide the others
                row |= {
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}"[:300],
                }
        progress(_line(row))
        trials.append(row)
    fitting = [
        row
        for row in trials
        if row["status"] == "ok"
        and (ceiling is None or (row["peak_allocated_bytes"] or 0) <= ceiling)
    ]
    best = max(fitting, key=lambda row: row["tokens_per_second"], default=None)
    family = hints_for_device(runtime.backend, runtime.device_name)
    return {
        "profile_version": PROFILE_VERSION,
        "measured_at": datetime.now(UTC).isoformat(),
        "device": {
            "family": family.family if family else None,
            "name": runtime.device_name,
            "backend": runtime.backend,
            "total_bytes": runtime.device_total_bytes,
            "memory_ceiling_bytes": ceiling,
        },
        "software": {
            "torch": runtime.framework_version,
            "runtime": runtime.runtime_version,
            "driver": runtime.driver_version,
            "os": runtime.os,
        },
        "config_shape": {
            "layers": config.model.num_layers,
            "hidden_dim": config.model.hidden_dim,
            "heads": config.model.num_heads,
            "ffn_dim": config.model.ffn_dim,
            "vocab_size": config.model.vocab_size,
            "seq_len": config.training.seq_len,
            "micro_batch_size": config.training.micro_batch_size,
            "precision": config.runtime.precision,
            "memory": config.model.memory,
        },
        "protocol": {
            "updates": updates,
            "warmup_updates": warmup,
            "microbatches_per_update": microbatches,
            "tokens": "synthetic uniform ids",
            "devices": 1,
        },
        "trials": trials,
        "best_fitting": best["name"] if best else None,
    }


def _line(row: dict[str, Any]) -> str:
    if row["status"] != "ok":
        return f"{row['name']}: {row['status']} ({row.get('reason', '')})"
    gib = (row["peak_allocated_bytes"] or 0) / 1024**3
    return (
        f"{row['name']}: {row['tokens_per_second']:,.0f} tokens/s, peak {gib:.1f} GiB"
    )


def write_profile(profile: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def default_profile_path(root: Path, profile: dict[str, Any]) -> Path:
    name = _SAFE.sub(
        "-", str(profile["device"]["name"] or profile["device"]["backend"])
    )
    return root / "tune" / f"{name.strip('-')}.yaml"
