# Persisted campaign schema failures intentionally remain ValueError.
# ruff: noqa: TRY004
"""Immutable workflow records for the learned Engram portability campaign.

This module deliberately owns campaign publication and state transitions, rather than
reusing the historical compiled-world campaign.  Model execution is supplied through
``execute_learned_coordinate`` in :mod:`klab.research.portability_runner`; keeping
that narrow boundary makes build/plan/report completely inert.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil

from klab import host_capacity
from klab.canonical import canonical_digest, canonical_json
from klab.training.manifest import sha256_file

_EXPERIMENT = "learned-engram-portability-v1"
_PROTOCOL_FORMAT = "klab-portability-protocol"
_RECEIPT_FORMAT = "klab-portability-execution"
_SEEDS = (17, 41, 73)
_WIDTHS = (64, 128)
_SOURCE_CONDITIONS = ("source-real", "source-dense")
_ADAPTER_CONDITIONS = ("constant", "random", "permuted", "real-adapter")
_RECIPIENT_CONDITIONS = (
    "baseline",
    "constant",
    "random",
    "permuted",
    "real-zero-shot",
    "real-adapter",
    "native",
)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    claimed = value.get("sha256")
    if not isinstance(claimed, str) or claimed != canonical_digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError(f"{description} hash mismatch: {path}")
    if path.read_bytes() != canonical_json(value) + b"\n":
        raise ValueError(f"{description} must be canonical JSON: {path}")
    return value


def _verified_checkpoint_pointer(
    run_root: Path, path: Path, description: str
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is missing or symlinked: {path}")
    from klab.training.checkpoints import CheckpointManager

    verification = CheckpointManager(run_root).verify(path)
    if not verification.valid:
        raise ValueError(
            f"{description} failed checkpoint verification: {verification.errors}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _read_learned_run_audit(root: Path, run_id: str) -> dict[str, Any]:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("learned run audit has an unsafe run identity")
    run_root = root / "runs" / run_id
    audit_path = run_root / "portability_audit.json"
    database = root / "runs" / "experiments.sqlite3"
    if (
        audit_path.is_symlink()
        or not audit_path.is_file()
        or database.is_symlink()
        or not database.is_file()
    ):
        raise ValueError("learned run audit or durable experiment database is missing")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM events "
                "WHERE run_id = ? AND kind = ? ORDER BY id DESC LIMIT 1",
                (run_id, "portability_audit_recorded"),
            ).fetchone()
        event = None if row is None else json.loads(row[0])
    except (OSError, sqlite3.Error, json.JSONDecodeError) as error:
        raise ValueError("learned run audit evidence is unreadable") from error
    if (
        not isinstance(audit, dict)
        or audit.get("format") != "klab-portability-audit"
        or audit.get("version") != 2
        or audit.get("valid") is not True
        or not isinstance(event, dict)
        or event.get("valid") is not True
        or event.get("sha256") != sha256_file(audit_path)
    ):
        raise ValueError(f"learned run audit hash mismatch: {audit_path}")
    return audit


def _write_immutable_json(path: Path, value: dict[str, Any]) -> Path:
    body = {key: item for key, item in value.items() if key != "sha256"}
    encoded = canonical_json({**body, "sha256": canonical_digest(body)}) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise FileExistsError(f"immutable learned portability file differs: {path}")
        return path
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise FileExistsError(
                    f"immutable learned portability file differs: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _descriptor(root: Path, path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"learned portability input must be a regular file: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _directory_file_bytes(path: Path) -> int:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"learned resource inventory is not a directory: {path}")
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"learned resource inventory contains a symlink: {item}")
        if item.is_file():
            total += item.stat().st_size
    return total


def _updates(scale: str, fact_count: int) -> dict[str, int]:
    if scale == "smoke":
        # Smoke is intentionally a wiring-only, fixed policy.
        return {"source": 2048, "native": 2048, "preparation": 512, "adapter": 256}
    return {
        "source": 16 * fact_count,
        "native": 16 * fact_count,
        "preparation": 4096,
        "adapter": 4 * fact_count,
    }


def _coordinate_plan() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in _SEEDS:
        for condition in _SOURCE_CONDITIONS:
            rows.append(
                {
                    "role": "source",
                    "condition": condition,
                    "recipient": "source",
                    "seed": seed,
                }
            )
    for seed in _SEEDS:
        for recipient in _WIDTHS:
            rows.append(
                {
                    "role": "preparation",
                    "condition": "prepare",
                    "recipient": f"width{recipient}",
                    "seed": seed,
                }
            )
            for condition in _RECIPIENT_CONDITIONS:
                role = "native" if condition == "native" else "recipient"
                rows.append(
                    {
                        "role": role,
                        "condition": condition,
                        "recipient": f"width{recipient}",
                        "seed": seed,
                    }
                )
    return rows


def _coordinate_id(row: dict[str, object]) -> str:
    return "-".join(
        (
            str(row["role"]),
            str(row["recipient"]),
            str(row["condition"]),
            f"s{row['seed']}",
        )
    )


def _protocol(
    root: Path, scale: str, backend: str, data: dict[str, dict[str, object]]
) -> dict[str, Any]:
    """The protocol is deliberately unselected for nano until measured calibration."""
    smoke = scale == "smoke"
    selected = 128 if smoke else None
    updates = _updates(scale, 128) if smoke else None
    body = {
        "format": _PROTOCOL_FORMAT,
        "version": 2,
        "experiment": _EXPERIMENT,
        "campaign_id": f"{_EXPERIMENT}-{scale}-seed-20260925",
        "scale": scale,
        "data_seed": 20260925,
        "pair_seeds": list(_SEEDS),
        "data": data,
        "models": {
            "source": {
                "hidden_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_dim": 512,
            },
            "width64": {
                "hidden_dim": 64,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_dim": 256,
            },
            "width128": {
                "hidden_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_dim": 512,
            },
        },
        "memory": {
            "kind": "byte",
            "table_size": 65521,
            "embedding_dim": 32,
            "ngram_size": 32,
            "normalization": "raw-utf8-v1",
            "hashing": "poly257-terminal-v1",
        },
        "training": {
            "backend_request": backend,
            "policy": "smoke-wiring-only" if smoke else "unselected",
            "source": None if updates is None else updates["source"],
            "preparation": None if updates is None else updates["preparation"],
            "adapter": None if updates is None else updates["adapter"],
            "native": None if updates is None else updates["native"],
        },
        "observation_steps": {
            "source_native": [0, 32, 128, 512, 2048, 8192],
            "preparation": [0, 128, 512, 2048],
            "adapter": [0, 8, 32, 128, 512, 2048, 8192],
        },
        "gates": {
            "source_monitor_accuracy": 0.75,
            "enabled_accuracy_advantage": 0.10,
            "disabled_nll_advantage": 0.10,
            "gradient_and_changed_row_fraction": 0.90,
            "preparation_accuracy": 0.95,
            "preparation_per_symbol_accuracy": 0.75,
        },
        "controls": list(_RECIPIENT_CONDITIONS),
        "resource_selection": {
            "status": "sealed" if smoke else "uncalibrated",
            "selected_fact_count": selected,
            "candidate_fact_counts": [128] if smoke else [512, 1024, 2048],
            "ceiling_seconds": 3600 if smoke else 43200,
            "selection_path": None if smoke else "execution/resource-selection.json",
        },
        "limitations": [
            "Smoke is wiring-only and cannot support the behavioral claim."
            if smoke
            else "Timing remains uncalibrated until disposable pilots complete.",
            "This protocol tests artifact and adapter portability, not representation portability.",
        ],
    }
    return {**body, "sha256": canonical_digest(body)}


def build_learned_portability_campaign(
    output_root: Path,
    *,
    seed: int = 20260925,
    scale: str = "nano",
    backend: str = "auto",
) -> Path:
    """Materialize and seal learned inputs.  This function never constructs a model."""
    started = time.monotonic()
    if seed != 20260925:
        raise ValueError("learned portability data seed is fixed at 20260925")
    if scale not in {"smoke", "nano"}:
        raise ValueError("learned portability scale must be smoke or nano")
    if backend not in {"auto", "cpu", "mps"}:
        raise ValueError("learned portability backend must be auto, cpu, or mps")
    root = Path(output_root)
    protocol_path = root / "portability_protocol.json"
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise FileExistsError(f"learned portability root is not a directory: {root}")
    if protocol_path.exists():
        protocol = _read_json(protocol_path, "learned portability protocol")
        if protocol.get("experiment") != _EXPERIMENT:
            raise FileExistsError(
                "existing campaign root belongs to another experiment"
            )
        if (
            protocol.get("scale") != scale
            or protocol.get("training", {}).get("backend_request") != backend
        ):
            raise FileExistsError(
                "existing learned campaign conflicts with requested build"
            )
        build_receipt = root / "execution" / "build.json"
        if not build_receipt.is_file():
            raise ValueError("existing campaign lacks its immutable build receipt")
        receipt = _read_json(build_receipt, "learned build receipt")
        if receipt.get("protocol_sha256") != protocol["sha256"]:
            raise ValueError("learned build receipt belongs to another protocol")
        return protocol_path
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"refusing to reuse nonempty learned campaign root: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    from klab.data.learned_portability import materialize_learned_portability_data

    counts = (128,) if scale == "smoke" else (512, 1024, 2048)
    manifests: dict[str, dict[str, object]] = {}
    for count in counts:
        manifest_path = materialize_learned_portability_data(
            root / "data-candidates" / str(count), seed=seed, fact_count=count
        )
        manifest = _read_json(manifest_path, "learned portability data manifest")
        if (
            manifest.get("format") != "klab-learned-portability-data"
            or manifest.get("version") != 1
        ):
            raise ValueError(
                "materializer returned an unsupported learned data manifest"
            )
        manifests[str(count)] = _descriptor(root, manifest_path)
    protocol = _protocol(root, scale, backend, manifests)
    _write_immutable_json(protocol_path, protocol)
    # Nano has no scientific configuration until calibration selects a candidate.
    selected_count = 128 if scale == "smoke" else None
    updates = _updates(scale, 128) if scale == "smoke" else None
    configurations = {
        "experiment": _EXPERIMENT,
        "data_manifest": None
        if selected_count is None
        else manifests[str(selected_count)],
        "candidate_data": manifests,
        "optimizer": {
            "name": "adamw",
            "peak": 0.001,
            "floor": 0.0001,
            "warmup": "min(64, max_steps // 8)",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "micro_batch_size": 8,
            "gradient_accumulation": 1,
            "seq_len": 128,
            "precision": "fp32",
        },
        "roles": {
            "source": {
                "model": protocol["models"]["source"],
                "max_steps": None if updates is None else updates["source"],
            },
            "preparation": {
                "max_steps": None if updates is None else updates["preparation"]
            },
            "adapter": {
                "max_steps": None if updates is None else updates["adapter"],
                "trainable_parameters": ["memory.output.weight", "memory.gate.weight"],
            },
            "native": {"max_steps": None if updates is None else updates["native"]},
        },
    }
    config_path = _write_immutable_json(
        root / "execution" / "configs.json", configurations
    )
    plan = {
        "format": "klab-learned-portability-plan",
        "version": 1,
        "experiment": _EXPERIMENT,
        "protocol": _descriptor(root, protocol_path),
        "configs": _descriptor(root, config_path),
        "coordinates": [
            {**row, "coordinate_id": _coordinate_id(row)} for row in _coordinate_plan()
        ],
        "read_only": True,
    }
    plan_path = _write_immutable_json(root / "plan.json", plan)
    design_path = _write_immutable_json(
        root / "execution" / "design.json",
        {
            "experiment": _EXPERIMENT,
            "protocol": _descriptor(root, protocol_path),
            "configs": _descriptor(root, config_path),
            "timing": "uncalibrated" if scale == "nano" else "not-required-smoke",
            "coordinates": plan["coordinates"],
        },
    )
    _write_immutable_json(
        root / "execution" / "build.json",
        {
            "format": "klab-learned-portability-build",
            "version": 1,
            "experiment": _EXPERIMENT,
            "protocol_sha256": protocol["sha256"],
            "elapsed_seconds": time.monotonic() - started,
            "data_candidates": manifests,
            "config": _descriptor(root, config_path),
            "plan": _descriptor(root, plan_path),
            "design": _descriptor(root, design_path),
        },
    )
    return protocol_path


def _selection(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol["scale"] == "smoke":
        backend = protocol["training"]["backend_request"]
        return {
            "status": "sealed",
            "fact_count": 128,
            "policy": "smoke-wiring-only",
            "updates": _updates("smoke", 128),
            "backend": "cpu" if backend == "auto" else backend,
        }
    path = root / "execution" / "resource-selection.json"
    if not path.exists():
        raise ValueError(
            "nano execution requires completed immutable timing calibration"
        )
    result = _read_json(path, "learned timing selection")
    if (
        result.get("protocol_sha256") != protocol["sha256"]
        or result.get("status") != "sealed"
    ):
        raise ValueError(
            "learned timing selection is not a sealed decision for this protocol"
        )
    if (
        result.get("fact_count")
        not in protocol["resource_selection"]["candidate_fact_counts"]
    ):
        raise ValueError("learned timing selection has an invalid fact count")
    if result.get("policy") not in {"full", "half"} or not isinstance(
        result.get("updates"), dict
    ):
        raise ValueError("learned timing selection has an invalid budget policy")
    accounted = result.get("accounted_elapsed_seconds")
    if (
        isinstance(accounted, bool)
        or not isinstance(accounted, (int, float))
        or not math.isfinite(accounted)
        or accounted < 0
    ):
        raise ValueError("learned timing selection has invalid elapsed accounting")
    return result


def _seal_smoke_or_calibration(
    root: Path, protocol: dict[str, Any], allowance: float
) -> dict[str, Any]:
    """Measure timing-only runs and seal the largest feasible nano policy."""
    if protocol["scale"] == "smoke":
        return _selection(root, protocol)
    selection_path = root / "execution" / "resource-selection.json"
    if selection_path.exists():
        return _selection(root, protocol)
    started = time.monotonic()

    import statistics
    import uuid
    from datetime import datetime

    import torch

    from klab.data.packing import prepare_data
    from klab.data.tokenizer import load_tokenizer
    from klab.evaluation.inference import load_run
    from klab.evaluation.learned_portability import (
        evaluate_learned_portability,
        evaluate_learned_preparation,
    )
    from klab.research.portability import _verify_learned_data_manifest
    from klab.training.trainer import train

    prior_receipt = _receipt(root)

    calibration_limit = min(1800.0, float(allowance))
    build_receipt = _read_json(
        root / "execution" / "build.json", "learned build receipt"
    )
    prior_elapsed = (
        float(prior_receipt["elapsed_seconds"]) if prior_receipt is not None else 0.0
    )
    build_elapsed = float(build_receipt["elapsed_seconds"])
    elapsed_before_calibration = (
        prior_elapsed if prior_receipt is not None else build_elapsed
    )
    ceiling = float(protocol["resource_selection"]["ceiling_seconds"])

    pilots_path = root / "execution" / "timing-pilots.json"
    pilots_existed = pilots_path.exists()
    if pilots_existed:
        pilots = _read_json(pilots_path, "learned timing pilots")
        if pilots.get("protocol_sha256") != protocol["sha256"]:
            raise ValueError("timing pilots belong to another protocol")
        candidate_costs = pilots.get("candidate_costs")
        completed = pilots.get("all_backends")
        if not isinstance(completed, list):
            raise RuntimeError("calibration_incomplete")
        chosen = next(
            (
                item
                for item in completed
                if isinstance(item, dict)
                and item.get("backend") == pilots.get("backend")
                and item.get("complete") is True
            ),
            None,
        )
        if not isinstance(candidate_costs, dict) or not isinstance(chosen, dict):
            raise RuntimeError("calibration_incomplete")
    else:
        candidate_costs: dict[str, dict[str, Any]] = {}
        counts = protocol["resource_selection"]["candidate_fact_counts"]
        for count in counts:
            manifest_path = root / protocol["data"][str(count)]["path"]
            verification_started = time.monotonic()
            verified = _verify_learned_data_manifest(manifest_path)
            data = verified["payload"]
            if data["fact_count"] != count:
                raise ValueError("candidate data manifest fact count differs")
            verification_seconds = time.monotonic() - verification_started
            candidate_costs[str(count)] = {
                "data_verification_seconds": verification_seconds,
                "data_inventory_bytes": sum(
                    int(item["size_bytes"]) for item in data["files"]
                ),
            }

        timing_specs = (
            ("source-byte-128", "source"),
            ("dense-128", "source"),
            ("native-64", "width64"),
            ("preparation-64", "width64"),
            ("adapter-64", "width64"),
            ("adapter-128", "width128"),
        )
        timing_coordinates = {
            label: {
                "role": "calibration",
                "condition": label,
                "recipient": recipient,
                "seed": 20260925,
            }
            for label, recipient in timing_specs
        }
        timing_manifests = {
            label: build_learned_run_manifest(root, coordinate, purpose="timing")
            for label, coordinate in timing_coordinates.items()
        }

        preparation_config = _learned_config(
            root,
            timing_coordinates["source-byte-128"],
            timing_manifests["source-byte-128"],
            max_steps=16,
            purpose="timing",
            backend_override="cpu",
        )
        for count in counts:
            candidate_root = (root / protocol["data"][str(count)]["path"]).parent
            candidate_config = preparation_config.model_copy(
                update={
                    "dataset": preparation_config.dataset.model_copy(
                        update={
                            "train_path": candidate_root / "source_train.jsonl",
                            "validation_path": candidate_root
                            / "preparation_validation.jsonl",
                            "cache_dir": root
                            / "timing-cache"
                            / f"candidate-{count}-{uuid.uuid4().hex}",
                        }
                    )
                }
            )
            prep_started = time.monotonic()
            prepare_data(
                candidate_config,
                load_tokenizer(candidate_root / "tokenizer" / "tokenizer.json"),
            )
            candidate_costs[str(count)]["data_preparation_seconds"] = (
                time.monotonic() - prep_started
            )

        requested = str(protocol["training"]["backend_request"])
        mps_available = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        if requested == "auto":
            backends = ("cpu", "mps") if mps_available else ("cpu",)
        elif requested == "mps":
            if not mps_available:
                raise RuntimeError("requested MPS calibration backend is unavailable")
            backends = ("mps",)
        else:
            backends = ("cpu",)

        all_backends: list[dict[str, Any]] = []
        for backend in backends:
            rows: list[dict[str, Any]] = []
            for label, _recipient in timing_specs:
                remaining = calibration_limit - (time.monotonic() - started)
                if remaining <= 0:
                    rows.append({"role": label, "error": "calibration_deadline"})
                    break
                config = _learned_config(
                    root,
                    timing_coordinates[label],
                    timing_manifests[label],
                    max_steps=16,
                    purpose="timing",
                    backend_override=backend,
                )
                config = config.model_copy(
                    update={
                        "logging": config.logging.model_copy(
                            update={"root_dir": root / "timing-runs"}
                        )
                    }
                )
                run_id = f"timing-{backend}-{label}-{uuid.uuid4().hex[:10]}"
                train_started = time.monotonic()
                try:
                    train(
                        config,
                        run_id=run_id,
                        max_wall_seconds=remaining,
                        experiment_id=_EXPERIMENT,
                        attempt_id=run_id,
                    )
                    database = root / "timing-runs" / "experiments.sqlite3"
                    with sqlite3.connect(
                        f"file:{database}?mode=ro", uri=True
                    ) as connection:
                        status = connection.execute(
                            "SELECT status FROM runs WHERE run_id = ?",
                            (run_id,),
                        ).fetchone()
                        timings = connection.execute(
                            "SELECT step,value FROM metrics WHERE run_id=? "
                            "AND name='performance/step_seconds' ORDER BY step",
                            (run_id,),
                        ).fetchall()
                        resource_measurements = connection.execute(
                            "SELECT name,MAX(value) FROM metrics WHERE run_id=? "
                            "AND name IN ('memory/process_peak_rss_bytes', "
                            "'memory/driver_allocated_bytes') GROUP BY name",
                            (run_id,),
                        ).fetchall()
                    if status is None or status[0] != "completed":
                        raise RuntimeError("timing run did not complete")
                    values = [
                        float(value) for step, value in timings if 1 <= int(step) <= 16
                    ]
                    if len(values) != 16 or any(
                        not math.isfinite(value) or value <= 0 for value in values
                    ):
                        raise RuntimeError("timing update series is incomplete")
                    progress_path = root / "timing-runs" / run_id / "progress.json"
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    measured_memory = {
                        str(name): float(value)
                        for name, value in resource_measurements
                        if value is not None
                    }
                    peak_process_rss = measured_memory.get(
                        "memory/process_peak_rss_bytes"
                    )
                    peak_driver_allocated = measured_memory.get(
                        "memory/driver_allocated_bytes", 0.0
                    )
                    if (
                        peak_process_rss is None
                        or not math.isfinite(peak_process_rss)
                        or peak_process_rss <= 0
                        or not math.isfinite(peak_driver_allocated)
                        or peak_driver_allocated < 0
                    ):
                        raise RuntimeError(
                            "timing host-resource telemetry is incomplete"
                        )
                    run_root = root / "timing-runs" / run_id
                    checkpoint_root = run_root / "checkpoints"
                    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
                        raise RuntimeError("timing checkpoint inventory is unavailable")
                    checkpoint_generations = []
                    for item in checkpoint_root.iterdir():
                        if item.is_symlink():
                            raise RuntimeError(
                                "timing checkpoint inventory contains a symlink"
                            )
                        if item.is_dir() and item.name.startswith("step_"):
                            checkpoint_generations.append(item)
                    checkpoint_sizes = [
                        _directory_file_bytes(item) for item in checkpoint_generations
                    ]
                    if not checkpoint_sizes or any(
                        size <= 0 for size in checkpoint_sizes
                    ):
                        raise RuntimeError("timing checkpoint inventory is empty")
                    audit_path = run_root / "portability_audit.json"
                    if audit_path.is_symlink() or not audit_path.is_file():
                        raise RuntimeError("timing audit inventory is unavailable")
                    audit = json.loads(audit_path.read_text(encoding="utf-8"))
                    audit_history = audit.get("gradient_update_history")
                    if not isinstance(audit_history, list) or len(audit_history) != 16:
                        raise RuntimeError(
                            "timing audit update inventory is incomplete"
                        )
                    audit_history_bytes = len(canonical_json(audit_history))
                    audit_fixed_bytes = audit_path.stat().st_size - audit_history_bytes
                    if audit_fixed_bytes < 0:
                        raise RuntimeError("timing audit size is invalid")
                    stage_seconds: dict[str, float] = {}
                    for stage in progress.get("stages", []):
                        name = stage.get("stage")
                        start_value = stage.get("started_at")
                        finish_value = stage.get("finished_at")
                        if (
                            not isinstance(name, str)
                            or not isinstance(start_value, str)
                            or not isinstance(finish_value, str)
                        ):
                            continue
                        duration = (
                            datetime.fromisoformat(finish_value)
                            - datetime.fromisoformat(start_value)
                        ).total_seconds()
                        if duration < 0 or not math.isfinite(duration):
                            raise RuntimeError("timing stage interval is invalid")
                        stage_seconds[name] = stage_seconds.get(name, 0.0) + duration
                    evaluation_seconds = stage_seconds.get("EVALUATING", 0.0)
                    checkpoint_seconds = stage_seconds.get("CHECKPOINTED", 0.0)
                    train_elapsed = time.monotonic() - train_started
                    non_update = (
                        train_elapsed
                        - sum(values)
                        - evaluation_seconds
                        - checkpoint_seconds
                    )
                    if non_update <= 0:
                        raise RuntimeError("timing setup overhead is not measurable")
                    rows.append(
                        {
                            "role": label,
                            "run_id": run_id,
                            "median_update_seconds": statistics.median(values[-14:]),
                            "setup_and_data_preparation_seconds": non_update,
                            "checkpoint_seconds_per_pilot": checkpoint_seconds,
                            "evaluation_seconds_per_pilot": evaluation_seconds,
                            "checkpoint_boundary_count": sum(
                                stage.get("stage") == "CHECKPOINTED"
                                for stage in progress.get("stages", [])
                            ),
                            "evaluation_boundary_count": sum(
                                stage.get("stage") == "EVALUATING"
                                for stage in progress.get("stages", [])
                            ),
                            "elapsed_seconds": train_elapsed,
                            "peak_process_rss_bytes": int(peak_process_rss),
                            "peak_driver_allocated_bytes": int(peak_driver_allocated),
                            "checkpoint_generation_count": len(checkpoint_sizes),
                            "max_checkpoint_generation_bytes": max(checkpoint_sizes),
                            "audit_fixed_bytes": audit_fixed_bytes,
                            "audit_bytes_per_update": (
                                audit_history_bytes / len(audit_history)
                            ),
                        }
                    )
                except Exception as error:  # noqa: BLE001
                    rows.append(
                        {
                            "role": label,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    break
            complete = len(rows) == len(timing_specs) and all(
                "median_update_seconds" in row for row in rows
            )
            all_backends.append(
                {
                    "backend": backend,
                    "complete": complete,
                    "pilots": rows,
                    "elapsed_seconds": sum(
                        float(row.get("elapsed_seconds", 0.0)) for row in rows
                    ),
                }
            )
            if time.monotonic() - started >= calibration_limit:
                break
        completed = [item for item in all_backends if item["complete"]]
        if not completed:
            raise RuntimeError("calibration_incomplete")
        chosen = min(completed, key=lambda item: item["elapsed_seconds"])

        selected_backend = str(chosen["backend"])
        pilot_rows = {str(row["role"]): row for row in chosen["pilots"]}
        source_loaded = load_run(
            str(pilot_rows["source-byte-128"]["run_id"]),
            root / "timing-runs",
            backend=selected_backend,
        )
        preparation_loaded = load_run(
            str(pilot_rows["preparation-64"]["run_id"]),
            root / "timing-runs",
            backend=selected_backend,
        )
        adapter_loaded = load_run(
            str(pilot_rows["adapter-128"]["run_id"]),
            root / "timing-runs",
            backend=selected_backend,
        )
        for count in counts:
            candidate_manifest = root / protocol["data"][str(count)]["path"]
            source_seconds: list[float] = []
            for partitions, wording in (
                (("source_monitor",), "source_monitor"),
                (("calibration",), "adapter_return"),
                (("held_out",), "final_report"),
                (("held_out",), "final_state"),
            ):
                began = time.monotonic()
                evaluate_learned_portability(
                    source_loaded,
                    candidate_manifest,
                    partitions=partitions,
                    wording=wording,
                )
                source_seconds.append(time.monotonic() - began)
            preparation_started = time.monotonic()
            evaluate_learned_preparation(
                preparation_loaded, candidate_manifest, split="validation"
            )
            preparation_seconds = time.monotonic() - preparation_started
            recipient_seconds: list[float] = []
            for partitions, wording in (
                (("calibration",), "adapter_return"),
                (("held_out",), "final_report"),
                (("held_out",), "final_state"),
            ):
                began = time.monotonic()
                evaluate_learned_portability(
                    adapter_loaded,
                    candidate_manifest,
                    partitions=partitions,
                    wording=wording,
                )
                recipient_seconds.append(time.monotonic() - began)
            candidate_costs[str(count)].update(
                {
                    "source_monitor_evaluation_seconds": source_seconds[0],
                    "source_query_evaluation_seconds": sum(source_seconds[1:]),
                    "preparation_evaluation_seconds": preparation_seconds,
                    "recipient_query_evaluation_seconds": sum(recipient_seconds),
                }
            )
        pilots = {
            "format": "klab-learned-portability-timing",
            "version": 1,
            "experiment": _EXPERIMENT,
            "protocol_sha256": protocol["sha256"],
            "backend": selected_backend,
            "pilots": chosen["pilots"],
            "all_backends": all_backends,
            "candidate_costs": candidate_costs,
            "elapsed_seconds": time.monotonic() - started,
        }
        if pilots["elapsed_seconds"] > calibration_limit:
            raise RuntimeError("calibration_incomplete")
        _write_immutable_json(pilots_path, pilots)

    rows = pilots["pilots"]
    required_roles = {
        "source-byte-128",
        "dense-128",
        "native-64",
        "preparation-64",
        "adapter-64",
        "adapter-128",
    }
    measured = {
        str(row["role"]): float(row["median_update_seconds"])
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("median_update_seconds"), (int, float))
    }
    if set(measured) != required_roles:
        raise RuntimeError("calibration_incomplete")
    timing_by_role = {str(row["role"]): row for row in rows}
    resource_fields = (
        "peak_process_rss_bytes",
        "peak_driver_allocated_bytes",
        "checkpoint_generation_count",
        "max_checkpoint_generation_bytes",
        "audit_fixed_bytes",
        "audit_bytes_per_update",
    )
    if (
        not isinstance(candidate_costs, dict)
        or "512" not in candidate_costs
        or not isinstance(candidate_costs["512"], dict)
        or not isinstance(candidate_costs["512"].get("data_inventory_bytes"), int)
        or any(
            not isinstance(timing_by_role[role].get(field), (int, float))
            or not math.isfinite(float(timing_by_role[role][field]))
            or float(timing_by_role[role][field]) < 0
            for role in required_roles
            for field in resource_fields
        )
    ):
        raise RuntimeError("calibration_incomplete")
    if any(
        not isinstance(
            timing_by_role[role].get("setup_and_data_preparation_seconds"),
            (int, float),
        )
        or timing_by_role[role]["setup_and_data_preparation_seconds"] <= 0
        for role in required_roles
    ):
        raise RuntimeError("calibration_incomplete")
    if not pilots_existed:
        del source_loaded, preparation_loaded, adapter_loaded
        import gc

        gc.collect()
    try:
        resource_capacity = {
            "host_available_bytes": int(host_capacity.measure_memory().available),
            "campaign_volume_free_bytes": int(shutil.disk_usage(root).free),
        }
    except (OSError, psutil.Error) as error:
        raise RuntimeError("resource_measurement_unavailable") from error
    if any(value <= 0 for value in resource_capacity.values()):
        raise RuntimeError("resource_measurement_unavailable")

    pilot_seconds = float(pilots["elapsed_seconds"])
    unreceipted_pilot_seconds = 0.0
    if pilots_existed:
        pilot_descriptor = _descriptor(root, pilots_path)
        if (
            prior_receipt is None
            or prior_receipt.get("timing_pilots") != pilot_descriptor
        ):
            unreceipted_pilot_seconds = pilot_seconds
    calibration_elapsed = time.monotonic() - started
    accounted_elapsed = (
        elapsed_before_calibration + unreceipted_pilot_seconds + calibration_elapsed
    )
    remaining = min(
        float(allowance) - calibration_elapsed,
        ceiling - accounted_elapsed,
    )
    if remaining <= 0:
        raise RuntimeError("resource_blocked")
    checkpoint_bytes_by_role = {
        role: int(timing_by_role[role]["max_checkpoint_generation_bytes"])
        for role in required_roles
    }
    audit_fixed_bytes_by_role = {
        role: int(timing_by_role[role]["audit_fixed_bytes"]) for role in required_roles
    }
    audit_bytes_per_update_by_role = {
        role: float(timing_by_role[role]["audit_bytes_per_update"])
        for role in required_roles
    }
    observed_training_bytes = max(
        max(
            int(row["peak_process_rss_bytes"]),
            int(row["peak_driver_allocated_bytes"]),
        )
        for row in timing_by_role.values()
    )
    base_inventory_bytes = int(candidate_costs["512"]["data_inventory_bytes"])
    data_bundle_count = len(_coordinate_plan())
    transferred_conditions = {
        "constant",
        "random",
        "permuted",
        "real-zero-shot",
        "real-adapter",
    }
    memory_package_count = len(_SEEDS) + sum(
        row["role"] == "recipient" and row["condition"] in transferred_conditions
        for row in _coordinate_plan()
    )
    addressing = _addressing()
    portable_table_bytes = (
        int(addressing["table_size"]) * int(addressing["embedding_dim"]) * 4
    )
    candidate_projection: list[dict[str, object]] = []
    for policy, divisor in (("full", 1), ("half", 2)):
        for count in (512, 1024, 2048):
            updates = _updates("nano", count)
            if divisor == 2:
                updates = {key: value // 2 for key, value in updates.items()}
            source_row = timing_by_role["source-byte-128"]
            dense_row = timing_by_role["dense-128"]
            prep_row = timing_by_role["preparation-64"]
            native_row = timing_by_role["native-64"]
            adapter64_row = timing_by_role["adapter-64"]
            adapter128_row = timing_by_role["adapter-128"]
            schedule_counts = {
                "source_native": len(
                    {
                        0,
                        *[
                            int(step)
                            for step in protocol["observation_steps"]["source_native"]
                            if int(step) <= updates["source"]
                        ],
                        updates["source"],
                    }
                ),
                "preparation": len(
                    {
                        0,
                        *[
                            int(step)
                            for step in protocol["observation_steps"]["preparation"]
                            if int(step) <= updates["preparation"]
                        ],
                        updates["preparation"],
                    }
                ),
                "adapter": len(
                    {
                        0,
                        *[
                            int(step)
                            for step in protocol["observation_steps"]["adapter"]
                            if int(step) <= updates["adapter"]
                        ],
                        updates["adapter"],
                    }
                ),
            }
            candidate = candidate_costs[str(count)]
            base_prep = float(candidate_costs["512"]["data_preparation_seconds"])
            base_verification = float(
                candidate_costs["512"]["data_verification_seconds"]
            )
            prep_adjustment = (
                float(candidate["data_preparation_seconds"])
                - base_prep
                + float(candidate["data_verification_seconds"])
                - base_verification
            )
            trained_runs = {
                "source": 6,
                "preparation": 6,
                "native": 6,
                "adapter": 24,
            }
            setup = (
                3
                * (
                    float(source_row["setup_and_data_preparation_seconds"])
                    + float(dense_row["setup_and_data_preparation_seconds"])
                )
                + 3
                * (
                    float(prep_row["setup_and_data_preparation_seconds"])
                    + float(dense_row["setup_and_data_preparation_seconds"])
                )
                + 3
                * (
                    float(native_row["setup_and_data_preparation_seconds"])
                    + float(source_row["setup_and_data_preparation_seconds"])
                )
                + 12
                * (
                    float(adapter64_row["setup_and_data_preparation_seconds"])
                    + float(adapter128_row["setup_and_data_preparation_seconds"])
                )
                + sum(trained_runs.values()) * prep_adjustment
            )
            update_cost = (
                3
                * updates["source"]
                * (measured["source-byte-128"] + measured["dense-128"])
                + 3
                * updates["preparation"]
                * (measured["preparation-64"] + measured["dense-128"])
                + 3
                * updates["native"]
                * (measured["native-64"] + measured["source-byte-128"])
                + 12
                * updates["adapter"]
                * (measured["adapter-64"] + measured["adapter-128"])
            )
            checkpoint_cost = (
                3
                * schedule_counts["source_native"]
                * (
                    float(source_row["checkpoint_seconds_per_pilot"])
                    + float(dense_row["checkpoint_seconds_per_pilot"])
                )
                / 2
                + 3
                * schedule_counts["preparation"]
                * (
                    float(prep_row["checkpoint_seconds_per_pilot"])
                    + float(dense_row["checkpoint_seconds_per_pilot"])
                )
                / 2
                + 3
                * schedule_counts["source_native"]
                * (
                    float(native_row["checkpoint_seconds_per_pilot"])
                    + float(source_row["checkpoint_seconds_per_pilot"])
                )
                / 2
                + 12
                * schedule_counts["adapter"]
                * (
                    float(adapter64_row["checkpoint_seconds_per_pilot"])
                    + float(adapter128_row["checkpoint_seconds_per_pilot"])
                )
                / 2
            )
            training_evaluation_cost = (
                3
                * (
                    float(source_row["evaluation_seconds_per_pilot"])
                    + float(dense_row["evaluation_seconds_per_pilot"])
                )
                + 3
                * (
                    float(prep_row["evaluation_seconds_per_pilot"])
                    + float(dense_row["evaluation_seconds_per_pilot"])
                )
                + 3
                * (
                    float(native_row["evaluation_seconds_per_pilot"])
                    + float(source_row["evaluation_seconds_per_pilot"])
                )
                + 12
                * (
                    float(adapter64_row["evaluation_seconds_per_pilot"])
                    + float(adapter128_row["evaluation_seconds_per_pilot"])
                )
            )
            evidence_evaluation_cost = (
                6
                * schedule_counts["source_native"]
                * float(candidate["source_monitor_evaluation_seconds"])
                + 6
                * schedule_counts["preparation"]
                * float(candidate["preparation_evaluation_seconds"])
                + 6
                * schedule_counts["source_native"]
                * float(candidate["recipient_query_evaluation_seconds"])
                + 24
                * schedule_counts["adapter"]
                * float(candidate["recipient_query_evaluation_seconds"])
                + 12 * float(candidate["recipient_query_evaluation_seconds"])
                + 3 * 2 * float(candidate["source_monitor_evaluation_seconds"])
            )
            raw = (
                update_cost
                + setup
                + checkpoint_cost
                + training_evaluation_cost
                + evidence_evaluation_cost
            )
            data_inventory_bytes = int(candidate["data_inventory_bytes"])
            data_growth_bytes = max(0, data_inventory_bytes - base_inventory_bytes)
            projected_host_memory_bytes = math.ceil(
                1.25 * (observed_training_bytes + data_growth_bytes)
            )
            host_memory_feasible = (
                projected_host_memory_bytes <= resource_capacity["host_available_bytes"]
            )
            checkpoint_projection_bytes = (
                (schedule_counts["source_native"] + 1)
                * 3
                * (
                    checkpoint_bytes_by_role["source-byte-128"]
                    + checkpoint_bytes_by_role["dense-128"]
                )
                + (schedule_counts["preparation"] + 1)
                * 3
                * (
                    checkpoint_bytes_by_role["preparation-64"]
                    + checkpoint_bytes_by_role["dense-128"]
                )
                + (schedule_counts["source_native"] + 1)
                * 3
                * (
                    checkpoint_bytes_by_role["native-64"]
                    + checkpoint_bytes_by_role["source-byte-128"]
                )
                + (schedule_counts["adapter"] + 1)
                * 12
                * (
                    checkpoint_bytes_by_role["adapter-64"]
                    + checkpoint_bytes_by_role["adapter-128"]
                )
            )
            audit_scale = count / 512
            audit_projection_bytes = math.ceil(
                sum(
                    multiplicity
                    * (
                        audit_fixed_bytes_by_role[role] * audit_scale
                        + audit_bytes_per_update_by_role[role] * updates[update_role]
                    )
                    for role, multiplicity, update_role in (
                        ("source-byte-128", 3, "source"),
                        ("dense-128", 3, "source"),
                        ("preparation-64", 3, "preparation"),
                        ("dense-128", 3, "preparation"),
                        ("native-64", 3, "native"),
                        ("source-byte-128", 3, "native"),
                        ("adapter-64", 12, "adapter"),
                        ("adapter-128", 12, "adapter"),
                    )
                )
            )
            data_bundle_bytes = data_inventory_bytes * data_bundle_count
            portable_memory_bytes = portable_table_bytes * memory_package_count
            projected_disk_bytes = math.ceil(
                1.25
                * (
                    checkpoint_projection_bytes
                    + audit_projection_bytes
                    + data_bundle_bytes
                    + portable_memory_bytes
                )
            )
            disk_feasible = (
                projected_disk_bytes <= resource_capacity["campaign_volume_free_bytes"]
            )
            time_feasible = 1.25 * raw <= 0.80 * remaining
            resource_projection = {
                "host_memory": {
                    "observed_training_peak_bytes": observed_training_bytes,
                    "candidate_inventory_growth_bytes": data_growth_bytes,
                    "projected_bytes_with_margin": projected_host_memory_bytes,
                    "available_bytes": resource_capacity["host_available_bytes"],
                    "feasible": host_memory_feasible,
                },
                "disk": {
                    "checkpoint_bytes": checkpoint_projection_bytes,
                    "audit_bytes": audit_projection_bytes,
                    "owned_data_bundle_bytes": data_bundle_bytes,
                    "portable_memory_package_bytes": portable_memory_bytes,
                    "projected_bytes_with_margin": projected_disk_bytes,
                    "available_bytes": resource_capacity["campaign_volume_free_bytes"],
                    "feasible": disk_feasible,
                },
            }
            candidate_projection.append(
                {
                    "policy": policy,
                    "fact_count": count,
                    "updates": updates,
                    "schedule_boundaries": schedule_counts,
                    "components_seconds": {
                        "updates": update_cost,
                        "setup_and_data_preparation": setup,
                        "checkpoints": checkpoint_cost,
                        "training_evaluation": training_evaluation_cost,
                        "query_evaluation_and_gates": evidence_evaluation_cost,
                    },
                    "raw_seconds": raw,
                    "expanded_seconds": 1.25 * raw,
                    "data_inventory_bytes": data_inventory_bytes,
                    "resource_projection": resource_projection,
                    "time_feasible": time_feasible,
                    "feasible": (
                        time_feasible and host_memory_feasible and disk_feasible
                    ),
                }
            )
    feasible = [row for row in candidate_projection if row["feasible"]]
    full = [row for row in feasible if row["policy"] == "full"]
    choices = full or [row for row in feasible if row["policy"] == "half"]
    if not choices:
        raise RuntimeError("resource_blocked")
    choice = max(choices, key=lambda row: int(row["fact_count"]))
    _write_immutable_json(
        selection_path,
        {
            "experiment": _EXPERIMENT,
            "protocol_sha256": protocol["sha256"],
            "status": "sealed",
            "backend": pilots["backend"],
            "fact_count": choice["fact_count"],
            "policy": choice["policy"],
            "updates": choice["updates"],
            "timing_pilots": _descriptor(root, pilots_path),
            "candidate_costs": candidate_costs,
            "projections": candidate_projection,
            "resource_capacity": resource_capacity,
            "resource_margin": 1.25,
            "selected_resource_projection": choice["resource_projection"],
            "resource_formula": (
                "1.25*projected_host_memory <= measured_available_host_memory "
                "and 1.25*projected_disk <= measured_campaign_volume_free_space"
            ),
            "formula": "1.25*projected_remaining <= 0.80*remaining_allowance",
            "build_elapsed_seconds": build_elapsed,
            "prior_elapsed_seconds": prior_elapsed,
            "calibration_elapsed_seconds": pilot_seconds,
            "accounted_elapsed_seconds": accounted_elapsed,
            "remaining_allowance_seconds": remaining,
        },
    )
    return _selection(root, protocol)


def _directory_descriptor(root: Path, path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"learned portability bundle is not a directory: {path}")
    files = [
        _descriptor(path, item) for item in sorted(path.rglob("*")) if item.is_file()
    ]
    if not files or any(item.is_symlink() for item in path.rglob("*")):
        raise ValueError("learned portability bundle is empty or unsafe")
    return {"path": path.relative_to(root).as_posix(), "files": files}


def _addressing() -> dict[str, object]:
    return {
        "format_version": 1,
        "normalization": "raw-utf8-v1",
        "hashing": "poly257-terminal-v1",
        "ngram_size": 32,
        "table_size": 65521,
        "embedding_dim": 32,
    }


def build_learned_run_manifest(
    campaign_root: Path,
    coordinate: dict[str, object],
    *,
    portable_artifact: Path | None = None,
    backbone: dict[str, object] | None = None,
    purpose: str = "scientific",
) -> Path:
    """Create an immutable run-owned v2 contract and exact owned bundle."""
    import shutil

    root = Path(campaign_root)
    protocol = _read_json(
        root / "portability_protocol.json", "learned portability protocol"
    )
    coordinate_id = _coordinate_id(coordinate)
    planned = {_coordinate_id(row): row for row in _coordinate_plan()}.get(
        coordinate_id
    )
    timing_widths = {
        "source-byte-128": "source",
        "dense-128": "source",
        "native-64": "width64",
        "preparation-64": "width64",
        "adapter-64": "width64",
        "adapter-128": "width128",
    }
    if purpose == "scientific" and planned != coordinate:
        raise ValueError("coordinate is not in the sealed learned campaign plan")
    if purpose == "timing" and coordinate != {
        "role": "calibration",
        "condition": coordinate.get("condition"),
        "recipient": timing_widths.get(str(coordinate.get("condition"))),
        "seed": 20260925,
    }:
        raise ValueError(
            "timing coordinate is not one of six sealed calibration pilots"
        )
    if purpose not in {"scientific", "timing"}:
        raise ValueError("learned run purpose must be scientific or timing")
    seed = int(coordinate["seed"])
    role, condition = str(coordinate["role"]), str(coordinate["condition"])
    recipient = str(coordinate["recipient"])
    selection = None if purpose == "timing" else _selection(root, protocol)
    selected = (
        min(protocol["resource_selection"]["candidate_fact_counts"])
        if purpose == "timing"
        else str(selection["fact_count"])
    )
    data_source = root / protocol["data"][str(selected)]["path"]
    manifest_root = root / "execution" / "run-manifests" / coordinate_id
    manifest = manifest_root / "manifest.json"
    if manifest.exists():
        return manifest
    bundle = manifest_root / "portability"
    bundle.mkdir(parents=True)
    shutil.copytree(data_source.parent, bundle / "data")
    tokenizer = bundle / "data" / "tokenizer" / "tokenizer.json"
    if not tokenizer.is_file():
        raise ValueError("learned data bundle lacks tokenizer.json")
    shutil.copy2(tokenizer, manifest_root / "tokenizer.json")
    shutil.copy2(root / "portability_protocol.json", bundle / "protocol.json")
    data = _read_json(bundle / "data" / "manifest.json", "bundled learned data")
    observations_root = bundle / "observations"
    observations_root.mkdir()
    schedule_name = (
        "source_native"
        if role in {"source", "native"}
        or condition in {"source-byte-128", "dense-128", "native-64"}
        else "preparation"
        if role == "preparation" or condition == "preparation-64"
        else "adapter"
    )
    budget_key = (
        "adapter"
        if role == "recipient" or condition in _ADAPTER_CONDITIONS
        else "preparation"
        if role == "preparation"
        else role
    )
    step_limit = 16 if purpose == "timing" else int(selection["updates"][budget_key])
    observed_steps = (
        {0}
        if role == "recipient" and condition in {"baseline", "real-zero-shot"}
        else {
            int(step)
            for step in protocol["observation_steps"][schedule_name]
            if int(step) <= step_limit
        }
        | {step_limit}
    )
    schedule = {
        "format": "klab-portability-observation-schedule",
        "version": 1,
        "role": role,
        "condition": condition,
        "steps": sorted(observed_steps),
        "query_partitions": (
            ["source_monitor"]
            if role == "source"
            else ["calibration", "held_out"]
            if role in {"recipient", "native"}
            else []
        ),
        "source_monitor_partitions": ["source_monitor"] if role == "source" else [],
        "scorer": "portability/data/scorer.jsonl",
        "ownership": "portability/data/ownership.json",
    }
    schedule_path = _write_immutable_json(observations_root / "schedule.json", schedule)

    if role in {"recipient", "native"}:
        if backbone is None:
            raise ValueError("recipient/native run requires its preparation checkpoint")
        run_root = backbone.get("run_root")
        checkpoint_root = backbone.get("checkpoint_root")
        if not isinstance(run_root, Path) or not isinstance(checkpoint_root, Path):
            raise ValueError("preparation backbone paths are invalid")
        target_root = bundle / "backbone"
        target_root.mkdir()
        shutil.copy2(run_root / "manifest.json", target_root / "manifest.json")
        target_checkpoint = target_root / "checkpoints" / checkpoint_root.name
        target_checkpoint.parent.mkdir()
        shutil.copytree(checkpoint_root, target_checkpoint)
        checkpoint_manifest = _read_json(
            target_checkpoint / "manifest.json", "preparation checkpoint"
        )
        run_manifest_path = target_root / "manifest.json"
        from klab.training.manifest import read_manifest

        prepared_manifest = read_manifest(run_manifest_path)
        backbone = {
            "checkpoint": _directory_descriptor(manifest_root, target_checkpoint),
            "checkpoint_sha256": checkpoint_manifest["sha256"],
            "architecture_sha256": checkpoint_manifest["architecture_sha256"],
            "tokenizer_sha256": sha256_file(manifest_root / "tokenizer.json"),
            "run_manifest": _descriptor(manifest_root, run_manifest_path),
        }
        if (
            prepared_manifest.get("architecture_sha256")
            != backbone["architecture_sha256"]
        ):
            raise ValueError("preparation checkpoint architecture binding differs")
    else:
        if backbone is not None:
            raise ValueError(
                "source/preparation runs cannot bind a preparation checkpoint"
            )
        backbone = None

    ownership = data["ownership"]
    prep = data["preparation"]
    all_facts = [
        fact_id for section in ownership.values() for fact_id in section["fact_ids"]
    ]
    if purpose == "timing":
        training_fact_ids = (
            all_facts
            if condition in {"source-byte-128", "dense-128", "native-64"}
            else list(ownership["calibration"]["fact_ids"])
            if condition in {"adapter-64", "adapter-128"}
            else list(prep["training_fact_ids"])
        )
    elif role in {"source", "native"}:
        training_fact_ids = all_facts
    elif role == "preparation":
        training_fact_ids = list(prep["training_fact_ids"])
    elif condition in _ADAPTER_CONDITIONS:
        training_fact_ids = list(ownership["calibration"]["fact_ids"])
    else:
        training_fact_ids = []

    memory_path = None
    tensor_sha256 = None
    transferred = (
        purpose == "scientific"
        and role == "recipient"
        and condition
        in {
            "constant",
            "random",
            "permuted",
            "real-zero-shot",
            "real-adapter",
        }
    )
    if transferred:
        if portable_artifact is None:
            raise ValueError(f"{condition} requires an exported source artifact")
        memory_dir = bundle / "memory"
        memory_dir.mkdir()
        memory_path = memory_dir / "source.engram"
        shutil.copy2(portable_artifact, memory_path)
        from klab.model.portable_engram import load_portable_engram

        tensor_sha256 = load_portable_engram(memory_path).manifest.table_sha256

    source = None
    if transferred:
        source_row = next(
            row
            for row in _coordinate_plan()
            if row["role"] == "source"
            and row["condition"] == "source-real"
            and row["seed"] == seed
        )
        source_entry = _outcome_records(root).get(_coordinate_id(source_row))
        source_outcome = source_entry.get("outcome") if source_entry else None
        if (
            not isinstance(source_outcome, dict)
            or source_outcome.get("status") != "completed"
            or not _source_gate(source_outcome)[0]
        ):
            raise ValueError("recipient source gate has not passed")
        source_run = root / "runs" / str(source_outcome["run_id"])
        source_audit = source_run / "portability_audit.json"
        source_audit_value = _read_learned_run_audit(
            root, str(source_outcome["run_id"])
        )
        if source_audit_value.get("run_status") != "completed":
            raise ValueError("source audit does not prove completed training")
        source_manifest = source_run / "manifest.json"
        pointer_path = source_run / "checkpoints" / "latest.json"
        pointer = _verified_checkpoint_pointer(
            source_run, pointer_path, "source checkpoint pointer"
        )
        generation = pointer.get("relative_path")
        if (
            not isinstance(generation, str)
            or Path(generation).is_absolute()
            or ".." in Path(generation).parts
        ):
            raise ValueError("source checkpoint pointer is unsafe")
        source_checkpoint = source_run / "checkpoints" / generation
        source_checkpoint_manifest = source_checkpoint / "manifest.json"
        provenance_path = root / "exports" / f"source-s{seed}.provenance.json"
        _read_json(provenance_path, "source export provenance")
        source_dir = bundle / "source"
        source_dir.mkdir()
        gate_path = _write_immutable_json(
            source_dir / "gate.json",
            {
                "status": "passed",
                "coordinate_id": _coordinate_id(source_row),
                "source_gate": source_outcome,
            },
        )
        for source_path, name in (
            (source_manifest, "run_manifest.json"),
            (source_checkpoint_manifest, "checkpoint_manifest.json"),
            (source_audit, "audit.json"),
            (provenance_path, "provenance.json"),
        ):
            if source_path.is_symlink() or not source_path.is_file():
                raise ValueError(f"source provenance file is missing: {source_path}")
            shutil.copy2(source_path, source_dir / name)
        from klab.model.portable_engram import load_portable_engram

        original_package = load_portable_engram(
            root / "exports" / f"source-s{seed}.engram"
        )
        source_manifest_value = read_manifest(source_dir / "run_manifest.json")
        source_checkpoint_value = _read_json(
            source_dir / "checkpoint_manifest.json", "source checkpoint manifest"
        )
        source = {
            "run_id": str(source_outcome["run_id"]),
            "checkpoint_sha256": source_checkpoint_value["sha256"],
            "architecture_sha256": source_manifest_value["architecture_sha256"],
            "table_sha256": original_package.manifest.table_sha256,
            "gate": _descriptor(manifest_root, gate_path),
            "run_manifest": _descriptor(
                manifest_root, source_dir / "run_manifest.json"
            ),
            "checkpoint_manifest": _descriptor(
                manifest_root, source_dir / "checkpoint_manifest.json"
            ),
            "audit": _descriptor(manifest_root, source_dir / "audit.json"),
            "provenance": _descriptor(manifest_root, source_dir / "provenance.json"),
        }
        if source["checkpoint_sha256"] != pointer.get("manifest_sha256"):
            raise ValueError("source checkpoint pointer differs from provenance")

    external_coordinate = {
        "role": role,
        "recipient": recipient,
        "representation": "byte",
        "condition": condition,
    }
    families = (
        ["source-backbone", "source-memory"]
        if role == "source" and condition == "source-real"
        else ["source-backbone"]
        if role == "source"
        else ["preparation-backbone"]
        if role == "preparation"
        else ["native-memory"]
        if role == "native"
        else ["recipient-adapter"]
        if role == "recipient"
        else {
            "source-byte-128": ["source-backbone", "source-memory"],
            "dense-128": ["source-backbone"],
            "native-64": ["native-memory"],
            "preparation-64": ["preparation-backbone"],
            "adapter-64": ["recipient-adapter"],
            "adapter-128": ["recipient-adapter"],
        }[condition]
    )
    observations = {
        "schedule": _descriptor(manifest_root, schedule_path),
        "queries": _descriptor(
            manifest_root, bundle / "data" / data["queries"]["path"]
        ),
        "scorer": _descriptor(manifest_root, bundle / "data" / data["scorer"]["path"]),
        "ownership": _descriptor(
            manifest_root, bundle / "data" / data["ownership_file"]["path"]
        ),
    }
    memory = {
        "kind": "byte",
        "artifact": None
        if memory_path is None
        else _descriptor(manifest_root, memory_path),
        "pack_id": None,
        "tensor_sha256": tensor_sha256,
        "addressing": _addressing(),
        "encoder_contract": None,
        "replacements": [],
    }
    return _write_immutable_json(
        manifest,
        {
            "format": "klab-portability-run",
            "version": 2,
            "experiment": _EXPERIMENT,
            "protocol": _descriptor(manifest_root, bundle / "protocol.json"),
            "world_manifest": _descriptor(
                manifest_root, bundle / "data" / "manifest.json"
            ),
            "coordinate": external_coordinate,
            "seed": seed,
            "initialization": {
                "purpose": purpose,
                "derivation_version": "init-v1",
                "pair_seed": seed,
                "width": 128
                if recipient == "source" or recipient == "width128"
                else 64,
                "families": families,
            },
            "initial_backbone": backbone,
            "memory": memory,
            "source": source,
            "observations": observations,
            "training_fact_ids": training_fact_ids,
        },
    )


def learned_portability_plan(campaign_root: Path) -> dict[str, object]:
    """Return a verified, read-only plan; it performs neither calibration nor training."""
    root = Path(campaign_root)
    protocol = _read_json(
        root / "portability_protocol.json", "learned portability protocol"
    )
    plan = _read_json(root / "plan.json", "learned portability plan")
    if (
        protocol.get("experiment") != _EXPERIMENT
        or plan.get("experiment") != _EXPERIMENT
    ):
        raise ValueError("campaign is not a learned Engram portability campaign")
    selection = protocol.get("resource_selection")
    if not isinstance(selection, dict):
        raise ValueError("learned protocol lacks resource selection")
    return {
        "experiment": _EXPERIMENT,
        "campaign_id": protocol["campaign_id"],
        "protocol_sha256": protocol["sha256"],
        "coordinates": plan["coordinates"],
        "resource_selection": selection,
        "timing": "uncalibrated"
        if selection.get("status") == "uncalibrated"
        else selection.get("status"),
        "prerequisites": [
            "all three source gates must pass before export or recipient work",
            "timing calibration must seal a resource selection before behavioral execution",
        ],
    }


def _receipt(root: Path) -> dict[str, Any] | None:
    index = root / "execution" / "receipt-index.json"
    if not index.exists():
        return None
    value = _read_json(index, "learned receipt index")
    revision = value.get("current")
    if not isinstance(revision, str):
        raise ValueError("learned receipt index has no current revision")
    return _read_json(
        root / "execution" / "receipts" / revision, "learned execution receipt"
    )


def _publish_receipt(
    root: Path,
    protocol: dict[str, Any],
    coordinates: list[dict[str, Any]],
    *,
    elapsed: float,
    allowance: float,
) -> Path:
    attempts = []
    attempt_root = root / "execution" / "attempts"
    if attempt_root.exists():
        for path in sorted(attempt_root.glob("*/*.json")):
            if path.is_symlink() or not path.is_file():
                raise ValueError("learned attempt inventory contains an unsafe path")
            item = _read_json(path, "learned attempt")
            attempts.append(
                {
                    "artifact": _descriptor(root, path),
                    "coordinate_id": item.get("coordinate_id"),
                    "run_id": item.get("run_id"),
                    "status": item.get("status"),
                    "started_at": item.get("started_at"),
                }
            )
    attempts.sort(
        key=lambda item: (str(item["started_at"]), str(item["artifact"]["path"]))
    )
    timing_path = root / "execution" / "timing-pilots.json"
    timing_artifact = _descriptor(root, timing_path) if timing_path.exists() else None
    body = {
        "format": _RECEIPT_FORMAT,
        "version": 2,
        "experiment": _EXPERIMENT,
        "campaign_id": protocol["campaign_id"],
        "protocol_sha256": protocol["sha256"],
        "coordinates": coordinates,
        "attempts": attempts,
        "timing_pilots": timing_artifact,
        "elapsed_seconds": elapsed,
        "remaining_seconds": max(0.0, allowance - elapsed),
    }
    digest = canonical_digest(body)
    receipt = _write_immutable_json(
        root / "execution" / "receipts" / f"{digest}.json", body
    )
    index = root / "execution" / "receipt-index.json"
    revisions: list[str] = []
    if index.exists():
        existing = _read_json(index, "learned receipt index")
        stored = existing.get("revisions")
        if not isinstance(stored, list) or not all(
            isinstance(item, str) for item in stored
        ):
            raise ValueError("learned receipt index is malformed")
        revisions = stored
    index_body = {"current": receipt.name, "revisions": [*revisions, receipt.name]}
    temporary = index.with_suffix(".tmp")
    temporary.write_bytes(
        canonical_json({**index_body, "sha256": canonical_digest(index_body)}) + b"\n"
    )
    os.replace(temporary, index)
    return receipt


def _outcome_records(root: Path) -> dict[str, dict[str, Any]]:
    directory = root / "execution" / "outcomes"
    if not directory.exists():
        return {}
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("learned outcome directory is unsafe")
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        record = _read_json(path, "learned coordinate outcome")
        coordinate_id = record.get("coordinate_id")
        if not isinstance(coordinate_id, str) or coordinate_id in records:
            raise ValueError("learned coordinate outcome inventory is malformed")
        records[coordinate_id] = record
    return records


def record_learned_coordinate_outcome(
    campaign_root: Path, coordinate: dict[str, object], outcome: dict[str, object]
) -> Path:
    """Publish one immutable coordinate outcome for the execution owner.

    Source outcomes are never overwritten: recovery publishes a new child attempt
    before this final outcome is recorded, and duplicate logical outcomes fail closed.
    """
    root = Path(campaign_root)
    protocol = _read_json(
        root / "portability_protocol.json", "learned portability protocol"
    )
    if protocol.get("experiment") != _EXPERIMENT:
        raise ValueError("campaign is not a learned Engram portability campaign")
    coordinate_id = _coordinate_id(coordinate)
    planned = {_coordinate_id(row): row for row in _coordinate_plan()}
    if coordinate_id not in planned or planned[coordinate_id] != coordinate:
        raise ValueError("coordinate is not in the sealed learned campaign plan")
    status = outcome.get("status")
    if status not in {"completed", "failed"}:
        raise ValueError(
            "only terminal outcomes are immutable; interruptions remain attempt records"
        )
    existing = _outcome_records(root)
    if coordinate_id in existing:
        raise FileExistsError(
            f"learned coordinate already has an immutable outcome: {coordinate_id}"
        )
    body = {
        "format": "klab-learned-portability-coordinate-outcome",
        "version": 1,
        "experiment": _EXPERIMENT,
        "protocol_sha256": protocol["sha256"],
        "coordinate": coordinate,
        "coordinate_id": coordinate_id,
        "outcome": outcome,
    }
    return _write_immutable_json(
        root / "execution" / "outcomes" / f"{coordinate_id}.json", body
    )


def _source_gate(outcome: dict[str, Any]) -> tuple[bool, str | None]:
    required = (
        "full_fact_exposure",
        "valid_update_audit",
        "finite_tensors",
        "nonzero_table_delta",
    )
    if not all(outcome.get(key) is True for key in required):
        return False, "source_audit_failed"
    if float(outcome.get("gradient_changed_row_fraction", -1.0)) < 0.90:
        return False, "source_row_ownership_gate_failed"
    if float(outcome.get("source_monitor_accuracy", -1.0)) < 0.75:
        return False, "source_monitor_gate_failed"
    if float(outcome.get("enabled_minus_disabled_accuracy", -1.0)) < 0.10:
        return False, "source_memory_accuracy_gate_failed"
    if float(outcome.get("disabled_minus_enabled_nll", -1.0)) < 0.10:
        return False, "source_memory_nll_gate_failed"
    return True, None


def _preparation_gate(outcome: dict[str, Any]) -> tuple[bool, str | None]:
    metrics = outcome.get("preparation_metrics")
    if not isinstance(metrics, dict):
        observation = outcome.get("preparation")
        metrics = observation.get("metrics") if isinstance(observation, dict) else None
    if not isinstance(metrics, dict):
        return False, "preparation_observation_missing"
    if float(metrics.get("accuracy", -1.0)) < 0.95:
        return False, "preparation_accuracy_gate_failed"
    per_symbol = metrics.get("per_symbol_accuracy")
    if not isinstance(per_symbol, dict) or not per_symbol:
        return False, "preparation_symbol_metrics_missing"
    if any(
        not isinstance(item, dict) or float(item.get("accuracy", -1.0)) < 0.75
        for item in per_symbol.values()
    ):
        return False, "preparation_per_symbol_gate_failed"
    return True, None


def _materialize_control_package(
    root: Path, source_path: Path, coordinate: dict[str, object]
) -> Path:
    import torch

    from klab.model.portable_engram import (
        export_portable_engram,
        load_portable_engram,
    )

    condition = str(coordinate["condition"])
    if condition not in {"constant", "random", "permuted"}:
        raise ValueError("control package requested for a non-control condition")
    seed = int(coordinate["seed"])
    package_path = root / "exports" / "controls" / f"source-s{seed}-{condition}.engram"
    provenance_path = package_path.with_suffix(".provenance.json")
    source_descriptor = _descriptor(root, source_path)
    if package_path.exists() or provenance_path.exists():
        if not package_path.is_file() or not provenance_path.is_file():
            raise ValueError(
                "control package and provenance must be published together"
            )
        provenance = _read_json(provenance_path, "control package provenance")
        package = load_portable_engram(package_path)
        if (
            provenance.get("experiment") != _EXPERIMENT
            or provenance.get("coordinate_id") != f"control-source-s{seed}-{condition}"
            or provenance.get("source_package") != source_descriptor
            or provenance.get("package") != _descriptor(root, package_path)
            or provenance.get("condition") != condition
            or provenance.get("seed") != seed
            or provenance.get("table_sha256") != package.manifest.table_sha256
        ):
            raise ValueError("control package provenance differs from its artifacts")
        return package_path

    source = load_portable_engram(source_path)
    table = source.table.detach().cpu().contiguous()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if condition == "constant":
        transformed = table.mean(dim=0, keepdim=True).expand_as(table).clone()
        derivation = "source-row-mean-replicated-v1"
    elif condition == "random":
        transformed = torch.randn(
            table.shape, generator=generator, dtype=table.dtype, device="cpu"
        )
        transformed.mul_(table.std(unbiased=False)).add_(table.mean())
        derivation = "source-moments-cpu-normal-v1"
    else:
        permutation = torch.randperm(table.shape[0], generator=generator)
        transformed = table.index_select(0, permutation)
        derivation = "seeded-cpu-row-permutation-v1"
    export_portable_engram(
        transformed,
        package_path,
        ngram_size=source.manifest.ngram_size,
        normalization=source.manifest.normalization,
        hashing=source.manifest.hashing,
    )
    package = load_portable_engram(package_path)
    _write_immutable_json(
        provenance_path,
        {
            "experiment": _EXPERIMENT,
            "coordinate_id": f"control-source-s{seed}-{condition}",
            "condition": condition,
            "seed": seed,
            "derivation": derivation,
            "source_package": source_descriptor,
            "source_table_sha256": source.manifest.table_sha256,
            "package": _descriptor(root, package_path),
            "table_sha256": package.manifest.table_sha256,
        },
    )
    return package_path


def learned_portability_status(campaign_root: Path) -> dict[str, object]:
    """Resolve immutable outcomes into the dependency graph without executing it."""
    root = Path(campaign_root)
    protocol = _read_json(
        root / "portability_protocol.json", "learned portability protocol"
    )
    if protocol.get("experiment") != _EXPERIMENT:
        raise ValueError("campaign is not a learned Engram portability campaign")
    outcomes = _outcome_records(root)
    source_failure: str | None = None
    source_ready = True
    source_rows = [row for row in _coordinate_plan() if row["role"] == "source"]
    for row in source_rows:
        result = outcomes.get(_coordinate_id(row))
        if result is None:
            source_ready = False
            continue
        outcome = result.get("outcome")
        if not isinstance(outcome, dict) or outcome.get("status") != "completed":
            source_failure = f"{_coordinate_id(row)}:source_execution_not_completed"
            source_ready = False
            break
        if row["condition"] == "source-real":
            passed, reason = _source_gate(outcome)
            if not passed:
                source_failure = f"{_coordinate_id(row)}:{reason}"
                source_ready = False
                break
    preparation_failures: dict[str, str] = {}
    for row in _coordinate_plan():
        if row["role"] != "preparation":
            continue
        coordinate_id = _coordinate_id(row)
        published = outcomes.get(coordinate_id)
        if published is None:
            continue
        outcome = published.get("outcome")
        if not isinstance(outcome, dict) or outcome.get("status") != "completed":
            preparation_failures[coordinate_id] = "preparation_execution_not_completed"
        else:
            passed, reason = _preparation_gate(outcome)
            if not passed:
                preparation_failures[coordinate_id] = str(reason)
    coordinates: list[dict[str, object]] = []
    for row in _coordinate_plan():
        coordinate_id = _coordinate_id(row)
        published = outcomes.get(coordinate_id)
        if published is not None:
            outcome = published["outcome"]
            assert isinstance(outcome, dict)
            reason = outcome.get("reason")
            status = str(outcome["status"])
            if row["role"] == "preparation" and coordinate_id in preparation_failures:
                status = "gate_failed"
                reason = preparation_failures[coordinate_id]
            coordinates.append(
                {
                    **row,
                    "coordinate_id": coordinate_id,
                    "status": status,
                    "reason": reason,
                }
            )
            continue
        if row["role"] != "source" and not source_ready:
            reason = (
                "source_signal_not_established"
                if source_failure is not None
                else "source_gates_pending"
            )
            coordinates.append(
                {
                    **row,
                    "coordinate_id": coordinate_id,
                    "status": "blocked",
                    "reason": reason,
                }
            )
            continue
        if row["role"] in {"recipient", "native"}:
            prep = next(
                item
                for item in _coordinate_plan()
                if item["role"] == "preparation"
                and item["seed"] == row["seed"]
                and item["recipient"] == row["recipient"]
            )
            prep_id = _coordinate_id(prep)
            if prep_id in preparation_failures:
                coordinates.append(
                    {
                        **row,
                        "coordinate_id": coordinate_id,
                        "status": "blocked",
                        "reason": preparation_failures[prep_id],
                    }
                )
                continue
            if prep_id not in outcomes:
                coordinates.append(
                    {
                        **row,
                        "coordinate_id": coordinate_id,
                        "status": "blocked",
                        "reason": "preparation_gate_pending",
                    }
                )
                continue
        coordinates.append(
            {
                **row,
                "coordinate_id": coordinate_id,
                "status": "planned",
                "reason": None,
            }
        )
    return {
        "experiment": _EXPERIMENT,
        "protocol_sha256": protocol["sha256"],
        "source_gate_failure": source_failure,
        "preparation_gate_failures": preparation_failures,
        "coordinates": coordinates,
    }


def _learned_config(
    root: Path,
    coordinate: dict[str, object],
    manifest_path: Path,
    *,
    max_steps: int,
    purpose: str = "scientific",
    backend_override: str | None = None,
) -> Any:
    """Construct the ordinary trainer config for one learned coordinate."""
    from importlib.resources import files

    from klab.config.loading import load_config
    from klab.config.models import AttentionConfig, ModelConfig, RunConfig
    from klab.model.inspection import named_tensor_inventory

    protocol = _read_json(
        root / "portability_protocol.json", "learned portability protocol"
    )
    data_root = manifest_path.parent / "portability" / "data"
    role, condition, recipient = (
        str(coordinate[key]) for key in ("role", "condition", "recipient")
    )
    width = 128 if recipient == "source" else int(recipient.removeprefix("width"))
    transferred = (
        purpose == "scientific"
        and role == "recipient"
        and condition
        in {"constant", "random", "permuted", "real-zero-shot", "real-adapter"}
    )
    local_byte = (
        (role == "source" and condition == "source-real")
        or (role == "native" and condition == "native")
        or (
            purpose == "timing"
            and condition
            in {"source-byte-128", "native-64", "adapter-64", "adapter-128"}
        )
    )
    memory = "portable" if transferred else "byte" if local_byte else "none"
    base = load_config(
        Path(str(files("klab.research").joinpath("resources/base.yaml")))
    ).model_dump(mode="python")
    base["name"] = f"learned-portability-{_coordinate_id(coordinate)}"
    base["seed"] = coordinate["seed"]
    base["tokenizer"] = {"path": manifest_path.parent / "tokenizer.json"}
    base["model"] = {
        "vocab_size": 260,
        "hidden_dim": width,
        "num_layers": 2,
        "num_heads": 4,
        "ffn_dim": 512 if width == 128 else 256,
        "max_seq_len": 128,
        "rms_norm_eps": 1e-6,
        "tie_embeddings": True,
        "ffn": "dense",
        "num_experts": 1,
        "experts_per_token": 1,
        "shared_expert": False,
        "router_aux_loss_coefficient": 0.0,
        "memory": memory,
        "memory_injection": "final",
        "memory_table_size": 65521 if memory != "none" else 0,
        "memory_ngram_size": 32 if memory != "none" else 0,
        "memory_dim": 32 if memory != "none" else 0,
        "memory_package_path": (
            manifest_path.parent / "portability" / "memory" / "source.engram"
            if transferred
            else None
        ),
        "memory_ngram_orders": [],
        "memory_hash_heads": 1,
        "semantic_memory_dim": None,
    }
    train_name = (
        "preparation_train.jsonl"
        if role == "preparation" or condition == "preparation-64"
        else "adapter_calibration.jsonl"
        if role == "recipient" or condition in {"adapter-64", "adapter-128"}
        else "source_train.jsonl"
    )
    train_path = data_root / train_name
    validation_path = data_root / "preparation_validation.jsonl"
    train_document_count = len(train_path.read_text(encoding="utf-8").splitlines())
    validation_document_count = len(
        validation_path.read_text(encoding="utf-8").splitlines()
    )
    base["dataset"] = {
        "source": "local_chat",
        "cache_dir": root / "cache",
        "train_max_documents": train_document_count,
        "validation_max_documents": validation_document_count,
        "train_max_tokens": 10_000_000,
        "validation_max_tokens": 10_000_000,
        "synthetic_seed": coordinate["seed"],
        "train_path": train_path,
        "validation_path": validation_path,
        "license": "CC0-1.0",
        "allocation_manifest_path": None,
    }
    tensors = named_tensor_inventory(
        ModelConfig.model_validate(base["model"]),
        AttentionConfig.model_validate(base["attention"]),
    )
    full_trainable = tuple(
        name
        for name, spec in tensors.items()
        if spec.trainable and spec.alias_of is None
    )
    adapter_tuning = (role == "recipient" and condition in _ADAPTER_CONDITIONS) or (
        purpose == "timing" and condition in {"adapter-64", "adapter-128"}
    )
    native_tuning = role == "native" or (
        purpose == "timing" and condition == "native-64"
    )
    trainable = (
        ("memory.output.weight", "memory.gate.weight")
        if adapter_tuning
        else (
            "memory.table.weight",
            "memory.output.weight",
            "memory.gate.weight",
        )
        if native_tuning
        else full_trainable
    )
    base["training"] = {
        "micro_batch_size": 8,
        "gradient_accumulation": 1,
        "seq_len": 128,
        "max_steps": max_steps,
        "max_tokens": max_steps * 8 * 128,
        "grad_clip_norm": 1.0,
        "neural_loss_weight": 1.0,
        "deterministic": True,
        "trainable_parameters": trainable,
        "portability_manifest_path": manifest_path,
    }
    base["optimizer"] = {
        "name": "adamw",
        "peak": 0.001,
        "floor": 0.0001,
        "warmup_steps": min(64, max_steps // 8),
        "weight_decay": 0.0,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "state_offload": False,
    }
    schedule_name = (
        "source_native"
        if role in {"source", "native"}
        or condition in {"source-byte-128", "dense-128", "native-64"}
        else "preparation"
        if role == "preparation" or condition == "preparation-64"
        else "adapter"
    )
    boundaries = {
        int(value)
        for value in protocol["observation_steps"][schedule_name]
        if type(value) is int and 0 <= value <= max_steps
    }
    boundaries.add(max_steps)
    base["evaluation"] = {
        "every_steps": max_steps,
        "max_batches": 2,
    }
    base["checkpoint"] = {
        "every_steps": None,
        "every_tokens": None,
        "every_minutes": None,
        "steps": tuple(sorted(boundaries)),
        "keep_periodic": True,
    }
    base["logging"] = {
        **base["logging"],
        "root_dir": root / "runs",
        "every_steps": max_steps,
        "architecture_diagnostics": "scalar",
    }
    backend = backend_override or _selection(root, protocol).get("backend")
    if backend not in {"cpu", "mps"}:
        backend = protocol["training"]["backend_request"]
    if backend == "auto":
        backend = "cpu"
    base["runtime"] = {
        **base["runtime"],
        "engine": "pytorch",
        "backend": backend,
        "precision": "fp32",
    }
    return RunConfig.model_validate(base)


def _resume_config_for_checkpoint(
    campaign_root: Path, requested: Any, checkpoint: Path
) -> Any:
    """Reuse the parent run's owned asset paths for a full-state child resume."""
    from klab.config.loading import load_config

    runs_root = (campaign_root / "runs").resolve()
    parent_root = checkpoint.parent.parent
    try:
        parent_root.resolve(strict=True).relative_to(runs_root)
    except (OSError, ValueError) as error:
        raise ValueError(
            "resume checkpoint is outside the campaign run store"
        ) from error
    resolved_path = parent_root / "resolved_config.yaml"
    if (
        parent_root.is_symlink()
        or resolved_path.is_symlink()
        or not resolved_path.is_file()
    ):
        raise ValueError("resume parent lacks its owned resolved configuration")
    parent = load_config(resolved_path)
    requested_payload = requested.model_dump(mode="json")
    parent_payload = parent.model_dump(mode="json")
    for section, field in (
        ("dataset", "cache_dir"),
        ("dataset", "train_path"),
        ("dataset", "validation_path"),
        ("tokenizer", "path"),
        ("training", "portability_manifest_path"),
        ("model", "memory_package_path"),
    ):
        if (
            section in requested_payload
            and section in parent_payload
            and field in requested_payload[section]
            and field in parent_payload[section]
        ):
            requested_payload[section][field] = parent_payload[section][field]
    normalized = type(requested).model_validate(requested_payload)
    requested_payload = normalized.model_dump(mode="json")
    for payload in (requested_payload, parent_payload):
        for field in ("name", "logging", "checkpoint"):
            payload.pop(field, None)
        payload.get("runtime", {}).pop("device_index", None)
    if requested_payload != parent_payload:
        raise ValueError("resume run settings differ from the saved coordinate")
    return normalized


def build_learned_portability_evidence(campaign_root: Path) -> Path:
    """Rebuild hashed scientific summaries exclusively from published artifacts."""
    root = Path(campaign_root)
    protocol_path = root / "portability_protocol.json"
    protocol = _read_json(protocol_path, "learned portability protocol")
    state = learned_portability_status(root)
    receipt = _receipt(root)
    try:
        selection: dict[str, Any] | None = _selection(root, protocol)
    except ValueError:
        selection = None
    outcomes = _outcome_records(root)
    observations: list[dict[str, object]] = []
    by_coordinate: dict[str, list[dict[str, object]]] = {}
    for coordinate_id, record in sorted(outcomes.items()):
        coordinate = record.get("coordinate")
        outcome = record.get("outcome")
        if not isinstance(coordinate, dict) or not isinstance(outcome, dict):
            raise ValueError("learned outcome lacks its coordinate or result")
        steps = outcome.get("step_observations", [])
        if not isinstance(steps, list):
            raise ValueError("learned outcome observation inventory is invalid")
        for step_result in steps:
            if (
                not isinstance(step_result, dict)
                or type(step_result.get("step")) is not int
            ):
                raise ValueError("learned observation step is malformed")
            entries = step_result.get("observations")
            if not isinstance(entries, list):
                raise ValueError("learned observation result inventory is invalid")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("learned observation reference is malformed")
                descriptor = entry.get("artifact")
                if not isinstance(descriptor, dict) or set(descriptor) != {
                    "path",
                    "sha256",
                    "size_bytes",
                }:
                    raise ValueError("learned observation descriptor is malformed")
                relative = Path(str(descriptor["path"]))
                if (
                    relative.is_absolute()
                    or relative.as_posix() != descriptor["path"]
                    or any(part in {"", ".", ".."} for part in relative.parts)
                ):
                    raise ValueError("learned observation path is unsafe")
                result_path = root / relative
                if _descriptor(root, result_path) != descriptor:
                    raise ValueError("learned observation artifact digest differs")
                result_artifact = _read_json(result_path, "learned observation result")
                if (
                    result_artifact.get("coordinate_id") != coordinate_id
                    or result_artifact.get("step") != step_result["step"]
                    or result_artifact.get("kind") != entry.get("kind")
                    or result_artifact.get("partition") != entry.get("partition")
                    or result_artifact.get("wording") != entry.get("wording")
                ):
                    raise ValueError(
                        "learned observation reference differs from artifact"
                    )
                result = result_artifact.get("result")
                metrics = result.get("metrics") if isinstance(result, dict) else None
                if not isinstance(metrics, dict) or metrics != entry.get("metrics"):
                    raise ValueError("learned observation metrics differ from artifact")
                observation = {
                    "coordinate_id": coordinate_id,
                    "coordinate": coordinate,
                    "step": step_result["step"],
                    "kind": entry["kind"],
                    "partition": entry["partition"],
                    "wording": entry["wording"],
                    "metrics": metrics,
                    "checkpoint": result_artifact.get("checkpoint"),
                    "artifact": descriptor,
                }
                observations.append(observation)
                by_coordinate.setdefault(coordinate_id, []).append(observation)

    coordinate_status = {str(row["coordinate_id"]): row for row in state["coordinates"]}
    sources: list[dict[str, object]] = []
    preparations: list[dict[str, object]] = []
    arms: list[dict[str, object]] = []
    source_gates: list[dict[str, object]] = []
    preparation_gates: list[dict[str, object]] = []
    for coordinate_id, status in coordinate_status.items():
        record = outcomes.get(coordinate_id)
        outcome = record.get("outcome") if record is not None else None
        outcome = outcome if isinstance(outcome, dict) else {}
        role = str(status["role"])
        condition = str(status["condition"])
        schedule_name = (
            "source_native"
            if role in {"source", "native"}
            else "preparation"
            if role == "preparation"
            else "adapter"
        )
        budget_key = (
            "adapter"
            if role == "recipient"
            else "preparation"
            if role == "preparation"
            else role
        )
        maximum = (
            int(selection["updates"][budget_key]) if selection is not None else None
        )
        scheduled_steps = (
            [0]
            if role == "recipient" and condition in {"baseline", "real-zero-shot"}
            else None
            if maximum is None
            else sorted(
                {
                    0,
                    *[
                        int(step)
                        for step in protocol["observation_steps"][schedule_name]
                        if int(step) <= maximum
                    ],
                    maximum,
                }
            )
        )
        item: dict[str, object] = {
            **status,
            "run_id": outcome.get("run_id"),
            "execution": outcome.get("execution"),
            "final_metrics": outcome.get("final_metrics", {}),
            "schedule_steps": scheduled_steps,
        }
        outcome_path = root / "execution" / "outcomes" / f"{coordinate_id}.json"
        if outcome_path.is_file():
            item["outcome_artifact"] = _descriptor(root, outcome_path)
        run_id = outcome.get("run_id")
        if isinstance(run_id, str):
            audit_path = root / "runs" / run_id / "portability_audit.json"
            if audit_path.is_file():
                item["audit_artifact"] = _descriptor(root, audit_path)
            run_root = root / "runs" / run_id
            for key, path in (
                ("run_manifest_artifact", run_root / "manifest.json"),
                (
                    "portability_manifest_artifact",
                    run_root / "portability_manifest.json",
                ),
                (
                    "latest_checkpoint_pointer",
                    run_root / "checkpoints" / "latest.json",
                ),
            ):
                if path.is_file():
                    item[key] = _descriptor(root, path)
            pointer_path = run_root / "checkpoints" / "latest.json"
            if pointer_path.is_file():
                pointer = _verified_checkpoint_pointer(
                    run_root, pointer_path, "learned run checkpoint pointer"
                )
                generation = pointer.get("relative_path")
                if isinstance(generation, str):
                    checkpoint_manifest_path = (
                        run_root / "checkpoints" / generation / "manifest.json"
                    )
                    if checkpoint_manifest_path.is_file():
                        item["latest_checkpoint_manifest"] = _descriptor(
                            root, checkpoint_manifest_path
                        )
        if status["role"] == "source" and status["condition"] == "source-real":
            package_path = root / "exports" / f"source-s{status['seed']}.engram"
            provenance_path = package_path.with_suffix(".provenance.json")
            if package_path.is_file():
                item["memory_package_artifact"] = _descriptor(root, package_path)
            if provenance_path.is_file():
                item["memory_package_provenance"] = _descriptor(root, provenance_path)
        if status["role"] == "recipient" and status["condition"] in {
            "constant",
            "random",
            "permuted",
            "real-zero-shot",
            "real-adapter",
        }:
            if status["condition"] in {"constant", "random", "permuted"}:
                package_path = (
                    root
                    / "exports"
                    / "controls"
                    / f"source-s{status['seed']}-{status['condition']}.engram"
                )
            else:
                package_path = root / "exports" / f"source-s{status['seed']}.engram"
            provenance_path = package_path.with_suffix(".provenance.json")
            if package_path.is_file():
                item["memory_package_artifact"] = _descriptor(root, package_path)
            if provenance_path.is_file():
                item["memory_package_provenance"] = _descriptor(root, provenance_path)
        if status["role"] == "source":
            gate_passed, gate_reason = (
                _source_gate(outcome)
                if outcome.get("status") == "completed"
                and status["condition"] == "source-real"
                else (None, None)
            )
            if status["condition"] == "source-real":
                gate = {
                    "coordinate_id": coordinate_id,
                    "status": "passed"
                    if gate_passed is True
                    else "failed"
                    if gate_passed is False
                    else "pending",
                    "reason": gate_reason,
                    "source_monitor_accuracy": outcome.get("source_monitor_accuracy"),
                    "enabled_minus_disabled_accuracy": outcome.get(
                        "enabled_minus_disabled_accuracy"
                    ),
                    "disabled_minus_enabled_nll": outcome.get(
                        "disabled_minus_enabled_nll"
                    ),
                    "gradient_changed_row_fraction": outcome.get(
                        "gradient_changed_row_fraction"
                    ),
                    "full_fact_exposure": outcome.get("full_fact_exposure"),
                    "valid_update_audit": outcome.get("valid_update_audit"),
                    "finite_tensors": outcome.get("finite_tensors"),
                    "nonzero_table_delta": outcome.get("nonzero_table_delta"),
                }
                source_gates.append(gate)
                item["gate"] = gate
            sources.append(item)
        elif status["role"] == "preparation":
            gate_passed, gate_reason = (
                _preparation_gate(outcome)
                if outcome.get("status") == "completed"
                else (None, None)
            )
            gate = {
                "coordinate_id": coordinate_id,
                "status": "passed"
                if gate_passed is True
                else "failed"
                if gate_passed is False
                else "pending",
                "reason": gate_reason,
                "metrics": outcome.get("preparation_metrics"),
            }
            preparation_gates.append(gate)
            item["gate"] = gate
            preparations.append(item)
        else:
            arms.append(item)

    def contrast(
        name: str,
        left_condition: str,
        right_condition: str,
        partition: str,
        wording: str,
        metric_name: str,
    ) -> dict[str, object]:
        matched: list[dict[str, object]] = []
        for seed in protocol["pair_seeds"]:
            for recipient in ("width64", "width128"):
                left_row = next(
                    (
                        row
                        for row in state["coordinates"]
                        if row["seed"] == seed
                        and row["recipient"] == recipient
                        and row["condition"] == left_condition
                        and row["role"]
                        == ("native" if left_condition == "native" else "recipient")
                    ),
                    None,
                )
                right_row = next(
                    (
                        row
                        for row in state["coordinates"]
                        if row["seed"] == seed
                        and row["recipient"] == recipient
                        and row["condition"] == right_condition
                        and row["role"]
                        == ("native" if right_condition == "native" else "recipient")
                    ),
                    None,
                )
                if left_row is None or right_row is None:
                    continue
                left_id, right_id = (
                    str(left_row["coordinate_id"]),
                    str(right_row["coordinate_id"]),
                )
                left_observation = max(
                    (
                        item
                        for item in by_coordinate.get(left_id, [])
                        if item["kind"] == "query"
                        and item["partition"] == partition
                        and item["wording"] == wording
                    ),
                    key=lambda item: int(item["step"]),
                    default=None,
                )
                right_observation = max(
                    (
                        item
                        for item in by_coordinate.get(right_id, [])
                        if item["kind"] == "query"
                        and item["partition"] == partition
                        and item["wording"] == wording
                    ),
                    key=lambda item: int(item["step"]),
                    default=None,
                )
                left_metrics = (
                    left_observation.get("metrics")
                    if isinstance(left_observation, dict)
                    else None
                )
                right_metrics = (
                    right_observation.get("metrics")
                    if isinstance(right_observation, dict)
                    else None
                )
                left_value = (
                    left_metrics.get(metric_name)
                    if isinstance(left_metrics, dict)
                    else None
                )
                right_value = (
                    right_metrics.get(metric_name)
                    if isinstance(right_metrics, dict)
                    else None
                )
                if (
                    isinstance(left_value, (int, float))
                    and not isinstance(left_value, bool)
                    and isinstance(right_value, (int, float))
                    and not isinstance(right_value, bool)
                ):
                    matched.append(
                        {
                            "seed": seed,
                            "recipient": recipient,
                            "left_coordinate_id": left_id,
                            "right_coordinate_id": right_id,
                            "left": float(left_value),
                            "right": float(right_value),
                            "delta": float(left_value) - float(right_value),
                        }
                    )
        return {
            "name": name,
            "left_condition": left_condition,
            "right_condition": right_condition,
            "partition": partition,
            "wording": wording,
            "metric": metric_name,
            "matched_pairs": len(matched),
            "per_seed": matched,
            "mean_delta": (
                sum(float(item["delta"]) for item in matched) / len(matched)
                if matched
                else None
            ),
        }

    contrasts: list[dict[str, object]] = []
    query_sets = (
        ("calibration", "adapter_return"),
        ("held_out", "final_report"),
        ("held_out", "final_state"),
    )
    contrast_pairs = (
        ("artifact_portability", "real-zero-shot", "baseline"),
        ("adapter_gain", "real-adapter", "real-zero-shot"),
        ("constant_control", "real-adapter", "constant"),
        ("random_control", "real-adapter", "random"),
        ("permuted_control", "real-adapter", "permuted"),
        ("native_memory", "native", "real-adapter"),
    )
    for name, left_condition, right_condition in contrast_pairs:
        for partition, wording in query_sets:
            for metric_name in ("accuracy", "answer_nll"):
                contrasts.append(
                    contrast(
                        name,
                        left_condition,
                        right_condition,
                        partition,
                        wording,
                        metric_name,
                    )
                )

    source_ready = (
        len(source_gates) == len(protocol["pair_seeds"])
        and all(item["status"] == "passed" for item in source_gates)
        and all(item["status"] == "completed" for item in sources)
    )
    adapter_pairs = [
        item
        for item in contrasts
        if item["name"] == "adapter_gain"
        and item["metric"] == "accuracy"
        and item["partition"] == "held_out"
    ]
    artifact_pairs = [
        item
        for item in contrasts
        if item["name"] == "artifact_portability"
        and item["metric"] == "accuracy"
        and item["partition"] == "held_out"
    ]
    artifact_status = (
        "blocked"
        if state["source_gate_failure"] is not None
        else "observed"
        if any(int(item["matched_pairs"]) for item in artifact_pairs)
        else "incomplete"
        if source_ready
        else "unavailable"
    )
    adapter_status = (
        "blocked"
        if state["source_gate_failure"] is not None
        else "observed"
        if any(int(item["matched_pairs"]) for item in adapter_pairs)
        else "incomplete"
        if source_ready
        else "unavailable"
    )
    selection_path = root / "execution" / "resource-selection.json"
    selection_descriptor = (
        _descriptor(root, selection_path) if selection_path.is_file() else None
    )
    selection_record = selection
    receipt_index = root / "execution" / "receipt-index.json"
    receipt_descriptor = None
    if receipt is not None and receipt_index.is_file():
        index = _read_json(receipt_index, "learned receipt index")
        receipt_path = root / "execution" / "receipts" / str(index["current"])
        if _read_json(receipt_path, "learned execution receipt") != receipt:
            raise ValueError("learned receipt index does not identify current receipt")
        receipt_descriptor = _descriptor(root, receipt_path)
    data_descriptor = None
    if selection is not None:
        data_path = root / protocol["data"][str(selection["fact_count"])]["path"]
        data_descriptor = _descriptor(root, data_path)
    timing_path = root / "execution" / "timing-pilots.json"
    timing_descriptor = (
        _descriptor(root, timing_path) if timing_path.is_file() else None
    )
    body = {
        "format": "klab-portability-evidence",
        "version": 2,
        "experiment": _EXPERIMENT,
        "campaign_id": protocol["campaign_id"],
        "protocol": _descriptor(root, protocol_path),
        "data": {
            "fact_count": None if selection is None else selection["fact_count"],
            "manifest": data_descriptor,
        },
        "resource_selection": selection_record,
        "resource_selection_artifact": selection_descriptor,
        "timing_pilots": timing_descriptor,
        "receipt": receipt_descriptor,
        "source_gate_failure": state["source_gate_failure"],
        "preparation_gate_failures": state["preparation_gate_failures"],
        "sources": sources,
        "source_gates": source_gates,
        "preparations": preparations,
        "preparation_gates": preparation_gates,
        "arms": arms,
        "observations": observations,
        "contrasts": contrasts,
        "thresholds": {
            "source": protocol["gates"],
            "source_results": source_gates,
            "preparation_results": preparation_gates,
        },
        "conclusions": {
            "artifact_portability": {
                "status": artifact_status,
                "contrasts": artifact_pairs,
            },
            "adapter_portability": {
                "status": adapter_status,
                "contrasts": adapter_pairs,
            },
            "representation_portability": {
                "status": "not_tested",
                "reason": "The protocol fixes byte-addressed memory.",
            },
        },
        "costs": {
            "elapsed_seconds": None if receipt is None else receipt["elapsed_seconds"],
            "remaining_seconds": None
            if receipt is None
            else receipt["remaining_seconds"],
            "resource_selection": selection_record,
            "timing_pilots": timing_descriptor,
        },
        "outcome_counts": {
            "planned": len(state["coordinates"]),
            "completed": sum(
                row["status"] == "completed" for row in state["coordinates"]
            ),
            "failed": sum(
                row["status"] in {"failed", "gate_failed"}
                for row in state["coordinates"]
            ),
            "blocked": sum(row["status"] == "blocked" for row in state["coordinates"]),
        },
        "limitations": list(protocol["limitations"])
        + [
            "Contrasts are matched descriptive differences, not significance tests or winner claims.",
            "A smoke-scale campaign verifies execution and evidence wiring only.",
            "Representation portability is not tested by this byte-only protocol.",
        ],
    }
    digest = canonical_digest(body)
    evidence_dir = root / "execution" / "evidence"
    evidence_path = _write_immutable_json(evidence_dir / f"{digest}.json", body)
    index_path = root / "execution" / "evidence-index.json"
    revisions: list[str] = []
    if index_path.exists():
        index = _read_json(index_path, "learned evidence index")
        stored = index.get("revisions")
        if not isinstance(stored, list) or not all(
            isinstance(item, str) for item in stored
        ):
            raise ValueError("learned evidence index is malformed")
        revisions = stored
    if evidence_path.name not in revisions:
        revisions.append(evidence_path.name)
    index_body = {"current": evidence_path.name, "revisions": revisions}
    temporary = index_path.with_suffix(".tmp")
    temporary.write_bytes(
        canonical_json({**index_body, "sha256": canonical_digest(index_body)}) + b"\n"
    )
    os.replace(temporary, index_path)
    return evidence_path


def execute_learned_portability_campaign(
    campaign_root: Path,
    *,
    max_wall_seconds: float = 43200,
    continue_existing: bool = False,
) -> dict[str, object]:
    """Execute sealed scientific coordinates and publish observed evidence."""
    import uuid
    from datetime import UTC, datetime

    import torch

    from klab.data.tokenizer import load_tokenizer
    from klab.engines.pytorch import PyTorchEngine
    from klab.evaluation.inference import InferenceRun, load_run
    from klab.evaluation.learned_portability import (
        evaluate_learned_portability,
        evaluate_learned_preparation,
    )
    from klab.model.portable_engram import (
        export_portable_engram,
        load_portable_engram,
    )
    from klab.training.trainer import train

    if (
        isinstance(max_wall_seconds, bool)
        or not isinstance(max_wall_seconds, (int, float))
        or not math.isfinite(max_wall_seconds)
        or max_wall_seconds <= 0
    ):
        raise ValueError("max_wall_seconds must be a positive finite number")
    root = Path(campaign_root)
    protocol = _read_json(
        root / "portability_protocol.json", "learned portability protocol"
    )
    ceiling = float(protocol["resource_selection"]["ceiling_seconds"])
    prior = _receipt(root)
    build_receipt = _read_json(
        root / "execution" / "build.json", "learned build receipt"
    )
    consumed = (
        float(prior["elapsed_seconds"])
        if prior is not None
        else float(build_receipt["elapsed_seconds"])
    )
    selection_path = root / "execution" / "resource-selection.json"
    if selection_path.exists():
        sealed_selection = _selection(root, protocol)
        consumed = max(consumed, float(sealed_selection["accounted_elapsed_seconds"]))
    remaining = ceiling - consumed
    if remaining <= 0:
        return {
            "status": "interrupted",
            "reason": "wall_time_allowance_exhausted",
            "remaining_seconds": 0.0,
        }
    if continue_existing and max_wall_seconds > remaining:
        raise ValueError("continue may not raise the sealed learned campaign allowance")
    started = time.monotonic()
    try:
        selection = _seal_smoke_or_calibration(
            root, protocol, min(remaining, float(max_wall_seconds))
        )
    except RuntimeError as error:
        elapsed = consumed + time.monotonic() - started
        state = learned_portability_status(root)
        _publish_receipt(
            root, protocol, state["coordinates"], elapsed=elapsed, allowance=ceiling
        )
        return {
            **state,
            "status": "blocked",
            "reason": str(error),
            "remaining_seconds": max(0.0, ceiling - elapsed),
        }
    limit = min(float(max_wall_seconds), remaining)

    def left() -> float:
        return limit - (time.monotonic() - started)

    def observation_specs(role: str) -> tuple[tuple[str, str, str], ...]:
        if role == "source":
            return (("source_monitor", "source_monitor", "source_monitor"),)
        if role in {"recipient", "native"}:
            return (
                ("calibration", "adapter_return", "calibration"),
                ("held_out", "final_report", "held_out_final_report"),
                ("held_out", "final_state", "held_out_final_state"),
            )
        return ()

    def persist_observation(
        logical: str,
        step: int,
        key: str,
        result_kind: str,
        partition: str,
        wording: str,
        result: dict[str, object],
        checkpoint: dict[str, object],
    ) -> dict[str, object]:
        observations_root = root / "execution" / "observations" / logical
        observations_root.mkdir(parents=True, exist_ok=True)
        result_path = observations_root / f"step_{step:08d}_{key}.json"
        _write_immutable_json(
            result_path,
            {
                "experiment": _EXPERIMENT,
                "coordinate_id": logical,
                "step": step,
                "kind": result_kind,
                "partition": partition,
                "wording": wording,
                "checkpoint": checkpoint,
                "result": result,
            },
        )
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("learned evaluator returned no metric object")
        return {
            "step": step,
            "kind": result_kind,
            "partition": partition,
            "wording": wording,
            "metrics": metrics,
            "artifact": _descriptor(root, result_path),
        }

    def evaluate_snapshot(
        coordinate: dict[str, object],
        logical: str,
        step: int,
        loaded: InferenceRun,
        checkpoint: dict[str, object],
        owned_data_manifest: Path,
    ) -> list[dict[str, object]]:
        role = str(coordinate["role"])
        if role == "preparation":
            result = evaluate_learned_preparation(
                loaded, owned_data_manifest, split="validation"
            )
            return [
                persist_observation(
                    logical,
                    step,
                    "preparation_validation",
                    "preparation",
                    "preparation_validation",
                    "preparation_copy",
                    result,
                    checkpoint,
                )
            ]
        observations: list[dict[str, object]] = []
        for partitions, wording, key in observation_specs(role):
            result = evaluate_learned_portability(
                loaded,
                owned_data_manifest,
                partitions=(partitions,),
                wording=wording,
            )
            observations.append(
                persist_observation(
                    logical,
                    step,
                    key,
                    "query",
                    partitions,
                    wording,
                    result,
                    checkpoint,
                )
            )
        return observations

    def checkpoint_for_step(run_root: Path, step: int) -> tuple[Path, dict[str, Any]]:
        checkpoint_root = run_root / "checkpoints"
        matches: list[tuple[int, Path, dict[str, Any]]] = []
        for candidate in checkpoint_root.glob("step_*_gen_*"):
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            value = _read_json(candidate / "manifest.json", "checkpoint manifest")
            if value.get("step") == step:
                try:
                    generation = int(candidate.name.rsplit("_", 1)[1])
                except IndexError, ValueError:
                    raise ValueError("checkpoint generation name is invalid")
                matches.append((generation, candidate, value))
        if not matches:
            raise ValueError(f"scheduled checkpoint is missing at step {step}")
        _generation, path, value = max(matches, key=lambda item: item[0])
        return path, value

    def observe_trained_run(
        coordinate: dict[str, object],
        logical: str,
        run_id: str,
        expected_steps: list[int],
        backend: str,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        run_root = root / "runs" / run_id
        owned_data = run_root / "portability" / "data" / "manifest.json"
        if not owned_data.is_file():
            raise ValueError("completed learned run lacks its owned data manifest")
        step_results: list[dict[str, object]] = []
        for step in expected_steps:
            if left() <= 0:
                raise _ObservationBudgetExpired
            checkpoint_root, checkpoint_manifest = checkpoint_for_step(run_root, step)
            loaded = load_run(
                run_id,
                root / "runs",
                checkpoint=checkpoint_root.name,
                backend=backend,
            )
            checkpoint = {
                "path": checkpoint_root.relative_to(root).as_posix(),
                "manifest_sha256": checkpoint_manifest["sha256"],
                "step": step,
            }
            results = evaluate_snapshot(
                coordinate, logical, step, loaded, checkpoint, owned_data
            )
            step_results.append(
                {"step": step, "checkpoint": checkpoint, "observations": results}
            )
        last = step_results[-1]["observations"]
        final_metrics = {
            str(item["kind"])
            + ":"
            + str(item["partition"])
            + ":"
            + str(item["wording"]): item["metrics"]
            for item in last
        }
        return step_results, final_metrics

    class _ObservationBudgetExpired(Exception):
        pass

    def _attempt_rows(logical: str) -> list[tuple[Path, dict[str, Any]]]:
        directory = root / "execution" / "attempts" / logical
        if not directory.exists():
            return []
        records = [
            (path, _read_json(path, "learned attempt"))
            for path in directory.glob("*.json")
            if path.is_file() and not path.is_symlink()
        ]
        return sorted(
            records,
            key=lambda item: str(item[1].get("started_at", "")),
        )

    def attempt(
        coordinate: dict[str, object],
        *,
        portable: Path | None = None,
        backbone: dict[str, object] | None = None,
        observation_only: bool = False,
    ) -> bool:
        logical = _coordinate_id(coordinate)
        existing_outcome = _outcome_records(root).get(logical)
        if existing_outcome is not None:
            outcome = existing_outcome.get("outcome")
            return isinstance(outcome, dict) and outcome.get("status") == "completed"
        if left() <= 0:
            return False
        role = str(coordinate["role"])
        condition = str(coordinate["condition"])
        if portable is not None and condition in {"constant", "random", "permuted"}:
            portable = _materialize_control_package(root, portable, coordinate)
        manifest = build_learned_run_manifest(
            root,
            coordinate,
            portable_artifact=portable,
            backbone=backbone,
        )
        step_key = "adapter" if role == "recipient" else role
        max_steps = int(selection["updates"][step_key])
        config = _learned_config(
            root,
            coordinate,
            manifest,
            max_steps=max_steps,
            backend_override=str(selection["backend"]),
        )
        attempts_dir = root / "execution" / "attempts" / logical
        attempts_dir.mkdir(parents=True, exist_ok=True)
        prior_attempts = _attempt_rows(logical)
        last_record = prior_attempts[-1][1] if prior_attempts else None
        reuse_run = (
            last_record.get("run_id")
            if isinstance(last_record, dict)
            and last_record.get("status") == "observation_pending"
            and isinstance(last_record.get("run_id"), str)
            else None
        )
        reuse_observation = (
            isinstance(last_record, dict)
            and last_record.get("status") == "observation_pending"
        )
        resume = None
        if reuse_run is None and last_record is not None:
            parent = last_record.get("run_id")
            if last_record.get("status") == "interrupted" and isinstance(parent, str):
                checkpoint = root / "runs" / parent / "checkpoints" / "latest.json"
                if checkpoint.is_file() and not checkpoint.is_symlink():
                    resume = checkpoint
        if resume is not None:
            config = _resume_config_for_checkpoint(root, config, resume)
        run_id = (
            str(reuse_run)
            if reuse_run is not None
            else f"{logical}-{uuid.uuid4().hex[:12]}"
        )
        attempt_id = uuid.uuid4().hex
        attempt_started = datetime.now(UTC).isoformat()
        began = time.monotonic()
        try:
            if observation_only:
                engine = PyTorchEngine()
                engine.validate(config)
                engine.initialize(config)
                if engine.model is None or engine.device is None:
                    raise RuntimeError("observation-only model initialization failed")
                world = _read_json(
                    manifest.parent / "portability" / "data" / "manifest.json",
                    "observation-only world manifest",
                )
                loaded = InferenceRun(
                    manifest.parent,
                    config,
                    engine.model,
                    load_tokenizer(config.tokenizer.path),
                    engine.device,
                    {
                        "tokenizer_sha256": world["tokenizer"]["sha256"],
                    },
                )
                loaded.model.eval()
                schedule = _read_json(
                    manifest.parent / "portability" / "observations" / "schedule.json",
                    "observation-only schedule",
                )
                if schedule["steps"] != [0]:
                    raise ValueError(
                        "observation-only coordinates must observe step zero"
                    )
                checkpoint = {
                    "kind": "preparation_backbone",
                    "manifest_sha256": backbone["checkpoint_sha256"]
                    if backbone is not None
                    else None,
                    "step": 0,
                }
                if left() <= 0:
                    raise _ObservationBudgetExpired
                results = evaluate_snapshot(
                    coordinate,
                    logical,
                    0,
                    loaded,
                    checkpoint,
                    manifest.parent / "portability" / "data" / "manifest.json",
                )
                step_results = [
                    {"step": 0, "checkpoint": checkpoint, "observations": results}
                ]
                last = step_results[-1]["observations"]
                final_metrics = {
                    str(item["kind"])
                    + ":"
                    + str(item["partition"])
                    + ":"
                    + str(item["wording"]): item["metrics"]
                    for item in last
                }
            else:
                if not reuse_observation:
                    budget = left()
                    if budget <= 0:
                        return False
                    train(
                        config,
                        run_id=run_id,
                        resume=resume,
                        max_wall_seconds=budget,
                        experiment_id=_EXPERIMENT,
                        attempt_id=attempt_id,
                    )
                database = root / "runs" / "experiments.sqlite3"
                if not database.is_file():
                    raise RuntimeError(
                        "trainer did not publish the durable experiment database"
                    )
                with sqlite3.connect(
                    f"file:{database}?mode=ro", uri=True
                ) as connection:
                    status_row = connection.execute(
                        "SELECT status FROM runs WHERE run_id = ?",
                        (run_id,),
                    ).fetchone()
                if status_row is None or status_row[0] not in {
                    "completed",
                    "interrupted",
                    "failed",
                }:
                    raise RuntimeError(
                        "trainer did not publish a durable terminal run status"
                    )
                run_status = str(status_row[0])
                audit = _read_learned_run_audit(root, run_id)
                if audit and audit.get("run_status") != run_status:
                    raise RuntimeError(
                        "portability audit status differs from durable run status"
                    )
                if run_status == "interrupted":
                    _write_immutable_json(
                        attempts_dir / f"{attempt_id}.json",
                        {
                            "experiment": _EXPERIMENT,
                            "coordinate_id": logical,
                            "attempt_id": attempt_id,
                            "run_id": run_id,
                            "parent_checkpoint": None
                            if resume is None
                            else resume.relative_to(root).as_posix(),
                            "started_at": attempt_started,
                            "status": "interrupted",
                            "elapsed_seconds": time.monotonic() - began,
                        },
                    )
                    return False
                if run_status == "failed":
                    reason = "trainer_run_failed"
                    _write_immutable_json(
                        attempts_dir / f"{attempt_id}.json",
                        {
                            "experiment": _EXPERIMENT,
                            "coordinate_id": logical,
                            "attempt_id": attempt_id,
                            "run_id": run_id,
                            "started_at": attempt_started,
                            "status": "failed",
                            "elapsed_seconds": time.monotonic() - began,
                            "reason": reason,
                        },
                    )
                    record_learned_coordinate_outcome(
                        root,
                        coordinate,
                        {"status": "failed", "run_id": run_id, "reason": reason},
                    )
                    return False
                schedule = _read_json(
                    root
                    / "runs"
                    / run_id
                    / "portability"
                    / "observations"
                    / "schedule.json",
                    "owned learned observation schedule",
                )
                try:
                    step_results, final_metrics = observe_trained_run(
                        coordinate,
                        logical,
                        run_id,
                        [int(step) for step in schedule["steps"]],
                        config.runtime.backend,
                    )
                except _ObservationBudgetExpired:
                    _write_immutable_json(
                        attempts_dir / f"{attempt_id}.json",
                        {
                            "experiment": _EXPERIMENT,
                            "coordinate_id": logical,
                            "attempt_id": attempt_id,
                            "run_id": run_id,
                            "started_at": attempt_started,
                            "status": "observation_pending",
                            "elapsed_seconds": time.monotonic() - began,
                        },
                    )
                    return False
            outcome: dict[str, object] = {
                "status": "completed",
                "run_id": None if observation_only else run_id,
                "execution": "observation_only" if observation_only else "trained",
                "step_observations": step_results,
                "final_metrics": final_metrics,
                "elapsed_seconds": time.monotonic() - began,
            }
            if role == "preparation":
                preparation_metrics = final_metrics.get(
                    "preparation:preparation_validation:preparation_copy"
                )
                if not isinstance(preparation_metrics, dict):
                    raise ValueError("preparation validation result is missing")
                outcome["preparation_metrics"] = preparation_metrics
            if role == "source":
                source_metrics = final_metrics.get(
                    "query:source_monitor:source_monitor"
                )
                if not isinstance(source_metrics, dict):
                    raise ValueError("source monitor result is missing")
                outcome.update(
                    {
                        "source_monitor_accuracy": source_metrics.get("accuracy", -1.0),
                        "enabled_minus_disabled_accuracy": audit.get(
                            "enabled_minus_disabled_accuracy", -1.0
                        ),
                        "disabled_minus_enabled_nll": audit.get(
                            "disabled_minus_enabled_nll", -1.0
                        ),
                        "gradient_changed_row_fraction": audit.get(
                            "gradient_changed_row_fraction", -1.0
                        ),
                        "full_fact_exposure": audit.get("full_fact_exposure", False),
                        "valid_update_audit": audit.get("valid_update_audit", False),
                        "finite_tensors": audit.get("finite_tensors", False),
                        "nonzero_table_delta": audit.get("nonzero_table_delta", False),
                    }
                )
            attempt_path = attempts_dir / f"{attempt_id}.json"
            _write_immutable_json(
                attempt_path,
                {
                    "experiment": _EXPERIMENT,
                    "coordinate_id": logical,
                    "attempt_id": attempt_id,
                    "run_id": None if observation_only else run_id,
                    "started_at": attempt_started,
                    "status": "completed",
                    "observation_only": observation_only,
                    "elapsed_seconds": time.monotonic() - began,
                },
            )
            record_learned_coordinate_outcome(root, coordinate, outcome)
            return True
        except _ObservationBudgetExpired:
            _write_immutable_json(
                attempts_dir / f"{attempt_id}.json",
                {
                    "experiment": _EXPERIMENT,
                    "coordinate_id": logical,
                    "attempt_id": attempt_id,
                    "run_id": None if observation_only else run_id,
                    "started_at": attempt_started,
                    "status": "observation_pending",
                    "elapsed_seconds": time.monotonic() - began,
                },
            )
            return False
        except Exception as error:  # noqa: BLE001
            reason = f"{type(error).__name__}: {error}"
            attempt_path = attempts_dir / f"{attempt_id}.json"
            if not attempt_path.exists():
                _write_immutable_json(
                    attempt_path,
                    {
                        "experiment": _EXPERIMENT,
                        "coordinate_id": logical,
                        "attempt_id": attempt_id,
                        "run_id": None if observation_only else run_id,
                        "started_at": attempt_started,
                        "status": "failed",
                        "elapsed_seconds": time.monotonic() - began,
                        "reason": reason,
                    },
                )
            if logical not in _outcome_records(root):
                record_learned_coordinate_outcome(
                    root,
                    coordinate,
                    {
                        "status": "failed",
                        "run_id": None if observation_only else run_id,
                        "reason": reason,
                    },
                )
            return False

    plan = _coordinate_plan()
    source_rows = [row for row in plan if row["role"] == "source"]
    for row in source_rows:
        attempt(row)
        if left() <= 0:
            break
    state = learned_portability_status(root)
    source_outcomes = _outcome_records(root)
    source_stage_complete = all(
        _coordinate_id(row) in source_outcomes
        and source_outcomes[_coordinate_id(row)]["outcome"].get("status") == "completed"
        for row in source_rows
    )
    if source_stage_complete and state["source_gate_failure"] is None:
        for row in source_rows:
            if row["condition"] != "source-real":
                continue
            outcome = source_outcomes[_coordinate_id(row)]["outcome"]
            package = root / "exports" / f"source-s{row['seed']}.engram"
            provenance_path = package.with_suffix(".provenance.json")
            package.parent.mkdir(parents=True, exist_ok=True)
            if package.exists() or provenance_path.exists():
                if not package.is_file() or not provenance_path.is_file():
                    raise ValueError(
                        "source package and provenance must be published together"
                    )
                provenance = _read_json(provenance_path, "source export provenance")
                if (
                    provenance.get("source_run_id") != outcome.get("run_id")
                    or provenance.get("package") != _descriptor(root, package)
                    or provenance.get("source_gate") != outcome
                ):
                    raise ValueError("existing source export provenance differs")
                load_portable_engram(package)
                continue
            loaded = load_run(str(outcome["run_id"]), root / "runs", backend=None)
            if loaded.model.memory is None or not hasattr(loaded.model.memory, "table"):
                raise ValueError("passed source gate has no trainable byte table")
            export_portable_engram(
                loaded.model.memory.table.weight, package, ngram_size=32
            )
            package_value = load_portable_engram(package)
            table_bytes = (
                loaded.model.memory.table.weight.detach()
                .to(device="cpu")
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            if (
                hashlib.sha256(table_bytes).hexdigest()
                != package_value.manifest.table_sha256
            ):
                raise ValueError("exported source table differs from trained source")
            _write_immutable_json(
                provenance_path,
                {
                    "experiment": _EXPERIMENT,
                    "source_run_id": outcome["run_id"],
                    "package": _descriptor(root, package),
                    "source_gate": outcome,
                },
            )

        for row in plan:
            if row["role"] == "preparation":
                attempt(row)
                if left() <= 0:
                    break
        if left() > 0:
            for row in plan:
                if row["role"] not in {"recipient", "native"}:
                    continue
                prep = next(
                    item
                    for item in plan
                    if item["role"] == "preparation"
                    and item["seed"] == row["seed"]
                    and item["recipient"] == row["recipient"]
                )
                prep_outcome = (
                    _outcome_records(root)
                    .get(_coordinate_id(prep), {})
                    .get("outcome", {})
                )
                if (
                    not isinstance(prep_outcome, dict)
                    or prep_outcome.get("status") != "completed"
                    or not _preparation_gate(prep_outcome)[0]
                ):
                    continue
                prep_root = root / "runs" / str(prep_outcome["run_id"])
                pointer_path = prep_root / "checkpoints" / "latest.json"
                pointer = _verified_checkpoint_pointer(
                    prep_root, pointer_path, "preparation checkpoint pointer"
                )
                relative = pointer.get("relative_path")
                if (
                    not isinstance(relative, str)
                    or Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                ):
                    raise ValueError("preparation checkpoint pointer is unsafe")
                checkpoint_root = prep_root / "checkpoints" / relative
                checkpoint_manifest = _read_json(
                    checkpoint_root / "manifest.json",
                    "preparation checkpoint manifest",
                )
                if checkpoint_manifest.get("sha256") != pointer.get("manifest_sha256"):
                    raise ValueError("preparation checkpoint pointer digest differs")
                backbone = {
                    "run_root": prep_root,
                    "checkpoint_root": checkpoint_root,
                    "checkpoint_sha256": checkpoint_manifest["sha256"],
                    "run_id": prep_outcome["run_id"],
                }
                package = root / "exports" / f"source-s{row['seed']}.engram"
                if row["condition"] not in {
                    "constant",
                    "random",
                    "permuted",
                    "real-zero-shot",
                    "real-adapter",
                }:
                    package = None
                observation_only = row["condition"] in {
                    "baseline",
                    "real-zero-shot",
                }
                attempt(
                    row,
                    portable=package,
                    backbone=backbone,
                    observation_only=observation_only,
                )
                if left() <= 0:
                    break
    state = learned_portability_status(root)
    elapsed = consumed + time.monotonic() - started
    _publish_receipt(
        root, protocol, state["coordinates"], elapsed=elapsed, allowance=ceiling
    )
    evidence = build_learned_portability_evidence(root)
    return {
        **state,
        "evidence": str(evidence),
        "elapsed_seconds": elapsed,
        "remaining_seconds": max(0.0, ceiling - elapsed),
    }


def continue_learned_portability_campaign(
    campaign_root: Path, *, max_wall_seconds: float
) -> dict[str, object]:
    return execute_learned_portability_campaign(
        campaign_root, max_wall_seconds=max_wall_seconds, continue_existing=True
    )


def report_learned_portability_campaign(
    campaign_root: Path,
) -> dict[str, object]:
    """Regenerate a static bundle from campaign evidence without model execution."""
    root = Path(campaign_root)
    protocol = _read_json(
        root / "portability_protocol.json", "learned portability protocol"
    )
    if protocol.get("experiment") != _EXPERIMENT:
        raise ValueError("campaign is not a learned Engram portability campaign")
    try:
        _selection(root, protocol)
    except ValueError as error:
        return {"status": "unavailable", "reason": str(error)}
    evidence = build_learned_portability_evidence(root)
    from klab.experiments.reporting import (
        build_portability_report,
        load_report_bundle,
        write_study_report,
    )

    report = build_portability_report(evidence)
    bundle = write_study_report(report, root / "reports")
    loaded = load_report_bundle(bundle)
    manifest = loaded.get("manifest")
    return {
        "status": "published",
        "evidence": str(evidence),
        "report": str(bundle),
        "report_sha256": manifest.get("bundle_sha256")
        if isinstance(manifest, dict)
        else None,
    }
