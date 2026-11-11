"""Pure runtime-history matching and optimizer-only forecasts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

from klab.canonical import canonical_digest

if TYPE_CHECKING:
    from klab.config.models import RunConfig
    from klab.runtime import RuntimeInfo


_SCHEMA_VERSION = 1
RUNTIME_OBSERVATION_KIND = "runtime_optimizer_throughput_v1"
MAX_HISTORICAL_OBSERVATIONS = 25
_SIGNATURE_FIELDS = (
    "engine",
    "backend",
    "os",
    "framework_version",
    "runtime_version",
    "driver_version",
    "device_identity",
    "precision",
    "architecture_identity",
    "attention_family",
    "optimizer",
    "sequence_length",
    "micro_batch_size",
    "gradient_accumulation",
    "activation_recomputation",
    "activation_offload",
)


def _json_model(value: object) -> object:
    """Return a JSON-safe model dump without changing the source model."""
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        raise TypeError("runtime forecasting requires configuration models")
    return model_dump(mode="json")


def runtime_signature(config: RunConfig, runtime: RuntimeInfo) -> dict[str, object]:
    """Describe the execution dimensions that affect optimizer update speed.

    Deliberately excludes dataset and tokenizer identity: compatible optimizer
    measurements can be reused across data sources.
    """
    model = _json_model(config.model)
    if not isinstance(model, dict):
        raise TypeError("model configuration must serialize to an object")
    model.pop("memory_package_path", None)
    mapping = model.get("memory_token_mapping")
    if isinstance(mapping, Mapping):
        model["memory_token_mapping"] = {"sha256": canonical_digest(mapping)}
    attention = _json_model(config.attention)
    optimizer = _json_model(config.optimizer)
    if not isinstance(attention, dict) or not isinstance(optimizer, dict):
        raise TypeError(
            "attention and optimizer configuration must serialize to objects"
        )

    physical_id = runtime.physical_device_id
    device_identity: dict[str, object] = {
        "physical_device_id": physical_id,
        "device_name": runtime.device_name,
        "device_index": runtime.device_index,
    }
    return {
        "engine": runtime.engine,
        "os": runtime.os,
        "framework_version": runtime.framework_version,
        "runtime_version": runtime.runtime_version,
        "driver_version": runtime.driver_version,
        "backend": runtime.backend,
        "device_identity": device_identity,
        "precision": (
            "fp32" if config.runtime.precision == "auto" else config.runtime.precision
        ),
        "architecture_identity": {"model": model, "attention": attention},
        "attention_family": config.attention.kind,
        "optimizer": optimizer,
        "sequence_length": config.training.seq_len,
        "micro_batch_size": config.training.micro_batch_size,
        "gradient_accumulation": config.training.gradient_accumulation,
        "activation_recomputation": (
            config.runtime.memory.activation_checkpointing.enabled
        ),
        "activation_offload": config.runtime.memory.activation_offload.enabled,
    }


def _finite_positive(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and value > 0


def _observation_signature(
    observation: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Support the explicit runtime_signature name and concise signature alias."""
    value = observation.get("runtime_signature", observation.get("signature"))
    return value if isinstance(value, Mapping) else None


def _run_id(observation: Mapping[str, object]) -> str | None:
    value = observation.get("run_id")
    return value if isinstance(value, str) and value else None


def match_runtime_observations(
    signature: Mapping[str, object], observations: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Select exact speed-compatible observations and explain every rejection."""
    expected = {field: signature.get(field) for field in _SIGNATURE_FIELDS}
    compatible: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            rejected.append(
                {
                    "index": index,
                    "run_id": None,
                    "reason": "observation is not an object",
                    "mismatches": [
                        {"field": "observation", "reason": "is not an object"}
                    ],
                }
            )
            continue
        run_id = _run_id(observation)
        observed_signature = _observation_signature(observation)
        if observed_signature is None:
            rejected.append(
                {
                    "index": index,
                    "run_id": run_id,
                    "reason": "runtime signature is missing or not an object",
                    "mismatches": [
                        {
                            "field": "runtime_signature",
                            "reason": "missing or not an object",
                        }
                    ],
                }
            )
            continue
        mismatches: list[dict[str, object]] = []
        for field in _SIGNATURE_FIELDS:
            if field not in observed_signature:
                mismatches.append({"field": field, "reason": "missing"})
            elif observed_signature[field] != expected[field]:
                mismatches.append({"field": field, "reason": "does not match"})
        identity = expected.get("device_identity")
        identity_known = isinstance(identity, Mapping) and bool(
            identity.get("physical_device_id") or identity.get("device_name")
        )
        if not identity_known:
            mismatches.append(
                {"field": "device_identity", "reason": "current identity unavailable"}
            )
        if run_id is None:
            mismatches.append(
                {"field": "run_id", "reason": "missing or not a nonempty string"}
            )
        rate = observation.get("optimizer_targets_per_second")
        if not _finite_positive(rate):
            mismatches.append(
                {
                    "field": "optimizer_targets_per_second",
                    "reason": "must be a finite positive number",
                }
            )
        if mismatches:
            rejected.append(
                {
                    "index": index,
                    "run_id": run_id,
                    "reason": "incompatible",
                    "mismatches": mismatches,
                }
            )
        else:
            assert isinstance(rate, (int, float)) and not isinstance(rate, bool)
            compatible.append(
                {
                    "run_id": run_id,
                    "runtime_signature": deepcopy(dict(observed_signature)),
                    "matching_fields": list(_SIGNATURE_FIELDS),
                    "optimizer_targets_per_second": float(rate),
                }
            )
    return {
        "schema_version": _SCHEMA_VERSION,
        "signature": deepcopy(expected),
        "compatible_observations": compatible,
        "rejected_observations": rejected,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _estimate_record(
    *,
    total_targets: int,
    rates: Sequence[float],
    source: str,
    run_ids: list[str],
    pilot_run_id: str | None = None,
) -> dict[str, object]:
    common: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "estimate_kind": "optimizer_only",
        "source": source,
        "total_targets": total_targets,
        "contributing_run_ids": run_ids,
    }
    if pilot_run_id is not None:
        common["pilot_run_id"] = pilot_run_id
    if not rates:
        return {
            **common,
            "availability": "unavailable",
            "confidence": "preliminary",
            "optimizer_targets_per_second": None,
            "optimizer_only_seconds": None,
            "low_optimizer_only_seconds": None,
            "high_optimizer_only_seconds": None,
            "reason": "no usable optimizer throughput observations",
        }
    p25 = _percentile(rates, 0.25)
    p75 = _percentile(rates, 0.75)
    typical = float(median(rates))
    return {
        **common,
        "availability": "available",
        "confidence": "preliminary",
        "optimizer_targets_per_second": typical,
        "optimizer_only_seconds": total_targets / typical,
        "low_optimizer_only_seconds": total_targets / p75,
        "high_optimizer_only_seconds": total_targets / p25,
        "reason": None,
    }


def optimizer_only_estimate(
    total_targets: int, compatible_observations: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    if type(total_targets) is not int or total_targets <= 0:
        raise ValueError("total_targets must be a positive integer")
    rates: list[float] = []
    run_ids: list[str] = []
    for observation in compatible_observations:
        if not isinstance(observation, Mapping):
            continue
        rate = observation.get("optimizer_targets_per_second")
        run_id = _run_id(observation)
        if not _finite_positive(rate) or run_id is None:
            continue
        assert isinstance(rate, (int, float)) and not isinstance(rate, bool)
        rates.append(float(rate))
        run_ids.append(run_id)
    return _estimate_record(
        total_targets=total_targets,
        rates=rates,
        source="historical",
        run_ids=run_ids,
    )


def warmup_estimate(
    total_targets: int,
    updates: Sequence[tuple[int, float]],
    *,
    pilot_run_id: str,
    discard_first_updates: int = 1,
) -> dict[str, object]:
    if type(total_targets) is not int or total_targets <= 0:
        raise ValueError("total_targets must be a positive integer")
    if type(discard_first_updates) is not int or discard_first_updates < 0:
        raise ValueError("discard_first_updates must be a nonnegative integer")
    rates: list[float] = []
    for update in updates[discard_first_updates:]:
        if not isinstance(update, tuple) or len(update) != 2:
            continue
        targets, seconds = update
        if type(targets) is not int or targets <= 0 or not _finite_positive(seconds):
            continue
        assert isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
        rates.append(targets / float(seconds))
    return _estimate_record(
        total_targets=total_targets,
        rates=rates,
        source="warmup-calibrated",
        run_ids=[pilot_run_id],
        pilot_run_id=pilot_run_id,
    )


def runtime_signature_key(signature: Mapping[str, object]) -> str:
    """Return a stable digest suitable for the calibration table key."""
    encoded = json.dumps(
        dict(signature),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_forecast_planning(
    root_dir: Path,
    config: RunConfig,
    runtime: RuntimeInfo,
    *,
    total_targets: int,
) -> dict[str, object]:
    """Build an explainable optimizer-only forecast from exact runtime matches."""
    from klab.training.metrics import ExperimentStore

    signature = runtime_signature(config, runtime)
    stored = ExperimentStore.get_all_calibrations(root_dir)
    observations: list[dict[str, object]] = []
    for row in stored:
        observation = row.get("observation")
        if (
            isinstance(observation, Mapping)
            and observation.get("kind") == RUNTIME_OBSERVATION_KIND
        ):
            observations.append({**dict(observation), "run_id": row.get("run_id")})
    history = match_runtime_observations(signature, observations)
    compatible = history["compatible_observations"]
    if not isinstance(compatible, list):
        raise TypeError("runtime history matcher returned invalid observations")
    planning = optimizer_only_estimate(
        total_targets, compatible[-MAX_HISTORICAL_OBSERVATIONS:]
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "runtime_signature": signature,
        "planning": planning,
        "history_match": history,
        "history_limit": MAX_HISTORICAL_OBSERVATIONS,
    }
