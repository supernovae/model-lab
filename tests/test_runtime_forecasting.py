from __future__ import annotations

from copy import deepcopy

import pytest

from sparselab.config.models import RunConfig
from sparselab.runtime import RuntimeInfo
from sparselab.runtime_forecasting import (
    match_runtime_observations,
    optimizer_only_estimate,
    runtime_forecast_planning,
    runtime_signature,
    runtime_signature_key,
    warmup_estimate,
)
from sparselab.training.metrics import ExperimentStore


def _config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "schema_version": 2,
            "name": "forecast",
            "seed": 1,
            "runtime": {
                "engine": "pytorch",
                "backend": "cpu",
                "precision": "auto",
                "memory": {
                    "activation_checkpointing": {"enabled": True},
                    "activation_offload": {"enabled": True},
                },
            },
            "model": {
                "vocab_size": 512,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "ffn_dim": 32,
                "max_seq_len": 16,
            },
            "tokenizer": {"path": "tokenizer.json"},
            "dataset": {
                "source": "synthetic",
                "cache_dir": "cache",
                "train_max_documents": 2,
                "validation_max_documents": 2,
                "train_max_tokens": 64,
                "validation_max_tokens": 64,
            },
            "training": {
                "micro_batch_size": 2,
                "gradient_accumulation": 3,
                "seq_len": 8,
                "max_steps": 20,
                "max_tokens": 160,
            },
            "logging": {"root_dir": "runs"},
        }
    )


def _runtime() -> RuntimeInfo:
    return RuntimeInfo(
        engine="pytorch",
        backend="cpu",
        torch_device="cpu",
        device_index=0,
        device_name="Test CPU",
        physical_device_id="cpu:test",
        framework_version="test",
        runtime_version=None,
        driver_version=None,
        os="test",
        system_total_bytes=1,
        system_available_bytes=1,
        device_total_bytes=None,
        device_free_bytes=None,
        device_recommended_bytes=None,
        measurement_source="test",
        measured_at="now",
        precision_capabilities=("fp32",),
    )


def _observation(signature: dict[str, object], rate: float = 10.0) -> dict[str, object]:
    return {
        "run_id": "run-a",
        "runtime_signature": deepcopy(signature),
        "optimizer_targets_per_second": rate,
    }


def test_signature_has_all_performance_dimensions_and_ignores_memory_path() -> None:
    config = _config()
    before = config.model_dump(mode="json")

    signature = runtime_signature(config, _runtime())

    assert config.model_dump(mode="json") == before
    assert signature["precision"] == "fp32"
    assert set(signature) == {
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
    }
    assert "memory_package_path" not in signature["architecture_identity"]["model"]


def test_matcher_requires_each_dimension_and_explains_rejections() -> None:
    signature = runtime_signature(_config(), _runtime())
    compatible = _observation(signature)
    observations = [compatible]
    for index, field in enumerate(signature):
        incompatible = _observation(signature)
        incompatible["run_id"] = f"bad-{field}"
        incompatible["runtime_signature"][field] = f"changed-{index}"
        observations.append(incompatible)

    result = match_runtime_observations(signature, observations)

    assert [item["run_id"] for item in result["compatible_observations"]] == ["run-a"]
    assert set(result["compatible_observations"][0]["matching_fields"]) == set(
        signature
    )
    assert {
        item["run_id"]: item["mismatches"][0]["field"]
        for item in result["rejected_observations"]
    } == {f"bad-{field}": field for field in signature}


def test_matcher_rejects_malformed_and_zero_rate_without_mutating_inputs() -> None:
    signature = runtime_signature(_config(), _runtime())
    observations: list[object] = [
        "not-an-object",
        {"run_id": "missing", "optimizer_targets_per_second": 3.0},
        _observation(signature, 0.0),
    ]
    before = deepcopy(observations)

    result = match_runtime_observations(  # type: ignore[arg-type]
        signature, observations
    )

    assert observations == before
    assert result["compatible_observations"] == []
    assert [item["reason"] for item in result["rejected_observations"]] == [
        "observation is not an object",
        "runtime signature is missing or not an object",
        "incompatible",
    ]
    assert result["rejected_observations"][2]["mismatches"] == [
        {
            "field": "optimizer_targets_per_second",
            "reason": "must be a finite positive number",
        }
    ]


def test_historical_estimate_uses_median_and_interpolated_range() -> None:
    observations = [
        {"run_id": "slow", "optimizer_targets_per_second": 10.0},
        {"run_id": "middle", "optimizer_targets_per_second": 20.0},
        {"run_id": "fast", "optimizer_targets_per_second": 30.0},
    ]
    before = deepcopy(observations)

    result = optimizer_only_estimate(600, observations)

    assert observations == before
    assert result == {
        "schema_version": 1,
        "estimate_kind": "optimizer_only",
        "source": "historical",
        "total_targets": 600,
        "contributing_run_ids": ["slow", "middle", "fast"],
        "availability": "available",
        "confidence": "preliminary",
        "optimizer_targets_per_second": 20.0,
        "optimizer_only_seconds": 30.0,
        "low_optimizer_only_seconds": 24.0,
        "high_optimizer_only_seconds": 40.0,
        "reason": None,
    }


def test_historical_estimate_is_unavailable_without_usable_observations() -> None:
    result = optimizer_only_estimate(
        100, [{"run_id": "bad", "optimizer_targets_per_second": float("nan")}]
    )

    assert result["availability"] == "unavailable"
    assert result["optimizer_only_seconds"] is None
    assert result["low_optimizer_only_seconds"] is None
    assert result["high_optimizer_only_seconds"] is None
    assert result["reason"] == "no usable optimizer throughput observations"


def test_warmup_discards_initial_updates_and_rejects_invalid_measurements() -> None:
    updates = [(100, 100.0), (100, 10.0), (100, 5.0), (0, 1.0), (100, 0.0)]
    before = deepcopy(updates)

    result = warmup_estimate(1_000, updates, pilot_run_id="pilot-1")

    assert updates == before
    assert result["source"] == "warmup-calibrated"
    assert result["pilot_run_id"] == "pilot-1"
    assert result["contributing_run_ids"] == ["pilot-1"]
    assert result["optimizer_targets_per_second"] == 15.0
    assert result["optimizer_only_seconds"] == pytest.approx(1000 / 15)
    assert result["low_optimizer_only_seconds"] == pytest.approx(1000 / 17.5)
    assert result["high_optimizer_only_seconds"] == pytest.approx(80.0)


def test_warmup_is_unavailable_when_no_measured_update_remains() -> None:
    result = warmup_estimate(
        100,
        [(100, 10.0)],
        pilot_run_id="pilot",
        discard_first_updates=1,
    )

    assert result["availability"] == "unavailable"
    assert result["optimizer_only_seconds"] is None


def test_planning_reads_only_compatible_recent_runtime_calibrations(tmp_path) -> None:
    config = _config()
    runtime = _runtime()
    signature = runtime_signature(config, runtime)
    store = ExperimentStore(tmp_path)
    for index in range(30):
        run_id = f"run-{index:02d}"
        store.create_run(run_id, {"name": run_id}, {})
        store.record_calibration(
            runtime_signature_key(signature),
            run_id,
            {
                "kind": "runtime_optimizer_throughput_v1",
                "runtime_signature": signature,
                "optimizer_targets_per_second": float(10 + index),
            },
        )
    store.create_run("incompatible", {"name": "incompatible"}, {})
    other_signature = {**signature, "sequence_length": 99}
    store.record_calibration(
        runtime_signature_key(other_signature),
        "incompatible",
        {
            "kind": "runtime_optimizer_throughput_v1",
            "runtime_signature": other_signature,
            "optimizer_targets_per_second": 20.0,
        },
    )
    store.create_run("memory-only", {"name": "memory-only"}, {})
    store.record_calibration("memory", "memory-only", {"observed": {"peak": 1}})
    before = store.export_records()

    result = runtime_forecast_planning(tmp_path, config, runtime, total_targets=1_000)

    assert result["planning"]["availability"] == "available"
    assert result["planning"]["contributing_run_ids"] == [
        f"run-{index:02d}" for index in range(5, 30)
    ]
    assert result["history_match"]["rejected_observations"][0]["mismatches"] == [
        {"field": "sequence_length", "reason": "does not match"}
    ]
    assert store.export_records() == before


def test_runtime_signature_binds_large_mapping_without_embedding_it(tmp_path):
    import json

    from tokenizers import Tokenizer, models

    from sparselab.canonical import canonical_digest
    from sparselab.config.models import RunConfig
    from sparselab.memory_tokenizer import build_memory_token_map

    path = tmp_path / "tokenizer.json"
    Tokenizer(models.WordLevel({f"token{i}": i for i in range(32768)})).save(str(path))
    mapping = build_memory_token_map(path, mode="identity").to_dict()
    raw = _config().model_dump(mode="json")
    raw["model"].update(
        vocab_size=32768,
        memory="ngram",
        memory_table_size=31,
        memory_dim=8,
        memory_ngram_size=2,
        memory_token_mapping=mapping,
    )
    config = RunConfig.model_validate(raw)
    signature = runtime_signature(config, _runtime())
    assert signature["architecture_identity"]["model"]["memory_token_mapping"] == {
        "sha256": canonical_digest(mapping)
    }
    assert len(json.dumps(signature).encode()) < 16384
    assert len(config.model.memory_token_mapping.token_ids) == 32768
