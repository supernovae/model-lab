from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

import sparselab.memory as memory_module
from sparselab.config.loading import load_config
from sparselab.memory import (
    MemoryMonitor,
    calibrated_estimate,
    calibration_key,
    calibration_multiplier,
    estimate_memory,
    parameter_inventory,
    plan_memory,
    registered_runtime_buffers_bytes,
    write_resource_proposal,
)
from sparselab.model.inspection import inspection_report, named_tensor_inventory
from sparselab.runtime import RuntimeInfo
from sparselab.training.manifest import config_sha256
from sparselab.training.metrics import ExperimentStore


def _runtime(
    *,
    backend: str = "cpu",
    total: int | None = 16_000_000_000,
    available: int | None = 12_000_000_000,
    device_total: int | None = None,
    device_free: int | None = None,
    recommended: int | None = None,
    driver_allocated: int | None = None,
) -> RuntimeInfo:
    return RuntimeInfo(
        engine="pytorch",
        backend=backend,
        torch_device=backend,
        device_index=0,
        device_name="test",
        physical_device_id=None,
        framework_version="test",
        runtime_version=None,
        driver_version=None,
        os="test",
        system_total_bytes=total,
        system_available_bytes=available,
        device_total_bytes=device_total,
        device_free_bytes=device_free,
        device_recommended_bytes=recommended,
        measurement_source="test",
        measured_at="test",
        precision_capabilities=("fp32",),
        device_driver_allocated_bytes=driver_allocated,
    )


def test_unified_offload_rejects_fit_without_host_staging_headroom(monkeypatch) -> None:
    config = load_config(Path("configs/runtime_smoke_cpu.yaml"))
    raw = config.model_dump(mode="json")
    raw["runtime"]["backend"] = "mps"
    raw["runtime"]["memory"]["activation_offload"]["enabled"] = True
    config = type(config).model_validate(raw)
    runtime = _runtime(backend="mps", recommended=8_000_000_000, driver_allocated=0)
    estimate = estimate_memory(config, runtime, parameter_inventory(config))
    assert estimate.result == "LIKELY_TO_FIT"
    memory = memory_module.psutil.virtual_memory()._replace(
        total=1 << 40, available=estimate.peak_bytes
    )
    monkeypatch.setattr(memory_module.psutil, "virtual_memory", lambda: memory)
    with pytest.raises(MemoryError):
        memory_module.validate_offload_headroom(config, runtime, estimate)


def test_shape_report_never_opens_portable_package() -> None:
    base = load_config(Path("configs/smoke_byte_memory_cpu.yaml"))
    config = base.model_copy(
        update={
            "model": base.model.model_copy(
                update={
                    "memory": "portable",
                    "memory_package_path": Path("/definitely/unavailable.engram"),
                }
            )
        }
    )
    report = inspection_report(config)
    tensors = named_tensor_inventory(config.model, config.attention)
    assert report["frozen"] == config.model.memory_table_size * config.model.memory_dim
    assert tensors["memory.embedding.weight"].trainable is False
    assert tensors["output.weight"].alias_of == "embedding.weight"


def test_billion_scale_shape_inspection_has_no_model_state() -> None:
    base = load_config(Path("configs/smoke_mla_cpu.yaml"))
    config = base.model_copy(
        update={
            "model": base.model.model_copy(
                update={
                    "vocab_size": 1_000_000,
                    "hidden_dim": 4096,
                    "num_heads": 32,
                    "ffn_dim": 16_384,
                    "num_layers": 48,
                }
            ),
            "attention": base.attention.model_copy(update={"latent_dim": 1024}),
        }
    )
    report = inspection_report(config)
    assert report["total"] > 1_000_000_000
    assert report["model_weight_bytes"] == report["total"] * 4


def test_artificial_budget_cannot_certify_a_physical_fit() -> None:
    config = load_config(Path("configs/smoke_mla_cpu.yaml"))
    budgeted = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "memory": config.runtime.memory.model_copy(
                        update={"budget_bytes": 10**18}
                    )
                }
            )
        }
    )
    estimate = estimate_memory(
        budgeted, _runtime(total=None, available=None), parameter_inventory(budgeted)
    )
    assert estimate.result == "UNKNOWN"
    assert estimate.capacity_ceiling_bytes == 10**18
    assert "cannot certify" in " ".join(estimate.assumptions)


def test_unified_memory_is_one_conservative_pool() -> None:
    config = load_config(Path("configs/smoke_mla_cpu.yaml"))
    estimate = estimate_memory(
        config,
        _runtime(
            backend="mps",
            total=64_000,
            available=4_000,
            recommended=10_000,
            driver_allocated=7_000,
        ),
        parameter_inventory(config),
    )
    assert estimate.capacity_ceiling_bytes == max(
        0, int(config.runtime.memory.max_device_memory_fraction * 10_000) - 7_000
    )
    assert estimate.result == "LIKELY_TO_EXCEED"
    unknown = estimate_memory(
        config,
        _runtime(backend="mps", available=4_000, recommended=10_000),
        parameter_inventory(config),
    )
    assert unknown.result == "UNKNOWN"


def test_proposal_is_complete_recomputed_and_preserves_effective_batch() -> None:
    config = load_config(Path("configs/smoke_mla_cpu.yaml"))
    constrained = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "memory": config.runtime.memory.model_copy(
                        update={"budget_bytes": 1, "policy": "balanced"}
                    )
                }
            )
        }
    )
    original_dump = constrained.model_dump(mode="json")
    original = estimate_memory(
        constrained, _runtime(), parameter_inventory(constrained)
    )
    proposal = plan_memory(constrained, _runtime(), original)
    assert constrained.model_dump(mode="json") == original_dump
    assert constrained.training.micro_batch_size == 4
    assert proposal.config["training"]["micro_batch_size"] < 4
    assert (
        proposal.config["training"]["micro_batch_size"]
        * proposal.config["training"]["gradient_accumulation"]
        == 4
    )
    candidate = constrained.__class__.model_validate(proposal.config)
    assert proposal.estimate == estimate_memory(
        candidate, _runtime(), parameter_inventory(candidate)
    )
    assert any(
        decision["requested"] == "activation_checkpointing"
        for decision in proposal.decisions
    )


def test_unsupported_offload_has_no_imaginary_saving() -> None:
    config = load_config(Path("configs/smoke_mla_cpu.yaml"))
    offload = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "memory": config.runtime.memory.model_copy(
                        update={
                            "budget_bytes": 1,
                            "activation_offload": config.runtime.memory.activation_offload.model_copy(
                                update={"enabled": True}
                            ),
                        }
                    )
                }
            )
        }
    )
    estimate = estimate_memory(offload, _runtime(), parameter_inventory(offload))
    proposal = plan_memory(offload, _runtime(), estimate)
    assert any("no saving" in decision["reason"] for decision in proposal.decisions)
    assert any(
        "activation offload is unsupported" in assumption
        for assumption in proposal.estimate.assumptions
    )


def test_capacity_exact_boundary_is_fit_and_one_byte_over_is_exceed() -> None:
    config = load_config(Path("configs/smoke_mla_cpu.yaml"))
    runtime = _runtime(total=10**18, available=10**18)
    baseline = estimate_memory(config, runtime, parameter_inventory(config))
    at_boundary = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "memory": config.runtime.memory.model_copy(
                        update={"budget_bytes": baseline.peak_bytes}
                    )
                }
            )
        }
    )
    over = at_boundary.model_copy(
        update={
            "runtime": at_boundary.runtime.model_copy(
                update={
                    "memory": at_boundary.runtime.memory.model_copy(
                        update={"budget_bytes": baseline.peak_bytes - 1}
                    )
                }
            )
        }
    )
    assert (
        estimate_memory(at_boundary, runtime, parameter_inventory(at_boundary)).result
        == "LIKELY_TO_FIT"
    )
    assert (
        estimate_memory(over, runtime, parameter_inventory(over)).result
        == "LIKELY_TO_EXCEED"
    )


def test_registered_buffers_are_disjoint_and_scale_with_layers() -> None:
    config = load_config(Path("configs/smoke_mla_cpu.yaml"))
    buffers = registered_runtime_buffers_bytes(config)
    assert buffers > 0
    doubled = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={"num_layers": config.model.num_layers * 2}
            )
        }
    )
    assert registered_runtime_buffers_bytes(doubled) == 2 * buffers


def test_calibration_uses_native_peaks_not_sampled_mps_lower_bounds() -> None:
    assert calibration_multiplier(100, 150, peak_method="native") == 1.5
    assert calibration_multiplier(100, 50, peak_method="native") == 1.0
    assert (
        calibration_multiplier(100, 10_000, peak_method="sampled_lower_bound") is None
    )


def test_monitor_lifetime_records_sampled_mps_without_fake_native_peak(
    monkeypatch,
) -> None:
    class FakeDevice:
        type = "mps"

    monkeypatch.setattr(memory_module, "allocated_memory_bytes", lambda device: 123)
    monitor = MemoryMonitor(FakeDevice(), sample_interval_seconds=0.01)
    monkeypatch.setattr(monitor, "_native", lambda name: None)
    monitor.begin_update()
    monitor.sample("after-forward")
    metrics = monitor.end_update()
    monitor.close()
    assert metrics["memory/device_sampled_peak_bytes"] == 123.0
    assert "memory/device_peak_allocated_bytes" not in metrics
    assert (
        metrics["memory/process_peak_rss_bytes"] >= metrics["memory/process_rss_bytes"]
    )
    assert monitor._observer is None


def test_monitor_prefers_reported_native_peak_over_sampled_metrics(monkeypatch) -> None:
    class FakeDevice:
        type = "cuda"

    monkeypatch.setattr(memory_module, "allocated_memory_bytes", lambda device: 7)
    monitor = MemoryMonitor(FakeDevice())
    monkeypatch.setattr(monitor, "_native", lambda name: 19)
    monkeypatch.setattr(
        memory_module.torch.cuda, "reset_peak_memory_stats", lambda device: None
    )
    monitor.begin_update()
    metrics = monitor.end_update()
    assert metrics["memory/device_peak_allocated_bytes"] == 19.0
    assert "memory/device_sampled_peak_bytes" not in metrics


def test_monitor_reports_current_allocation_separately_from_earlier_peak(
    monkeypatch,
) -> None:
    class FakeDevice:
        type = "mps"

    allocated = iter([1, 1000, *([9] * 1001)])
    monkeypatch.setattr(
        memory_module, "allocated_memory_bytes", lambda device: next(allocated)
    )
    monitor = MemoryMonitor(FakeDevice())
    monkeypatch.setattr(monitor, "_native", lambda name: None)
    monitor.begin_update()
    monitor.sample("forward")
    for _ in range(1000):
        monitor.sample("backward")
    metrics = monitor.end_update()
    assert metrics["memory/device_allocated_bytes"] == 9
    assert metrics["memory/device_sampled_peak_bytes"] == 1000


def test_failed_peak_reset_cannot_report_a_stale_native_peak(monkeypatch) -> None:
    class FakeDevice:
        type = "cuda"

    def failed_reset(device):
        raise RuntimeError("reset failed")

    monkeypatch.setattr(
        memory_module.torch.cuda, "reset_peak_memory_stats", failed_reset
    )
    monkeypatch.setattr(memory_module, "allocated_memory_bytes", lambda device: 7)
    monitor = MemoryMonitor(FakeDevice())
    monkeypatch.setattr(monitor, "_native", lambda name: 9999)
    monitor.begin_update()
    metrics = monitor.end_update()
    assert "memory/device_peak_allocated_bytes" not in metrics
    assert "memory/device_peak_reserved_bytes" not in metrics
    assert metrics["memory/device_allocated_bytes"] == 7


def _publication_proposal():
    config = load_config(Path("configs/smoke_mla_cpu.yaml"))
    runtime = _runtime()
    return plan_memory(
        config, runtime, estimate_memory(config, runtime, parameter_inventory(config))
    )


def test_proposal_report_binds_exact_yaml_and_loadable_configuration(
    tmp_path: Path,
) -> None:
    output, report_path = write_resource_proposal(
        _publication_proposal(), tmp_path / "proposal.yaml"
    )
    report = json.loads(report_path.read_text())
    assert report["format_version"] == 1
    assert report["yaml_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    reloaded = load_config(output).model_dump(mode="json")
    assert report["config_sha256"] == config_sha256(reloaded)


@pytest.mark.parametrize("existing", ["proposal.yaml", "proposal.yaml.decisions.json"])
def test_proposal_conflict_preserves_user_artifact_and_publishes_nothing_else(
    tmp_path: Path, existing: str
) -> None:
    original = b"user-owned contents, not a generated proposal\n"
    destination = tmp_path / existing
    destination.write_bytes(original)
    with pytest.raises(FileExistsError):
        write_resource_proposal(_publication_proposal(), tmp_path / "proposal.yaml")
    assert destination.read_bytes() == original
    assert {path.name for path in tmp_path.iterdir()} == {existing}


@pytest.mark.parametrize("failure_point", ["temporary_sync", "config_publication"])
def test_proposal_storage_failure_exposes_no_config_or_orphan_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    output = tmp_path / "proposal.yaml"
    failure = OSError("injected storage failure")
    if failure_point == "temporary_sync":

        def fail_sync(_descriptor):
            raise failure

        monkeypatch.setattr(memory_module.os, "fsync", fail_sync)
    else:
        link = memory_module.os.link

        def fail_publication(source, destination):
            if destination == output:
                raise failure
            return link(source, destination)

        monkeypatch.setattr(memory_module.os, "link", fail_publication)
    with pytest.raises(OSError):
        write_resource_proposal(_publication_proposal(), output)
    assert list(tmp_path.iterdir()) == []


def test_native_calibration_is_additive_stable_and_source_specific(tmp_path):
    config = load_config(Path("configs/smoke_mla_cpu.yaml"))
    runtime = _runtime(backend="cuda", device_total=10**12, device_free=10**12)
    inventory = parameter_inventory(config)
    baseline = estimate_memory(config, runtime, inventory)
    native_peak = 2 * (baseline.peak_bytes - baseline.headroom_bytes)
    observation = {
        "estimate": asdict(baseline),
        "peak_method": "native",
        "observed": {"memory/device_peak_allocated_bytes": native_peak},
    }
    store = ExperimentStore(tmp_path)
    store.create_run("warmup", {}, {})
    key = calibration_key(config, runtime, source_digest="a" * 64)
    store.record_calibration(key, "warmup", observation)
    records = ExperimentStore.get_calibration(tmp_path, key)
    corrected = calibrated_estimate(config, runtime, inventory, records)
    assert corrected.peak_bytes == 2 * baseline.peak_bytes
    assert corrected.calibration_sample_count == 1
    assert corrected.peak_bytes == sum(
        getattr(corrected, field)
        for field in (
            "resident_weights_bytes",
            "runtime_buffers_bytes",
            "gradients_bytes",
            "optimizer_bytes",
            "activations_bytes",
            "attention_working_bytes",
            "workspace_bytes",
            "headroom_bytes",
        )
    )
    repeated = {"observation": {**observation, "estimate": asdict(corrected)}}
    sampled = {"observation": {**observation, "peak_method": "sampled_lower_bound"}}
    again = calibrated_estimate(
        config, runtime, inventory, records + [repeated, sampled]
    )
    assert again.peak_bytes == corrected.peak_bytes
    assert again.calibration_sample_count == 2
    changed_source = calibration_key(config, runtime, source_digest="b" * 64)
    assert (
        calibrated_estimate(
            config,
            runtime,
            inventory,
            ExperimentStore.get_calibration(tmp_path, changed_source),
        )
        == baseline
    )


def test_max_fit_applies_only_explicit_optimizer_alternative():
    base = load_config(Path("configs/smoke_mla_cpu.yaml"))
    payload = base.model_dump(mode="json")
    payload["model"].update(hidden_dim=256, num_layers=4, num_heads=4, ffn_dim=1024)
    payload["training"].update(micro_batch_size=1, seq_len=16)
    payload["runtime"]["memory"].update(
        policy="max_fit", allowed_optimizers=["adafactor"]
    )
    payload["runtime"]["memory"]["activation_checkpointing"]["enabled"] = True
    config = base.__class__.model_validate(payload)
    runtime = _runtime()
    adam = estimate_memory(config, runtime, parameter_inventory(config))
    alternate = config.model_dump(mode="json")
    alternate["optimizer"] = {
        "name": "adafactor",
        **{
            name: getattr(config.optimizer, name)
            for name in ("peak", "floor", "warmup_steps", "weight_decay")
        },
    }
    factored = config.__class__.model_validate(alternate)
    lower = estimate_memory(factored, runtime, parameter_inventory(factored))
    payload["runtime"]["memory"]["budget_bytes"] = (
        adam.peak_bytes + lower.peak_bytes
    ) // 2
    config = config.__class__.model_validate(payload)
    original = config.model_dump(mode="json")
    proposal = plan_memory(
        config, runtime, estimate_memory(config, runtime, parameter_inventory(config))
    )
    assert proposal.config["optimizer"]["name"] == "adafactor"
    assert proposal.estimate.result == "LIKELY_TO_FIT"
    assert config.model_dump(mode="json") == original
    assert any(
        item["requested"] == "optimizer" and item["scientifically_significant"]
        for item in proposal.decisions
    )


def test_policy_cannot_discard_measured_uncertainty_to_claim_a_fit() -> None:
    base = load_config(Path("configs/smoke_mla_cpu.yaml"))
    payload = base.model_dump(mode="json")
    payload["training"].update(micro_batch_size=4, gradient_accumulation=1)
    payload["runtime"]["memory"]["policy"] = "low_memory"
    config = type(base).model_validate(payload)
    small = config.model_dump(mode="json")
    small["training"].update(micro_batch_size=1, gradient_accumulation=4)
    small["runtime"]["memory"]["activation_checkpointing"]["enabled"] = True
    candidate = type(base).model_validate(small)
    runtime = _runtime()
    raw_candidate = estimate_memory(candidate, runtime, parameter_inventory(candidate))
    payload["runtime"]["memory"]["budget_bytes"] = int(raw_candidate.peak_bytes * 1.5)
    config = type(base).model_validate(payload)
    baseline = estimate_memory(config, runtime, parameter_inventory(config))
    corrected = estimate_memory(
        config, runtime, parameter_inventory(config), calibration=2
    )
    proposal = plan_memory(config, runtime, corrected)
    assert proposal.estimate.result == "LIKELY_TO_EXCEED"
    assert proposal.estimate.peak_bytes >= (
        raw_candidate.peak_bytes + corrected.peak_bytes - baseline.peak_bytes
    )
    assert proposal.estimate.calibration_sample_count == 0


def test_sdpa_buffer_estimate_matches_registered_runtime_allocations():
    import torch

    from sparselab.model.transformer import DenseLM

    config = load_config(Path("configs/smoke_cpu.yaml"))
    config = config.model_copy(
        update={
            "attention": config.attention.model_copy(update={"implementation": "sdpa"})
        }
    )
    model = DenseLM(config.model, config.attention).eval()
    with torch.inference_mode():
        model(torch.zeros((1, config.training.seq_len), dtype=torch.long))
    actual = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    assert registered_runtime_buffers_bytes(config) == actual
