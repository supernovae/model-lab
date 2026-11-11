from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from klab import runtime


@pytest.mark.parametrize(
    ("system", "release", "distro", "interop", "host_os", "environment"),
    [
        ("Linux", "6.8.0-generic", "", "", "linux", "native"),
        ("Darwin", "25.0.0", "Ubuntu", "", "macos", "native"),
        ("Windows", "11", "Ubuntu", "", "windows", "native"),
        ("Linux", "6.6.87.2-microsoft-standard-WSL2", "", "", "linux", "wsl2"),
        ("Linux", "5.4.72-microsoft-standard", "", "", "linux", "wsl2"),
        ("Linux", "4.4.0-Microsoft", "", "", "linux", "unknown-wsl"),
        ("Linux", "custom", "Ubuntu", "", "linux", "unknown-wsl"),
        ("Linux", "custom", "", "/run/WSL/1_interop", "linux", "unknown-wsl"),
    ],
)
def test_host_environment_detection(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    release: str,
    distro: str,
    interop: str,
    host_os: str,
    environment: str,
) -> None:
    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    monkeypatch.setattr(runtime.platform, "release", lambda: release)
    monkeypatch.setattr(runtime.platform, "machine", lambda: "test-architecture")
    monkeypatch.setenv("WSL_DISTRO_NAME", distro)
    monkeypatch.setenv("WSL_INTEROP", interop)

    assert runtime.detect_host_environment() == {
        "host_os": host_os,
        "host_environment": environment,
        "host_architecture": "test-architecture",
    }


@pytest.mark.parametrize("backend", ["cpu", "cuda", "rocm", "xpu"])
@pytest.mark.parametrize("wsl", [False, True])
def test_host_inventory_is_independent_of_backend(
    monkeypatch: pytest.MonkeyPatch, backend: str, wsl: bool
) -> None:
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        runtime.platform,
        "release",
        lambda: "6.6-microsoft-standard-WSL2" if wsl else "6.6-generic",
    )
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: backend in {"cuda", "rocm"})
    monkeypatch.setattr(torch.version, "hip", "test-hip" if backend == "rocm" else None)
    monkeypatch.setattr(
        torch, "xpu", SimpleNamespace(is_available=lambda: backend == "xpu")
    )
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _: None)

    infos = runtime.discover_runtimes()

    assert runtime._auto_backend() == backend
    assert {info.host_os for info in infos} == {"linux"}
    assert {info.host_environment for info in infos} == {"wsl2" if wsl else "native"}
    selected = next(info for info in infos if info.backend == backend)
    assert selected.torch_device == str(runtime.torch_device_for(backend))
    assert runtime.RuntimeInfo.from_dict(selected.as_dict()) == selected


def test_legacy_runtime_does_not_infer_host_from_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _mlx_probe_result(features=[])["runtime"]
    assert isinstance(value, dict)
    for name in ("host_os", "host_environment", "host_architecture"):
        value.pop(name)

    def unexpected_detection():
        pytest.fail("historical runtime readers must not detect the current host")

    monkeypatch.setattr(runtime, "detect_host_environment", unexpected_detection)
    info = runtime.RuntimeInfo.from_dict(value)
    assert info.os == "test"
    assert info.host_os is None
    assert info.host_environment is None
    assert info.host_architecture is None
    assert runtime.RuntimeInfo.from_dict(info.as_dict()) == info


def _config(
    *,
    backend: str = "cpu",
    precision: str = "fp32",
    index: int = 0,
    checkpointing: bool = False,
):
    return SimpleNamespace(
        runtime=SimpleNamespace(
            engine="pytorch",
            backend=backend,
            precision=precision,
            device_index=index,
            memory=SimpleNamespace(
                activation_checkpointing=SimpleNamespace(enabled=checkpointing),
                activation_offload=SimpleNamespace(enabled=False),
            ),
        ),
        optimizer=SimpleNamespace(name="adamw", state_offload=False),
    )


def _mlx_probe_result(*, features: list[str]) -> dict[str, object]:
    info = runtime.RuntimeInfo(
        engine="mlx",
        backend="metal",
        torch_device=None,
        device_index=0,
        device_name="Apple Metal",
        physical_device_id="apple-metal:test",
        framework_version="0.32.2",
        runtime_version="0.32.2",
        driver_version="test",
        os="test",
        system_total_bytes=1,
        system_available_bytes=1,
        device_total_bytes=None,
        device_free_bytes=None,
        device_recommended_bytes=None,
        measurement_source="mlx native",
        measured_at="now",
        precision_capabilities=("fp32",),
        tested_precisions=("fp32",),
        tested_features=tuple(features),
        validated_at="now",
    )
    return {
        "format_version": 1,
        "ok": True,
        "tested_precision": "fp32",
        "tested_features": features,
        "runtime": info.as_dict(),
    }


def test_cpu_probe_isolated_from_parent_rng() -> None:
    torch.manual_seed(918)
    before = torch.get_rng_state().clone()

    info = runtime.validate_runtime(_config())

    assert torch.equal(torch.get_rng_state(), before)
    assert info.backend == "cpu"
    assert info.tested_precisions == ("fp32",)
    assert info.tested_features == ("forward_backward_optimizer", "optimizer:adamw")
    assert info.validated_at is not None


def test_explicit_unavailable_backend_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_backend_available", lambda backend: backend == "cpu")
    import klab.runtime_profile as profiles

    monkeypatch.setattr(
        profiles, "require_authorization", lambda config, authorization: None
    )

    with pytest.raises(ValueError, match="requested backend unavailable: mps"):
        runtime.validate_runtime(_config(backend="mps"))


def test_hip_inventory_does_not_claim_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime, "_backend_available", lambda backend: backend == "rocm"
    )
    monkeypatch.setattr(torch.version, "hip", "6.2")

    infos = {
        info.backend: info
        for info in runtime.discover_runtimes()
        if info.engine == "pytorch"
    }

    assert infos["rocm"].torch_device == "cuda:0"
    assert "PyTorch build targets the other CUDA/HIP API" in infos["cuda"].limitations
    assert infos["cuda"].precision_capabilities == ()


def test_cpu_fp16_is_rejected_before_probe() -> None:
    with pytest.raises(ValueError, match="fp16 is unsupported on CPU"):
        runtime.validate_runtime(_config(precision="fp16"))


def test_mps_inventory_does_not_offer_rejected_fp16(monkeypatch):
    import klab.runtime_profile as profiles

    monkeypatch.setattr(
        profiles, "require_authorization", lambda config, authorization: None
    )
    monkeypatch.setattr(runtime, "_backend_available", lambda backend: backend == "mps")
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _: None)
    info = next(item for item in runtime.discover_runtimes() if item.backend == "mps")
    assert info.precision_capabilities == ("fp32", "bf16")
    assert info.tested_precisions == ()
    with pytest.raises(ValueError, match="fp16 requires a supported GradScaler on mps"):
        runtime.validate_runtime(_config(backend="mps", precision="fp16"))


def test_state_offload_stays_explicitly_unsupported() -> None:
    config = _config()
    config.optimizer.state_offload = True

    with pytest.raises(ValueError, match="state_offload is deferred"):
        runtime.validate_runtime(config)


def test_bf16_checkpoint_probe_reports_only_exercised_precision_and_features():
    requested = _config(precision="bf16", checkpointing=True)
    info = runtime.validate_runtime(requested)
    assert info.tested_precisions == ("bf16",)
    assert "activation_checkpointing" in info.tested_features
    assert requested.runtime.precision == "bf16"
