from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from test_training import config as training_config

from klab import runtime
from klab.cli.main import main
from klab.runtime_profile import (
    RuntimeProfile,
    authorize_profile,
    load_runtime_profile,
    probe_runtime_profile,
)


def profile_value(**changes):
    return {
        "runtime_profile_version": 1,
        "id": "offline-cpu",
        "python": str(Path(sys.executable).absolute()),
        "engine": "pytorch",
        "backend": "cpu",
        "device_index": 0,
        "requirements": {},
        **changes,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"unexpected": True},
        {"runtime_profile_version": 2},
        {"id": "../unsafe"},
        {"python": "relative/python"},
        {"engine": "mlx", "backend": "cpu"},
        {"device_index": -1},
        {"requirements": {"device_name_regex": "["}},
        {"requirements": {"torch_hip": True}},
        {"requirements": {"bf16": 1}},
    ],
)
def test_profile_rejects_unsafe_or_ambiguous_values(tmp_path, changes):
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile_value(**changes)))
    with pytest.raises((ValueError, TypeError)):
        load_runtime_profile(path)


def test_profile_rejects_nonexecutable_file(tmp_path):
    python = tmp_path / "python"
    python.write_text("not an interpreter")
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile_value(python=str(python))))
    with pytest.raises(ValueError):
        load_runtime_profile(path)


def test_real_cpu_probe_and_authorization(tmp_path):
    profile = RuntimeProfile.model_validate(profile_value())
    result = probe_runtime_profile(profile)
    assert result["schema_version"] == 1
    authorization = authorize_profile(profile, training_config(tmp_path))
    info = runtime.validate_runtime(
        training_config(tmp_path), authorization=authorization
    )
    assert info.backend == "cpu"
    assert "forward_backward_optimizer" in info.tested_features


def test_mlx_namespace_package_probe_uses_loaded_core_identity(tmp_path, monkeypatch):
    from klab.runtime_identity_probe import probe

    namespace = ModuleType("mlx")
    namespace.__file__ = None
    namespace.__path__ = [str(tmp_path)]
    core = ModuleType("mlx.core")
    core.__file__ = str(tmp_path / "core.so")
    core.__version__ = "0.32.2"
    core.metal = SimpleNamespace(is_available=lambda: True)
    namespace.core = core
    monkeypatch.setitem(sys.modules, "mlx", namespace)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "torch", None)

    result = probe(
        {
            "id": "mlx-namespace",
            "engine": "mlx",
            "backend": "metal",
            "device_index": 0,
        }
    )
    assert result["framework_path"] == str(Path(core.__file__).resolve())
    assert result["framework_version"] == "0.32.2"
    assert result["available"] is True
    assert result["torch_path"] is None


@pytest.mark.parametrize("command", ["train", "stage", "experiment submit"])
def test_accelerator_cli_rejects_before_workdir(tmp_path, monkeypatch, command):
    config = training_config(tmp_path)
    value = config.model_dump(mode="json")
    value["runtime"]["backend"] = "cuda"
    path = tmp_path / "accelerator.yaml"
    path.write_text(yaml.safe_dump(value))
    work = tmp_path / "must-not-exist"
    monkeypatch.setenv("KLAB_WORK_DIR", str(work))
    args = ["klab", *command.split(), str(path)]
    if command == "experiment submit":
        args += ["--wait"]
    if command == "stage":
        args += ["--output", str(tmp_path / "stage")]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit, match="runtime-profile|registered"):
        main()
    assert not work.exists()
    assert not (tmp_path / "stage").exists()
    assert not config.logging.root_dir.exists()


@pytest.mark.parametrize("backend", ["cuda", "metal"])
def test_unregistered_acceleration_is_not_authorized(tmp_path, backend):
    config = training_config(tmp_path)
    config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "backend": backend,
                    "engine": "mlx" if backend == "metal" else "pytorch",
                }
            )
        }
    )
    with pytest.raises(ValueError, match="runtime-profile|registered"):
        runtime.validate_runtime(config)


def fake_runtime(monkeypatch, config, backend):
    import klab.runtime_identity_probe as identity
    import klab.runtime_profile as profiles

    engine = "mlx" if backend == "metal" else "pytorch"
    evidence = {
        "schema_version": 1,
        "profile_id": "offline-cpu",
        "engine": engine,
        "backend": backend,
        "device_index": 0,
        "python": str(Path(sys.executable).resolve()),
        "available": True,
        "device_count": 1,
        "device_name": "Fixture Accelerator",
        "device_names": ["Fixture Accelerator"],
        "torch_hip": "fixture-hip" if backend == "rocm" else None,
        "torch_cuda": "fixture-cuda" if backend == "cuda" else None,
        "bf16_supported": None,
        **identity.source_identity(),
    }
    monkeypatch.setattr(profiles, "_probe", lambda *args, **kwargs: dict(evidence))
    monkeypatch.setattr(identity, "probe", lambda request: dict(evidence))
    config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={"engine": engine, "backend": backend}
            )
        }
    )
    return evidence, config


@pytest.mark.parametrize("backend", ["cuda", "rocm", "metal"])
def test_matched_vendor_probe_authorizes_without_hardware(
    tmp_path, monkeypatch, backend
):
    from klab.runtime_profile import require_authorization

    _, config = fake_runtime(monkeypatch, training_config(tmp_path), backend)
    profile = RuntimeProfile.model_validate(
        profile_value(engine=config.runtime.engine, backend=backend)
    )
    authorization = authorize_profile(profile, config)
    require_authorization(config, authorization)


@pytest.mark.parametrize(
    ("backend", "requirements", "changed", "message"),
    [
        ("rocm", {"torch_hip": True}, {"torch_hip": None}, "CUDA/HIP|torch_hip"),
        ("cuda", {}, {"torch_hip": "fixture-hip"}, "CUDA/HIP"),
        ("xpu", {}, {"available": False}, "unavailable"),
        ("mps", {"device_name_regex": "Different"}, {}, "device name"),
    ],
)
def test_profile_capability_mismatch_rejected(
    tmp_path, monkeypatch, backend, requirements, changed, message
):
    evidence, config = fake_runtime(monkeypatch, training_config(tmp_path), backend)
    evidence.update(changed)
    profile = RuntimeProfile.model_validate(
        profile_value(backend=backend, requirements=requirements)
    )
    with pytest.raises(ValueError, match=message):
        authorize_profile(profile, config)


def test_bf16_requirement_needs_disposable_update(tmp_path, monkeypatch):
    _, config = fake_runtime(monkeypatch, training_config(tmp_path), "cuda")
    profile = RuntimeProfile.model_validate(
        profile_value(backend="cuda", requirements={"bf16": True})
    )
    calls = []

    def precision_update(requested, *, authorization):
        calls.append((requested.runtime.precision, authorization))
        return SimpleNamespace(tested_precisions=("fp32",), as_dict=dict)

    monkeypatch.setattr(runtime, "validate_runtime", precision_update)
    with pytest.raises(ValueError, match="BF16.*not verified"):
        authorize_profile(profile, config)
    assert [precision for precision, _ in calls] == ["bf16"]


def test_requested_bf16_without_profile_requirement_still_tests_optimizer(
    tmp_path, monkeypatch
):
    _, config = fake_runtime(monkeypatch, training_config(tmp_path), "rocm")
    config = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"precision": "bf16"})}
    )
    profile = RuntimeProfile.model_validate(profile_value(backend="rocm"))
    calls = []

    def tested_update(requested, *, authorization):
        calls.append(requested.runtime.precision)
        return SimpleNamespace(tested_precisions=("fp32",), as_dict=dict)

    monkeypatch.setattr(runtime, "validate_runtime", tested_update)
    with pytest.raises(ValueError, match="BF16.*not verified"):
        authorize_profile(profile, config)
    assert calls == ["bf16"]


def test_registered_ssh_worker_validates_probe_source_and_remote_transport(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from klab.runtime_identity_probe import probe
    from klab.training.manifest import source_identity
    from klab.workers import transport
    from klab.workers.controller import Controller
    from klab.workers.models import WorkerCapabilities, WorkerDefinition

    config = training_config(tmp_path)
    definition = WorkerDefinition(
        worker_id="remote-cpu",
        name="remote-cpu",
        transport="ssh",
        host="fixture-host",
        python=Path(sys.executable).absolute(),
        root=tmp_path,
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    request = {
        "id": definition.worker_id,
        "engine": "pytorch",
        "backend": "cpu",
        "device_index": 0,
    }
    observed = probe(request)
    cpu = next(info for info in runtime.discover_runtimes() if info.backend == "cpu")
    tested = replace(
        cpu,
        device_name=observed["device_name"],
        framework_version=observed["framework_version"],
        tested_precisions=("fp32",),
        tested_features=(),
        validated_at="fixture-validation",
    )
    from klab.workers.execution import _caps

    capability = WorkerCapabilities.model_validate(
        _caps(definition, validation_status="unverified", runtime=tested)
    )
    controller = Controller(tmp_path / "controller")
    monkeypatch.setattr(controller, "_worker_for_attempt", lambda _: definition)
    monkeypatch.setattr(controller.store, "save_worker", lambda *args: None)
    seen = []

    def fake_call_worker(worker, operation, payload, **kwargs):
        seen.append(payload["config"])
        return SimpleNamespace(
            result={
                "validation_status": "passed",
                "reason": None,
                "runtime": tested.as_dict(),
                "runtime_authorization": {
                    "authorization_version": 1,
                    "kind": "worker",
                    "descriptor": {
                        **definition.model_dump(mode="json"),
                        "transport": "local",
                        "host": None,
                    },
                    "probe": observed,
                },
            }
        )

    monkeypatch.setattr(transport, "call_worker", fake_call_worker)
    verified, evidence, reason = controller._validate_candidate(
        capability, config, source_sha256=source_identity()["sha256"]
    )
    assert reason is None
    assert verified.runtime["tested_precisions"] == ["fp32"]
    assert evidence["probe"]["source_sha256"] == observed["source_sha256"]
    assert seen == [config.model_dump(mode="json")]
    observed["source_sha256"] = "0" * 64
    verified, _, reason = controller._validate_candidate(
        capability, config, source_sha256=source_identity()["sha256"]
    )
    assert verified is None
    assert "identity" in reason


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("source_sha256", "0" * 64), ("device_name", "Changed Accelerator")],
)
def test_authorization_rejects_runtime_drift(tmp_path, monkeypatch, field, replacement):
    from klab.runtime_profile import require_authorization

    evidence, config = fake_runtime(monkeypatch, training_config(tmp_path), "rocm")
    profile = RuntimeProfile.model_validate(profile_value(backend="rocm"))
    authorization = authorize_profile(profile, config)
    evidence[field] = replacement
    with pytest.raises(ValueError, match="identity changed|source changed"):
        require_authorization(config, authorization)


@pytest.mark.parametrize(
    "target", [{"device_index": 1}, {"engine": "mlx", "backend": "metal"}]
)
def test_sealed_authorization_cannot_select_another_runtime(
    tmp_path, monkeypatch, target
):
    from klab.runtime_profile import require_authorization

    evidence, config = fake_runtime(monkeypatch, training_config(tmp_path), "rocm")
    evidence["device_count"] = 2
    profile = RuntimeProfile.model_validate(profile_value(backend="rocm"))
    authorization = authorize_profile(profile, config)
    selected = config.model_copy(
        update={"runtime": config.runtime.model_copy(update=target)}
    )
    with pytest.raises(ValueError, match="runtime selection"):
        require_authorization(selected, authorization)
