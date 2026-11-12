from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

from klab.runtime_environments import (
    RuntimeEntry,
    profile_for_id,
    read_registry,
    register_runtime,
    resolve_registry_path,
    resolve_runtime_dir,
    unregister_runtime,
)


def entry(**changes):
    return RuntimeEntry.model_validate(
        {
            "python": str(Path(sys.executable).absolute()),
            "engine": "pytorch",
            "backend": "cpu",
            "device_index": 0,
            **changes,
        }
    )


def test_missing_registry_is_passive_and_missing_interpreter_stays_visible(tmp_path):
    path = tmp_path / "config/runtimes.yaml"
    assert not read_registry(path).runtimes
    assert not path.parent.exists()
    missing = entry(python=str(tmp_path / "missing/bin/python"))
    register_runtime("missing", missing, registry_path=path)
    assert read_registry(path).runtimes["missing"] == missing
    with pytest.raises(ValueError, match="absolute executable"):
        profile_for_id("missing", registry_path=path)


def test_registry_publication_duplicate_and_unregister_preserve_interpreter(tmp_path):
    path = tmp_path / "runtimes.yaml"
    register_runtime("cpu", entry(), registry_path=path)
    before = path.read_bytes()
    assert profile_for_id("cpu", registry_path=path).backend == "cpu"
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="already registered"):
        register_runtime("cpu", entry(), registry_path=path)
    assert path.read_bytes() == before
    unregister_runtime("cpu", registry_path=path)
    assert not read_registry(path).runtimes
    assert Path(sys.executable).is_file()
    with pytest.raises(ValueError, match="unknown runtime"):
        profile_for_id("cpu", registry_path=path)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "runtime_registry_version: 2\nruntimes: {}",
        "runtime_registry_version: 1\nruntimes:\n  cpu: {}\n  cpu: {}",
    ],
)
def test_registry_malformed_fails_closed(tmp_path, raw):
    path = tmp_path / "runtimes.yaml"
    path.write_text(raw)
    with pytest.raises(ValueError):
        read_registry(path)
    with pytest.raises(ValueError):
        register_runtime("new", entry(), registry_path=path)
    assert path.read_text() == raw


@pytest.mark.parametrize(
    "changes",
    [
        {"python": "relative/python"},
        {"python": 1},
        {"device_index": -1},
        {"backend": "metal"},
        {"unknown": 1},
        {"requirements": {"torch_hip": True}},
    ],
)
def test_registry_entry_strict(changes):
    with pytest.raises(ValueError):
        entry(**changes)


def test_registry_bad_id(tmp_path):
    with pytest.raises(ValueError):
        register_runtime("../bad", entry(), registry_path=tmp_path / "r.yaml")
    assert not (tmp_path / "r.yaml").exists()


def test_roots_are_external_independent_and_passive(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "science"))
    monkeypatch.delenv("KLAB_RUNTIME_DIR", raising=False)
    assert resolve_runtime_dir() == tmp_path / "data/klab/runtimes"
    assert resolve_registry_path() == tmp_path / "config/klab/runtimes.yaml"
    assert not (tmp_path / "data").exists()
    monkeypatch.setenv("KLAB_RUNTIME_DIR", "relative")
    with pytest.raises(ValueError, match="absolute"):
        resolve_runtime_dir()
    monkeypatch.setenv("KLAB_RUNTIME_DIR", str(tmp_path / "science"))
    with pytest.raises(ValueError, match="differ"):
        resolve_runtime_dir()
    monkeypatch.setenv("KLAB_RUNTIME_DIR", str(Path.cwd() / "runtimes"))
    with pytest.raises(ValueError, match="checkout"):
        resolve_runtime_dir()


def test_registries_are_host_local_not_environment_search(tmp_path, monkeypatch):
    a, b = tmp_path / "host-a", tmp_path / "host-b"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(a))
    register_runtime("cpu", entry())
    monkeypatch.setenv("XDG_CONFIG_HOME", str(b))
    assert not read_registry().runtimes
    with pytest.raises(ValueError, match="unknown"):
        profile_for_id("cpu")
    assert not b.exists()
    assert os.access(Path(sys.executable), os.X_OK)
    assert (
        yaml.safe_load((a / "klab/runtimes.yaml").read_text())[
            "runtime_registry_version"
        ]
        == 1
    )


def test_discovery_caps_deduplicates_and_preserves_venv_prefixes(tmp_path):
    from klab.runtime_env_inventory import canonical_python, discover_candidates
    from klab.runtime_environments import RuntimeRegistry

    root = tmp_path / "runtimes"
    for index in range(35):
        (root / f"env-{index:02}/bin").mkdir(parents=True)
    shared = root / "env-00/bin/python"
    shared.symlink_to(sys.executable)
    other = root / "env-01/bin/python"
    other.symlink_to(sys.executable)
    entries = {
        f"id-{index:02}": entry(python=str(tmp_path / f"registered-{index}/python"))
        for index in range(35)
    }
    entries["alias-a"] = entry(python=str(shared))
    entries["alias-b"] = entry(python=str(shared))
    candidates, limits = discover_candidates(
        registry=RuntimeRegistry(runtimes=entries),
        runtime_dir=root,
        project_dir=tmp_path / "project",
    )
    assert limits["truncated"]
    assert limits["runtime_root_candidates"] == 32
    assert limits["registered_candidates"] == 32
    assert len(candidates) <= 66
    candidate = next(
        item for item in candidates if item["python"] == canonical_python(shared)
    )
    assert candidate["ids"] == ["alias-a", "alias-b"]
    assert candidate["origins"] == ["runtime_root", "registered"]
    assert canonical_python(shared) != canonical_python(other)


def test_inventory_missing_candidate_and_real_cpu():
    from klab.runtime_env_inventory import inspect_python

    missing = inspect_python(Path("/nonexistent/klab-runtime-python"))
    assert missing["status"] == "NOT_PROVISIONED"
    observed = inspect_python(Path(sys.executable).absolute())
    assert observed["status"] == "READY"
    assert observed["torch"]["installed"]
    assert observed["torch"]["hip"] is None
    # A CPU-capable environment can also expose host accelerators such as MPS.
    assert "cpu" in observed["backends"]
    assert observed["klab_import"]["success"]


def test_inventory_source_drift_and_declared_backend(monkeypatch):
    from klab import runtime_env_inventory as module
    from klab.runtime_identity_probe import source_identity

    environment = {
        "python": str(module.canonical_python(Path(sys.executable))),
        "status": "READY",
        "reason": None,
        "backends": ["cpu"],
        "devices": {},
    }
    declared = entry(backend="rocm")
    result = module.observed_entry("false-rocm", declared, [environment])
    assert result["status"] == "UNAVAILABLE"
    monkeypatch.setattr(
        module,
        "source_identity",
        lambda: {**source_identity(), "source_sha256": "drift"},
    )
    assert (
        module.inspect_python(Path(sys.executable).absolute())["status"]
        == "SOURCE_MISMATCH"
    )


def test_venv_aliases_deduplicate_without_merging_independent_prefixes(tmp_path):
    from klab.runtime_env_inventory import canonical_python, discover_candidates
    from klab.runtime_environments import RuntimeRegistry

    root = tmp_path / "runtimes"
    for name in ("one", "two"):
        prefix = root / name
        (prefix / "bin").mkdir(parents=True)
        (prefix / "pyvenv.cfg").write_text("home = /usr/bin")
        (prefix / "bin/python").symlink_to(sys.executable)
        (prefix / "bin/python3.14").symlink_to(sys.executable)
    first = root / "one/bin/python"
    alias = root / "one/bin/python3.14"
    second = root / "two/bin/python"
    assert canonical_python(first) == canonical_python(alias)
    assert canonical_python(first) != canonical_python(second)
    registry = RuntimeRegistry(
        runtimes={
            "first": entry(python=str(first)),
            "alias": entry(python=str(alias)),
            "second": entry(python=str(second)),
        }
    )
    candidates, _ = discover_candidates(
        registry=registry, runtime_dir=root, project_dir=tmp_path / "project"
    )
    row = next(item for item in candidates if item["python"] == first)
    assert row["ids"] == ["alias", "first"]
    assert len([item for item in candidates if "registered" in item["origins"]]) == 2


@pytest.mark.parametrize(
    "failed_lock", ["requirements.lock", "source-requirements.lock"]
)
def test_rocm_provision_hash_failure_retains_evidence_without_device_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_lock: str
) -> None:
    import json
    from types import SimpleNamespace

    from klab import runtime_env_provision as provisioner
    from klab.training.manifest import sha256_file

    root = tmp_path / "runtimes"
    monkeypatch.setattr(provisioner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(provisioner.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        provisioner, "read_registry", lambda: SimpleNamespace(runtimes={})
    )
    monkeypatch.setattr(provisioner, "resolve_runtime_dir", lambda: root)
    monkeypatch.setattr(provisioner, "preflight_runtime_root", lambda _: {})
    monkeypatch.setattr(provisioner, "_validate_python", lambda _: {})
    monkeypatch.setattr(provisioner, "_source_commit", lambda _: "a" * 40)
    monkeypatch.setattr(
        provisioner,
        "inspect_python",
        lambda _: pytest.fail("hash failure reached device probe"),
    )
    installs = []

    def fake_uv(command, *, environment, checkout, log):
        assert environment["ROCM_SDK_TARGET_FAMILY"] == "gfx1100"
        if command[1] == "export":
            Path(command[command.index("--output-file") + 1]).write_text(
                "pyyaml==6.0.3\n"
            )
        if "-r" not in command:
            return
        lock = Path(command[command.index("-r") + 1])
        if lock.name not in {"requirements.lock", "source-requirements.lock"}:
            return
        assert "--require-hashes" in command and "--no-deps" in command
        assert (
            "--only-binary=:all:"
            if lock.name == "requirements.lock"
            else "--no-build-isolation"
        ) in command
        installs.append(lock)
        if lock.name == failed_lock:
            raise ValueError("fixture: package hash mismatch")

    monkeypatch.setattr(provisioner, "_run_uv", fake_uv)
    result = provisioner.provision("rocm-test", recipe="rocm-gfx1100-v1")
    assert result["status"] == "ERROR"
    assert "hash mismatch" in result["reason"]
    assert installs[-1].name == failed_lock
    assert len(installs) == (1 if failed_lock == "requirements.lock" else 2)
    receipt = json.loads(Path(result["failure_record"]).read_text())
    assert receipt["dependency_lock_sha256"] == {
        str(path): sha256_file(path)
        for path in (installs[0], installs[0].with_name("source-requirements.lock"))
    }
    assert not (root / "rocm-test/provision-receipt.json").exists()
    with pytest.raises(ValueError, match="target already exists"):
        provisioner.provision("rocm-test", recipe="rocm-gfx1100-v1")
