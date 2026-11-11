from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from klab.canonical import canonical_digest, canonical_json
from klab.config.loading import load_config
from klab.config.models import RunConfig
from klab.model.inspection import named_tensor_inventory
from klab.model.transformer import DenseLM
from klab.training.checkpoints import CheckpointManager, TrainingSnapshot
from klab.training.mlx_checkpoints import validate_optimizer_state


def _zero_optimizer(groups: list[list[str]], learning_rate: float) -> dict[str, object]:
    return {
        "state": {
            **{
                f"states.{group}.{field}": np.asarray(
                    0 if field == "step" else learning_rate,
                    dtype=np.uint64 if field == "step" else np.float32,
                )
                for group in range(2)
                for field in ("step", "learning_rate")
            },
        },
    }


def _mlx_snapshot() -> TrainingSnapshot:
    resolved = load_config(Path("configs/smoke_cpu.yaml"))
    config = resolved.model_dump(mode="json")
    config["runtime"]["engine"] = "mlx"
    config["runtime"]["backend"] = "metal"
    numpy = np.random.RandomState(7).get_state()
    inventory = named_tensor_inventory(resolved.model, resolved.attention)
    groups = [
        sorted(
            name
            for name, spec in inventory.items()
            if spec.trainable and spec.alias_of is None and len(spec.shape) >= 2
        ),
        sorted(
            name
            for name, spec in inventory.items()
            if spec.trainable and spec.alias_of is None and len(spec.shape) < 2
        ),
    ]
    model = DenseLM(resolved.model, resolved.attention)
    return TrainingSnapshot(
        model=model.state_dict(),
        optimizer=_zero_optimizer(groups, resolved.optimizer.peak),
        schedule={
            "kind": "warmup_cosine_v1",
            "completed_updates": 0,
            "max_steps": resolved.training.max_steps,
            "warmup_steps": resolved.optimizer.warmup_steps,
            "peak": resolved.optimizer.peak,
            "floor": resolved.optimizer.floor,
        },
        step=0,
        tokens_seen=0,
        cursor=(0, 0),
        config=config,
        run_id="mlx-codec",
        engine="mlx",
        backend="metal",
        optimizer_parameter_names=groups,
        rng={
            "python": random.Random(3).getstate(),
            "numpy_kind": numpy[0],
            "numpy_keys": numpy[1],
            "numpy_pos": numpy[2],
            "numpy_has_gauss": numpy[3],
            "numpy_cached_gaussian": numpy[4],
            "mlx": np.array([1, 2], dtype=np.uint32),
        },
    )


def test_common_manager_promotes_mlx_canonical_weights_without_native_codec(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(tmp_path)
    snapshot = _mlx_snapshot()
    record = manager.save(snapshot)
    generation = tmp_path / "checkpoints" / record.relative_path
    # Weights-only promotion still checks every declared file's integrity, but
    # must not deserialize a native state that another engine cannot consume.
    native = generation / "training_state.json"
    native.write_text("{")
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        if entry["name"] == native.name:
            entry["bytes"] = native.stat().st_size
            entry["sha256"] = hashlib.sha256(native.read_bytes()).hexdigest()
    manifest.pop("sha256")
    manifest["sha256"] = canonical_digest(manifest)
    manifest_path.write_bytes(canonical_json(manifest))
    promoted = manager.load(generation, mode="promote")

    config = RunConfig.model_validate(snapshot.config)
    expected = DenseLM(config.model, config.attention).eval()
    actual = DenseLM(config.model, config.attention).eval()
    expected.load_state_dict(snapshot.model)
    actual.load_state_dict(promoted.model)
    inputs = torch.tensor([[0, 1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(actual(inputs), expected(inputs), rtol=0, atol=0)
    with pytest.raises(ValueError):
        manager.load(generation, mode="resume")


def test_common_manager_rejects_unknown_native_codec_and_strict_pointer(
    tmp_path: Path,
) -> None:
    from klab.workers.models import (
        current_required_versions,
        validate_required_versions,
    )

    manager = CheckpointManager(tmp_path)
    record = manager.save(_mlx_snapshot())
    root = tmp_path / "checkpoints"
    generation = root / record.relative_path
    manifest = json.loads((generation / "manifest.json").read_text())
    required = current_required_versions()
    required["state_codecs"][manifest["state_codec"]] = [
        manifest["state_codec_version"]
    ]
    validate_required_versions(required)
    manifest["state_codec_version"] = 2
    manifest.pop("sha256")

    manifest["sha256"] = canonical_digest(manifest)
    (generation / "manifest.json").write_text(json.dumps(manifest))
    assert not manager.verify(generation).valid

    (root / "latest.json").write_text(
        json.dumps(
            {
                "format_version": True,
                "relative_path": record.relative_path,
                "manifest_sha256": record.manifest_sha256,
            }
        )
    )
    assert not manager.verify(root / "latest.json").valid


def test_mlx_optimizer_validator_accepts_initialized_and_updated_adamw_state() -> None:
    groups = [["weight"], ["bias"]]
    shapes = {"weight": (2, 2), "bias": (2,)}
    state = _zero_optimizer(groups, 0.003)

    assert (
        validate_optimizer_state(state, groups, shapes, step=0, learning_rate=0.003)
        == 0
    )

    state["state"]["states.0.weight.m"] = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="incomplete"):
        validate_optimizer_state(state, groups, shapes, step=0, learning_rate=0.003)

    state["state"].update(
        {
            "states.0.weight.v": np.zeros((2, 2), dtype=np.float32),
            "states.1.bias.m": np.zeros((2,), dtype=np.float32),
            "states.1.bias.v": np.zeros((2,), dtype=np.float32),
        }
    )
    assert (
        validate_optimizer_state(state, groups, shapes, step=0, learning_rate=0.003)
        == 0
    )

    for group in range(2):
        state["state"][f"states.{group}.step"] = np.asarray(1, dtype=np.uint64)
    for name in (
        "states.0.weight.m",
        "states.0.weight.v",
        "states.1.bias.m",
        "states.1.bias.v",
    ):
        state["state"][name].fill(1)
    assert (
        validate_optimizer_state(state, groups, shapes, step=1, learning_rate=0.003)
        == 1
    )


def test_corrupt_native_optimizer_is_rejected_and_recovery_falls_back(tmp_path):
    manager = CheckpointManager(tmp_path)
    first = manager.save(_mlx_snapshot())
    second = manager.save(_mlx_snapshot())
    generation = tmp_path / "checkpoints" / second.relative_path
    optimizer = generation / "optimizer.safetensors"
    optimizer.write_bytes(b"invalid safetensors")
    manifest_path = generation / "manifest.json"
    raw = json.loads(manifest_path.read_text())
    for entry in raw["files"]:
        if entry["name"] == optimizer.name:
            entry["bytes"] = optimizer.stat().st_size
            entry["sha256"] = hashlib.sha256(optimizer.read_bytes()).hexdigest()
    raw.pop("sha256")
    raw["sha256"] = canonical_digest(raw)
    manifest_path.write_bytes(canonical_json(raw))
    assert not manager.verify(generation).valid
    recovered = manager.recovery_report()
    assert recovered.record is not None
    assert recovered.record.manifest_sha256 == first.manifest_sha256


@pytest.mark.parametrize(
    "native_file", ["training_state.json", "optimizer.safetensors"]
)
def test_native_sync_failure_cannot_publish_a_committed_generation(
    tmp_path, monkeypatch, native_file
):
    sync = os.fsync

    def fail_native_sync(descriptor):
        opened = os.fstat(descriptor)
        for candidate in tmp_path.rglob(native_file):
            saved = candidate.stat()
            if (opened.st_dev, opened.st_ino) == (saved.st_dev, saved.st_ino):
                raise OSError("injected native storage failure")
        return sync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_native_sync)
    manager = CheckpointManager(tmp_path)
    with pytest.raises(OSError):
        manager.save(_mlx_snapshot())
    assert manager.recovery_report().record is None
    assert not (tmp_path / "checkpoints" / "latest.json").exists()
