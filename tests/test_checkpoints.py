from __future__ import annotations

import hashlib
import json
import random
import weakref
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch

from sparselab.canonical import canonical_digest
from sparselab.config.loading import load_config
from sparselab.config.models import RunConfig
from sparselab.engines.base import CanonicalTensor
from sparselab.model.transformer import DenseLM
from sparselab.training.checkpoints import (
    CheckpointManager,
    LineageBest,
    TrainingSnapshot,
    existing_run_recovery_report,
    load_legacy_checkpoint,
    verify_legacy_checkpoint,
)
from sparselab.training.manifest import config_sha256
from sparselab.training.optimizer import (
    learning_rate_for_step,
    make_optimizer,
    schedule_payload,
    set_optimizer_learning_rate,
)


def _cpu_rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy_kind": numpy_state[0],
        "numpy_keys": torch.from_numpy(numpy_state[1].copy()),
        "numpy_pos": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gaussian": float(numpy_state[4]),
        "torch": torch.get_rng_state(),
        "device_type": "cpu",
        "device_index": 0,
        "device_rng": None,
    }


def state() -> dict[str, object]:
    return {
        "format_version": 1,
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {},
        "schedule": {"max_steps": 2},
        "step": 1,
        "tokens_seen": 4,
        "cursor": (0, 1),
        "config": {"schema_version": 1},
    }


def write_legacy_fixture(path: Path) -> None:
    torch.save(state(), path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "filename": path.name,
                "format_version": 1,
                "step": 1,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    )


def test_legacy_checkpoint_reader_validates_existing_artifact(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    write_legacy_fixture(path)
    report = verify_legacy_checkpoint(path)
    assert report.valid
    loaded = load_legacy_checkpoint(path)
    assert loaded["step"] == 1
    path.write_bytes(path.read_bytes() + b"corrupt")
    assert not verify_legacy_checkpoint(path).valid
    with pytest.raises(ValueError, match="invalid legacy checkpoint"):
        load_legacy_checkpoint(path)


def test_legacy_checkpoint_requires_manifest(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save(state(), path)
    assert not verify_legacy_checkpoint(path).valid


def test_pointer_digest_must_bind_selected_generation(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save(_snapshot(1))
    pointer = tmp_path / "checkpoints/latest.json"
    payload = json.loads(pointer.read_text())
    payload["manifest_sha256"] = "0" * 64
    pointer.write_text(json.dumps(payload))
    report = manager.verify(pointer)
    assert not report.valid
    assert any(error["field"] == "pointer" for error in report.errors)


def test_config_verification_distinguishes_promotion_and_full_resume(tmp_path):
    manager = CheckpointManager(tmp_path)
    snapshot = _snapshot(1)
    manager.save(snapshot)
    config = load_config(Path("configs/smoke_cpu.yaml"))
    pointer = tmp_path / "checkpoints" / "latest.json"
    assert manager.verify(pointer, expected_config=config).valid
    different_optimizer = config.model_copy(
        update={
            "optimizer": config.optimizer.model_copy(
                update={"peak": config.optimizer.peak * 2}
            )
        }
    )
    assert not manager.verify(pointer, expected_config=different_optimizer).valid
    assert manager.verify(
        pointer, expected_config=different_optimizer, require_training_state=False
    ).valid
    different_attention = config.model_copy(
        update={
            "attention": config.attention.model_copy(
                update={"kind": "sliding", "window_size": 4}
            )
        }
    )
    assert not manager.verify(
        pointer, expected_config=different_attention, require_training_state=False
    ).valid


def test_checkpoint_verification_rejects_tampered_weight_file(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    record = manager.save(_snapshot(1))
    generation = tmp_path / "checkpoints" / record.relative_path
    weights = next(generation.glob("*.safetensors"))
    weights.write_bytes(weights.read_bytes() + b"tampered")
    assert not manager.verify(generation).valid


def test_checkpoint_allows_symlinked_ancestor_but_not_member(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    manager = CheckpointManager(alias / "run")
    snapshot = _snapshot(1)
    record = manager.save(snapshot)
    generation = alias / "run/checkpoints" / record.relative_path
    loaded = manager.load(generation)
    assert torch.equal(
        loaded.model["embedding.weight"], snapshot.model["embedding.weight"]
    )
    weights = next(generation.glob("*.safetensors"))
    outside = real / "outside.safetensors"
    weights.replace(outside)
    weights.symlink_to(outside)
    assert not manager.verify(generation).valid


def _snapshot(
    step: int,
    loss: float | None = None,
    *,
    decay_steps: int | None = None,
    memory_policy: bool = False,
    stable_steps: int = 0,
) -> TrainingSnapshot:
    config = load_config(Path("configs/smoke_cpu.yaml"))
    values = config.model_dump(mode="json")
    if decay_steps is not None:
        values["optimizer"]["decay_steps"] = decay_steps
    values["training"]["stable_steps"] = stable_steps
    if memory_policy:
        values["model"].update(
            memory="ngram", memory_table_size=17, memory_dim=8, memory_ngram_size=2
        )
        values["optimizer"].update(
            memory_table_lr_multiplier=5.0, memory_table_weight_decay=0.0
        )
    config = RunConfig.model_validate(values)
    model = DenseLM(config.model, config.attention)
    settings = config.optimizer
    optimizer = make_optimizer(
        model,
        settings.peak,
        settings.weight_decay,
        settings.betas,
        settings.eps,
        memory_table_lr_multiplier=settings.memory_table_lr_multiplier,
        memory_table_weight_decay=settings.memory_table_weight_decay,
    )
    inputs = torch.arange(8).reshape(1, 8)
    for update in range(1, step + 1):
        optimizer.zero_grad(set_to_none=True)
        set_optimizer_learning_rate(
            optimizer,
            learning_rate_for_step(
                update,
                config.training.max_steps,
                settings.warmup_steps,
                settings.peak,
                settings.floor,
                decay_steps=decay_steps,
                stable_steps=stable_steps,
            ),
        )
        model(inputs).square().mean().backward()
        optimizer.step()
    optimizer_state = optimizer.state_dict()
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return TrainingSnapshot(
        model=model.state_dict(),
        optimizer=optimizer_state,
        schedule=schedule_payload(config, step),
        step=step,
        tokens_seen=step * 8,
        cursor=(0, step),
        config=config.model_dump(mode="json"),
        run_id="tiny",
        validation_loss=loss,
        rng=_cpu_rng_state(),
        optimizer_parameter_names={
            stored_id: names[id(parameter)]
            for stored, live in zip(
                optimizer_state["param_groups"], optimizer.param_groups, strict=True
            )
            for stored_id, parameter in zip(
                stored["params"], live["params"], strict=True
            )
        },
    )


def test_explicit_decay_schedule_matches_legacy_through_horizon() -> None:
    peak, floor, warmup, decay, maximum = 3e-4, 3e-5, 100, 4096, 16384
    for step in (4095, 4096, 4097, 8192):
        actual = learning_rate_for_step(
            step, maximum, warmup, peak, floor, decay_steps=decay
        )
        expected = learning_rate_for_step(min(step, decay), decay, warmup, peak, floor)
        assert actual == expected
        if step >= decay:
            assert actual == floor


@pytest.mark.parametrize("mutation", ["kind", "decay_steps", "learning_rate"])
def test_explicit_decay_checkpoint_verifies_schedule_and_optimizer_lr(
    tmp_path: Path, mutation: str
) -> None:
    snapshot = _snapshot(6, decay_steps=5)
    assert snapshot.schedule == {
        "kind": "warmup_cosine_floor_v1",
        "completed_updates": 6,
        "max_steps": 40,
        "warmup_steps": 4,
        "peak": 0.003,
        "floor": 0.0003,
        "decay_steps": 5,
    }
    manager = CheckpointManager(tmp_path)
    record = manager.save(snapshot)
    assert manager.verify(tmp_path / "checkpoints" / record.relative_path).valid

    if mutation == "kind":
        snapshot.schedule["kind"] = "warmup_cosine_v1"
    elif mutation == "decay_steps":
        snapshot.schedule["decay_steps"] = 6
    else:
        snapshot.optimizer["param_groups"][0]["lr"] = 0.003
    failure = (
        "optimizer learning rate differs from saved schedule"
        if mutation == "learning_rate"
        else "schedule does not match configuration and completed updates"
    )
    with pytest.raises(ValueError, match=failure):
        manager.save(snapshot)
    assert manager.verify(tmp_path / "checkpoints" / record.relative_path).valid


def test_selected_parent_excludes_later_same_step_and_future_best(tmp_path) -> None:
    manager = CheckpointManager(tmp_path)
    selected = manager.save(_snapshot(1), 0.5)
    manager.save(_snapshot(1), 0.4)
    manager.save(_snapshot(2), 0.1)
    lineage = manager.lineage_best_for_child("source-run", selected.manifest_sha256)
    assert lineage == LineageBest("source-run", selected.manifest_sha256, 1, 0.5)


def test_weight_source_saves_with_bounded_host_staging(tmp_path, monkeypatch) -> None:
    from sparselab.training import checkpoints

    snapshot = _snapshot(1)
    original = snapshot.model
    references = []

    class BoundedSource:
        aliases: ClassVar[dict[str, str]] = {"output.weight": "embedding.weight"}

        def tensors(self):
            for name, value in sorted(original.items()):
                if name in self.aliases:
                    continue
                array = value.detach().cpu().numpy().copy()
                references.append(weakref.ref(array))
                if sum(reference() is not None for reference in references) > 2:
                    raise MemoryError("bounded CPU staging budget exceeded")
                yield CanonicalTensor(name=name, array=array, trainable=True)
                del array

    monkeypatch.setattr(checkpoints, "SHARD_BYTES", 64)
    snapshot.model = {}
    snapshot.weight_source = BoundedSource()
    manager = CheckpointManager(tmp_path)
    saved = manager.save(snapshot)
    restored = manager.load(tmp_path / "checkpoints" / saved.relative_path)
    config = load_config(Path("configs/smoke_cpu.yaml"))
    expected = DenseLM(config.model, config.attention).eval()
    observed = DenseLM(config.model, config.attention).eval()
    expected.load_state_dict(original)
    observed.load_state_dict(restored.model)
    inputs = torch.arange(8).reshape(1, 8)
    with torch.no_grad():
        torch.testing.assert_close(observed(inputs), expected(inputs), rtol=0, atol=0)


def test_reconcile_repairs_pointers_after_finalized_generations(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    first = manager.save(_snapshot(2, 0.5), 0.5)
    second = manager.save(_snapshot(10, 0.5), 0.5)
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "latest.json").write_text('{"relative_path":"missing"}')
    (checkpoints / "best.json").write_text('{"relative_path":"missing"}')
    recovered = manager.reconcile()
    assert recovered.record is not None
    assert recovered.record.relative_path == second.relative_path
    # Equal finite losses use the earliest generation, not lexical pointer order.
    assert (
        json.loads((checkpoints / "best.json").read_text())["relative_path"]
        == first.relative_path
    )


def test_latest_uses_numeric_step_then_generation_order(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    higher_step = manager.save(_snapshot(10, 0.8), 0.8)
    manager.save(_snapshot(2, 0.7), 0.7)
    report = manager.reconcile()
    assert report.record is not None
    assert report.record.relative_path == higher_step.relative_path


def test_explicit_corrupt_resume_never_falls_back(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    first = manager.save(_snapshot(1, 0.8), 0.8)
    second = manager.save(_snapshot(2, 0.7), 0.7)
    generation = tmp_path / "checkpoints" / second.relative_path
    next(generation.glob("*.safetensors")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="invalid checkpoint"):
        manager.load(generation)
    recovered = manager.reconcile()
    assert recovered.record is not None
    assert recovered.record.relative_path == first.relative_path
    assert recovered.rejected


def test_fault_before_finalized_rename_preserves_published_pointers(
    tmp_path: Path,
) -> None:
    stable = CheckpointManager(tmp_path)
    first = stable.save(_snapshot(1, 0.5), 0.5)

    def fail_at(point: str) -> None:
        if point == "before_finalized_rename":
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        CheckpointManager(tmp_path, fault_injector=fail_at).save(_snapshot(2, 0.4), 0.4)
    latest = json.loads((tmp_path / "checkpoints/latest.json").read_text())
    assert latest["relative_path"] == first.relative_path
    assert len(list((tmp_path / "checkpoints").glob("step_*_gen_*"))) == 1


def test_fault_after_finalized_rename_is_recoverable(tmp_path: Path) -> None:
    stable = CheckpointManager(tmp_path)
    first = stable.save(_snapshot(1, 0.5), 0.5)

    def fail_at(point: str) -> None:
        if point == "after_finalized_rename":
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        CheckpointManager(tmp_path, fault_injector=fail_at).save(_snapshot(2, 0.4), 0.4)
    assert (
        json.loads((tmp_path / "checkpoints/latest.json").read_text())["relative_path"]
        == first.relative_path
    )
    recovered = CheckpointManager(tmp_path).reconcile()
    assert recovered.record is not None
    assert recovered.record.step == 2


def test_fault_between_pointer_publications_reconciles_best(tmp_path: Path) -> None:
    stable = CheckpointManager(tmp_path)
    first = stable.save(_snapshot(1, 0.5), 0.5)

    projection_count = 0

    def fail_at(point: str) -> None:
        nonlocal projection_count
        if point == "after_latest_projection_before_best_projection":
            projection_count += 1
            if projection_count == 2:
                raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        CheckpointManager(tmp_path, fault_injector=fail_at).save(_snapshot(2, 0.4), 0.4)
    checkpoints = tmp_path / "checkpoints"
    assert json.loads((checkpoints / "latest.json").read_text())["step"] == 2
    assert json.loads((checkpoints / "best.json").read_text())["relative_path"] == (
        first.relative_path
    )
    CheckpointManager(tmp_path).reconcile()
    assert json.loads((checkpoints / "best.json").read_text())["step"] == 2


def test_lineage_best_is_available_metadata_not_local_best_pointer(
    tmp_path: Path,
) -> None:
    source = CheckpointManager(tmp_path / "source")
    inherited = LineageBest("unavailable-grandparent", "a" * 64, 3, 0.1)
    parent = _snapshot(1, 0.2)
    parent.lineage_best = inherited
    record = source.save(parent, 0.2)
    lineage = source.lineage_best_for_child("source-run", record.manifest_sha256)
    assert lineage == inherited

    child = CheckpointManager(tmp_path / "child")
    snapshot = _snapshot(1, 0.3)
    snapshot.lineage_best = lineage
    child.save(snapshot, 0.3)
    loaded = child.load(tmp_path / "child/checkpoints/latest.json")
    assert loaded.lineage_best == inherited
    assert (
        json.loads((tmp_path / "child/checkpoints/best.json").read_text())["step"] == 1
    )


def test_existing_run_recovery_report_is_read_only(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    record = manager.save(_snapshot(1, 0.5), 0.5)
    pointer = tmp_path / "checkpoints/latest.json"
    pointer.write_text('{"relative_path":"missing"}')
    report = existing_run_recovery_report(tmp_path)
    assert report.record is not None
    assert report.record.relative_path == record.relative_path
    assert json.loads(pointer.read_text())["relative_path"] == "missing"


def test_native_identity_hashes_the_raw_saved_config(tmp_path: Path) -> None:
    snapshot = _snapshot(1, 0.5)
    # A later reader may supply this default, but it must not reinterpret identity.
    snapshot.config["logging"].pop("architecture_diagnostics")
    record = CheckpointManager(tmp_path).save(snapshot, 0.5)
    generation = tmp_path / "checkpoints" / record.relative_path
    native = torch.load(generation / "training_state.pt", weights_only=True)
    assert native["config_sha256"] == config_sha256(snapshot.config)
    assert CheckpointManager(tmp_path).verify(generation).valid


@pytest.mark.parametrize(
    "mutation",
    [
        "counter",
        "moment_shape",
        "missing_moment",
        "missing_parameter",
        "schedule",
        "rng",
        "config",
    ],
)
def test_full_verify_rejects_tampered_native_state(
    tmp_path: Path, mutation: str
) -> None:
    manager = CheckpointManager(tmp_path)
    record = manager.save(_snapshot(1, 0.25), 0.25)
    generation = tmp_path / "checkpoints" / record.relative_path
    state_path = generation / "training_state.pt"
    state = torch.load(state_path, weights_only=True)
    parameter_id = next(iter(state["optimizer"]["state"]))
    if mutation == "counter":
        state["step"] = 99
    elif mutation == "moment_shape":
        state["optimizer"]["state"][parameter_id]["exp_avg"] = torch.zeros(1)
    elif mutation == "missing_moment":
        state["optimizer"]["state"][parameter_id].pop("exp_avg_sq")
    elif mutation == "missing_parameter":
        state["optimizer"]["state"].pop(parameter_id)
    elif mutation == "schedule":
        state["schedule"]["peak"] *= 2
    elif mutation == "rng":
        state["rng"]["python"] = (3, (), None)
    elif mutation == "config":
        state["config"]["model"]["hidden_dim"] *= 2
    torch.save(state, state_path)
    manifest = json.loads((generation / "manifest.json").read_text())
    for entry in manifest["files"]:
        if entry["name"] == "training_state.pt":
            entry["sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
            entry["bytes"] = state_path.stat().st_size
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    manifest["sha256"] = canonical_digest(payload)
    (generation / "manifest.json").write_text(json.dumps(manifest))
    report = manager.verify(generation)
    assert not report.valid
    assert manager.verify(generation, require_training_state=False).valid


def test_checkpoint_reader_does_not_create_missing_run(tmp_path: Path) -> None:
    run = tmp_path / "missing"
    report = CheckpointManager(run).verify(run / "checkpoints/latest.json")
    assert not report.valid
    assert not run.exists()


def test_writer_lease_excludes_another_writer_and_recovery(tmp_path: Path) -> None:
    first, second = CheckpointManager(tmp_path), CheckpointManager(tmp_path)
    with first.writer_lease():
        record = first.save(_snapshot(0))
        with pytest.raises(RuntimeError, match="lease is held"):
            second.save(_snapshot(0))
        with pytest.raises(RuntimeError, match="lease is held"):
            second.reconcile()
    recovered = second.reconcile()
    assert recovered.record is not None
    assert recovered.record.manifest_sha256 == record.manifest_sha256


def test_retention_keeps_best_latest_and_recovery_predecessor(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, keep_periodic=False)
    with manager.writer_lease():
        oldest = manager.save(_snapshot(0), 1.0)
        best = manager.save(_snapshot(1), 0.5)
        previous = manager.save(_snapshot(2), 0.75)
        latest = manager.save(_snapshot(3), 0.8)
    checkpoints = tmp_path / "checkpoints"
    assert not (checkpoints / oldest.relative_path).exists()
    for record in (best, previous, latest):
        assert manager.verify(checkpoints / record.relative_path).valid
    weights = next((checkpoints / latest.relative_path).glob("*.safetensors"))
    weights.write_bytes(weights.read_bytes() + b"corrupt")
    recovered = manager.reconcile()
    assert recovered.record is not None
    assert recovered.record.manifest_sha256 == previous.manifest_sha256
    assert recovered.rejected
    assert (
        json.loads((checkpoints / "best.json").read_text())["manifest_sha256"]
        == best.manifest_sha256
    )


def test_checkpoint_binds_memory_table_optimizer_and_stable_schedule(tmp_path):
    manager = CheckpointManager(tmp_path)
    snapshot = _snapshot(1, memory_policy=True, stable_steps=3)
    manager.save(snapshot)
    assert manager.verify(tmp_path / "checkpoints/latest.json").valid
    assert snapshot.schedule["kind"] == "warmup_stable_cosine_v1"
    assert snapshot.schedule["stable_steps"] == 3
    snapshot.optimizer["param_groups"][-1]["lr"] /= 5
    with pytest.raises(ValueError, match="learning rate"):
        CheckpointManager(tmp_path / "tampered").save(snapshot)


@pytest.mark.parametrize("corrupt_best", [False, True])
def test_lineage_verifies_only_candidates_that_can_contribute(
    tmp_path, monkeypatch, corrupt_best
) -> None:
    manager = CheckpointManager(tmp_path)
    records = [
        manager.save(_snapshot(step), loss)
        for step, loss in enumerate((0.6, 0.2, 0.5, 0.7), start=1)
    ]
    if corrupt_best:
        shard = next(
            (manager.root / records[1].relative_path).glob("weights-*.safetensors")
        )
        shard.write_bytes(shard.read_bytes() + b"corrupt")
    original = manager.verify
    opened = []

    def verify(path, *args, **kwargs):
        opened.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(manager, "verify", verify)
    winner = records[2 if corrupt_best else 1]
    assert manager.lineage_best_for_child("parent", records[-1].manifest_sha256) == (
        LineageBest(
            "parent", winner.manifest_sha256, winner.step, winner.validation_loss
        )
    )
    assert opened == [
        records[-1].relative_path,
        records[1].relative_path,
        *([records[2].relative_path] if corrupt_best else []),
    ]
    monkeypatch.setattr(manager, "verify", original)
    assert bool(manager.recovery_report().rejected) == corrupt_best


@pytest.mark.parametrize("corrupt_newer_ancestor", [False, True])
def test_lineage_keeps_inherited_tie_order_and_verifies_contributor(
    tmp_path, corrupt_newer_ancestor
) -> None:
    manager = CheckpointManager(tmp_path)
    ancestors = [
        LineageBest("older", "a" * 64, 1, 0.1),
        LineageBest("newer", "b" * 64, 1, 0.1),
    ]
    records = []
    for step, ancestor in enumerate(ancestors, start=1):
        snapshot = _snapshot(step)
        snapshot.lineage_best = ancestor
        records.append(manager.save(snapshot, 0.5))
    selected = manager.save(_snapshot(3), 0.1)
    if corrupt_newer_ancestor:
        shard = next(
            (manager.root / records[-1].relative_path).glob("weights-*.safetensors")
        )
        shard.write_bytes(shard.read_bytes() + b"corrupt")
    assert (
        manager.lineage_best_for_child("parent", selected.manifest_sha256)
        == (ancestors[0 if corrupt_newer_ancestor else 1])
    )


def test_lineage_rejects_corrupt_selected_parent(tmp_path) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save(_snapshot(1), 0.1)
    selected = manager.save(_snapshot(2), 0.2)
    shard = next((manager.root / selected.relative_path).glob("weights-*.safetensors"))
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(
        ValueError, match="selected parent is not a verified generation"
    ):
        manager.lineage_best_for_child("parent", selected.manifest_sha256)


def test_lineage_local_tie_keeps_earliest_generation(tmp_path) -> None:
    manager = CheckpointManager(tmp_path)
    first = manager.save(_snapshot(1), 0.1)
    manager.save(_snapshot(2), 0.1)
    selected = manager.save(_snapshot(3), 0.3)
    assert manager.lineage_best_for_child("parent", selected.manifest_sha256) == (
        LineageBest("parent", first.manifest_sha256, 1, 0.1)
    )
