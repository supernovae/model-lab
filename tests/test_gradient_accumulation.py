from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from test_training import config, equal
from torch.nn import functional

from klab.config.models import AttentionConfig, ModelConfig
from klab.engines.base import (
    EngineNonFiniteError,
    EngineOutOfMemory,
    Microbatch,
)
from klab.engines.pytorch import PyTorchEngine
from klab.model.transformer import DenseLM


def _moe_model() -> DenseLM:
    return DenseLM(
        ModelConfig(
            vocab_size=260,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            ffn_dim=16,
            max_seq_len=8,
            ffn="moe",
            num_experts=3,
            experts_per_token=2,
            shared_expert=True,
            router_aux_loss_coefficient=0.2,
        ),
        AttentionConfig(),
    )


def _masked_objective(
    model: DenseLM,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    checkpointed: bool,
) -> torch.Tensor:
    logits, auxiliary = model.forward_with_aux(
        inputs,
        valid_target_mask=labels != -100,
        activation_checkpointing=checkpointed,
        diagnostics="full",
    )
    return (
        functional.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=-100
        )
        + auxiliary
    )


@pytest.mark.parametrize("checkpointed", [False, True])
def test_masked_final_window_ignores_tail_for_dense_attention_moe(
    checkpointed: bool,
) -> None:
    torch.manual_seed(13)
    original = _moe_model()
    changed_tail = copy.deepcopy(original)
    inputs = torch.randint(0, 260, (1, 6))
    tail_changed_inputs = inputs.clone()
    tail_changed_inputs[:, 4:] = torch.randint(0, 260, (1, 2))
    labels = torch.randint(0, 260, (1, 6))
    labels[:, 4:] = -100

    original_loss = _masked_objective(
        original, inputs, labels, checkpointed=checkpointed
    )
    changed_tail_loss = _masked_objective(
        changed_tail, tail_changed_inputs, labels, checkpointed=checkpointed
    )
    original_loss.backward()
    changed_tail_loss.backward()

    assert torch.allclose(original_loss, changed_tail_loss, atol=1e-6, rtol=1e-5)
    for (name, parameter), (other_name, other_parameter) in zip(
        original.named_parameters(), changed_tail.named_parameters(), strict=True
    ):
        assert name == other_name
        assert parameter.grad is not None and other_parameter.grad is not None
        assert torch.allclose(
            parameter.grad, other_parameter.grad, atol=1e-6, rtol=1e-5
        )


def test_scaler_overflow_leaves_engine_window_uncommitted(tmp_path):
    cfg = config(tmp_path).model_copy(
        update={
            "training": config(tmp_path).training.model_copy(
                update={"micro_batch_size": 1, "gradient_accumulation": 2}
            )
        }
    )
    engine = PyTorchEngine()
    engine.initialize(cfg)
    assert engine.model is not None
    assert engine.optimizer is not None
    engine.scaler = torch.amp.GradScaler("cpu", init_scale=128)
    before = copy.deepcopy(engine.model.state_dict())
    handle = next(engine.model.parameters()).register_hook(
        lambda gradient: torch.full_like(gradient, torch.inf)
    )
    try:
        update = engine.train_update(
            [
                Microbatch(
                    torch.randint(0, 512, (1, 16)).numpy(),
                    torch.randint(0, 512, (1, 16)).numpy(),
                ),
                Microbatch(
                    torch.randint(0, 512, (1, 16)).numpy(),
                    torch.randint(0, 512, (1, 16)).numpy(),
                ),
            ],
            1,
            32,
        )
    finally:
        handle.remove()
        engine.close()
    assert update.outcome == "OVERFLOW"
    assert update.committed_targets == 0
    equal(engine.model.state_dict(), before)


def _optimizer_config(tmp_path, name: str):
    base = config(tmp_path)
    if name == "adamw":
        return base
    payload = base.model_dump(mode="json")
    payload["optimizer"] = {
        "name": "adafactor",
        "peak": base.optimizer.peak,
        "floor": base.optimizer.floor,
        "warmup_steps": base.optimizer.warmup_steps,
        "weight_decay": base.optimizer.weight_decay,
    }
    return type(base).model_validate(payload)


def _full_batch(seed: int, batch_size: int = 1) -> Microbatch:
    generator = np.random.default_rng(seed)
    return Microbatch(
        generator.integers(0, 512, (batch_size, 16), dtype=np.int64),
        generator.integers(0, 512, (batch_size, 16), dtype=np.int64),
    )


def _named_optimizer_slots(engine: PyTorchEngine) -> dict[str, dict[str, torch.Tensor]]:
    assert engine.model is not None and engine.optimizer is not None
    names = {id(parameter): name for name, parameter in engine.model.named_parameters()}
    return {
        names[id(parameter)]: {
            key: value.detach().clone()
            for key, value in values.items()
            if isinstance(value, torch.Tensor)
        }
        for parameter, values in engine.optimizer.state.items()
    }


@pytest.mark.parametrize("optimizer_name", ["adamw", "adafactor"])
def test_named_optimizer_state_survives_serialized_group_reordering(
    tmp_path, optimizer_name: str
) -> None:
    cfg = _optimizer_config(tmp_path, optimizer_name)
    source = PyTorchEngine()
    source.initialize(cfg)
    source.train_update([_full_batch(1)], 1, 16)
    state = source.export_training_state()
    reordered = copy.deepcopy(state.optimizer)
    for group in reordered["param_groups"]:
        group["params"].reverse()
    reordered_state = type(state)(
        reordered,
        state.rng,
        state.scaler,
        copy.deepcopy(state.optimizer_parameter_names),
    )
    expected = _named_optimizer_slots(source)
    restored = PyTorchEngine()
    restored.initialize(cfg)
    try:
        restored.restore_training_state(reordered_state)
        actual = _named_optimizer_slots(restored)
    finally:
        source.close()
        restored.close()
    assert actual.keys() == expected.keys()
    for name in expected:
        equal(actual[name], expected[name])


def test_mismatched_or_empty_target_window_mutates_no_engine_state(tmp_path) -> None:
    engine = PyTorchEngine()
    engine.initialize(config(tmp_path))
    assert engine.model is not None
    before = copy.deepcopy(engine.model.state_dict())
    supervised = _full_batch(2)
    empty = Microbatch(supervised.inputs.copy(), np.full_like(supervised.targets, -100))
    try:
        with pytest.raises(ValueError, match="valid-target count"):
            engine.train_update([supervised], 1, 15)
        with pytest.raises(ValueError, match="at least one supervised target"):
            engine.train_update([empty], 1, 1)
    finally:
        engine.close()
    equal(engine.model.state_dict(), before)


def test_unscaled_nonfinite_gradient_rejects_optimizer_commit(tmp_path) -> None:
    engine = PyTorchEngine()
    engine.initialize(config(tmp_path))
    assert engine.model is not None
    before = copy.deepcopy(engine.model.state_dict())
    handle = next(engine.model.parameters()).register_hook(
        lambda gradient: torch.full_like(gradient, torch.inf)
    )
    try:
        with pytest.raises(EngineNonFiniteError, match="unscaled gradients"):
            engine.train_update([_full_batch(3)], 1, 16)
    finally:
        handle.remove()
        engine.close()
    equal(engine.model.state_dict(), before)


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("CUDA out of memory"),
        RuntimeError("MPS backend out of memory"),
    ],
)
def test_backend_allocation_failures_are_typed(
    error: BaseException, monkeypatch, tmp_path
):
    engine = PyTorchEngine()
    engine.initialize(config(tmp_path))
    assert engine.model is not None

    def fail_forward(*args, **kwargs):
        raise error

    monkeypatch.setattr(engine.model, "forward_with_aux", fail_forward)
    try:
        with pytest.raises(EngineOutOfMemory) as raised:
            engine.train_update([_full_batch(5)], 1, 16)
    finally:
        engine.close()
    assert raised.value.__cause__ is error


def test_recompute_ratio_counts_only_executed_microbatches(tmp_path) -> None:
    base = config(tmp_path)
    cfg = base.model_copy(
        update={
            "runtime": base.runtime.model_copy(
                update={
                    "memory": base.runtime.memory.model_copy(
                        update={
                            "activation_checkpointing": (
                                base.runtime.memory.activation_checkpointing.model_copy(
                                    update={"enabled": True}
                                )
                            )
                        }
                    )
                }
            )
        }
    )
    engine = PyTorchEngine()
    engine.initialize(cfg)
    valid = _full_batch(4)
    empty = Microbatch(valid.inputs.copy(), np.full_like(valid.targets, -100))
    try:
        update = engine.train_update([empty, valid], 1, 16)
    finally:
        engine.close()
    assert update.metrics["recompute/block_call_ratio"] == pytest.approx(1.0)


@pytest.mark.parametrize("padded", [False, True])
def test_accumulated_dense_updates_match_large_batch_over_schedule(
    tmp_path, padded
) -> None:
    base = config(tmp_path)
    # Direct engine initialization does not apply the trainer's configured seed.
    # Keep the numerical comparison independent of earlier tests in this worker.
    torch.manual_seed(base.seed)
    accumulated_cfg = base.model_copy(
        update={
            "training": base.training.model_copy(
                update={"micro_batch_size": 1, "gradient_accumulation": 4}
            )
        }
    )
    large_cfg = base.model_copy(
        update={
            "training": base.training.model_copy(
                update={"micro_batch_size": 4, "gradient_accumulation": 1}
            )
        }
    )
    accumulated = PyTorchEngine()
    accumulated.initialize(accumulated_cfg)
    assert accumulated.model is not None
    initial_weights = copy.deepcopy(accumulated.model.state_dict())
    large = PyTorchEngine()
    large.initialize(large_cfg, initial_weights=initial_weights)
    try:
        for step in range(1, 4):
            microbatches = [_full_batch(step * 10 + offset) for offset in range(4)]
            if padded:
                for offset, batch in enumerate(microbatches):
                    batch.targets[:, 3 + offset * 3 :] = -100
            valid_targets = sum(
                int(np.count_nonzero(batch.targets != -100)) for batch in microbatches
            )
            merged = Microbatch(
                np.concatenate([batch.inputs for batch in microbatches]),
                np.concatenate([batch.targets for batch in microbatches]),
            )
            accumulated.train_update(microbatches, step, valid_targets)
            large.train_update([merged], step, valid_targets)
        assert accumulated.model is not None and large.model is not None
        for name, parameter in accumulated.model.state_dict().items():
            assert torch.allclose(
                parameter,
                large.model.state_dict()[name],
                atol=1e-6,
                rtol=1e-5,
            ), name
        for name, parameter in accumulated.model.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                large.model.get_parameter(name).grad,
                atol=1e-6,
                rtol=1e-5,
            )
        assert accumulated.optimizer is not None and large.optimizer is not None
        torch.testing.assert_close(
            accumulated.optimizer.state_dict(),
            large.optimizer.state_dict(),
            atol=1e-6,
            rtol=1e-5,
        )
    finally:
        accumulated.close()
        large.close()
