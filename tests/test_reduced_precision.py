from __future__ import annotations

from pathlib import Path

import pytest
import torch
from test_memory_planner import _runtime

from klab.config.loading import load_config
from klab.config.models import RunConfig
from klab.memory import estimate_memory, plan_memory
from klab.model.inspection import parameter_inventory
from klab.training.reduced_precision import (
    Bf16GradientBuffers,
    ReducedPrecisionAdamW,
    seeded_generator,
    stochastic_round_bf16,
)


def test_fp32_storage_reproduces_torch_adamw() -> None:
    torch.manual_seed(0)
    reference = [
        torch.nn.Parameter(torch.randn(6, 5)),
        torch.nn.Parameter(torch.randn(5)),
    ]
    ours = [torch.nn.Parameter(p.detach().clone()) for p in reference]
    options = {"lr": 1e-2, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1}
    expected = torch.optim.AdamW(reference, foreach=False, **options)
    actual = ReducedPrecisionAdamW(ours, state_dtype=torch.float32, seed=1, **options)
    for _ in range(6):
        for first, second in zip(reference, ours, strict=True):
            first.grad = torch.randn_like(first)
            second.grad = first.grad.clone()
        expected.step()
        actual.step()
    for first, second in zip(reference, ours, strict=True):
        assert torch.allclose(first, second, rtol=1e-6, atol=1e-7)


def test_stochastic_rounding_is_unbiased_and_keeps_representable_values() -> None:
    generator = seeded_generator(torch.device("cpu"), seed=3, index=1)
    # Between the bf16 neighbours 1.0 and 1 + 2**-7: round-to-nearest always
    # gives 1.0 and loses the increment; stochastic rounding keeps its mean.
    value = torch.full((200_000,), 1.0 + 2**-10)
    mean = stochastic_round_bf16(value, generator).float().mean().item()
    assert mean == pytest.approx(1.0 + 2**-10, abs=2e-4)
    exact = torch.tensor([-1.5, 0.0, 0.25, 3.0])
    assert torch.equal(stochastic_round_bf16(exact, generator).float(), exact)


def _train_teacher_student(*, reduced: bool) -> float:
    """Fit a fixed random teacher with accumulation, as the engine updates."""
    torch.manual_seed(0)
    teacher = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4)
    )
    student = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4)
    )
    parameters = list(student.parameters())
    options = {"lr": 1e-2, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1}
    optimizer = (
        ReducedPrecisionAdamW(parameters, state_dtype=torch.bfloat16, seed=1, **options)
        if reduced
        else torch.optim.AdamW(parameters, foreach=False, **options)
    )
    buffers = Bf16GradientBuffers(parameters, seed=1) if reduced else None
    loss = torch.zeros(())
    for update in range(1, 301):
        optimizer.zero_grad(set_to_none=True)
        if buffers is not None:
            buffers.begin_update(update)
        for _ in range(4):
            inputs = torch.randn(32, 16)
            loss = (student(inputs) - teacher(inputs).detach()).square().mean()
            (loss / 4).backward()
        if buffers is not None:
            buffers.publish()
            buffers.clip_(1.0, update)
        else:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, foreach=False)
        optimizer.step()
    return loss.item()


def test_bf16_gradients_and_moments_reach_the_fp32_loss() -> None:
    reference = _train_teacher_student(reduced=False)
    narrowed = _train_teacher_student(reduced=True)
    assert reference < 0.05
    assert narrowed == pytest.approx(reference, abs=0.02)


def test_estimate_counts_each_narrowed_byte_and_the_planner_offers_them() -> None:
    base = load_config(Path("configs/smoke_cpu.yaml"))
    inventory = parameter_inventory(base)
    runtime = _runtime()
    reference = estimate_memory(base, runtime, inventory)
    narrowed = base.model_copy(
        update={
            "optimizer": base.optimizer.model_copy(update={"state_dtype": "bf16"}),
            "runtime": base.runtime.model_copy(update={"gradient_dtype": "bf16"}),
        }
    )
    estimate = estimate_memory(narrowed, runtime, inventory)
    assert (
        reference.gradients_bytes - estimate.gradients_bytes == 2 * inventory.trainable
    )
    assert (
        reference.optimizer_bytes - estimate.optimizer_bytes == 4 * inventory.trainable
    )
    assert reference.resident_weights_bytes == estimate.resident_weights_bytes

    memory = base.runtime.memory.model_copy(
        update={
            "policy": "max_fit",
            "budget_bytes": 1,
            "allowed_low_precision": ("bf16_gradients", "bf16_optimizer_state"),
        }
    )
    constrained = base.model_copy(
        update={"runtime": base.runtime.model_copy(update={"memory": memory})}
    )
    proposal = plan_memory(
        constrained, runtime, estimate_memory(constrained, runtime, inventory)
    )
    assert proposal.config["runtime"]["gradient_dtype"] == "bf16"
    assert proposal.config["optimizer"]["state_dtype"] == "bf16"
    assert {
        decision["requested"]
        for decision in proposal.decisions
        if decision.get("scientifically_significant")
    } >= {"bf16_gradients", "bf16_optimizer_state"}


def test_reduced_precision_is_refused_where_it_cannot_hold() -> None:
    base = load_config(Path("configs/smoke_cpu.yaml")).model_dump(mode="json")
    adafactor = {**base, "optimizer": {"name": "adafactor"}}
    adafactor["runtime"] = {**base["runtime"], "gradient_dtype": "bf16"}
    with pytest.raises(ValueError, match="requires AdamW"):
        RunConfig.model_validate(adafactor)
    replicas = {**base}
    replicas["runtime"] = {
        **base["runtime"],
        "gradient_dtype": "bf16",
        "secondary_device_indices": [1],
    }
    with pytest.raises(ValueError, match="gradient_dtype: bf16"):
        RunConfig.model_validate(replicas)
