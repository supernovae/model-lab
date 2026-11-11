"""Table-only AdamW policy and stable-before-decay schedule checks."""

from itertools import pairwise

import pytest
import torch
from torch import nn

from klab.model.memory import TokenNgramMemory
from klab.training.optimizer import (
    is_ngram_table_parameter,
    learning_rate_for_step,
    make_optimizer,
    set_optimizer_learning_rate,
)


def test_table_multiplier_updates_all_ports_without_changing_projections():
    model = nn.Module()
    model.memory = TokenNgramMemory(4, 7, 2, 4, hash_heads=2)
    model.extra_memories = nn.ModuleDict({"3": TokenNgramMemory(4, 7, 2, 4)})
    model.backbone = nn.Linear(4, 4)
    optimizer = make_optimizer(
        model,
        0.01,
        0.1,
        (0.0, 0.0),
        1e-8,
        memory_table_lr_multiplier=5.0,
        memory_table_weight_decay=0.0,
    )
    table_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if is_ngram_table_parameter(name)
    }
    assert len(table_ids) == 3
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(
        list(model.parameters())
    )
    for base_lr in (0.01, 0.003):
        set_optimizer_learning_rate(optimizer, base_lr)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                table = id(parameter) in table_ids
                assert group["lr"] == base_lr * (5 if table else 1)
                assert group["weight_decay"] == (
                    0 if table or parameter.ndim < 2 else 0.1
                )
    before = {
        id(parameter): parameter.detach().clone() for parameter in model.parameters()
    }
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for parameter in model.parameters():
        if id(parameter) in table_ids:
            assert torch.allclose(
                before[id(parameter)] - parameter,
                torch.full_like(parameter, 0.015),
                atol=1e-7,
            )
    saved = optimizer.state_dict()
    restored = make_optimizer(
        model,
        0.01,
        0.1,
        (0.0, 0.0),
        1e-8,
        memory_table_lr_multiplier=5.0,
        memory_table_weight_decay=0.0,
    )
    restored.load_state_dict(saved)
    set_optimizer_learning_rate(restored, 0.002)
    assert restored.param_groups[-1]["lr"] == 0.01


def test_default_optimizer_preserves_two_groups_and_dense_policy_refuses():
    model = nn.Linear(3, 3)
    optimizer = make_optimizer(model, 0.01, 0.1, (0.9, 0.95), 1e-8)
    assert len(optimizer.param_groups) == 2
    assert all("lr_multiplier" not in group for group in optimizer.param_groups)
    with pytest.raises(ValueError, match="requires token-ngram"):
        make_optimizer(
            model, 0.01, 0.1, (0.9, 0.95), 1e-8, memory_table_lr_multiplier=5
        )


def test_warmup_stable_decay_boundaries_and_floor():
    rates = [
        learning_rate_for_step(step, 20, 2, 1.0, 0.1, decay_steps=10, stable_steps=3)
        for step in range(1, 21)
    ]
    assert rates[:5] == [0.5, 1.0, 1.0, 1.0, 1.0]
    assert all(left > right for left, right in pairwise(rates[4:10]))
    assert rates[9:] == [0.1] * 11
    for step in range(1, 21):
        assert learning_rate_for_step(
            step, 20, 2, 1.0, 0.1, stable_steps=0
        ) == learning_rate_for_step(step, 20, 2, 1.0, 0.1)
