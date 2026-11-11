from __future__ import annotations

import torch

from klab.config.models import AttentionConfig, ModelConfig
from klab.model.inspection import inspect_model
from klab.model.moe import TopKMoE
from klab.model.transformer import DenseLM


def test_top_k_dispatch_preserves_token_order_and_normalizes_weights() -> None:
    moe = TopKMoE(
        hidden_dim=2,
        ffn_dim=4,
        num_experts=2,
        experts_per_token=2,
        auxiliary_loss_coefficient=0.1,
    )
    with torch.no_grad():
        moe.router.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        for parameter in moe.experts[0].parameters():
            parameter.zero_()
        for parameter in moe.experts[1].parameters():
            parameter.zero_()
    values = torch.tensor([[[2.0, 0.0], [-2.0, 0.0], [3.0, 0.0]]])
    output = moe(values)
    assert output.shape == values.shape
    assert moe.last_diagnostics is not None
    assert moe.last_diagnostics.counts.tolist() == [3, 3]
    assert torch.isclose(moe.last_diagnostics.fractions.sum(), torch.tensor(1.0))
    assert moe.last_diagnostics.auxiliary_loss > 0


def test_moe_decoder_runs_backward_and_reports_active_expert() -> None:
    config = ModelConfig(
        vocab_size=512,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        ffn_dim=32,
        max_seq_len=16,
        ffn="moe",
        num_experts=3,
        experts_per_token=2,
        shared_expert=True,
        router_aux_loss_coefficient=0.01,
    )
    model = DenseLM(config, AttentionConfig())
    logits = model(torch.randint(0, config.vocab_size, (2, 8)))
    logits.mean().backward()
    inspection = inspect_model(model)
    assert inspection["expert"] > 0
    assert inspection["ffn"] == 0
    assert inspection["active_per_token"] < inspection["total"]
    for block in model.blocks:
        assert isinstance(block.ffn, TopKMoE)
        assert block.ffn.last_diagnostics is not None
        assert int(block.ffn.last_diagnostics.counts.sum()) == 32


def test_selected_probability_mass_is_measured_before_renormalization() -> None:
    moe = TopKMoE(2, 4, 3, experts_per_token=2)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe(torch.ones(1, 2, 2))
    assert torch.isclose(
        moe.last_diagnostics.mean_topk_probability, torch.tensor(2 / 3)
    )


def test_full_routing_snapshot_contains_only_supervised_tokens() -> None:
    torch.manual_seed(19)
    moe = TopKMoE(4, 8, 3, experts_per_token=2)
    values = torch.randn(1, 4, 4)
    valid = torch.tensor([[True, True, False, False]])
    moe(values, valid_target_mask=valid, diagnostics="full")
    first = moe.last_diagnostics
    changed = values.clone()
    changed[:, 2:] += 100
    moe(changed, valid_target_mask=valid, diagnostics="full")
    second = moe.last_diagnostics
    assert first is not None and second is not None
    for name in ("router_logits", "selected_experts", "selected_weights"):
        original, altered = getattr(first, name), getattr(second, name)
        assert original.shape[0] == 2
        torch.testing.assert_close(original, altered, rtol=0, atol=0)


def test_dispatch_matches_dense_weighted_expert_sum_with_idle_experts() -> None:
    torch.manual_seed(0)
    moe = TopKMoE(hidden_dim=8, ffn_dim=16, num_experts=16, experts_per_token=2)
    values = torch.randn(1, 3, 8)
    output = moe(values)
    tokens = values.reshape(-1, 8)
    probabilities = torch.softmax(moe.router(tokens), dim=-1)
    weights, selected = probabilities.topk(2, dim=-1)
    dense = torch.zeros(tokens.shape[0], 16).scatter(
        1, selected, weights / weights.sum(dim=-1, keepdim=True)
    )
    expected = sum(
        dense[:, index : index + 1] * expert(tokens)
        for index, expert in enumerate(moe.experts)
    )
    assert torch.allclose(output.reshape(-1, 8), expected, atol=1e-6)
    output.sum().backward()
    # Three tokens choose at most six of sixteen experts; the rest never run.
    used = set(selected.flatten().tolist())
    for index, expert in enumerate(moe.experts):
        assert (expert.down.weight.grad is not None) == (index in used)
