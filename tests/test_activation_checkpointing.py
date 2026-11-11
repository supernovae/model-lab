from __future__ import annotations

import copy

import pytest
import torch
from torch.nn import functional

from klab.config.models import AttentionConfig, ModelConfig
from klab.model.attention.sparse import BlockSparseAttention
from klab.model.moe import TopKMoE
from klab.model.transformer import DenseLM


def _model(
    *,
    ffn: str = "dense",
    memory: str = "none",
    attention: str = "dense",
    memory_injection: str = "final",
    memory_variant: str = "hidden_gate",
    memory_conv_kernel_size: int = 0,
) -> DenseLM:
    options = (
        {"memory_table_size": 32, "memory_ngram_size": 2, "memory_dim": 8}
        if memory in {"byte", "ngram"}
        else {}
    )
    config = ModelConfig(
        vocab_size=260,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        ffn_dim=16,
        max_seq_len=8,
        ffn=ffn,
        num_experts=3 if ffn == "moe" else 1,
        experts_per_token=2 if ffn == "moe" else 1,
        shared_expert=ffn == "moe",
        router_aux_loss_coefficient=0.1 if ffn == "moe" else 0.0,
        memory_injection=memory_injection,
        memory_block_index=0 if memory_injection == "after_block" else None,
        memory_variant=memory_variant,
        memory_conv_kernel_size=memory_conv_kernel_size,
        memory=memory,
        **options,
    )
    return DenseLM(
        config,
        AttentionConfig(kind="block_sparse", block_size=2, selected_blocks=1)
        if attention == "sparse"
        else AttentionConfig(),
    )


def _loss_and_update(
    model: DenseLM,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    checkpointed: bool,
    addresses: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logits, auxiliary = model.forward_with_aux(
        inputs,
        byte_addresses=addresses,
        valid_target_mask=labels != -100,
        activation_checkpointing=checkpointed,
        diagnostics="full",
    )
    loss = functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), ignore_index=-100
    )
    (loss + auxiliary).backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer.step()
    return logits.detach(), auxiliary.detach(), gradients


@pytest.mark.parametrize(
    ("ffn", "memory", "placement"),
    [
        ("dense", "none", "final"),
        ("moe", "none", "final"),
        ("moe", "byte", "final"),
        ("dense", "byte", "embedding"),
        ("dense", "ngram", "embedding"),
    ],
)
def test_nonreentrant_recomputation_matches_uncheckpointed_forward_gradients_and_update(
    ffn: str, memory: str, placement: str
) -> None:
    torch.manual_seed(17)
    plain = _model(ffn=ffn, memory=memory, memory_injection=placement)
    checkpointed = copy.deepcopy(plain)
    inputs = torch.randint(0, 260, (2, 6))
    labels = torch.randint(0, 260, (2, 6))
    labels[:, -1] = -100
    addresses = torch.randint(0, 32, (2, 6)) if memory == "byte" else None

    plain_logits, plain_auxiliary, plain_gradients = _loss_and_update(
        plain, inputs, labels, checkpointed=False, addresses=addresses
    )
    checkpointed_logits, checkpointed_auxiliary, checkpointed_gradients = (
        _loss_and_update(
            checkpointed, inputs, labels, checkpointed=True, addresses=addresses
        )
    )

    assert torch.allclose(plain_logits, checkpointed_logits, atol=1e-6, rtol=1e-5)
    assert torch.allclose(plain_auxiliary, checkpointed_auxiliary, atol=1e-6, rtol=1e-5)
    assert plain_gradients.keys() == checkpointed_gradients.keys()
    for name, gradient in plain_gradients.items():
        assert torch.allclose(
            gradient, checkpointed_gradients[name], atol=1e-6, rtol=1e-5
        )
        assert torch.allclose(
            dict(plain.named_parameters())[name],
            dict(checkpointed.named_parameters())[name],
            atol=1e-6,
            rtol=1e-5,
        )
    assert checkpointed.recomputed_block_call_ratio == 1.0


def test_checkpoint_recomputation_preserves_forward_diagnostic_snapshot() -> None:
    torch.manual_seed(4)
    model = _model(ffn="moe", attention="sparse")
    inputs = torch.randint(0, 260, (2, 6))
    labels = torch.randint(0, 260, (2, 6))
    logits, auxiliary = model.forward_with_aux(
        inputs,
        valid_target_mask=labels != -100,
        activation_checkpointing=True,
        diagnostics="full",
    )
    moe = model.blocks[0].ffn
    attention = model.blocks[0].attention
    assert isinstance(moe, TopKMoE)
    assert isinstance(attention, BlockSparseAttention)
    forward_routing = moe.last_diagnostics
    forward_selection = attention.last_diagnostics
    assert forward_routing is not None and forward_routing.router_logits is not None
    assert (
        forward_selection is not None and forward_selection.selected_blocks is not None
    )
    (logits.square().mean() + auxiliary).backward()
    assert moe.last_diagnostics is forward_routing
    assert attention.last_diagnostics is forward_selection
    assert model.recomputed_block_call_ratio == 1.0


def test_scalar_diagnostics_avoid_full_router_and_dense_teacher_snapshots() -> None:
    model = _model(ffn="moe", attention="sparse")
    _, _ = model.forward_with_aux(torch.randint(0, 260, (1, 5)), diagnostics="scalar")
    moe = model.blocks[0].ffn
    attention = model.blocks[0].attention
    assert isinstance(moe, TopKMoE)
    assert isinstance(attention, BlockSparseAttention)
    assert moe.last_diagnostics is not None
    assert moe.last_diagnostics.router_logits is None
    assert moe.last_diagnostics.selected_experts is None
    assert attention.last_diagnostics is not None
    assert attention.last_diagnostics.selected_blocks is None
    assert attention.last_diagnostics.dense_teacher_mass is None


def test_embedding_memory_diagnostics_are_not_recomputed() -> None:
    torch.manual_seed(29)
    model = _model(memory="ngram", memory_injection="embedding")
    inputs = torch.randint(0, 260, (2, 6))
    logits, auxiliary = model.forward_with_aux(
        inputs,
        activation_checkpointing=True,
        diagnostics="full",
    )
    assert model.memory is not None and model.memory.last_diagnostics is not None
    forward_diagnostic = model.memory.last_diagnostics
    before = model.architecture_metric_tensors()
    indicator_name = "engram/injection/embedding"
    assert {name for name in before if name.startswith("engram/injection/")} == {
        indicator_name
    }
    before_indicator = before[indicator_name]

    (logits.square().mean() + auxiliary).backward()

    after = model.architecture_metric_tensors()
    assert model.memory.last_diagnostics is forward_diagnostic
    assert model.recomputed_block_call_ratio == 1.0
    assert after.keys() == before.keys()
    torch.testing.assert_close(after[indicator_name], before_indicator)


@pytest.mark.parametrize("kernel", [0, 4])
def test_context_memory_checkpointed_gradients_and_update(kernel: int) -> None:
    torch.manual_seed(17)
    plain = _model(
        memory="ngram",
        memory_injection="after_block",
        memory_variant="context_key",
        memory_conv_kernel_size=kernel,
    )
    checkpointed = copy.deepcopy(plain)
    inputs = torch.randint(0, 260, (2, 6))
    labels = torch.randint(0, 260, (2, 6))
    left = _loss_and_update(plain, inputs, labels, checkpointed=False)
    right = _loss_and_update(checkpointed, inputs, labels, checkpointed=True)
    torch.testing.assert_close(left[0], right[0])
    for name in left[2]:
        torch.testing.assert_close(left[2][name], right[2][name])
    for name, value in plain.state_dict().items():
        torch.testing.assert_close(value, checkpointed.state_dict()[name])
