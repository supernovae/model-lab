from __future__ import annotations

import copy

import pytest
import torch

from sparselab.config.models import AttentionConfig, ModelConfig
from sparselab.model.memory import ByteAddressMemory, TokenNgramMemory
from sparselab.model.portable_engram import (
    PortableEngramAdapter,
    export_portable_engram,
)
from sparselab.model.transformer import DenseLM


def test_ngram_address_does_not_read_future_ids() -> None:
    memory = TokenNgramMemory(hidden_dim=4, table_size=97, ngram_size=3, value_dim=3)
    left = torch.tensor([[4, 8, 15, 16, 23]])
    right = left.clone()
    right[:, 3:] = torch.tensor([42, 99])
    assert torch.equal(memory.addresses(left)[:, :3], memory.addresses(right)[:, :3])


def test_multi_order_multi_head_addresses_remain_causal() -> None:
    memory = TokenNgramMemory(
        hidden_dim=4,
        table_size=97,
        ngram_size=3,
        value_dim=3,
        ngram_orders=(2, 3, 4),
        hash_heads=2,
    )
    left = torch.tensor([[4, 8, 15, 16, 23]])
    right = left.clone()
    right[:, 3:] = torch.tensor([42, 99])
    for order in memory.ngram_orders:
        for head in range(memory.hash_heads):
            assert torch.equal(
                memory.addresses(left, order, head)[:, :3],
                memory.addresses(right, order, head)[:, :3],
            )
    assert len(memory.extra_tables) == 5


def test_multi_stream_collision_diagnostics_aggregate_all_lookups() -> None:
    memory = TokenNgramMemory(
        hidden_dim=4,
        table_size=2,
        ngram_size=2,
        value_dim=3,
        ngram_orders=(2, 3),
        hash_heads=2,
    )
    hidden = torch.randn(1, 8, 4)
    memory(hidden, torch.zeros(1, 8, dtype=torch.long))
    diagnostics = memory.last_diagnostics
    assert diagnostics is not None
    assert int(diagnostics.lookup_count) == 32
    assert len(diagnostics.streams) == 4
    assert all(int(stream.lookup_count) == 8 for stream in diagnostics.streams)
    assert sum(int(stream.lookup_count) for stream in diagnostics.streams) == int(
        diagnostics.lookup_count
    )
    assert all(int(stream.collision_count) > 0 for stream in diagnostics.streams)
    assert int(diagnostics.collision_count) > 0
    assert float(diagnostics.bucket_reuse_rate) > 0


def test_disabled_memory_is_absent_and_enabled_memory_runs_backward() -> None:
    plain = ModelConfig(
        vocab_size=512,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        max_seq_len=16,
    )
    assert DenseLM(plain, AttentionConfig()).memory is None
    config = plain.model_copy(
        update={
            "memory": "ngram",
            "memory_table_size": 31,
            "memory_ngram_size": 3,
            "memory_dim": 7,
        }
    )
    model = DenseLM(config, AttentionConfig())
    logits = model(torch.randint(0, 512, (2, 8)))
    logits.mean().backward()
    assert model.memory is not None
    assert model.memory.last_diagnostics is not None
    diagnostics = model.memory.last_diagnostics
    assert int(diagnostics.lookup_count) == 16
    assert 0 < int(diagnostics.unique_addresses) <= 16
    assert int(diagnostics.collision_count) == 16 - int(diagnostics.unique_addresses)
    assert 0 <= float(diagnostics.bucket_reuse_rate) < 1
    assert 0 < float(diagnostics.table_utilization) <= 1
    assert 0 < float(diagnostics.gate_mean) < 1


def _legacy_final_forward(
    model: DenseLM,
    input_ids: torch.Tensor,
    byte_addresses: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = model.embedding(input_ids)
    auxiliary_loss = hidden.new_zeros(())
    for block in model.blocks:
        hidden, block_auxiliary = block.forward_with_aux(hidden)
        auxiliary_loss = auxiliary_loss + block_auxiliary
    hidden = model.norm(hidden)
    if isinstance(model.memory, TokenNgramMemory):
        hidden = model.memory(hidden, input_ids)
    elif isinstance(model.memory, (ByteAddressMemory, PortableEngramAdapter)):
        if byte_addresses is None:
            raise ValueError("test reference requires byte addresses")
        hidden = model.memory(hidden, byte_addresses)
    return model.output(hidden), auxiliary_loss


@pytest.mark.parametrize("memory_kind", ["none", "ngram", "byte", "portable"])
def test_final_memory_matches_legacy_forward_outputs_and_gradients(
    memory_kind: str, tmp_path
) -> None:
    package = None
    if memory_kind == "portable":
        package = tmp_path / "memory.engram"
        export_portable_engram(torch.zeros(17, 5), package, ngram_size=3)
    config = ModelConfig(
        vocab_size=260,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        ffn_dim=16,
        max_seq_len=8,
        memory=memory_kind,
        memory_table_size=17 if memory_kind != "none" else 0,
        memory_ngram_size=3 if memory_kind != "none" else 0,
        memory_dim=5 if memory_kind != "none" else 0,
        memory_package_path=package,
    )
    torch.manual_seed(19)
    actual = DenseLM(config, AttentionConfig())
    reference = copy.deepcopy(actual)
    placement_peer = DenseLM(
        config.model_copy(update={"memory_injection": "embedding"}),
        AttentionConfig(),
    )
    assert actual.state_dict().keys() == placement_peer.state_dict().keys()
    assert {
        name: (tuple(parameter.shape), parameter.requires_grad)
        for name, parameter in actual.named_parameters()
    } == {
        name: (tuple(parameter.shape), parameter.requires_grad)
        for name, parameter in placement_peer.named_parameters()
    }

    input_ids = torch.tensor([[3, 7, 11, 13], [5, 9, 15, 17]])
    addresses = (
        torch.randint(0, 17, input_ids.shape)
        if memory_kind in {"byte", "portable"}
        else None
    )
    logits, auxiliary = actual.forward_with_aux(input_ids, byte_addresses=addresses)
    expected, expected_auxiliary = _legacy_final_forward(
        reference, input_ids, addresses
    )
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(auxiliary, expected_auxiliary)
    if memory_kind == "portable":
        assert isinstance(actual.memory, PortableEngramAdapter)
        assert not actual.memory.embedding.weight.requires_grad

    (logits.square().mean() + auxiliary).backward()
    (expected.square().mean() + expected_auxiliary).backward()
    actual_parameters = dict(actual.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    assert actual_parameters.keys() == reference_parameters.keys()
    for name, parameter in actual_parameters.items():
        reference_parameter = reference_parameters[name]
        if parameter.grad is None:
            assert reference_parameter.grad is None
        else:
            assert reference_parameter.grad is not None
            torch.testing.assert_close(parameter.grad, reference_parameter.grad)


def test_embedding_injection_carries_prefix_memory_into_later_logits() -> None:
    torch.manual_seed(23)
    config = ModelConfig(
        vocab_size=260,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        ffn_dim=32,
        max_seq_len=8,
        memory="ngram",
        memory_table_size=97,
        memory_ngram_size=3,
        memory_dim=8,
    )
    final_template = DenseLM(config, AttentionConfig()).eval()
    input_ids = torch.tensor([[3, 7, 11, 13]])
    assert isinstance(final_template.memory, TokenNgramMemory)
    addresses = final_template.memory.addresses(input_ids)
    start_address = int(addresses[0, 0])
    assert start_address not in addresses[0, 1:].tolist()

    models = [copy.deepcopy(final_template) for _ in range(4)]
    models[2].config = config.model_copy(update={"memory_injection": "embedding"})
    models[3].config = config.model_copy(update={"memory_injection": "embedding"})
    with torch.no_grad():
        for model in models:
            memory = model.memory
            assert isinstance(memory, TokenNgramMemory)
            memory.table.weight.zero_()
            memory.output.weight.zero_()
            memory.output.weight[:8, :8] = torch.eye(8)
            memory.gate.weight.zero_()
        for model in (models[1], models[3]):
            memory = model.memory
            assert isinstance(memory, TokenNgramMemory)
            memory.table.weight[start_address].fill_(1.0)

    final_before, final_after, early_before, early_after = (
        model(input_ids)[:, -1] for model in models
    )
    torch.testing.assert_close(final_before, final_after, rtol=0, atol=0)
    assert not torch.allclose(early_before, early_after, rtol=0, atol=1e-10)


@pytest.mark.parametrize("kernel", [0, 4])
def test_context_key_gate_reads_memory_and_backpropagates(kernel: int) -> None:
    from sparselab.model.memory import ContextKeyGate

    gate = ContextKeyGate(4, 1e-6)
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    assert torch.sigmoid(gate(hidden, hidden)).item() > 0.85
    assert torch.sigmoid(gate(hidden, -hidden)).item() < 0.15
    torch.manual_seed(7)
    memory = TokenNgramMemory(
        8, 31, 3, 4, (2, 3), 2, variant="context_key", conv_kernel_size=kernel
    )
    ids = torch.tensor([[3, 7, 11, 13, 17, 19]])
    states = torch.randn(1, 6, 8, requires_grad=True)
    output = memory(states, ids)
    future_ids = ids.clone()
    future_ids[:, 4:] += 1
    future_states = states.detach().clone()
    future_states[:, 4:] += 5
    torch.testing.assert_close(output[:, :4], memory(future_states, future_ids)[:, :4])
    output.square().sum().backward()
    for name, parameter in memory.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert not memory.table.weight.grad.is_sparse
    assert states.grad is not None and states.grad.abs().sum() > 0


def test_context_gate_changes_with_retrieved_content_at_fixed_hidden() -> None:
    memory = TokenNgramMemory(4, 31, 2, 4, variant="context_key")
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    ids = torch.tensor([[7]])
    with torch.no_grad():
        memory.key.weight.copy_(torch.eye(4))
        memory.table.weight[:] = hidden[0, 0]
        memory(hidden, ids)
        aligned = memory.last_diagnostics.gate_mean.item()
        memory.table.weight.neg_()
        memory(hidden, ids)
        opposed = memory.last_diagnostics.gate_mean.item()
    assert aligned > 0.85 and opposed < 0.15


def test_convolved_memory_rejects_masked_allocation() -> None:
    config = ModelConfig(
        vocab_size=260,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        ffn_dim=16,
        max_seq_len=8,
        memory="ngram",
        memory_table_size=31,
        memory_ngram_size=3,
        memory_dim=4,
        memory_variant="context_key",
        memory_conv_kernel_size=4,
    )
    model = DenseLM(config, AttentionConfig())
    with pytest.raises(ValueError, match="memory_mask"):
        model(
            torch.ones(1, 4, dtype=torch.long),
            memory_mask=torch.ones(1, 4, dtype=torch.bool),
        )
