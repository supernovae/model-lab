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


def test_vectorized_stream_diagnostics_match_per_stream_reference() -> None:
    from sparselab.model.memory import _diagnostics

    torch.manual_seed(7)
    memory = TokenNgramMemory(
        hidden_dim=8,
        table_size=31,
        ngram_size=3,
        value_dim=5,
        ngram_orders=(2, 3, 4),
        hash_heads=2,
    )
    hidden = torch.randn(3, 12, 8)
    input_ids = torch.randint(0, 64, (3, 12))
    output = memory(hidden, input_ids)
    actual = memory.last_diagnostics
    assert actual is not None

    addresses = [
        memory.addresses(input_ids, order, head)
        for order in memory.ngram_orders
        for head in range(memory.hash_heads)
    ]
    for left, right in zip(addresses, memory._all_addresses(input_ids), strict=True):
        assert torch.equal(left, right)
    tables = (memory.table, *memory.extra_tables)
    stream_values = [
        memory.output(table(address))
        for table, address in zip(tables, addresses, strict=True)
    ]
    values = torch.stack(stream_values).mean(dim=0)
    gate = torch.sigmoid(memory.gate(hidden))
    expected_streams = tuple(
        _diagnostics(address, gate, value, hidden, memory.table_size)
        for address, value in zip(addresses, stream_values, strict=True)
    )
    expected = _diagnostics(
        torch.cat([address.flatten() for address in addresses]),
        gate,
        values,
        hidden,
        memory.table_size,
    )
    torch.testing.assert_close(output, hidden + gate * values, rtol=0, atol=0)
    for field in (
        "lookup_count",
        "unique_addresses",
        "collision_count",
        "bucket_reuse_rate",
        "table_utilization",
        "maximum_address_fraction",
        "gate_mean",
        "value_norm",
        "hidden_norm",
    ):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field))
        for actual_stream, expected_stream in zip(
            actual.streams, expected_streams, strict=True
        ):
            torch.testing.assert_close(
                getattr(actual_stream, field), getattr(expected_stream, field)
            )


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
