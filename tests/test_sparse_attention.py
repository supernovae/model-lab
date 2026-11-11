from __future__ import annotations

import pytest
import torch

from klab.model.attention.dense import DenseAttention
from klab.model.attention.sparse import BlockSparseAttention, select_sparse_backend


def test_sparse_backend_dispatches_cpu_and_torch_devices() -> None:
    assert select_sparse_backend(torch.device("cpu")) == "cpu"
    expected = "hip" if torch.version.hip else "torch"
    assert select_sparse_backend(torch.device("cuda")) == expected


@pytest.mark.rocm
@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="requires an actual ROCm device",
)
@pytest.mark.parametrize(
    ("batch", "length", "hidden_dim", "block_size", "selected_blocks"),
    [(2, 5, 8, 3, 2), (1, 9, 80, 4, 1)],
)
def test_native_hip_sparse_forward_and_backward_match_reference(
    batch: int,
    length: int,
    hidden_dim: int,
    block_size: int,
    selected_blocks: int,
) -> None:
    torch.manual_seed(811)
    reference = BlockSparseAttention(
        hidden_dim, 2, length, 10_000.0, block_size, selected_blocks
    )
    native = BlockSparseAttention(
        hidden_dim, 2, length, 10_000.0, block_size, selected_blocks
    ).cuda()
    native.load_state_dict(reference.state_dict())
    values = torch.randn(batch, length, hidden_dim)
    reference_input = values.clone().requires_grad_()
    native_input = values.cuda().requires_grad_()

    expected = reference(reference_input, diagnostics="scalar")
    actual = native(native_input, diagnostics="scalar")
    assert native.last_backend == "hip"
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(
        native_input.grad.cpu(), reference_input.grad, rtol=2e-4, atol=2e-5
    )
    for (_, actual_parameter), (_, expected_parameter) in zip(
        native.named_parameters(), reference.named_parameters()
    ):
        torch.testing.assert_close(
            actual_parameter.grad.cpu(),
            expected_parameter.grad,
            rtol=3e-4,
            atol=3e-5,
        )


def test_block_sparse_attention_is_causal_and_reports_selection() -> None:
    attention = BlockSparseAttention(8, 2, 8, 10_000.0, block_size=2, selected_blocks=1)
    first = torch.randn(1, 6, 8)
    second = first.clone()
    second[:, 4:] = torch.randn_like(second[:, 4:])
    assert torch.allclose(attention(first)[:, :4], attention(second)[:, :4], atol=1e-6)
    diagnostics = attention.last_diagnostics
    assert attention.last_backend == "cpu"
    assert diagnostics is not None
    assert 0 < int(diagnostics.selected_tokens) < int(diagnostics.available_tokens)
    assert 0 < float(diagnostics.selection_ratio) < 1
    assert 0 < float(diagnostics.dense_teacher_mass) <= 1
    assert 0 < float(diagnostics.dense_teacher_topk_recall) <= 1


def test_sparse_matches_dense_when_every_causal_block_is_selected() -> None:
    dense = DenseAttention(8, 2, 8, 10_000.0)
    sparse = BlockSparseAttention(8, 2, 8, 10_000.0, block_size=2, selected_blocks=8)
    sparse.load_state_dict(dense.state_dict(), strict=True)
    values = torch.randn(2, 6, 8)
    assert torch.allclose(dense(values), sparse(values), atol=1e-6, rtol=1e-5)
    diagnostics = sparse.last_diagnostics
    assert diagnostics is not None
    assert torch.allclose(diagnostics.dense_teacher_mass, torch.ones(()))
    assert torch.allclose(diagnostics.dense_teacher_topk_recall, torch.ones(()))
