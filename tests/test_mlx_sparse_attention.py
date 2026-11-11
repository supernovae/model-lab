"""MLX block-sparse attention agrees with the canonical PyTorch reference."""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
from mlx import nn
from mlx.utils import tree_flatten

from klab.model.attention.mlx_sparse import MLXBlockSparseAttention
from klab.model.attention.sparse import BlockSparseAttention

pytestmark = pytest.mark.mlx


def _paired_attention(
    *, block_size: int, selected_blocks: int, max_seq_len: int = 8, hidden_dim: int = 8
) -> tuple[BlockSparseAttention, MLXBlockSparseAttention]:
    torch.manual_seed(71)
    reference = BlockSparseAttention(
        hidden_dim, 2, max_seq_len, 10_000.0, block_size, selected_blocks
    )
    native = MLXBlockSparseAttention(
        hidden_dim, 2, max_seq_len, 10_000.0, block_size, selected_blocks
    )
    native.load_weights(
        [
            (name, mx.array(parameter.detach().cpu().numpy()))
            for name, parameter in reference.named_parameters()
        ],
        strict=True,
    )
    return reference, native


def _native_loss_and_gradients(
    attention: MLXBlockSparseAttention, values: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    inputs = mx.array(values)

    def loss_fn(x: mx.array) -> mx.array:
        return mx.sum(attention(x))

    loss, gradients = nn.value_and_grad(attention, loss_fn)(inputs)
    input_gradient = mx.grad(lambda x: mx.sum(attention(x)))(inputs)
    mx.eval(loss, gradients, input_gradient)
    return (
        np.array(attention(inputs)),
        {name: np.array(gradient) for name, gradient in tree_flatten(gradients)},
        np.array(input_gradient),
    )


def _reference_loss_and_gradients(
    attention: BlockSparseAttention, values: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    inputs = torch.tensor(values, requires_grad=True)
    output = attention(inputs)
    output.sum().backward()
    return (
        output.detach().numpy(),
        {
            name: parameter.grad.detach().numpy()
            for name, parameter in attention.named_parameters()
        },
        inputs.grad.detach().numpy(),
    )


def _assert_reference_agreement(
    *, block_size: int, selected_blocks: int, values: np.ndarray
) -> None:
    reference, native = _paired_attention(
        block_size=block_size,
        selected_blocks=selected_blocks,
        max_seq_len=values.shape[1],
        hidden_dim=values.shape[2],
    )
    expected, expected_grads, expected_input_grad = _reference_loss_and_gradients(
        reference, values
    )
    actual, actual_grads, actual_input_grad = _native_loss_and_gradients(native, values)

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        actual_input_grad, expected_input_grad, rtol=2e-4, atol=2e-5
    )
    assert actual_grads.keys() == expected_grads.keys()
    for name, expected_gradient in expected_grads.items():
        np.testing.assert_allclose(
            actual_grads[name], expected_gradient, rtol=2e-4, atol=2e-5, err_msg=name
        )


def test_mlx_sparse_preserves_prefix_under_future_input_changes() -> None:
    reference, native = _paired_attention(block_size=2, selected_blocks=1)
    torch.manual_seed(72)
    first = torch.randn(2, 6, 8).numpy()
    second = first.copy()
    second[:, 4:] = torch.randn(2, 2, 8).numpy()

    expected_first = reference(torch.tensor(first)).detach().numpy()
    expected_second = reference(torch.tensor(second)).detach().numpy()
    mx.eval(native(mx.array(first)), native(mx.array(second)))
    actual_first = np.array(native(mx.array(first)))
    actual_second = np.array(native(mx.array(second)))

    np.testing.assert_allclose(actual_first[:, :4], actual_second[:, :4], atol=1e-5)
    np.testing.assert_allclose(actual_first, expected_first, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(actual_second, expected_second, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    ("shape", "block_size", "selected_blocks"),
    [((2, 5, 8), 3, 2), ((1, 37, 8), 8, 1), ((2, 17, 80), 6, 2)],
)
def test_mlx_sparse_matches_projection_and_input_gradients_for_partial_blocks(
    shape: tuple[int, int, int], block_size: int, selected_blocks: int
) -> None:
    values = np.random.default_rng(73).normal(size=shape).astype(np.float32)
    _assert_reference_agreement(
        block_size=block_size, selected_blocks=selected_blocks, values=values
    )


def test_mlx_sparse_matches_batch_head_union_selection_gradients() -> None:
    # One selected block per query can still yield up to B*H selected blocks in
    # the union.  This checks the shared union gather, including repeated K/V
    # contributions in backward, against the reference semantics.
    values = np.random.default_rng(74).normal(size=(3, 7, 8)).astype(np.float32)
    _assert_reference_agreement(block_size=2, selected_blocks=1, values=values)
