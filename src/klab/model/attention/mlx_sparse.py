"""Native MLX causal selection with bounded-memory Metal attention kernels."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from klab.model.attention.mlx_sparse_kernels import selected_attention

__all__ = ["MLXBlockSparseAttention"]


class MLXBlockSparseAttention(nn.Module):
    """Preserve the reference's causal block selection and batch/head union.

    Block-index scores occupy ``[B,H,T,ceil(T/block_size)]``; with block size
    one the selector itself approaches dense work. The union may include up
    to ``min(ceil(T/block_size), B*H*selected_blocks)`` blocks per position.
    FP32 Metal kernels attend directly to those keys without gathered K/V or
    token-square softmax activations. Key-owned backward scans the compact
    membership mask and uses no atomic accumulation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int,
        rope_base: float,
        block_size: int,
        selected_blocks: int,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if hidden_dim // num_heads % 2:
            raise ValueError("RoPE requires an even per-head dimension")
        if max_seq_len <= 0 or block_size <= 0 or selected_blocks <= 0:
            raise ValueError(
                "max_seq_len, block_size, and selected_blocks must be positive"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self.block_size = block_size
        self.selected_blocks = selected_blocks

        # These names intentionally match BlockSparseAttention's canonical
        # projection names used by cross-engine canonical weight mapping.
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 3:
            raise ValueError(
                "attention input must have shape [batch, sequence, hidden]"
            )
        batch, length, hidden = x.shape
        if batch == 0 or length == 0:
            raise ValueError("attention input batch and sequence must be nonempty")
        if hidden != self.num_heads * self.head_dim:
            raise ValueError(
                "attention input hidden dimension does not match projections"
            )
        if length > self.max_seq_len:
            raise ValueError("sequence length exceeds configured attention context")

        def split(projection: nn.Linear) -> mx.array:
            return mx.transpose(
                projection(x).reshape(batch, length, self.num_heads, self.head_dim),
                (0, 2, 1, 3),
            )

        query = mx.fast.rope(
            split(self.q_proj),
            self.head_dim,
            traditional=True,
            base=self.rope_base,
            scale=1.0,
            offset=0,
        )
        key = mx.fast.rope(
            split(self.k_proj),
            self.head_dim,
            traditional=True,
            base=self.rope_base,
            scale=1.0,
            offset=0,
        )
        value = split(self.v_proj)

        # Full block means plus one partial mean per position avoid a
        # [B,H,T,blocks,D] summary tensor. Selection is nondifferentiable.
        block_count = (length + self.block_size - 1) // self.block_size
        positions = mx.arange(length)
        block_ids = mx.arange(block_count)
        starts = block_ids * self.block_size
        ends = mx.minimum(starts + self.block_size, length)
        selection_key, selection_query = mx.stop_gradient(key), mx.stop_gradient(query)
        key_prefix = mx.concatenate(
            (mx.zeros_like(selection_key[:, :, :1]), mx.cumsum(selection_key, axis=2)),
            axis=2,
        )
        summaries = (
            mx.take(key_prefix, ends, axis=2) - mx.take(key_prefix, starts, axis=2)
        ) / (ends - starts)[None, None, :, None]
        index_scores = mx.matmul(selection_query, mx.swapaxes(summaries, -1, -2))
        current_starts = (positions // self.block_size) * self.block_size
        partial_means = (
            mx.take(key_prefix, positions + 1, axis=2)
            - mx.take(key_prefix, current_starts, axis=2)
        ) / (positions + 1 - current_starts)[None, None, :, None]
        partial_scores = mx.sum(selection_query * partial_means, axis=-1)
        current_block = block_ids[None, :] == (positions // self.block_size)[:, None]
        index_scores = mx.where(
            current_block[None, None], partial_scores[..., None], index_scores
        )
        block_is_causal = starts[None, :] <= positions[:, None]
        index_scores = mx.where(
            mx.expand_dims(mx.expand_dims(block_is_causal, 0), 0), index_scores, -1e30
        )
        per_query_count = min(self.selected_blocks, block_count)
        chosen = mx.argsort(index_scores, axis=-1)[..., -per_query_count:]

        # Union block choices over batch and heads.  Sorting selected IDs into
        # ascending block order preserves the reference gather/token order.
        all_blocks = mx.broadcast_to(
            mx.expand_dims(block_ids, 0), (length, block_count)
        )
        chosen_by_position = mx.transpose(chosen, (2, 0, 1, 3)).reshape(length, -1)
        union = (
            mx.zeros((length, block_count), dtype=mx.int32)
            .at[positions[:, None], chosen_by_position]
            .add(1)
            > 0
        )
        union_capacity = min(block_count, batch * self.num_heads * per_query_count)
        union_sort_keys = mx.where(union, all_blocks, block_count + all_blocks)
        selected_block_ids = mx.argsort(union_sort_keys, axis=-1)[:, :union_capacity]
        output, _ = selected_attention(self.block_size)(
            query,
            key,
            value,
            mx.stop_gradient(selected_block_ids),
            mx.stop_gradient(union),
        )
        return self.out_proj(
            mx.transpose(output, (0, 2, 1, 3)).reshape(batch, length, hidden)
        )
