"""Optional MLX implementation of the canonical dense Model Lab decoder.

This module deliberately imports MLX: callers that only inspect checkpoint metadata use
``training.mlx_checkpoints`` instead and therefore remain usable without MLX installed.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from klab.config.models import AttentionConfig, ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Keep the reduction explicitly fp32: autocast is not an MLX capability yet.
        scale = mx.rsqrt(
            mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + self.eps
        )
        return x * scale.astype(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dims: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dims, hidden, bias=False)
        self.up = nn.Linear(dims, hidden, bias=False)
        self.down = nn.Linear(hidden, dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    """Bias-free causal multi-head attention with adjacent-pair RoPE."""

    def __init__(self, dims: int, heads: int, rope_base: float) -> None:
        super().__init__()
        self.heads, self.head_dim, self.rope_base = heads, dims // heads, rope_base
        self.q_proj = nn.Linear(dims, dims, bias=False)
        self.k_proj = nn.Linear(dims, dims, bias=False)
        self.v_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, _ = x.shape

        def split(layer: nn.Linear) -> mx.array:
            return mx.transpose(
                layer(x).reshape(batch, length, self.heads, self.head_dim), (0, 2, 1, 3)
            )

        q = mx.fast.rope(
            split(self.q_proj),
            self.head_dim,
            traditional=True,
            base=self.rope_base,
            scale=1.0,
            offset=0,
        )
        k = mx.fast.rope(
            split(self.k_proj),
            self.head_dim,
            traditional=True,
            base=self.rope_base,
            scale=1.0,
            offset=0,
        )
        v = split(self.v_proj)
        scores = mx.matmul(q, mx.swapaxes(k, -1, -2)) / (self.head_dim**0.5)
        causal = mx.triu(mx.ones((length, length), dtype=mx.bool_), k=1)
        scores = mx.where(causal, mx.array(-1e9, dtype=scores.dtype), scores)
        output = mx.matmul(mx.softmax(scores, axis=-1), v)
        return self.out_proj(
            mx.transpose(output, (0, 2, 1, 3)).reshape(batch, length, -1)
        )


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, attention: AttentionConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        if attention.kind == "dense":
            self.attention = Attention(
                config.hidden_dim, config.num_heads, attention.rope_base
            )
        elif attention.kind == "block_sparse":
            from klab.model.attention.mlx_sparse import MLXBlockSparseAttention

            assert attention.block_size is not None
            assert attention.selected_blocks is not None
            self.attention = MLXBlockSparseAttention(
                config.hidden_dim,
                config.num_heads,
                config.max_seq_len,
                attention.rope_base,
                attention.block_size,
                attention.selected_blocks,
            )
        else:
            raise ValueError(
                f"MLXDenseLM does not implement {attention.kind} attention"
            )
        self.norm2 = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.ffn = SwiGLU(config.hidden_dim, config.ffn_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class MLXDenseLM(nn.Module):
    """Dense FFN decoder with dense or reference MLX block-sparse attention."""

    def __init__(
        self,
        config: ModelConfig,
        attention: AttentionConfig | None = None,
        *,
        recompute_blocks: bool = False,
    ) -> None:
        super().__init__()
        attention = attention or AttentionConfig()
        if config.num_kv_heads is not None and config.num_kv_heads != config.num_heads:
            raise ValueError("MLXDenseLM does not support grouped-query attention")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = [
            DecoderBlock(config, attention) for _ in range(config.num_layers)
        ]
        self._block_calls = [
            nn.utils.checkpoint(block) if recompute_blocks else block
            for block in self.blocks
        ]
        self.norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.output = (
            None
            if config.tie_embeddings
            else nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        )

    def __call__(self, input_ids: mx.array) -> mx.array:
        hidden = self.embedding(input_ids)
        for block in self._block_calls:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        return (
            self.embedding.as_linear(hidden)
            if self.output is None
            else self.output(hidden)
        )
