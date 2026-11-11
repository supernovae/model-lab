"""Reference causal multi-head latent attention."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from klab.model.cache import AttentionKVCache
from klab.model.rope import RoPE


class LatentAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        latent_dim: int,
        max_seq_len: int,
        rope_base: float,
        *,
        implementation: Literal["reference", "sdpa"] = "reference",
    ) -> None:
        super().__init__()
        self.implementation = implementation
        if latent_dim % num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.value_dim = latent_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kv_down = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.k_up = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.rope = RoPE(self.head_dim, rope_base)
        self.max_seq_len = max_seq_len

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = x.shape
        query = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim)
        latent = self.kv_down(x)
        key = self.k_up(latent).view(batch, length, self.num_heads, self.head_dim)
        value = latent.view(batch, length, self.num_heads, self.value_dim)
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        if length > self.max_seq_len:
            raise ValueError("sequence length exceeds configured attention context")
        query, key, value = self._project(x)
        query, key = self.rope(query), self.rope(key)
        if self.implementation == "sdpa":
            # No [T, T] weights are materialized; the latent values may be
            # narrower than the keys.
            output = functional.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=True
            )
        else:
            scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
            # Built per call: no [max_seq_len, max_seq_len] buffer is kept.
            scores.masked_fill_(
                torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1),
                float("-inf"),
            )
            output = torch.softmax(scores.float(), dim=-1).to(value.dtype) @ value
        return self.out_proj(output.transpose(1, 2).reshape(batch, length, -1))

    def create_cache(
        self, batch: int, capacity: int, *, device: torch.device, dtype: torch.dtype
    ) -> AttentionKVCache:
        """Allocate one bounded inference request's expanded K/V storage."""
        if capacity <= 0:
            raise ValueError("KV cache capacity must be positive")
        return AttentionKVCache(
            key=torch.empty(
                (batch, self.num_heads, capacity, self.head_dim),
                device=device,
                dtype=dtype,
            ),
            value=torch.empty(
                (batch, self.num_heads, capacity, self.value_dim),
                device=device,
                dtype=dtype,
            ),
            length=0,
            position=0,
        )

    def forward_cached(
        self, x: Tensor, cache: AttentionKVCache
    ) -> tuple[Tensor, AttentionKVCache]:
        """Evaluate appended tokens using bounded latent projected K/V storage."""
        batch, length, _ = x.shape
        offset = cache.position
        prior_length = cache.length
        query, key, value = self._project(x)
        query = self.rope(query, position_offset=offset)
        key = self.rope(key, position_offset=offset)
        cache.append(key, value, position=offset + length)
        keys = cache.key[:, :, : cache.length]
        values = cache.value[:, :, : cache.length]
        key_positions = torch.arange(
            offset - prior_length, offset + length, device=x.device
        )
        query_positions = torch.arange(offset, offset + length, device=x.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        if self.implementation == "sdpa":
            output = functional.scaled_dot_product_attention(
                query, keys, values, attn_mask=allowed[None, None], dropout_p=0.0
            )
        else:
            scores = query @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)
            scores.masked_fill_(~allowed[None, None], float("-inf"))
            output = torch.softmax(scores.float(), dim=-1).to(values.dtype) @ values
        return (
            self.out_proj(output.transpose(1, 2).reshape(batch, length, -1)),
            cache,
        )
