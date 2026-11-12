"""Bias-free dense causal self-attention over [batch, sequence, hidden]."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from klab.model.cache import AttentionKVCache
from klab.model.rope import RoPE


class DenseAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int,
        rope_base: float,
        window_size: int | None = None,
        num_kv_heads: int | None = None,
        *,
        implementation: Literal["reference", "sdpa"] = "reference",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if (
            self.num_kv_heads <= 0
            or self.num_kv_heads > num_heads
            or num_heads % self.num_kv_heads
        ):
            raise ValueError(
                "num_kv_heads must be positive, no greater than num_heads, and divide num_heads"
            )
        if implementation == "sdpa" and self.num_kv_heads != self.num_heads:
            raise ValueError("SDPA requires equal Q/KV head counts")
        if window_size is not None and window_size <= 0:
            raise ValueError("window_size must be positive")
        self.implementation = implementation
        self.max_seq_len = max_seq_len
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rope = RoPE(self.head_dim, rope_base)

    def _heads(self, projection: nn.Linear, x: Tensor, heads: int) -> Tensor:
        batch, length, _ = x.shape
        return projection(x).view(batch, length, heads, self.head_dim).transpose(1, 2)

    def _attention(self, query: Tensor, key: Tensor) -> Tensor:
        if self.num_kv_heads == self.num_heads:
            return query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        groups = self.num_heads // self.num_kv_heads
        grouped_query = query.unflatten(1, (self.num_kv_heads, groups))
        return torch.einsum("bkgtd,bksd->bkgts", grouped_query, key) / math.sqrt(
            self.head_dim
        )

    def _attend_values(self, scores: Tensor, value: Tensor) -> Tensor:
        if self.num_kv_heads == self.num_heads:
            return scores @ value
        output = torch.einsum("bkgts,bksd->bkgtd", scores, value)
        return output.flatten(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, hidden = x.shape
        if length > self.max_seq_len:
            raise ValueError("sequence length exceeds configured attention context")
        query, key, value = (
            self.rope(self._heads(self.q_proj, x, self.num_heads)),
            self.rope(self._heads(self.k_proj, x, self.num_kv_heads)),
            self._heads(self.v_proj, x, self.num_kv_heads),
        )
        if self.window_size is not None and length > self.window_size:
            output = self._windowed(query, key, value)
        elif self.implementation == "sdpa":
            output = functional.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=True
            )
        else:
            probabilities = self._probabilities(query, key, length).to(value.dtype)
            output = self._attend_values(probabilities, value)
        return self.out_proj(
            output.transpose(1, 2).contiguous().view(batch, length, hidden)
        )

    def blocked_mask(self, length: int, device: torch.device | None = None) -> Tensor:
        """``[query, key]`` mask, true where attention is not allowed.

        Built on demand: no ``[max_seq_len, max_seq_len]`` buffer is kept, so the
        configured context does not cost memory until a sequence that long runs.
        """
        positions = torch.arange(length, device=device)
        distance = positions[:, None] - positions[None, :]
        blocked = distance < 0
        if self.window_size is not None:
            blocked |= distance >= self.window_size
        return blocked

    def _masked_attention(
        self, query: Tensor, key: Tensor, value: Tensor, allowed: Tensor
    ) -> Tensor:
        """Attend ``query`` to ``key``/``value`` where ``allowed[query, key]``."""
        if self.implementation == "sdpa":
            return functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=allowed[None, None],
                dropout_p=0.0,
                is_causal=False,
            )
        scores = self._attention(query, key)
        scores.masked_fill_(~allowed, float("-inf"))
        probabilities = torch.softmax(scores.float(), dim=-1).to(value.dtype)
        return self._attend_values(probabilities, value)

    def _windowed(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Sliding-window attention in window-sized query chunks.

        Each chunk scores only the keys its queries can see, at most
        ``2 * window_size - 1`` of them, so work and memory grow with
        ``length * window_size`` instead of ``length ** 2``. The weights are the
        ones a full masked score matrix would give.
        """
        window = self.window_size
        assert window is not None
        length = query.shape[2]
        outputs = []
        for start in range(0, length, window):
            stop = min(start + window, length)
            low = max(0, start - window + 1)
            query_positions = torch.arange(start, stop, device=query.device)
            key_positions = torch.arange(low, stop, device=query.device)
            distance = query_positions[:, None] - key_positions[None, :]
            outputs.append(
                self._masked_attention(
                    query[:, :, start:stop],
                    key[:, :, low:stop],
                    value[:, :, low:stop],
                    (distance >= 0) & (distance < window),
                )
            )
        return torch.cat(outputs, dim=2)

    def _probabilities(self, query: Tensor, key: Tensor, length: int) -> Tensor:
        """Masked softmax attention weights (fp32) over the full prefix."""
        scores = self._attention(query, key)
        scores.masked_fill_(self.blocked_mask(length, query.device), float("-inf"))
        return torch.softmax(scores.float(), dim=-1)

    @torch.no_grad()
    def attention_probabilities(self, x: Tensor) -> Tensor:
        """Attention weights ``[batch, heads, query, key]`` for input X.

        The same projections, RoPE, causal/window mask and softmax as the
        reference forward (for SDPA too: SDPA computes the same weights
        internally without exposing them). Used by the model explorer.
        """
        length = x.shape[1]
        if length > self.max_seq_len:
            raise ValueError("sequence length exceeds configured attention context")
        query = self.rope(self._heads(self.q_proj, x, self.num_heads))
        key = self.rope(self._heads(self.k_proj, x, self.num_kv_heads))
        weights = self._probabilities(query, key, length)
        return weights.flatten(1, 2) if weights.ndim == 5 else weights

    @property
    def cache_backend(self) -> str:
        return "dense" if self.window_size is None else "sliding"

    @property
    def cache_layout(self) -> tuple[int, int, int]:
        """K/V heads, key width and value width of one cached layer."""
        return self.num_kv_heads, self.head_dim, self.head_dim

    def create_cache(
        self, batch: int, capacity: int, *, device: torch.device, dtype: torch.dtype
    ) -> AttentionKVCache:
        """Allocate one bounded inference request's projected K/V storage."""
        if capacity <= 0:
            raise ValueError("KV cache capacity must be positive")
        shape = (batch, self.num_kv_heads, capacity, self.head_dim)
        return AttentionKVCache(
            key=torch.empty(shape, device=device, dtype=dtype),
            value=torch.empty(shape, device=device, dtype=dtype),
            length=0,
            position=0,
        )

    def forward_cached(
        self, x: Tensor, cache: AttentionKVCache
    ) -> tuple[Tensor, AttentionKVCache]:
        """Evaluate appended tokens using bounded request-local projected K/V."""
        batch, length, hidden = x.shape
        offset = cache.position
        if offset + length > self.max_seq_len:
            raise ValueError("sequence length exceeds configured attention context")
        query = self.rope(
            self._heads(self.q_proj, x, self.num_heads), position_offset=offset
        )
        key = self.rope(
            self._heads(self.k_proj, x, self.num_kv_heads), position_offset=offset
        )
        value = self._heads(self.v_proj, x, self.num_kv_heads)
        cache.append(key, value, position=offset + length)
        # A window never reaches further back than its size, so only that tail
        # of the stored keys is read: decode cost follows the window, not the
        # whole cached prefix.
        visible = (
            cache.length
            if self.window_size is None
            else min(cache.length, self.window_size - 1 + length)
        )
        keys = cache.key[:, :, cache.length - visible : cache.length]
        values = cache.value[:, :, cache.length - visible : cache.length]
        query_positions = torch.arange(offset, offset + length, device=x.device)
        key_positions = torch.arange(
            offset + length - visible, offset + length, device=x.device
        )
        allowed = key_positions[None, :] <= query_positions[:, None]
        if self.window_size is not None:
            allowed &= (
                key_positions[None, :] > query_positions[:, None] - self.window_size
            )
        output = self._masked_attention(query, keys, values, allowed)
        return (
            self.out_proj(
                output.transpose(1, 2).contiguous().view(batch, length, hidden)
            ),
            cache,
        )
