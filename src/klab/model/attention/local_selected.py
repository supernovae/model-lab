"""Causal sparse attention: a local window plus selected earlier blocks.

The sequence is cut into blocks of ``block_size`` tokens. Every query attends to

* its own block, causally;
* the ``window_size // block_size`` blocks before it, always; and
* ``selected_blocks`` further-back blocks, chosen per head.

Selection scores the first query of a block against the mean key of each earlier
block. That query sees no later token, so every position stays causal, and all
queries of a block share one gather, which keeps the whole computation batched:
there is no per-position loop and no ``[T, T]`` tensor. Attention work grows with
``T * (window_size + (selected_blocks + 1) * block_size)``; block scoring grows
with ``(T / block_size) ** 2`` and is done in bounded chunks.

Keys and values are either ordinary per-head projections or, with ``latent_dim``,
the shared low-rank latent of multi-head latent attention, so sparse selection and
latent K/V compression can be combined.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from klab.model.rope import RoPE

# Upper bound on masked score cells evaluated per chunk of query blocks.
_CHUNK_CELLS = 1 << 26


class LocalSelectedAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int,
        rope_base: float,
        block_size: int,
        window_size: int,
        selected_blocks: int,
        latent_dim: int | None = None,
    ) -> None:
        super().__init__()
        if block_size <= 0 or selected_blocks < 0:
            raise ValueError("block_size must be positive and selected_blocks >= 0")
        if window_size <= 0 or window_size % block_size:
            raise ValueError("window_size must be a positive multiple of block_size")
        if latent_dim is not None and latent_dim % num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.window_size = window_size
        self.local_blocks = window_size // block_size
        self.selected_blocks = selected_blocks
        self.latent_dim = latent_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if latent_dim is None:
            self.value_dim = self.head_dim
            self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.value_dim = latent_dim // num_heads
            self.kv_down = nn.Linear(hidden_dim, latent_dim, bias=False)
            self.k_up = nn.Linear(latent_dim, hidden_dim, bias=False)
            self.out_proj = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.rope = RoPE(self.head_dim, rope_base)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = x.shape

        def heads(tensor: Tensor, dim: int) -> Tensor:
            return tensor.view(batch, length, self.num_heads, dim).transpose(1, 2)

        query = heads(self.q_proj(x), self.head_dim)
        if self.latent_dim is None:
            key = heads(self.k_proj(x), self.head_dim)
            value = heads(self.v_proj(x), self.value_dim)
        else:
            latent = self.kv_down(x)
            key = heads(self.k_up(latent), self.head_dim)
            value = heads(latent, self.value_dim)
        return self.rope(query), self.rope(key), value

    def visible_blocks(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        """Key-block indices each query block attends to, and which are real.

        ``query`` and ``key`` are ``[groups, blocks, block_size, head_dim]``. The
        result is ``[groups, blocks, local_blocks + 1 + selected]``: the previous
        local blocks, the query's own block, then the selected earlier blocks.
        """
        groups, blocks = query.shape[:2]
        local = self.local_blocks
        own = torch.arange(blocks, device=query.device)
        nearby = own[:, None] - local + torch.arange(local + 1, device=query.device)
        indices = nearby.clamp_min(0).expand(groups, -1, -1)
        real = (nearby >= 0).expand(groups, -1, -1)
        chosen = min(self.selected_blocks, max(blocks - local - 1, 0))
        if not chosen:
            return indices, real
        with torch.no_grad():
            selector = query[:, :, 0].float()
            summary = key.float().mean(dim=2)
            step = max(1, _CHUNK_CELLS // max(1, groups * blocks))
            picked, found = [], []
            for start in range(0, blocks, step):
                rows = own[start : start + step]
                scores = selector[:, start : start + step] @ summary.transpose(1, 2)
                # Only blocks before the local window compete.
                scores.masked_fill_(own[None, :] >= rows[:, None] - local, -torch.inf)
                best = scores.topk(chosen, dim=-1)
                found.append(best.values > -torch.inf)
                picked.append(best.indices)
            selected, selected_real = torch.cat(picked, dim=1), torch.cat(found, dim=1)
        return (
            torch.cat((indices, selected.masked_fill(~selected_real, 0)), dim=-1),
            torch.cat((real, selected_real), dim=-1),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        if length > self.max_seq_len:
            raise ValueError("sequence length exceeds configured attention context")
        size = self.block_size
        query, key, value = self._project(x)
        padding = -length % size
        blocks = (length + padding) // size
        groups = batch * self.num_heads

        def blocked(tensor: Tensor) -> Tensor:
            # Padding sits after every real position, so causality hides it.
            tensor = functional.pad(tensor, (0, 0, 0, padding))
            return tensor.reshape(groups, blocks, size, tensor.shape[-1])

        query, key, value = blocked(query), blocked(key), blocked(value)
        indices, real = self.visible_blocks(query, key)
        slots = indices.shape[-1]
        # Earlier blocks are visible whole; the query's own block only causally.
        causal = torch.ones(size, slots, size, dtype=torch.bool, device=x.device)
        causal[:, self.local_blocks] = torch.ones_like(causal[:, 0]).tril()
        causal = causal.reshape(size, slots * size)
        group = torch.arange(groups, device=x.device)[:, None, None]
        step = max(1, _CHUNK_CELLS // (groups * size * slots * size))
        outputs = []
        for start in range(0, blocks, step):
            chosen = indices[:, start : start + step]
            allowed = (
                real[:, start : start + step].repeat_interleave(size, dim=-1)[
                    :, :, None
                ]
                & causal
            )
            outputs.append(
                functional.scaled_dot_product_attention(
                    query[:, start : start + step],
                    key[group, chosen].flatten(2, 3),
                    value[group, chosen].flatten(2, 3),
                    attn_mask=allowed,
                    dropout_p=0.0,
                )
            )
        output = torch.cat(outputs, dim=1).reshape(
            batch, self.num_heads, blocks * size, self.value_dim
        )[:, :, :length]
        return self.out_proj(output.transpose(1, 2).reshape(batch, length, -1))
