"""Causal sparse attention: a local window plus selected earlier blocks.

The sequence is cut into blocks of ``block_size`` tokens. Every query attends to

* its own block, causally;
* the ``window_size // block_size`` blocks before it, always; and
* ``selected_blocks`` further-back blocks, chosen per head.

Selection scores the first query of a block against one summary key per earlier
block. That query sees no later token, so every position stays causal, and all
queries of a block share one gather, which keeps the whole computation batched:
there is no per-position loop and no ``[T, T]`` tensor. Attention work grows with
``T * (window_size + (selected_blocks + 1) * block_size)``; block scoring grows
with ``(T / block_size) ** 2`` and is done in bounded chunks.

``selector`` decides what a block summary is:

``mean_key``
    The mean key of the block. Nothing is learned for selection.

``learned``
    A learned weighting of the positions in a block, trained through a coarse
    attention branch: every query also attends to the summaries of all complete
    earlier blocks, and a per-head gate adds that result to the output. The
    summaries therefore receive gradient from the loss, and block selection
    reuses them. The weighting starts uniform and the gate at zero, so a fresh
    layer computes exactly what ``mean_key`` computes.

Keys and values are either ordinary per-head projections or, with ``latent_dim``,
the shared low-rank latent of multi-head latent attention, so sparse selection and
latent K/V compression can be combined.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from klab.model.cache import AttentionKVCache
from klab.model.rope import RoPE, RopeScaling

# Upper bound on masked score cells evaluated per chunk of query blocks.
CHUNK_CELLS = 1 << 26


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
        *,
        rope_scaling: RopeScaling | None = None,
        selector: Literal["mean_key", "learned"] = "mean_key",
    ) -> None:
        super().__init__()
        if block_size <= 0 or selected_blocks < 0:
            raise ValueError("block_size must be positive and selected_blocks >= 0")
        if window_size <= 0 or window_size % block_size:
            raise ValueError("window_size must be a positive multiple of block_size")
        if latent_dim is not None and latent_dim % num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        if selector not in ("mean_key", "learned"):
            raise ValueError(f"unknown block selector: {selector}")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.window_size = window_size
        self.local_blocks = window_size // block_size
        self.selected_blocks = selected_blocks
        self.latent_dim = latent_dim
        self.selector = selector
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
        if selector == "learned":
            # Uniform weighting and a closed gate reproduce ``mean_key`` exactly.
            self.pool_logits = nn.Parameter(torch.zeros(num_heads, block_size))
            self.coarse_gate = nn.Parameter(torch.zeros(num_heads))
        self.rope = RoPE(self.head_dim, rope_base, rope_scaling)
        # Set by training (activation_checkpointing.strategy); not a weight.
        self.recompute_chunks = False

    def _project(
        self, x: Tensor, position_offset: int = 0
    ) -> tuple[Tensor, Tensor, Tensor]:
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
        return (
            self.rope(query, position_offset=position_offset),
            self.rope(key, position_offset=position_offset),
            value,
        )

    def _summarize(self, blocks: Tensor) -> Tensor:
        """One vector per block from ``[batch, heads, blocks, block_size, dim]``."""
        if self.selector == "mean_key":
            return blocks.mean(dim=3)
        weights = torch.softmax(self.pool_logits.float(), dim=-1).to(blocks.dtype)
        return (blocks * weights[None, :, None, :, None]).sum(dim=3)

    def _select(self, selector: Tensor, summary: Tensor, rows: Tensor) -> Tensor:
        """Top block indices for query blocks ``rows``; ``-1`` where none exists.

        ``selector`` is ``[batch, heads, len(rows), dim]`` and ``summary`` holds
        every block before the last row's local window.
        """
        candidates = summary.shape[2]
        chosen = min(self.selected_blocks, candidates)
        shape = (*selector.shape[:3], chosen)
        if not chosen:
            return torch.empty(shape, dtype=torch.long, device=selector.device)
        with torch.no_grad():
            scores = selector.float() @ summary.float().transpose(-2, -1)
            # Only blocks before a row's local window compete.
            block = torch.arange(candidates, device=selector.device)
            scores.masked_fill_(
                block[None, :] >= rows[:, None] - self.local_blocks, -torch.inf
            )
            best = scores.topk(chosen, dim=-1)
            return best.indices.masked_fill(best.values == -torch.inf, -1)

    def visible_blocks(self, query: Tensor, summary: Tensor) -> Tensor:
        """Key blocks each query block attends to, ``-1`` where a slot is empty.

        ``query`` is ``[batch, heads, blocks, block_size, dim]``. The result is
        ``[batch, heads, blocks, local_blocks + 1 + selected]``: the previous
        local blocks, the query's own block, then the selected earlier blocks.
        """
        batch, heads, blocks = query.shape[:3]
        local = self.local_blocks
        own = torch.arange(blocks, device=query.device)
        nearby = own[:, None] - local + torch.arange(local + 1, device=query.device)
        nearby = nearby.clamp_min(-1).expand(batch, heads, -1, -1)
        candidates = summary[:, :, : max(blocks - local - 1, 0)]
        step = max(1, CHUNK_CELLS // max(1, batch * heads * blocks))
        selected = [
            self._select(
                query[:, :, start : start + step, 0],
                candidates,
                own[start : start + step],
            )
            for start in range(0, blocks, step)
        ]
        return torch.cat((nearby, torch.cat(selected, dim=2)), dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        if length > self.max_seq_len:
            raise ValueError("sequence length exceeds configured attention context")
        size, heads = self.block_size, self.num_heads
        query, key, value = self._project(x)
        padding = -length % size
        blocks = (length + padding) // size

        def blocked(tensor: Tensor) -> Tensor:
            # Padding sits after every real position, so causality hides it.
            tensor = functional.pad(tensor, (0, 0, 0, padding))
            return tensor.reshape(batch, heads, blocks, size, tensor.shape[-1])

        query, key, value = blocked(query), blocked(key), blocked(value)
        summary = self._summarize(key)
        indices = self.visible_blocks(query, summary)
        real = indices >= 0
        indices = indices.clamp_min(0)
        slots = indices.shape[-1]
        # Earlier blocks are visible whole; the query's own block only causally.
        causal = torch.ones(size, slots, size, dtype=torch.bool, device=x.device)
        causal[:, self.local_blocks] = torch.ones_like(causal[:, 0]).tril()
        causal = causal.reshape(size, slots * size)
        # One row per (batch, head, block): a selected block is a row index, so
        # the gather is one index_select instead of broadcast advanced indexing.
        rows = (
            torch.arange(batch * heads, device=x.device).view(batch, heads, 1, 1)
            * blocks
            + indices
        )
        key_rows, value_rows = key.flatten(0, 2), value.flatten(0, 2)
        coarse = self.selector == "learned"
        summary_value = self._summarize(value) if coarse else None
        step = max(1, CHUNK_CELLS // (batch * heads * size * slots * size))
        # With ``recompute_chunks`` each chunk keeps only its inputs and
        # recomputes its gathered keys, mask and scores in backward, so a block
        # holds one chunk of attention work instead of all of them.
        recompute = (
            self.recompute_chunks and torch.is_grad_enabled() and query.requires_grad
        )
        outputs = []
        for start in range(0, blocks, step):
            stop = min(start + step, blocks)
            arguments = (
                query[:, :, start:stop],
                key_rows,
                value_rows,
                rows[:, :, start:stop],
                real[:, :, start:stop],
                causal,
                summary[:, :, :stop] if coarse else None,
                summary_value[:, :, :stop] if coarse else None,
                start,
            )
            outputs.append(
                checkpoint(self._attend, *arguments, use_reentrant=False)
                if recompute
                else self._attend(*arguments)
            )
        output = torch.cat(outputs, dim=2).reshape(
            batch, heads, blocks * size, self.value_dim
        )[:, :, :length]
        return self.out_proj(output.transpose(1, 2).reshape(batch, length, -1))

    def _attend(
        self,
        query: Tensor,
        key_rows: Tensor,
        value_rows: Tensor,
        rows: Tensor,
        real: Tensor,
        causal: Tensor,
        summary: Tensor | None,
        summary_value: Tensor | None,
        start: int,
    ) -> Tensor:
        """Attention of one chunk of query blocks ``[batch, heads, n, size, dim]``."""
        batch, heads, count, size = query.shape[:4]

        def gather(source: Tensor) -> Tensor:
            picked = source.index_select(0, rows.flatten())
            return picked.view(batch, heads, count, -1, source.shape[-1])

        allowed = real.repeat_interleave(size, dim=-1)[..., None, :] & causal
        output = functional.scaled_dot_product_attention(
            query,
            gather(key_rows),
            gather(value_rows),
            attn_mask=allowed,
            dropout_p=0.0,
        )
        if summary is None or summary_value is None:
            return output
        stop = start + count
        own = torch.arange(stop, device=query.device)
        return output + self._coarse(
            query.flatten(2, 3),
            summary,
            summary_value,
            own[None, :] < own[start:].repeat_interleave(size)[:, None],
        ).unflatten(2, (count, size))

    def _coarse(
        self, query: Tensor, summary_key: Tensor, summary_value: Tensor, allowed: Tensor
    ) -> Tensor:
        """Gated attention of ``query`` over block summaries.

        A zero key and value slot is always available, so a query with no
        complete earlier block attends to nothing instead of dividing by zero.
        """
        output = functional.scaled_dot_product_attention(
            query,
            functional.pad(summary_key, (0, 0, 1, 0)),
            functional.pad(summary_value, (0, 0, 1, 0)),
            attn_mask=functional.pad(allowed, (1, 0), value=True),
            dropout_p=0.0,
        )
        return output * self.coarse_gate.to(output.dtype)[None, :, None, None]

    @staticmethod
    def _visible(stored: Tensor, tokens: Tensor, near: int, stop: int) -> Tensor:
        """Selected-block rows followed by the contiguous local rows."""
        # An empty slot points at block zero and is masked; a request smaller
        # than one block has no such rows, so the index is kept in range.
        rows = tokens.clamp_max(stored.shape[2] - 1)
        gathered = stored.gather(
            2, rows[..., None].expand(-1, -1, -1, stored.shape[-1])
        )
        return torch.cat((gathered, stored[:, :, near:stop]), dim=2)

    cache_backend = "local_selected"

    @property
    def cache_layout(self) -> tuple[int, int, int]:
        """K/V heads, key width and value width of one cached layer."""
        return self.num_heads, self.head_dim, self.value_dim

    def create_cache(
        self, batch: int, capacity: int, *, device: torch.device, dtype: torch.dtype
    ) -> AttentionKVCache:
        """Allocate one request's K/V plus per-block summaries and selections."""
        if capacity <= 0:
            raise ValueError("KV cache capacity must be positive")
        heads, blocks = self.num_heads, -(-capacity // self.block_size)
        extra = {
            "summary_key": torch.zeros(
                (batch, heads, blocks, self.head_dim), device=device, dtype=dtype
            ),
            "selected": torch.full(
                (batch, heads, blocks, self.selected_blocks), -1, device=device
            ),
        }
        if self.selector == "learned":
            extra["summary_value"] = torch.zeros(
                (batch, heads, blocks, self.value_dim), device=device, dtype=dtype
            )
        # Zeroed, not empty: an unselected slot is masked but still read, and a
        # masked weight of zero times an uninitialized value is not zero.
        return AttentionKVCache(
            key=torch.zeros(
                (batch, heads, capacity, self.head_dim), device=device, dtype=dtype
            ),
            value=torch.zeros(
                (batch, heads, capacity, self.value_dim), device=device, dtype=dtype
            ),
            length=0,
            position=0,
            extra=extra,
        )

    def forward_cached(
        self, x: Tensor, cache: AttentionKVCache
    ) -> tuple[Tensor, AttentionKVCache]:
        """Evaluate appended tokens against stored K/V, summaries and selections.

        Each block's summary is stored when the next block begins and its
        selection when its first token arrives, so a token reads only its local
        window and selected blocks. Positions before a rewind keep their stored
        selections; later blocks are selected again when they are re-entered.
        """
        batch, length, _ = x.shape
        offset = cache.position
        if offset != cache.length:
            raise ValueError("local_selected cache must start at position zero")
        if offset + length > self.max_seq_len:
            raise ValueError("sequence length exceeds configured attention context")
        size, local = self.block_size, self.local_blocks
        query, key, value = self._project(x, offset)
        cache.append(key, value, position=offset + length)
        summary_key, selected = cache.extra["summary_key"], cache.extra["selected"]
        outputs = []
        start = offset
        while start < offset + length:
            block = start // size
            stop = min((block + 1) * size, offset + length)
            segment = query[:, :, start - offset : stop - offset]
            if start % size == 0:
                if block:
                    previous = slice((block - 1) * size, block * size)
                    summary_key[:, :, block - 1] = self._summarize(
                        cache.key[:, :, None, previous]
                    )[:, :, 0]
                    if self.selector == "learned":
                        cache.extra["summary_value"][:, :, block - 1] = self._summarize(
                            cache.value[:, :, None, previous]
                        )[:, :, 0]
                chosen = self._select(
                    segment[:, :, :1],
                    summary_key[:, :, : max(block - local, 0)],
                    torch.tensor([block], device=x.device),
                )[:, :, 0]
                selected[:, :, block] = -1
                selected[:, :, block, : chosen.shape[-1]] = chosen
            chosen = selected[:, :, block]
            near = max(block - local, 0) * size
            tokens = (
                chosen.clamp_min(0)[..., None] * size
                + torch.arange(size, device=x.device)
            ).flatten(2)
            positions = torch.arange(start, stop, device=x.device)
            allowed = torch.cat(
                (
                    (chosen >= 0)
                    .repeat_interleave(size, dim=-1)[:, :, None, :]
                    .expand(-1, -1, stop - start, -1),
                    (
                        torch.arange(near, stop, device=x.device)[None, :]
                        <= positions[:, None]
                    ).expand(batch, self.num_heads, -1, -1),
                ),
                dim=-1,
            )

            output = functional.scaled_dot_product_attention(
                segment,
                self._visible(cache.key, tokens, near, stop),
                self._visible(cache.value, tokens, near, stop),
                attn_mask=allowed,
                dropout_p=0.0,
            )
            if self.selector == "learned":
                output = output + self._coarse(
                    segment,
                    summary_key[:, :, :block],
                    cache.extra["summary_value"][:, :, :block],
                    torch.ones(stop - start, block, dtype=torch.bool, device=x.device),
                )
            outputs.append(output)
            start = stop
        output = torch.cat(outputs, dim=2)
        return (
            self.out_proj(output.transpose(1, 2).reshape(batch, length, -1)),
            cache,
        )
