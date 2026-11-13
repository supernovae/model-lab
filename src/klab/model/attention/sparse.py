"""Reference causal block-selection attention without a dense QK matrix."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from klab.model.moe import diagnostic_writes_enabled
from klab.model.rope import RoPE, RopeScaling


@dataclass(frozen=True)
class SparseAttentionDiagnostics:
    available_tokens: Tensor
    selected_tokens: Tensor
    selection_ratio: Tensor
    selected_blocks: Tensor | None
    estimated_attention_flops: Tensor
    dense_teacher_mass: Tensor | None
    dense_teacher_topk_recall: Tensor | None


SparseBackend = Literal["cpu", "mps", "torch", "hip"]


def select_sparse_backend(device: torch.device) -> SparseBackend:
    """Report the implementation selected for sparse attention."""
    if device.type == "cpu":
        return "cpu"
    if device.type == "mps":
        return "mps"
    if device.type == "cuda" and torch.version.hip:
        return "hip"
    return "torch"


class BlockSparseAttention(nn.Module):
    """Select causal K/V blocks from compressed keys, then attend to raw K/V.

    This deliberately loops over query positions for clarity.  It never forms
    a ``[B, H, T, T]`` score matrix: index scores use one mean key per causal
    block and final scores use only the selected original keys.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int,
        rope_base: float,
        block_size: int,
        selected_blocks: int,
        *,
        rope_scaling: RopeScaling | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.block_size = block_size
        self.selected_blocks = selected_blocks
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rope = RoPE(self.head_dim, rope_base, rope_scaling)
        self.max_seq_len = max_seq_len
        self.last_diagnostics: SparseAttentionDiagnostics | None = None
        self.last_backend: SparseBackend | None = None

    cache_unsupported_reason = (
        "block_sparse selection is prefix-dependent; generation recomputes "
        "the active prefix for exact reference semantics"
    )

    def forward_with_aux(
        self,
        x: Tensor,
        *,
        valid_target_mask: Tensor | None = None,
        diagnostics: Literal["scalar", "full"] = "scalar",
    ) -> tuple[Tensor, None]:
        """Attend with the block's target mask; this attention adds no loss."""
        return (
            self(x, valid_target_mask=valid_target_mask, diagnostics=diagnostics),
            None,
        )

    def metric_tensors(self) -> dict[str, Tensor]:
        diagnostic = self.last_diagnostics
        if diagnostic is None:
            return {}
        return {
            metric: value
            for metric, value in (
                ("available_tokens", diagnostic.available_tokens),
                ("selected_tokens", diagnostic.selected_tokens),
                ("selection_ratio", diagnostic.selection_ratio),
                ("estimated_flops", diagnostic.estimated_attention_flops),
                ("dense_teacher_mass", diagnostic.dense_teacher_mass),
                ("dense_teacher_topk_recall", diagnostic.dense_teacher_topk_recall),
            )
            if value is not None
        }

    def forward(
        self,
        x: Tensor,
        *,
        valid_target_mask: Tensor | None = None,
        diagnostics: Literal["scalar", "full"] = "full",
    ) -> Tensor:
        if diagnostics not in ("scalar", "full"):
            raise ValueError(f"unknown diagnostics policy: {diagnostics}")
        if valid_target_mask is not None and valid_target_mask.shape != x.shape[:2]:
            raise ValueError("valid_target_mask must align with attention inputs")
        if valid_target_mask is not None:
            valid_target_mask = valid_target_mask.to(dtype=torch.bool)
        batch, length, hidden = x.shape
        if length > self.max_seq_len:
            raise ValueError("sequence length exceeds configured attention context")

        def heads(projection: nn.Linear) -> Tensor:
            return (
                projection(x)
                .view(batch, length, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        record_diagnostics = diagnostic_writes_enabled()
        if record_diagnostics:
            self.last_backend = select_sparse_backend(x.device)

        query, key, value = (
            self.rope(heads(self.q_proj)),
            self.rope(heads(self.k_proj)),
            heads(self.v_proj),
        )
        key_prefix = torch.cat(
            (torch.zeros_like(key[:, :, :1]), key.cumsum(dim=2)),
            dim=2,
        )
        hip_selected = x.device.type == "cuda" and bool(torch.version.hip)
        membership = (
            torch.zeros(
                (length, (length + self.block_size - 1) // self.block_size),
                dtype=torch.bool,
                device=x.device,
            )
            if hip_selected
            else None
        )
        result = torch.empty_like(query)
        selected_total = 0
        available_total = 0
        selected_block_ids: list[Tensor] = []
        diagnostic_positions = 0
        dense_mass_total = (
            torch.zeros((), device=x.device)
            if record_diagnostics and diagnostics == "full"
            else None
        )
        dense_recall_total = (
            torch.zeros_like(dense_mass_total) if dense_mass_total is not None else None
        )
        for position in range(length):
            starts = torch.arange(0, position + 1, self.block_size, device=x.device)
            ends = (starts + self.block_size).clamp_max(position + 1)
            summaries = (
                key_prefix.index_select(2, ends) - key_prefix.index_select(2, starts)
            ) / (ends - starts).view(1, 1, -1, 1)
            block_count = starts.numel()
            index_scores = (query[:, :, position].unsqueeze(2) * summaries).sum(-1)
            chosen = index_scores.topk(
                min(self.selected_blocks, block_count), dim=-1
            ).indices
            indices = torch.cat(
                [
                    torch.arange(
                        block * self.block_size,
                        min((block + 1) * self.block_size, position + 1),
                        device=x.device,
                    )
                    for block in range(block_count)
                ]
            )
            mask = torch.zeros_like(indices, dtype=torch.bool)
            for block in chosen.flatten().unique():
                mask |= (indices // self.block_size) == block
            indices = indices[mask]
            if membership is not None:
                membership[position, chosen] = True
            else:
                chosen_key, chosen_value = (
                    key.index_select(2, indices),
                    value.index_select(2, indices),
                )
                scores = (query[:, :, position].unsqueeze(2) * chosen_key).sum(
                    -1
                ) / math.sqrt(self.head_dim)
                result[:, :, position] = (
                    torch.softmax(scores.float(), dim=-1)
                    .unsqueeze(-1)
                    .mul(chosen_value)
                    .sum(dim=2)
                )
            if record_diagnostics:
                valid_rows = (
                    None
                    if valid_target_mask is None
                    else valid_target_mask[:, position]
                )
                valid_count = batch if valid_rows is None else int(valid_rows.sum())
                if valid_count:
                    query_count = valid_count * self.num_heads
                    diagnostic_positions += query_count
                    selected_total += indices.numel() * query_count
                    available_total += (position + 1) * query_count
                    if diagnostics == "full":
                        assert (
                            dense_mass_total is not None
                            and dense_recall_total is not None
                        )
                        with torch.no_grad():
                            selected_block_ids.append(
                                (chosen if valid_rows is None else chosen[valid_rows])
                                .detach()
                                .flatten()
                            )
                            diagnostic_query = (
                                query[:, :, position]
                                if valid_rows is None
                                else query[valid_rows, :, position]
                            )
                            diagnostic_key = (
                                key[:, :, : position + 1]
                                if valid_rows is None
                                else key[valid_rows, :, : position + 1]
                            )
                            dense_scores = (
                                diagnostic_query.unsqueeze(2) * diagnostic_key
                            ).sum(-1) / math.sqrt(self.head_dim)
                            dense_probabilities = torch.softmax(
                                dense_scores.float(), dim=-1
                            )
                            dense_mass_total += dense_probabilities.index_select(
                                2, indices
                            ).sum()
                            dense_topk = dense_scores.topk(
                                indices.numel(), dim=-1
                            ).indices
                            dense_recall_total += (
                                torch.isin(dense_topk, indices).float().mean(-1).sum()
                            )
        if membership is not None:
            from klab.model.attention.hip_sparse import selected_attention

            result = selected_attention(query, key, value, membership, self.block_size)
        if record_diagnostics:
            self.last_diagnostics = SparseAttentionDiagnostics(
                torch.tensor(available_total, device=x.device),
                torch.tensor(selected_total, device=x.device),
                torch.tensor(selected_total / max(1, available_total), device=x.device),
                (
                    torch.cat(selected_block_ids)
                    if diagnostics == "full" and selected_block_ids
                    else None
                ),
                torch.tensor(selected_total * self.head_dim, device=x.device),
                (
                    (dense_mass_total / max(1, diagnostic_positions)).detach()
                    if dense_mass_total is not None and diagnostic_positions
                    else None
                ),
                (
                    (dense_recall_total / max(1, diagnostic_positions)).detach()
                    if dense_recall_total is not None and diagnostic_positions
                    else None
                ),
            )
        return self.out_proj(
            result.transpose(1, 2).contiguous().view(batch, length, hidden)
        )
