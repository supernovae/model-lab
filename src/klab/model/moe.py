"""Local top-one routed SwiGLU experts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from klab.model.ffn import SwiGLU


@dataclass(frozen=True)
class RoutingDiagnostics:
    counts: Tensor
    fractions: Tensor
    entropy: Tensor
    maximum_fraction: Tensor
    mean_topk_probability: Tensor
    auxiliary_loss: Tensor
    router_logits: Tensor | None
    selected_experts: Tensor | None
    selected_weights: Tensor | None


_diagnostic_writes_enabled: ContextVar[bool] = ContextVar(
    "klab_diagnostic_writes_enabled", default=True
)


def diagnostic_writes_enabled() -> bool:
    """Whether this forward may update diagnostic snapshots."""
    return _diagnostic_writes_enabled.get()


@contextmanager
def suppress_diagnostic_writes() -> Iterator[None]:
    """Prevent checkpoint recomputation from replacing forward diagnostics."""
    token = _diagnostic_writes_enabled.set(False)
    try:
        yield
    finally:
        _diagnostic_writes_enabled.reset(token)


class TopKMoE(nn.Module):
    """Readable local top-k SwiGLU dispatch with normalized router weights.

    Tokens are flattened from ``[B, T, D]`` to ``[B*T, D]``.  Each selected
    expert receives only its selected rows; weighted ``index_add_`` restores
    the original token order without materializing an expert-by-token tensor.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        experts_per_token: int = 1,
        shared_expert: bool = False,
        auxiliary_loss_coefficient: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.auxiliary_loss_coefficient = auxiliary_loss_coefficient
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(hidden_dim, ffn_dim) for _ in range(num_experts)
        )
        self.shared_expert = SwiGLU(hidden_dim, ffn_dim) if shared_expert else None
        self.last_diagnostics: RoutingDiagnostics | None = None

    def forward_with_aux(
        self,
        x: Tensor,
        *,
        valid_target_mask: Tensor | None = None,
        diagnostics: Literal["scalar", "full"] = "scalar",
    ) -> tuple[Tensor, Tensor]:
        if diagnostics not in ("scalar", "full"):
            raise ValueError(f"unknown diagnostics policy: {diagnostics}")
        original_shape = x.shape
        tokens = x.reshape(-1, original_shape[-1])
        router_logits = self.router(tokens)
        probabilities = torch.softmax(router_logits.float(), dim=-1)
        weights, selected = probabilities.topk(self.experts_per_token, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(tokens)
        for expert_index, expert in enumerate(self.experts):
            token_indices, choice_indices = (selected == expert_index).nonzero(
                as_tuple=True
            )
            if token_indices.numel():
                contribution = expert(tokens.index_select(0, token_indices))
                contribution = contribution * weights[
                    token_indices, choice_indices
                ].unsqueeze(-1)
                output.index_add_(0, token_indices, contribution.to(output.dtype))
        if self.shared_expert is not None:
            output = output + self.shared_expert(tokens)
        if valid_target_mask is None:
            valid = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        else:
            valid = valid_target_mask.reshape(-1).to(dtype=torch.bool)
            if valid.shape != (tokens.shape[0],):
                raise ValueError("valid_target_mask must align with routed tokens")
        valid_probabilities = probabilities[valid]
        valid_selected = selected[valid]
        counts = torch.bincount(
            valid_selected.flatten(), minlength=self.num_experts
        ).detach()
        fractions = counts.float() / max(1, valid_selected.numel())
        importance = (
            valid_probabilities.mean(dim=0)
            if valid_probabilities.numel()
            else probabilities.new_zeros(self.num_experts)
        )
        load = fractions.detach()
        auxiliary_loss = (
            self.auxiliary_loss_coefficient
            * self.num_experts
            * (importance * load).sum()
        )
        if diagnostic_writes_enabled():
            # Diagnostic-only operations must not add saved autograd tensors:
            # this branch is intentionally absent during block recomputation.
            with torch.no_grad():
                entropy = (
                    -(
                        valid_probabilities
                        * valid_probabilities.clamp_min(
                            torch.finfo(probabilities.dtype).tiny
                        ).log()
                    )
                    .sum(dim=-1)
                    .mean()
                    if valid_probabilities.numel()
                    else probabilities.new_zeros(())
                )
                self.last_diagnostics = RoutingDiagnostics(
                    counts=counts,
                    fractions=fractions,
                    entropy=entropy,
                    maximum_fraction=fractions.max(),
                    mean_topk_probability=(
                        valid_probabilities.gather(1, valid_selected).sum(dim=-1).mean()
                        if valid_probabilities.numel()
                        else probabilities.new_zeros(())
                    ),
                    auxiliary_loss=auxiliary_loss.detach(),
                    router_logits=router_logits.detach()[valid]
                    if diagnostics == "full"
                    else None,
                    selected_experts=valid_selected.detach()
                    if diagnostics == "full"
                    else None,
                    selected_weights=weights.detach()[valid]
                    if diagnostics == "full"
                    else None,
                )
        return output.reshape(original_shape), auxiliary_loss

    def forward(
        self,
        x: Tensor,
        *,
        valid_target_mask: Tensor | None = None,
        diagnostics: Literal["scalar", "full"] = "scalar",
    ) -> Tensor:
        return self.forward_with_aux(
            x, valid_target_mask=valid_target_mask, diagnostics=diagnostics
        )[0]
