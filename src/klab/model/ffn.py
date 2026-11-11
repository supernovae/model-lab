"""Bias-free SwiGLU feed-forward network."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(functional.silu(self.gate(x)) * self.up(x))
