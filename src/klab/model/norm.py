"""Root-mean-square normalization."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return x * scale.to(dtype=x.dtype) * self.weight
