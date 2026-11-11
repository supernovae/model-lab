"""Real-valued adjacent-pair rotary position embedding."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        inverse_frequency = base ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

    def _cache(self, length: int, device: torch.device) -> tuple[Tensor, Tensor]:
        if (
            self.cos.numel() == 0
            or self.cos.shape[2] < length
            or self.cos.device != device
        ):
            # Evaluation may populate this cache before the next training update.
            # Persistent caches must be ordinary constants, not inference tensors.
            with torch.inference_mode(False), torch.no_grad():
                positions = torch.arange(length, device=device, dtype=torch.float32)
                angles = torch.outer(positions, self.inverse_frequency.to(device))
                self.cos = angles.cos()[None, None, :, :]
                self.sin = angles.sin()[None, None, :, :]
        return self.cos[:, :, :length], self.sin[:, :, :length]

    def reserve(self, length: int, device: torch.device) -> None:
        """Precompute bounded constants before tracing dynamic training shapes."""
        if length < 1:
            raise ValueError("RoPE reservation requires a positive length")
        self._cache(length, device)

    def forward(self, x: Tensor, *, position_offset: int = 0) -> Tensor:
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative")
        end = position_offset + x.shape[-2]
        cos, sin = self._cache(end, x.device)
        cos = cos[:, :, position_offset:end]
        sin = sin[:, :, position_offset:end]
        even, odd = x[..., 0::2], x[..., 1::2]
        output = torch.empty_like(x)
        output[..., 0::2] = even * cos + -odd * sin
        output[..., 1::2] = even * sin + odd * cos
        return output
