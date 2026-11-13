"""Real-valued adjacent-pair rotary position embedding, with context scaling.

The scaled schedules follow the Hugging Face ``transformers`` definitions
(``modeling_rope_utils``), so an exported model reproduces Lab's frequencies:

* ``linear`` (position interpolation): every frequency divided by ``factor``.
* ``ntk`` (static NTK-aware): the base raised to
  ``base * factor ** (head_dim / (head_dim - 2))``; it exports as that base.
* ``yarn``: frequencies blended between extrapolation and interpolation by a
  ramp over wavelength, and cos/sin scaled by ``0.1 * ln(factor) + 1``.
* ``llama3``: Llama 3.1's piecewise schedule over wavelength.

``factor`` is always ``max_seq_len / original_max_seq_len``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

RopeScalingType = Literal["linear", "ntk", "yarn", "llama3"]


@dataclass(frozen=True)
class RopeScaling:
    type: RopeScalingType
    factor: float
    original_max_seq_len: int
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0

    def __post_init__(self) -> None:
        if self.factor <= 1:
            raise ValueError("RoPE scaling factor must be greater than 1")


def resolve_rope_scaling(settings: Any, max_seq_len: int) -> RopeScaling | None:
    """The schedule for ``attention.rope_scaling`` settings, or ``None``.

    Accepts the config model or the plain mapping an exported model carries;
    unset optional settings keep their defaults.
    """
    if settings is None:
        return None
    if not isinstance(settings, Mapping):
        settings = settings.model_dump(exclude_none=True)
    values = {key: value for key, value in settings.items() if value is not None}
    original = int(values.pop("original_max_seq_len"))
    return RopeScaling(
        type=values.pop("type"),
        factor=max_seq_len / original,
        original_max_seq_len=original,
        **values,
    )


def _default_inverse_frequency(head_dim: int, base: float) -> Tensor:
    return base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)


def ntk_base(head_dim: int, base: float, factor: float) -> float:
    """The base static NTK-aware scaling uses in place of ``base``."""
    return base * factor ** (head_dim / (head_dim - 2))


def rope_schedule(
    head_dim: int, base: float, scaling: RopeScaling | None
) -> tuple[Tensor, float]:
    """``(inverse_frequency, attention_factor)`` for one head dimension."""
    if scaling is None:
        return _default_inverse_frequency(head_dim, base), 1.0
    factor = scaling.factor
    if scaling.type == "linear":
        return _default_inverse_frequency(head_dim, base) / factor, 1.0
    if scaling.type == "ntk":
        return _default_inverse_frequency(
            head_dim, ntk_base(head_dim, base, factor)
        ), 1.0
    if scaling.type == "yarn":
        return _yarn(head_dim, base, scaling)
    if scaling.type == "llama3":
        return _llama3(head_dim, base, scaling), 1.0
    raise ValueError(f"unknown RoPE scaling type {scaling.type!r}")


def _yarn(head_dim: int, base: float, scaling: RopeScaling) -> tuple[Tensor, float]:
    factor = scaling.factor
    original = scaling.original_max_seq_len

    def correction_dim(rotations: float) -> float:
        return (head_dim * math.log(original / (rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = max(math.floor(correction_dim(scaling.beta_fast)), 0)
    high = min(math.ceil(correction_dim(scaling.beta_slow)), head_dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp(
        (torch.arange(head_dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1
    )
    extrapolation_weight = 1 - ramp
    position_frequency = base ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    )
    extrapolation = 1.0 / position_frequency
    interpolation = 1.0 / (factor * position_frequency)
    inverse_frequency = (
        interpolation * (1 - extrapolation_weight)
        + extrapolation * extrapolation_weight
    )
    return inverse_frequency, 0.1 * math.log(factor) + 1.0


def _llama3(head_dim: int, base: float, scaling: RopeScaling) -> Tensor:
    factor = scaling.factor
    original = scaling.original_max_seq_len
    low, high = scaling.low_freq_factor, scaling.high_freq_factor
    inverse_frequency = _default_inverse_frequency(head_dim, base)
    wavelength = 2 * math.pi / inverse_frequency
    low_wavelength = original / low
    high_wavelength = original / high
    scaled = torch.where(
        wavelength > low_wavelength, inverse_frequency / factor, inverse_frequency
    )
    smooth = (original / wavelength - low) / (high - low)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    medium = ~(wavelength < high_wavelength) & ~(wavelength > low_wavelength)
    return torch.where(medium, smoothed, scaled)


class RoPE(nn.Module):
    def __init__(
        self, head_dim: int, base: float, scaling: RopeScaling | None = None
    ) -> None:
        super().__init__()
        inverse_frequency, attention_factor = rope_schedule(head_dim, base, scaling)
        self.attention_factor = attention_factor
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
                self.cos = (angles.cos() * self.attention_factor)[None, None, :, :]
                self.sin = (angles.sin() * self.attention_factor)[None, None, :, :]
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
