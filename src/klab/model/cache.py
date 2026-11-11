"""Ephemeral, bounded inference-only key/value cache contracts.

These plain dataclasses deliberately are not modules.  A cache is owned by one
``DenseLM`` instance and one request; it is never part of a state dict,
checkpoint, or reusable model state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class IncrementalCacheCapability:
    """Whether an attention implementation can decode from appended K/V state."""

    supported: bool
    reason: str | None = None
    backend: Literal["dense", "sliding", "mla"] | None = None


@dataclass
class AttentionKVCache:
    """Preallocated projected K/V storage for one attention layer."""

    key: Tensor
    value: Tensor
    length: int
    position: int

    @property
    def capacity(self) -> int:
        return self.key.shape[2]

    @property
    def allocated_bytes(self) -> int:
        return (
            self.key.numel() * self.key.element_size()
            + self.value.numel() * self.value.element_size()
        )

    def append(self, key: Tensor, value: Tensor, *, position: int) -> None:
        """Append projected values without reallocating or retaining autograd state."""
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "projected K/V tensors must have shape [batch, heads, sequence, dim]"
            )
        if key.shape[:3] != value.shape[:3]:
            raise ValueError(
                "KV cache key/value batch, head, and sequence shapes differ"
            )
        if (
            key.shape[:2] != self.key.shape[:2]
            or value.shape[:2] != self.value.shape[:2]
        ):
            raise ValueError("KV cache batch/head shape does not match attention input")
        if (
            key.shape[-1] != self.key.shape[-1]
            or value.shape[-1] != self.value.shape[-1]
        ):
            raise ValueError("KV cache head dimensions do not match attention input")
        if key.device != self.key.device or value.device != self.value.device:
            raise ValueError("KV cache device does not match attention input")
        if key.dtype != self.key.dtype or value.dtype != self.value.dtype:
            raise ValueError("KV cache dtype does not match attention input")
        end = self.length + key.shape[2]
        if end > self.capacity:
            raise ValueError("KV cache capacity exceeded; rebuild the active context")
        self.key[:, :, self.length : end].copy_(key.detach())
        self.value[:, :, self.length : end].copy_(value.detach())
        self.length = end
        self.position = position


@dataclass
class GenerationCache:
    """Bounded request-local transformer state used only by autoregressive decoding."""

    layers: list[AttentionKVCache]
    input_ids: Tensor
    length: int
    capacity: int
    mode: Literal["dense", "sliding", "mla"]
    owner: object
    batch_size: int
    device: torch.device
    input_dtype: torch.dtype
    memory_history: Tensor | None = None
    memory_histories: dict[str, Tensor] = field(default_factory=dict)

    @property
    def allocated_bytes(self) -> int:
        memory_bytes = (
            self.memory_history.numel() * self.memory_history.element_size()
            if self.memory_history is not None
            else 0
        )
        return (
            self.input_ids.numel() * self.input_ids.element_size()
            + sum(layer.allocated_bytes for layer in self.layers)
            + memory_bytes
            + sum(
                value.numel() * value.element_size()
                for value in self.memory_histories.values()
            )
        )

    def append_input_ids(self, input_ids: Tensor) -> None:
        """Append request IDs into preallocated storage without a ``torch.cat`` copy."""
        if input_ids.ndim != 2 or input_ids.shape[0] != self.batch_size:
            raise ValueError("generation cache batch size does not match input_ids")
        if input_ids.device != self.device or input_ids.dtype != self.input_dtype:
            raise ValueError(
                "generation cache input IDs device or dtype does not match"
            )
        end = self.length + input_ids.shape[1]
        if end > self.capacity:
            raise ValueError(
                "generation cache capacity exceeded; rebuild the active context"
            )
        self.input_ids[:, self.length : end].copy_(input_ids.detach())
        self.length = end

    def rewind(
        self,
        length: int,
        *,
        memory_history: Tensor | None,
        memory_histories: dict[str, Tensor],
    ) -> None:
        """Drop appended positions, for a caller that fed tokens speculatively.

        Stored K/V and IDs past ``length`` become unused and are overwritten by
        the next append. Rolling memory state cannot be shortened, so the caller
        supplies the histories that belong to ``length``.
        """
        if type(length) is not int or not 0 <= length <= self.length:
            raise ValueError("rewind length must be within the cached positions")
        self.length = length
        for layer in self.layers:
            layer.length = length
            layer.position = length
        self.memory_history = memory_history
        self.memory_histories = dict(memory_histories)
