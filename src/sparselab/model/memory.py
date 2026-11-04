"""Local causal token n-gram memory adapter."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sparselab.address_hash import (
    engram_address_finish,
    engram_address_step,
    engram_multipliers,
    token_address_finish,
    token_address_init,
    token_address_step,
    validate_engram_table_sizes,
)
from sparselab.model.norm import RMSNorm

if TYPE_CHECKING:
    from sparselab.config.models import ModelConfig


@dataclass(frozen=True)
class MemoryDiagnostics:
    lookup_count: Tensor
    unique_addresses: Tensor
    collision_count: Tensor
    bucket_reuse_rate: Tensor
    table_utilization: Tensor
    maximum_address_fraction: Tensor
    gate_mean: Tensor
    value_norm: Tensor
    hidden_norm: Tensor
    streams: tuple[MemoryDiagnostics, ...] = ()


def _diagnostics(
    addresses: Tensor,
    gate: Tensor,
    values: Tensor,
    hidden: Tensor,
    table_size: int,
) -> MemoryDiagnostics:
    counts = torch.bincount(addresses.flatten(), minlength=table_size)
    lookup_count = torch.tensor(addresses.numel(), device=addresses.device)
    unique = (counts > 0).sum()
    collisions = lookup_count - unique
    denominator = lookup_count.clamp_min(1)
    return MemoryDiagnostics(
        lookup_count=lookup_count.detach(),
        unique_addresses=unique.detach(),
        collision_count=collisions.detach(),
        bucket_reuse_rate=(collisions.float() / denominator).detach(),
        table_utilization=(unique.float() / table_size).detach(),
        maximum_address_fraction=(counts.max().float() / denominator).detach(),
        gate_mean=gate.mean().detach(),
        value_norm=values.norm(dim=-1).mean().detach(),
        hidden_norm=hidden.norm(dim=-1).mean().detach(),
    )


class ContextKeyGate(nn.Module):
    """Engram equation (4) logits; sigmoid is applied by the memory adapter."""

    def __init__(
        self, hidden_dim: int, eps: float, transform: str = "identity"
    ) -> None:
        super().__init__()
        self.query_norm = RMSNorm(hidden_dim, eps)
        self.key_norm = RMSNorm(hidden_dim, eps)
        self.scale = hidden_dim**-0.5
        self.transform = transform

    def forward(self, hidden: Tensor, key: Tensor) -> Tensor:
        query = self.query_norm(hidden).float()
        normalized_key = self.key_norm(key).float()
        logits = (query * normalized_key).sum(dim=-1, keepdim=True) * self.scale
        if self.transform == "signed_sqrt":
            logits = logits.abs().clamp_min(1e-6).sqrt() * logits.sign()
        return logits.to(hidden.dtype)


class TokenNgramMemory(nn.Module):
    """Map causal token-ID suffixes into trainable memory values."""

    def __init__(
        self,
        hidden_dim: int,
        table_size: int,
        ngram_size: int,
        value_dim: int,
        ngram_orders: tuple[int, ...] = (),
        hash_heads: int = 1,
        variant: Literal["hidden_gate", "context_key"] = "hidden_gate",
        norm_eps: float = 1e-6,
        conv_kernel_size: int = 0,
        addressing: str = "legacy",
        table_sizes: tuple[int, ...] = (),
        hash_seed: int = 0,
        token_mapping: tuple[int, ...] | None = None,
        vocab_size: int | None = None,
        layer_id: int = 0,
        padding_token_id: int = 0,
        gate_transform: str = "identity",
    ) -> None:
        super().__init__()
        self.table_size = table_size
        self.ngram_size = ngram_size
        self.ngram_orders = ngram_orders or (ngram_size,)
        self.hash_heads = hash_heads
        self.addressing = addressing
        self.padding_id = (
            token_mapping[padding_token_id]
            if token_mapping is not None
            else padding_token_id
        )
        streams = len(self.ngram_orders) * hash_heads
        if addressing == "prime_xor_v1":
            self.table_sizes = validate_engram_table_sizes(table_sizes, streams)
            if self.table_sizes[0] != table_size or vocab_size is None:
                raise ValueError(
                    "prime XOR requires vocabulary and matching first table size"
                )
            memory_vocab = vocab_size  # Fixed original-vocabulary bound isolates mapping ablations.
            self.multipliers = engram_multipliers(
                hash_seed, max(self.ngram_orders), memory_vocab, layer_id=layer_id
            )
        elif addressing == "legacy" and not table_sizes and not hash_seed:
            self.table_sizes = (table_size,) * streams
            self.multipliers = ()
        else:
            raise ValueError("invalid token memory addressing settings")
        self.register_buffer(
            "token_mapping",
            None
            if token_mapping is None
            else torch.tensor(token_mapping, dtype=torch.long),
            persistent=False,
        )

        if variant not in {"hidden_gate", "context_key"}:
            raise ValueError(f"unknown token memory variant: {variant}")
        self.variant = variant
        if conv_kernel_size and variant != "context_key":
            raise ValueError("memory convolution requires context_key memory")
        self.conv_history_length = max(self.ngram_orders) * max(conv_kernel_size - 1, 0)
        self.conv = (
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                conv_kernel_size,
                groups=hidden_dim,
                dilation=max(self.ngram_orders),
                bias=False,
            )
            if conv_kernel_size
            else None
        )
        self.conv_norm = RMSNorm(hidden_dim, norm_eps) if conv_kernel_size else None
        self.table = nn.Embedding(table_size, value_dim)
        self.extra_tables = nn.ModuleList(
            nn.Embedding(size, value_dim) for size in self.table_sizes[1:]
        )
        if variant == "context_key":
            concatenated_dim = value_dim * len(self.ngram_orders) * hash_heads
            self.output = nn.Linear(concatenated_dim, hidden_dim, bias=False)
            self.key = nn.Linear(concatenated_dim, hidden_dim, bias=False)
            self.gate = ContextKeyGate(hidden_dim, norm_eps, gate_transform)
        else:
            self.output = nn.Linear(value_dim, hidden_dim, bias=False)
            self.gate = nn.Linear(hidden_dim, 1, bias=False)
        self.last_diagnostics: MemoryDiagnostics | None = None

    def addresses(
        self, input_ids: Tensor, order: int | None = None, seed: int = 0
    ) -> Tensor:
        """Hash only IDs at or before each position for one order/hash head."""
        batch, length = input_ids.shape
        order = self.ngram_size if order is None else order
        if self.token_mapping is not None:
            input_ids = self.token_mapping[input_ids]
        if self.addressing == "prime_xor_v1":
            stream = self.ngram_orders.index(order) * self.hash_heads + seed
            address = torch.zeros_like(input_ids, dtype=torch.long)
            # Padding is mapped through the same memory vocabulary as input tokens.
            for offset in range(order):
                shifted = torch.full_like(input_ids, self.padding_id)
                if offset == 0:
                    shifted = input_ids
                elif offset < length:
                    shifted[:, offset:] = input_ids[:, :-offset]
                address = engram_address_step(
                    address, shifted, self.multipliers[offset]
                )
            return engram_address_finish(address, self.table_sizes[stream])
        address = torch.full(
            (batch, length),
            token_address_init(seed),
            dtype=torch.long,
            device=input_ids.device,
        )
        order = self.ngram_size if order is None else order
        for offset in range(order):
            shifted = torch.full_like(input_ids, self.padding_id)
            if offset == 0:
                shifted = input_ids
            elif offset < length:
                shifted[:, offset:] = input_ids[:, :-offset]
            address = token_address_step(address, shifted, self.table_size, seed)
        return token_address_finish(address, self.table_size, seed)

    def _apply_addresses(
        self, hidden: Tensor, addresses: list[Tensor], history: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        tables = (self.table, *self.extra_tables)
        retrieved = [
            table(address) for table, address in zip(tables, addresses, strict=True)
        ]
        if self.variant == "context_key":
            concatenated = torch.cat(retrieved, dim=-1)
            values = self.output(concatenated)
            gate = torch.sigmoid(self.gate(hidden, self.key(concatenated)))
            stream_values = retrieved
        else:
            stream_values = [self.output(value) for value in retrieved]
            values = torch.stack(stream_values).mean(dim=0)
            gate = torch.sigmoid(self.gate(hidden))
        stream_diagnostics = tuple(
            _diagnostics(address, gate, value, hidden, size)
            for address, value, size in zip(
                addresses, stream_values, self.table_sizes, strict=True
            )
        )
        self.last_diagnostics = replace(
            _diagnostics(
                torch.cat(
                    [
                        address.flatten() + sum(self.table_sizes[:index])
                        for index, address in enumerate(addresses)
                    ]
                ),
                gate,
                values,
                hidden,
                sum(self.table_sizes),
            ),
            streams=stream_diagnostics,
        )
        gated = gate * values
        next_history = None
        if self.conv is not None:
            normalized = self.conv_norm(gated)
            if history is None:
                history = normalized.new_zeros(
                    (hidden.shape[0], self.conv_history_length, hidden.shape[-1])
                )
            expected = (hidden.shape[0], self.conv_history_length, hidden.shape[-1])
            if (
                history.shape != expected
                or history.device != normalized.device
                or history.dtype != normalized.dtype
            ):
                raise ValueError(
                    "memory convolution cache shape, device or dtype mismatch"
                )
            sequence = torch.cat((history, normalized), dim=1)
            refined = self.conv(sequence.transpose(1, 2)).transpose(1, 2)
            gated = gated + F.silu(refined)
            next_history = (
                sequence[:, -self.conv_history_length :].detach().clone()
                if self.conv_history_length
                else sequence[:, :0].detach().clone()
            )
        elif history is not None:
            raise ValueError("memory history supplied without convolution")
        return hidden + gated, next_history

    def forward(self, hidden: Tensor, input_ids: Tensor) -> Tensor:
        return self._apply_addresses(
            hidden,
            [
                self.addresses(input_ids, order, head)
                for order in self.ngram_orders
                for head in range(self.hash_heads)
            ],
        )[0]

    def forward_last(
        self,
        hidden: Tensor,
        history_ids: Tensor,
        convolution_history: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Apply causal n-gram memory to appended hidden states only.

        The bounded suffix retains exactly the token history each hash order
        needs, including the leading zero-padding semantics at a fresh context.
        """
        if hidden.shape[1] > history_ids.shape[1]:
            raise ValueError("n-gram history is shorter than appended hidden states")
        required = max(self.ngram_orders)
        suffix = history_ids[:, -max(required + hidden.shape[1] - 1, 1) :]
        addresses = [
            self.addresses(suffix, order, head)[:, -hidden.shape[1] :]
            for order in self.ngram_orders
            for head in range(self.hash_heads)
        ]
        return self._apply_addresses(hidden, addresses, convolution_history)


def build_token_memory(
    model: ModelConfig, block_index: int | None = None
) -> TokenNgramMemory:
    """Construct token memory from its complete architecture-bound settings."""
    sites = model.memory_block_sites
    block_index = sites[0] if block_index is None and sites else block_index
    if block_index is not None and block_index not in sites:
        raise ValueError("memory block is not a configured port")
    port = sites.index(block_index) if sites else 0
    streams = (len(model.memory_ngram_orders) or 1) * model.memory_hash_heads
    sizes = model.memory_table_sizes[port * streams : (port + 1) * streams]
    return TokenNgramMemory(
        model.hidden_dim,
        sizes[0] if sizes else model.memory_table_size,
        model.memory_ngram_size,
        model.memory_dim,
        model.memory_ngram_orders,
        model.memory_hash_heads,
        model.memory_variant,
        model.rms_norm_eps,
        model.memory_conv_kernel_size,
        addressing=model.memory_addressing,
        table_sizes=sizes,
        hash_seed=model.memory_hash_seed,
        token_mapping=None
        if model.memory_token_mapping is None
        else model.memory_token_mapping.token_ids,
        vocab_size=model.vocab_size,
        layer_id=block_index or 0,
        padding_token_id=model.memory_padding_token_id,
        gate_transform=model.memory_gate_transform,
    )


class ByteAddressMemory(nn.Module):
    """Apply precomputed causal byte addresses supplied by prepared data."""

    def __init__(self, hidden_dim: int, table_size: int, value_dim: int) -> None:
        super().__init__()
        self.table_size = table_size
        self.table = nn.Embedding(table_size, value_dim)
        self.output = nn.Linear(value_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, 1, bias=False)
        self.last_diagnostics: MemoryDiagnostics | None = None

    def forward(self, hidden: Tensor, addresses: Tensor) -> Tensor:
        if addresses.shape != hidden.shape[:2]:
            raise ValueError("byte addresses must have shape [batch, sequence]")
        if addresses.dtype != torch.long:
            raise ValueError("byte addresses must be int64")
        if addresses.numel() and (
            addresses.min() < 0 or addresses.max() >= self.table_size
        ):
            raise ValueError("byte address is outside configured memory table")
        values = self.output(self.table(addresses))
        gate = torch.sigmoid(self.gate(hidden))
        self.last_diagnostics = _diagnostics(
            addresses, gate, values, hidden, self.table_size
        )
        return hidden + gate * values
