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
    document_scoped: bool = False


def _diagnostics(
    addresses: Tensor,
    gate: Tensor,
    values: Tensor,
    hidden: Tensor,
    table_size: int,
    position_mask: Tensor | None = None,
) -> MemoryDiagnostics:
    if position_mask is not None:
        # Document-scoped occupancy; additive across microbatch boundaries.
        # Bincount weights exclude padding without dynamic-sized indexing.
        weights = position_mask.to(torch.float32)
        offsets = (
            torch.arange(addresses.shape[0], device=addresses.device)[:, None]
            * table_size
        )
        counts = torch.bincount(
            (addresses + offsets).flatten(),
            weights=weights.flatten(),
            minlength=addresses.shape[0] * table_size,
        ).reshape(addresses.shape[0], table_size)
        lookups = weights.sum()
        unique = (counts > 0).sum()
        collisions = lookups - unique
        denominator = lookups.clamp_min(1)
        lengths = weights.sum(-1)
        return MemoryDiagnostics(
            document_scoped=True,
            lookup_count=lookups.detach(),
            unique_addresses=unique.detach(),
            collision_count=collisions.detach(),
            bucket_reuse_rate=(collisions / denominator).detach(),
            table_utilization=(
                ((counts > 0).sum(-1) * lengths).sum() / denominator / table_size
            ).detach(),
            maximum_address_fraction=(
                counts.max(-1).values.sum() / denominator
            ).detach(),
            gate_mean=((gate.squeeze(-1) * weights).sum() / denominator).detach(),
            value_norm=((values.norm(dim=-1) * weights).sum() / denominator).detach(),
            hidden_norm=((hidden.norm(dim=-1) * weights).sum() / denominator).detach(),
        )
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

    def _shifted_inputs(self, input_ids: Tensor, max_order: int) -> list[Tensor]:
        if self.token_mapping is not None:
            input_ids = self.token_mapping[input_ids]
        length = input_ids.shape[1]
        shifts = [input_ids]
        for offset in range(1, max_order):
            shifted = torch.full_like(input_ids, self.padding_id)
            if offset < length:
                shifted[:, offset:] = input_ids[:, :-offset]
            shifts.append(shifted)
        return shifts

    def _addresses_from_shifts(
        self, shifts: list[Tensor], order: int, seed: int
    ) -> Tensor:
        base = shifts[0]
        if self.addressing == "prime_xor_v1":
            stream = self.ngram_orders.index(order) * self.hash_heads + seed
            address = torch.zeros_like(base, dtype=torch.long)
            for offset in range(order):
                address = engram_address_step(
                    address, shifts[offset], self.multipliers[offset]
                )
            return engram_address_finish(address, self.table_sizes[stream])
        address = torch.full(
            base.shape,
            token_address_init(seed),
            dtype=torch.long,
            device=base.device,
        )
        for offset in range(order):
            address = token_address_step(address, shifts[offset], self.table_size, seed)
        return token_address_finish(address, self.table_size, seed)

    def addresses(
        self, input_ids: Tensor, order: int | None = None, seed: int = 0
    ) -> Tensor:
        """Hash only IDs at or before each position for one order/hash head."""
        order = self.ngram_size if order is None else order
        shifts = self._shifted_inputs(input_ids, order)
        return self._addresses_from_shifts(shifts, order, seed)

    def _all_addresses(self, input_ids: Tensor) -> list[Tensor]:
        if (
            "addresses" in self.__dict__
            or getattr(type(self).addresses, "__func__", type(self).addresses)
            is not _DEFAULT_ADDRESSES
        ):
            return [
                self.addresses(input_ids, order, head)
                for order in self.ngram_orders
                for head in range(self.hash_heads)
            ]
        shifts = self._shifted_inputs(input_ids, max(self.ngram_orders))
        return [
            self._addresses_from_shifts(shifts, order, head)
            for order in self.ngram_orders
            for head in range(self.hash_heads)
        ]

    def _apply_addresses(
        self,
        hidden: Tensor,
        addresses: list[Tensor],
        history: Tensor | None = None,
        position_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        tables = (self.table, *self.extra_tables)
        retrieved = [
            table(address) for table, address in zip(tables, addresses, strict=True)
        ]
        # Preserve the vectorized legacy path and its shared-table diagnostics.
        if (
            self.variant == "hidden_gate"
            and self.addressing == "legacy"
            and position_mask is None
        ):
            stream_values = [self.output(value) for value in retrieved]
            stacked_values = torch.stack(stream_values)
            values = stacked_values.mean(dim=0)
            gate = torch.sigmoid(self.gate(hidden))
            num_streams = len(addresses)
            gate_mean = gate.mean().detach()
            hidden_norm = hidden.norm(dim=-1).mean().detach()
            stream_value_norms = stacked_values.norm(dim=-1).mean(dim=(1, 2)).detach()
            per_stream_lookups = addresses[0].numel()
            lookup_tensor = torch.tensor(per_stream_lookups, device=hidden.device)
            stream_denom = lookup_tensor.clamp_min(1)
            flat_addresses = torch.stack([address.flatten() for address in addresses])
            stream_offsets = (
                torch.arange(num_streams, device=hidden.device)[:, None]
                * self.table_size
            )
            counts_by_stream = torch.bincount(
                (flat_addresses + stream_offsets).flatten(),
                minlength=num_streams * self.table_size,
            ).view(num_streams, self.table_size)
            uniques = (counts_by_stream > 0).sum(dim=-1)
            collisions = lookup_tensor - uniques
            reuse_rates = (collisions.float() / stream_denom).detach()
            utilizations = (uniques.float() / self.table_size).detach()
            max_fractions = (
                counts_by_stream.max(dim=-1).values.float() / stream_denom
            ).detach()
            stream_diagnostics = tuple(
                MemoryDiagnostics(
                    lookup_count=lookup_tensor.detach(),
                    unique_addresses=uniques[index].detach(),
                    collision_count=collisions[index].detach(),
                    bucket_reuse_rate=reuse_rates[index],
                    table_utilization=utilizations[index],
                    maximum_address_fraction=max_fractions[index],
                    gate_mean=gate_mean,
                    value_norm=stream_value_norms[index],
                    hidden_norm=hidden_norm,
                )
                for index in range(num_streams)
            )
            combined_counts = counts_by_stream.sum(dim=0)
            total_lookups = torch.tensor(
                per_stream_lookups * num_streams, device=hidden.device
            )
            total_unique = (combined_counts > 0).sum()
            total_collisions = total_lookups - total_unique
            total_denom = total_lookups.clamp_min(1)
            self.last_diagnostics = MemoryDiagnostics(
                lookup_count=total_lookups.detach(),
                unique_addresses=total_unique.detach(),
                collision_count=total_collisions.detach(),
                bucket_reuse_rate=(total_collisions.float() / total_denom).detach(),
                table_utilization=(total_unique.float() / self.table_size).detach(),
                maximum_address_fraction=(
                    combined_counts.max().float() / total_denom
                ).detach(),
                gate_mean=gate_mean,
                value_norm=values.norm(dim=-1).mean().detach(),
                hidden_norm=hidden_norm,
                streams=stream_diagnostics,
            )
            return hidden + gate * values, None
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
            _diagnostics(address, gate, value, hidden, size, position_mask)
            for address, value, size in zip(
                addresses, stream_values, self.table_sizes, strict=True
            )
        )
        if position_mask is not None:
            combined = stream_diagnostics
            first = combined[0]
            weights = position_mask.to(torch.float32)
            self.last_diagnostics = replace(
                first,
                lookup_count=sum(d.lookup_count for d in combined),
                unique_addresses=sum(d.unique_addresses for d in combined),
                collision_count=sum(d.collision_count for d in combined),
                bucket_reuse_rate=torch.stack(
                    [d.bucket_reuse_rate for d in combined]
                ).mean(),
                table_utilization=sum(
                    d.table_utilization * size
                    for d, size in zip(combined, self.table_sizes, strict=True)
                )
                / sum(self.table_sizes),
                maximum_address_fraction=torch.stack(
                    [d.maximum_address_fraction for d in combined]
                ).mean(),
                value_norm=(
                    (values.norm(dim=-1) * weights).sum() / weights.sum().clamp_min(1)
                ).detach(),
                streams=stream_diagnostics,
            )
        else:
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

    def forward(
        self, hidden: Tensor, input_ids: Tensor, *, position_mask: Tensor | None = None
    ) -> Tensor:
        return self._apply_addresses(
            hidden,
            self._all_addresses(input_ids),
            position_mask=position_mask,
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
            address[:, -hidden.shape[1] :] for address in self._all_addresses(suffix)
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


_DEFAULT_ADDRESSES = TokenNgramMemory.addresses


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
