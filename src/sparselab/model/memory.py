"""Local causal token n-gram memory adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sparselab.address_hash import (
    token_address_finish,
    token_address_init,
    token_address_step,
)


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
    ) -> None:
        super().__init__()
        self.table_size = table_size
        self.ngram_size = ngram_size
        self.ngram_orders = ngram_orders or (ngram_size,)
        self.hash_heads = hash_heads
        self.table = nn.Embedding(table_size, value_dim)
        self.extra_tables = nn.ModuleList(
            nn.Embedding(table_size, value_dim)
            for _ in range(len(self.ngram_orders) * hash_heads - 1)
        )
        self.output = nn.Linear(value_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, 1, bias=False)
        self.last_diagnostics: MemoryDiagnostics | None = None

    def _shifted_inputs(self, input_ids: Tensor, max_order: int) -> list[Tensor]:
        length = input_ids.shape[1]
        shifts = [input_ids]
        for offset in range(1, max_order):
            shifted = torch.zeros_like(input_ids)
            if offset < length:
                shifted[:, offset:] = input_ids[:, :-offset]
            shifts.append(shifted)
        return shifts

    def _addresses_from_shifts(
        self, shifts: list[Tensor], order: int, seed: int
    ) -> Tensor:
        base = shifts[0]
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

    def _apply_addresses(self, hidden: Tensor, addresses: list[Tensor]) -> Tensor:
        tables = (self.table, *self.extra_tables)
        stream_values = [
            self.output(table(address))
            for table, address in zip(tables, addresses, strict=True)
        ]
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
            torch.arange(num_streams, device=hidden.device)[:, None] * self.table_size
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
        return hidden + gate * values

    def forward(self, hidden: Tensor, input_ids: Tensor) -> Tensor:
        return self._apply_addresses(hidden, self._all_addresses(input_ids))

    def forward_last(self, hidden: Tensor, history_ids: Tensor) -> Tensor:
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
        return self._apply_addresses(hidden, addresses)


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
