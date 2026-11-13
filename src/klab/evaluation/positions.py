"""Held-out loss by position in the window: the context-extension curve.

A target at window index ``t`` is predicted from ``t + 1`` tokens of context.
Buckets double in width up to the panel's ``sequence_length``: the first covers
``0`` up to ``min(1024, the largest power of two below sequence_length)``, then
``1024-2048``, ``2048-4096`` and so on, the last ending at ``sequence_length``.
A context extension that works keeps the late buckets' loss at or below the
early ones'. Buckets only report the scored targets; they never change which
targets are scored or the panel's loss.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FIRST_BUCKET = 1024


def bucket_edges(sequence_length: int) -> list[int]:
    """``[0, first, 2 * first, ..., sequence_length]`` (see the module docstring)."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    first = FIRST_BUCKET
    while first > 1 and first >= sequence_length:
        first //= 2
    edges = [0]
    edge = first
    while edge < sequence_length:
        edges.append(edge)
        edge *= 2
    return [*edges, sequence_length]


class PositionLoss:
    """Summed NLL and target count at every window position of a panel."""

    def __init__(self, sequence_length: int) -> None:
        self.edges = bucket_edges(sequence_length)
        self.sums = np.zeros(sequence_length, dtype=np.float64)
        self.counts = np.zeros(sequence_length, dtype=np.int64)
        # Why the evaluator could not score single targets, when it could not.
        self.unavailable: str | None = None

    def add(self, nll: Any, targets: Any) -> None:
        """Add one batch: per-target NLL and targets, both ``[batch, positions]``."""
        import torch

        valid = targets != -100
        width = targets.shape[-1]
        if width > self.sums.size:
            raise ValueError("batch is wider than the panel's sequence_length")
        sums = torch.where(valid, nll.double(), 0.0).sum(dim=0)
        self.sums[:width] += sums.cpu().numpy()
        self.counts[:width] += valid.sum(dim=0).cpu().numpy()

    def buckets(self) -> list[dict[str, Any]]:
        """Per bucket: ``start``/``end`` positions, ``targets`` and mean ``loss``."""
        out = []
        for start, end in zip(self.edges, self.edges[1:], strict=False):
            targets = int(self.counts[start:end].sum())
            total = float(self.sums[start:end].sum())
            out.append(
                {
                    "start": start,
                    "end": end,
                    "targets": targets,
                    "loss": total / targets if targets else None,
                }
            )
        return out

    def report(self) -> dict[str, Any]:
        """``position_buckets``, or None with the reason they are unavailable."""
        if self.unavailable is not None:
            return {
                "position_buckets": None,
                "position_buckets_unavailable": self.unavailable,
            }
        return {"position_buckets": self.buckets()}
