"""Output-head cross-entropy that never holds the logits of a whole microbatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.nn import functional
from torch.utils.checkpoint import checkpoint


def chunked_output_head_cross_entropy(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    chunk_tokens: int,
    precision: Callable[[], Any],
) -> torch.Tensor:
    """Sum cross-entropy over ``chunk_tokens`` positions at a time.

    Each chunk is projected under the run's autocast context and reduced in
    FP32, as the unchunked head does. When gradients are needed a chunk's
    logits are dropped after its loss and rebuilt during backward, so the peak
    is one chunk's logits instead of ``[tokens, vocabulary]``.
    """

    def chunk_loss(
        chunk: torch.Tensor, labels: torch.Tensor, projection: torch.Tensor
    ) -> torch.Tensor:
        with precision():
            logits = functional.linear(chunk, projection)
        return functional.cross_entropy(
            logits.float(), labels, ignore_index=-100, reduction="sum"
        )

    flat_hidden = hidden.flatten(0, 1)
    flat_targets = targets.flatten()
    recompute = torch.is_grad_enabled() and (
        flat_hidden.requires_grad or weight.requires_grad
    )
    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    for start in range(0, flat_hidden.shape[0], chunk_tokens):
        chunk = flat_hidden[start : start + chunk_tokens]
        labels = flat_targets[start : start + chunk_tokens]
        if recompute:
            total = total + checkpoint(
                chunk_loss, chunk, labels, weight, use_reentrant=False
            )
        else:
            total = total + chunk_loss(chunk, labels, weight)
    return total


@torch.no_grad()
def chunked_output_head_token_losses(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    chunk_tokens: int,
    precision: Callable[[], Any],
) -> torch.Tensor:
    """Per-target cross-entropy ``[batch, positions]`` (0 where masked), chunked.

    Evaluation only: the same projection and FP32 reduction as
    :func:`chunked_output_head_cross_entropy`, kept per target.
    """
    flat_hidden = hidden.flatten(0, 1)
    flat_targets = targets.flatten()
    parts = []
    for start in range(0, flat_hidden.shape[0], chunk_tokens):
        with precision():
            logits = functional.linear(
                flat_hidden[start : start + chunk_tokens], weight
            )
        parts.append(
            functional.cross_entropy(
                logits.float(),
                flat_targets[start : start + chunk_tokens],
                ignore_index=-100,
                reduction="none",
            )
        )
    return torch.cat(parts).view_as(targets)
