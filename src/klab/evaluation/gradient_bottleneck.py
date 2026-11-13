"""How much of the logit gradient the output head lets through.

The output head maps a hidden state to the vocabulary, ``logits = W h``. The
loss gradient for the logits, ``G = softmax(logits) - onehot(target)``, has one
entry per vocabulary token, but the gradient the rest of the network receives,
``G W``, keeps only the part of ``G`` in the column space of ``W``: at most as
many directions as the head's rank, out of the vocabulary's. The rest, the
projection of ``G`` onto the kernel of ``Wᵀ``, is lost (Godey and Artzi, "Lost
in Backpropagation: The LM Head is a Gradient Bottleneck", arXiv 2603.10145,
section 3.3).

Over every scored held-out target, this reports the share of the Frobenius
norm of ``G`` that is lost (``kernel_norm_fraction``) and the cosine between
``G`` and what is kept (``kept_cosine``). A gradient pointing in a random
direction would keep a ``rank / vocab_size`` share of its squared norm
(``random_kept_squared_fraction``); a head that keeps much more than that
aligns its column space with the errors it makes.
"""

from __future__ import annotations

import math
from typing import Any

import torch

# Rows of the logit gradient handled at once; bounds the extra memory.
_ROWS = 1024


class GradientBottleneck:
    """An evaluation observer: ``observer(logits, targets)`` for each batch."""

    def __init__(self, model: Any) -> None:
        weight = model.output.weight.detach().to("cpu", torch.float64)
        vocab, width = weight.shape
        basis, singular, _ = torch.linalg.svd(weight, full_matrices=False)
        tolerance = singular.max() * max(vocab, width) * torch.finfo(torch.float32).eps
        rank = int((singular > tolerance).sum())
        self.vocab_size = vocab
        self.rank = rank
        self.basis = basis[:, :rank].to(model.output.weight.device, torch.float32)
        self.total = 0.0
        self.kept = 0.0
        self.targets = 0

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        with torch.no_grad():
            flat = logits.reshape(-1, logits.shape[-1])
            labels = targets.reshape(-1)
            valid = labels != -100
            flat, labels = flat[valid], labels[valid]
            for start in range(0, len(labels), _ROWS):
                rows = torch.softmax(flat[start : start + _ROWS].float(), dim=-1)
                rows[torch.arange(len(rows)), labels[start : start + _ROWS]] -= 1.0
                self.total += float(rows.double().square().sum())
                self.kept += float((rows @ self.basis).double().square().sum())
            self.targets += len(labels)

    def result(self) -> dict[str, Any]:
        if not self.targets or not self.total:
            raise ValueError("gradient bottleneck needs scored targets")
        kept = min(1.0, self.kept / self.total)
        return {
            "targets": self.targets,
            "vocab_size": self.vocab_size,
            "head_rank": self.rank,
            "kept_squared_fraction": kept,
            "kept_cosine": math.sqrt(kept),
            "kernel_norm_fraction": math.sqrt(1.0 - kept),
            "random_kept_squared_fraction": self.rank / self.vocab_size,
        }


def observers(*callables: Any) -> Any:
    """One observer that calls each given observer (None entries skipped)."""
    present = [call for call in callables if call is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]

    def observe(logits: torch.Tensor, targets: torch.Tensor) -> None:
        for call in present:
            call(logits, targets)

    return observe
