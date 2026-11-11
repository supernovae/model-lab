"""Held-out next-token evaluation with exact target accounting."""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import numpy as np
import torch
from torch.nn import functional

from klab.data.allocation import OWNER_HYBRID, OWNER_LEXICAL
from klab.data.packing import TokenBlockDataset
from klab.engram.semantic import SemanticQueryBatch


class NonFiniteLoss(ValueError, FloatingPointError):
    """Native evaluation produced a NaN/inf loss (a numerical failure)."""


def _device_rng_state(device: torch.device) -> torch.Tensor | None:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
    ):
        return torch.mps.get_rng_state()
    if device.type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu.get_rng_state(device)  # type: ignore[attr-defined]
    return None


def _restore_device_rng_state(device: torch.device, state: torch.Tensor | None) -> None:
    if state is None:
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    elif (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.set_rng_state(state, device)  # type: ignore[attr-defined]


def evaluate(
    model: torch.nn.Module,
    dataset: TokenBlockDataset,
    *,
    batch_size: int,
    max_batches: int,
    device: torch.device,
    observer: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    before_batch: Callable[[int], None] | None = None,
    progress: Callable[[float, int, int], None] | None = None,
) -> dict[str, float | int | str | None]:
    """Evaluate no more than ``max_batches`` and report exactly scored labels.

    ``observer(logits, targets)`` sees every scored batch (e.g. the probe
    battery's per-window and calibration statistics) without a second pass.
    ``before_batch`` receives forward-input positions before each model call;
    ``progress`` receives cumulative NLL, valid targets and completed batches.
    """
    if batch_size <= 0 or max_batches <= 0:
        raise ValueError("batch_size and max_batches must be positive")
    was_training = model.training
    cpu_rng = torch.get_rng_state()
    device_rng = _device_rng_state(device)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    total = 0.0
    count = 0
    batches = 0
    try:
        model.eval()
        with torch.inference_mode():
            limit = min(len(dataset), batch_size * max_batches)
            for start in range(0, limit, batch_size):
                records = [
                    dataset.numpy_microblock(i)
                    for i in range(start, min(start + batch_size, limit))
                ]
                if not records:
                    continue
                inputs, targets, addresses, owners, queries, masks = zip(
                    *records, strict=True
                )
                x = torch.from_numpy(np.stack(inputs)).to(device)
                y = torch.from_numpy(np.stack(targets)).to(device)
                byte_addresses = (
                    None
                    if addresses[0] is None
                    else torch.from_numpy(np.stack(addresses)).to(device)
                )
                memory_mask = None
                if owners[0] is not None:
                    owner_batch = np.stack(owners)
                    if np.any(owner_batch > OWNER_HYBRID):
                        raise ValueError(
                            "evaluation owner sidecar contains unknown codes"
                        )
                    memory_mask = torch.from_numpy(
                        (owner_batch == OWNER_LEXICAL) | (owner_batch == OWNER_HYBRID)
                    ).to(device)
                semantic_queries = None
                if queries[0] is not None or masks[0] is not None:
                    memories = getattr(model, "semantic_memories", None)
                    if (
                        not memories
                        or len(memories) != 1
                        or queries[0] is None
                        or masks[0] is None
                    ):
                        raise ValueError(
                            "evaluation semantic queries require one verified attached pack"
                        )
                    semantic_queries = SemanticQueryBatch(
                        next(iter(memories.values())).retriever.key_encoder,
                        torch.from_numpy(np.stack(queries)).to(device),
                        torch.from_numpy(np.stack(masks)).to(device),
                    )
                model_inputs: dict[str, object] = {}
                if byte_addresses is not None:
                    model_inputs["byte_addresses"] = byte_addresses
                if semantic_queries is not None:
                    model_inputs["semantic_queries"] = semantic_queries
                if memory_mask is not None:
                    model_inputs["memory_mask"] = memory_mask
                if before_batch is not None:
                    before_batch(x.numel())
                logits = model(x, **model_inputs)
                loss = functional.cross_entropy(
                    logits.float().flatten(0, 1),
                    y.flatten(),
                    ignore_index=-100,
                    reduction="sum",
                )
                if not torch.isfinite(loss):
                    raise NonFiniteLoss("nonfinite validation loss")
                if observer is not None:
                    observer(logits, y)
                valid = int((y != -100).sum())
                if valid:
                    total += float(loss)
                    count += valid
                batches += 1
                if progress is not None:
                    progress(total, count, batches)
        if count == 0:
            raise ValueError("validation contains no valid labels")
        loss = total / count
        perplexity: float | None
        if loss > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
            perplexity = None
            reason = "loss exceeds finite exp range"
        else:
            perplexity = math.exp(loss)
            reason = None
        return {
            "loss": loss,
            "perplexity": perplexity,
            "perplexity_unavailable_reason": reason,
            "valid_targets": count,
            "batches": batches,
        }
    finally:
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        _restore_device_rng_state(device, device_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
