"""Teacher-forced, byte-normalized likelihood of raw stories under native tokenizers."""

from __future__ import annotations

import math

import torch
from tokenizers import Tokenizer
from torch.nn import functional


@torch.inference_mode()
def score_story(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    text: str,
    *,
    device: torch.device,
    window_size: int = 128,
    stride: int = 64,
) -> dict[str, float | int]:
    """Score every document token and final EOS exactly once, with no padding.

    EOS is the start context (not a scored initial target). Each forward sees at
    most ``window_size`` input tokens. After the first window, the preceding
    ``window_size - stride`` inputs supply context for ``stride`` new targets;
    position indices restart at each window. The default is the prespecified
    128/64 cross-tokenizer control; 512/256 is the separate native-context loss.
    """
    byte_count = len(text.encode("utf-8"))
    if window_size <= 0 or stride <= 0 or stride > window_size:
        raise ValueError(
            "window_size and stride must define positive overlapping windows"
        )
    if byte_count == 0:
        raise ValueError("story must contain at least one UTF-8 byte")
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("tokenizer has no <eos> token")
    document_ids = tokenizer.encode(text, add_special_tokens=False).ids
    if tokenizer.decode(document_ids, skip_special_tokens=False) != text:
        raise ValueError("tokenizer does not reproduce the raw story")
    if eos in document_ids:
        raise ValueError("raw story encoding contains the EOS token")

    # One input per scored target: EOS -> first document ID -> ... -> last ID.
    inputs = [eos, *document_ids]
    targets = [*document_ids, eos]
    was_training = model.training
    nll = 0.0
    try:
        model.eval()
        start = 0
        while start < len(targets):
            context_start = max(0, start - (window_size - stride))
            stop = min(start + (window_size if start == 0 else stride), len(targets))
            x = torch.tensor(
                [inputs[context_start:stop]], dtype=torch.long, device=device
            )
            y = torch.tensor(targets[start:stop], dtype=torch.long, device=device)
            logits = model(x)
            if logits.ndim != 3 or logits.shape[:2] != x.shape:
                raise ValueError(
                    "model must return [batch, input length, vocabulary] logits"
                )
            selected = logits[:, start - context_start : stop - context_start]
            loss = functional.cross_entropy(
                selected.float().reshape(-1, selected.shape[-1]), y, reduction="sum"
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite text likelihood")
            nll += float(loss)
            start = stop
    finally:
        model.train(was_training)
    return {
        "nll_nats": nll,
        "nll_per_byte": nll / byte_count,
        "bits_per_byte": nll / (byte_count * math.log(2)),
        "utf8_bytes": byte_count,
        "token_count": len(document_ids),
        "target_count": len(targets),
    }
