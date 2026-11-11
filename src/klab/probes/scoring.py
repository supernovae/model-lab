"""One token-scoring path for every probe and for the lm-eval adapter.

All log-probabilities come from :func:`log_probs`, which never sees more than
the model's ``max_seq_len`` inputs. :func:`score_tokens` scores any span of
targets completely, windowing with maximal left context, so long continuations
are never silently truncated. Context/continuation boundaries follow
lm-evaluation-harness (``TemplateLM._encode_pair``): trailing context whitespace
moves to the continuation, the pair is encoded together and split at the
context's token count.

Held-out validation is scored by the native evaluator
(:meth:`klab.evaluation.inference.InferenceRun.evaluate`); probes only add
a :class:`ValidationStats` observer, so there is one forward pass and one
windowing/masking implementation for ``try`` and ``probe``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def encode(loaded: Any, text: str) -> list[int]:
    return list(loaded.tokenizer.encode(text, add_special_tokens=False).ids)


def eot(loaded: Any) -> int:
    declared = getattr(loaded, "eot_token_id", None)  # reference models
    if declared is not None:
        return int(declared)
    token = loaded.tokenizer.token_to_id("<eos>")
    return int(token) if token is not None else 0


def _byte_inputs(loaded: Any, ids: list[int]) -> dict[str, Any]:
    config = loaded.config.model
    if config.memory not in {"byte", "portable"}:
        return {}
    import torch

    from klab.evaluation.generation import _addresses_from_ids

    addresses = _addresses_from_ids(
        loaded.tokenizer, ids, config.memory_table_size, config.memory_ngram_size
    )
    return {
        "byte_addresses": torch.tensor(
            [addresses], dtype=torch.long, device=loaded.device
        )
    }


def log_probs(loaded: Any, ids: Sequence[int]) -> np.ndarray:
    """Log-softmax over the vocabulary at every position of IDS -> [T, V]."""
    import torch

    ids = list(ids)
    limit = loaded.config.model.max_seq_len
    if not ids or len(ids) > limit:
        raise ValueError(f"log_probs needs 1..{limit} tokens, got {len(ids)}")
    x = torch.tensor([ids], dtype=torch.long, device=loaded.device)
    with torch.inference_mode():
        logits = loaded.model(x, **_byte_inputs(loaded, ids))[0].float()
        return torch.log_softmax(logits, dim=-1).cpu().numpy().astype(np.float64)


def score_tokens(
    loaded: Any, tokens: Sequence[int], first_target: int
) -> tuple[np.ndarray, np.ndarray]:
    """Log-prob and greedy-hit of every target ``tokens[first_target:]``.

    Each target is scored exactly once with as much left context as fits in
    ``max_seq_len``; no target is dropped however long the span is.
    """
    tokens = list(tokens)
    if first_target < 1 or first_target > len(tokens):
        raise ValueError("first_target must leave at least one context token")
    length = loaded.config.model.max_seq_len
    values: list[np.ndarray] = []
    hits: list[np.ndarray] = []
    start = first_target
    while start < len(tokens):
        end = min(start + length, len(tokens))  # targets tokens[start:end]
        first = max(0, end - 1 - length)  # inputs tokens[first:end - 1]
        rows = log_probs(loaded, tokens[first : end - 1])[-(end - start) :]
        targets = np.asarray(tokens[start:end])
        values.append(rows[np.arange(len(targets)), targets])
        hits.append(rows.argmax(axis=-1) == targets)
        start = end
    if not values:
        return np.zeros(0), np.zeros(0, dtype=bool)
    return np.concatenate(values), np.concatenate(hits)


def split_pair(
    loaded: Any, context: str, continuation: str
) -> tuple[list[int], list[int]]:
    """Harness boundary handling: whitespace moves right, encode whole, split."""
    spaces = len(context) - len(context.rstrip())
    if spaces:
        continuation = context[-spaces:] + continuation
        context = context[:-spaces]
    if not context:
        return [eot(loaded)], encode(loaded, continuation)
    whole = encode(loaded, context + continuation)
    context_ids = encode(loaded, context)
    return context_ids, whole[len(context_ids) :]


def continuation_score(
    loaded: Any, context: str, continuation: str
) -> tuple[float, int, bool]:
    """(sum log P(continuation | context), scored tokens, greedy reproduces it)."""
    context_ids, continuation_ids = split_pair(loaded, context, continuation)
    if not continuation_ids:
        return 0.0, 0, True
    values, hits = score_tokens(
        loaded, [*context_ids, *continuation_ids], len(context_ids)
    )
    return float(values.sum()), len(continuation_ids), bool(hits.all())


def rolling_score(loaded: Any, text: str) -> float:
    """Every token scored exactly once after an end-of-text token."""
    tokens = [eot(loaded), *encode(loaded, text)]
    if len(tokens) < 2:
        return 0.0
    return float(score_tokens(loaded, tokens, 1)[0].sum())


def tie_credit(scores: Mapping[str, float], answer: str, rtol: float = 1e-9) -> float:
    """1/k when the answer is among k candidates tied at the top score, else 0.

    Answer-independent tie handling: a model that scores every candidate the
    same earns chance (1/k), never full credit.
    """
    best = max(scores.values())
    tolerance = rtol * max(1.0, abs(best))
    tied = [c for c, v in scores.items() if v >= best - tolerance]
    return 1.0 / len(tied) if answer in tied else 0.0


RANKING_PROTOCOL = "klab-ranking-v1: mean token log-prob, fractional ties"


def ranking_scores(
    loaded: Any, items: Sequence[Mapping[str, Any]]
) -> list[dict[str, float]]:
    """Per item, each candidate's mean token log-prob after the item's prefix."""
    out = []
    for item in items:
        scores = {}
        for candidate in item["candidates"]:
            total, count, _ = continuation_score(
                loaded, item["prefix"], f" {candidate}"
            )
            scores[candidate] = total / max(count, 1)
        out.append(scores)
    return out


def top_candidate(scores: Mapping[str, float], rtol: float = 1e-9) -> str:
    """The model's pick, or ``"tie: a / b"`` when several share the top score.

    Display only: credit comes from :func:`tie_credit`. A tie is never
    reported as the first candidate, so candidate order cannot look like a
    preference.
    """
    best = max(scores.values())
    tolerance = rtol * max(1.0, abs(best))
    tied = [c for c, v in scores.items() if v >= best - tolerance]
    return tied[0] if len(tied) == 1 else "tie: " + " / ".join(tied)


def constant_pick(items: Sequence[Mapping[str, Any]], picks: Sequence[str]) -> bool:
    """Every question with the same candidate set got the same pick.

    The signature of a context-free answer prior: the model ranks candidates
    the same way whatever the prompt says, as near-uniform untrained models do
    (their pick follows tiny prior differences, not candidate order). Accuracy
    is then exactly the share of items whose answer is that pick: chance when
    each candidate is the answer equally often. Needs a repeated candidate set.
    """
    by_set: dict[tuple[str, ...], set[str]] = {}
    for item, pick in zip(items, picks, strict=True):
        by_set.setdefault(tuple(item["candidates"]), set()).add(pick)
    repeated = len(items) > len(by_set)
    return repeated and all(len(found) == 1 for found in by_set.values())


def ranking_credit(loaded: Any, items: Sequence[Mapping[str, Any]]) -> list[float]:
    """Fractional credit per item; candidates ranked by mean token log-prob."""
    return [
        tie_credit(scores, item["answer"])
        for scores, item in zip(ranking_scores(loaded, items), items, strict=True)
    ]


class ValidationStats:
    """Observer for the native evaluator: per-window sums and top-1 confidence."""

    def __init__(self) -> None:
        self.sums: list[float] = []
        self.counts: list[int] = []
        self.confidences: list[np.ndarray] = []
        self.correct: list[np.ndarray] = []
        self.tokens = 0
        self._started = time.monotonic()

    def __call__(self, logits: Any, targets: Any) -> None:
        import torch

        log_p = torch.log_softmax(logits.float(), dim=-1)
        valid = targets != -100
        safe = targets.clamp(min=0)
        nll = -log_p.gather(-1, safe[..., None])[..., 0]
        for row in range(targets.shape[0]):
            mask = valid[row]
            self.sums.append(float(nll[row][mask].sum()))
            self.counts.append(int(mask.sum()))
        top = log_p.max(dim=-1)
        self.confidences.append(top.values.exp()[valid].cpu().numpy())
        self.correct.append((top.indices == safe)[valid].cpu().numpy())
        self.tokens += int(targets.numel())

    def result(self, native: Mapping[str, Any]) -> dict[str, Any]:
        elapsed = time.monotonic() - self._started
        correct = np.concatenate(self.correct) if self.correct else np.zeros(0)
        return {
            "loss": float(native["loss"]),
            "valid_targets": int(native["valid_targets"]),
            "window_sums": list(self.sums),
            "window_counts": list(self.counts),
            "windows": len(self.counts),
            "confidences": np.concatenate(self.confidences)
            if self.confidences
            else np.zeros(0),
            "correct": correct,
            "accuracy": float(correct.mean()) if correct.size else None,
            "ms_per_token": 1000.0 * elapsed / max(self.tokens, 1),
        }
