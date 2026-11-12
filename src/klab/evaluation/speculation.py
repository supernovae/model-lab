"""Draft-and-verify greedy decoding: drafters propose, the model decides.

A drafter guesses the next few tokens cheaply. The model then scores the whole
guess in one cached forward and keeps the longest prefix it would have produced
itself, so the completion is the model's own greedy completion and only the
number of model calls changes. ``PromptLookupSpeculation`` drafts from n-gram
repeats inside the running context and needs no second model or table;
``DraftModelSpeculation`` drafts with a smaller model that shares the tokenizer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


class Drafter(Protocol):
    """What greedy generation needs from a drafter."""

    model_calls: int
    drafted_tokens: int
    accepted_tokens: int

    def begin(self) -> None: ...

    def draft(self, ids: Sequence[int], limit: int) -> list[int]: ...


@dataclass
class PromptLookupSpeculation:
    """Draft the tokens that followed the latest earlier copy of the last n-gram.

    Counters accumulate over every generation that uses this object:
    ``model_calls`` cached forwards that verified a draft (prompt encoding and
    full-window re-encoding are not counted), ``drafted_tokens`` proposed and
    ``accepted_tokens`` confirmed by the model.
    """

    ngram: int = 3
    max_draft_tokens: int = 8
    model_calls: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    _following: dict[tuple[int, ...], int] = field(default_factory=dict, repr=False)
    _indexed: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        for name in ("ngram", "max_draft_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def begin(self) -> None:
        """Forget the previous request's context; counters are kept."""
        self._following.clear()
        self._indexed = 0

    def draft(self, ids: Sequence[int], limit: int) -> list[int]:
        """Return at most ``limit`` proposed continuation tokens for ``ids``."""
        # Index every n-gram that already has a successor, latest copy winning.
        for end in range(max(self._indexed, self.ngram), len(ids)):
            self._following[tuple(ids[end - self.ngram : end])] = end
        self._indexed = max(self._indexed, len(ids))
        limit = min(limit, self.max_draft_tokens)
        if limit <= 0 or len(ids) < self.ngram:
            return []
        start = self._following.get(tuple(ids[-self.ngram :]))
        return [] if start is None else list(ids[start : start + limit])

    @property
    def acceptance_rate(self) -> float:
        return (
            self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0
        )


class DraftModelSpeculation:
    """Draft with a smaller model's own greedy continuation.

    The draft model must share the target's tokenizer. It keeps its own KV
    cache across calls within one request: each call rewinds to where the
    previous call's context ended and feeds only the tokens added since, so
    a draft costs one forward per proposed token plus one for the new context.
    ``draft_model_calls`` counts those forwards; the other counters mean what
    they mean for prompt lookup.
    """

    def __init__(
        self,
        model: Any,
        device: torch.device,
        *,
        max_draft_tokens: int = 4,
        blocked_ids: Sequence[int] = (),
    ) -> None:
        if (
            isinstance(max_draft_tokens, bool)
            or not isinstance(max_draft_tokens, int)
            or max_draft_tokens <= 0
        ):
            raise ValueError("max_draft_tokens must be a positive integer")
        if not callable(getattr(model, "forward_cached", None)):
            raise TypeError("draft model does not expose incremental KV-cache decoding")
        capability = getattr(model, "incremental_cache_capability", None)
        if capability is not None and not capability.supported:
            raise ValueError(
                f"draft model cannot decode incrementally: {capability.reason}"
            )
        self.model = model
        self.device = device
        self.max_draft_tokens = max_draft_tokens
        self.capacity = int(model.config.max_seq_len)
        self.blocked_ids = tuple(blocked_ids)
        self.model_calls = 0
        self.drafted_tokens = 0
        self.accepted_tokens = 0
        self.draft_model_calls = 0
        self._cache: Any = None
        self._boundary: tuple[int, Any, dict[str, Any]] | None = None

    def begin(self) -> None:
        """Drop the previous request's cache; counters are kept."""
        self.model.eval()
        self._cache = None
        self._boundary = None

    def _forward(self, ids: Sequence[int], **options: Any) -> torch.Tensor:
        logits, self._cache = self.model.forward_cached(
            torch.tensor([list(ids)], device=self.device), **options
        )
        self.draft_model_calls += 1
        return logits[0, -1]

    def _greedy(self, logits: torch.Tensor) -> int:
        scores = logits.float().clone()
        for token_id in self.blocked_ids:
            if 0 <= token_id < scores.numel():
                scores[token_id] = -torch.inf
        return int(scores.argmax())

    def draft(self, ids: Sequence[int], limit: int) -> list[int]:
        """Return at most ``limit`` tokens the draft model would produce next.

        ``ids`` only grows between calls of one request, as generation calls it.
        """
        limit = min(limit, self.max_draft_tokens, self.capacity - len(ids))
        if limit <= 0:
            return []
        with torch.inference_mode():
            if self._boundary is None or len(ids) <= self._boundary[0]:
                logits = self._forward(ids, cache_capacity=self.capacity)
            else:
                length, history, histories = self._boundary
                self._cache.rewind(
                    length, memory_history=history, memory_histories=histories
                )
                logits = self._forward(ids[length:], cache=self._cache)
            cache = self._cache
            self._boundary = (
                len(ids),
                cache.memory_history,
                dict(cache.memory_histories),
            )
            proposed: list[int] = []
            while True:
                proposed.append(self._greedy(logits))
                if len(proposed) == limit:
                    return proposed
                logits = self._forward(proposed[-1:], cache=self._cache)
