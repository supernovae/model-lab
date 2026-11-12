"""Draft-and-verify greedy decoding: drafters propose, the model decides.

A drafter guesses the next few tokens cheaply. The model then scores the whole
guess in one cached forward and keeps the longest prefix it would have produced
itself, so the completion is the model's own greedy completion and only the
number of model calls changes. ``PromptLookupSpeculation`` drafts from n-gram
repeats inside the running context and needs no second model or table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


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
