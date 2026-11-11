"""Descriptive continuation-to-source overlap, independent of corpus selection.

Scores measure textual similarity only; they are not copyright or training gates.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

_WORD = re.compile(r"\w+", re.UNICODE)
_MAX_EDIT_CHARS = 2048


@dataclass(frozen=True)
class SourcePassage:
    source_id: str
    text: str


@dataclass(frozen=True)
class SourceMatch:
    source_id: str
    passage_sha256: str
    longest_exact_match: str
    longest_exact_match_chars: int
    character_overlap: float
    word_overlap: float
    ngram_overlap: float
    edit_similarity: float | None


@dataclass(frozen=True)
class MemorizationDiagnostic:
    normalization: str
    ngram_size: int
    best_source: SourceMatch
    sources: tuple[SourceMatch, ...]


def normalize_text(text: str) -> str:
    """NFC, Unicode casefold, and collapse Unicode whitespace to single spaces.

    Leading/trailing whitespace is removed; punctuation is retained. Scores and
    longest-match excerpts use this normalized text, not the original bytes.
    """
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _longest_common_substring(source: str, continuation: str) -> str:
    """Find an exact contiguous match in linear time using a suffix automaton."""
    if not source or not continuation:
        return ""
    edges: list[dict[str, int]] = [{}]
    suffix = [-1]
    length = [0]
    state = 0
    for char in source:
        current = len(edges)
        edges.append({})
        suffix.append(0)
        length.append(length[state] + 1)
        parent = state
        while parent >= 0 and char not in edges[parent]:
            edges[parent][char] = current
            parent = suffix[parent]
        if parent >= 0:
            target = edges[parent][char]
            if length[parent] + 1 == length[target]:
                suffix[current] = target
            else:
                clone = len(edges)
                edges.append(edges[target].copy())
                length.append(length[parent] + 1)
                suffix.append(suffix[target])
                while parent >= 0 and edges[parent].get(char) == target:
                    edges[parent][char] = clone
                    parent = suffix[parent]
                suffix[target] = suffix[current] = clone
        state = current

    state = matched = best = best_end = 0
    for index, char in enumerate(continuation):
        while state and char not in edges[state]:
            state = suffix[state]
            matched = length[state]
        if char in edges[state]:
            state = edges[state][char]
            matched += 1
        else:
            matched = 0
        if matched > best:
            best, best_end = matched, index + 1
    return continuation[best_end - best : best_end]


def _overlap(query: str | list[str], source: str | list[str]) -> float:
    """Multiset recall with the continuation as denominator."""
    if not query:
        return 0.0
    return sum((Counter(query) & Counter(source)).values()) / len(query)


def _edit_similarity(left: str, right: str) -> float:
    """Levenshtein similarity, normalized to the longer string length."""
    if len(left) < len(right):
        left, right = right, left
    if not left:
        return 1.0
    previous = list(range(len(right) + 1))
    for index, char in enumerate(left, start=1):
        current = [index]
        for column, other in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (char != other),
                )
            )
        previous = current
    return 1 - previous[-1] / len(left)


def diagnose_memorization(
    continuation: str,
    passages: Sequence[SourcePassage],
    *,
    ngram_size: int = 3,
    max_edit_chars: int = 512,
) -> MemorizationDiagnostic:
    """Compare a continuation against identified passages without policy decisions.

    Overlap metrics are continuation recall: character/word multiset overlap and
    presence of distinct word n-grams in the source. For strings longer than
    ``max_edit_chars`` on either side edit similarity is unavailable (``None``).
    A best source is chosen by longest match, then n-gram, word, character,
    edit similarity, and finally lexicographically smallest source ID.
    """
    if (
        not isinstance(ngram_size, int)
        or isinstance(ngram_size, bool)
        or not 1 <= ngram_size <= 10
    ):
        raise ValueError("ngram_size must be an integer between 1 and 10")
    if (
        not isinstance(max_edit_chars, int)
        or isinstance(max_edit_chars, bool)
        or not 1 <= max_edit_chars <= _MAX_EDIT_CHARS
    ):
        raise ValueError(f"max_edit_chars must be between 1 and {_MAX_EDIT_CHARS}")
    if not passages:
        raise ValueError("at least one source passage is required")
    identifiers = [passage.source_id for passage in passages]
    if any(not identifier for identifier in identifiers) or len(
        set(identifiers)
    ) != len(identifiers):
        raise ValueError("source IDs must be nonempty and unique")
    normalized = normalize_text(continuation)
    words = _WORD.findall(normalized)
    ngrams = {
        tuple(words[index : index + ngram_size])
        for index in range(len(words) - ngram_size + 1)
    }
    results = []
    for passage in sorted(passages, key=lambda item: item.source_id):
        source = normalize_text(passage.text)
        source_words = _WORD.findall(source)
        source_ngrams = {
            tuple(source_words[index : index + ngram_size])
            for index in range(len(source_words) - ngram_size + 1)
        }
        longest = _longest_common_substring(source, normalized)
        results.append(
            SourceMatch(
                source_id=passage.source_id,
                passage_sha256=hashlib.sha256(passage.text.encode("utf-8")).hexdigest(),
                longest_exact_match=longest,
                longest_exact_match_chars=len(longest),
                character_overlap=_overlap(normalized, source),
                word_overlap=_overlap(words, source_words),
                ngram_overlap=(
                    len(ngrams & source_ngrams) / len(ngrams) if ngrams else 0.0
                ),
                edit_similarity=(
                    _edit_similarity(normalized, source)
                    if len(normalized) <= max_edit_chars
                    and len(source) <= max_edit_chars
                    else None
                ),
            )
        )
    best = max(
        results,
        key=lambda item: (
            item.longest_exact_match_chars,
            item.ngram_overlap,
            item.word_overlap,
            item.character_overlap,
            item.edit_similarity if item.edit_similarity is not None else -1,
        ),
    )
    return MemorizationDiagnostic(
        normalization="NFC+casefold+whitespace-collapse-v1",
        ngram_size=ngram_size,
        best_source=best,
        sources=tuple(results),
    )
