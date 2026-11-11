"""Deterministic exact-budget selection for controlled generated-token comparisons."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

_MAX_STATES = 100_000
_PRIMARY = frozenset({"primary_source", "human_authored"})


def select_fraction(
    candidates: list[dict[str, Any]], generated_share: float, train_tokens: int
) -> set[str]:
    """Select complete examples with exact token/share budgets and family coverage.

    Infeasible budgets fail rather than silently changing the requested experiment.
    Same-source shape studies can compare releases using identical candidate snapshots.
    """
    if train_tokens <= 0 or not 0 <= generated_share <= 1:
        raise ValueError("invalid generated-token selection budget")
    generated_target = Fraction(str(generated_share)) * train_tokens
    if generated_target.denominator != 1:
        raise ValueError("generated share is not an integer number of tokens")
    generated_tokens = int(generated_target)
    primary_tokens = train_tokens - generated_tokens
    ids = [item["record_id"] for item in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate record identity")
    families = sorted(
        {family for item in candidates for family in item["source_family_ids"]}
    )
    if len(families) > 20:
        raise ValueError("too many families for exact token selection")
    family_bits = {family: 1 << index for index, family in enumerate(families)}
    partitions: list[list[tuple[str, int, int]]] = [[], []]
    for item in candidates:
        count = item["tokens"]
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("candidate token count must be positive")
        bits = 0
        for family in item["source_family_ids"]:
            bits |= family_bits[family]
        category = 0 if item["origin"] in _PRIMARY else 1
        partitions[category].append((item["record_id"], count, bits))

    def possibilities(
        entries: list[tuple[str, int, int]], target: int
    ) -> dict[int, dict[int, tuple[str, ...]]]:
        states: dict[tuple[int, int], tuple[str, ...]] = {(0, 0): ()}
        for identifier, count, bits in sorted(entries):
            for (total, covered), chosen in tuple(states.items()):
                next_total = total + count
                if next_total > target:
                    continue
                key = (next_total, covered | bits)
                replacement = (*chosen, identifier)
                if key not in states or replacement < states[key]:
                    states[key] = replacement
            if len(states) > _MAX_STATES:
                raise ValueError("exact fraction selection exceeds bounded state limit")
        return {
            mask: chosen for (total, mask), chosen in states.items() if total == target
        }

    primary = possibilities(partitions[0], primary_tokens)
    generated = possibilities(partitions[1], generated_tokens)
    required = (1 << len(families)) - 1
    matches = [
        left + right
        for left_mask, left in primary.items()
        for right_mask, right in generated.items()
        if left_mask | right_mask == required
    ]
    if not matches:
        raise ValueError(
            "exact token fraction with source-family coverage is infeasible"
        )
    return set(min(matches))
