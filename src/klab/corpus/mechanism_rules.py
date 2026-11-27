"""A closed integer transfer system for objective procedural experiments."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from klab.canonical import canonical_digest
from klab.config.models import StrictModel
from klab.corpus.relation_sources import Digest, Identifier, Split

Amount = Annotated[int, Field(strict=True, ge=0, le=10**9)]
Pair = tuple[Amount, Amount]
Direction = Literal[0, 1]
Violation = Literal["request_limit", "source_minimum", "destination_capacity"]


class TransferRule(StrictModel):
    kind: Literal["bounded_transfer"]
    capacity: Pair
    minimum: Pair
    request_limit: Annotated[int, Field(strict=True, ge=1, le=10**6)]

    @model_validator(mode="after")
    def bounds(self):
        if any(
            low > high for low, high in zip(self.minimum, self.capacity, strict=True)
        ):
            raise ValueError("minimum exceeds capacity")
        return self


class World(StrictModel):
    id: Identifier
    family: Identifier
    split: Split
    rule: TransferRule
    initial: Pair
    steps: Annotated[int, Field(strict=True, ge=1, le=64)]
    prose: Annotated[str, Field(min_length=1, max_length=32768)] | None = None

    @model_validator(mode="after")
    def initial_bounds(self):
        if any(
            not low <= value <= high
            for low, value, high in zip(
                self.rule.minimum, self.initial, self.rule.capacity, strict=True
            )
        ):
            raise ValueError("initial state is outside its rule bounds")
        if self.prose is not None and not self.prose.strip():
            raise ValueError("world prose must contain non-whitespace text")
        return self


class Transfer(StrictModel):
    amount: Amount
    after: Pair
    violations: tuple[Violation, ...]


class Observation(StrictModel):
    step: int
    before: Pair
    source: Direction
    available: Amount
    headroom: Amount
    legal_maximum: Amount
    actual: Transfer
    counterfactual: Transfer


class ObservedWorld(StrictModel):
    world: World
    mechanism: Digest
    source: Identifier
    line: int
    observations: tuple[Observation, ...]


def _apply(rule, before, source, amount, available, headroom):
    violations = tuple(
        name
        for name, failed in (
            ("request_limit", amount > rule.request_limit),
            ("source_minimum", amount > available),
            ("destination_capacity", amount > headroom),
        )
        if failed
    )
    after = list(before)
    if not violations:
        after[source] -= amount
        after[1 - source] += amount
    return Transfer(amount=amount, after=tuple(after), violations=violations)


def simulate_world(world: World, seed: int, source: str, line: int) -> ObservedWorld:
    """Replay a seeded action trace with conservation, exact constraints and a boundary foil."""
    before = world.initial
    rule = world.rule
    observations = []
    for step in range(world.steps):
        random_word = int(canonical_digest([seed, world.id, step]), 16)
        sender = random_word & 1
        requested = 1 + (random_word >> 1) % rule.request_limit
        available = before[sender] - rule.minimum[sender]
        headroom = rule.capacity[1 - sender] - before[1 - sender]
        legal = min(available, headroom, rule.request_limit)
        actual = _apply(rule, before, sender, requested, available, headroom)
        foil = legal if actual.violations else legal + 1
        alternative = _apply(rule, before, sender, foil, available, headroom)
        observations.append(
            Observation(
                step=step,
                before=before,
                source=sender,
                available=available,
                headroom=headroom,
                legal_maximum=legal,
                actual=actual,
                counterfactual=alternative,
            )
        )
        before = actual.after
    return ObservedWorld(
        world=world,
        mechanism=canonical_digest(rule.model_dump(mode="json")),
        source=source,
        line=line,
        observations=tuple(observations),
    )
