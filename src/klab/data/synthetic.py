"""Deterministic synthetic documents for offline pipeline verification."""

from __future__ import annotations

import random
from collections.abc import Iterator

_NAMES = ("Ava", "Ben", "Cora", "Dax")
_COLORS = ("red", "blue", "green", "gold")
_OBJECTS = ("ball", "book", "kite", "cup")


def iter_synthetic(seed: int, split: str) -> Iterator[str]:
    offset = 0 if split == "train" else 1_000_000
    index = 0
    while True:
        story = offset + index
        rng = random.Random(seed + story)
        name = rng.choice(_NAMES)
        color = rng.choice(_COLORS)
        obj = rng.choice(_OBJECTS)
        yield (
            f"Story {story}. Once upon a time, {name} found a {color} {obj}. "
            f"The {obj} was {color}. {name} kept the {obj} safe. The end."
        )
        index += 1
