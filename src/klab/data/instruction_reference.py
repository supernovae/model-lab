"""Deterministic, synthetic instruction-format corpus for local reference training."""

from __future__ import annotations

import random
from collections.abc import Iterator

_NAMES = ("Ava", "Ben", "Cora", "Dax", "Eli", "Faye")
_COLORS = ("red", "blue", "green", "gold", "violet")
_OBJECTS = ("book", "kite", "cup", "ball", "map")
_WORDS = ("causal", "attention", "memory", "token", "sparse", "latent")


def _dialogue(user: str, assistant: str) -> str:
    return (
        "System: You are a concise local assistant.\n\n"
        f"User: {user}\n\nAssistant: {assistant}"
    )


def _document(seed: int, index: int) -> str:
    rng = random.Random(seed + index)
    kind = index % 6
    if kind == 0:
        left, right = rng.randrange(100), rng.randrange(100)
        return _dialogue(
            f"Reference {index}: what is {left} plus {right}?", str(left + right)
        )
    if kind == 1:
        word = rng.choice(_WORDS)
        return _dialogue(f"Reference {index}: reverse the word {word}.", word[::-1])
    if kind == 2:
        name, color, object_name = (
            rng.choice(_NAMES),
            rng.choice(_COLORS),
            rng.choice(_OBJECTS),
        )
        return _dialogue(
            f"Reference {index}: {name} has a {color} {object_name}. What color is the {object_name}?",
            f"The {object_name} is {color}.",
        )
    if kind == 3:
        return _dialogue(
            f"Reference {index}: explain causal attention in one sentence.",
            "Causal attention lets each position use earlier positions but not future ones.",
        )
    if kind == 4:
        word = rng.choice(_WORDS)
        return _dialogue(
            f"Reference {index}: use {word} in a short sentence.",
            f"This model studies {word} patterns.",
        )
    return _dialogue(
        f"Reference {index}: reply with a friendly greeting.",
        "Hello! How can I help?",
    )


def iter_instruction_reference(seed: int, split: str) -> Iterator[str]:
    """Yield deterministic train/validation-disjoint instruction dialogues."""
    if split not in {"train", "validation"}:
        raise ValueError(f"unknown split {split!r}")
    offset = 0 if split == "train" else 1_000_000
    index = 0
    while True:
        yield _document(seed, offset + index)
        index += 1
