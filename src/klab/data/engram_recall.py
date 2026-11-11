"""Deterministic associative-recall documents for dense-versus-Engram experiments."""

from __future__ import annotations

import random
from collections.abc import Iterator

_KEYS = (
    "amber",
    "birch",
    "coral",
    "dune",
    "ember",
    "frost",
    "grove",
    "harbor",
    "ivory",
    "jade",
    "kestrel",
    "linen",
    "meadow",
    "north",
    "orchid",
    "piper",
)
_VALUES = (
    "lumen",
    "orbit",
    "quartz",
    "ripple",
    "saffron",
    "thistle",
    "umber",
    "velvet",
    "willow",
    "xenon",
    "yarrow",
    "zephyr",
    "apricot",
    "bronze",
    "cinder",
    "dahlia",
)
_FILLERS = (
    "The archive is organized by short labels.",
    "Keep the record exact and concise.",
    "This entry belongs to a fixed reference table.",
    "The lookup must use the stated label only.",
)


def recall_case(seed: int, case_id: int) -> tuple[str, str]:
    """Return a deterministic prompt and single-token expected answer."""
    rng = random.Random(seed + case_id)
    key_index = case_id % len(_KEYS)
    key, value = _KEYS[key_index], _VALUES[key_index]
    filler = rng.choice(_FILLERS)
    return (
        (
            f"Memory record {case_id}. The code {key} maps to {value}. {filler} "
            f"Question: what does {key} map to?\nAnswer:"
        ),
        value,
    )


def iter_engram_recall(seed: int, split: str) -> Iterator[str]:
    """Yield train/validation-disjoint supervised recall documents."""
    if split not in {"train", "validation"}:
        raise ValueError(f"unknown split {split!r}")
    offset = 0 if split == "train" else 1_000_000
    index = 0
    while True:
        prompt, answer = recall_case(seed, offset + index)
        yield f"{prompt} {answer}"
        index += 1
