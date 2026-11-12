"""Finite chat-native alias recall corpora and held-out evaluation cases.

The train corpus deliberately cycles only its finite training documents.  Evaluation
uses the same learned aliases with wording that is absent from training, so it is a
held-out phrasing check rather than a record-number split.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from klab.evaluation.chat import ChatMessage, format_chat_prompt

_ALIASES = (
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
    "quill",
    "raven",
    "sable",
    "thrush",
    "umber",
    "violet",
    "willow",
    "xenon",
)
_VALUES = (
    "lumen",
    "orbit",
    "quartz",
    "ripple",
    "saffron",
    "thistle",
    "velvet",
    "yarrow",
    "zephyr",
    "apricot",
    "bronze",
    "cinder",
    "dahlia",
    "fable",
    "garnet",
    "helium",
    "indigo",
    "jasper",
    "kelp",
    "lilac",
    "mango",
    "nectar",
    "onyx",
    "pearl",
)
_MAPPING = dict(zip(_ALIASES, _VALUES, strict=True))
_SYSTEM = "Answer the requested alias with only its value."
_TRAIN_REQUESTS = (
    "What value belongs to the alias {alias}?",
    "Please recall the value for {alias}.",
    "Give the stored value associated with {alias}.",
)
_EVALUATION_REQUESTS = (
    "Which value should I return when I see {alias}?",
    "Complete this alias lookup: {alias} becomes what value?",
)


@dataclass(frozen=True)
class ChatRecallCase:
    """A plain transcript case with a single exact assistant response."""

    identifier: str
    prompt: str
    expected: str
    kind: str


def _prompt(history: tuple[ChatMessage, ...], message: str) -> str:
    return format_chat_prompt(history, message, system=_SYSTEM)


def _training_cases() -> tuple[ChatRecallCase, ...]:
    cases: list[ChatRecallCase] = []
    for alias, value in _MAPPING.items():
        for template_index, template in enumerate(_TRAIN_REQUESTS):
            # Include both single-turn and real multi-turn transcripts.  The answer
            # has the same leading space used by the chat training convention.
            history = (
                ()
                if template_index < 2
                else (
                    ChatMessage("user", "I will ask about one stored alias."),
                    ChatMessage("assistant", "I am ready."),
                )
            )
            cases.append(
                ChatRecallCase(
                    f"train-{alias}-{template_index}",
                    _prompt(history, template.format(alias=alias)),
                    value,
                    "alias_recall",
                )
            )
    return tuple(cases)


def _evaluation_cases(split: str) -> tuple[ChatRecallCase, ...]:
    if split not in {"validation", "test"}:
        raise ValueError(f"unknown held-out split {split!r}")
    # Alternate aliases by split.  They have all appeared in training, but their
    # evaluation wording and prompt combinations have not.
    pairs = tuple(_MAPPING.items())
    selected = pairs[::2] if split == "validation" else pairs[1::2]
    cases: list[ChatRecallCase] = []
    for index, (alias, value) in enumerate(selected):
        template = _EVALUATION_REQUESTS[index % len(_EVALUATION_REQUESTS)]
        history = (
            ()
            if index % 2 == 0
            else (
                ChatMessage("user", "Use the alias table you learned."),
                ChatMessage("assistant", "Understood."),
            )
        )
        cases.append(
            ChatRecallCase(
                f"{split}-alias-{alias}",
                _prompt(history, template.format(alias=alias)),
                value,
                "alias_recall",
            )
        )
    return tuple(cases)


def cases(split: str) -> tuple[ChatRecallCase, ...]:
    """Return finite, deterministic cases for ``train``, ``validation``, or ``test``."""
    if split == "train":
        return _training_cases()
    return _evaluation_cases(split)


def iter_chat_recall(seed: int, split: str) -> Iterator[str]:
    """Yield documents; only training is cyclic and no seed changes membership."""
    del seed
    if split == "train":
        train_cases = cases("train")
        while True:
            for case in train_cases:
                yield f"{case.prompt} {case.expected}"
        return
    if split == "validation":
        for case in cases("validation"):
            yield f"{case.prompt} {case.expected}"
        return
    raise ValueError(f"unknown split {split!r}")
