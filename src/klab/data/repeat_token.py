"""SpamLang: documents that repeat one symbol, drawn from the tokenizer's words.

Godey and Artzi ("Lost in Backpropagation", arXiv 2603.10145, section 3.2)
train on sequences that each repeat one uniformly drawn symbol from a
vocabulary of V. Any model can learn it in principle, so a failure to learn it
is an optimization failure, which they attribute to the output head.

Here the symbols are the tokenizer's own whole-word tokens: a space followed by
letters, which the tokenizer encodes back to exactly that one token. The
pre-tokenizer splits repeated words apart, so a document of one symbol repeated
tokenizes to that token repeated, and the task runs on the ordinary text
pipeline. The symbol count is somewhat below the vocabulary size (no bytes,
special tokens or word pieces); :func:`symbols` reports it. A tokenizer whose
vocabulary is exactly the symbols would match the paper exactly; see the data
guide.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator
from typing import Any

_WORD = re.compile(r" [A-Za-z]+")


def symbols(tokenizer: Any) -> list[str]:
    """The tokenizer's whole-word tokens that encode to themselves, in ID order."""
    found = []
    for token in range(tokenizer.get_vocab_size()):
        text = tokenizer.decode([token], skip_special_tokens=False)
        if _WORD.fullmatch(text) and tokenizer.encode(
            text, add_special_tokens=False
        ).ids == [token]:
            found.append(text)
    return found


def iter_repeat_token(
    tokenizer: Any, seed: int, split: str, length: int
) -> Iterator[str]:
    """Documents of LENGTH tokens, each one uniformly drawn symbol repeated."""
    words = symbols(tokenizer)
    if len(words) < 2:
        raise ValueError(
            "repeat_token needs a tokenizer with at least two whole-word tokens"
        )
    offset = 0 if split == "train" else 1_000_000_000
    for index in range(2**62):
        word = random.Random(f"{seed}:{offset + index}").choice(words)
        yield word * length
