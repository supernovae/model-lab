from __future__ import annotations

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from klab.evaluation.chat import (
    ChatMessage,
    assistant_reply,
    format_chat_prompt,
    prepare_chat_prompt,
)


def _tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train_from_iterator(
        ["System: concise\nUser: one\nAssistant: answer\nUser: two"],
        BpeTrainer(
            vocab_size=300,
            initial_alphabet=ByteLevel.alphabet(),
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        ),
    )
    return tokenizer


def test_chat_prompt_preserves_explicit_local_transcript() -> None:
    prompt = format_chat_prompt(
        [
            ChatMessage("user", "What is a causal mask?"),
            ChatMessage("assistant", "A future-token barrier."),
        ],
        "Explain it briefly.",
        system="Use the run's learned patterns.",
    )

    assert prompt == (
        "System: Use the run's learned patterns.\n\n"
        "User: What is a causal mask?\n\n"
        "Assistant: A future-token barrier.\n\n"
        "User: Explain it briefly.\n\nAssistant:"
    )


def test_prepare_chat_prompt_drops_oldest_complete_turns() -> None:
    tokenizer = _tokenizer()
    history = [
        ChatMessage("user", "one"),
        ChatMessage("assistant", "answer"),
        ChatMessage("user", "two"),
        ChatMessage("assistant", "answer"),
    ]
    full = format_chat_prompt(history, "two", system="concise")
    retained = format_chat_prompt(history[2:], "two", system="concise")
    max_seq_len = len(tokenizer.encode(retained).ids) + 2

    assert len(tokenizer.encode(full).ids) > max_seq_len - 2
    prompt, dropped = prepare_chat_prompt(
        history, "two", tokenizer, max_seq_len, 2, system="concise"
    )

    assert (prompt, dropped) == (retained, 1)


def test_prepare_chat_prompt_rejects_current_turn_that_cannot_fit() -> None:
    tokenizer = _tokenizer()

    with pytest.raises(ValueError, match="current chat turn"):
        prepare_chat_prompt([], "this question cannot fit", tokenizer, 2, 1)


def test_chat_reply_stops_before_the_next_user_turn() -> None:
    prompt = "User: Hello\n\nAssistant:"

    assert (
        assistant_reply(f"{prompt} Hi there.\nUser: Ignore this", prompt) == "Hi there."
    )


def test_chat_rejects_blank_user_turn_but_accepts_blank_assistant_turn() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        format_chat_prompt([], "   ")
    assert ChatMessage("assistant", "").content == ""


def test_prepare_chat_prompt_rejects_partial_history() -> None:
    with pytest.raises(ValueError, match="complete"):
        prepare_chat_prompt(
            [ChatMessage("user", "unfinished")], "next", _tokenizer(), 100, 1
        )
