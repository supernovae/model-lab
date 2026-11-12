"""Plain-transcript chat over local generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from tokenizers import Tokenizer

from klab.chat_template import PLAIN, ChatTemplate

if TYPE_CHECKING:
    from klab.engines.mlx import MLXEngine


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError(f"unsupported chat role: {self.role}")
        if self.role == "user" and not self.content.strip():
            raise ValueError("user chat messages must not be blank")


def format_chat_prompt(
    history: Sequence[ChatMessage],
    message: str,
    *,
    system: str | None = None,
    template: ChatTemplate = PLAIN,
) -> str:
    """Render a transcript that ends where the assistant's reply begins."""
    return template.prompt(
        [(item.role, item.content) for item in history], message, system=system
    )


def _validate_complete_turns(history: Sequence[ChatMessage]) -> None:
    if len(history) % 2:
        raise ValueError("chat history must contain complete user/assistant turns")
    for index in range(0, len(history), 2):
        if history[index].role != "user" or history[index + 1].role != "assistant":
            raise ValueError("chat history must alternate user and assistant turns")


def prepare_chat_prompt(
    history: Sequence[ChatMessage],
    message: str,
    tokenizer: Tokenizer,
    max_seq_len: int,
    max_new_tokens: int,
    *,
    system: str | None = None,
    template: ChatTemplate = PLAIN,
) -> tuple[str, int]:
    """Return a completion-reserved chat prompt and dropped complete-turn count.

    History is removed only as whole oldest user/assistant pairs.  The system and
    current user message are never cropped: if those cannot fit, callers receive a
    clear error rather than an altered question.
    """
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must not be negative")
    budget = max_seq_len - max_new_tokens
    if budget < 0:
        raise ValueError("max_new_tokens exceeds max_seq_len")

    turns = list(history)
    _validate_complete_turns(turns)
    dropped = 0
    while True:
        prompt = format_chat_prompt(
            turns[dropped * 2 :], message, system=system, template=template
        )
        if len(tokenizer.encode(prompt, add_special_tokens=False).ids) <= budget:
            return prompt, dropped
        if dropped == len(turns) // 2:
            raise ValueError(
                "current chat turn and completion budget exceed max_seq_len"
            )
        dropped += 1


def assistant_reply(
    completion: str, prompt: str, template: ChatTemplate = PLAIN
) -> str:
    """Extract one assistant turn and stop where the template says a reply ends."""
    if not completion.startswith(prompt):
        raise ValueError("completion does not begin with its chat prompt")
    reply = completion.removeprefix(prompt)
    for marker in template.stop:
        reply = reply.split(marker, 1)[0]
    return reply.strip()


def chat_turn(
    model: Any,
    tokenizer: Tokenizer,
    history: Sequence[ChatMessage],
    message: str,
    max_seq_len: int,
    max_new_tokens: int,
    device: torch.device | str,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int = 0,
    engine: MLXEngine | None = None,
    template: ChatTemplate = PLAIN,
) -> tuple[str, str]:
    """Generate one bounded assistant turn and return its prompt and reply."""
    from klab.evaluation.generation import generate

    prompt, _ = prepare_chat_prompt(
        history,
        message,
        tokenizer,
        max_seq_len,
        max_new_tokens,
        system=system,
        template=template,
    )
    completion = generate(
        model,
        tokenizer,
        prompt,
        max_seq_len,
        max_new_tokens,
        device,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        stop_sequences=template.stop,
        stop_token_ids=template.stop_token_ids(tokenizer),
        strict_context=True,
        engine=engine,
    )
    return prompt, assistant_reply(completion, prompt, template)
