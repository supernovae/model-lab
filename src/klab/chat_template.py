"""Chat templates: how a conversation becomes the text a model reads.

One template renders training conversations, builds the prompt for ``klab chat``
and the local API, names where a reply stops, and exports as the Jinja template
Hugging Face tokenizers carry. Lab ships the common formats; a config may also
declare its own::

    dataset:
      chat_template: chatml            # plain | chatml | llama3

    dataset:
      chat_template:
        name: bracketed
        system: "[SYS] {content}\n"
        user: "[USER] {content}\n"
        assistant: "[BOT] {content}\n"
        generation_prompt: "[BOT]"
        stop: ["\n[USER]"]

Each role is a pattern containing ``{content}`` exactly once. The assistant
pattern starts with ``generation_prompt``; everything after that prompt, the
closing text included, is what an assistant-only loss supervises. Text listed
under ``special_tokens`` becomes single tokens when a tokenizer is trained with
the same template.

This module does not import Torch.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SLOT = "{content}"


class ChatTemplate(BaseModel):
    """Role patterns, the prompt that opens a reply, and where a reply ends."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    system: str
    user: str
    assistant: str
    generation_prompt: str = Field(min_length=1)
    # Text placed between consecutive messages.
    separator: str = ""
    stop: tuple[str, ...] = Field(min_length=1)
    special_tokens: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_patterns(self) -> ChatTemplate:
        for role in ("system", "user", "assistant"):
            if getattr(self, role).count(_SLOT) != 1:
                raise ValueError(
                    f"chat template {role} pattern must contain {_SLOT} exactly once"
                )
        if not self.assistant.startswith(self.generation_prompt) or (
            _SLOT in self.generation_prompt
        ):
            raise ValueError(
                "chat template assistant pattern must start with generation_prompt"
            )
        if any(not stop for stop in self.stop):
            raise ValueError("chat template stop strings must not be empty")
        if len(set(self.special_tokens)) != len(self.special_tokens) or any(
            not token.strip() for token in self.special_tokens
        ):
            raise ValueError("chat template special_tokens must be unique and nonblank")
        return self

    @property
    def answer_lead(self) -> str:
        """Assistant text between the generation prompt and the reply."""
        return self.assistant.split(_SLOT)[0][len(self.generation_prompt) :]

    @property
    def answer_tail(self) -> str:
        """Assistant text that closes a reply."""
        return self.assistant.split(_SLOT)[1]

    def message(self, role: str, content: str) -> str:
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported chat role: {role}")
        return getattr(self, role).replace(_SLOT, content.strip())

    def prompt(
        self,
        history: Sequence[tuple[str, str]],
        message: str,
        *,
        system: str | None = None,
    ) -> str:
        """Render ``(role, content)`` turns and a new user message, ready for a reply."""
        if not message.strip():
            raise ValueError("chat message must not be blank")
        parts = [self.message("system", system)] if system else []
        parts.extend(self.message(role, content) for role, content in history)
        parts.extend((self.message("user", message), self.generation_prompt))
        return self.separator.join(parts)

    def stop_token_ids(self, tokenizer: Any) -> tuple[int, ...]:
        """IDs of stop strings the tokenizer holds as single tokens."""
        ids = (tokenizer.token_to_id(stop) for stop in self.stop)
        return tuple(token for token in ids if token is not None)

    def missing_special_tokens(self, tokenizer: Any) -> tuple[str, ...]:
        return tuple(
            token
            for token in self.special_tokens
            if tokenizer.token_to_id(token) is None
        )

    def to_jinja(self) -> str:
        """The same rendering as a Hugging Face ``chat_template`` string."""

        def literal(text: str) -> str:
            return "{{ " + json.dumps(text) + " }}" if text else ""

        branches = []
        for index, role in enumerate(("system", "user", "assistant")):
            before, after = getattr(self, role).split(_SLOT)
            keyword = "if" if index == 0 else "elif"
            branches.append(
                f"{{%- {keyword} message['role'] == '{role}' -%}}"
                f"{literal(before)}{{{{ message['content'] | trim }}}}{literal(after)}"
            )
        return (
            "{%- for message in messages -%}"
            + (
                "{%- if not loop.first -%}" + literal(self.separator) + "{%- endif -%}"
                if self.separator
                else ""
            )
            + "".join(branches)
            + "{%- endif -%}{%- endfor -%}"
            "{%- if add_generation_prompt -%}"
            + literal(self.separator + self.generation_prompt)
            + "{%- endif -%}"
        )


BUILTIN: dict[str, ChatTemplate] = {
    template.name: template
    for template in (
        # Readable text with no added tokens; works with any tokenizer.
        ChatTemplate(
            name="plain",
            system="System: {content}",
            user="User: {content}",
            assistant="Assistant: {content}",
            generation_prompt="Assistant:",
            separator="\n\n",
            stop=("\nUser:", "\nSystem:", "\nAssistant:"),
        ),
        # ChatML, as used by Qwen, SmolLM and many fine-tunes.
        ChatTemplate(
            name="chatml",
            system="<|im_start|>system\n{content}<|im_end|>\n",
            user="<|im_start|>user\n{content}<|im_end|>\n",
            assistant="<|im_start|>assistant\n{content}<|im_end|>\n",
            generation_prompt="<|im_start|>assistant\n",
            stop=("<|im_end|>",),
            special_tokens=("<|im_start|>", "<|im_end|>"),
        ),
        # Llama 3 header format.
        ChatTemplate(
            name="llama3",
            system="<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
            user="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
            assistant=(
                "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
            ),
            generation_prompt="<|start_header_id|>assistant<|end_header_id|>\n\n",
            stop=("<|eot_id|>",),
            special_tokens=("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"),
        ),
    )
}
PLAIN = BUILTIN["plain"]


def resolve(value: str | ChatTemplate | dict[str, Any] | None) -> ChatTemplate:
    """Return the template for a built-in name, a mapping or a template."""
    if value is None:
        return PLAIN
    if isinstance(value, ChatTemplate):
        return value
    if isinstance(value, str):
        if value not in BUILTIN:
            raise ValueError(
                f"unknown chat template {value!r}: Lab ships "
                f"{', '.join(BUILTIN)}; declare any other as a mapping"
            )
        return BUILTIN[value]
    template = ChatTemplate.model_validate(value)
    if template.name in BUILTIN and template != BUILTIN[template.name]:
        raise ValueError(
            f"chat template name {template.name!r} is built in; choose another name"
        )
    return template
