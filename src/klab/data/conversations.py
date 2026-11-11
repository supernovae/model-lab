"""Strict streaming reader for licensed local chat corpora."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from klab.canonical import reject_duplicate_keys
from klab.evaluation.chat import ChatMessage, format_chat_prompt


@dataclass(frozen=True)
class RenderedConversation:
    """Rendered corpus document and character spans which receive chat loss."""

    text: str
    supervision_spans: tuple[tuple[int, int], ...]
    loss_mode: str


def _error(path: Path, line_number: int, detail: str) -> ValueError:
    return ValueError(f"{path}:{line_number}: {detail}")


def _nonblank_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _canonical_tool_calls(
    value: object, path: Path, line_number: int, index: int
) -> str:
    if not isinstance(value, list) or not value:
        raise _error(path, line_number, f"message {index} tool_calls must be nonempty")
    calls: list[dict[str, object]] = []
    for call_index, call in enumerate(value):
        if not isinstance(call, dict) or set(call) != {"id", "name", "arguments"}:
            raise _error(
                path,
                line_number,
                f"message {index} tool call {call_index} has invalid keys",
            )
        if not _nonblank_string(call["id"]) or not _nonblank_string(call["name"]):
            raise _error(
                path,
                line_number,
                f"message {index} tool call {call_index} needs id and name",
            )
        arguments = call["arguments"]
        if not isinstance(arguments, dict):
            raise _error(
                path,
                line_number,
                f"message {index} tool call {call_index} arguments must be an object",
            )
        try:
            encoded = json.dumps(
                call,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise _error(
                path,
                line_number,
                f"message {index} tool call {call_index} is not canonical JSON: {error}",
            ) from error
        # json.dumps accepts non-string mapping keys; canonical JSON does not.
        if json.loads(encoded, object_pairs_hook=reject_duplicate_keys) != call:
            raise _error(
                path,
                line_number,
                f"message {index} tool call {call_index} is not canonical JSON",
            )
        calls.append(call)
    ids = [str(call["id"]) for call in calls]
    if len(ids) != len(set(ids)):
        raise _error(path, line_number, f"message {index} has duplicate tool call IDs")
    return json.dumps(
        calls,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _v1_document(record: object, path: Path, line_number: int) -> RenderedConversation:
    if not isinstance(record, dict) or set(record) != {"messages"}:
        raise _error(path, line_number, "record must contain only messages")
    messages = record["messages"]
    if not isinstance(messages, list):
        raise _error(path, line_number, "messages must be an array")
    parsed: list[ChatMessage] = []
    system: str | None = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise _error(
                path, line_number, f"message {index} must contain only role and content"
            )
        role, content = message["role"], message["content"]
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise _error(path, line_number, f"message {index} has unsupported role")
        if not _nonblank_string(content):
            raise _error(
                path, line_number, f"message {index} content must not be blank"
            )
        if role == "system":
            if index != 0:
                raise _error(path, line_number, "system message is allowed only first")
            system = content
        else:
            parsed.append(ChatMessage(role, content))
    if len(parsed) < 2 or len(parsed) % 2:
        raise _error(
            path, line_number, "conversation requires complete user/assistant pairs"
        )
    for index in range(0, len(parsed), 2):
        if parsed[index].role != "user" or parsed[index + 1].role != "assistant":
            raise _error(
                path, line_number, "messages must alternate user then assistant"
            )
    history: list[ChatMessage] = []
    document = ""
    for index in range(0, len(parsed), 2):
        user, assistant = parsed[index : index + 2]
        prompt = format_chat_prompt(history, user.content, system=system)
        if document and not prompt.startswith(document):
            raise AssertionError("chat transcript rendering is not append-only")
        document = f"{prompt} {assistant.content.strip()}"
        history.extend((user, assistant))
    return RenderedConversation(document, ((0, len(document)),), "all_tokens")


def _v2_document(
    record: dict[str, object], path: Path, line_number: int
) -> RenderedConversation:
    if (
        set(record) != {"format_version", "loss_mode", "messages"}
        or type(record["format_version"]) is not int
        or record["format_version"] != 2
    ):
        raise _error(
            path,
            line_number,
            "v2 record must contain exactly format_version=2, loss_mode, and messages",
        )
    loss_mode = record["loss_mode"]
    if not isinstance(loss_mode, str) or loss_mode not in {
        "assistant_only",
        "all_tokens",
    }:
        raise _error(
            path, line_number, "v2 loss_mode must be assistant_only or all_tokens"
        )
    messages = record["messages"]
    if not isinstance(messages, list):
        raise _error(path, line_number, "messages must be an array")
    system: str | None = None
    rendered = ""
    spans: list[tuple[int, int]] = []
    pending: list[str] = []
    seen_call_ids: set[str] = set()
    expect_user = True
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise _error(
                path, line_number, f"message {index} must be an object with a role"
            )
        role = message["role"]
        if role == "system":
            if (
                index != 0
                or set(message) != {"role", "content"}
                or not _nonblank_string(message.get("content"))
            ):
                raise _error(
                    path,
                    line_number,
                    "system message is allowed only first with nonblank content",
                )
            system = str(message["content"])
            continue
        if role == "user":
            if (
                pending
                or not expect_user
                or set(message) != {"role", "content"}
                or not _nonblank_string(message.get("content"))
            ):
                raise _error(
                    path, line_number, f"message {index} is not a complete user turn"
                )
            prefix = (
                format_chat_prompt([], str(message["content"]), system=system)
                if not rendered
                else f"\n\nUser: {str(message['content']).strip()}\n\nAssistant:"
            )
            rendered += prefix
            expect_user = False
            continue
        if role == "assistant":
            if (
                expect_user
                or pending
                or not set(message).issubset({"role", "content", "tool_calls"})
            ):
                raise _error(
                    path,
                    line_number,
                    f"message {index} is not an expected assistant turn",
                )
            calls = message.get("tool_calls")
            content = message.get("content")
            if calls is None:
                if set(message) != {"role", "content"} or not _nonblank_string(content):
                    raise _error(
                        path,
                        line_number,
                        f"message {index} final assistant answer must have nonblank content",
                    )
                answer = str(content).strip()
                start = len(rendered)
                rendered += " " + answer
                spans.append((start, len(rendered)))
                expect_user = True
                continue
            if set(message) != {"role", "content", "tool_calls"} or (
                content is not None and not isinstance(content, str)
            ):
                raise _error(
                    path,
                    line_number,
                    f"message {index} tool-call assistant has invalid content",
                )
            encoded_calls = _canonical_tool_calls(calls, path, line_number, index)
            call_ids = [str(call["id"]) for call in calls]  # type: ignore[index]
            if seen_call_ids.intersection(call_ids):
                raise _error(
                    path, line_number, f"message {index} reuses a tool call ID"
                )
            seen_call_ids.update(call_ids)
            if isinstance(content, str) and content:
                start = len(rendered)
                rendered += " " + content
                spans.append((start, len(rendered)))
            rendered += "\nToolCalls[v1]:"
            start = len(rendered)
            rendered += " " + encoded_calls
            spans.append((start, len(rendered)))
            pending = call_ids
            continue
        if role == "tool":
            if (
                not pending
                or set(message) != {"role", "tool_call_id", "content"}
                or not _nonblank_string(message.get("tool_call_id"))
                or not _nonblank_string(message.get("content"))
            ):
                raise _error(
                    path,
                    line_number,
                    f"message {index} is not a valid pending tool result",
                )
            call_id = str(message["tool_call_id"])
            if call_id != pending[0]:
                raise _error(
                    path,
                    line_number,
                    f"message {index} tool results must match pending calls in order",
                )
            pending.pop(0)
            result = json.dumps(
                {"tool_call_id": call_id, "content": str(message["content"]).strip()},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            rendered += f"\nToolResult[v1]: {result}"
            if not pending:
                rendered += "\n\nAssistant:"
            continue
        raise _error(path, line_number, f"message {index} has unsupported role")
    if pending or not expect_user:
        raise _error(
            path,
            line_number,
            "conversation must end with a complete assistant final answer",
        )
    if not rendered:
        raise _error(
            path,
            line_number,
            "conversation requires at least one user/assistant exchange",
        )
    return RenderedConversation(
        rendered,
        tuple(spans) if loss_mode == "assistant_only" else ((0, len(rendered)),),
        str(loss_mode),
    )


def _conversation_document(
    record: object, path: Path, line_number: int
) -> RenderedConversation:
    if isinstance(record, dict) and "format_version" in record:
        return _v2_document(record, path, line_number)
    return _v1_document(record, path, line_number)


def iter_rendered_conversations(path: Path) -> Iterator[RenderedConversation]:
    """Yield validated local-chat records with explicit loss spans for v2."""
    with path.open("r", encoding="utf-8", newline="") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(
                    line,
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"nonfinite JSON value: {value}")
                    ),
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise _error(path, line_number, f"invalid JSON: {error}") from error
            yield _conversation_document(record, path, line_number)


def iter_conversations(path: Path) -> Iterator[str]:
    """Yield the historical text-only view; v1 rendering is byte-for-byte stable."""
    yield from (conversation.text for conversation in iter_rendered_conversations(path))
