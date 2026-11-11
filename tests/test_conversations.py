from __future__ import annotations

import json
from pathlib import Path

import pytest

from klab.data.conversations import iter_conversations


def _write_jsonl(path: Path, records: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_iter_conversations_streams_chat_format_with_system_and_multiple_turns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "licensed.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Second question"},
                    {"role": "assistant", "content": "Second answer"},
                ]
            }
        ],
    )

    assert list(iter_conversations(path)) == [
        (
            "System: Be concise.\n\n"
            "User: First question\n\nAssistant: First answer\n\n"
            "User: Second question\n\nAssistant: Second answer"
        )
    ]


@pytest.mark.parametrize(
    "record",
    [
        {"messages": []},
        {"messages": [{"role": "user", "content": "orphan"}]},
        {
            "messages": [
                {"role": "assistant", "content": "wrong"},
                {"role": "user", "content": "wrong"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "q", "extra": 1},
                {"role": "assistant", "content": "a"},
            ]
        },
        {"unexpected": []},
        {
            "messages": [
                {"role": [], "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        },
    ],
)
def test_iter_conversations_rejects_invalid_schema_and_turns(
    tmp_path: Path, record: object
) -> None:
    path = tmp_path / "invalid.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises(ValueError, match=r"invalid\.jsonl:1"):
        list(iter_conversations(path))


def test_iter_conversations_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    path.write_text('{"messages": [], "messages": []}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        list(iter_conversations(path))


def test_v2_assistant_only_renders_versioned_tool_transcript_and_spans(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tools.jsonl"
    _write_jsonl(
        path,
        [
            {
                "format_version": 2,
                "loss_mode": "assistant_only",
                "messages": [
                    {"role": "user", "content": "Search μ"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call-1", "name": "lookup", "arguments": {"q": "μ"}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call-1", "content": "found"},
                    {"role": "assistant", "content": "The result is μ."},
                ],
            }
        ],
    )

    from klab.data.conversations import iter_rendered_conversations

    rendered = next(iter_rendered_conversations(path))
    assert rendered.loss_mode == "assistant_only"
    assert (
        'ToolCalls[v1]: [{"arguments":{"q":"μ"},"id":"call-1","name":"lookup"}]'
        in rendered.text
    )
    assert (
        'ToolResult[v1]: {"content":"found","tool_call_id":"call-1"}' in rendered.text
    )
    assert "\n\nAssistant: The result is μ." in rendered.text
    supervised = {
        position
        for start, end in rendered.supervision_spans
        for position in range(start, end)
    }
    for excluded in ("Search μ", "found", "ToolCalls[v1]:"):
        start = rendered.text.index(excluded)
        assert supervised.isdisjoint(range(start, start + len(excluded)))
    start = rendered.text.index("The result is μ.")
    assert set(range(start, len(rendered.text))) <= supervised


@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "a", "name": "f", "arguments": {}}],
            },
            {"role": "tool", "tool_call_id": "wrong", "content": "r"},
        ],
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "a", "name": "f", "arguments": {}}],
            },
            {"role": "tool", "tool_call_id": "a", "content": "r"},
        ],
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "a", "name": "f", "arguments": {"x": float("nan")}}
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "r"},
            {"role": "assistant", "content": "done"},
        ],
    ],
)
def test_v2_rejects_incomplete_or_noncanonical_tool_turns(
    tmp_path: Path, messages: list[object]
) -> None:
    path = tmp_path / "invalid-v2.jsonl"
    _write_jsonl(
        path,
        [{"format_version": 2, "loss_mode": "assistant_only", "messages": messages}],
    )
    with pytest.raises(ValueError, match=r"invalid-v2\.jsonl:1"):
        list(iter_conversations(path))


def test_v2_rejects_float_format_version(tmp_path):
    path = tmp_path / "version.jsonl"
    _write_jsonl(
        path,
        [
            {
                "format_version": 2.0,
                "loss_mode": "assistant_only",
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
            }
        ],
    )
    with pytest.raises(ValueError):
        list(iter_conversations(path))
