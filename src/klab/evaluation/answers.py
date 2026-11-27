"""Evaluate bounded greedy responses against independently supplied answers."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from klab.config.models import StrictModel
from klab.evaluation.generation_request import generate_result
from klab.evaluation.inference import InferenceRun

_MAX_PANEL_BYTES = 4 * 1024**2
Label = Annotated[str, StringConstraints(min_length=1, max_length=128)]
Answer = Annotated[str, StringConstraints(min_length=1, max_length=1024)]
AnswerKind = Literal["integer", "yes_no", "text"]


def _normalize(kind: AnswerKind, text: str) -> str | None:
    line = text.strip()
    if not line or line.splitlines() != [line]:
        return None
    if kind == "integer":
        if not re.fullmatch(r"[+-]?[0-9]+", line):
            return None
        digits = line.lstrip("+-").lstrip("0")
        return ("-" if line.startswith("-") and digits else "") + (digits or "0")
    if kind == "yes_no":
        value = line.lower().removesuffix(".")
        return value if value in ("yes", "no") else None
    return line


class AnswerItem(StrictModel):
    id: Label
    category: Label
    prompt: Annotated[str, StringConstraints(min_length=1, max_length=8192)]
    kind: AnswerKind
    accepted: Annotated[list[Answer], Field(min_length=1, max_length=16)]

    @model_validator(mode="after")
    def valid_answers(self):
        if not self.prompt.strip():
            raise ValueError("answer prompt must have visible text")
        normalized = {_normalize(self.kind, value) for value in self.accepted}
        if None in normalized or len(normalized) != len(self.accepted):
            raise ValueError(
                "accepted answers must be valid and distinct after normalization"
            )
        return self


class AnswerPanel(StrictModel):
    schema_version: Literal[1]
    name: Label
    max_new_tokens: Annotated[int, Field(strict=True, ge=1, le=32)]
    items: Annotated[list[AnswerItem], Field(min_length=1, max_length=2000)]

    @model_validator(mode="after")
    def distinct_ids(self):
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("answer item IDs must be unique")
        return self


def read_answers(path: Path) -> tuple[AnswerPanel, str]:
    with path.open("rb") as stream:
        data = stream.read(_MAX_PANEL_BYTES + 1)
    if len(data) > _MAX_PANEL_BYTES:
        raise ValueError("answer declaration exceeds 4 MiB")
    return AnswerPanel.model_validate_json(data), hashlib.sha256(data).hexdigest()


def score_answers(loaded: InferenceRun, panel: AnswerPanel, digest: str) -> dict:
    results = []
    categories = defaultdict(list)
    kinds = defaultdict(list)
    for item in panel.items:
        try:
            response = generate_result(
                loaded.model,
                loaded.tokenizer,
                item.prompt,
                loaded.config.model.max_seq_len,
                panel.max_new_tokens,
                loaded.device,
                temperature=0,
                stop_sequences=("\n",),
                strict_context=True,
                engine=loaded.engine,
                accounting_label=f"answers:{loaded.identity['checkpoint_sha256']}:{digest}:{item.id}",
            )
        except ValueError as error:
            raise ValueError(f"answer item {item.id}: {error}") from error
        text = response.text[len(item.prompt) :]
        answer = (
            _normalize(item.kind, text) if response.finish_reason == "stop" else None
        )
        row = {
            "id": item.id,
            "category": item.category,
            "kind": item.kind,
            "accepted": item.accepted,
            "response": text,
            "normalized_answer": answer,
            "correct": answer is not None
            and answer in {_normalize(item.kind, value) for value in item.accepted},
            "finish_reason": response.finish_reason,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "forward_input_positions": response.forward_input_positions,
        }
        results.append(row)
        categories[item.category].append(row)
        kinds[item.kind].append(row)

    def summarize(rows):
        return {
            "items": len(rows),
            "correct": sum(row["correct"] for row in rows),
            "valid_form": sum(row["normalized_answer"] is not None for row in rows),
            "length_limited": sum(row["finish_reason"] == "length" for row in rows),
            "completion_tokens": sum(row["completion_tokens"] for row in rows),
            "forward_input_positions": sum(
                row["forward_input_positions"] for row in rows
            ),
        }

    return {
        "source": "standalone_answers",
        "identity": loaded.identity,
        "answers": {
            "name": panel.name,
            "sha256": digest,
            "protocol": "greedy_first_line_v1",
            "max_new_tokens": panel.max_new_tokens,
            "overall": summarize(results),
            "categories": {key: summarize(rows) for key, rows in categories.items()},
            "kinds": {key: summarize(rows) for key, rows in kinds.items()},
            "items": results,
        },
    }
