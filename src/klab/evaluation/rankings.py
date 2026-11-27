"""Score declarative answer choices on the existing token-scoring path."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator

from klab.config.models import StrictModel

_MAX_PANEL_BYTES = 4 * 1024**2
Label = Annotated[str, StringConstraints(min_length=1, max_length=128)]
Choice = Annotated[str, StringConstraints(min_length=1, max_length=1024)]
Prompt = Annotated[str, StringConstraints(min_length=1, max_length=8192)]


class RankingItem(StrictModel):
    id: Label
    category: Label
    prefix: Prompt
    candidates: Annotated[list[Choice], Field(min_length=2, max_length=16)]
    answer: Choice

    @model_validator(mode="after")
    def valid_choices(self):
        if not self.prefix.strip():
            raise ValueError("ranking prefix must have visible text")
        if any(value != value.strip() for value in self.candidates):
            raise ValueError("ranking choices must have no surrounding whitespace")
        if self.answer not in self.candidates or len(set(self.candidates)) != len(
            self.candidates
        ):
            raise ValueError("ranking choices must be distinct and include the answer")
        return self


class RankingPair(StrictModel):
    id: Label
    left: Label
    right: Label
    question: Prompt


class RankingPanel(StrictModel):
    schema_version: Literal[1]
    name: Label
    items: Annotated[list[RankingItem], Field(min_length=1, max_length=2000)]
    pairs: Annotated[list[RankingPair], Field(max_length=1000)] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_items_and_pairs(self):
        by_id = {item.id: item for item in self.items}
        if len(by_id) != len(self.items):
            raise ValueError("ranking item IDs must be unique")
        if len({pair.id for pair in self.pairs}) != len(self.pairs):
            raise ValueError("ranking pair IDs must be unique")
        paired_items = set()
        for pair in self.pairs:
            if (
                pair.left == pair.right
                or pair.left not in by_id
                or pair.right not in by_id
                or pair.left in paired_items
                or pair.right in paired_items
            ):
                raise ValueError(f"ranking pair {pair.id} needs two unused item IDs")
            left, right = by_id[pair.left], by_id[pair.right]
            if (
                left.category != right.category
                or left.candidates != right.candidates
                or left.answer == right.answer
            ):
                raise ValueError(
                    f"ranking pair {pair.id} needs the same category and ordered choices with different answers"
                )
            suffix = "\n" + pair.question
            if (
                not pair.question.strip()
                or any(not item.prefix.endswith(suffix) for item in (left, right))
                or any(
                    not item.prefix[: -len(suffix)].strip() for item in (left, right)
                )
                or left.prefix[: -len(suffix)].strip()
                == right.prefix[: -len(suffix)].strip()
            ):
                raise ValueError(
                    f"ranking pair {pair.id} needs distinct contexts ending in the same declared question line"
                )
            paired_items.update((pair.left, pair.right))
        return self


def read_ranking(path: Path) -> tuple[RankingPanel, str]:
    with path.open("rb") as stream:
        raw = stream.read(_MAX_PANEL_BYTES + 1)
    if len(raw) > _MAX_PANEL_BYTES:
        raise ValueError("ranking declaration exceeds 4 MiB")
    return RankingPanel.model_validate_json(raw), hashlib.sha256(raw).hexdigest()


def _paired_results(pairs, rows):
    by_id = {row["id"]: row for row in rows}
    groups = defaultdict(list)
    results = []
    for pair in pairs:
        left, right = by_id[pair.left], by_id[pair.right]
        a, b = left["answer"], right["answer"]
        ls, rs = left["scores"], right["scores"]
        row = {
            "id": pair.id,
            "category": left["category"],
            "left": pair.left,
            "right": pair.right,
            "joint_credit": left["credit"] * right["credit"],
            "chance": left["chance"] * right["chance"],
            "answer_margin": ((ls[a] - ls[b]) + (rs[b] - rs[a])) / 2,
        }
        groups[row["category"]].append(row)
        results.append(row)

    def summarize(group):
        count = len(group)
        return {
            "pairs": count,
            "joint_accuracy": sum(row["joint_credit"] for row in group) / count,
            "chance": sum(row["chance"] for row in group) / count,
            "mean_answer_margin": sum(row["answer_margin"] for row in group) / count,
        }

    return {
        "overall": summarize(results),
        "categories": {
            category: summarize(group) for category, group in groups.items()
        },
        "items": results,
    }


def score_ranking(loaded: Any, panel: RankingPanel, digest: str) -> dict[str, Any]:
    from klab.probes import scoring

    if loaded.config.runtime.engine != "pytorch":
        raise ValueError("answer ranking requires the PyTorch scoring runtime")
    groups = defaultdict(list)
    results = []
    for item in panel.items:
        continuations = {}
        seen = set()
        prefix = item.prefix.rstrip()
        prefix_ids = scoring.encode(loaded, prefix)
        if not prefix_ids:
            raise ValueError(f"ranking item {item.id} has no encoded prefix")
        for choice in item.candidates:
            entire = scoring.encode(loaded, item.prefix + " " + choice)
            start = len(prefix_ids)
            continuation = tuple(entire[start:])
            if entire[:start] != prefix_ids or not continuation or continuation in seen:
                raise ValueError(
                    f"ranking item {item.id} has an ambiguous token boundary or choice"
                )
            if len(entire) - 1 > loaded.config.model.max_seq_len:
                raise ValueError(f"ranking item {item.id} exceeds checkpoint context")
            seen.add(continuation)
            continuations[choice] = continuation
        scores = scoring.score_encoded_choices(loaded, prefix_ids, continuations)
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError(f"ranking item {item.id} produced a nonfinite score")
        credit = scoring.tie_credit(scores, item.answer)
        row = {
            "id": item.id,
            "category": item.category,
            "answer": item.answer,
            "credit": credit,
            "chance": 1 / len(scores),
            "scores": scores,
        }
        groups[item.category].append(row)
        results.append(row)

    def summarize(rows):
        return {
            "items": len(rows),
            "accuracy": sum(row["credit"] for row in rows) / len(rows),
            "chance": sum(row["chance"] for row in rows) / len(rows),
        }

    return {
        "source": "standalone_ranking",
        "identity": loaded.identity,
        "ranking": {
            "name": panel.name,
            "sha256": digest,
            "protocol": scoring.RANKING_PROTOCOL,
            "overall": summarize(results),
            "categories": {
                category: summarize(group) for category, group in groups.items()
            },
            "items": results,
            **({"pairs": _paired_results(panel.pairs, results)} if panel.pairs else {}),
        },
    }
