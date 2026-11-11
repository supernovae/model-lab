"""Bind independent item-level reviews to a verified descriptive panel."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from klab.campaign.state import read_canonical
from klab.canonical import canonical_json
from klab.evaluation.panel import verify_panel_result
from klab.training.manifest import sha256_file

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _review_file(
    path: Path, expected_sha: str, expected_count: int
) -> list[dict[str, Any]]:
    if not _SHA.fullmatch(expected_sha) or sha256_file(path) != expected_sha:
        raise ValueError("review ledger identity changed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != expected_count:
        raise ValueError("review ledger has incomplete item coverage")
    return rows


def _load_prompt_renderer(path: Path, expected_sha: str) -> str:
    if not _SHA.fullmatch(expected_sha) or sha256_file(path) != expected_sha:
        raise ValueError("prompt renderer identity changed")
    renderer = read_canonical(path)
    if sha256_file(path) != expected_sha:
        raise ValueError("prompt renderer changed while reading")
    if (
        not isinstance(renderer, dict)
        or set(renderer) != {"format", "template"}
        or renderer["format"] != "frozen-question-prompt-renderer-v1"
        or not isinstance(renderer["template"], str)
    ):
        raise ValueError("invalid frozen-question prompt renderer")
    template = renderer["template"]
    remainder = template.replace("{question}", "")
    if (
        template.count("{question}") != 1
        or not template.strip()
        or "{" in remainder
        or "}" in remainder
    ):
        raise ValueError("prompt renderer must contain exactly one question field")
    return template


def verify_reviewed_scores(
    path: Path,
    *,
    index_sha256: str,
    checkpoint_sha256: str,
    frozen_content_sha256: str,
    expected_items: int,
    expected_axes: int,
    items_per_axis: int,
    min_correct: int,
    min_axis_correct: int,
    min_format: int,
    min_axis_format: int,
) -> dict[str, Any]:
    """Verify immutable sources, complete independent reviews, and derived totals.

    Score imports are observational evidence. The readiness policy and named
    model review decide whether a checkpoint can progress.
    """
    receipt = read_canonical(Path(path))
    if receipt.get("format") != "reviewed-panel-scores-v1":
        raise ValueError("invalid reviewed-score receipt")
    if (
        receipt.get("evaluation_index_sha256") != index_sha256
        or receipt.get("checkpoint_sha256") != checkpoint_sha256
        or receipt.get("frozen_content_sha256") != frozen_content_sha256
    ):
        raise ValueError("reviewed scores differ from readiness binding")
    panel = verify_panel_result(Path(receipt["panel_path"]))
    if (
        panel["record_sha256"] != receipt.get("panel_record_sha256")
        or panel["evaluation_index_sha256"] != index_sha256
        or panel["checkpoint_sha256"] != checkpoint_sha256
    ):
        raise ValueError("reviewed scores differ from verified panel")
    order_path = Path(receipt["order_path"])
    if sha256_file(order_path) != receipt.get("order_file_sha256"):
        raise ValueError("frozen item order changed")
    order = json.loads(order_path.read_text())
    if sha256_file(order_path) != receipt.get("order_file_sha256"):
        raise ValueError("frozen item order changed while reading")
    if (
        order.get("frozen_content_sha256") != frozen_content_sha256
        or len(order.get("item_ids", [])) != expected_items
        or len(order.get("item_content_sha256", [])) != expected_items
        or len(set(order["item_ids"])) != expected_items
        or len(panel["rows"]) != expected_items
    ):
        raise ValueError("frozen item or panel coverage differs")
    frozen = read_canonical(Path(receipt["frozen_items_path"]))
    if (
        frozen.get("format") != "kml-card03-evaluation-manifest-v1"
        or frozen.get("content_sha256") != frozen_content_sha256
        or hashlib.sha256(
            canonical_json(
                {key: value for key, value in frozen.items() if key != "content_sha256"}
            )
        ).hexdigest()
        != frozen_content_sha256
    ):
        raise ValueError("frozen evaluation manifest identity changed")
    closed_items = [item for item in frozen["items"] if item["suite"] == "closed_book"]
    items = {item["id"]: item for item in closed_items}
    if (
        len(closed_items) != expected_items
        or len(items) != expected_items
        or set(order["item_ids"]) != set(items)
        or any(
            item_id not in items
            or items[item_id]["content_sha256"] != digest
            or items[item_id]["split"] != "test"
            for item_id, digest in zip(
                order["item_ids"], order["item_content_sha256"], strict=True
            )
        )
    ):
        raise ValueError("frozen item order or content differs from accepted suite")
    renderer_path = receipt.get("renderer_path")
    renderer_sha = receipt.get("renderer_file_sha256")
    if not isinstance(renderer_path, str) or not isinstance(renderer_sha, str):
        raise TypeError("content-pinned prompt renderer required for readiness")
    template = _load_prompt_renderer(Path(renderer_path), renderer_sha)
    for ordinal, item_id in enumerate(order["item_ids"]):
        question = items[item_id].get("question")
        observed = panel["rows"][ordinal]
        if (
            not isinstance(question, str)
            or not question.strip()
            or observed.get("prompt") != template.replace("{question}", question)
        ):
            raise ValueError("panel prompt differs from rendered frozen question")
    reviewer_paths = [Path(value) for value in receipt["review_paths"]]
    reviewer_shas = receipt["review_file_sha256"]
    reviewers = receipt["reviewers"]
    if (
        len(reviewer_paths) != 2
        or len(reviewer_shas) != 2
        or len(reviewers) != 2
        or reviewers[0] == reviewers[1]
        or any(
            not isinstance(value, str) or not _ID.fullmatch(value)
            for value in reviewers
        )
        or reviewer_paths[0].resolve() == reviewer_paths[1].resolve()
    ):
        raise ValueError("two distinct named reviewer ledgers required")
    ledgers = [
        _review_file(source, digest, expected_items)
        for source, digest in zip(reviewer_paths, reviewer_shas, strict=True)
    ]
    adjudications = receipt.get("adjudications")
    if not isinstance(adjudications, list):
        raise TypeError("adjudication list required")
    resolved: dict[int, dict[str, Any]] = {}
    for row in adjudications:
        ordinal = row.get("index") if isinstance(row, dict) else None
        if (
            type(ordinal) is not int
            or ordinal in resolved
            or not 0 <= ordinal < expected_items
        ):
            raise ValueError("invalid or duplicate adjudication")
        if not isinstance(row.get("adjudicator"), str) or not _ID.fullmatch(
            row["adjudicator"]
        ):
            raise ValueError("named adjudicator required")
        if not isinstance(row.get("note"), str) or not row["note"].strip():
            raise ValueError("adjudication rationale required")
        if (
            type(row.get("score")) is not int
            or row["score"] not in (0, 1)
            or type(row.get("format_ok")) is not bool
        ):
            raise ValueError("invalid adjudicated score")
        resolved[ordinal] = row
    axis: dict[str, dict[str, int]] = {}
    correct = formatting = 0
    for ordinal, (left, right, observed) in enumerate(
        zip(*ledgers, panel["rows"], strict=True)
    ):
        if observed.get("status") != "COMPLETED":
            raise ValueError("failed or missing panel generation cannot be scored")
        for reviewer, row in zip(reviewers, (left, right), strict=True):
            if (
                row.get("index") != ordinal
                or row.get("id") != order["item_ids"][ordinal]
                or row.get("content_sha256") != order["item_content_sha256"][ordinal]
                or row.get("category") != items.get(row.get("id"), {}).get("category")
                or row.get("reviewer") != reviewer
                or row.get("prompt_sha256")
                != hashlib.sha256(observed["prompt"].encode()).hexdigest()
                or type(row.get("score")) is not int
                or row["score"] not in (0, 1)
                or type(row.get("format_ok")) is not bool
                or not isinstance(row.get("category"), str)
                or not _ID.fullmatch(row["category"])
            ):
                raise ValueError("review row differs from frozen item or panel prompt")
        if left["category"] != right["category"]:
            raise ValueError("reviewers disagree on category")
        disagreement = (left["score"], left["format_ok"]) != (
            right["score"],
            right["format_ok"],
        )
        if disagreement != (ordinal in resolved):
            raise ValueError("unresolved or unnecessary adjudication")
        selected = resolved.get(ordinal, left)
        if ordinal in resolved and (
            selected.get("id") != left["id"]
            or selected.get("content_sha256") != left["content_sha256"]
        ):
            raise ValueError("adjudication differs from frozen item")
        counts = axis.setdefault(
            left["category"], {"items": 0, "correct": 0, "format": 0}
        )
        counts["items"] += 1
        counts["correct"] += selected["score"]
        counts["format"] += int(selected["format_ok"])
        correct += selected["score"]
        formatting += int(selected["format_ok"])
    if len(axis) != expected_axes or any(
        row["items"] != items_per_axis for row in axis.values()
    ):
        raise ValueError("reviewed-score axis denominators differ")
    passed = (
        correct >= min_correct
        and formatting >= min_format
        and all(
            row["correct"] >= min_axis_correct and row["format"] >= min_axis_format
            for row in axis.values()
        )
    )
    return {
        "receipt_sha256": sha256_file(Path(path)),
        "panel_record_sha256": panel["record_sha256"],
        "correct": correct,
        "format_ok": formatting,
        "axes": axis,
        "passed": passed,
    }
