"""The serial, SQLite-backed `measure_views` that the sharded one must equal exactly.

This is the implementation the sharded measurement replaced, kept as the oracle
for the equality test. It is slow and single-process by design.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import zip_longest
from pathlib import Path
from typing import Any

from klab.corpus.measurement import (
    SPLITS,
    _Concentration,
    _Distribution,
    _rows,
)
from klab.training.manifest import sha256_file


@contextmanager
def _lineage_index(root: Path) -> Iterator[sqlite3.Connection]:
    """Keep the release read-only and remove the on-disk index on exit."""
    with tempfile.TemporaryDirectory(
        prefix=".measurement-", dir=root.parent
    ) as directory:
        connection = sqlite3.connect(str(Path(directory) / "index.sqlite"))
        try:
            connection.execute(
                "CREATE TABLE lineage (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
            )
            connection.executemany(
                "INSERT INTO lineage VALUES (?, ?)",
                (
                    (row["record_id"], json.dumps(row))
                    for row in _rows(root / "lineage.jsonl")
                ),
            )
            connection.commit()
            yield connection
        finally:
            connection.close()


def _record(connection: sqlite3.Connection, record_id: str) -> dict[str, Any] | None:
    match = connection.execute(
        "SELECT data FROM lineage WHERE id = ?", (record_id,)
    ).fetchone()
    return json.loads(match[0]) if match is not None else None


def measure_views(
    root: Path,
    tokenizer: Path | None = None,
    *,
    release_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Count exact rendered tokenizer inputs and classify records independently."""
    root = Path(root)
    with _lineage_index(root) as connection:
        return _measure_views(root, connection, tokenizer, release_spec)


def _measure_views(
    root: Path,
    connection: sqlite3.Connection,
    tokenizer: Path | None,
    release_spec: dict[str, Any] | None,
) -> dict[str, Any]:
    model = None
    if tokenizer is not None:
        from tokenizers import Tokenizer

        model = Tokenizer.from_file(str(tokenizer))
    generations = {row["record_id"]: row for row in _rows(root / "generations.jsonl")}
    scenarios = {row["scenario_id"]: row for row in _rows(root / "scenarios.jsonl")}
    views = {}
    training = _Distribution(counted_tokens=model is not None)
    for view in ("lm", "chat"):
        views[view] = {}
        for split in SPLITS:
            payload = root / view / f"{split}.jsonl"
            links = _rows(root / view / f"{split}.lineage.jsonl")
            if view == "lm":
                rows = ((row["text"], None) for row in _rows(payload))
            else:
                from klab.data.conversations import iter_rendered_conversations

                rows = (
                    (rendered.text, row)
                    for row, rendered in zip(
                        _rows(payload),
                        iter_rendered_conversations(payload),
                        strict=True,
                    )
                )
            distribution = _Distribution(counted_tokens=model is not None)
            concentration = _Concentration(chat=view == "chat")
            connection.execute(
                "CREATE TABLE parents (id TEXT PRIMARY KEY, count INTEGER NOT NULL)"
            )
            families: Counter[str] = Counter()
            for link, rendered in zip_longest(links, rows):
                if link is None or rendered is None:
                    raise ValueError("view and lineage lengths disagree")
                text, chat_row = rendered
                if (
                    link["split"] != split
                    or (record := _record(connection, link["record_id"])) is None
                ):
                    raise ValueError("view lineage reference mismatch")
                generation = generations.get(record.get("generation_id"))
                scenario = scenarios.get(record.get("scenario_id"))
                if generation:
                    record["generator"] = generation["generator"]
                elif scenario:
                    record["generator"] = {
                        "adapter_id": scenario["generator_id"],
                    }
                elif record.get("semantic_id"):
                    record["generator"] = {"generator_version": "semantic_chat_v1"}
                count = len(model.encode(text).ids) if model is not None else None
                distribution.add(record, count)
                if (
                    release_spec is not None
                    and split == "train"
                    and release_spec[view]["selected"]
                    and split in release_spec[view]["training_splits"]
                ):
                    training.add(record, count)
                concentration.add(text, chat_row)
                connection.executemany(
                    "INSERT INTO parents VALUES (?, 1) "
                    "ON CONFLICT(id) DO UPDATE SET count = count + 1",
                    ((parent,) for parent in record.get("parent_document_ids", [])),
                )
                families.update(record.get("source_family_ids", []))
            summary = distribution.result()
            summary["concentration"] = concentration.result()
            parent_count, parent_max = connection.execute(
                "SELECT count(*), max(count) FROM parents"
            ).fetchone()
            summary["concentration"]["parent_document_counts"] = (
                dict(connection.execute("SELECT id, count FROM parents ORDER BY id"))
                if parent_count <= 100_000
                else {}
            )
            if parent_count > 100_000:
                summary["concentration"]["parent_document_count_reason"] = (
                    "omitted_above_100k_parent_bound"
                )
            connection.execute("DROP TABLE parents")
            summary["concentration"]["source_family_counts"] = dict(
                sorted(families.items())
            )
            for dimension in ("generator_identity", "template_family"):
                groups = summary["dimensions"].get(dimension, {})
                if (
                    (
                        item_count := max(
                            (item["records"] for item in groups.values()), default=0
                        )
                    )
                    > 1
                    and item_count == summary["records"]
                    and len(groups) == 1
                ):
                    summary["concentration"]["advisories"].append(
                        f"All {view}/{split} records share {dimension}; inspect lineage concentration."
                    )
            if parent_max is not None and parent_max > 1:
                summary["concentration"]["advisories"].append(
                    f"Repeated parent document lineage in {view}/{split}; inspect ancestry."
                )
            if (
                families
                and summary["records"] > 1
                and max(families.values()) == summary["records"]
            ):
                summary["concentration"]["advisories"].append(
                    f"All {view}/{split} records inherit one source family; inspect source concentration."
                )
            views[view][split] = summary
    result = {
        "views": views,
        "denominators": {
            "records": "exported rows",
            "actual_tokens": "tokens in rendered tokenizer inputs; domain counts may overlap",
        },
    }
    if release_spec is not None:
        result["training_mixture"] = training.result()
    if tokenizer is not None:
        result["tokenizer_sha256"] = sha256_file(tokenizer)
    return result


def measure_source_rights(
    root: Path, tokenizer: Path
) -> dict[str, dict[str, int | None]]:
    """Measure each retained train source passage once, not its derived views."""
    from tokenizers import Tokenizer

    model = Tokenizer.from_file(str(tokenizer))
    counted: set[str] = set()
    totals: dict[str, dict[str, int | None]] = {
        state: {"documents": 0, "source_tokens": None}
        for state in (
            "eligible",
            "eligible_with_obligations",
            "review_required",
            "ineligible",
        )
    }
    for state in ("eligible", "eligible_with_obligations"):
        totals[state]["source_tokens"] = 0
    for document in _rows(Path(root) / "documents.jsonl"):
        if document["split"] != "train" or document.get("drop_reason"):
            continue
        rights = document.get("rights")
        if rights is None:
            continue
        state = rights["training_eligibility"]
        if state not in ("eligible", "eligible_with_obligations"):
            raise ValueError(
                "ineligible source document in retained training inventory"
            )
        sha = document["content_sha256"]
        if sha in counted:
            continue
        counted.add(sha)
        totals[state]["documents"] += 1
        totals[state]["source_tokens"] += len(model.encode(document["text"]).ids)
    return totals
