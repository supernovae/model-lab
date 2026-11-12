"""Descriptive, read-only measurements of verified corpus release views."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import zip_longest
from pathlib import Path
from typing import Any

from klab.training.manifest import sha256_file

SPLITS = ("train", "validation", "test")
DIMENSIONS = ("origin", "verification_status", "shape")
_WORD = re.compile(r"\w+", re.UNICODE)


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line:
                yield json.loads(line)


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


class _Distribution:
    """Incremental row and token denominators for one view or the training mix."""

    def __init__(self, *, counted_tokens: bool) -> None:
        self.counted_tokens = counted_tokens
        self.records = 0
        self.token_total = 0
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.tokens: dict[str, Counter[str]] = defaultdict(Counter)

    def add(self, row: dict[str, Any], token_count: int | None) -> None:
        self.records += 1
        self.token_total += token_count or 0
        counts, tokens = self.counts, self.tokens
        shape = row.get("shape") or {}
        attributes = shape.get("attributes") or {}
        generator = row.get("generator") or {}
        verification = row.get("verification") or {}
        fields = {
            "origin": row.get("origin", "legacy_unclassified"),
            "verification_status": verification.get(
                "status", row.get("validation_status", "unverified")
            ),
            "shape": shape.get("id", "legacy_unclassified"),
            "generator_identity": row.get("generator_identity") or "none",
            "generator_provider": generator.get("provider") or "none",
            "generator_model": generator.get("model") or "none",
            "generator_adapter_id": generator.get("adapter_id") or "none",
            "generator_model_revision": generator.get("model_revision") or "none",
            "generator_version": generator.get("generator_version")
            or generator.get("adapter_version")
            or "none",
            "template_family": row.get("template_family_id") or "none",
            "template_id": generator.get("template_id")
            or row.get("template_family_id")
            or "none",
        }
        fields.update(
            {
                f"shape.{key}": value
                for key, value in attributes.items()
                if isinstance(value, str)
            }
        )
        for key, value in fields.items():
            counts[key][str(value)] += 1
            if token_count is not None:
                tokens[key][str(value)] += token_count
        for domain in attributes.get("source_domains", row.get("domains", [])):
            counts["source_domain"][domain] += 1
            if token_count is not None:
                tokens["source_domain"][domain] += token_count
        for modality in row.get("modalities", []):
            counts["source_modality"][modality] += 1
            if token_count is not None:
                tokens["source_modality"][modality] += token_count

    def result(self) -> dict[str, Any]:
        records, token_total = self.records, self.token_total
        counted_tokens = self.counted_tokens
        counts, tokens = self.counts, self.tokens
        return {
            "records": records,
            "actual_tokens": token_total if counted_tokens else None,
            "token_count_reason": None if counted_tokens else "tokenizer_not_declared",
            "dimensions": {
                key: {
                    value: {
                        "records": count,
                        "record_fraction": count / records,
                        "actual_tokens": tokens[key][value] if counted_tokens else None,
                        "token_fraction": (
                            tokens[key][value] / token_total if token_total else None
                        ),
                    }
                    for value, count in sorted(values.items())
                }
                for key, values in sorted(counts.items())
            },
        }


class _Concentration:
    """Track overlap without retaining rendered texts or whole chat records."""

    def __init__(self, *, chat: bool) -> None:
        self.chat = chat
        self.exact: set[bytes] = set()
        self.normalized: set[bytes] = set()
        self.prefixes: set[str] = set()
        self.prompt_prefixes: set[str] = set()
        self.answer_prefixes: set[str] = set()
        self.duplicate = self.normalized_duplicate = self.prefix_rows = 0
        self.prompt_rows = self.answer_rows = 0
        self.chars = 0
        self.ngrams: Counter[tuple[str, ...]] = Counter()

    def add(self, text: str, row: dict[str, Any] | None = None) -> None:
        digest = hashlib.sha256(text.encode()).digest()
        self.duplicate += digest in self.exact
        self.exact.add(digest)
        normalized = unicodedata.normalize(
            "NFC", text.replace("\r\n", "\n").replace("\r", "\n")
        )
        digest = hashlib.sha256(normalized.encode()).digest()
        self.normalized_duplicate += digest in self.normalized
        self.normalized.add(digest)
        if len(text) >= 32:
            prefix = text[:32]
            self.prefix_rows += prefix in self.prefixes
            self.prefixes.add(prefix)
        previous_chars = self.chars
        self.chars += len(text)
        if previous_chars <= 20_000_000 < self.chars:
            self.ngrams.clear()
        if self.chars <= 20_000_000:
            words = _WORD.findall(normalized.casefold())
            self.ngrams.update(zip(words, words[1:], words[2:], strict=False))
        if row is not None and self.chars <= 20_000_000:
            prompt = next(
                (
                    message["content"]
                    for message in row["messages"]
                    if message["role"] == "user"
                ),
                "",
            )
            answer = next(
                (
                    message["content"]
                    for message in reversed(row["messages"])
                    if message["role"] == "assistant"
                ),
                "",
            )
            if len(prompt) >= 32:
                prefix = prompt[:32]
                self.prompt_rows += prefix in self.prompt_prefixes
                self.prompt_prefixes.add(prefix)
            if len(answer) >= 32:
                prefix = answer[:32]
                self.answer_rows += prefix in self.answer_prefixes
                self.answer_prefixes.add(prefix)

    def result(self) -> dict[str, Any]:
        if self.chars > 20_000_000:
            return {
                "duplicate_text_rows": self.duplicate,
                "normalized_duplicate_rows": self.normalized_duplicate,
                "prefix_32_rows": self.prefix_rows,
                "ngram_3_distinct": None,
                "ngram_3_total": None,
                "ngram_count_reason": "omitted_above_20m_character_bound",
                "advisories": [
                    "Large corpus: lexical n-gram concentration not exhaustively measured."
                ],
            }
        ngram_total = sum(self.ngrams.values())
        advisories = []
        if self.normalized_duplicate:
            advisories.append(
                "Repeated exact or NFC/newline-normalized text; inspect origins."
            )
        if self.prefix_rows:
            advisories.append(
                "Shared rendered prefixes; inspect templates and boilerplate."
            )
        if ngram_total >= 20 and len(self.ngrams) * 4 < ngram_total:
            advisories.append(
                "Repeated 3-word sequences; inspect source and template concentration."
            )
        result = {
            "duplicate_text_rows": self.duplicate,
            "normalized_duplicate_rows": self.normalized_duplicate,
            "prefix_32_rows": self.prefix_rows,
            "ngram_3_distinct": len(self.ngrams) if ngram_total >= 20 else None,
            "ngram_3_total": ngram_total if ngram_total >= 20 else None,
            "advisories": advisories,
        }
        if self.chat:
            result["prompt_prefix_32_rows"] = self.prompt_rows
            result["answer_prefix_32_rows"] = self.answer_rows
        return result


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
                        "generator_version": scenario["generator_version"],
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


def summarize_release(
    root: Path,
    *,
    release_spec: dict[str, Any] | None = None,
    tokenizer: Path | None = None,
) -> dict[str, Any]:
    """Frozen descriptive census, with measured tokens only for a pinned tokenizer."""
    root = Path(root)
    with _lineage_index(root) as connection:
        summary = _measure_views(root, connection, tokenizer, release_spec)
        return _summarize_release(root, connection, summary, release_spec)


def _summarize_release(
    root: Path,
    connection: sqlite3.Connection,
    summary: dict[str, Any],
    release_spec: dict[str, Any] | None,
) -> dict[str, Any]:
    views = summary["views"]
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for splits in views.values():
        for item in splits.values():
            for dimension, values in item["dimensions"].items():
                for value, count in values.items():
                    totals[dimension][value] += count["records"]
    connection.execute("CREATE TABLE selected (id TEXT PRIMARY KEY)")
    for view in ("lm", "chat"):
        if (
            release_spec is not None
            and release_spec[view]["selected"]
            and "train" in release_spec[view]["training_splits"]
        ):
            connection.executemany(
                "INSERT OR IGNORE INTO selected VALUES (?)",
                (
                    (link["record_id"],)
                    for link in _rows(root / view / "train.lineage.jsonl")
                ),
            )
    heldout_keys = (
        "parent_document_ids",
        "source_family_ids",
        "generator_world_id",
        "scenario_family_id",
        "template_family_id",
    )
    heldout: dict[str, set[str]] = {key: set() for key in heldout_keys}
    test_parents: set[str] = set()
    connection.execute("CREATE TABLE test_ids (id TEXT PRIMARY KEY)")
    for view in ("lm", "chat"):
        connection.executemany(
            "INSERT OR IGNORE INTO test_ids VALUES (?)",
            (
                (link["record_id"],)
                for link in _rows(root / view / "test.lineage.jsonl")
            ),
        )
    for (data,) in connection.execute(
        "SELECT lineage.data FROM lineage JOIN test_ids ON lineage.id = test_ids.id"
    ):
        row = json.loads(data)
        test_parents.update(row["parent_document_ids"])
        for key in heldout_keys:
            values = row.get(key, []) if key.endswith("_ids") else [row.get(key)]
            heldout[key].update(value for value in values if value is not None)
    generated_count = unverified_generated = 0
    generated_parents: set[str] = set()
    shared: dict[str, dict[str, set[str]]] = {}
    for (data,) in connection.execute(
        "SELECT lineage.data FROM lineage JOIN selected ON lineage.id = selected.id"
    ):
        row = json.loads(data)
        if row["origin"] in {"primary_source", "human_authored"}:
            continue
        generated_count += 1
        unverified_generated += row["verification"]["status"] in {
            "unverified",
            "schema_validated",
        }
        generated_parents.update(
            parent for parent in row["parent_document_ids"] if parent in test_parents
        )
        shape = row["shape"]["id"]
        groups = shared.setdefault(shape, {key: set() for key in heldout_keys})
        for key in heldout_keys:
            values = row.get(key, []) if key.endswith("_ids") else [row.get(key)]
            groups[key].update(value for value in values if value in heldout[key])
    generated_test_overlap = sorted(generated_parents & test_parents)
    shared_by_shape = {
        shape: {key: sorted(values) for key, values in groups.items()}
        for shape, groups in sorted(shared.items())
    }
    lexical_count = 0
    lexical_terms: set[str] = set()
    for row in _rows(root / "lexical/candidates.jsonl"):
        lexical_count += 1
        lexical_terms.add(row["term"])
    summary["inventory"] = {
        "lexical_candidate_count": lexical_count,
        "lexical_distinct_terms": len(lexical_terms),
        "view_record_distribution": {
            key: dict(sorted(values.items())) for key, values in sorted(totals.items())
        },
    }
    summary["inventory"]["generated_training_records"] = generated_count
    summary["inventory"]["unverified_generated_training_records"] = unverified_generated
    summary["inventory"]["generated_test_shared_parent_ids"] = generated_test_overlap
    summary["inventory"]["generated_heldout_lineage_by_shape"] = shared_by_shape
    summary["advisories"] = sorted(
        {
            warning
            for splits in views.values()
            for item in splits.values()
            for warning in item["concentration"]["advisories"]
        }
    )
    if unverified_generated:
        summary["advisories"].append(
            f"{unverified_generated} generated training records have no correctness verification beyond schema validation."
        )
    if generated_test_overlap:
        summary["advisories"].append(
            "Generated training and test examples share source document lineage."
        )
    summary["advisories"].sort()
    return summary
