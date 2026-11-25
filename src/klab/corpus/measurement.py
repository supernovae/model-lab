"""Descriptive, read-only measurements of verified corpus release views."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from klab.corpus.release_verify import (
    current_settings,
    line_boundaries,
    range_records,
    scratch_root,
)
from klab.corpus.shards import ShardPlan, chunk_ranges, declare_stage, run_sharded
from klab.safe_io import read_file_range
from klab.training.manifest import sha256_file

SPLITS = ("train", "validation", "test")
DIMENSIONS = ("origin", "verification_status", "shape")
_WORD = re.compile(r"\w+", re.UNICODE)
_NGRAM_CHARS = 20_000_000


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


RIGHTS_STAGE = declare_stage("describe: source rights")
_ELIGIBLE = ("eligible", "eligible_with_obligations")


def _rights_range(
    path: str, start: int, end: int, tokenizer: str
) -> dict[str, list[Any]]:
    """Retained train documents in one byte range: content digest, rights state, tokens."""
    model = _tokenizer(tokenizer)
    digests, states, tokens, seen = [], [], [], set()
    for row, _, _ in range_records(Path(path), start, end):
        if row["split"] != "train" or row.get("drop_reason"):
            continue
        rights = row.get("rights")
        if rights is None:
            continue
        state = rights["training_eligibility"]
        if state not in _ELIGIBLE:
            raise ValueError(
                "ineligible source document in retained training inventory"
            )
        sha = row["content_sha256"]
        if sha in seen:
            continue
        seen.add(sha)
        digests.append(bytes.fromhex(sha))
        states.append(_ELIGIBLE.index(state))
        tokens.append(len(model.encode(row["text"]).ids))
    return {
        "digests": base64.b64encode(b"".join(digests)).decode("ascii"),
        "states": states,
        "tokens": tokens,
    }


def measure_source_rights(
    root: Path, tokenizer: Path, *, cache: Path | None = None
) -> dict[str, dict[str, int | None]]:
    """Measure each retained train source passage once, not its derived views.

    A passage that appears in several documents counts for the first one in file
    order, wherever its duplicates are, so the byte ranges of `documents.jsonl`
    can be measured independently and merged.
    """
    root = Path(root)
    documents = root / "documents.jsonl"
    info = documents.stat()
    cache = cache or _scratch_cache()
    plan = ShardPlan(
        RIGHTS_STAGE,
        [(str(documents), *r, str(tokenizer)) for r in chunk_ranges(documents)],
        info.st_size,
        {
            "documents": [info.st_size, info.st_mtime_ns],
            "tokenizer": sha256_file(tokenizer),
            "measurement": sha256_file(Path(__file__)),
        },
        None if cache is None else cache / "shards",
    )
    parts = list(run_sharded(plan, _rights_range, ordered=True))
    totals: dict[str, dict[str, int | None]] = {
        state: {"documents": 0, "source_tokens": None}
        for state in (*_ELIGIBLE, "review_required", "ineligible")
    }
    for state in _ELIGIBLE:
        totals[state]["source_tokens"] = 0
    if parts:
        digests = np.frombuffer(
            b"".join(base64.b64decode(part["digests"]) for part in parts), dtype="S32"
        )
        states = np.array([s for part in parts for s in part["states"]], dtype=np.int64)
        tokens = np.array([t for part in parts for t in part["tokens"]], dtype=np.int64)
        _, first = np.unique(digests, return_index=True)
        for index, state in enumerate(_ELIGIBLE):
            chosen = first[states[first] == index]
            totals[state]["documents"] = int(chosen.size)
            totals[state]["source_tokens"] = int(tokens[chosen].sum())
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

    def state(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "token_total": self.token_total,
            "counts": {key: dict(value) for key, value in self.counts.items()},
            "tokens": {key: dict(value) for key, value in self.tokens.items()},
        }

    def absorb(self, state: dict[str, Any]) -> None:
        """Add another shard's counts; every field is a sum, so order never matters."""
        self.records += state["records"]
        self.token_total += state["token_total"]
        for key, values in state["counts"].items():
            self.counts[key].update(values)
        for key, values in state["tokens"].items():
            self.tokens[key].update(values)

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
        if previous_chars <= _NGRAM_CHARS < self.chars:
            self.ngrams.clear()
        if self.chars <= _NGRAM_CHARS:
            words = _WORD.findall(normalized.casefold())
            self.ngrams.update(zip(words, words[1:], words[2:], strict=False))
        if row is not None and self.chars <= _NGRAM_CHARS:
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
        if self.chars > _NGRAM_CHARS:
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


INDEX_STAGE = declare_stage("describe: lineage index")
COUNT_STAGE = declare_stage("describe: view lines")
VIEW_STAGE = declare_stage("describe: view rows")
_PARENT_ID_LIMIT = 100_000
# Every Nth link line gets an offset, so a shard finds its first link by skipping
# fewer than this many lines.
_LINK_STRIDE = 4096


def _scratch_cache() -> Path | None:
    """Where finished runs are kept when the caller names no cache: the scratch directory."""
    scratch = current_settings().scratch_dir
    return None if scratch is None else scratch / "describe"


def _encode(digests: list[bytes]) -> str:
    return base64.b64encode(b"".join(sorted(set(digests)))).decode("ascii")


def _decode(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype="S32")


def _union(states: list[dict[str, Any]], key: str) -> int:
    arrays = [_decode(state[key]) for state in states]
    return int(np.unique(np.concatenate(arrays)).size) if arrays else 0


def _id_hash(identity: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(identity, digest_size=8).digest(), "big")


def _record_id(line: bytes) -> bytes:
    """A lineage row's `record_id`, found without parsing the row when it is plain."""
    start = line.find(b'"record_id":"')
    if start >= 0:
        start += 13
        end = line.find(b'"', start)
        if end > 0 and b"\\" not in line[start:end]:
            return line[start:end]
    return json.loads(line)["record_id"].encode()


def _index_range(path: str, start: int, end: int) -> tuple[np.ndarray, ...]:
    """Hash, offset and length of every lineage row in one byte range."""
    hashes, offsets, lengths = [], [], []
    with Path(path).open("rb") as stream:
        stream.seek(start)
        position = start
        for line in stream:
            if position >= end:
                break
            if line.strip():
                hashes.append(_id_hash(_record_id(line)))
                offsets.append(position)
                lengths.append(len(line))
            position += len(line)
    return (
        np.array(hashes, dtype=np.uint64),
        np.array(offsets, dtype=np.uint64),
        np.array(lengths, dtype=np.uint32),
    )


def _build_lineage_index(lineage: Path, directory: Path) -> None:
    """Write the sorted (id hash, offset, length) arrays of `lineage.jsonl`.

    A lookup reads the one row it needs and checks its `record_id`, so a hash
    collision or a stale index is detected, never silently used.
    """
    if (directory / "complete").exists():
        return
    plan = ShardPlan(
        INDEX_STAGE,
        [(str(lineage), *r) for r in chunk_ranges(lineage)],
        lineage.stat().st_size,
    )
    parts = list(run_sharded(plan, _index_range, ordered=True))
    hashes, offsets, lengths = (
        (np.concatenate([part[column] for part in parts]) for column in range(3))
        if parts
        else (np.empty(0, np.uint64), np.empty(0, np.uint64), np.empty(0, np.uint32))
    )
    order = np.argsort(hashes, kind="stable")
    staging = directory.with_name(directory.name + f".partial-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=True)
    for name, column in (
        ("hashes", hashes),
        ("offsets", offsets),
        ("lengths", lengths),
    ):
        np.save(staging / f"{name}.npy", column[order])
    (staging / "complete").write_text("1")
    if directory.exists():
        for old in directory.iterdir():
            old.unlink()
        directory.rmdir()
    staging.rename(directory)


class _LineageIndex:
    def __init__(self, lineage: Path, directory: Path) -> None:
        self.path = lineage
        self.hashes = np.load(directory / "hashes.npy", mmap_mode="r")
        self.offsets = np.load(directory / "offsets.npy", mmap_mode="r")
        self.lengths = np.load(directory / "lengths.npy", mmap_mode="r")
        self.stream = lineage.open("rb")

    def get(self, record_id: str) -> dict[str, Any] | None:
        wanted = record_id.encode()
        key = np.uint64(_id_hash(wanted))
        low = int(np.searchsorted(self.hashes, key, side="left"))
        high = int(np.searchsorted(self.hashes, key, side="right"))
        for position in range(low, high):
            row = json.loads(
                read_file_range(
                    self.stream,
                    int(self.offsets[position]),
                    int(self.lengths[position]),
                )
            )
            if row["record_id"] == record_id:
                return row
        return None


def _count_lines(path: str, start: int, end: int) -> int:
    """Rows in a byte range; a final row without a newline counts too."""
    count, position, last = 0, start, b"\n"
    with Path(path).open("rb") as stream:
        stream.seek(start)
        while position < end:
            block = stream.read(min(16 * 1024**2, end - position))
            if not block:
                break
            count += block.count(b"\n")
            position += len(block)
            last = block[-1:]
    return count + (last != b"\n")


def _link_lines(path: Path, offset: int, skip: int, count: int) -> Iterator[bytes]:
    with path.open("rb") as stream:
        stream.seek(offset)
        for _ in range(skip):
            stream.readline()
        for _ in range(count):
            yield stream.readline()


_tokenizers: dict[str, Any] = {}


def _tokenizer(path: str) -> Any:
    if path not in _tokenizers:
        # One thread per worker process: the pool already uses every core.
        os.environ.setdefault("RAYON_NUM_THREADS", "1")
        from tokenizers import Tokenizer

        _tokenizers[path] = Tokenizer.from_file(path)
    return _tokenizers[path]


def _measure_shard(
    view: str,
    split: str,
    payload: str,
    start: int,
    end: int,
    links: str,
    link_offset: int,
    link_skip: int,
    lines: int,
    lineage: str,
    index: str,
    generations: str,
    scenarios: str,
    tokenizer: str | None,
) -> dict[str, Any]:
    """Measure one run of consecutive view rows and return a mergeable state."""
    model = _tokenizer(tokenizer) if tokenizer is not None else None
    lookup = _LineageIndex(Path(lineage), Path(index))
    generated = {row["record_id"]: row for row in _rows(Path(generations))}
    scenario_rows = {row["scenario_id"]: row for row in _rows(Path(scenarios))}
    if view == "lm":
        texts = (
            (json.loads(line)["text"], None)
            for line, _, _ in _payload_lines(Path(payload), start, end)
        )
    else:
        from klab.data.conversations import iter_rendered_conversations

        texts = (
            (rendered.text, row)
            for row, rendered in zip(
                _rows(Path(payload)),
                iter_rendered_conversations(Path(payload)),
                strict=True,
            )
        )
    distribution = _Distribution(counted_tokens=model is not None)
    concentration = _Concentration(chat=view == "chat")
    parents: Counter[str] = Counter()
    families: Counter[str] = Counter()
    linked = _link_lines(Path(links), link_offset, link_skip, lines)
    rows = 0
    for raw, (text, chat_row) in zip(linked, texts, strict=False):
        link = json.loads(raw)
        record = lookup.get(link["record_id"])
        if link["split"] != split or record is None:
            raise ValueError("view lineage reference mismatch")
        generation = generated.get(record.get("generation_id"))
        scenario = scenario_rows.get(record.get("scenario_id"))
        if generation:
            record["generator"] = generation["generator"]
        elif scenario:
            record["generator"] = {"adapter_id": scenario["generator_id"]}
        elif record.get("semantic_id"):
            record["generator"] = {"generator_version": "semantic_chat_v1"}
        count = len(model.encode(text).ids) if model is not None else None
        distribution.add(record, count)
        concentration.add(text, chat_row)
        parents.update(record.get("parent_document_ids", []))
        families.update(record.get("source_family_ids", []))
        rows += 1
    if rows != lines:
        raise ValueError("view and lineage lengths disagree")
    in_bound = concentration.chars <= _NGRAM_CHARS
    parent_digests = sorted(
        (hashlib.sha256(identity.encode()).digest(), count)
        for identity, count in parents.items()
    )
    return {
        "distribution": distribution.state(),
        "chars": concentration.chars,
        "exact": _encode(list(concentration.exact)),
        "normalized": _encode(list(concentration.normalized)),
        "prefixes": _encode(
            [hashlib.sha256(p.encode()).digest() for p in concentration.prefixes]
        ),
        "prefix_eligible": concentration.prefix_rows + len(concentration.prefixes),
        "prompts": _encode(
            [hashlib.sha256(p.encode()).digest() for p in concentration.prompt_prefixes]
        )
        if in_bound
        else "",
        "prompt_eligible": concentration.prompt_rows
        + len(concentration.prompt_prefixes),
        "answers": _encode(
            [hashlib.sha256(p.encode()).digest() for p in concentration.answer_prefixes]
        )
        if in_bound
        else "",
        "answer_eligible": concentration.answer_rows
        + len(concentration.answer_prefixes),
        "ngrams": [[*gram, count] for gram, count in concentration.ngrams.items()]
        if in_bound
        else None,
        "families": dict(families),
        "parent_digests": base64.b64encode(
            b"".join(digest for digest, _ in parent_digests)
        ).decode("ascii"),
        "parent_counts": [count for _, count in parent_digests],
        "parent_ids": dict(parents) if len(parents) <= _PARENT_ID_LIMIT else None,
    }


def _payload_lines(
    path: Path, start: int, end: int
) -> Iterator[tuple[bytes, int, int]]:
    with path.open("rb") as stream:
        stream.seek(start)
        position = start
        for raw in stream:
            if position >= end:
                return
            yield raw, position, len(raw)
            position += len(raw)


def _merge_concentration(chat: bool, states: list[dict[str, Any]]) -> _Concentration:
    """Combine shard states into the state a single serial pass would have ended in."""
    merged = _Concentration(chat=chat)
    rows = sum(state["distribution"]["records"] for state in states)
    merged.chars = sum(state["chars"] for state in states)
    merged.duplicate = rows - _union(states, "exact")
    merged.normalized_duplicate = rows - _union(states, "normalized")
    merged.prefix_rows = sum(s["prefix_eligible"] for s in states) - _union(
        states, "prefixes"
    )
    if merged.chars <= _NGRAM_CHARS:
        merged.prompt_rows = sum(s["prompt_eligible"] for s in states) - _union(
            states, "prompts"
        )
        merged.answer_rows = sum(s["answer_eligible"] for s in states) - _union(
            states, "answers"
        )
        for state in states:
            merged.ngrams.update({tuple(item[:3]): item[3] for item in state["ngrams"]})
    return merged


def _merge_parents(states: list[dict[str, Any]]) -> tuple[int, int, dict[str, int]]:
    """Distinct parents, the most rows any one parent has, and the ids if few."""
    keys = [
        np.frombuffer(base64.b64decode(state["parent_digests"]), dtype="S32")
        for state in states
    ]
    counts = [np.array(state["parent_counts"], dtype=np.int64) for state in states]
    if not keys:
        return 0, 0, {}
    unique, inverse = np.unique(np.concatenate(keys), return_inverse=True)
    totals = np.bincount(inverse, weights=np.concatenate(counts)).astype(np.int64)
    ids: Counter[str] = Counter()
    if unique.size <= _PARENT_ID_LIMIT:
        for state in states:
            ids.update(state["parent_ids"])
    return int(unique.size), int(totals.max(initial=0)), dict(ids)


def _plan_views(
    root: Path, lineage_index: Path, tokenizer: Path | None, cache: Path | None
) -> tuple[list[tuple], ShardPlan, list[tuple[str, str]]]:
    """One task per run of view rows, in view/split order, and the key that names them."""
    tasks, owners, total, files = [], [], 0, {}
    generations = str(root / "generations.jsonl")
    scenarios = str(root / "scenarios.jsonl")
    for view in ("lm", "chat"):
        for split in SPLITS:
            payload, links = (
                root / view / f"{split}.jsonl",
                root / view / f"{split}.lineage.jsonl",
            )
            size = payload.stat().st_size
            for path in (payload, links):
                info = path.stat()
                files[str(path)] = [info.st_size, info.st_mtime_ns]
            if size == 0:
                if links.stat().st_size:
                    raise ValueError("view and lineage lengths disagree")
                owners.append((view, split))
                continue
            offsets, link_lines = line_boundaries(links, _LINK_STRIDE)
            if view == "lm":
                ranges = chunk_ranges(payload)
                counts = list(
                    run_sharded(
                        ShardPlan(
                            COUNT_STAGE,
                            [(str(payload), *r) for r in ranges],
                            size,
                            {"file": files[str(payload)]},
                            None if cache is None else cache / "shards",
                        ),
                        _count_lines,
                        ordered=True,
                    )
                )
            else:
                ranges, counts = [(0, size)], [link_lines]
            if sum(counts) != link_lines:
                raise ValueError("view and lineage lengths disagree")
            first = 0
            for (start, end), lines in zip(ranges, counts, strict=True):
                tasks.append(
                    (
                        view,
                        split,
                        str(payload),
                        start,
                        end,
                        str(links),
                        offsets[first // _LINK_STRIDE],
                        first % _LINK_STRIDE,
                        lines,
                        str(root / "lineage.jsonl"),
                        str(lineage_index),
                        generations,
                        scenarios,
                        str(tokenizer) if tokenizer is not None else None,
                    )
                )
                owners.append((view, split))
                first += lines
            total += size
    inputs = {
        "files": files,
        "lineage": [
            (info := (root / "lineage.jsonl").stat()).st_size,
            info.st_mtime_ns,
        ],
        "tokenizer": sha256_file(tokenizer) if tokenizer is not None else None,
        "measurement": sha256_file(Path(__file__)),
    }
    plan = ShardPlan(
        VIEW_STAGE, tasks, total, inputs, None if cache is None else cache / "shards"
    )
    return tasks, plan, owners


def measure_views(
    root: Path,
    tokenizer: Path | None = None,
    *,
    release_spec: dict[str, Any] | None = None,
    cache: Path | None = None,
) -> dict[str, Any]:
    """Count exact rendered tokenizer inputs and classify records independently.

    The view files are cut into runs of rows that worker processes measure
    independently; each run's result is a small mergeable state, kept under
    `cache` so a rerun, or a run stopped part way, measures only the runs that
    are not there. Merging reproduces the serial pass exactly.
    """
    root = Path(root)
    cache = cache or _scratch_cache()
    with tempfile.TemporaryDirectory(prefix=".describe-", dir=scratch_root()) as work:
        directory = (cache or Path(work)) / "lineage-index"
        info = (root / "lineage.jsonl").stat()
        directory = directory.with_name(
            f"lineage-index-{info.st_size}-{info.st_mtime_ns}"
        )
        directory.parent.mkdir(parents=True, exist_ok=True)
        _build_lineage_index(root / "lineage.jsonl", directory)
        tasks, plan, owners = _plan_views(root, directory, tokenizer, cache)
        states: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        results = run_sharded(plan, _measure_shard, ordered=True)
        for task, state in zip(tasks, results, strict=True):
            states[task[0], task[1]].append(state)
    return _combine(root, states, owners, tokenizer, release_spec)


def _combine(
    root: Path,
    states: dict[tuple[str, str], list[dict[str, Any]]],
    owners: list[tuple[str, str]],
    tokenizer: Path | None,
    release_spec: dict[str, Any] | None,
) -> dict[str, Any]:
    counted = tokenizer is not None
    views: dict[str, dict[str, Any]] = {"lm": {}, "chat": {}}
    training = _Distribution(counted_tokens=counted)
    for view, split in dict.fromkeys(owners):
        shards = states.get((view, split), [])
        distribution = _Distribution(counted_tokens=counted)
        families: Counter[str] = Counter()
        for state in shards:
            distribution.absorb(state["distribution"])
            families.update(state["families"])
        if (
            release_spec is not None
            and split == "train"
            and release_spec[view]["selected"]
            and split in release_spec[view]["training_splits"]
        ):
            for state in shards:
                training.absorb(state["distribution"])
        summary = distribution.result()
        summary["concentration"] = _merge_concentration(view == "chat", shards).result()
        parent_count, parent_max, parent_ids = _merge_parents(shards)
        summary["concentration"]["parent_document_counts"] = (
            dict(sorted(parent_ids.items())) if parent_count <= _PARENT_ID_LIMIT else {}
        )
        if parent_count > _PARENT_ID_LIMIT:
            summary["concentration"]["parent_document_count_reason"] = (
                "omitted_above_100k_parent_bound"
            )
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
        if parent_max > 1:
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
    summary = measure_views(root, tokenizer, release_spec=release_spec)
    with _lineage_index(root) as connection:
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
