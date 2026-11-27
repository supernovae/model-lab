"""Descriptive, read-only measurements of verified corpus release views."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from klab.canonical import canonical_digest
from klab.corpus.build_stages import implementation_digests
from klab.corpus.measurement_inputs import MeasurementTokenizer, TokenizerGeneration
from klab.corpus.normalized_documents import (
    document_chunks,
    document_range,
    source_generations,
)
from klab.corpus.provenance import HUMAN_ORIGIN, SOURCE_ORIGIN
from klab.corpus.release_verify import (
    current_settings,
    line_boundaries,
    scratch_root,
)
from klab.corpus.shards import (
    ShardPlan,
    chunk_ranges,
    declare_stage,
    range_file_identity,
    run_sharded,
    run_sharded_files,
)
from klab.result_publication import sync_directory
from klab.safe_io import read_file_range, rename_noreplace
from klab.training.manifest import sha256_file

if TYPE_CHECKING:
    from klab.corpus.release import ReleaseMembers

SPLITS = ("train", "validation", "test")
DIMENSIONS = ("origin", "verification_status", "shape")
_WORD = re.compile(r"\w+", re.UNICODE)
_NGRAM_CHARS = 20_000_000


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line:
                yield json.loads(line)


RIGHTS_STAGE = declare_stage("describe: source rights")
_ELIGIBLE = ("eligible", "eligible_with_obligations")


def _rights_range(
    path: str, start: int, end: int, tokenizer: TokenizerGeneration
) -> dict[str, list[Any]]:
    """Retained train documents in one byte range: content digest, rights state, tokens."""
    model = _tokenizer(*tokenizer)
    digests, states, tokens, seen = [], [], [], set()
    for row, _, _ in document_range(Path(path), start, end):
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
    release: ReleaseMembers,
    tokenizer: MeasurementTokenizer,
    *,
    cache: Path | None = None,
) -> dict[str, dict[str, int | None]]:
    """Measure each retained train source passage once, not its derived views.

    A passage that appears in several documents counts for the first one in file
    order, wherever its duplicates are, so the byte ranges of `documents.jsonl`
    can be measured independently and merged.
    """
    documents = release.member("documents.jsonl")
    generation = range_file_identity(documents)
    normalized = source_generations(documents)
    cache = cache or _scratch_cache()
    plan = ShardPlan(
        RIGHTS_STAGE,
        [(str(documents), *r, tokenizer.source) for r in document_chunks(documents)],
        generation[2] + sum(item[2] for item in normalized),
        {
            "documents": generation,
            "normalized_documents": normalized,
            "tokenizer": tokenizer.identity(),
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
LINK_STAGE = declare_stage("describe: view links")
VIEW_STAGE = declare_stage("describe: view rows")
_PARENT_ID_LIMIT = 100_000
# Every Nth link line gets an offset, so a shard finds its first link by skipping
# fewer than this many lines.
_LINK_STRIDE = 4096
_INDEX_FIELDS = (("hashes", np.uint64), ("offsets", np.uint64), ("lengths", np.uint32))


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


def _write_index_range(
    staging: Path, path: str, start: int, end: int
) -> dict[str, int]:
    columns = _index_range(path, start, end)
    rows = np.empty(len(columns[0]), dtype=list(_INDEX_FIELDS))
    for (name, _), column in zip(_INDEX_FIELDS, columns, strict=True):
        rows[name] = column
    np.save(staging / "rows.npy", rows)
    return {"rows": rows.size}


def _lineage_columns(plan: ShardPlan) -> tuple[np.ndarray, ...]:
    if plan.store is None:
        parts = list(run_sharded(plan, _index_range, ordered=True))
        return tuple(
            np.concatenate([part[index] for part in parts])
            if parts
            else np.empty(0, dtype)
            for index, (_, dtype) in enumerate(_INDEX_FIELDS)
        )

    ranges = list(run_sharded_files(plan, _write_index_range, ordered=True))
    size = sum(result["rows"] for _, result in ranges)
    columns = tuple(np.empty(size, dtype) for _, dtype in _INDEX_FIELDS)
    start = 0
    for directory, result in ranges:
        rows = np.load(directory / "rows.npy", mmap_mode="r")
        end = start + result["rows"]
        for (name, _), column in zip(_INDEX_FIELDS, columns, strict=True):
            column[start:end] = rows[name]
        del rows
        start = end
    return columns


def _build_lineage_index(
    lineage: Path, work: Path, *, cache: Path | None = None
) -> Path:
    """Assemble the sorted lookup, with verified range checkpoints in persistent scratch."""
    lineage = lineage.resolve()
    generation = range_file_identity(lineage)
    inputs = {
        "file": generation,
        "lineage": str(lineage),
        "producer": implementation_digests("measurement.py"),
    }
    directory = cache or work
    index = directory / f"lineage-index-{canonical_digest(inputs)}"
    staging = index.with_suffix(".pending")
    runs = None if cache is None else index.with_suffix(".ranges")
    directory.mkdir(parents=True, exist_ok=True)
    with index.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if staging.exists():
            shutil.rmtree(staging)
        complete = (index / "complete").is_file() and all(
            (index / f"{name}.npy").is_file() for name, _ in _INDEX_FIELDS
        )
        if not complete:
            plan = ShardPlan(
                INDEX_STAGE,
                [(str(lineage), *r) for r in chunk_ranges(lineage)],
                generation[2],
                inputs,
                runs,
            )
            columns = _lineage_columns(plan)
            order = np.argsort(columns[0], kind="stable")
            staging.mkdir()
            try:
                for (name, _), column in zip(_INDEX_FIELDS, columns, strict=True):
                    with (staging / f"{name}.npy").open("xb") as stream:
                        np.save(stream, column[order])
                        if runs is not None:
                            stream.flush()
                            os.fsync(stream.fileno())
                (staging / "complete").touch()
                if runs is not None:
                    sync_directory(staging)
                if range_file_identity(lineage) != generation:
                    raise ValueError("lineage changed during index construction")
                if index.exists():
                    shutil.rmtree(index)
                rename_noreplace(staging, index)
                if runs is not None:
                    sync_directory(directory)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        if runs is not None and runs.exists():
            shutil.rmtree(runs)
    return index


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


@lru_cache(maxsize=1)
def _tokenizer(path: str, generation: tuple[int, ...]) -> Any:
    # One thread per worker process: the pool already uses every core.
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    from tokenizers import Tokenizer

    file = Path(path)
    if tuple(range_file_identity(file)) != generation:
        raise ValueError("tokenizer changed before Corpus measurement")
    model = Tokenizer.from_file(path)
    if tuple(range_file_identity(file)) != generation:
        raise ValueError("tokenizer changed while loading for Corpus measurement")
    return model


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
    tokenizer: TokenizerGeneration | None,
) -> dict[str, Any]:
    """Measure one run of consecutive view rows and return a mergeable state."""
    model = _tokenizer(*tokenizer) if tokenizer is not None else None
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


def _scan_view_links(
    path: str, stride: int, generation: list[int]
) -> tuple[list[int], int]:
    links = Path(path)
    result = line_boundaries(links, stride)
    if range_file_identity(links) != generation:
        raise ValueError(f"view links changed during measurement: {links}")
    return result


def _plan_views(
    member: Callable[[str], Path],
    lineage: Path,
    lineage_index: Path,
    tokenizer: MeasurementTokenizer | None,
    cache: Path | None,
) -> tuple[list[tuple], ShardPlan, list[tuple[str, str]]]:
    """One task per run of view rows, in view/split order, and the key that names them."""
    tasks, owners, total = [], [], 0
    generations = str(member("generations.jsonl"))
    scenarios = str(member("scenarios.jsonl"))
    views = [
        (
            view,
            split,
            member(f"{view}/{split}.jsonl"),
            member(f"{view}/{split}.lineage.jsonl"),
        )
        for view in ("lm", "chat")
        for split in SPLITS
    ]
    files = {
        str(path): range_file_identity(path)
        for _, _, payload, links in views
        for path in (payload, links)
    }
    link_tasks = [
        (str(links), _LINK_STRIDE, files[str(links)])
        for _, _, payload, links in views
        if files[str(payload)][2]
    ]
    implementation = implementation_digests("measurement.py", "release_verify.py")
    link_plan = ShardPlan(
        LINK_STAGE,
        link_tasks,
        sum(generation[2] for _, _, generation in link_tasks),
        implementation,
        None if cache is None else cache / "shards",
        # Keep sequential reads and avoid starting a pool for a few split files.
        operator_cap=1,
    )
    indexed_links = dict(
        zip(
            (name for name, _, _ in link_tasks),
            run_sharded(link_plan, _scan_view_links, ordered=True),
            strict=True,
        )
    )
    for view, split, payload, links in views:
        size = files[str(payload)][2]
        if size == 0:
            if files[str(links)][2]:
                raise ValueError("view and lineage lengths disagree")
            owners.append((view, split))
            continue
        offsets, link_lines = indexed_links[str(links)]
        if view == "lm":
            ranges = chunk_ranges(payload)
            counts = list(
                run_sharded(
                    ShardPlan(
                        COUNT_STAGE,
                        [(str(payload), *r) for r in ranges],
                        size,
                        {
                            "file": files[str(payload)],
                            "measurement": implementation["measurement.py"],
                        },
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
                    str(lineage),
                    str(lineage_index),
                    generations,
                    scenarios,
                    tokenizer.source if tokenizer is not None else None,
                )
            )
            owners.append((view, split))
            first += lines
        total += size
    inputs = {
        # Empty outputs are recreated by a resumed native build. They cannot
        # affect a nonempty view run, so their file generations are immaterial.
        "files": {
            name: generation for name, generation in files.items() if generation[2]
        },
        "lineage": range_file_identity(lineage),
        "generations": _nonempty_generation(Path(generations)),
        "scenarios": _nonempty_generation(Path(scenarios)),
        "tokenizer": tokenizer.identity() if tokenizer is not None else None,
        "measurement": implementation["measurement.py"],
    }
    plan = ShardPlan(
        VIEW_STAGE, tasks, total, inputs, None if cache is None else cache / "shards"
    )
    return tasks, plan, owners


def _nonempty_generation(path: Path) -> list[int] | None:
    generation = range_file_identity(path)
    return generation if generation[2] else None


def measure_views(
    release: ReleaseMembers,
    tokenizer: MeasurementTokenizer | None = None,
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
    with _measured_views(release.member, tokenizer, release_spec, cache) as (
        summary,
        _,
        _,
    ):
        return summary


@contextmanager
def _measured_views(
    member: Callable[[str], Path],
    tokenizer: MeasurementTokenizer | None,
    release_spec: dict[str, Any] | None,
    cache: Path | None,
) -> Iterator[tuple[dict[str, Any], Path, Path]]:
    cache = cache or _scratch_cache()
    with tempfile.TemporaryDirectory(prefix=".describe-", dir=scratch_root()) as work:
        lineage = member("lineage.jsonl")
        directory = _build_lineage_index(lineage, Path(work), cache=cache)
        tasks, plan, owners = _plan_views(member, lineage, directory, tokenizer, cache)
        states: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        results = run_sharded(plan, _measure_shard, ordered=True)
        for task, state in zip(tasks, results, strict=True):
            states[task[0], task[1]].append(state)
        yield (
            _combine(states, owners, tokenizer, release_spec),
            directory,
            Path(work),
        )


def _combine(
    states: dict[tuple[str, str], list[dict[str, Any]]],
    owners: list[tuple[str, str]],
    tokenizer: MeasurementTokenizer | None,
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
        result["tokenizer_sha256"] = tokenizer.digest
    return result


def summarize_release(
    root: Path,
    *,
    release_spec: dict[str, Any] | None = None,
    tokenizer: Path | None = None,
    cache: Path | None = None,
) -> dict[str, Any]:
    """Frozen descriptive census, with measured tokens only for a pinned tokenizer."""
    root = Path(root)
    bound = MeasurementTokenizer.from_path(tokenizer) if tokenizer is not None else None
    with _measured_views(root.joinpath, bound, release_spec, cache) as (
        summary,
        index,
        work,
    ):
        training_views = [
            view
            for view, splits in summary["views"].items()
            if release_spec is not None
            and release_spec[view]["selected"]
            and "train" in release_spec[view]["training_splits"]
            and set(splits.get("train", {}).get("dimensions", {}).get("origin", {}))
            - {SOURCE_ORIGIN, HUMAN_ORIGIN}
        ]
        if training_views:
            generated = _generated_lineage(root, index, work, training_views)
        else:
            generated = (0, 0, [], {})
    if bound is not None:
        bound.require_unchanged()
    return _summarize_release(root, summary, *generated)


def _generated_lineage(
    root: Path, index: Path, work: Path, training_views: list[str]
) -> tuple[int, int, list[str], dict[str, dict[str, list[str]]]]:
    heldout_keys = (
        "parent_document_ids",
        "source_family_ids",
        "generator_world_id",
        "scenario_family_id",
        "template_family_id",
    )
    heldout: dict[str, set[str]] = {key: set() for key in heldout_keys}
    lineage = root / "lineage.jsonl"
    lookup = _LineageIndex(lineage, index)
    with lookup.stream:
        for view in ("lm", "chat"):
            for link in _rows(root / view / "test.lineage.jsonl"):
                row = lookup.get(link["record_id"])
                if row is not None:
                    for key in heldout_keys:
                        values = (
                            row.get(key, []) if key.endswith("_ids") else [row.get(key)]
                        )
                        heldout[key].update(
                            value for value in values if value is not None
                        )
    generated_count = unverified_generated = 0
    shared: dict[str, dict[str, set[str]]] = {}
    with closing(sqlite3.connect(work / "training-ids.sqlite")) as connection:
        # Membership is on disk; the training lineage payload is read sequentially.
        connection.execute("CREATE TABLE selected (id TEXT PRIMARY KEY) WITHOUT ROWID")
        for view in training_views:
            connection.executemany(
                "INSERT OR IGNORE INTO selected VALUES (?)",
                (
                    (link["record_id"],)
                    for link in _rows(root / view / "train.lineage.jsonl")
                ),
            )
        for row in _rows(lineage):
            if (
                row["origin"] in {SOURCE_ORIGIN, HUMAN_ORIGIN}
                or not connection.execute(
                    "SELECT 1 FROM selected WHERE id = ?", (row["record_id"],)
                ).fetchone()
            ):
                continue
            generated_count += 1
            unverified_generated += row["verification"]["status"] in {
                "unverified",
                "schema_validated",
            }
            shape = row["shape"]["id"]
            groups = shared.setdefault(shape, {key: set() for key in heldout_keys})
            for key in heldout_keys:
                values = row.get(key, []) if key.endswith("_ids") else [row.get(key)]
                groups[key].update(value for value in values if value in heldout[key])
    shared_by_shape = {
        shape: {key: sorted(values) for key, values in groups.items()}
        for shape, groups in sorted(shared.items())
    }
    generated_parents = {
        parent for groups in shared.values() for parent in groups["parent_document_ids"]
    }
    return (
        generated_count,
        unverified_generated,
        sorted(generated_parents),
        shared_by_shape,
    )


def _summarize_release(
    root: Path,
    summary: dict[str, Any],
    generated_count: int,
    unverified_generated: int,
    generated_test_overlap: list[str],
    shared_by_shape: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    views = summary["views"]
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for splits in views.values():
        for item in splits.values():
            for dimension, values in item["dimensions"].items():
                for value, count in values.items():
                    totals[dimension][value] += count["records"]
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
