"""Read-only lexical screening of cross-split normalized document overlap.

Lowercase Unicode word tokens form sets of three-word shingles (short texts use
all their words). Sixty-four deterministic MinHashes in sixteen four-hash bands
propose pairs; exact shingle Jaccard scores every proposed pair. Identical text
hashes independently propose pairs. LSH can miss similar pairs: this is not a
proof of semantic independence, and neither paraphrases nor all lexical matches
are guaranteed to appear.

The screen indexes the held-out splits (validation and test) and streams the
training split past that index, so memory grows with the held-out side and the
longest single document, not with the corpus. Training documents are compared
with held-out ones and held-out splits with each other; training documents are
never compared with each other. Every document is screened: a held-out side
over its caps raises before anything is reported. Both passes use line-aligned
byte shards. The checkpointed first pass writes compact evidence through the
shared runner. Forked training-pass workers share the held-out index, and the
report is the same for any number of workers.

Template-aware mode also scores each pair on its content: shingles that are
boilerplate (very frequent among the held-out documents or a deterministic
sample of the training documents of a source) are set aside, and a pair that is
similar only through them is counted as `template_only` instead of a candidate.
"""

from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import multiprocessing
import os
import re
import shutil
import struct
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.build_stages import implementation_digests
from klab.corpus.release_verify import verification_settings
from klab.corpus.shards import (
    ShardPlan,
    declare_stage,
    range_file_identity,
    run_sharded_files,
)

EVIDENCE_STAGE = declare_stage("near-duplicate evidence")

_SPLITS = ("train", "validation", "test")
_HELD_OUT = ("validation", "test")
_HASHES = 64
_BANDS = 16
_ROWS = _HASHES // _BANDS
# Shingles whose MinHashes are computed at once; bounds the working matrix.
_BLOCK = 8192
_WORD_CACHE = 1 << 20
_DEFAULT_SAMPLE_STRIDE = 256


def _seeded(label: str, count: int) -> np.ndarray:
    return np.array(
        [
            int.from_bytes(
                hashlib.sha256(f"near-duplicate-v2:{label}:{i}".encode()).digest()[:8],
                "big",
            )
            | 1
            for i in range(count)
        ],
        dtype=np.uint64,
    )


_POSITION = _seeded("position", 3)
_MULTIPLY = _seeded("multiply", _HASHES)[:, None]
_ADD = _seeded("add", _HASHES)[:, None]


def _mix(values: np.ndarray) -> np.ndarray:
    """SplitMix64's finalizer: spreads every input bit over the output."""
    values = values ^ (values >> np.uint64(30))
    values = values * np.uint64(0xBF58476D1CE4E5B9)
    values = values ^ (values >> np.uint64(27))
    values = values * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


class _Shingler:
    """Turns text into its sorted, distinct three-word shingle hashes."""

    def __init__(self) -> None:
        self._words: dict[str, int] = {}

    def _word(self, word: str) -> int:
        value = self._words.get(word)
        if value is None:
            if len(self._words) >= _WORD_CACHE:
                self._words.clear()
            value = int.from_bytes(
                hashlib.blake2b(word.encode(), digest_size=8).digest(), "big"
            )
            self._words[word] = value
        return value

    def __call__(self, text: str) -> np.ndarray:
        words = re.findall(r"\w+", text.casefold())
        if not words:
            return np.empty(0, dtype=np.uint64)
        hashes = np.fromiter(map(self._word, words), dtype=np.uint64, count=len(words))
        width = min(3, len(words))
        count = len(words) - width + 1
        combined = np.zeros(count, dtype=np.uint64)
        for offset in range(width):
            combined += hashes[offset : offset + count] * _POSITION[offset]
        return np.unique(_mix(combined))


def _band_keys(shingles: np.ndarray) -> list[int]:
    if not len(shingles):
        return []
    signature = np.full(_HASHES, np.iinfo(np.uint64).max, dtype=np.uint64)
    for start in range(0, len(shingles), _BLOCK):
        block = shingles[start : start + _BLOCK][None, :]
        np.minimum(signature, (block * _MULTIPLY + _ADD).min(axis=1), out=signature)
    # One 64-bit key per band. Keys that collide only propose extra pairs, and
    # every proposed pair is scored exactly.
    rows = signature.reshape(_BANDS, _ROWS)
    combined = np.arange(_BANDS, dtype=np.uint64)
    for row in range(_ROWS):
        combined = _mix(combined ^ rows[:, row])
    return combined.tolist()


_FIELDS = ("document_id", "split", "source_id", "source_location", "content_sha256")


def _order(pair: dict[str, Any]) -> tuple[str, str]:
    return pair["left_document_id"], pair["right_document_id"]


def _coverage_key(row: dict[str, Any]) -> tuple[str, str]:
    return row["split"], str(row.get("source_id"))


class Checkpoint:
    """Progress of the training-split pass, saved so a killed screen resumes.

    Each shard of ``documents.jsonl`` has its own file. A file is trusted only
    when its digest verifies and its key equals the key of this exact screen
    (release, algorithm, every parameter and the shard plan); anything else is
    ignored and the shard starts over. The state is every accumulator of the
    shard, so a resumed screen reports exactly what an uninterrupted one would.
    """

    def __init__(self, path: Path, key: str, every: int = 200_000) -> None:
        if every < 1:
            raise ValueError("checkpoint interval must be at least 1 document")
        self.path, self.key, self.every = Path(path), key, every

    def shard(self, index: int) -> Checkpoint:
        name = f"{self.path.stem}-shard-{index:06d}{self.path.suffix}"
        return Checkpoint(self.path.with_name(name), self.key, self.every)

    def load(self) -> dict[str, Any] | None:
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            body = saved["state"]
            if (
                saved["key"] != self.key
                or saved["sha256"] != canonical_digest(body)
                or body["key"] != self.key
            ):
                return None
            return body
        except OSError, ValueError, KeyError, TypeError:
            return None

    def save(self, state: dict[str, Any]) -> None:
        state = {"key": self.key, **state}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "wb", dir=self.path.parent, prefix=".checkpoint-", delete=False
        ) as handle:
            handle.write(
                canonical_json(
                    {"key": self.key, "sha256": canonical_digest(state), "state": state}
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


_COUNTERS = ("documents", "text_bytes", "without_words")


def _better(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge two summaries of one held-out document's template-only pairs."""
    closest = min(
        (a, b), key=lambda e: (-e["closest_content_jaccard"], e["closest_document_id"])
    )
    return {
        "document_id": a["document_id"],
        "split": a["split"],
        "source_id": a["source_id"],
        "pairs": a["pairs"] + b["pairs"],
        "max_jaccard": max(a["max_jaccard"], b["max_jaccard"]),
        "min_boilerplate_share": min(
            a["min_boilerplate_share"], b["min_boilerplate_share"]
        ),
        "closest_document_id": closest["closest_document_id"],
        "closest_content_jaccard": closest["closest_content_jaccard"],
    }


class _Tally:
    """Everything one stretch of the screen accumulates; shards merge exactly."""

    def __init__(self, max_candidates: int) -> None:
        self.max_candidates = max_candidates
        self.found: list[dict[str, Any]] = []
        self.candidate_count = 0
        self.identical_count = 0
        self.comparisons = 0
        self.template_only_count = 0
        self.classes: Counter[str] = Counter()
        self.template_docs: dict[str, dict[str, Any]] = {}
        self.counters: dict[str, Counter[tuple[str, str]]] = {
            name: Counter() for name in _COUNTERS
        }
        self.trained = 0

    def keep_listed(self, limit: int | None = None) -> None:
        self.found.sort(key=_order)
        del self.found[limit or self.max_candidates :]

    def add_found(self, pair: dict[str, Any]) -> None:
        self.found.append(pair)
        if len(self.found) >= 2 * self.max_candidates:
            self.keep_listed()

    def add_template_only(self, entry: dict[str, Any]) -> None:
        self.template_only_count += entry["pairs"]
        known = self.template_docs.get(entry["document_id"])
        self.template_docs[entry["document_id"]] = (
            entry if known is None else _better(known, entry)
        )

    def account(self, row: dict[str, Any], shingles: np.ndarray) -> None:
        key = _coverage_key(row)
        self.counters["documents"][key] += 1
        self.counters["text_bytes"][key] += len(row["text"].encode("utf-8"))
        if not len(shingles):
            self.counters["without_words"][key] += 1

    def merge(self, other: _Tally) -> None:
        self.found += other.found
        self.keep_listed()
        for name in (
            "candidate_count",
            "identical_count",
            "comparisons",
            "template_only_count",
            "trained",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.classes.update(other.classes)
        for entry in other.template_docs.values():
            known = self.template_docs.get(entry["document_id"])
            self.template_docs[entry["document_id"]] = (
                entry if known is None else _better(known, entry)
            )
        for name in _COUNTERS:
            self.counters[name].update(other.counters[name])

    def state(self) -> dict[str, Any]:
        self.keep_listed()
        return {
            "found": self.found,
            "candidate_count": self.candidate_count,
            "identical_count": self.identical_count,
            "comparisons": self.comparisons,
            "template_only_count": self.template_only_count,
            "classes": dict(sorted(self.classes.items())),
            "template_docs": [
                self.template_docs[k] for k in sorted(self.template_docs)
            ],
            "trained": self.trained,
            **{
                name: [[a, b, n] for (a, b), n in sorted(counter.items())]
                for name, counter in self.counters.items()
            },
        }

    @classmethod
    def from_state(cls, saved: dict[str, Any], max_candidates: int) -> _Tally:
        tally = cls(max_candidates)
        tally.found = list(saved["found"])
        for name in (
            "candidate_count",
            "identical_count",
            "comparisons",
            "template_only_count",
            "trained",
        ):
            setattr(tally, name, saved[name])
        tally.classes.update(saved["classes"])
        tally.template_docs = {e["document_id"]: e for e in saved["template_docs"]}
        for name in _COUNTERS:
            tally.counters[name].update({(a, b): n for a, b, n in saved[name]})
        return tally


def _member(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    if not len(sorted_values):
        return np.zeros(len(values), dtype=bool)
    at = np.searchsorted(sorted_values, values)
    at[at == len(sorted_values)] = 0
    return sorted_values[at] == values


def _frequent(arrays: list[np.ndarray], at_least: int) -> np.ndarray:
    """Sorted shingles present in at least `at_least` of the arrays.

    Each array holds distinct values, so a shingle's count is its document
    frequency. Hash-partitioned so the working set is a sixteenth of the input.
    """
    found = []
    for part in range(16):
        top = np.uint64(part)
        pieces = [a[(a >> np.uint64(60)) == top] for a in arrays]
        values, counts = np.unique(np.concatenate(pieces), return_counts=True)
        found.append(values[counts >= at_least])
    return np.concatenate(found)


def _salt(label: str) -> np.uint64:
    return np.uint64(
        int.from_bytes(hashlib.blake2b(label.encode(), digest_size=8).digest(), "big")
    )


class _Sketch:
    """Bounded document-frequency counts of sampled training shingles.

    A two-row count-min sketch of saturating one-byte counters, keyed by source
    and shingle. Collisions can only raise a count, so a shingle can be taken
    for boilerplate wrongly (rarely: both rows must reach the threshold), never
    the reverse.
    """

    def __init__(self, bits: int) -> None:
        self.shift = np.uint64(64 - bits)
        self.rows = [np.zeros(1 << bits, dtype=np.uint8) for _ in range(2)]

    def _slots(self, source: str, shingles: np.ndarray) -> list[np.ndarray]:
        salt = _salt(source)
        first = _mix(shingles ^ salt) >> self.shift
        second = _mix(shingles * _POSITION[0] + salt) >> self.shift
        return [first.astype(np.intp), second.astype(np.intp)]

    def add(self, source: str, shingles: np.ndarray) -> None:
        """Count one document per occurrence; `shingles` joins sampled documents."""
        for row, slots in zip(self.rows, self._slots(source, shingles), strict=True):
            at, counts = np.unique(slots, return_counts=True)
            row[at] = np.minimum(row[at].astype(np.int64) + counts, 255).astype(
                np.uint8
            )

    def count(self, source: str, shingles: np.ndarray) -> np.ndarray:
        first, second = (
            row[slots]
            for row, slots in zip(self.rows, self._slots(source, shingles), strict=True)
        )
        return np.minimum(first, second)


class _Options:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class _Index:
    """The held-out documents, their LSH buckets and what counts as boilerplate."""

    def __init__(self, options: _Options) -> None:
        self.options = options
        self.held: list[dict[str, Any]] = []
        self.held_shingles: list[np.ndarray] = []
        self.buckets: dict[int, list[int]] = defaultdict(list)
        self.identical: dict[str, list[int]] = defaultdict(list)
        self.frequent: dict[str, np.ndarray] = {}
        self.sketch: _Sketch | None = None
        self.sampled: Counter[str] = Counter()

    def find_boilerplate(self, sample: _Sketch | None, sampled: Counter[str]) -> None:
        by_source: dict[str, list[np.ndarray]] = defaultdict(list)
        for row, shingles in zip(self.held, self.held_shingles, strict=True):
            by_source[row["source_id"]].append(shingles)
        self.frequent = {
            source: _frequent(arrays, self.options.boilerplate_heldout_documents)
            for source, arrays in sorted(by_source.items())
        }
        self.sketch, self.sampled = sample, sampled

    def boilerplate(self, sources: Iterable[str], shingles: np.ndarray) -> np.ndarray:
        """Which of `shingles` are template text under any of the `sources`."""
        found = np.zeros(len(shingles), dtype=bool)
        for source in set(sources):
            if source in self.frequent:
                found |= _member(self.frequent[source], shingles)
            if self.sketch is not None and self.sampled[source]:
                found |= (
                    self.sketch.count(source, shingles)
                    >= self.options.boilerplate_sample_documents
                )
        return found

    def keys(self, source: str, shingles: np.ndarray) -> list[int]:
        keys = _band_keys(shingles)
        if self.options.template_aware and len(shingles):
            template = self.boilerplate((source,), shingles)
            if template.any():
                # The same text in different surroundings must still meet.
                keys += _band_keys(shingles[~template])
        return list(dict.fromkeys(keys))

    def proposals(self, digest: str, keys: list[int]) -> list[int]:
        matches = set(self.identical.get(digest, ()))
        for key in keys:
            matches.update(self.buckets.get(key, ()))
        return sorted(matches)

    def add(self, row: dict[str, Any], shingles: np.ndarray, keys: list[int]) -> None:
        index = len(self.held)
        self.held.append({field: row.get(field) for field in _FIELDS})
        self.held_shingles.append(shingles)
        self.identical[row["content_sha256"]].append(index)
        for key in keys:
            self.buckets[key].append(index)

    def compare(
        self,
        tally: _Tally,
        row: dict[str, Any],
        shingles: np.ndarray,
        other: int,
    ) -> None:
        tally.comparisons += 1
        subject, right = self.held[other], row
        a, b = self.held_shingles[other], shingles
        exact = subject["content_sha256"] == right["content_sha256"]
        shared = np.intersect1d(a, b, assume_unique=True)
        union = len(a) + len(b) - len(shared)
        similarity = len(shared) / union if union else float(exact)
        extra: dict[str, Any] = {}
        if self.options.template_aware:
            sources = (subject["source_id"], right.get("source_id"))
            in_a = self.boilerplate(sources, a)
            in_b = self.boilerplate(sources, b)
            template_shared = int(self.boilerplate(sources, shared).sum())
            content_shared = len(shared) - template_shared
            content_union = (
                len(a) - int(in_a.sum()) + len(b) - int(in_b.sum()) - content_shared
            )
            content = content_shared / content_union if content_union else None
            share = template_shared / len(shared) if len(shared) else 0.0
            reached = similarity >= self.options.min_jaccard
            if exact or (content is not None and content >= self.options.min_jaccard):
                kind = "leak"
            elif not reached:
                return
            elif content is None:
                # No content on either side leaves nothing to tell them apart by.
                kind = "content_too_short"
            elif share >= self.options.boilerplate_min_share:
                tally.add_template_only(
                    {
                        "document_id": subject["document_id"],
                        "split": subject["split"],
                        "source_id": subject["source_id"],
                        "pairs": 1,
                        "max_jaccard": similarity,
                        "min_boilerplate_share": share,
                        "closest_document_id": right["document_id"],
                        "closest_content_jaccard": content,
                    }
                )
                return
            else:
                # Template explains too little of the match to set it aside.
                kind = "partly_template"
            extra = {
                "class": kind,
                "content_jaccard": content,
                "boilerplate_share": share,
            }
        elif not (exact or similarity >= self.options.min_jaccard):
            return
        tally.candidate_count += 1
        tally.classes[extra.get("class", "leak")] += 1
        tally.identical_count += exact
        left = subject
        if right["document_id"] < left["document_id"]:
            left, right = right, left
        tally.add_found(
            {
                **{f"left_{field}": left.get(field) for field in _FIELDS},
                **{f"right_{field}": right.get(field) for field in _FIELDS},
                "jaccard": similarity,
                "shared_shingles": len(shared),
                "union_shingles": union,
                "identical_content_sha256": exact,
                **extra,
            }
        )


_JOB: dict[str, Any] = {}


def _sampled(document_id: str, stride: int) -> bool:
    digest = hashlib.blake2b(document_id.encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") % stride == 0


def _scan_rows(rows, sample_stride: int | None, add_held, add_sample):
    """Read one shard into its held-out shingle evidence and training sample."""
    shingler = _Shingler()
    tally = _Tally(0)  # This pass records coverage and proposes no candidates.
    sampled: Counter[str] = Counter()
    for _, row in rows:
        if row["split"] not in _SPLITS:
            raise ValueError("near-duplicate invalid document split")
        if row["split"] == "train":
            if sample_stride is not None and _sampled(
                row["document_id"], sample_stride
            ):
                source = str(row.get("source_id"))
                add_sample(source, shingler(row["text"]))
                sampled[source] += 1
            continue
        shingles = shingler(row["text"])
        add_held({field: row.get(field) for field in _FIELDS}, shingles)
        tally.account(row, shingles)
    return {"tally": tally.state(), "sampled": sampled}


def _scan(shard: int) -> dict[str, Any]:
    held: list[tuple[dict[str, Any] | str, np.ndarray]] = []
    sample: dict[str, list[np.ndarray]] = defaultdict(list)
    result = _scan_rows(
        _JOB["read"](*_JOB["shards"][shard]),
        _JOB["sample_stride"],
        lambda fields, shingles: held.append((fields, shingles)),
        lambda source, shingles: sample[source].append(shingles),
    )
    held.extend((source, np.concatenate(parts)) for source, parts in sample.items())
    return {**result, "evidence": held}


_EVIDENCE_HEADER = struct.Struct("<II")


def _write_evidence(stream: BinaryIO, fields, shingles: np.ndarray) -> None:
    header = canonical_json(fields)
    stream.write(_EVIDENCE_HEADER.pack(len(header), len(shingles)))
    stream.write(header)
    stream.write(memoryview(shingles.astype("<u8", copy=False)))


def _read_evidence(path: Path):
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        offset = 0
        while raw := stream.read(_EVIDENCE_HEADER.size):
            length, count = _EVIDENCE_HEADER.unpack(raw)
            offset += len(raw) + length + count * 8
            if offset > size:
                raise ValueError("truncated near-duplicate shingle evidence")
            fields = json.loads(stream.read(length))
            values = stream.read(count * 8)
            if len(values) != count * 8:
                raise ValueError("truncated near-duplicate shingle evidence")
            yield fields, np.frombuffer(values, dtype="<u8")


def _scan_file(staging, path, start, end, sample_stride):
    with (staging / "evidence.bin").open("xb") as stream:

        def emit(fields, shingles):
            _write_evidence(stream, fields, shingles)

        return _scan_rows(
            _document_rows(Path(path), start, end),
            sample_stride,
            emit,
            emit,
        )


def _scan_results(shards, workers, sample_stride, evidence):
    if evidence is None:
        yield from _run(_scan, len(shards), workers)
        return
    plan = replace(evidence, tasks=[(*task, sample_stride) for task in evidence.tasks])
    with (
        verification_settings(workers=workers),
        closing(run_sharded_files(plan, _scan_file, ordered=True)) as segments,
    ):
        for directory, result in segments:
            with closing(_read_evidence(directory / "evidence.bin")) as rows:
                yield {**result, "evidence": rows}


def _stream(shard: int) -> dict[str, Any]:
    """Second pass over one shard's training documents, resumable by offset."""
    options: _Options = _JOB["options"]
    index: _Index = _JOB["index"]
    checkpoint: Checkpoint | None = _JOB["checkpoint"]
    checkpoint = checkpoint.shard(shard) if checkpoint is not None else None
    start, end = _JOB["shards"][shard]
    tally = _Tally(options.max_candidates)
    saved = checkpoint.load() if checkpoint is not None else None
    if saved is not None:
        tally = _Tally.from_state(saved, options.max_candidates)
        start = saved["offset"]
        if saved["done"]:
            return tally.state()
    shingler = _Shingler()
    seen = _JOB["seen"]
    for position, row in _JOB["read"](start, end):
        if row["split"] != "train":
            continue
        if row["document_id"] in seen:
            raise ValueError("near-duplicate duplicate document ID")
        shingles = shingler(row["text"])
        keys = index.keys(str(row.get("source_id")), shingles)
        for other in index.proposals(row["content_sha256"], keys):
            index.compare(tally, row, shingles, other)
        tally.account(row, shingles)
        tally.trained += 1
        if checkpoint is not None and tally.trained % checkpoint.every == 0:
            checkpoint.save({**tally.state(), "offset": position, "done": False})
    state = tally.state()
    if checkpoint is not None:
        checkpoint.save({**state, "offset": end, "done": True})
    return state


def _run(function: Callable[[int], Any], count: int, workers: int) -> Iterable[Any]:
    """Results of `function` over the shards, in shard order.

    Workers are forked, so they share the job and the index read-only.
    """
    if workers <= 1 or count <= 1:
        yield from map(function, range(count))
        return
    with multiprocessing.get_context("fork").Pool(min(workers, count)) as pool:
        yield from pool.imap(function, range(count), chunksize=1)


def screen(
    read: Callable[[int, int], Iterable[tuple[int, dict[str, Any]]]],
    shards: Sequence[tuple[int, int]],
    *,
    max_heldout_documents: int = 1_000_000,
    max_heldout_shingles: int = 200_000_000,
    max_candidates: int = 20_000,
    min_jaccard: float = 0.65,
    template_aware: bool = False,
    boilerplate_heldout_documents: int = 20,
    boilerplate_sample_documents: int = 4,
    boilerplate_sample_stride: int = _DEFAULT_SAMPLE_STRIDE,
    boilerplate_min_share: float = 0.5,
    boilerplate_sketch_bits: int = 27,
    workers: int = 1,
    checkpoint: Checkpoint | None = None,
    _evidence: ShardPlan | None = None,
) -> dict[str, Any]:
    """Screen every normalized document, including excluded or duplicate records.

    ``read(start, end)`` yields ``(end offset, row)`` for the documents whose
    lines start in ``[start, end)`` of the source; ``shards`` are consecutive
    line-aligned ranges that cover it. Both passes use up to ``workers``
    processes. The first indexes the held-out splits and
    refuses past ``max_heldout_documents`` or ``max_heldout_shingles``: the
    index is about 8 bytes per held-out shingle plus about 4 KB per held-out
    document, shared by the workers. The second streams the training split past
    that index; with a ``checkpoint`` each shard saves its progress and a later
    call resumes the unfinished shards only. Shards merge in a fixed order, so
    the report does not depend on ``workers``. Every candidate is counted; the
    first ``max_candidates`` in document-ID order are listed.
    The threshold deliberately retains ambiguous lexical overlaps for review.

    With ``template_aware``, shingles that are boilerplate are left out of a
    second, content similarity. A shingle is boilerplate in a source when at
    least ``boilerplate_heldout_documents`` held-out documents of the source
    contain it, or when at least ``boilerplate_sample_documents`` of the training
    documents sampled one in ``boilerplate_sample_stride`` (by document ID) do;
    the sample is counted in a fixed-size sketch of ``2 ** boilerplate_sketch_bits``
    bytes per row. A pair at the threshold whose content similarity is below it
    is ``template_only`` when boilerplate is at least ``boilerplate_min_share``
    of what the two share (else ``partly_template``): counted and summarized per
    held-out document, and not a candidate. Identical text, and pairs with no content on either side,
    always stay candidates.
    """
    if (
        min(
            max_heldout_documents,
            max_heldout_shingles,
            max_candidates,
            boilerplate_heldout_documents,
            boilerplate_sample_documents,
            boilerplate_sample_stride,
            workers,
        )
        < 1
        or not 0 < min_jaccard <= 1
        or not 0 <= boilerplate_min_share <= 1
        or not 8 <= boilerplate_sketch_bits <= 32
    ):
        raise ValueError("near-duplicate caps must be positive and threshold in (0, 1]")
    options = _Options(
        max_candidates=max_candidates,
        min_jaccard=min_jaccard,
        template_aware=template_aware,
        boilerplate_heldout_documents=boilerplate_heldout_documents,
        boilerplate_sample_documents=boilerplate_sample_documents,
        boilerplate_min_share=boilerplate_min_share,
    )
    index = _Index(options)
    base = _Tally(max_candidates)
    sketch = _Sketch(boilerplate_sketch_bits) if template_aware else None
    sample_stride = boilerplate_sample_stride if template_aware else None
    sampled: Counter[str] = Counter()
    seen: set[str] = set()
    _JOB.clear()
    _JOB.update(
        read=read, shards=list(shards), options=options, sample_stride=sample_stride
    )
    held: list[tuple[dict[str, Any], np.ndarray]] = []
    total_shingles = 0
    with closing(_scan_results(shards, workers, sample_stride, _evidence)) as results:
        for result in results:
            base.merge(_Tally.from_state(result["tally"], max_candidates))
            sampled.update(result["sampled"])
            for fields, shingles in result["evidence"]:
                if isinstance(fields, str):
                    sketch.add(fields, shingles)
                    continue
                if fields["document_id"] in seen:
                    raise ValueError("near-duplicate duplicate document ID")
                seen.add(fields["document_id"])
                if len(held) >= max_heldout_documents:
                    raise ValueError(
                        "near-duplicate held-out document cap exceeded "
                        f"({max_heldout_documents})"
                    )
                total_shingles += len(shingles)
                if total_shingles > max_heldout_shingles:
                    raise ValueError(
                        "near-duplicate held-out shingle cap exceeded "
                        f"({max_heldout_shingles})"
                    )
                held.append((fields, shingles))
    index.held = [fields for fields, _ in held]
    index.held_shingles = [shingles for _, shingles in held]
    if template_aware:
        index.find_boilerplate(sketch, sampled)
    held_fields, held_shingles = index.held, index.held_shingles
    index.held, index.held_shingles = [], []
    for fields, shingles in zip(held_fields, held_shingles, strict=True):
        keys = index.keys(str(fields["source_id"]), shingles)
        for other in index.proposals(fields["content_sha256"], keys):
            if index.held[other]["split"] != fields["split"]:
                index.compare(base, fields, shingles, other)
        index.add(fields, shingles, keys)
    del held, held_fields, held_shingles

    _JOB.update(index=index, seen=seen, checkpoint=checkpoint)
    for state in _run(_stream, len(shards), workers):
        base.merge(_Tally.from_state(state, max_candidates))
    if checkpoint is not None:
        for number in range(len(shards)):
            checkpoint.shard(number).clear()
    _JOB.clear()

    base.keep_listed()
    counters = base.counters
    coverage = [
        {
            "split": split,
            "source_id": source,
            "documents": counters["documents"][split, source],
            "text_bytes": counters["text_bytes"][split, source],
            "documents_without_words": counters["without_words"][split, source],
        }
        for split, source in sorted(counters["documents"])
    ]
    report = {
        "algorithm": "word_shingle_minhash_lsh_v2",
        "limitations": (
            "Lexical candidates only; LSH is not exhaustive and does not establish "
            "semantic independence. Documents without words are matched only by "
            "identical text."
            + (
                " Template-aware: a shingle many documents share is taken for "
                "template text, so repeated real content can be mistaken for it; "
                "the template_only pairs are counted and summarized, not hidden."
                if template_aware
                else ""
            )
        ),
        "parameters": {
            "word_shingle_width": 3,
            "short_document_width": "min(3, word_count)",
            "hashes": _HASHES,
            "bands": _BANDS,
            "hashes_per_band": _ROWS,
            "min_jaccard": min_jaccard,
            "max_heldout_documents": max_heldout_documents,
            "max_heldout_shingles": max_heldout_shingles,
            "max_candidates": max_candidates,
            "template_aware": template_aware,
            **(
                {
                    "boilerplate_heldout_documents": boilerplate_heldout_documents,
                    "boilerplate_sample_documents": boilerplate_sample_documents,
                    "boilerplate_sample_stride": boilerplate_sample_stride,
                    "boilerplate_min_share": boilerplate_min_share,
                    "boilerplate_sketch_bits": boilerplate_sketch_bits,
                }
                if template_aware
                else {}
            ),
        },
        "document_count": sum(counters["documents"].values()),
        "heldout_shingles": total_shingles,
        "coverage": coverage,
        "cross_split_comparisons": base.comparisons,
        "candidate_count": base.candidate_count,
        "identical_text_count": base.identical_count,
        "candidates_truncated": base.candidate_count > max_candidates,
        "candidates": base.found,
    }
    if template_aware:
        listed = [base.template_docs[k] for k in sorted(base.template_docs)]
        report |= {
            "boilerplate": {
                "heldout_shingles": sum(len(v) for v in index.frequent.values()),
                "sampled_documents": sum(sampled.values()),
            },
            "class_counts": {
                **{
                    name: base.classes[name]
                    for name in ("leak", "content_too_short", "partly_template")
                },
                "template_only": base.template_only_count,
            },
            "template_only_count": base.template_only_count,
            "template_only_document_count": len(listed),
            "template_only_truncated": len(listed) > max_candidates,
            "template_only": listed[:max_candidates],
        }
    return report


def _line_ranges(path: Path, shard_bytes: int) -> list[tuple[int, int]]:
    """Consecutive line-aligned byte ranges of about `shard_bytes` each."""
    size = path.stat().st_size
    bounds = [0]
    with path.open("rb") as handle:
        for target in range(shard_bytes, size, shard_bytes):
            handle.seek(target - 1)
            handle.readline()
            if bounds[-1] < handle.tell() < size:
                bounds.append(handle.tell())
    bounds.append(size)
    return list(itertools.pairwise(bounds))


def _document_rows(
    path: Path, start: int, end: int
) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("rb") as handle:
        handle.seek(start)
        offset = start
        while offset < end:
            line = handle.readline()
            if not line:
                break
            offset += len(line)
            if line.strip():
                yield offset, json.loads(line)


def screen_documents(
    documents: Path,
    *,
    identity: str,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 200_000,
    workers: int = 1,
    shard_bytes: int = 256 << 20,
    retain_evidence: bool = False,
    template_aware: bool = False,
    boilerplate_sample_stride: int = _DEFAULT_SAMPLE_STRIDE,
    **options: Any,
) -> dict[str, Any]:
    """Screen a `documents.jsonl` file in line-aligned shards of `shard_bytes`.

    With `checkpoint_dir` each shard saves first-pass shingle evidence and
    training progress. `retain_evidence` keeps verified first-pass shards after
    success so later screens can reuse them with different comparison options.
    """
    if retain_evidence and checkpoint_dir is None:
        raise ValueError(
            "retaining near-duplicate evidence requires a checkpoint directory"
        )
    documents = documents.resolve(strict=True)
    observed = range_file_identity(documents)
    shards = _line_ranges(documents, shard_bytes)
    options = {
        **options,
        "template_aware": template_aware,
        "boilerplate_sample_stride": boilerplate_sample_stride,
    }

    def read(start: int, end: int):
        return _document_rows(documents, start, end)

    if checkpoint_dir is None:
        report = screen(read, shards, workers=workers, **options)
    else:
        evidence_inputs = {
            "file": observed,
            "numpy_version": np.__version__,
            "unicode_version": unicodedata.unidata_version,
            "implementation": implementation_digests("near_duplicates.py"),
            "sample_stride": boilerplate_sample_stride if template_aware else None,
        }
        inputs = {
            **evidence_inputs,
            "identity": identity,
            "options": options,
            "shards": shards,
        }
        key = canonical_digest(inputs)
        root = Path(checkpoint_dir)
        checkpoint = Checkpoint(
            root / f"near-duplicates-{key[:16]}.json", key, checkpoint_every
        )
        evidence = checkpoint.path.with_suffix(".evidence")
        if retain_evidence:
            evidence_key = canonical_digest(
                {**evidence_inputs, "shard_bytes": shard_bytes}
            )
            evidence = root / f"near-duplicates-{evidence_key}.retained-evidence"
        plan = ShardPlan(
            EVIDENCE_STAGE,
            [(str(documents), start, end) for start, end in shards],
            observed[2],
            inputs=evidence_inputs if retain_evidence else {"screen": key},
            store=evidence,
        )
        root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            checkpoint.path.with_suffix(".lock"),
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as lock:
            # Unlike flock, a process record lock does not survive in forked workers.
            fcntl.lockf(lock, fcntl.LOCK_EX)
            if evidence.is_symlink():
                raise ValueError(f"symlinked near-duplicate evidence: {evidence}")
            report = screen(
                read,
                shards,
                workers=workers,
                checkpoint=checkpoint,
                _evidence=plan,
                **options,
            )
            if not retain_evidence and evidence.exists():
                shutil.rmtree(evidence)
    if range_file_identity(documents) != observed:
        raise ValueError("documents changed during near-duplicate screen")
    return report


def audit_release(path: Path, **options: Any) -> dict[str, Any]:
    """Verify the immutable release, then screen its original normalized text."""
    from klab.corpus.release import open_release

    with open_release(path) as release:
        release_id = release.manifest["release_id"]
        report = screen_documents(
            release.member("documents.jsonl"), identity=release_id, **options
        )
        return {"release_id": release_id, **report}


DECISION_CHOICES = ("merged", "dropped", "kept")


def pair_key(pair: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted((pair["left_document_id"], pair["right_document_id"])))


def check_decisions(
    report: dict[str, Any], report_sha256: str, record: dict[str, Any]
) -> dict[str, Any]:
    """Compare a reviewer's decision record with the screen report it decides.

    The record names the report by the SHA-256 of its file and lists decisions
    per reported pair (`documents: [ID, ID]`, `decision`: one of `merged`,
    `dropped` or `kept`, and a `reason` for `kept`). It may also carry a
    `default` (`decision: kept` and a `reason`) that decides every reported pair
    not listed individually, so a large report of templated or boilerplate
    matches is accepted as a class and only the exceptions are listed. Returns
    the problems found; an empty `problems` list means every pair is decided by
    a record that was written for this exact report.
    """
    problems: list[str] = []
    if record.get("schema_version") != 1:
        problems.append("decision record schema_version must be 1")
    if record.get("report_sha256") != report_sha256:
        problems.append("decision record was written for a different report")
    if record.get("release_id") != report.get("release_id"):
        problems.append("decision record names a different release")
    for field in ("accepted_by", "accepted_on"):
        if not str(record.get(field) or "").strip():
            problems.append(f"decision record lacks {field}")
    decided: dict[tuple[str, str], str] = {}
    for item in record.get("decisions") or []:
        documents = item.get("documents")
        if (
            not isinstance(documents, list)
            or len(documents) != 2
            or not all(isinstance(value, str) for value in documents)
        ):
            problems.append(f"malformed decision entry: {item!r}"[:200])
            continue
        key = tuple(sorted(documents))
        if item.get("decision") not in DECISION_CHOICES:
            problems.append(f"pair {key[0]} / {key[1]} has no valid decision")
        elif item["decision"] == "kept" and not str(item.get("reason") or "").strip():
            problems.append(f"pair {key[0]} / {key[1]} is kept without a reason")
        elif key in decided:
            problems.append(f"pair {key[0]} / {key[1]} is decided twice")
        decided[key] = item.get("decision")
    default = record.get("default")
    if default is not None and not (
        isinstance(default, dict)
        and default.get("decision") == "kept"
        and str(default.get("reason") or "").strip()
    ):
        problems.append("default decision must be `kept` with a reason")
        default = None
    reported = {pair_key(pair) for pair in report["candidates"]}
    explicit = reported & decided.keys()
    undecided = sorted(reported - decided.keys()) if default is None else []
    unknown = sorted(decided.keys() - reported)
    if unknown:
        problems.append(f"{len(unknown)} decided pair(s) are not in the report")
    return {
        "reported_pairs": len(reported),
        "decided_pairs": len(explicit) if default is None else len(reported),
        "defaulted_pairs": len(reported) - len(explicit) if default else 0,
        "undecided": [list(pair) for pair in undecided[:20]],
        "undecided_count": len(undecided),
        "problems": problems,
    }
