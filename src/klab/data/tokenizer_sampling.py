"""Whole-document TRAIN byte quotas by seeded hash, and disjoint TRAIN audits.

Release verification belongs to the caller. Inside each stratum the TRAIN
documents are ordered by a hash of (`sampling.seed`, content digest), and the fit
takes the longest prefix of that order that fits the stratum's byte quota. The
sample is therefore a uniform random subset, independent of where a document sits
in `documents.jsonl`, and a smaller quota selects a prefix of a larger quota's
sample. The audit takes the documents that follow the fit in the same order. No
validation/test text is selected or scored.

The scan is parallel and keeps bounded memory:

1. A histogram pass sums each stratum's bytes per rank bucket (the top 12 bits of
   the rank), and reports any document that breaks a rule.
2. A candidate pass keeps, per stratum, only the documents in the lowest buckets
   holding about twice the quota in bytes (and enough documents after the fit for
   the audit).
3. The coordinator orders the candidates, drops repeated content within a
   stratum and fills the quota. The result is exact when the walk overflowed the
   quota (or reached the end of a stratum) inside the candidates, because every
   document ranked before that point is then among them. If it did not (for
   example repeated content made the threshold too tight) the stratum's
   threshold doubles and only the candidate pass runs again.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from itertools import batched
from pathlib import Path
from typing import BinaryIO, NamedTuple

from tokenizers import Tokenizer

from klab.canonical import canonical_json
from klab.config.models import TokenizerTrainConfig
from klab.corpus import release_verify
from klab.corpus.mixture import _source_rights
from klab.corpus.mixture_index import MAX_ROW_BYTES
from klab.corpus.release_verify import range_records
from klab.corpus.shards import (
    declare_stage,
    plan_ranges,
    range_file_identity,
    run_sharded,
)
from klab.safe_io import read_file_range
from klab.training.manifest import sha256_file

BUCKET_BITS = 12
BUCKETS = 1 << BUCKET_BITS
OVERSAMPLE = 2


def quotas(config: TokenizerTrainConfig) -> dict[str, int]:
    spec = config.sampling
    assert spec is not None
    total = sum(spec.byte_weights.values())
    result = {
        key: spec.max_fit_bytes * weight // total
        for key, weight in spec.byte_weights.items()
    }
    ordered = sorted(
        result,
        key=lambda key: (-(spec.max_fit_bytes * spec.byte_weights[key] % total), key),
    )
    for key in ordered[: spec.max_fit_bytes - sum(result.values())]:
        result[key] += 1
    return result


LINEAGE_STAGE = declare_stage("tokenizer sampling: TRAIN lineage")
HISTOGRAM_STAGE = declare_stage("tokenizer sampling: rank histogram")
CANDIDATE_STAGE = declare_stage("tokenizer sampling: candidates")
_ELIGIBLE = frozenset({"eligible", "eligible_with_obligations"})


class _Candidate(NamedTuple):
    """One retained TRAIN document, located but not carrying its text."""

    document_id: str
    content_sha256: str
    stratum: str
    size: int
    offset: int
    length: int
    rank: int

    @property
    def order(self) -> tuple[int, str, int]:
        return self.rank, self.content_sha256, self.offset


class Selection(NamedTuple):
    """The fit and the audit reserve of each stratum, both in hash order."""

    fit: dict[str, list[_Candidate]]
    reserve: dict[str, list[_Candidate]]
    train_documents: int
    inputs: dict[Path, list[int]]

    def check_inputs(self) -> None:
        """Refuse to reuse offsets and rights decisions from a changed release."""
        for path, identity in self.inputs.items():
            try:
                unchanged = range_file_identity(path) == identity
            except OSError as error:
                raise ValueError(f"tokenizer sampling input changed: {path}") from error
            if not unchanged:
                raise ValueError(f"tokenizer sampling input changed: {path}")


def rank(seed: int, content_sha256: str) -> int:
    """The 64-bit position of a document in its stratum's sampling order."""
    digest = hashlib.sha256(f"{seed}:{content_sha256}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _lineage_ids(path: str, start: int, end: int) -> list[str]:
    return [row["record_id"] for row, _, _ in range_records(Path(path), start, end)]


def _train_documents(
    path: str, start: int, end: int, source_strata: dict[str, str], seed: int
) -> Iterator[tuple[_Candidate, str | None]]:
    """One byte range's undropped TRAIN documents, with the rule each breaks."""
    for row, offset, length in range_records(Path(path), start, end):
        if length > MAX_ROW_BYTES:
            raise ValueError("mixture record-byte cap exceeded")
        if row["drop_reason"] is not None or row["split"] != "train":
            continue
        problem = None
        stratum = source_strata.get(row["source_id"])
        if row.get("rights", {}).get("training_eligibility") not in _ELIGIBLE:
            problem = "ineligible"
        elif stratum is None:
            problem = "source"
        elif stratum not in row["domains"]:
            problem = "domains"
        yield (
            _Candidate(
                row["document_id"],
                row["content_sha256"],
                stratum or row["source_id"],
                len(row["text"].encode("utf-8")),
                offset,
                length,
                rank(seed, row["content_sha256"]),
            ),
            problem,
        )


def _histogram_range(
    path: str, start: int, end: int, source_strata: dict[str, str], seed: int
) -> dict:
    """Bytes per stratum and rank bucket, and the documents that break a rule."""
    histogram: dict[str, dict[int, list[int]]] = {}
    problems = []
    for candidate, problem in _train_documents(path, start, end, source_strata, seed):
        if problem is not None:
            problems.append([candidate.document_id, problem, candidate.stratum])
        elif candidate.size:
            buckets = histogram.setdefault(candidate.stratum, {})
            bucket = candidate.rank >> (64 - BUCKET_BITS)
            counted = buckets.setdefault(bucket, [0, 0])
            counted[0] += candidate.size
            counted[1] += 1
    return {
        "histogram": {
            stratum: [[bucket, *counted] for bucket, counted in sorted(buckets.items())]
            for stratum, buckets in histogram.items()
        },
        "problems": problems,
    }


def _candidate_range(
    path: str,
    start: int,
    end: int,
    source_strata: dict[str, str],
    seed: int,
    limits: dict[str, int],
) -> list[list]:
    """The range's documents whose rank bucket is within their stratum's limit."""
    found = []
    for candidate, problem in _train_documents(path, start, end, source_strata, seed):
        if (
            problem is None
            and candidate.size
            and candidate.rank >> (64 - BUCKET_BITS) <= limits[candidate.stratum]
        ):
            found.append(list(candidate))
    return found


def _limit(buckets: list[list[int]], cap: int, depth: int, factor: int) -> int:
    """The last bucket holding `factor` times the quota's bytes and its documents.

    The document target is the quota's documents at the stratum's mean size plus
    the audit depth. The final bucket is returned if the stratum never gets there.
    """
    size = sum(row[0] for row in buckets)
    count = sum(row[1] for row in buckets)
    if not count:
        return BUCKETS - 1
    wanted_documents = factor * (cap * count // size + depth)
    total = documents = 0
    for bucket, (nbytes, ndocs) in enumerate(buckets):
        total += nbytes
        documents += ndocs
        if total >= factor * cap and documents >= wanted_documents:
            return bucket
    return BUCKETS - 1


def _fill(
    ordered: list[_Candidate], cap: int
) -> tuple[list[_Candidate], list[_Candidate], bool]:
    """The longest prefix within `cap`, the rest, and whether the cap stopped it."""
    used = 0
    for index, candidate in enumerate(ordered):
        if used + candidate.size > cap:
            return ordered[:index], ordered[index:], True
        used += candidate.size
    return ordered, [], False


def _unique(found: list[_Candidate]) -> list[_Candidate]:
    """Hash order with repeated content kept once, at its first file position."""
    unique, seen = [], set()
    for candidate in sorted(found, key=lambda item: item.order):
        if candidate.content_sha256 not in seen:
            seen.add(candidate.content_sha256)
            unique.append(candidate)
    return unique


def select_documents(
    config: TokenizerTrainConfig, caps: dict[str, int] | None = None
) -> Selection:
    """Choose each stratum's fit prefix and audit reserve from the frozen release."""
    spec = config.sampling
    assert spec is not None
    release = config.dataset.corpus_release_path
    assert release is not None
    caps = quotas(config) if caps is None else caps
    lineage = release / "lm/train.lineage.jsonl"
    documents_path = release / "documents.jsonl"
    source_generation = {
        path: range_file_identity(path)
        for path in (lineage, documents_path, release / "sources.json")
    }
    _source_rights(release, spec.source_strata)
    inputs = {}
    if release_verify.current_settings().scratch_dir is not None:
        inputs = {
            "producer": sha256_file(Path(__file__)),
            "range_reader": sha256_file(Path(release_verify.__file__)),
            "max_row_bytes": MAX_ROW_BYTES,
        }
    selected: set[str] = set()
    for ids in run_sharded(
        plan_ranges(LINEAGE_STAGE, lineage, inputs=inputs), _lineage_ids
    ):
        selected.update(ids)
    if len(selected) > config.max_documents:
        raise ValueError(
            "tokenizer seeded sampling needs every TRAIN document: "
            f"max_documents {config.max_documents} is below {len(selected)}"
        )
    totals = {stratum: [[0, 0] for _ in range(BUCKETS)] for stratum in caps}
    results = run_sharded(
        plan_ranges(
            HISTOGRAM_STAGE,
            documents_path,
            spec.source_strata,
            spec.seed,
            inputs=inputs,
        ),
        _histogram_range,
        ordered=True,
    )
    try:
        for result in results:
            for document_id, problem, stratum in result["problems"]:
                if document_id not in selected:
                    continue
                if problem == "ineligible":
                    raise ValueError(
                        "tokenizer sampling encountered ineligible TRAIN document"
                    )
                if problem == "source":
                    raise KeyError(stratum)
                raise ValueError(
                    "tokenizer stratum disagrees with authenticated document domains"
                )
            for stratum, buckets in result["histogram"].items():
                for bucket, nbytes, ndocs in buckets:
                    totals[stratum][bucket][0] += nbytes
                    totals[stratum][bucket][1] += ndocs
    finally:
        results.close()
    oversample = dict.fromkeys(caps, OVERSAMPLE)
    chosen: dict[str, tuple[list[_Candidate], list[_Candidate]]] = {}
    pending = set(caps)
    while pending:
        limits = {
            stratum: _limit(
                totals[stratum],
                caps[stratum],
                spec.audit_documents_per_stratum,
                oversample[stratum],
            )
            for stratum in caps
        }
        found: dict[str, list[_Candidate]] = {stratum: [] for stratum in caps}
        for rows in run_sharded(
            plan_ranges(
                CANDIDATE_STAGE,
                documents_path,
                spec.source_strata,
                spec.seed,
                limits,
                inputs=inputs,
            ),
            _candidate_range,
        ):
            for row in rows:
                if row[0] in selected:
                    candidate = _Candidate(*row)
                    found[candidate.stratum].append(candidate)
        for stratum in sorted(pending):
            fit, rest, full = _fill(_unique(found[stratum]), caps[stratum])
            exact = limits[stratum] == BUCKETS - 1 or (
                full and len(rest) >= spec.audit_documents_per_stratum
            )
            if exact:
                chosen[stratum] = fit, rest
                pending.discard(stratum)
            else:
                oversample[stratum] *= 2
    selection = Selection(
        {stratum: chosen[stratum][0] for stratum in caps},
        {stratum: chosen[stratum][1] for stratum in caps},
        len(selected),
        source_generation,
    )
    selection.check_inputs()
    return selection


def _text(stream: BinaryIO, candidate: _Candidate) -> str:
    return json.loads(read_file_range(stream, candidate.offset, candidate.length))[
        "text"
    ]


def _prefetched_rows(stream: BinaryIO, rows: list[_Candidate]) -> Iterator[_Candidate]:
    """Give the kernel a bounded look ahead without reordering the training stream."""
    advise = getattr(os, "posix_fadvise", None)
    descriptor = stream.fileno()
    for batch in batched(rows, 256):
        if advise is not None:
            remaining = 1024**2
            try:
                for row in batch:
                    count = min(row.length, remaining)
                    advise(descriptor, row.offset, count, os.POSIX_FADV_WILLNEED)
                    remaining -= count
                    if not remaining:
                        break
            except OSError:
                # Advice is optional. The exact reads still check every row.
                advise = None
        yield from batch


def documents(
    config: TokenizerTrainConfig,
    stats: dict,
    selected_hashes: set[str] | None = None,
    *,
    selection: Selection | None = None,
):
    """The fit documents; pass the `select_documents` result to reuse a scan already made."""
    spec = config.sampling
    assert spec is not None
    caps = quotas(config)
    selection = selection or select_documents(config, caps)
    selection.check_inputs()
    used = {key: sum(c.size for c in selection.fit[key]) for key in caps}
    counts = {key: len(selection.fit[key]) for key in caps}
    missing = [key for key, count in counts.items() if not count]
    if missing:
        raise ValueError(
            "tokenizer byte sampling has no fitting TRAIN document in strata: "
            + ", ".join(sorted(missing))
        )
    digest = hashlib.sha256()
    identities = hashlib.sha256()
    release = config.dataset.corpus_release_path
    assert release is not None
    with (release / "documents.jsonl").open("rb") as stream:
        for stratum in caps:
            for row in _prefetched_rows(stream, selection.fit[stratum]):
                text = _text(stream, row)
                digest.update(text.encode("utf-8") + b"\0")
                identities.update(
                    canonical_json([row.document_id, stratum, row.content_sha256])
                    + b"\n"
                )
                if selected_hashes is not None:
                    selected_hashes.add(row.content_sha256)
                yield text
    selection.check_inputs()
    stats.update(
        acquired=selection.train_documents,
        selected=sum(counts.values()),
        bytes=sum(used.values()),
        digest=digest.hexdigest(),
        stop_reason="seeded_hash_prefix",
        sampling={
            "order": "seeded_hash",
            "seed": spec.seed,
            "split": "train",
            "requested_bytes": caps,
            "selected_bytes": used,
            "selected_documents": counts,
            "unfilled_bytes": {key: caps[key] - used[key] for key in caps},
            "selected_identity_sha256": identities.hexdigest(),
            "train_documents": selection.train_documents,
        },
    )


def audit(
    config: TokenizerTrainConfig, tokenizer, *, selection: Selection | None = None
) -> dict:
    # Audit ordinary text, including literal special-token spellings, without
    # mutating the fitted artifact or its production encoding behavior.
    tokenizer = Tokenizer.from_str(tokenizer.to_str())
    tokenizer.encode_special_tokens = True
    spec = config.sampling
    assert spec is not None
    selection = selection or select_documents(config)
    selection.check_inputs()
    fit_hashes = {row.content_sha256 for rows in selection.fit.values() for row in rows}
    groups = {
        key: {
            "documents": 0,
            "bytes": 0,
            "tokens": 0,
            "unknown_tokens": 0,
            "roundtrip_failures": 0,
        }
        for key in spec.byte_weights
    }
    total_bytes = 0
    digest = hashlib.sha256()
    unknown = tokenizer.token_to_id("<unk>")
    release = config.dataset.corpus_release_path
    assert release is not None
    # The documents after each stratum's fit, one stratum at a time in turn, so
    # the byte cap is shared evenly and the choice never depends on the scan.
    depth = spec.audit_documents_per_stratum
    with (release / "documents.jsonl").open("rb") as stream:
        for position in range(depth):
            for stratum, reserve in selection.reserve.items():
                if position >= len(reserve):
                    continue
                row = reserve[position]
                if row.content_sha256 in fit_hashes or total_bytes + row.size > (
                    spec.max_audit_bytes
                ):
                    continue
                fit_hashes.add(row.content_sha256)
                group = groups[stratum]
                text = _text(stream, row)
                ids = tokenizer.encode(text).ids
                group["documents"] += 1
                group["bytes"] += row.size
                group["tokens"] += len(ids)
                group["unknown_tokens"] += ids.count(unknown)
                group["roundtrip_failures"] += int(
                    tokenizer.decode(ids, skip_special_tokens=False) != text
                )
                total_bytes += row.size
                digest.update(
                    canonical_json([row.document_id, row.content_sha256]) + b"\n"
                )
    selection.check_inputs()
    for group in groups.values():
        group["bytes_per_token"] = (
            group["bytes"] / group["tokens"] if group["tokens"] else None
        )
        group["document_shortfall"] = max(
            0, spec.audit_documents_per_stratum - group["documents"]
        )
    probes = {
        "identifier": "some_identifier42 camelCase HTTPServer2",
        "numeric": "-0.125 1e-9 1234567890",
        "multilingual": "café Ελληνικά 日本語 العربية",
        "code_whitespace": "def example():\n\treturn 42  \n",
    }
    probe_results = {}
    for label, text in probes.items():
        ids = tokenizer.encode(text).ids
        probe_results[label] = {
            "roundtrip": tokenizer.decode(ids, skip_special_tokens=False) == text,
            "unknown_tokens": ids.count(unknown),
            "tokens": len(ids),
        }
    return {
        "split": "train",
        "disjoint_from_fit": True,
        "special_tokens_as_text": True,
        "requested_documents_per_stratum": spec.audit_documents_per_stratum,
        "max_bytes": spec.max_audit_bytes,
        "selected_bytes": total_bytes,
        "selected_identity_sha256": digest.hexdigest(),
        "strata": groups,
        "fixed_probes": probe_results,
        "complete": all(not group["document_shortfall"] for group in groups.values()),
    }
