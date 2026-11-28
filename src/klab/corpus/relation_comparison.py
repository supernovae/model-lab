"""Compare verified Corpus representations using the normal text reader and encoder."""

from __future__ import annotations

import json
import os
import re
import stat
from contextlib import ExitStack
from heapq import nsmallest
from itertools import islice
from pathlib import Path

from tokenizers import __version__ as tokenizers_version

from klab.corpus.build_stages import RECEIPT, implementation_digests, run_stage
from klab.corpus.relation_compile import COMPILED, MAX_RELATION_SEQUENCE_BYTES
from klab.corpus.relations import SPLITS, relation_store
from klab.corpus.release_members import shared_payload_lease
from klab.corpus.shards import range_file_identity
from klab.corpus.stage_inspection import open_verified_stage
from klab.data.datasets import iter_local_text
from klab.data.encoding import (
    TOKENIZER_BATCH_DOCUMENTS,
    TOKENIZER_BATCH_SOURCE_BYTES,
    PreparationEncoder,
)
from klab.data.tokenizer import load_tokenizer
from klab.safe_io import symlinked, write_canonical_json
from klab.training.manifest import sha256_file

_STAGE = "relation-comparison"
_REPORT = "comparison.json"
_KEY = re.compile(r"[0-9a-f]{64}")
_METADATA_CAP = 4 * 1024**2


def _count_tokens(path, encoder):
    documents = content_tokens = 0
    batch, size = [], 0
    for text, _ in iter_local_text(path, source_bytes=MAX_RELATION_SEQUENCE_BYTES):
        weight = len(text.encode("utf-8"))
        if batch and (
            len(batch) == TOKENIZER_BATCH_DOCUMENTS
            or size + weight > TOKENIZER_BATCH_SOURCE_BYTES
        ):
            content_tokens += sum(map(len, encoder.encode_arrays(batch)))
            batch, size = [], 0
        batch.append(text)
        size += weight
        documents += 1
    content_tokens += sum(map(len, encoder.encode_arrays(batch)))
    return documents, content_tokens


def compare_relations(root, keys, tokenizer_path, *, max_bytes, workers=None):
    """Measure or reuse serialized content and one EOS per row for matched views."""
    if not 2 <= len(keys) <= 16 or len(set(keys)) != len(keys):
        raise ValueError("compare requires 2–16 distinct compiled keys")
    if max_bytes <= 0:
        raise ValueError("comparison byte limit must be positive")
    if any(not _KEY.fullmatch(key) for key in keys):
        raise ValueError("compare requires full compiled SHA-256 keys")
    store = relation_store(root)
    names = tuple(f"{split}.jsonl" for split in SPLITS)
    expected_bytes = 0
    # Reserve the serialization read budget before a cold open can hash the inputs.
    for key in keys:
        directory = store.directory(COMPILED, key)
        if symlinked(directory):
            raise ValueError(f"unsafe compiled relation stage: {key}")
        for name in names:
            member = (directory / name).lstat()
            if not stat.S_ISREG(member.st_mode):
                raise ValueError(
                    f"compiled relation member is not a regular file: {key}/{name}"
                )
            expected_bytes += member.st_size
            if expected_bytes > max_bytes:
                raise ValueError(
                    "compiled text inputs exceed the comparison byte limit"
                )
    with ExitStack() as stack:
        arms = [
            stack.enter_context(open_verified_stage(store, COMPILED, key))[:2]
            for key in keys
        ]
        index = arms[0][1]["inputs"]["index"]
        if any(receipt["inputs"]["index"] != index for _, receipt in arms):
            raise ValueError("compare requires the same canonical relation index")
        if (
            sum(receipt["files"][name]["size"] for _, receipt in arms for name in names)
            > max_bytes
        ):
            raise ValueError("compiled text inputs exceed the comparison byte limit")
        generation = range_file_identity(tokenizer_path)
        tokenizer_sha256 = sha256_file(tokenizer_path)
        if generation != range_file_identity(tokenizer_path):
            raise ValueError("tokenizer changed while loading relation comparison")
        data = Path(__file__).parents[1] / "data"
        inputs = {
            "index": index,
            "arms": [
                {"key": receipt["key"], "files": receipt["files"]}
                for _, receipt in arms
            ],
            "tokenizer_sha256": tokenizer_sha256,
            "tokenizers_version": tokenizers_version,
            "implementation": {
                **implementation_digests("relation_comparison.py"),
                **{
                    name: sha256_file(data / name)
                    for name in ("datasets.py", "encoding.py", "tokenizer.py")
                },
            },
        }
        key = store.key(inputs)

        def produce(directory, _result):
            report = _measure(
                arms, index, tokenizer_path, tokenizer_sha256, workers=workers
            )
            if generation != range_file_identity(tokenizer_path):
                raise ValueError("tokenizer changed during relation comparison")
            # Finish all input generation checks before an output becomes visible.
            stack.close()
            write_canonical_json(directory / _REPORT, report, exclusive=True)

        reused = False

        def reuse():
            nonlocal reused
            reused = True

        destination = store.directory(_STAGE, key)
        if symlinked(destination):
            raise ValueError("unsafe relation comparison destination")
        with shared_payload_lease(destination):
            path, _ = run_stage(store, _STAGE, inputs, produce, on_reuse=reuse)
            report = json.loads((path / _REPORT).read_bytes())
        return {**report, "comparison_key": key, "reused": reused}


def _measure(arms, index, tokenizer_path, tokenizer_sha256, *, workers):
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.token_to_id("<eos>") is None:
        raise ValueError("tokenizer has no <eos> special token")
    with PreparationEncoder(tokenizer, max_workers=workers) as encoder:
        representations = []
        for path, receipt in arms:
            stats = receipt["result"]
            partitions = {}
            for split in SPLITS:
                name = f"{split}.jsonl"
                documents, content = _count_tokens(path / name, encoder)
                partition = stats["partitions"][split]
                if documents != partition["sequences"]:
                    raise ValueError(
                        f"compiled sequence count differs from receipt in {split}"
                    )
                lineage = receipt["files"][f"{split}.lineage.jsonl"]
                partitions[split] = {
                    **partition,
                    "content_tokens": content,
                    "eos_tokens": documents,
                    "tokens": content + documents,
                    "lineage_bytes": lineage["size"],
                    "text_bytes": receipt["files"][name]["size"],
                }
            tokens = sum(row["tokens"] for row in partitions.values())
            baseline = representations[0]["tokens"] if representations else tokens
            representations.append(
                {
                    "key": receipt["key"],
                    "options": receipt["inputs"]["options"],
                    "fidelity": stats.get("fidelity", "assertion_visits"),
                    "payload_bytes": stats["payload_bytes"],
                    "scratch_bytes": stats.get("scratch_bytes", 0),
                    "tokens": tokens,
                    "token_delta": tokens - baseline,
                    "partitions": partitions,
                }
            )
        return {
            "index": index,
            "tokenizer_sha256": tokenizer_sha256,
            "baseline": representations[0]["key"],
            "representations": representations,
        }


def comparison_inventory(root, *, limit=20, after=None):
    """Page published comparison keys without opening their inputs or reports."""
    if not 1 <= limit <= 100 or (after is not None and not _KEY.fullmatch(after)):
        raise ValueError(
            "comparison inventory needs a limit of 1–100 and a full key cursor"
        )
    path = relation_store(root).root / _STAGE
    if symlinked(path):
        raise ValueError("unsafe relation comparison store")
    if not path.is_dir():
        return {"comparison_keys": [], "next_after": None}
    with os.scandir(path) as entries:
        keys = nsmallest(
            limit + 1,
            (
                entry.name
                for entry in entries
                if _KEY.fullmatch(entry.name)
                and (after is None or entry.name > after)
                and entry.is_dir(follow_symlinks=False)
            ),
        )
    return {
        "comparison_keys": keys[:limit],
        "next_after": keys[limit - 1] if len(keys) > limit else None,
    }


def read_comparison(root, key):
    """Check the compact report's bytes without scanning any compiled input."""
    if not _KEY.fullmatch(key):
        raise ValueError("comparison inspection requires a full SHA-256 key")
    store = relation_store(root)
    directory = store.directory(_STAGE, key)
    if symlinked(directory):
        raise ValueError("unsafe relation comparison path")
    with os.scandir(directory) as entries:
        if {entry.name for entry in islice(entries, 3)} != {RECEIPT, _REPORT}:
            raise ValueError("comparison directory does not contain its expected files")
    for name in (RECEIPT, _REPORT):
        info = (directory / name).lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > _METADATA_CAP:
            raise ValueError("comparison control file is unsafe or exceeds 4 MiB")
    with open_verified_stage(store, _STAGE, key) as (path, receipt, _):
        report = json.loads((path / _REPORT).read_bytes())
        return {
            **report,
            "comparison_key": key,
            "report_sha256": receipt["files"][_REPORT]["sha256"],
        }
