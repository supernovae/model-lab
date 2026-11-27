"""Compare verified Corpus representations using the normal text reader and encoder."""

from __future__ import annotations

from contextlib import ExitStack

from klab.corpus.relation_compile import COMPILED, MAX_RELATION_SEQUENCE_BYTES
from klab.corpus.relations import SPLITS, relation_store
from klab.corpus.shards import range_file_identity
from klab.corpus.stage_inspection import open_verified_stage
from klab.data.datasets import iter_local_text
from klab.data.encoding import (
    TOKENIZER_BATCH_DOCUMENTS,
    TOKENIZER_BATCH_SOURCE_BYTES,
    PreparationEncoder,
)
from klab.data.tokenizer import load_tokenizer
from klab.training.manifest import sha256_file


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
    """Count serialized content and one EOS per row for paired artifact views."""
    if not 2 <= len(keys) <= 16 or len(set(keys)) != len(keys):
        raise ValueError("compare requires 2–16 distinct compiled keys")
    if max_bytes <= 0:
        raise ValueError("comparison byte limit must be positive")
    store = relation_store(root)
    with ExitStack() as stack:
        arms = [
            stack.enter_context(open_verified_stage(store, COMPILED, key))[:2]
            for key in keys
        ]
        index = arms[0][1]["inputs"]["index"]
        if any(receipt["inputs"]["index"] != index for _, receipt in arms):
            raise ValueError("compare requires the same canonical relation index")
        if (
            sum(
                receipt["files"][f"{split}.jsonl"]["size"]
                for _, receipt in arms
                for split in SPLITS
            )
            > max_bytes
        ):
            raise ValueError("compiled text inputs exceed the comparison byte limit")
        generation = range_file_identity(tokenizer_path)
        tokenizer = load_tokenizer(tokenizer_path)
        tokenizer_sha256 = sha256_file(tokenizer_path)
        if generation != range_file_identity(tokenizer_path):
            raise ValueError("tokenizer changed while loading relation comparison")
        if tokenizer.token_to_id("<eos>") is None:
            raise ValueError("tokenizer has no <eos> special token")
        encoder = stack.enter_context(
            PreparationEncoder(tokenizer, max_workers=workers)
        )
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
            "baseline": keys[0],
            "representations": representations,
        }
