"""Predeclared train-only ByteLevel BPE selection on one saved source snapshot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import statistics
import time
from itertools import islice
from pathlib import Path

from klab.config.models import DatasetConfig, TokenizerTrainConfig
from klab.data.datasets import iter_documents
from klab.data.tokenizer import (
    SPECIAL_TOKENS,
    load_tokenizer,
    train_tokenizer,
    verify_snapshot,
    verify_tokenizer_artifact,
)

VOCABS = (8192, 12000, 16384)
TRAIN_DOCS = 250_000
TRAIN_BYTES = 256 * 1024 * 1024
DEV_DOCS = 2_000


def choose_candidate(candidates: list[dict[str, object]]) -> int:
    eligible = [candidate for candidate in candidates if candidate["valid"]]
    if not eligible:
        raise ValueError("no valid tokenizer candidates")
    minimum = min(float(candidate["tokens_per_byte"]) for candidate in eligible)
    return min(
        int(candidate["vocab_size"])
        for candidate in eligible
        if float(candidate["tokens_per_byte"]) <= 1.02 * minimum
    )


def _percentile(values: list[int], quantile: float) -> int:
    return sorted(values)[math.ceil(quantile * len(values)) - 1]


def bakeoff(dataset: DatasetConfig, output_dir: Path) -> Path:
    """Train three BPEs on the identical whole-story prefix, evaluate only validation[:2000]."""
    if dataset.source not in {"local_stories", "snapshot"}:
        raise ValueError("bakeoff requires a verified dataset snapshot")
    source = verify_snapshot(dataset)
    if (
        source["splits"]["train"]["count"] < TRAIN_DOCS
        or source["splits"]["validation"]["count"] < DEV_DOCS
    ):
        raise ValueError(
            "snapshot lacks required train or tokenizer-development stories"
        )
    if output_dir.exists():
        raise FileExistsError(f"bakeoff destination exists: {output_dir}")
    assert dataset.source_manifest_path is not None
    source_hash = hashlib.sha256(dataset.source_manifest_path.read_bytes()).hexdigest()
    bounded = dataset.model_copy(
        update={"train_max_tokens": TRAIN_BYTES, "train_max_documents": TRAIN_DOCS}
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates: list[dict[str, object]] = []
    for vocab in VOCABS:
        config = TokenizerTrainConfig(
            schema_version=1,
            vocab_size=vocab,
            min_frequency=2,
            max_documents=TRAIN_DOCS,
            output_dir=output_dir / str(vocab),
            dataset=bounded,
        )
        started = time.monotonic()
        path = train_tokenizer(config)
        fit_seconds = time.monotonic() - started
        process_peak_rss_bytes = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        manifest = verify_tokenizer_artifact(
            path,
            source="local_stories",
            revision=dataset.revision,
            vocab_size=vocab,
            dataset=bounded,
        )
        tokenizer = load_tokenizer(path)
        counts: list[int] = []
        total_bytes = total_tokens = 0
        started = time.monotonic()
        for index, story in enumerate(
            islice(iter_documents(dataset, "validation"), DEV_DOCS)
        ):
            ids = tokenizer.encode(story, add_special_tokens=False).ids
            if tokenizer.decode(ids, skip_special_tokens=False) != story:
                raise ValueError(
                    f"vocabulary {vocab} fails validation roundtrip at development story {index}"
                )
            counts.append(len(ids))
            total_tokens += len(ids)
            total_bytes += len(story.encode("utf-8"))
        elapsed = max(time.monotonic() - started, 1e-9)
        if len(counts) != DEV_DOCS or not total_bytes:
            raise ValueError("incomplete tokenizer-development sample")
        contract = manifest["training_contract"]
        if (
            contract["stop_reason"] == "source_exhausted"
            or (
                contract["stop_reason"] == "document_limit"
                and manifest["selected_documents"] != TRAIN_DOCS
            )
            or manifest["selected_input_bytes_utf8"] > TRAIN_BYTES
        ):
            raise ValueError(
                "tokenizer did not reach its declared whole-document prefix boundary"
            )
        if manifest["special_ids"] != {
            token: index for index, token in enumerate(SPECIAL_TOKENS)
        }:
            raise ValueError("invalid special token IDs")
        candidates.append(
            {
                "vocab_size": vocab,
                "valid": True,
                "tokens_per_byte": total_tokens / total_bytes,
                "total_tokens": total_tokens,
                "utf8_bytes": total_bytes,
                "median_tokens": statistics.median(counts),
                "p90_tokens": _percentile(counts, 0.90),
                "p99_tokens": _percentile(counts, 0.99),
                "fraction_over_512": sum(count > 512 for count in counts) / DEV_DOCS,
                "encode_tokens_per_second": total_tokens / elapsed,
                "encode_seconds": elapsed,
                "bpe_fit_seconds": manifest["bpe_fit_seconds"],
                "total_training_seconds": fit_seconds,
                "process_peak_rss_bytes": process_peak_rss_bytes,
                "artifact_bytes": path.stat().st_size,
                "tokenizer_sha256": manifest["sha256"],
                "training_content_sha256": manifest["content_digest_sha256"],
                "training_documents": manifest["selected_documents"],
                "training_bytes": manifest["selected_input_bytes_utf8"],
            }
        )
    if len({candidate["training_content_sha256"] for candidate in candidates}) != 1:
        raise ValueError("bakeoff candidates trained on different story prefixes")
    receipt = {
        "schema_version": 1,
        "source_manifest_sha256": source_hash,
        "development_split": "validation",
        "development_start": 0,
        "development_count": DEV_DOCS,
        "training_max_documents": TRAIN_DOCS,
        "training_max_utf8_bytes": TRAIN_BYTES,
        "min_frequency": 2,
        "rule": "smallest valid vocab with held-out tokens/byte <= 1.02 * minimum",
        "candidates": candidates,
        "selected_vocab_size": choose_candidate(candidates),
    }
    path = output_dir / "selection_receipt.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path
