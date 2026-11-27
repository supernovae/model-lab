"""Compare two prepared backings of the same training stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import time
from pathlib import Path


def _measure(data, *, seq_len: int, count: int, batch: int, seed: int, evict: bool):
    import numpy as np
    import psutil

    from klab.data.packing import TokenBlockDataset, epoch_order

    process = psutil.Process()
    started = time.perf_counter()
    with data.open_train() as source:
        dataset = TokenBlockDataset(source, seq_len, supervision=data.train_supervision)
        open_seconds = time.perf_counter() - started
        order = epoch_order(len(dataset), seed, 0)[:count].tolist()
        if not order:
            raise ValueError("prepared data has no usable training blocks")

        if evict:
            paths = (
                [data.root / "train_supervision.npy"]
                if data.train_supervision is not None
                else []
            )
            if data.train is not None:
                paths.append(data.root / "train.npy")
            else:
                schedule = data.mixture_schedule
                assert schedule is not None
                paths.extend(
                    file.path for file in (schedule.schedule, *schedule.tokens)
                )
            for values in (source, data.train_supervision):
                if (mapping := getattr(values, "_mmap", None)) is not None:
                    mapping.madvise(mmap.MADV_DONTNEED)
            for path in paths:
                with path.open("rb") as stream:
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

        results = []
        for phase in ("first", "repeated"):
            # Hash every input and target, including mask sentinels, outside
            # the timed reads. Retain only one block at a time.
            digest = hashlib.sha256()
            latencies = []
            rss_before = process.memory_info().rss
            before = process.io_counters() if hasattr(process, "io_counters") else None
            for start in range(0, len(order), batch):
                elapsed = 0.0
                for index in order[start : start + batch]:
                    began = time.perf_counter()
                    inputs, targets, _ = dataset.numpy_microblock(index)
                    elapsed += time.perf_counter() - began
                    digest.update(inputs)
                    digest.update(targets)
                latencies.append(elapsed)
            after = process.io_counters() if before is not None else None
            seconds = sum(latencies)
            results.append(
                {
                    "phase": phase,
                    "blocks": len(order),
                    "positions": len(order) * seq_len,
                    "seconds": seconds,
                    "positions_per_second": len(order) * seq_len / seconds,
                    "p95_seconds_per_batch": float(np.percentile(latencies, 95)),
                    "storage_read_bytes": (
                        after.read_bytes - before.read_bytes
                        if after is not None
                        else None
                    ),
                    "read_call_bytes": (
                        after.read_chars - before.read_chars
                        if after is not None and hasattr(after, "read_chars")
                        else None
                    ),
                    "rss_before_bytes": rss_before,
                    "rss_after_bytes": process.memory_info().rss,
                    "batch_sha256": digest.hexdigest(),
                }
            )
        return {"open_seconds": open_seconds, "passes": results}


def _compare(args: argparse.Namespace) -> dict:
    from klab.data.packing import load_prepared_data

    if not (1 <= args.blocks <= 65_536 and 1 <= args.batch_blocks <= args.blocks):
        raise ValueError("require 1 <= batch-blocks <= blocks <= 65536")
    if not 0 <= args.seed < 2**63:
        raise ValueError("loader benchmark seed must be between 0 and 2**63 - 1")
    if args.evict_file_pages and not (
        hasattr(os, "POSIX_FADV_DONTNEED") and hasattr(mmap.mmap, "madvise")
    ):
        raise ValueError("file-page eviction advice is unavailable on this host")

    verification = {}
    data = {}
    for mode, path in (
        ("contiguous", args.contiguous),
        ("references", args.references),
    ):
        started = time.perf_counter()
        data[mode] = load_prepared_data(path, byte_enabled=False)
        verification[mode] = time.perf_counter() - started
    contiguous, references = data.values()
    if contiguous.train is None or references.train is not None:
        raise ValueError(
            "provide a contiguous cache followed by a reference-only cache"
        )
    left, right = contiguous.manifest, references.manifest
    if (
        left["tokenizer_sha256"] != right["tokenizer_sha256"]
        or left["train"]["sha256"] != right["train"]["sha256"]
        or left["supervision"].get("train") != right["supervision"].get("train")
    ):
        raise ValueError(
            "prepared loader inputs differ in tokenizer, targets or supervision"
        )
    seq_len = right["train_reference"]["seq_len"]
    results = {}
    for mode, prepared in data.items():
        results[mode] = _measure(
            prepared,
            seq_len=seq_len,
            count=args.blocks,
            batch=args.batch_blocks,
            seed=args.seed,
            evict=args.evict_file_pages,
        )
    digests = {
        sample["batch_sha256"]
        for result in results.values()
        for sample in result["passes"]
    }
    if len(digests) != 1:
        raise ValueError("prepared loaders produced different batches")
    return {
        "status": "completed",
        "cache_advice": "evict_file_pages" if args.evict_file_pages else "none",
        "seed": args.seed,
        "sequence_length": seq_len,
        "batch_blocks": args.batch_blocks,
        "train_sha256": right["train"]["sha256"],
        "verification_seconds": verification,
        "loaders": results,
    }


def _handle(args: argparse.Namespace) -> None:
    try:
        result = _compare(args)
    except (OSError, TypeError, ValueError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}))
        raise SystemExit(2) from None
    print(json.dumps(result, sort_keys=True))


def register_parser(data_commands: argparse._SubParsersAction) -> None:
    parser = data_commands.add_parser(
        "benchmark-loader", help="Compare prepared contiguous and reference loaders"
    )
    parser.add_argument("contiguous", type=Path)
    parser.add_argument("references", type=Path)
    parser.add_argument("--blocks", type=int, default=4096)
    parser.add_argument("--batch-blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evict-file-pages", action="store_true")
    parser.set_defaults(handler=_handle)
