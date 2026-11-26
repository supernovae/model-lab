"""Bounded synthetic retained-mixture benchmark; no pytest, downloads or training.

Run this same file with each pinned Lab interpreter. This exercises the actual
retained index and schedule, with explicitly synthetic release evidence. It is
not a release-verification benchmark or scientific acceptance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace


def benchmark_loaders(root, schedule, receipt, records, *, blocks, seq_len):
    """Compare the same shuffled training blocks with and without segment reads."""
    import mmap

    import numpy as np
    import psutil

    from klab.data.mixture_batches import ReferencedTokenArray
    from klab.data.packing import TokenBlockDataset

    root.mkdir()
    padded = -(-schedule.target_count // seq_len) * seq_len
    path = root / "train.npy"
    ids = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.int32, shape=(padded + 1,)
    )
    mask = np.lib.format.open_memmap(
        root / "train_supervision.npy", mode="w+", dtype=bool, shape=(padded + 1,)
    )
    ids[0] = 2
    position = 1
    for raw in schedule.blocks(64 * 1024**2):
        values = np.frombuffer(raw, dtype="<i4")
        ids[position : position + len(values)] = values
        position += len(values)
    ids[position:] = 1
    mask[:] = False
    mask[1:position] = True
    ids.flush()
    mask.flush()
    inline = TokenBlockDataset(ids, seq_len, supervision=mask)
    order = np.random.default_rng(42).permutation(len(inline))[:blocks].tolist()
    process = psutil.Process()
    results = []

    def run(dataset, mode, paths):
        if hasattr(os, "posix_fadvise"):
            for path in paths:
                with path.open("rb") as stream:
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        for cache in ("first", "warm"):
            latencies = []
            digest = hashlib.sha256()
            before = process.io_counters() if hasattr(process, "io_counters") else None
            for offset in range(0, len(order), 4):
                started = time.perf_counter()
                batch = [
                    dataset.numpy_microblock(index)
                    for index in order[offset : offset + 4]
                ]
                latencies.append(time.perf_counter() - started)
                for inputs, targets, _ in batch:
                    digest.update(inputs.tobytes())
                    digest.update(targets.tobytes())
            after = process.io_counters() if before is not None else None
            results.append(
                {
                    "loader": mode,
                    "cache": cache,
                    "blocks": len(order),
                    "seconds": sum(latencies),
                    "p95_ms_per_batch": 1000 * float(np.percentile(latencies, 95)),
                    "bytes_read_from_storage": (
                        after.read_bytes - before.read_bytes
                        if after is not None
                        else None
                    ),
                    "rss_bytes": process.memory_info().rss,
                    "sha256": digest.hexdigest(),
                }
            )

    if hasattr(ids._mmap, "madvise"):
        ids._mmap.madvise(mmap.MADV_DONTNEED)
    run(inline, "contiguous", [path])
    opened_at = time.perf_counter()
    with ReferencedTokenArray(
        schedule, records, receipt, seq_len=seq_len, bos=2, pad=1
    ) as source:
        open_seconds = time.perf_counter() - opened_at
        references = TokenBlockDataset(source, seq_len, supervision=mask)
        run(references, "references", [proof.path for proof in schedule.tokens])
    if len({row["sha256"] for row in results}) != 1:
        raise ValueError("train batches differ across loaders")
    return {
        "row_tokens": seq_len,
        "reference_open_seconds": open_seconds,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--documents", type=int, default=512)
    parser.add_argument("--text-bytes", type=int, default=512)
    parser.add_argument("--token-cache", choices=("append", "sealed"), default="append")
    parser.add_argument(
        "--token-output", choices=("contiguous", "references"), default="contiguous"
    )
    parser.add_argument("--loader-blocks", type=int, default=0)
    parser.add_argument("--loader-row-tokens", type=int, default=1024)
    args = parser.parse_args()
    if (
        not 1 <= args.workers <= 8
        or not 8 <= args.documents <= 65_536
        or not 32 <= args.text_bytes <= 2048
    ):
        parser.error("bounds: workers 1..8, documents 8..65536, text-bytes 32..2048")
    if args.token_output == "references" and args.token_cache != "sealed":
        parser.error("reference token output requires the sealed cache")
    if (
        not 0 <= args.loader_blocks <= 16_384
        or not 16 <= args.loader_row_tokens <= 8192
    ):
        parser.error("bounds: loader-blocks 0..16384, loader-row-tokens 16..8192")
    if args.loader_blocks and args.token_output != "references":
        parser.error("loader comparison requires reference token output")
    # Affinity also bounds the baseline's default Rayon planner. Do this before
    # importing Lab or tokenizers; no nested BLAS/Rayon pool inherits more cores.
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[: args.workers])
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    root = args.root.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.environ["XDG_CONFIG_HOME"] = str(root / "config")

    import psutil
    from tokenizers import Tokenizer, models, pre_tokenizers

    from klab.canonical import canonical_json
    from klab.corpus.mixture import _native_mixture_completion, _schedule
    from klab.corpus.mixture_pipeline import MixtureFrontier, pipeline_request
    from klab.corpus.mixture_schedule import token_cache_root, verify_schedule
    from klab.corpus.release import ReleaseMembers
    from klab.corpus.release_verify import verification_settings
    from klab.corpus.retained_mixture import RetainedMixtureIndex
    from klab.corpus.split_runs import write_rows
    from klab.training.manifest import sha256_file

    workspace = root / "corpora" / "synthetic"
    release = workspace / "builds" / "build"
    release.mkdir(parents=True)
    (release / "manifest.json").write_text("{}\n")
    source = workspace / "render.jsonl"
    refs = workspace / "references.jsonl"
    parents = workspace / "parents.bin"
    origins = workspace / "origins.jsonl"
    evidence = workspace / "evidence.json"
    inventory = workspace / "family-inventory.jsonl"
    evidence.write_text("{}\n")
    origins.write_bytes(b"")
    inventory.write_bytes(b"synthetic-only\n")
    references = []
    with source.open("wb") as stream:
        for i in range(args.documents):
            # Deterministic duplicates, empty records and uneven lengths.
            key = i - 1 if i % 17 == 1 else i
            text = (
                ""
                if i % 31 == 0
                else (f"record {key:08d} " + "abcdefgh " * args.text_bytes)[
                    : args.text_bytes + (i % 7)
                ]
            )
            content = hashlib.sha256(text.encode()).hexdigest()
            row = {
                "document_id": str(i),
                "source_id": "source",
                "content_sha256": content,
                "text": text,
                "domains": ["general"],
                "rights": {"training_eligibility": "eligible"},
            }
            raw = canonical_json(row) + b"\n"
            offset = stream.tell()
            stream.write(raw)
            references.append(
                [
                    str(i),
                    0,
                    offset,
                    len(raw),
                    "train",
                    None,
                    content,
                    None,
                    None,
                    None,
                    ["general"],
                    f"family-{i // 3}",
                    "general",
                ]
            )
    write_rows(refs, references)
    parents.write_bytes(b"".join(struct.pack("<Q", i) for i in range(args.documents)))
    proof = {"completion_path": str(evidence), "cache_path": str(evidence)}
    cohort = {
        "release_id": "a" * 64,
        "manifest": {"corpus_id": "synthetic"},
        "documents": args.documents,
        "references": str(refs),
        "parents": str(parents),
        "origins": str(origins),
        "sources": [str(source)],
        "source_ids": ["source"],
        "evidence": {
            "references": proof,
            "deduplication": proof,
            "origins": proof,
            "sources": {"source": proof},
        },
    }
    spec = SimpleNamespace(
        release_path=release,
        target_quotas={"general": args.documents * args.text_bytes // 2},
        unique_train_content_floors={"general": 1},
        max_exposures=1,
        strata={},
        packing="contiguous",
        token_cache=args.token_cache,
        token_output=args.token_output,
        seed=42,
        order_policy=None,
        family_inventory=inventory,
        additional_releases=[],
    )
    special = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    tokenizer = Tokenizer(
        models.BPE(
            vocab={
                **special,
                **{
                    c: i + 4
                    for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
                },
            },
            merges=[],
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer_sha = hashlib.sha256(tokenizer.to_str().encode()).hexdigest()
    members = ReleaseMembers(release, {"release_id": cohort["release_id"]})
    declaration = b"synthetic benchmark"
    results = []
    for name in ("cold", "warm"):
        staging = root / name
        staging.mkdir()
        scratch = staging / "index"
        scratch.mkdir()
        started = time.monotonic()
        with RetainedMixtureIndex(
            scratch, root, [cohort], 42, {"source": "general"}
        ) as index:
            indexed = time.monotonic()
            cache, request = _native_mixture_completion(
                spec, declaration, staging, members.manifest, tokenizer_sha
            )
            frontier = MixtureFrontier(
                root / f".{name}.assembly", cache, pipeline_request(request, index)
            )
            with verification_settings(workers=args.workers):
                receipt = _schedule(
                    spec,
                    declaration,
                    members,
                    tokenizer,
                    tokenizer_sha,
                    3,
                    index,
                    staging,
                    frontier=frontier,
                )
        scheduled_at = time.monotonic()
        if args.token_output == "references":
            with verification_settings(
                workers=args.workers, verification_mode="verified_reuse"
            ):
                schedule = verify_schedule(
                    staging, token_cache_root(release, root), receipt, 3
                )
            verified_at = time.monotonic()
            digest = hashlib.sha256()
            for block in schedule.blocks(64 * 1024**2):
                digest.update(block)
            output_digest = digest.hexdigest()
        else:
            verified_at = scheduled_at
            output_digest = sha256_file(staging / "train.tokens.ids")
        results.append(
            {
                "cache": name,
                "index_seconds": indexed - started,
                "schedule_seconds": scheduled_at - indexed,
                "reference_verification_seconds": verified_at - scheduled_at,
                "target_scan_seconds": time.monotonic() - verified_at,
                "receipt": receipt,
                "output_sha256": [
                    output_digest,
                    sha256_file(staging / "train.records.jsonl"),
                ],
                "rss_bytes": psutil.Process().memory_info().rss,
                "progress": json.loads((frontier.root / "progress.json").read_bytes()),
            }
        )
    if (
        results[0]["receipt"] != results[1]["receipt"]
        or results[0]["output_sha256"] != results[1]["output_sha256"]
    ):
        raise ValueError("cold/warm outputs differ")
    import klab

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(klab.__file__).parents[2],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    result = {
        "revision": revision,
        "workers": args.workers,
        "documents": args.documents,
        "text_bytes": args.text_bytes,
        "token_cache": args.token_cache,
        "token_output": args.token_output,
        "fixture_sha256": sha256_file(source),
        "scratch_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        "results": results,
    }
    if args.loader_blocks:
        result["loaders"] = benchmark_loaders(
            root / "loader",
            schedule,
            receipt,
            staging / "train.records.jsonl",
            blocks=args.loader_blocks,
            seq_len=args.loader_row_tokens,
        )
    (root / "benchmark.json").write_bytes(canonical_json(result) + b"\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
