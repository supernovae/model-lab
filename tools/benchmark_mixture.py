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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--documents", type=int, default=512)
    parser.add_argument("--text-bytes", type=int, default=512)
    args = parser.parse_args()
    if (
        not 1 <= args.workers <= 8
        or not 8 <= args.documents <= 2048
        or not 32 <= args.text_bytes <= 2048
    ):
        parser.error("bounds: workers 1..8, documents 8..2048, text-bytes 32..2048")
    # Affinity also bounds the baseline's default Rayon planner. Do this before
    # importing Lab or tokenizers; no nested BLAS/Rayon pool inherits more cores.
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[: args.workers])
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["SPARSELAB_MIXTURE_WORKERS"] = str(args.workers)
    root = args.root.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.environ["XDG_CONFIG_HOME"] = str(root / "config")

    import psutil
    from tokenizers import Tokenizer, models, pre_tokenizers

    from sparselab.canonical import canonical_json
    from sparselab.corpus.mixture import _schedule
    from sparselab.corpus.retained_mixture import RetainedMixtureIndex
    from sparselab.corpus.split_runs import write_rows
    from sparselab.training.manifest import sha256_file

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
            # Same harness on baseline and candidate: include native durability
            # when provided by that revision, without installing test packages.
            import inspect

            options = {}
            if "frontier" in inspect.signature(_schedule).parameters:
                from sparselab.corpus.mixture import _native_mixture_completion
                from sparselab.corpus.mixture_pipeline import (
                    MixtureFrontier,
                    pipeline_request,
                )

                cache, request = _native_mixture_completion(
                    spec,
                    b"synthetic benchmark",
                    staging,
                    {"release_id": "a" * 64},
                    tokenizer_sha,
                )
                options["frontier"] = MixtureFrontier(
                    staging, cache, pipeline_request(request, index)
                )
            receipt = _schedule(
                spec,
                b"synthetic benchmark",
                release,
                {"release_id": "a" * 64},
                tokenizer,
                tokenizer_sha,
                3,
                index,
                staging,
                **options,
            )
        progress_path = (
            (root / ("." + name + ".assembly") / "progress.json")
            if options
            else staging / "progress.json"
        )
        results.append(
            {
                "cache": name,
                "index_seconds": indexed - started,
                "schedule_seconds": time.monotonic() - indexed,
                "receipt": receipt,
                "output_sha256": sha256_file(staging / "train.tokens.jsonl"),
                "rss_bytes": psutil.Process().memory_info().rss,
                "progress": json.loads(progress_path.read_bytes())
                if progress_path.exists()
                else None,
            }
        )
    if (
        results[0]["receipt"] != results[1]["receipt"]
        or results[0]["output_sha256"] != results[1]["output_sha256"]
    ):
        raise ValueError("cold/warm outputs differ")
    import sparselab

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(sparselab.__file__).parents[2],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    result = {
        "revision": revision,
        "workers": args.workers,
        "documents": args.documents,
        "text_bytes": args.text_bytes,
        "fixture_sha256": sha256_file(source),
        "scratch_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        "results": results,
    }
    (root / "benchmark.json").write_bytes(canonical_json(result) + b"\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
