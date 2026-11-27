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
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace


def benchmark_loaders(root, schedule, receipt, records, *, blocks, seq_len):
    """Compare the same shuffled training blocks with and without segment reads."""
    import mmap

    import numpy as np
    import psutil

    from klab.data.mixture_batches import ReferencedTokenArray
    from klab.data.packing import TokenBlockDataset, collect_token_mixture
    from klab.owned_copy import owned_copy

    root.mkdir()
    configurations = SimpleNamespace(
        dataset=SimpleNamespace(train_path=schedule.schedule.path),
        training=SimpleNamespace(seq_len=seq_len),
        model=SimpleNamespace(vocab_size=260),
    )
    specials = {"<bos>": 2, "<pad>": 1, "<eos>": 3}
    tokenizer = SimpleNamespace(token_to_id=specials.get)
    prep = {}
    for kind in ("contiguous", "references"):
        packed = root / kind
        packed.mkdir()
        proofs = {}
        started = time.perf_counter()
        packed_ids, packed_mask, _ = collect_token_mixture(
            configurations,
            tokenizer,
            receipt,
            packed,
            proofs=proofs,
            schedule=schedule,
            store_train=kind == "contiguous",
        )
        if kind == "contiguous":
            ids, mask = packed_ids, packed_mask
            path = packed / "train.npy"
            packed_hash = proofs["train.npy"].sha256
        else:
            owned_copy(
                schedule.schedule.path,
                packed / schedule.schedule.path.name,
                proof=schedule.schedule,
            )
            owned_copy(
                records,
                packed / records.name,
                expected_sha256=receipt["train_records_sha256"],
            )
            packed_hash = packed_ids.metadata()["sha256"]
        prep[kind] = {
            "seconds": time.perf_counter() - started,
            "physical_bytes": sum(path.stat().st_size for path in packed.iterdir()),
            "train_sha256": packed_hash,
        }
        if kind == "references" and not np.array_equal(mask, packed_mask):
            raise ValueError("prepared supervision differs")
    if prep["references"]["train_sha256"] != prep["contiguous"]["train_sha256"]:
        raise ValueError("logical prepared training differs")
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
        "preparation": prep,
    }


def benchmark_projection(root, schedule, receipt, records, *, seq_len):
    """Measure selected-range packaging and exact replay on the same mixture."""
    import psutil

    from klab.canonical import canonical_digest
    from klab.corpus.mixture_projection import (
        load_projection,
        project_dependencies,
        verify_projection_replay,
    )
    from klab.data.packing import PreparedData, collect_token_mixture

    configuration = SimpleNamespace(
        dataset=SimpleNamespace(train_path=schedule.schedule.path),
        training=SimpleNamespace(seq_len=seq_len),
        model=SimpleNamespace(vocab_size=260),
    )
    reference = {"seq_len": seq_len, "bos": 2, "pad": 1}
    specials = {"<bos>": reference["bos"], "<pad>": reference["pad"], "<eos>": 3}
    root.mkdir()
    logical, mask, _ = collect_token_mixture(
        configuration,
        SimpleNamespace(token_to_id=specials.get),
        receipt,
        root,
        schedule=schedule,
        store_train=False,
    )
    manifest = {
        "train": logical.metadata(),
        "token_mixture": receipt,
        "train_reference": reference,
    }
    manifest["manifest_sha256"] = canonical_digest(manifest)
    process = psutil.Process()
    io_counters = getattr(process, "io_counters", None)
    before = io_counters() if io_counters is not None else None
    started = time.monotonic()
    destination = root / "selected"
    files = project_dependencies(schedule, destination, manifest["manifest_sha256"])
    packaged = time.monotonic()
    after = io_counters() if io_counters is not None else None
    compact = load_projection(
        destination,
        manifest,
        memo={(proof.path, proof.fingerprint): proof for proof in files.values()},
    )
    data = PreparedData(
        records.parent,
        None,
        None,
        mask,
        None,
        None,
        None,
        manifest,
        SimpleNamespace(manifest_sha256=manifest["manifest_sha256"]),
        compact,
    )
    verify_projection_replay(data, proof_store=None, verification_mode="cold")
    verified = time.monotonic()
    checked = io_counters() if io_counters is not None else None
    return {
        "whole_bytes": schedule.dependency_footprint()[0],
        "projected_bytes": compact.dependency_footprint()[0],
        "projected_schedule_bytes": compact.schedule.size_bytes,
        "logical_sha256": manifest["train"]["sha256"],
        "package_seconds": packaged - started,
        "verify_seconds": verified - packaged,
        "package_process_reads": after.read_chars - before.read_chars
        if after is not None
        else None,
        "verify_process_reads": checked.read_chars - after.read_chars
        if checked is not None
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--documents", type=int, default=512)
    parser.add_argument("--text-bytes", type=int, default=512)
    parser.add_argument("--token-cache", choices=("append", "sealed"), default="append")
    parser.add_argument("--segment-dtype", choices=("int32", "uint16"), default="int32")
    parser.add_argument(
        "--token-output", choices=("contiguous", "references"), default="contiguous"
    )
    parser.add_argument("--loader-blocks", type=int, default=0)
    parser.add_argument("--loader-row-tokens", type=int, default=1024)
    parser.add_argument("--portable-segments", action="store_true")
    parser.add_argument("--transfer-plan", action="store_true")
    parser.add_argument("--project-selected", action="store_true")
    parser.add_argument("--source-order-sources", type=int, default=0)
    parser.add_argument("--lexical-topics", type=int, default=0)
    parser.add_argument("--link-neighborhoods", type=int, default=0)
    parser.add_argument("--entity-neighborhoods", type=int, default=0)
    args = parser.parse_args()
    if (
        not 1 <= args.workers <= 8
        or not 8 <= args.documents <= 65_536
        or not 32 <= args.text_bytes <= 2048
    ):
        parser.error("bounds: workers 1..8, documents 8..65536, text-bytes 32..2048")
    if args.token_output == "references" and args.token_cache != "sealed":
        parser.error("reference token output requires the sealed cache")
    if args.segment_dtype == "uint16" and args.token_cache != "sealed":
        parser.error("uint16 segments require the sealed cache")
    if (
        not 0 <= args.loader_blocks <= 16_384
        or not 16 <= args.loader_row_tokens <= 8192
    ):
        parser.error("bounds: loader-blocks 0..16384, loader-row-tokens 16..8192")
    if args.loader_blocks and args.token_output != "references":
        parser.error("loader comparison requires reference token output")
    if args.portable_segments and args.token_output != "references":
        parser.error("portable segment packaging requires reference token output")
    if args.transfer_plan and args.token_output != "references":
        parser.error("transfer planning requires reference token output")
    if args.project_selected and args.token_output != "references":
        parser.error("selected-range packaging requires reference token output")
    if args.source_order_sources and (
        args.token_output != "references"
        or not 2 <= args.source_order_sources <= min(64, args.documents)
    ):
        parser.error("source ordering requires reference output and 2–64 sources")
    if args.lexical_topics and (
        args.token_output != "references" or not 1 <= args.lexical_topics <= 32
    ):
        parser.error("lexical ordering requires reference output and 1–32 anchors")
    for count in (args.link_neighborhoods, args.entity_neighborhoods):
        if count and (args.token_output != "references" or not 1 <= count <= 32):
            parser.error(
                "neighborhood ordering requires reference output and 1–32 groups"
            )
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

    from klab.canonical import canonical_digest, canonical_json
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
    source_ids = [f"source-{i:02d}" for i in range(args.source_order_sources)] or [
        "source"
    ]
    sources = [workspace / f"{name}.jsonl" for name in source_ids]
    source_strata = dict.fromkeys(source_ids, "general")
    refs = workspace / "references.jsonl"
    parents = workspace / "parents.bin"
    origins = workspace / "origins.jsonl"
    evidence = workspace / "evidence.json"
    inventory = workspace / "family-inventory.jsonl"
    evidence.write_text("{}\n")
    origins.write_bytes(b"")
    inventory.write_bytes(b"synthetic-only\n")
    topic_words = (
        "metallurgy",
        "horticulture",
        "seismology",
        "meteorology",
        "woodworking",
        "architecture",
        "cartography",
        "hydrology",
    )
    references = []
    with ExitStack() as files:
        streams = [files.enter_context(source.open("wb")) for source in sources]
        for i in range(args.documents):
            # Deterministic duplicates, empty records and uneven lengths.
            key = i - 1 if i % 17 == 1 else i
            marker = (
                topic_words[(key // len(sources)) % len(topic_words)] + " "
                if args.lexical_topics
                else ""
            )
            doc_id = (
                f"{i:064x}"
                if args.lexical_topics
                or args.link_neighborhoods
                or args.entity_neighborhoods
                else str(i)
            )
            text = (
                ""
                if i % 31 == 0
                else (marker + f"record {key:08d} " + "abcdefgh " * args.text_bytes)[
                    : args.text_bytes + (i % 7)
                ]
            )
            content = hashlib.sha256(text.encode()).hexdigest()
            shard = i % len(sources)
            stream = streams[shard]
            row = {
                "document_id": doc_id,
                "source_id": source_ids[shard],
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
                    doc_id,
                    shard,
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
        "sources": [str(source) for source in sources],
        "source_ids": source_ids,
        "evidence": {
            "references": proof,
            "deduplication": proof,
            "origins": proof,
            "sources": dict.fromkeys(source_ids, proof),
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
        segment_dtype=args.segment_dtype,
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
        with RetainedMixtureIndex(scratch, root, [cohort], 42, source_strata) as index:
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
        "segment_dtype": args.segment_dtype,
        "token_output": args.token_output,
        "fixture_sha256": (
            sha256_file(sources[0])
            if len(sources) == 1
            else canonical_digest([sha256_file(source) for source in sources])
        ),
        "scratch_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        "results": results,
    }
    if args.token_output == "references":
        result["segment_storage"] = {
            "segments": len(schedule.tokens),
            "token_bytes": sum(proof.size_bytes for proof in schedule.tokens),
            "index_bytes": sum(proof.size_bytes for proof in schedule.indexes),
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
    orders = []
    if args.source_order_sources:
        orders.append("source_order")
    if args.lexical_topics:
        orders.append("lexical_order")
    if args.link_neighborhoods:
        orders.append("link_order")
    if args.entity_neighborhoods:
        orders.append("entity_order")
    for order in orders:
        from klab.corpus.sequence_order import order_sequences

        ordered = root / order
        ordered.mkdir()
        process = psutil.Process()
        io_counters = getattr(process, "io_counters", None)
        topics = None
        neighborhoods = None
        if order == "lexical_order":
            from klab.corpus.lexical_topics import LexicalTopics, build_lexical_index
            from klab.corpus.split_runs import rows

            lexical = root / "lexical-index"
            lexical.mkdir()
            before = io_counters() if io_counters is not None else None
            started = time.monotonic()
            index = build_lexical_index(
                lexical,
                lexical / "scratch.sqlite",
                (
                    (row["document_id"], "general", row["text"])
                    for source in sources
                    for row in rows(source)
                ),
                max_topics=args.lexical_topics,
                max_documents=args.documents,
            )
            elapsed = time.monotonic() - started
            after = io_counters() if io_counters is not None else None
            topics = LexicalTopics(index, "synthetic-index", args.lexical_topics)
            result["lexical_build"] = {
                "seconds": elapsed,
                "records_per_second": args.documents / elapsed,
                "index_bytes": index["files"][0]["bytes"],
                "process_read_bytes": after.read_chars - before.read_chars
                if after is not None
                else None,
                "process_write_bytes": after.write_chars - before.write_chars
                if after is not None
                else None,
            }
        elif order in {"link_order", "entity_order"}:
            import sqlite3

            from klab.corpus.neighborhood_index import (
                NeighborhoodIndex,
                build_neighborhood_index,
            )

            kind = "links" if order == "link_order" else "entities"
            directory = root / f"{kind}-index"
            directory.mkdir()
            if kind == "links":
                rows = (
                    {"from_document_id": f"{i:064x}", "to_document_id": f"{i % 8:064x}"}
                    for i in range(8, args.documents)
                )
                limit = args.link_neighborhoods
                build_key, count_key = "link_build", "edges"
            else:
                rows = (
                    {
                        "document_id": f"{i:064x}",
                        "entity_id": f"synthetic:entity-{i % 8}",
                    }
                    for i in range(args.documents)
                )
                limit = args.entity_neighborhoods
                build_key, count_key = "entity_build", "mentions"
            facts = write_rows(root / f"{kind}.jsonl", rows)
            evidence = SimpleNamespace(sha256=facts["sha256"], max_neighborhoods=limit)
            before = io_counters() if io_counters is not None else None
            started = time.monotonic()
            index = build_neighborhood_index(
                directory,
                directory / "scratch.sqlite",
                Path(facts["path"]),
                evidence.sha256,
                (
                    {"document_id": f"{i:064x}", "split": "train", "stratum": "general"}
                    for i in range(args.documents)
                ),
                kind=kind,
                strata=("general",),
                max_neighborhoods=limit,
            )
            elapsed = time.monotonic() - started
            after = io_counters() if io_counters is not None else None
            result[build_key] = {
                "seconds": elapsed,
                count_key: facts["rows"],
                "evidence_bytes": facts["bytes"],
                "index_bytes": index["files"][0]["bytes"],
                "process_read_bytes": after.read_chars - before.read_chars
                if after is not None
                else None,
                "process_write_bytes": after.write_chars - before.write_chars
                if after is not None
                else None,
            }
            db = sqlite3.connect(
                Path(index["files"][0]["path"]).as_uri() + "?mode=ro&immutable=1",
                uri=True,
            )
            db.execute("PRAGMA cache_size=-8192")
            neighborhoods = NeighborhoodIndex(
                db, index, "synthetic-index", evidence, kind
            )
        before = io_counters() if io_counters is not None else None
        started = time.monotonic()
        with verification_settings(scratch_dir=root):
            members, coverage = order_sequences(
                staging / "train.records.jsonl",
                receipt,
                schedule,
                ordered,
                source_strata,
                {"general": 1},
                topics=topics,
                neighborhoods=neighborhoods,
            )
        if neighborhoods is not None:
            db.close()
        elapsed = time.monotonic() - started
        after = io_counters() if io_counters is not None else None
        result[order] = {
            **coverage,
            "sources": len(source_ids),
            "seconds": elapsed,
            "input_metadata_bytes": receipt["train_records_bytes"]
            + schedule.schedule.size_bytes,
            "process_read_bytes": after.read_chars - before.read_chars
            if after is not None
            else None,
            "process_write_bytes": after.write_chars - before.write_chars
            if after is not None
            else None,
            "schedule_sha256": members["token_schedule"]["sha256"],
        }
        if args.loader_blocks:
            ordered_receipt = {**receipt, **members}
            ordered_schedule = verify_schedule(
                ordered,
                schedule.cache_root,
                ordered_receipt,
                schedule.eos,
                memo={
                    (file.path, file.fingerprint): file
                    for file in (*schedule.tokens, *schedule.indexes)
                },
            )
            result[order]["loaders"] = benchmark_loaders(
                root / f"{order}-loader",
                ordered_schedule,
                ordered_receipt,
                ordered / "train.records.jsonl",
                blocks=args.loader_blocks,
                seq_len=args.loader_row_tokens,
            )
    if args.transfer_plan:
        from klab.corpus.mixture_transfer import plan_transfer

        process = psutil.Process()
        io_counters = getattr(process, "io_counters", None)
        before = io_counters() if io_counters is not None else None
        started = time.monotonic()
        with verification_settings(scratch_dir=root):
            plan = plan_transfer(schedule)
        elapsed = time.monotonic() - started
        after = io_counters() if io_counters is not None else None
        result["transfer_plan"] = {
            **plan,
            "seconds": elapsed,
            "process_read_bytes": after.read_chars - before.read_chars
            if after is not None
            else None,
        }
    if args.portable_segments:
        destination = root / "portable-segments"
        process = psutil.Process()
        io_counters = getattr(process, "io_counters", None)
        before = io_counters() if io_counters is not None else None
        started = time.monotonic()
        copies = schedule.copy_dependencies(destination)
        copied = time.monotonic()
        after = io_counters() if io_counters is not None else None
        verify_schedule(
            staging,
            destination,
            receipt,
            3,
            portable=True,
            memo={
                (proof.path, proof.fingerprint): proof
                for proof in (schedule.schedule, *copies.values())
            },
        )
        verified = time.monotonic()
        checked = io_counters() if io_counters is not None else None
        result["portable_segments"] = {
            "segments": len(schedule.tokens),
            "segment_files": len(copies),
            "logical_bytes": sum(proof.size_bytes for proof in copies.values()),
            "copy_seconds": copied - started,
            "copy_read_chars": after.read_chars - before.read_chars
            if after is not None
            else None,
            "verify_seconds": verified - copied,
            "verify_read_chars": checked.read_chars - after.read_chars
            if checked is not None
            else None,
        }
    if args.project_selected:
        with verification_settings(scratch_dir=root):
            result["selected_packaging"] = benchmark_projection(
                root / "projected",
                schedule,
                receipt,
                staging / "train.records.jsonl",
                seq_len=args.loader_row_tokens,
            )
    (root / "benchmark.json").write_bytes(canonical_json(result) + b"\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
