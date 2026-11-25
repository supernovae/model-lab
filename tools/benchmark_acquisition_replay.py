"""Measure synthetic acquisition replay with local fixtures and no network."""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import io
import json
import random
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import huggingface_hub
import pytest

from klab.canonical import canonical_digest
from klab.corpus import acquisition, wikimedia_ranges
from klab.corpus.project import load_project
from klab.corpus.release_verify import verification_settings, worker_count

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from corpus_fixtures import (
    serve_pinned_acquisition,
    write_acquisition_recipe,
)

MIB = 1024**2
CACHE_CASES = {"archive", "mediawiki_dump", "wikimedia_dump", "git", "http_document"}
DUMP_CASES = ("mediawiki_dump", "wikimedia_dump")
INDEXED_INTERRUPTS = ("member-checkpoint", "plan-group")
CASES = (
    "archive",
    "mediawiki_dump",
    "wikimedia_dump",
    "git",
    "huggingface_dataset",
    "local",
    "http_document",
)


class Interrupted(OSError):
    """Simulate interruption at the requested replay boundary."""


def synthetic_records(size: int, count: int) -> list[bytes]:
    generator = random.Random(485)
    return [
        generator.randbytes(size // count // 2).hex().encode() for _ in range(count)
    ]


def archive_body(records: list[bytes]) -> bytes:
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(
            fileobj=buffer, mode="wb", compresslevel=1, mtime=0
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        for index, record in enumerate(records, 1):
            member = tarfile.TarInfo(f"doc-{index:03d}.txt")
            member.size = len(record)
            archive.addfile(member, io.BytesIO(record))
    return buffer.getvalue()


def wiki_xml(records: list[bytes], dbname: str) -> bytes:
    prefix = (
        '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" xml:lang="en">'
        f"<siteinfo><dbname>{dbname}</dbname></siteinfo>"
    )
    output = io.BytesIO(prefix.encode())
    output.seek(0, io.SEEK_END)
    for index, record in enumerate(records, 1):
        output.write(
            f"<page><title>Article {index}</title><ns>0</ns><id>{index}</id>"
            f"<revision><id>{index + len(records)}</id>"
            "<timestamp>2026-10-01T00:00:00Z</timestamp><text>".encode()
        )
        output.write(record)
        output.write(b"</text></revision></page>")
    output.write(b"</mediawiki>")
    return output.getvalue()


def indexed_wiki_xml(
    xml: bytes, index_uri: str, pages_per_member: int
) -> tuple[bytes, bytes, dict]:
    header, _, articles = xml.partition(b"<page>")
    pages = (b"<page>" + articles).split(b"</page>")
    footer = pages.pop()
    parts = [header]
    parts.extend(
        b"".join(page + b"</page>" for page in pages[start : start + pages_per_member])
        for start in range(0, len(pages), pages_per_member)
    )
    parts.append(footer)
    body = io.BytesIO()
    rows = []
    for number, part in enumerate(parts):
        if 0 < number < len(parts) - 1:
            rows.extend(
                f"{body.tell()}:{page}:Article {page}\n"
                for page in range(
                    (number - 1) * pages_per_member + 1,
                    min(number * pages_per_member, len(pages)) + 1,
                )
            )
        body.write(bz2.compress(part, compresslevel=3))
    index_xml = "".join(rows).encode()
    index = bz2.compress(index_xml)
    return (
        body.getvalue(),
        index,
        {
            "uri": index_uri,
            "expected_sha256": hashlib.sha256(index).hexdigest(),
            "max_bytes": len(index),
            "max_expanded_bytes": len(index_xml),
            "max_members": len(parts),
            "max_member_expanded_bytes": max(map(len, parts)),
        },
    )


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def fixture(
    root: Path,
    kind: str,
    size: int,
    patch: pytest.MonkeyPatch,
    *,
    pages_per_member: int | None = None,
    selected_pages: int | None = None,
):
    records = synthetic_records(size, 256 if kind in CASES[:3] else 64)
    site = "https://fixture.example"
    uri, revision, license_id = f"{site}/manual.txt", "fixture", "MIT"
    bodies: dict[str, bytes] = {}
    hub_calls: list[str] = []
    if kind == "archive":
        uri = f"{site}/documents.tar.gz"
        body = bodies[uri] = archive_body(records)
        spec = {
            "expected_sha256": hashlib.sha256(body).hexdigest(),
            "format": "tar.gz",
            "include": ["*.txt"],
            "max_archive_bytes": len(body),
            "max_expanded_bytes": size + len(records) * 2048 + 16384,
            "max_members": len(records),
            "max_member_bytes": len(records[0]),
            "max_bytes": size,
        }
    elif kind in ("mediawiki_dump", "wikimedia_dump"):
        revision, license_id = "20261101", "CC-BY-SA-4.0"
        official = kind == "wikimedia_dump"
        dbname = "enwiki" if official else "fixturewiki"
        xml = wiki_xml(records, dbname)
        if official:
            prefix = f"https://dumps.wikimedia.org/enwiki/{revision}"
            name = f"enwiki-{revision}-pages-articles-multistream1.xml-p1p256.bz2"
            uri = f"{prefix}/{name}"
            checksum_uri = f"{prefix}/enwiki-{revision}-sha1sums.txt"
            index_uri = uri.replace("multistream1.xml", "multistream-index1.txt")
            if pages_per_member is not None:
                body, index_data, index = indexed_wiki_xml(
                    xml, index_uri, pages_per_member
                )
                spec = {"multistream_index": index}
            else:
                body = bz2.compress(xml, compresslevel=3)
                spec = {}
            bodies[uri] = body
            sha1 = hashlib.sha1(body).hexdigest()
            bodies[checksum_uri] = f"{sha1}  {name}\n".encode()
            spec.update(checksum_uri=checksum_uri, expected_sha1=sha1)
            if pages_per_member is not None:
                bodies[index_uri] = index_data
        else:
            uri = f"{site}/fixturewiki-{revision}.xml.gz"
            body = bodies[uri] = gzip.compress(xml, compresslevel=1, mtime=0)
            spec = {
                "compression": "gzip",
                "dbname": dbname,
                "language": "en",
                "site_url": site,
            }
        spec.update(
            expected_sha256=hashlib.sha256(body).hexdigest(),
            max_compressed_bytes=len(body),
            max_decompressed_bytes=len(xml),
            max_scanned_pages=len(records),
            max_selected_pages=selected_pages or len(records),
            max_emitted_bytes=len(xml) + size,
        )
    elif kind == "http_document":
        body = bodies[uri] = b"".join(records)
        spec = {
            "expected_sha256": hashlib.sha256(body).hexdigest(),
            "max_bytes": size,
        }
    else:
        revision = "d" * 40 if kind == "huggingface_dataset" else revision
        upstream = root / "upstream"
        folder = (
            upstream / revision / "default/train"
            if kind == "huggingface_dataset"
            else upstream
        )
        folder.mkdir(parents=True)
        files = []
        for index, record in enumerate(records, 1):
            path = folder / f"doc-{index:03d}.txt"
            if kind == "huggingface_dataset":
                path = path.with_suffix(".jsonl")
                record = b'{"text":"' + record + b'"}\n'
            path.write_bytes(record)
            files.append({"name": path.name, "path": str(path)})
        spec = {"max_bytes": size}
        if kind == "git":
            git(upstream, "init", "--object-format=sha1")
            git(upstream, "add", ".")
            git(
                upstream,
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "commit",
                "-m",
                "Synthetic acquisition replay",
            )
            revision, uri = git(upstream, "rev-parse", "HEAD"), str(upstream)
            spec["include"] = ["*.txt"]
        elif kind == "local":
            spec["files"] = files
        else:
            uri = "fixture/benchmark"
            spec.update(
                config="default",
                split="train",
                include=["default/train/*.jsonl"],
                text_field="text",
                max_rows=len(records),
                max_bytes=size + len(records) * 12,
            )

            def snapshot_download(**kwargs) -> str:
                hub_calls.append(kwargs["repo_id"])
                return str(upstream / revision)

            patch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    http_calls = serve_pinned_acquisition(patch, bodies)
    recipe = write_acquisition_recipe(
        root,
        kind=kind,
        acquisition=spec,
        revision=revision,
        uri=uri,
        license=license_id,
    )
    return load_project(recipe), http_calls, hub_calls, len(bodies.get(uri, b""))


def benchmark(
    root: Path,
    kind: str,
    size: int,
    trials: int,
    interruption: str,
    prefetch_bytes: int,
    selected_pages: int | None,
) -> None:
    with pytest.MonkeyPatch.context() as patch:
        indexed = interruption in INDEXED_INTERRUPTS
        project, http, hub, pinned_bytes = fixture(
            root,
            kind,
            size,
            patch,
            pages_per_member={"plan-group": 2, "member-checkpoint": 100}.get(
                interruption
            ),
            selected_pages=selected_pages,
        )
        fetches: list[str] = []
        blob_processes: list[str] = []
        popen = subprocess.Popen

        def record_git(args, *options, **kwargs):
            if isinstance(args, list) and args[0] == "git":
                if "fetch" in args:
                    fetches.append("git fetch")
                if "cat-file" in args:
                    blob_processes.append("git cat-file")
            return popen(args, *options, **kwargs)

        patch.setattr(subprocess, "Popen", record_git)
        verify = acquisition.verify_snapshot
        iterparse = ET.iterparse
        run_stage = acquisition.run_stage
        run_plan_stage = wikimedia_ranges.run_stage

        def interrupt(path, **options):
            if options.get("_staged"):
                raise Interrupted("replay benchmark")
            return verify(path, **options)

        def interrupt_xml(*args, **options):
            midpoint = project.sources[0].acquisition.max_selected_pages // 2
            pages = 0
            for event, element in iterparse(*args, **options):
                yield event, element
                if event == "end" and element.tag.endswith("}page"):
                    pages += 1
                    if pages == midpoint:
                        raise Interrupted("replay benchmark")

        def interrupt_member(store, stage, inputs, produce):
            result = run_stage(store, stage, inputs, produce)
            if stage == "wikimedia-member":
                raise Interrupted("replay benchmark")
            return result

        def interrupt_plan_group(store, stage, inputs, produce, **options):
            result = run_plan_stage(store, stage, inputs, produce, **options)
            if stage == "wikimedia-plan-group":
                raise Interrupted("replay benchmark")
            return result

        modes = (False, True) if kind in CACHE_CASES else (False,)
        expected_digest = None
        for cached in modes:
            elapsed = []
            cold_elapsed = []
            for _ in range(trials):
                with tempfile.TemporaryDirectory(dir=root) as attempt:
                    trial_root = Path(attempt)
                    cold_entry = None
                    if indexed and not cached:
                        started = time.perf_counter()
                        cold_lock = acquisition.acquire(
                            project,
                            trial_root / "cold",
                            wikimedia_prefetch_bytes=prefetch_bytes,
                        )
                        cold_elapsed.append(time.perf_counter() - started)
                        cold_entry = cold_lock["sources"]["one"]
                    work = trial_root / "work"
                    cache = trial_root / "cache" if cached else None
                    parent = work / "corpora/example"
                    snapshots = parent / "snapshots/one"
                    with pytest.MonkeyPatch.context() as fault:
                        if interruption == "mid-xml":
                            fault.setattr(ET, "iterparse", interrupt_xml)
                        elif interruption == "member-checkpoint":
                            fault.setattr(acquisition, "run_stage", interrupt_member)
                        elif interruption == "plan-group":
                            fault.setattr(
                                wikimedia_ranges, "run_stage", interrupt_plan_group
                            )
                        else:
                            fault.setattr(acquisition, "verify_snapshot", interrupt)
                        try:
                            acquisition.acquire(
                                project,
                                work,
                                transport_cache=cache,
                                wikimedia_prefetch_bytes=prefetch_bytes,
                            )
                        except (Interrupted, ValueError) as error:
                            # The range planner wraps I/O errors.
                            if not isinstance(error, Interrupted) and not isinstance(
                                error.__cause__, Interrupted
                            ):
                                raise
                        else:
                            raise RuntimeError(
                                "did not reach the interruption boundary"
                            )
                    if (parent / "acquisition.json").exists() or list(
                        snapshots.glob("*/manifest.json")
                    ):
                        raise RuntimeError(
                            "interrupted acquisition published a snapshot or lock"
                        )
                    for calls in (http, hub, fetches, blob_processes):
                        calls.clear()
                    started = time.perf_counter()
                    lock = acquisition.acquire(
                        project,
                        work,
                        transport_cache=cache,
                        wikimedia_prefetch_bytes=prefetch_bytes,
                    )
                    elapsed.append(time.perf_counter() - started)
                    entry = lock["sources"]["one"]
                    if (
                        cold_entry
                        and entry["snapshot_sha256"] != cold_entry["snapshot_sha256"]
                    ):
                        raise RuntimeError("snapshot identity changed on resume")
                    manifest = verify(Path(entry["snapshot_path"]))
                    digest = canonical_digest(manifest["files"])
                    if expected_digest is None:
                        expected_digest = digest
                    if digest != expected_digest:
                        raise RuntimeError(
                            "payload identity changed across retries or cache modes"
                        )
                    receipts = [
                        json.loads(path.read_text())
                        for path in (snapshots / ".resource-receipts").glob(
                            "attempt-*.json"
                        )
                    ]
                    completed = [
                        receipt["resources"]
                        for receipt in receipts
                        if receipt["status"] == "complete"
                    ]
                    peak = (
                        manifest["retrieval"]
                        .get("resources", {})
                        .get("peak_staging_bytes")
                    )
                    if peak is None:
                        peak = max(
                            (readings["peak_staging_bytes"] for readings in completed),
                            default=None,
                        )
            print(
                json.dumps(
                    {
                        "adapter": kind,
                        "interruption": interruption,
                        "warm_transport_cache": cached,
                        "worker_limit": worker_count(),
                        "wikimedia_prefetch_bytes": prefetch_bytes,
                        "selected_pages_limit": selected_pages,
                        "median_cold_seconds": round(statistics.median(cold_elapsed), 6)
                        if cold_elapsed
                        else None,
                        "median_retry_seconds": round(statistics.median(elapsed), 6),
                        "trials": trials,
                        "pinned_body_bytes": pinned_bytes or None,
                        "emitted_bytes": sum(
                            file["size"] for file in manifest["files"]
                        ),
                        "payload_inventory_sha256": digest,
                        "snapshot_sha256": entry["snapshot_sha256"],
                        "peak_staging_bytes": peak,
                        "retry_expanded_bytes": completed[0]["expanded_bytes"]
                        if completed
                        else None,
                        "http_requests": len(http),
                        "hub_lookups": len(hub),
                        "git_fetches": len(fetches),
                        "git_cat_file_processes": len(blob_processes),
                    }
                ),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=CASES, action="append")
    parser.add_argument(
        "--size-mib", type=int, choices=range(1, 65), default=16, metavar="1..64"
    )
    parser.add_argument(
        "--trials", type=int, choices=range(1, 11), default=3, metavar="1..10"
    )
    parser.add_argument("--scratch-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--wikimedia-prefetch-bytes", type=int, default=0)
    parser.add_argument(
        "--selected-pages", type=int, choices=range(1, 257), metavar="1..256"
    )
    parser.add_argument(
        "--interrupt-at",
        choices=("prepublish", "mid-xml", *INDEXED_INTERRUPTS),
        default="prepublish",
        help="Stop before publication, halfway through XML, or after the first indexed member or plan group",
    )
    args = parser.parse_args()
    allowed = DUMP_CASES if args.interrupt_at == "mid-xml" else CASES
    if args.interrupt_at in INDEXED_INTERRUPTS:
        allowed = ("wikimedia_dump",)
    kinds = args.adapter or allowed
    if any(kind not in allowed for kind in kinds):
        parser.error("--interrupt-at requires an eligible adapter")
    if args.selected_pages is not None and args.interrupt_at not in INDEXED_INTERRUPTS:
        parser.error("--selected-pages requires an indexed interruption case")
    if args.workers < 1 or args.wikimedia_prefetch_bytes < 0:
        parser.error(
            "--workers must be positive and --wikimedia-prefetch-bytes nonnegative"
        )
    for kind in kinds:
        with (
            tempfile.TemporaryDirectory(
                prefix="klab-replay-", dir=args.scratch_dir
            ) as scratch,
            pytest.MonkeyPatch.context() as patch,
            verification_settings(workers=args.workers),
        ):
            root = Path(scratch)
            patch.setenv("XDG_CONFIG_HOME", str(root / "config"))
            patch.delenv("KLAB_TRANSPORT_CACHE", raising=False)
            benchmark(
                root,
                kind,
                args.size_mib * MIB,
                args.trials,
                args.interrupt_at,
                args.wikimedia_prefetch_bytes,
                args.selected_pages,
            )


if __name__ == "__main__":
    main()
