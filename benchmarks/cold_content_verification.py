"""Read-only hash comparison with a common sequential OS-cache pre-read.

Run with uv run --locked --no-sync --with blake3 --with xxhash python
benchmarks/cold_content_verification.py --prepared-root /external/prepared \
  --fixture-root /external/fixtures --output /external/new-result.json
The pre-read does not drop caches or imply physical-disk cold I/O. It makes
all timed cases start after the same sequential file read policy instead.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from klab.host_capacity import hardware_observation, sha_work_plan

MIB = 1024 * 1024
FIXTURE_BYTES = 64 * MIB
CASES = (
    "sha_1",
    "sha_4",
    "sha_8",
    "sha_16",
    "file_digest",
    "blake3_1",
    "blake3_bounded",
    "xxh3_128",
)


def fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "mode": stat.st_mode,
    }


def fixture(path: Path) -> None:
    """Create one small, deterministic, exclusively owned, immutable input."""
    block = bytes(range(256)) * (MIB // 256)
    with path.open("xb") as stream:
        for _ in range(FIXTURE_BYTES // len(block)):
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def hash_stream(path: Path, case: str, threads: int) -> str:
    if case == "file_digest":
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    if case.startswith("sha_"):
        hasher = hashlib.sha256()
        chunk = int(case.split("_")[1]) * MIB
    elif case.startswith("blake3_"):
        import blake3

        hasher = blake3.blake3(max_threads=threads)
        chunk = 16 * MIB
    elif case == "xxh3_128":
        import xxhash

        hasher = xxhash.xxh3_128()
        chunk = 16 * MIB
    else:
        raise ValueError(f"unknown case: {case}")
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            hasher.update(data)
    return hasher.hexdigest()


def worker(path: Path, case: str, threads: int) -> dict:
    initial = fingerprint(path)
    # Identical sequential pre-read immediately before EACH timed sample.
    # This is an OS-cache policy, not an assertion of physical disk I/O.
    with path.open("rb") as stream:
        while stream.read(16 * MIB):
            pass
    if fingerprint(path) != initial:
        raise RuntimeError(f"file changed during pre-read: {path}")
    cpu_start = time.process_time()
    start = time.perf_counter()
    digest = hash_stream(path, case, threads)
    elapsed = time.perf_counter() - start
    cpu = time.process_time() - cpu_start
    final = fingerprint(path)
    if final != initial:
        raise RuntimeError(f"file changed during hash: {path}")
    return {
        "case": case,
        "digest": digest,
        "digest_authority": "non_authoritative_non_cryptographic"
        if case == "xxh3_128"
        else (
            "alternative_cryptographic_not_sha256_identity"
            if case.startswith("blake3_")
            else "sha256"
        ),
        "bytes": initial["size"],
        "wall_seconds": elapsed,
        "cpu_seconds": cpu,
        "throughput_mib_s": initial["size"] / MIB / elapsed,
        "threads": threads,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "fingerprint": initial,
    }


def benchmark(
    output: Path, root: Path, repetitions: int, timeout: int, prepared_root: Path
) -> dict:
    checkout = Path(__file__).resolve().parent.parent
    output = output.absolute()
    root = root.absolute()
    if checkout in output.parents or checkout in root.parents or root == checkout:
        raise ValueError("output and fixture root must be external to checkout")
    if output.exists():
        raise FileExistsError(output)
    if repetitions < 3 or timeout < 1:
        raise ValueError(
            "at least three repetitions and positive per-sample timeout required"
        )
    train_array = prepared_root.absolute() / "train.npy"
    if not train_array.is_file():
        raise FileNotFoundError(train_array)
    manifest_path = train_array.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_train_sha256 = manifest["train"]["sha256"]
    if manifest["train"]["size_bytes"] != train_array.stat().st_size:
        raise ValueError("prepared manifest train size differs from array")
    root.mkdir(parents=True, exist_ok=True)
    storage = os.statvfs(root)
    if (
        storage.f_bavail * storage.f_frsize < FIXTURE_BYTES + 256 * MIB
        or storage.f_favail < 10
    ):
        raise ValueError("insufficient free bytes/inodes for fixture")
    fixture_path = root / "cold-hash-fixture-64m-v1.bin"
    if not fixture_path.exists():
        fixture(fixture_path)
    elif fixture_path.stat().st_size != FIXTURE_BYTES:
        raise ValueError("existing fixture has unexpected size")
    # Bounded by the existing host planner; an operator may cap this below host capacity.
    plan = sha_work_plan(operator_cap=4)
    hardware = hardware_observation()
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "framework": {
            name: importlib.metadata.version(name)
            for name in ("blake3", "xxhash", "psutil")
        },
        "hardware": hardware,
        "host_plan": asdict(plan),
        "cache_policy": "16 MiB sequential pre-read of entire input before each timed case; OS cache uncontrolled; no cache drop; NOT physical disk cold",
        "timing_boundary": "hash only, after pre-read; process startup excluded; peak RSS includes fresh subprocess pre-read and hashing",
        "fixture_recipe": "64 MiB: bytes(range(256)) repeated, fsync, chmod 0444; existing fixture must match digest across repetitions",
    }
    inputs = {"fixture": fixture_path, "prepared_train": train_array}
    fixture_hasher = hashlib.sha256()
    block = bytes(range(256)) * (MIB // 256)
    for _ in range(FIXTURE_BYTES // MIB):
        fixture_hasher.update(block)
    expected_fixture_sha256 = fixture_hasher.hexdigest()
    expected_sha256 = {
        "fixture": expected_fixture_sha256,
        "prepared_train": expected_train_sha256,
    }
    baselines = {name: fingerprint(path) for name, path in inputs.items()}
    rows: list[dict] = []
    try:
        for name, path in inputs.items():
            for repeat in range(repetitions):
                order = CASES if repeat % 2 == 0 else tuple(reversed(CASES))
                for case in order:
                    threads = plan.workers if case == "blake3_bounded" else 1
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--sample",
                            str(path),
                            "--case",
                            case,
                            "--threads",
                            str(threads),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        check=True,
                    )
                    row = json.loads(completed.stdout)
                    if (
                        row["fingerprint"] != baselines[name]
                        or fingerprint(path) != baselines[name]
                    ):
                        raise RuntimeError(f"file changed: {path}")
                    row.update(input=name, repetition=repeat)
                    rows.append(row)
                    print(
                        f"{name} {repeat + 1}/{repetitions} {case}: {row['wall_seconds']:.3f}s",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        for name, path in inputs.items():
            if fingerprint(path) != baselines[name]:
                raise RuntimeError(f"input mutated: {path}")
    aggregate = {}
    for name in inputs:
        digests = {}
        for case in CASES:
            samples = [
                row for row in rows if row["input"] == name and row["case"] == case
            ]
            if (
                len(samples) != repetitions
                or len({row["digest"] for row in samples}) != 1
            ):
                raise RuntimeError(f"repetitions disagree: {name}/{case}")
            digests[case] = samples[0]["digest"]
            aggregate[f"{name}/{case}"] = {
                "median_wall_seconds": statistics.median(
                    row["wall_seconds"] for row in samples
                ),
                "median_cpu_seconds": statistics.median(
                    row["cpu_seconds"] for row in samples
                ),
                "median_throughput_mib_s": statistics.median(
                    row["throughput_mib_s"] for row in samples
                ),
                "max_peak_rss_bytes": max(row["peak_rss_bytes"] for row in samples),
                "threads": samples[0]["threads"],
            }
        if (
            len(
                {
                    digests[case]
                    for case in CASES
                    if case.startswith("sha_") or case == "file_digest"
                }
            )
            != 1
        ):
            raise RuntimeError(f"SHA256 variants disagree: {name}")
        if digests["sha_1"] != expected_sha256[name]:
            raise RuntimeError(f"SHA256 differs from expected fixture/manifest: {name}")
        if digests["blake3_1"] != digests["blake3_bounded"]:
            raise RuntimeError(f"BLAKE3 thread variants disagree: {name}")
    return {
        "format": "cold-content-verification-v2",
        "metadata": metadata,
        "inputs": {
            name: {"path": str(path), "fingerprint": baselines[name]}
            for name, path in inputs.items()
        },
        "expected_sha256": {
            "fixture_recipe": expected_fixture_sha256,
            "prepared_train_manifest": expected_train_sha256,
            "prepared_manifest_path": str(manifest_path),
        },
        "repetitions": repetitions,
        "samples": rows,
        "aggregate": aggregate,
        "status": "all_inputs_unchanged_and_digests_repeated",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new external JSON result path")
    parser.add_argument(
        "--fixture-root",
        type=Path,
        help="explicit external directory for the deterministic benchmark fixture",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        help="prepared train.npy and manifest.json directory",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--sample", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="per-sample operational deadline in seconds",
    )
    args = parser.parse_args()
    if args.sample is not None:
        if args.case is None or args.threads < 1:
            parser.error("sample needs --case and positive --threads")
        print(json.dumps(worker(args.sample, args.case, args.threads)))
    elif args.output is None or args.fixture_root is None or args.prepared_root is None:
        parser.error("--output, --fixture-root and --prepared-root are required")
    else:
        result = benchmark(
            args.output,
            args.fixture_root,
            args.repetitions,
            args.timeout,
            args.prepared_root,
        )
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(
            json.dumps({"output": str(args.output), "samples": len(result["samples"])})
        )


if __name__ == "__main__":
    main()
