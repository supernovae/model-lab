"""Bounded, offline preparation benchmark native interface."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from sparselab.canonical import canonical_json
from sparselab.config.models import StrictModel
from sparselab.documents import read_document
from sparselab.experiments.lock import _exclusive_bytes
from sparselab.training.manifest import sha256_file, source_identity
from sparselab.workspace_preflight import check_storage, require_storage

MIB = 1024 * 1024
_RESERVE_BYTES = 512 * MIB
_RESERVE_INODES = 2048
_MAX_SIZE = 32 * MIB
_MAX_TOTAL = 64 * MIB


class BenchmarkDeclaration(StrictModel):
    format: Literal["sparselab-preparation-benchmark-v1"]
    workspace: Path
    sizes_bytes: list[int] = Field(default_factory=lambda: [MIB])
    seed: int = Field(default=1701, strict=True, ge=0, le=2**32 - 1)
    batch_documents: list[int] = Field(default_factory=lambda: [1, 16])
    batch_source_bytes: int = Field(default=MIB, strict=True, ge=1, le=8 * MIB)
    threads: list[int] = Field(default_factory=lambda: [1])
    shard_workers: int = Field(default=1, strict=True, ge=1, le=32)
    worker_timeout_seconds: float = Field(default=120, strict=True, gt=0, le=3600)

    @field_validator("sizes_bytes", "batch_documents", "threads", mode="before")
    @classmethod
    def _strict_coordinates(cls, value: object) -> object:
        if not isinstance(value, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in value
        ):
            raise ValueError("coordinates must be nonempty strict integer lists")
        return value

    @field_validator("sizes_bytes", "batch_documents", "threads")
    @classmethod
    def _unique(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("coordinates must be nonempty strict integer lists")
        if len(value) != len(set(value)):
            raise ValueError("benchmark coordinates must be unique")
        return value

    @field_validator("sizes_bytes")
    @classmethod
    def _sizes(cls, value: list[int]) -> list[int]:
        if (
            any(item <= 0 or item > _MAX_SIZE for item in value)
            or sum(value) > _MAX_TOTAL
        ):
            raise ValueError(
                "sizes_bytes must be positive <=32 MiB with total <=64 MiB"
            )
        return value

    @field_validator("batch_documents")
    @classmethod
    def _documents(cls, value: list[int]) -> list[int]:
        if any(not 1 <= item <= 256 for item in value):
            raise ValueError("batch_documents must be in 1..256")
        return value

    @field_validator("threads")
    @classmethod
    def _threads(cls, value: list[int]) -> list[int]:
        cpu_count = os.cpu_count() or 1
        if any(item <= 0 or item > cpu_count for item in value):
            raise ValueError(f"threads must be in 1..{cpu_count}")
        return value

    @field_validator("worker_timeout_seconds")
    @classmethod
    def _finite_timeout(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("worker_timeout_seconds must be finite")
        return value


def _regular(path: Path, *, label: str) -> Path:
    if any(part == ".." for part in path.parts):
        raise ValueError(f"{label} must not contain traversal: {path}")
    candidate = path.absolute()
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} has symlinked component: {current}")
    if not candidate.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return candidate


def _safe_workspace(source: Path, workspace: Path) -> Path:
    if any(part == ".." for part in workspace.parts):
        raise ValueError("workspace must not contain traversal")
    candidate = workspace if workspace.is_absolute() else source.parent / workspace
    parent = candidate.parent
    if not parent.exists() or not parent.is_dir() or parent.is_symlink():
        raise ValueError("workspace parent must already exist and not be a symlink")
    current = parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"workspace has symlinked ancestor: {current}")
        current = current.parent
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"benchmark workspace already exists: {candidate}")
    return candidate.resolve(strict=False)


def _write_json(path: Path, payload: object) -> None:
    _exclusive_bytes(path, canonical_json(payload) + b"\n")


def _corpus(path: Path, target_bytes: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    records = source_bytes = 0
    with path.open("xb") as handle:
        while handle.tell() < target_bytes:
            number = rng.randrange(1_000_000)
            text = (
                f"{path.stem} record-{records}: The café fox 🦊 considers sample {number}, a sparse memory and a blue stone. "
                "Keep document boundaries, Unicode and EOS unchanged.\n"
                f"```python\ndef sparse_{number}(values):\n    return [x + {number % 101} for x in values if x % 3]\n```\n"
            ) * rng.choice((2, 4, 8, 16))
            handle.write(
                (json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8")
            )
            records += 1
            source_bytes += len(text.encode("utf-8"))
    return {
        "records": records,
        "source_utf8_bytes": source_bytes,
        "jsonl_bytes": path.stat().st_size,
    }


def _worker(case_file: Path) -> None:
    case = json.loads(_regular(case_file, label="case").read_text(encoding="utf-8"))

    result_path = Path(case["result"])
    started = time.perf_counter()
    before, children_before = (
        resource.getrusage(resource.RUSAGE_SELF),
        resource.getrusage(resource.RUSAGE_CHILDREN),
    )
    import psutil

    process, stop = psutil.Process(), threading.Event()
    sampled: dict[str, int | None] = {
        "parent_peak_rss_bytes": None,
        "process_tree_peak_rss_bytes": None,
    }

    def sample() -> None:
        while not stop.is_set():
            try:
                parent = process.memory_info().rss
                tree = parent + sum(
                    child.memory_info().rss
                    for child in process.children(recursive=True)
                )
                sampled["parent_peak_rss_bytes"] = max(
                    sampled["parent_peak_rss_bytes"] or 0, parent
                )
                sampled["process_tree_peak_rss_bytes"] = max(
                    sampled["process_tree_peak_rss_bytes"] or 0, tree
                )
            except psutil.Error, OSError:
                pass
            stop.wait(0.025)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        from sparselab.config.models import RunConfig
        from sparselab.data.encoding import PreparationEncoder
        from sparselab.data.packing import prepare_data
        from sparselab.data.tokenizer import load_tokenizer
        from sparselab.resource_envelope import ResourceEnvelope

        run = RunConfig.model_validate_json(
            Path(case["config"]).read_text(encoding="utf-8")
        )
        run = run.model_copy(
            update={
                "dataset": run.dataset.model_copy(
                    update={"cache_dir": Path(case["cache"])}
                )
            }
        )
        observed = {
            "maximum_batch_documents": 0,
            "maximum_batch_source_bytes": 0,
            "encode_calls": 0,
        }
        observation_lock = threading.Lock()
        original = PreparationEncoder.encode

        def bounded(self: PreparationEncoder, texts: list[str]):
            size = sum(len(text.encode("utf-8")) for text in texts)
            if len(texts) > case["documents"] or size > case["batch_source_bytes"]:
                raise ValueError("observed tokenizer batch exceeds benchmark caps")
            with observation_lock:
                observed["maximum_batch_documents"] = max(
                    observed["maximum_batch_documents"], len(texts)
                )
                observed["maximum_batch_source_bytes"] = max(
                    observed["maximum_batch_source_bytes"], size
                )
                observed["encode_calls"] += 1
            return original(self, texts)

        PreparationEncoder.encode = bounded
        prep_started = time.perf_counter()
        prepared = prepare_data(
            run,
            load_tokenizer(run.tokenizer.path),
            tokenizer_batch_documents=case["documents"],
            tokenizer_batch_source_bytes=case["batch_source_bytes"],
            resource_envelope=ResourceEnvelope(
                resource_envelope_version=1,
                max_workers=case["threads"],
                preparation_shard_workers=case["shard_workers"],
            ),
        )
        prep_seconds = time.perf_counter() - prep_started
        receipt = json.loads(
            (
                prepared.root.with_name(prepared.root.name + ".preparation.json")
            ).read_text(encoding="utf-8")
        )
        after = resource.getrusage(resource.RUSAGE_SELF)
        children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        elapsed = time.perf_counter() - started
        try:
            current_rss: int | None = process.memory_info().rss
        except psutil.Error, OSError:
            current_rss = None
        high_water = int(after.ru_maxrss)
        if sys.platform != "darwin":
            high_water *= 1024
        result: dict[str, Any] = {
            "status": "completed",
            "preparation_seconds": prep_seconds,
            "whole_worker_seconds": elapsed,
            "cpu_user_seconds": after.ru_utime
            - before.ru_utime
            + children_after.ru_utime
            - children_before.ru_utime,
            "cpu_system_seconds": after.ru_stime
            - before.ru_stime
            + children_after.ru_stime
            - children_before.ru_stime,
            "current_rss_bytes": current_rss,
            "os_high_water_rss_bytes": high_water,
            "sampled_process_tree_memory": {
                "scope": "worker_process_tree_sampled_25ms",
                **sampled,
            },
            "backend": "cpu",
            "runtime": {
                "python": str(Path(sys.executable).resolve()),
                "torch_version": str(sys.modules["torch"].__version__),
                "host": platform.platform(),
            },
            "rayon_threads": receipt["tokenizer_rayon_threads"],
            "shard_workers": receipt["shard_workers"],
            "requested_batch_documents": case["documents"],
            "requested_batch_source_bytes": case["batch_source_bytes"],
            "requested_threads": case["threads"],
            "observed_batches": observed,
            "arrays_sha256": {
                path.name: sha256_file(path)
                for path in sorted(prepared.root.glob("*.npy"))
            },
            "manifest_sha256": prepared.manifest["manifest_sha256"],
            "cache_identity": prepared.manifest["cache_identity"],
            "operational_receipt": receipt,
        }
        if receipt["tokenizer_host_work_plan"]["workers"] != case["threads"]:
            raise ValueError(
                "actual preparation total worker cap differs from benchmark request"
            )
        _write_json(result_path, result)
    finally:
        stop.set()
        sampler.join()


def _storage(declaration: BenchmarkDeclaration) -> tuple[int, int, str]:
    variants = len(declaration.batch_documents) * len(declaration.threads)
    scale = math.ceil(variants / 8)
    projected = sum((size + 2 * MIB) * 32 * scale for size in declaration.sizes_bytes)
    return (
        projected,
        _RESERVE_INODES,
        "(size_bytes + 2MiB generator allowance) * 32 * ceil(case_variants_per_size / 8), plus 512MiB/2048-inode reserve",
    )


def _failed_report(
    binding: dict[str, Any],
    storage: list[dict[str, object]],
    cases: list[dict[str, Any]],
    error: str,
) -> dict[str, Any]:
    return {
        "format": "sparselab-preparation-benchmark-report-v1",
        "status": "failed",
        "declaration": binding,
        "storage_admission": storage,
        "error": error,
        "cases": cases,
        "exact_scientific_parity_per_size": {},
    }


def _unmeasured_case(
    *, size: int, documents: int, threads: int, status: str, reason: str
) -> dict[str, Any]:
    return {
        "size_bytes": size,
        "batch_documents": documents,
        "threads": threads,
        "status": status,
        "reason": reason,
        "requested_batch_documents": documents,
        "requested_threads": threads,
        "observed_batches": None,
        "preparation_seconds": None,
        "whole_worker_seconds": None,
        "parent_worker_seconds": None,
        "cpu_user_seconds": None,
        "cpu_system_seconds": None,
        "current_rss_bytes": None,
        "os_high_water_rss_bytes": None,
        "sampled_process_tree_memory": {
            "scope": "worker_process_tree_sampled_25ms",
            "parent_peak_rss_bytes": None,
            "process_tree_peak_rss_bytes": None,
        },
        "arrays_sha256": None,
        "manifest_sha256": None,
        "cache_identity": None,
    }


def run_preparation_benchmark(source: Path) -> dict[str, Any]:
    source = _regular(source, label="benchmark declaration")
    raw = source.read_bytes()
    declaration = BenchmarkDeclaration.model_validate(read_document(source))
    if (
        len(declaration.sizes_bytes)
        * len(declaration.batch_documents)
        * len(declaration.threads)
        > 32
    ):
        raise ValueError("benchmark declaration exceeds 32 cases")
    workspace = _safe_workspace(source, declaration.workspace)
    projected, inodes, policy = _storage(declaration)
    storage = require_storage(
        [
            check_storage(
                workspace.parent,
                projected_bytes=projected,
                projected_inodes=inodes,
                reserve_bytes=_RESERVE_BYTES,
                reserve_inodes=_RESERVE_INODES,
            )
        ]
    )
    binding = {"path": str(source), "file_sha256": hashlib.sha256(raw).hexdigest()}
    workspace.mkdir(exist_ok=False)
    report_path = workspace / "report.json"
    cases: list[dict[str, Any]] = []
    corpus_started = time.perf_counter()
    corpus_seconds: float | None = None
    tokenizer_seconds: float | None = None
    fixtures: dict[int, tuple[Path, dict[str, int], dict[str, int]]] = {}
    try:
        from sparselab.config.models import RunConfig, TokenizerTrainConfig
        from sparselab.data.tokenizer import train_tokenizer

        for size in declaration.sizes_bytes:
            folder = workspace / f"size-{size}"
            folder.mkdir()
            train = _corpus(folder / "train.jsonl", size, declaration.seed)
            validation_target = max(1, min(size // 32, MIB))
            validation = _corpus(
                folder / "validation.jsonl",
                validation_target,
                (declaration.seed + 1202) % 2**32,
            )
            validation["requested_bytes"] = validation_target
            train["requested_bytes"] = size
            fixtures[size] = (folder, train, validation)
        corpus_seconds = time.perf_counter() - corpus_started
        first, train, validation = fixtures[declaration.sizes_bytes[0]]
        dataset = {
            "source": "local_text",
            "license": "generated-offline-fixture",
            "cache_dir": str(first / "unused-cache"),
            "train_path": str(first / "train.jsonl"),
            "validation_path": str(first / "validation.jsonl"),
            "train_max_documents": train["records"],
            "validation_max_documents": validation["records"],
            "train_max_tokens": train["source_utf8_bytes"] + train["records"],
            "validation_max_tokens": validation["source_utf8_bytes"]
            + validation["records"],
        }
        tokenizer_started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            tokenizer = train_tokenizer(
                TokenizerTrainConfig.model_validate(
                    {
                        "schema_version": 1,
                        "vocab_size": 512,
                        "min_frequency": 1,
                        "max_documents": min(256, train["records"]),
                        "output_dir": workspace / "tokenizer",
                        "dataset": dataset,
                    }
                )
            )
        tokenizer_seconds = time.perf_counter() - tokenizer_started
        for size, (folder, train, validation) in fixtures.items():
            values = dict(
                dataset,
                train_path=str(folder / "train.jsonl"),
                validation_path=str(folder / "validation.jsonl"),
                train_max_documents=train["records"],
                validation_max_documents=validation["records"],
                train_max_tokens=train["source_utf8_bytes"] + train["records"],
                validation_max_tokens=validation["source_utf8_bytes"]
                + validation["records"],
            )
            run = RunConfig.model_validate(
                {
                    "schema_version": 2,
                    "name": "preparation-benchmark",
                    "seed": declaration.seed,
                    "runtime": {
                        "engine": "pytorch",
                        "backend": "cpu",
                        "precision": "fp32",
                    },
                    "model": {
                        "vocab_size": 512,
                        "hidden_dim": 16,
                        "num_layers": 1,
                        "num_heads": 2,
                        "ffn_dim": 32,
                        "max_seq_len": 32,
                    },
                    "tokenizer": {"path": tokenizer},
                    "dataset": values,
                    "training": {
                        "micro_batch_size": 1,
                        "seq_len": 16,
                        "max_steps": 2,
                        "max_tokens": 32,
                    },
                    "optimizer": {"name": "adamw", "warmup_steps": 1},
                    "logging": {"root_dir": workspace / "runs"},
                }
            )
            config = folder / "run.json"
            config.write_text(run.model_dump_json(), encoding="utf-8")
            for threads in declaration.threads:
                for documents in declaration.batch_documents:
                    case_root = folder / f"threads-{threads}-batch-{documents}"
                    case_root.mkdir()
                    case_file = case_root / "case.json"
                    result = case_root / "result.json"
                    case_file.write_bytes(
                        canonical_json(
                            {
                                "config": str(config),
                                "cache": str(case_root / "cache"),
                                "documents": documents,
                                "threads": threads,
                                "batch_source_bytes": declaration.batch_source_bytes,
                                "shard_workers": declaration.shard_workers,
                                "result": str(result),
                            }
                        )
                    )
                    env = os.environ.copy()
                    env.update(
                        {
                            "HF_HUB_OFFLINE": "1",
                            "HF_DATASETS_OFFLINE": "1",
                            "UV_OFFLINE": "1",
                            "SPARSELAB_WORK_DIR": str(workspace),
                            "RAYON_NUM_THREADS": str(threads),
                            "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                            "VECLIB_MAXIMUM_THREADS": "1",
                        }
                    )
                    command = [
                        sys.executable,
                        "-m",
                        "sparselab.data.preparation_benchmark",
                        "--worker",
                        str(case_file),
                    ]
                    parent_worker_started = time.perf_counter()
                    try:
                        with (
                            (case_root / "stdout.log").open("wb") as stdout,
                            (case_root / "stderr.log").open("wb") as stderr,
                        ):
                            completed = subprocess.run(
                                command,
                                env=env,
                                stdout=stdout,
                                stderr=stderr,
                                timeout=declaration.worker_timeout_seconds,
                                check=False,
                            )
                        parent_worker_seconds = (
                            time.perf_counter() - parent_worker_started
                        )
                        row: dict[str, Any] = {
                            "size_bytes": size,
                            "batch_documents": documents,
                            "threads": threads,
                            "log_path": str(case_root),
                            "result_path": str(result),
                            "parent_worker_seconds": parent_worker_seconds,
                        }
                        if completed.returncode or not result.is_file():
                            row = _unmeasured_case(
                                size=size,
                                documents=documents,
                                threads=threads,
                                status="failed",
                                reason="worker_failed",
                            )
                            row.update(
                                log_path=str(case_root),
                                result_path=str(result),
                                parent_worker_seconds=parent_worker_seconds,
                                returncode=completed.returncode,
                            )
                        else:
                            row.update(json.loads(result.read_text(encoding="utf-8")))
                            row["parent_worker_seconds"] = parent_worker_seconds
                    except subprocess.TimeoutExpired:
                        row = _unmeasured_case(
                            size=size,
                            documents=documents,
                            threads=threads,
                            status="timed_out",
                            reason="worker_timeout",
                        )
                        row.update(
                            log_path=str(case_root),
                            result_path=str(result),
                            parent_worker_seconds=time.perf_counter()
                            - parent_worker_started,
                        )
                    cases.append(row)
        if _regular(source, label="benchmark declaration").read_bytes() != raw:
            raise ValueError("benchmark declaration changed during execution")

        parity: dict[str, bool] = {}
        for size in declaration.sizes_bytes:
            rows = [row for row in cases if row["size_bytes"] == size]
            identities = {
                (
                    json.dumps(row["arrays_sha256"], sort_keys=True),
                    row["manifest_sha256"],
                    json.dumps(row["cache_identity"], sort_keys=True),
                )
                for row in rows
                if row["status"] == "completed"
            }
            parity[str(size)] = (
                len(identities) == 1
                and len(rows)
                == len(declaration.batch_documents) * len(declaration.threads)
                and all(row["status"] == "completed" for row in rows)
            )
        report: dict[str, Any] = {
            "format": "sparselab-preparation-benchmark-report-v1",
            "status": "completed" if all(parity.values()) else "failed",
            "declaration": binding,
            "workspace": str(workspace),
            "storage_admission": storage,
            "storage_estimation_policy": policy,
            "projected_bytes": projected,
            "phase_seconds": {
                "corpus_generation": corpus_seconds,
                "tokenizer_training": tokenizer_seconds,
            },
            "host": {
                "platform": platform.platform(),
                "python": str(Path(sys.executable).resolve()),
                "logical_cpu_count": os.cpu_count(),
                "framework": {
                    "torch_version": getattr(
                        sys.modules.get("torch"), "__version__", None
                    )
                },
                "source_identity": source_identity(),
            },
            "generated": {
                str(size): {"train": train, "validation": validation}
                for size, (_, train, validation) in fixtures.items()
            },
            "cases": cases,
            "exact_scientific_parity_per_size": parity,
        }
    except BaseException as error:  # noqa: BLE001 -- retain failed/interrupted setup evidence after the exclusive claim
        report = _failed_report(
            binding, storage, cases, f"{type(error).__name__}: {error}"
        )
        for size in declaration.sizes_bytes:
            for threads in declaration.threads:
                for documents in declaration.batch_documents:
                    if not any(
                        row.get("size_bytes") == size
                        and row.get("threads") == threads
                        and row.get("batch_documents") == documents
                        for row in cases
                    ):
                        cases.append(
                            _unmeasured_case(
                                size=size,
                                documents=documents,
                                threads=threads,
                                status="failed",
                                reason="setup_failed",
                            )
                        )
        report["cases"] = cases
        report["exact_scientific_parity_per_size"] = {
            str(size): False for size in declaration.sizes_bytes
        }
    report.setdefault("workspace", str(workspace))
    report.setdefault("storage_estimation_policy", policy)
    report.setdefault("projected_bytes", projected)
    report.setdefault(
        "phase_seconds",
        {"corpus_generation": corpus_seconds, "tokenizer_training": tokenizer_seconds},
    )
    report.setdefault(
        "generated",
        {
            str(size): {"train": train, "validation": validation}
            for size, (_, train, validation) in fixtures.items()
        },
    )
    report.setdefault(
        "exact_scientific_parity_per_size",
        {str(size): False for size in declaration.sizes_bytes},
    )
    report.setdefault(
        "host",
        {
            "platform": platform.platform(),
            "python": str(Path(sys.executable).resolve()),
            "logical_cpu_count": os.cpu_count(),
            "framework": {
                "torch_version": getattr(sys.modules.get("torch"), "__version__", None)
            },
            "source_identity": source_identity(),
        },
    )
    report["observed_storage_after"] = asdict(
        check_storage(
            workspace.parent,
            projected_bytes=0,
            projected_inodes=0,
            reserve_bytes=0,
            reserve_inodes=0,
        )
    )
    published = canonical_json(report)
    _exclusive_bytes(report_path, published + b"\n")
    return json.loads(published)


def _handle(args: argparse.Namespace) -> None:
    try:
        report = run_preparation_benchmark(args.declaration)
    except (OSError, ValueError, TypeError) as error:
        if args.json:
            print(json.dumps({"status": "error", "error": str(error)}, sort_keys=True))
        else:
            print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"{report['status']}: {report.get('workspace', args.declaration)}")
    if report["status"] != "completed":
        raise SystemExit(1)


def register_parser(data_commands: argparse._SubParsersAction) -> None:
    command = data_commands.add_parser(
        "benchmark-preparation", help="Bounded offline preparation benchmark"
    )
    command.add_argument("declaration", type=Path)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_handle, deferred_workspace=True)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker is None:
        parser.error("--worker is required")
    _worker(args.worker)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
