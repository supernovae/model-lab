"""Read-only input benchmarks; all generated outputs belong to a fresh workspace.

No training, historical authority rewriting, cache dropping or timing assertions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import ssl
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import psutil

MIB = 1024 * 1024


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def check_identities(cases: list[dict], paths: dict) -> dict:
    """Compare authenticated outputs, never independent runtime timestamp receipts."""
    values: dict[str, set[str]] = {}
    for case in cases:
        if case["status"] != "ok":
            raise RuntimeError(f"unavailable benchmark case: {case}")
        identity = case["identities"]
        arrays = identity.get("array_sha256", identity.get("sha256"))
        if isinstance(arrays, dict) and arrays != paths["array_sha256"]:
            raise ValueError(f"array SHA mismatch: {case['case']}")
        for field in (
            "manifest_sha256",
            "plan_sha256",
            "scientific_sha256",
            "stage_sha256",
            "bundle_digest",
        ):
            if field in identity:
                values.setdefault(field, set()).add(identity[field])
        if (
            case["case"].endswith("warm")
            and case["case"] != "stage_verify_warm"
            and case["logical_sha_measurement"]["payload_sha_calls"] != 0
        ):
            raise ValueError(f"warm payload rescan: {case['case']}")
    if any(len(group) != 1 for group in values.values()):
        raise ValueError(f"within-revision closure identity mismatch: {values}")
    return {
        "arrays_equal_1_2_N_cold_warm_copy_methods": True,
        "frozen_stage_manifest_equal_native_buffered_cold_warm": True,
        "dispatch_manifest_equal_native_buffered_cold_warm": True,
        "plan_scientific_and_prepared_manifest_equal_cold_warm": True,
        "warm_zero_payload_sha_except_private_hardlinked_stage": True,
    }


def metadata() -> dict:
    from klab.host_capacity import hardware_observation
    from klab.training.manifest import source_identity

    hardware = hardware_observation()
    versions = {}
    for name in ("tokenizers", "numpy", "zstandard", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_measured"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "host": platform.platform(),
        "machine": platform.machine(),
        "openssl": ssl.OPENSSL_VERSION,
        "hashlib_sha256_module": hashlib.sha256.__module__,
        "versions": versions,
        "rayon": "tokenizers native runtime; separately observed in preparation receipt",
        "compression": {"zlib": __import__("zlib").ZLIB_RUNTIME_VERSION},
        "isa": hardware["isa"],
        "hardware_observation": hardware,
        "source_implementation_sha256": source_identity()["sha256"],
        "dirty_tree": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def fixture(root: Path, medium: bool) -> dict:
    from preparation_benchmark import corpus

    from klab.config.models import RunConfig, TokenizerTrainConfig
    from klab.data.packing import prepare_data
    from klab.data.tokenizer import load_tokenizer, train_tokenizer
    from klab.experiments.lock import publish_lock, resolve_plan
    from klab.experiments.plan import ExperimentPlan
    from klab.staging import stage
    from klab.training.manifest import sha256_file
    from klab.workers.controller import Controller

    root.mkdir(mode=0o700)
    train = corpus(root / "train.jsonl", (64 * MIB if medium else 16 * 1024), 1701)
    validation = corpus(root / "validation.jsonl", (MIB if medium else 4096), 2903)
    dataset = {
        "source": "synthetic",
        "cache_dir": str(root / "cache"),
        "train_max_documents": 600000 if medium else 200,
        "validation_max_documents": 10000 if medium else 40,
        "train_max_tokens": 16 * MIB if medium else 8192,
        "validation_max_tokens": MIB // 4 if medium else 2048,
        "synthetic_seed": 1701,
    }
    tokenizer = train_tokenizer(
        TokenizerTrainConfig.model_validate(
            {
                "schema_version": 1,
                "vocab_size": 512,
                "min_frequency": 1,
                "max_documents": 256 if medium else 200,
                "output_dir": root / "tokenizer",
                "dataset": {
                    **dataset,
                    "train_max_tokens": max(32768, dataset["train_max_tokens"]),
                },
            }
        )
    )
    config = RunConfig.model_validate(
        {
            "schema_version": 2,
            "name": "capacity-fixture",
            "seed": 1701,
            "runtime": {"engine": "pytorch", "backend": "cpu", "precision": "fp32"},
            "model": {
                "vocab_size": 512,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "ffn_dim": 32,
                "max_seq_len": 32,
            },
            "tokenizer": {"path": tokenizer},
            "dataset": dataset,
            "training": {
                "micro_batch_size": 1,
                "seq_len": 16,
                "max_steps": 2,
                "max_tokens": 32,
            },
            "optimizer": {"name": "adamw", "warmup_steps": 1},
            "logging": {"root_dir": root / "runs"},
        }
    )
    prepared = prepare_data(config, load_tokenizer(tokenizer))
    plan = ExperimentPlan.model_validate(
        {
            "plan_version": 1,
            "id": "capacity-fixture",
            "base_run": config,
            "artifacts": {
                "tokenizer": {
                    "kind": "tokenizer",
                    "version": 1,
                    "producer": "klab",
                    "identifier": tokenizer.parent.name,
                    "sha256": sha256_file(tokenizer),
                    "path": str(tokenizer),
                },
                "packed": {
                    "kind": "prepared_data",
                    "version": 1,
                    "producer": "klab",
                    "identifier": prepared.manifest["settings_sha256"],
                    "sha256": prepared.manifest["manifest_sha256"],
                    "path": str(prepared.root),
                },
            },
            "inputs": {"tokenizer": "tokenizer", "training": "packed"},
        }
    )
    source = root / "plan.yaml"
    source.write_text(plan.model_dump_json())
    lock = publish_lock(resolve_plan(plan, source), root)
    config_path = root / "config.json"
    config_path.write_text(config.model_dump_json())
    bundle = stage(config, root / "stage", through="validate")
    controller = Controller(root / "controller")
    from klab.verification_proofs import verification_options
    from klab.workers.bundles import (
        install_dispatch_bundle,
        prepare_dispatch_bundle,
    )

    dispatch_root = root / "dispatch"
    dispatch = prepare_dispatch_bundle(config, dispatch_root, stage_bundle=bundle)
    worker_root = root / "worker"
    worker_root.mkdir(mode=0o700)
    install_dispatch_bundle(
        worker_root,
        dispatch,
        {
            f"assets/{item.sha256}": dispatch_root / item.relative_path
            for item in dispatch.files
        },
        check_only=False,
        **verification_options(root),
    )
    # Prime in this process; all measured warm consumers run in new processes.
    from klab.data.packing import load_prepared_data
    from klab.experiments.lock import open_lock

    options = verification_options(root)
    load_prepared_data(prepared.root, byte_enabled=False, **options)
    resolve_plan(plan, source, **options)
    open_lock(lock, **options)
    controller.store.enqueue_many(
        [
            {
                "experiment_id": "fixture-experiment",
                "attempt_id": "fixture-attempt",
                "run_id": "fixture-run",
                "spec": {"schema_version": 1, "experiment_id": "fixture-experiment"},
            }
        ]
    )
    campaign = root / "campaign.yaml"
    save(
        campaign,
        {
            "campaign_version": 1,
            "id": "capacity-fixture",
            "stages": [
                {
                    "id": "tokenizer",
                    "kind": "tokenizer_reference",
                    "scope": "tokenizer",
                    "artifact": plan.artifacts["tokenizer"].model_dump(
                        mode="json", exclude_none=True
                    ),
                }
            ],
        },
    )
    return {
        "config": str(config_path),
        "plan": str(source),
        "lock": str(lock),
        "stage": str(bundle),
        "prepared": str(prepared.root),
        "controller": str(controller.root),
        "campaign": str(campaign),
        "work_dir": str(root / "campaign-work"),
        "proof_root": str(root),
        "dispatch": str(dispatch_root),
        "worker": str(worker_root),
        "bundle_digest": dispatch.digest(),
        "array_sha256": {k: p.sha256 for k, p in prepared.receipt.proofs.items()},
        "source_counts": {"train": train, "validation": validation},
    }


def operation(case: str, paths: dict, destination: Path, workers: int) -> dict:
    from klab.config.models import RunConfig
    from klab.data.packing import load_prepared_data
    from klab.experiments.lock import open_lock, resolve_plan
    from klab.experiments.plan import load_plan
    from klab.staging import materialize_prepared_inputs, verify_stage_bundle
    from klab.training.manifest import sha256_file
    from klab.workers.controller import Controller

    if case == "campaign_read":
        from klab.campaign.engine import CampaignEngine

        engine = CampaignEngine(Path(paths["campaign"]), Path(paths["work_dir"]))
        result = engine.inspect("status")
        return {"status": result.get("status"), "id": engine.plan.id}

    root = Path(paths["prepared"])
    if case == "prepared_cold":
        data = load_prepared_data(root, byte_enabled=False)
        return {"manifest_sha256": data.manifest["manifest_sha256"]}
    if case == "sha":
        names = sorted(root.glob("*.npy"))
        if workers == 1:
            values = [sha256_file(p) for p in names]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as executor:
                values = list(executor.map(sha256_file, names))
        return {"sha256": dict(zip((p.name for p in names), values, strict=True))}
    if case == "resolve":
        locked = resolve_plan(load_plan(Path(paths["plan"])), Path(paths["plan"]))
        return {
            "plan_sha256": locked.plan_sha256,
            "scientific_sha256": locked.scientific_sha256,
        }
    if case in {"open", "explain"}:
        locked = open_lock(Path(paths["lock"]))
        return {
            "plan_sha256": locked.plan_sha256,
            "scientific_sha256": locked.scientific_sha256,
            "cells": len(locked.cells),
        }
    if case == "controller_read":
        controller = Controller(Path(paths["controller"]), read_only=True)
        attempts = controller.store.attempts()
        return {
            "attempt_states": [(r["attempt_id"], r["status"]) for r in attempts],
            "expired_tick": controller.tick(deadline=-1.0),
        }
    config = RunConfig.model_validate_json(Path(paths["config"]).read_text())
    if case == "stage_verify":
        return {
            "stage_sha256": verify_stage_bundle(Path(paths["stage"]), config)["sha256"]
        }
    if case == "materialize":
        out = materialize_prepared_inputs(config, destination)
        return {
            "sha256": {
                p.name: sha256_file(p)
                for p in sorted((out / "assets/data").glob("*.npy"))
            }
        }
    raise ValueError(f"unknown benchmark case: {case}")


def extended_operation(case: str, paths: dict, destination: Path, workers: int) -> dict:
    import errno
    from contextlib import ExitStack
    from unittest.mock import patch

    from klab.config.models import RunConfig
    from klab.data.packing import load_prepared_data
    from klab.experiments.lock import open_lock, resolve_plan
    from klab.experiments.plan import load_plan
    from klab.host_capacity import sha_work_plan
    from klab.staging import (
        materialize_prepared_inputs,
        verify_stage_bundle,
    )
    from klab.verification_proofs import verification_options
    from klab.workers.bundles import (
        install_dispatch_bundle,
        materialize_dispatch_bundle,
        prepare_dispatch_bundle,
        verify_dispatch_bundle,
    )

    root = Path(paths["prepared"])
    options = verification_options(Path(paths["proof_root"]))
    if case == "real_owned_copy":
        from klab.owned_copy import owned_copy

        data = load_prepared_data(root, byte_enabled=False)
        destination.mkdir(mode=0o700)
        copies = []
        for member in sorted(root.iterdir()):
            if not member.is_file() or member.is_symlink():
                raise ValueError("real prepared source must contain regular files")
            copied = owned_copy(
                member,
                destination / member.name,
                proof=data.receipt.proofs.get(member.name),
            )
            copies.append(
                {
                    "mechanism": copied.mechanism,
                    "logical_bytes": copied.logical_bytes,
                }
            )
        verified = load_prepared_data(destination, byte_enabled=False, **options)
        return {
            "manifest_sha256": verified.manifest["manifest_sha256"],
            "array_sha256": {k: p.sha256 for k, p in verified.receipt.proofs.items()},
            "copy_observations": copies,
        }
    if case.startswith("prepared_"):
        plan = sha_work_plan(operator_cap=workers)
        with patch("klab.data.packing.sha_work_plan", return_value=plan):
            data = load_prepared_data(
                root,
                byte_enabled=False,
                **(options if case == "prepared_warm" else {}),
            )
        return {
            "manifest_sha256": data.manifest["manifest_sha256"],
            "array_sha256": {k: p.sha256 for k, p in data.receipt.proofs.items()},
            "capacity_plan": asdict(plan),
            "payload_bytes": sum(p.size_bytes for p in data.receipt.proofs.values()),
        }
    if case in {"resolve_warm", "open_warm", "explain_warm"}:
        locked = (
            resolve_plan(load_plan(Path(paths["plan"])), Path(paths["plan"]), **options)
            if case == "resolve_warm"
            else open_lock(Path(paths["lock"]), **options)
        )
        return {
            "plan_sha256": locked.plan_sha256,
            "scientific_sha256": locked.scientific_sha256,
        }
    config = RunConfig.model_validate_json(Path(paths["config"]).read_text())
    observations: list[dict[str, object]] = []
    if case in {"worker_install_warm", "worker_install_transfer"}:
        from klab.workers.models import BundleManifest

        manifest = BundleManifest.model_validate_json(
            (Path(paths["dispatch"]) / "bundle.json").read_text()
        )
        if case == "worker_install_transfer":
            destination.mkdir(mode=0o700)
        result = install_dispatch_bundle(
            destination if case == "worker_install_transfer" else Path(paths["worker"]),
            manifest,
            {
                f"assets/{item.sha256}": Path(paths["dispatch"]) / item.relative_path
                for item in manifest.files
            }
            if case == "worker_install_transfer"
            else {},
            check_only=False,
            copy_observations=observations,
            **options,
        )
        return {**result, "copy_observations": observations}
    if case.startswith("worker_materialize"):
        manifest = materialize_dispatch_bundle(
            Path(paths["worker"]),
            paths["bundle_digest"],
            destination,
            copy_observations=observations,
            **(options if case.endswith("warm") else {}),
        )
        return {
            "bundle_digest": manifest.digest(),
            "array_sha256": paths["array_sha256"],
            "copy_observations": observations,
            "_validation_root": str(destination),
        }
    with ExitStack() as stack:
        if case.endswith("buffered"):
            stack.enter_context(
                patch(
                    "klab.owned_copy.fcntl.ioctl",
                    side_effect=OSError(errno.EOPNOTSUPP, "benchmark forced fallback"),
                )
            )
            if hasattr(os, "copy_file_range"):
                stack.enter_context(
                    patch(
                        "klab.owned_copy.os.copy_file_range",
                        side_effect=OSError(
                            errno.EOPNOTSUPP, "benchmark forced fallback"
                        ),
                    )
                )
        if case.startswith("stage_copy"):
            from klab.owned_copy import owned_copy

            source = Path(paths["stage"])
            destination.mkdir(mode=0o700)
            for member in sorted(source.rglob("*")):
                if member.is_symlink():
                    raise ValueError("stage source contains symlink")
                if not member.is_file():
                    continue
                copied = owned_copy(member, destination / member.relative_to(source))
                observations.append(
                    {
                        "mechanism": copied.mechanism,
                        "logical_bytes": copied.logical_bytes,
                    }
                )
            return {
                "stage_sha256": verify_stage_bundle(destination, config)["sha256"],
                "array_sha256": paths["array_sha256"],
                "_validation_root": str(destination),
                "copy_observations": observations,
            }
        if case.startswith("dispatch_create"):
            manifest = prepare_dispatch_bundle(
                config,
                destination,
                stage_bundle=Path(paths["stage"]),
                **options,
            )
            verified = verify_dispatch_bundle(destination)
            assert verified.digest() == manifest.digest()
            return {
                "bundle_digest": manifest.digest(),
                "array_sha256": paths["array_sha256"],
                "_validation_root": str(destination),
            }
        if case == "stage_verify_warm":
            return {
                "stage_sha256": verify_stage_bundle(
                    Path(paths["stage"]),
                    config,
                    **options,
                )["sha256"],
            }
        if case == "materialize_warm":
            materialize_prepared_inputs(
                config,
                destination,
                copy_observations=observations,
                **options,
            )
            return {
                "array_sha256": paths["array_sha256"],
                "copy_observations": observations,
                "_validation_root": str(destination),
            }
    raise ValueError(f"unknown extended case: {case}")


def measured_extended_operation(
    case: str,
    paths: dict,
    destination: Path,
    workers: int,
) -> dict:
    """Count logical SHA work, not physical I/O; preserve cold baseline boundaries."""
    from unittest.mock import patch

    from klab import owned_copy as copy_module
    from klab.training import manifest as manifest_module

    digests = set(paths.get("array_sha256", {}).values())
    counters = {
        "payload_sha_calls": 0,
        "logical_payload_sha_read_bytes": 0,
        "copy_authentication_sha_read_bytes": 0,
    }
    original_hash = manifest_module.sha256_file
    original_fd_hash = copy_module._hash_fd

    def counted(path: Path, **kwargs: object) -> str:
        path = Path(path)
        if path.suffix == ".npy" or path.name in digests:
            counters["payload_sha_calls"] += 1
            counters["logical_payload_sha_read_bytes"] += path.stat().st_size
        return original_hash(path, **kwargs)

    def counted_fd(fd: int) -> str:
        counters["copy_authentication_sha_read_bytes"] += os.fstat(fd).st_size
        return original_fd_hash(fd)

    with (
        patch("klab.training.manifest.sha256_file", counted),
        patch("klab.staging.sha256_file", counted),
        patch("klab.workers.bundles.sha256_file", counted),
        patch("klab.owned_copy._hash_fd", counted_fd),
    ):
        result = extended_operation(case, paths, destination, workers)
    payload_bytes = sum(
        path.stat().st_size for path in Path(paths["prepared"]).glob("*.npy")
    )
    counters["logical_payload_bytes_avoided"] = (
        max(0, payload_bytes - counters["logical_payload_sha_read_bytes"])
        if case.endswith("warm")
        else 0
    )
    result["_measurement"] = counters
    return result


def process_io(process: psutil.Process) -> dict[str, int] | None:
    try:
        counters = process.io_counters()
        return {"read_bytes": counters.read_bytes, "write_bytes": counters.write_bytes}
    except psutil.Error, AttributeError:
        try:
            with Path(f"/proc/{process.pid}/io").open() as stream:
                raw = stream.read(4096)
            return {
                key: int(value.strip())
                for key, value in (line.split(":", 1) for line in raw.splitlines())
            }
        except OSError, ValueError:
            return None


def worker(args: argparse.Namespace) -> None:
    paths = json.loads(args.paths.read_text())
    process = psutil.Process()
    stop = threading.Event()
    observed = {"peak_tree_rss_bytes": 0, "peak_tree_swap_bytes": 0}

    def sample() -> None:
        while not stop.is_set():
            rss = swap = 0
            for child in [process, *process.children(recursive=True)]:
                try:
                    rss += child.memory_info().rss
                    swap += getattr(child.memory_full_info(), "swap", 0)
                except psutil.Error, OSError:
                    pass
            observed["peak_tree_rss_bytes"] = max(observed["peak_tree_rss_bytes"], rss)
            observed["peak_tree_swap_bytes"] = max(
                observed["peak_tree_swap_bytes"], swap
            )
            stop.wait(0.025)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    before = resource.getrusage(resource.RUSAGE_SELF)
    io_before = process_io(process)
    started = time.perf_counter()
    try:
        identity = (
            operation(args.case, paths, args.destination, args.workers)
            if args.case
            in {
                "prepared_cold",
                "resolve",
                "open",
                "explain",
                "stage_verify",
                "materialize",
                "controller_read",
                "campaign_read",
                "sha",
            }
            else measured_extended_operation(
                args.case,
                paths,
                args.destination,
                args.workers,
            )
        )
        result = {"status": "ok", "identities": identity}
    except (OSError, ValueError, TypeError) as error:
        result = {"status": "unavailable", "reason": str(error)}
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_SELF)
    stop.set()
    sampler.join()
    io_after = process_io(process)
    result.update(
        {
            "case": args.case,
            "workers": args.workers,
            "elapsed_seconds": elapsed,
            "cpu_user_seconds": after.ru_utime - before.ru_utime,
            "cpu_system_seconds": after.ru_stime - before.ru_stime,
            "read_bytes": io_after["read_bytes"] - io_before["read_bytes"]
            if io_before and io_after
            else "not_measured",
            "write_bytes": io_after["write_bytes"] - io_before["write_bytes"]
            if io_before and io_after
            else "not_measured",
            **observed,
        }
    )
    validation_root = result.get("identities", {}).pop("_validation_root", None)
    measurement = result.get("identities", {}).pop("_measurement", None)
    if measurement is not None:
        result["logical_sha_measurement"] = measurement
    if validation_root is not None:
        verification_started = time.perf_counter()
        hashes: dict[str, str] = {}
        for path in sorted(Path(validation_root).rglob("*.npy")):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if path.name in hashes and hashes[path.name] != digest:
                raise ValueError("copied prepared aliases disagree")
            hashes[path.name] = digest
        if hashes != paths["array_sha256"]:
            raise ValueError("copied arrays differ from canonical source")
        result["identities"]["array_sha256"] = hashes
        result["independent_post_timer_validation_seconds"] = (
            time.perf_counter() - verification_started
        )
    save(args.result, result)


def benchmark(args: argparse.Namespace) -> None:
    workspace = args.workspace.absolute()
    if workspace.exists() or args.summary.exists():
        raise FileExistsError("workspace and summary must both be new")
    checkout = Path(__file__).resolve().parent.parent
    if workspace == checkout or checkout in workspace.parents:
        raise ValueError("workspace must be external to checkout")
    parent = workspace.parent
    parent.mkdir(parents=True, exist_ok=True)
    storage = os.statvfs(parent)
    required = (
        12 * 1024**3
    )  # two medium fixtures, one real copy, and conservative margin
    if storage.f_bavail * storage.f_frsize < required or storage.f_favail < 4096:
        raise ValueError("benchmark needs 12 GiB free and 4096 free inodes")
    workspace.mkdir(mode=0o700)
    repetitions = args.repetitions or (3 if args.fixture == "medium" else 1)
    paths = fixture(workspace / "fixture", args.fixture == "medium")
    paths_file = workspace / "paths.json"
    save(paths_file, paths)
    cases = []
    for repeat in range(repetitions):
        order = [1, 2, 4] if repeat % 2 == 0 else [4, 2, 1]
        for case in [
            "prepared_cold",
            "resolve",
            "open",
            "explain",
            "stage_verify",
            "materialize",
            "controller_read",
            "campaign_read",
            "sha",
        ]:
            for workers in order if case == "sha" else [1]:
                result = workspace / f"{case}-{repeat}-{workers}.json"
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--paths",
                        str(paths_file),
                        "--case",
                        case,
                        "--workers",
                        str(workers),
                        "--destination",
                        str(workspace / f"copy-{repeat}"),
                        "--result",
                        str(result),
                    ],
                    check=True,
                )
                cases.append(
                    {
                        **json.loads(result.read_text()),
                        "repetition": repeat,
                        "cache_condition": "uncontrolled_OS_cache; fresh_process_cold_verifier; no_cache_drop",
                    }
                )
        for case in [
            "prepared_cold_planned",
            "prepared_warm",
            "resolve_warm",
            "open_warm",
            "explain_warm",
            "stage_verify_warm",
            "materialize_warm",
            "worker_install_warm",
            "worker_materialize_cold",
            "worker_materialize_warm",
            "stage_copy_native",
            "stage_copy_buffered",
            "dispatch_create_native",
            "worker_install_transfer",
            "dispatch_create_buffered",
        ]:
            for workers in order if case == "prepared_cold_planned" else [1]:
                result = workspace / f"{case}-{repeat}-{workers}.json"
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--paths",
                        str(paths_file),
                        "--case",
                        case,
                        "--workers",
                        str(workers),
                        "--destination",
                        str(workspace / f"{case}-copy-{repeat}-{workers}"),
                        "--result",
                        str(result),
                    ],
                    check=True,
                )
                cases.append(
                    {
                        **json.loads(result.read_text()),
                        "repetition": repeat,
                        "cache_condition": "uncontrolled_OS_cache; fresh_process_warm_or_cold_as_named; no_cache_drop",
                        "measurement_boundary": "extended_operation_including_imports",
                    }
                )
    real = {
        "status": "unavailable",
        "reason": "MODEL-0 prepared root absent/unreadable",
    }
    if args.model0_prepared and args.model0_prepared.is_dir():
        real_paths = workspace / "real-paths.json"
        save(real_paths, {"prepared": str(args.model0_prepared.absolute())})
        result = workspace / "real-cold.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--paths",
                str(real_paths),
                "--case",
                "prepared_cold",
                "--workers",
                "1",
                "--result",
                str(result),
            ],
            check=True,
        )
        real = json.loads(result.read_text())
        real_rows = []
        real_owned = workspace / "real-owned"
        source_paths = workspace / "real-transfer-paths.json"
        save(
            source_paths,
            {
                "prepared": str(args.model0_prepared.absolute()),
                "proof_root": str(workspace),
            },
        )
        copied_result = workspace / "real-owned-copy.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--paths",
                str(source_paths),
                "--case",
                "real_owned_copy",
                "--destination",
                str(real_owned),
                "--result",
                str(copied_result),
            ],
            check=True,
        )
        copied = json.loads(copied_result.read_text())
        real_rows.append(copied)
        if copied["status"] != "ok":
            raise RuntimeError(f"real owned copy failed: {copied}")
        warm_paths = workspace / "real-warm-paths.json"
        save(
            warm_paths,
            {
                "prepared": str(real_owned),
                "proof_root": str(workspace),
                "array_sha256": copied["identities"]["array_sha256"],
            },
        )
        for repetition in range(repetitions):
            result = workspace / f"real-warm-{repetition}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--paths",
                    str(warm_paths),
                    "--case",
                    "prepared_warm",
                    "--result",
                    str(result),
                ],
                check=True,
            )
            row = json.loads(result.read_text())
            if (
                row["status"] != "ok"
                or row["logical_sha_measurement"]["payload_sha_calls"]
            ):
                raise RuntimeError(f"real warm verification failed: {row}")
            real_rows.append(row)
        real["task_owned_transfer_and_warm"] = real_rows
    save(
        args.summary,
        {
            "format": "capacity-aware-execution-v1-benchmark",
            "metadata": metadata(),
            "workspace": str(workspace),
            "fixture": args.fixture,
            "paths": paths,
            "cases": cases,
            "identity_checks": check_identities(cases, paths),
            "model0_read_only": real,
            "historical_lock": {
                "status": "not_measured",
                "reason": "older source authority not rewritten",
            },
            "campaign_reconciliation": {
                "status": "read_only_fixture",
                "reason": "no real Campaign mutation authorized",
            },
        },
    )
    failures = [c for c in cases if c["status"] != "ok"]
    if failures:
        raise RuntimeError(f"benchmark cases unavailable: {failures}")
    print(
        json.dumps(
            {
                "summary": str(args.summary),
                "cases": len(cases),
                "model0": real["status"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--fixture", choices=["small", "medium", "real"], default="small"
    )
    parser.add_argument("--model0-prepared", type=Path)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--paths", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--destination", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    elif not args.workspace or not args.summary:
        parser.error("--workspace and --summary are required")
    elif args.repetitions is not None and args.repetitions < 1:
        parser.error("--repetitions must be positive")
    elif args.fixture == "real" and not args.model0_prepared:
        parser.error("--fixture real requires --model0-prepared")
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
