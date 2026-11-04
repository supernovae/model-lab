"""Explicit, isolated disposable replay. Never called by preparation or tests."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import UTC
from pathlib import Path

ARMS = ("baseline", "padded_mb1", "mb4", "mb8")

OWNED_OUTPUT = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--envelope", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--host-idle-evidence", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        parser.error(
            "Execution requires --execute after GPU reservation and idle review"
        )
    from datetime import datetime

    evidence = json.loads(args.host_idle_evidence.read_text())
    age = (
        datetime.now(UTC) - datetime.fromisoformat(evidence["observed_at_utc"])
    ).total_seconds()
    if not 0 <= age <= 1800 or evidence.get("all_reported_ccs_percent") != 0.0:
        raise ValueError("Fresh host-level compute-idle evidence required")
    root = Path(
        "/mnt/sparselab/projects/devmind/experiments/reasoning-spine-continuation-20261104-v1"
    )
    terminal = json.loads(
        (root / "continuation-main-v1/terminal-verification.json").read_text()
    )
    if terminal["status"] != "completed" or not terminal["verification"]["valid"]:
        raise ValueError("Training completion and final cold verification required")
    if any(Path("/proc", str(pid)).exists() for pid in (59674, 59686)):
        raise ValueError("Prior trainer or supervisor still present")
    lock = open("/tmp/sparselab-performance-replay.lock", "a")  # noqa: SIM115 -- held through lease and replay
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from sparselab.workers.leases import acquire_lease

    lease = acquire_lease(
        worker_id="performance-recovery-v1",
        backend="xpu",
        physical_device_id=evidence["physical_device_id"],
    )
    if lease is None:
        raise RuntimeError("Physical device is leased; benchmark refused")
    import atexit

    atexit.register(lease.close)
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    global OWNED_OUTPUT
    OWNED_OUTPUT = args.output
    for variable, folder in [
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("SYCL_CACHE_DIR", "sycl"),
        ("XDG_CACHE_HOME", "cache"),
    ]:
        os.environ[variable] = str(args.output.resolve() / folder)
    os.environ["TORCH_LOGS"] = "recompiles,graph_breaks"
    # Dependencies that might initialize an accelerator are imported ONLY after the gate.
    from dataclasses import asdict

    import numpy as np
    import torch

    from sparselab.config.models import RunConfig
    from sparselab.data.document_batching import collate_document_update
    from sparselab.data.story_packing import MatchedStoryDataset
    from sparselab.engines.base import EngineState, Microbatch
    from sparselab.engines.pytorch import PyTorchEngine
    from sparselab.resource_envelope import (
        check_envelope,
        current_process_rss_bytes,
        load_resource_envelope,
    )
    from sparselab.runtime import seed_everything
    from sparselab.runtime_profile import authorize_profile, load_runtime_profile
    from sparselab.training.checkpoints import CheckpointManager
    from sparselab.training.continuation import runtime_optimization_transition
    from sparselab.training.manifest import (
        canonical_digest,
        config_sha256,
        sha256_file,
        source_identity,
    )

    plan = json.loads(args.plan.read_text())
    digest = plan.pop("plan_sha256")
    if canonical_digest(plan) != digest:
        raise ValueError("Plan changed")
    source = Path(__file__).resolve().parents[2]
    if (
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        != plan["source_commit"]
    ):
        raise ValueError("Source revision changed")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("Source is dirty")
    import sparselab

    if not Path(sparselab.__file__).resolve().is_relative_to(source):
        raise ValueError("Wrong imported source")
    checkpoint = Path(plan["checkpoint"])
    if checkpoint.name in {"latest.json", "best.json"} or not checkpoint.is_dir():
        raise ValueError("Explicit completed generation required")
    for path, expected in [
        (plan["config"], plan["config_file_sha256"]),
        (plan["data"], plan["data_sha256"]),
        (checkpoint / "manifest.json", plan["checkpoint_manifest_file_sha256"]),
    ]:
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"Pinned input changed: {path}")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if manifest["sha256"] != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint identity changed")
    envelope = load_resource_envelope(args.envelope)
    check_envelope(
        envelope, workspace=args.output, rss_bytes=current_process_rss_bytes()
    )
    manager = CheckpointManager(checkpoint.parent.parent)
    verified = manager.verify(checkpoint, verification_mode="cold")
    if not verified.valid:
        raise ValueError(str(verified))
    config = RunConfig.model_validate(plan["arms"][args.arm])
    transition = runtime_optimization_transition(
        RunConfig.model_validate(plan["parent_config"]),
        config,
        plan["checkpoint_sha256"],
    )
    if transition != plan["transitions"][args.arm]:
        raise ValueError("Continuation transition changed")
    receipt = json.loads(args.preflight_receipt.read_text())
    if (
        receipt.get("decision")
        not in {"qualified_match", "explicit_unqualified_deviation"}
        or receipt.get("plan_sha256") != digest
        or receipt.get("arm") != args.arm
        or receipt.get("source_commit") != plan["source_commit"]
        or receipt.get("config_sha256")
        != canonical_digest(config.model_dump(mode="json"))
        or receipt.get("runtime_profile_sha256") != sha256_file(args.runtime_profile)
    ):
        raise ValueError("Missing/mismatched launch-preflight admission")
    config = config.model_copy(
        update={
            "logging": config.logging.model_copy(
                update={"root_dir": args.output / "unused-runs"}
            )
        }
    )
    ids = np.load(plan["data"], mmap_mode="r")
    matched = config.training.matched_stories
    dataset = MatchedStoryDataset(
        ids,
        config.training.seq_len,
        plan["eos"],
        layout=matched.layout,
        targets_per_update=matched.targets_per_update,
        unpadded_documents=matched.unpadded_documents,
        frozen_update_ends=tuple(matched.frozen_update_ends),
    )
    snapshot = manager.load(checkpoint, mode="resume")
    if (
        config_sha256(snapshot.config) != config_sha256(plan["parent_config"])
        or snapshot.source_identity_sha256 != plan["parent_source_identity_sha256"]
        or source_identity()["sha256"] != plan["candidate_source_identity_sha256"]
    ):
        raise ValueError("Saved config or parent/candidate source provenance changed")
    if list(snapshot.cursor) != plan["cursor"] or snapshot.step != plan["step"]:
        raise ValueError("Checkpoint cursor/step mismatch")
    if snapshot.schedule["completed_updates"] != plan["step"]:
        raise ValueError("Checkpoint schedule mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)
    seed_everything(config.seed, deterministic_cpu=False)
    authorization = authorize_profile(
        load_runtime_profile(args.runtime_profile), config
    )
    engine = PyTorchEngine()
    runtime = engine.validate(config, authorization=authorization)
    if (
        runtime.device_name != "Intel(R) Arc(TM) Pro B70 Graphics"
        or torch.__version__ != "2.15.0+xpu"
    ):
        raise ValueError("Unqualified hardware/framework; replan explicitly")
    engine.initialize(config, initial_weights=snapshot.model)
    # Restore into the SAME module for the warm replay, preserving compiled graphs.
    initial_weights = {
        name: value.detach().cpu().clone()
        for name, value in engine.model.state_dict().items()
    }
    del snapshot
    result = {
        "arm": args.arm,
        "plan_sha256": digest,
        "status": "running",
        "transition": transition,
        "source_drift": plan["source_drift"],
        "parent_plan_sha256": plan["parent_plan_sha256"],
        "host_idle_evidence": evidence,
        "device_lease": lease.metadata.copy(),
        "launch_admission": receipt,
        "qualification": "unqualified_throughput_experiment",
        "checkpoint_verification": asdict(verified),
        "runtime": runtime.as_dict(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "config": config.model_dump(mode="json"),
        "passes": [],
        "runtime_profile_sha256": sha256_file(args.runtime_profile),
        "envelope_sha256": sha256_file(args.envelope),
        "loaded_gpu_libraries": sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if any(
                    name in line
                    for name in ("libze_intel_gpu.so", "libigc.so", "libze_loader.so")
                )
            }
        ),
        "numerical_evidence": "Per-update loss, grad norm, LR; deterministic spaced parameter samples. Not full tensor equivalence.",
        "compile_evidence": "stderr has TORCH_LOGS recompiles/graph_breaks; Dynamo counters below are diagnostic, version-dependent.",
    }
    if not all(
        "intel-26.40/prefix/" in library for library in result["loaded_gpu_libraries"]
    ) or not all(
        any(
            Path(library).name == expected for library in result["loaded_gpu_libraries"]
        )
        for expected in ("libze_intel_gpu.so.1.18.40196", "libze_loader.so.1.34.0")
    ):
        engine.close()
        raise ValueError(
            "Unexpected loaded GPU stack: " + repr(result["loaded_gpu_libraries"])
        )
    try:
        for phase in ("cold", "warm"):
            engine.optimizer.zero_grad(set_to_none=True)
            engine.model.load_state_dict(initial_weights)
            for module in engine.model.modules():
                if hasattr(module, "last_diagnostics"):
                    module.last_diagnostics = None
            state = manager.load(checkpoint, mode="resume")
            engine.restore_training_state(
                EngineState(
                    state.optimizer,
                    state.rng,
                    state.scaler,
                    state.optimizer_parameter_names,
                )
            )
            del state
            check_envelope(
                envelope, workspace=args.output, rss_bytes=current_process_rss_bytes()
            )
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
            torch.xpu.reset_peak_memory_stats()
            from torch._dynamo.utils import counters

            before = {str(k): dict(v) for k, v in counters.items()}
            rows = []
            result["in_progress"] = {"phase": phase, "updates": rows}
            wall = time.perf_counter()
            for offset, group in enumerate(plan["groups"]):
                lease.heartbeat()
                check_envelope(
                    envelope,
                    workspace=args.output,
                    rss_bytes=current_process_rss_bytes(),
                )
                records = list(dataset.update_rows(group["index"], len(ids)))
                batches = [Microbatch(r[0][None, :], r[1][None, :]) for r in records]
                if matched.document_batching == "padded":
                    batches = collate_document_update(
                        records,
                        width=config.training.micro_batch_size,
                        eos=plan["eos"],
                        max_length=config.training.seq_len,
                        padding_multiple=matched.padding_multiple,
                    )
                targets = sum(int((b.targets != -100).sum()) for b in batches)
                compute_slots = sum(b.inputs.size for b in batches)
                real_positions = sum(len(r[0]) for r in records)
                attention_pairs = sum(
                    b.inputs.shape[0] * b.inputs.shape[1] ** 2 for b in batches
                )
                expected_cost = plan["shape_audit"][args.arm]["updates"][offset]
                if [compute_slots, real_positions, attention_pairs] != [
                    expected_cost[k]
                    for k in ("compute_slots", "real_positions", "attention_pairs")
                ]:
                    raise ValueError("Prepared batching shapes changed")
                if (
                    targets != group["targets"]
                    or [len(r[0]) for r in records] != group["lengths"]
                ):
                    raise ValueError("Update membership changed")
                update = engine.train_update(
                    batches, plan["step"] + offset + 1, targets
                )
                capacity = torch.xpu.get_device_properties().total_memory
                if (
                    torch.xpu.max_memory_reserved()
                    > capacity * config.runtime.memory.max_device_memory_fraction
                ):
                    raise MemoryError(
                        "Reserved device memory exceeds configured ceiling; no retry"
                    )
                if offset == 0:
                    mapped = sorted(
                        {
                            line.split()[-1]
                            for line in Path("/proc/self/maps").read_text().splitlines()
                            if any(
                                name in line
                                for name in (
                                    "libze_intel_gpu.so",
                                    "libigc.so",
                                    "libze_loader.so",
                                )
                            )
                        }
                    )
                    result["loaded_gpu_libraries"] = mapped
                    if not all(
                        "intel-26.40/prefix/" in path for path in mapped
                    ) or not all(
                        any(Path(path).name == expected for path in mapped)
                        for expected in (
                            "libze_intel_gpu.so.1.18.40196",
                            "libigc.so.2.42.6+1792520529",
                            "libze_loader.so.1.34.0",
                        )
                    ):
                        raise ValueError(
                            "Unexpected post-kernel GPU stack: " + repr(mapped)
                        )
                if update.outcome != "APPLIED" or not np.isfinite(update.loss_sum):
                    raise RuntimeError("Nonfinite or unapplied update")
                rows.append(
                    {
                        "group": group["index"],
                        "step": plan["step"] + offset + 1,
                        "targets": targets,
                        "compute_slots": compute_slots,
                        "real_positions": real_positions,
                        "attention_pairs": attention_pairs,
                        "shapes": [list(b.inputs.shape) for b in batches],
                        "loss": update.loss_sum / targets,
                        "seconds": update.elapsed_seconds,
                        "grad_norm": update.metrics["optimizer/grad_norm"],
                        "learning_rate": update.metrics["optimizer/learning_rate"],
                    }
                )
            torch.xpu.synchronize()
            wall = time.perf_counter() - wall
            peak_allocated = torch.xpu.max_memory_allocated()
            peak_reserved = torch.xpu.max_memory_reserved()
            result["in_progress"].update(
                update_seconds=sum(r["seconds"] for r in rows),
                wall_seconds=wall,
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
            )
            (args.output / "result.json").write_text(json.dumps(result, indent=2))
            samples = {}
            for name, parameter in engine.model.named_parameters():
                flat = parameter.detach().flatten()
                count = min(32, flat.numel())
                indices = torch.tensor(
                    [i * (flat.numel() - 1) // max(count - 1, 1) for i in range(count)],
                    dtype=torch.long,
                    device=flat.device,
                )
                samples[name] = flat[indices].float().cpu().tolist()
            seconds = sum(r["seconds"] for r in rows)
            targets = sum(r["targets"] for r in rows)
            result["passes"].append(
                {
                    "phase": phase,
                    "updates": rows,
                    "update_seconds": seconds,
                    "wall_seconds": wall,
                    "targets_per_update_second": targets / seconds,
                    "targets_per_wall_second": targets / wall,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "device_capacity_bytes": capacity,
                    "reserved_headroom_bytes": capacity - peak_reserved,
                    "ceiling_bytes": int(
                        capacity * config.runtime.memory.max_device_memory_fraction
                    ),
                    "compute_slots": sum(r["compute_slots"] for r in rows),
                    "attention_pairs": sum(r["attention_pairs"] for r in rows),
                    "memory_headroom_qualification": "Measured one replay; sustained qualification pending",
                    "compiler_counters_before": before,
                    "compiler_counters_after": {
                        str(k): dict(v) for k, v in counters.items()
                    },
                    "parameter_samples": samples,
                }
            )
            result.pop("in_progress", None)
            (args.output / "result.json").write_text(json.dumps(result, indent=2))
        result["status"] = "completed"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        raise
    finally:
        result["total_wall_seconds_including_verification_and_setup"] = (
            time.perf_counter() - started
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        engine.close()
        lease.close()
        atexit.unregister(lease.close)
        fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if OWNED_OUTPUT is not None and not (OWNED_OUTPUT / "result.json").exists():
            (OWNED_OUTPUT / "result.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "stage": "setup_or_preflight",
                        "error": repr(error),
                        "qualification": "unqualified",
                    }
                )
            )
        raise
