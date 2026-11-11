"""Command-line entry point for Model Lab."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import stat
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from klab.batch_calibration import calibrate_batch
from klab.config.loading import load_config, load_tokenizer_config
from klab.config.migrate import migrate_file
from klab.dashboard.stores import configured_stores, parse_run_store
from klab.data.bakeoff import bakeoff
from klab.data.encoding import (
    TOKENIZER_BATCH_DOCUMENTS,
    TOKENIZER_BATCH_SOURCE_BYTES,
    validate_tokenizer_batch_limits,
)
from klab.data.packing import prepare_data
from klab.data.tokenizer import (
    load_tokenizer,
    train_tokenizer,
    verify_run_tokenizer,
)
from klab.data.withheld_facts import (
    audit_manifest,
    verify_manifest,
    write_manifest,
)
from klab.engram.packs import (
    _rename_noreplace,
    compile_pack,
    inspect_pack,
    verify_pack,
)
from klab.evaluation.byte_memory_transfer import transfer_byte_memory
from klab.evaluation.capabilities import (
    capability_card,
    compare_results,
    describe_capability_card,
    evaluate_capability,
    list_capability_cards,
    phase_e_behavior_suite,
    task_card_payload,
    write_capability_result,
)
from klab.evaluation.chat import ChatMessage, assistant_reply, prepare_chat_prompt
from klab.evaluation.evidence import experiment_evidence
from klab.evaluation.generation import generate
from klab.evaluation.inference import load_run, write_inference_result
from klab.evaluation.post_train_triage import read_triage, triage_summary
from klab.evaluation.surface_overlay import surface_review_status
from klab.evaluation.withheld_facts import (
    evaluate_withheld_facts,
    write_withheld_evaluation,
)
from klab.hf_auth import (
    HuggingFaceAccessError,
    HuggingFaceCredentialError,
    set_token_file,
)
from klab.memory import (
    calibrated_estimate,
    calibration_key,
    plan_memory,
    write_resource_proposal,
)
from klab.model.inspection import inspection_report, parameter_inventory
from klab.model.memory import ByteAddressMemory
from klab.model.portable_engram import export_portable_engram, load_portable_engram
from klab.readiness import SMOKE_FAMILIES, smoke_readiness
from klab.resource_envelope import (
    check_envelope,
    current_process_rss_bytes,
    load_resource_envelope,
    require_envelope_applicability,
)
from klab.runtime_forecasting import runtime_forecast_planning
from klab.runtime_profile import (
    authorize_profile,
    load_runtime_profile,
    probe_runtime_profile,
    require_authorization,
)
from klab.staging import inspect_runtime, stage
from klab.training.checkpoints import CheckpointManager, _safe_member
from klab.training.manifest import source_identity
from klab.training.metrics import ExperimentStore
from klab.training.trainer import train
from klab.verification_proofs import verification_options
from klab.workdir import (
    ensure_scratch_dir,
    ensure_work_dir,
    record_storage_observation,
    resolve_work_dir,
    storage_checks,
    warn_storage_checks,
)
from klab.workspace_cleanup import apply_cleanup, plan_cleanup, write_plan
from klab.workspace_preflight import (
    tokenizer_storage_checks,
    training_storage_checks,
)

if TYPE_CHECKING:
    from klab.research.lifecycle import LifecycleRegistry


from klab.canonical import reject_duplicate_keys


def _dashboard(args: argparse.Namespace) -> None:
    configured_stores(Path(args.runs_dir), args.run_store)
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        args.address,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        # One accessible accent (blue) instead of Streamlit's default red,
        # which reads as "fail" next to the verdict colors.
        "--theme.primaryColor",
        "#0969da",
        "--",
        "--runs-dir",
        args.runs_dir,
        "--reports-dir",
        args.reports_dir,
    ]
    for label, directory in args.run_store:
        command.extend(["--run-store", f"{label}={directory}"])
    if args.lifecycle is not None:
        command.extend(["--lifecycle", args.lifecycle])
    command.extend(["--evidence-root", args.evidence_root])
    if args.surface_dir is not None:
        command.extend(["--surface-dir", args.surface_dir])
    from klab.lab_mode import lab_root

    lab_dir = args.lab_dir or lab_root(
        resolve_work_dir(getattr(args, "work_dir", None))
    )
    command.extend(["--lab-dir", str(lab_dir)])
    # Replace the launcher so container stop signals and exit status reach the
    # actual server, including when it is PID 1.
    # Nothing may depend on Python cleanup or buffered output after this exec.
    os.execv(sys.executable, command)


def _workspace_cleanup_plan(args: argparse.Namespace) -> None:
    plan = plan_cleanup(
        Path(args.workspace),
        max_extra_periodic=args.max_extra_periodic,
        max_cache_entries=args.max_cache_entries,
    )
    write_plan(plan, Path(args.output))
    print(json.dumps(plan, sort_keys=True))


def _workspace_cleanup_apply(args: argparse.Namespace) -> None:
    print(json.dumps(apply_cleanup(Path(args.plan)), sort_keys=True))


def _facts_manifest(args: argparse.Namespace) -> None:
    output = Path(args.output)
    write_manifest(output, args.seed)
    print(output)


def _facts_verify(args: argparse.Namespace) -> None:
    manifest = verify_manifest(Path(args.path))
    print(json.dumps({"sha256": manifest["sha256"], "valid": True}, sort_keys=True))


def _facts_audit(args: argparse.Namespace) -> None:
    print(json.dumps(audit_manifest(Path(args.path)), indent=2, sort_keys=True))


def _facts_evaluate(args: argparse.Namespace) -> None:
    loaded = load_run(args.run_id, Path(args.runs_dir), args.checkpoint, args.backend)
    if loaded.engine is not None:
        raise ValueError(
            "withheld-facts evaluation is unsupported for native MLX runs because "
            "candidate likelihood scoring currently requires the PyTorch evaluator"
        )
    result = evaluate_withheld_facts(
        loaded.model,
        loaded.tokenizer,
        Path(args.manifest),
        max_seq_len=loaded.config.model.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        device=loaded.device,
    )
    result["identity"] = loaded.identity
    output = (
        Path(args.output)
        if args.output
        else Path(args.runs_dir)
        / args.run_id
        / "evaluations"
        / f"withheld-v2-{result['manifest_sha256']}.json"
    )
    write_withheld_evaluation(output, result)
    print(output)


def _facts_transfer_evaluate(args: argparse.Namespace) -> None:
    source = load_run(
        args.source_run_id, Path(args.runs_dir), args.source_checkpoint, args.backend
    )
    target = load_run(
        args.target_run_id, Path(args.runs_dir), args.target_checkpoint, args.backend
    )
    if source.engine is not None or target.engine is not None:
        raise ValueError(
            "withheld-facts transfer evaluation is unsupported for native MLX runs; "
            "it transfers PyTorch byte-memory tables"
        )
    transfer_byte_memory(source.model.state_dict(), target.model)
    result = evaluate_withheld_facts(
        target.model,
        target.tokenizer,
        Path(args.manifest),
        max_seq_len=target.config.model.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        device=target.device,
    )
    result["source_run_id"] = args.source_run_id
    result["target_run_id"] = args.target_run_id
    result["source_identity"] = source.identity
    result["target_identity"] = target.identity
    output = (
        Path(args.output)
        if args.output
        else Path(args.runs_dir)
        / args.target_run_id
        / "evaluations"
        / f"transferred-v2-{args.source_run_id}-{result['manifest_sha256']}.json"
    )
    write_withheld_evaluation(output, result)
    print(output)


def _engram_export(args: argparse.Namespace) -> None:
    from klab.training.manifest import read_manifest, trained_memory_addressing

    # load_run refuses a run trained with pre-fix degenerate addressing; the
    # scheme exported is the one the hash-verified run manifest recorded at
    # training time, never one inferred from the table's row count.
    loaded = load_run(args.run_id, Path(args.runs_dir), args.checkpoint, args.backend)
    verified = read_manifest(Path(args.runs_dir) / args.run_id / "manifest.json")
    trained = trained_memory_addressing(
        verified["effective_config"], verified["architecture_sha256"]
    )
    if loaded.engine is not None:
        raise ValueError(
            "Engram export is unsupported for native MLX runs; MLX supports dense "
            "and block-sparse attention only"
        )
    if not isinstance(loaded.model.memory, ByteAddressMemory):
        raise TypeError("Engram export requires a byte-memory run")
    hashing = trained.get("byte")
    if not isinstance(hashing, str):
        raise TypeError("run records no raw-byte memory addressing")
    manifest = export_portable_engram(
        loaded.model.memory.table.weight,
        Path(args.output),
        ngram_size=loaded.config.model.memory_ngram_size,
        hashing=hashing,
    )
    print(json.dumps(manifest.as_dict(), sort_keys=True))


def _engram_inspect(args: argparse.Namespace) -> None:
    package = load_portable_engram(Path(args.path))
    print(json.dumps(package.manifest.as_dict(), sort_keys=True))


def _engram_pack_compile(args: argparse.Namespace) -> None:
    from pickle import UnpicklingError

    from pyarrow import ArrowException
    from safetensors import SafetensorError

    try:
        manifest = compile_pack(
            Path(args.source),
            Path(args.output),
            name=args.name,
            namespace=args.namespace,
            default_license=args.license,
            source_name=args.source_name,
            source_revision=args.source_revision,
            created_at=args.created_at,
            lexical_package=Path(args.lexical_package)
            if args.lexical_package
            else None,
            semantic_keys=Path(args.semantic_keys) if args.semantic_keys else None,
            semantic_values=Path(args.semantic_values)
            if args.semantic_values
            else None,
            semantic_metadata=Path(args.semantic_metadata)
            if args.semantic_metadata
            else None,
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        EOFError,
        UnpicklingError,
        ArrowException,
        SafetensorError,
    ) as error:
        payload = {
            "valid": False,
            "errors": [{"field": "compile", "reason": str(error)}],
        }
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(1) from None
    print(
        json.dumps(
            {
                "pack_id": manifest.pack_id,
                "output": str(args.output),
                "record_count": manifest.record_count,
                "lexical_count": manifest.lexical.table_size
                if manifest.lexical is not None
                else 0,
                "semantic_count": manifest.semantic.entry_count
                if manifest.semantic is not None
                else 0,
            },
            sort_keys=True,
        )
    )


def _engram_pack_inspect(args: argparse.Namespace) -> None:
    try:
        payload = inspect_pack(Path(args.path))
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        print(
            json.dumps(
                {
                    "valid": False,
                    "errors": [{"field": "inspect", "reason": str(error)}],
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None
    print(json.dumps(payload, sort_keys=True))


def _engram_pack_verify(args: argparse.Namespace) -> None:
    try:
        report = verify_pack(Path(args.path), expected_pack_id=args.expected_pack_id)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        payload = {
            "valid": False,
            "pack_id": None,
            "errors": [{"field": "verify", "reason": str(error)}],
            "files": [],
        }
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(1) from None
    payload = {
        "valid": report.valid,
        "pack_id": report.pack_id,
        "errors": [
            {"field": item.field, "reason": item.reason} for item in report.errors
        ],
        "files": [
            {
                "relative_path": item.relative_path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in report.files
        ],
    }
    print(json.dumps(payload, sort_keys=True))
    if not report.valid:
        raise SystemExit(1)


def _config_migrate(args: argparse.Namespace) -> None:
    changes = migrate_file(Path(args.input), Path(args.output))
    print("\n".join(changes))


def _config_derive(args: argparse.Namespace) -> None:
    """Publish a schema-validated config variant without replacing either input."""
    try:
        from klab.derivation import derive_config, parse_assignments

        receipt = derive_config(
            Path(args.source),
            Path(args.output),
            parse_assignments(args.settings),
        )
    except (OSError, TypeError, ValueError) as error:
        message = str(error)
        if args.json:
            print(json.dumps({"status": "error", "error": message}, allow_nan=False))
        else:
            print(f"klab config derive: {message}", file=sys.stderr)
        raise SystemExit(2) from None

    payload = {
        "status": "ok",
        "receipt": str(
            Path(args.output)
            .absolute()
            .with_name(Path(args.output).name + ".derivation.json")
        ),
        **receipt,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True, allow_nan=False, default=str))
        return
    print(f"output: {receipt['output']['path']}")
    print(f"receipt: {payload['receipt']}")
    validation = receipt.get("validation", {})
    print(f"validation scope: {validation.get('scope', 'typed_config')}")
    for field, change in receipt.get("delta", {}).items():
        print(f"delta {field}: {change['base']!r} -> {change['variant']!r}")


def _legacy_checkpoint_inventory(path: Path, expected_config=None) -> dict[str, object]:
    from klab.training.weight_import import load_legacy_weights

    legacy = load_legacy_weights(path, expected_config)
    return {
        **legacy.metadata,
        "engine": "pytorch",
        "requested_backend": legacy.config.runtime.backend,
        "resume_level": "weights_only",
        "source_config_sha256": legacy.source_config_sha256,
        "tensor_inventory": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in legacy.tensors.items()
        },
        "full_resume_unavailable_reason": (
            "Legacy v1 lacks resumable RNG and immutable provenance; "
            "use weights import, then train --promote."
        ),
    }


def _checkpoint_inspect(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if path.suffix == ".pt":
        manifest = _legacy_checkpoint_inventory(path)
    else:
        directory = CheckpointManager(path.parent.parent)._resolve(path)
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("unsafe checkpoint directory")
        manifest_path = _safe_member(directory, "manifest.json")
        if manifest_path is None:
            raise ValueError("unsafe checkpoint manifest path")
        manifest = json.loads(manifest_path.read_text())
    print(
        json.dumps(manifest, sort_keys=True)
        if args.json
        else "\n".join(f"{key}: {value}" for key, value in manifest.items())
    )


def _checkpoint_verify(args: argparse.Namespace) -> None:
    path = Path(args.path)
    try:
        expected_config = load_config(Path(args.config)) if args.config else None
    except (ValueError, TypeError) as error:
        payload = {
            "valid": False,
            "errors": [{"field": "config", "reason": str(error)}],
            "files": [],
            "resume_level": None,
        }
        print(json.dumps(payload, sort_keys=True) if args.json else payload)
        raise SystemExit(1) from None
    if path.suffix == ".pt":
        try:
            metadata = _legacy_checkpoint_inventory(path, expected_config)
            payload = {
                "valid": True,
                "errors": [],
                "files": [{"name": path.name, "sha256": metadata["sha256"]}],
                "resume_level": "weights_only",
                "full_resume_unavailable_reason": metadata[
                    "full_resume_unavailable_reason"
                ],
            }
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            payload = {
                "valid": False,
                "errors": [{"field": "legacy_checkpoint", "reason": str(error)}],
                "files": [],
                "resume_level": "weights_only",
            }
    else:
        root = path.parent.parent
        report = CheckpointManager(root).verify(
            path,
            require_training_state=not args.weights_only,
            expected_config=expected_config,
        )
        payload = {
            "valid": report.valid,
            "errors": list(report.errors),
            "files": list(report.verified_files),
            "resume_level": report.resume_level,
        }
    print(json.dumps(payload, sort_keys=True) if args.json else payload)
    if not payload["valid"]:
        raise SystemExit(1)


def _inspect(args: argparse.Namespace) -> None:
    if args.estimate_runtime and args.write_proposal:
        raise ValueError("--estimate-runtime is read-only; write proposals separately")
    config = load_config(Path(args.config))
    inventory = parameter_inventory(config)
    runtime = inspect_runtime(config)
    observations = ExperimentStore.get_calibration(
        config.logging.root_dir,
        calibration_key(
            config, runtime, source_digest=str(source_identity()["sha256"])
        ),
    )
    estimate = calibrated_estimate(config, runtime, inventory, observations)
    values = {
        **inspection_report(config),
        "parameter_inventory": inventory.__dict__,
        "memory_estimate": estimate.__dict__,
        "runtime": runtime.as_dict(),
    }
    proposal = plan_memory(config, runtime, estimate)
    values["resource_proposal"] = asdict(proposal)
    if args.write_proposal:
        paths = write_resource_proposal(proposal, Path(args.write_proposal))
        values["proposal_paths"] = [str(path) for path in paths]
    if args.estimate_runtime:
        forecast = runtime_forecast_planning(
            config.logging.root_dir,
            config,
            runtime,
            total_targets=config.training.max_tokens,
        )
        values["runtime_forecast"] = {
            "schema_version": forecast["schema_version"],
            "runtime_signature": forecast["runtime_signature"],
            "planning": forecast["planning"],
            "warmup_calibrated": None,
            "live": None,
            "final_observed": None,
            "history_match": forecast["history_match"],
            "history_limit": forecast["history_limit"],
        }
    if args.json or args.estimate_runtime:
        print(json.dumps(values, indent=2, sort_keys=True))
        return
    print("Model")
    for field in ("total", "trainable", "active_per_token"):
        print(f"  {field}: {values[field]:,}")
    print("Estimated training memory")
    for field, value in asdict(estimate).items():
        if field.endswith("_bytes") and value is not None:
            print(f"  {field}: {value:,} bytes ({value / 1024**3:.3f} GiB)")
    print("Device")
    print(f"  {runtime.engine}/{runtime.backend}, index {runtime.device_index}")
    for limitation in runtime.limitations:
        print(f"  {limitation}")
    print(f"Result: {estimate.result}")
    for assumption in estimate.assumptions:
        print(f"  {assumption}")


def _runtime_status(args: argparse.Namespace) -> None:
    run_id = args.run_id
    records = ExperimentStore.get_runtime_progress_records(
        Path(args.runs_dir), run_id, limit=args.limit
    )
    latest_record = next(
        (item for item in reversed(records) if item["kind"] == "runtime_progress"),
        None,
    )
    final_record = next(
        (
            item
            for item in reversed(records)
            if item["kind"] == "runtime_final_observation"
        ),
        None,
    )
    latest_payload = latest_record.get("payload") if latest_record is not None else None
    latest = dict(latest_payload) if isinstance(latest_payload, dict) else None
    final_payload = final_record.get("payload") if final_record is not None else None
    final_observed = dict(final_payload) if isinstance(final_payload, dict) else None
    snapshot_age_seconds: float | None = None
    if latest is not None:
        try:
            observed = datetime.fromisoformat(str(latest["observed_at_utc"]))
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=UTC)
            snapshot_age_seconds = max(
                (datetime.now(UTC) - observed).total_seconds(), 0.0
            )
        except KeyError, TypeError, ValueError:
            snapshot_age_seconds = None
        if snapshot_age_seconds is not None:
            derived = latest.get("derived")
            derived_record = dict(derived) if isinstance(derived, dict) else {}
            live_value = derived_record.get("live")
            live = dict(live_value) if isinstance(live_value, dict) else {}
            last_progress_age = live.get("last_meaningful_progress_age_seconds")
            stall_after = live.get("stalled_after_seconds", 300.0)
            if (
                latest.get("state") not in {"COMPLETE", "FAILED", "INTERRUPTED"}
                and isinstance(stall_after, (int, float))
                and not isinstance(stall_after, bool)
                and (
                    snapshot_age_seconds
                    + (
                        float(last_progress_age)
                        if isinstance(last_progress_age, (int, float))
                        and not isinstance(last_progress_age, bool)
                        else 0.0
                    )
                    >= float(stall_after)
                )
            ):
                progress_age = (
                    float(last_progress_age) + snapshot_age_seconds
                    if isinstance(last_progress_age, (int, float))
                    and not isinstance(last_progress_age, bool)
                    else snapshot_age_seconds
                )
                live.update(
                    state="NO_PROGRESS",
                    last_meaningful_progress_age_seconds=progress_age,
                    eta_low_seconds=None,
                    eta_high_seconds=None,
                    eta_status="suspended",
                    eta_basis=None,
                )
                optimizer_eta = live.get("optimizer_only_eta")
                if isinstance(optimizer_eta, dict):
                    optimizer_eta = dict(optimizer_eta)
                    optimizer_eta.update(
                        low_seconds=None,
                        high_seconds=None,
                        status="suspended",
                        basis=None,
                    )
                    live["optimizer_only_eta"] = optimizer_eta
                derived_record["live"] = live
                latest["derived"] = derived_record
                latest["state"] = "NO_PROGRESS"
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "availability": "available"
        if latest_record is not None or final_record is not None
        else "unavailable",
        "reason": None
        if latest_record is not None or final_record is not None
        else "no runtime records are available for this run",
        "snapshot_age_seconds": snapshot_age_seconds,
        "latest": latest,
        "final_observed": final_observed,
        "records": records,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


def _runtime_attention(args: argparse.Namespace) -> None:
    from klab.runtime_attention import attention_probe
    from klab.runtime_environments import profile_for_id

    config = load_config(Path(args.config))
    if args.worker is not None:
        if args.runtime is not None or args.runtime_profile is not None:
            raise ValueError("--worker cannot be combined with a local runtime profile")
        from klab.workers.controller import Controller
        from klab.workers.transport import call_worker

        controller = Controller(Path(args.store), read_only=True)
        workers = [
            controller._model("WorkerDefinition", row["definition"])
            for row in controller.store.worker_records()
            if row["definition"]["name"] == args.worker
        ]
        if len(workers) != 1:
            raise ValueError("registered worker unavailable or ambiguous")
        result = call_worker(
            workers[0],
            "attention_probe",
            {"config": config.model_dump(mode="json")},
            timeout=1800,
        ).result
    else:
        profile = (
            profile_for_id(args.runtime)
            if args.runtime is not None
            else load_runtime_profile(Path(args.runtime_profile))
            if args.runtime_profile is not None
            else None
        )
        authorization = None
        if profile is not None:
            _ensure_runtime_interpreter(
                profile.python,
                hashlib.sha256(
                    json.dumps(profile.model_dump(mode="json"), sort_keys=True).encode()
                ).hexdigest(),
                lambda: probe_runtime_profile(profile),
            )
            authorization = authorize_profile(profile, config)
        result = attention_probe(config, authorization=authorization)
    print(json.dumps(result, indent=2, sort_keys=True))


def _runtime_probe(args: argparse.Namespace) -> None:
    profile = load_runtime_profile(Path(args.profile))
    result = probe_runtime_profile(profile)
    print(json.dumps(result, indent=2 if args.json else None, sort_keys=True))


def _data_bakeoff(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    print(bakeoff(config.dataset, Path(args.output)))


def _data_prepare(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    if config.dataset.source != "local_token_mixture":
        verify_run_tokenizer(config)
    tokenizer = load_tokenizer(config.tokenizer.path)
    print(
        prepare_data(
            config,
            tokenizer,
            resource_envelope=args.resource_envelope_value,
            tokenizer_batch_documents=args.tokenizer_batch_documents,
            tokenizer_batch_source_bytes=args.tokenizer_batch_source_bytes,
            observer=getattr(args, "observer", None),
        ).root
    )


def _prepared_input_summary(
    root: Path, config: Any, verified: dict[str, Any]
) -> dict[str, Any]:
    """Report supervised targets from the cold-authenticated prepared assets."""
    import numpy as np

    manifest = json.loads((root / "assets" / "data" / "manifest.json").read_text())
    train = manifest["train"]
    result = {
        "bundle": str(root.resolve()),
        "bundle_manifest_sha256": verified["sha256"],
        "source_identity_sha256": verified["source_identity_sha256"],
        "prepared_data_manifest_sha256": manifest["manifest_sha256"],
        "supervised_target_positions": train.get("supervised_target_positions"),
        "scheduled_target_positions": train.get("scheduled_target_positions"),
        "masked_padding_positions": train.get("masked_padding_positions"),
        "supervised_target_positions_by_stratum": train.get(
            "supervised_target_positions_by_stratum"
        ),
    }
    if config.dataset.source == "local_token_mixture":
        if manifest.get("supervision", {}).get("kind") != "token-loss-mask-v1":
            raise ValueError("token mixture prepared inputs lack a supervision mask")
        mask = np.load(
            root / "assets" / "data" / "train_supervision.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        actual = sum(
            int(np.count_nonzero(mask[start : start + 1_048_576]))
            for start in range(1, len(mask), 1_048_576)
        )
        mixture = manifest["token_mixture"]
        expected = sum(mixture["actual_target_tokens"].values())
        if (
            actual != expected
            or actual != train.get("supervised_target_positions")
            or actual != train.get("scheduled_target_positions")
            or actual != config.training.max_tokens
            or train.get("supervised_target_positions_by_stratum")
            != mixture["actual_target_tokens"]
        ):
            raise ValueError("prepared supervision differs from verified token mixture")
    return result


def _data_prepared_inputs(args: argparse.Namespace) -> None:
    from klab.data.legacy import require_current_dataset
    from klab.staging import materialize_prepared_inputs, verify_prepared_inputs

    config = load_config(Path(args.config))
    require_current_dataset(config.dataset)
    root = Path(args.output if args.inputs_command == "publish" else args.bundle)
    if args.inputs_command == "publish":
        materialize_prepared_inputs(
            config,
            root,
            resource_envelope=args.resource_envelope_value,
            tokenizer_batch_documents=args.tokenizer_batch_documents,
            tokenizer_batch_source_bytes=args.tokenizer_batch_source_bytes,
            verification_mode="cold",
        )
    verified = verify_prepared_inputs(root, config, verification_mode="cold")
    print(json.dumps(_prepared_input_summary(root, config, verified), sort_keys=True))


def _workspace_preflight(args: argparse.Namespace) -> None:
    checks = (
        tokenizer_storage_checks(load_tokenizer_config(Path(args.config)))
        if args.tokenizer
        else training_storage_checks(load_config(Path(args.config)))
    )
    status = (
        "adequate"
        if all(item.status == "adequate" for item in checks)
        else "insufficient"
    )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": status,
                "filesystems": [asdict(item) for item in checks],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if status == "insufficient":
        raise SystemExit(1)


def _tokenizer_train(args: argparse.Namespace) -> None:
    print(train_tokenizer(load_tokenizer_config(Path(args.config))))


def _stage(args: argparse.Namespace) -> None:
    from klab.training.pilot_deadline import load_pilot_deadline_policy

    options = verification_options(
        resolve_work_dir(args.work_dir), cold=args.cold_verify
    )
    config = load_config(Path(args.config))
    if args.through != "inspect":
        from klab.data.legacy import require_current_dataset

        require_current_dataset(config.dataset)
    if args.prepared_inputs is not None:
        from klab.experiments.artifacts import _safe_path
        from klab.staging import verify_prepared_inputs

        output = Path(args.output)
        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"existing-input stage output already exists: {output}"
            )
        prepared = _safe_path(
            str(args.prepared_inputs.absolute()), Path(args.config).absolute()
        )
        verify_prepared_inputs(prepared, config, **options)

    policy = (
        load_pilot_deadline_policy(args.pilot_deadline_policy)
        if args.pilot_deadline_policy is not None
        else None
    )
    output = stage(
        config,
        Path(args.output),
        args.through,
        prepared_inputs=args.prepared_inputs,
        authorization=args.runtime_authorization,
        resource_envelope=args.resource_envelope_value,
        tokenizer_batch_documents=args.tokenizer_batch_documents,
        tokenizer_batch_source_bytes=args.tokenizer_batch_source_bytes,
        pilot_deadline_policy=policy,
        observer=getattr(args, "observer", None),
        **options,
    )
    if getattr(args, "observer", None) is not None:
        try:
            from klab.phase_observations import set_observation_runtime

            report = json.loads((output / "stage.json").read_text())
            set_observation_runtime(
                args,
                requested=config.runtime.model_dump(mode="json"),
                resolved=report.get("runtime"),
                reason=None
                if report.get("runtime") is not None
                else "stage_runtime_unavailable",
                observation_scope="host_process_tree",
            )
        except Exception as error:  # noqa: BLE001 -- optional metadata must not change workflow success
            print(
                f"klab: optional observation runtime unavailable: {error}",
                file=sys.stderr,
            )
    print(output)


def _batch_calibrate(args: argparse.Namespace) -> None:
    print(
        calibrate_batch(
            load_config(Path(args.config)),
            Path(args.output),
            max_candidates=args.max_candidates,
        )
    )


def _readiness_smoke(args: argparse.Namespace) -> None:
    print(
        smoke_readiness(
            Path(args.configs_root),
            Path(args.output),
            families=tuple(args.family),
            command_timeout=None if args.command_timeout == 0 else args.command_timeout,
        )
    )


def _triage_summary(
    report: dict[str, object] | None,
    error: str | None = None,
    *,
    run_id: str | None = None,
    runs_dir: Path | None = None,
) -> str:
    if report is None:
        return f"POST-TRAIN TRIAGE\n  UNKNOWN: {error or 'no verified report'}"
    triggers = report["triggers"]
    fired = sorted(
        code for code, details in triggers.items() if details["fired"] is True
    )
    lines = [
        "POST-TRAIN TRIAGE",
        f"  {triage_summary(report)}",
        f"  Triggers: {', '.join(fired) if fired else 'none verified'}",
    ]
    integrity = report["core"]["integrity"]
    if integrity["status"] != "PASS":
        lines.append(
            f"  Checkpoint integrity: {integrity.get('reason') or 'unknown reason'}"
        )
    if run_id is not None and runs_dir is not None:
        from klab.canonical import canonical_json

        digest = hashlib.sha256(canonical_json(report) + b"\n").hexdigest()
        lines.append(
            f"  Report: {runs_dir / run_id / 'post-train-triage' / (digest + '.json')}"
        )
    return "\n".join(lines)


def _triage_report(
    run_id: str, runs_dir: Path
) -> tuple[dict[str, object] | None, str | None]:
    try:
        report = read_triage(run_id, runs_dir)
    except (OSError, ValueError) as error:
        return None, f"invalid or ambiguous artifact: {error}"
    if report is None:
        return None, "no verified post-train triage artifact"
    return report, None


def _checkpoint_plan_spot(args: argparse.Namespace) -> None:
    from klab.training.spot_safety import check_spot_capacity, load_spot_policy

    print(
        json.dumps(
            check_spot_capacity(load_spot_policy(args.policy), args.workspace), indent=2
        )
    )


def _train(args: argparse.Namespace) -> None:
    from klab.training.spot_safety import load_spot_policy

    config = load_config(Path(args.config))
    if args.runs_dir is not None:
        config = config.model_copy(
            update={
                "logging": config.logging.model_copy(
                    update={"root_dir": Path(args.runs_dir).expanduser().resolve()}
                )
            }
        )
    if args.backend:
        config = config.model_copy(
            update={
                "runtime": config.runtime.model_copy(update={"backend": args.backend})
            }
        )
    run_id = train(
        config,
        spot_policy=load_spot_policy(args.spot_policy) if args.spot_policy else None,
        resume=Path(args.resume) if args.resume else None,
        extend_budget=Path(args.extend_budget) if args.extend_budget else None,
        promote=Path(args.promote) if args.promote else None,
        add_ngram_memory=args.add_ngram_memory,
        extend_context=args.extend_context,
        append_data=args.append_data,
        append_from_checkpoint=args.append_from_checkpoint,
        recover=Path(args.recover) if args.recover else None,
        run_id=args.run_id,
        stop_after_step=args.stop_after_step,
        allow_runtime_drift=args.allow_runtime_drift,
        allow_runtime_optimization=args.allow_runtime_optimization,
        stage_bundle=Path(args.stage_bundle) if args.stage_bundle else None,
        authorization=args.runtime_authorization,
        resource_envelope=args.resource_envelope_value,
        tokenizer_batch_documents=args.tokenizer_batch_documents,
        tokenizer_batch_source_bytes=args.tokenizer_batch_source_bytes,
        require_prepared=args.require_prepared,
    )
    print(run_id)
    progress = config.logging.root_dir / run_id / "progress.json"
    try:
        if progress.is_symlink():
            raise ValueError("run progress is a symlink")
        status = json.loads(progress.read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError, TypeError) as error:
        print(_triage_summary(None, f"cannot verify training completion: {error}"))
        return
    if status == "completed":
        report, error = _triage_report(run_id, config.logging.root_dir)
        print(
            _triage_summary(
                report, error, run_id=run_id, runs_dir=config.logging.root_dir
            )
        )


def _eval(args: argparse.Namespace) -> None:
    if args.panel is not None:
        from klab.evaluation.panels import run_panel

        if not args.checkpoint or args.output is None:
            raise ValueError(
                "named panel evaluation requires --checkpoint and --output"
            )
        result = run_panel(
            args.run_id,
            Path(args.runs_dir),
            args.checkpoint,
            Path(args.panel),
            Path(args.output),
            backend=args.backend,
            authorization=args.runtime_authorization,
        )
        print(json.dumps({**result, "output": str(args.output)}, indent=2))
        if not result["evidence_complete"]:
            raise SystemExit(1)
        return
    if args.output is not None:
        raise ValueError("--output requires --panel")
    loaded = load_run(
        args.run_id,
        Path(args.runs_dir),
        args.checkpoint,
        args.backend,
        authorization=args.runtime_authorization,
    )
    result = loaded.evaluate()
    result.update({"source": "standalone_eval", "identity": loaded.identity})
    path = write_inference_result(loaded.run, "eval", result)
    print(json.dumps({**result, "output": str(path)}, indent=2))


def _inference_context(args: argparse.Namespace, native: int) -> int:
    context = args.context_length if args.context_length is not None else native
    if not 1 <= context <= native:
        raise ValueError(f"context-length must be between 1 and {native}")
    return context


def _generate(args: argparse.Namespace) -> None:
    loaded = load_run(
        args.run_id,
        Path(args.runs_dir),
        args.checkpoint,
        args.backend,
        authorization=args.runtime_authorization,
    )
    context = _inference_context(args, loaded.config.model.max_seq_len)
    settings = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
        "stop_sequences": args.stop,
        "use_cache": not args.no_cache,
        "context_length": context,
        "strict_context": args.strict_context,
    }
    speculation = None
    if args.prompt_lookup is not None:
        from klab.evaluation.speculation import PromptLookupSpeculation

        speculation = PromptLookupSpeculation(
            ngram=args.prompt_lookup, max_draft_tokens=args.draft_tokens
        )
    if args.show_prompt:
        print(json.dumps({"format": "raw", "prompt": args.prompt}), file=sys.stderr)
    text = generate(
        loaded.model,
        loaded.tokenizer,
        args.prompt,
        context,
        args.max_new_tokens,
        loaded.device,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        stop_sequences=args.stop,
        use_cache=not args.no_cache,
        strict_context=args.strict_context,
        engine=loaded.engine,
        speculation=speculation,
    )
    speculation_report = (
        None
        if speculation is None
        else {
            "drafter": "prompt_lookup",
            "ngram": speculation.ngram,
            "max_draft_tokens": speculation.max_draft_tokens,
            "model_calls": speculation.model_calls,
            "drafted_tokens": speculation.drafted_tokens,
            "accepted_tokens": speculation.accepted_tokens,
        }
    )
    if args.json:
        print(
            json.dumps(
                {
                    "identity": loaded.identity,
                    "format": "raw",
                    "generation": settings,
                    "prompt": args.prompt,
                    "prompt_tokens": max(
                        1,
                        len(
                            loaded.tokenizer.encode(
                                args.prompt, add_special_tokens=False
                            ).ids
                        ),
                    ),
                    "response": text[len(args.prompt) :],
                    "text": text,
                    **(
                        {"speculation": speculation_report}
                        if speculation_report is not None
                        else {}
                    ),
                },
                sort_keys=True,
            )
        )
    else:
        print(text)
        if speculation_report is not None:
            print(json.dumps({"speculation": speculation_report}), file=sys.stderr)


def _model_exercise(args: argparse.Namespace) -> None:
    from klab.evaluation.reference_exercise import (
        exercise_checkpoint,
        write_observation,
    )

    observation = exercise_checkpoint(
        args.run_id,
        Path(args.checkpoint),
        runs_dir=Path(args.runs_dir),
        backend=args.backend,
        prompt_panel=Path(args.prompt_panel) if args.prompt_panel else None,
    )
    output_root = (
        Path(args.output)
        if args.output
        else Path(args.runs_dir).resolve().parent / "exercises"
    )
    output = write_observation(observation, output_root)
    print(json.dumps({**observation, "output": str(output)}, indent=2, sort_keys=True))


def _capability_list(args: argparse.Namespace) -> None:
    print(json.dumps(list_capability_cards(), indent=2, sort_keys=True))


def _capability_describe(args: argparse.Namespace) -> None:
    print(json.dumps(describe_capability_card(args.card), indent=2, sort_keys=True))


def _capability_result(args: argparse.Namespace, run_id: str, checkpoint: str | None):
    loaded = load_run(run_id, Path(args.runs_dir), checkpoint, args.backend)
    result = evaluate_capability(
        capability_card(args.card),
        loaded.model,
        loaded.tokenizer,
        loaded.config.model.max_seq_len,
        loaded.device,
        engine=loaded.engine,
    )
    result["identity"] = loaded.identity
    return result, write_capability_result(loaded.run, result)


def _code_work_evaluate(args: argparse.Namespace) -> None:
    from klab.evaluation.code_work import run_code_work

    result = run_code_work(
        args.run_id,
        Path(args.runs_dir),
        args.checkpoint,
        Path(args.declaration),
        Path(args.output),
        backend=args.backend,
        authorization=getattr(args, "runtime_authorization", None),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _capability_evaluate(args: argparse.Namespace) -> None:
    result, path = _capability_result(args, args.run_id, args.checkpoint)
    print(json.dumps({**result, "output": str(path)}, indent=2, sort_keys=True))


def _capability_compare(args: argparse.Namespace) -> None:
    # One model resident at a time, including at user-selected larger scales.
    base, base_path = _capability_result(args, args.base_run_id, args.base_checkpoint)
    variant, variant_path = _capability_result(
        args, args.variant_run_id, args.variant_checkpoint
    )
    result = compare_results(base, variant, vary=args.vary)
    result.update({"base_output": str(base_path), "variant_output": str(variant_path)})
    path = write_inference_result(
        Path(args.runs_dir) / args.variant_run_id, "comparison", result
    )
    print(json.dumps({**result, "output": str(path)}, indent=2, sort_keys=True))


def _capability_suite(args: argparse.Namespace) -> None:
    del args
    print(json.dumps(phase_e_behavior_suite(), indent=2, sort_keys=True))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_strict_json(path: Path, *, max_bytes: int = 2 * 1024 * 1024) -> object:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ValueError(f"cannot open JSON input {path}: {error}") from error
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"JSON input must be a regular file: {path}")
        if info.st_size > max_bytes:
            raise ValueError(f"JSON input exceeds {max_bytes} byte limit: {path}")
        content = handle.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"JSON input exceeds {max_bytes} byte limit: {path}")
    try:
        return json.loads(
            content.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read strict JSON from {path}: {error}") from error


def _write_json_exclusive(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _task_jsonl(cases: Iterable[Any]) -> bytes:
    records: list[bytes] = []
    for case in cases:
        record = {
            "messages": [
                {"role": "user", "content": case.prompt},
                {"role": "assistant", "content": case.expected},
            ]
        }
        records.append(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    if not records:
        raise ValueError("task split must not be empty")
    return b"".join(records)


def _task_inventory(directory: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"generated task bundle contains a symlink: {path}")
        if path.is_file() and path.relative_to(directory) != Path("manifest.json"):
            content = path.read_bytes()
            files.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                }
            )
    return files


def _build_phase_e_tasks(args: argparse.Namespace) -> None:
    if args.task == "memory-allocation":
        if args.seed != 17:
            raise ValueError("memory-allocation uses fixed provenance and seeds")
        inputs = (
            "tokenizer",
            "train_jsonl",
            "validation_jsonl",
            "oracle_audit",
            "provenance",
            "cards_dir",
        )
        missing = [
            "--" + name.replace("_", "-") for name in inputs if not getattr(args, name)
        ]
        if missing:
            raise ValueError(
                "memory-allocation requires explicit inputs: " + ", ".join(missing)
            )
        from klab.data.allocation_tasks import build_memory_allocation

        manifest = build_memory_allocation(
            Path(args.output),
            tokenizer_path=Path(args.tokenizer),
            train_path=Path(args.train_jsonl),
            validation_path=Path(args.validation_jsonl),
            audit_path=Path(args.oracle_audit),
            provenance_path=Path(args.provenance),
            cards_dir=Path(args.cards_dir),
        )
        print(manifest)
        return
    from klab.data.phase_e_tasks import (
        build_wikidata_mini,
        load_wikidata_mini_cases,
        math_identity_test_cases,
        math_identity_training_cases,
        math_identity_validation_cases,
        python_stdlib_test_cases,
        python_stdlib_training_cases,
        python_stdlib_validation_cases,
        wikidata_mini_spec,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite task bundle: {output}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.phase-e-", dir=output.parent)
    )
    try:
        if args.task == "wikidata-mini":
            if args.seed != 17:
                raise ValueError(
                    "wikidata-mini uses fixed source revisions and no seed"
                )
            source = staging / "source"
            build_wikidata_mini(source)
            cases = load_wikidata_mini_cases(source, split="test")
            factual = tuple(case for case in cases if case.kind == "factual_recall")
            paraphrase = tuple(case for case in cases if case.kind == "paraphrase")
            cards = staging / "cards"
            cards.mkdir()
            payloads = (
                task_card_payload(
                    "wikidata-mini-factual-recall-v1",
                    "Recalls date facts from train-only exposure to pinned Wikidata entities.",
                    "Four-entity fixture with CC0 source facts and MIT prompts; not a broad factual-knowledge benchmark.",
                    factual,
                ),
                task_card_payload(
                    "wikidata-mini-paraphrase-v1",
                    "Retrieves held-out entity facts under alternate question wording.",
                    "The answers derive only from pinned Wikidata test-entity revisions and must not enter training.",
                    paraphrase,
                ),
            )
            for payload in payloads:
                card_path = cards / f"{payload['name']}.json"
                card_path.write_text(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            source_manifest = json.loads(
                (source / "manifest.json").read_text(encoding="utf-8")
            )
            details = {
                "task": args.task,
                "seed": None,
                "license": "MIT prompts/format; CC0-1.0 Wikidata facts",
                "source_data_license": "CC0-1.0",
                "generated_artifact_license": "MIT",
                "source_manifest": wikidata_mini_spec(),
                "source_manifest_sha256": source_manifest["source_manifest_sha256"],
                "train_paths": ["source/train.jsonl"],
                "validation_paths": ["source/validation.jsonl"],
                "held_out_cards": [
                    f"cards/{payload['name']}.json" for payload in payloads
                ],
                "split_unit": "entity",
                "test_answers_used_for_training": False,
            }
        else:
            if args.task == "math-identities":
                train = math_identity_training_cases(args.seed)
                validation = math_identity_validation_cases(args.seed)
                test = math_identity_test_cases(args.seed)
                card_name = "math-identity-novel-operands-v1"
                hypothesis = "Applies practiced arithmetic identities to disjoint held-out operands."
                limitations = "Synthetic integer tasks measure this fixed identity set, not broad mathematical reasoning."
                answer_method = "integer arithmetic over generated operands"
            elif args.task == "python-stdlib":
                train = python_stdlib_training_cases(args.seed)
                validation = python_stdlib_validation_cases(args.seed)
                test = python_stdlib_test_cases(args.seed)
                card_name = "python-stdlib-api-v1"
                hypothesis = "Applies documented Python standard-library operations to held-out inputs."
                limitations = "Expected outputs use a finite trusted standard-library allowlist; generated Python is never executed."
                answer_method = "direct calls to fixed standard-library functions"
            else:
                raise ValueError(f"unknown Phase E task bundle: {args.task}")

            (staging / "train.jsonl").write_bytes(_task_jsonl(train))
            (staging / "validation.jsonl").write_bytes(_task_jsonl(validation))
            cases_path = staging / "test-cases.json"
            cases_path.write_text(
                json.dumps(
                    [asdict(case) for case in test],
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            cards = staging / "cards"
            cards.mkdir()
            payload = task_card_payload(card_name, hypothesis, limitations, test)
            (cards / f"{card_name}.json").write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            details = {
                "task": args.task,
                "seed": args.seed,
                "license": "MIT",
                "provenance": "Kybern Model Lab-authored synthetic tasks; no upstream benchmark solutions.",
                "answer_method": answer_method,
                "train_paths": ["train.jsonl"],
                "validation_paths": ["validation.jsonl"],
                "held_out_cards": [f"cards/{card_name}.json"],
                "test_answers_used_for_training": False,
            }

        manifest = {
            "format": "klab-phase-e-task-bundle",
            "version": 1,
            **details,
            "files": _task_inventory(staging),
        }
        (staging / "manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _rename_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps({"output": str(output), "manifest": str(output / "manifest.json")})
    )


def _research_corpus_mine(args: argparse.Namespace) -> None:
    from klab.data.lexical_mining import (
        analyze_prepared_training,
        analyze_training_corpus,
    )

    model_run = load_config(Path(args.model_config)) if args.model_config else None
    if model_run is None and (
        args.table_size is None or args.memory_dim is None or args.ngram_orders is None
    ):
        raise ValueError("supply --model-config or all table/dimension/order settings")
    if model_run is not None and any(
        value is not None
        for value in (
            args.table_size,
            args.memory_dim,
            args.ngram_orders,
            args.hash_heads,
        )
    ):
        raise ValueError(
            "--model-config replaces explicit table/dimension/order settings"
        )
    settings = {
        "table_size": model_run.model.memory_table_size
        if model_run
        else args.table_size,
        "memory_dim": model_run.model.memory_dim if model_run else args.memory_dim,
        "ngram_orders": (
            model_run.model.memory_ngram_orders or (model_run.model.memory_ngram_size,)
        )
        if model_run
        else tuple(args.ngram_orders),
        "hash_heads": model_run.model.memory_hash_heads
        if model_run
        else (args.hash_heads or 1),
        "model_config": model_run.model if model_run else None,
        "memory_block_index": args.memory_block,
    }
    if args.prepared_train is not None:
        if args.tokenizer is not None:
            raise ValueError(
                "--prepared-train uses its recorded tokenizer identity; omit --tokenizer"
            )
        if args.sequence_length is None or args.max_blocks is None:
            raise ValueError(
                "--prepared-train requires --sequence-length and --max-blocks"
            )
        result = analyze_prepared_training(
            Path(args.prepared_train),
            sequence_length=args.sequence_length,
            max_blocks=args.max_blocks,
            tokenizer_path=model_run.tokenizer.path if model_run else None,
            **settings,
        )
    else:
        if args.tokenizer is None:
            raise ValueError("--train-jsonl requires --tokenizer")
        if args.sequence_length is not None or args.max_blocks is not None:
            raise ValueError("packed sampling options require --prepared-train")
        result = analyze_training_corpus(
            Path(args.train_jsonl),
            Path(args.tokenizer),
            **settings,
        )
    output = _write_json_exclusive(Path(args.output), result)
    print(
        json.dumps(
            {"output": str(output), "analysis_sha256": result["analysis_sha256"]}
        )
    )


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _review_records_from_results(base: object, variant: object) -> list[dict[str, str]]:
    if not isinstance(base, dict) or not isinstance(variant, dict):
        raise TypeError("capability results must be JSON objects")
    for label, result in (("base", base), ("variant", variant)):
        if (
            result.get("format") != "capability_result_v2"
            or result.get("valid") is not True
            or not isinstance(result.get("identity"), dict)
            or not isinstance(result.get("results"), list)
        ):
            raise ValueError(f"{label} capability result is invalid or incomplete")
        result_digest = result.get("result_digest")
        if not _valid_sha256(result_digest):
            raise ValueError(f"{label} capability result lacks a content digest")
        body = {key: value for key, value in result.items() if key != "result_digest"}
        canonical = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != result_digest:
            raise ValueError(f"{label} capability result digest mismatch")
        if not _valid_sha256(result.get("card_digest")) or not _valid_sha256(
            result.get("evaluation_source_sha256")
        ):
            raise ValueError(f"{label} capability result lacks card/evaluator digests")
        if not _valid_sha256(result["identity"].get("checkpoint_sha256")):
            raise ValueError(f"{label} capability result lacks checkpoint identity")
    if (
        base["card_digest"] != variant["card_digest"]
        or base["evaluation_source_sha256"] != variant["evaluation_source_sha256"]
    ):
        raise ValueError("capability results must share card and evaluator identities")

    def cases(result: dict[str, object], label: str) -> dict[str, dict[str, str]]:
        indexed: dict[str, dict[str, str]] = {}
        for raw in result["results"]:
            if not isinstance(raw, dict):
                raise TypeError(f"{label} capability result has a malformed case")
            required = ("id", "prompt", "expected", "response")
            if not all(isinstance(raw.get(key), str) and raw[key] for key in required):
                raise ValueError(f"{label} capability result case is incomplete")
            identifier = raw["id"]
            if identifier in indexed:
                raise ValueError(f"{label} capability result has duplicate case IDs")
            indexed[identifier] = {key: raw[key] for key in required}
        return indexed

    base_cases, variant_cases = cases(base, "base"), cases(variant, "variant")
    if not base_cases or set(base_cases) != set(variant_cases):
        raise ValueError("capability results must contain the same nonempty case set")
    records: list[dict[str, str]] = []
    for identifier in sorted(base_cases):
        left, right = base_cases[identifier], variant_cases[identifier]
        if left["prompt"] != right["prompt"] or left["expected"] != right["expected"]:
            raise ValueError(f"capability case {identifier} differs across results")
        records.extend(
            (
                {
                    "case_id": identifier,
                    "prompt": left["prompt"],
                    "condition": "baseline",
                    "response": left["response"],
                    "source_id": base["result_digest"],
                },
                {
                    "case_id": identifier,
                    "prompt": right["prompt"],
                    "condition": "variant",
                    "response": right["response"],
                    "source_id": variant["result_digest"],
                },
            )
        )
    return records


def _review_bundle(args: argparse.Namespace) -> None:
    from klab.evaluation.human_review import create_review_bundle

    if args.records:
        if args.variant_result:
            raise ValueError("--variant-result requires --base-result")
        records = _read_strict_json(Path(args.records))
    else:
        if not args.variant_result:
            raise ValueError(
                "--base-result and --variant-result must be supplied together"
            )
        records = _review_records_from_results(
            _read_strict_json(Path(args.base_result)),
            _read_strict_json(Path(args.variant_result)),
        )
    criteria = _read_strict_json(Path(args.criteria))
    bundle, reveal_map = create_review_bundle(records, criteria, args.seed)
    bundle_path, reveal_path = Path(args.bundle), Path(args.reveal_map)
    if bundle_path.resolve() == reveal_path.resolve():
        raise ValueError("review bundle and reveal map must be separate files")
    created_bundle = _write_json_exclusive(bundle_path, bundle)
    try:
        _write_json_exclusive(reveal_path, reveal_map)
    except BaseException:
        created_bundle.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "bundle": str(bundle_path),
                "bundle_digest": bundle["bundle_digest"],
                "reveal_map": str(reveal_path),
                "reveal_map_digest": reveal_map["reveal_map_digest"],
            },
            sort_keys=True,
        )
    )


def _review_validate(args: argparse.Namespace) -> None:
    from klab.evaluation.human_review import validate_review_judgments

    bundle_path, judgments_path, output_path = (
        Path(args.bundle),
        Path(args.judgments),
        Path(args.output),
    )
    if output_path.resolve() in {bundle_path.resolve(), judgments_path.resolve()}:
        raise ValueError("normalized judgment output must not overwrite an input")
    bundle = _read_strict_json(bundle_path)
    judgments = _read_strict_json(judgments_path)
    normalized = validate_review_judgments(bundle, judgments)
    output = _write_json_exclusive(output_path, normalized)
    print(
        json.dumps(
            {
                "output": str(output),
                "bundle_digest": normalized["bundle_digest"],
                "judgment_count": len(normalized["judgments"]),
            },
            sort_keys=True,
        )
    )


def _lifecycle_registry(args: argparse.Namespace) -> tuple[LifecycleRegistry, Path]:
    from klab.research.lifecycle import load_lifecycle

    return (
        load_lifecycle(Path(args.lifecycle) if args.lifecycle else None),
        Path(args.evidence_root),
    )


def _research_diagnostics(
    registry: LifecycleRegistry,
    *,
    study_sha256: str | None = None,
    protocol_sha256: str | None = None,
) -> list[dict[str, object]]:
    from klab.research.lifecycle import prior_evidence_warnings

    return prior_evidence_warnings(
        registry, study_sha256=study_sha256, protocol_sha256=protocol_sha256
    )


def _study_plan(args: argparse.Namespace) -> None:
    from klab.experiments.study import plan_study, study_plan_payload

    plan = plan_study(Path(args.path), max_runs=args.max_runs)
    payload = study_plan_payload(plan)
    registry, _ = _lifecycle_registry(args)
    payload["research_diagnostics"] = _research_diagnostics(
        registry, study_sha256=plan.study_sha256
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _study_submit(args: argparse.Namespace) -> None:
    from klab.experiments.study import (
        plan_study,
        study_execution_paths,
        submit_study,
    )

    plan = plan_study(Path(args.path), max_runs=args.max_runs)
    registry, _ = _lifecycle_registry(args)
    for warning in _research_diagnostics(registry, study_sha256=plan.study_sha256):
        print(json.dumps(warning, sort_keys=True), file=sys.stderr)
    store, receipt_path = study_execution_paths(
        plan,
        store=Path(args.store) if args.store else None,
        receipt=Path(args.receipt) if args.receipt else None,
    )
    receipt = submit_study(
        plan,
        receipt_path,
        store=store,
        worker=args.worker,
        stage_bundle=Path(args.stage_bundle) if args.stage_bundle else None,
    )
    print(
        json.dumps(
            {
                "study_sha256": plan.study_sha256,
                "receipt": str(receipt_path),
                "store": str(store),
                "runs": receipt["runs"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _study_recover(args: argparse.Namespace) -> None:
    from klab.experiments.study import plan_study, recover_study_receipt

    receipt = recover_study_receipt(
        plan_study(Path(args.path), max_runs=args.max_runs),
        Path(args.intent),
        Path(args.output),
        store=Path(args.store),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


def _study_collect(args: argparse.Namespace) -> None:
    from klab.experiments.study import collect_study, plan_study

    plan = plan_study(Path(args.path), max_runs=args.max_runs)
    report, report_path = collect_study(
        plan,
        Path(args.receipt),
        runs_dir=Path(args.runs_dir)
        if args.runs_dir
        else Path(args.receipt).parent / "runs",
        checkpoint=args.checkpoint,
        backend=args.backend,
    )
    print(json.dumps({**report, "output": str(report_path)}, indent=2, sort_keys=True))


def _research_list(args: argparse.Namespace) -> None:
    from klab.research.catalog import list_research

    entries = list_research()
    if args.json:
        print(
            json.dumps(
                [entry.model_dump(mode="json") for entry in entries],
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for entry in entries:
            print(f"{entry.id}\t{entry.title}\n  {entry.question}")


def _research_describe(args: argparse.Namespace) -> None:
    from klab.research.catalog import load_recipe, load_research

    entry = load_research(args.reference)
    payload = entry.model_dump(mode="json")
    recipe = load_recipe(
        entry,
        local_root=Path(args.reference).resolve().parent
        if str(args.reference).endswith(".json")
        else None,
    )
    payload["available_designs"] = sorted(recipe.designs)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{entry.title}\n{entry.question}\n\nHypothesis: {entry.hypothesis}")
        print(f"Failure interpretation: {entry.failure_interpretation}")
        print(f"Fixed controls: {'; '.join(entry.controls)}")
        print(f"Varied fields: {', '.join(entry.independent_variables)}")
        print(f"Designs: {', '.join(sorted(recipe.designs))}")
        print(f"Limitations: {'; '.join(entry.cannot_establish)}")


def _research_scaffold(args: argparse.Namespace) -> None:
    from klab.experiments.study import plan_study
    from klab.research.scaffold import scaffold_research

    if args.resource_root is not None and args.lifecycle is None:
        from klab.research.catalog import resource_path

        args.lifecycle = str(resource_path("lifecycle.json", root=args.resource_root))
    registry, _ = _lifecycle_registry(args)
    with tempfile.TemporaryDirectory(
        prefix="klab-scaffold-preview-", dir=ensure_scratch_dir(args.work_dir)
    ) as preview_root:
        preview = scaffold_research(
            args.reference,
            Path(preview_root) / "scaffold",
            scale=args.scale,
            data=args.data,
            backend=args.backend,
            design=args.design,
            root=args.resource_root,
        )
        study_sha256 = plan_study(preview / "study.yaml").study_sha256
    for warning in _research_diagnostics(registry, study_sha256=study_sha256):
        print(json.dumps(warning, sort_keys=True), file=sys.stderr)
    path = scaffold_research(
        args.reference,
        Path(args.output),
        scale=args.scale,
        data=args.data,
        backend=args.backend,
        design=args.design,
        root=args.resource_root,
    )
    print(path)


def _research_baseline_list(args: argparse.Namespace) -> None:
    from klab.research.lifecycle import research_status

    registry, evidence_root = _lifecycle_registry(args)
    print(
        json.dumps(
            research_status(registry, evidence_root=evidence_root).get("baselines", []),
            indent=2,
            sort_keys=True,
        )
    )


def _research_baseline_describe(args: argparse.Namespace) -> None:
    from klab.research.lifecycle import research_status

    registry, evidence_root = _lifecycle_registry(args)
    baselines = research_status(registry, evidence_root=evidence_root).get(
        "baselines", []
    )
    if not isinstance(baselines, list):
        raise TypeError("lifecycle status omitted its baseline inventory")
    match = next(
        (
            item
            for item in baselines
            if isinstance(item, dict) and item.get("id") == args.identifier
        ),
        None,
    )
    if match is None:
        raise ValueError(f"unknown baseline: {args.identifier}")
    print(json.dumps(match, indent=2, sort_keys=True))


def _research_validate(args: argparse.Namespace) -> None:
    from klab.research.lifecycle import validate_baseline, validate_lifecycle

    registry, evidence_root = _lifecycle_registry(args)
    if args.baseline is None:
        payload = validate_lifecycle(registry, evidence_root=evidence_root)
        valid = payload["valid"]
    else:
        payload = validate_baseline(
            registry, args.baseline, evidence_root=evidence_root
        )
        baseline = payload["baseline"]
        valid = isinstance(baseline, dict) and baseline.get("valid") is True
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not valid:
        raise SystemExit(1)


def _is_learned_portability_root(root: Path) -> bool:
    protocol = root / "portability_protocol.json"
    if protocol.is_symlink() or not protocol.is_file():
        return False
    try:
        payload = json.loads(protocol.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return False
    return (
        isinstance(payload, dict)
        and payload.get("experiment") == "learned-engram-portability-v1"
    )


def _research_portability_build(args: argparse.Namespace) -> None:
    if args.experiment == "learned-engram-portability-v1":
        from klab.research.learned_portability_campaign import (
            build_learned_portability_campaign,
        )

        if args.updates is not None:
            raise ValueError(
                "--updates is not supported by learned-engram-portability-v1"
            )
        protocol = build_learned_portability_campaign(
            Path(args.output), seed=args.seed, scale=args.scale, backend=args.backend
        )
    else:
        from klab.research.portability_campaign import build_portability_campaign

        if args.backend != "auto":
            raise ValueError(
                "--backend is only supported by learned-engram-portability-v1"
            )
        protocol = build_portability_campaign(
            Path(args.output),
            seed=args.seed,
            scale=args.scale,
            updates=4 if args.updates is None else args.updates,
        )
    print(
        json.dumps(
            {
                "protocol": str(protocol),
                "plan": str(protocol.parent / "plan.json"),
                "campaign_id": json.loads(protocol.read_text())["campaign_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _research_portability_plan(args: argparse.Namespace) -> None:
    root = Path(args.campaign_root)
    if _is_learned_portability_root(root):
        from klab.research.learned_portability_campaign import (
            learned_portability_plan,
        )

        payload = learned_portability_plan(root)
    else:
        from klab.research.portability_runner import portability_resource_plan

        payload = portability_resource_plan(root)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _research_portability_run(args: argparse.Namespace) -> None:
    root = Path(args.campaign_root)
    if _is_learned_portability_root(root):
        from klab.research.learned_portability_campaign import (
            execute_learned_portability_campaign,
        )

        if any(
            value is not None
            for value in (
                args.recipient,
                args.representation,
                args.condition,
                args.seed,
            )
        ):
            raise ValueError(
                "learned portability run does not accept coordinate overrides"
            )
        if args.max_wall_seconds is None:
            raise ValueError("learned portability run requires --max-wall-seconds")
        print(
            json.dumps(
                execute_learned_portability_campaign(
                    root, max_wall_seconds=args.max_wall_seconds
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.max_wall_seconds is not None:
        raise ValueError("--max-wall-seconds is only supported by learned portability")
    if any(
        value is None
        for value in (args.recipient, args.representation, args.condition, args.seed)
    ):
        raise ValueError(
            "compiled portability run requires recipient, representation, condition, and seed"
        )
    from klab.research.portability_runner import (
        build_portability_evidence,
        execute_portability_arm,
        record_portability_failure,
    )

    try:
        receipt = execute_portability_arm(
            root,
            recipient=args.recipient,
            representation=args.representation,
            condition=args.condition,
            seed=args.seed,
            updates=json.loads((root / "portability_protocol.json").read_text())[
                "training"
            ]["planned_updates"],
        )
    except Exception as error:
        record_portability_failure(
            root,
            recipient=args.recipient,
            representation=args.representation,
            condition=args.condition,
            seed=args.seed,
            error=error,
        )
        raise
    evidence = build_portability_evidence(root)
    print(
        json.dumps(
            {"receipt": receipt, "evidence": str(evidence)}, indent=2, sort_keys=True
        )
    )


def _research_portability_continue(args: argparse.Namespace) -> None:
    root = Path(args.campaign_root)
    if _is_learned_portability_root(root):
        from klab.research.learned_portability_campaign import (
            continue_learned_portability_campaign,
        )

        if args.max_wall_seconds is None:
            raise ValueError("learned portability continue requires --max-wall-seconds")
        print(
            json.dumps(
                continue_learned_portability_campaign(
                    root, max_wall_seconds=args.max_wall_seconds
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.max_wall_seconds is not None:
        raise ValueError("--max-wall-seconds is only supported by learned portability")
    from klab.research.portability_runner import (
        build_portability_evidence,
        continue_portability_campaign,
    )

    result = continue_portability_campaign(root, max_arms=args.max_arms)
    result["evidence"] = str(build_portability_evidence(root))
    print(json.dumps(result, indent=2, sort_keys=True))


def _research_portability_report(args: argparse.Namespace) -> None:
    root = Path(args.campaign_root)
    if _is_learned_portability_root(root):
        from klab.research.learned_portability_campaign import (
            report_learned_portability_campaign,
        )

        print(
            json.dumps(
                report_learned_portability_campaign(root), indent=2, sort_keys=True
            )
        )
        return
    from klab.experiments.reporting import _validate_portability_evidence
    from klab.research.portability_runner import build_portability_evidence

    path = build_portability_evidence(root)
    evidence, _ = _validate_portability_evidence(path)
    print(
        json.dumps(
            {"evidence": str(path), "summary": evidence["summary"]},
            indent=2,
            sort_keys=True,
        )
    )


def _learn_list(args: argparse.Namespace) -> None:
    from klab.research.catalog import list_lessons

    lessons = list_lessons()
    if args.json:
        print(
            json.dumps(
                [lesson.model_dump(mode="json") for lesson in lessons],
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for lesson in lessons:
            print(f"{lesson.id}\t{lesson.title}\n  {lesson.summary}")


def _learn_describe(args: argparse.Namespace) -> None:
    from klab.research.catalog import load_lesson

    lesson = load_lesson(args.identifier)
    payload = lesson.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{lesson.title}\n{lesson.summary}")
        for step in lesson.steps:
            print(f"\n{step.source_path}:{step.source_symbol}")
            print(step.explanation)
            print("Shapes: " + "; ".join(step.shapes))
            print("Observe: " + ", ".join(step.observe))
        print("Limits: " + "; ".join(lesson.limits))


def _learn_scaffold(args: argparse.Namespace) -> None:
    from klab.research.scaffold import scaffold_lesson

    path = scaffold_lesson(
        args.identifier,
        Path(args.output),
        root=args.resource_root,
        scale=args.scale,
        data=args.data,
        backend=args.backend,
        memory_package=Path(args.memory_package) if args.memory_package else None,
    )
    print(path)


def _learn_probe(args: argparse.Namespace) -> None:
    from klab.research.probe import probe_model

    payload = probe_model(load_config(Path(args.config)), args.prompt)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"Execution: {payload['execution']} (configured runtime shown separately)"
        )
        print(f"Input {payload['input_shape']} -> logits {payload['logits_shape']}")
        print(f"Evidence scope: {payload['evidence_scope']}")
        print(json.dumps(payload["diagnostics"], indent=2, sort_keys=True))


def _study_report(args: argparse.Namespace) -> None:
    from klab.experiments.reporting import build_study_report, write_study_report

    if bool(args.identity_expectations) != bool(args.candidate_commit):
        raise ValueError(
            "identity expectations and candidate commit are required together"
        )
    expectations = None
    if args.identity_expectations:
        from klab.experiments.identity_acceptance import (
            accept_identity,
            read_expectations,
        )

        expectations = read_expectations(Path(args.identity_expectations))

    report = build_study_report(
        Path(args.path),
        Path(args.receipt),
        Path(args.evidence),
        research_path=Path(args.research) if args.research else None,
        runs_dir=Path(args.runs_dir) if args.runs_dir else None,
        portability_evidence_path=(
            Path(args.portability_evidence) if args.portability_evidence else None
        ),
    )
    if expectations is not None:
        report["identity_acceptance"] = accept_identity(
            report, expectations, args.candidate_commit
        )
    print(write_study_report(report, Path(args.output)))


def _reference_pythia_trajectory(args: argparse.Namespace) -> None:
    from klab.research.pythia import evaluate_pythia_trajectory

    print(
        evaluate_pythia_trajectory(
            load_config(Path(args.config)),
            Path(args.output),
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            device=args.device,
            steps=tuple(args.steps),
        )
    )


def _chat(args: argparse.Namespace) -> None:
    loaded = load_run(
        args.run_id,
        Path(args.runs_dir),
        args.checkpoint,
        args.backend,
        authorization=args.runtime_authorization,
    )
    context = _inference_context(args, loaded.config.model.max_seq_len)
    stops = ["\nUser:", "\nSystem:", "\nAssistant:", *args.stop]
    if not args.json:
        print(
            "Verified checkpoint: " + json.dumps(loaded.identity, sort_keys=True),
            file=sys.stderr,
        )
    history: list[ChatMessage] = []
    turns = []
    settings = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
        "stop_sequences": stops,
        "use_cache": not args.no_cache,
        "context_length": context,
        "strict_context": True,
    }
    if args.transcript and Path(args.transcript).exists():
        raise FileExistsError(f"transcript already exists: {args.transcript}")

    def respond(message: str) -> None:
        prompt, dropped = prepare_chat_prompt(
            history,
            message,
            loaded.tokenizer,
            context,
            args.max_new_tokens,
            system=args.system,
        )
        if args.show_prompt:
            print(
                json.dumps({"format": "chat_transcript_v1", "prompt": prompt}),
                file=sys.stderr,
            )
        completion = generate(
            loaded.model,
            loaded.tokenizer,
            prompt,
            context,
            args.max_new_tokens,
            loaded.device,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            strict_context=True,
            stop_sequences=stops,
            use_cache=not args.no_cache,
            engine=loaded.engine,
        )
        reply = assistant_reply(completion, prompt)
        turn = {
            "format": "chat_transcript_v1",
            "user": message,
            "prompt": prompt,
            "response": reply,
            "dropped_history_turns": dropped,
            "prompt_tokens": len(
                loaded.tokenizer.encode(prompt, add_special_tokens=False).ids
            ),
        }
        turns.append(turn)
        if args.json:
            print(
                json.dumps(
                    {"identity": loaded.identity, "generation": settings, **turn},
                    sort_keys=True,
                )
            )
        else:
            if dropped:
                print(
                    f"[Context: dropped {dropped} oldest complete turn(s)]",
                    file=sys.stderr,
                )
            print(f"Assistant: {reply}")
        history.extend((ChatMessage("user", message), ChatMessage("assistant", reply)))

    try:
        if args.message is not None:
            respond(args.message)
            return
        if not args.json:
            print(
                "Local chat. /exit ends; /reset clears context. Greedy unless --temperature is set."
            )
        while True:
            try:
                message = input("" if args.json else "You: ")
            except EOFError:
                return
            if message.strip().lower() in {"/exit", "/quit"}:
                return
            if message.strip().lower() == "/reset":
                history.clear()
                continue
            if not message.strip():
                continue
            try:
                respond(message)
            except ValueError as error:
                print(f"Chat input rejected: {error}", file=sys.stderr)
    finally:
        if args.transcript:
            path = Path(args.transcript)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as handle:
                json.dump(
                    {
                        "format": "chat_transcript_v1",
                        "identity": loaded.identity,
                        "system": args.system,
                        "generation": settings,
                        "turns": turns,
                        "history": [asdict(item) for item in history],
                    },
                    handle,
                    indent=2,
                )
                handle.write("\n")


def _weights_import(args: argparse.Namespace) -> None:
    from klab.training.weight_import import import_weights

    result = import_weights(
        Path(args.source),
        Path(args.destination),
        load_config(Path(args.config)),
        source_format=args.format,
        source_tokenizer=Path(args.source_tokenizer),
        provenance=Path(args.provenance),
    )
    print(json.dumps(asdict(result), sort_keys=True, default=str))


def _tokenizer_batch_integer(value: str, *, maximum: int) -> int:
    if not value.isascii() or not value.isdecimal():
        raise argparse.ArgumentTypeError("must be an exact positive decimal integer")
    number = int(value)
    if not 1 <= number <= maximum:
        raise argparse.ArgumentTypeError(f"must be between 1 and {maximum}")
    return number


def _tokenizer_batch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tokenizer-batch-documents",
        type=lambda value: _tokenizer_batch_integer(value, maximum=256),
        default=TOKENIZER_BATCH_DOCUMENTS,
        metavar="N",
        help=f"Preparation documents per Rust tokenizer batch (1..256; default: {TOKENIZER_BATCH_DOCUMENTS})",
    )
    parser.add_argument(
        "--tokenizer-batch-source-bytes",
        type=lambda value: _tokenizer_batch_integer(value, maximum=8388608),
        default=TOKENIZER_BATCH_SOURCE_BYTES,
        metavar="N",
        help="UTF-8 source bytes per tokenizer batch (1..8388608; default: 1048576)",
    )


def _lab_lookup_failure(args: argparse.Namespace, error: Exception) -> None:
    """Lookup errors have one payload, whether found in preflight or a reader."""
    from klab.lab_results import ResultLookupError

    if isinstance(error, ResultLookupError):
        if (
            args.command in {"explore", "probe"}
            and "--work-dir/--lab-dir" in error.hint
        ):
            error = ResultLookupError(
                error.spec,
                error.status,
                error.reason,
                error.hint.replace(
                    "--work-dir/--lab-dir", "--work-dir or --config lab_dir"
                ),
            )
        if args.json:
            print(json.dumps(error.as_json(), indent=2, sort_keys=True))
            raise SystemExit(1) from None
        raise SystemExit(f"klab {args.command}: {error}") from None


def _try(args: argparse.Namespace) -> None:
    """Lab mode: YAML delta (or config) vs baseline -> one scored comparison."""
    from klab.lab_mode import run_try, summarize

    try:
        record, path = run_try(
            Path(args.candidate),
            Path(args.vs),
            work_dir=resolve_work_dir(args.work_dir),
            lab_dir=args.lab_dir,
            seed=args.seed,
            backend=args.backend,
            authorization=args.runtime_authorization,
            resource_envelope=args.resource_envelope_value,
            fresh_baseline=args.fresh_baseline,
            repeat_baseline=args.repeat_baseline,
            continuation=args.continuation,
            probe_tier=None if args.probe_tier == "none" else args.probe_tier,
            request={
                **args.try_config_loaded.model_dump(mode="json"),
                "baseline": args.vs,
                "backend_override": args.backend,
                "runtime_id": args.runtime,
                "runtime_profile_path": str(args.try_config_loaded.runtime_profile)
                if args.try_config_loaded.runtime_profile is not None
                else None,
                "resource_envelope_path": str(args.try_config_loaded.resource_envelope)
                if args.try_config_loaded.resource_envelope is not None
                else None,
                "runtime_profile": args.runtime_profile_loaded.model_dump(mode="json")
                if args.runtime_profile_loaded is not None
                else None,
                "resource_envelope": args.resource_envelope_value.model_dump(
                    mode="json"
                )
                if args.resource_envelope_value is not None
                else None,
            },
        )
    except (ValueError, OSError) as error:
        raise SystemExit(f"klab try: {error}") from None
    if args.json:
        print(json.dumps({**record, "record": str(path)}, indent=2, sort_keys=True))
    else:
        from klab.probes.render import use_color

        print(summarize(record, path, color=use_color()))
    if record["exit_status"]:
        raise SystemExit(record["exit_status"])


def _report_run(args: argparse.Namespace) -> None:
    """Keep verified observations and advisory triage separate in one run view."""
    run_id = args.record
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise SystemExit("klab report: run ID must be one safe path component")
    runs_dir = (
        Path(args.runs_dir)
        if args.runs_dir
        else resolve_work_dir(args.work_dir) / "runs"
    )
    try:
        manifest = runs_dir / run_id / "manifest.json"
        if not manifest.exists() and not manifest.is_symlink():
            raise FileNotFoundError(f"run manifest unavailable: {manifest}")
        evidence = experiment_evidence(runs_dir / run_id)
    except FileNotFoundError as error:
        evidence = {"status": "missing_evidence", "reason": str(error)}
    except (OSError, ValueError, TypeError, KeyError) as error:
        evidence = {"status": "invalid_evidence", "reason": str(error)}
    triage, error = _triage_report(run_id, runs_dir)
    payload = {
        "format": "klab-run-report-v1",
        "run_id": run_id,
        "evidence": evidence,
        "triage": triage
        if triage is not None
        else {"status": "UNKNOWN", "reason": error},
    }
    if args.surface_dir is not None:
        payload["surface_review"] = surface_review_status(
            run_id, runs_dir, Path(args.surface_dir)
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"RUN EVIDENCE: {run_id}")
        print(json.dumps(evidence, indent=2, sort_keys=True))
        print(_triage_summary(triage, error, run_id=run_id, runs_dir=runs_dir))
        if "surface_review" in payload:
            print("SURFACE REVIEW (read-only)")
            print(json.dumps(payload["surface_review"], indent=2, sort_keys=True))
    if evidence.get("status") in {"missing_evidence", "invalid_evidence"}:
        raise SystemExit(1)


def _report_research(args: argparse.Namespace) -> None:
    """Show declared lifecycle state and next tests, never authorize a run."""
    from klab.research.lifecycle import next_experiments, research_status

    registry, evidence_root = _lifecycle_registry(args)
    status = research_status(registry, evidence_root=evidence_root)
    if args.study:
        from klab.experiments.study import plan_study

        status["research_diagnostics"] = _research_diagnostics(
            registry, study_sha256=plan_study(Path(args.study)).study_sha256
        )
    elif args.protocol_sha256:
        status["research_diagnostics"] = _research_diagnostics(
            registry, protocol_sha256=args.protocol_sha256
        )
    print(
        json.dumps(
            {
                "format": "klab-research-report-v1",
                "status": status,
                "next": next_experiments(registry, evidence_root=evidence_root),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _report(args: argparse.Namespace) -> None:
    """Read results through native readers without rerunning or publishing anything."""
    from klab.report_config import ReportConfig, load_report_config

    try:
        config = (
            load_report_config(args.config)
            if args.config is not None
            else ReportConfig(report_config_version=1)
        )
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit(f"klab report: invalid report config: {error}") from None
    for field in ("surface_dir", "lifecycle", "study", "protocol_sha256"):
        setattr(args, field, getattr(config, field))
    args.evidence_root = str(config.evidence_root)
    selectors = {
        "record": ("lab_dir",),
        "run": ("runs_dir", "surface_dir"),
        "research": ("lifecycle", "study", "protocol_sha256"),
    }
    for kind, fields in selectors.items():
        if kind != args.kind and any(
            getattr(args, field) is not None for field in fields
        ):
            raise SystemExit(f"klab report: {kind} selectors require --kind {kind}")
    if args.kind != "research" and "evidence_root" in config.model_fields_set:
        raise SystemExit("klab report: evidence_root requires --kind research")
    if args.kind == "record" and args.config is not None:
        raise SystemExit("klab report: --config requires --kind run or research")
    if args.kind == "research":
        if args.record is not None:
            raise SystemExit(
                "klab report: --kind research uses lifecycle in --config, not a record ID"
            )
        _report_research(args)
        return
    if args.record is None:
        raise SystemExit("klab report: a record or run ID is required")
    if args.kind == "run":
        _report_run(args)
        return
    from klab.lab_mode import lab_root, summarize
    from klab.lab_records import read_lab_record
    from klab.lab_results import resolve_record
    from klab.probes.render import render, use_color

    root = lab_root(resolve_work_dir(args.work_dir), args.lab_dir)
    try:
        target = resolve_record(args.record, root)
        kind, record = read_lab_record(target)
    except (ValueError, OSError) as error:
        _lab_lookup_failure(args, error)
        raise SystemExit(f"klab report: {error}") from None
    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    elif kind == "probe":
        print(render(record, color=use_color()))
        print(f"  record: {target}")
    else:
        print(summarize(record, target, color=use_color()))


def _compare(args: argparse.Namespace) -> None:
    """Place sealed results against each other and pinned reference models."""
    from klab.lab_mode import lab_root
    from klab.probes.compare import (
        as_json,
        compare,
        references_as_json,
        render,
        render_references,
    )
    from klab.probes.points import collect_points
    from klab.probes.render import use_color

    lab_dir = lab_root(resolve_work_dir(args.work_dir), args.lab_dir)
    try:
        if args.list_references:
            pool = collect_points(lab_dir)
            if args.json:
                print(json.dumps(references_as_json(pool), indent=2, sort_keys=True))
            else:
                print(render_references(pool, color=use_color()))
            return
        if not args.results:
            raise ValueError("name a result (try/probe id, record path or ref:NAME)")
        report = compare(
            args.results[0],
            args.results[1:],
            lab_dir=lab_dir,
            references=args.references,
            final=args.final,
        )
    except (ValueError, OSError, KeyError) as error:
        _lab_lookup_failure(args, error)
        raise SystemExit(f"klab compare: {error}") from None
    if args.json:
        print(json.dumps(as_json(report), indent=2, sort_keys=True))
    else:
        print(render(report, color=use_color()))


def _probe(args: argparse.Namespace) -> None:
    """Run the fast-fail probe battery on a run or checkpoint."""
    from klab.lab_mode import lab_root
    from klab.probes.cli import run_probe
    from klab.probes.render import render, use_color
    from klab.reference_models import ReferenceUnavailable

    try:
        result, path = run_probe(
            args.target,
            args.vs,
            lab_dir=lab_root(resolve_work_dir(args.work_dir), args.lab_dir),
            runs_dir=Path(args.runs_dir) if args.runs_dir else None,
            tier=args.tier,
            fast_fail=not args.no_fast_fail,
            backend=args.backend,
            authorization=args.runtime_authorization,
            resource_envelope=args.resource_envelope_value,
            lm_eval_tasks=args.lm_eval_tasks,
            lm_eval_limit=args.lm_eval_limit,
            lm_eval_protocol=args.lm_eval_protocol,
            final=args.final,
            request={
                **args.probe_config_loaded.model_dump(mode="json"),
                "baseline": args.vs,
                "backend_override": args.backend,
                "runtime_id": args.runtime,
                "runtime_profile_path": str(args.probe_config_loaded.runtime_profile)
                if args.probe_config_loaded.runtime_profile is not None
                else None,
                "resource_envelope_path": str(
                    args.probe_config_loaded.resource_envelope
                )
                if args.probe_config_loaded.resource_envelope is not None
                else None,
                "runtime_profile": args.runtime_profile_loaded.model_dump(mode="json")
                if args.runtime_profile_loaded is not None
                else None,
                "resource_envelope": args.resource_envelope_value.model_dump(
                    mode="json"
                )
                if args.resource_envelope_value is not None
                else None,
            },
        )
    except (ValueError, OSError, ReferenceUnavailable) as error:
        _lab_lookup_failure(args, error)
        raise SystemExit(f"klab probe: {error}") from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    if args.json:
        print(json.dumps({**result, "record": str(path)}, indent=2, sort_keys=True))
    else:
        print(render(result, color=use_color()))
        print(f"  record: {path}")


def _explore(args: argparse.Namespace) -> None:
    """Look inside a small checkpoint; cache the views for the dashboard."""
    from klab.explorer import (
        DEFAULT_TEXT,
        ExplorerUnavailable,
        explore,
        runtime_label,
    )
    from klab.lab_context import LabCancelled
    from klab.lab_mode import lab_root

    # Same runtime wiring as probe/try: _prepare_runtime_command resolved the
    # run, applied the selected runtime profile and authorized the device; the
    # envelope is checked at every explorer stage.
    try:
        result, path = explore(
            args.target,
            lab_dir=lab_root(resolve_work_dir(args.work_dir), args.lab_dir),
            runs_dir=Path(args.runs_dir) if args.runs_dir else None,
            text=args.text or DEFAULT_TEXT,
            backend=args.backend,
            authorization=args.runtime_authorization,
            resource_envelope=args.resource_envelope_value,
            refresh=args.refresh,
        )
    except (ValueError, OSError, ExplorerUnavailable) as error:
        _lab_lookup_failure(args, error)
        raise SystemExit(f"klab explore: {error}") from None
    except LabCancelled as error:
        raise SystemExit(f"klab explore: stopped: {error}") from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    if args.json:
        print(json.dumps({**result, "cache": str(path) if path else None}, indent=2))
        return
    arch = result["architecture"]
    inventory = arch["inventory"]
    print(
        f"EXPLORE  {result['target'].get('run_id')} · {inventory['total']:,} params "
        f"({inventory['active_per_token']:,} active/token) · {len(arch['layers'])} layers"
    )
    if result.get("unavailable"):
        print(f"  {result['unavailable']}")
        return
    print(f"  diagnostics runtime: {runtime_label(result)}")
    reuse = result.get("cache_reuse") or {}
    print(
        f"  cache reuse: {reuse.get('status')}"
        + (f" · {reuse['reason']}" if reuse.get("reason") else "")
    )
    tokens = [t for t in result["tokens"] if t["loss"] is not None]
    mean = sum(t["loss"] for t in tokens) / len(tokens)
    print(
        f"  {len(result['tokens'])} tokens · mean loss {mean:.3f} · "
        f"{len(result['attention']['layers'])} attention map(s) · "
        f"routing {'yes' if result.get('routing') else 'n/a'} · "
        f"memory {'yes' if result.get('memory') else 'n/a'} · {result['seconds']:.1f}s"
    )
    print(f"  cache: {path}")
    print("  open the dashboard's Explorer page to see it: klab dashboard")


# Help groups: lab-mode fast path first; release/full-provenance last. Every
# registered command appears exactly once (see tests/test_lab_mode.py).
FAST_PATH_COMMANDS = (
    "try",
    "probe",
    "report",
    "compare",
    "explore",
    "inspect",
    "train",
    "eval",
    "generate",
    "chat",
    "dashboard",
    "readiness",
)
RELEASE_COMMANDS = (
    "campaign",
    "experiment",
    "attempt",
    "iteration",
    "corpus",
    "research",
    "study",
    "review",
    "recovery",
    "archive",
    "family",
    "controller",
    "worker",
    "hosted",
    "monitor",
    "monitor-baseline",
    "runner-status",
    "lab-job",
)


_FALLBACK_HELP = {
    "inspect": "Show effective config, shape and memory estimate (read-only)",
    "train": "Train one config locally (native trainer)",
    "eval": "Score a run's checkpoint on its held-out validation split",
    "generate": "Generate a raw continuation from a verified checkpoint",
    "dashboard": "Local read-only training/research dashboard",
    "readiness": "CPU wiring smoke checks for the shipped mechanisms",
    "weights": "Import external weights with provenance",
    "config": "Derive or migrate typed run configs",
    "checkpoint": "Inspect or verify checkpoints",
    "workspace": "Storage preflight and cleanup plans",
    "runtime": "Runtime probes and status",
    "tokenizer": "Train or verify tokenizers",
    "data": "Prepare, verify and inspect training data",
    "facts": "Synthetic fact manifests and recall evaluation",
    "engram": "Export, inspect and pack Engram memory",
    "stage": "Pilot a config through staged gates",
    "batch": "Calibrate micro-batch size",
    "model": "Checkpoint-bound reference exercises",
    "evaluation": "Checkpoint-bound evaluation suites and fixed slices",
}


def _group_command_help(
    parser: argparse.ArgumentParser,
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    helps = {
        action.dest: action.help or _FALLBACK_HELP.get(action.dest, "")
        for action in commands._choices_actions
    }
    for name in commands.choices:
        helps.setdefault(name, _FALLBACK_HELP.get(name, ""))
    names = list(commands.choices)
    tools = [
        name
        for name in names
        if name not in FAST_PATH_COMMANDS and name not in RELEASE_COMMANDS
    ]
    missing = [
        name for name in (*FAST_PATH_COMMANDS, *RELEASE_COMMANDS) if name not in names
    ]
    if missing:
        raise AssertionError(f"help groups name unregistered commands: {missing}")
    width = max(len(name) for name in names) + 2

    def section(title: str, members: Sequence[str]) -> str:
        rows = [f"  {name:<{width}}{helps.get(name, '')}".rstrip() for name in members]
        return "\n".join((title, *rows))

    parser.epilog = "\n\n".join(
        (
            section(
                "Fast path (lab mode: YAML delta -> scored comparison):",
                FAST_PATH_COMMANDS,
            ),
            section("Data, models and diagnostics:", tools),
            section(
                "Advanced / release (full provenance, approvals, campaigns):",
                RELEASE_COMMANDS,
            ),
            "Start with: klab try DELTA.yaml --vs BASELINE.yaml",
        )
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    commands.metavar = "COMMAND"
    commands.help = "one of the commands grouped below"
    # The grouped epilog replaces argparse's flat, registration-ordered list.
    commands._choices_actions = []


def build_parser(work_dir: Path | None = None) -> argparse.ArgumentParser:
    runs_dir_default = str(resolve_work_dir(work_dir) / "runs")
    parser = argparse.ArgumentParser(
        prog="klab", description="Kybern Model Lab: train, compare and inspect models."
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Persistent state root (flag > KLAB_WORK_DIR > XDG data/home default)",
    )
    parser.add_argument(
        "--hf-token-file",
        type=Path,
        help="Read a Hugging Face token from a file (overrides HF_TOKEN and stored login; before the subcommand)",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    lab_try = commands.add_parser(
        "try",
        help="Lab mode: train/score a YAML delta against a baseline in one command",
        description=(
            "Derive the candidate from a lab delta (or take a full config), train "
            "both arms locally, score both on the same held-out split and write one "
            "compact record. Skips plan locks, approvals, admission reviews and "
            "campaign reconciliation; keeps storage/resource limits, data identity, "
            "safe cancellation and held-out checks."
        ),
    )
    lab_try.add_argument(
        "candidate", help="Lab delta YAML ({set: {dotted.field: value}}) or RunConfig"
    )
    lab_try.add_argument("--vs", required=True, help="Baseline RunConfig YAML")
    lab_try.add_argument(
        "--config",
        default="fast",
        metavar="PRESET_OR_YAML",
        help="Validated try policy: fast, standard, full, no-probes or YAML path",
    )
    lab_try.add_argument(
        "--backend", choices=("auto", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    lab_try.add_argument("--runtime", metavar="ID")
    lab_try.add_argument("--json", action="store_true")
    lab_try.set_defaults(handler=_try)
    lab_report = commands.add_parser(
        "report",
        help="Read sealed results, run evidence or declared research (no compute)",
    )
    lab_report.add_argument(
        "record", nargs="?", help="try/probe record; run ID with --kind run"
    )
    lab_report.add_argument(
        "--kind", choices=("record", "run", "research"), default="record"
    )
    lab_report.add_argument("--lab-dir", type=Path)
    lab_report.add_argument("--json", action="store_true")
    lab_report.add_argument("--runs-dir", help="Saved run location for --kind run")
    lab_report.add_argument(
        "--config", type=Path, help="Advanced read-only report selectors YAML"
    )
    lab_report.set_defaults(handler=_report)
    lab_compare = commands.add_parser(
        "compare",
        help="Compare sealed results with each other and with reference models",
        description=(
            "Place a try/probe result against other results and, with "
            "--references, against pinned SmolLM2/Pythia checkpoints. Each "
            "metric is compared only within one comparison group (held-out "
            "loss: same eval group; lm-eval: same benchmark group); otherwise "
            "the pair is reported NOT COMPARABLE or MISSING EVIDENCE. Reads "
            "records only; nothing is re-scored or downloaded."
        ),
    )
    lab_compare.add_argument(
        "results",
        nargs="*",
        metavar="RESULT",
        help="try/probe id, record path or ref:NAME; the first is the subject",
    )
    lab_compare.add_argument(
        "--references",
        action="store_true",
        help="Also compare with every pinned reference model result",
    )
    lab_compare.add_argument(
        "--list-references",
        action="store_true",
        help="List pinned reference models and which have results",
    )
    lab_compare.add_argument(
        "--final",
        action="store_true",
        help="Report `probe --config final` results (held-back split) against each "
        "other. Ordinary comparisons never include final evidence.",
    )
    lab_compare.add_argument("--lab-dir", type=Path)
    lab_compare.add_argument("--json", action="store_true")
    lab_compare.set_defaults(handler=_compare)
    probe = commands.add_parser(
        "probe",
        help="Fast-fail probe battery on a run or checkpoint (optionally vs a baseline)",
        description=(
            "Score a run or checkpoint with the versioned probe suite, cheapest "
            "probes first: held-out loss and calibration, top-1 agreement and "
            "divergence vs a baseline, degeneration, reworded fact recall, "
            "needle-in-context and (full tier, optional extra) lm-eval tasks. "
            "Stops early on a hard failure and recommends a next action."
        ),
    )
    probe.add_argument(
        "target",
        help="Try/probe record (its candidate), run id/directory, checkpoint, "
        "or ref:NAME for a pinned "
        "reference model (--config full; see `klab compare --list-references`)",
    )
    probe.add_argument(
        "--vs",
        metavar="BASELINE",
        help="Baseline run/checkpoint or another record's candidate",
    )
    probe.add_argument(
        "--config",
        default="fast",
        metavar="NAME_OR_YAML",
        help="Validated policy: fast (default), standard, full, final, final-full or YAML",
    )
    probe.add_argument(
        "--backend",
        choices=("auto", "mps", "cuda", "rocm", "xpu", "cpu"),
        help="Default: saved backend (CPU extra does not override it). "
        "Use --backend cpu explicitly to inspect an accelerator run on CPU.",
    )
    probe.add_argument("--runtime", metavar="ID", help="Registered native runtime")
    probe.add_argument("--json", action="store_true")
    probe.set_defaults(
        handler=_probe,
        runtime_profile=None,
        resource_envelope=None,
        lab_dir=None,
        runs_dir=None,
    )
    explore = commands.add_parser(
        "explore",
        help="Model explorer: architecture, weights, attention, routing, memory",
        description=(
            "Look inside a small checkpoint: architecture and parameter counts, "
            "per-tensor weight statistics, per-token loss and top-k predictions, "
            "attention patterns, MoE routing and memory-table lookups over a "
            "text. Results are cached under LAB/explorer for the dashboard."
        ),
    )
    explore.add_argument(
        "target",
        help="Try/probe record (its candidate), run id/directory or checkpoint",
    )
    explore.add_argument(
        "--config", type=Path, help="ExploreConfig YAML: text, refresh and lookup roots"
    )
    explore.add_argument(
        "--resource-envelope",
        type=Path,
        help="ResourceEnvelope YAML checked before every explorer stage",
    )
    explore_runtime = explore.add_mutually_exclusive_group()
    explore_runtime.add_argument("--runtime-profile", type=Path)
    explore_runtime.add_argument("--runtime", metavar="ID")
    explore.add_argument("--json", action="store_true")
    explore.set_defaults(
        handler=_explore,
        backend="cpu",
        text=None,
        refresh=False,
        lab_dir=None,
        runs_dir=None,
    )
    from klab.operational_monitor_cli import register_monitor_parser
    from klab.training.attempt_contract_cli import register_attempt_parser

    register_monitor_parser(commands)
    register_attempt_parser(commands)
    weights = commands.add_parser("weights")
    weight_commands = weights.add_subparsers(dest="weights_command", required=True)
    weight_import = weight_commands.add_parser("import")
    weight_import.add_argument("source")
    weight_import.add_argument("destination")
    weight_import.add_argument("--config", required=True)
    weight_import.add_argument(
        "--format",
        required=True,
        choices=("klab_legacy_v1", "hf_llama_safetensors"),
    )
    weight_import.add_argument("--source-tokenizer", required=True)
    weight_import.add_argument("--provenance", required=True)
    weight_import.set_defaults(handler=_weights_import)
    config = commands.add_parser("config")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    from klab.config.validation import add_arguments as validation_arguments

    validation_arguments(
        config_commands.add_parser(
            "validate",
            help="Validate native declaration schema without runtime probing",
        )
    )
    config_derive = config_commands.add_parser(
        "derive",
        help="Publish a typed config variant; validates schema compatibility, not artifacts or runtime",
    )
    config_derive.add_argument("source", help="Source v2 run configuration")
    config_derive.add_argument(
        "--set",
        dest="settings",
        action="append",
        required=True,
        metavar="FIELD=JSON",
        help="Typed JSON value at a dotted configuration field (repeatable)",
    )
    config_derive.add_argument("--output", required=True, help="New YAML output path")
    config_derive.add_argument("--json", action="store_true")
    config_derive.set_defaults(handler=_config_derive)
    from klab.training.schedule import register_parser as register_schedule

    register_schedule(config_commands)
    config_migrate = config_commands.add_parser("migrate")
    config_migrate.add_argument("input")
    config_migrate.add_argument("--output", required=True)
    config_migrate.set_defaults(handler=_config_migrate)
    checkpoint = commands.add_parser("checkpoint")
    checkpoint_commands = checkpoint.add_subparsers(
        dest="checkpoint_command", required=True
    )
    checkpoint_spot = checkpoint_commands.add_parser("plan-spot")
    checkpoint_spot.add_argument("policy", type=Path)
    checkpoint_spot.add_argument("--workspace", type=Path, required=True)
    checkpoint_spot.set_defaults(handler=_checkpoint_plan_spot)
    checkpoint_inspect = checkpoint_commands.add_parser("inspect")
    checkpoint_inspect.add_argument("path")
    checkpoint_inspect.add_argument("--json", action="store_true")
    checkpoint_inspect.set_defaults(handler=_checkpoint_inspect)
    checkpoint_verify = checkpoint_commands.add_parser("verify")
    checkpoint_verify.add_argument("path")
    checkpoint_verify.add_argument("--config")
    checkpoint_verify.add_argument("--weights-only", action="store_true")
    checkpoint_verify.add_argument("--json", action="store_true")
    checkpoint_verify.set_defaults(handler=_checkpoint_verify)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("config")
    inspect.add_argument("--json", action="store_true")
    inspect.add_argument("--write-proposal")
    inspect.add_argument(
        "--estimate-runtime",
        action="store_true",
        help="Include a read-only, compatibility-filtered optimizer runtime forecast as JSON",
    )
    inspect.set_defaults(handler=_inspect)
    workspace = commands.add_parser("workspace")
    workspace_commands = workspace.add_subparsers(
        dest="workspace_command", required=True
    )
    workspace_preflight = workspace_commands.add_parser("preflight")
    workspace_preflight.add_argument("config")
    workspace_preflight.add_argument("--tokenizer", action="store_true")
    workspace_preflight.set_defaults(handler=_workspace_preflight)
    workspace_cleanup = workspace_commands.add_parser("cleanup")
    cleanup_commands = workspace_cleanup.add_subparsers(
        dest="cleanup_command", required=True
    )
    cleanup_plan = cleanup_commands.add_parser("plan")
    cleanup_plan.add_argument("workspace")
    cleanup_plan.add_argument("--output", required=True)
    cleanup_plan.add_argument("--max-extra-periodic", type=int, default=2)
    cleanup_plan.add_argument("--max-cache-entries", type=int, default=2)
    cleanup_plan.set_defaults(handler=_workspace_cleanup_plan)
    cleanup_apply = cleanup_commands.add_parser("apply")
    cleanup_apply.add_argument("plan")
    cleanup_apply.set_defaults(handler=_workspace_cleanup_apply)
    runtime = commands.add_parser(
        "runtime",
        description="Runtime probes and status.",
        epilog="Manage machine-local environments with klab runtime env --help (Torch-free).",
    )
    runtime_commands = runtime.add_subparsers(dest="runtime_command", required=True)
    runtime_status = runtime_commands.add_parser("status")
    runtime_status.add_argument("run_id")
    runtime_status.add_argument("--runs-dir", default=runs_dir_default)
    runtime_status.add_argument("--limit", type=int, default=50)
    runtime_status.add_argument("--json", action="store_true", required=True)
    runtime_status.set_defaults(handler=_runtime_status)
    runtime_probe = runtime_commands.add_parser("probe")
    runtime_probe.add_argument("profile", help="Strict runtime profile YAML")
    runtime_probe.add_argument("--json", action="store_true")
    runtime_probe.set_defaults(handler=_runtime_probe)
    runtime_attention = runtime_commands.add_parser("attention")
    runtime_attention.add_argument("config")
    selection = runtime_attention.add_mutually_exclusive_group()
    selection.add_argument("--runtime")
    selection.add_argument("--runtime-profile", type=Path)
    runtime_attention.add_argument("--worker")
    runtime_attention.add_argument("--store", default=runs_dir_default)
    runtime_attention.add_argument("--json", action="store_true", required=True)
    runtime_attention.set_defaults(handler=_runtime_attention)
    tokenizer = commands.add_parser("tokenizer")
    tokenizer_commands = tokenizer.add_subparsers(
        dest="tokenizer_command", required=True
    )
    tokenizer_train = tokenizer_commands.add_parser("train")
    tokenizer_train.add_argument("config")
    tokenizer_train.set_defaults(handler=_tokenizer_train)
    from klab.data.tokenizer_cli import (
        register_memory_map_parser,
        register_verify_parser,
    )

    register_verify_parser(tokenizer_commands)
    register_memory_map_parser(tokenizer_commands)
    data = commands.add_parser("data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_prepare = data_commands.add_parser("prepare")
    data_prepare.add_argument("config")
    data_prepare.add_argument("--resource-envelope", type=Path)
    data_prepare.add_argument("--cold-verify", action="store_true")
    data_prepare.add_argument("--observations-output", type=Path)
    _tokenizer_batch_arguments(data_prepare)
    prepared_inputs = data_commands.add_parser(
        "prepared-inputs", help="Publish or cold-verify immutable training inputs"
    )
    prepared_commands = prepared_inputs.add_subparsers(
        dest="inputs_command", required=True
    )
    prepared_publish = prepared_commands.add_parser("publish")
    prepared_publish.add_argument("config")
    prepared_publish.add_argument("--output", required=True, type=Path)
    prepared_publish.add_argument("--resource-envelope", type=Path)
    _tokenizer_batch_arguments(prepared_publish)
    prepared_publish.set_defaults(handler=_data_prepared_inputs)
    prepared_verify = prepared_commands.add_parser("verify")
    prepared_verify.add_argument("config")
    prepared_verify.add_argument("bundle", type=Path)
    prepared_verify.set_defaults(handler=_data_prepared_inputs)
    from klab.data.preparation_benchmark import (
        register_parser as register_benchmark,
    )

    register_benchmark(data_commands)
    from klab.data.coverage import register_parser as register_coverage
    from klab.data.legacy import register_parser as register_data_migration
    from klab.data.source_cli import register_parser as register_dataset_sources

    register_dataset_sources(data_commands)
    register_coverage(data_commands)
    register_data_migration(data_commands)
    data_bakeoff = data_commands.add_parser("bakeoff")
    data_bakeoff.add_argument("config")
    data_bakeoff.add_argument("output", type=Path)
    data_bakeoff.set_defaults(handler=_data_bakeoff)
    facts = commands.add_parser("facts")
    fact_commands = facts.add_subparsers(dest="fact_command", required=True)
    manifest = fact_commands.add_parser("manifest")
    manifest.add_argument("--seed", type=int, default=0)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(handler=_facts_manifest)
    verify = fact_commands.add_parser("verify")
    verify.add_argument("path")
    verify.set_defaults(handler=_facts_verify)
    audit = fact_commands.add_parser("audit")
    audit.add_argument("path")
    audit.set_defaults(handler=_facts_audit)
    facts_evaluate = fact_commands.add_parser("evaluate")
    facts_evaluate.add_argument("run_id")
    facts_evaluate.add_argument("manifest")
    facts_evaluate.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: selected persistent state root/runs)",
    )
    facts_evaluate.add_argument("--checkpoint")
    facts_evaluate.add_argument(
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    facts_evaluate.add_argument("--max-new-tokens", type=int, default=16)
    facts_evaluate.add_argument("--output")
    facts_evaluate.set_defaults(handler=_facts_evaluate)
    transfer_evaluate = fact_commands.add_parser("transfer-evaluate")
    transfer_evaluate.add_argument("source_run_id")
    transfer_evaluate.add_argument("target_run_id")
    transfer_evaluate.add_argument("manifest")
    transfer_evaluate.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: selected persistent state root/runs)",
    )
    transfer_evaluate.add_argument("--source-checkpoint")
    transfer_evaluate.add_argument("--target-checkpoint")
    transfer_evaluate.add_argument(
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    transfer_evaluate.add_argument("--max-new-tokens", type=int, default=16)
    transfer_evaluate.add_argument("--output")
    transfer_evaluate.set_defaults(handler=_facts_transfer_evaluate)
    data_prepare.set_defaults(handler=_data_prepare)
    engram = commands.add_parser("engram")
    engram_commands = engram.add_subparsers(dest="engram_command", required=True)
    engram_export = engram_commands.add_parser("export")
    engram_export.add_argument("run_id")
    engram_export.add_argument("--output", required=True)
    engram_export.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: selected persistent state root/runs)",
    )
    engram_export.add_argument("--checkpoint")
    engram_export.add_argument(
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    engram_export.set_defaults(handler=_engram_export)
    engram_inspect = engram_commands.add_parser("inspect")
    engram_inspect.add_argument("path")
    engram_inspect.set_defaults(handler=_engram_inspect)
    pack = engram_commands.add_parser("pack")
    pack_commands = pack.add_subparsers(dest="engram_pack_command", required=True)
    pack_compile = pack_commands.add_parser("compile")
    pack_compile.add_argument("source")
    pack_compile.add_argument("--output", required=True)
    pack_compile.add_argument("--name", required=True)
    pack_compile.add_argument("--namespace", required=True)
    pack_compile.add_argument("--license")
    pack_compile.add_argument("--source-name")
    pack_compile.add_argument("--source-revision")
    pack_compile.add_argument("--created-at")
    pack_compile.add_argument("--lexical-package")
    pack_compile.add_argument("--semantic-keys")
    pack_compile.add_argument("--semantic-values")
    pack_compile.add_argument("--semantic-metadata")
    pack_compile.set_defaults(handler=_engram_pack_compile)
    pack_inspect = pack_commands.add_parser("inspect")
    pack_inspect.add_argument("path")
    pack_inspect.set_defaults(handler=_engram_pack_inspect)
    pack_verify = pack_commands.add_parser("verify")
    pack_verify.add_argument("path")
    pack_verify.add_argument("--expected-pack-id")
    pack_verify.set_defaults(handler=_engram_pack_verify)
    staging = commands.add_parser("stage")
    staging.add_argument("config")
    staging.add_argument(
        "--through", default="smoke", choices=("inspect", "validate", "smoke", "warmup")
    )
    staging.add_argument("--output", required=True)
    staging.add_argument("--observations-output", type=Path)
    staging.add_argument(
        "--prepared-inputs",
        type=Path,
        help="Verify and stage an existing materialized input bundle without preparing data",
    )
    staging.add_argument(
        "--cold-verify",
        action="store_true",
        help="Independently rehash and verify every staged input without proof reuse",
    )
    stage_runtime = staging.add_mutually_exclusive_group()
    stage_runtime.add_argument("--runtime-profile", type=Path)
    stage_runtime.add_argument("--runtime", metavar="ID")
    staging.add_argument("--resource-envelope", type=Path)
    staging.add_argument(
        "--pilot-deadline-policy",
        type=Path,
        help="Versioned operational initialization, no-progress and absolute pilot limits",
    )
    _tokenizer_batch_arguments(staging)
    staging.set_defaults(handler=_stage)
    batch = commands.add_parser("batch")
    batch_commands = batch.add_subparsers(dest="batch_command", required=True)
    batch_calibrate = batch_commands.add_parser("calibrate")
    batch_calibrate.add_argument("config")
    batch_calibrate.add_argument("--output", required=True)
    batch_calibrate.add_argument("--max-candidates", type=int, default=8)
    batch_calibrate.set_defaults(handler=_batch_calibrate)
    readiness = commands.add_parser("readiness")
    readiness_commands = readiness.add_subparsers(
        dest="readiness_command", required=True
    )
    readiness_smoke = readiness_commands.add_parser("smoke")
    readiness_smoke.add_argument("--configs-root", default="configs")
    readiness_smoke.add_argument("--output", required=True)
    readiness_smoke.add_argument(
        "--family", action="append", choices=tuple(SMOKE_FAMILIES), default=[]
    )
    readiness_smoke.add_argument(
        "--command-timeout",
        type=float,
        default=3600.0,
        metavar="SECONDS",
        help="Per-command bound for slow hosts (default: 3600; 0 disables)",
    )
    readiness_smoke.set_defaults(handler=_readiness_smoke)
    from klab.evaluation.cli import register_readiness_parser

    register_readiness_parser(readiness_commands)
    training = commands.add_parser("train")
    training.add_argument("config")
    training.add_argument(
        "--backend", choices=("auto", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    training.add_argument(
        "--runs-dir",
        help="Explicit run store override; otherwise preserve logging.root_dir from the config",
    )
    training.add_argument("--run-id")
    training.add_argument("--resume")
    training.add_argument(
        "--spot-policy", type=Path, help="Separate operational spot-safety JSON policy"
    )
    training.add_argument("--extend-budget")
    training.add_argument(
        "--append-data",
        action="store_true",
        help="verified append-only document data; requires --extend-budget",
    )
    training.add_argument(
        "--append-from-checkpoint",
        action="store_true",
        help="explicit data phase after a complete checkpoint update; preserve consumed prefix and defer untrained source tail",
    )
    training.add_argument("--promote")
    training.add_argument(
        "--add-ngram-memory",
        action="store_true",
        help="explicit zero-impact dense-checkpoint migration; requires --promote",
    )
    training.add_argument(
        "--extend-context",
        action="store_true",
        help="explicit larger unscaled RoPE context; requires --promote --add-ngram-memory",
    )
    training.add_argument("--recover")
    training.add_argument("--allow-runtime-drift", action="store_true")
    training.add_argument(
        "--allow-runtime-optimization",
        action="store_true",
        help="Explicitly resume full state with supported runtime changes; numerical trajectory may differ",
    )
    training.add_argument("--stop-after-step", type=int)
    training.add_argument("--stage-bundle")
    training.add_argument(
        "--require-prepared",
        action="store_true",
        help="never tokenize or pack in this process: fail before the model is "
        "built unless `data prepare` already published this config's cache",
    )
    train_runtime = training.add_mutually_exclusive_group()
    train_runtime.add_argument("--runtime-profile", type=Path)
    train_runtime.add_argument("--runtime", metavar="ID")
    training.add_argument("--resource-envelope", type=Path)
    _tokenizer_batch_arguments(training)
    training.set_defaults(handler=_train)
    model = commands.add_parser("model")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    model_exercise = model_commands.add_parser(
        "exercise", help="Run the checkpoint-bound dense-LM reference exercise suite."
    )
    model_exercise.add_argument("run_id")
    model_exercise.add_argument("--checkpoint", required=True)
    model_exercise.add_argument("--runs-dir", default=runs_dir_default)
    model_exercise.add_argument(
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    model_exercise.add_argument("--prompt-panel")
    model_exercise.add_argument(
        "--output",
        help="Observation file/directory (default: exercises/ beside the run store)",
    )
    model_exercise.set_defaults(handler=_model_exercise)

    evaluation = commands.add_parser("eval")
    evaluation.add_argument("run_id")
    evaluation.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: selected persistent state root/runs)",
    )
    evaluation.add_argument("--checkpoint")
    evaluation.add_argument(
        "--panel", type=Path, help="Pinned development heldout panel JSON"
    )
    evaluation.add_argument(
        "--output", type=Path, help="New panel result file outside input roots"
    )
    evaluation.add_argument(
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    eval_runtime = evaluation.add_mutually_exclusive_group()
    eval_runtime.add_argument("--runtime-profile", type=Path)
    eval_runtime.add_argument("--runtime", metavar="ID")
    evaluation.set_defaults(handler=_eval)
    capability = commands.add_parser(
        "capability",
        help="Evaluate or compare a versioned narrow capability card.",
    )
    capability_commands = capability.add_subparsers(
        dest="capability_command", required=True
    )
    capability_list = capability_commands.add_parser("list")
    capability_list.set_defaults(handler=_capability_list)
    capability_suite = capability_commands.add_parser(
        "suite", help="List separated Phase E outcome categories and frozen cards."
    )
    capability_suite.set_defaults(handler=_capability_suite)
    capability_describe = capability_commands.add_parser("describe")
    capability_describe.add_argument(
        "card", help="Built-in card name or declarative JSON file"
    )
    capability_describe.set_defaults(handler=_capability_describe)
    code_work = capability_commands.add_parser(
        "code-work",
        help="Score native generations with a pinned optional Inspect Docker sandbox.",
    )
    code_work.add_argument("run_id")
    code_work.add_argument("declaration")
    code_work.add_argument("--output", required=True)
    code_work.add_argument("--runs-dir", default=runs_dir_default)
    code_work.add_argument("--checkpoint", required=True)
    code_work.add_argument("--backend", choices=("cpu", "mps", "cuda", "rocm", "xpu"))
    code_work.set_defaults(handler=_code_work_evaluate)
    capability_evaluate = capability_commands.add_parser("evaluate")
    capability_evaluate.add_argument("run_id")
    capability_evaluate.add_argument("card")
    capability_evaluate.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: selected persistent state root/runs)",
    )
    capability_evaluate.add_argument(
        "--checkpoint", help="latest.json, best.json, or a generation path"
    )
    capability_evaluate.add_argument(
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    capability_evaluate.set_defaults(handler=_capability_evaluate)
    capability_compare = capability_commands.add_parser("compare")
    capability_compare.add_argument("base_run_id")
    capability_compare.add_argument("variant_run_id")
    capability_compare.add_argument("card")
    capability_compare.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: selected persistent state root/runs)",
    )
    capability_compare.add_argument("--base-checkpoint")
    capability_compare.add_argument("--variant-checkpoint")
    capability_compare.add_argument(
        "--vary",
        choices=("memory", "attention", "ffn", "scale", "none"),
        default="memory",
    )
    capability_compare.add_argument(
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    capability_compare.set_defaults(handler=_capability_compare)

    study = commands.add_parser(
        "study", help="Plan and compare matched architecture experiments."
    )
    study_commands = study.add_subparsers(dest="study_command", required=True)
    study_plan = study_commands.add_parser("plan")
    study_plan.add_argument("path", help="Architecture study YAML")
    study_plan.add_argument("--max-runs", type=int, default=1000)
    study_plan.add_argument(
        "--lifecycle",
        help="Explicit registry JSON; default is an empty current registry",
    )
    study_plan.add_argument("--evidence-root", default=".")
    study_plan.set_defaults(handler=_study_plan)
    study_submit = study_commands.add_parser("submit")
    study_submit.add_argument("path", help="Architecture study YAML")
    study_submit.add_argument("--max-runs", type=int, default=1000)
    study_submit.add_argument(
        "--receipt", help="Defaults to the experiment workspace receipt.json"
    )
    study_submit.add_argument("--worker")
    study_submit.add_argument("--stage-bundle")
    study_submit.add_argument(
        "--store",
        help="Defaults to selected persistent root/experiments/<study-name>/runs",
    )
    study_submit.add_argument(
        "--lifecycle",
        help="Explicit registry JSON; default is an empty current registry",
    )
    study_submit.add_argument("--evidence-root", default=".")
    study_submit.set_defaults(handler=_study_submit)
    study_recover = study_commands.add_parser(
        "recover",
        help="Recover a receipt from bound native submission evidence; no dispatch.",
    )
    study_recover.add_argument("path", help="Original architecture study YAML")
    study_recover.add_argument("--intent", required=True)
    study_recover.add_argument("--store", required=True)
    study_recover.add_argument(
        "--output", required=True, help="Distinct unused recovery receipt path"
    )
    study_recover.add_argument("--max-runs", type=int, default=1000)
    study_recover.set_defaults(handler=_study_recover)
    study_collect = study_commands.add_parser("collect")
    study_collect.add_argument("path", help="Architecture study YAML")
    study_collect.add_argument("receipt")
    study_collect.add_argument(
        "--runs-dir", help="Defaults to runs/ beside the study receipt"
    )
    study_collect.add_argument("--checkpoint")
    study_collect.add_argument(
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    study_collect.add_argument("--max-runs", type=int, default=1000)
    study_collect.set_defaults(handler=_study_collect)
    study_report = study_commands.add_parser("report")
    study_report.add_argument("path", help="Architecture study YAML")
    study_report.add_argument("receipt")
    study_report.add_argument(
        "--evidence", required=True, help="One explicit collected report JSON"
    )
    study_report.add_argument("--output", required=True)
    study_report.add_argument("--research")
    study_report.add_argument("--runs-dir")
    study_report.add_argument("--portability-evidence")
    study_report.add_argument(
        "--identity-expectations",
        help="Fixed metadata identity assertions; no execution",
    )
    study_report.add_argument(
        "--candidate-commit",
        help="Exact verifier checkout commit, required with expectations",
    )
    study_report.set_defaults(handler=_study_report)

    reference = commands.add_parser(
        "reference",
        help="Run explicitly pinned, observation-only external model adapters.",
    )
    reference_commands = reference.add_subparsers(
        dest="reference_command", required=True
    )
    pythia = reference_commands.add_parser(
        "pythia",
        help="Observe registered Pythia checkpoints on pinned FineWeb-Edu validation.",
    )
    pythia_commands = pythia.add_subparsers(dest="pythia_command", required=True)
    trajectory = pythia_commands.add_parser("trajectory")
    trajectory.add_argument("config")
    trajectory.add_argument("--output", required=True)
    trajectory.add_argument("--cache-dir")
    trajectory.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    trajectory.add_argument(
        "--steps",
        nargs="+",
        choices=("step0", "step10000", "step143000"),
        default=("step0", "step10000", "step143000"),
    )
    trajectory.set_defaults(handler=_reference_pythia_trajectory)

    research = commands.add_parser(
        "research",
        help="Discover studies, build Phase E tasks, and mine training-only corpus statistics.",
    )
    research_commands = research.add_subparsers(dest="research_command", required=True)
    from klab.recovery.evidence import (
        register_research_evidence_parser as register_evidence_parser,
    )
    from klab.recovery.snapshot import register_parser as register_snapshot_parser
    from klab.research.lint import register_parser as register_lint_parser

    register_lint_parser(research_commands)
    register_snapshot_parser(research_commands)
    register_evidence_parser(research_commands)
    research_list = research_commands.add_parser("list")
    research_list.add_argument("--json", action="store_true")
    research_list.set_defaults(handler=_research_list)
    research_describe = research_commands.add_parser("describe")
    research_describe.add_argument(
        "reference", help="Packaged research ID or explicit JSON fork"
    )
    research_describe.add_argument("--json", action="store_true")
    research_describe.set_defaults(handler=_research_describe)
    research_scaffold = research_commands.add_parser("scaffold")
    research_scaffold.add_argument(
        "reference", help="Packaged research ID or explicit JSON fork"
    )
    research_scaffold.add_argument(
        "--resource-root",
        type=Path,
        help="Complete local authoring resource directory; defaults to packaged resources",
    )
    research_scaffold.add_argument("--output", required=True)
    research_scaffold.add_argument(
        "--scale", choices=("smoke", "nano", "micro", "tiny"), default="micro"
    )
    research_scaffold.add_argument(
        "--data",
        default="offline",
        help="Dataset profile name from the selected resource directory",
    )
    research_scaffold.add_argument(
        "--backend", choices=("cpu", "mps", "cuda", "rocm", "xpu"), default="cpu"
    )
    research_scaffold.add_argument(
        "--design",
        default="default",
        help="Design name from the selected native recipe",
    )
    research_scaffold.add_argument(
        "--lifecycle",
        help="Explicit registry JSON; otherwise resource-root lifecycle.json or empty packaged registry",
    )
    research_scaffold.add_argument("--evidence-root", default=".")
    research_scaffold.set_defaults(handler=_research_scaffold)

    def lifecycle_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--lifecycle",
            help="Explicit registry JSON; default is an empty current registry",
        )
        command.add_argument("--evidence-root", default=".")
        command.add_argument("--json", action="store_true")

    baseline = research_commands.add_parser("baseline")
    baseline_commands = baseline.add_subparsers(dest="baseline_command", required=True)
    baseline_list = baseline_commands.add_parser("list")
    lifecycle_arguments(baseline_list)
    baseline_list.set_defaults(handler=_research_baseline_list)
    baseline_describe = baseline_commands.add_parser("describe")
    baseline_describe.add_argument("identifier")
    lifecycle_arguments(baseline_describe)
    baseline_describe.set_defaults(handler=_research_baseline_describe)
    research_validate = research_commands.add_parser("validate")
    lifecycle_arguments(research_validate)
    research_validate.add_argument(
        "--baseline",
        help="validate one established baseline independently of global registry health",
    )
    research_validate.set_defaults(handler=_research_validate)
    research_tasks = research_commands.add_parser(
        "tasks", help="Build provenance-bound train and held-out task artifacts."
    )
    task_commands = research_tasks.add_subparsers(dest="task_command", required=True)
    task_build = task_commands.add_parser("build")
    task_build.add_argument(
        "task",
        choices=(
            "math-identities",
            "python-stdlib",
            "wikidata-mini",
            "memory-allocation",
        ),
    )
    task_build.add_argument("--output", required=True)
    task_build.add_argument("--seed", type=int, default=17)
    task_build.add_argument(
        "--tokenizer", help="required for memory-allocation; no packaged default"
    )
    task_build.add_argument(
        "--train-jsonl", help="required for memory-allocation; no packaged default"
    )
    task_build.add_argument(
        "--validation-jsonl", help="required for memory-allocation; no packaged default"
    )
    task_build.add_argument(
        "--oracle-audit", help="required for memory-allocation; no packaged default"
    )
    task_build.add_argument(
        "--provenance", help="required for memory-allocation; no packaged default"
    )
    task_build.add_argument(
        "--cards-dir", help="required for memory-allocation; no packaged default"
    )
    task_build.set_defaults(handler=_build_phase_e_tasks)
    research_corpus = research_commands.add_parser(
        "corpus",
        help="Measure token statistics on one explicitly supplied train split.",
    )
    corpus_commands = research_corpus.add_subparsers(
        dest="corpus_command", required=True
    )
    corpus_mine = corpus_commands.add_parser("mine")
    corpus_source = corpus_mine.add_mutually_exclusive_group(required=True)
    corpus_source.add_argument("--train-jsonl")
    corpus_source.add_argument("--prepared-train")
    corpus_mine.add_argument("--sequence-length", type=int)
    corpus_mine.add_argument("--max-blocks", type=int)
    corpus_mine.add_argument("--tokenizer")
    corpus_mine.add_argument("--model-config")
    corpus_mine.add_argument("--memory-block", type=int)
    corpus_mine.add_argument("--table-size", type=int)
    corpus_mine.add_argument("--memory-dim", type=int)
    corpus_mine.add_argument("--ngram-orders", type=int, nargs="+")
    corpus_mine.add_argument("--hash-heads", type=int)
    corpus_mine.add_argument("--output", required=True)
    corpus_mine.set_defaults(handler=_research_corpus_mine)
    portability = research_commands.add_parser(
        "portability", help="Build and execute controlled Engram portability campaigns."
    )
    portability_commands = portability.add_subparsers(
        dest="portability_command", required=True
    )
    portability_build = portability_commands.add_parser("build")
    portability_build.add_argument("--output", required=True)
    portability_build.add_argument(
        "--experiment",
        choices=("engram-portability-v1", "learned-engram-portability-v1"),
    )
    portability_build.add_argument(
        "--scale", choices=("smoke", "nano", "micro", "tiny"), default="smoke"
    )
    portability_build.add_argument(
        "--backend", choices=("auto", "cpu", "mps"), default="auto"
    )
    portability_build.add_argument("--seed", type=int, default=20260925)
    portability_build.add_argument("--updates", type=int)
    portability_build.set_defaults(handler=_research_portability_build)
    portability_plan = portability_commands.add_parser("plan")
    portability_plan.add_argument("--campaign-root", required=True)
    portability_plan.set_defaults(handler=_research_portability_plan)
    portability_run = portability_commands.add_parser("run")
    portability_run.add_argument("--campaign-root", required=True)
    portability_run.add_argument("--recipient", choices=("width32", "width64"))
    portability_run.add_argument(
        "--representation", choices=("token", "byte", "semantic")
    )
    portability_run.add_argument(
        "--condition",
        choices=(
            "native",
            "adapter-tuned",
            "frozen-only",
            "joint",
            "random",
            "corrupt",
            "disabled",
        ),
    )
    portability_run.add_argument("--seed", type=int, choices=(17, 41, 73))
    portability_run.add_argument("--max-wall-seconds", type=float)
    portability_run.set_defaults(handler=_research_portability_run)
    portability_continue = portability_commands.add_parser("continue")
    portability_continue.add_argument("--campaign-root", required=True)
    portability_continue.add_argument("--max-arms", type=int)
    portability_continue.add_argument("--max-wall-seconds", type=float)
    portability_continue.set_defaults(handler=_research_portability_continue)
    portability_report = portability_commands.add_parser("report")
    portability_report.add_argument("--campaign-root", required=True)
    portability_report.set_defaults(handler=_research_portability_report)
    review = commands.add_parser(
        "review", help="Create blinded human-review bundles and validate judgments."
    )
    review_commands = review.add_subparsers(dest="review_command", required=True)
    review_bundle = review_commands.add_parser("bundle")
    source = review_bundle.add_mutually_exclusive_group(required=True)
    source.add_argument("--records", help="Paired candidate record JSON")
    source.add_argument("--base-result", help="First capability_result_v2 JSON")
    review_bundle.add_argument(
        "--variant-result", help="Second capability_result_v2 JSON, paired by case ID"
    )
    review_bundle.add_argument("--criteria", required=True)
    review_bundle.add_argument("--seed", type=int, required=True)
    review_bundle.add_argument("--bundle", required=True)
    review_bundle.add_argument("--reveal-map", required=True)
    review_bundle.set_defaults(handler=_review_bundle)
    review_validate = review_commands.add_parser("validate")
    review_validate.add_argument("--bundle", required=True)
    review_validate.add_argument("--judgments", required=True)
    review_validate.add_argument("--output", required=True)
    review_validate.set_defaults(handler=_review_validate)

    learn = commands.add_parser(
        "learn", help="Learn and probe one mechanism without a campaign."
    )
    learn_commands = learn.add_subparsers(dest="learn_command", required=True)
    learn_list = learn_commands.add_parser("list")
    learn_list.add_argument("--json", action="store_true")
    learn_list.set_defaults(handler=_learn_list)
    learn_describe = learn_commands.add_parser("describe")
    learn_describe.add_argument("identifier")
    learn_describe.add_argument("--json", action="store_true")
    learn_describe.set_defaults(handler=_learn_describe)
    learn_scaffold = learn_commands.add_parser("scaffold")
    learn_scaffold.add_argument("identifier")
    learn_scaffold.add_argument(
        "--resource-root",
        type=Path,
        help="Complete local authoring resource directory; defaults to packaged resources",
    )
    learn_scaffold.add_argument("--output", required=True)
    learn_scaffold.add_argument(
        "--scale", choices=("smoke", "nano", "micro", "tiny"), default="smoke"
    )
    learn_scaffold.add_argument(
        "--data",
        default="offline",
        help="Dataset profile name from the selected resource directory",
    )
    learn_scaffold.add_argument(
        "--backend", choices=("cpu", "mps", "cuda", "rocm", "xpu"), default="cpu"
    )
    learn_scaffold.add_argument("--memory-package")
    learn_scaffold.set_defaults(handler=_learn_scaffold)
    learn_probe = learn_commands.add_parser("probe")
    learn_probe.add_argument("config")
    learn_probe.add_argument("--prompt", required=True)
    learn_probe.add_argument("--json", action="store_true")
    learn_probe.set_defaults(handler=_learn_probe)

    from klab.cli.serve import add_command as add_serve_command

    add_serve_command(commands, runs_dir_default)
    generation = commands.add_parser("generate")
    generation.add_argument("run_id")
    generation.add_argument("--prompt", required=True)
    generation.add_argument("--max-new-tokens", type=int, default=64)
    generation.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: selected persistent state root/runs)",
    )
    generation.add_argument("--checkpoint")
    generation.add_argument("--temperature", type=float, default=0.0)
    generation.add_argument("--top-k", type=int, default=0)
    generation.add_argument("--seed", type=int, default=0)
    generation.add_argument(
        "--stop",
        action="append",
        default=[],
        help="Literal stop string; repeat for multiple stops",
    )
    generation.add_argument(
        "--no-cache", action="store_true", help="Disable request-local KV caching"
    )
    generation.add_argument(
        "--context-length",
        type=int,
        help="Context cap, at most the checkpoint native context",
    )
    generation.add_argument(
        "--prompt-lookup",
        type=int,
        metavar="N",
        help="Draft tokens from earlier N-gram repeats in the context and verify "
        "them in one model call; greedy output is unchanged",
    )
    generation.add_argument(
        "--draft-tokens",
        type=int,
        default=8,
        help="Most tokens drafted per model call with --prompt-lookup",
    )
    generation.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the exact serialized prompt as JSON to stderr",
    )
    generation.add_argument(
        "--json",
        action="store_true",
        help="Emit identity, prompt, policy and completion JSON",
    )
    generation.add_argument(
        "--strict-context",
        action="store_true",
        help="Reject prompt plus completion beyond context; default uses a sliding window",
    )
    generation.add_argument(
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    generation_runtime = generation.add_mutually_exclusive_group()
    generation_runtime.add_argument("--runtime-profile", type=Path)
    generation_runtime.add_argument("--runtime", metavar="ID")
    generation.set_defaults(handler=_generate)
    chat = commands.add_parser(
        "chat", help="Chat with a verified local PyTorch or native MLX checkpoint."
    )
    chat.add_argument("run_id")
    chat.add_argument("--message")
    chat.add_argument("--system")
    chat.add_argument("--max-new-tokens", type=int, default=16)
    chat.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: selected persistent state root/runs)",
    )
    chat.add_argument(
        "--checkpoint", help="latest.json, best.json, or a generation path"
    )
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.add_argument("--top-k", type=int, default=0)
    chat.add_argument("--seed", type=int, default=0)
    chat.add_argument(
        "--stop",
        action="append",
        default=[],
        help="Literal stop string; repeat for multiple stops",
    )
    chat.add_argument(
        "--no-cache", action="store_true", help="Disable request-local KV caching"
    )
    chat.add_argument(
        "--context-length",
        type=int,
        help="Context cap, at most the checkpoint native context",
    )
    chat.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the exact serialized prompt as JSON to stderr",
    )
    chat.add_argument(
        "--transcript",
        help="Save a checkpoint-bound JSON transcript without overwriting",
    )
    chat.add_argument(
        "--json", action="store_true", help="Emit one JSON object per response"
    )
    chat.add_argument(
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    chat_runtime = chat.add_mutually_exclusive_group()
    chat_runtime.add_argument("--runtime-profile", type=Path)
    chat_runtime.add_argument("--runtime", metavar="ID")
    chat.set_defaults(handler=_chat)
    dashboard = commands.add_parser("dashboard")
    dashboard.add_argument(
        "--runs-dir",
        default=runs_dir_default,
        help="Run directory (default: selected persistent state root/runs)",
    )
    dashboard.add_argument(
        "--run-store",
        action="append",
        type=parse_run_store,
        default=[],
        metavar="LABEL=DIRECTORY",
        help="Additional read-only run store shown in the dashboard selector (repeatable)",
    )
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.add_argument(
        "--address",
        default="127.0.0.1",
        help="Bind address; use 0.0.0.0 only with deliberate network isolation",
    )
    dashboard.add_argument("--reports-dir", default="artifacts/research-reports")
    dashboard.add_argument(
        "--lifecycle",
        help="Explicit registry JSON; default is an empty current registry",
    )
    dashboard.add_argument("--evidence-root", default=".")
    dashboard.add_argument(
        "--surface-dir", help="Optional read-only Surface Review overlay"
    )
    dashboard.add_argument(
        "--lab-dir",
        type=Path,
        help="Lab root for the Probes page (default: WORK_DIR/lab)",
    )
    dashboard.set_defaults(handler=_dashboard)
    from klab.iteration_cli import register_parser as register_iteration_parser

    register_iteration_parser(commands)
    from klab.corpus.cli import add_commands as add_corpus_commands
    from klab.corpus.memorization_cli import (
        register_parser as register_memorization,
    )
    from klab.engram.semantic_cli import register_parser as register_semantic

    register_memorization(commands)
    register_semantic(commands)
    add_corpus_commands(commands)
    from klab.campaign.cli import add_commands as add_campaign_commands

    add_campaign_commands(commands)
    from klab.cli.surface import add_commands as add_surface_commands

    add_surface_commands(commands)
    from klab.workers.cli import add_commands

    add_commands(commands, default_store=Path(runs_dir_default))
    from klab.hosted.cli import add_parser as add_hosted_parser

    add_hosted_parser(commands)
    from klab.archive import register_parser as register_archive_parser
    from klab.evaluation.cli import register_evaluation_parser
    from klab.family.cli import register_parser as register_family_parser
    from klab.recovery.cli import register_parser as register_recovery_parser

    register_recovery_parser(commands)
    register_evaluation_parser(commands, runs_dir_default=runs_dir_default)
    register_family_parser(commands)
    register_archive_parser(commands)
    _group_command_help(parser, commands)
    return parser


_PROFILE_EXEC_MARKER = "_KLAB_RUNTIME_PROFILE_EXEC"


def _ensure_runtime_interpreter(python: Path, digest: str, probe: Any) -> None:
    """Validate an external interpreter once, then verify it again in its child."""
    expected = Path(os.path.abspath(python))
    marker = os.environ.get(_PROFILE_EXEC_MARKER)
    if marker is not None:
        previous, separator, nonce = marker.partition(":")
        if not separator or not nonce or previous != digest:
            raise ValueError("runtime interpreter re-exec marker mismatch")
        if (
            Path(os.path.abspath(sys.executable)) != expected
            or Path(sys.executable).resolve() != expected.resolve()
        ):
            raise ValueError("runtime child interpreter differs from selected python")
    elif Path(os.path.abspath(sys.executable)) != expected:
        probe()
        environment = os.environ.copy()
        environment[_PROFILE_EXEC_MARKER] = f"{digest}:{secrets.token_hex(16)}"
        os.execve(
            str(python),
            [
                str(python),
                "-c",
                "from klab.cli.main import main; main()",
                *sys.argv[1:],
            ],
            environment,
        )
        raise AssertionError("os.execve unexpectedly returned")
    probe()


def _registered_local_worker(
    store: Path, name: str, *, optional: bool = False
) -> Any | None:
    from klab.workers.controller import Controller
    from klab.workers.models import WorkerDefinition

    if optional and not (store / "experiments.sqlite3").is_file():
        return None
    controller = Controller(store, read_only=True)
    definitions = [
        WorkerDefinition.model_validate(record["definition"])
        for record in controller.store.worker_records()
        if record["definition"]["name"] == name
    ]
    if len(definitions) != 1:
        if optional and not definitions:
            return None
        raise ValueError(f"registered local worker {name} is unavailable or ambiguous")
    definition = definitions[0]
    if definition.transport != "local":
        if optional:
            return None
        raise ValueError("remote worker cannot run checkpoint-bound local evaluation")
    return definition


def _prepare_local_worker(args: argparse.Namespace, definition: Any) -> None:
    from klab.runtime_profile import RuntimeProfile

    profile = RuntimeProfile.model_validate(
        {
            "runtime_profile_version": 1,
            "id": definition.worker_id,
            "python": definition.python,
            "engine": definition.engine,
            "backend": definition.backend,
            "device_index": definition.device_index,
        }
    )
    digest = hashlib.sha256(
        json.dumps(definition.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    _ensure_runtime_interpreter(
        definition.python, digest, lambda: probe_runtime_profile(profile)
    )
    args.runtime_worker_definition = definition


def _campaign_evaluation_workers(args: argparse.Namespace) -> list[Any]:
    from klab.campaign.engine import CampaignEngine
    from klab.campaign.plan import (
        Evaluation,
        RuntimeAcceptance,
        safe_path,
    )
    from klab.experiments.plan import load_plan

    source = Path(args.source).absolute()
    engine = CampaignEngine(source, resolve_work_dir(args.work_dir))
    stages = {stage.id: stage for stage in engine.plan.stages}
    actionable = engine.inspect("next")["next_action"].get("stage")
    stage = stages.get(actionable)
    if (
        not isinstance(stage, Evaluation)
        or stage.runtime is None
        or stage.backend == "cpu"
    ):
        return []
    runtime_stage = stages[stage.runtime]
    if not isinstance(runtime_stage, RuntimeAcceptance) or runtime_stage.worker is None:
        return []
    plan_stage = stages[runtime_stage.plan]
    locked_plan = load_plan(safe_path(source.parent, plan_stage.source))
    store = (
        engine.store.root
        / "experiments"
        / f"{plan_stage.id}-{locked_plan.id}"
        / "controller"
    )
    definition = _registered_local_worker(store, runtime_stage.worker, optional=True)
    return [definition] if definition is not None else []


def _is_config_submission(args: argparse.Namespace) -> bool:
    return (
        args.command == "experiment"
        and args.experiment_command == "submit"
        and args.config is not None
    )


def _prepare_runtime_command(args: argparse.Namespace) -> None:
    """Gate direct execution before the CLI creates its scratch directory."""
    if args.command == "try":
        from klab.try_config import prepare_try_config

        prepare_try_config(args)
    if args.command == "explore":
        from klab.explore_config import prepare_explore_config

        prepare_explore_config(args)
    if args.command == "probe":
        from klab.probes.config import prepare_probe_config

        prepare_probe_config(args)
    args.runtime_authorization = None
    args.runtime_profile_loaded = None
    is_submission = _is_config_submission(args)
    is_legacy = args.command in {
        "try",
        "probe",
        "explore",
        "train",
        "stage",
        "eval",
        "generate",
        "chat",
        "serve",
    }
    is_experiment = args.command == "experiment" and args.experiment_command in {
        "bind",
        "run",
    }
    is_campaign = args.command == "campaign" and args.campaign_command in {
        "apply",
        "resume",
    }
    is_snapshot = args.command == "research" and args.research_command == "snapshot"
    is_suite = (
        args.command == "evaluation"
        and args.evaluation_command == "suite"
        and args.suite_command == "run"
    )
    is_fixed_slices = (
        args.command == "evaluation"
        and args.evaluation_command == "fixed-slices"
        and args.fixed_command in {"score", "continuations"}
    )
    if not (
        is_legacy
        or is_submission
        or is_experiment
        or is_campaign
        or is_snapshot
        or is_suite
        or is_fixed_slices
    ):
        return
    from klab.runtime_environments import profile_for_id

    runtime_id = getattr(args, "runtime", None)
    profile_path = getattr(args, "runtime_profile", None)
    if runtime_id is not None and profile_path is not None:
        raise ValueError("--runtime and --runtime-profile are mutually exclusive")
    if is_submission and (
        args.worker is not None
        or (not args.wait and runtime_id is None and profile_path is None)
    ):
        if runtime_id is not None or profile_path is not None:
            raise ValueError(
                "--worker and --runtime/--runtime-profile are mutually exclusive"
            )
        return
    profile = (
        profile_for_id(runtime_id)
        if runtime_id is not None
        else load_runtime_profile(profile_path)
        if profile_path is not None
        else None
    )
    if args.command == "explore":
        # CPU is the documented default. An explicit profile selects its
        # backend; native authorization still checks engine and device index.
        args.backend = profile.backend if profile is not None else "cpu"
    if args.command == "stage" and args.through == "inspect":
        return
    if is_experiment and args.experiment_command == "run" and args.binding is not None:
        from klab.experiments.binding import inspect_runtime_binding
        from klab.experiments.cli import _run_cells
        from klab.experiments.lock import open_lock
        from klab.runtime_profile import RuntimeProfile

        locked = open_lock(Path(args.lock))
        cell = _run_cells(locked, args)[0]
        binding = inspect_runtime_binding(Path(args.binding), locked, cell)
        if binding["source_kind"] == "profile":
            profile = RuntimeProfile.model_validate(binding["descriptor"])
    workers = _campaign_evaluation_workers(args) if is_campaign else []
    if is_suite and getattr(args, "worker", None) is not None:
        if args.store is None:
            raise ValueError("--worker evaluation requires --store CONTROLLER_STORE")
        workers.append(_registered_local_worker(Path(args.store), args.worker))
    if workers and (
        len({str(worker.python) for worker in workers}) != 1
        or (
            profile is not None
            and any(
                worker.python.resolve() != profile.python.resolve()
                for worker in workers
            )
        )
    ):
        raise ValueError("one command cannot select different runtime interpreters")
    if profile is not None:
        digest = hashlib.sha256(profile.model_dump_json().encode("utf-8")).hexdigest()
        _ensure_runtime_interpreter(
            profile.python, digest, lambda: probe_runtime_profile(profile)
        )
        args.runtime_profile_python = Path(os.path.abspath(profile.python))
        args.runtime_profile_loaded = profile
    elif workers:
        _prepare_local_worker(args, workers[0])
    if is_suite or is_fixed_slices:
        from klab.evaluation.inference import evaluation_config
        from klab.runtime_profile import authorize_worker

        config = evaluation_config(
            args.run_id, Path(args.runs_dir), args.checkpoint, args.backend
        )
        if profile is not None:
            args.runtime_authorization = authorize_profile(profile, config)
        elif workers:
            args.runtime_authorization = authorize_worker(workers[0], config)
        return
    if is_experiment or is_campaign or is_snapshot:
        return

    if args.command in {"train", "stage"} or is_submission:
        config = load_config(Path(args.config))
    elif args.command == "try":
        config = load_config(Path(args.vs))
    elif args.command in {"probe", "explore"}:
        from klab.config.models import RunConfig
        from klab.lab_mode import lab_root
        from klab.lab_results import resolve_checkpoint
        from klab.reference_models import is_reference

        if args.command == "probe" and is_reference(args.target):
            # Pinned public checkpoints load in-process through transformers
            # on the CPU; there is no Model Lab run config to authorize.
            if profile is not None or workers:
                raise ValueError("runtime profiles do not apply to ref: models")
            return
        target = resolve_checkpoint(
            args.target,
            lab_dir=lab_root(resolve_work_dir(args.work_dir), args.lab_dir),
            runs_dir=Path(args.runs_dir) if args.runs_dir else None,
        )
        config = RunConfig.model_validate_json(
            (target.run / "resolved_config.yaml").read_text(encoding="utf-8")
        )
    else:
        from klab.config.models import RunConfig

        run = (Path(args.runs_dir) / args.run_id).resolve()
        config = RunConfig.model_validate_json(
            (run / "resolved_config.yaml").read_text(encoding="utf-8")
        )
    backend_override = (
        args.backend
        if args.command
        in {"try", "probe", "explore", "train", "eval", "generate", "chat", "serve"}
        else None
    )
    if backend_override is not None:
        config = config.model_copy(
            update={
                "runtime": config.runtime.model_copy(
                    update={"backend": backend_override}
                )
            }
        )
    if profile is None:
        try:
            require_authorization(config, None)
        except ValueError as error:
            if args.command == "probe" and args.backend is None:
                from klab.lab_results import ResultLookupError

                raise ResultLookupError(
                    args.target,
                    "runtime_required",
                    f"saved backend {config.runtime.backend}: {error}",
                    "the CPU extra installs a framework, not a backend override; "
                    "select a compatible --runtime or declaration runtime_profile, or explicitly "
                    "pass --backend cpu for a PyTorch run (no automatic fallback)",
                ) from None
            raise
    else:
        args.runtime_authorization = authorize_profile(profile, config)


def _read_only_command(args: argparse.Namespace) -> bool:
    from klab.campaign.cli import READ_ONLY_COMMANDS

    if args.command == "tokenizer" and args.tokenizer_command == "verify":
        return True
    if args.command == "iteration":
        return True
    if args.command in {"semantic", "memorization"}:
        return True
    if args.command == "data" and args.data_command == "coverage":
        return True
    if args.command == "data" and args.data_command == "prepared-inputs":
        return args.inputs_command == "verify"
    if (
        args.command == "evaluation"
        and args.evaluation_command == "fixed-slices"
        and args.fixed_command == "verify-selection"
    ):
        return True
    if args.command == "campaign":
        return args.campaign_command in READ_ONLY_COMMANDS
    if args.command == "recovery":
        return args.recovery_command in {"inspect", "plan"}
    if args.command == "family":
        return args.family_command in {"show", "graph", "compare", "verify"}
    if args.command == "archive":
        return args.archive_command == "verify"
    if args.command == "experiment":
        return args.experiment_command in {"validate", "inspect", "diff", "explain"}
    if args.command == "research":
        return args.research_command in {
            "snapshot",
            "list",
            "describe",
            "validate",
            "baseline",
        }
    if args.command == "runtime":
        return args.runtime_command in {"probe", "status"}
    if args.command == "attempt":
        return args.attempt_command == "status"
    if args.command == "corpus":
        return args.corpus_command not in {
            "acquire",
            "build",
            "freeze",
            "export",
            "tokenizer-bakeoff",
        }
    return args.command in {
        "runner-status",
        "inspect",
        "checkpoint",
        "report",
        "compare",
    }


def _command_storage_checks(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.command == "runner-status":
        return []
    if args.command == "tokenizer" and args.tokenizer_command == "verify":
        return []
    checks = storage_checks(resolve_work_dir(args.work_dir))
    for field in ("runs_dir", "store", "output"):
        value = getattr(args, field, None)
        if value:
            checks.extend(storage_checks(Path(value), kind=field))
    config_path = getattr(args, "config", None)
    if config_path and (
        args.command in {"train", "stage", "data", "tokenizer"}
        or _is_config_submission(args)
    ):
        config = (
            load_tokenizer_config(Path(config_path))
            if args.command == "tokenizer"
            else load_config(Path(config_path))
        )
        checks.extend(storage_checks(config.dataset.cache_dir, kind="dataset_cache"))
        if hasattr(config, "logging"):
            checks.extend(storage_checks(config.logging.root_dir, kind="logging_root"))
        if args.command == "tokenizer":
            checks.extend(storage_checks(config.output_dir, kind="tokenizer_output"))
    return checks


def main() -> None:
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument("--work-dir", type=Path)
    global_parser.add_argument("--hf-token-file", type=Path)
    # Only the prefix before the verb contains global arguments.
    # Consume other global values too, so they cannot become the verb boundary.
    global_parser.add_argument("command", nargs=argparse.REMAINDER)
    global_args, _ = global_parser.parse_known_args(sys.argv[1:])
    parser = build_parser(global_args.work_dir)
    args, extras = parser.parse_known_args()
    if extras:
        if args.command == "campaign" and args.json:
            from klab.campaign.cli import argument_error

            argument_error(
                args.campaign_command, f"unrecognized arguments: {' '.join(extras)}"
            )
        parser.error(f"unrecognized arguments: {' '.join(extras)}")
    if args.command == "experiment" and args.experiment_command == "recipe":
        args.handler(args)
        return
    if args.command == "experiment" and args.experiment_command == "submit":
        from klab.workers.cli import validate_submit_args

        try:
            validate_submit_args(args)
        except ValueError as error:
            parser.error(str(error))
    if (
        args.command in {"stage", "train"}
        or _is_config_submission(args)
        or (args.command == "data" and args.data_command == "prepare")
        or (
            args.command == "data"
            and args.data_command == "prepared-inputs"
            and args.inputs_command == "publish"
        )
        or (args.command == "experiment" and args.experiment_command == "prepare")
    ):
        validate_tokenizer_batch_limits(
            args.tokenizer_batch_documents
            if args.tokenizer_batch_documents is not None
            else TOKENIZER_BATCH_DOCUMENTS,
            args.tokenizer_batch_source_bytes
            if args.tokenizer_batch_source_bytes is not None
            else TOKENIZER_BATCH_SOURCE_BYTES,
        )
    if args.command == "try":
        from klab.try_config import prepare_try_config

        try:
            prepare_try_config(args)
        except (ValueError, OSError) as error:
            raise SystemExit(f"klab try: {error}") from None
    if args.command == "probe":
        from klab.probes.config import prepare_probe_config

        try:
            prepare_probe_config(args)
        except (ValueError, OSError) as error:
            raise SystemExit(f"klab probe: {error}") from None
    if getattr(args, "resource_envelope", None) is not None:
        try:
            args.resource_envelope_value = load_resource_envelope(
                args.resource_envelope
            )
            require_envelope_applicability(
                args.resource_envelope_value, operation=args.command
            )
        except (ValueError, OSError) as error:
            raise SystemExit(f"klab: {error}") from None
    else:
        args.resource_envelope_value = None
    try:
        from klab.phase_observations import validate_observations_destination

        try:
            validate_observations_destination(args)
        except (ValueError, OSError) as error:
            if args.command == "campaign" and args.json:
                from klab.campaign.cli import argument_error

                argument_error(args.campaign_command, str(error))
            raise
        from klab.data.legacy import require_current_dataset

        if (
            args.command in {"train", "stage"}
            or _is_config_submission(args)
            or (
                args.command == "data"
                and args.data_command in {"prepare", "bakeoff", "prepared-inputs"}
            )
        ):
            require_current_dataset(load_config(Path(args.config)).dataset)
        elif args.command == "tokenizer" and args.tokenizer_command == "train":
            require_current_dataset(load_tokenizer_config(Path(args.config)).dataset)
        _prepare_runtime_command(args)
    except (ValueError, OSError) as error:
        _lab_lookup_failure(args, error)
        raise SystemExit(f"klab: {error}") from None
    if args.resource_envelope_value is not None:
        if _is_config_submission(args):
            workspace = Path(args.store)
        elif args.command == "stage":
            workspace = Path(args.output)
        elif args.command == "train":
            config = load_config(Path(args.config))
            workspace = (
                Path(args.runs_dir)
                if args.runs_dir is not None
                else config.logging.root_dir
            )
        elif args.command == "data":
            workspace = (
                args.output
                if args.data_command == "prepared-inputs"
                else load_config(Path(args.config)).dataset.cache_dir
            )
        elif args.command in {"try", "explore", "probe"}:
            from klab.lab_mode import lab_root

            workspace = lab_root(resolve_work_dir(args.work_dir), args.lab_dir)
        elif args.command == "experiment" and args.experiment_command == "prepare":
            from klab.experiments.plan import load_plan

            plan = load_plan(Path(args.source).absolute())
            workspace = resolve_work_dir(args.work_dir) / "experiments" / plan.id
        else:
            workspace = args.work_dir or Path.cwd()
        try:
            check_envelope(
                # Try resolves scoped limits at its controller boundary; this
                # preflight only measures the common host/storage constraints.
                args.resource_envelope_value.model_copy(update={"lab_try": None})
                if args.command == "try"
                else args.resource_envelope_value,
                workspace=workspace,
                rss_bytes=current_process_rss_bytes(),
            )
        except (ValueError, OSError) as error:
            raise SystemExit(f"klab: {error}") from None
    set_token_file(args.hf_token_file)
    args.work_dir = resolve_work_dir(args.work_dir)
    os.environ["KLAB_WORK_DIR"] = str(args.work_dir)
    args.storage_checks = _command_storage_checks(args)
    read_only = _read_only_command(args)
    deferred = (
        args.command in {"campaign", "recovery", "archive", "attempt", "lab-job"}
        or (
            args.command == "experiment"
            and args.experiment_command in {"prepare", "run"}
        )
        or (args.command == "data" and args.data_command == "benchmark-preparation")
    )
    if not read_only:
        if args.command not in {"campaign", "recovery"}:
            warn_storage_checks(args.storage_checks)
        if not deferred:
            ensure_work_dir(args.work_dir)
            if (
                args.command in {"train", "tokenizer"}
                or (args.command == "data" and args.data_command == "prepare")
                or (
                    args.command == "data"
                    and args.data_command == "prepared-inputs"
                    and args.inputs_command == "publish"
                )
                or (
                    args.command == "corpus"
                    and args.corpus_command
                    in {
                        "acquire",
                        "build",
                        "freeze",
                        "export",
                        "tokenizer-bakeoff",
                    }
                )
            ):
                record_storage_observation(
                    args.work_dir,
                    args.storage_checks,
                    args.command,
                    target=getattr(args, "config", None)
                    or getattr(args, "project", None),
                )
    try:
        from klab.data.sources import snapshot_verification_operation
        from klab.phase_observations import observation_session

        reuse = (
            args.command == "stage"
            or (args.command == "data" and args.data_command == "prepare")
            or (
                args.command == "campaign"
                and args.campaign_command
                in {"plan", "status", "next", "explain", "apply", "resume", "approve"}
            )
        )
        with (
            snapshot_verification_operation(
                mode="verified_reuse"
                if reuse and not getattr(args, "cold_verify", False)
                else "cold"
            ),
            observation_session(args),
        ):
            args.handler(args)
    except (HuggingFaceAccessError, HuggingFaceCredentialError) as error:
        raise SystemExit(f"klab: {error}") from None
