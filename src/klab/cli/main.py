"""Command-line entry point for Model Lab."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from klab.batch_calibration import calibrate_batch
from klab.config.loading import load_config, load_tokenizer_config
from klab.dashboard.stores import configured_stores, parse_run_store
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
from klab.engram.packs import (
    compile_pack,
    inspect_pack,
    verify_pack,
)
from klab.evaluation.chat import ChatMessage, assistant_reply, prepare_chat_prompt
from klab.evaluation.evidence import experiment_evidence
from klab.evaluation.generation import generate
from klab.evaluation.inference import load_run, write_inference_result
from klab.evaluation.post_train_triage import read_triage, triage_summary
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
    ]
    for label, directory in args.run_store:
        command.extend(["--run-store", f"{label}={directory}"])
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


def _config_compose(args: argparse.Namespace) -> None:
    """Publish one run config assembled from several declaration files."""
    try:
        from klab.derivation import compose_config, parse_parts

        receipt = compose_config(parse_parts(args.parts), Path(args.output))
    except (OSError, TypeError, ValueError) as error:
        if args.json:
            print(json.dumps({"status": "error", "error": str(error)}))
        else:
            print(f"klab config compose: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    if args.json:
        print(json.dumps({"status": "ok", **receipt}, sort_keys=True))
    else:
        print(f"Composed {len(receipt['parts'])} parts: {receipt['output']['path']}")
        print("Schema valid; no artifact, runtime or capacity was checked.")


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


def _architecture_list(args: argparse.Namespace) -> None:
    from klab.architectures import count_parameters, load_preset, preset_names

    rows = []
    for name in preset_names():
        _, preset = load_preset(name)
        rows.append(
            {
                "name": name,
                "description": preset.description,
                "reference_parameters": count_parameters(
                    preset.model, preset.attention
                ),
            }
        )
    if args.json:
        print(json.dumps(rows, sort_keys=True))
        return
    for row in rows:
        print(f"{row['name']:<10} {row['description']}")


def _architecture_derive(args: argparse.Namespace) -> None:
    try:
        from klab.architectures import derive_run_config, parse_parameter_count

        result = derive_run_config(
            args.preset,
            parse_parameter_count(args.params),
            Path(args.base),
            Path(args.output),
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            num_layers=args.layers,
            name=args.name,
        )
    except (OSError, TypeError, ValueError) as error:
        if args.json:
            print(json.dumps({"status": "error", "error": str(error)}))
        else:
            print(f"klab architecture derive: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    if args.json:
        print(json.dumps({"status": "ok", **result}, sort_keys=True))
        return
    print(
        f"{result['preset']}: {result['parameters']:,} parameters "
        f"({result['relative_error']:+.2%} of the {result['target_parameters']:,} "
        f"target), {result['num_layers']} layers x {result['hidden_dim']} wide, "
        f"{result['num_heads']} heads"
    )
    print(f"output: {result['output']}")


def _checkpoint_inspect(args: argparse.Namespace) -> None:
    path = Path(args.path)
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


def _data_tokens(args: argparse.Namespace) -> None:
    from klab.data.packing import token_store_root
    from klab.data.token_store import list_stores

    root = token_store_root(load_config(Path(args.config)).dataset.cache_dir)
    stores = list_stores(root)
    if args.json:
        print(json.dumps({"root": str(root), "stores": stores}, sort_keys=True))
        return
    print(f"Token stores under {root}: {len(stores)}")
    for store in stores:
        print(
            f"  {Path(str(store['path'])).name}  "
            f"{store['documents']:,} documents  {store['tokens']:,} tokens  "
            f"{store['size_bytes'] / 2**30:.2f} GiB  "
            f"{'whole source' if store['complete'] else 'partial source'}  "
            f"source {str(store['source_sha256'])[:12]}  "
            f"tokenizer {str(store['tokenizer_sha256'])[:12]}"
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
    from klab.staging import materialize_prepared_inputs, verify_prepared_inputs

    config = load_config(Path(args.config))
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


def _chat(args: argparse.Namespace) -> None:
    loaded = load_run(
        args.run_id,
        Path(args.runs_dir),
        args.checkpoint,
        args.backend,
        authorization=args.runtime_authorization,
    )
    context = _inference_context(args, loaded.config.model.max_seq_len)
    template = loaded.config.dataset.template
    stops = [*template.stop, *args.stop]
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
            template=template,
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
            stop_token_ids=template.stop_token_ids(loaded.tokenizer),
            use_cache=not args.no_cache,
            engine=loaded.engine,
        )
        reply = assistant_reply(completion, prompt, template)
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


def _weights_export(args: argparse.Namespace) -> None:
    from klab.training.weight_export import export_weights

    print(
        json.dumps(
            export_weights(
                Path(args.checkpoint), Path(args.output), export_format=args.format
            ),
            sort_keys=True,
        )
    )


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
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"RUN EVIDENCE: {run_id}")
        print(json.dumps(evidence, indent=2, sort_keys=True))
        print(_triage_summary(triage, error, run_id=run_id, runs_dir=runs_dir))
    if evidence.get("status") in {"missing_evidence", "invalid_evidence"}:
        raise SystemExit(1)


def _report(args: argparse.Namespace) -> None:
    """Read results through native readers without rerunning or publishing anything."""
    selectors = {"record": "lab_dir", "run": "runs_dir"}
    for kind, field in selectors.items():
        if kind != args.kind and getattr(args, field) is not None:
            raise SystemExit(
                f"klab report: --{field.replace('_', '-')} requires --kind {kind}"
            )
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


# Help groups: building a model first, then the lab-mode comparison loop;
# release/full-provenance last. Every registered command appears exactly once
# (see tests/test_lab_mode.py).
BUILD_COMMANDS = (
    "architecture",
    "inspect",
    "tokenizer",
    "data",
    "train",
    "generate",
)
FAST_PATH_COMMANDS = (
    "try",
    "probe",
    "report",
    "compare",
    "explore",
    "eval",
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
    "controller",
    "worker",
    "hosted",
    "monitor",
    "monitor-baseline",
    "lab-job",
)


_FALLBACK_HELP = {
    "inspect": "Show effective config, shape and memory estimate (read-only)",
    "train": "Train one config locally (native trainer)",
    "eval": "Score a run's checkpoint on its held-out validation split",
    "generate": "Generate a raw continuation from a verified checkpoint",
    "dashboard": "Local read-only training/research dashboard",
    "readiness": "CPU wiring smoke checks for the shipped mechanisms",
    "weights": "Import external weights with provenance, or export for Hugging Face",
    "config": "Validate, compose and derive run configs",
    "checkpoint": "Inspect or verify checkpoints",
    "workspace": "Storage preflight and cleanup plans",
    "runtime": "Runtime probes and status",
    "tokenizer": "Train or verify a tokenizer on your data",
    "data": "Prepare, verify and inspect training data (data prepare CONFIG)",
    "engram": "Export, inspect and pack Engram memory",
    "stage": "Pilot a config through staged gates",
    "batch": "Calibrate micro-batch size",
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
        if name not in (*BUILD_COMMANDS, *FAST_PATH_COMMANDS, *RELEASE_COMMANDS)
    ]
    missing = [
        name
        for name in (*BUILD_COMMANDS, *FAST_PATH_COMMANDS, *RELEASE_COMMANDS)
        if name not in names
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
                "Build (your text -> tokenizer -> prepared data -> model):",
                BUILD_COMMANDS,
            ),
            section(
                "Compare (lab mode: YAML delta -> scored comparison):",
                FAST_PATH_COMMANDS,
            ),
            section("Models and diagnostics:", tools),
            section(
                "Advanced / release (full provenance, approvals, campaigns):",
                RELEASE_COMMANDS,
            ),
            "Build a model: klab train CONFIG.yaml (docs/build.md)",
            "Test a change: klab try DELTA.yaml --vs BASELINE.yaml",
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
        help="Read a try or probe result, or a run's evidence (no compute)",
    )
    lab_report.add_argument(
        "record", nargs="?", help="try/probe record; run ID with --kind run"
    )
    lab_report.add_argument("--kind", choices=("record", "run"), default="record")
    lab_report.add_argument("--lab-dir", type=Path)
    lab_report.add_argument("--json", action="store_true")
    lab_report.add_argument("--runs-dir", help="Saved run location for --kind run")
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
        choices=("hf_llama_safetensors",),
    )
    weight_import.add_argument("--source-tokenizer", required=True)
    weight_import.add_argument("--provenance", required=True)
    weight_import.set_defaults(handler=_weights_import)
    weight_export = weight_commands.add_parser(
        "export",
        help="Write a checkpoint as a Hugging Face Llama directory",
    )
    weight_export.add_argument(
        "checkpoint", help="run directory, checkpoint generation or latest.json"
    )
    weight_export.add_argument("output", help="new directory to create")
    weight_export.add_argument(
        "--format", default="hf_llama_safetensors", choices=("hf_llama_safetensors",)
    )
    weight_export.set_defaults(handler=_weights_export)
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

    config_compose = config_commands.add_parser(
        "compose",
        help="Assemble one run config from section and local-path files",
    )
    config_compose.add_argument(
        "parts",
        nargs="+",
        metavar="PATH|FIELD=PATH",
        help="YAML/JSON mappings merged in order; FIELD=PATH places a file at a field",
    )
    config_compose.add_argument("--output", required=True, help="New YAML output path")
    config_compose.add_argument("--json", action="store_true")
    config_compose.set_defaults(handler=_config_compose)
    register_schedule(config_commands)
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
    from klab.data.source_cli import register_parser as register_dataset_sources

    register_dataset_sources(data_commands)
    register_coverage(data_commands)
    data_prepare.set_defaults(handler=_data_prepare)
    data_tokens = data_commands.add_parser(
        "tokens",
        help="List the reusable token stores beside a config's data cache",
    )
    data_tokens.add_argument("config")
    data_tokens.add_argument("--json", action="store_true")
    data_tokens.set_defaults(handler=_data_tokens)
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
    architecture = commands.add_parser(
        "architecture",
        help="List named architecture presets or scale one to a parameter target.",
    )
    architecture_commands = architecture.add_subparsers(
        dest="architecture_command", required=True
    )
    architecture_list = architecture_commands.add_parser("list")
    architecture_list.add_argument("--json", action="store_true")
    architecture_list.set_defaults(handler=_architecture_list)
    architecture_derive = architecture_commands.add_parser(
        "derive",
        help="Scale a preset to a parameter count and write a run config from a base",
    )
    architecture_derive.add_argument(
        "preset", help="Shipped preset name or a preset YAML path"
    )
    architecture_derive.add_argument(
        "--params", required=True, help="Target total parameters, such as 300M or 1B"
    )
    architecture_derive.add_argument(
        "--base",
        required=True,
        help="Run config supplying everything except model and attention",
    )
    architecture_derive.add_argument("--output", required=True, help="New YAML path")
    architecture_derive.add_argument(
        "--vocab-size", type=int, help="Default: the base config's vocabulary"
    )
    architecture_derive.add_argument(
        "--max-seq-len", type=int, help="Default: the base config's context"
    )
    architecture_derive.add_argument(
        "--layers", type=int, help="Fix the depth and search width only"
    )
    architecture_derive.add_argument("--name", help="Run name for the new config")
    architecture_derive.add_argument("--json", action="store_true")
    architecture_derive.set_defaults(handler=_architecture_derive)
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
    dashboard.add_argument(
        "--lab-dir",
        type=Path,
        help="Lab root for the Probes page (default: WORK_DIR/lab)",
    )
    dashboard.set_defaults(handler=_dashboard)
    from klab.iteration_cli import register_parser as register_iteration_parser

    register_iteration_parser(commands)
    from klab.corpus.cli import add_commands as add_corpus_commands

    add_corpus_commands(commands)
    from klab.campaign.cli import add_commands as add_campaign_commands

    add_campaign_commands(commands)
    from klab.workers.cli import add_commands

    add_commands(commands, default_store=Path(runs_dir_default))
    from klab.hosted.cli import add_parser as add_hosted_parser

    add_hosted_parser(commands)
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
