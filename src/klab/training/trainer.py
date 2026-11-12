# Malformed persisted portability identity is reported as ValueError.
# ruff: noqa: TRY004
"""Engine-neutral single-host training lifecycle with explicit update boundaries."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import signal
import socket
import time
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Literal

import numpy as np
import psutil
import torch

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import RunConfig
from klab.corpus.export import verify_release_export
from klab.data.allocation import (
    copy_allocation_bundle,
    load_allocation_manifest,
)
from klab.data.encoding import (
    TOKENIZER_BATCH_DOCUMENTS,
    TOKENIZER_BATCH_SOURCE_BYTES,
    validate_tokenizer_batch_limits,
)
from klab.data.packing import (
    BatchCursor,
    PairedBlockDataset,
    PreparedData,
    TokenBlockDataset,
    epoch_order,
    load_prepared_data,
    prepare_data,
    supervision_requires_mask,
)
from klab.data.story_packing import MatchedStoryDataset
from klab.data.tokenizer import load_tokenizer
from klab.engines.base import EngineState, EvaluationResult, Microbatch
from klab.engines.pytorch import PyTorchEngine
from klab.evaluation.panel_data import (
    copy_fixed_panels,
    load_fixed_panels,
    panel_protocol,
    panel_protocol_sha256,
)
from klab.evaluation.post_train_triage import triage_completed_run
from klab.memory import (
    calibrated_estimate,
    calibration_key,
    parameter_inventory,
    validate_offload_headroom,
)
from klab.objectives import objective as registered_objective
from klab.progress import ProgressReporter
from klab.resource_envelope import (
    ResourceEnvelope,
    check_envelope,
    current_process_rss_bytes,
)
from klab.runtime import seed_everything
from klab.runtime_forecasting import (
    RUNTIME_OBSERVATION_KIND,
    runtime_forecast_planning,
    runtime_signature_key,
)
from klab.runtime_profile import RuntimeAuthorization, require_authorization
from klab.training.attempt_budget import AttemptBudget
from klab.training.checkpoints import (
    CheckpointManager,
    CheckpointRecord,
    LineageBest,
    TrainingSnapshot,
    _atomic_json,
    choose_lineage_best,
    existing_run_recovery_report,
)
from klab.training.continuation import load_continuation
from klab.training.exposure import checkpoint_exposure, exposure_context
from klab.training.manifest import (
    ArtifactIdentity,
    RunManifest,
    _is_sha256,
    architecture_sha256,
    checkpoint_selection,
    config_sha256,
    sha256_file,
    source_identity,
    write_manifest,
)
from klab.training.metrics import ExperimentStore
from klab.training.optimizer import schedule_payload
from klab.training.pilot_progress import (
    current_pilot_progress,
    emit_pilot_progress,
    pilot_phase,
)
from klab.training.preparation_guard import require_model_runtime_allowed
from klab.training.spot_safety import SpotSafetyPolicy, check_spot_capacity
from klab.training.stages import ExperimentStage, StageHistory
from klab.training.throughput import summarize_training_progress

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore

_LOGGER = logging.getLogger(__name__)


class _WallTimeExpired(Exception):
    """Internal signal that discards an incomplete cooperative evaluation."""


def _load_run_data(
    run: Path,
    config: RunConfig,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: Literal["cold", "verified_reuse"] = "cold",
) -> PreparedData:
    return load_prepared_data(
        run / "data",
        byte_enabled=config.model.memory in {"byte", "portable"},
        proof_store=proof_store,
        verification_mode=verification_mode,
    )


def _check_training_storage(
    config: RunConfig,
    data: PreparedData,
    *,
    tokenizer_path: Path | None = None,
    completed_steps: int = 0,
    completed_tokens: int = 0,
) -> None:
    from klab.workspace_preflight import require_storage, training_storage_checks

    # Fresh packing checks new-cache growth before allocating. Native loading
    # has now authenticated these inputs, so reserve only their run copy and
    # checkpoints. Bind the actual cache/stage location for this storage query.
    storage_config = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(
                update={"cache_dir": data.root.parent}
            ),
            "tokenizer": config.tokenizer.model_copy(
                update={"path": tokenizer_path or config.tokenizer.path}
            ),
        }
    )
    require_storage(
        training_storage_checks(
            storage_config,
            verified_prepared=data.receipt,
            completed_steps=completed_steps,
            completed_tokens=completed_tokens,
        )
    )


def _stack_optional(
    records: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
        ]
    ],
    field: int,
) -> np.ndarray | None:
    values = [record[field] for record in records]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("microbatch sidecars must be present consistently")
    return np.stack([value for value in values if value is not None])


def _portability_manifest_v2(path: Path | None) -> dict[str, object] | None:
    if path is None or path.is_symlink() or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        isinstance(payload, dict)
        and payload.get("format") == "klab-portability-run"
        and type(payload.get("version")) is int
        and payload["version"] == 2
    ):
        return payload
    return None


def _bind_learned_inputs(config: RunConfig, manifest_path: Path) -> RunConfig:
    payload = _portability_manifest_v2(manifest_path)
    if payload is None:
        raise ValueError("learned portability run manifest is not v2")
    memory = payload.get("memory")
    if not isinstance(memory, dict):
        raise ValueError("learned portability manifest lacks memory identity")
    artifact = memory.get("artifact")
    package_path: Path | None = None
    if artifact is not None:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValueError("learned portable memory descriptor is invalid")
        package_path = manifest_path.parent / artifact["path"]
    role = payload["coordinate"]["role"]
    condition = payload["coordinate"]["condition"]
    default_train = {
        "source": "source_train.jsonl",
        "preparation": "preparation_train.jsonl",
        "recipient": "adapter_calibration.jsonl",
        "native": "source_train.jsonl",
        "calibration": (
            "preparation_train.jsonl"
            if condition.startswith("preparation-")
            else "adapter_calibration.jsonl"
            if condition.startswith("adapter-")
            else "source_train.jsonl"
        ),
    }[role]
    train_name = (
        config.dataset.train_path.name
        if config.dataset.train_path is not None
        else default_train
    )
    default_validation = "preparation_validation.jsonl"
    validation_name = (
        config.dataset.validation_path.name
        if config.dataset.validation_path is not None
        else default_validation
    )
    data_root = manifest_path.parent / "portability" / "data"
    return config.model_copy(
        update={
            "tokenizer": config.tokenizer.model_copy(
                update={"path": manifest_path.parent / "tokenizer.json"}
            ),
            "model": config.model.model_copy(
                update={"memory_package_path": package_path}
            ),
            "dataset": config.dataset.model_copy(
                update={
                    "train_path": data_root / train_name,
                    "validation_path": data_root / validation_name,
                }
            ),
            "training": config.training.model_copy(
                update={"portability_manifest_path": manifest_path}
            ),
        }
    )


def _copy_artifacts(
    run: Path, config: RunConfig, data: PreparedData, source_run: Path | None = None
) -> tuple[ArtifactIdentity, ...]:
    artifacts: list[ArtifactIdentity] = []
    copied_data = {}
    tokenizer_source = (
        source_run / "tokenizer.json"
        if source_run is not None
        else config.tokenizer.path
    )
    tokenizer = run / "tokenizer.json"
    shutil.copy2(tokenizer_source, tokenizer)
    if source_run is None:
        for source in (config.tokenizer.path.with_name("tokenizer_manifest.json"),):
            if source.is_file():
                shutil.copy2(source, run / source.name)
    elif (source_run / "tokenizer_manifest.json").is_file():
        shutil.copy2(
            source_run / "tokenizer_manifest.json", run / "tokenizer_manifest.json"
        )
    if config.dataset.corpus_release_path is not None:
        portable = (
            source_run / "corpus" / "binding.json" if source_run is not None else None
        )
        corpus_dir = run / "corpus"
        corpus_dir.mkdir()
        if portable is not None and portable.is_file():
            from klab.workers.bundles import verify_portable_corpus_binding

            binding = verify_portable_corpus_binding(config, source_run)
            for name in (*binding["files"], "binding.json"):
                shutil.copy2(source_run / "corpus" / name, corpus_dir / name)
            verify_portable_corpus_binding(
                config,
                run if (run / "tokenizer_manifest.json").is_file() else source_run,
            )
        else:
            binding = verify_release_export(config.dataset)
            evidence = (
                "manifest.json",
                "report.json",
                "license-report.json",
                "audit.json",
            )
            for name in evidence:
                shutil.copy2(
                    config.dataset.corpus_release_path / name, corpus_dir / name
                )
            shutil.copy2(
                config.dataset.corpus_export_path / "export.json",
                corpus_dir / "export.json",
            )
            if sha256_file(corpus_dir / "export.json") != binding["export_sha256"]:
                raise ValueError("corpus export changed while copying run evidence")
    # Keep run ownership while using native reflink/clone support where available.
    from klab.staging import _copy_tree

    copied_data = _copy_tree(
        data.root, run / "data", verified=dict(data.receipt.proofs)
    )
    copy_fixed_panels(
        config, run, inputs_root=source_run, training_manifest=data.manifest
    )
    mask_required = supervision_requires_mask(data.manifest)
    for split in ("train", "validation"):
        mask = run / "data" / f"{split}_supervision.npy"
        if mask.is_file() != mask_required:
            raise ValueError(
                f"prepared {split} supervision inventory does not match descriptor"
            )
    manifest_source = (
        source_run / "portability_manifest.json"
        if source_run is not None
        and (source_run / "portability_manifest.json").is_file()
        else config.training.portability_manifest_path
    )
    learned_manifest = _portability_manifest_v2(manifest_source)
    if learned_manifest is not None:
        assert manifest_source is not None
        bundle_source = manifest_source.parent / "portability"
        if bundle_source.is_symlink() or not bundle_source.is_dir():
            raise ValueError("learned portability owned bundle is missing")
        shutil.copytree(bundle_source, run / "portability")
        shutil.copy2(manifest_source, run / "portability_manifest.json")
    package_source = (
        source_run / "portable_package"
        if source_run is not None
        else config.model.memory_package_path
    )
    if (
        learned_manifest is None
        and package_source is not None
        and package_source.exists()
    ):
        destination = run / "portable_package"
        if package_source.is_dir():
            shutil.copytree(package_source, destination)
        else:
            shutil.copy2(package_source, destination)
    allocation_source = config.dataset.allocation_manifest_path
    if allocation_source is not None:
        cache_identity = data.manifest.get("cache_identity")
        allocation_metadata = data.manifest.get("allocation")
        if not isinstance(cache_identity, dict) or not isinstance(
            allocation_metadata, dict
        ):
            raise ValueError("prepared data lacks allocation identity")
        source_identity_sha256 = cache_identity.get("source_identity_sha256")
        tokenizer_sha256 = cache_identity.get("tokenizer_sha256")
        if not isinstance(source_identity_sha256, str) or not isinstance(
            tokenizer_sha256, str
        ):
            raise ValueError("prepared data allocation identity is malformed")
        if source_run is not None:
            allocation_source = source_run / "allocation" / allocation_source.name
        allocation = load_allocation_manifest(
            allocation_source,
            source_identity_sha256=source_identity_sha256,
            tokenizer_sha256=tokenizer_sha256,
        )
        if allocation.sha256 != allocation_metadata.get("manifest_sha256"):
            raise ValueError("prepared data and allocation manifests differ")
        copy_allocation_bundle(allocation, run / "allocation")
    for path in sorted(
        item
        for item in run.rglob("*")
        if item.is_file()
        and item != run / "manifest.json"
        and not item.is_relative_to(run / "checkpoints")
    ):
        relative = path.relative_to(run)
        proof = (
            copied_data.get(relative.relative_to("data").as_posix())
            if copied_data and relative.parts[0] == "data"
            else None
        )
        if proof is not None:
            from klab.data.verification import _fingerprint, _owned_file

            if not _owned_file(proof) or proof.fingerprint != _fingerprint(path):
                raise ValueError(f"copied run input changed before manifest: {path}")
        artifacts.append(
            ArtifactIdentity(
                str(relative),
                proof.sha256 if proof is not None else sha256_file(path),
                proof.size_bytes if proof is not None else path.stat().st_size,
            )
        )
    return tuple(artifacts)


def _checkpoint_due(
    config: RunConfig,
    step: int,
    tokens: int,
    elapsed: float,
    watermarks: dict[str, float],
) -> bool:
    checks = (
        (config.checkpoint.every_steps, step - watermarks.get("step", 0)),
        (config.checkpoint.every_tokens, tokens - watermarks.get("tokens", 0)),
        (
            config.checkpoint.every_minutes,
            (elapsed - watermarks.get("minutes", 0)) / 60,
        ),
    )
    policy = config.checkpoint.retention
    milestone_due = policy is not None and any(
        step >= item.step
        and tokens >= item.committed_targets
        and not (
            watermarks.get("step", -1) >= item.step
            and watermarks.get("tokens", -1) >= item.committed_targets
        )
        for item in policy.milestones
    )
    return (
        milestone_due
        or step in config.checkpoint.steps
        or any(interval is not None and value >= interval for interval, value in checks)
    )


def _checkpoint_watermarks(
    step: int,
    tokens: int,
    elapsed: float,
    previous: dict[str, float],
    *,
    operational_only: bool,
) -> dict[str, float]:
    if operational_only:
        return previous.copy()
    return {"step": float(step), "tokens": float(tokens), "minutes": elapsed}


# Per-step progress.json rewrites are limited to this interval; checkpoints,
# stage changes and the final state are always written.
PROGRESS_INTERVAL_SECONDS = 1.0


def _host_copy(value: object) -> object:
    """Copy every tensor in a nested state to host memory."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _host_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_host_copy(item) for item in value)
    return value


def _tensor_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _snapshot(
    manager: CheckpointManager,
    engine: object,
    config: RunConfig,
    run_id: str,
    cursor: BatchCursor,
    step: int,
    tokens: int,
    watermarks: dict[str, float],
    validation_loss: float | None = None,
    *,
    source_digest: str,
    parent_digest: str | None,
    wall_seconds: float,
    update_seconds: float,
    lineage_best: LineageBest | None = None,
    exposure: dict[str, object] | None = None,
    detach: bool = False,
) -> TrainingSnapshot | None:
    """Capture the training state; ``detach`` copies it to host memory.

    A detached snapshot shares nothing with the live engine, so it can be
    written while training continues. Returns None when the host lacks the
    memory for that copy.
    """
    state = engine.export_training_state()
    weights = engine.export_weights()
    optimizer = state.optimizer
    if detach:
        needed = _tensor_bytes(optimizer) + sum(
            parameter.numel() * parameter.element_size()
            for parameter in engine.model.state_dict().values()
        )
        if psutil.virtual_memory().available < 1.5 * needed:
            return None
        weights = weights.detached()
        optimizer = _host_copy(optimizer)
    config_payload = config.model_dump(mode="json")
    return TrainingSnapshot(
        model={},
        weight_source=weights,
        optimizer=optimizer,
        schedule=schedule_payload(config, step),
        step=step,
        tokens_seen=tokens,
        cursor=(cursor.epoch, cursor.next_block),
        config=config_payload,
        run_id=run_id,
        rng=state.rng,
        scaler=state.scaler,
        validation_loss=validation_loss,
        cadence=watermarks.copy(),
        engine=config.runtime.engine,
        backend=config.runtime.backend,
        optimizer_parameter_names=state.optimizer_parameter_names,
        config_sha256=config_sha256(config_payload),
        architecture_sha256=architecture_sha256(config_payload),
        source_identity_sha256=source_digest,
        manifest_sha256=manager.manifest_sha256,
        parent_checkpoint_sha256=parent_digest,
        cumulative_wall_seconds=wall_seconds,
        cumulative_update_seconds=update_seconds,
        lineage_best=lineage_best,
        exposure=exposure,
    )


def _write_validation_report(
    run: Path,
    record: CheckpointRecord,
    result: dict[str, object],
    config: RunConfig,
    source_digest: str,
    mask_required: bool,
) -> Path:
    evaluations = run / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    report = (
        evaluations
        / f"validation_step_{record.step:08d}_gen_{record.generation_id:06d}.json"
    )
    identities = {
        "data/validation.npy": sha256_file(run / "data" / "validation.npy"),
        "tokenizer.json": sha256_file(run / "tokenizer.json"),
        "source_identity_sha256": source_digest,
        "protocol": "next-token-cross-entropy-v1",
    }
    supervision = run / "data" / "validation_supervision.npy"
    if supervision.is_file() != mask_required:
        raise ValueError("validation supervision inventory does not match descriptor")
    if mask_required:
        identities["data/validation_supervision.npy"] = sha256_file(supervision)
    addresses = run / "data" / "validation_byte_addresses.npy"
    if addresses.is_file():
        identities["data/validation_byte_addresses.npy"] = sha256_file(addresses)
    if config.evaluation.panels:
        for path in sorted((run / "panels").rglob("*")):
            if path.is_file():
                identities[path.relative_to(run).as_posix()] = sha256_file(path)
    payload: dict[str, object] = {
        "kind": "held_out_validation_v2",
        "checkpoint": record.relative_path,
        "checkpoint_sha256": record.manifest_sha256,
        "step": record.step,
        "tokens_seen": record.tokens_seen,
        "identities": identities,
        "batch_size": None
        if config.evaluation.panels
        else config.evaluation.batch_size or config.training.micro_batch_size,
        "max_batches": None
        if config.evaluation.panels
        else config.evaluation.max_batches,
        "seq_len": config.training.seq_len,
        **result,
    }
    payload["sha256"] = canonical_digest(payload)
    content = canonical_json(payload) + b"\n"
    if report.exists():
        if report.read_bytes() != content:
            raise FileExistsError(
                f"immutable validation report already exists: {report}"
            )
        return report
    temporary = report.with_name(report.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        __import__("os").fsync(handle.fileno())
    temporary.replace(report)
    return report


def _check_startup_cancel(cancel_path: Path | None) -> None:
    if cancel_path is not None and cancel_path.exists():
        raise InterruptedError(
            f"training cancelled before run creation: {cancel_path}; no new checkpoint"
        )


def train(
    config: RunConfig,
    *,
    resume: Path | None = None,
    extend_budget: Path | None = None,
    promote: Path | None = None,
    add_ngram_memory: bool = False,
    extend_context: bool = False,
    append_data: bool = False,
    append_from_checkpoint: bool = False,
    recover: Path | None = None,
    run_id: str | None = None,
    stop_after_step: int | None = None,
    worker_id: str | None = None,
    allow_runtime_drift: bool = False,
    allow_runtime_optimization: bool = False,
    max_wall_seconds: float | None = None,
    cancel_path: Path | None = None,
    experiment_id: str | None = None,
    attempt_id: str | None = None,
    dispatch_metadata: dict[str, object] | None = None,
    stage_bundle: Path | None = None,
    requested_config_override: dict[str, object] | None = None,
    authorization: RuntimeAuthorization | None = None,
    resource_envelope: ResourceEnvelope | None = None,
    tokenizer_batch_documents: int = TOKENIZER_BATCH_DOCUMENTS,
    tokenizer_batch_source_bytes: int = TOKENIZER_BATCH_SOURCE_BYTES,
    proof_store: ProofStore | None = None,
    verification_mode: Literal["cold", "verified_reuse"] = "cold",
    require_prepared: bool = False,
    checkpoint_committed: Callable[[Path, CheckpointRecord], None] | None = None,
    spot_policy: SpotSafetyPolicy | None = None,
) -> str:
    """Run one independent experiment, optionally bound to a stage bundle."""
    if add_ngram_memory and promote is None:
        raise ValueError("--add-ngram-memory requires --promote")
    if extend_context and not add_ngram_memory:
        raise ValueError("--extend-context requires --add-ngram-memory and --promote")
    if append_from_checkpoint and not append_data:
        raise ValueError("--append-from-checkpoint requires --append-data")
    if append_data and extend_budget is None:
        raise ValueError("--append-data requires --extend-budget")
    require_model_runtime_allowed()
    require_authorization(config, authorization)
    validate_tokenizer_batch_limits(
        tokenizer_batch_documents, tokenizer_batch_source_bytes
    )
    if resource_envelope is not None:
        check_envelope(
            resource_envelope,
            workspace=config.logging.root_dir,
            rss_bytes=current_process_rss_bytes(),
        )
    if max_wall_seconds is not None and (
        isinstance(max_wall_seconds, bool)
        or not isinstance(max_wall_seconds, (int, float))
        or not math.isfinite(max_wall_seconds)
        or max_wall_seconds <= 0
    ):
        raise ValueError("max_wall_seconds must be positive and finite")
    _check_startup_cancel(cancel_path)
    started = time.monotonic()
    completed_run_id = _train_impl(
        config,
        resume=resume,
        extend_budget=extend_budget,
        promote=promote,
        add_ngram_memory=add_ngram_memory,
        extend_context=extend_context,
        append_data=append_data,
        append_from_checkpoint=append_from_checkpoint,
        recover=recover,
        run_id=run_id,
        stop_after_step=stop_after_step,
        worker_id=worker_id,
        allow_runtime_drift=allow_runtime_drift,
        allow_runtime_optimization=allow_runtime_optimization,
        stage_bundle=stage_bundle,
        cancel_path=cancel_path,
        purpose="training",
        experiment_id=experiment_id,
        attempt_id=attempt_id,
        dispatch_metadata=dispatch_metadata,
        requested_config_override=requested_config_override,
        max_wall_seconds=max_wall_seconds,
        authorization=authorization,
        resource_envelope=resource_envelope,
        tokenizer_batch_documents=tokenizer_batch_documents,
        tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
        proof_store=proof_store,
        verification_mode=verification_mode,
        require_prepared=require_prepared,
        checkpoint_committed=checkpoint_committed,
        spot_policy=spot_policy,
    )
    if _attempt_contract_suppresses_triage():
        return completed_run_id
    try:
        progress = config.logging.root_dir / completed_run_id / "progress.json"
        if progress.is_symlink():
            raise ValueError("run progress is a symlink")
        if json.loads(progress.read_text(encoding="utf-8"))["status"] == "completed":
            remaining = (
                None
                if max_wall_seconds is None
                else max(0.0, max_wall_seconds - (time.monotonic() - started))
            )
            triage_completed_run(
                completed_run_id,
                config.logging.root_dir,
                remaining_seconds=remaining,
                authorization=authorization,
            )
    except Exception as error:  # noqa: BLE001 - diagnostics must not fail completed training
        _LOGGER.warning("POST-TRAIN TRIAGE UNKNOWN for %s: %s", completed_run_id, error)
    return completed_run_id


def _attempt_contract_suppresses_triage() -> bool:
    """A train-phase reservation has no budget for implicit diagnostic generations."""
    return bool(os.environ.get("KLAB_ATTEMPT_BUDGET_LEDGER")) and (
        os.environ.get("KLAB_ATTEMPT_ACTIVITY") == "train"
    )


def _train_impl(
    config: RunConfig,
    *,
    resume: Path | None = None,
    promote: Path | None = None,
    add_ngram_memory: bool = False,
    extend_context: bool = False,
    append_data: bool = False,
    append_from_checkpoint: bool = False,
    extend_budget: Path | None = None,
    recover: Path | None = None,
    run_id: str | None = None,
    stop_after_step: int | None = None,
    worker_id: str | None = None,
    allow_runtime_drift: bool = False,
    allow_runtime_optimization: bool = False,
    stage_bundle: Path | None = None,
    cancel_path: Path | None = None,
    purpose: str = "training",
    experiment_id: str | None = None,
    attempt_id: str | None = None,
    dispatch_metadata: dict[str, object] | None = None,
    requested_config_override: dict[str, object] | None = None,
    max_wall_seconds: float | None = None,
    authorization: RuntimeAuthorization | None = None,
    resource_envelope: ResourceEnvelope | None = None,
    tokenizer_batch_documents: int = TOKENIZER_BATCH_DOCUMENTS,
    tokenizer_batch_source_bytes: int = TOKENIZER_BATCH_SOURCE_BYTES,
    proof_store: ProofStore | None = None,
    verification_mode: Literal["cold", "verified_reuse"] = "cold",
    require_prepared: bool = False,
    checkpoint_committed: Callable[[Path, CheckpointRecord], None] | None = None,
    spot_policy: SpotSafetyPolicy | None = None,
) -> str:

    require_model_runtime_allowed()
    require_authorization(config, authorization)
    validate_tokenizer_batch_limits(
        tokenizer_batch_documents, tokenizer_batch_source_bytes
    )

    def wall_expired() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    if max_wall_seconds is not None and (
        isinstance(max_wall_seconds, bool)
        or not isinstance(max_wall_seconds, (int, float))
        or not math.isfinite(max_wall_seconds)
        or max_wall_seconds <= 0
    ):
        raise ValueError("max_wall_seconds must be positive and finite")
    deadline = None if max_wall_seconds is None else time.monotonic() + max_wall_seconds
    _check_startup_cancel(cancel_path)
    with pilot_phase("input_validation"):
        if config.dataset.corpus_release_path is not None:
            portable = stage_bundle / "assets" if stage_bundle is not None else None
            if (
                portable is not None
                and (portable / "corpus" / "binding.json").is_file()
            ):
                from klab.workers.bundles import verify_portable_corpus_binding

                verify_portable_corpus_binding(config, portable)
            else:
                from klab.config.loading import load_config
                from klab.experiments.artifacts import Artifact, verify_artifact

                export_path = config.dataset.corpus_export_path
                if export_path is None:
                    raise ValueError("corpus release requires a verified export")

                export_config = load_config(export_path / "run.yaml")
                if (
                    export_config.dataset.corpus_export_path != export_path
                    or export_config.dataset.corpus_release_path
                    != config.dataset.corpus_release_path
                    or export_config.dataset.source != config.dataset.source
                    or export_config.dataset.revision != config.dataset.revision
                ):
                    raise ValueError("tokenizer export does not match selected corpus")
                tokenizer_path = config.tokenizer.path
                manifest_path = tokenizer_path.with_name("tokenizer_manifest.json")
                if manifest_path.is_symlink():
                    raise ValueError("tokenizer provenance must not be a symlink")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict) or not isinstance(
                    manifest.get("sha256"), str
                ):
                    raise ValueError("tokenizer provenance lacks a digest")
                verify_artifact(
                    Artifact(
                        kind="tokenizer",
                        version=1,
                        producer="klab",
                        identifier=tokenizer_path.parent.name,
                        sha256=manifest["sha256"],
                        path=str(tokenizer_path),
                    ),
                    export_path / "run.yaml",
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
                if manifest.get("vocab_size") != config.model.vocab_size:
                    raise ValueError("verified tokenizer vocabulary differs from model")
    _check_startup_cancel(cancel_path)
    operation_started = time.perf_counter()
    with ExitStack() as resources:
        if (experiment_id is None) != (attempt_id is None):
            raise ValueError("experiment_id and attempt_id must be supplied together")
        for identity in (experiment_id, attempt_id):
            if identity is not None and (
                not isinstance(identity, str)
                or not identity
                or identity in {".", ".."}
                or Path(identity).name != identity
            ):
                raise ValueError("execution identities must be nonempty identifiers")
        dispatch_decisions: tuple[dict[str, object], ...] = ()
        if dispatch_metadata is not None:
            if experiment_id is None or not isinstance(dispatch_metadata, dict):
                raise ValueError(
                    "dispatch metadata requires experiment and attempt identities"
                )
            if not {"spec_digest", "bundle_digest"} <= dispatch_metadata.keys() or (
                dispatch_metadata.keys()
                - {"spec_digest", "bundle_digest", "matrix", "plan"}
            ):
                raise ValueError("unsupported dispatch metadata fields")
            if not all(
                _is_sha256(dispatch_metadata[key])
                for key in ("spec_digest", "bundle_digest")
            ):
                raise ValueError("dispatch identities must be SHA-256 digests")
            matrix = dispatch_metadata.get("matrix")
            if matrix is not None and (
                not isinstance(matrix, dict)
                or set(matrix) != {"matrix_sha256", "coordinate"}
                or not _is_sha256(matrix["matrix_sha256"])
                or not isinstance(matrix["coordinate"], dict)
                or not matrix["coordinate"]
                or not all(
                    isinstance(key, str) and key and isinstance(value, str) and value
                    for key, value in matrix["coordinate"].items()
                )
            ):
                raise ValueError("invalid dispatch matrix identity")
            if dispatch_metadata.get("plan") is not None:
                from klab.workers.models import PlanMetadata

                plan_identity = PlanMetadata.model_validate(dispatch_metadata["plan"])
                if plan_identity.config_sha256 != config_sha256(
                    config.model_dump(mode="json")
                ):
                    raise ValueError(
                        "dispatch plan config differs from training config"
                    )
            if len(canonical_json(dispatch_metadata)) > 16 * 1024:
                raise ValueError("dispatch metadata exceeds 16 KiB")
            dispatch_decisions = (
                {"kind": "worker_dispatch", "schema_version": 1, **dispatch_metadata},
            )
        requested_config = config.model_dump(mode="json")
        if requested_config_override is not None:
            original = RunConfig.model_validate(requested_config_override)
            comparable = original.model_dump(mode="json")
            if original.runtime.backend == "auto":
                comparable["runtime"]["backend"] = config.runtime.backend
            if (
                original.runtime.precision == "auto"
                and config.runtime.precision != "auto"
            ):
                comparable["runtime"]["precision"] = "fp32"
            if config_sha256(comparable) != config_sha256(requested_config):
                raise ValueError(
                    "requested execution override changes scientific configuration"
                )
            requested_config = original.model_dump(mode="json")
        if config.training.parent_checkpoint is not None and not any(
            path is not None for path in (resume, recover, extend_budget)
        ):
            # A config that names its parent starts from it, exactly as
            # --promote does; continuing this run later resumes its own state.
            declared = checkpoint_selection(config.training.parent_checkpoint)
            if promote is not None and promote.resolve() != declared.resolve():
                raise ValueError(
                    "--promote differs from the config's training.parent_checkpoint"
                )
            promote = declared
        choices = [
            path
            for path in (resume, promote, recover, extend_budget)
            if path is not None
        ]
        if len(choices) > 1:
            raise ValueError(
                "resume, promote, recover, and extend-budget are mutually exclusive"
            )
        if purpose not in {"training", "smoke", "warmup"}:
            raise ValueError("invalid execution purpose")
        if purpose != "training" and (choices or stage_bundle is None):
            raise ValueError(
                "pilots require immutable staged inputs and fresh initialization"
            )
        staged = None
        staged_proofs = {}
        if stage_bundle is not None:
            from klab.staging import verify_stage_bundle

            with pilot_phase("stage_bundle_verification"):
                staged = verify_stage_bundle(
                    stage_bundle,
                    config,
                    purpose=purpose,
                    _proofs=staged_proofs,
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
        _check_startup_cancel(cancel_path)
        history = StageHistory()
        history.start(ExperimentStage.CONFIGURED)
        history.finish()
        history.start(ExperimentStage.INSPECTED)
        inventory = parameter_inventory(config)
        history.finish()
        history.start(ExperimentStage.VALIDATED)
        run_id = run_id or uuid.uuid4().hex
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be a single nonempty directory name")
        run = config.logging.root_dir / run_id
        if run.exists():
            report = existing_run_recovery_report(run)
            raise FileExistsError(
                f"run exists: {run_id}; explicit recovery creates a child; "
                f"read-only recovery report: {json.dumps(report, sort_keys=True)}"
            )
        spot_decision = (
            check_spot_capacity(spot_policy, run) if spot_policy is not None else None
        )
        if config.runtime.engine == "pytorch":
            engine = PyTorchEngine()
        elif config.runtime.engine == "mlx":
            from klab.engines.mlx import MLXEngine

            engine = MLXEngine()
        else:
            raise ValueError(f"unsupported execution engine: {config.runtime.engine}")
        with pilot_phase("runtime_initialization"):
            runtime = engine.validate(config, authorization=authorization)
        _check_startup_cancel(cancel_path)
        config = config.model_copy(
            update={
                "runtime": config.runtime.model_copy(
                    update={
                        "backend": runtime.backend,
                        "precision": "fp32"
                        if config.runtime.precision == "auto"
                        else config.runtime.precision,
                    }
                )
            }
        )
        current_source = source_identity()
        if promote is None and (
            resume is not None or recover is not None or extend_budget is not None
        ):
            continuation_root = (
                recover.resolve()
                if recover is not None
                else (resume or extend_budget).parent.parent.resolve()  # type: ignore[union-attr]
            )
            owned_manifest = continuation_root / "portability_manifest.json"
            if _portability_manifest_v2(owned_manifest) is not None:
                config = _bind_learned_inputs(config, owned_manifest)
        continuation_state = load_continuation(
            config,
            runtime,
            current_source,
            resume=resume,
            extend_budget=extend_budget,
            promote=promote,
            add_ngram_memory=add_ngram_memory,
            extend_context=extend_context,
            append_data=append_data,
            append_from_checkpoint=append_from_checkpoint,
            recover=recover,
            allow_runtime_drift=allow_runtime_drift,
            allow_runtime_optimization=allow_runtime_optimization,
        )
        snapshot = continuation_state.snapshot
        source_run = continuation_state.source_run
        continuation = continuation_state.kind
        artifact_source = (
            source_run if continuation == "RESUMED" and not append_data else None
        )
        if staged is not None and continuation != "RESUMED":
            artifact_source = Path(staged["assets_root"])
            config = config.model_copy(
                update={
                    "tokenizer": config.tokenizer.model_copy(
                        update={"path": artifact_source / "tokenizer.json"}
                    ),
                    "model": config.model.model_copy(
                        update={
                            "memory_package_path": artifact_source / "portable_package"
                            if config.model.memory_package_path is not None
                            else None,
                        }
                    ),
                }
            )
        _check_startup_cancel(cancel_path)
        preparation_started = time.perf_counter()
        seed_everything(config.seed, deterministic_cpu=config.training.deterministic)
        with pilot_phase("data_open"):
            if continuation == "RESUMED" and not append_data:
                assert source_run is not None
                data = _load_run_data(
                    source_run,
                    config,
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
            elif artifact_source is not None:
                if staged is not None and artifact_source == Path(
                    staged["assets_root"]
                ):
                    from klab.staging import _load_inventory_prepared

                    data = _load_inventory_prepared(
                        artifact_source,
                        staged_proofs,
                        byte_enabled=config.model.memory in {"byte", "portable"},
                    )
                else:
                    data = _load_run_data(
                        artifact_source,
                        config,
                        proof_store=proof_store,
                        verification_mode=verification_mode,
                    )
            else:
                tokenizer = load_tokenizer(config.tokenizer.path)
                data = prepare_data(
                    config,
                    tokenizer,
                    resource_envelope=resource_envelope,
                    tokenizer_batch_documents=tokenizer_batch_documents,
                    tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                    require_prepared=require_prepared,
                )
        if config.model.memory_token_mapping is not None:
            from klab.memory_tokenizer import validate_memory_token_map

            validate_memory_token_map(
                config.model.memory_token_mapping.model_dump(mode="json"),
                tokenizer_path=artifact_source / "tokenizer.json"
                if artifact_source
                else config.tokenizer.path,
            )
        load_fixed_panels(
            config,
            inputs_root=artifact_source,
            tokenizer_path=(artifact_source / "tokenizer.json")
            if artifact_source
            else None,
            training_manifest=data.manifest,
        )
        if purpose == "training":
            _check_training_storage(
                config,
                data,
                completed_steps=snapshot.step if continuation == "RESUMED" else 0,
                completed_tokens=snapshot.tokens_seen
                if continuation == "RESUMED"
                else 0,
                # Native continuation/stage verification selected this owned
                # input source. A resumed run need not retain caller-local files.
                tokenizer_path=(
                    artifact_source / "tokenizer.json"
                    if artifact_source is not None
                    else None
                ),
            )
            _check_startup_cancel(cancel_path)
        dataset = TokenBlockDataset(
            data.train,
            config.training.seq_len,
            data.train_byte_addresses,
            data.train_supervision,
            data.train_owner_ids,
            data.train_semantic_queries,
            data.train_semantic_mask,
        )
        matched = config.training.matched_stories
        if matched is not None:
            if (
                config.model.memory != "none"
                and not (config.model.memory == "ngram" and matched.unpadded_documents)
            ) or any(
                value is not None
                for value in (
                    data.train_supervision,
                    data.train_byte_addresses,
                    data.train_owner_ids,
                    data.train_semantic_queries,
                    data.train_semantic_mask,
                )
            ):
                raise ValueError(
                    "matched stories require plain all-token data; ngram memory requires unpadded whole documents"
                )
            story_tokenizer = load_tokenizer(
                artifact_source / "tokenizer.json"
                if artifact_source
                else config.tokenizer.path
            )
            eos = story_tokenizer.token_to_id("<eos>")
            if eos is None:
                raise ValueError("matched stories require an EOS token")
            dataset = MatchedStoryDataset(
                data.train,
                config.training.seq_len,
                eos,
                layout=matched.layout,
                targets_per_update=matched.targets_per_update,
                target_group_transition=(
                    matched.target_group_transition.model_dump()
                    if matched.target_group_transition
                    else None
                ),
                unpadded_documents=matched.unpadded_documents,
                frozen_update_ends=matched.frozen_update_ends,
            )
            if snapshot is not None and any(
                decision["kind"] == "target_group_transition"
                for decision in continuation_state.decisions
            ):
                from klab.training.continuation import verify_regroup_cursor

                verify_regroup_cursor(dataset, snapshot)
            if config.training.max_steps != dataset.horizon(config.training.max_tokens):
                raise ValueError(
                    "matched story max_steps must equal the exact shared budget horizon"
                )
        validation_dataset = TokenBlockDataset(
            data.validation,
            config.training.seq_len,
            data.validation_byte_addresses,
            data.validation_supervision,
            data.validation_owner_ids,
            data.validation_semantic_queries,
            data.validation_semantic_mask,
        )
        if matched is not None and matched.unpadded_documents:
            if any(
                value is not None
                for value in (
                    data.validation_supervision,
                    data.validation_byte_addresses,
                    data.validation_owner_ids,
                    data.validation_semantic_queries,
                    data.validation_semantic_mask,
                )
            ):
                raise ValueError(
                    "unpadded document validation requires plain all-token data"
                )
            validation_dataset = MatchedStoryDataset(
                data.validation,
                config.training.seq_len,
                eos,
                layout="story",
                targets_per_update=1,
                unpadded_documents=True,
            )
        if registered_objective(config.training.objective.kind).pairs:
            pad = load_tokenizer(
                artifact_source / "tokenizer.json"
                if artifact_source
                else config.tokenizer.path
            ).token_to_id("<pad>")
            dataset = PairedBlockDataset(
                data.train, config.training.seq_len, data.train_supervision, pad
            )
            validation_dataset = PairedBlockDataset(
                data.validation,
                config.training.seq_len,
                data.validation_supervision,
                pad,
            )
        if not len(dataset):
            raise ValueError("training data contains no supervised next-token targets")
        if not len(validation_dataset):
            raise ValueError(
                "validation data contains no supervised next-token targets"
            )
        calibration_identity = calibration_key(
            config, runtime, source_digest=str(current_source["sha256"])
        )
        observations = ExperimentStore.get_calibration(
            config.logging.root_dir, calibration_identity
        )
        if staged is not None and stage_bundle is not None:
            for pilot_purpose in ("smoke", "warmup"):
                observations.extend(
                    ExperimentStore.get_calibration(
                        stage_bundle / "pilots" / pilot_purpose,
                        calibration_identity,
                    )
                )
        memory_estimate = calibrated_estimate(config, runtime, inventory, observations)
        if memory_estimate.result == "LIKELY_TO_EXCEED":
            if config.runtime.memory.estimate_policy == "enforce":
                raise MemoryError(
                    "estimated training memory exceeds the safe ceiling; inspect --write-proposal before training"
                )
            _LOGGER.warning(
                "Estimated training memory exceeds the planning ceiling; "
                "estimate_policy=observe continues with the unchanged configuration"
            )
        offload_headroom = validate_offload_headroom(config, runtime, memory_estimate)
        history.finish()
        planning_started = time.perf_counter()
        forecast = runtime_forecast_planning(
            config.logging.root_dir,
            config,
            runtime,
            total_targets=config.training.max_tokens,
        )
        planning_seconds = time.perf_counter() - planning_started
        preparation_seconds = max(
            planning_started - preparation_started,
            0.0,
        )
        staged_forecast = staged.get("runtime_forecast") if staged else None
        warmup_forecast = (
            staged_forecast.get("warmup_calibrated")
            if isinstance(staged_forecast, dict)
            and staged_forecast.get("runtime_signature")
            == forecast["runtime_signature"]
            else None
        )
        runtime_forecast: dict[str, object] = {
            "schema_version": forecast["schema_version"],
            "runtime_signature": forecast["runtime_signature"],
            "planning": forecast["planning"],
            "warmup_calibrated": warmup_forecast,
            "live": None,
            "final_observed": None,
        }
        if (
            continuation == "RESUMED"
            and config.model.memory_package_path is not None
            and (
                config.training.portability_manifest_path is None
                or _portability_manifest_v2(config.training.portability_manifest_path)
                is None
            )
        ):
            assert source_run is not None
            owned_package = source_run / "portable_package"
            if not owned_package.exists():
                raise ValueError("resume run lacks immutable portable package")
            config = config.model_copy(
                update={
                    "model": config.model.model_copy(
                        update={"memory_package_path": owned_package}
                    )
                }
            )
        initial_weights = snapshot.model if snapshot is not None else None
        first_update_step = (
            snapshot.step + 1
            if snapshot is not None and continuation == "RESUMED"
            else 1
        )
        _check_startup_cancel(cancel_path)
        with pilot_phase("engine_initialization"):
            if isinstance(engine, PyTorchEngine):
                engine.initialize(
                    config,
                    initial_weights=initial_weights,
                    resume_portability=(
                        continuation == "RESUMED"
                        and _portability_manifest_v2(
                            config.training.portability_manifest_path
                        )
                        is not None
                    ),
                )
            else:
                engine.initialize(config, initial_weights=initial_weights)
        resources.callback(engine.close)
        step = tokens = 0
        cursor = BatchCursor()
        parent_run_id = continuation_state.parent_run_id
        lineage_best = continuation_state.lineage_best
        cumulative_wall = 0.0
        cumulative_updates = 0.0
        watermarks: dict[str, float] = {}
        prior_exposure = snapshot.exposure if snapshot is not None else None
        wall_complete = updates_complete = continuation != "RESUMED"
        if snapshot is not None and continuation == "RESUMED":
            engine.restore_training_state(
                EngineState(
                    snapshot.optimizer,
                    snapshot.rng or {},
                    snapshot.scaler,
                    snapshot.optimizer_parameter_names or [],
                )
            )
            step, tokens = snapshot.step, snapshot.tokens_seen
            cursor = BatchCursor(*snapshot.cursor)
            watermarks = snapshot.cadence or {}
            cumulative_wall = snapshot.cumulative_wall_seconds or 0.0
            cumulative_updates = snapshot.cumulative_update_seconds or 0.0
            prior_timing = (prior_exposure or {}).get("timing", {})
            # Defensively require native counters too, even though new exposure
            # timing is derived from those same counters at save time.
            wall_complete = (
                snapshot.cumulative_wall_seconds is not None
                and prior_timing.get("training_loop_elapsed_seconds") is not None
            )
            updates_complete = (
                snapshot.cumulative_update_seconds is not None
                and prior_timing.get("optimizer_update_seconds") is not None
            )
        if snapshot is not None:
            snapshot.model.clear()
            snapshot.optimizer.clear()
            snapshot.rng = None
            snapshot.scaler = None
        del initial_weights, snapshot
        initial_step = step
        initial_cumulative_updates = cumulative_updates
        if continuation == "RESUMED" and (
            step >= config.training.max_steps or tokens >= config.training.max_tokens
        ):
            raise ValueError("cannot resume a completed training budget")
        if stop_after_step is not None and stop_after_step <= step:
            raise ValueError("stop-after-step must exceed current step")
        if resource_envelope is not None:
            check_envelope(
                resource_envelope, workspace=run, rss_bytes=current_process_rss_bytes()
            )
        _check_startup_cancel(cancel_path)
        run.mkdir(parents=True)
        if spot_decision is not None:
            _atomic_json(run / "spot_safety.json", spot_decision)
        if authorization is not None:
            _atomic_json(run / "runtime_authorization.json", authorization.as_dict())
        inherited = None
        if config.checkpoint.retention is not None and continuation == "RESUMED":
            from klab.training.retention import inherited_milestones

            assert source_run is not None
            inherited = inherited_milestones(
                source_run, config.checkpoint.retention, step, tokens
            )
        manager = CheckpointManager(
            run,
            keep_periodic=config.checkpoint.keep_periodic,
            retention=config.checkpoint.retention,
            inherited_milestones=inherited,
        )
        resources.enter_context(manager.writer_lease())
        with pilot_phase("run_input_materialization"):
            artifacts = _copy_artifacts(run, config, data, artifact_source)
            fixed_panels = load_fixed_panels(
                config,
                inputs_root=run,
                tokenizer_path=run / "tokenizer.json",
                training_manifest=data.manifest,
            )
        exposure_base = exposure_context(
            dataset,
            artifacts,
            seed=config.seed,
            run_id=run_id,
            continuation=continuation,
            prior=prior_exposure,
        )
        owned_manifest = run / "portability_manifest.json"
        if _portability_manifest_v2(owned_manifest) is not None:
            config = _bind_learned_inputs(config, owned_manifest)
            from klab.research.portability import load_portability_manifest

            assert isinstance(engine, PyTorchEngine)
            engine.portability_run = load_portability_manifest(config)
            engine.config = config
        pilot_reports = tuple(staged["pilot_reports"]) if staged else ()
        stage_digest = str(staged["sha256"]) if staged else None
        if staged is not None:
            evidence = run / "stage_evidence.json"
            _atomic_json(
                evidence,
                {
                    "stage_bundle_sha256": stage_digest,
                    "pilot_reports": pilot_reports,
                    "stages": staged["stages"],
                },
            )
            artifacts = (
                *artifacts,
                ArtifactIdentity(
                    evidence.name,
                    sha256_file(evidence),
                    evidence.stat().st_size,
                ),
            )
        (run / "resolved_config.yaml").write_text(
            json.dumps(config.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"
        )
        manifest = RunManifest(
            run_id,
            config.name,
            runtime,
            requested_config,
            config.model_dump(mode="json"),
            architecture_sha256(config.model_dump(mode="json")),
            current_source,
            worker_id or socket.gethostname(),
            continuation,
            experiment_id=experiment_id,
            attempt_id=attempt_id,
            parent_run_id=parent_run_id,
            checkpoint_sha256=continuation_state.parent_checkpoint_sha256,
            purpose=purpose,
            stage_bundle_sha256=stage_digest,
            pilot_reports=pilot_reports,
            artifacts=artifacts,
            resource_decisions=continuation_state.decisions
            + dispatch_decisions
            + (
                (
                    {
                        "requested": "activation_offload",
                        "effective": True,
                        "reason": "conservative host headroom checked before model allocation",
                        "host_budget": offload_headroom,
                    },
                )
                if offload_headroom is not None
                else ()
            ),
        )
        manifest_digest = write_manifest(run / "manifest.json", manifest)
        manager.manifest_sha256 = manifest_digest
        store = ExperimentStore(config.logging.root_dir)
        resources.enter_context(store.session())
        store.create_run(
            run_id,
            config.model_dump(mode="json"),
            {
                "inspection": asdict(inventory),
                "runtime": runtime.as_dict(),
                "manifest_sha256": manifest_digest,
                "memory_estimate": asdict(memory_estimate),
                "purpose": purpose,
                "runtime_authorization": (
                    authorization.as_dict() if authorization is not None else None
                ),
                "resource_envelope": (
                    resource_envelope.model_dump(mode="json")
                    if resource_envelope is not None
                    else None
                ),
                "tokenizer_batch_documents": tokenizer_batch_documents,
                "tokenizer_batch_source_bytes": tokenizer_batch_source_bytes,
                "runtime_forecast": runtime_forecast,
            },
            parent_run_id,
            config_manifest=manifest.payload(),
        )
        store.register_manifest(run_id, manifest_digest, manifest.payload())
        for sequence, stage_record in enumerate(history.records):
            store.record_stage(
                run_id,
                sequence,
                stage_record.stage,
                stage_record.status,
                stage_record.step,
                stage_record.tokens_seen,
                stage_record.started_at,
                stage_record.finished_at,
                stage_record.payload,
            )
        for decision in continuation_state.decisions:
            store.log_event(
                run_id, step, tokens, cumulative_wall, str(decision["kind"]), decision
            )
        if config.external_components:
            # The config names each kind; this records which package and
            # version supplied the code that actually ran.
            store.log_event(
                run_id,
                step,
                tokens,
                cumulative_wall,
                "external_components",
                config.external_components,
            )
        objective_identity = getattr(engine, "objective_identity", None)
        if objective_identity is not None:
            # Which objective code ran and which frozen checkpoints it used.
            store.log_event(
                run_id,
                step,
                tokens,
                cumulative_wall,
                "training_objective",
                objective_identity,
            )
        started = time.perf_counter()
        interrupted = False
        latest_record: CheckpointRecord | None = None
        local_best: CheckpointRecord | None = None
        observed_peaks: dict[str, float] = {}
        phase_seconds: dict[str, float] = {
            "preparation": preparation_seconds,
            "planning": planning_seconds,
        }
        phase_counts: dict[str, int] = {"preparation": 1, "planning": 1}
        progress_samples: deque[tuple[float, int]] = deque([(0.0, tokens)], maxlen=121)
        last_optimizer_update_elapsed: float | None = None
        optimizer_observations: deque[tuple[int, float]] = deque(maxlen=1000)

        def add_phase_time(name: str, phase_started: float) -> None:
            phase_seconds[name] = phase_seconds.get(name, 0.0) + max(
                time.perf_counter() - phase_started, 0.0
            )
            phase_counts[name] = phase_counts.get(name, 0) + 1

        def live_summary() -> dict[str, object]:
            now = max(time.perf_counter() - started, 0.0)
            points = list(progress_samples)
            if last_optimizer_update_elapsed is not None and (
                not points or last_optimizer_update_elapsed > points[-1][0]
            ):
                points.append((last_optimizer_update_elapsed, tokens))
            summary = summarize_training_progress(
                points,
                total_targets=config.training.max_tokens,
                now=now,
                recent_window_seconds=60.0,
                long_window_seconds=600.0,
                stalled_after_seconds=300.0,
            )
            summary.update(
                recent_window_seconds=60.0,
                long_window_seconds=600.0,
                stalled_after_seconds=300.0,
            )
            if last_optimizer_update_elapsed is None:
                summary.update(
                    state="INITIALIZING",
                    eta_low_seconds=None,
                    eta_high_seconds=None,
                    eta_status="unavailable",
                    eta_basis=None,
                    optimizer_only_eta={
                        "low_seconds": None,
                        "high_seconds": None,
                        "status": "unavailable",
                        "basis": None,
                    },
                )
            return summary

        def persist_runtime_progress(record: dict[str, object]) -> None:
            raw_counters = record.get("raw_counters")
            counters = raw_counters if isinstance(raw_counters, dict) else {}
            record_step = counters.get("optimizer_step", step)
            record_targets = counters.get("completed_targets", tokens)
            record_elapsed = record.get("elapsed_seconds", 0.0)
            if type(record_step) is not int:
                record_step = step
            if type(record_targets) is not int:
                record_targets = tokens
            if isinstance(record_elapsed, (int, float)) and not isinstance(
                record_elapsed, bool
            ):
                elapsed = float(record_elapsed)
            else:
                elapsed = 0.0
            store.upsert_runtime_progress_snapshot(
                run_id,
                record_step,
                record_targets,
                cumulative_wall + elapsed,
                record,
            )

        runtime_forecast["live"] = live_summary()
        runtime_reporter = ProgressReporter(
            "training",
            operation_id=run_id,
            run_id=run_id,
            completed_work=tokens,
            total_work=config.training.max_tokens,
            unit="targets",
            raw_counters={
                "optimizer_step": step,
                "completed_targets": tokens,
                "optimizer_update_seconds": cumulative_updates,
            },
            derived=runtime_forecast,
            on_record=persist_runtime_progress,
        )
        resources.callback(runtime_reporter.close, failed=True, state="FAILED")

        def elapsed_seconds() -> float:
            return cumulative_wall + time.perf_counter() - started

        # A checkpoint being written off the training thread, with what its
        # bookkeeping needs once the write is verified. At most one is pending.
        pending_checkpoint: (
            tuple[Future[tuple[CheckpointRecord, float]], dict[str, object]] | None
        ) = None
        progress_written = float("-inf")
        # Update metrics waiting for one batched, synced store write.
        buffered_metrics: list[tuple[int, int, float, dict[str, float]]] = []
        metrics_flushed = time.perf_counter()

        def flush_metrics() -> None:
            nonlocal metrics_flushed
            metrics_flushed = time.perf_counter()
            if not buffered_metrics:
                return
            rows = buffered_metrics.copy()
            buffered_metrics.clear()
            try:
                store.log_metrics_batch(run_id, rows)
            finally:
                add_phase_time("reporting", metrics_flushed)

        def progress(status: str = "running", *, force: bool = False) -> None:
            """Publish progress.json; routine per-step updates are rate limited."""
            nonlocal progress_written
            now = time.perf_counter()
            # A small synced write waits behind a checkpoint's disk writeback,
            # so routine updates also hold back while one is being written.
            if (
                status == "running"
                and not force
                and (
                    now - progress_written < PROGRESS_INTERVAL_SECONDS
                    or pending_checkpoint is not None
                )
            ):
                return
            progress_written = now
            flush_metrics()
            live = live_summary()
            runtime_forecast["live"] = live
            reporter_state = (
                live["state"]
                if status == "running"
                else {
                    "completed": "COMPLETE",
                    "interrupted": "INTERRUPTED",
                    "failed": "FAILED",
                }.get(status, status.upper())
            )
            runtime_reporter.update(
                completed_work=tokens,
                total_work=config.training.max_tokens,
                unit="targets",
                raw_counters={
                    "optimizer_step": step,
                    "completed_targets": tokens,
                    "optimizer_update_seconds": cumulative_updates,
                },
                state=str(reporter_state),
                derived=runtime_forecast,
                emit=False,
            )
            reporting_started = time.perf_counter()
            try:
                _atomic_json(
                    run / "progress.json",
                    {
                        "manifest_sha256": manifest_digest,
                        "status": status,
                        "step": step,
                        "tokens_seen": tokens,
                        "cumulative_wall_seconds": elapsed_seconds(),
                        "cumulative_update_seconds": cumulative_updates,
                        "current_stage": asdict(history.current)
                        if history.current
                        else None,
                        "stages": [asdict(item) for item in history.records],
                        "latest": asdict(latest_record) if latest_record else None,
                        "best": asdict(local_best) if local_best else None,
                        "lineage_best": lineage_best.as_dict()
                        if lineage_best
                        else None,
                        "runtime_forecast": runtime_forecast,
                    },
                )
            finally:
                add_phase_time("reporting", reporting_started)

        def finish_stage(status: str = "complete", reason: str | None = None) -> None:
            record = history.finish(
                status, reason=reason, step=step, tokens_seen=tokens
            )
            payload = dict(record.payload or {})
            if reason is not None:
                payload["reason"] = reason
            store.record_stage(
                run_id,
                len(history.records) - 1,
                record.stage,
                record.status,
                step,
                tokens,
                record.started_at,
                record.finished_at,
                payload,
            )

        def enter_stage(
            stage: ExperimentStage, payload: dict[str, object] | None = None
        ) -> None:
            flush_metrics()
            if history.current is not None:
                finish_stage()
            current = history.start(
                stage, step=step, tokens_seen=tokens, payload=payload
            )
            store.log_event(
                run_id,
                step,
                tokens,
                elapsed_seconds(),
                "stage_started",
                asdict(current),
            )
            progress()

        spot_watermark = elapsed_seconds()

        checkpoint_writer = ThreadPoolExecutor(1, thread_name_prefix="klab-checkpoint")
        resources.callback(checkpoint_writer.shutdown, wait=True)
        background_checkpoints = (
            config.checkpoint.background
            and isinstance(engine, PyTorchEngine)
            and config.checkpoint.retention is None
        )
        # The best validation loss handed to a checkpoint, which leads
        # ``local_best`` while that checkpoint is still being written.
        best_validation_loss: float | None = None

        def commit_checkpoint(
            record: CheckpointRecord, context: dict[str, object]
        ) -> None:
            """Record a verified checkpoint in the run's books."""
            nonlocal latest_record, local_best, lineage_best
            at_step, at_tokens = context["step"], context["tokens"]
            validation, loss = context["validation"], context["loss"]
            latest_record = record
            if loss is not None:
                if (
                    local_best is None
                    or local_best.validation_loss is None
                    or loss < local_best.validation_loss
                ):
                    local_best = record
                lineage_best = choose_lineage_best(
                    lineage_best,
                    LineageBest(run_id, record.manifest_sha256, at_step, loss),
                )
                assert validation is not None
                reporting_started = time.perf_counter()
                try:
                    _write_validation_report(
                        run,
                        record,
                        validation,
                        config,
                        str(context["source_digest"]),
                        supervision_requires_mask(data.manifest),
                    )
                finally:
                    add_phase_time("reporting", reporting_started)
            reporting_started = time.perf_counter()
            try:
                store.record_checkpoint(run_id, record)
                if config.checkpoint.retention is not None:
                    store.log_event(
                        run_id,
                        at_step,
                        at_tokens,
                        elapsed_seconds(),
                        "checkpoint_retention",
                        json.loads((run / "checkpoints/retention.json").read_text()),
                    )
                store.log_event(
                    run_id,
                    at_step,
                    at_tokens,
                    elapsed_seconds(),
                    "checkpoint_saved",
                    asdict(record),
                )
            finally:
                add_phase_time("reporting", reporting_started)
            progress(force=True)
            if checkpoint_committed is not None:
                relay_started = time.perf_counter()
                try:
                    checkpoint_committed(run, record)
                finally:
                    add_phase_time("relay_transfer", relay_started)

        def drain_checkpoint(*, wait: bool = True, tolerate: bool = False) -> None:
            """Commit the pending checkpoint; ``wait`` blocks until it is written."""
            nonlocal pending_checkpoint
            if pending_checkpoint is None:
                return
            future, context = pending_checkpoint
            if not wait and not future.done():
                return
            waiting_started = time.perf_counter()
            pending_checkpoint = None
            try:
                record, write_seconds = future.result()
            except BaseException:
                if tolerate:
                    return
                raise
            finally:
                # Only the time training actually waited counts as checkpoint time.
                phase_seconds["checkpoint"] = phase_seconds.get("checkpoint", 0.0) + (
                    time.perf_counter() - waiting_started
                )
            phase_seconds["checkpoint_background"] = (
                phase_seconds.get("checkpoint_background", 0.0) + write_seconds
            )
            phase_counts["checkpoint_background"] = (
                phase_counts.get("checkpoint_background", 0) + 1
            )
            commit_checkpoint(record, context)

        def write_checkpoint(
            snapshot: TrainingSnapshot, loss: float | None
        ) -> tuple[CheckpointRecord, float]:
            write_started = time.perf_counter()
            record = manager.save(snapshot, loss)
            return record, time.perf_counter() - write_started

        def save_boundary(
            validation: dict[str, object] | None,
            *,
            operational_only: bool = False,
            wait: bool = True,
        ) -> None:
            """Checkpoint the current state.

            With ``wait`` false the state is copied to host memory and written
            off the training thread; its record is committed by a later
            ``drain_checkpoint``.
            """
            nonlocal watermarks, spot_watermark, pending_checkpoint
            nonlocal best_validation_loss
            # One write at a time: the previous checkpoint is the parent of
            # this one's lineage.
            drain_checkpoint()
            if spot_policy is not None:
                check_spot_capacity(spot_policy, run)
            enter_stage(ExperimentStage.CHECKPOINTED)
            elapsed = elapsed_seconds()
            spot_watermark = elapsed
            watermarks = _checkpoint_watermarks(
                step,
                tokens,
                elapsed,
                watermarks,
                operational_only=operational_only,
            )
            loss_value = validation.get("loss") if validation else None
            if validation is not None and not isinstance(loss_value, (int, float)):
                raise TypeError("validation report loss must be numeric")
            loss = float(loss_value) if isinstance(loss_value, (int, float)) else None
            if loss is not None and (
                best_validation_loss is None or loss < best_validation_loss
            ):
                best_validation_loss = loss
            phase_reporter = ProgressReporter(
                "checkpoint",
                operation_id=run_id,
                run_id=run_id,
                completed_work=0,
                total_work=1,
                unit="checkpoint_writes",
                raw_counters={"optimizer_step": step, "completed_targets": tokens},
            )
            context: dict[str, object] = {
                "step": step,
                "tokens": tokens,
                "validation": validation,
                "loss": loss,
                "source_digest": current_source["sha256"],
            }
            checkpoint_started = time.perf_counter()
            record: CheckpointRecord | None = None
            try:
                with pilot_phase(
                    "checkpoint", completed_steps=step, completed_targets=tokens
                ):
                    capture = partial(
                        _snapshot,
                        manager,
                        engine,
                        config,
                        run_id,
                        cursor,
                        step,
                        tokens,
                        watermarks,
                        loss,
                        source_digest=str(current_source["sha256"]),
                        parent_digest=continuation_state.parent_checkpoint_sha256,
                        wall_seconds=elapsed,
                        update_seconds=cumulative_updates,
                        lineage_best=lineage_best,
                        exposure=checkpoint_exposure(
                            exposure_base,
                            tokens=tokens,
                            cursor=cursor,
                            wall_seconds=elapsed,
                            invocation_seconds=checkpoint_started - operation_started,
                            update_seconds=cumulative_updates,
                            wall_complete=wall_complete,
                            updates_complete=updates_complete,
                        ),
                    )
                    snapshot = (
                        capture(detach=True)
                        if background_checkpoints and not wait
                        else None
                    )
                    if snapshot is not None:
                        pending_checkpoint = (
                            checkpoint_writer.submit(write_checkpoint, snapshot, loss),
                            context,
                        )
                    else:
                        if background_checkpoints and not wait:
                            store.log_event(
                                run_id,
                                step,
                                tokens,
                                elapsed,
                                "checkpoint_background_skipped",
                                {"reason": "insufficient host memory for a copy"},
                            )
                        snapshot = capture()
                        assert snapshot is not None
                        record = manager.save(snapshot, loss)
            except BaseException:
                add_phase_time("checkpoint_attempt", checkpoint_started)
                phase_reporter.close(failed=True)
                raise
            add_phase_time("checkpoint", checkpoint_started)
            phase_reporter.update(
                completed_work=1,
                total_work=1,
                raw_counters={"optimizer_step": step, "completed_targets": tokens},
            )
            phase_reporter.close(state="COMPLETE")
            if record is not None:
                commit_checkpoint(record, context)

        def finish_run(status: str, reason: str | None = None) -> None:
            # A failing run still waits for its writer, but keeps its own error.
            drain_checkpoint(tolerate=status == "failed")
            flush_metrics()
            final_reporting_started = time.perf_counter()
            reporting_before = phase_seconds.get("reporting", 0.0)
            if isinstance(engine, PyTorchEngine) and engine.portability_run is not None:
                audit_path = run / "portability_audit.json"
                audit_version = (
                    2 if engine.portability_run.payload.get("version") == 2 else 1
                )
                try:
                    audit: dict[str, object] = {
                        "format": "klab-portability-audit",
                        "version": audit_version,
                        "valid": True,
                        "run_status": status,
                        **engine.portability_audit(completed=status == "completed"),
                    }
                except Exception as error:  # noqa: BLE001
                    audit = {
                        "format": "klab-portability-audit",
                        "version": 1,
                        "valid": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    status = "failed"
                    reason = f"portability integrity audit failed: {error}"
                report_started = time.perf_counter()
                try:
                    _atomic_json(audit_path, audit)
                    store.log_event(
                        run_id,
                        step,
                        tokens,
                        elapsed_seconds(),
                        "portability_audit_recorded",
                        {
                            "valid": audit["valid"],
                            "sha256": sha256_file(audit_path),
                        },
                    )
                finally:
                    add_phase_time("reporting", report_started)
            if status == "failed" and history.current is not None:
                finish_stage("failed", reason)
            enter_stage(
                {
                    "completed": ExperimentStage.COMPLETE,
                    "interrupted": ExperimentStage.INTERRUPTED,
                    "failed": ExperimentStage.FAILED,
                }[status]
            )
            finish_stage("complete" if status == "completed" else status, reason)
            store.finish_run(
                run_id,
                status,
                str(manager.root / "latest.json") if latest_record else None,
            )
            store.log_event(
                run_id,
                step,
                tokens,
                elapsed_seconds(),
                f"run_{status}",
                {"reason": reason},
            )
            if latest_record is not None and status != "completed":
                store.log_event(
                    run_id,
                    step,
                    tokens,
                    elapsed_seconds(),
                    "resumable",
                    {
                        "checkpoint": latest_record.relative_path,
                        "checkpoint_digest": latest_record.manifest_sha256,
                        "resume_level": latest_record.resume_level,
                    },
                )
            if observed_peaks:
                store.record_calibration(
                    calibration_identity,
                    run_id,
                    {
                        "estimate": asdict(memory_estimate),
                        "observed": observed_peaks,
                        "peak_method": "native"
                        if "memory/device_peak_allocated_bytes" in observed_peaks
                        else "sampled_lower_bound",
                        "runtime": runtime.as_dict(),
                    },
                )
            if status == "completed" and len(optimizer_observations) >= 2:
                signature = runtime_forecast["runtime_signature"]
                if isinstance(signature, dict):
                    rates = [
                        targets / update_seconds
                        for targets, update_seconds in optimizer_observations
                    ]
                    store.record_calibration(
                        runtime_signature_key(signature),
                        run_id,
                        {
                            "kind": RUNTIME_OBSERVATION_KIND,
                            "schema_version": 1,
                            "runtime_signature": signature,
                            "optimizer_targets_per_second": median(rates),
                            "observed_at_utc": datetime.now(UTC).isoformat(
                                timespec="milliseconds"
                            ),
                            "calibration_update_count": len(optimizer_observations),
                            "optimizer_targets_observed": sum(
                                targets for targets, _ in optimizer_observations
                            ),
                            "optimizer_update_seconds_observed": sum(
                                seconds for _, seconds in optimizer_observations
                            ),
                            "first_update_discarded": True,
                            "sample_window": "last_1000_updates_after_initialization",
                        },
                    )
            nested_reporting = phase_seconds.get("reporting", 0.0) - reporting_before
            phase_seconds["reporting"] = phase_seconds.get("reporting", 0.0) + max(
                time.perf_counter() - final_reporting_started - nested_reporting, 0.0
            )
            phase_counts["reporting"] = phase_counts.get("reporting", 0) + 1
            optimizer_update_seconds = cumulative_updates - initial_cumulative_updates
            update_count = max(step - initial_step, 0)
            if update_count:
                phase_seconds["optimizer_update"] = optimizer_update_seconds
                phase_counts["optimizer_update"] = update_count
            end_to_end_seconds = max(time.perf_counter() - operation_started, 0.0)
            final_state = {
                "completed": "COMPLETE",
                "interrupted": "INTERRUPTED",
                "failed": "FAILED",
            }[status]
            final_live = live_summary()
            final_live.update(
                state=final_state,
                eta_low_seconds=0.0 if status == "completed" else None,
                eta_high_seconds=0.0 if status == "completed" else None,
                eta_status="complete" if status == "completed" else "unavailable",
                eta_basis="configured_stop" if status == "completed" else None,
                optimizer_only_eta={
                    "low_seconds": 0.0 if status == "completed" else None,
                    "high_seconds": 0.0 if status == "completed" else None,
                    "status": "complete" if status == "completed" else "unavailable",
                    "basis": "configured_stop" if status == "completed" else None,
                },
            )

            def phase_observation(name: str) -> dict[str, object]:
                count = phase_counts.get(name, 0)
                return {
                    "seconds": phase_seconds.get(name) if count else None,
                    "availability": "observed" if count else "unavailable",
                    "observation_count": count,
                }

            observed_components = (
                "preparation",
                "planning",
                "batch_assembly",
                "optimizer_update",
                "validation",
                "checkpoint",
                "relay_transfer",
                "reporting",
            )
            measured = sum(
                phase_seconds[name]
                for name in observed_components
                if phase_counts.get(name, 0)
            )
            observed = {
                "schema_version": 1,
                "operation_id": run_id,
                "run_id": run_id,
                "run_status": status,
                "reason": reason,
                "observed_at_utc": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "completed_steps": step,
                "completed_targets": tokens,
                "total_targets": config.training.max_tokens,
                "end_to_end_seconds": end_to_end_seconds,
                "cumulative_run_wall_seconds": elapsed_seconds(),
                "phases": {
                    **{name: phase_observation(name) for name in observed_components},
                    "validation_attempt": phase_observation("validation_attempt"),
                    "checkpoint_attempt": phase_observation("checkpoint_attempt"),
                    # Written off the training thread: overlaps optimizer updates
                    # and is not part of the measured wall-time sum.
                    "checkpoint_background": phase_observation("checkpoint_background"),
                    "evaluation": {
                        "seconds": None,
                        "availability": "not_observed",
                        "reason": "training records inline validation separately",
                    },
                    "generation": {
                        "seconds": None,
                        "availability": "not_observed",
                        "reason": "generation is a separate inference operation",
                    },
                    "unclassified_overhead": {
                        "seconds": max(end_to_end_seconds - measured, 0.0),
                        "availability": "derived_residual",
                    },
                },
            }
            runtime_forecast["live"] = final_live
            runtime_forecast["final_observed"] = observed
            store.log_event(
                run_id,
                step,
                tokens,
                elapsed_seconds(),
                "runtime_final_observation",
                observed,
            )
            progress(status)
            runtime_reporter.close(
                failed=status == "failed",
                state=final_state,
            )

        validation_pass_index = 0
        # Targets trained at the last validation pass, for a token cadence.
        evaluated_tokens = tokens

        def evaluation_batches(
            selected=validation_dataset,
            batch_size=None,
            limit=config.evaluation.max_batches,
            name="default",
        ):
            batch_size = (
                batch_size
                or config.evaluation.batch_size
                or config.training.micro_batch_size
            )
            progress_total = 0
            if current_pilot_progress() is not None:
                count = (len(selected) + batch_size - 1) // batch_size
                progress_total = count if limit is None else min(count, limit)
            for emitted, offset in enumerate(
                range(0, len(selected), batch_size),
                start=1,
            ):
                if limit is not None and emitted > limit:
                    break
                if wall_expired():
                    raise _WallTimeExpired
                records = [
                    selected.numpy_microblock(index)
                    for index in range(
                        offset,
                        min(
                            offset + batch_size,
                            len(selected),
                        ),
                    )
                ]
                inputs = np.stack([record[0] for record in records])
                if AttemptBudget.forward_allocation_active_from_environment():
                    AttemptBudget.reserve_forward_from_environment(
                        f"validation:{run_id}:{validation_pass_index}:{name}:{emitted}",
                        kind="operational_validation",
                        positions=int(inputs.size),
                    )
                yield Microbatch(
                    inputs,
                    np.stack([record[1] for record in records]),
                    _stack_optional(records, 2),
                    _stack_optional(records, 3),
                    _stack_optional(records, 4),
                    _stack_optional(records, 5),
                )
                if progress_total:
                    emit_pilot_progress(
                        "progress",
                        "validation",
                        current_step=step,
                        completed_steps=step,
                        completed_targets=tokens,
                        counter="items",
                        value=emitted,
                        total=progress_total,
                        subject=f"validation_step_{step}_consumed_batches",
                    )

        def evaluate_and_record() -> dict[str, object] | None:
            nonlocal validation_pass_index
            if wall_expired():
                return None
            validation_pass_index += 1
            enter_stage(ExperimentStage.EVALUATING)
            phase_reporter = ProgressReporter(
                "validation",
                operation_id=run_id,
                run_id=run_id,
                completed_work=0,
                total_work=1,
                unit="validation_passes",
                raw_counters={"optimizer_step": step, "completed_targets": tokens},
            )
            validation_started = time.perf_counter()
            try:
                with pilot_phase(
                    "validation",
                    current_step=step,
                    completed_steps=step,
                    completed_targets=tokens,
                ):
                    if fixed_panels:
                        named = {}
                        totals = []
                        for name, (panel, selected, identity) in fixed_panels.items():
                            measured = engine.evaluate(
                                evaluation_batches(
                                    selected, panel.batch_size, None, name
                                )
                            )
                            if (
                                measured.valid_targets != panel.supervised_targets
                                or measured.batches
                                != (panel.blocks + panel.batch_size - 1)
                                // panel.batch_size
                            ):
                                raise ValueError(
                                    "active evaluation did not complete panel coverage"
                                )
                            totals.append(measured)
                            named[name] = {
                                **measured.to_report(),
                                "nll_sum": measured.loss_sum,
                                "protocol": panel_protocol(panel),
                                "protocol_sha256": panel_protocol_sha256(panel),
                                "data_identity": identity,
                            }
                        combined = EvaluationResult(
                            math.fsum(item.loss_sum for item in totals),
                            sum(item.valid_targets for item in totals),
                            sum(item.batches for item in totals),
                        )
                        evaluation_result = {
                            **combined.to_report(),
                            "nll_sum": combined.loss_sum,
                            "panels": named,
                        }
                    else:
                        evaluation_result = engine.evaluate(
                            evaluation_batches()
                        ).to_report()
                result: dict[str, object] = {}
                result.update(evaluation_result)
            except _WallTimeExpired:
                add_phase_time("validation_attempt", validation_started)
                phase_reporter.close(state="INTERRUPTED")
                finish_stage("interrupted", "wall_time_limit")
                return None
            except BaseException:
                add_phase_time("validation_attempt", validation_started)
                phase_reporter.close(failed=True)
                raise
            add_phase_time("validation", validation_started)
            phase_reporter.update(
                completed_work=1,
                total_work=1,
                raw_counters={
                    "optimizer_step": step,
                    "completed_targets": tokens,
                    "validation_batches": result.get("batches"),
                },
            )
            phase_reporter.close(state="COMPLETE")
            loss_value = result.get("loss")
            if not isinstance(loss_value, (int, float)):
                raise TypeError("evaluation report loss must be numeric")
            metrics: dict[str, float] = {"validation/loss": float(loss_value)}
            perplexity = result.get("perplexity")
            if isinstance(perplexity, (int, float)):
                metrics["validation/perplexity"] = float(perplexity)
            for name, panel_result in result.get("panels", {}).items():
                metrics[f"validation/{name}/loss"] = float(panel_result["loss"])
                if panel_result["perplexity"] is not None:
                    metrics[f"validation/{name}/perplexity"] = float(
                        panel_result["perplexity"]
                    )
            reporting_started = time.perf_counter()
            try:
                completion_elapsed = elapsed_seconds()
                store.log_metrics(run_id, step, tokens, completion_elapsed, metrics)
                store.log_event(
                    run_id,
                    step,
                    tokens,
                    completion_elapsed,
                    "validation_completed",
                    result,
                )
            finally:
                add_phase_time("reporting", reporting_started)
            return result

        enter_stage(ExperimentStage.TRAINING)

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal interrupted
            interrupted = True

        old_int, old_term = (
            signal.signal(signal.SIGINT, request_stop),
            signal.signal(signal.SIGTERM, request_stop),
        )
        try:
            if interrupted or (cancel_path is not None and cancel_path.exists()):
                reason = (
                    "cancelled"
                    if cancel_path is not None and cancel_path.exists()
                    else "signal"
                )
                finish_run("interrupted", reason)
                return run_id
            initial_validation = evaluate_and_record()
            save_boundary(initial_validation, wait=False)
            if wall_expired():
                finish_run("interrupted", "wall_time_limit")
                return run_id
            enter_stage(ExperimentStage.TRAINING)
            emit_pilot_progress(
                "start", "training", completed_steps=step, completed_targets=tokens
            )
            while (
                step < config.training.max_steps and tokens < config.training.max_tokens
            ):
                if cancel_path is not None and cancel_path.exists():
                    interrupted = True
                if wall_expired():
                    interrupted = True
                if interrupted:
                    drain_checkpoint()
                    if latest_record is None or latest_record.step != step:
                        validation = evaluate_and_record()
                        if (
                            validation is None
                            and wall_expired()
                            and history.current is not None
                        ):
                            finish_stage("interrupted", "wall_time_limit")
                        save_boundary(validation)
                    reason = (
                        "wall_time_limit"
                        if wall_expired()
                        else "cancelled"
                        if cancel_path is not None and cancel_path.exists()
                        else "signal"
                    )
                    finish_run("interrupted", reason)
                    return run_id
                assembly_started = time.perf_counter()
                remaining = config.training.max_tokens - tokens
                if isinstance(dataset, MatchedStoryDataset):
                    epoch, group = cursor.epoch, cursor.next_block
                    if group == len(dataset):
                        epoch, group = epoch + 1, 0
                    records = list(dataset.update_rows(group, remaining))
                    valid_targets = sum(
                        int(np.count_nonzero(row[1] != -100)) for row in records
                    )
                    committed_cursor = BatchCursor(epoch, group + 1)
                else:
                    window_size = (
                        config.training.micro_batch_size
                        * config.training.gradient_accumulation
                    )
                    candidates: list[tuple[int, BatchCursor]] = []
                    next_cursor = cursor
                    while len(candidates) < window_size:
                        order = epoch_order(
                            len(dataset), config.seed, next_cursor.epoch
                        )
                        count = min(
                            window_size - len(candidates),
                            len(order) - next_cursor.next_block,
                        )
                        if count == 0:
                            next_cursor = BatchCursor(next_cursor.epoch + 1, 0)
                            continue
                        candidates.extend(
                            (
                                int(order[offset]),
                                BatchCursor(next_cursor.epoch, offset + 1),
                            )
                            for offset in range(
                                next_cursor.next_block, next_cursor.next_block + count
                            )
                        )
                        next_cursor = BatchCursor(
                            next_cursor.epoch, next_cursor.next_block + count
                        )
                    # Start reading the following update's blocks now, so that
                    # disk I/O overlaps with this update instead of delaying it.
                    upcoming = epoch_order(len(dataset), config.seed, next_cursor.epoch)
                    dataset.advise(
                        upcoming[
                            next_cursor.next_block : next_cursor.next_block
                            + window_size
                        ].tolist()
                    )
                    records = []
                    valid_targets = 0
                    committed_cursor = cursor
                    for index, candidate_cursor in candidates:
                        inputs, targets, addresses, owners, queries, query_mask = (
                            dataset.numpy_microblock(index)
                        )
                        labels = targets.copy()
                        available = int(np.count_nonzero(labels != -100))
                        allowed = min(remaining - valid_targets, available)
                        if allowed < available:
                            flat = labels.reshape(-1)
                            flat[np.flatnonzero(flat != -100)[allowed:]] = -100
                        records.append(
                            (inputs, labels, addresses, owners, queries, query_mask)
                        )
                        committed_cursor = candidate_cursor
                        valid_targets += allowed
                        if valid_targets == remaining:
                            break
                if valid_targets == 0:
                    raise RuntimeError(
                        "selected training window has no supervised targets"
                    )
                padded_documents = (
                    isinstance(dataset, MatchedStoryDataset)
                    and config.training.matched_stories.document_batching == "padded"
                )
                microbatches = (
                    []
                    if padded_documents
                    else [
                        Microbatch(
                            np.stack(
                                [
                                    record[0]
                                    for record in records[
                                        offset : offset
                                        + config.training.micro_batch_size
                                    ]
                                ]
                            ),
                            np.stack(
                                [
                                    record[1]
                                    for record in records[
                                        offset : offset
                                        + config.training.micro_batch_size
                                    ]
                                ]
                            ),
                            _stack_optional(
                                records[
                                    offset : offset + config.training.micro_batch_size
                                ],
                                2,
                            ),
                            _stack_optional(
                                records[
                                    offset : offset + config.training.micro_batch_size
                                ],
                                3,
                            ),
                            _stack_optional(
                                records[
                                    offset : offset + config.training.micro_batch_size
                                ],
                                4,
                            ),
                            _stack_optional(
                                records[
                                    offset : offset + config.training.micro_batch_size
                                ],
                                5,
                            ),
                        )
                        for offset in range(
                            0, len(records), config.training.micro_batch_size
                        )
                    ]
                )
                if (
                    isinstance(dataset, MatchedStoryDataset)
                    and config.training.matched_stories.document_batching == "padded"
                ):
                    from klab.data.document_batching import collate_document_update

                    microbatches = collate_document_update(
                        records,
                        width=config.training.micro_batch_size,
                        eos=dataset.eos,
                        max_length=config.training.seq_len,
                        padding_multiple=config.training.matched_stories.padding_multiple,
                        max_padded_slots=config.training.matched_stories.max_padded_slots,
                        max_attention_cells=config.training.matched_stories.max_attention_cells,
                    )
                add_phase_time("batch_assembly", assembly_started)
                retry_seconds = 0.0
                emit_pilot_progress(
                    "start",
                    "optimizer_update",
                    current_step=step + 1,
                    completed_steps=step,
                    completed_targets=tokens,
                )
                for retries in range(3):
                    update = engine.train_update(microbatches, step + 1, valid_targets)
                    retry_seconds += update.elapsed_seconds
                    if update.outcome == "APPLIED":
                        break
                else:
                    raise FloatingPointError(
                        "GradScaler overflow persisted after three attempts"
                    )
                metric_values = dict(update.metrics)
                metric_values["optimizer/overflow_retries"] = float(retries)
                metric_values["performance/step_seconds"] = retry_seconds
                metric_values["performance/tokens_per_second"] = valid_targets / max(
                    retry_seconds, 1e-9
                )
                if (
                    step + 1 > first_update_step
                    and update.committed_targets > 0
                    and retry_seconds > 0
                ):
                    optimizer_observations.append(
                        (update.committed_targets, retry_seconds)
                    )
                cumulative_updates += retry_seconds
                step, tokens = step + 1, tokens + update.committed_targets
                cursor = committed_cursor
                emit_pilot_progress(
                    "complete",
                    "optimizer_update",
                    current_step=step,
                    completed_steps=step,
                    completed_targets=tokens,
                )
                update_elapsed = max(time.perf_counter() - started, 0.0)
                last_optimizer_update_elapsed = update_elapsed
                if update_elapsed - progress_samples[-1][0] >= 5.0:
                    progress_samples.append((update_elapsed, tokens))
                elapsed = cumulative_wall + time.perf_counter() - started
                for name, value in metric_values.items():
                    if name.startswith("memory/") and "peak" in name:
                        observed_peaks[name] = max(observed_peaks.get(name, 0.0), value)
                unavailable = getattr(engine, "unavailable_memory_reasons", {})
                reporting_started = time.perf_counter()
                try:
                    if (
                        step % config.logging.every_steps == 0
                        or step == first_update_step
                        or step >= config.training.max_steps
                        or tokens >= config.training.max_tokens
                    ):
                        buffered_metrics.append((step, tokens, elapsed, metric_values))
                    if isinstance(engine, PyTorchEngine):
                        row_evidence = engine.portability_update_evidence()
                        if row_evidence is not None:
                            store.log_event(
                                run_id,
                                step,
                                tokens,
                                elapsed,
                                "learned_engram_gradient_rows",
                                row_evidence,
                            )
                    if unavailable and step == first_update_step:
                        store.log_event(
                            run_id,
                            step,
                            tokens,
                            elapsed,
                            "memory_measurements_unavailable",
                            unavailable,
                        )
                finally:
                    add_phase_time("reporting", reporting_started)
                progress()
                drain_checkpoint(wait=False)
                if cancel_path is not None and cancel_path.exists():
                    interrupted = True
                terminal = (
                    step >= config.training.max_steps
                    or tokens >= config.training.max_tokens
                    or step == stop_after_step
                    or interrupted
                )
                deadline_stopping = wall_expired()
                evaluation_due = not deadline_stopping and (
                    terminal
                    or (
                        config.evaluation.every_steps is not None
                        and step % config.evaluation.every_steps == 0
                    )
                    or (
                        config.evaluation.every_tokens is not None
                        and tokens - evaluated_tokens >= config.evaluation.every_tokens
                    )
                )
                validation = evaluate_and_record() if evaluation_due else None
                if validation is not None:
                    evaluated_tokens = tokens
                terminal = terminal or wall_expired()
                validation_loss = (
                    validation.get("loss") if validation is not None else None
                )
                new_best = (
                    config.checkpoint.save_on_new_best
                    and validation is not None
                    and isinstance(validation_loss, (int, float))
                    and (
                        best_validation_loss is None
                        or float(validation_loss) < best_validation_loss
                    )
                )
                scientific_checkpoint = (
                    _checkpoint_due(config, step, tokens, elapsed, watermarks)
                    or terminal
                    or new_best
                )
                operational_checkpoint = spot_policy is not None and spot_policy.due(
                    elapsed_seconds(), spot_watermark
                )
                if scientific_checkpoint or operational_checkpoint:
                    save_boundary(
                        validation,
                        operational_only=not scientific_checkpoint,
                        wait=terminal,
                    )
                if terminal:
                    status = (
                        "completed"
                        if step >= config.training.max_steps
                        or tokens >= config.training.max_tokens
                        else "interrupted"
                    )
                    finish_run(
                        status,
                        None
                        if status == "completed"
                        else "wall_time_limit"
                        if wall_expired()
                        else "cancelled"
                        if cancel_path is not None and cancel_path.exists()
                        else "signal"
                        if interrupted
                        else "stop_after_step",
                    )
                    emit_pilot_progress(
                        "complete",
                        "training",
                        current_step=step,
                        completed_steps=step,
                        completed_targets=tokens,
                    )
                    return run_id
                if (
                    history.current is not None
                    and history.current.stage != ExperimentStage.TRAINING
                ):
                    enter_stage(ExperimentStage.TRAINING)
            raise RuntimeError(
                "training stopped without completing a valid target window"
            )
        except BaseException as error:
            finish_run("failed", str(error))
            raise
        finally:
            # No completion event is emitted for an interrupted or failed run.
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
