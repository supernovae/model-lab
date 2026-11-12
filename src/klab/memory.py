"""Conservative memory accounting, policy proposals, and update measurements."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
from typing import Literal

import psutil
import torch
import yaml

from klab import host_capacity
from klab.canonical import canonical_json
from klab.config.models import RunConfig
from klab.model.inspection import (
    ParameterInventory,
    named_tensor_inventory,
    parameter_inventory,
)
from klab.runtime import RuntimeInfo, allocated_memory_bytes, process_rss_bytes
from klab.training.manifest import config_sha256

_MIB = 1024 * 1024


@dataclass(frozen=True)
class MemoryEstimate:
    """Reference-v1 estimate with disjoint byte categories."""

    version: str
    result: Literal["LIKELY_TO_FIT", "LIKELY_TO_EXCEED", "UNKNOWN"]
    peak_bytes: int
    capacity_ceiling_bytes: int | None
    resident_weights_bytes: int
    runtime_buffers_bytes: int
    gradients_bytes: int
    optimizer_bytes: int
    activations_bytes: int
    attention_working_bytes: int
    workspace_bytes: int
    headroom_bytes: int
    assumptions: tuple[str, ...]
    calibration_sample_count: int = 0


@dataclass(frozen=True)
class ResourceProposal:
    """A complete candidate config plus ordered, non-applied decisions."""

    config: dict[str, object]
    decisions: tuple[dict[str, object], ...]
    estimate: MemoryEstimate


def _capacity_ceiling(
    config: RunConfig, runtime: RuntimeInfo
) -> tuple[int | None, str | None]:
    """Return a single physical capacity ceiling, never inventing an extra pool."""
    fraction = config.runtime.memory.max_device_memory_fraction
    if runtime.backend == "cpu":
        if runtime.system_total_bytes is None or runtime.system_available_bytes is None:
            return None, "CPU total or available RAM is unavailable"
        return max(
            0,
            min(
                int(fraction * runtime.system_total_bytes),
                runtime.system_available_bytes,
            ),
        ), None
    if runtime.backend in {"cuda", "rocm", "xpu"}:
        if runtime.device_total_bytes is None or runtime.device_free_bytes is None:
            return None, "discrete device total or free memory is unavailable"
        return max(
            0,
            min(int(fraction * runtime.device_total_bytes), runtime.device_free_bytes),
        ), None
    if runtime.backend in {"mps", "metal"}:
        if any(
            value is None
            for value in (
                runtime.device_recommended_bytes,
                runtime.system_available_bytes,
                runtime.device_driver_allocated_bytes,
            )
        ):
            return (
                None,
                "unified-memory recommendation, driver allocation or available system RAM is unavailable",
            )
        assert runtime.device_recommended_bytes is not None
        assert runtime.system_available_bytes is not None
        assert runtime.device_driver_allocated_bytes is not None
        return max(
            0,
            min(
                int(fraction * runtime.device_recommended_bytes)
                - runtime.device_driver_allocated_bytes,
                runtime.system_available_bytes,
            ),
        ), None
    return None, f"unknown backend {runtime.backend!r} has no capacity policy"


def _adafactor_state_bytes(config: RunConfig) -> int:
    """Exact factor-storage estimate from canonical trainable tensor shapes.

    Matrix states use row and column factors; vectors retain a full second-moment
    tensor. Every parameter has a scalar step counter. This is intentionally
    independent of sparse direct-use accounting: inactive expert state is resident.
    """
    total = 0
    for spec in named_tensor_inventory(
        config.model,
        config.attention,
        trainable_parameters=config.training.trainable_parameters,
    ).values():
        if spec.alias_of is not None or not spec.trainable:
            continue
        if len(spec.shape) >= 2:
            factors = spec.numel // spec.shape[-1] + spec.numel // spec.shape[-2]
        else:
            factors = spec.numel
        total += 4 * (factors + 1)  # factor(s) and scalar step
    return total


def optimizer_state_bytes(
    config: RunConfig, inventory: ParameterInventory | None = None
) -> int:
    """Return native optimizer-state bytes without counting parameters twice."""
    if config.optimizer.name == "adamw":
        steps = sum(
            spec.trainable and spec.alias_of is None
            for spec in named_tensor_inventory(
                config.model,
                config.attention,
                trainable_parameters=config.training.trainable_parameters,
            ).values()
        )
        return (inventory or parameter_inventory(config)).trainable * 8 + steps * 4
    return _adafactor_state_bytes(config)


def registered_runtime_buffers_bytes(config: RunConfig) -> int:
    """Account for nonpersistent RoPE caches actually registered by blocks."""
    d, layers = config.model.hidden_dim, config.model.num_layers
    head_dim = d // config.model.num_heads
    # Attention keeps no [C,C] mask buffer: masks are built per call. RoPE owns
    # fp32 inverse frequency plus fp32 cos/sin caches built through the requested T.
    rope = 4 * (head_dim // 2 + config.training.seq_len * head_dim)
    mapping_bytes = (
        8 * config.model.vocab_size * max(1, len(config.model.memory_block_sites))
        if config.model.memory_token_mapping is not None
        else 0
    )
    return layers * rope + mapping_bytes


def _attention_reach(config: RunConfig) -> tuple[int, int, str]:
    """Keys scored per query, gathered K/V width per position, and the assumption.

    Score and probability tensors are sized by how many keys each query
    actually scores, so a bounded attention is not charged for a ``[T, T]``
    matrix it never builds.
    """
    attention, model, t = config.attention, config.model, config.training.seq_len
    if attention.kind == "local_selected":
        assert attention.block_size and attention.window_size
        assert attention.selected_blocks is not None
        slots = attention.window_size // attention.block_size + 1
        slots += attention.selected_blocks
        reach = min(t, slots * attention.block_size)
        # The learned selector's coarse branch also scores one summary per block.
        coarse = (
            -(-t // attention.block_size) + 1 if attention.selector == "learned" else 0
        )
        value_width = attention.latent_dim or model.hidden_dim
        return (
            reach + coarse,
            # Selected key and value blocks are copied once per query block.
            (reach // attention.block_size) * (model.hidden_dim + value_width),
            (
                "local_selected attention scores its local window and selected "
                "blocks only, plus one gathered copy of those keys and values"
            ),
        )
    if attention.kind == "sliding_window":
        assert attention.window_size is not None
        return (
            min(t, 2 * attention.window_size),
            0,
            "sliding-window attention scores at most two windows of keys per query",
        )
    if (
        attention.implementation == "sdpa"
        and attention.kind in {"dense", "mla"}
        and config.runtime.backend in {"cuda", "xpu"}
    ):
        return (
            0,
            0,
            (
                "fused scaled-dot-product attention on CUDA/XPU is assumed not to "
                "retain a [T, T] score matrix; verify with a measured warmup"
            ),
        )
    return (
        t,
        0,
        (
            "a full [T, T] score/probability allowance is kept for dense, MLA and "
            "block-sparse attention on this implementation and backend"
        ),
    )


def estimate_memory(
    config: RunConfig,
    runtime_info: RuntimeInfo,
    inventory: ParameterInventory,
    *,
    calibration: float | None = None,
) -> MemoryEstimate:
    """Estimate peak training memory without model or table materialization.

    ``calibration`` is a conservative multiplier derived from a matching native
    peak observation. Callers must not derive it from sampled MPS observations.
    """
    b, t, d, f, l, h, v = (
        config.training.micro_batch_size,
        config.training.seq_len,
        config.model.hidden_dim,
        config.model.ffn_dim,
        config.model.num_layers,
        config.model.num_heads,
        config.model.vocab_size,
    )
    model, attention = config.model, config.attention
    weights = inventory.total * 4
    buffers = registered_runtime_buffers_bytes(config)
    gradients = inventory.trainable * 4
    optimizer = optimizer_state_bytes(config, inventory)
    if model.ffn == "dense":
        per_block_nonattention = b * t * (4 * 6 * d + 4 * 3 * f)
    else:
        direct = model.experts_per_token + int(model.shared_expert)
        per_block_nonattention = (
            b * t * (4 * 6 * d + 4 * (3 * f * direct + 2 * model.num_experts))
        )
    projected_width = (
        2 * d + (attention.latent_dim or d)
        if attention.kind == "mla"
        else d + 2 * (model.num_kv_heads or h) * (d // h)
    )
    keys_per_query, gathered_width, attention_note = _attention_reach(config)
    per_block_attention = (
        4 * b * t * projected_width
        + 2 * 4 * b * h * t * keys_per_query
        + 4 * b * t * gathered_width
    )
    streams = (
        (len(model.memory_ngram_orders) or 1) * model.memory_hash_heads
        if model.memory == "ngram"
        else int(model.memory != "none")
    )
    logits_and_memory = 2 * 4 * b * t * v + 2 * 4 * b * t * streams * d
    if config.runtime.memory.activation_checkpointing.enabled:
        activations = (
            (l + 1) * 4 * b * t * d + per_block_nonattention + logits_and_memory
        )
        attention_working = per_block_attention
    else:
        activations = l * per_block_nonattention + logits_and_memory
        attention_working = l * per_block_attention
    workspace = max(
        64 * _MIB, int(0.1 * (weights + gradients + optimizer)) + 8 * b * t * v
    )
    subtotal = (
        weights
        + buffers
        + gradients
        + optimizer
        + activations
        + attention_working
        + workspace
    )
    headroom = int(0.15 * subtotal)
    peak = subtotal + headroom
    if calibration is not None:
        if not math.isfinite(calibration):
            raise ValueError("calibration multiplier must be finite")
        correction = math.ceil(peak * max(1.0, calibration)) - peak
        headroom += correction
        peak += correction

    physical_ceiling, missing = _capacity_ceiling(config, runtime_info)
    budget = config.runtime.memory.budget_bytes
    ceiling = (
        min(physical_ceiling, budget)
        if physical_ceiling is not None and budget is not None
        else physical_ceiling
        if physical_ceiling is not None
        else budget
    )
    if physical_ceiling is None:
        result: Literal["LIKELY_TO_FIT", "LIKELY_TO_EXCEED", "UNKNOWN"] = (
            "LIKELY_TO_EXCEED" if budget is not None and peak > budget else "UNKNOWN"
        )
    else:
        assert ceiling is not None
        result = "LIKELY_TO_FIT" if peak <= ceiling else "LIKELY_TO_EXCEED"
    assumptions = [
        "reference-v2: FP32 master parameters and gradients; autocast does not reduce resident model or optimizer state",
        "categories are disjoint: registered buffers, retained activations, attention working tensors, workspace, and headroom",
        attention_note,
    ]
    if config.runtime.memory.activation_offload.enabled:
        assumptions.append(
            "activation offload is unsupported here and receives no estimated saving"
        )
    if missing:
        assumptions.append(missing)
    if budget is not None and physical_ceiling is None:
        assumptions.append(
            "artificial budget can only demonstrate an exceedance; it cannot certify a physical fit"
        )
    return MemoryEstimate(
        "reference-v2",
        result,
        peak,
        ceiling,
        weights,
        buffers,
        gradients,
        optimizer,
        activations,
        attention_working,
        workspace,
        headroom,
        tuple(assumptions),
    )


def validate_offload_headroom(
    config: RunConfig, runtime: RuntimeInfo, estimate: MemoryEstimate
) -> dict[str, object] | None:
    """Check incremental host allocations before creating a run or a large model."""
    if not config.runtime.memory.activation_offload.enabled:
        return None
    try:
        memory = host_capacity.measure_memory()
        rss = process_rss_bytes()
    except (OSError, psutil.Error) as error:
        raise MemoryError(
            "activation offload requires readable host memory counters"
        ) from error
    staged = estimate.activations_bytes + estimate.attention_working_bytes
    # Allow both CPU snapshots and serialization buffers, without claiming that
    # all checkpoint implementations materialize this upper bound simultaneously.
    checkpoint = 2 * (
        estimate.resident_weights_bytes
        + estimate.runtime_buffers_bytes
        + estimate.optimizer_bytes
    )
    host_increment = math.ceil(1.15 * (staged + checkpoint))
    unified = runtime.backend in {"mps", "metal"}
    required = host_increment + (estimate.peak_bytes if unified else 0)
    ceiling = max(0, min(memory.available, memory.total - rss))
    if unified and estimate.capacity_ceiling_bytes is not None:
        ceiling = min(ceiling, estimate.capacity_ceiling_bytes)
    if required > ceiling:
        raise MemoryError(
            "activation offload host headroom is insufficient: "
            f"estimated incremental {required} bytes exceeds {ceiling} bytes; "
            f"current process RSS {rss}, staged activations {staged}, "
            f"checkpoint staging {checkpoint}"
        )
    return {
        "process_rss_bytes": rss,
        "available_host_bytes": memory.available,
        "staged_activation_bytes": staged,
        "checkpoint_staging_bytes": checkpoint,
        "incremental_peak_bytes": required,
        "ceiling_bytes": ceiling,
        "unified_memory": unified,
    }


def calibration_key(
    config: RunConfig, runtime: RuntimeInfo, *, source_digest: str
) -> str:
    """Stable key for exact execution settings; observations never cross settings."""
    model = config.model.model_dump(mode="json")
    # Local artifact paths do not affect allocation shape and must not fragment
    # otherwise identical warmup observations.
    model.pop("memory_package_path", None)
    payload = {
        "engine": runtime.engine,
        "backend": runtime.backend,
        "device": runtime.physical_device_id or runtime.device_name,
        "framework": runtime.framework_version,
        "source_identity_sha256": source_digest,
        "runtime_version": runtime.runtime_version,
        "driver_version": runtime.driver_version,
        "model": model,
        "attention": config.attention.model_dump(mode="json"),
        "precision": "fp32"
        if config.runtime.precision == "auto"
        else config.runtime.precision,
        "diagnostics": config.logging.architecture_diagnostics,
        "seq_len": config.training.seq_len,
        "micro_batch_size": config.training.micro_batch_size,
        "gradient_accumulation": config.training.gradient_accumulation,
        "optimizer": config.optimizer.name,
        "checkpointing": config.runtime.memory.activation_checkpointing.enabled,
        "offload": config.runtime.memory.activation_offload.enabled,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def calibration_multiplier(
    estimated_without_headroom: int,
    observed_native_peak: int | None,
    *,
    peak_method: str,
) -> float | None:
    """Return a non-decreasing correction only for trustworthy native peaks."""
    if (
        peak_method != "native"
        or observed_native_peak is None
        or estimated_without_headroom <= 0
        or observed_native_peak < 0
        or not math.isfinite(observed_native_peak)
    ):
        return None
    return max(1.0, observed_native_peak / estimated_without_headroom)


def calibrated_estimate(
    config: RunConfig,
    runtime: RuntimeInfo,
    inventory: ParameterInventory,
    observations: list[dict[str, object]],
) -> MemoryEstimate:
    """Apply only exact-key native observations; keep corrected categories additive."""
    multiplier = 1.0
    samples = 0
    for record in observations:
        observation = record.get("observation")
        if (
            not isinstance(observation, dict)
            or observation.get("peak_method") != "native"
        ):
            continue
        prior, observed = observation.get("estimate"), observation.get("observed")
        if not isinstance(prior, dict) or not isinstance(observed, dict):
            continue
        native_peaks = [
            observed.get(name)
            for name in (
                "memory/device_peak_allocated_bytes",
                "memory/device_peak_reserved_bytes",
            )
        ]
        peak = max(
            (
                value
                for value in native_peaks
                if type(value) in (int, float) and math.isfinite(value) and value >= 0
            ),
            default=None,
        )
        total, headroom = prior.get("peak_bytes"), prior.get("headroom_bytes")
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in (peak, total, headroom)
        ):
            continue
        correction = calibration_multiplier(
            total - headroom, peak, peak_method="native"
        )
        if correction is not None:
            multiplier = max(multiplier, correction)
            samples += 1
    estimate = estimate_memory(config, runtime, inventory, calibration=multiplier)
    return replace(
        estimate,
        calibration_sample_count=samples,
        assumptions=estimate.assumptions
        + (
            f"native calibration: {samples} exact-key observations, conservative multiplier {multiplier:g}",
        )
        if samples
        else estimate.assumptions,
    )


def _decision(
    name: str,
    before: MemoryEstimate,
    after: MemoryEstimate,
    reason: str,
    **extra: object,
) -> dict[str, object]:
    return {
        "requested": name,
        "reason": reason,
        "before_peak_bytes": before.peak_bytes,
        "after_peak_bytes": after.peak_bytes,
        **extra,
    }


def _replace_microbatch(
    config: RunConfig, micro_batch_size: int, effective_batch: int
) -> RunConfig:
    return config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "micro_batch_size": micro_batch_size,
                    "gradient_accumulation": effective_batch // micro_batch_size,
                }
            )
        }
    )


def plan_memory(
    config: RunConfig, runtime_info: RuntimeInfo, estimate: MemoryEstimate
) -> ResourceProposal:
    """Produce an explicit policy-ordered proposal, retaining measured baseline correction."""
    candidate, current = config, estimate
    decisions: list[dict[str, object]] = []
    policy = config.runtime.memory.policy
    effective = config.training.micro_batch_size * config.training.gradient_accumulation
    reference = estimate_memory(config, runtime_info, parameter_inventory(config))
    inherited_margin = max(0, estimate.peak_bytes - reference.peak_bytes)

    def candidate_estimate(proposed: RunConfig) -> MemoryEstimate:
        after = estimate_memory(proposed, runtime_info, parameter_inventory(proposed))
        if not inherited_margin:
            return after
        peak = after.peak_bytes + inherited_margin
        result = after.result
        if (
            after.capacity_ceiling_bytes is not None
            and peak > after.capacity_ceiling_bytes
        ):
            result = "LIKELY_TO_EXCEED"
        return replace(
            after,
            peak_bytes=peak,
            headroom_bytes=after.headroom_bytes + inherited_margin,
            result=result,
            assumptions=after.assumptions
            + (
                (
                    f"inherited baseline uncertainty margin: {inherited_margin} bytes; "
                    "this changed configuration has no matching calibration and requires warmup"
                ),
            ),
        )

    def consider(
        name: str, proposed: RunConfig, reason: str, **details: object
    ) -> None:
        nonlocal candidate, current
        if proposed == candidate:
            return
        proposed = RunConfig.model_validate(proposed.model_dump(mode="json"))
        after = candidate_estimate(proposed)
        decisions.append(
            _decision(
                name,
                current,
                after,
                reason + "; changed settings require their own warmup",
                **details,
            )
        )
        candidate, current = proposed, after

    def checkpointing(enabled: bool) -> None:
        if candidate.runtime.memory.activation_checkpointing.enabled == enabled:
            return
        if (
            enabled
            and runtime_info.engine != "pytorch"
            and "activation_checkpointing" not in runtime_info.tested_features
        ):
            decisions.append(
                {
                    "requested": "activation_checkpointing",
                    "effective": False,
                    "reason": "unsupported engine; no saving was assumed",
                }
            )
            return
        if enabled and candidate.runtime.compile_transformer_blocks:
            decisions.append(
                {
                    "requested": "activation_checkpointing",
                    "effective": False,
                    "reason": "compiled transformer blocks exclude activation "
                    "checkpointing; no saving was assumed",
                }
            )
            return
        memory = candidate.runtime.memory.model_copy(
            update={
                "activation_checkpointing": candidate.runtime.memory.activation_checkpointing.model_copy(
                    update={"enabled": enabled}
                ),
            }
        )
        consider(
            "activation_checkpointing",
            candidate.model_copy(
                update={
                    "runtime": candidate.runtime.model_copy(update={"memory": memory})
                }
            ),
            "trade backward recomputation against retained activations",
            effective=enabled,
        )

    if policy == "fast":
        checkpointing(False)
        if candidate.runtime.memory.activation_offload.enabled:
            memory = candidate.runtime.memory.model_copy(
                update={
                    "activation_offload": candidate.runtime.memory.activation_offload.model_copy(
                        update={"enabled": False}
                    ),
                }
            )
            consider(
                "activation_offload",
                candidate.model_copy(
                    update={
                        "runtime": candidate.runtime.model_copy(
                            update={"memory": memory}
                        )
                    }
                ),
                "avoid transfer overhead under fast policy",
                effective=False,
            )
        for micro in range(effective, candidate.training.micro_batch_size, -1):
            if effective % micro:
                continue
            proposed = _replace_microbatch(candidate, micro, effective)
            after = candidate_estimate(proposed)
            if after.result == "LIKELY_TO_FIT":
                consider(
                    "micro_batch_size",
                    proposed,
                    "larger divisible microbatch preserves effective batch",
                    effective=micro,
                    effective_batch_size=effective,
                )
                break
    else:
        if policy == "max_fit" and current.result != "LIKELY_TO_FIT":
            for precision in ("bf16", "fp16"):
                supported = precision in runtime_info.tested_precisions
                decisions.append(
                    _decision(
                        "precision",
                        current,
                        current,
                        "validated compute precision but saved-tensor savings are not instrumented; no precision change or saving was assumed"
                        if supported
                        else "unsupported: precision lacks a successful runtime probe; no saving was assumed",
                        candidate=precision,
                        effective=candidate.runtime.precision,
                        supported=supported,
                        scientifically_significant=True,
                        applied=False,
                    )
                )
        if policy == "low_memory" or current.result != "LIKELY_TO_FIT":
            micros = (
                [1]
                if policy == "low_memory"
                else [
                    value
                    for value in range(candidate.training.micro_batch_size - 1, 0, -1)
                    if effective % value == 0
                ]
            )
            for micro in micros:
                consider(
                    "micro_batch_size",
                    _replace_microbatch(candidate, micro, effective),
                    "preserve effective batch while reducing activation dimensions",
                    effective=micro,
                    effective_batch_size=effective,
                )
                if current.result == "LIKELY_TO_FIT":
                    break
            if policy == "low_memory" or current.result != "LIKELY_TO_FIT":
                checkpointing(True)

        if policy == "max_fit" and current.result != "LIKELY_TO_FIT":
            decisions.append(
                {
                    "requested": "selective_checkpointing",
                    "effective": False,
                    "reason": "deferred; no saving was assumed",
                }
            )
            for length in sorted(
                set(config.runtime.memory.allowed_sequence_lengths), reverse=True
            ):
                if length >= candidate.training.seq_len:
                    continue
                proposed = candidate.model_copy(
                    update={
                        "training": candidate.training.model_copy(
                            update={"seq_len": length}
                        ),
                    }
                )
                consider(
                    "sequence_length",
                    proposed,
                    "explicit allowed_sequence_lengths candidate",
                    effective=length,
                    scientifically_significant=True,
                    tokens_per_update=effective * length,
                )
                if current.result == "LIKELY_TO_FIT":
                    break

        if candidate.runtime.memory.activation_offload.enabled or (
            policy in {"low_memory", "max_fit"} and current.result != "LIKELY_TO_FIT"
        ):
            decisions.append(
                {
                    "requested": "activation_offload",
                    "effective": candidate.runtime.memory.activation_offload.enabled,
                    "reason": "no validated capacity model for this backend; no saving was assumed",
                }
            )

        if policy == "max_fit" and current.result != "LIKELY_TO_FIT":
            for optimizer in config.runtime.memory.allowed_optimizers:
                if optimizer == candidate.optimizer.name:
                    continue
                if runtime_info.engine != "pytorch":
                    decisions.append(
                        {
                            "requested": "optimizer",
                            "candidate": optimizer,
                            "effective": candidate.optimizer.name,
                            "reason": "unsupported engine; no saving was assumed",
                        }
                    )
                    continue
                payload = candidate.model_dump(mode="json")
                payload["optimizer"] = {
                    "name": optimizer,
                    **{
                        name: getattr(candidate.optimizer, name)
                        for name in (
                            "peak",
                            "floor",
                            "warmup_steps",
                            "weight_decay",
                            "state_offload",
                        )
                    },
                }
                proposed = RunConfig.model_validate(payload)
                after = candidate_estimate(proposed)
                if after.peak_bytes >= current.peak_bytes:
                    continue
                consider(
                    "optimizer",
                    proposed,
                    "explicit optimizer alternative changes update semantics; Adafactor uses relative step-size scaling",
                    effective=optimizer,
                    scientifically_significant=True,
                )
                if current.result == "LIKELY_TO_FIT":
                    break
            if current.result != "LIKELY_TO_FIT":
                for name in (
                    "optimizer_state_offload",
                    "expert_offload",
                    "parameter_offload",
                ):
                    decisions.append(
                        {
                            "requested": name,
                            "effective": False,
                            "reason": "deferred; no saving was assumed",
                        }
                    )
    return ResourceProposal(
        candidate.model_dump(mode="json"), tuple(decisions), current
    )


def write_resource_proposal(
    proposal: ResourceProposal, path: Path
) -> tuple[Path, Path]:
    """Publish a durable report first and its hash-bound YAML commit marker last."""
    config = RunConfig.model_validate(proposal.config).model_dump(mode="json")
    yaml_content = yaml.safe_dump(config, sort_keys=False, allow_unicode=True).encode(
        "utf-8"
    )
    report_content = (
        canonical_json(
            {
                "format_version": 1,
                "yaml_sha256": hashlib.sha256(yaml_content).hexdigest(),
                "config_sha256": config_sha256(config),
                "estimate": asdict(proposal.estimate),
                "decisions": proposal.decisions,
            }
        )
        + b"\n"
    )
    report_path = path.with_suffix(path.suffix + ".decisions.json")
    for destination in (path, report_path):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite proposal artifact: {destination}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_files: dict[Path, Path] = {}
    created: list[Path] = []
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        for destination, content in (
            (report_path, report_content),
            (path, yaml_content),
        ):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=path.parent
            )
            temporary_files[destination] = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        # Link, never replace: a racing writer cannot be overwritten. Make the
        # report directory entry durable before exposing a usable config path.
        for destination in (report_path, path):
            os.link(temporary_files[destination], destination)
            created.append(destination)
            os.fsync(directory_fd)
    except BaseException:
        for destination in reversed(created):
            destination.unlink(missing_ok=True)
        if created:
            os.fsync(directory_fd)
        raise
    finally:
        try:
            for temporary in temporary_files.values():
                temporary.unlink(missing_ok=True)
        finally:
            os.close(directory_fd)
    return path, report_path


class MemoryMonitor:
    """Bounded per-update aggregates; sampled peaks are not native high-water marks."""

    def __init__(
        self,
        device: object,
        *,
        runtime_info: RuntimeInfo | None = None,
        sample_interval_seconds: float | None = None,
    ) -> None:
        if sample_interval_seconds is not None and sample_interval_seconds < 0.01:
            raise ValueError("sample_interval_seconds must be at least 0.01 seconds")
        self.device = device
        self.runtime_info = runtime_info
        self.sample_interval_seconds = sample_interval_seconds
        self.unavailable_reasons: dict[str, str] = {}
        self._latest: dict[str, int | float | str] = {}
        self._peaks: dict[str, float] = {}
        self._sampling_seconds = 0.0
        self._native_peaks_valid = False
        self._lock, self._stop = Lock(), Event()
        self._observer: Thread | None = None
        self._observer_started: float | None = None

    @property
    def _device_type(self) -> str:
        return str(getattr(self.device, "type", self.device))

    def _native(self, name: str) -> int | None:
        kind = self._device_type
        try:
            if kind == "metal":
                import mlx.core as mx

                method = {
                    "allocated": mx.get_active_memory,
                    "peak_allocated": mx.get_peak_memory,
                    "cache": mx.get_cache_memory,
                }.get(name)
                return int(method()) if method is not None else None
            if kind in {"cuda", "xpu"}:
                api = getattr(torch, kind, None)
                method = getattr(
                    api,
                    {
                        "reserved": "memory_reserved",
                        "peak_allocated": "max_memory_allocated",
                        "peak_reserved": "max_memory_reserved",
                    }.get(name, ""),
                    None,
                )
                return int(method(self.device)) if method else None
            if kind == "mps" and name == "driver_allocated":
                method = getattr(torch.mps, "driver_allocated_memory", None)
                return int(method()) if method else None
        except ImportError, RuntimeError, AttributeError:
            return None
        return None

    def begin_update(self) -> None:
        self._stop_observer()
        with self._lock:
            self._latest.clear()
            self._peaks.clear()
            self._sampling_seconds = 0.0
            self.unavailable_reasons.clear()
        self._native_peaks_valid = False
        if self._device_type in {"cuda", "xpu"}:
            api = getattr(torch, self._device_type, None)
            reset = getattr(api, "reset_peak_memory_stats", None)
            try:
                if reset is not None:
                    reset(self.device)
                    self._native_peaks_valid = True
            except RuntimeError, AttributeError:
                pass
        elif self._device_type == "metal":
            try:
                import mlx.core as mx

                mx.reset_peak_memory()
                self._native_peaks_valid = True
            except ImportError, RuntimeError, AttributeError:
                pass
        self.sample("begin")
        if self.sample_interval_seconds is not None:
            self._stop.clear()
            self._observer_started = monotonic()
            self._observer = Thread(
                target=self._observe,
                name="klab-memory-observer",
                daemon=True,
            )
            self._observer.start()

    def _observe(self) -> None:
        while not self._stop.wait(self.sample_interval_seconds):
            self.sample("observer")

    def sample(self, phase: str) -> dict[str, int | float | str]:
        started = monotonic()
        sample: dict[str, int | float | str] = {
            "phase": phase,
            "monotonic_seconds": started,
        }
        try:
            sample["memory/process_rss_bytes"] = process_rss_bytes()
        except OSError, RuntimeError, psutil.Error:
            self.unavailable_reasons["memory/process_rss_bytes"] = (
                "process RSS API failed"
            )
        try:
            sample["memory/system_available_bytes"] = int(
                host_capacity.measure_memory().available
            )
        except OSError, RuntimeError, psutil.Error:
            self.unavailable_reasons["memory/system_available_bytes"] = (
                "system RAM API failed"
            )
        try:
            allocated = (
                self._native("allocated")
                if self._device_type == "metal"
                else allocated_memory_bytes(self.device)  # type: ignore[arg-type]
            )
        except RuntimeError, AttributeError:
            allocated = None
        if allocated is None:
            self.unavailable_reasons["memory/device_allocated_bytes"] = (
                f"{self._device_type} allocated-memory API unavailable or failed"
            )
        else:
            sample["memory/device_allocated_bytes"] = allocated
        native_metrics = [
            ("memory/device_reserved_bytes", "reserved"),
            ("memory/device_peak_allocated_bytes", "peak_allocated"),
            ("memory/device_peak_reserved_bytes", "peak_reserved"),
            ("memory/driver_allocated_bytes", "driver_allocated"),
        ]
        if self._device_type == "metal":
            native_metrics.append(("memory/device_cache_bytes", "cache"))
        for metric, native in native_metrics:
            if native.startswith("peak_") and not self._native_peaks_valid:
                self.unavailable_reasons[metric] = (
                    f"{self._device_type} per-update native peak reset unavailable or failed"
                )
                continue
            value = self._native(native)
            if value is None:
                self.unavailable_reasons[metric] = (
                    f"{self._device_type} native {native} API unavailable or failed"
                )
            else:
                sample[metric] = value
        with self._lock:
            self._latest = sample
            for metric, value in sample.items():
                if metric.startswith("memory/"):
                    self._peaks[metric] = max(
                        self._peaks.get(metric, 0.0), float(value)
                    )
            self._sampling_seconds += monotonic() - started
        return sample

    def _stop_observer(self) -> float:
        if self._observer is None:
            return 0.0
        self._stop.set()
        self._observer.join()
        self._observer = None
        assert self._observer_started is not None
        seconds = monotonic() - self._observer_started
        self._observer_started = None
        return seconds

    def end_update(self) -> dict[str, float]:
        observer_seconds = self._stop_observer()
        self.sample("end")
        with self._lock:
            result = {
                metric: float(value)
                for metric, value in self._latest.items()
                if metric.startswith("memory/")
            }
            result.update(
                {
                    metric: peak
                    for metric, peak in self._peaks.items()
                    if "peak" in metric
                }
            )
            if "memory/process_rss_bytes" in self._peaks:
                result["memory/process_peak_rss_bytes"] = self._peaks[
                    "memory/process_rss_bytes"
                ]
            if (
                self._device_type == "mps"
                and "memory/device_allocated_bytes" in self._peaks
            ):
                result["memory/device_sampled_peak_bytes"] = self._peaks[
                    "memory/device_allocated_bytes"
                ]
            result["memory/sampling_seconds"] = self._sampling_seconds
        result["memory/observer_active_seconds"] = observer_seconds
        return result

    def close(self) -> None:
        self._stop_observer()
