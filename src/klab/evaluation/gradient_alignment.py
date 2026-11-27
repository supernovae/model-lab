"""Compare two verified panel loss gradients at one saved checkpoint."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from klab.canonical import canonical_digest
from klab.config.models import RunConfig
from klab.data.packing import canonical_tokenizer_sha256
from klab.evaluation.inference import InferenceRun, load_run
from klab.evaluation.language_model import evaluate
from klab.evaluation.panel_data import select_panel
from klab.evaluation.panels import HeldoutPanel, read_heldout_panel, validate_panel_run
from klab.model.inspection import parameter_inventory
from klab.training.attempt_budget import AttemptBudget

DEFAULT_GRADIENT_BYTES = 64 * 1024**2
_REDUCTION_ELEMENTS = 65_536


def _measure_gradient(loaded: InferenceRun, panel: HeldoutPanel, root: Path, row: dict):
    dataset, row["data_identity"] = select_panel(panel, root)

    def before_batch(positions: int) -> None:
        attempted = row["attempted_forward_positions"]
        if attempted + positions > panel.max_forward_positions:
            raise ValueError("panel forward budget exhausted")
        if AttemptBudget.forward_allocation_active_from_environment():
            AttemptBudget.reserve_forward_from_environment(
                f"gradient:{panel.checkpoint_sha256}:{row['side']}:{row['protocol_sha256']}:{attempted}",
                kind="fixed_profile",
                positions=positions,
            )
        row["attempted_forward_positions"] += positions

    def progress(total: float, targets: int, batches: int) -> None:
        row.update(nll_sum=total, valid_targets=targets, batches=batches)

    def backward(loss: torch.Tensor) -> None:
        (loss / panel.supervised_targets).backward()

    metrics = evaluate(
        loaded.model,
        dataset,
        batch_size=panel.batch_size,
        max_batches=math.ceil(panel.blocks / panel.batch_size),
        device=loaded.device,
        before_batch=before_batch,
        progress=progress,
        backward=backward,
    )
    if metrics["valid_targets"] != panel.supervised_targets:
        raise ValueError("gradient evaluation did not complete panel coverage")
    row.update(metrics)


def _gradient_moments(left, right) -> dict[str, float | None]:
    dot = left_square = right_square = 0.0
    for left_tensor, right_tensor in zip(left, right, strict=True):
        if left_tensor is None and right_tensor is None:
            continue
        left_flat = None if left_tensor is None else left_tensor.reshape(-1)
        right_flat = None if right_tensor is None else right_tensor.reshape(-1)
        size = (left_flat if left_flat is not None else right_flat).numel()
        for start in range(0, size, _REDUCTION_ELEMENTS):
            stop = start + _REDUCTION_ELEMENTS
            a = None if left_flat is None else left_flat[start:stop].double()
            b = None if right_flat is None else right_flat[start:stop].double()
            if a is not None:
                left_square += float(torch.dot(a, a))
            if b is not None:
                right_square += float(torch.dot(b, b))
            if a is not None and b is not None:
                dot += float(torch.dot(a, b))
    if not all(math.isfinite(value) for value in (dot, left_square, right_square)):
        raise ValueError("nonfinite panel gradients")
    left_norm, right_norm = math.sqrt(left_square), math.sqrt(right_square)
    scale = left_norm * right_norm
    return {
        "dot_product": dot,
        "left_norm": left_norm,
        "right_norm": right_norm,
        "cosine": max(-1.0, min(1.0, dot / scale)) if scale else None,
    }


def run_gradient_alignment(
    run_id: str,
    runs_dir: Path,
    checkpoint: str,
    left_declaration: Path,
    right_declaration: Path,
    output: Path,
    *,
    max_gradient_bytes: int = DEFAULT_GRADIENT_BYTES,
    authorization: Any = None,
) -> dict[str, Any]:
    left, left_root = read_heldout_panel(left_declaration)
    right, right_root = read_heldout_panel(right_declaration)
    if left.checkpoint_sha256 != right.checkpoint_sha256:
        raise ValueError("gradient panels must pin the same checkpoint")
    if max_gradient_bytes <= 0:
        raise ValueError("gradient storage bound must be positive")
    for root in (runs_dir / run_id, left_root, right_root):
        if output.resolve().is_relative_to(root.resolve()):
            raise ValueError(
                "gradient output must be outside the immutable input roots"
            )
    result: dict[str, Any] = {
        "source": "heldout_panel_gradient_alignment",
        "version": 1,
        "precision": "fp32",
        "max_gradient_bytes": max_gradient_bytes,
        "status": "incomplete",
        "evidence_complete": False,
        "reason": "gradient evaluation did not complete",
        "requested_checkpoint_sha256": left.checkpoint_sha256,
        "panels": {},
    }
    started = time.monotonic()
    with output.open("x", encoding="utf-8") as handle:

        def persist() -> None:
            handle.seek(0)
            handle.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())

        persist()
        try:

            def check_config(config: RunConfig) -> None:
                if (
                    config.runtime.engine != "pytorch"
                    or config.attention_kinds != {"dense"}
                    or config.ffn_kinds != {"dense"}
                    or config.model.memory not in {"none", "ngram"}
                    or config.training.objective.kind != "next_token"
                ):
                    raise ValueError(
                        "gradient alignment requires dense next-token PyTorch"
                    )
                planned_bytes = 8 * parameter_inventory(config).total
                result["planned_gradient_storage_bytes"] = planned_bytes
                if planned_bytes > max_gradient_bytes:
                    raise ValueError("gradient storage exceeds --gradient-max-bytes")

            positions = (
                left.blocks * left.sequence_length
                + right.blocks * right.sequence_length
            )
            AttemptBudget.require_fixed_evaluation_allocation_from_environment(
                fixed_positions=positions, total_positions=positions, calls=0, tokens=0
            )
            loaded = load_run(
                run_id,
                runs_dir,
                checkpoint,
                "cpu",
                authorization=authorization,
                expected_checkpoint_sha256=left.checkpoint_sha256,
                hash_training_state=False,
                hash_run_data=False,
                check_config=check_config,
            )
            result["identity"] = loaded.identity
            tokenizer_sha = canonical_tokenizer_sha256(loaded.tokenizer)
            for panel in (left, right):
                validate_panel_run(panel, loaded, tokenizer_sha)
            assert (
                isinstance(loaded.device, torch.device) and loaded.device.type == "cpu"
            )
            parameters = tuple(p for p in loaded.model.parameters() if p.requires_grad)
            if not parameters or any(p.dtype != torch.float32 for p in parameters):
                raise ValueError(
                    "gradient alignment requires fp32 trainable parameters"
                )
            gradient_bytes = 2 * sum(p.numel() * p.element_size() for p in parameters)
            result["gradient_storage_bytes"] = gradient_bytes
            if gradient_bytes > max_gradient_bytes:
                raise ValueError("gradient storage exceeds --gradient-max-bytes")

            gradients = []
            for side, panel, root in (
                ("left", left, left_root),
                ("right", right, right_root),
            ):
                protocol = panel.model_dump(
                    mode="json", exclude={"prepared_path", "checkpoint_sha256"}
                )
                row = {
                    "side": side,
                    "protocol": protocol,
                    "protocol_sha256": canonical_digest(protocol),
                    "attempted_forward_positions": 0,
                    "valid_targets": 0,
                }
                result["panels"][side] = row
                _measure_gradient(loaded, panel, root, row)
                gradients.append(tuple(p.grad for p in parameters))
                loaded.model.zero_grad(set_to_none=True)
            result.update(_gradient_moments(*gradients))
            result.update(status="complete", evidence_complete=True, reason=None)
        except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - retain failed evidence
            result.update(
                status="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                reason=f"{type(error).__name__}: {error}",
            )
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            persist()
    return result
