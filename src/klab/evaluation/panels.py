"""One immutable development panel per invocation, using native loss evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from pydantic import Field

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import FixedValidationPanel
from klab.data.packing import (
    canonical_tokenizer_sha256,
)
from klab.evaluation.inference import load_run
from klab.evaluation.language_model import evaluate
from klab.evaluation.panel_data import select_panel as _dataset
from klab.training.attempt_budget import AttemptBudget

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class HeldoutPanel(FixedValidationPanel):
    schema_version: Literal[1]
    purpose: Literal["development"]
    checkpoint_sha256: Digest


def run_panel(
    run_id: str,
    runs_dir: Path,
    checkpoint: str,
    declaration: Path,
    output: Path,
    *,
    backend: str | None = None,
    authorization: Any = None,
) -> dict[str, Any]:
    """Preserve incomplete evidence; output must be a new file outside input roots."""
    panel = HeldoutPanel.model_validate_json(declaration.read_text(encoding="utf-8"))
    prepared_root = panel.prepared_path
    if not prepared_root.is_absolute():
        prepared_root = declaration.resolve().parent / prepared_root
    for root in ((runs_dir / run_id).resolve(), prepared_root.resolve()):
        if output.resolve().is_relative_to(root):
            raise ValueError("panel output must be outside the immutable input roots")
    protocol = panel.model_dump(
        mode="json", exclude={"prepared_path", "checkpoint_sha256"}
    )
    protocol_sha = canonical_digest(protocol)
    result: dict[str, Any] = {
        "source": "heldout_panel",
        "version": 1,
        "panel": panel.name,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "requested_checkpoint_sha256": panel.checkpoint_sha256,
        "identity": None,
        "data_identity": None,
        "status": "incomplete",
        "evidence_complete": False,
        "accounting_complete": False,
        "reason": "evaluation did not complete",
        "loss": None,
        "nll_sum": 0.0,
        "valid_targets": 0,
        "batches": 0,
        "attempted_forward_positions": 0,
        "evaluation_seconds": 0.0,
    }
    started = time.monotonic()
    evaluation_started = None
    with output.open("x", encoding="utf-8") as handle:

        def persist() -> None:
            handle.seek(0)
            handle.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())

        persist()
        try:
            dataset, result["data_identity"] = _dataset(panel, prepared_root)
            AttemptBudget.require_fixed_evaluation_allocation_from_environment(
                fixed_positions=panel.blocks * panel.sequence_length,
                total_positions=panel.blocks * panel.sequence_length,
                calls=0,
                tokens=0,
            )
            loaded = load_run(
                run_id,
                runs_dir,
                checkpoint,
                backend,
                authorization=authorization,
                expected_checkpoint_sha256=panel.checkpoint_sha256,
            )
            result["identity"] = loaded.identity
            result["comparison_group"] = hashlib.sha256(
                canonical_json(
                    {
                        "protocol_sha256": protocol_sha,
                        "runtime": loaded.identity["runtime"],
                    }
                )
            ).hexdigest()
            if (
                loaded.engine is not None
                or loaded.config.model.memory not in {"none", "ngram"}
                or loaded.config.model.ffn != "dense"
                or loaded.config.dataset.allocation_manifest_path is not None
                or loaded.config.model.semantic_memory_dim is not None
                or not isinstance(loaded.device, torch.device)
            ):
                raise ValueError(
                    "named panels require dense-FFN PyTorch with no memory or token ngram memory"
                )
            if (
                loaded.identity["tokenizer_sha256"] != panel.tokenizer_file_sha256
                or canonical_tokenizer_sha256(loaded.tokenizer)
                != panel.tokenizer_canonical_sha256
            ):
                raise ValueError("panel checkpoint tokenizer identity mismatch")
            if panel.sequence_length > loaded.config.model.max_seq_len:
                raise ValueError("panel sequence length exceeds checkpoint context")

            def before_batch(positions: int) -> None:
                if (
                    result["attempted_forward_positions"] + positions
                    > panel.max_forward_positions
                ):
                    raise ValueError("panel forward budget exhausted")
                if AttemptBudget.forward_allocation_active_from_environment():
                    AttemptBudget.reserve_forward_from_environment(
                        f"panel:{panel.checkpoint_sha256}:{protocol_sha}:{result['batches']}",
                        kind="fixed_profile",
                        positions=positions,
                    )
                result["attempted_forward_positions"] += positions

            def progress(total: float, targets: int, batches: int) -> None:
                result.update(nll_sum=total, valid_targets=targets, batches=batches)

            evaluation_started = time.monotonic()
            metrics = evaluate(
                loaded.model,
                dataset,
                batch_size=panel.batch_size,
                max_batches=math.ceil(panel.blocks / panel.batch_size),
                device=loaded.device,
                before_batch=before_batch,
                progress=progress,
            )
            if metrics["valid_targets"] != panel.supervised_targets:
                raise ValueError("native evaluation did not complete panel coverage")
            result.update(metrics)
            result.update(status="complete", evidence_complete=True, reason=None)
        except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - retain failed panel evidence
            result.update(
                status="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                reason=f"{type(error).__name__}: {error}",
            )
        finally:
            if evaluation_started is not None:
                result["evaluation_seconds"] = time.monotonic() - evaluation_started
            result["accounting_complete"] = True
            result["elapsed_seconds"] = time.monotonic() - started
            persist()
    return result
