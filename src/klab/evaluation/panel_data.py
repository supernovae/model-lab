"""Authenticated fixed-panel inputs shared by standalone and active evaluation."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np

from klab.canonical import canonical_digest
from klab.config.models import FixedValidationPanel, RunConfig
from klab.data.packing import (
    TokenBlockDataset,
    canonical_tokenizer_sha256,
    load_prepared_data,
)
from klab.data.tokenizer import load_tokenizer
from klab.training.manifest import sha256_file


def select_panel(
    panel: FixedValidationPanel, root: Path
) -> tuple[TokenBlockDataset, dict]:
    prepared = load_prepared_data(root, byte_enabled=False)
    manifest = prepared.manifest
    if manifest["manifest_sha256"] != panel.prepared_manifest_sha256:
        raise ValueError("panel prepared manifest identity mismatch")
    if manifest["tokenizer_sha256"] != panel.tokenizer_canonical_sha256:
        raise ValueError("panel prepared tokenizer identity mismatch")
    if manifest.get("allocation") is not None:
        raise ValueError("named panels do not support allocation sidecars")
    dataset = TokenBlockDataset(
        prepared.validation,
        panel.sequence_length,
        supervision=prepared.validation_supervision,
    )
    if len(dataset) < panel.blocks:
        raise ValueError("panel has insufficient eligible blocks (possibly empty)")
    dataset.block_indices = dataset.block_indices[: panel.blocks].copy()
    targets = sum(
        int(np.count_nonzero(dataset.numpy_block(i)[1] != -100))
        for i in range(len(dataset))
    )
    if targets != panel.supervised_targets:
        raise ValueError("panel supervised-target coverage differs from declaration")
    return dataset, {
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "validation_sha256": manifest["validation"]["sha256"],
        "supervision": manifest["supervision"],
        "packing_version": manifest["packing_version"],
        "source_identity_sha256": manifest.get("source_identity_sha256"),
    }


def panel_protocol(panel: FixedValidationPanel) -> dict:
    return panel.model_dump(mode="json", exclude={"prepared_path"})


def panel_protocol_sha256(panel: FixedValidationPanel) -> str:
    return canonical_digest(panel_protocol(panel))


def load_fixed_panels(
    config: RunConfig,
    *,
    inputs_root: Path | None = None,
    tokenizer_path: Path | None = None,
    training_manifest: dict | None = None,
) -> dict:
    if not config.evaluation.panels:
        return {}
    tokenizer_path = tokenizer_path or config.tokenizer.path
    file_sha = sha256_file(tokenizer_path)
    canonical_sha = canonical_tokenizer_sha256(load_tokenizer(tokenizer_path))
    result = {}
    manifests = set()
    validation_arrays = set()
    for panel in config.evaluation.panels:
        if (
            file_sha != panel.tokenizer_file_sha256
            or canonical_sha != panel.tokenizer_canonical_sha256
        ):
            raise ValueError("panel training tokenizer identity mismatch")
        path = (
            inputs_root / "panels" / panel.name if inputs_root else panel.prepared_path
        )
        dataset, identity = select_panel(panel, path)
        validation_sha = identity["validation_sha256"]
        if (
            panel.prepared_manifest_sha256 in manifests
            or validation_sha in validation_arrays
        ):
            raise ValueError(
                "fixed panels must use distinct prepared validation inputs"
            )
        if (
            training_manifest is not None
            and validation_sha == training_manifest["train"]["sha256"]
        ):
            raise ValueError("fixed panel validation aliases the training array")
        manifests.add(panel.prepared_manifest_sha256)
        validation_arrays.add(validation_sha)
        result[panel.name] = (panel, dataset, identity)
    return result


def copy_fixed_panels(
    config: RunConfig,
    destination: Path,
    *,
    inputs_root: Path | None = None,
    training_manifest: dict | None = None,
) -> None:
    """Copy only native prepared inputs; run/stage inventories bind every byte."""
    if not config.evaluation.panels:
        return
    tokenizer_path = (inputs_root or destination) / "tokenizer.json"
    load_fixed_panels(
        config,
        inputs_root=inputs_root,
        tokenizer_path=tokenizer_path,
        training_manifest=training_manifest,
    )
    for panel in config.evaluation.panels:
        source = (
            inputs_root / "panels" / panel.name if inputs_root else panel.prepared_path
        )
        prepared = load_prepared_data(source, byte_enabled=False)
        target = destination / "panels" / panel.name
        target.mkdir(parents=True)
        for name in ("manifest.json", *prepared.receipt.proofs):
            shutil.copy2(source / name, target / name)
    load_fixed_panels(
        config,
        inputs_root=destination,
        tokenizer_path=destination / "tokenizer.json",
        training_manifest=training_manifest,
    )


def panel_coverage(evaluation: dict) -> tuple[int, int]:
    panels = evaluation.get("panels", [])
    return (
        sum(p["supervised_targets"] for p in panels),
        sum((p["blocks"] + p["batch_size"] - 1) // p["batch_size"] for p in panels),
    )


def validate_panel_results(payload: dict, declarations: list, artifacts: dict) -> None:
    """Check named coverage and weighting against the authenticated run contract."""
    from klab.engines.base import EvaluationResult

    panels = [FixedValidationPanel.model_validate(item) for item in declarations]
    reports = payload.get("panels")
    if not isinstance(reports, dict) or set(reports) != {p.name for p in panels}:
        raise ValueError("fixed-panel report names differ from run contract")
    totals = []
    for panel in panels:
        report = reports[panel.name]
        if not isinstance(report, dict):
            raise TypeError("fixed-panel result must be an object")
        if report.get("protocol") != panel_protocol(panel) or report.get(
            "protocol_sha256"
        ) != panel_protocol_sha256(panel):
            raise ValueError("fixed-panel report protocol mismatch")
        prefix = f"panels/{panel.name}/"
        owned = {
            key: value for key, value in artifacts.items() if key.startswith(prefix)
        }
        if not {prefix + "manifest.json", prefix + "validation.npy"} <= owned.keys():
            raise ValueError("fixed-panel inputs are not owned run artifacts")
        if any(
            payload["identities"].get(key) != digest for key, digest in owned.items()
        ):
            raise ValueError("fixed-panel artifact identity mismatch")
        identity = report.get("data_identity", {})
        if (
            identity.get("prepared_manifest_sha256") != panel.prepared_manifest_sha256
            or identity.get("validation_sha256") != artifacts[prefix + "validation.npy"]
            or panel.tokenizer_file_sha256 != artifacts.get("tokenizer.json")
        ):
            raise ValueError("fixed-panel data identity mismatch")
        measured = EvaluationResult(
            report["nll_sum"], report["valid_targets"], report["batches"]
        )
        if (
            type(measured.loss_sum) not in (int, float)
            or not math.isfinite(measured.loss_sum)
            or type(measured.valid_targets) is not int
            or type(measured.batches) is not int
            or measured.valid_targets != panel.supervised_targets
            or measured.batches
            != (panel.blocks + panel.batch_size - 1) // panel.batch_size
            or any(
                report.get(key) != value for key, value in measured.to_report().items()
            )
        ):
            raise ValueError("fixed-panel result coverage or loss mismatch")
        totals.append(measured)
    aggregate = EvaluationResult(
        math.fsum(item.loss_sum for item in totals),
        sum(item.valid_targets for item in totals),
        sum(item.batches for item in totals),
    )
    if payload.get("nll_sum") != aggregate.loss_sum or any(
        payload.get(key) != value for key, value in aggregate.to_report().items()
    ):
        raise ValueError("fixed-panel weighted aggregate mismatch")
