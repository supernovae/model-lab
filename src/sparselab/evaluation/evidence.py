"""Summarize only integrity-bound local evaluation observations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from sparselab.canonical import canonical_digest
from sparselab.data.packing import (
    TokenBlockDataset,
    load_prepared_data,
    supervision_requires_mask,
)
from sparselab.data.verification import VerifiedFile, _receipt_from_proofs, verify_file
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import read_manifest
from sparselab.verification_proofs import (
    ProofStore,
    VerificationMode,
    file_binding,
    validate_mode,
)


def _inside(root: Path, path: Path) -> bool:
    try:
        return root.resolve() in path.resolve().parents
    except OSError:
        return False


def _validated_artifacts(
    run: Path,
    manifest: dict[str, object],
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> tuple[dict[str, str], dict[str, VerifiedFile]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("run manifest lacks artifact inventory")
    verified: dict[str, str] = {}
    proofs: dict[str, VerifiedFile] = {}
    for item in artifacts:
        if not isinstance(item, dict):
            raise TypeError("invalid artifact inventory")
        name, digest = item.get("relative_path"), item.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str):
            raise TypeError("invalid artifact identity")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"artifact integrity failure: {name}")
        path = run / name
        if not _inside(run, path) or path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact integrity failure: {name}")
        proof = verify_file(
            path,
            expected_sha256=digest,
            proof_store=proof_store,
            verification_mode=verification_mode,
            binding=file_binding(
                path,
                digest,
                kind="run_artifact",
                identifier=name,
                closure={"run_id": manifest["run_id"], "inventory": artifacts},
            ),
        )
        if Path(name).parts[0] == "data" and path.suffix == ".npy":
            proofs[path.name] = proof
        verified[name] = digest
    return verified, proofs


def validate_held_out_report(
    path: Path,
    checkpoints: dict[str, dict[str, object]],
    artifacts: dict[str, str],
    manifest: dict[str, object],
    expected_targets: int,
    expected_batches: int,
) -> tuple[dict[str, object] | None, str | None]:
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return None, "report is not an object"
        digest = payload.pop("sha256", None)
        if not isinstance(digest, str) or canonical_digest(payload) != digest:
            return None, "report hash mismatch"
        checkpoint = payload.get("checkpoint")
        bound = checkpoints.get(checkpoint) if isinstance(checkpoint, str) else None
        if bound is None or bound.get("digest") != payload.get("checkpoint_sha256"):
            return None, "checkpoint binding mismatch"
        if payload.get("kind") != "held_out_validation_v2":
            return None, "unknown evaluation protocol"
        identities = payload.get("identities")
        if (
            not isinstance(identities, dict)
            or identities.get("protocol") != "next-token-cross-entropy-v1"
        ):
            return None, "evaluation identity mismatch"
        for required in ("data/validation.npy", "tokenizer.json"):
            if (
                required not in artifacts
                or identities.get(required) != artifacts[required]
            ):
                return None, f"{required} identity mismatch"
        for optional in (
            "data/validation_supervision.npy",
            "data/validation_byte_addresses.npy",
        ):
            if (
                optional in artifacts
                and identities.get(optional) != artifacts[optional]
            ):
                return None, f"{optional} identity mismatch"
            if optional in identities and optional not in artifacts:
                return None, f"unexpected {optional} identity"
        if identities.get("source_identity_sha256") != manifest.get(
            "source_identity", {}
        ).get("sha256"):
            return None, "source identity mismatch"
        if payload.get("step") != bound.get("step") or payload.get(
            "tokens_seen"
        ) != bound.get("tokens_seen"):
            return None, "checkpoint counter mismatch"
        loss = payload.get("loss")
        valid_targets = payload.get("valid_targets")
        if (
            not isinstance(loss, (int, float))
            or not math.isfinite(loss)
            or not isinstance(valid_targets, int)
            or valid_targets <= 0
        ):
            return None, "invalid loss or target count"
        checkpoint_loss = bound.get("validation_loss")
        if not isinstance(checkpoint_loss, (int, float)) or loss != checkpoint_loss:
            return None, "checkpoint loss mismatch"
        perplexity = payload.get("perplexity")
        reason = payload.get("perplexity_unavailable_reason")
        if perplexity is None:
            if not isinstance(reason, str) or loss <= math.log(
                float.fromhex("0x1.fffffffffffffp+1023")
            ):
                return None, "invalid perplexity unavailability reason"
        elif (
            not isinstance(perplexity, (int, float))
            or not math.isfinite(perplexity)
            or reason is not None
            or loss > math.log(float.fromhex("0x1.fffffffffffffp+1023"))
            or not math.isclose(perplexity, math.exp(loss), rel_tol=1e-12)
        ):
            return None, "invalid perplexity"
        declared_panels = manifest["effective_config"]["evaluation"].get("panels", [])
        if declared_panels:
            from sparselab.evaluation.panel_data import validate_panel_results

            validate_panel_results(payload, declared_panels, artifacts)
        elif payload.get("panels"):
            return None, "unexpected fixed-panel results"
        for field in (
            ("seq_len",)
            if declared_panels
            else ("batch_size", "max_batches", "seq_len")
        ):
            if not isinstance(payload.get(field), int) or payload[field] <= 0:
                return None, f"missing {field}"
        config = manifest["effective_config"]
        expected_protocol = {
            "batch_size": None
            if declared_panels
            else config["evaluation"].get("batch_size")
            or config["training"]["micro_batch_size"],
            "seq_len": config["training"]["seq_len"],
            "max_batches": None
            if declared_panels
            else config["evaluation"]["max_batches"],
        }
        if any(payload[key] != value for key, value in expected_protocol.items()):
            return None, "evaluation protocol differs from run config"
        if (
            valid_targets != expected_targets
            or payload.get("batches") != expected_batches
        ):
            return None, "validation target or batch count mismatch"
        return payload, None
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return None, str(error)


def experiment_evidence(
    run: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, object]:
    """Return observations only when every referenced immutable artifact verifies."""
    validate_mode(verification_mode)
    manifest = read_manifest(run / "manifest.json")
    artifacts, array_proofs = _validated_artifacts(
        run, manifest, proof_store=proof_store, verification_mode=verification_mode
    )
    config = manifest["effective_config"]
    data_root = run / "data"
    data_manifest = json.loads(
        (data_root / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        supervision_requires_mask(data_manifest)
        and not {"data/train_supervision.npy", "data/validation_supervision.npy"}
        <= artifacts.keys()
    ):
        raise ValueError("run inventory lacks required supervision masks")
    prepared = load_prepared_data(
        data_root,
        byte_enabled=config["model"]["memory"] in {"byte", "portable"},
        verification="structural",
        receipt=_receipt_from_proofs(data_root, data_manifest, array_proofs),
    )
    corpus_release_sha256 = None
    corpus_report_path = None
    if "corpus/export.json" in artifacts:
        export = json.loads(
            (run / "corpus" / "export.json").read_text(encoding="utf-8")
        )
        if not isinstance(export, dict) or export.get("release_id") != config[
            "dataset"
        ].get("revision"):
            raise ValueError("run corpus export and config identities disagree")
        for name, digest_key in (
            ("manifest.json", "release_manifest_sha256"),
            ("report.json", "report_sha256"),
            ("license-report.json", "license_report_sha256"),
        ):
            if artifacts.get("corpus/" + name) != export.get(digest_key):
                raise ValueError(f"run corpus evidence digest mismatch: {name}")
        corpus_release_sha256 = export["release_id"]
        corpus_report_path = "corpus/report.json"
    seq_len = config["training"]["seq_len"]
    batch_size = (
        config["evaluation"].get("batch_size") or config["training"]["micro_batch_size"]
    )
    validation = TokenBlockDataset(
        prepared.validation,
        seq_len,
        supervision=prepared.validation_supervision,
    )
    blocks = min(len(validation), batch_size * config["evaluation"]["max_batches"])
    expected_targets = blocks * seq_len
    if validation.supervision is not None:
        expected_targets = 0
        for block in validation.block_indices[:blocks]:
            start = int(block) * seq_len + 1
            expected_targets += int(
                np.count_nonzero(validation.supervision[start : start + seq_len])
            )
    expected_batches = math.ceil(blocks / batch_size)
    if config["evaluation"].get("panels"):
        from sparselab.evaluation.panel_data import panel_coverage

        expected_targets, expected_batches = panel_coverage(config["evaluation"])
    manifest_digest = canonical_digest(manifest)
    manager = CheckpointManager(run, manifest_sha256=manifest_digest)
    checkpoints: list[dict[str, object]] = []
    checkpoint_lookup: dict[str, dict[str, object]] = {}
    for path in sorted((run / "checkpoints").glob("step_*_gen_*")):
        report = manager.verify(
            path,
            manifest_digest,
            require_training_state=False,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        item: dict[str, object] = {
            "path": path.name,
            "verified": report.valid,
            "errors": list(report.errors),
            "verification_scope": report.resume_level,
        }
        if report.valid:
            raw = json.loads((path / "manifest.json").read_text())
            item.update(
                {
                    "step": raw["step"],
                    "tokens_seen": raw["tokens_seen"],
                    "exposure": raw.get("exposure"),
                    "validation_loss": raw["validation_loss"],
                    "digest": raw["sha256"],
                }
            )
            checkpoint_lookup[path.name] = item
        checkpoints.append(item)
    observations: list[dict[str, object]] = []
    rejected_reports: list[dict[str, str]] = []
    for path in sorted((run / "evaluations").glob("validation_step_*_gen_*.json")):
        result, reason = validate_held_out_report(
            path,
            checkpoint_lookup,
            artifacts,
            manifest,
            expected_targets,
            expected_batches,
        )
        if result is None:
            rejected_reports.append(
                {"path": path.name, "reason": reason or "invalid report"}
            )
        else:
            observations.append(result)
    verified = bool(checkpoints) and all(item["verified"] for item in checkpoints)
    observed_generations = {item["checkpoint"] for item in observations}
    missing_reports = sorted(
        name
        for name, item in checkpoint_lookup.items()
        if item["validation_loss"] is not None and name not in observed_generations
    )
    level = "artifact_only"
    if verified and observations:
        level = (
            "partial_held_out"
            if rejected_reports or missing_reports
            else "checkpointed_held_out"
        )
    return {
        "format": "experiment_evidence_v2",
        "run_id": manifest["run_id"],
        "corpus_release_sha256": corpus_release_sha256,
        "corpus_report_path": corpus_report_path,
        "source_identity_sha256": manifest["source_identity"]["sha256"],
        "verified_checkpoints": verified,
        "checkpoint_count": len(checkpoints),
        "quality_observations": observations,
        "rejected_reports": rejected_reports,
        "missing_reports": missing_reports,
        "evidence_level": level,
        "interpretation": "Verified checkpoints are paired with integrity-bound validation observations. Hashes bind artifacts but do not by themselves prove split separation or general model quality."
        if level == "checkpointed_held_out"
        else "Some observations verify, but reports are missing or rejected; inspect details before claiming complete evidence."
        if level == "partial_held_out"
        else "No complete integrity-bound checkpoint and validation evidence is available.",
        "checkpoints": checkpoints,
    }
