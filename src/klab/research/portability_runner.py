"""Training, checkpoint observation, and campaign receipts for portability runs."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from klab.canonical import canonical_digest
from klab.data.toy_worlds import StructuredKey, frozen_encoders
from klab.engram.semantic import SemanticQueryBatch, SemanticRetriever
from klab.evaluation.chat import format_chat_prompt
from klab.evaluation.generation import generate
from klab.model.portable_engram import load_portable_engram
from klab.research.portability import (
    PortabilityRun,
    load_portability_manifest,
    verify_portability_assets_unchanged,
)
from klab.research.portability_campaign import (
    _file_descriptor,
    _read_jsonl,
    _write_immutable_json,
    build_portability_run_manifest,
)
from klab.runtime import seed_everything
from klab.training.checkpoints import CheckpointManager

_SYMBOLS = "ABCDEFGHIJKLMNOP"


def _sha256_tensor(tensor: torch.Tensor) -> str:
    data = memoryview(tensor.detach().cpu().contiguous().numpy()).cast("B")
    return hashlib.sha256(data).hexdigest()


def _load_replacement(
    model: torch.nn.Module,
    run: PortabilityRun,
    world_id: str,
) -> str | None:
    memory = run.memory
    descriptor = memory["replacements"].get(world_id)
    if descriptor is None:
        raise ValueError(f"portability manifest has no replacement for {world_id}")
    artifact_descriptor = descriptor["artifact"]
    artifact_path = run.root / artifact_descriptor["path"]
    kind = memory["kind"]
    if kind == "semantic":
        retriever = SemanticRetriever.from_pack(
            artifact_path, expected_pack_id=descriptor["pack_id"]
        )
        model.semantic_memories["allocation"].replace_retriever(retriever)
        return descriptor["pack_id"] or descriptor["tensor_sha256"]
    if kind == "token":
        loaded = load_file(artifact_path, device="cpu")
        if set(loaded) != {"memory.table.weight"}:
            raise ValueError("replacement token table has an invalid tensor inventory")
        table = loaded["memory.table.weight"]
        destination = model.memory.table.weight
        if (
            tuple(table.shape) != tuple(destination.shape)
            or table.dtype != destination.dtype
        ):
            raise ValueError(
                "replacement token table differs from recipient dimensions"
            )
        if _sha256_tensor(table) != descriptor["tensor_sha256"]:
            raise ValueError("replacement token table digest mismatch")
    elif kind == "byte":
        package = load_portable_engram(
            artifact_path,
            expected_shape=(model.config.memory_table_size, model.config.memory_dim),
            expected_ngram_size=model.config.memory_ngram_size,
        )
        table = package.table
        destination = model.memory.table.weight
        if package.manifest.table_sha256 != descriptor["tensor_sha256"]:
            raise ValueError("replacement byte package tensor digest mismatch")
    else:
        raise ValueError(f"unsupported replacement memory kind: {kind}")
    if (
        tuple(table.shape) != tuple(destination.shape)
        or table.dtype != destination.dtype
    ):
        raise ValueError("replacement memory tensor differs from recipient dimensions")
    if _sha256_tensor(table) != descriptor["tensor_sha256"]:
        raise ValueError("replacement memory tensor digest mismatch")
    with torch.no_grad():
        destination.copy_(table.to(device=destination.device))
    return descriptor["tensor_sha256"]


def _start_subject(key: dict[str, Any]) -> tuple[int, str]:
    subject = key["subject"]
    slot_value, node_nonce = subject.split(".", 1)
    node = node_nonce.split("~", 1)[0]
    return int(slot_value[1:]), node


def _predict_case(
    model: torch.nn.Module,
    tokenizer: Any,
    row: dict[str, Any],
    world_manifest: dict[str, Any],
    device: torch.device,
    *,
    seed: int,
) -> dict[str, Any]:
    key = row["start_key"]
    slot, start_node = _start_subject(key)
    nonce_inventory = world_manifest["key_nonce_by_slot_node_relation"]
    base_prompt = row["prompt"][:-32]
    relations = row["relations"]
    current_subject = key["subject"]
    relation_encoder = (
        frozen_encoders()[0]
        if "allocation" in getattr(model, "semantic_memories", {})
        else None
    )
    answers: list[str] = []
    semantic_statuses: list[dict[str, Any]] = []
    for hop, relation in enumerate(relations):
        if hop == 0:
            subject = current_subject
        else:
            if not answers or answers[-1] not in _SYMBOLS:
                break
            node = answers[-1]
            nonce_key = f"{slot}|{node}|{relation}"
            if nonce_key not in nonce_inventory:
                break
            subject = f"w{slot:02d}.{node}~{nonce_inventory[nonce_key]}"
        lookup = f"{subject}.{relation}"
        suffix = "|" * (32 - len(lookup)) + lookup
        prompt = format_chat_prompt([], base_prompt + suffix)
        semantic_queries = None
        if "allocation" in getattr(model, "semantic_memories", {}):
            adapter = model.semantic_memories["allocation"]
            encoder = adapter.retriever.key_encoder
            assert relation_encoder is not None
            vector = relation_encoder.encode(StructuredKey(subject, relation))
            ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            vectors = torch.zeros(
                (1, len(ids), len(vector)), dtype=torch.float32, device=device
            )
            mask = torch.zeros((1, len(ids)), dtype=torch.bool, device=device)
            vectors[0, -1] = torch.as_tensor(vector, dtype=torch.float32, device=device)
            mask[0, -1] = True
            semantic_queries = SemanticQueryBatch(
                encoder=encoder,
                vectors=vectors,
                mask=mask,
                as_of=row.get("as_of"),
            )
        generated = generate(
            model,
            tokenizer,
            prompt,
            model.config.max_seq_len,
            1,
            device,
            temperature=0.0,
            seed=seed + hop,
            semantic_queries=semantic_queries,
        )
        answer = generated[len(prompt) :].strip()
        answers.append(answer)
        if semantic_queries is not None:
            adapter = model.semantic_memories["allocation"]
            semantic_statuses.extend(asdict(trace) for trace in adapter.last_traces)
    predicted_path = [start_node, *[answer for answer in answers if answer in _SYMBOLS]]
    expected = row["scorer"]["expected_answer"]
    expected_path = row["scorer"]["expected_path"]
    return {
        "case_id": row["case_id"],
        "world_id": row["world_id"],
        "task": row["task"],
        "expected_answer": expected,
        "predicted_answer": answers[-1] if answers else "",
        "answer_correct": bool(answers and answers[-1] == expected),
        "expected_path": expected_path,
        "predicted_path": predicted_path,
        "path_correct": bool(expected_path and predicted_path == expected_path),
        "semantic_traces": semantic_statuses,
        "as_of": row.get("as_of"),
    }


def _evaluate_cases(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    world_manifest: dict[str, Any],
    device: torch.device,
    *,
    seed: int,
    run: PortabilityRun | None = None,
) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    active_world: str | None = None
    for row in cases:
        world_id = row["world_id"]
        if run is not None and run.memory["replacements"] and world_id != active_world:
            _load_replacement(model, run, world_id)
            active_world = world_id
        outputs.append(
            _predict_case(model, tokenizer, row, world_manifest, device, seed=seed)
        )
    if not outputs:
        raise ValueError("portability evaluation has no cases")
    return {
        "case_count": len(outputs),
        "exact_answer_accuracy": sum(row["answer_correct"] for row in outputs)
        / len(outputs),
        "exact_path_accuracy": sum(row["path_correct"] for row in outputs)
        / len(outputs),
        "semantic_trace_status_counts": {
            status: sum(
                trace["status"] == status
                for row in outputs
                for trace in row["semantic_traces"]
            )
            for status in ("hit", "unknown", "conflict", "temporal_miss")
        },
        "cases": outputs,
    }


def _same_prediction(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return (
        first["predicted_answer"] == second["predicted_answer"]
        and first["predicted_path"] == second["predicted_path"]
    )


def _swap_probes(
    model: torch.nn.Module,
    tokenizer: Any,
    run: PortabilityRun,
    cases: list[dict[str, Any]],
    world_manifest: dict[str, Any],
    device: torch.device,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if not run.memory["replacements"]:
        return []
    world_specs = {entry["world_id"]: entry for entry in world_manifest["worlds"]}
    selected: list[tuple[str, str, dict[str, Any]]] = []
    for case in cases:
        if (
            case["task"] != "three-hop"
            or case["start_key"]["subject"].split(".")[1].split("~")[0] != "A"
        ):
            continue
        spec = world_specs[case["world_id"]]
        if spec.get("replacement") != "a":
            continue
        partner = next(
            candidate["world_id"]
            for candidate in world_manifest["worlds"]
            if candidate["partition"] == spec["partition"]
            and candidate["slot"] == spec["slot"]
            and candidate.get("replacement") == "b"
        )
        selected.append((case["world_id"], partner, case))
    results: list[dict[str, Any]] = []
    for world_a, world_b, case in selected:
        verify_portability_assets_unchanged(run)
        pack_a = _load_replacement(model, run, world_a)
        prediction_a1 = _predict_case(
            model, tokenizer, case, world_manifest, device, seed=seed
        )
        pack_b = _load_replacement(model, run, world_b)
        prediction_b = _predict_case(
            model, tokenizer, case, world_manifest, device, seed=seed
        )
        pack_a2 = _load_replacement(model, run, world_a)
        prediction_a2 = _predict_case(
            model, tokenizer, case, world_manifest, device, seed=seed
        )
        results.append(
            {
                "case_id": case["case_id"],
                "world_a": world_a,
                "world_b": world_b,
                "pack_id_a_first": pack_a,
                "pack_id_b": pack_b,
                "pack_id_a_final": pack_a2,
                "a_predictions_equal": _same_prediction(prediction_a1, prediction_a2),
                "a_first": prediction_a1,
                "b": prediction_b,
                "a_final": prediction_a2,
                "interpretation": (
                    "attachment-only A/B/A probe; not a natural-language query producer"
                ),
            }
        )
    return results


def _checkpoint_records(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    manager = CheckpointManager(run_dir)
    records: list[tuple[Path, dict[str, Any]]] = []
    for checkpoint in manager.root.glob("step_*_gen_*"):
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            continue
        report = manager.verify(checkpoint)
        if not report.valid:
            continue
        manifest = json.loads(
            (checkpoint / "manifest.json").read_text(encoding="utf-8")
        )
        records.append((checkpoint, manifest))
    records.sort(key=lambda item: (item[1]["step"], item[1]["generation_id"]))
    return records


def _evaluate_checkpoint(
    config: Any,
    portability_run: PortabilityRun,
    checkpoint_path: Path | None,
    development: list[dict[str, Any]],
    final: list[dict[str, Any]] | None,
    world_manifest: dict[str, Any],
    tokenizer: Any,
    *,
    run_id: str,
    step: int,
    tokens_seen: int,
    checkpoint_sha256: str,
    observation_path: Path,
) -> dict[str, Any]:
    from klab.engines.pytorch import PyTorchEngine

    seed_everything(config.seed, deterministic_cpu=True)
    engine = PyTorchEngine()
    engine.validate(config)
    engine.initialize(config)
    if engine.model is None:
        raise RuntimeError("PyTorch engine did not initialize a model")
    if checkpoint_path is not None:
        snapshot = CheckpointManager(checkpoint_path.parents[1]).load(
            checkpoint_path, mode="promote"
        )
        engine.model.load_state_dict(snapshot.model, strict=True)
    engine.model.eval()
    device = engine.device
    dev_results = _evaluate_cases(
        engine.model,
        tokenizer,
        development,
        world_manifest,
        device,
        seed=config.seed,
        run=portability_run,
    )
    swap_results = _swap_probes(
        engine.model,
        tokenizer,
        portability_run,
        development,
        world_manifest,
        device,
        seed=config.seed,
    )
    final_results = (
        _evaluate_cases(
            engine.model,
            tokenizer,
            final,
            world_manifest,
            device,
            seed=config.seed,
            run=portability_run,
        )
        if final is not None
        else None
    )
    verify_portability_assets_unchanged(portability_run)
    body = {
        "format": "klab-portability-checkpoint-observation",
        "version": 1,
        "run_id": run_id,
        "step": step,
        "tokens_seen": tokens_seen,
        "checkpoint_sha256": checkpoint_sha256,
        "development": dev_results,
        "final": final_results,
        "replacement_swaps": swap_results,
        "query_producer": "generator-provided structured keys; no text encoder",
    }
    _write_immutable_json(observation_path, body)
    digest = hashlib.sha256(observation_path.read_bytes()).hexdigest()
    return {
        "run_id": run_id,
        "step": step,
        "tokens_seen": tokens_seen,
        "checkpoint_sha256": checkpoint_sha256,
        "observation_sha256": digest,
        "observation_path": observation_path,
        "development_exact_answer_accuracy": dev_results["exact_answer_accuracy"],
        "development_exact_path_accuracy": dev_results["exact_path_accuracy"],
    }


def execute_portability_arm(
    campaign_root: Path,
    *,
    recipient: str,
    representation: str,
    condition: str,
    seed: int,
    updates: int,
) -> dict[str, Any]:
    """Train or attach one declared arm, then bind every checkpoint to observations."""
    from klab.data.tokenizer import load_tokenizer
    from klab.research.portability_campaign import _run_if_needed

    campaign_root = Path(campaign_root)
    arm_id = f"{recipient}-{representation}-{condition}-s{seed}"
    receipt_path = campaign_root / "evidence" / "arms" / f"{arm_id}.json"
    if receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_path, config, _ = build_portability_run_manifest(
        campaign_root,
        recipient=recipient,
        representation=representation,
        condition=condition,
        seed=seed,
        updates=updates,
    )
    portability_run = load_portability_manifest(config)
    world_path = campaign_root / "worlds" / "world_manifest.json"
    world_manifest = json.loads(world_path.read_text(encoding="utf-8"))
    development = _read_jsonl(campaign_root / "observations" / "development.jsonl")
    final_cases = _read_jsonl(campaign_root / "observations" / "final.jsonl")
    tokenizer = load_tokenizer(config.tokenizer.path)
    started = time.monotonic()
    run_id: str | None
    audit: dict[str, Any] | None = None
    if condition == "frozen-only":
        run_id = None
        backbone = portability_run.backbone
        if backbone is None:
            raise ValueError(
                "frozen-only arm is missing its recipient preparation checkpoint"
            )
        checkpoint_records: list[tuple[Path | None, dict[str, Any]]] = [
            (
                None,
                {
                    "step": 0,
                    "tokens_seen": 0,
                    "sha256": backbone["checkpoint_sha256"],
                },
            )
        ]
    else:
        run_id = arm_id
        _run_if_needed(config, run_id)
        run_dir = config.logging.root_dir / run_id
        checkpoint_records = _checkpoint_records(run_dir)
        if not checkpoint_records or checkpoint_records[-1][1]["step"] != updates:
            raise RuntimeError(
                f"arm did not save its final planned checkpoint: {arm_id}"
            )
        audit_path = run_dir / "portability_audit.json"
        if audit_path.is_symlink() or not audit_path.is_file():
            raise ValueError(f"portability arm did not write its final audit: {arm_id}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("format") != "klab-portability-audit"
            or audit.get("valid") is not True
            or audit.get("frozen_parameters_unchanged") is not True
            or audit.get("assets_unchanged") is not True
            or set(audit.get("trainable_parameter_names", []))
            != set(config.training.trainable_parameters or ())
        ):
            raise ValueError(f"portability update-isolation audit failed: {arm_id}")
        if condition in {
            "adapter-tuned",
            "random",
            "corrupt",
            "frozen-only",
        } and not audit.get("backbone_unchanged", False):
            raise ValueError(f"frozen recipient backbone changed: {arm_id}")
        final_checkpoint = checkpoint_records[-1][0]
        snapshot = CheckpointManager(run_dir).load(final_checkpoint, mode="resume")
        raw_names = snapshot.optimizer_parameter_names
        if isinstance(raw_names, dict):
            optimizer_names = set(raw_names.values())
        elif isinstance(raw_names, list):
            optimizer_names = {name for group in raw_names for name in group}
        else:
            raise ValueError("checkpoint omits canonical optimizer parameter names")
        if optimizer_names != set(config.training.trainable_parameters or ()):
            raise ValueError(
                f"checkpoint optimizer membership differs from plan: {arm_id}"
            )
    threshold = float(
        json.loads((campaign_root / "portability_protocol.json").read_text())[
            "training"
        ]["threshold"]["minimum"]
    )
    observations: list[dict[str, Any]] = []
    threshold_step: int | None = None
    for index, (checkpoint_path, checkpoint) in enumerate(checkpoint_records):
        step = int(checkpoint["step"])
        is_final = index == len(checkpoint_records) - 1
        observation_run_id = run_id or f"frozen-{arm_id}"
        observation_path = (
            campaign_root
            / "evidence"
            / "observations"
            / arm_id
            / f"step-{step:08d}.json"
        )
        result = _evaluate_checkpoint(
            config,
            portability_run,
            checkpoint_path,
            development,
            final_cases if is_final else None,
            world_manifest,
            tokenizer,
            run_id=observation_run_id,
            step=step,
            tokens_seen=int(checkpoint["tokens_seen"]),
            checkpoint_sha256=str(checkpoint["sha256"]),
            observation_path=observation_path,
        )
        if (
            threshold_step is None
            and result["development_exact_answer_accuracy"] >= threshold
        ):
            threshold_step = step
        observations.append(
            {
                "run_id": observation_run_id,
                "step": step,
                "tokens_seen": int(checkpoint["tokens_seen"]),
                "checkpoint_sha256": str(checkpoint["sha256"]),
                "observation_sha256": result["observation_sha256"],
                "observation": _file_descriptor(campaign_root, observation_path),
                "development_exact_answer_accuracy": result[
                    "development_exact_answer_accuracy"
                ],
                "development_exact_path_accuracy": result[
                    "development_exact_path_accuracy"
                ],
                "censored": bool(is_final and threshold_step is None),
            }
        )
    verify_portability_assets_unchanged(portability_run)
    reached = threshold_step is not None
    from klab.model.inspection import named_tensor_inventory

    inventory = named_tensor_inventory(config.model, config.attention)
    trainable_parameter_count = sum(
        inventory[name].numel for name in config.training.trainable_parameters or ()
    )
    artifact_descriptor = portability_run.memory["artifact"]
    artifact_bytes = (
        0
        if artifact_descriptor is None
        else sum(item["size_bytes"] for item in artifact_descriptor["files"])
        if "files" in artifact_descriptor
        else artifact_descriptor["size_bytes"]
    )
    receipt = {
        "format": "klab-portability-arm-receipt",
        "version": 1,
        "arm_id": arm_id,
        "recipient": recipient,
        "representation": representation,
        "condition": condition,
        "seed": seed,
        "run_id": run_id,
        "status": "completed" if reached else "censored",
        "threshold_reached_step": threshold_step,
        "threshold": {
            "metric": "development_exact_answer_accuracy",
            "minimum": threshold,
        },
        "checkpoint_observations": observations,
        "audit": audit,
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_artifact_bytes": artifact_bytes,
        "elapsed_seconds": time.monotonic() - started,
        "manifest": _file_descriptor(campaign_root, manifest_path),
    }
    _write_immutable_json(receipt_path, receipt)
    return json.loads(receipt_path.read_text(encoding="utf-8"))


def continue_portability_campaign(
    campaign_root: Path, *, max_arms: int | None = None, updates: int | None = None
) -> dict[str, Any]:
    """Run pending arms in declared order; immutable arm receipts make replay safe."""
    campaign_root = Path(campaign_root)
    protocol = json.loads(
        (campaign_root / "portability_protocol.json").read_text(encoding="utf-8")
    )
    planned_updates = protocol["training"]["planned_updates"]
    if updates is not None and updates != planned_updates:
        raise ValueError("requested update count differs from the immutable protocol")
    updates = planned_updates
    plan = json.loads((campaign_root / "plan.json").read_text(encoding="utf-8"))
    completed_now: list[str] = []
    for arm in plan["arms"]:
        arm_id = f"{arm['recipient']}-{arm['representation']}-{arm['condition']}-s{arm['seed']}"
        receipt = campaign_root / "evidence" / "arms" / f"{arm_id}.json"
        if receipt.exists():
            continue
        try:
            execute_portability_arm(
                campaign_root,
                recipient=arm["recipient"],
                representation=arm["representation"],
                condition=arm["condition"],
                seed=arm["seed"],
                updates=updates,
            )
        except Exception as error:
            record_portability_failure(
                campaign_root,
                recipient=arm["recipient"],
                representation=arm["representation"],
                condition=arm["condition"],
                seed=arm["seed"],
                error=error,
            )
            raise
        completed_now.append(arm_id)
        if max_arms is not None and len(completed_now) >= max_arms:
            break
    return {
        "campaign_id": plan["campaign_id"],
        "planned_arms": len(plan["arms"]),
        "completed_this_invocation": completed_now,
        "remaining_arms": sum(
            not (
                campaign_root
                / "evidence"
                / "arms"
                / f"{arm['recipient']}-{arm['representation']}-{arm['condition']}-s{arm['seed']}.json"
            ).exists()
            for arm in plan["arms"]
        ),
    }


def build_portability_evidence(campaign_root: Path) -> Path:
    """Build the strict report envelope from immutable observations and receipts."""
    campaign_root = Path(campaign_root)
    protocol = json.loads(
        (campaign_root / "portability_protocol.json").read_text(encoding="utf-8")
    )
    world = json.loads(
        (campaign_root / "worlds" / "world_manifest.json").read_text(encoding="utf-8")
    )
    plan = json.loads((campaign_root / "plan.json").read_text(encoding="utf-8"))
    arms: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    counts = {"completed_arms": 0, "failed_arms": 0, "censored_arms": 0}
    for planned in plan["arms"]:
        arm_id = (
            f"{planned['recipient']}-{planned['representation']}-"
            f"{planned['condition']}-s{planned['seed']}"
        )
        receipt_path = campaign_root / "evidence" / "arms" / f"{arm_id}.json"
        if not receipt_path.is_file():
            failure_paths = sorted(
                (campaign_root / "evidence" / "failures" / arm_id).glob(
                    "attempt-*.json"
                )
            )
            if failure_paths:
                failure = json.loads(failure_paths[-1].read_text(encoding="utf-8"))
                counts["failed_arms"] += 1
                arms.append(
                    {
                        "arm_id": arm_id,
                        **planned,
                        "status": "failed",
                        "run_id": failure["run_id"],
                    }
                )
            else:
                arms.append(
                    {
                        "arm_id": arm_id,
                        **planned,
                        "status": "planned",
                        "run_id": None,
                    }
                )
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        status = receipt["status"]
        if status == "completed":
            counts["completed_arms"] += 1
        elif status == "failed":
            counts["failed_arms"] += 1
        elif status == "censored":
            counts["censored_arms"] += 1
        arms.append(
            {
                "arm_id": arm_id,
                **planned,
                "status": status,
                "run_id": receipt["run_id"],
            }
        )
        observations.extend(
            {
                "run_id": row["run_id"],
                "step": row["step"],
                "tokens_seen": row["tokens_seen"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "observation_sha256": row["observation_sha256"],
                "censored": row["censored"],
            }
            for row in receipt["checkpoint_observations"]
        )
    evidence = {
        "format": "klab-portability-evidence",
        "version": 1,
        "campaign_id": protocol["campaign_id"],
        "scale": protocol["scale"],
        "protocol_sha256": protocol["sha256"],
        "world_manifest_sha256": world["sha256"],
        "arms": arms,
        "observations": observations,
        "summary": {
            "planned_arms": len(plan["arms"]),
            **counts,
        },
        "limitations": list(protocol["limitations"]),
    }
    digest = canonical_digest(evidence)
    evidence_path = campaign_root / f"portability_evidence-{digest[:16]}.json"
    _write_immutable_json(evidence_path, evidence)
    return evidence_path


def portability_resource_plan(campaign_root: Path) -> dict[str, Any]:
    """Estimate sequential CPU campaign memory and storage without loading weights."""
    from klab.config.models import AttentionConfig, ModelConfig
    from klab.model.inspection import named_tensor_inventory

    campaign_root = Path(campaign_root)
    protocol = json.loads(
        (campaign_root / "portability_protocol.json").read_text(encoding="utf-8")
    )
    world = json.loads(
        (campaign_root / "worlds" / "world_manifest.json").read_text(encoding="utf-8")
    )
    plan = json.loads((campaign_root / "plan.json").read_text(encoding="utf-8"))
    vocab_size = world["tokenizer"]["vocab_size"]
    updates = protocol["training"]["planned_updates"]
    resource_by_arm: dict[str, Any] = {}
    for arm in plan["arms"]:
        width = int(arm["recipient"].removeprefix("width"))
        representation, condition = arm["representation"], arm["condition"]
        memory = (
            "ngram"
            if representation == "token" and condition != "disabled"
            else "byte"
            if representation == "byte" and condition != "disabled"
            else "none"
        )
        semantic_dim = (
            8 if representation == "semantic" and condition != "disabled" else None
        )
        model = ModelConfig.model_validate(
            {
                "vocab_size": vocab_size,
                "hidden_dim": width,
                "num_layers": 1,
                "num_heads": 4,
                "ffn_dim": width * 2,
                "max_seq_len": 128,
                "memory": memory,
                "memory_table_size": 8192 if memory != "none" else 0,
                "memory_ngram_size": 32 if memory != "none" else 0,
                "memory_dim": 8 if memory != "none" else 0,
                "semantic_memory_dim": semantic_dim,
            }
        )
        inventory = named_tensor_inventory(model, AttentionConfig())
        parameters = sum(
            spec.numel for spec in inventory.values() if spec.alias_of is None
        )
        adapter_parameters = 9 * width
        selected_parameters = (
            adapter_parameters
            if condition in {"adapter-tuned", "random", "corrupt", "frozen-only"}
            else parameters
        )
        optimizer_parameters = 0 if condition == "frozen-only" else selected_parameters
        model_bytes = parameters * 4
        optimizer_bytes = optimizer_parameters * 12
        activation_bytes = 64 * width * 8 * 4
        peak_bytes = model_bytes + optimizer_bytes + activation_bytes + 64 * 1024**2
        checkpoint_bytes = model_bytes + optimizer_parameters * 8
        resource_by_arm[arm["recipient"] + "-" + representation + "-" + condition] = {
            "parameters": parameters,
            "selected_parameter_count": selected_parameters,
            "optimizer_parameter_count": optimizer_parameters,
            "model_bytes_fp32": model_bytes,
            "optimizer_and_gradient_bytes_estimate": optimizer_bytes,
            "activation_bytes_estimate": activation_bytes,
            "peak_ram_bytes_estimate": peak_bytes,
            "checkpoint_bytes_estimate": checkpoint_bytes,
            "condition": condition,
            "updates": updates,
        }
    unique_coordinates = {key: item for key, item in resource_by_arm.items()}
    completed_receipts = list((campaign_root / "evidence" / "arms").glob("*.json"))
    observed_seconds = [
        json.loads(path.read_text(encoding="utf-8"))["elapsed_seconds"]
        for path in completed_receipts
        if path.is_file()
    ]
    per_arm = list(unique_coordinates.values())
    training_checkpoint_storage = sum(
        item["checkpoint_bytes_estimate"]
        * (updates + 1)
        * len(protocol["recipient_seeds"])
        for item in per_arm
        if item["condition"] != "frozen-only"
    )
    preparation_parameters = [
        item["parameters"]
        for key, item in unique_coordinates.items()
        if key.startswith(("width32-token-disabled", "width64-token-disabled"))
    ]
    preparation_checkpoint_storage = (
        sum(preparation_parameters)
        * 12
        * (updates + 1)
        * len(protocol["recipient_seeds"])
    )
    return {
        "campaign_id": protocol["campaign_id"],
        "scale": protocol["scale"],
        "planned_arms": len(plan["arms"]),
        "recipient_seeds": protocol["recipient_seeds"],
        "sequential_cpu_only": True,
        "architecture": "DenseLM, dense attention, 1 layer, 4 heads, FFN=2x width",
        "estimates": {
            "maximum_single_arm_ram_bytes": max(
                item["peak_ram_bytes_estimate"] for item in per_arm
            ),
            "training_checkpoint_storage_bytes_estimate": training_checkpoint_storage,
            "preparation_checkpoint_storage_bytes_estimate": preparation_checkpoint_storage,
            "total_checkpoint_storage_bytes_estimate": (
                training_checkpoint_storage + preparation_checkpoint_storage
            ),
            "runtime_calibration": {
                "observed_arm_count": len(observed_seconds),
                "median_wall_seconds_per_arm": (
                    statistics.median(observed_seconds) if observed_seconds else None
                ),
                "interpretation": (
                    "observed train plus checkpoint-observation wall time"
                    if observed_seconds
                    else "not calibrated; estimates are analytical only"
                ),
            },
        },
        "coordinate_estimates": unique_coordinates,
        "limitations": [
            "RAM and checkpoint estimates exclude Python, tokenizer, allocator, filesystem, and process overhead beyond the fixed 64 MiB allowance.",
            "Runtime estimates are unavailable until an arm receipt provides observed wall time.",
            "This resource plan does not establish behavioral transfer.",
        ],
    }


def record_portability_failure(
    campaign_root: Path,
    *,
    recipient: str,
    representation: str,
    condition: str,
    seed: int,
    error: BaseException,
) -> Path:
    """Persist an immutable failed attempt without blocking later recovery."""
    campaign_root = Path(campaign_root)
    arm_id = f"{recipient}-{representation}-{condition}-s{seed}"
    directory = campaign_root / "evidence" / "failures" / arm_id
    directory.mkdir(parents=True, exist_ok=True)
    attempt = 1 + sum(path.is_file() for path in directory.glob("attempt-*.json"))
    path = directory / f"attempt-{attempt:04d}.json"
    _write_immutable_json(
        path,
        {
            "format": "klab-portability-failure",
            "version": 1,
            "arm_id": arm_id,
            "recipient": recipient,
            "representation": representation,
            "condition": condition,
            "seed": seed,
            "run_id": None if condition == "frozen-only" else arm_id,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    return path
