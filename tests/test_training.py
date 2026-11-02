from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from sparselab.canonical import canonical_digest
from sparselab.config.loading import load_config
from sparselab.config.migrate import migrate_v1
from sparselab.config.models import RunConfig, TokenizerTrainConfig
from sparselab.data.tokenizer import train_tokenizer
from sparselab.engines.base import EngineCapabilityError
from sparselab.engines.mlx_policy import validate as validate_mlx
from sparselab.evaluation.evidence import experiment_evidence
from sparselab.evaluation.post_train_triage import read_triage
from sparselab.evaluation.reference_exercise import PROMPTS
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import read_manifest
from sparselab.training.metrics import ExperimentStore
from sparselab.training.trainer import train


def equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            equal(first, second)
    else:
        assert left == right


_CACHED_TOKENIZER: tuple[bytes, bytes] | None = None


def config(root: Path) -> RunConfig:
    global _CACHED_TOKENIZER
    tok_dir = root / "tokenizer"
    if _CACHED_TOKENIZER is not None and not tok_dir.exists():
        tok_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = tok_dir / "tokenizer.json"
        tokenizer.write_bytes(_CACHED_TOKENIZER[0])
        (tok_dir / "tokenizer_manifest.json").write_bytes(_CACHED_TOKENIZER[1])
    else:
        tokenizer = train_tokenizer(
            TokenizerTrainConfig.model_validate(
                {
                    "schema_version": 1,
                    "vocab_size": 512,
                    "min_frequency": 1,
                    "max_documents": 200,
                    "output_dir": tok_dir,
                    "dataset": {
                        "source": "synthetic",
                        "cache_dir": root / "cache",
                        "train_max_documents": 200,
                        "validation_max_documents": 40,
                        "train_max_tokens": 32768,
                        "validation_max_tokens": 2048,
                        "synthetic_seed": 7,
                    },
                }
            )
        )
        if _CACHED_TOKENIZER is None:
            _CACHED_TOKENIZER = (
                tokenizer.read_bytes(),
                (tok_dir / "tokenizer_manifest.json").read_bytes(),
            )
    return RunConfig.model_validate(
        migrate_v1(
            {
                "schema_version": 1,
                "name": "test",
                "seed": 7,
                "device": "cpu",
                "model": {
                    "vocab_size": 512,
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 2,
                    "ffn_dim": 32,
                    "max_seq_len": 32,
                },
                "tokenizer": {"path": tokenizer},
                "dataset": {
                    "source": "synthetic",
                    "cache_dir": root / "cache",
                    "train_max_documents": 200,
                    "validation_max_documents": 40,
                    "train_max_tokens": 8192,
                    "validation_max_tokens": 2048,
                    "synthetic_seed": 7,
                },
                "training": {
                    "batch_size": 2,
                    "seq_len": 16,
                    "max_steps": 12,
                    "max_tokens": 384,
                    "deterministic": True,
                },
                "optimizer": {
                    "learning_rate": 0.003,
                    "min_learning_rate": 0.0003,
                    "warmup_steps": 2,
                },
                "logging": {"root_dir": root / "runs", "checkpoint_every_steps": 6},
            }
        )
    )


@pytest.mark.parametrize(
    ("memory", "optimizer_name"),
    [("none", "adamw"), ("byte", "adamw"), ("none", "adafactor")],
)
def test_interrupted_resume_matches_uninterrupted(
    tmp_path: Path, memory: str, optimizer_name: str
) -> None:
    memory_settings = (
        {
            "memory": "byte",
            "memory_table_size": 64,
            "memory_dim": 8,
            "memory_ngram_size": 5,
        }
        if memory == "byte"
        else {"memory": "none"}
    )
    full = config(tmp_path / "full")
    full = full.model_copy(
        update={"model": full.model.model_copy(update=memory_settings)}
    )
    if optimizer_name == "adafactor":
        payload = full.model_dump(mode="json")
        payload["optimizer"] = {
            "name": "adafactor",
            "peak": full.optimizer.peak,
            "floor": full.optimizer.floor,
            "warmup_steps": full.optimizer.warmup_steps,
            "weight_decay": full.optimizer.weight_decay,
        }
        full = RunConfig.model_validate(payload)
    train(full, run_id="full")
    split = config(tmp_path / "split")
    split = split.model_copy(
        update={
            "model": split.model.model_copy(update=memory_settings),
            "optimizer": full.optimizer,
        }
    )
    train(split, run_id="part", stop_after_step=5)
    train(
        split,
        run_id="resumed",
        resume=split.logging.root_dir / "part/checkpoints/latest.json",
    )
    left = CheckpointManager(full.logging.root_dir / "full").load(
        full.logging.root_dir / "full/checkpoints/latest.json"
    )
    right = CheckpointManager(split.logging.root_dir / "resumed").load(
        split.logging.root_dir / "resumed/checkpoints/latest.json"
    )
    assert left.tokens_seen == right.tokens_seen == 384
    assert left.cursor == right.cursor
    equal(left.optimizer, right.optimizer)
    equal(left.model, right.model)
    equal(left.rng, right.rng)
    parent = CheckpointManager(split.logging.root_dir / "part").load(
        split.logging.root_dir / "part/checkpoints/latest.json"
    )
    manifest = read_manifest(split.logging.root_dir / "resumed/manifest.json")
    assert manifest["checkpoint_sha256"] == parent.checkpoint_sha256
    assert right.parent_checkpoint_sha256 == parent.checkpoint_sha256


def test_training_pairs_validation_with_verified_checkpoints(tmp_path: Path) -> None:
    original = config(tmp_path)
    measured = original.model_copy(
        update={
            "evaluation": original.evaluation.model_copy(
                update={"every_steps": 2, "max_batches": 1}
            )
        }
    )
    run_id = train(measured, run_id="measured")
    run = measured.logging.root_dir / run_id
    prepared_manifest = json.loads((run / "data/manifest.json").read_text())
    assert prepared_manifest["supervision"] == {"kind": "all_tokens"}
    assert not (run / "data/train_supervision.npy").exists()
    assert not (run / "data/validation_supervision.npy").exists()
    run_manifest = read_manifest(run / "manifest.json")
    inventory = {item["relative_path"] for item in run_manifest["artifacts"]}
    assert "data/train_supervision.npy" not in inventory
    assert "data/validation_supervision.npy" not in inventory
    report_payload = json.loads(next((run / "evaluations").glob("*.json")).read_text())
    assert "data/validation_supervision.npy" not in report_payload["identities"]
    assert report_payload["valid_targets"] == (
        report_payload["batches"]
        * measured.training.seq_len
        * measured.training.micro_batch_size
    )

    evidence = experiment_evidence(measured.logging.root_dir / run_id)
    observations = evidence["quality_observations"]
    assert evidence["evidence_level"] == "checkpointed_held_out"
    assert evidence["verified_checkpoints"]
    assert isinstance(observations, list)
    report = read_triage(run_id, measured.logging.root_dir)
    assert report is not None
    assert report["format"] == "sparselab_post_train_triage_v1"
    assert report["core"]["evidence"]["expected_validation_targets"] == (
        measured.training.micro_batch_size * measured.training.seq_len
    )
    assert (
        len(
            list(
                (measured.logging.root_dir / run_id / "post-train-triage").glob(
                    "*.json"
                )
            )
        )
        == 1
    )
    assert [case["id"] for case in report["tier1"]["greedy"]] == [
        name for name, _ in PROMPTS
    ]
    assert len(report["tier1"]["sampled"]) <= 8
    assert all(
        case["settings"]
        == {
            "temperature": 0,
            "top_k": 0,
            "seed": 42042,
            "max_new_tokens": 32,
        }
        and len(case["token_ids"]) <= 32
        for case in report["tier1"]["greedy"]
        if case["status"] == "OBSERVED"
    )
    assert all(
        case["settings"]["seed"] in (11, 29)
        and case["settings"]["top_k"] in (0, 40)
        and case["settings"]["max_new_tokens"] == 24
        and len(case["token_ids"]) <= 24
        for case in report["tier1"]["sampled"]
        if case["status"] == "OBSERVED"
    )
    assert [item["step"] for item in observations] == [0, 2, 4, 6, 8, 10, 12]


def test_post_train_triage_failure_preserves_completed_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from sparselab.training import trainer

    def unavailable(*args: object, **kwargs: object) -> None:
        raise RuntimeError("diagnostic unavailable")

    monkeypatch.setattr(trainer, "triage_completed_run", unavailable)
    configured = config(tmp_path)
    run_id = train(configured, run_id="triage-failure")
    progress = json.loads(
        (configured.logging.root_dir / run_id / "progress.json").read_text()
    )
    assert progress["status"] == "completed"
    assert read_triage(run_id, configured.logging.root_dir) is None
    assert "POST-TRAIN TRIAGE UNKNOWN" in caplog.text


def test_interrupted_training_does_not_produce_completed_triage(tmp_path: Path) -> None:
    configured = config(tmp_path)
    run_id = train(configured, run_id="interrupted-triage", stop_after_step=2)
    assert read_triage(run_id, configured.logging.root_dir) is None


def test_trainer_saves_only_explicit_checkpoint_boundaries(tmp_path: Path) -> None:
    original = config(tmp_path)
    configuration = original.model_dump(mode="json")
    configuration["evaluation"].update({"every_steps": 12, "max_batches": 1})
    configuration["checkpoint"].update(
        {
            "every_steps": None,
            "every_tokens": None,
            "every_minutes": None,
            "steps": [0, 2, 6, 12],
            "keep_periodic": True,
        }
    )
    measured = RunConfig.model_validate(configuration)
    run_id = train(measured, run_id="explicit-boundaries")

    manifests = sorted(
        (measured.logging.root_dir / run_id / "checkpoints").glob(
            "step_*_gen_*/manifest.json"
        )
    )
    assert [json.loads(path.read_text())["step"] for path in manifests] == [
        0,
        2,
        6,
        12,
    ]

    invalid = original.model_dump(mode="json")
    invalid["checkpoint"].update(
        {
            "every_steps": None,
            "every_tokens": None,
            "every_minutes": None,
            "steps": [0, 4, 2],
        }
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        RunConfig.model_validate(invalid)


def test_assistant_only_evidence_counts_and_binds_supervised_targets(
    tmp_path: Path,
) -> None:
    raw = config(tmp_path).model_dump(mode="json")
    for split in ("train", "validation"):
        path = tmp_path / f"{split}.jsonl"
        records = [
            {
                "format_version": 2,
                "loss_mode": "assistant_only",
                "messages": [
                    {
                        "role": "user",
                        "content": f"{split} {index}: " + "rain falls gently. " * 40,
                    },
                    {"role": "assistant", "content": "the quiet river flows"},
                ],
            }
            for index in range(2)
        ]
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        raw["dataset"][f"{split}_path"] = str(path)
        raw["dataset"][f"{split}_max_tokens"] = 4096
    raw["dataset"].update(source="local_chat", license="MIT")
    raw["training"].update(max_steps=2, max_tokens=8)
    raw["optimizer"]["warmup_steps"] = 0
    raw["evaluation"].update(every_steps=1, max_batches=20)
    measured = RunConfig.model_validate(raw)
    run_id = train(measured, run_id="assistant-evidence")
    run = measured.logging.root_dir / run_id
    prepared_manifest = json.loads((run / "data/manifest.json").read_text())
    assert prepared_manifest["supervision"]["kind"] == "token-loss-mask-v1"
    inventory = {
        item["relative_path"]
        for item in read_manifest(run / "manifest.json")["artifacts"]
    }
    assert {
        "data/train_supervision.npy",
        "data/validation_supervision.npy",
    } <= inventory
    supervision = np.load(run / "data/validation_supervision.npy")
    seq_len = measured.training.seq_len
    usable = (len(supervision) - 1) // seq_len * seq_len
    counts = supervision[1 : usable + 1].reshape(-1, seq_len).sum(axis=1)
    selected = counts[counts > 0][
        : measured.training.micro_batch_size * measured.evaluation.max_batches
    ]
    assert (counts == 0).any()
    assert 0 < selected.sum() < len(selected) * seq_len

    evidence = experiment_evidence(run)
    assert evidence["evidence_level"] == "checkpointed_held_out"
    for observation in evidence["quality_observations"]:
        assert observation["valid_targets"] == int(selected.sum())
        assert (
            observation["batches"]
            == (len(selected) + measured.training.micro_batch_size - 1)
            // measured.training.micro_batch_size
        )
    triage = read_triage(run_id, measured.logging.root_dir)
    assert triage is not None
    assert triage["core"]["evidence"]["expected_validation_targets"] == int(
        selected.sum()
    )

    report = next((run / "evaluations").glob("*.json"))
    payload = json.loads(report.read_text())
    payload["identities"]["data/validation_supervision.npy"] = "0" * 64
    payload.pop("sha256")
    payload["sha256"] = canonical_digest(payload)
    report.write_text(json.dumps(payload))
    rejected = experiment_evidence(run)
    assert report.name in {item["path"] for item in rejected["rejected_reports"]}
    assert payload["checkpoint"] in rejected["missing_reports"]


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_interrupted_checkpoint_resumes_locally(tmp_path: Path) -> None:
    original = config(tmp_path)
    mps = original.model_copy(
        update={"runtime": original.runtime.model_copy(update={"backend": "auto"})}
    )
    import sys

    from sparselab.runtime_profile import RuntimeProfile, authorize_profile

    authorization = authorize_profile(
        RuntimeProfile(
            runtime_profile_version=1,
            id="mps-resume-regression",
            python=Path(sys.executable).absolute(),
            engine="pytorch",
            backend="mps",
            device_index=0,
        ),
        mps,
    )
    train(mps, run_id="part", stop_after_step=2, authorization=authorization)
    train(
        mps,
        run_id="resumed",
        resume=mps.logging.root_dir / "part/checkpoints/latest.json",
        authorization=authorization,
    )

    resumed = CheckpointManager(mps.logging.root_dir / "resumed").load(
        mps.logging.root_dir / "resumed/checkpoints/latest.json"
    )
    assert resumed.step == mps.training.max_steps
    assert resumed.backend == "mps"
    manifest = read_manifest(mps.logging.root_dir / "resumed/manifest.json")
    assert manifest["requested_config"]["runtime"]["backend"] == "auto"
    assert manifest["effective_config"]["runtime"]["backend"] == "mps"


def test_adafactor_state_offload_is_rejected(tmp_path: Path) -> None:
    payload = config(tmp_path).model_dump(mode="json")
    payload["optimizer"] = {
        "name": "adafactor",
        "peak": 0.003,
        "floor": 0.0003,
        "warmup_steps": 2,
        "weight_decay": 0.1,
        "state_offload": True,
    }

    with pytest.raises(ValueError, match="state_offload is deferred"):
        RunConfig.model_validate(payload)


def test_resume_rejects_model_configuration_mismatch(tmp_path: Path) -> None:
    original = config(tmp_path / "original")
    train(original, run_id="part", stop_after_step=5)
    incompatible = original.model_copy(
        update={"model": original.model.model_copy(update={"ffn_dim": 48})}
    )
    with pytest.raises(ValueError, match="configuration differs"):
        train(
            incompatible,
            run_id="rejected",
            resume=original.logging.root_dir / "part/checkpoints/latest.json",
        )


def test_memory_placement_cannot_resume_or_promote_across_sites(
    tmp_path: Path,
) -> None:
    original = config(tmp_path / "source")
    memory_config = original.model.model_copy(
        update={
            "memory": "ngram",
            "memory_table_size": 31,
            "memory_ngram_size": 3,
            "memory_dim": 8,
            "memory_injection": "final",
        }
    )
    original = original.model_copy(update={"model": memory_config})
    train(original, run_id="final", stop_after_step=1)
    checkpoint = original.logging.root_dir / "final/checkpoints/latest.json"
    embedding = original.model_copy(
        update={
            "model": original.model.model_copy(update={"memory_injection": "embedding"})
        }
    )

    with pytest.raises(ValueError, match="architecture semantics"):
        train(
            embedding,
            run_id="rejected-resume",
            resume=checkpoint,
            allow_runtime_drift=True,
        )
    with pytest.raises(ValueError, match="architecture semantics"):
        train(
            embedding,
            run_id="rejected-promotion",
            promote=checkpoint,
            allow_runtime_drift=True,
            stop_after_step=1,
        )


@pytest.mark.parametrize("placement", ["final", "embedding"])
def test_mlx_validation_rejects_enabled_memory_at_every_placement(
    placement: str,
) -> None:
    base = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "runtime_smoke_cpu.yaml"
    )
    model = base.model.model_copy(
        update={
            "memory": "ngram",
            "memory_table_size": 31,
            "memory_ngram_size": 3,
            "memory_dim": 8,
            "memory_injection": placement,
        }
    )
    config_value = base.model_copy(
        update={
            "model": model,
            "runtime": base.runtime.model_copy(
                update={"engine": "mlx", "backend": "metal"}
            ),
        }
    )

    with pytest.raises(EngineCapabilityError, match="does not support memory modules"):
        validate_mlx(config_value)


def test_non_multiple_token_budget_commits_exactly_77_targets(tmp_path: Path) -> None:
    original = config(tmp_path)
    bounded = original.model_copy(
        update={"training": original.training.model_copy(update={"max_tokens": 77})}
    )
    run_id = train(bounded, run_id="bounded")
    snapshot = CheckpointManager(bounded.logging.root_dir / run_id).load(
        bounded.logging.root_dir / run_id / "checkpoints/latest.json"
    )
    assert snapshot.tokens_seen == 77
    assert snapshot.cursor == (0, 5)


def test_whole_update_crosses_epoch_without_losing_targets(tmp_path: Path) -> None:
    raw = config(tmp_path).model_dump(mode="json")
    raw["dataset"]["train_max_tokens"] = 256
    raw["training"].update(max_steps=10, max_tokens=320)
    raw["checkpoint"]["steps"] = [0, 10]
    raw["evaluation"].update(every_steps=10, max_batches=1)
    bounded = RunConfig.model_validate(raw)
    train(bounded, run_id="cross-epoch")
    root = bounded.logging.root_dir / "cross-epoch"
    snapshot = CheckpointManager(root).load(root / "checkpoints/latest.json")
    assert (snapshot.step, snapshot.tokens_seen) == (10, 320)
    assert snapshot.cursor[0] == 1
    assert CheckpointManager(root).verify(root / "checkpoints/latest.json").valid


def test_allow_runtime_drift_does_not_allow_dataset_change(tmp_path: Path) -> None:
    original = config(tmp_path / "original")
    train(original, run_id="part", stop_after_step=5)
    changed = original.model_copy(
        update={"dataset": original.dataset.model_copy(update={"synthetic_seed": 99})}
    )
    with pytest.raises(ValueError, match="configuration differs"):
        train(
            changed,
            run_id="rejected",
            resume=original.logging.root_dir / "part/checkpoints/latest.json",
            allow_runtime_drift=True,
        )


def test_promotion_prepares_destination_dataset(tmp_path: Path) -> None:
    source = config(tmp_path / "source")
    train(source, run_id="source", stop_after_step=2)
    destination = config(tmp_path / "destination").model_copy(
        update={"dataset": source.dataset.model_copy(update={"synthetic_seed": 91})}
    )
    train(
        destination,
        run_id="promoted",
        promote=source.logging.root_dir / "source/checkpoints/latest.json",
        stop_after_step=1,
    )
    source_manifest = (
        source.logging.root_dir / "source/data/manifest.json"
    ).read_text()
    destination_manifest = (
        destination.logging.root_dir / "promoted/data/manifest.json"
    ).read_text()
    assert source_manifest != destination_manifest


def test_tampered_validation_report_is_not_held_out_evidence(tmp_path: Path) -> None:
    original = config(tmp_path)
    run_id = train(original, run_id="measured")
    report = next((original.logging.root_dir / run_id / "evaluations").glob("*.json"))
    payload = json.loads(report.read_text())
    payload["loss"] = 999
    report.write_text(json.dumps(payload))
    evidence = experiment_evidence(original.logging.root_dir / run_id)
    assert evidence["evidence_level"] != "checkpointed_held_out"
    assert report.name in {item["path"] for item in evidence["rejected_reports"]}
    assert payload["checkpoint"] in evidence["missing_reports"]
    assert all(
        item["checkpoint"] != payload["checkpoint"]
        for item in evidence["quality_observations"]
    )


def test_source_drift_requires_explicit_best_effort_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = config(tmp_path)
    train(original, run_id="part", stop_after_step=1)
    parent = original.logging.root_dir / "part"
    previous_source = read_manifest(parent / "manifest.json")["source_identity"]
    changed_source = {**previous_source, "sha256": "0" * 64}
    monkeypatch.setattr(
        "sparselab.training.trainer.source_identity", lambda: changed_source
    )
    with pytest.raises(ValueError, match="allow-runtime-drift"):
        train(original, run_id="rejected", resume=parent / "checkpoints/latest.json")
    assert not (original.logging.root_dir / "rejected").exists()
    train(
        original,
        run_id="allowed",
        resume=parent / "checkpoints/latest.json",
        allow_runtime_drift=True,
        stop_after_step=2,
    )
    manifest = read_manifest(original.logging.root_dir / "allowed/manifest.json")
    decision = next(
        item
        for item in manifest["resource_decisions"]
        if item["kind"] == "runtime_drift"
    )
    assert decision["resume_level"] == "best_effort"
    assert decision["changes"][0]["requested"] == previous_source["sha256"]
    assert decision["changes"][0]["effective"] == changed_source["sha256"]


@pytest.mark.parametrize("explicit_parent_decay", [False, True])
def test_budget_extension_preserves_full_state_and_parent(
    tmp_path: Path, explicit_parent_decay: bool
) -> None:
    raw = config(tmp_path).model_dump(mode="json")
    raw["training"].update(max_steps=4, max_tokens=128)
    raw["checkpoint"]["steps"] = [0, 4]
    raw["evaluation"].update(every_steps=4, max_batches=1)
    if explicit_parent_decay:
        raw["optimizer"]["decay_steps"] = 4
    parent_config = RunConfig.model_validate(raw)
    train(parent_config, run_id="parent")
    parent_run = parent_config.logging.root_dir / "parent"
    generation = next((parent_run / "checkpoints").glob("step_00000004_gen_*"))
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in generation.iterdir()
    }
    parent = CheckpointManager(parent_run).load(generation)
    raw["name"] = "budget-extension"
    raw["training"].update(max_steps=8, max_tokens=256)
    raw["optimizer"]["decay_steps"] = 4
    raw["checkpoint"]["steps"] = [0, 4, 8]
    if explicit_parent_decay:
        relocated = tmp_path / "relocated-tokenizer.json"
        relocated.write_bytes(parent_config.tokenizer.path.read_bytes())
        raw["tokenizer"]["path"] = str(relocated)
    extended = RunConfig.model_validate(raw)
    with pytest.raises(ValueError, match="configuration differs"):
        train(extended, run_id="strict-rejected", resume=generation)
    assert not (extended.logging.root_dir / "strict-rejected").exists()
    with pytest.raises(ValueError, match="generation directory"):
        train(
            extended,
            run_id="pointer-rejected",
            extend_budget=parent_run / "checkpoints/latest.json",
        )
    assert not (extended.logging.root_dir / "pointer-rejected").exists()
    train(extended, run_id="extended", extend_budget=generation, stop_after_step=5)
    child_run = extended.logging.root_dir / "extended"
    child = CheckpointManager(child_run).load(child_run / "checkpoints/latest.json")
    assert (child.step, child.tokens_seen, child.schedule["kind"]) == (
        5,
        160,
        "warmup_cosine_floor_v1",
    )
    assert child.cursor == (0, 10)
    assert child.parent_checkpoint_sha256 == parent.checkpoint_sha256
    assert (
        CheckpointManager(child_run).verify(child_run / "checkpoints/latest.json").valid
    )
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in generation.iterdir()
    } == before
    decision = next(
        item
        for item in read_manifest(child_run / "manifest.json")["resource_decisions"]
        if item["kind"] == "budget_extension"
    )
    assert (
        decision["old_max_tokens"],
        decision["new_max_tokens"],
        decision["decay_steps"],
    ) == (128, 256, 4)
    train(extended, run_id="resumed", resume=child_run / "checkpoints/latest.json")
    finished = CheckpointManager(extended.logging.root_dir / "resumed").load(
        extended.logging.root_dir / "resumed/checkpoints/latest.json"
    )
    assert (finished.step, finished.tokens_seen) == (8, 256)
    if explicit_parent_decay:
        from sparselab.training.continuation import _extension_settings

        # Equal tampered files must not authorize a new tokenizer identity.
        (parent_run / "tokenizer.json").write_bytes(b"tampered tokenizer")
        extended.tokenizer.path.write_bytes(b"tampered tokenizer")
        with pytest.raises(ValueError, match="bound run artifact inventory"):
            _extension_settings(
                parent_config, extended, parent_tokenizer=parent_run / "tokenizer.json"
            )


def test_budget_extension_rejects_other_changes_before_child(tmp_path: Path) -> None:
    raw = config(tmp_path).model_dump(mode="json")
    raw["training"].update(max_steps=4, max_tokens=128)
    raw["checkpoint"]["steps"] = [0, 4]
    raw["evaluation"].update(every_steps=4, max_batches=1)
    original = RunConfig.model_validate(raw)
    train(original, run_id="parent")
    checkpoints = original.logging.root_dir / "parent/checkpoints"
    terminal = next(checkpoints.glob("step_00000004_gen_*"))
    nonterminal = next(checkpoints.glob("step_00000000_gen_*"))
    raw["training"].update(max_steps=8, max_tokens=256)
    raw["optimizer"]["decay_steps"] = 4
    raw["checkpoint"]["steps"] = [0, 4, 8]
    valid = RunConfig.model_validate(raw)
    invalid = [
        (nonterminal, valid),
        (
            terminal,
            valid.model_copy(
                update={
                    "dataset": valid.dataset.model_copy(update={"synthetic_seed": 99})
                }
            ),
        ),
        (
            terminal,
            valid.model_copy(
                update={
                    "training": valid.training.model_copy(update={"max_tokens": 128})
                }
            ),
        ),
        (
            terminal,
            valid.model_copy(
                update={
                    "training": valid.training.model_copy(
                        update={"micro_batch_size": 1}
                    )
                }
            ),
        ),
        (
            terminal,
            valid.model_copy(
                update={"optimizer": valid.optimizer.model_copy(update={"peak": 0.004})}
            ),
        ),
        (
            terminal,
            valid.model_copy(
                update={"runtime": valid.runtime.model_copy(update={"backend": "rocm"})}
            ),
        ),
    ]
    for index, (generation, changed) in enumerate(invalid):
        run_id = f"rejected-{index}"
        with pytest.raises(ValueError):
            train(changed, run_id=run_id, extend_budget=generation)
        assert not (valid.logging.root_dir / run_id).exists()


@pytest.mark.parametrize("entrypoint", ["train", "_train_impl"])
def test_preexisting_cancel_marker_stops_before_runtime(
    tmp_path, monkeypatch, entrypoint
):
    from sparselab.training import trainer

    original = config(tmp_path)
    cancel = tmp_path / "cancel"
    cancel.write_text("retain cancellation")

    def forbidden(*args, **kwargs):
        pytest.fail("preexisting cancellation reached runtime, data or model work")

    monkeypatch.setattr(trainer.PyTorchEngine, "validate", forbidden)
    monkeypatch.setattr(trainer, "prepare_data", forbidden)
    monkeypatch.setattr(trainer.PyTorchEngine, "initialize", forbidden)
    with pytest.raises(InterruptedError, match="before run creation"):
        getattr(trainer, entrypoint)(original, run_id="cancelled", cancel_path=cancel)
    assert not (original.logging.root_dir / "cancelled").exists()
    assert cancel.read_text() == "retain cancellation"


def test_late_cancellation_checkpoints_the_last_committed_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.training import trainer

    original = config(tmp_path)
    cancel = tmp_path / "late-cancel"
    checkpoint_due = trainer._checkpoint_due

    def cancel_after_update_check(config, step, *args):
        due = checkpoint_due(config, step, *args)
        if step == 1:
            cancel.touch()
        return due

    with monkeypatch.context() as patch:
        patch.setattr(trainer, "_checkpoint_due", cancel_after_update_check)
        train(original, run_id="cancelled", cancel_path=cancel)
    train(original, run_id="reference", stop_after_step=1)
    observed = CheckpointManager(original.logging.root_dir / "cancelled").load(
        original.logging.root_dir / "cancelled/checkpoints/latest.json"
    )
    expected = CheckpointManager(original.logging.root_dir / "reference").load(
        original.logging.root_dir / "reference/checkpoints/latest.json"
    )
    assert (observed.step, observed.tokens_seen) == (1, 32)
    equal(observed.model, expected.model)
    equal(observed.optimizer, expected.optimizer)
    equal(observed.rng, expected.rng)


def test_completed_training_calibrates_and_reuses_runtime_forecast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from sparselab.training import trainer

    validate_runtime = trainer.PyTorchEngine.validate

    def identified_runtime(engine, config: RunConfig, *, authorization=None):
        return replace(
            validate_runtime(engine, config, authorization=authorization),
            device_name="Fixture CPU",
            physical_device_id="fixture:cpu",
        )

    monkeypatch.setattr(trainer.PyTorchEngine, "validate", identified_runtime)
    original = config(tmp_path)
    short_training = original.training.model_copy(
        update={"max_steps": 3, "max_tokens": 96}
    )
    short = original.model_copy(update={"training": short_training})

    train(short, run_id="calibration")
    train(short, run_id="forecast")
    captured = capsys.readouterr()
    progress_records = []
    for line in captured.err.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(record, dict)
            and record.get("run_id") == "forecast"
            and record.get("phase") == "training"
        ):
            progress_records.append(record)
    assert progress_records[0]["event"] == "started"
    assert progress_records[-1]["event"] == "finished"
    assert progress_records[-1]["completed_work"] == 96
    assert progress_records[-1]["total_work"] == 96

    store = ExperimentStore(short.logging.root_dir)
    calibrations = ExperimentStore.get_all_calibrations(short.logging.root_dir)
    runtime_calibration = next(
        row["observation"]
        for row in calibrations
        if row["run_id"] == "calibration"
        and isinstance(row["observation"], dict)
        and row["observation"].get("kind") == "runtime_optimizer_throughput_v1"
    )
    assert isinstance(runtime_calibration, dict)
    assert runtime_calibration["first_update_discarded"] is True
    assert runtime_calibration["calibration_update_count"] == 2

    progress = next(
        record["payload"]
        for record in store.get_runtime_progress_records(
            short.logging.root_dir, "forecast"
        )
        if record["kind"] == "runtime_progress"
    )
    assert isinstance(progress, dict)
    derived = progress["derived"]
    assert isinstance(derived, dict)
    forecast = derived["planning"]
    assert isinstance(forecast, dict)
    assert forecast["availability"] == "available"
    assert "calibration" in forecast["contributing_run_ids"]
    live = derived["live"]
    assert isinstance(live, dict)
    optimizer_eta = live["optimizer_only_eta"]
    assert isinstance(optimizer_eta, dict)
    assert optimizer_eta["status"] == "complete"
    assert optimizer_eta["low_seconds"] == optimizer_eta["high_seconds"] == 0.0

    final = next(
        record["payload"]
        for record in store.get_runtime_progress_records(
            short.logging.root_dir, "forecast"
        )
        if record["kind"] == "runtime_final_observation"
    )
    assert isinstance(final, dict)
    phases = final["phases"]
    assert isinstance(phases, dict)
    assert phases["generation"]["availability"] == "not_observed"
    assert phases["evaluation"]["availability"] == "not_observed"


@pytest.mark.parametrize("layout", ["story", "contiguous"])
def test_matched_story_resume_and_config_identity(tmp_path: Path, layout: str) -> None:
    from sparselab.config.models import MatchedStoryConfig
    from sparselab.data.packing import prepare_data
    from sparselab.data.story_packing import MatchedStoryDataset
    from sparselab.data.tokenizer import load_tokenizer

    base = config(tmp_path)
    tokenizer = load_tokenizer(base.tokenizer.path)
    prepared = prepare_data(base, tokenizer)
    plan = MatchedStoryDataset(
        prepared.train,
        16,
        tokenizer.token_to_id("<eos>"),
        layout=layout,
        targets_per_update=128,
    )
    measured = base.model_copy(
        update={
            "training": base.training.model_copy(
                update={
                    "matched_stories": MatchedStoryConfig(
                        layout=layout, targets_per_update=128
                    ),
                    "max_steps": plan.horizon(1025),
                    "max_tokens": 1025,
                }
            ),
            "optimizer": base.optimizer.model_copy(update={"warmup_steps": 0}),
        }
    )
    train(measured, run_id="full")
    train(measured, run_id="part", stop_after_step=1)
    source = measured.logging.root_dir / "part/checkpoints/latest.json"
    changed = measured.model_copy(
        update={
            "training": measured.training.model_copy(
                update={
                    "matched_stories": MatchedStoryConfig(
                        layout="contiguous" if layout == "story" else "story",
                        targets_per_update=128,
                    ),
                }
            )
        }
    )
    with pytest.raises(ValueError, match="settings|config"):
        train(changed, run_id="incompatible", resume=source)
    train(measured, run_id="resumed", resume=source)
    snapshots = [
        CheckpointManager(measured.logging.root_dir / name).load(
            measured.logging.root_dir / name / "checkpoints/latest.json"
        )
        for name in ("full", "resumed")
    ]
    left, right = snapshots
    assert left.tokens_seen == right.tokens_seen == 1025
    assert left.cursor == right.cursor
    assert left.step == right.step == plan.horizon(1025)
    equal(left.model, right.model)
    equal(left.optimizer, right.optimizer)
    equal(left.rng, right.rng)
