from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from test_training import config as training_config
from test_training import equal

import sparselab.staging as staging_module
from sparselab.config.models import RunConfig
from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer
from sparselab.experiments.artifacts import verify_artifact
from sparselab.experiments.plan import Artifact
from sparselab.model.portable_engram import export_portable_engram
from sparselab.resource_envelope import ResourceEnvelope
from sparselab.staging import _read_sealed, stage
from sparselab.training import manifest as manifest_module
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import read_manifest
from sparselab.training.stages import ExperimentStage, StageHistory
from sparselab.training.trainer import train

FIXTURES = Path(__file__).with_name("fixtures")


def _config(root: Path) -> RunConfig:
    base = training_config(root)
    return base.model_copy(
        update={
            "training": base.training.model_copy(
                update={
                    "micro_batch_size": 1,
                    "max_steps": 3,
                    "max_tokens": 48,
                }
            ),
            "optimizer": base.optimizer.model_copy(update={"warmup_steps": 1}),
            "checkpoint": base.checkpoint.model_copy(update={"every_steps": 2}),
            "staging": base.staging.model_copy(
                update={"smoke_steps": 2, "warmup_steps": 3}
            ),
        }
    )


def test_warmup_pilots_are_isolated_and_bundle_is_self_contained(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    bundle_root = stage(config, tmp_path / "stage", through="warmup")
    bundle = _read_sealed(bundle_root / "bundle.json")

    direct_id = train(config, run_id="direct")
    direct_manifest = read_manifest(
        config.logging.root_dir / direct_id / "manifest.json"
    )
    direct_progress = json.loads(
        (config.logging.root_dir / direct_id / "progress.json").read_text(
            encoding="utf-8"
        )
    )
    assert direct_manifest["stage_bundle_sha256"] is None
    assert direct_manifest["pilot_reports"] == []
    assert "sha256" not in direct_manifest
    assert {record["stage"] for record in direct_progress["stages"]}.isdisjoint(
        {"SMOKE_TEST", "WARMUP"}
    )

    moved_tokenizer = tmp_path / "moved-tokenizer.json"
    shutil.move(config.tokenizer.path, moved_tokenizer)
    moved_cache = tmp_path / "moved-cache"
    shutil.move(config.dataset.cache_dir, moved_cache)
    assert not config.tokenizer.path.exists()
    assert not config.dataset.cache_dir.exists()

    bundled_id = train(config, run_id="bundled", stage_bundle=bundle_root)
    bundled_manifest = read_manifest(
        config.logging.root_dir / bundled_id / "manifest.json"
    )
    assert bundled_manifest["stage_bundle_sha256"] == bundle["sha256"]
    assert [report["purpose"] for report in bundled_manifest["pilot_reports"]] == [
        "smoke",
        "warmup",
    ]
    assert (config.logging.root_dir / bundled_id / "tokenizer.json").is_file()
    assert (config.logging.root_dir / bundled_id / "data" / "train.npy").is_file()

    direct = CheckpointManager(config.logging.root_dir / direct_id).load(
        config.logging.root_dir / direct_id / "checkpoints/latest.json"
    )
    bundled = CheckpointManager(config.logging.root_dir / bundled_id).load(
        config.logging.root_dir / bundled_id / "checkpoints/latest.json"
    )
    assert (direct.step, direct.tokens_seen, direct.cursor) == (
        bundled.step,
        bundled.tokens_seen,
        bundled.cursor,
    )
    equal(direct.model, bundled.model)
    equal(direct.optimizer, bundled.optimizer)
    equal(direct.rng, bundled.rng)


def test_resource_envelope_rejects_stage_and_train_before_creating_output(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    envelope = ResourceEnvelope(resource_envelope_version=1, min_disk_bytes=1 << 62)
    stage_root = tmp_path / "unused-stage" / "bundle"
    with pytest.raises(ValueError, match="min_disk_bytes"):
        stage(config, stage_root, through="validate", resource_envelope=envelope)
    assert not stage_root.parent.exists()

    run = config.logging.root_dir / "never-started"
    with pytest.raises(ValueError, match="min_disk_bytes"):
        train(config, run_id=run.name, resource_envelope=envelope)
    assert not run.exists()
    assert not config.dataset.cache_dir.exists()


def test_tampered_stage_bundle_rejects_before_run_creation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    bundle_root = stage(config, tmp_path / "stage", through="validate")
    asset = bundle_root / "assets" / "tokenizer.json"
    asset.write_bytes(asset.read_bytes() + b"tampered")

    with pytest.raises(ValueError):
        train(config, run_id="rejected", stage_bundle=bundle_root)

    assert not (config.logging.root_dir / "rejected").exists()


def test_bundle_inventory_hashes_prepared_arrays_once_per_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    bundle = stage(config, tmp_path / "stage", through="validate")
    original = manifest_module.sha256_file
    hashed: list[Path] = []

    def count(path: Path) -> str:
        hashed.append(Path(path))
        return original(path)

    monkeypatch.setattr(manifest_module, "sha256_file", count)
    for attempt in range(2):
        staging_module.verify_stage_bundle(bundle, config)
        arrays = [
            path.name
            for path in hashed
            if path.parent == bundle / "assets" / "data" and path.suffix == ".npy"
        ]
        assert sorted(arrays) == sorted(["train.npy", "validation.npy"] * (attempt + 1))
    train_path = bundle / "assets" / "data" / "train.npy"
    content = bytearray(train_path.read_bytes())
    content[-1] ^= 1
    train_path.write_bytes(content)
    with pytest.raises(ValueError, match="digest mismatch"):
        staging_module.verify_stage_bundle(bundle, config)


def test_prepared_inputs_inventory_hashes_arrays_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    root = staging_module.materialize_prepared_inputs(
        config, tmp_path / "prepared-inputs"
    )
    original = manifest_module.sha256_file
    hashed: list[Path] = []

    def count(path: Path) -> str:
        hashed.append(Path(path))
        return original(path)

    monkeypatch.setattr(manifest_module, "sha256_file", count)
    staging_module.verify_prepared_inputs(root, config)
    assert sorted(
        path.name
        for path in hashed
        if path.parent == root / "assets" / "data" and path.suffix == ".npy"
    ) == ["train.npy", "validation.npy"]


def test_prepared_artifact_memo_is_per_operation_and_fingerprint_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    prepared = prepare_data(config, load_tokenizer(config.tokenizer.path))
    manifest = json.loads((prepared.root / "manifest.json").read_text())
    artifact = Artifact(
        kind="prepared_data",
        version=1,
        producer="sparselab",
        identifier=manifest["settings_sha256"],
        sha256=manifest["manifest_sha256"],
        path=str(prepared.root),
    )
    original = manifest_module.sha256_file
    hashed: list[Path] = []

    def count(path: Path) -> str:
        hashed.append(Path(path))
        return original(path)

    monkeypatch.setattr(manifest_module, "sha256_file", count)
    source = tmp_path / "plan.yaml"
    memo: dict[tuple[object, ...], dict[str, object]] = {}
    verify_artifact(artifact, source, memo=memo)
    verify_artifact(artifact, source, memo=memo)
    arrays = [
        path
        for path in hashed
        if path.parent == prepared.root and path.suffix == ".npy"
    ]
    assert sorted(path.name for path in arrays) == ["train.npy", "validation.npy"]
    verify_artifact(artifact, source, memo={})
    arrays = [
        path
        for path in hashed
        if path.parent == prepared.root and path.suffix == ".npy"
    ]
    assert sorted(path.name for path in arrays) == [
        "train.npy",
        "train.npy",
        "validation.npy",
        "validation.npy",
    ]
    array = prepared.root / "train.npy"
    content = bytearray(array.read_bytes())
    content[-1] ^= 1
    array.write_bytes(content)
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_artifact(artifact, source, memo=memo)


def test_inspect_needs_no_assets_but_failed_preflight_is_not_resumable(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    missing = config.model_copy(
        update={
            "tokenizer": config.tokenizer.model_copy(
                update={"path": tmp_path / "missing.json"}
            )
        }
    )
    inspected = stage(missing, tmp_path / "inspect-only", through="inspect")
    report = _read_sealed(inspected / "stage.json")
    assert report["status"] == "complete"
    assert not (inspected / "assets").exists()

    failed_root = tmp_path / "failed-validate"
    with pytest.raises(Exception, match="missing.json"):
        stage(missing, failed_root, through="validate")

    failed = _read_sealed(failed_root / "stage.json")
    assert failed["status"] == "failed"
    assert not (failed_root / "bundle.json").exists()
    assert {record["stage"] for record in failed["stages"]} == {
        "CONFIGURED",
        "INSPECTED",
        "VALIDATED",
        "FAILED",
    }
    assert all(record["status"] != "resumable" for record in failed["stages"])


def test_over_budget_staging_writes_proposal_without_mutating_config(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    constrained = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "memory": config.runtime.memory.model_copy(
                        update={"budget_bytes": 1}
                    )
                }
            )
        }
    )
    before = constrained.model_dump(mode="json")
    output = tmp_path / "over-budget"

    with pytest.raises(MemoryError, match="exceeds the explicit safe ceiling"):
        stage(constrained, output, through="validate")

    assert constrained.model_dump(mode="json") == before
    assert (output / "proposal.yaml").is_file()
    assert _read_sealed(output / "stage.json")["status"] == "failed"
    assert not (output / "bundle.json").exists()


@pytest.mark.parametrize("package_ngram", [2, 3])
def test_portable_file_bundle_validates_config_and_survives_source_removal(
    tmp_path, package_ngram
):
    base = _config(tmp_path)
    package = tmp_path / "memory.engram"
    table = torch.arange(136, dtype=torch.float32).reshape(17, 8) / 136
    export_portable_engram(table, package, ngram_size=package_ngram)
    payload = base.model_dump(mode="json")
    payload["model"].update(
        memory="portable",
        memory_table_size=17,
        memory_dim=8,
        memory_ngram_size=2,
        memory_package_path=str(package),
    )
    config = RunConfig.model_validate(payload)
    output = tmp_path / "portable-stage"
    if package_ngram != 2:
        with pytest.raises(ValueError):
            stage(config, output, through="validate")
        assert _read_sealed(output / "stage.json")["status"] == "failed"
        assert not (output / "bundle.json").exists()
        return
    stage(config, output, through="validate")
    assert (output / "assets" / "portable_package").is_file()
    package.unlink()
    config.tokenizer.path.unlink()
    shutil.rmtree(config.dataset.cache_dir)
    run_id = train(config, run_id="portable-bundled", stage_bundle=output)
    run = config.logging.root_dir / run_id
    snapshot = CheckpointManager(run).load(run / "checkpoints" / "latest.json")
    assert (snapshot.step, snapshot.tokens_seen) == (3, 48)
    assert torch.equal(snapshot.model["memory.embedding.weight"], table)


def test_stage_history_allows_repeated_evaluation_checkpoint_boundaries() -> None:
    history = StageHistory()
    for stage_name, step, tokens in (
        (ExperimentStage.CONFIGURED, 0, 0),
        (ExperimentStage.INSPECTED, 0, 0),
        (ExperimentStage.VALIDATED, 0, 0),
        (ExperimentStage.TRAINING, 0, 0),
        (ExperimentStage.EVALUATING, 1, 16),
        (ExperimentStage.CHECKPOINTED, 1, 16),
        (ExperimentStage.TRAINING, 1, 16),
        (ExperimentStage.CHECKPOINTED, 2, 32),
        (ExperimentStage.TRAINING, 2, 32),
        (ExperimentStage.EVALUATING, 3, 48),
        (ExperimentStage.CHECKPOINTED, 3, 48),
        (ExperimentStage.COMPLETE, 3, 48),
    ):
        history.start(stage_name, step=step, tokens_seen=tokens)
        history.finish(step=step, tokens_seen=tokens)

    assert [record.stage for record in history.records].count(
        ExperimentStage.CHECKPOINTED
    ) == 3
    assert history.records[-1].stage is ExperimentStage.COMPLETE


def test_stage_history_rejects_backward_counters_and_terminal_reentry() -> None:
    history = StageHistory()
    for stage_name, step, tokens in (
        (ExperimentStage.CONFIGURED, 0, 0),
        (ExperimentStage.INSPECTED, 0, 0),
        (ExperimentStage.VALIDATED, 0, 0),
        (ExperimentStage.TRAINING, 2, 32),
    ):
        history.start(stage_name, step=step, tokens_seen=tokens)
        history.finish(step=step, tokens_seen=tokens)

    with pytest.raises(ValueError, match="counters cannot move backward"):
        history.start(ExperimentStage.CHECKPOINTED, step=1, tokens_seen=32)

    history.start(ExperimentStage.FAILED, step=2, tokens_seen=32)
    history.finish("failed", step=2, tokens_seen=32)
    with pytest.raises(ValueError, match="illegal stage transition"):
        history.start(ExperimentStage.TRAINING, step=2, tokens_seen=32)


@pytest.mark.parametrize(
    ("fixture_name", "supported"),
    [
        ("runtime_stage_wire_v1.json", True),
        ("runtime_stage_wire_v2.json", False),
    ],
)
def test_stage_wire_version_compatibility(
    fixture_name: str, supported: bool, tmp_path: Path
) -> None:
    wire = tmp_path / "bundle.json"
    shutil.copy2(FIXTURES / fixture_name, wire)

    if supported:
        sealed = _read_sealed(wire)
        assert sealed["format_version"] == 1
        assert sealed["status"] == "complete"
    else:
        with pytest.raises(ValueError, match="unsupported stage bundle version"):
            _read_sealed(wire)


def test_fresh_stage_private_links_preserve_canonical_inputs_and_fallback_identity(
    tmp_path: Path, monkeypatch
) -> None:
    import os

    from sparselab.training.manifest import sha256_file

    config = _config(tmp_path)
    original = prepare_data(config, load_tokenizer(config.tokenizer.path))
    source = original.root / "train.npy"
    before = sha256_file(source)
    linked = stage(config, tmp_path / "linked", through="validate")
    private_source = linked / "prepared/assets/data/train.npy"
    private_stage = linked / "assets/data/train.npy"
    assert private_source.stat().st_ino == private_stage.stat().st_ino
    assert private_stage.stat().st_ino != source.stat().st_ino

    link = os.link

    def no_private_link(source_path, target_path, *args, **kwargs):
        if "/prepared/assets/" in str(source_path):
            raise OSError("injected unsupported private hardlink")
        return link(source_path, target_path, *args, **kwargs)

    monkeypatch.setattr(os, "link", no_private_link)
    copied = stage(config, tmp_path / "copied", through="validate")
    assert (
        _read_sealed(linked / "inputs.json")["sha256"]
        == _read_sealed(copied / "inputs.json")["sha256"]
    )
    assert (copied / "prepared/assets/data/train.npy").stat().st_ino != (
        copied / "assets/data/train.npy"
    ).stat().st_ino
    staging_module.verify_stage_bundle(linked, config)
    staging_module.verify_stage_bundle(copied, config)
    with private_stage.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        byte = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([byte[0] ^ 1]))
    assert sha256_file(source) == before
    with pytest.raises(ValueError):
        staging_module.verify_stage_bundle(linked, config)


@pytest.mark.parametrize("operation", ["stage", "materialize"])
def test_owned_directory_publication_never_replaces_a_late_destination(
    tmp_path: Path, monkeypatch, operation: str
) -> None:
    config = _config(tmp_path)
    destination = tmp_path / "contended"
    publish = staging_module._publish_prepared_directory

    def competing_destination(source, target):
        if target == destination:
            target.mkdir()
            (target / "owner.txt").write_bytes(b"competing transaction")
        return publish(source, target)

    monkeypatch.setattr(
        staging_module, "_publish_prepared_directory", competing_destination
    )
    with pytest.raises(FileExistsError):
        if operation == "stage":
            stage(config, destination, through="validate")
        else:
            staging_module.materialize_prepared_inputs(config, destination)
    assert (destination / "owner.txt").read_bytes() == b"competing transaction"
    assert sorted(path.name for path in destination.iterdir()) == ["owner.txt"]


@pytest.mark.parametrize("layout", ["contiguous", "story"])
def test_matched_story_pilots_derive_complete_groups_and_train(
    tmp_path: Path, layout: str
) -> None:
    from sparselab.config.models import MatchedStoryConfig
    from sparselab.data.story_packing import MatchedStoryDataset
    from sparselab.training.pilot import run_pilot

    base = _config(tmp_path)
    tokenizer = load_tokenizer(base.tokenizer.path)
    prepared = prepare_data(base, tokenizer)
    plan = MatchedStoryDataset(
        prepared.train,
        base.training.seq_len,
        tokenizer.token_to_id("<eos>"),
        layout=layout,
        targets_per_update=1,
    )
    config = base.model_copy(
        update={
            "training": base.training.model_copy(
                update={
                    "matched_stories": MatchedStoryConfig(
                        layout=layout, targets_per_update=1
                    ),
                    "micro_batch_size": 16,
                    "max_steps": 3,
                    "max_tokens": plan.targets_for_updates(3),
                }
            )
        }
    )
    root = stage(config, tmp_path / "stage", through="validate")
    # Use only the frozen bundle: neither derivation nor trainer may fall back
    # to mutable caller-owned tokenizer/cache paths.
    config.tokenizer.path.unlink()
    shutil.rmtree(config.dataset.cache_dir)
    for purpose in ("smoke", "warmup"):
        steps = getattr(config.staging, f"{purpose}_steps")
        expected_targets = plan.targets_for_updates(steps)
        assert expected_targets != steps * 16 * config.training.seq_len
        report = _read_sealed(run_pilot(root, purpose))
        assert report["step"] == report["timed_updates"] == steps
        assert report["tokens_seen"] == expected_targets
        assert report["derived_config"]["training"]["max_tokens"] == expected_targets
        assert (
            sum(row["targets"] for row in report["update_observations"])
            == expected_targets
        )
        assert report["checkpoint_reload"] == "verified_full_state_and_finite_forward"
