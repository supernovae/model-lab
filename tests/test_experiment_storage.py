from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_training import config as training_config

from klab.config.models import MatchedStoryConfig, RunConfig
from klab.data.packing import prepare_data
from klab.data.tokenizer import load_tokenizer
from klab.experiments.storage import checkpoint_generation_bounds, storage_preview
from klab.workspace_preflight import require_storage, training_storage_checks


def configured(tmp_path: Path, *, updates: int = 20, **cadence: object) -> RunConfig:
    base = training_config(tmp_path)
    checkpoint = base.checkpoint.model_copy(
        update={
            "every_steps": None,
            "every_tokens": None,
            "every_minutes": None,
            **cadence,
        }
    )
    return base.model_copy(
        update={
            "training": base.training.model_copy(
                update={"max_steps": updates, "max_tokens": updates * 100_000}
            ),
            "checkpoint": checkpoint,
            "evaluation": base.evaluation.model_copy(update={"every_steps": 2048}),
        }
    )


@pytest.mark.parametrize("save_best", [True, False])
def test_coincident_periodic_and_validation_without_minutes(
    tmp_path: Path, save_best: bool
) -> None:
    cfg = configured(
        tmp_path, updates=5525, every_steps=2048, save_on_new_best=save_best
    )
    result = storage_preview(cfg)
    assert result["write_count_upper"] == 4
    assert result["periodic_step_trigger_upper"] == 2
    assert result["validation_triggered_best_upper"] == (3 if save_best else 1)
    assert (
        result["estimated_total_write_bytes_upper"]
        == result["estimated_retained_bytes_upper"]
    )


def test_explicit_token_and_minute_cadences(tmp_path: Path) -> None:
    cfg = configured(tmp_path, updates=10, steps=(2, 4))
    assert storage_preview(cfg)["write_count_upper"] == 4
    assert checkpoint_generation_bounds(cfg, stop_after_step=4)["upper"] == 3
    assert checkpoint_generation_bounds(cfg, stop_after_step=20)["upper"] == 4
    token = cfg.model_copy(
        update={
            "checkpoint": cfg.checkpoint.model_copy(
                update={"steps": (), "every_tokens": 2 * 16 * 2}
            )
        }
    )
    assert storage_preview(token)["write_count_upper"] == 6
    minute = cfg.model_copy(
        update={
            "checkpoint": cfg.checkpoint.model_copy(
                update={"steps": (), "every_minutes": 0.1}
            )
        }
    )
    assert storage_preview(minute)["write_count_upper"] == 11
    assert checkpoint_generation_bounds(minute, stop_after_step=5)["upper"] == 6
    assert (
        checkpoint_generation_bounds(
            cfg, stop_after_step=5, operational_checkpoints=True
        )["upper"]
        == 6
    )


def test_best_and_retention_respect_manager_protection(tmp_path: Path) -> None:
    cfg = configured(tmp_path, updates=10, every_steps=1)
    assert storage_preview(cfg)["write_count_upper"] == 11
    requested = storage_preview(cfg, {"keep_periodic": False})
    # Experiment post-run retention is not the trainer's pruning policy.
    assert requested["retained_generations_upper"] == 11
    assert requested["peak_generations_upper"] == 11
    own_policy = cfg.model_copy(
        update={
            "checkpoint": cfg.checkpoint.model_copy(update={"keep_periodic": False})
        }
    )
    limited = storage_preview(own_policy)
    assert limited["retained_generations_upper"] == 3
    assert limited["peak_generations_upper"] == 4
    assert (
        limited["estimated_total_write_bytes_upper"]
        > limited["estimated_peak_checkpoint_bytes_upper"]
    )
    only_best = configured(tmp_path, updates=5525, every_steps=10_000)
    assert storage_preview(only_best)["write_count_upper"] == 4


def test_verified_cache_is_existing_but_run_copy_is_new(tmp_path: Path) -> None:
    cfg = configured(tmp_path, every_steps=5)
    prepared = prepare_data(cfg, load_tokenizer(cfg.tokenizer.path))
    preview = storage_preview(cfg, verified_prepared=prepared.receipt)
    size = sum(
        path.stat().st_size for path in prepared.root.rglob("*") if path.is_file()
    )
    assert preview["existing_prepared_bytes"] == size
    assert preview["future_cache_growth_bytes"] == 0
    assert preview["future_run_copy_bytes"] == size
    assert storage_preview(cfg)["future_cache_growth_bytes"] > 0
    other = cfg.model_copy(
        update={"dataset": cfg.dataset.model_copy(update={"train_max_tokens": 4096})}
    )
    with pytest.raises(ValueError, match="run configuration"):
        storage_preview(other, verified_prepared=prepared.receipt)
    with pytest.raises(TypeError, match="sealed"):
        storage_preview(cfg, verified_prepared={"size_bytes": 1})
    (prepared.root / "train.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="proof|metadata"):
        storage_preview(cfg, verified_prepared=prepared.receipt)


def test_authenticated_cache_changes_actual_free_space_acceptance(
    tmp_path, monkeypatch
):
    import klab.workspace_preflight as preflight

    cfg = configured(tmp_path, every_steps=5)
    prepared = prepare_data(cfg, load_tokenizer(cfg.tokenizer.path))
    baseline = training_storage_checks(
        cfg, work_dir=tmp_path / "work", verified_prepared=prepared.receipt
    )
    assert len(baseline) == 1
    required = baseline[0].projected_bytes + baseline[0].reserve_bytes
    real = preflight.os.statvfs

    def just_enough(path):
        original = real(path)
        return SimpleNamespace(
            f_bavail=(required + original.f_frsize - 1) // original.f_frsize,
            f_frsize=original.f_frsize,
            f_favail=original.f_favail,
            f_files=original.f_files,
        )

    monkeypatch.setattr(preflight.os, "statvfs", just_enough)
    bound = training_storage_checks(
        cfg, work_dir=tmp_path / "work", verified_prepared=prepared.receipt
    )
    unbound = training_storage_checks(cfg, work_dir=tmp_path / "work")
    assert bound[0].status == "adequate"
    assert unbound[0].status == "insufficient"


def test_insufficient_future_growth_is_not_excused_by_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = configured(tmp_path, every_steps=5)
    prepared = prepare_data(cfg, load_tokenizer(cfg.tokenizer.path))
    import klab.workspace_preflight as preflight

    real = preflight.os.statvfs

    def low_space(path):
        original = real(path)
        return SimpleNamespace(
            f_bavail=1,
            f_frsize=original.f_frsize,
            f_favail=original.f_favail,
            f_files=original.f_files,
        )

    monkeypatch.setattr(preflight.os, "statvfs", low_space)
    checks = training_storage_checks(cfg, verified_prepared=prepared.receipt)
    assert checks[0].status == "insufficient"
    assert checks[0].projected_bytes >= sum(
        proof.size_bytes for proof in prepared.receipt.proofs.values()
    )
    with pytest.raises(OSError, match="storage preflight failed"):
        require_storage(checks)


def test_resumed_storage_counts_only_future_target_cadence(tmp_path: Path) -> None:
    cfg = configured(tmp_path, updates=14087, every_tokens=5000000, keep_periodic=True)
    cfg = cfg.model_copy(
        update={
            "training": cfg.training.model_copy(
                update={
                    "max_tokens": 144235834,
                    "matched_stories": MatchedStoryConfig(layout="story"),
                }
            ),
            "evaluation": cfg.evaluation.model_copy(update={"every_steps": 100}),
        }
    )
    future = checkpoint_generation_bounds(
        cfg, completed_steps=13500, completed_tokens=66864692
    )
    assert future["steps"] == 587
    assert future["scheduled_tokens"] == 15
    assert future["validation_steps"] == 5
    assert future["peak_generations"] == 22
    old = training_storage_checks(cfg)
    remaining = training_storage_checks(
        cfg, completed_steps=13500, completed_tokens=66864692
    )
    assert sum(r.projected_bytes for r in remaining) < sum(
        r.projected_bytes for r in old
    )


def test_cold_cache_refuses_before_packing_when_the_run_cannot_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Room for the cache alone is not enough: the run copy and checkpoints count."""
    import klab.workspace_preflight as preflight
    from klab.training.trainer import train
    from klab.workspace_preflight import projected_data_bytes

    # Keep the work root on the same device as the run, whatever HOME is.
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    cfg = configured(tmp_path, every_steps=5)
    (combined,) = training_storage_checks(cfg)
    cache_only = projected_data_bytes(cfg) + combined.reserve_bytes
    room = (cache_only + combined.projected_bytes + combined.reserve_bytes) // 2
    assert cache_only < room < combined.projected_bytes + combined.reserve_bytes
    real = preflight.os.statvfs

    def partly_full(path):
        original = real(path)
        return SimpleNamespace(
            f_bavail=room // original.f_frsize,
            f_frsize=original.f_frsize,
            f_favail=original.f_favail,
            f_files=original.f_files,
        )

    monkeypatch.setattr(preflight.os, "statvfs", partly_full)
    with pytest.raises(OSError, match="storage preflight failed"):
        train(cfg, run_id="does-not-fit")
    assert not list(cfg.dataset.cache_dir.glob("*/manifest.json"))


def test_training_storage_admits_only_the_invocations_checkpoint_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long optimization schedule can run in short, resumable segments."""
    import klab.workspace_preflight as preflight
    from klab.training.manifest import read_manifest
    from klab.training.trainer import train

    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    base = configured(tmp_path, updates=244_141, every_steps=4, keep_periodic=True)
    cfg = base.model_copy(
        update={
            "optimizer": base.optimizer.model_copy(update={"warmup_steps": 2048}),
            "evaluation": base.evaluation.model_copy(update={"max_batches": 1}),
        }
    )
    real = preflight.os.statvfs

    def short_run_space(path):
        original = real(path)
        return SimpleNamespace(
            f_bavail=(512 * 1024**2) // original.f_frsize,
            f_frsize=original.f_frsize,
            f_favail=4096,
            f_files=original.f_files,
        )

    monkeypatch.setattr(preflight.os, "statvfs", short_run_space)

    with pytest.raises(OSError, match="storage preflight failed"):
        train(cfg, run_id="full")
    with pytest.raises(ValueError, match="current step"):
        train(cfg, run_id="invalid-stop", stop_after_step=0)
    for stop in (1.5, True):
        with pytest.raises(ValueError, match="must be an integer"):
            train(cfg, run_id="invalid-stop", stop_after_step=stop)
    assert not list(cfg.dataset.cache_dir.glob("*/manifest.json"))
    assert not (cfg.logging.root_dir / "full").exists()
    assert not (cfg.logging.root_dir / "invalid-stop").exists()

    parent = cfg.logging.root_dir / "parent"
    train(cfg, run_id=parent.name, stop_after_step=8)
    child = cfg.logging.root_dir / "child"
    with pytest.raises(ValueError, match="current step"):
        train(
            cfg,
            run_id=child.name,
            resume=parent / "checkpoints/latest.json",
            stop_after_step=8,
        )
    assert not child.exists()
    train(
        cfg,
        run_id=child.name,
        resume=parent / "checkpoints/latest.json",
        stop_after_step=14,
    )
    for run, steps in ((parent, [0, 4, 8]), (child, [8, 12, 14])):
        checkpoints = sorted((run / "checkpoints").glob("step_*_gen_*/manifest.json"))
        assert [json.loads(path.read_text())["step"] for path in checkpoints] == steps
        manifest = read_manifest(run / "manifest.json")
        assert manifest["requested_config"] == cfg.model_dump(mode="json")
        assert (
            json.loads((run / "progress.json").read_text())["status"] == "interrupted"
        )
