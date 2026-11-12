"""Lab mode (`klab try`): fast local loop with retained safety rails.

These tests train two-layer, 32-wide synthetic CPU models for a few dozen
steps. They check wiring and the retained rails (record, data identity,
held-out checks, safe cancellation, resource limits), not model quality.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from lab_fixtures import write_smoke_inputs

from klab.cli.main import (
    BUILD_COMMANDS,
    FAST_PATH_COMMANDS,
    RELEASE_COMMANDS,
    build_parser,
)
from klab.lab_mode import (
    EXIT_INTERRUPTED,
    EXIT_NOT_COMPARABLE,
    read_candidate,
    read_record,
    run_try,
)
from klab.resource_envelope import ResourceEnvelope
from klab.training.trainer import train

# Generous: the TODO target is 15 minutes on CPU; a quiet laptop needs ~15 s.
LOOP_BUDGET_SECONDS = float(os.environ.get("KLAB_LAB_LOOP_BUDGET_SECONDS", "900"))


def _delta(root: Path, name: str, settings: dict[str, object]) -> Path:
    path = root / name
    path.write_text(yaml.safe_dump({"question": name, "set": settings}))
    return path


@pytest.fixture
def lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    return tmp_path, write_smoke_inputs(tmp_path)


def test_try_writes_one_sealed_record_with_identity_and_heldout_scores(
    lab: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, baseline = lab
    delta = _delta(root, "wider.yaml", {"model.ffn_dim": 128})
    monkeypatch.chdir(root)
    record, path = run_try(delta, Path(baseline.name), work_dir=root / "work")
    assert record["request"]["baseline"] == baseline.name
    assert record["baseline_provenance"] == {
        "selector": baseline.name,
        "working_directory": str(root),
        "status": "resolved",
        "resolved": {"kind": "config", "path": str(baseline)},
    }

    assert path == root / "work/lab/tries" / record["try_id"] / "try.json"
    assert read_record(path) == record
    assert record["status"] == "completed" and record["exit_status"] == 0
    assert record["mode"] == "lab"
    assert record["delta"] == {"model.ffn_dim": {"base": 96, "variant": 128}}
    assert record["declared_delta"] == {"model.ffn_dim": 128}
    assert record["code"]["source_identity_sha256"]
    assert "campaign_approval_and_reconciliation" in record["skipped_release_gates"]
    for arm in ("baseline", "candidate"):
        row = record["arms"][arm]
        assert row["status"] == "completed"
        assert row["seed"] == 42
        assert row["heldout"]["split"] == "validation"
        assert row["heldout"]["valid_targets"] > 0
        assert Path(row["heldout"]["observation"]).is_file()
        data = row["data"]
        assert len(data["validation_sha256"]) == 64
        assert data["validation_sha256"] != data["train_sha256"]
        assert data["eval_data_sha256"]["validation"] == data["validation_sha256"]
        assert row["checkpoint_sha256"]
    comparison = record["comparison"]
    assert comparison["comparable"] is True
    assert comparison["verdict"] in {
        "CANDIDATE_LOWER_LOSS",
        "CANDIDATE_HIGHER_LOSS",
        "TIE",
    }
    assert comparison["heldout_loss_delta"] == pytest.approx(
        record["arms"]["candidate"]["heldout"]["loss"]
        - record["arms"]["baseline"]["heldout"]["loss"]
    )

    # A second idea against the same baseline reuses the verified baseline run.
    second, _ = run_try(
        _delta(root, "lr.yaml", {"optimizer.peak": 0.006}),
        baseline,
        work_dir=root / "work",
    )
    assert second["arms"]["baseline"]["reused"] is True
    assert second["arms"]["baseline"]["run_id"] == record["arms"]["baseline"]["run_id"]
    assert (
        second["arms"]["candidate"]["run_id"] != record["arms"]["candidate"]["run_id"]
    )


def test_tampered_record_is_rejected(lab: tuple[Path, Path]) -> None:
    root, baseline = lab
    _, path = run_try(
        _delta(root, "d.yaml", {"model.ffn_dim": 64}), baseline, work_dir=root / "work"
    )
    edited = json.loads(path.read_text())
    edited["comparison"]["verdict"] = "CANDIDATE_LOWER_LOSS"
    edited["arms"]["candidate"]["heldout"]["loss"] = 0.0
    path.chmod(0o644)
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="digest mismatch"):
        read_record(path)


@pytest.mark.parametrize(
    ("settings", "failed"),
    [
        ({"dataset.validation_max_tokens": 2048}, "same_validation_data"),
        # The baseline's 32-token eval windows cannot be shared with a model
        # whose context is 16 tokens, so the protocols differ (review fix #2).
        ({"model.max_seq_len": 16, "training.seq_len": 16}, "same_eval_protocol"),
    ],
)
def test_heldout_changes_are_not_comparable(
    lab: tuple[Path, Path], settings: dict[str, object], failed: str
) -> None:
    root, baseline = lab
    record, _ = run_try(
        _delta(root, "h.yaml", settings), baseline, work_dir=root / "work"
    )
    assert record["status"] == "completed"
    assert record["comparison"]["verdict"] == "NOT_COMPARABLE"
    assert failed in record["comparison"]["failed"]
    assert "heldout_loss_delta" not in record["comparison"]
    assert record["exit_status"] == EXIT_NOT_COMPARABLE


def test_startup_cancellation_is_recorded_without_fabricated_checkpoint(
    lab: tuple[Path, Path],
) -> None:
    root, baseline = lab
    calls: list[str] = []

    def cancel_then_train(config, **kwargs):
        calls.append(kwargs["run_id"])
        kwargs["cancel_path"].touch()  # as `touch CANCEL` from another shell
        return train(config, **kwargs)

    record, path = run_try(
        _delta(root, "c.yaml", {"model.ffn_dim": 128}),
        baseline,
        work_dir=root / "work",
        train_fn=cancel_then_train,
    )
    assert record["status"] == "interrupted"
    assert record["exit_status"] == EXIT_INTERRUPTED
    assert record["interruption"] == {
        "arm": "baseline",
        "phase": "training",
        "reason": "cancelled",
    }
    assert "comparison" not in record and "candidate" not in record["arms"]
    assert len(calls) == 1  # the candidate never started
    run = root / "work/lab/runs" / calls[0]
    assert record["arms"] == {}
    assert not run.exists()
    assert read_record(path)["status"] == "interrupted"

    # The sealed interruption is preserved; no baseline exists to reuse.
    record, _ = run_try(
        _delta(root, "c2.yaml", {"model.ffn_dim": 128}),
        baseline,
        work_dir=root / "work",
    )
    assert record["status"] == "completed"
    assert record["arms"]["baseline"]["reused"] is False
    assert record["arms"]["baseline"]["run_id"] != calls[0]
    assert not run.exists()
    assert read_record(path)["status"] == "interrupted"


def test_sigint_from_the_cli_stops_safely_and_records_interruption(
    tmp_path: Path,
) -> None:
    baseline = write_smoke_inputs(tmp_path)
    config = yaml.safe_load(baseline.read_text())
    # Long enough to still be training when SIGINT lands; small enough that the
    # storage preflight (which scales with max_steps) fits an ordinary disk.
    config["training"]["max_steps"] = 20_000
    config["training"]["max_tokens"] = 20_000 * 128
    config["checkpoint"]["every_steps"] = 10_000
    baseline.write_text(yaml.safe_dump(config, sort_keys=False))
    delta = _delta(tmp_path, "s.yaml", {"model.ffn_dim": 128})
    work = tmp_path / "work"
    process = subprocess.Popen(
        [sys.executable, "-m", "klab", "try", str(delta), "--vs", str(baseline)],
        env={**os.environ, "KLAB_WORK_DIR": str(work)},
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            progress = list((work / "lab/runs").glob("*/progress.json"))
            if progress and json.loads(progress[0].read_text()).get("step", 0) > 0:
                break
            assert process.poll() is None, "try exited before training started"
            time.sleep(0.2)
        else:
            pytest.fail("training never started")
        process.send_signal(signal.SIGINT)
        stdout, _ = process.communicate(timeout=300)
    finally:
        if process.poll() is None:
            process.kill()
    assert process.returncode == EXIT_INTERRUPTED
    assert "stopped safely: arm=baseline phase=training reason=signal" in stdout
    (record_path,) = (work / "lab/tries").glob("*/try.json")
    record = read_record(record_path)
    assert record["status"] == "interrupted"
    (progress_path,) = (work / "lab/runs").glob("*/progress.json")
    assert json.loads(progress_path.read_text())["status"] == "interrupted"
    assert (progress_path.parent / "checkpoints/latest.json").is_file()


@pytest.mark.parametrize("explicit_wall,declared_wall", [(None, 1e-9), (1e-9, 60)])
def test_wall_limit_is_opt_in_and_stops_safely(
    lab: tuple[Path, Path], explicit_wall: float | None, declared_wall: float
) -> None:
    root, baseline = lab
    record, _ = run_try(
        _delta(root, "w.yaml", {"model.ffn_dim": 128}),
        baseline,
        work_dir=root / "work",
        max_wall_seconds=explicit_wall,
        resource_envelope=ResourceEnvelope(
            resource_envelope_version=1,
            lab_try={"max_wall_seconds_per_arm": declared_wall},
        ),
    )
    assert record["status"] == "interrupted"
    assert record["interruption"]["reason"] == "wall_time_limit"
    assert record["resources"]["max_wall_seconds_per_arm"] == 1e-9
    assert (
        record["resources"]["resource_envelope"]["lab_try"]["max_wall_seconds_per_arm"]
        == declared_wall
    )


def test_resource_envelope_is_enforced_before_training(
    lab: tuple[Path, Path],
) -> None:
    root, baseline = lab
    calls: list[str] = []
    envelope = ResourceEnvelope(
        resource_envelope_version=1, min_disk_bytes=2**62
    )  # more disk than any host has
    with pytest.raises((ValueError, OSError)):
        run_try(
            _delta(root, "r.yaml", {"model.ffn_dim": 128}),
            baseline,
            work_dir=root / "work",
            resource_envelope=envelope,
            train_fn=lambda *a, **k: calls.append("trained") or "never",
        )
    assert calls == []
    (record_path,) = (root / "work/lab/tries").glob("*/try.json")
    assert read_record(record_path)["status"] == "failed"


def test_storage_preflight_refuses_runs_that_do_not_fit(
    lab: tuple[Path, Path],
) -> None:
    root, baseline = lab
    record, _ = run_try(
        _delta(
            root,
            "huge.yaml",
            {"training.max_steps": 10**9, "training.max_tokens": 128 * 10**9},
        ),
        baseline,
        work_dir=root / "work",
    )
    assert record["status"] == "failed"
    assert record["arms"]["baseline"]["status"] == "completed"
    assert "storage preflight failed" in record["failure"]
    assert "candidate" not in record["arms"]


def test_noop_and_malformed_deltas_are_refused(lab: tuple[Path, Path]) -> None:
    root, baseline = lab
    with pytest.raises(ValueError, match="does not change"):
        run_try(
            _delta(root, "same.yaml", {"model.ffn_dim": 96}),
            baseline,
            work_dir=root / "work",
        )
    bad = root / "bad.yaml"
    bad.write_text("set: {}\n")
    with pytest.raises(ValueError, match="nonempty"):
        read_candidate(bad)
    bad.write_text("sett:\n  model.ffn_dim: 1\n")
    with pytest.raises(ValueError, match="unexpected"):
        read_candidate(bad)
    assert read_candidate(baseline).kind == "config"


def test_help_lists_build_first_and_every_command_once() -> None:
    parser = build_parser()
    parser.format_help()
    epilog = parser.epilog or ""
    (commands,) = [action for action in parser._actions if action.dest == "command"]
    names = set(commands.choices)
    assert {"try", "report", "campaign", "experiment", "attempt"} <= names
    listed = [
        line.split()[0]
        for line in epilog.splitlines()
        if line.startswith("  ") and line.strip()
    ]
    assert sorted(listed) == sorted(names)
    assert listed[: len(BUILD_COMMANDS)] == list(BUILD_COMMANDS)
    assert listed[len(BUILD_COMMANDS) :][: len(FAST_PATH_COMMANDS)] == list(
        FAST_PATH_COMMANDS
    )
    assert listed[-len(RELEASE_COMMANDS) :] == list(RELEASE_COMMANDS)


def test_cpu_smoke_loop_yaml_to_report_within_budget(tmp_path: Path) -> None:
    """Loop-time guard: YAML delta -> scored report through the public CLI."""
    baseline = write_smoke_inputs(tmp_path, train_tokenizer_now=False)
    delta = _delta(tmp_path, "loop.yaml", {"model.ffn_dim": 128, "seed": 9})
    (tmp_path / "bounds.yaml").write_text(
        "resource_envelope_version: 1\nlab_try:\n"
        "  tokenizer_batch_documents: 8\n  tokenizer_batch_source_bytes: 1024\n"
    )
    request = tmp_path / "try-policy.yaml"
    request.write_text("seed: 7\nresource_envelope: bounds.yaml\n")
    environment = {**os.environ, "KLAB_WORK_DIR": str(tmp_path / "work")}

    def cli(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "klab", *args],
            env=environment,
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )

    started = time.monotonic()
    tokenizer = cli("tokenizer", "train", str(tmp_path / "tokenizer.yaml"))
    assert tokenizer.returncode == 0, tokenizer.stderr[-2000:]
    tried = cli(
        "try", str(delta), "--vs", str(baseline), "--config", str(request), "--json"
    )
    assert tried.returncode == 0, tried.stderr[-2000:]
    record = json.loads(tried.stdout)
    report = cli("report", record["try_id"])
    elapsed = time.monotonic() - started
    assert report.returncode == 0, report.stderr[-2000:]
    assert "verdict:" in report.stdout and record["try_id"] in report.stdout
    assert record["comparison"]["comparable"] is True
    assert record["seed"]["baseline"] == record["seed"]["candidate"] == 7
    assert record["request"]["seed"] == 7
    assert record["resources"]["tokenizer_batch_documents"] == 8
    assert record["resources"]["tokenizer_batch_source_bytes"] == 1024
    assert (
        record["resources"]["resource_envelope"]
        == record["request"]["resource_envelope"]
    )
    assert (
        record["resources"]["resource_envelope"]["lab_try"]["tokenizer_batch_documents"]
        == 8
    )
    # The fast probe tier is part of every try and lands in the same record.
    assert record["probe"]["tiers_run"] == ["fast"]
    assert record["probe"]["verdict"]["action"]
    assert "PROBE BATTERY" in report.stdout
    print(f"lab loop YAML->report: {elapsed:.1f}s (budget {LOOP_BUDGET_SECONDS}s)")
    assert elapsed < LOOP_BUDGET_SECONDS, f"lab loop took {elapsed:.1f}s"


# --- Review fixes (PR #60): each test failed before its fix. -----------------


def _local_text_baseline(root: Path) -> Path:
    """A local_text baseline whose train file can be edited between tries."""
    baseline = write_smoke_inputs(root)
    words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "a",
        "lazy",
        "dog",
        "while",
        "birds",
        "sing",
    ]
    for split, count in (("train", 400), ("validation", 60)):
        lines = [
            json.dumps(
                {
                    "text": f"{split} story {n}: "
                    + " ".join(words[(n * 5 + k) % len(words)] for k in range(12))
                }
            )
            for n in range(count)
        ]
        (root / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
    config = yaml.safe_load(baseline.read_text())
    config["dataset"].update(
        {
            "source": "local_text",
            "train_path": str(root / "train.jsonl"),
            "validation_path": str(root / "validation.jsonl"),
            "license": "CC0-1.0",
            "train_max_documents": 400,
            "validation_max_documents": 60,
        }
    )
    config["dataset"].pop("synthetic_seed", None)
    baseline.write_text(yaml.safe_dump(config, sort_keys=False))
    return baseline


def test_edited_training_data_retrains_instead_of_reusing_stale_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    baseline = _local_text_baseline(tmp_path)
    first, _ = run_try(
        _delta(tmp_path, "a.yaml", {"model.ffn_dim": 128}),
        baseline,
        work_dir=tmp_path / "work",
    )
    assert first["comparison"]["comparable"] is True
    # Same paths and config, different training bytes; validation unchanged.
    train = tmp_path / "train.jsonl"
    train.write_text(train.read_text().replace("fox", "cat"))
    second, _ = run_try(
        _delta(tmp_path, "b.yaml", {"optimizer.peak": 0.006}),
        baseline,
        work_dir=tmp_path / "work",
    )
    base, cand = second["arms"]["baseline"], second["arms"]["candidate"]
    assert base["reused"] is False
    assert base["run_id"] != first["arms"]["baseline"]["run_id"]
    assert base["data"]["train_sha256"] == cand["data"]["train_sha256"]
    assert (
        base["data"]["train_sha256"]
        != first["arms"]["baseline"]["data"]["train_sha256"]
    )
    assert (
        base["data"]["validation_sha256"]
        == first["arms"]["baseline"]["data"]["validation_sha256"]
    )
    assert second["comparison"]["comparable"] is True
    assert second["comparison"]["checks"]["same_training_data"] is True
    # Unchanged inputs still reuse the current baseline.
    third, _ = run_try(
        _delta(tmp_path, "c.yaml", {"model.ffn_dim": 64}),
        baseline,
        work_dir=tmp_path / "work",
    )
    assert third["arms"]["baseline"]["reused"] is True
    assert third["arms"]["baseline"]["run_id"] == base["run_id"]


def test_shared_eval_protocol_holds_context_windows_fixed(
    lab: tuple[Path, Path],
) -> None:
    root, baseline = lab
    # Same scored-token count (4x32 vs 8x16), different context boundaries.
    record, _ = run_try(
        _delta(
            root,
            "ctx.yaml",
            {"training.seq_len": 16, "training.micro_batch_size": 8},
        ),
        baseline,
        work_dir=root / "work",
    )
    base, cand = record["arms"]["baseline"], record["arms"]["candidate"]
    assert base["eval_protocol"]["seq_len"] == 32
    assert cand["eval_protocol"] == base["eval_protocol"]
    assert cand["eval_protocol_sha256"] == base["eval_protocol_sha256"]
    assert record["eval_protocol"] == base["eval_protocol"]
    # Before the fix each arm was scored with its own windows (16 vs 32 tokens)
    # while still reporting matching scored-target counts.
    assert cand["heldout"]["batches"] == base["heldout"]["batches"]
    assert record["comparison"]["checks"]["same_eval_protocol"] is True
    assert record["comparison"]["comparable"] is True


def test_cancel_during_baseline_scoring_is_never_reused(
    lab: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.evaluation import inference

    root, baseline = lab
    original = inference.InferenceRun.evaluate

    def cancel_while_scoring(self, **kwargs):  # type: ignore[no-untyped-def]
        for marker in (root / "work/lab/tries").glob("*"):
            (marker / "CANCEL").touch()
        return original(self, **kwargs)

    monkeypatch.setattr(inference.InferenceRun, "evaluate", cancel_while_scoring)
    record, _ = run_try(
        _delta(root, "x.yaml", {"model.ffn_dim": 128}), baseline, work_dir=root / "work"
    )
    assert record["status"] == "interrupted"
    assert record["interruption"] == {
        "arm": "baseline",
        "phase": "scoring",
        "reason": "cancelled",
    }
    interrupted_baseline = record["arms"]["baseline"]["run_id"]
    monkeypatch.setattr(inference.InferenceRun, "evaluate", original)
    again, _ = run_try(
        _delta(root, "y.yaml", {"model.ffn_dim": 128}), baseline, work_dir=root / "work"
    )
    assert again["status"] == "completed"
    assert again["arms"]["baseline"]["reused"] is False
    assert again["arms"]["baseline"]["run_id"] != interrupted_baseline


def test_seed_option_is_authoritative_for_both_arms(lab: tuple[Path, Path]) -> None:
    root, baseline = lab
    record, _ = run_try(
        _delta(root, "s.yaml", {"model.ffn_dim": 128, "seed": 42}),
        baseline,
        work_dir=root / "work",
        seed=17,
    )
    assert record["arms"]["baseline"]["seed"] == 17
    assert record["arms"]["candidate"]["seed"] == 17
    assert record["seed"] == {
        "override": 17,
        "baseline": 17,
        "candidate": 17,
        "changed_by_delta": False,
    }
    assert "seed" not in record["delta"]


def test_seed_change_in_delta_is_an_explicit_changed_variable(
    lab: tuple[Path, Path],
) -> None:
    root, baseline = lab
    record, _ = run_try(
        _delta(root, "s.yaml", {"seed": 43}), baseline, work_dir=root / "work"
    )
    assert record["seed"] == {
        "override": None,
        "baseline": 42,
        "candidate": 43,
        "changed_by_delta": True,
    }
    assert record["changed_variables"] == ["seed"]
    assert record["arms"]["candidate"]["seed"] == 43


# --- Second review (PR #60): each test failed before its fix. ---------------


def _portable_memory_baseline(root: Path) -> tuple[Path, Path]:
    """A baseline that trains with an external portable Engram package."""
    import torch

    from klab.model.portable_engram import export_portable_engram

    baseline = write_smoke_inputs(root)
    package = root / "memory.engram"
    export_portable_engram(
        torch.randn(17, 5, generator=torch.Generator().manual_seed(0)),
        package,
        ngram_size=3,
    )
    config = yaml.safe_load(baseline.read_text())
    config["model"].update(
        {
            "memory": "portable",
            "memory_table_size": 17,
            "memory_ngram_size": 3,
            "memory_dim": 5,
            "memory_package_path": str(package),
        }
    )
    baseline.write_text(yaml.safe_dump(config, sort_keys=False))
    return baseline, package


def test_changed_memory_package_retrains_baseline_and_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    from klab.model.portable_engram import export_portable_engram

    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    baseline, package = _portable_memory_baseline(tmp_path)
    first, _ = run_try(
        _delta(tmp_path, "a.yaml", {"model.ffn_dim": 128}),
        baseline,
        work_dir=tmp_path / "work",
    )
    assert first["comparison"]["comparable"] is True
    old_digest = first["arms"]["baseline"]["memory_packages"][
        "model.memory_package_path"
    ]
    # Same path, same config: new memory contents on disk, not in the delta.
    package.unlink()
    export_portable_engram(
        torch.randn(17, 5, generator=torch.Generator().manual_seed(1)),
        package,
        ngram_size=3,
    )
    second, _ = run_try(
        _delta(tmp_path, "b.yaml", {"model.ffn_dim": 64}),
        baseline,
        work_dir=tmp_path / "work",
    )
    base, cand = second["arms"]["baseline"], second["arms"]["candidate"]
    assert base["reused"] is False
    assert base["memory_packages"] == cand["memory_packages"]
    assert base["memory_packages"]["model.memory_package_path"] != old_digest
    assert second["comparison"]["checks"]["same_memory_packages"] is True
    assert second["memory_packages"]["changed_by_delta"] is False

    # A delta that swaps the package is an explicit changed variable.
    other = tmp_path / "other.engram"
    export_portable_engram(
        torch.randn(17, 5, generator=torch.Generator().manual_seed(2)),
        other,
        ngram_size=3,
    )
    third, _ = run_try(
        _delta(tmp_path, "c.yaml", {"model.memory_package_path": str(other)}),
        baseline,
        work_dir=tmp_path / "work",
    )
    assert third["arms"]["baseline"]["reused"] is True
    assert "model.memory_package_path" in third["changed_variables"]
    assert third["memory_packages"]["changed_by_delta"] is True
    assert third["comparison"]["checks"]["same_memory_packages"] is True
    assert (
        third["arms"]["baseline"]["memory_packages"]
        != third["arms"]["candidate"]["memory_packages"]
    )


def _chat_baseline(root: Path) -> Path:
    """A local_chat baseline whose loss mode can differ on identical tokens."""
    baseline = write_smoke_inputs(root)
    for mode in ("all_tokens", "assistant_only"):
        for split, count in (("train", 300), ("validation", 60)):
            lines = [
                json.dumps(
                    {
                        "format_version": 2,
                        "loss_mode": mode,
                        "messages": [
                            {"role": "user", "content": f"{split} question {n}?"},
                            {
                                "role": "assistant",
                                "content": f"the answer to {split} {n} is {n * 7}.",
                            },
                        ],
                    }
                )
                for n in range(count)
            ]
            (root / f"{split}-{mode}.jsonl").write_text("\n".join(lines) + "\n")
    config = yaml.safe_load(baseline.read_text())
    config["dataset"].update(
        {
            "source": "local_chat",
            "train_path": str(root / "train-all_tokens.jsonl"),
            "validation_path": str(root / "validation-all_tokens.jsonl"),
            "license": "CC0-1.0",
            "train_max_documents": 300,
            "validation_max_documents": 60,
        }
    )
    config["dataset"].pop("synthetic_seed", None)
    baseline.write_text(yaml.safe_dump(config, sort_keys=False))
    return baseline


def test_supervision_mask_is_part_of_the_eval_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    baseline = _chat_baseline(tmp_path)
    record, _ = run_try(
        _delta(
            tmp_path,
            "mask.yaml",
            {
                "dataset.train_path": str(tmp_path / "train-assistant_only.jsonl"),
                "dataset.validation_path": str(
                    tmp_path / "validation-assistant_only.jsonl"
                ),
            },
        ),
        baseline,
        work_dir=tmp_path / "work",
    )
    base, cand = record["arms"]["baseline"], record["arms"]["candidate"]
    # Identical validation tokens; only the scored positions differ.
    assert base["data"]["validation_sha256"] == cand["data"]["validation_sha256"]
    assert base["eval_protocol"]["objective"] == "all_tokens"
    assert cand["eval_protocol"]["objective"] == "token-loss-mask-v1"
    assert (
        base["eval_protocol"]["supervision_mask_sha256"]
        != cand["eval_protocol"]["supervision_mask_sha256"]
    )
    assert record["comparison"]["verdict"] == "NOT_COMPARABLE"
    assert "same_eval_protocol" in record["comparison"]["failed"]


def test_cancel_during_setup_finalizes_the_started_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import klab.lab_mode as lab

    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    baseline = _local_text_baseline(tmp_path)
    original = lab._path_digest
    seen: list[dict[str, object]] = []

    def digest(path: Path):  # type: ignore[no-untyped-def]
        (try_dir,) = (tmp_path / "work/lab/tries").glob("*")
        seen.append(read_record(try_dir / "try.json"))
        (try_dir / "CANCEL").touch()
        return original(path)

    monkeypatch.setattr(lab, "_path_digest", digest)
    record, path = run_try(
        _delta(tmp_path, "c.yaml", {"model.ffn_dim": 128}),
        baseline,
        work_dir=tmp_path / "work",
    )
    # A sealed 'started' record existed before any input hashing.
    assert seen and seen[0]["status"] == "started"
    assert seen[0]["try_id"] == record["try_id"]
    assert record["status"] == "interrupted"
    assert record["interruption"] == {
        "arm": None,
        "phase": "setup",
        "reason": "cancelled",
    }
    assert record["exit_status"] == EXIT_INTERRUPTED
    assert read_record(path) == record
    # Interrupted setup is never a reuse source: the next try trains a baseline.
    monkeypatch.setattr(lab, "_path_digest", original)
    again, _ = run_try(
        _delta(tmp_path, "d.yaml", {"model.ffn_dim": 128}),
        baseline,
        work_dir=tmp_path / "work",
    )
    assert again["status"] == "completed"
    assert again["arms"]["baseline"]["reused"] is False


def test_terminal_checkpoint_bracket_preserves_full_state_and_fresh_baselines(lab):
    from test_training import equal

    from klab.config.loading import load_config
    from klab.training.checkpoints import CheckpointManager
    from klab.training.manifest import sha256_file
    from klab.try_config import TryContinuation

    root, baseline = lab
    config = load_config(baseline)
    train(config, run_id="parent")
    parent_run = config.logging.root_dir / "parent"
    manager = CheckpointManager(parent_run)
    generation = next((parent_run / "checkpoints").glob("step_00000004_gen_*"))
    before = {p.name: sha256_file(p) for p in generation.iterdir()}
    parent = manager.load(generation)
    raw = config.model_dump(mode="json")
    raw["training"].update(max_steps=6, max_tokens=768)
    raw["optimizer"]["decay_steps"] = 4
    baseline.write_text(yaml.safe_dump(raw))
    delta = _delta(
        root,
        "runtime.yaml",
        {
            "training.deterministic": not config.training.deterministic,
        },
    )
    calls = []

    def observed_train(config, **kwargs):
        assert kwargs["extend_budget"] == generation
        calls.append(kwargs["run_id"])
        return train(config, **kwargs)

    record, path = run_try(
        delta,
        baseline,
        work_dir=root / "work",
        repeat_baseline=True,
        continuation=TryContinuation(
            checkpoint=generation, allow_runtime_optimization=True
        ),
        train_fn=observed_train,
        probe_tier=None,
    )
    assert record["status"] == "completed", record.get("failure")
    assert read_record(path) == record
    assert record["execution_order"] == ["baseline", "candidate", "baseline_repeat"]
    assert len(calls) == len(set(calls)) == 3
    children = []
    for arm in record["execution_order"]:
        row = record["arms"][arm]
        assert not row["reused"]
        assert row["reuse_key"] is None
        assert row["training_targets_applied"] == 256
        run = Path(row["run_dir"])
        child = CheckpointManager(run).load(run / "checkpoints/latest.json")
        assert child.parent_checkpoint_sha256 == parent.checkpoint_sha256
        assert child.step == 6 and child.tokens_seen == 768
        assert child.rng and child.optimizer
        assert {int(v["step"]) for v in child.optimizer["state"].values()} == {6}
        assert CheckpointManager(run).verify(run / "checkpoints/latest.json").valid
        children.append(child)
    equal(children[0].model, children[2].model)
    equal(children[0].optimizer, children[2].optimizer)
    equal(children[0].rng, children[2].rng)
    assert children[0].schedule == children[2].schedule
    assert {p.name: sha256_file(p) for p in generation.iterdir()} == before
    ordinary, _ = run_try(delta, baseline, work_dir=root / "work", probe_tier=None)
    assert ordinary["status"] == "completed"
    assert ordinary["arms"]["baseline"]["reused"] is False
    assert ordinary["arms"]["baseline"]["training_targets_applied"] == 768


def test_bracket_cancellation_keeps_completed_arms(lab):
    root, baseline = lab
    calls = []

    def cancel_third(config, **kwargs):
        calls.append(kwargs["run_id"])
        if len(calls) == 3:
            kwargs["cancel_path"].touch()
        return train(config, **kwargs)

    record, path = run_try(
        _delta(root, "bracket.yaml", {"model.ffn_dim": 128}),
        baseline,
        work_dir=root / "work",
        repeat_baseline=True,
        train_fn=cancel_third,
        probe_tier=None,
    )
    assert record["status"] == "interrupted"
    assert record["interruption"]["arm"] == "baseline_repeat"
    assert record["arms"]["baseline"]["status"] == "completed"
    assert record["arms"]["candidate"]["status"] == "completed"
    assert "comparison" not in record
    assert read_record(path) == record
