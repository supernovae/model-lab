from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_training import config as training_config

from klab.data.packing import prepare_data
from klab.data.tokenizer import load_tokenizer
from klab.operational_monitor import capture_workspace_baseline
from klab.training.attempt_budget import AttemptBudget
from klab.training.attempt_receipts import verify_native_phase_counters
from klab.training.manifest import read_manifest, sha256_file


@pytest.mark.skipif(os.name != "posix", reason="native attempt monitor needs POSIX")
def test_attempt_runs_prepared_fresh_training_to_absolute_stop(tmp_path: Path) -> None:
    stop_step = 4
    target_positions = stop_step * 32
    payload = training_config(tmp_path).model_dump(mode="json")
    payload["training"].update(max_steps=2961, max_tokens=2961 * 32)
    payload["optimizer"]["warmup_steps"] = 100
    payload["evaluation"].update(every_steps=400, max_batches=1)
    from klab.config.models import RunConfig

    config = RunConfig.model_validate(payload)
    prepare_data(config, load_tokenizer(config.tokenizer.path))
    declaration = tmp_path / "run.yaml"
    declaration.write_text(json.dumps(payload))
    runs = config.logging.root_dir
    runs.mkdir()
    policy = tmp_path / "monitor.yaml"
    # The wall limit guards a hung child. Startup time on a busy CI runner
    # is separate from the training bounds exercised below.
    policy.write_text("monitor_policy_version: 1\nmax_wall_seconds: 180\n")
    baseline = tmp_path / "baseline.json"
    baseline_sha = capture_workspace_baseline(tmp_path, baseline).sha256
    digest = sha256_file(declaration)
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "max_optimizer_updates": stop_step,
                "max_actual_target_positions": target_positions,
                "max_generation_calls": 0,
                "max_generated_tokens": 0,
                "max_wall_seconds": 240,
                "content_identity_sha256": digest,
                "monitor_policy_sha256": sha256_file(policy),
                "workspace_baseline_sha256": baseline_sha,
            }
        )
    )
    budget = AttemptBudget.create_contract(
        tmp_path / "claim.sqlite3",
        contract_path=contract,
        expected_sha256=sha256_file(contract),
    )
    native = [sys.executable, "-m", "klab", "--work-dir", str(tmp_path)]

    def run(label: str, updates: int, targets: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                *native,
                "attempt",
                "run",
                "--ledger",
                str(budget.path),
                "--activity",
                "train",
                "--label",
                label,
                "--content-identity-sha256",
                digest,
                "--policy",
                str(policy),
                "--baseline",
                str(baseline),
                "--workspace",
                str(tmp_path),
                "--completion",
                str(tmp_path / f"{label}.owned.json"),
                "--updates",
                str(updates),
                "--target-positions",
                str(targets),
                "--generation-calls",
                "0",
                "--generated-tokens",
                "0",
                "--receipt-kind",
                "train",
                "--receipt-path",
                str(runs / label),
                "--",
                *native,
                "train",
                str(declaration),
                "--runs-dir",
                str(runs),
                "--run-id",
                label,
                "--require-prepared",
                "--stop-after-step",
                str(stop_step),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=210,
        )

    finished = run("bounded", stop_step, target_positions)
    if finished.returncode != 0:
        monitor = tmp_path / "bounded.owned.json.monitor"
        evidence = "\n".join(
            (monitor / name).read_text(errors="replace")[-4000:]
            for name in ("completion.json", "command.stderr.log")
            if (monitor / name).is_file()
        )
        pytest.fail(
            f"attempt exited {finished.returncode}\n{finished.stderr}\n{evidence}"
        )
    (reservation,) = budget.status()["reservations"]
    assert (reservation["actual_updates"], reservation["actual_target_positions"]) == (
        stop_step,
        target_positions,
    )
    bounded = runs / "bounded"
    assert read_manifest(bounded / "manifest.json")["requested_config"] == payload
    progress = bounded / "progress.json"
    altered = json.loads(progress.read_text())
    altered["runtime_forecast"]["final_observed"]["reason"] = "signal"
    progress.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="completion"):
        verify_native_phase_counters("train", bounded, expected_stop_step=stop_step)

    for label, updates, targets in (
        ("step-cap", stop_step - 1, target_positions),
        ("target-cap", stop_step, target_positions - 1),
    ):
        rejected = run(label, updates, targets)
        assert rejected.returncode != 0
        assert "exceeds phase reservation" in rejected.stderr
        assert not (runs / label).exists()
    assert budget.status()["reservations"] == [reservation]
