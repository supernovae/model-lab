"""Matched immutable Tier-1 observations become sealed pairs, never generated fill-ins."""

import hashlib
import itertools
import json
from pathlib import Path

import pytest

from klab.evaluation import surface_triage_import
from klab.evaluation.surface_review import open_surface_bundle


def _report(run_id: str, digest: str, text: str):
    settings = {"max_new_tokens": 32, "seed": 42042, "temperature": 0, "top_k": 0}
    return {
        "identity": {
            "run_id": run_id,
            "manifest_sha256": "a" * 64,
            "generation_manifest_sha256": digest,
            "sha256": "b" * 64,
        },
        "inputs": {"manifest_sha256": "a" * 64, "generation_manifest_sha256": digest},
        "core": {
            "integrity": {
                "status": "PASS",
                "generation": "step_0001_gen_000001",
                "generation_manifest_sha256": digest,
            }
        },
        "tier1": {
            "status": "OBSERVED",
            "settings": {"greedy": settings},
            "greedy": [
                {
                    "id": "simple-continuation",
                    "prompt": "Once upon a time",
                    "text": text,
                    "status": "OBSERVED",
                    "settings": settings,
                }
            ],
            "sampled": [],
        },
    }


def test_triage_import_matches_distinct_verified_sources_without_generating(
    tmp_path: Path, monkeypatch
):
    reports = {
        f"run-{i}": _report(f"run-{i}", f"{i}" * 64, f"Once upon a time {i}")
        for i in range(1, 4)
    }
    inputs = []
    for run_id, report in reports.items():
        runs_dir = tmp_path / run_id
        folder = runs_dir / run_id / "post-train-triage"
        folder.mkdir(parents=True)
        content = json.dumps(report, sort_keys=True).encode()
        (folder / f"{hashlib.sha256(content).hexdigest()}.json").write_bytes(content)
        inputs.append((run_id, runs_dir))
    monkeypatch.setattr(
        surface_triage_import, "read_triage", lambda run_id, runs_dir: reports[run_id]
    )
    result = surface_triage_import.import_triage_reports(
        inputs, tmp_path / "paired", "full", 1, 2
    )
    assert result["eligible_count"] == 3
    view = open_surface_bundle(tmp_path / "paired", private=True)
    actual = {
        frozenset(side["run_id"] for side in case["sides"].values())
        for case in view["provenance"]["cases"]
    }
    assert actual == {frozenset(pair) for pair in itertools.combinations(reports, 2)}
    assert all(
        "overall_preference" in case["dimensions"] for case in view["blind"]["cases"]
    )
    reports["run-3"]["tier1"]["greedy"][0]["prompt"] = "A different prompt"
    # Changed report contents must never be substituted for existing observations.
    with pytest.raises(ValueError, match="artifact changed"):
        surface_triage_import.import_triage_reports(
            inputs, tmp_path / "not-published", "full", 1, 2
        )
    assert not (tmp_path / "not-published").exists()
