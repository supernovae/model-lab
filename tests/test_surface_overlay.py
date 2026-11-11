"""Read-only Surface Review status is tied to verified endpoint identity."""

from pathlib import Path

from klab.evaluation import surface_overlay
from klab.evaluation.surface_review import (
    complete_surface_review,
    create_surface_bundle,
    open_surface_bundle,
    record_surface_judgment,
)


def test_review_status_requires_matching_verified_endpoint_and_complete_votes(
    tmp_path: Path, monkeypatch
):
    report = {
        "identity": {"run_id": "run-one", "generation_manifest_sha256": "a" * 64},
        "inputs": {"manifest_sha256": "b" * 64, "tokenizer_sha256": "c" * 64},
        "core": {"integrity": {"status": "PASS", "generation": "step_0001_gen_000001"}},
    }
    monkeypatch.setattr(
        surface_overlay,
        "read_triage",
        lambda run_id, runs_dir: report if run_id == "run-one" else None,
    )
    bundles = tmp_path / "bundles"
    bundles.mkdir()
    source = {
        "kind": "klab_checkpoint",
        "run_id": "run-one",
        "checkpoint_digest": "a" * 64,
        "run_manifest_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
    }
    other = {
        "kind": "klab_checkpoint",
        "run_id": "run-two",
        "checkpoint_digest": "d" * 64,
    }
    cells = [
        {
            "candidate_id": str(i),
            "prompt_id": str(i),
            "prompt": "A prompt",
            "category": "cause_effect",
            "decoder": {"temperature": 0},
            "rng_seed": None,
            "a": {"source": source, "response": "First story"},
            "b": {"source": other, "response": "Second story"},
        }
        for i in range(2)
    ]
    create_surface_bundle(
        cells,
        profile="full",
        selection_seed=1,
        presentation_seed=2,
        output_dir=bundles / "matched",
    )
    status = lambda run: surface_overlay.surface_review_status(run, tmp_path, bundles)
    assert status("run-one")["independent_subjective_quality"] == "REVIEW_AVAILABLE"
    assert status("run-two")["independent_subjective_quality"] == "UNKNOWN"
    cases = open_surface_bundle(bundles / "matched")["blind"]["cases"]
    for case in cases:
        record_surface_judgment(
            bundles / "matched",
            case["blind_case_id"],
            {dimension: "neither" for dimension in case["dimensions"]},
            [],
            "2026-09-28T00:00:00Z",
        )
    assert status("run-one")["independent_subjective_quality"] == "REVIEW_AVAILABLE"
    complete_surface_review(bundles / "matched")
    assert (
        status("run-one")["independent_subjective_quality"]
        == "OBSERVED_SINGLE_REVIEWER"
    )
    report["identity"]["generation_manifest_sha256"] = "f" * 64
    assert status("run-one")["independent_subjective_quality"] == "UNKNOWN"
