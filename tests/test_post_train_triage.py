from __future__ import annotations

import json
from pathlib import Path

import pytest

from klab.evaluation.post_train_triage import _advice, loss_trajectory, read_triage


def _points(losses: tuple[float, ...]) -> list[dict[str, object]]:
    return [
        {"step": index, "targets": target, "loss": loss, "evidence": "verified"}
        for index, (target, loss) in enumerate(
            zip((0, 1_000_000, 2_000_000, 4_000_000), losses, strict=True)
        )
    ]


def test_loss_trajectory_uses_exposures_and_endpoint_rates() -> None:
    curve = loss_trajectory(_points((2.0, 1.8, 1.6, 1.595)))
    assert curve["initial_loss"] == 2.0
    assert curve["terminal_loss"] == 1.595
    assert curve["delta_loss"] == pytest.approx(-0.405)
    assert curve["trend"] == "plateau_possible"
    intervals = curve["trend_intervals"]
    assert intervals["early"]["rate"] == pytest.approx(0.2)
    assert intervals["recent"]["rate"] == pytest.approx(0.0025)
    assert intervals["recent"]["from"]["targets"] == 2_000_000
    assert intervals["recent"]["to"]["targets"] == 4_000_000


@pytest.mark.parametrize(
    ("losses", "expected"),
    [
        ((2.0, 1.8, 1.6, 1.61), "regressing"),
        ((2.0, 1.8, 1.6, 1.59), "still_improving"),
        ((2.0, 1.8, 1.6, 1.591), "plateau_possible"),
        ((2.0, 1.8, 1.6, 1.595), "plateau_possible"),
        ((2.0, 1.8, 1.6, 1.55), "still_improving"),
    ],
)
def test_endpoint_trend_boundaries(losses: tuple[float, ...], expected: str) -> None:
    assert loss_trajectory(_points(losses))["trend"] == expected


def test_short_or_unverified_curve_cannot_claim_plateau() -> None:
    assert loss_trajectory(_points((2.0, 1.8, 1.6, 1.595))[:3])["trend"] == (
        "insufficient_history"
    )


def test_decoder_sensitivity_distinguishes_missing_from_stable_and_reduced_repetition() -> (
    None
):
    core = {
        "integrity": {"status": "PASS"},
        "loss": {"trend": "insufficient_history", "points": []},
        "runtime": {},
    }
    tier1 = {"sampled": [], "parameter_inventory": {"total": 1000}}
    parent = {"status": "UNKNOWN"}
    panel = {"status": "UNKNOWN"}
    seed = {"fired": None, "comparisons": [], "reason": "no declared siblings"}

    def outcome() -> tuple[dict, dict, list]:
        return _advice(core, tier1, {}, parent, panel, seed)

    unknown, diagnostics, recommendations = outcome()
    assert unknown["DECODER_SENSITIVITY_DETECTED"]["fired"] is None
    assert diagnostics["DECODER_SENSITIVITY_DETECTED"]["status"] == "UNKNOWN"
    assert recommendations == []
    tier1["sampled"] = [
        {
            "id": "simple-continuation",
            "status": "OBSERVED",
            "trigram_excess_difference": 0,
            "settings": {"seed": seed_id},
        }
        for seed_id in range(8)
    ]
    stable, diagnostics, _ = outcome()
    assert stable["DECODER_SENSITIVITY_DETECTED"]["fired"] is False
    assert diagnostics["DECODER_SENSITIVITY_DETECTED"]["status"] == "NOT_NEEDED"
    tier1["sampled"][0]["trigram_excess_difference"] = -3
    sensitive, diagnostics, recommendations = outcome()
    assert sensitive["DECODER_SENSITIVITY_DETECTED"]["fired"] is True
    assert diagnostics["DECODER_SENSITIVITY_DETECTED"]["status"] == "RECOMMENDED"
    assert recommendations[0]["estimated_cost_class"] == "medium"


def test_viewer_rejects_unsafe_run_and_conflicting_artifacts(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    assert read_triage("run", tmp_path) is None
    with pytest.raises(ValueError):
        read_triage("../other", tmp_path)
    directory = tmp_path / "run" / "post-train-triage"
    directory.mkdir(parents=True)
    (directory / "forged.json").write_text(
        json.dumps({"format": "klab_post_train_triage_v1"})
    )
    with pytest.raises(ValueError):
        read_triage("run", tmp_path)
    (directory / "forged.json").unlink()
    (directory / "outside.json").symlink_to(tmp_path / "unowned.json")
    with pytest.raises(ValueError):
        read_triage("run", tmp_path)


def _triage_shape_fixture(tmp_path: Path) -> dict:
    """A sealed advisory fixture with unknown diagnostics; no model work."""
    from klab.evaluation import post_train_triage as triage

    return {
        "format": triage.FORMAT,
        "inputs": {},
        "core": {
            "integrity": {"status": "UNKNOWN"},
            "loss": {"trend": "insufficient_history", "points": []},
        },
        "tier1": {"status": "UNKNOWN", "greedy": [], "sampled": []},
        "surfaces": {
            "review": {"applicability": "UNKNOWN", "observation": "NOT_TESTED"}
        },
        "triggers": {
            code: {
                "fired": None,
                "operands": {},
                "threshold": None,
                "missing_data_reason": "fixture",
            }
            for code in triage._TRIGGERS
        },
        "diagnostics": {
            code: {
                "status": "UNKNOWN",
                "estimated_cost_class": "negligible",
                "requires_training": False,
            }
            for code in triage._TRIGGERS
        },
        "recommendations": [
            {
                "action": "inspect",
                "trigger": triage._TRIGGERS[0],
                "estimated_cost_class": "negligible",
                "requires_training": False,
            }
        ],
        "known_unknowns": [],
    }


def _write_triage_shape(tmp_path: Path, payload: object) -> Path:
    from klab.canonical import canonical_digest, canonical_json
    from klab.training.manifest import sha256_file

    if isinstance(payload, dict):
        payload["identity"] = {
            "sha256": canonical_digest(payload),
            "run_id": "run",
            "manifest_sha256": canonical_digest({"run_id": "run"}),
        }
    folder = tmp_path / "run" / "post-train-triage"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "fixture.json"
    path.write_bytes(canonical_json(payload) + b"\n")
    sealed = folder / (sha256_file(path) + ".json")
    path.rename(sealed)
    return sealed


@pytest.mark.parametrize(
    "field,value",
    [
        (("core",), None),
        (("recommendations",), {}),
        (("known_unknowns",), {}),
        (("core", "integrity", "status"), []),
        (("core", "loss", "trend"), {}),
        (("tier1", "status"), []),
        (("surfaces", "review", "applicability"), []),
        (("surfaces", "review", "observation"), {}),
        (("diagnostics", "LOSS_BEHAVIOR_DIVERGENCE", "status"), []),
        (("diagnostics", "LOSS_BEHAVIOR_DIVERGENCE", "estimated_cost_class"), {}),
        (("recommendations", 0, "estimated_cost_class"), []),
        ((), []),
    ],
)
def test_malformed_sealed_triage_is_value_error_and_cli_unknown(tmp_path, field, value):
    from klab.cli.main import _triage_report

    payload = _triage_shape_fixture(tmp_path)
    if field:
        current = payload
        for key in field[:-1]:
            current = current[key]
        current[field[-1]] = value
    else:
        payload = value
    path = _write_triage_shape(tmp_path, payload)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        read_triage("run", tmp_path)
    report, reason = _triage_report("run", tmp_path)
    assert report is None
    assert "invalid or ambiguous artifact" in reason
    assert path.read_bytes() == before


def test_valid_shape_fixture_and_nonobject_progress(tmp_path, monkeypatch):
    from klab.evaluation import post_train_triage as triage

    payload = _triage_shape_fixture(tmp_path)
    path = _write_triage_shape(tmp_path, payload)
    (path.parent.parent / "manifest.json").write_text("{}")
    before = path.read_bytes()
    monkeypatch.setattr(triage, "read_manifest", lambda path: {"run_id": "run"})
    progress = tmp_path / "run" / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "status": "completed",
                "manifest_sha256": payload["identity"]["manifest_sha256"],
            }
        )
    )
    assert read_triage("run", tmp_path) == payload
    progress.write_text("[]")
    with pytest.raises(ValueError, match="expected JSON object"):
        read_triage("run", tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("value", [[], "scalar", 42, None, True])
def test_nonobject_manifest_is_local_value_error(tmp_path, value):
    from klab.training.manifest import read_manifest

    path = _write_triage_shape(tmp_path, _triage_shape_fixture(tmp_path))
    manifest = path.parent.parent / "manifest.json"
    manifest.write_text(json.dumps(value))
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    # The native reader's contract for its other callers remains unchanged.
    with pytest.raises(TypeError, match="manifest must be a JSON object"):
        read_manifest(manifest)
    with pytest.raises(ValueError, match="expected JSON object: manifest.json"):
        read_triage("run", tmp_path)
    assert before == {
        str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("requested_config", []),
        ("effective_config", 1),
        ("model", []),
        ("attention", []),
        ("memory", []),
        ("artifacts", {}),
    ],
)
def test_manifest_nested_shape_errors_are_local_value_error(tmp_path, field, value):
    from klab.canonical import canonical_digest
    from klab.training.manifest import (
        ARCHITECTURE_VERSION,
        IDENTITY_VERSION,
        architecture_sha256,
        config_sha256,
        read_manifest,
    )

    path = _write_triage_shape(tmp_path, _triage_shape_fixture(tmp_path))
    data = {
        "manifest_version": 1,
        "architecture_version": ARCHITECTURE_VERSION,
        "identity_version": IDENTITY_VERSION,
        "requested_config": {},
        "effective_config": {"model": {}, "attention": {}},
        "artifacts": [],
    }
    data["architecture_sha256"] = architecture_sha256(data["effective_config"])
    if field in {"model", "attention"}:
        data["effective_config"][field] = value
    elif field == "memory":
        data["effective_config"]["model"][field] = value
        data["effective_config"]["model"]["memory_table_size"] = 2
    else:
        data[field] = value
    for config_name in ("requested_config", "effective_config"):
        config = data[config_name]
        data[config_name + "_sha256"] = (
            config_sha256(config) if isinstance(config, dict) else "malformed"
        )
    data["sha256"] = canonical_digest(data)
    manifest = path.parent.parent / "manifest.json"
    manifest.write_text(json.dumps(data))
    before = manifest.read_bytes(), path.read_bytes()
    with pytest.raises(TypeError):
        read_manifest(manifest)
    with pytest.raises(ValueError, match="post-train triage manifest"):
        read_triage("run", tmp_path)
    assert (manifest.read_bytes(), path.read_bytes()) == before


def _triage_pass_fixture(tmp_path):
    from klab.canonical import canonical_digest, canonical_json

    manifest = {"manifest_version": 1, "run_id": "run"}
    manifest["sha256"] = canonical_digest(manifest)
    bound_manifest = {key: value for key, value in manifest.items() if key != "sha256"}
    payload = _triage_shape_fixture(tmp_path)
    checkpoint = {
        "manifest_sha256": canonical_digest(bound_manifest),
        "step": 0,
        "tokens_seen": 0,
    }
    checkpoint["sha256"] = canonical_digest(checkpoint)
    generation = "step_00000000_gen_000001"
    payload["core"]["integrity"] = {
        "status": "PASS",
        "generation": generation,
        "generation_manifest_sha256": checkpoint["sha256"],
    }
    path = _write_triage_shape(tmp_path, payload)
    # Bind the fixture to the verified legacy-format run manifest.
    payload.pop("identity")
    payload["identity"] = {
        "run_id": "run",
        "sha256": canonical_digest(payload),
        "manifest_sha256": canonical_digest(bound_manifest),
        "generation_manifest_sha256": checkpoint["sha256"],
    }
    path.unlink()
    raw = canonical_json(payload) + b"\n"
    from klab.training.manifest import sha256_file

    path.write_bytes(raw)
    sealed = path.with_name(sha256_file(path) + ".json")
    path.rename(sealed)
    run = path.parent.parent
    (run / "manifest.json").write_bytes(canonical_json(manifest))
    checkpoint_path = run / "checkpoints" / generation / "manifest.json"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(canonical_json(checkpoint))
    progress = {
        "status": "completed",
        "manifest_sha256": canonical_digest(bound_manifest),
        "step": 0,
        "tokens_seen": 0,
        "latest": {
            "relative_path": generation,
            "manifest_sha256": checkpoint["sha256"],
        },
    }
    progress_path = run / "progress.json"
    progress_path.write_bytes(canonical_json(progress))
    return sealed, checkpoint_path, progress_path


@pytest.mark.parametrize(
    "broken",
    [
        "checkpoint_array",
        "checkpoint_scalar",
        "missing_step",
        "bad_step",
        "bad_tokens",
        "negative_step",
        "checkpoint_bool",
        "progress_step_bool",
        "progress_tokens_scalar",
        "latest_array",
        "latest_scalar",
    ],
)
def test_pass_binding_rejects_malformed_checkpoint_and_progress(tmp_path, broken):
    from klab.canonical import canonical_digest, canonical_json

    path, checkpoint, progress = _triage_pass_fixture(tmp_path)
    assert read_triage("run", tmp_path)["core"]["integrity"]["status"] == "PASS"
    if broken.startswith("checkpoint_"):
        checkpoint.write_text("[]" if broken.endswith("array") else "42")
    elif broken in {
        "missing_step",
        "bad_step",
        "bad_tokens",
        "negative_step",
        "checkpoint_bool",
    }:
        data = json.loads(checkpoint.read_text())
        data.pop("sha256")
        if broken == "missing_step":
            data.pop("step")
        elif broken == "negative_step":
            data["step"] = -1
        elif broken == "checkpoint_bool":
            data["step"] = False
        else:
            data["step" if broken == "bad_step" else "tokens_seen"] = []
        data["sha256"] = canonical_digest(data)
        checkpoint.write_bytes(canonical_json(data))
        # Keep digest/counter bindings matched so only malformed counter shape
        # distinguishes this from the valid fixture.
        report = json.loads(path.read_text())
        identity = report.pop("identity")
        report["core"]["integrity"]["generation_manifest_sha256"] = data["sha256"]
        identity["generation_manifest_sha256"] = data["sha256"]
        identity["sha256"] = canonical_digest(report)
        report["identity"] = identity
        path.write_bytes(canonical_json(report) + b"\n")
        from klab.training.manifest import sha256_file

        sealed = path.with_name(sha256_file(path) + ".json")
        path.rename(sealed)
        progress_data = json.loads(progress.read_text())
        progress_data["latest"]["manifest_sha256"] = data["sha256"]
        progress_data["step"] = data.get("step")
        progress_data["tokens_seen"] = data.get("tokens_seen")
        progress.write_text(json.dumps(progress_data))
    else:
        data = json.loads(progress.read_text())
        if broken == "progress_step_bool":
            data["step"] = False
        elif broken == "progress_tokens_scalar":
            data["tokens_seen"] = "0"
        else:
            data["latest"] = [] if broken.endswith("array") else 42
        progress.write_text(json.dumps(data))
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        read_triage("run", tmp_path)
    assert before == {
        str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }
