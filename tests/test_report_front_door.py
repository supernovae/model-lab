"""Result report consolidation, including offline and failed-evidence reads."""

from __future__ import annotations

import json
import sys

import pytest

from klab.cli import main as cli


def _report(monkeypatch, work, *words):
    monkeypatch.setattr(
        sys, "argv", ["klab", "--work-dir", str(work), "report", *words]
    )
    cli.main()


def _snapshot(root):
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def test_missing_and_invalid_run_inputs_never_create_store(
    tmp_path, monkeypatch, capsys
):
    work = tmp_path / "absent-work"
    runs = tmp_path / "runs"
    for invalid in (False, True):
        if invalid:
            (runs / "run").mkdir(parents=True)
            (runs / "run" / "manifest.json").write_text("{invalid")
        before = _snapshot(tmp_path)
        with pytest.raises(SystemExit) as error:
            _report(
                monkeypatch,
                work,
                "run",
                "--kind",
                "run",
                "--runs-dir",
                str(runs),
                "--json",
            )
        assert error.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["evidence"]["status"] == (
            "invalid_evidence" if invalid else "missing_evidence"
        )
        assert payload["triage"]["status"] == "UNKNOWN"
        assert not work.exists()
        assert _snapshot(tmp_path) == before


def test_run_report_retains_negative_and_rejected_observations(
    tmp_path, monkeypatch, capsys
):
    evidence = {
        "quality_observations": [{"loss_delta": -0.25}],
        "rejected_reports": [{"reason": "report hash mismatch"}],
        "missing_reports": ["step_2"],
    }
    triage = {
        "core": {"integrity": {"status": "PASS"}},
        "triggers": {"collapse": {"fired": True}},
    }
    monkeypatch.setattr(
        "klab.evaluation.evidence.experiment_evidence", lambda path: evidence
    )
    monkeypatch.setattr(
        "klab.evaluation.post_train_triage.read_triage", lambda *args: triage
    )
    monkeypatch.setattr(
        "klab.evaluation.post_train_triage.triage_summary",
        lambda report: "negative diagnostic retained",
    )
    work = tmp_path / "absent-work"
    runs = tmp_path / "runs"
    (runs / "run").mkdir(parents=True)
    (runs / "run" / "manifest.json").write_text("{}")
    words = ["run", "--kind", "run", "--runs-dir", str(runs)]
    _report(monkeypatch, work, *words, "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "format": "klab-run-report-v1",
        "run_id": "run",
        "evidence": evidence,
        "triage": triage,
    }
    _report(monkeypatch, work, *words)
    text = capsys.readouterr().out
    for value in (
        "-0.25",
        "report hash mismatch",
        "step_2",
        "negative diagnostic retained",
        "collapse",
    ):
        assert value in text
    assert not work.exists()


def test_invalid_triage_shape_stays_visible(tmp_path, monkeypatch, capsys):
    from test_post_train_triage import _triage_shape_fixture, _write_triage_shape

    payload = _triage_shape_fixture(tmp_path)
    payload["core"] = None
    path = _write_triage_shape(tmp_path, payload)
    monkeypatch.setattr(
        "klab.evaluation.evidence.experiment_evidence",
        lambda path: {"quality_observations": []},
    )
    # The envelope's evidence reader requires a manifest location; the malformed
    # triage is rejected by the real reader before its manifest-binding check.
    (path.parent.parent / "manifest.json").write_text("{}")
    before = _snapshot(tmp_path)
    _report(
        monkeypatch,
        tmp_path / "absent-work",
        "run",
        "--kind",
        "run",
        "--runs-dir",
        str(tmp_path),
        "--json",
    )
    result = json.loads(capsys.readouterr().out)
    assert result["triage"]["status"] == "UNKNOWN"
    assert "missing core" in result["triage"]["reason"]
    assert _snapshot(tmp_path) == before
    assert not (tmp_path / "absent-work").exists()


@pytest.mark.parametrize("source", ["_json", "_checkpoint_metadata"])
def test_unexpected_triage_reader_bugs_propagate(tmp_path, monkeypatch, source):
    from test_post_train_triage import _triage_shape_fixture, _write_triage_shape

    from klab.evaluation import post_train_triage as triage

    if source == "_checkpoint_metadata":
        from test_post_train_triage import _triage_pass_fixture

        path, _, _ = _triage_pass_fixture(tmp_path)
    else:
        path = _write_triage_shape(tmp_path, _triage_shape_fixture(tmp_path))
        (path.parent.parent / "manifest.json").write_text("{}")
    before = _snapshot(tmp_path)

    def broken(*args):
        raise TypeError("unexpected reader bug")

    monkeypatch.setattr(triage, source, broken)
    monkeypatch.setattr(
        "klab.evaluation.evidence.experiment_evidence",
        lambda path: {"quality_observations": []},
    )
    with pytest.raises(TypeError, match="unexpected reader bug"):
        _report(
            monkeypatch,
            tmp_path / "absent-work",
            "run",
            "--kind",
            "run",
            "--runs-dir",
            str(tmp_path),
            "--json",
        )
    assert _snapshot(tmp_path) == before
    assert not (tmp_path / "absent-work").exists()


@pytest.mark.parametrize("json_output", [True, False])
def test_nonobject_manifest_renders_invalid_evidence_and_unknown_triage(
    tmp_path, monkeypatch, capsys, json_output
):
    from test_post_train_triage import _triage_shape_fixture, _write_triage_shape

    from klab.engines.pytorch import PyTorchEngine
    from klab.evaluation import post_train_triage as triage
    from klab.training import trainer
    from klab.training.metrics import ExperimentStore

    path = _write_triage_shape(tmp_path, _triage_shape_fixture(tmp_path))
    (path.parent.parent / "manifest.json").write_text("[]")
    before = _snapshot(tmp_path)

    def no_compute(*args, **kwargs):
        pytest.fail("report attempted compute or opened a store")

    monkeypatch.setattr(trainer, "train", no_compute)
    monkeypatch.setattr(PyTorchEngine, "__init__", no_compute)
    monkeypatch.setattr(ExperimentStore, "__init__", no_compute)
    monkeypatch.setattr(triage, "generate_with_token_ids", no_compute)
    with pytest.raises(SystemExit) as error:
        _report(
            monkeypatch,
            tmp_path / "absent-work",
            "run",
            "--kind",
            "run",
            "--runs-dir",
            str(tmp_path),
            *(["--json"] if json_output else []),
        )
    assert error.value.code == 1
    output = capsys.readouterr().out
    if json_output:
        result = json.loads(output)
        assert result["evidence"]["status"] == "invalid_evidence"
        assert result["triage"]["status"] == "UNKNOWN"
        assert "manifest.json" in result["triage"]["reason"]
    else:
        assert "invalid_evidence" in output
        assert "UNKNOWN" in output
        assert "manifest.json" in output
    assert _snapshot(tmp_path) == before
    assert not (tmp_path / "absent-work").exists()


def test_attempt_grammar_uses_current_report_path():
    from klab.training.attempt_commands import classify_attempt_command

    command = [
        "klab",
        "report",
        "run",
        "--kind",
        "run",
        "--runs-dir",
        "/runs",
        "--json",
    ]
    assert classify_attempt_command(command).effect == "inspection"
    with pytest.raises(ValueError, match="unreviewed native attempt operation"):
        classify_attempt_command(["klab", "evidence", "run"])


@pytest.mark.slow
def test_tiny_cpu_run_report_uses_native_readers_without_retraining(
    tmp_path, monkeypatch, capsys
):
    from test_training import config

    from klab.evaluation.evidence import experiment_evidence
    from klab.evaluation.post_train_triage import read_triage
    from klab.training import trainer

    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "training-work"))
    declaration = config(tmp_path)
    declaration = declaration.model_copy(
        update={
            "training": declaration.training.model_copy(
                update={"max_steps": 1, "max_tokens": 32}
            ),
            "optimizer": declaration.optimizer.model_copy(update={"warmup_steps": 0}),
        }
    )
    trainer.train(declaration, run_id="measured")
    capsys.readouterr()
    monkeypatch.setattr(
        trainer, "train", lambda *a, **k: pytest.fail("report started training")
    )
    runs = declaration.logging.root_dir
    run = runs / "measured"
    words = ["measured", "--kind", "run", "--runs-dir", str(runs), "--json"]
    before = _snapshot(tmp_path)
    _report(monkeypatch, tmp_path / "absent-work", *words)
    payload = json.loads(capsys.readouterr().out)
    assert payload["evidence"] == experiment_evidence(run)
    assert payload["triage"] == read_triage("measured", runs)
    assert payload["evidence"]["quality_observations"]
    assert _snapshot(tmp_path) == before
    path = next((run / "evaluations").glob("*.json"))
    contents = json.loads(path.read_text())
    contents["loss"] = 999
    path.write_text(json.dumps(contents))
    before = _snapshot(tmp_path)
    _report(monkeypatch, tmp_path / "absent-work", *words)
    tampered = json.loads(capsys.readouterr().out)
    assert path.name in {
        row["path"] for row in tampered["evidence"]["rejected_reports"]
    }
    assert contents["checkpoint"] in tampered["evidence"]["missing_reports"]
    assert _snapshot(tmp_path) == before
    assert not (tmp_path / "absent-work").exists()

    triage_path = next((run / "post-train-triage").glob("*.json"))
    triage_path.write_text("{}")
    before = _snapshot(tmp_path)
    _report(monkeypatch, tmp_path / "absent-work", *words)
    invalid_triage = json.loads(capsys.readouterr().out)
    assert invalid_triage["triage"]["status"] == "UNKNOWN"
    assert "invalid" in invalid_triage["triage"]["reason"]
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    ("kind", "selector"), [("record", "--runs-dir"), ("run", "--lab-dir")]
)
def test_report_rejects_a_selector_for_the_other_kind(
    tmp_path, monkeypatch, kind, selector
):
    with pytest.raises(SystemExit, match=f"{selector} requires --kind"):
        _report(monkeypatch, tmp_path / "work", "x", "--kind", kind, selector, "y")
    assert not (tmp_path / "work").exists()
