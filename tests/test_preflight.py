"""Pretraining preflight: one report that proves the artifacts agree."""

from __future__ import annotations

import json

import preflight_fixtures as fixtures
import pytest

from klab.preflight import (
    Evidence,
    open_gate,
    read_report,
    requires_preflight,
    run_preflight,
    write_report,
)


def _run(setup, **options):
    evidence = Evidence(
        setup.config,
        stage_bundle=setup.stage,
        near_duplicates=setup.near_path,
        near_duplicate_decisions=setup.decisions,
        **options,
    )
    return run_preflight(evidence)


def _statuses(report) -> dict[str, str]:
    return {check["id"]: check["status"] for check in report["checks"]}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    return fixtures.build(tmp_path, monkeypatch)


def test_consistent_artifacts_give_a_sealed_report_the_trainer_accepts(setup, tmp_path):
    report = _run(setup)
    statuses = _statuses(report)
    # Unmeasured engram tables and an unrecorded recipe are warnings, not failures.
    assert {k: v for k, v in statuses.items() if v == "FAIL"} == {}
    assert statuses["warmup"] == statuses["quotas"] == statuses["sources"] == "PASS"
    near = next(c for c in report["checks"] if c["id"] == "near_duplicates")
    # Template-only pairs are shown in the report but need no decision.
    assert near["evidence"]["template_only_count"] == 7
    assert near["evidence"]["undecided_count"] == 0
    assert "7 template-only pair(s) on 2 held-out document(s)" in near["summary"]
    assert report["subject"]["prepared_manifest_sha256"] == "p1"
    assert report["subject"]["stage_bundle_sha256"] == "s" * 64
    warmup = next(c for c in report["checks"] if c["id"] == "warmup")["evidence"]
    assert warmup["devices"] == 1 and warmup["targets_per_second"] == 1000

    path = write_report(report, tmp_path / "out")
    assert read_report(path) == report
    gate = open_gate(str(path), setup.config)
    gate.bind(stage_bundle_sha256="s" * 64, prepared_manifest_sha256="p1")
    assert gate.record["report_sha256"] == report["sha256"]
    with pytest.raises(ValueError, match="different prepared data"):
        gate.bind(stage_bundle_sha256="s" * 64, prepared_manifest_sha256="p2")

    tampered = tmp_path / "tampered.json"
    tampered.write_text(path.read_text().replace('"PASS"', '"WARN"', 1))
    with pytest.raises(ValueError, match="not a sealed preflight report"):
        open_gate(str(tampered), setup.config)


def _rights_rule_removed_a_source(s):
    path = s.report_path
    report = json.loads(path.read_text())
    report["source_ledger"][2]["retained_documents"] = 0
    path.write_text(json.dumps(report))


def _quota_cannot_be_filled(s):
    path = s.mixture / "receipt.json"
    receipt = json.loads(path.read_text())
    receipt["available_unique_positions"]["web"] = 100
    path.write_text(json.dumps(receipt))
    s.receipt.update(receipt)


def _screen_truncated(s):
    s.near["candidates_truncated"] = True
    s.near_path.write_text(json.dumps(s.near))


def _pair_undecided(s):
    s.near["candidates"].append({"left_document_id": "c", "right_document_id": "d"})
    s.near_path.write_text(json.dumps(s.near))


def _peak_exceeds_card(s):
    s.pilot["report"]["observed_peak_bytes"] = 40 * fixtures.GIB


def _tokenizer_never_audited(s):
    manifest = json.loads(s.tokenizer_manifest.read_text())
    del manifest["sampling_audit"]
    s.tokenizer_manifest.write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    ("damage", "failing"),
    [
        (_rights_rule_removed_a_source, "sources"),
        (_quota_cannot_be_filled, "quotas"),
        (_screen_truncated, "near_duplicates"),
        (_pair_undecided, "near_duplicates"),
        (_peak_exceeds_card, "warmup"),
        (_tokenizer_never_audited, "tokenizer"),
    ],
)
def test_each_wrong_setup_fails_its_check_and_blocks_training(
    setup, tmp_path, damage, failing
):
    damage(setup)
    report = _run(setup)
    assert report["status"] == "FAIL"
    assert _statuses(report)[failing] == "FAIL"
    path = write_report(report, tmp_path / "out")
    with pytest.raises(ValueError, match=f"preflight failed: .*{failing}"):
        open_gate(str(path), setup.config)


def test_a_source_losing_most_files_warns_with_its_reasons(setup):
    report = json.loads(setup.report_path.read_text())
    row = report["source_ledger"][0]
    row.update(files_set_aside=3, file_set_aside_share=0.75)
    row["set_aside_reasons"] = [
        {"level": "files", "reason": "rights ineligible", "count": 3}
    ]
    setup.report_path.write_text(json.dumps(report))
    sources = next(c for c in _run(setup)["checks"] if c["id"] == "sources")
    assert sources["status"] == "WARN"
    assert (
        sources["evidence"]["over_threshold"][0]["reasons"][0]["reason"]
        == "rights ineligible"
    )


def test_a_check_that_cannot_run_is_a_failure_not_a_skip(setup):
    report = run_preflight(Evidence(setup.config))  # no stage bundle, no screen
    statuses = _statuses(report)
    assert statuses["warmup"] == statuses["near_duplicates"] == "FAIL"


def test_the_gate_refuses_a_missing_stale_or_unreasoned_preflight(setup, tmp_path):
    assert requires_preflight(setup.config, continuation=False)
    assert not requires_preflight(setup.config, continuation=True)
    with pytest.raises(ValueError, match="needs a preflight"):
        open_gate(None, setup.config)
    with pytest.raises(ValueError, match="needs a reason"):
        open_gate("waive:  ", setup.config)
    assert open_gate("waive:reviewed by owner", setup.config).record == {
        "waived": "reviewed by owner"
    }
    path = write_report(_run(setup), tmp_path / "out")
    changed = setup.config.model_copy(update={"seed": setup.config.seed + 1})
    with pytest.raises(ValueError, match="config_sha256 changed"):
        open_gate(str(path), changed)
    receipt = setup.mixture / "receipt.json"
    receipt.write_text(receipt.read_text() + " ")
    with pytest.raises(ValueError, match="mixture_receipt_sha256 changed"):
        open_gate(str(path), setup.config)


def test_cli_writes_the_report_and_exits_nonzero_on_a_failure(
    setup, tmp_path, monkeypatch, capsys
):
    import sys

    import yaml

    from klab.cli.main import main

    config = tmp_path / "pretrain.yaml"
    config.write_text(yaml.safe_dump(setup.config.model_dump(mode="json")))
    command = [
        "klab", "--work-dir", str(tmp_path / "work"), "preflight", str(config),
        "--output", str(tmp_path / "out"), "--stage-bundle", str(setup.stage),
        "--near-duplicates", str(setup.near_path),
        "--near-duplicate-decisions", str(setup.decisions),
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", command)
    main()
    written = list((tmp_path / "out").glob("preflight-*.json"))
    assert len(written) == 1 and read_report(written[0])["status"] != "FAIL"
    assert "quotas" in capsys.readouterr().out

    setup.pilot["report"]["observed_peak_bytes"] = 99 * fixtures.GIB
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 1
