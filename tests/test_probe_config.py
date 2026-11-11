"""Probe declaration contracts and public wiring, metadata only (no models)."""

import argparse
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_lab_results import _cli, _record, _run

from klab.probes.config import ProbeConfig, load_probe_config


@pytest.mark.parametrize(
    ("name", "tier", "final"),
    [
        ("fast", "fast", False),
        ("standard", "standard", False),
        ("full", "full", False),
        ("final", "standard", True),
        ("final-full", "full", True),
    ],
)
def test_named_declarations_match_validated_files(
    tmp_path: Path, name: str, tier: str, final: bool
) -> None:
    request = tmp_path / "probe.yaml"
    request.write_text(yaml.safe_dump({"tier": tier, "final": final}))
    assert load_probe_config(name) == load_probe_config(str(request))
    assert load_probe_config(name).fast_fail is True


@pytest.mark.parametrize("document", ["", "# defaults\n", "null", "{}"])
def test_empty_declaration_uses_default_policy(tmp_path: Path, document: str) -> None:
    request = tmp_path / "probe.yaml"
    request.write_text(document)
    assert load_probe_config(str(request)) == ProbeConfig()


@pytest.mark.parametrize(
    "document",
    [
        "[]",
        "false",
        "5",
        "tier: [",
        "unknown: true",
        "tier: bad",
        "probe_config_version: true",
        "probe_config_version: 1.0",
        "final: 'true'",
        "fast_fail: 0",
        "final: true",
        "lab_dir: 5",
        "tier: full\nlm_eval_tasks: piqa",
        "tier: full\nlm_eval_tasks: [1]",
        "tier: full\nlm_eval_tasks: []",
        "tier: full\nlm_eval_tasks: [piqa, piqa]",
        "tier: full\nlm_eval_tasks: [' piqa']",
        "tier: full\nlm_eval_tasks: ['']",
        "tier: full\nlm_eval_limit: 0",
        "tier: full\nlm_eval_limit: true",
        "tier: full\nlm_eval_limit: 1.0",
        "lm_eval_tasks: [piqa]",
        "lm_eval_limit: 50",
    ],
)
def test_invalid_declaration_fails_before_runtime_or_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, document: str
) -> None:
    from klab import runtime_environments
    from klab.cli.main import _prepare_runtime_command, build_parser

    request = tmp_path / "probe.yaml"
    request.write_text(document)
    monkeypatch.setattr(
        runtime_environments,
        "profile_for_id",
        lambda *_: pytest.fail("runtime reached"),
    )
    args = build_parser(tmp_path).parse_args(
        ["probe", "missing", "--config", str(request), "--runtime", "fixture"]
    )
    with pytest.raises(ValueError):
        _prepare_runtime_command(args)
    assert not (tmp_path / "lab").exists()


def test_paths_expand_home_and_anchor_to_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(task_home))
    request = tmp_path / "requests/probe.yaml"
    request.parent.mkdir()
    request.write_text(
        "lab_dir: ~/lab\nruns_dir: ../runs\nruntime_profile: ../profile.yaml\nresource_envelope: ~/bounds.yaml\n"
    )
    config = load_probe_config(str(request))
    assert config.lab_dir == task_home / "lab"
    assert config.runs_dir == tmp_path / "runs"
    assert config.runtime_profile == tmp_path / "profile.yaml"
    assert config.resource_envelope == task_home / "bounds.yaml"


def test_probe_has_five_executable_options_and_removes_old_flags(
    tmp_path: Path,
) -> None:
    from klab.cli.main import build_parser

    parser = build_parser(tmp_path)
    commands = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    options = {
        flag
        for a in commands.choices["probe"]._actions
        for flag in a.option_strings
        if a.dest != "help"
    }
    assert options == {"--vs", "--config", "--backend", "--runtime", "--json"}
    for flag, values in {
        "--tier": ["standard"],
        "--final": [],
        "--no-fast-fail": [],
        "--lm-eval-tasks": ["piqa"],
        "--lm-eval-limit": ["50"],
        "--lab-dir": ["lab"],
        "--runs-dir": ["runs"],
        "--runtime-profile": ["profile.yaml"],
        "--resource-envelope": ["bounds.yaml"],
    }.items():
        with pytest.raises(SystemExit):
            parser.parse_args(["probe", "candidate", flag, *values])


def test_runtime_selectors_conflict_before_lookup(tmp_path: Path) -> None:
    from klab.cli.main import _prepare_runtime_command, build_parser

    request = tmp_path / "probe.yaml"
    request.write_text("runtime_profile: missing.yaml")
    args = build_parser(tmp_path).parse_args(
        ["probe", "missing", "--config", str(request), "--runtime", "fixture"]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _prepare_runtime_command(args)


def test_validated_policy_and_native_envelope_reach_public_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import klab.probes.cli as probes

    lab = tmp_path / "custom-lab"
    target = _record(lab, _run(lab, backend="cuda"))
    request = tmp_path / "probe.yaml"
    request.write_text(
        "tier: full\nfinal: true\nfast_fail: false\nlm_eval_tasks: [piqa, winogrande]\nlm_eval_limit: 200\nlab_dir: custom-lab\nresource_envelope: bounds.yaml\n"
    )
    bounds = tmp_path / "bounds.yaml"
    bounds.write_text("resource_envelope_version: 1\nmax_rss_bytes: 1125899906842624\n")
    calls = []

    def score(
        target_spec: str, baseline: str | None, **options: Any
    ) -> tuple[dict, Path]:
        calls.append((target_spec, baseline, options))
        return {"request": options["request"]}, target

    monkeypatch.setattr(probes, "run_probe", score)
    _cli(
        monkeypatch,
        tmp_path,
        "probe",
        str(target),
        "--vs",
        "explicit-base",
        "--backend",
        "cpu",
        "--config",
        str(request),
        "--json",
    )
    payload = json.loads(capsys.readouterr().out)
    spec, baseline, policy = calls[0]
    assert spec == str(target) and baseline == "explicit-base"
    assert policy["tier"] == "full" and policy["final"] is True
    assert policy["fast_fail"] is False
    assert (
        policy["lm_eval_tasks"] == ["piqa", "winogrande"]
        and policy["lm_eval_limit"] == 200
    )
    assert (
        policy["lab_dir"] == lab
        and policy["resource_envelope"].max_rss_bytes == 1 << 50
    )
    assert payload["request"]["resource_envelope"]["max_rss_bytes"] == 1 << 50
    assert payload["request"]["resource_envelope_path"] == str(bounds)
    assert payload["request"]["baseline"] == "explicit-base"
    assert payload["request"]["backend_override"] == "cpu"
    assert not (lab / "probes").exists()


def test_envelope_failure_stops_before_model_or_probe_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import klab.probes.cli as probes

    run = _run(tmp_path / "lab")
    request = tmp_path / "probe.yaml"
    request.write_text("resource_envelope: bounds.yaml\n")
    (tmp_path / "bounds.yaml").write_text(
        "resource_envelope_version: 1\nmax_rss_bytes: 1\n"
    )
    monkeypatch.setattr(
        probes, "run_probe", lambda *_a, **_k: pytest.fail("scoring reached")
    )
    with pytest.raises(SystemExit, match="max_rss_bytes"):
        _cli(monkeypatch, tmp_path, "probe", str(run), "--config", str(request))
    assert not (tmp_path / "lab/probes").exists()


def test_profile_file_uses_native_loader_and_authorization_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    import klab.cli.main as cli
    import klab.probes.cli as probes

    run = _run(tmp_path / "lab", backend="cuda")
    original = (run / "resolved_config.yaml").read_bytes()
    profile = {
        "runtime_profile_version": 1,
        "id": "fixture",
        "python": str(Path(sys.executable).resolve()),
        "engine": "pytorch",
        "backend": "cpu",
        "device_index": 0,
    }
    (tmp_path / "profile.yaml").write_text(yaml.safe_dump(profile))
    request = tmp_path / "probe.yaml"
    request.write_text("tier: standard\nruntime_profile: profile.yaml")
    monkeypatch.setattr(cli, "_ensure_runtime_interpreter", lambda *_: None)
    authorization = object()
    seen = []

    def authorize(selected: Any, config: Any) -> object:
        seen.append((selected, config.runtime.backend))
        return authorization

    def score(_target: str, _baseline: str | None, **options: Any) -> tuple[dict, Path]:
        assert options["authorization"] is authorization
        assert options["request"]["runtime_profile_path"] == str(
            tmp_path / "profile.yaml"
        )
        assert options["request"]["runtime_profile"]["id"] == "fixture"
        assert options["request"]["runtime_profile"]["backend"] == "cpu"
        return {"request": options["request"]}, tmp_path / "unused.json"

    monkeypatch.setattr(cli, "authorize_profile", authorize)
    monkeypatch.setattr(probes, "run_probe", score)
    _cli(
        monkeypatch,
        tmp_path,
        "probe",
        str(run),
        "--backend",
        "cpu",
        "--config",
        str(request),
        "--json",
    )
    assert seen[0][0].id == "fixture" and seen[0][1] == "cpu"
    assert (
        json.loads(capsys.readouterr().out)["request"]["runtime_profile"][
            "device_index"
        ]
        == 0
    )
    assert (run / "resolved_config.yaml").read_bytes() == original


def test_named_preset_precedes_same_named_file_with_explicit_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "standard").write_text("tier: full")
    assert load_probe_config("standard").tier == "standard"
    assert load_probe_config("./standard").tier == "full"
    (tmp_path / "my-policy").write_text("tier: standard")
    assert load_probe_config("my-policy").tier == "standard"


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (
            "finall",
            "unknown probe config 'finall'; choose fast, standard, full, final, final-full",
        ),
        ("./finall", "cannot read probe config"),
        ("finall.yaml", "cannot read probe config"),
        ("finall.json", "cannot read probe config"),
        ("configs/finall", "cannot read probe config"),
    ],
)
def test_missing_config_diagnoses_bare_names_and_explicit_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str, message: str
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError) as failure:
        load_probe_config(selection)
    assert message in str(failure.value)


def test_sealed_request_retains_paths_and_values_when_sources_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    import klab.cli.main as cli
    from klab.lab_records import seal
    from klab.probes import cli as probes

    target = _record(tmp_path / "lab", _run(tmp_path / "lab"))
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "runtime_profile_version": 1,
                "id": "fixture",
                "python": str(Path(sys.executable).resolve()),
                "engine": "pytorch",
                "backend": "cpu",
                "device_index": 0,
            }
        )
    )
    bounds_path = tmp_path / "bounds.yaml"
    bounds_path.write_text(
        "resource_envelope_version: 1\nmax_rss_bytes: 1125899906842624\n"
    )
    request = tmp_path / "probe.yaml"
    request.write_text("runtime_profile: profile.yaml\nresource_envelope: bounds.yaml")
    monkeypatch.setattr(cli, "_ensure_runtime_interpreter", lambda *_: None)
    monkeypatch.setattr(cli, "authorize_profile", lambda *_: object())
    monkeypatch.setattr(
        probes,
        "run_battery",
        lambda *_a, **_k: {
            "format": "klab-probe-v1",
            "stop": {"kind": "resource_exceeded"},
            "probes": [],
            "verdict": {"action": "rerun"},
        },
    )
    _cli(
        monkeypatch, tmp_path, "probe", str(target), "--config", str(request), "--json"
    )
    result = json.loads(capsys.readouterr().out)
    saved = result["request"]
    assert saved["runtime_profile_path"] == str(profile_path)
    assert saved["resource_envelope_path"] == str(bounds_path)
    assert saved["runtime_profile"]["id"] == "fixture"
    assert saved["resource_envelope"]["max_rss_bytes"] == 1 << 50
    profile_path.write_text("changed")
    bounds_path.write_text("changed")
    _cli(monkeypatch, tmp_path, "report", result["probe_id"], "--json")
    reread = json.loads(capsys.readouterr().out)
    assert reread["request"] == saved
    assert reread["record_sha256"] == seal(reread)["record_sha256"]
