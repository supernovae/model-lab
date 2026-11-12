"""Everyday result lookup and dispatch: metadata only, no models or downloads."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from klab.lab_records import KINDS, seal
from klab.lab_results import (
    ResultLookupError,
    resolve_checkpoint,
    resolve_record,
    resolve_result,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", ["try", "probe"])
@pytest.mark.parametrize("missing_subject", [True, False])
def test_compare_missing_lm_eval_hint_pins_exported_record_after_latest_moves(
    tmp_path: Path, kind: str, missing_subject: bool
) -> None:
    from klab.probes.compare import as_json, compare, render

    lab = tmp_path / "lab"
    run = _run(lab)
    path = _record(lab, run, kind)
    record = json.loads(path.read_text())
    if kind == "try":
        record["arms"]["candidate"]["heldout"] = {"loss": 1.0}
    exported = tmp_path / "exported record.json"
    exported.write_text(json.dumps(seal(record)))
    (run / "checkpoints/step-2").mkdir()
    (run / "checkpoints/latest.json").write_text('{"relative_path":"step-2"}')
    specs = [str(exported), "ref:SmolLM2-135M"]
    if not missing_subject:
        specs.reverse()
    report = compare(specs[0], specs[1:], lab_dir=lab)
    row = next(p for p in report["comparisons"][0]["pairs"] if p["metric"] == "lm_eval")
    assert row["status"] == "missing_evidence"
    command = shlex.split(row["note"].split("`")[1])
    assert command == ["klab", "probe", str(exported), "--config", "full"]
    target = resolve_checkpoint(command[2], lab_dir=lab)
    assert target.checkpoint == str(run / "checkpoints/step-1")
    assert target.expected_sha256 == "a" * 64
    assert row["note"] in render(report)
    assert row in as_json(report)["comparisons"][0]["pairs"]
    (run / "checkpoints/step-1").rmdir()
    with pytest.raises(ResultLookupError, match="recorded checkpoint is unavailable"):
        resolve_checkpoint(command[2], lab_dir=lab)


def test_needle_and_reference_curve_hints_use_record_targets(tmp_path: Path) -> None:
    from klab.probes.compare import as_json, compare, render

    lab = tmp_path / "lab"
    run = _run(lab)
    path = _record(lab, run, "probe")
    exported = tmp_path / "exported record.json"
    exported.write_bytes(path.read_bytes())
    (run / "checkpoints/step-2").mkdir()
    (run / "checkpoints/latest.json").write_text('{"relative_path":"step-2"}')
    report = compare(str(exported), [], lab_dir=lab, references=True)
    row = next(p for p in report["comparisons"][0]["pairs"] if p["metric"] == "needle")
    assert row["status"] == "missing_evidence"
    command = shlex.split(row["note"].split("`")[1])
    assert command == ["klab", "probe", str(exported), "--config", "standard"]
    target = resolve_checkpoint(command[2], lab_dir=lab)
    assert target.checkpoint == str(run / "checkpoints/step-1")
    assert target.expected_sha256 == "a" * 64
    assert row["note"] in render(report)
    assert row in as_json(report)["comparisons"][0]["pairs"]
    curve_hint = report["curve"]["note"]
    assert "klab probe RECORD --config full" in curve_hint
    assert "probe RUN" not in render(report)
    assert curve_hint in render(report)
    assert as_json(report)["curve"]["note"] == curve_hint


def _run(lab: Path, name: str = "candidate", backend: str = "cpu") -> Path:
    from klab.config.models import RunConfig

    run = lab / "runs" / name
    run.mkdir(parents=True)
    (run / "manifest.json").write_text("{}")
    raw = yaml.safe_load((ROOT / "configs/smoke_moe_cpu.yaml").read_text())
    raw["runtime"]["backend"] = backend
    (run / "resolved_config.yaml").write_text(
        RunConfig.model_validate(raw).model_dump_json()
    )
    (run / "checkpoints" / "step-1").mkdir(parents=True)
    return run


def _record(lab: Path, run: Path, kind: str = "try", **updates: Any) -> Path:
    fmt, folder, filename = KINDS[kind]
    row = {
        "run_id": run.name,
        "run_dir": str(run),
        "checkpoint": "checkpoints/step-1",
        "checkpoint_sha256": "a" * 64,
    }
    record = {
        "format": fmt,
        "try_id" if kind == "try" else "probe_id": f"{kind}-example",
        "arms" if kind == "try" else "target": {"candidate": row}
        if kind == "try"
        else row,
        **updates,
    }
    path = lab / folder / f"{kind}-example" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seal(record)))
    return path


@pytest.mark.parametrize("kind", ["try", "probe"])
def test_one_record_handle_selects_exact_checkpoint_and_retains_record(
    tmp_path: Path, kind: str
) -> None:
    run = _run(tmp_path)
    path = _record(tmp_path, run, kind)
    # The latest pointer has advanced; lookup must never use it for a record.
    (run / "checkpoints" / "latest.json").write_text('{"relative_path":"step-2"}')
    for spec in (path.parent.name, str(path.parent), str(path)):
        assert resolve_record(spec, tmp_path) == path
        target = resolve_checkpoint(spec, lab_dir=tmp_path)
        assert target.run == run
        assert target.checkpoint == str(run / "checkpoints/step-1")
        assert target.expected_sha256 == "a" * 64


def test_direct_run_checkpoint_and_custom_runs_directory(tmp_path: Path) -> None:
    run = _run(tmp_path / "external")
    for spec in (run.name, str(run)):
        target = resolve_checkpoint(spec, lab_dir=tmp_path, runs_dir=run.parent)
        assert target.run == run and target.checkpoint is None
    checkpoint = run / "checkpoints/step-1"
    assert resolve_checkpoint(str(checkpoint), lab_dir=tmp_path).checkpoint == str(
        checkpoint
    )
    # Runs do not imply a protocol, record, final split or candidate role.
    with pytest.raises(ResultLookupError, match="unique sealed result"):
        resolve_record(str(run), tmp_path)


def test_exported_record_inside_run_is_still_a_record(tmp_path: Path) -> None:
    run = _run(tmp_path)
    source = _record(tmp_path, run)
    exported = run / "saved-result.json"
    exported.write_bytes(source.read_bytes())
    assert resolve_record(str(exported), tmp_path) == exported
    target = resolve_checkpoint(str(exported), lab_dir=tmp_path)
    assert target.checkpoint == str(run / "checkpoints/step-1")
    assert target.expected_sha256 == "a" * 64


def test_missing_recorded_checkpoint_never_falls_back_to_latest(tmp_path: Path) -> None:
    run = _run(tmp_path)
    path = _record(tmp_path, run)
    (run / "checkpoints/step-1").rmdir()
    (run / "checkpoints/step-2").mkdir()
    with pytest.raises(ResultLookupError, match="recorded checkpoint is unavailable"):
        resolve_checkpoint(path.parent.name, lab_dir=tmp_path)
    assert resolve_record(path.parent.name, tmp_path) == path


@pytest.mark.parametrize("checkpoint", ["../other", "checkpoints/latest.json"])
def test_record_cannot_redirect_to_a_pointer_or_another_run(
    tmp_path: Path, checkpoint: str
) -> None:
    run = _run(tmp_path)
    path = _record(tmp_path, run)
    record = json.loads(path.read_text())
    record["arms"]["candidate"]["checkpoint"] = checkpoint
    path.write_text(json.dumps(seal(record)))
    with pytest.raises(
        ResultLookupError, match="does not name a checkpoint generation"
    ):
        resolve_checkpoint(str(path), lab_dir=tmp_path)


def test_record_cannot_follow_checkpoint_symlink_outside_run(tmp_path: Path) -> None:
    run = _run(tmp_path)
    path = _record(tmp_path, run)
    selected = run / "checkpoints/step-1"
    selected.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    selected.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ResultLookupError, match="escapes its run"):
        resolve_checkpoint(str(path), lab_dir=tmp_path)


def test_stopped_record_remains_readable_without_a_scored_candidate(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)
    path = _record(
        tmp_path, run, arms={}, status="resources", stop={"kind": "resources"}
    )
    assert resolve_result(str(path), lab_dir=tmp_path).record["stop"] == {
        "kind": "resources"
    }
    with pytest.raises(ResultLookupError, match="retained stop/missing evidence"):
        resolve_checkpoint(str(path), lab_dir=tmp_path)


def test_colliding_ids_require_an_explicit_path(tmp_path: Path) -> None:
    run = _run(tmp_path, "try-example")
    record = _record(tmp_path, run)
    for resolver in (
        lambda: resolve_record("try-example", tmp_path),
        lambda: resolve_checkpoint("try-example", lab_dir=tmp_path),
    ):
        with pytest.raises(ResultLookupError, match="ambiguous_result"):
            resolver()
    assert resolve_record(str(record), tmp_path) == record
    assert resolve_checkpoint(str(run), lab_dir=tmp_path).run == run


def test_duplicate_run_ids_across_roots_are_not_silently_shadowed(
    tmp_path: Path,
) -> None:
    _run(tmp_path / "lab")
    external = _run(tmp_path / "external")
    with pytest.raises(ResultLookupError, match="ambiguous_result"):
        resolve_checkpoint(
            "candidate", lab_dir=tmp_path / "lab", runs_dir=external.parent
        )


def test_record_digest_is_checked_on_every_surface(tmp_path: Path) -> None:
    run = _run(tmp_path)
    path = _record(tmp_path, run)
    record = json.loads(path.read_text())
    record["arms"]["candidate"]["checkpoint_sha256"] = "b" * 64
    path.write_text(json.dumps(record))
    for resolver in (
        lambda: resolve_record(str(path), tmp_path),
        lambda: resolve_checkpoint(str(path), lab_dir=tmp_path),
    ):
        with pytest.raises(ResultLookupError, match="digest mismatch"):
            resolver()


def test_folder_with_two_records_is_ambiguous(tmp_path: Path) -> None:
    run = _run(tmp_path)
    first = _record(tmp_path, run)
    second = _record(tmp_path, run, "probe")
    (first.parent / "probe.json").write_bytes(second.read_bytes())
    with pytest.raises(ResultLookupError, match="multiple records"):
        resolve_record(str(first.parent), tmp_path)


def _cli(monkeypatch: pytest.MonkeyPatch, work: Path, *args: str) -> None:
    from klab.cli.main import main

    monkeypatch.setenv("KLAB_WORK_DIR", str(work))
    monkeypatch.setattr(sys, "argv", ["klab", "--work-dir", str(work), *args])
    main()


@pytest.mark.parametrize("command", ["report", "compare", "probe", "explore"])
def test_unknown_id_has_the_same_text_and_json_error_on_all_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    with pytest.raises(SystemExit) as text_error:
        _cli(monkeypatch, tmp_path, command, "missing")
    with pytest.raises(SystemExit) as json_error:
        _cli(monkeypatch, tmp_path, command, "missing", "--json")
    assert json_error.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "missing_evidence"
    assert payload["result"] == "missing"
    assert payload["reason"] in str(text_error.value)
    assert payload["hint"] in str(text_error.value)
    if command in {"explore", "probe"}:
        assert "--lab-dir" not in payload["hint"] and "--config" in payload["hint"]


def test_probe_saved_backend_error_is_explicit_in_text_and_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = "cuda"
    run = _run(tmp_path / "lab", backend=backend)
    for flag in ([], ["--json"]):
        with pytest.raises(SystemExit) as error:
            _cli(monkeypatch, tmp_path, "probe", run.name, *flag)
        if flag:
            payload = json.loads(capsys.readouterr().out)
            assert payload["status"] == "runtime_required"
            message = payload["reason"] + payload["hint"]
        else:
            message = str(error.value)
        assert f"saved backend {backend}" in message
        assert "--backend cpu" in message and "no automatic fallback" in message
    assert not (tmp_path / "lab/probes").exists()


def test_explicit_cpu_override_reaches_native_authorization_without_changing_saved_config(
    tmp_path: Path,
) -> None:
    from klab.cli.main import _prepare_runtime_command, build_parser

    run = _run(tmp_path / "lab", backend="cuda")
    original = (run / "resolved_config.yaml").read_bytes()
    for command in ("probe", "explore"):
        args = build_parser(tmp_path).parse_args(
            ["--work-dir", str(tmp_path), command, run.name]
            + (["--backend", "cpu"] if command == "probe" else [])
        )
        _prepare_runtime_command(args)
        assert args.backend == "cpu" and args.runtime_authorization is None
    assert (run / "resolved_config.yaml").read_bytes() == original


def test_record_digest_is_forwarded_to_native_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from klab.evaluation import inference
    from klab.probes.cli import load_target

    run = _run(tmp_path)
    path = _record(tmp_path, run)
    seen = []
    monkeypatch.setattr(inference, "load_run", lambda *a, **kw: seen.append((a, kw)))
    load_target(
        path.parent.name,
        lab_dir=tmp_path,
        runs_dir=None,
        backend="cpu",
        authorization=None,
    )
    assert seen == [
        (
            (run.name, run.parent, str(run / "checkpoints/step-1"), "cpu"),
            {"authorization": None, "expected_checkpoint_sha256": "a" * 64},
        )
    ]


def test_native_loader_rejects_record_digest_mismatch_before_device_or_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from klab.evaluation import inference

    monkeypatch.setattr(
        inference,
        "_verified_checkpoint",
        lambda *a, **kw: (None, None, None, {"sha256": "b" * 64}, None),
    )
    monkeypatch.setattr(
        inference,
        "_ephemeral_config",
        lambda *a: pytest.fail("must reject before runtime selection"),
    )
    with pytest.raises(ValueError, match="digest differs from the sealed lab record"):
        inference.load_run("candidate", tmp_path, expected_checkpoint_sha256="a" * 64)


def test_report_retains_resource_stop_and_compare_explains_missing_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lab = tmp_path / "lab"
    path = _record(
        lab,
        _run(lab),
        arms={},
        status="interrupted",
        interruption={"arm": "baseline", "phase": "training", "reason": "resources"},
    )
    original = path.read_bytes()
    _cli(monkeypatch, tmp_path, "report", path.parent.name)
    text = capsys.readouterr().out
    assert "stopped safely" in text and "resources" in text
    _cli(monkeypatch, tmp_path, "report", path.parent.name, "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload == json.loads(original)
    for command in ("probe", "explore", "compare"):
        with pytest.raises(SystemExit):
            _cli(monkeypatch, tmp_path, command, path.parent.name, "--json")
        error = json.loads(capsys.readouterr().out)
        assert error["status"] == "missing_evidence"
        assert "retained stop/missing evidence" in error["hint"]
    assert path.read_bytes() == original


def test_record_cli_dispatch_preserves_explicit_backend_and_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from klab import explorer

    lab = tmp_path / "lab"
    record = _record(lab, _run(lab, backend="cuda"))
    envelope = tmp_path / "envelope.yaml"
    envelope.write_text(
        "resource_envelope_version: 1\nmax_rss_bytes: 1125899906842624\n"
    )
    seen = []

    def inspect(target: str, **kwargs: Any) -> tuple[dict, None]:
        seen.append((target, kwargs))
        return {"unavailable": "fixture: no model loaded"}, None

    monkeypatch.setattr(explorer, "explore", inspect)
    _cli(
        monkeypatch,
        tmp_path,
        "explore",
        record.parent.name,
        "--resource-envelope",
        str(envelope),
        "--json",
    )
    assert (
        json.loads(capsys.readouterr().out)["unavailable"] == "fixture: no model loaded"
    )
    assert seen[0][0] == record.parent.name
    assert seen[0][1]["backend"] == "cpu"
    assert seen[0][1]["resource_envelope"].max_rss_bytes == 1 << 50


@pytest.mark.parametrize(
    "declaration",
    [
        "[]",
        "text: [",
        "text: '   '",
        "refresh: 'false'",
        "explore_config_version: 2",
        "backend: cuda",
    ],
)
def test_explore_declaration_fails_before_runtime_or_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declaration: str
) -> None:
    from klab import runtime_environments
    from klab.cli.main import _prepare_runtime_command, build_parser

    request = tmp_path / "explore.yaml"
    request.write_text(declaration)
    monkeypatch.setattr(
        runtime_environments,
        "profile_for_id",
        lambda *_: pytest.fail("invalid declaration reached runtime selection"),
    )
    args = build_parser(tmp_path).parse_args(
        ["explore", "missing", "--config", str(request), "--runtime", "accelerator"]
    )
    with pytest.raises(ValueError):
        _prepare_runtime_command(args)


def test_explore_declaration_anchors_roots_and_reaches_native_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from klab import explorer

    declarations = tmp_path / "requests"
    declarations.mkdir()
    lab = tmp_path / "custom-lab"
    run = _run(lab, backend="cuda")
    request = declarations / "explore.yaml"
    request.write_text(
        "text: declared story\nrefresh: true\nlab_dir: ../custom-lab\nruns_dir: ../custom-lab/runs\n"
    )
    original = (run / "resolved_config.yaml").read_bytes()
    seen = []

    def inspect(target: str, **kwargs: Any) -> tuple[dict, None]:
        seen.append(kwargs)
        return {"unavailable": "no model loaded"}, None

    monkeypatch.setattr(explorer, "explore", inspect)
    _cli(monkeypatch, tmp_path, "explore", run.name, "--config", str(request), "--json")
    assert json.loads(capsys.readouterr().out)["unavailable"] == "no model loaded"
    assert seen[0]["text"] == "declared story" and seen[0]["refresh"] is True
    assert seen[0]["lab_dir"] == lab and seen[0]["runs_dir"] == lab / "runs"
    assert seen[0]["backend"] == "cpu"
    assert (run / "resolved_config.yaml").read_bytes() == original


@pytest.mark.parametrize("selector", ["--runtime", "--runtime-profile"])
def test_explore_explicit_profile_selects_backend_before_native_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selector: str
) -> None:
    import klab.cli.main as cli
    from klab import runtime_environments
    from klab.runtime_profile import RuntimeProfile

    run = _run(tmp_path / "lab")
    original = (run / "resolved_config.yaml").read_bytes()
    profile = RuntimeProfile(
        runtime_profile_version=1,
        id="fixture",
        python=Path(sys.executable).resolve(),
        engine="pytorch",
        backend="cuda",
        device_index=0,
    )
    monkeypatch.setattr(runtime_environments, "profile_for_id", lambda _: profile)
    monkeypatch.setattr(cli, "load_runtime_profile", lambda _: profile)
    monkeypatch.setattr(cli, "_ensure_runtime_interpreter", lambda *_: None)
    seen = []
    authorization = object()

    def authorize(selected: Any, config: Any) -> object:
        assert selected is profile
        seen.append(config.runtime.backend)
        return authorization

    monkeypatch.setattr(cli, "authorize_profile", authorize)
    args = cli.build_parser(tmp_path).parse_args(
        ["explore", str(run), selector, "fixture"]
    )
    cli._prepare_runtime_command(args)
    assert seen == ["cuda"] and args.backend == "cuda"
    assert args.runtime_authorization is authorization
    assert (run / "resolved_config.yaml").read_bytes() == original

    def unavailable(*_: Any) -> None:
        raise ValueError("device unavailable")

    monkeypatch.setattr(cli, "authorize_profile", unavailable)
    with pytest.raises(ValueError, match="device unavailable"):
        cli._prepare_runtime_command(args)


@pytest.mark.parametrize("document", ["", "{}\n"])
def test_explore_empty_declaration_uses_defaults(tmp_path: Path, document: str) -> None:
    from klab.explore_config import ExploreConfig, load_explore_config

    request = tmp_path / "explore.yaml"
    request.write_text(document)
    assert load_explore_config(request) == ExploreConfig()


def test_explore_declaration_expands_home_and_preserves_exact_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.explore_config import load_explore_config

    task_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(task_home))
    request = tmp_path / "explore.yaml"
    prompt = " \tOnce upon a time\n "
    request.write_text(
        yaml.safe_dump({"text": prompt, "lab_dir": "~/lab", "runs_dir": "~/runs"})
    )
    config = load_explore_config(request)
    assert config.text == prompt
    assert config.lab_dir == task_home / "lab"
    assert config.runs_dir == task_home / "runs"
