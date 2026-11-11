"""Try declaration and operational-envelope contracts without model work."""

import argparse
import json
import sys
from pathlib import Path

import pytest
import yaml
from test_lab_results import _cli

from sparselab.resource_envelope import ResourceEnvelope, resolve_lab_try_limits
from sparselab.try_config import TryConfig, load_try_config, prepare_try_config


@pytest.mark.parametrize(
    "name,tier",
    [
        ("fast", "fast"),
        ("standard", "standard"),
        ("full", "full"),
        ("no-probes", "none"),
    ],
)
def test_presets_match_files(tmp_path, name, tier):
    path = tmp_path / "try.yaml"
    path.write_text(yaml.safe_dump({"probe_tier": tier}))
    assert load_try_config(name) == load_try_config(str(path))


@pytest.mark.parametrize("document", ["", "# defaults", "null", "{}"])
def test_empty_files_default(tmp_path, document):
    path = tmp_path / "try.yaml"
    path.write_text(document)
    assert load_try_config(str(path)) == TryConfig()


@pytest.mark.parametrize(
    "document",
    [
        "[]",
        "false",
        "5",
        "seed: [",
        "unknown: true",
        "probe_tier: final",
        "try_config_version: true",
        "try_config_version: 1.0",
        "seed: '7'",
        "seed: true",
        "seed: 7.0",
        "seed: -1",
        "seed: 4294967296",
        "fresh_baseline: 'true'",
        "repeat_baseline: 'true'",
        "continuation: {checkpoint: cp, append_data: 'true'}",
        "lab_dir: 5",
    ],
)
def test_invalid_policy_precedes_runtime(tmp_path, monkeypatch, document):
    from sparselab import runtime_environments
    from sparselab.cli.main import _prepare_runtime_command, build_parser

    path = tmp_path / "try.yaml"
    path.write_text(document)
    monkeypatch.setattr(
        runtime_environments,
        "profile_for_id",
        lambda *_: pytest.fail("runtime reached"),
    )
    args = build_parser(tmp_path).parse_args(
        [
            "try",
            "missing",
            "--vs",
            "missing",
            "--config",
            str(path),
            "--runtime",
            "fixture",
        ]
    )
    with pytest.raises(ValueError):
        _prepare_runtime_command(args)
    assert not (tmp_path / "lab").exists()


def test_paths_snapshot_and_runtime_conflict(tmp_path, monkeypatch):
    from sparselab.cli.main import build_parser

    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "requests/try.yaml"
    path.parent.mkdir()
    path.write_text(
        "seed: 7\nlab_dir: ~/lab\nruntime_profile: ../profile.yaml\nresource_envelope: bounds.yaml"
    )
    config = load_try_config(str(path))
    assert config.lab_dir == tmp_path / "lab"
    assert config.runtime_profile == tmp_path / "profile.yaml"
    assert config.resource_envelope == path.parent / "bounds.yaml"
    parser = build_parser(tmp_path)
    args = parser.parse_args(
        ["try", "missing", "--vs", "missing", "--config", str(path)]
    )
    prepare_try_config(args)
    path.write_text("seed: 9")
    prepare_try_config(args)
    assert args.seed == 7 and args.runtime_profile == config.runtime_profile
    path.write_text("runtime_profile: missing.yaml")
    args = parser.parse_args(
        [
            "try",
            "missing",
            "--vs",
            "missing",
            "--config",
            str(path),
            "--runtime",
            "fixture",
        ]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_try_config(args)


def test_five_options_and_removed_flags(tmp_path):
    from sparselab.cli.main import build_parser

    parser = build_parser(tmp_path)
    commands = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    options = {
        flag
        for a in commands.choices["try"]._actions
        for flag in a.option_strings
        if a.dest != "help"
    }
    assert options == {"--vs", "--config", "--backend", "--runtime", "--json"}
    for flag, values in {
        "--seed": ["7"],
        "--probe-tier": ["none"],
        "--fresh-baseline": [],
        "--lab-dir": ["lab"],
        "--runtime-profile": ["profile.yaml"],
        "--resource-envelope": ["bounds.yaml"],
        "--max-wall-seconds": ["1"],
        "--tokenizer-batch-documents": ["1"],
        "--tokenizer-batch-source-bytes": ["1"],
    }.items():
        with pytest.raises(SystemExit):
            parser.parse_args(["try", "candidate", "--vs", "baseline", flag, *values])


@pytest.mark.parametrize(
    "limits",
    [
        {"max_wall_seconds_per_arm": 0},
        {"max_wall_seconds_per_arm": float("inf")},
        {"max_wall_seconds_per_arm": True},
        {"max_wall_seconds_per_arm": "1"},
        {"tokenizer_batch_documents": 257},
        {"tokenizer_batch_documents": True},
        {"tokenizer_batch_documents": 1.0},
        {"tokenizer_batch_source_bytes": 0},
        {"tokenizer_batch_source_bytes": 8388609},
        {"typo": 1},
    ],
)
def test_envelope_limits_strict(limits):
    with pytest.raises(ValueError):
        ResourceEnvelope(resource_envelope_version=1, lab_try=limits)


def test_native_defaults_legacy_serialization_and_tighter_limits():
    legacy = ResourceEnvelope(resource_envelope_version=1)
    assert "lab_try" not in legacy.model_dump(mode="json")
    assert resolve_lab_try_limits(
        legacy,
        max_wall_seconds=None,
        tokenizer_batch_documents=None,
        tokenizer_batch_source_bytes=None,
    ) == (None, 256, 1048576)
    envelope = ResourceEnvelope(
        resource_envelope_version=1,
        lab_try={
            "max_wall_seconds_per_arm": 60,
            "tokenizer_batch_documents": 32,
            "tokenizer_batch_source_bytes": 2097152,
        },
    )
    assert resolve_lab_try_limits(
        envelope,
        max_wall_seconds=None,
        tokenizer_batch_documents=None,
        tokenizer_batch_source_bytes=None,
    ) == (60, 32, 2097152)
    assert resolve_lab_try_limits(
        envelope,
        max_wall_seconds=10,
        tokenizer_batch_documents=8,
        tokenizer_batch_source_bytes=4194304,
    ) == (10, 8, 2097152)
    for wall, docs, size in [
        (True, None, None),
        (float("nan"), None, None),
        (None, 0, None),
        (None, None, True),
    ]:
        with pytest.raises(ValueError):
            resolve_lab_try_limits(
                envelope,
                max_wall_seconds=wall,
                tokenizer_batch_documents=docs,
                tokenizer_batch_source_bytes=size,
            )


def test_public_try_uses_native_profile_and_seals_loaded_request(
    tmp_path, monkeypatch, capsys
):
    import sparselab.cli.main as cli
    import sparselab.lab_mode as lab

    baseline = Path(__file__).resolve().parents[1] / "configs/smoke_cpu.yaml"
    profile = {
        "runtime_profile_version": 1,
        "id": "fixture",
        "python": str(Path(sys.executable).resolve()),
        "engine": "pytorch",
        "backend": "cpu",
        "device_index": 0,
    }
    (tmp_path / "profile.yaml").write_text(yaml.safe_dump(profile))
    (tmp_path / "bounds.yaml").write_text(
        "resource_envelope_version: 1\nlab_try:\n  max_wall_seconds_per_arm: 60\n  tokenizer_batch_documents: 8\n"
    )
    path = tmp_path / "try.yaml"
    path.write_text(
        "seed: 7\nfresh_baseline: true\nprobe_tier: none\nlab_dir: custom-lab\nruntime_profile: profile.yaml\nresource_envelope: bounds.yaml"
    )
    authorization = object()
    monkeypatch.setattr(cli, "_ensure_runtime_interpreter", lambda *_: None)
    monkeypatch.setattr(cli, "authorize_profile", lambda profile, config: authorization)
    seen = []

    def run(candidate, base, **options):
        seen.append(options)
        assert options["authorization"] is authorization
        return {
            "exit_status": 0,
            "request": options["request"],
        }, tmp_path / "unused.json"

    monkeypatch.setattr(lab, "run_try", run)
    _cli(
        monkeypatch,
        tmp_path,
        "try",
        "candidate.yaml",
        "--vs",
        str(baseline),
        "--config",
        str(path),
        "--backend",
        "cpu",
        "--json",
    )
    payload = json.loads(capsys.readouterr().out)["request"]
    assert seen[0]["seed"] == 7 and seen[0]["fresh_baseline"] is True
    assert seen[0]["probe_tier"] is None
    assert seen[0]["lab_dir"] == tmp_path / "custom-lab"
    from sparselab.runtime_profile import RuntimeProfile

    assert payload["runtime_profile"] == RuntimeProfile.model_validate(
        profile
    ).model_dump(mode="json")
    assert payload["runtime_profile_path"] == str(tmp_path / "profile.yaml")
    assert payload["resource_envelope_path"] == str(tmp_path / "bounds.yaml")
    assert payload["resource_envelope"]["lab_try"]["max_wall_seconds_per_arm"] == 60
    path.write_text("seed: 9")
    assert payload["seed"] == 7
    assert not (tmp_path / "custom-lab/tries").exists()


def test_failed_setup_retains_request_and_report_roundtrip(
    tmp_path, monkeypatch, capsys
):
    from sparselab.lab_mode import read_record, run_try

    request = {"seed": 7, "probe_tier": "none", "resource_envelope": None}
    with pytest.raises((ValueError, OSError)):
        run_try(
            tmp_path / "missing",
            tmp_path / "baseline",
            work_dir=tmp_path,
            train_fn=lambda *_a, **_k: pytest.fail("training reached"),
            request=request,
        )
    (path,) = (tmp_path / "lab/tries").glob("*/try.json")
    record = read_record(path)
    assert record["status"] == "failed" and record["request"] == request
    _cli(monkeypatch, tmp_path, "report", str(path), "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload["request"] == request and payload["status"] == "failed"


def test_named_file_escape_and_typo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "standard").write_text("probe_tier: none")
    assert load_try_config("standard").probe_tier == "standard"
    assert load_try_config("./standard").probe_tier == "none"
    with pytest.raises(ValueError, match="unknown try config"):
        load_try_config("standrad")


def test_interrupted_setup_retains_request(tmp_path, monkeypatch):
    import sparselab.lab_mode as lab

    def stop(_path):
        raise KeyboardInterrupt

    monkeypatch.setattr(lab, "read_candidate", stop)
    request = {"seed": 7, "probe_tier": "none"}
    record, path = lab.run_try(
        tmp_path / "candidate",
        tmp_path / "base",
        work_dir=tmp_path,
        train_fn=lambda *_a, **_k: pytest.fail("training reached"),
        request=request,
    )
    assert record["status"] == "interrupted" and record["request"] == request
    assert lab.read_record(path) == record


@pytest.mark.parametrize(
    "command,count",
    [("try", 5), ("probe", 5), ("explore", 5), ("compare", 5), ("report", 5)],
)
def test_everyday_executable_option_budget(tmp_path, command, count):
    from sparselab.cli.main import build_parser

    parser = build_parser(tmp_path)
    commands = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    options = [
        a
        for a in commands.choices[command]._actions
        if a.option_strings and a.dest != "help"
    ]
    assert len(options) == count


def test_invalid_envelope_precedes_native_runtime(tmp_path, monkeypatch):
    import sparselab.cli.main as cli

    (tmp_path / "try.yaml").write_text(
        "runtime_profile: missing.yaml\nresource_envelope: bounds.yaml"
    )
    (tmp_path / "bounds.yaml").write_text(
        "resource_envelope_version: 1\nlab_try:\n  tokenizer_batch_documents: 257"
    )
    monkeypatch.setattr(
        cli, "load_runtime_profile", lambda *_: pytest.fail("runtime reached")
    )
    with pytest.raises(SystemExit, match="tokenizer_batch_documents"):
        _cli(
            monkeypatch,
            tmp_path,
            "try",
            "missing",
            "--vs",
            "missing",
            "--config",
            str(tmp_path / "try.yaml"),
        )
    assert not (tmp_path / "lab").exists()


@pytest.mark.parametrize("command", ["probe", "explore", "train", "stage"])
@pytest.mark.parametrize("limits", [{}, {"max_wall_seconds_per_arm": 60}])
def test_cli_rejects_try_limits_before_runtime_or_outputs(
    tmp_path, monkeypatch, command, limits
):
    import sparselab.cli.main as cli

    bounds = tmp_path / "bounds.yaml"
    bounds.write_text(
        yaml.safe_dump({"resource_envelope_version": 1, "lab_try": limits})
    )
    if command == "probe":
        policy = tmp_path / "probe.yaml"
        policy.write_text(
            "resource_envelope: bounds.yaml\nruntime_profile: missing.yaml"
        )
        args = ["probe", "missing", "--config", str(policy)]
    else:
        args = [
            command,
            "missing",
            "--resource-envelope",
            str(bounds),
            "--runtime-profile",
            "missing.yaml",
        ]
        if command == "stage":
            args.extend(["--output", str(tmp_path / "stage")])
    monkeypatch.setattr(
        cli, "load_runtime_profile", lambda *_: pytest.fail("runtime reached")
    )
    with pytest.raises(
        SystemExit, match=f"lab_try limits apply only to try, not {command}"
    ):
        _cli(monkeypatch, tmp_path, *args)
    assert not (tmp_path / "lab").exists()
    assert not (tmp_path / "stage").exists()


@pytest.mark.parametrize("command", ["probe", "explore", "train", "stage"])
def test_direct_native_consumers_reject_try_limits(tmp_path, monkeypatch, command):
    from sparselab import explorer, staging
    from sparselab.config.loading import load_config
    from sparselab.probes.cli import run_probe
    from sparselab.training import trainer

    envelope = ResourceEnvelope(
        resource_envelope_version=1, lab_try={"max_wall_seconds_per_arm": 60}
    )
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke_cpu.yaml")
    monkeypatch.setattr(trainer, "require_model_runtime_allowed", lambda: None)
    monkeypatch.setattr(staging, "require_model_runtime_allowed", lambda: None)
    monkeypatch.setattr(trainer, "require_authorization", lambda *_: None)
    monkeypatch.setattr(
        trainer, "_train_impl", lambda *_a, **_k: pytest.fail("training reached")
    )
    with pytest.raises(ValueError, match="lab_try limits apply only to try"):
        if command == "probe":
            run_probe(
                "missing", None, lab_dir=tmp_path / "lab", resource_envelope=envelope
            )
        elif command == "explore":
            explorer.explore(
                "missing", lab_dir=tmp_path / "lab", resource_envelope=envelope
            )
        elif command == "train":
            trainer.train(config, resource_envelope=envelope)
        else:
            staging.stage(
                config,
                tmp_path / "stage",
                through="inspect",
                resource_envelope=envelope,
            )
    assert not (tmp_path / "lab").exists()
    assert not (tmp_path / "stage").exists()


def test_shared_resource_boundary_rejects_unconsumed_try_limits(tmp_path, monkeypatch):
    from sparselab import resource_envelope as resource

    envelope = ResourceEnvelope(resource_envelope_version=1, lab_try={})
    monkeypatch.setattr(
        resource.host_capacity,
        "measure_memory",
        lambda: pytest.fail("measurement reached"),
    )
    with pytest.raises(ValueError, match="lab_try limits apply only to try"):
        resource.check_envelope(envelope, workspace=tmp_path, rss_bytes=0)


@pytest.mark.parametrize("command", ["try", "probe", "train"])
def test_accelerator_error_names_valid_runtime_selectors(
    tmp_path, monkeypatch, command
):
    from test_lab_results import _run

    baseline = Path(__file__).resolve().parents[1] / "configs/smoke_cpu.yaml"
    if command == "try":
        args = ["try", "missing", "--vs", str(baseline)]
    elif command == "probe":
        args = ["probe", str(_run(tmp_path / "lab", backend="cuda"))]
    else:
        args = ["train", str(baseline)]
    with pytest.raises(SystemExit) as error:
        _cli(monkeypatch, tmp_path, *args, "--backend", "cuda")
    message = str(error.value)
    assert "cuda requires --runtime ID" in message
    assert "try/probe: declaration runtime_profile" in message
    assert "other commands: --runtime-profile PATH" in message
    assert "registered compatible worker" in message
    assert not (tmp_path / "lab/tries").exists()
    assert not (tmp_path / "lab/probes").exists()


def test_continuation_path_and_flags_reach_cli(tmp_path):
    from sparselab.cli.main import build_parser

    path = tmp_path / "request.yaml"
    path.write_text(
        "repeat_baseline: true\ncontinuation:\n  checkpoint: parent/checkpoints/generation\n  append_data: true\n  allow_runtime_optimization: true\n"
    )
    args = build_parser(tmp_path).parse_args(
        ["try", "delta", "--vs", "base", "--config", str(path)]
    )
    prepare_try_config(args)
    assert args.repeat_baseline is True
    assert args.continuation.checkpoint == tmp_path / "parent/checkpoints/generation"
    assert args.continuation.append_data is True
    assert args.continuation.allow_runtime_optimization is True
    assert args.continuation.allow_runtime_drift is False
    assert args.continuation.append_from_checkpoint is False
