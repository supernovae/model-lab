"""Cross-contract checks for the retained location/runtime selectors."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from klab.explore_config import load_explore_config
from klab.probes.config import load_probe_config
from klab.try_config import load_try_config


@pytest.mark.parametrize(
    ("loader", "fields"),
    [
        (load_try_config, ("lab_dir", "runtime_profile", "resource_envelope")),
        (
            load_probe_config,
            ("lab_dir", "runs_dir", "runtime_profile", "resource_envelope"),
        ),
        (load_explore_config, ("lab_dir", "runs_dir")),
    ],
)
@pytest.mark.parametrize(
    ("invoking_dir", "relative_source"),
    [("first/nested", False), ("second/nested/deeper", True)],
)
def test_declaration_locations_do_not_follow_cwd_or_work_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader,
    fields: tuple[str, ...],
    invoking_dir: str,
    relative_source: bool,
) -> None:
    source = tmp_path / "declarations" / "request.yaml"
    source.parent.mkdir()
    source.write_text("\n".join(f"{field}: ../retained/{field}" for field in fields))
    expected = {field: tmp_path / "retained" / field for field in fields}
    before = source.read_bytes()
    cwd = tmp_path / invoking_dir
    cwd.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("KLAB_WORK_DIR", str(cwd / "work"))
    selected = Path(os.path.relpath(source, cwd)) if relative_source else source
    # Explore takes a Path; try/probe take a selection string (preset or path).
    config = loader(str(selected) if loader != load_explore_config else selected)
    assert {field: getattr(config, field) for field in fields} == expected
    assert not (cwd / "work").exists()
    assert source.read_bytes() == before
    assert not (tmp_path / "retained").exists()


def test_runtime_registry_and_install_root_do_not_follow_scientific_work_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.runtime_environments import (
        RuntimeEntry,
        profile_for_id,
        register_runtime,
        resolve_registry_path,
        resolve_runtime_dir,
    )

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("KLAB_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "first-work"))
    register_runtime(
        "retained-cpu",
        RuntimeEntry(
            python=Path(sys.executable).absolute(),
            engine="pytorch",
            backend="cpu",
            device_index=0,
        ),
    )
    registry = resolve_registry_path()
    before = registry.read_bytes()
    profile = profile_for_id("retained-cpu")
    for name in ("first-work", "second-work"):
        monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / name))
        assert resolve_registry_path() == registry
        assert resolve_runtime_dir() == tmp_path / "data/klab/runtimes"
        assert profile_for_id("retained-cpu") == profile
        assert not (tmp_path / name).exists()
    assert registry.read_bytes() == before
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    "command",
    [
        "list",
        "discover",
        "show",
        "register",
        "unregister",
        "doctor",
        "profile",
        "provision",
    ],
)
@pytest.mark.parametrize("work_prefix", ["none", "split", "equals"])
def test_all_environment_help_routes_avoid_framework_and_execution(
    tmp_path: Path, command: str, work_prefix: str
) -> None:
    # A fresh process catches eager imports hidden by pytest's application imports.
    script = """
import importlib.abc
import sys

class RefuseExecutionImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname == 'klab.cli.main':
            raise AssertionError('environment help imported execution: ' + fullname)

sys.meta_path.insert(0, RefuseExecutionImports())
def refuse_children(event, args):
    if event in {
        'subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec',
        'os.fork', 'os.forkpty', 'os.spawn',
    }:
        raise AssertionError('environment help started execution: ' + event)
sys.addaudithook(refuse_children)
from klab.cli.entry import main
command, work_prefix, work_dir = sys.argv[1:]
prefix = []
if work_prefix == 'split':
    prefix = ['--work-dir', work_dir]
elif work_prefix == 'equals':
    prefix = ['--work-dir=' + work_dir]
sys.argv = ['klab', *prefix, 'runtime', 'env', command, '--help']
main()
"""
    settings = {
        "KLAB_WORK_DIR": str(tmp_path / "work"),
        "KLAB_RUNTIME_DIR": str(tmp_path / "runtimes"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            command,
            work_prefix,
            str(tmp_path / "selected-work"),
        ],
        env={**os.environ, **settings, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"klab runtime env {command}" in result.stdout
    assert all(not Path(path).exists() for path in settings.values())
    assert not (tmp_path / "selected-work").exists()


def test_environment_work_prefix_preserves_separate_runtime_locations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab import runtime_env_cli
    from klab.cli import entry
    from klab.runtime_environments import (
        resolve_registry_path,
        resolve_runtime_dir,
    )
    from klab.workdir import resolve_work_dir

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "original-work"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("KLAB_RUNTIME_DIR", str(tmp_path / "runtimes"))
    seen = []

    def observe(argv):
        seen.append(argv)
        assert resolve_work_dir() == tmp_path / "selected-work"
        assert resolve_registry_path() == tmp_path / "config/klab/runtimes.yaml"
        assert resolve_runtime_dir() == tmp_path / "runtimes"

    monkeypatch.setattr(runtime_env_cli, "main", observe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "klab",
            "--work-dir",
            "selected-work",
            "runtime",
            "env",
            "list",
            "--json",
        ],
    )
    entry.main()
    assert seen == [["list", "--json"]]
    assert not (tmp_path / "selected-work").exists()
    assert not (tmp_path / "runtimes").exists()
    assert not (tmp_path / "config").exists()


def test_runtime_help_exposes_environment_management(tmp_path: Path) -> None:
    from klab.cli.main import build_parser

    commands = next(
        action.choices
        for action in build_parser(tmp_path / "work")._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert "klab runtime env --help" in commands["runtime"].format_help()
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("work_prefix", ["split", "equals"])
def test_environment_work_prefix_drives_native_runtime_root_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, work_prefix: str
) -> None:
    from klab.cli import entry

    monkeypatch.chdir(tmp_path)
    locations = {
        "KLAB_WORK_DIR": tmp_path / "original-work",
        "KLAB_RUNTIME_DIR": tmp_path / "selected-work",
        "XDG_CONFIG_HOME": tmp_path / "config",
        "XDG_DATA_HOME": tmp_path / "data",
    }
    for name, path in locations.items():
        monkeypatch.setenv(name, str(path))
    prefix = (
        ["--work-dir", "selected-work"]
        if work_prefix == "split"
        else ["--work-dir=selected-work"]
    )
    monkeypatch.setattr(
        sys, "argv", ["klab", *prefix, "runtime", "env", "list", "--json"]
    )
    # Exercise the real manager and native root check, not a mocked handoff.
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ERROR"
    assert payload["reason"] == "runtime root must differ from scientific work root"
    assert all(not path.exists() for path in locations.values())
