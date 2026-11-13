"""Global-prefix dispatch regressions (LAB69); no real credentials or models."""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import pytest

from klab.cli import entry

TOKEN = "/nonexistent-lab69/token"
# One case per parsing path: no prefix, separate values, and joined values
# after another prefix option.
PREFIXES = [
    [],
    ["--hf-token-file", TOKEN, "--work-dir", "selected"],
    ["--work-dir=selected", f"--hf-token-file={TOKEN}"],
]


@pytest.mark.parametrize("prefix", PREFIXES)
def test_installed_environment_help_is_framework_and_credential_free(tmp_path, prefix):
    # Run the actual installed console script in a fresh guarded interpreter.
    script = """
import importlib.abc
import runpy
import sys

class RefuseExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mlx'} or fullname in {
            'klab.cli.main', 'klab.hf_auth',
        }:
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, RefuseExecution())
def guard(event, args):
    if event == 'open' and str(args[0]) == '/nonexistent-lab69/token':
        raise AssertionError('token file opened')
    if event in {'subprocess.Popen', 'os.system', 'socket.connect'}:
        raise AssertionError('unexpected execution: ' + event)
sys.addaudithook(guard)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    console = Path(sysconfig.get_path("scripts")) / "klab"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(console),
            *prefix,
            "runtime",
            "env",
            "--help",
        ],
        cwd=tmp_path,
        env={**os.environ, "KLAB_WORK_DIR": str(tmp_path / "work")},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: klab runtime env" in result.stdout
    assert not (tmp_path / "selected").exists()
    assert not (tmp_path / "work").exists()


def test_parser_construction_leaves_model_frameworks_unloaded():
    # Every command pays this import; frameworks load in the handlers that use them.
    script = (
        "import sys, klab.cli.main as m; m.build_parser(); "
        "print(sorted(name for name in ('torch', 'datasets', 'pandas', 'pyarrow') "
        "if name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize("prefix", PREFIXES)
def test_environment_handoff_preserves_remainder_and_work_root(
    monkeypatch, tmp_path, prefix
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "original"))
    child = ["list", "--", "child", "--hf-token-file", "--work-dir"]
    seen = []
    monkeypatch.setitem(
        sys.modules, "klab.runtime_env_cli", SimpleNamespace(main=seen.append)
    )
    monkeypatch.setattr(sys, "argv", ["klab", *prefix, "runtime", "env", *child])
    entry.main()
    assert seen == [child]
    selected = (
        "selected"
        if any(arg.startswith("--work-dir") for arg in prefix)
        else "original"
    )
    assert os.environ["KLAB_WORK_DIR"] == str(tmp_path / selected)


@pytest.mark.parametrize(
    "argv",
    [
        ["--hf-token-file", TOKEN, "--help"],
        ["--work-dir", "root", "--hf-token-file", TOKEN, "report", "example"],
        ["monitor", "--", "child", "--hf-token-file", "--work-dir"],
        ["runtime", "status", "--hf-token-file", "--work-dir"],
        ["--unknown", "runtime", "env"],
    ],
)
def test_ordinary_dispatch_receives_unchanged_argv(monkeypatch, argv):
    seen = []
    monkeypatch.setitem(
        sys.modules,
        "klab.cli.main",
        SimpleNamespace(main=lambda: seen.append(sys.argv[1:])),
    )
    monkeypatch.setattr(sys, "argv", ["klab", *argv])
    entry.main()
    assert seen == [argv]


@pytest.mark.parametrize(
    "argv",
    [["--hf-token-file"], ["--hf-token-file", "--work-dir=root"]],
)
def test_missing_prefix_values_fail_cleanly(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", ["klab", *argv])
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
    assert "expected one argument" in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--unknown", "--work-dir"])
def test_runtime_local_options_are_not_consumed(monkeypatch, capsys, option):
    monkeypatch.setattr(sys, "argv", ["klab", "runtime", "env", "list", option])
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
