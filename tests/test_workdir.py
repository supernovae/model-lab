from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from klab.workdir import (
    CHECKOUT_STORAGE_REASON,
    WORK_DIR_ENV,
    ensure_scratch_dir,
    ensure_work_dir,
    record_storage_observation,
    resolve_work_dir,
    storage_checks,
    warn_storage_checks,
)
from klab.workers import transport
from klab.workers.models import WorkerDefinition

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("caller", ["scratch", "work", "worker"])
@pytest.mark.parametrize("kind", ["outside", "checkout", "dangling", "inside", "file"])
@pytest.mark.parametrize("configured", [False, True])
def test_unsafe_scratch_refuses_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caller: str,
    kind: str,
    configured: bool,
) -> None:
    root = tmp_path / "work"
    root.mkdir()
    scratch = root / "scratch"
    target = root / "target" if kind == "inside" else tmp_path / kind
    if kind != "dangling":
        target.mkdir()
        (target / "sentinel").write_bytes(b"destination unchanged\n")
    if kind == "checkout":
        (target / ".git").mkdir()
    if kind == "file":
        scratch.write_bytes(b"scratch file unchanged\n")
    else:
        scratch.symlink_to(target, target_is_directory=True)
    before_stat = scratch.lstat()
    for name in (WORK_DIR_ENV, "TMPDIR", "TEMP", "TMP"):
        if configured:
            monkeypatch.setenv(name, str(tmp_path / f"previous-{name}"))
        else:
            monkeypatch.delenv(name, raising=False)
    if caller == "worker":
        monkeypatch.setenv(WORK_DIR_ENV, str(root))
    monkeypatch.setattr(tempfile, "tempdir", "previous-cache" if configured else None)
    before_env = {
        name: os.environ.get(name) for name in (WORK_DIR_ENV, "TMPDIR", "TEMP", "TMP")
    }
    before_tempdir = tempfile.tempdir
    mkdtemp = Mock(side_effect=AssertionError("must refuse before allocation"))
    popen = Mock(side_effect=AssertionError("must refuse before worker launch"))
    monkeypatch.setattr(transport.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(transport.subprocess, "Popen", popen)

    error = FileExistsError if kind == "file" else ValueError
    match = None if kind == "file" else "scratch directory must not be a symlink"
    with pytest.raises(error, match=match):
        if caller == "scratch":
            ensure_scratch_dir(root)
        elif caller == "work":
            ensure_work_dir(root)
        else:
            worker = WorkerDefinition(
                worker_id="test-worker",
                name="test-worker",
                transport="local",
                python=Path(sys.executable),
                root=tmp_path,
                engine="pytorch",
                backend="cpu",
                device_index=0,
            )
            transport.call_worker(worker, "status", {"attempt_id": None})

    mkdtemp.assert_not_called()
    popen.assert_not_called()
    assert {name: os.environ.get(name) for name in before_env} == before_env
    assert tempfile.tempdir == before_tempdir
    assert scratch.lstat() == before_stat
    if kind == "file":
        assert scratch.read_bytes() == b"scratch file unchanged\n"
    else:
        assert scratch.is_symlink()
        assert scratch.readlink() == target
    if kind == "dangling":
        assert not target.exists()
    else:
        assert (target / "sentinel").read_bytes() == b"destination unchanged\n"
        assert {p.name for p in target.iterdir()} == (
            {"sentinel", ".git"} if kind == "checkout" else {"sentinel"}
        )
    assert {p.name for p in root.iterdir()} == (
        {"scratch", "target"} if kind == "inside" else {"scratch"}
    )


@pytest.mark.parametrize("caller", ["work", "worker"])
@pytest.mark.parametrize(
    "redirect", ["symlink-at-mkdir", "resolve-before-mkdir", "resolve-after-mkdir"]
)
def test_scratch_location_checks_refuse_before_environment_or_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caller: str, redirect: str
) -> None:
    root = tmp_path / "work"
    root.mkdir()
    scratch = root / "scratch"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_bytes(b"keep\n")
    missing = outside / "missing"
    original_mkdir = Path.mkdir
    original_resolve = Path.resolve
    mkdir_calls = []

    def controlled_mkdir(path, *args, **kwargs):
        if path == scratch:
            mkdir_calls.append(path)
            if redirect == "symlink-at-mkdir":
                path.symlink_to(outside, target_is_directory=True)
        return original_mkdir(path, *args, **kwargs)

    def controlled_resolve(path, *args, **kwargs):
        if path == scratch and (
            redirect == "resolve-before-mkdir"
            or (redirect == "resolve-after-mkdir" and mkdir_calls)
        ):
            return missing
        return original_resolve(path, *args, **kwargs)

    # Deterministic branch coverage, not an adversarial concurrency guarantee.
    monkeypatch.setattr(Path, "mkdir", controlled_mkdir)
    monkeypatch.setattr(Path, "resolve", controlled_resolve)
    for name in (WORK_DIR_ENV, "TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(root) if name == WORK_DIR_ENV else "previous")
    monkeypatch.setattr(tempfile, "tempdir", "previous-cache")
    before_env = {
        name: os.environ[name] for name in (WORK_DIR_ENV, "TMPDIR", "TEMP", "TMP")
    }
    mkdtemp = Mock(side_effect=AssertionError("must refuse before allocation"))
    popen = Mock(side_effect=AssertionError("must refuse before worker launch"))
    monkeypatch.setattr(transport.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(transport.subprocess, "Popen", popen)
    message = (
        "scratch directory must not be a symlink"
        if redirect == "symlink-at-mkdir"
        else "scratch directory must remain inside work root"
    )
    with pytest.raises(ValueError, match=message):
        if caller == "work":
            ensure_work_dir(root)
        else:
            worker = WorkerDefinition(
                worker_id="test-worker",
                name="test-worker",
                transport="local",
                python=Path(sys.executable),
                root=tmp_path,
                engine="pytorch",
                backend="cpu",
                device_index=0,
            )
            transport.call_worker(worker, "status", {"attempt_id": None})

    assert mkdir_calls == ([] if redirect == "resolve-before-mkdir" else [scratch])
    assert {name: os.environ[name] for name in before_env} == before_env
    assert tempfile.tempdir == "previous-cache"
    mkdtemp.assert_not_called()
    popen.assert_not_called()
    assert (outside / "sentinel").read_bytes() == b"keep\n"
    assert {p.name for p in outside.iterdir()} == {"sentinel"}
    assert not missing.exists()
    if redirect == "symlink-at-mkdir":
        assert scratch.is_symlink()
        assert scratch.readlink() == outside
    elif redirect == "resolve-before-mkdir":
        assert not scratch.exists()
    else:
        assert scratch.is_dir() and not scratch.is_symlink()
        assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("linked_root", [False, True])
def test_ordinary_scratch_supports_temporary_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
    linked_root: bool,
) -> None:
    root = tmp_path / "work"
    selected = tmp_path / "selected" if linked_root else root
    if linked_root:
        selected.symlink_to(root, target_is_directory=True)
    scratch = root / "scratch"
    if existing:
        scratch.mkdir(parents=True)
        (scratch / "sentinel").write_bytes(b"keep\n")
    for name in (WORK_DIR_ENV, "TMPDIR", "TEMP", "TMP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tempfile, "tempdir", None)

    assert ensure_scratch_dir(selected) == scratch
    assert ensure_work_dir(selected) == root
    with tempfile.TemporaryDirectory(prefix="lab50-") as allocated:
        assert Path(allocated).parent == scratch
    with tempfile.TemporaryFile() as allocated:
        allocated.write(b"temporary fixture\n")
        allocated.seek(0)
        assert allocated.read() == b"temporary fixture\n"
    assert {p.name for p in scratch.iterdir()} == ({"sentinel"} if existing else set())
    if existing:
        assert (scratch / "sentinel").read_bytes() == b"keep\n"


def test_default_work_dir_uses_external_xdg_or_home_without_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    nested = project / "src" / "klab"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv(WORK_DIR_ENV, raising=False)
    assert resolve_work_dir(cwd=nested) == tmp_path / "data" / "klab"
    assert not (tmp_path / "data").exists()
    monkeypatch.setenv("XDG_DATA_HOME", "relative-data")
    assert resolve_work_dir(cwd=nested) == home / ".local" / "share" / "klab"
    monkeypatch.setenv("XDG_DATA_HOME", " ")
    assert resolve_work_dir(cwd=nested) == home / ".local" / "share" / "klab"
    assert not home.exists()


def test_explicit_path_precedes_environment_and_retains_relative_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv(WORK_DIR_ENV, "legacy/klab-work")
    assert resolve_work_dir(cwd=tmp_path) == tmp_path / "legacy" / "klab-work"
    assert resolve_work_dir("chosen", cwd=tmp_path) == tmp_path / "chosen"
    assert (
        resolve_work_dir(tmp_path / "absolute", cwd=tmp_path) == tmp_path / "absolute"
    )
    assert not (tmp_path / "chosen").exists()
    assert not (tmp_path / "legacy").exists()


def test_configured_relative_work_dir_redirects_tempfile_allocations(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(WORK_DIR_ENV, "relative-scratch")
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, "previous-temp")
    monkeypatch.setattr(tempfile, "tempdir", None)

    work_dir = ensure_work_dir()

    assert work_dir == tmp_path / "relative-scratch"
    assert (work_dir / "scratch").is_dir()
    assert os.environ[WORK_DIR_ENV] == str(work_dir)
    assert all(
        os.environ[name] == str(work_dir / "scratch")
        for name in ("TMPDIR", "TEMP", "TMP")
    )
    with tempfile.TemporaryDirectory(prefix="klab-test-") as temp_dir:
        assert Path(temp_dir).parent == work_dir / "scratch"


def test_worker_rpc_fallback_allocates_under_work_dir_and_cleans_up(
    tmp_path: Path, monkeypatch
) -> None:
    work_dir = tmp_path / "scratch"
    monkeypatch.setenv(WORK_DIR_ENV, str(work_dir))
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, "previous-temp")
    monkeypatch.setattr(tempfile, "tempdir", None)
    allocated_dirs: list[Path] = []
    original_mkdtemp = tempfile.mkdtemp

    def record_mkdtemp(*, prefix: str, dir: str | Path | None = None) -> str:
        assert dir is not None
        allocated_dirs.append(Path(dir))
        return original_mkdtemp(prefix=prefix, dir=dir)

    def unavailable_endpoint(*args, **kwargs):
        raise FileNotFoundError("endpoint unavailable")

    worker = WorkerDefinition(
        worker_id="test-worker",
        name="test-worker",
        transport="local",
        python=Path(sys.executable),
        root=tmp_path,
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    monkeypatch.setattr(transport.tempfile, "mkdtemp", record_mkdtemp)
    monkeypatch.setattr(transport, "_endpoint_argv", lambda _: ["unused-endpoint"])
    monkeypatch.setattr(transport.subprocess, "Popen", unavailable_endpoint)

    with pytest.raises(FileNotFoundError, match="endpoint unavailable"):
        transport.call_worker(worker, "status", {"attempt_id": None})

    assert allocated_dirs == [work_dir / "scratch"]
    assert list(work_dir.iterdir()) == [work_dir / "scratch"]
    assert list((work_dir / "scratch").iterdir()) == []


def test_checkout_storage_checks_find_git_ancestor_without_creating_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    nested = checkout / "not-created" / "persistent"
    outside = tmp_path / "persistent"
    monkeypatch.setenv(WORK_DIR_ENV, str(nested))
    checks = storage_checks(resolve_work_dir())
    assert checks == [
        {
            "reason_code": CHECKOUT_STORAGE_REASON,
            "kind": "work_root",
            "path": str(nested),
            "git_root": str(checkout),
        }
    ]
    assert warn_storage_checks(checks) == checks
    assert CHECKOUT_STORAGE_REASON in capsys.readouterr().err
    assert (
        storage_checks(checkout / "not-created" / "output", kind="output")[0]["kind"]
        == "output"
    )
    assert storage_checks(outside) == []
    assert not nested.exists()
    assert not outside.exists()


def test_two_checkouts_share_external_root_without_checkout_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "persistent-state"
    monkeypatch.setenv(WORK_DIR_ENV, str(root))
    for name in ("first", "second"):
        checkout = tmp_path / name
        checkout.mkdir()
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        assert resolve_work_dir(cwd=checkout) == root
        assert storage_checks(resolve_work_dir(cwd=checkout)) == []
    assert not root.exists()


def test_implicit_runs_follow_selected_root_and_explicit_destination_is_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.cli.main import build_parser

    environment = tmp_path / "environment-state"
    explicit = tmp_path / "flag-state"
    monkeypatch.setenv(WORK_DIR_ENV, str(environment))
    assert build_parser().parse_args(["eval", "run"]).runs_dir == str(
        environment / "runs"
    )
    parser = build_parser(explicit)
    assert parser.parse_args(
        ["--work-dir", str(explicit), "eval", "run"]
    ).runs_dir == str(explicit / "runs")
    selected = parser.parse_args(["eval", "run", "--runs-dir", "legacy/runs"])
    assert selected.runs_dir == "legacy/runs"
    assert not environment.exists()
    assert not explicit.exists()


def test_storage_observation_replay_keeps_original_bytes_and_detects_tamper(
    tmp_path: Path,
) -> None:
    import json

    from klab.result_publication import read_canonical

    root = tmp_path / "persistent-state"
    checks = [
        {
            "reason_code": CHECKOUT_STORAGE_REASON,
            "kind": "output",
            "path": str(tmp_path / "output"),
            "git_root": str(tmp_path),
        }
    ]
    receipt = record_storage_observation(
        root, checks, "train", target=tmp_path / "output"
    )
    before = receipt.read_bytes()
    assert read_canonical(receipt)["storage_checks"] == checks
    assert (
        record_storage_observation(root, checks, "train", target=tmp_path / "output")
        == receipt
    )
    assert receipt.read_bytes() == before
    value = json.loads(before)
    value["observed_at_utc"] = "2000-01-01T00:00:00Z"
    receipt.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="conflicting storage observation"):
        record_storage_observation(root, checks, "train", target=tmp_path / "output")
