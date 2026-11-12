"""Operational guards count the owned process tree, not host-wide used swap."""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from klab import operational_monitor as monitor


class FakeProcess:
    def __init__(self, pid: int, parent: int, ctime: float, rss: int, swap: int):
        self.pid, self.parent, self.ctime = pid, parent, ctime
        self.rss, self.swap = rss, swap

    def ppid(self) -> int:
        return self.parent

    def create_time(self) -> float:
        return self.ctime

    def status(self) -> str:
        return "running"

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=self.rss)

    def memory_full_info(self) -> SimpleNamespace:
        return SimpleNamespace(swap=self.swap)


def _sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **limits: int
) -> monitor.MonitorSample:
    root = FakeProcess(100, 1, 1.0, 50, 2)
    child = FakeProcess(101, 100, 2.0, 70, 9)
    unrelated = FakeProcess(102, 1, 3.0, 900, 2000)
    monkeypatch.setattr(
        monitor.psutil, "process_iter", lambda _: [root, child, unrelated]
    )
    monkeypatch.setattr(
        monitor.psutil, "virtual_memory", lambda: SimpleNamespace(available=75)
    )
    monkeypatch.setattr(
        monitor.psutil,
        "swap_memory",
        lambda: SimpleNamespace(free=55, used=2 * 1024**3),
    )
    monkeypatch.setattr(monitor, "_storage", lambda _: (1000, 200))
    return monitor.sample(
        monitor.ProcessIdentity(pid=100, create_time=1.0),
        {100: monitor.ProcessIdentity(pid=100, create_time=1.0)},
        monitor.MonitorPolicy(monitor_policy_version=1, **limits),
        tmp_path,
        reserved_bytes=300,
        reserved_inodes=10,
    )


def test_swap_only_counts_owned_descendants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _sample(
        monkeypatch, tmp_path, max_tree_swap_bytes=12, min_host_free_swap_bytes=50
    )
    assert result.tree_swap_bytes == 11
    assert result.host_used_swap_bytes == 2 * 1024**3
    assert result.violations == []
    assert [(p.pid, p.create_time) for p in result.processes] == [
        (100, 1.0),
        (101, 2.0),
    ]


@pytest.mark.parametrize(
    ("limit", "value"),
    [
        ("max_tree_swap_bytes", 10),
        ("min_host_free_swap_bytes", 56),
        ("min_host_available_ram_bytes", 76),
        ("max_tree_rss_bytes", 119),
        ("min_disk_free_bytes", 1001),
        ("min_disk_free_inodes", 201),
        ("min_projected_disk_free_bytes", 701),
        ("min_projected_disk_free_inodes", 191),
    ],
)
def test_each_guard_violation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, limit: str, value: int
) -> None:
    assert _sample(monkeypatch, tmp_path, **{limit: value}).violations == [limit]


def test_unavailable_child_swap_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        monitor,
        "_swap_bytes",
        lambda p: (
            (_ for _ in ()).throw(RuntimeError("unavailable")) if p.pid == 101 else 2
        ),
    )
    assert _sample(monkeypatch, tmp_path, max_tree_swap_bytes=12).violations == [
        "max_tree_swap_bytes"
    ]


def test_unrequested_unavailable_swap_does_not_invalidate_known_rss(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        monitor,
        "_swap_bytes",
        lambda p: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    result = _sample(monkeypatch, tmp_path, max_tree_rss_bytes=120)
    assert result.tree_rss_bytes == 120
    assert result.tree_swap_bytes is None
    assert result.violations == []


def test_pid_reuse_never_targets_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    root = monitor.ProcessIdentity(pid=100, create_time=1.0)
    monkeypatch.setattr(
        monitor.psutil, "process_iter", lambda _: [FakeProcess(100, 1, 2.0, 1, 0)]
    )
    with pytest.raises(RuntimeError, match="root PID was reused"):
        monitor._owned(root, {100: root})
    replacement = FakeProcess(101, 100, 4.0, 1, 0)
    old = monitor.ProcessIdentity(pid=101, create_time=3.0)
    monkeypatch.setattr(
        monitor.psutil,
        "process_iter",
        lambda _: [FakeProcess(100, 1, 1.0, 1, 0), replacement],
    )
    known = {100: root, 101: old}
    assert [p.pid for p in monitor._owned(root, known)] == [100]
    assert 101 not in known


def test_observed_orphan_remains_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    root = monitor.ProcessIdentity(pid=100, create_time=1.0)
    child = FakeProcess(101, 100, 2.0, 1, 5)
    monkeypatch.setattr(
        monitor.psutil,
        "process_iter",
        lambda _: [FakeProcess(100, 1, 1.0, 1, 0), child],
    )
    known = {100: root}
    assert [p.pid for p in monitor._owned(root, known)] == [100, 101]
    child.parent = 1
    monkeypatch.setattr(monitor.psutil, "process_iter", lambda _: [child])
    assert [p.pid for p in monitor._owned(root, known)] == [101]


@pytest.mark.parametrize(
    "payload",
    [
        {"monitor_policy_version": 1, "scientific_sha256": "abc"},
        {"monitor_policy_version": True},
    ],
)
def test_policy_rejects_invalid_or_scientific_fields(payload) -> None:
    with pytest.raises(ValueError):
        monitor.MonitorPolicy.model_validate(payload)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux pidfd ownership guard")
def test_guard_terminates_owned_command_without_touching_unrelated_process(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "monitor_policy_version: 1\nmax_tree_rss_bytes: 1\n"
        "interval_seconds: 0.02\ntermination_grace_seconds: 0.1\n"
    )
    command = [sys.executable, "-c", "import signal; signal.pause()"]
    unrelated = subprocess.Popen(command)
    try:
        result = monitor.monitor_command(
            command,
            monitor.load_monitor_policy(policy_path),
            workspace=tmp_path,
            log_dir=tmp_path / "logs",
            policy_path=policy_path,
        )
        assert result.status == "VIOLATED"
        assert result.violations == ["max_tree_rss_bytes"]
        assert result.returncode == -signal.SIGTERM
        assert unrelated.poll() is None
        assert not monitor.psutil.pid_exists(result.launch.root.pid)
    finally:
        unrelated.kill()
        unrelated.wait()


def test_workspace_sample_counts_files_reported_on_another_overlay_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overlayfs without xino gives files a different st_dev than directories."""
    import os
    import stat as stat_module

    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "payload.bin").write_bytes(b"x" * 1000)
    (tmp_path / "top.txt").write_bytes(b"y" * 24)
    real_stat = os.stat

    def overlay_stat(path, *args, **kwargs):
        info = real_stat(path, *args, **kwargs)
        if stat_module.S_ISDIR(info.st_mode):
            return info
        fields = list(info)
        fields[stat_module.ST_DEV] = info.st_dev + 1
        return os.stat_result(fields)

    monkeypatch.setattr(os, "stat", overlay_stat)
    assert monitor.sample_workspace_tree(tmp_path) == (1024, 4)
