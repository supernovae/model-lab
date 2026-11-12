from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from klab.canonical import canonical_digest, canonical_json
from klab.training.checkpoints import CheckpointManager
from klab.training.metrics import ExperimentStore
from klab.workspace_cleanup import (
    apply_cleanup,
    mark_prepared_cache,
    plan_cleanup,
    write_plan,
)


def _workspace(tmp_path: Path, *, status: str = "completed") -> tuple[Path, Path]:
    workspace = tmp_path / "experiment"
    cache = workspace / "cache"
    store = ExperimentStore(workspace / "runs")
    store.create_run("tiny", {"dataset": {"cache_dir": str(cache)}}, {})
    if status != "running":
        store.finish_run("tiny", status)
    return workspace, cache


def _cache(workspace: Path, cache: Path, number: int, *, owned: bool = True) -> Path:
    digest = f"{number:016x}" + "a" * 48
    root = cache / digest[:16]
    root.mkdir(parents=True)
    manifest = {
        "settings_sha256": digest,
        "packing_version": "contiguous-eos-v6",
        "supervision": {"kind": "all_tokens"},
    }
    for split in ("train", "validation"):
        path = root / f"{split}.npy"
        np.save(path, np.arange(4, dtype=np.int32), allow_pickle=False)
        manifest[split] = {
            "dtype": "int32",
            "shape": [4],
            "tokens": 4,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    (root / "manifest.json").write_text(json.dumps(manifest))
    if owned:
        mark_prepared_cache(workspace, root)
    os.utime(root, ns=(number, number))
    return root


def test_cleanup_plan_and_apply_only_owned_old_caches(tmp_path: Path) -> None:
    workspace, cache = _workspace(tmp_path)
    oldest = _cache(workspace, cache, 1)
    newer = _cache(workspace, cache, 2)
    newest = _cache(workspace, cache, 3)
    unowned = _cache(workspace, cache, 4, owned=False)
    plan = plan_cleanup(workspace, max_cache_entries=1)
    assert [item["path"] for item in plan["candidates"]] == [
        str(oldest.relative_to(workspace)),
        str(newer.relative_to(workspace)),
    ]
    path = tmp_path / "proposal.json"
    write_plan(plan, path)
    assert apply_cleanup(path)["removed"] == 2
    assert not oldest.exists() and not newer.exists()
    assert newest.exists() and unowned.exists()


def test_cleanup_rejects_stale_plan_without_deleting(tmp_path: Path) -> None:
    workspace, cache = _workspace(tmp_path)
    candidate = _cache(workspace, cache, 1)
    _cache(workspace, cache, 2)
    path = tmp_path / "proposal.json"
    write_plan(plan_cleanup(workspace, max_cache_entries=1), path)
    (candidate / "train.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="stale"):
        apply_cleanup(path)
    assert candidate.exists()


def test_cleanup_skips_active_runs_and_symlink_members(tmp_path: Path) -> None:
    workspace, cache = _workspace(tmp_path, status="running")
    candidate = _cache(workspace, cache, 1)
    newest = _cache(workspace, cache, 2)
    assert plan_cleanup(workspace, max_cache_entries=0)["candidates"] == []
    ExperimentStore(workspace / "runs").finish_run("tiny", "completed")
    (candidate / "unsafe").symlink_to(tmp_path)
    assert [
        item["path"]
        for item in plan_cleanup(workspace, max_cache_entries=0)["candidates"]
    ] == [str(newest.relative_to(workspace))]


def test_cleanup_excludes_cache_with_unowned_extra_file(tmp_path: Path) -> None:
    workspace, cache = _workspace(tmp_path)
    candidate = _cache(workspace, cache, 1)
    _cache(workspace, cache, 2)
    (candidate / "notes.txt").write_text("user file")
    assert plan_cleanup(workspace, max_cache_entries=1)["candidates"] == []


def test_cleanup_skips_cache_during_worker_attempt(tmp_path: Path) -> None:
    workspace, cache = _workspace(tmp_path)
    _cache(workspace, cache, 1)
    store = ExperimentStore(workspace / "runs")
    with store._connect() as con:
        con.execute(
            "INSERT INTO experiments VALUES('campaign', '{}', 'QUEUED', datetime('now'))"
        )
        con.execute(
            "INSERT INTO attempts(attempt_id,experiment_id,run_id,status) "
            "VALUES('attempt', 'campaign', 'tiny', 'QUEUED')"
        )
    assert plan_cleanup(workspace, max_cache_entries=0)["candidates"] == []


def test_prepared_data_marks_new_workspace_cache(tmp_path: Path, monkeypatch) -> None:
    from test_training import config

    from klab.data.packing import prepare_data
    from klab.data.tokenizer import load_tokenizer

    workspace = tmp_path / "experiment"
    configured = config(workspace)
    monkeypatch.setenv("KLAB_WORK_DIR", str(workspace))
    prepared = prepare_data(configured, load_tokenizer(configured.tokenizer.path))
    marker = json.loads((prepared.root / ".klab-cache-owner.json").read_text())
    assert marker["workspace"] == str(workspace)
    assert (
        prepare_data(configured, load_tokenizer(configured.tokenizer.path)).root
        == prepared.root
    )


def test_registered_checkpoint_is_never_proposed(tmp_path: Path) -> None:
    from test_checkpoints import _snapshot

    workspace, _ = _workspace(tmp_path)
    store = ExperimentStore(workspace / "runs")
    digest = canonical_digest({"run_id": "tiny"})
    store.register_manifest("tiny", digest, {"run_id": "tiny"})
    run = workspace / "runs" / "tiny"
    manager = CheckpointManager(run, manifest_sha256=digest)
    records = [manager.save(_snapshot(step)) for step in range(3)]
    store.record_checkpoint("tiny", records[0])
    assert plan_cleanup(workspace, max_extra_periodic=0)["candidates"] == []
    with store._connect() as con:
        con.execute(
            "DELETE FROM checkpoints WHERE run_id=? AND relative_path=?",
            ("tiny", records[0].relative_path),
        )
    store.create_run(
        "child", {"dataset": {"cache_dir": str(workspace / "cache")}}, {}, "tiny"
    )
    store.register_manifest(
        "child",
        hashlib.sha256(
            canonical_json({"checkpoint_sha256": records[0].manifest_sha256})
        ).hexdigest(),
        {"checkpoint_sha256": records[0].manifest_sha256},
    )
    store.finish_run("child", "completed")
    assert plan_cleanup(workspace, max_extra_periodic=0)["candidates"] == []
    with store._connect() as con:
        con.execute("DELETE FROM manifests WHERE run_id='child'")
    plan = plan_cleanup(workspace, max_extra_periodic=0)
    assert [item["path"] for item in plan["candidates"]] == [
        f"runs/tiny/checkpoints/{records[0].relative_path}"
    ]
    path = tmp_path / "checkpoint-plan.json"
    write_plan(plan, path)
    with (
        CheckpointManager(run).writer_lease(),
        pytest.raises(RuntimeError, match="lease is held"),
    ):
        apply_cleanup(path)
    assert (run / "checkpoints" / records[0].relative_path).exists()
    assert apply_cleanup(path)["removed"] == 1
    assert not (run / "checkpoints" / records[0].relative_path).exists()
    assert all(
        (run / "checkpoints" / record.relative_path).exists() for record in records[1:]
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_keeps_persistent_root_separate_from_all_temporary_factories(
    tmp_path: Path, explicit: bool
) -> None:
    workspace, _ = _workspace(tmp_path)
    root = tmp_path / "persistent-state"
    platform_temp = tmp_path / "platform-temp"
    platform_temp.mkdir()
    env = os.environ.copy()
    env.update(
        TMPDIR=str(platform_temp),
        TEMP=str(platform_temp),
        TMP=str(platform_temp),
        KLAB_WORK_DIR=str(tmp_path / "overridden") if explicit else str(root),
    )
    options = ["--work-dir", str(root)] if explicit else []
    script = (
        "import json, os, subprocess, sys, tempfile; "
        "from klab.cli.main import main; main(); "
        "parent = tempfile.mkdtemp(); "
        "child = subprocess.check_output([sys.executable, '-c', "
        "'import tempfile; print(tempfile.mkdtemp())'], text=True).strip(); "
        "external = subprocess.check_output(['mktemp', '-d', "
        "os.path.join(os.environ['TMPDIR'], 'tool.XXXXXXXX')], text=True).strip(); "
        "print(json.dumps({'parent': parent, 'child': child, 'external': external, "
        "'persistent': os.environ['KLAB_WORK_DIR']})); "
        "os.rmdir(parent); os.rmdir(child); os.rmdir(external)"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            *options,
            "workspace",
            "cleanup",
            "plan",
            str(workspace),
            "--output",
            str(tmp_path / "plan.json"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(result.stdout.splitlines()[-1])
    assert observed["persistent"] == str(root)
    assert {Path(observed[key]).parent for key in ("parent", "child", "external")} == {
        root / "scratch"
    }
    assert root.is_dir() and (root / "scratch").is_dir()
    assert not (tmp_path / "overridden").exists()
    assert not list(platform_temp.iterdir())


def test_cleanup_refuses_unresolved_manifest_lineage(tmp_path):
    source = ExperimentStore(tmp_path / "source")
    source.create_run("tiny", {}, {})
    payload = {"checkpoint_sha256": "a" * 64}
    digest = canonical_digest(payload)
    source.register_manifest("tiny", digest, payload)
    source.finish_run("tiny", "completed")
    workspace = tmp_path / "experiment"
    target = ExperimentStore(workspace / "runs")
    target.import_records(source.export_records()["records"])
    with pytest.raises(ValueError, match="unresolved manifest reference for tiny"):
        plan_cleanup(workspace)
    target.materialize_manifest("tiny", digest, payload)
    assert plan_cleanup(workspace)["candidates"] == []
