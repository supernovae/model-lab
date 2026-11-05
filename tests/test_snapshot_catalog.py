import json
import sqlite3
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from sparselab.dashboard.snapshot_catalog import (
    SavedSnapshot,
    SnapshotOwner,
    catalog,
    resolve_candidate,
)


def selected():
    return SavedSnapshot(
        "run", "checkpoints/step_0001_gen_0001", "a" * 64, 1, "2026-01-01", 100
    )


def test_catalog_excludes_incomplete_unverified_and_unsafe_rows(tmp_path):
    with sqlite3.connect(tmp_path / "experiments.sqlite3") as c:
        c.execute(
            "create table checkpoints(run_id,relative_path,digest,step,created_at,size_bytes,verification_status)"
        )
        for path, status, date in [
            ("step_0001_gen_0001", "verified", "2026-01-01"),
            ("step_0002_gen_0002", "verified", "2026-01-02"),
            (".partial", "verified", "2026-01-03"),
            ("step_0003_gen_0003", "pending", "2026-01-04"),
        ]:
            c.execute(
                "insert into checkpoints values(?,?,?,?,?,?,?)",
                ("run", path, "a" * 64, 1, date, 100, status),
            )
    rows = catalog(tmp_path)
    assert [row.checkpoint for row in rows] == [
        "checkpoints/step_0002_gen_0002",
        "checkpoints/step_0001_gen_0001",
    ]


def test_missing_corrupt_and_archived_candidates_fail_without_fallback(tmp_path):
    item = selected()
    mapping = tmp_path / "mapping.json"
    with pytest.raises(FileNotFoundError):
        resolve_candidate(tmp_path, item, mapping)
    path = tmp_path / item.run / item.checkpoint
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"sha256": "wrong", "step": 1}))
    with pytest.raises(ValueError, match="identity"):
        resolve_candidate(tmp_path, item, mapping)
    (path / "manifest.json").write_text(json.dumps({"sha256": item.digest, "step": 1}))
    assert resolve_candidate(tmp_path, item, mapping) == path
    mapping.write_text(
        json.dumps(
            {"checkpoints": [{"source": str(path), "archive": "/archive/generation"}]}
        )
    )
    with pytest.raises(ValueError, match="Archived"):
        resolve_candidate(tmp_path, item, mapping)


def test_owner_serializes_load_and_generation_and_reuses_exact_identity():
    owner = SnapshotOwner()
    item = selected()
    entered = Event()
    release = Event()
    calls = []
    results = []

    def loader(value):
        calls.append(value.key)
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(
            identity={"run_id": value.run, "checkpoint_sha256": value.digest}
        )

    def action(model):
        return model.identity.copy()

    thread = Thread(target=lambda: results.append(owner.execute(item, loader, action)))
    thread.start()
    assert entered.wait(5)
    with pytest.raises(ValueError, match="busy"):
        owner.execute(replace(item, digest="b" * 64), loader, action)
    release.set()
    thread.join(5)
    assert results[0]["checkpoint_sha256"] == item.digest
    assert owner.execute(item, loader, action) == results[0] and len(calls) == 1
    entered.clear()
    release.clear()

    def blocked_action(model):
        entered.set()
        assert release.wait(5)
        return model.identity.copy()

    thread = Thread(target=lambda: owner.execute(item, loader, blocked_action))
    thread.start()
    assert entered.wait(5)
    with pytest.raises(ValueError, match="busy"):
        owner.execute(replace(item, digest="b" * 64), loader, action)
    release.set()
    thread.join(5)

    def failed(value):
        assert owner.loaded is None
        raise ValueError("integrity check failed")

    with pytest.raises(ValueError, match="integrity"):
        owner.execute(replace(item, digest="b" * 64), failed, action)
    assert owner.loaded is None and owner.key is None and owner.status[0] == "failed"
    owner.execute(item, loader, action)
    assert owner.status[0] == "ready"


def test_successful_switch_evicts_previous_model_before_loading():
    owner = SnapshotOwner()
    first = selected()
    second = replace(
        first, checkpoint="checkpoints/step_0002_gen_0002", digest="b" * 64, step=2
    )
    loads = []

    def loader(value):
        assert owner.loaded is None and owner.key is None
        loads.append(value.key)
        return SimpleNamespace(
            identity={"run_id": value.run, "checkpoint_sha256": value.digest}
        )

    action = lambda model: model.identity.copy()
    assert owner.execute(first, loader, action)["checkpoint_sha256"] == first.digest
    assert owner.execute(second, loader, action)["checkpoint_sha256"] == second.digest
    assert owner.execute(second, loader, action)["checkpoint_sha256"] == second.digest
    assert loads == [first.key, second.key]
