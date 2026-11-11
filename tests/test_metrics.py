from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from klab.training.metrics import SCHEMA_VERSION, ExperimentStore, _envelope_digest


def _run(store: ExperimentStore, run_id: str = "run") -> None:
    store.create_run(run_id, {"name": run_id}, {"purpose": "test"})


def test_reopen_keeps_origin_and_exposes_committed_records(tmp_path) -> None:
    store = ExperimentStore(tmp_path)
    _run(store)
    first = store.export_records()
    reopened = ExperimentStore(tmp_path)
    second = reopened.export_records()
    assert first["origin_id"] == second["origin_id"]
    assert second["next_sequence"] == 1
    assert second["records"][0]["kind"] == "run_created"


def test_v2_migrates_with_backup_and_preserves_stage(tmp_path) -> None:
    path = tmp_path / "experiments.sqlite3"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE runs(run_id TEXT PRIMARY KEY,name TEXT,status TEXT,parent_run_id TEXT,created_at TEXT,updated_at TEXT,config_json TEXT,metadata_json TEXT,latest_checkpoint TEXT);"
        "CREATE TABLE metrics(run_id TEXT,step INTEGER,tokens_seen INTEGER,wall_time REAL,name TEXT,value REAL,PRIMARY KEY(run_id,step,name));"
        "CREATE TABLE events(id INTEGER PRIMARY KEY,run_id TEXT,step INTEGER,tokens_seen INTEGER,wall_time REAL,kind TEXT,payload_json TEXT);"
        "CREATE TABLE stage_history(run_id TEXT,sequence INTEGER,stage TEXT,status TEXT,step INTEGER,tokens_seen INTEGER,started_at TEXT,finished_at TEXT,payload_json TEXT,PRIMARY KEY(run_id,sequence));"
        "INSERT INTO runs VALUES('old','old','complete',NULL,'t','t','{}','{}',NULL);"
        "INSERT INTO stage_history VALUES('old',1,'TRAINING','COMPLETE',1,2,'t','t','{}');"
        "PRAGMA user_version=2;"
    )
    con.commit()
    con.close()

    store = ExperimentStore(tmp_path)
    assert (tmp_path / "experiments.sqlite3.v2.bak").is_file()
    assert store.path.exists()
    with sqlite3.connect(store.path) as migrated:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert (
            migrated.execute("SELECT stage FROM stage_history").fetchone()[0]
            == "TRAINING"
        )


def test_unknown_schema_and_orphan_legacy_rows_are_rejected(tmp_path) -> None:
    path = tmp_path / "experiments.sqlite3"
    with sqlite3.connect(path) as con:
        con.execute("PRAGMA user_version=99")
    with pytest.raises(ValueError, match="unsupported"):
        ExperimentStore(tmp_path)

    path.unlink()
    with sqlite3.connect(path) as con:
        con.executescript(
            "CREATE TABLE runs(run_id TEXT PRIMARY KEY);"
            "CREATE TABLE metrics(run_id TEXT);"
            "INSERT INTO metrics VALUES('missing'); PRAGMA user_version=2;"
        )
    with pytest.raises(ValueError, match="orphan"):
        ExperimentStore(tmp_path)


def test_failed_domain_write_rolls_back_its_outbox_record(tmp_path) -> None:
    store = ExperimentStore(tmp_path)
    _run(store)
    with pytest.raises(sqlite3.IntegrityError):
        _run(store)
    exported = store.export_records()
    assert [record["kind"] for record in exported["records"]] == ["run_created"]


def test_import_is_checked_contiguous_idempotent_and_never_echoes(tmp_path) -> None:
    source = ExperimentStore(tmp_path / "source")
    _run(source)
    source.log_metrics("run", 1, 8, 0.1, {"train/loss": 1.0})
    batch = source.export_records()
    target = ExperimentStore(tmp_path / "target")
    target.import_records(batch["records"])
    target.import_records(batch["records"])
    assert target.export_records()["records"] == []

    corrupt = dict(batch["records"][0])
    corrupt["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        target.import_records([corrupt])
    conflicting_replay = dict(batch["records"][0])
    conflicting_replay["payload"] = {
        **conflicting_replay["payload"],
        "name": "different",
    }
    conflicting_replay["sha256"] = _envelope_digest(conflicting_replay)
    with pytest.raises(ValueError, match="conflicting"):
        target.import_records([conflicting_replay])

    gap = dict(batch["records"][1])
    gap["origin_id"] = "00000000-0000-4000-8000-000000000001"
    gap["sha256"] = _envelope_digest(gap)
    with pytest.raises(ValueError, match="gap"):
        target.import_records([gap])


def test_stage_checkpoint_and_calibration_are_preserved_in_replication(
    tmp_path,
) -> None:
    source = ExperimentStore(tmp_path / "source")
    _run(source)
    source.record_stage("run", 1, "TRAINING", "RUNNING", 0, 0, "now")
    source.record_checkpoint(
        "run",
        {
            "generation_id": 1,
            "relative_path": "checkpoints/one",
            "manifest_sha256": "digest",
            "step": 1,
            "tokens_seen": 8,
            "created_at": "now",
            "bytes": 10,
            "validation_loss": 0.5,
            "engine": "pytorch",
            "backend": "cpu",
        },
    )
    source.record_calibration("key", "run", {"peak": 10})
    target = ExperimentStore(tmp_path / "target")
    target.import_records(source.export_records()["records"])
    assert ExperimentStore.get_calibration(tmp_path / "target", "key") == [
        {"run_id": "run", "observation": {"peak": 10}}
    ]
    with sqlite3.connect(target.path) as con:
        assert con.execute("SELECT COUNT(*) FROM stage_history").fetchone()[0] == 1
        assert con.execute("SELECT digest FROM checkpoints").fetchone()[0] == "digest"


def test_duplicate_keys_and_nonfinite_values_are_rejected(tmp_path) -> None:
    store = ExperimentStore(tmp_path)
    _run(store)
    with pytest.raises(ValueError, match="nonfinite"):
        store.log_metrics("run", 1, 1, 1.0, {"train/loss": float("nan")})
    with pytest.raises(ValueError, match="duplicate JSON key"):
        store.import_records(['{"schema_version":1,"schema_version":1}'])


@pytest.mark.parametrize("field", ["schema_version", "sequence"])
def test_boolean_protocol_numbers_are_rejected_before_projection(tmp_path, field):
    source = ExperimentStore(tmp_path / "source")
    _run(source)
    record = source.export_records()["records"][0]
    record[field] = True
    record["sha256"] = _envelope_digest(record)
    target = ExperimentStore(tmp_path / "target")
    with pytest.raises(ValueError):
        target.import_records([record])
    with sqlite3.connect(target.path) as con:
        assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_export_byte_budget_includes_canonical_array_framing(tmp_path):
    from klab.training.metrics import _canonical_json

    store = ExperimentStore(tmp_path)
    _run(store)
    store.log_metrics("run", 1, 8, 0.1, {"train/loss": 1.0})
    first = store.export_records(limit=1)["records"]
    size = len(_canonical_json(first).encode("utf-8"))
    with pytest.raises(ValueError):
        store.export_records(max_bytes=size - 1)
    batch = store.export_records(max_bytes=size)
    assert batch["records"] == first
    assert batch["next_sequence"] == 1 and batch["has_more"]
    assert store.export_records(after_sequence=1)["next_sequence"] == 2


def test_oversized_envelope_rolls_back_domain_write(tmp_path):
    store = ExperimentStore(tmp_path)
    with pytest.raises(ValueError):
        store.create_run("large", {"name": "large"}, {"report": "x" * 65_536})
    assert store.export_records()["records"] == []
    with sqlite3.connect(store.path) as con:
        assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    _run(store)
    assert store.export_records()["next_sequence"] == 1


def test_reopening_rejects_missing_columns_and_sequence_disagreement(tmp_path):
    store = ExperimentStore(tmp_path / "columns")
    with sqlite3.connect(store.path) as con:
        con.execute("ALTER TABLE calibration RENAME COLUMN observation_json TO broken")
    with pytest.raises(ValueError):
        ExperimentStore(tmp_path / "columns")
    store = ExperimentStore(tmp_path / "sequence")
    _run(store)
    with sqlite3.connect(store.path) as con:
        con.execute("UPDATE store_metadata SET next_sequence=9")
    with pytest.raises(ValueError):
        ExperimentStore(tmp_path / "sequence")


def test_v3_migrates_queue_tables_with_backup_and_preserves_history(tmp_path) -> None:
    source = ExperimentStore(tmp_path)
    _run(source)
    with sqlite3.connect(source.path) as con:
        con.execute("PRAGMA user_version=3")
        con.execute("DROP TABLE attempts")
        con.execute("DROP TABLE experiments")
        con.execute("DROP TABLE workers")
        con.execute("DROP TABLE runtime_progress_snapshots")
    migrated = ExperimentStore(tmp_path)
    assert (tmp_path / "experiments.sqlite3.v3.bak").is_file()
    with sqlite3.connect(migrated.path) as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert con.execute("SELECT run_id FROM runs").fetchone()[0] == "run"
        assert con.execute("SELECT COUNT(*) FROM workers").fetchone()[0] == 0


def test_runtime_progress_snapshot_replaces_latest_without_outbox_growth(
    tmp_path,
) -> None:
    store = ExperimentStore(tmp_path)
    _run(store)
    first = {
        "schema_version": 1,
        "phase": "training",
        "event": "heartbeat",
        "state": "RUNNING",
        "observed_at_utc": "2026-01-01T00:00:00+00:00",
    }
    latest = {
        **first,
        "event": "finished",
        "state": "COMPLETE",
        "completed_work": 64,
    }

    store.upsert_runtime_progress_snapshot("run", 1, 32, 1.0, first)
    store.upsert_runtime_progress_snapshot("run", 2, 64, 2.0, latest)

    assert [record["kind"] for record in store.export_records()["records"]] == [
        "run_created"
    ]
    with sqlite3.connect(store.path) as con:
        assert (
            con.execute("SELECT COUNT(*) FROM runtime_progress_snapshots").fetchone()[0]
            == 1
        )
    assert ExperimentStore.get_runtime_progress_records(tmp_path, "run") == [
        {
            "kind": "runtime_progress",
            "step": 2,
            "tokens_seen": 64,
            "wall_time": 2.0,
            "payload": latest,
        }
    ]

    final = {"schema_version": 1, "run_status": "completed"}
    store.log_event("run", 2, 64, 3.0, "runtime_final_observation", final)
    records = ExperimentStore.get_runtime_progress_records(tmp_path, "run")
    assert [record["kind"] for record in records] == [
        "runtime_progress",
        "runtime_final_observation",
    ]
    assert records[-1]["payload"] == final
    assert ExperimentStore.get_runtime_progress_records(tmp_path, "run", limit=1) == [
        records[-1]
    ]


@pytest.mark.parametrize("size", [16, 80_000])
def test_manifest_reference_preserves_complete_payload_and_bounded_transport(
    tmp_path, size
):
    from types import SimpleNamespace

    from klab.training.metrics import MAX_ENVELOPE_BYTES, _canonical_json
    from klab.workers.artifacts import _restore_crash_projection

    source = ExperimentStore(tmp_path / "source")
    _run(source)
    payload = {"run_id": "run", "source_identity": {"inventory": "x" * size}}
    encoded = _canonical_json(payload)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    source.register_manifest("run", digest, payload)
    batch = source.export_records()["records"]
    assert batch[-1]["kind"] == "manifest_reference"
    assert len(_canonical_json(batch[-1]).encode()) < MAX_ENVELOPE_BYTES
    assert batch[-1]["payload"]["sha256"] == digest
    with source._connect() as con:
        assert con.execute("SELECT json FROM manifests").fetchone()[0] == encoded
    target = ExperimentStore(tmp_path / "target")
    target.import_records(batch)
    with target._connect() as con:
        assert "manifest_reference" in json.loads(
            con.execute("SELECT json FROM manifests").fetchone()[0]
        )
    with pytest.raises(ValueError, match="digest differs"):
        target.materialize_manifest("run", digest, {**payload, "tampered": True})
    _restore_crash_projection(
        target, SimpleNamespace(run_id="run", state="COMPLETE"), (payload, [])
    )
    with target._connect() as con:
        assert (
            json.loads(con.execute("SELECT json FROM manifests").fetchone()[0])
            == payload
        )
    target.import_records(batch)
    with target._connect() as con:
        assert con.execute("SELECT digest,json FROM manifests").fetchone() == (
            digest,
            encoded,
        )
    assert target.export_records()["records"] == []


def test_manifest_reference_rejects_malformed_and_mismatched_evidence(tmp_path):
    from klab.training.metrics import _canonical_json

    source = ExperimentStore(tmp_path / "source")
    _run(source)
    payload = {"run_id": "run"}
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    with pytest.raises(ValueError, match="digest differs"):
        source.register_manifest("run", "0" * 64, payload)
    assert source.export_records()["next_sequence"] == 1
    source.register_manifest("run", digest, payload)
    records = source.export_records()["records"]
    invalid = [
        ("relative_path", "../manifest.json"),
        ("sha256", "bad"),
        ("sha256", None),
        ("sha256", "A" * 64),
        ("sha256", "g" * 64),
        ("size_bytes", True),
        ("size_bytes", 0),
        ("size_bytes", -1),
        ("extra", "not permitted"),
    ]
    for index, (field, value) in enumerate(invalid):
        bad = json.loads(json.dumps(records))
        bad[-1]["payload"][field] = value
        bad[-1]["sha256"] = _envelope_digest(bad[-1])
        target = ExperimentStore(tmp_path / f"bad-{index}")
        with pytest.raises(ValueError, match="malformed manifest"):
            target.import_records(bad)
        with target._connect() as con:
            assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    bad = json.loads(json.dumps(records))
    bad[-1]["payload"]["size_bytes"] += 1
    bad[-1]["sha256"] = _envelope_digest(bad[-1])
    target = ExperimentStore(tmp_path / "wrong-size")
    target.import_records(bad)
    with pytest.raises(ValueError, match="reference differs"):
        target.materialize_manifest("run", digest, payload)


def test_historical_inline_manifest_remains_readable(tmp_path):
    from klab.training.metrics import _canonical_json

    source = ExperimentStore(tmp_path / "source")
    _run(source)
    payload = {"run_id": "run"}
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    source.register_manifest("run", digest, payload)
    records = source.export_records()["records"]
    records[-1].update(kind="manifest", payload={"digest": digest, "payload": payload})
    records[-1]["sha256"] = _envelope_digest(records[-1])
    target = ExperimentStore(tmp_path / "target")
    target.import_records(records)
    target.materialize_manifest("run", digest, payload)
    with target._connect() as con:
        assert (
            json.loads(con.execute("SELECT json FROM manifests").fetchone()[0])
            == payload
        )


def test_manifest_reference_conflict_and_inline_digest_mismatch(tmp_path):
    from klab.training.metrics import _canonical_json

    source = ExperimentStore(tmp_path / "source")
    _run(source)
    payload = {"run_id": "run"}
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    source.register_manifest("run", digest, payload)
    records = source.export_records()["records"]
    target = ExperimentStore(tmp_path / "target")
    target.import_records(records)
    conflict = json.loads(json.dumps(records[-1]))
    conflict["sequence"] += 1
    conflict["payload"]["sha256"] = "0" * 64
    conflict["sha256"] = _envelope_digest(conflict)
    with pytest.raises(ValueError, match="conflicting manifest reference"):
        target.import_records([conflict])
    with target._connect() as con:
        con.execute(
            "UPDATE manifests SET digest=?,json=?", ("0" * 64, _canonical_json(payload))
        )
    with pytest.raises(ValueError, match="different digest"):
        target.materialize_manifest("run", digest, payload)
    with pytest.raises(ValueError, match="absent"):
        target.materialize_manifest("missing", digest, payload)


def test_session_commits_each_write_and_recovers_from_outbox_failure(tmp_path) -> None:
    store = ExperimentStore(tmp_path)
    _run(store)
    with store.session():
        with store.session():
            store.log_metrics("run", 1, 8, 0.1, {"train/loss": 1.0})
        # An independent reader sees the commit before the session exits.
        reader = ExperimentStore(tmp_path, read_only=True)
        assert len(reader.export_records()["records"]) == 2
        with pytest.raises(ValueError, match="envelope exceeds"):
            store.log_event("run", 1, 8, 0.1, "oversized", {"text": "x" * 65536})
        store.log_event("run", 1, 8, 0.1, "recovered", {})
        with sqlite3.connect(store.path) as con:
            assert con.execute("SELECT kind FROM events").fetchall() == [("recovered",)]
        assert [r["sequence"] for r in reader.export_records()["records"]] == [1, 2, 3]
    # A later call and a later session can both use the same store.
    store.log_metrics("run", 2, 16, 0.2, {"train/loss": 0.9})
    with store.session():
        assert len(store.export_records()["records"]) == 4


def test_session_allows_heartbeat_thread_writes(tmp_path) -> None:
    store = ExperimentStore(tmp_path)
    _run(store)
    with store.session(), ThreadPoolExecutor(max_workers=1) as executor:
        for step in range(3):
            pending = executor.submit(
                store.upsert_runtime_progress_snapshot,
                "run",
                step,
                step * 8,
                float(step),
                {"state": "RUNNING"},
            )
            store.log_metrics("run", step, step * 8, float(step), {"train/loss": 1.0})
            pending.result(timeout=5)
        assert (
            ExperimentStore.get_runtime_progress_records(tmp_path, "run")[0]["step"]
            == 2
        )
        assert len(store.export_records()["records"]) == 4


def test_session_commits_survive_process_exit_without_cleanup(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import sys
from pathlib import Path
from klab.training.metrics import ExperimentStore
store = ExperimentStore(Path(sys.argv[1]))
store.create_run("run", {}, {})
with store.session():
    store.log_metrics("run", 1, 8, 0.1, {"train/loss": 1.0})
    store.log_event("run", 1, 8, 0.1, "runtime_final_observation", {"status": "failed"})
    os._exit(0)
""",
            str(tmp_path),
        ],
        check=True,
        timeout=15,
    )
    reopened = ExperimentStore(tmp_path)
    records = reopened.export_records()["records"]
    assert [r["kind"] for r in records] == ["run_created", "metrics", "event"]
    assert records[-1]["payload"]["payload"] == {"status": "failed"}
    with sqlite3.connect(reopened.path) as con:
        assert con.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert con.execute("SELECT value FROM metrics").fetchall() == [(1.0,)]


def test_large_configuration_replication_resolves_only_from_matching_manifest(tmp_path):
    from klab.canonical import canonical_digest
    from klab.training.metrics import MAX_ENVELOPE_BYTES

    config = {
        "name": "mapped",
        "model": {"memory_token_mapping": {"token_ids": list(range(32768))}},
    }
    manifest = {"run_id": "mapped", "effective_config": config}
    digest = canonical_digest(manifest)
    source = ExperimentStore(tmp_path / "source")
    source.create_run(
        "mapped", config, {"manifest_sha256": digest}, config_manifest=manifest
    )
    source.register_manifest("mapped", digest, manifest)
    records = source.export_records()["records"]
    assert all(len(json.dumps(r).encode()) < MAX_ENVELOPE_BYTES for r in records)
    assert records[0]["payload"]["config"]["field"] == "effective_config"
    with source._connect() as con:
        assert (
            json.loads(con.execute("SELECT config_json FROM runs").fetchone()[0])
            == config
        )
    target = ExperimentStore(tmp_path / "target")
    target.import_records(records)
    with pytest.raises(ValueError, match="digest differs"):
        target.materialize_manifest("mapped", digest, manifest | {"tampered": True})
    target.materialize_manifest("mapped", digest, manifest)
    with target._connect() as con:
        assert (
            json.loads(con.execute("SELECT config_json FROM runs").fetchone()[0])
            == config
        )
    assert target.export_records()["records"] == []
    with pytest.raises(ValueError, match="complete manifest"):
        ExperimentStore(tmp_path / "unbound").create_run(
            "mapped", config, {"manifest_sha256": digest}
        )
