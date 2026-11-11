from __future__ import annotations

import sqlite3

import pytest

from klab.resource_envelope import ResourceEnvelope
from klab.workers.store import ControllerStore


def _entry(index: int) -> dict[str, object]:
    return {
        "experiment_id": f"experiment-{index}",
        "attempt_id": f"attempt-{index}",
        "run_id": f"run-{index}",
        "spec": {"schema_version": 1, "experiment_id": f"experiment-{index}"},
    }


def test_enqueue_many_is_atomic_on_late_duplicate(tmp_path) -> None:
    store = ControllerStore(tmp_path)
    first, duplicate = _entry(1), _entry(2)
    duplicate["run_id"] = first["run_id"]
    with pytest.raises(sqlite3.IntegrityError):
        store.enqueue_many([first, duplicate])
    assert store.attempts() == []


def test_queue_cancel_is_durable_and_never_reopens_terminal_attempt(tmp_path) -> None:
    store = ControllerStore(tmp_path)
    entry = _entry(1)
    store.enqueue_many([entry])
    assert store.request_cancel("run-1")["status"] == "CANCELLED"
    assert store.request_cancel("run-1")["status"] == "CANCELLED"
    assert store.attempt_by_run("run-1")["status"] == "CANCELLED"
    assert store.attempt_by_run("run-1")["ingestion_status"] == "NOT_REQUIRED"


def test_assignment_is_compare_and_set_and_keeps_identity(tmp_path) -> None:
    store = ControllerStore(tmp_path)
    entry = _entry(1)
    store.enqueue_many([entry])
    assert store.assign("attempt-1", "worker-1")
    assert not store.assign("attempt-1", "worker-2")
    attempt = store.attempt_by_run("run-1")
    assert attempt is not None
    assert attempt["worker_id"] == "worker-1"
    assert attempt["status"] == "ASSIGNED"


def test_durable_cancel_survives_running_receipt_and_connection_loss(tmp_path) -> None:
    store = ControllerStore(tmp_path)
    store.enqueue_many([_entry(1)])
    store.assign("attempt-1", "worker-1")
    store.request_cancel("run-1")
    store.set_receipt("attempt-1", {"state": "RUNNING"}, "RUNNING")
    store.mark_unknown_if_nonterminal("attempt-1", "connection lost")
    assert (
        ControllerStore(tmp_path).attempt_by_run("run-1")["status"]
        == "CANCEL_REQUESTED"
    )
    store.set_receipt(
        "attempt-1",
        {"state": "INTERRUPTED", "cancellation_acknowledged": True},
        "CANCELLED",
    )
    store.set_receipt("attempt-1", {"state": "RUNNING"}, "RUNNING")
    assert store.attempt_by_run("run-1")["status"] == "CANCELLED"


def test_rejected_envelope_does_not_create_attempt_or_dispatch_bundle(tmp_path) -> None:
    from test_training import config as training_config

    from klab.workers.controller import Controller

    controller = Controller(tmp_path / "controller")
    config = training_config(tmp_path / "inputs")
    envelope = ResourceEnvelope(resource_envelope_version=1, min_disk_bytes=1 << 62)
    with pytest.raises(ValueError, match="min_disk_bytes"):
        controller.submit(config, resource_envelope=envelope)
    assert controller.store.attempts() == []
    assert not (controller.root / ".dispatch").exists()
    assert not config.dataset.cache_dir.exists()


def test_queue_limit_rejects_second_submission_before_dispatch(tmp_path) -> None:
    from test_training import config as training_config

    from klab.workers.controller import Controller

    controller = Controller(tmp_path / "controller")
    config = training_config(tmp_path / "inputs")
    envelope = ResourceEnvelope(resource_envelope_version=1, max_queue_depth=1)
    first = controller.submit(config, resource_envelope=envelope)
    attempt = controller.store.attempt_by_run(first.run_id)
    assert attempt["status"] == "QUEUED"
    assert attempt["spec"]["resource_envelope"] == envelope.model_dump(mode="json")
    with pytest.raises(ValueError, match="max_queue_depth"):
        controller.submit(config, resource_envelope=envelope)
    assert len(controller.store.attempts()) == 1
    assert len(list((controller.root / ".dispatch").iterdir())) == 1


def test_receipt_for_another_attempt_cannot_change_durable_assignment(tmp_path) -> None:
    from test_training import config as training_config

    from klab.workers.controller import Controller
    from klab.workers.transport import ProtocolError

    controller = Controller(tmp_path / "controller")
    submission = controller.submit(training_config(tmp_path / "inputs"))
    assert (
        "resource_envelope"
        not in controller.store.attempt_by_run(submission.run_id)["spec"]
    )
    controller.store.assign(submission.attempt_id, "worker")
    attempt = controller.store.attempt_by_run(submission.run_id)
    spec = controller._model("ExperimentSpec", attempt["spec"])
    receipt = {
        "attempt_id": "another-attempt",
        "run_id": submission.run_id,
        "experiment_id": submission.experiment_id,
        "worker_id": "worker",
        "spec_digest": spec.digest(),
        "bundle_digest": spec.dispatch_bundle_digest,
        "state": "COMPLETE",
    }
    with pytest.raises(ProtocolError):
        controller._reconcile_receipt(attempt, receipt)
    observed = controller.store.attempt_by_run(submission.run_id)
    assert observed["status"] == "ASSIGNED"
    assert observed["terminal_receipt"] is None


def test_failure_before_run_creation_finishes_ingestion(tmp_path) -> None:
    from klab.workers.controller import Controller

    controller = Controller(tmp_path)
    controller.store.enqueue_many([_entry(1)])
    controller.store.assign("attempt-1", "worker")
    receipt = {
        "state": "FAILED",
        "training_started_at": "2026-09-23T00:00:00+00:00",
        "artifacts": [
            {"relative_path": "stderr.log", "size_bytes": 0, "sha256": "0" * 64}
        ],
    }
    controller.store.set_receipt("attempt-1", receipt, "FAILED")
    controller._ingest(
        controller.store.attempt_by_run("run-1"), None, receipt, "worker"
    )
    observed = controller.store.attempt_by_run("run-1")
    assert observed["status"] == "FAILED"
    assert observed["ingestion_status"] == "NOT_REQUIRED"


def test_recovery_preserves_receipt_and_fences_stale_observations(tmp_path) -> None:
    from test_training import config as training_config

    from klab.workers.controller import Controller

    controller = Controller(tmp_path / "controller")
    submission = controller.submit(training_config(tmp_path / "inputs"))
    store = controller.store
    store.assign(submission.attempt_id, "lost-worker")
    original = {"state": "RUNNING", "heartbeat_at": "2026-09-23T00:00:00+00:00"}
    store.set_receipt(submission.attempt_id, original, "RUNNING")
    row = store.attempt_by_run(submission.run_id)
    spec = controller._model("ExperimentSpec", row["spec"])
    observation = {
        "reason": "HOSTED_WORKER_LOST",
        "commit_sha256": "a" * 64,
        "confirmed_at": "2026-09-23T00:01:00+00:00",
        "instance_id": "owned-instance",
    }
    identity = {
        "expected_worker_id": "lost-worker",
        "expected_spec_digest": spec.digest(),
        "expected_bundle_digest": spec.dispatch_bundle_digest,
    }
    store.record_recovery(submission.attempt_id, observation, **identity)
    store.record_recovery(submission.attempt_id, observation, **identity)
    store.set_receipt(submission.attempt_id, {"state": "RUNNING"}, "RUNNING")
    store.mark_ingestion_complete(submission.attempt_id)
    recovered = store.attempt_by_run(submission.run_id)
    assert recovered["status"] == "UNKNOWN"
    assert recovered["receipt"] == original
    assert recovered["terminal_receipt"] is None
    assert recovered["recovery_observation"] == observation
    assert recovered["ingestion_status"] == "COMPLETE"
    with pytest.raises(ValueError, match="conflicting recovery"):
        store.record_recovery(
            submission.attempt_id,
            {**observation, "commit_sha256": "b" * 64},
            **identity,
        )
    store.set_receipt(submission.attempt_id, {"state": "COMPLETE"}, "COMPLETE")
    conflict = store.attempt_by_run(submission.run_id)
    assert conflict["terminal_receipt"] is None
    assert "CONFLICT" in conflict["ingestion_error"]


def test_old_readonly_queue_projects_null_recovery_without_migrating(tmp_path) -> None:
    store = ControllerStore(tmp_path)
    store.enqueue_many([_entry(1)])
    with store.transaction() as con:
        con.execute("ALTER TABLE attempts DROP COLUMN recovery_observation_json")
    readonly = ControllerStore(tmp_path, read_only=True)
    assert readonly.attempt_by_run("run-1")["recovery_observation"] is None
    with readonly.metrics._connect() as con:
        assert "recovery_observation_json" not in {
            row[1] for row in con.execute("PRAGMA table_info(attempts)")
        }
    ControllerStore(tmp_path)
    with store.metrics._connect() as con:
        assert "recovery_observation_json" in {
            row[1] for row in con.execute("PRAGMA table_info(attempts)")
        }
