from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from klab.workers.leases import acquire_lease, lease_key, process_matches


def test_accelerator_aliases_contend_across_worker_roots(tmp_path: Path) -> None:
    state = tmp_path / "host-state"
    first = acquire_lease(
        worker_id="mps", backend="mps", physical_device_id=None, state_dir=state
    )
    assert first is not None
    try:
        assert (
            acquire_lease(
                worker_id="mlx",
                backend="metal",
                physical_device_id=None,
                state_dir=state,
            )
            is None
        )
    finally:
        first.close()


def test_unknown_accelerator_identity_contends_with_known_device(
    tmp_path: Path,
) -> None:
    known = acquire_lease(
        worker_id="cuda",
        backend="cuda",
        physical_device_id="pci-0000",
        state_dir=tmp_path,
    )
    assert known is not None
    try:
        assert (
            acquire_lease(
                worker_id="unknown",
                backend="rocm",
                physical_device_id=None,
                state_dir=tmp_path,
            )
            is None
        )
    finally:
        known.close()


def test_distinct_known_accelerators_share_gate_but_keep_device_slots(
    tmp_path: Path,
) -> None:
    first = acquire_lease(
        worker_id="one",
        backend="cuda",
        physical_device_id="pci-one",
        state_dir=tmp_path,
    )
    second = acquire_lease(
        worker_id="two",
        backend="cuda",
        physical_device_id="pci-two",
        state_dir=tmp_path,
    )
    assert first is not None and second is not None
    try:
        assert (
            acquire_lease(
                worker_id="one",
                backend="cuda",
                physical_device_id="pci-three",
                state_dir=tmp_path,
            )
            is None
        )
    finally:
        first.close()
        second.close()


def test_cpu_workers_use_distinct_logical_slots(tmp_path: Path) -> None:
    one = acquire_lease(
        worker_id="cpu-one", backend="cpu", physical_device_id=None, state_dir=tmp_path
    )
    two = acquire_lease(
        worker_id="cpu-two", backend="cpu", physical_device_id=None, state_dir=tmp_path
    )
    assert one is not None and two is not None
    try:
        assert one.key != two.key
    finally:
        one.close()
        two.close()


def test_duplicate_executor_claim_is_nonblocking(tmp_path: Path) -> None:
    from klab.workers.execution import _claim_file

    first = _claim_file(tmp_path)
    assert first is not None
    try:
        assert _claim_file(tmp_path) is None
    finally:
        first.close()


def test_dead_running_receipt_becomes_unknown_without_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KLAB_WORKER_LEASE_DIR", str(tmp_path / "leases"))
    from klab.workers.execution import (
        _receipt_payload,
        _write_receipt,
        status_worker,
    )
    from klab.workers.models import WorkerDefinition

    worker = WorkerDefinition(
        worker_id=f"worker-{tmp_path.name}",
        name=f"worker-{tmp_path.name}",
        transport="local",
        host=None,
        python=Path(sys.executable).absolute(),
        root=tmp_path,
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    payload = _receipt_payload(
        worker,
        {
            "attempt_id": "attempt",
            "run_id": "run",
            "experiment_id": "experiment",
            "spec_digest": "a" * 64,
            "bundle_digest": "b" * 64,
        },
    )
    payload.update(state="RUNNING", pid=999999, process_start="0", boot_id="wrong")
    _write_receipt(worker, payload)
    holder = acquire_lease(
        worker_id=worker.worker_id, backend="cpu", physical_device_id=None
    )
    assert holder is not None
    try:
        assert status_worker(worker, "attempt")["receipt"]["state"] == "RUNNING"
    finally:
        holder.close()
    assert status_worker(worker, "attempt")["receipt"]["state"] == "UNKNOWN"


def test_cancel_marks_attempt_without_signalling_pid(tmp_path: Path) -> None:
    from klab.workers.execution import (
        _receipt_payload,
        _write_receipt,
        cancel_attempt,
    )
    from klab.workers.models import WorkerDefinition

    worker = WorkerDefinition(
        worker_id="worker",
        name="worker",
        transport="local",
        host=None,
        python=Path(os.__file__).resolve(),
        root=tmp_path,
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    payload = _receipt_payload(
        worker,
        {
            "attempt_id": "attempt",
            "run_id": "run",
            "experiment_id": "experiment",
            "spec_digest": "a" * 64,
            "bundle_digest": "b" * 64,
        },
    )
    _write_receipt(worker, payload)
    receipt = cancel_attempt(worker, "attempt")
    assert receipt["cancellation_requested"]
    assert receipt["cancellation_acknowledged"]
    assert receipt["state"] == "INTERRUPTED"
    assert (tmp_path / "attempts" / "attempt" / "cancel.json").is_file()
    from klab.workers.execution import execute_attempt

    assert cancel_attempt(worker, "future") is None
    future = _receipt_payload(
        worker,
        {
            "attempt_id": "future",
            "run_id": "future-run",
            "experiment_id": "future-experiment",
            "spec_digest": "a" * 64,
            "bundle_digest": "b" * 64,
        },
    )
    _write_receipt(worker, future)
    acknowledged = execute_attempt(worker, "future")
    assert acknowledged["state"] == "INTERRUPTED"
    assert acknowledged["cancellation_acknowledged"]
    assert not (worker.root / "runs/future-run").exists()


def test_late_cancel_cannot_regress_complete_receipt(tmp_path: Path) -> None:
    from klab.workers.execution import (
        _receipt_payload,
        _write_receipt,
        cancel_attempt,
    )
    from klab.workers.models import WorkerDefinition

    worker = WorkerDefinition(
        worker_id="worker",
        name="worker",
        transport="local",
        host=None,
        python=Path(os.__file__).resolve(),
        root=tmp_path,
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    receipt = _receipt_payload(
        worker,
        {
            "attempt_id": "attempt",
            "run_id": "run",
            "experiment_id": "experiment",
            "spec_digest": "a" * 64,
            "bundle_digest": "b" * 64,
        },
    )
    receipt.update(state="COMPLETE", phase="complete")
    _write_receipt(worker, receipt)
    assert cancel_attempt(worker, "attempt")["state"] == "COMPLETE"
    assert not (tmp_path / "attempts" / "attempt" / "cancel.json").exists()


def test_stale_heartbeat_mutation_cannot_replace_terminal_receipt(
    tmp_path: Path,
) -> None:
    from klab.workers.execution import (
        _mutate_receipt,
        _receipt_payload,
        _write_receipt,
    )
    from klab.workers.models import WorkerDefinition

    worker = WorkerDefinition(
        worker_id="worker",
        name="worker",
        transport="local",
        host=None,
        python=Path(os.__file__).resolve(),
        root=tmp_path,
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    receipt = _receipt_payload(
        worker,
        {
            "attempt_id": "attempt",
            "run_id": "run",
            "experiment_id": "experiment",
            "spec_digest": "a" * 64,
            "bundle_digest": "b" * 64,
        },
    )
    receipt.update(state="COMPLETE", phase="complete")
    _write_receipt(worker, receipt)

    result = _mutate_receipt(
        worker,
        "attempt",
        lambda stale: stale | {"state": "RUNNING", "heartbeat_at": "later"},
    )
    assert result["state"] == "COMPLETE"


def test_recycled_or_dead_pid_never_counts_as_live_claim() -> None:
    assert not process_matches(os.getpid(), "0", "wrong-boot")


def test_unknown_accelerator_identity_uses_host_wide_key() -> None:
    assert lease_key(
        worker_id="a", backend="mps", physical_device_id=None
    ) == lease_key(worker_id="b", backend="metal", physical_device_id=None)


def test_inherited_pilot_keeps_lease_after_wrapper_closes(tmp_path: Path) -> None:
    import subprocess
    import sys

    lease = acquire_lease(
        worker_id="owner", backend="mps", physical_device_id=None, state_dir=tmp_path
    )
    assert lease is not None
    with subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; print('ready', flush=True); sys.stdin.buffer.read()",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        pass_fds=lease.inherited_fds,
    ) as pilot:
        try:
            assert pilot.stdout.readline() == b"ready\n"
            lease.close()
            assert (
                acquire_lease(
                    worker_id="contender",
                    backend="metal",
                    physical_device_id=None,
                    state_dir=tmp_path,
                )
                is None
            )
        finally:
            pilot.communicate(timeout=10)
            lease.close()
    next_owner = acquire_lease(
        worker_id="contender",
        backend="metal",
        physical_device_id=None,
        state_dir=tmp_path,
    )
    assert next_owner is not None
    next_owner.close()


def test_complete_worker_inventory_preserves_verified_triage_through_ingestion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil
    import sys

    from test_training import config as training_config

    from klab.evaluation.post_train_triage import read_triage
    from klab.training.metrics import ExperimentStore
    from klab.training.trainer import train
    from klab.workers import artifacts
    from klab.workers.controller import Controller
    from klab.workers.execution import _final_artifacts, _receipt_payload
    from klab.workers.models import AttemptReceipt, WorkerDefinition

    original = training_config(tmp_path / "worker")
    configured = original.model_copy(
        update={
            "training": original.training.model_copy(
                update={"max_steps": 4, "max_tokens": 128}
            )
        }
    )
    controller = Controller(tmp_path / "controller")
    submission = controller.submit(configured)
    attempt = controller.store.attempt_by_run(submission.run_id)
    spec = controller._model("ExperimentSpec", attempt["spec"])
    _, bundle = controller._dispatch_bundle(spec)
    worker = WorkerDefinition(
        worker_id="triage-worker",
        name="triage-worker",
        transport="local",
        python=Path(sys.executable),
        root=configured.logging.root_dir.parent,
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    train(
        configured,
        run_id=submission.run_id,
        worker_id=worker.worker_id,
        experiment_id=submission.experiment_id,
        attempt_id=submission.attempt_id,
        dispatch_metadata={
            "spec_digest": spec.digest(),
            "bundle_digest": bundle.digest(),
        },
    )
    source_run = configured.logging.root_dir / submission.run_id
    assert read_triage(submission.run_id, configured.logging.root_dir) is not None
    controller.store.metrics.import_records(
        ExperimentStore(configured.logging.root_dir).export_records()["records"]
    )
    raw = _receipt_payload(
        worker,
        {
            "attempt_id": submission.attempt_id,
            "run_id": submission.run_id,
            "experiment_id": submission.experiment_id,
            "spec_digest": spec.digest(),
            "bundle_digest": bundle.digest(),
        },
    )
    raw.update(state="COMPLETE", artifacts=_final_artifacts(worker, raw))
    triage_items = [
        item
        for item in raw["artifacts"]
        if item["relative_path"].startswith("run/post-train-triage/")
    ]
    assert len(triage_items) == 1
    receipt = AttemptReceipt.model_validate(raw)

    def local_transfer(_worker, _receipt, item, destination, **kwargs):
        shutil.copyfile(
            source_run / item.relative_path.removeprefix("run/"), destination
        )

    monkeypatch.setattr(artifacts, "_download", local_transfer)
    artifacts.ingest_attempt_artifacts(
        worker,
        receipt,
        controller.root,
        spec=spec,
        bundle=bundle,
        records=controller.store.metrics,
    )
    target = (
        controller.root
        / submission.run_id
        / triage_items[0]["relative_path"].removeprefix("run/")
    )
    assert (
        target.read_bytes()
        == (
            source_run / triage_items[0]["relative_path"].removeprefix("run/")
        ).read_bytes()
    )
    assert read_triage(submission.run_id, controller.root) is not None


def test_receipt_includes_manifest_declared_hidden_cache_owner(tmp_path: Path) -> None:
    import json

    from klab.canonical import canonical_digest
    from klab.workers.execution import _final_artifacts

    run = tmp_path / "runs" / "run"
    (run / "data").mkdir(parents=True)
    (run / "data" / ".klab-cache-owner.json").write_text("{}")
    (run / ".transient").write_text("not declared")
    manifest = {
        "manifest_version": 1,
        "artifacts": [{"relative_path": "data/.klab-cache-owner.json"}],
    }
    (run / "manifest.json").write_text(
        json.dumps(
            {
                **manifest,
                "sha256": canonical_digest(manifest),
            }
        )
    )
    files = _final_artifacts(
        {"root": str(tmp_path)}, {"run_id": "run", "attempt_id": "attempt"}
    )
    names = {item["relative_path"] for item in files}
    assert "run/data/.klab-cache-owner.json" in names
    assert "run/.transient" not in names


def test_same_attempt_cold_request_is_sticky_and_outside_scientific_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from test_training import config

    from klab.canonical import canonical_json
    from klab.workers import execution
    from klab.workers.controller import Controller
    from klab.workers.models import WorkerDefinition

    controller = Controller(tmp_path / "controller")
    submission = controller.submit(config(tmp_path / "inputs"))
    attempt = controller.store.attempt_by_run(submission.run_id)
    spec = controller._model("ExperimentSpec", attempt["spec"])
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(canonical_json(spec.model_dump(mode="json")))
    worker = WorkerDefinition(
        worker_id="worker",
        name="worker",
        transport="local",
        python=Path(sys.executable).absolute(),
        root=tmp_path / "worker",
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    # Model a crash after durable preparation and before the detached executor.
    monkeypatch.setattr(execution, "_spawn_executor", lambda *_: None)
    payload = {
        "attempt_id": submission.attempt_id,
        "run_id": submission.run_id,
        "experiment_id": submission.experiment_id,
        "spec_digest": spec.digest(),
        "bundle_digest": spec.dispatch_bundle_digest,
    }
    receipts = []
    directory = worker.root / "attempts" / submission.attempt_id
    for requested, expected in ((False, False), (True, True), (False, True)):
        receipts.append(
            execution.launch_attempt(
                worker, {**payload, "cold_verify": requested}, spec_path
            )
        )
        assert json.loads((directory / "verification.json").read_bytes()) == {
            "verification_version": 1,
            "cold_verify": expected,
        }
        assert (directory / "spec.json").read_bytes() == spec_path.read_bytes()
    assert {
        (row["attempt_id"], row["spec_digest"], row["bundle_digest"])
        for row in receipts
    } == {(submission.attempt_id, spec.digest(), spec.dispatch_bundle_digest)}
    assert "cold_verify" not in json.loads(spec_path.read_bytes())


@pytest.mark.parametrize("marker", ["regular", "absent", "directory", "symlink"])
def test_trainer_startup_interrupt_requires_regular_worker_marker(
    tmp_path, monkeypatch, caplog, marker
):
    from types import SimpleNamespace

    from klab import staging
    from klab.config.loading import load_config
    from klab.training import trainer
    from klab.workers import bundles, execution
    from klab.workers.models import (
        ExperimentSpec,
        WorkerDefinition,
        current_required_versions,
    )

    config = load_config(
        Path(__file__).resolve().parents[1] / "examples/lab-job-try/baseline.yaml"
    )
    worker = WorkerDefinition(
        worker_id=f"fixture-{tmp_path.name}",
        name="fixture",
        transport="local",
        python=Path(sys.executable).absolute(),
        root=tmp_path / "worker",
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    spec = ExperimentSpec(
        experiment_id="fixture",
        config=config,
        config_sha256="a" * 64,
        required_versions=current_required_versions(),
        source_identity_sha256="b" * 64,
        dispatch_bundle_digest="c" * 64,
    )
    receipt = execution._receipt_payload(
        worker,
        {
            "attempt_id": "attempt",
            "run_id": "never-created",
            "experiment_id": "fixture",
            "spec_digest": "d" * 64,
            "bundle_digest": "c" * 64,
        },
    )
    execution._write_receipt(worker, receipt)
    lease_root = tmp_path / "leases"
    monkeypatch.setenv("KLAB_WORKER_LEASE_DIR", str(lease_root))
    monkeypatch.setattr(
        execution,
        "_spec_from_path",
        lambda *a: {"payload": spec.model_dump(mode="json")},
    )
    monkeypatch.setattr(execution, "source_identity", lambda: {"sha256": "b" * 64})
    monkeypatch.setattr(execution, "validate_runtime", lambda *a, **k: None)
    monkeypatch.setattr(execution, "_caps", lambda *a, **k: {})
    monkeypatch.setattr(
        execution,
        "_runtime_for_definition",
        lambda *a: SimpleNamespace(backend="cpu", physical_device_id=None),
    )
    monkeypatch.setattr(
        bundles,
        "materialize_dispatch_bundle",
        lambda *a, **k: SimpleNamespace(
            digest=lambda: "c" * 64,
            config_sha256=spec.config_sha256,
            config=config,
            source_identity_sha256=spec.source_identity_sha256,
            continuation=spec.continuation,
        ),
    )
    phases = []
    monkeypatch.setattr(staging, "stage", lambda *a, **k: phases.append("stage"))
    error = InterruptedError("fixture startup interruption")

    def stop(config, **kwargs):
        assert phases == ["stage"]
        phases.append("train")
        cancel = kwargs["cancel_path"]
        if marker == "regular":
            cancel.write_text("retain cancellation")
        elif marker == "directory":
            cancel.mkdir()
        elif marker == "symlink":
            target = tmp_path / "target"
            target.write_text("retain target")
            cancel.symlink_to(target)
        raise error

    def forbidden(*args, **kwargs):
        pytest.fail("fixture executed a model")

    monkeypatch.setattr(trainer, "train", stop)
    monkeypatch.setattr(trainer.PyTorchEngine, "initialize", forbidden)
    if marker == "regular":
        result = execution.execute_attempt(worker, "attempt")
        assert result["state"] == "INTERRUPTED"
        assert result["phase"] == "cancelled_before_training"
        assert result["cancellation_requested"] is True
        assert result["cancellation_acknowledged"] is True
    else:
        with pytest.raises(InterruptedError) as caught:
            execution.execute_attempt(worker, "attempt")
        assert caught.value is error
        result = execution._load_receipt(worker, "attempt")
        assert result["state"] == "FAILED"
        assert result["phase"] == "failed"
        assert result["cancellation_acknowledged"] is False
        assert result["error"] == {"code": "INTERRUPTEDERROR", "message": str(error)}
    assert phases == ["stage", "train"]
    assert not (worker.root / "runs/never-created").exists()
    # Terminal receipts are durable and another execute cannot retry the attempt.
    assert execution.execute_attempt(worker, "attempt") == result
    assert phases == ["stage", "train"]
    cancel = worker.root / "attempts/attempt/cancel.json"
    if marker == "regular":
        assert cancel.read_text() == "retain cancellation"
    elif marker == "directory":
        assert cancel.is_dir()
    elif marker == "symlink":
        assert cancel.is_symlink() and cancel.read_text() == "retain target"
    else:
        assert not cancel.exists()

    assert lease_root.is_dir()
    assert list(lease_root.glob("*.lock"))
    failures = [
        record
        for record in caplog.records
        if record.name == execution._LOGGER.name
        and record.getMessage() == "Worker training attempt failed"
    ]
    assert len(failures) == (0 if marker == "regular" else 1)
    if failures:
        assert failures[0].exc_info[1] is error
