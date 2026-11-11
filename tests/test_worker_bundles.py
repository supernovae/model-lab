"""Boundary regressions for immutable worker bundle artifacts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from klab.training.manifest import ArtifactIdentity, sha256_file
from klab.workers.artifacts import (
    _download,
    artifact_inventory,
    read_artifact_chunk,
)


def _receipt(root: Path, content: bytes) -> SimpleNamespace:
    source = root / "runs" / "run-1" / "manifest.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    return SimpleNamespace(
        attempt_id="attempt-1",
        run_id="run-1",
        artifacts=(
            ArtifactIdentity("run/manifest.json", sha256_file(source), len(content)),
        ),
    )


def test_known_inventory_chunk_is_bounded_and_hashed(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path, b"immutable artifact")
    result, chunk = read_artifact_chunk(
        tmp_path, receipt, "run/manifest.json", 2, 4, tmp_path / "chunk"
    )
    assert chunk.read_bytes() == b"muta"
    assert result["offset"] == 2
    assert result["next_offset"] == 6
    assert result["eof"] is False


def test_artifact_rejects_escaped_or_unadvertised_members(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path, b"immutable artifact")
    with pytest.raises(ValueError):
        read_artifact_chunk(
            tmp_path, receipt, "run/../manifest.json", 0, 1, tmp_path / "x"
        )
    with pytest.raises(ValueError):
        read_artifact_chunk(tmp_path, receipt, "run/other", 0, 1, tmp_path / "x")


def test_artifact_inventory_rejects_post_receipt_mutation(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path, b"immutable artifact")
    (tmp_path / "runs" / "run-1" / "manifest.json").write_bytes(b"changed")
    with pytest.raises(ValueError):
        artifact_inventory(tmp_path, receipt)


def test_download_uses_scoped_wire_chunks_and_verifies_final_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.workers import transport

    receipt = _receipt(tmp_path, b"abcdefgh")
    item = receipt.artifacts[0]
    calls: list[Path] = []

    def artifact_reply(_worker, _op, payload, *, receive_dir, timeout):
        calls.append(receive_dir)
        start = payload["offset"]
        content = b"abcdefgh"[start : start + 3]
        attachment = receive_dir / "chunk"
        attachment.write_bytes(content)
        return SimpleNamespace(
            result={
                "file_sha256": item.sha256,
                "total_length": item.size_bytes,
                "offset": start,
                "chunk_sha256": sha256_file(attachment),
                "next_offset": start + len(content),
                "eof": start + len(content) == item.size_bytes,
            },
            attachments={"chunk": attachment},
        )

    monkeypatch.setattr(transport, "call_worker", artifact_reply)
    output = tmp_path / "downloaded"
    _download(None, receipt, item, output, receive_root=tmp_path, timeout=30)
    assert output.read_bytes() == b"abcdefgh"
    assert calls and all(not path.exists() for path in calls)


def test_check_install_race_accepts_verified_redundant_assets(tmp_path: Path) -> None:
    from test_training import config as training_config

    from klab.staging import verify_prepared_inputs
    from klab.workers.bundles import (
        install_dispatch_bundle,
        materialize_dispatch_bundle,
        prepare_dispatch_bundle,
    )

    config = training_config(tmp_path / "source")
    bundle = tmp_path / "dispatch"
    manifest = prepare_dispatch_bundle(config, bundle)
    worker = tmp_path / "worker"
    checked = install_dispatch_bundle(worker, manifest, {}, check_only=True)
    attachments = {
        f"assets/{item.sha256}": bundle / item.relative_path for item in manifest.files
    }
    assert set(checked["missing_asset_digests"]) == {
        name.removeprefix("assets/") for name in attachments
    }
    install_dispatch_bundle(worker, manifest, attachments, check_only=False)
    # Another caller can install the checked assets before our original upload arrives.
    install_dispatch_bundle(worker, manifest, attachments, check_only=False)
    materialized = tmp_path / "materialized"
    materialize_dispatch_bundle(worker, manifest.digest(), materialized)
    verify_prepared_inputs(materialized, config)
    assert (
        materialized / "assets/tokenizer.json"
    ).read_bytes() == config.tokenizer.path.read_bytes()
    incoming = next(iter(attachments.values()))
    incoming.write_bytes(incoming.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        install_dispatch_bundle(worker, manifest, attachments, check_only=False)
    # Rejected redundant bytes did not poison the immutable cached publication.
    materialize_dispatch_bundle(worker, manifest.digest(), tmp_path / "after-rejection")


def test_resume_and_promotion_bundle_survive_original_input_removal(
    tmp_path: Path,
) -> None:
    import json
    import shutil

    from test_training import config as training_config

    from klab.training.trainer import train
    from klab.workers.bundles import prepare_dispatch_bundle

    config = training_config(tmp_path / "source")
    parent_id = train(config, run_id="parent")
    parent = config.logging.root_dir / parent_id
    pointer = json.loads((parent / "checkpoints/latest.json").read_text())
    generation = parent / "checkpoints" / pointer["relative_path"]
    generation_digest = json.loads((generation / "manifest.json").read_text())["sha256"]
    promoted = prepare_dispatch_bundle(config, tmp_path / "promote", promote=generation)
    shutil.rmtree(config.tokenizer.path.parent)
    shutil.rmtree(config.dataset.cache_dir)
    resumed = prepare_dispatch_bundle(config, tmp_path / "resume", resume=generation)
    assert resumed.continuation.checkpoint_sha256 == generation_digest
    assert promoted.continuation.checkpoint_sha256 == generation_digest
    assert (tmp_path / "resume/assets/data/train.npy").read_bytes() == (
        parent / "data/train.npy"
    ).read_bytes()
    assert (tmp_path / "promote/assets/tokenizer.json").read_bytes() == (
        parent / "tokenizer.json"
    ).read_bytes()


def test_aggregate_receipt_limit_rejects_before_disk_or_rpc(tmp_path: Path) -> None:
    from klab.workers.artifacts import ingest_attempt_artifacts
    from klab.workers.transport import MAX_ATTACHMENT_BYTES

    receipt = SimpleNamespace(
        run_id="oversized",
        artifacts=tuple(
            ArtifactIdentity(f"run/file-{index}", "a" * 64, MAX_ATTACHMENT_BYTES)
            for index in range(5)
        ),
    )
    target = tmp_path / "controller"
    with pytest.raises(ValueError):
        ingest_attempt_artifacts(
            None, receipt, target, spec=None, bundle=None, records=None
        )
    assert not target.exists()


def test_continuation_honors_pointer_digest_and_requested_tokenizer(
    tmp_path: Path,
) -> None:
    import json

    from test_training import config as training_config

    from klab.training.trainer import train
    from klab.workers.bundles import prepare_dispatch_bundle

    config = training_config(tmp_path / "source")
    train(config, run_id="parent", stop_after_step=1)
    pointer = config.logging.root_dir / "parent/checkpoints/latest.json"
    original = pointer.read_bytes()
    changed = json.loads(original)
    changed["manifest_sha256"] = "0" * 64
    pointer.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        prepare_dispatch_bundle(config, tmp_path / "wrong-pointer", resume=pointer)
    pointer.write_bytes(original)
    config.tokenizer.path.unlink()
    with pytest.raises(ValueError):
        prepare_dispatch_bundle(config, tmp_path / "missing-tokenizer", promote=pointer)
    assert not (tmp_path / "wrong-pointer/bundle.json").exists()
    assert not (tmp_path / "missing-tokenizer/bundle.json").exists()


@pytest.mark.parametrize(
    "corruption",
    ["configuration", "projection", "premature_complete", "missing_crash_projection"],
)
def test_publication_binds_native_run_to_dispatch_and_replica(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    import json
    import shutil
    import sys

    from test_training import config as training_config

    from klab.training.checkpoints import CheckpointManager
    from klab.training.metrics import ExperimentStore
    from klab.training.trainer import train
    from klab.workers import artifacts
    from klab.workers.controller import Controller
    from klab.workers.execution import _final_artifacts, _receipt_payload
    from klab.workers.models import AttemptReceipt, WorkerDefinition

    config = training_config(tmp_path / "worker")
    controller = Controller(tmp_path / "controller")
    submission = controller.submit(config)
    attempt = controller.store.attempt_by_run(submission.run_id)
    spec = controller._model("ExperimentSpec", attempt["spec"])
    _, bundle = controller._dispatch_bundle(spec)
    worker = WorkerDefinition(
        worker_id="native-worker",
        name="native-worker",
        transport="local",
        python=Path(sys.executable),
        root=config.logging.root_dir.parent,
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    executed = (
        config.model_copy(update={"seed": config.seed + 1})
        if corruption == "configuration"
        else config
    )
    train(
        executed,
        run_id=submission.run_id,
        worker_id=worker.worker_id,
        experiment_id=submission.experiment_id,
        attempt_id=submission.attempt_id,
        dispatch_metadata={
            "spec_digest": spec.digest(),
            "bundle_digest": bundle.digest(),
        },
        stop_after_step=1,
    )
    run = config.logging.root_dir / submission.run_id
    assert CheckpointManager(run).verify(run / "checkpoints/latest.json").valid
    controller.store.metrics.import_records(
        ExperimentStore(config.logging.root_dir).export_records()["records"]
    )
    if corruption == "projection":
        with controller.store.transaction() as con:
            con.execute(
                "UPDATE checkpoints SET tokens_seen=tokens_seen+1 WHERE run_id=?",
                (submission.run_id,),
            )
    if corruption == "missing_crash_projection":
        with controller.store.transaction() as con:
            con.execute("DELETE FROM checkpoints WHERE run_id=?", (submission.run_id,))
            con.execute("DELETE FROM manifests WHERE run_id=?", (submission.run_id,))
        (run / "checkpoints/latest.json").unlink()
    if corruption == "premature_complete":
        progress_path = run / "progress.json"
        progress = json.loads(progress_path.read_bytes())
        progress["status"] = "completed"
        progress_path.write_text(json.dumps(progress))
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
    state = (
        "UNKNOWN"
        if corruption == "missing_crash_projection"
        else "COMPLETE"
        if corruption == "premature_complete"
        else "INTERRUPTED"
    )
    raw.update(state=state, artifacts=_final_artifacts(worker, raw))
    receipt = AttemptReceipt.model_validate(raw)

    def local_transfer(_worker, _receipt, item, destination, **kwargs):
        shutil.copyfile(run / item.relative_path.removeprefix("run/"), destination)

    monkeypatch.setattr(artifacts, "_download", local_transfer)
    if corruption == "missing_crash_projection":
        controller.store.assign(submission.attempt_id, worker.worker_id)
        controller._reconcile_receipt(attempt, raw)
        artifacts.ingest_attempt_artifacts(
            worker,
            receipt,
            controller.root,
            spec=spec,
            bundle=bundle,
            records=controller.store.metrics,
        )
        controller.store.mark_ingestion_complete(submission.attempt_id)
        with controller.store.metrics._connect() as con:
            assert con.execute(
                "SELECT MAX(step),MAX(tokens_seen) FROM checkpoints WHERE run_id=?",
                (submission.run_id,),
            ).fetchone() == (1, 32)
            assert con.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0
        config.tokenizer.path.unlink()
        shutil.rmtree(config.dataset.cache_dir)
        child = controller.resume(submission.run_id, worker=worker.name)
        assert child.run_id != submission.run_id
        assert controller.store.attempt_by_run(child.run_id)["status"] == "QUEUED"
        assert not (controller.root / child.run_id).exists()
        return
    with pytest.raises(ValueError):
        artifacts.ingest_attempt_artifacts(
            worker,
            receipt,
            controller.root,
            spec=spec,
            bundle=bundle,
            records=controller.store.metrics,
        )
    assert not (controller.root / submission.run_id).exists()
