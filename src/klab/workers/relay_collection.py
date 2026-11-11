"""Controller-only authenticated offline collection and explicit lost-worker recovery."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from klab.canonical import canonical_json
from klab.training.manifest import ArtifactIdentity, read_manifest
from klab.training.metrics import _strict_json_loads
from klab.workers.relay import RelayStore, read_attempt_key
from klab.workers.relay_models import RelayBinding, RelayProfile
from klab.workspace_preflight import check_storage, require_storage


class RelayArtifactSource:
    def __init__(self, store: RelayStore) -> None:
        self.store = store

    def fetch(
        self, item: ArtifactIdentity, destination: Path, *, deadline: float | None
    ) -> None:
        old = self.store.timeout_seconds
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("relay collection deadline expired")
            self.store.timeout_seconds = min(old, remaining)
        try:
            self.store.get_verified(item, destination)
        finally:
            self.store.timeout_seconds = old


class _VerifiedRecoverySource:
    def __init__(self, root: Path) -> None:
        self.root = root

    def fetch(
        self, item: ArtifactIdentity, destination: Path, *, deadline: float | None
    ) -> None:
        from .artifacts import _target

        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("recovery publication deadline expired")
        source = _target(self.root, item.relative_path)
        if source.is_symlink() or not source.is_file():
            raise ValueError("verified recovery cache changed")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)


def _cold_recovery_source(
    controller: Any,
    relay: RelayStore,
    worker: Any,
    receipt: Any,
    descriptor: Any,
    spec: Any,
    bundle: Any,
) -> _VerifiedRecoverySource:
    from .artifacts import _target, _verify_complete_run

    items = [
        ArtifactIdentity(item.relative_path, item.sha256, item.size_bytes)
        for item in relay.flattened_inventory(descriptor)
        if item.relative_path.startswith("run/")
    ]
    if not items or descriptor.checkpoint is None:
        raise ValueError("NO_DURABLE_CHECKPOINT: missing full run closure")
    cache = relay.scratch_root / f"recovery-{descriptor.descriptor_digest()}"
    cache.mkdir(mode=0o700, exist_ok=True)
    if cache.is_symlink() or cache.stat().st_mode & 0o077:
        raise ValueError("recovery cache must be a private real directory")
    require_storage(
        [
            check_storage(
                cache,
                projected_bytes=sum(item.size_bytes for item in items),
                projected_inodes=len(items) + 4,
            )
        ]
    )
    source = RelayArtifactSource(relay)
    deadline = time.monotonic() + controller.transfer_timeout
    for item in items:
        path = _target(cache, item.relative_path)
        if not path.exists():
            source.fetch(item, path, deadline=deadline)
    _verify_complete_run(
        cache,
        receipt,
        items,
        worker,
        spec,
        bundle,
        controller.store.metrics,
        recovery_frontier=descriptor.checkpoint.model_dump(mode="json"),
    )
    return _VerifiedRecoverySource(cache)


def import_commit_records(
    controller: Any, relay: RelayStore, descriptor: Any, *, verify_only: bool = False
) -> None:
    cursor_by_origin: dict[str, int] = {}
    inventory = {
        item.relative_path: item for item in relay.flattened_inventory(descriptor)
    }
    with tempfile.TemporaryDirectory(
        prefix=".relay-records-", dir=controller.root
    ) as directory:
        for index, item in enumerate(descriptor.outbox_pages):
            path = Path(directory) / f"{index}.json"
            relay.get_verified(item.artifact, path)
            page = _strict_json_loads(path.read_bytes())
            if not isinstance(page, dict) or set(page) != {"origin_id", "records"}:
                raise ValueError("invalid relay outbox page fields")
            origin = page["origin_id"]
            cursor = cursor_by_origin.get(origin, 0)
            if origin != item.origin_id or item.first_sequence != cursor + 1:
                raise ValueError("relay outbox declared range or origin mismatch")
            records = page["records"]
            if not isinstance(records, list) or not records:
                raise ValueError("empty or invalid relay outbox page")
            for record in records:
                record = controller.store.metrics._validate_envelope(record)
                if (
                    record.get("origin_id") != origin
                    or record.get("sequence") != cursor + 1
                ):
                    raise ValueError(
                        "relay outbox prefix has a gap or origin substitution"
                    )
                if record.get("kind") == "manifest_reference":
                    from klab.training.metrics import (
                        _validate_manifest_reference,
                        manifest_reference,
                    )

                    payload = record["payload"]
                    _validate_manifest_reference(payload)
                    prefix = (
                        "run/"
                        if record["run_id"] == descriptor.run_id
                        else f"history/{record['run_id']}/run/"
                    )
                    member = inventory.get(prefix + "manifest.json")
                    if member is None:
                        raise ValueError(
                            "relay manifest reference lacks its immutable closure"
                        )
                    manifest_path = Path(directory) / f"manifest-{member.sha256}.json"
                    if not manifest_path.exists():
                        relay.get_verified(member, manifest_path)
                    manifest = read_manifest(manifest_path)
                    artifact_digest = hashlib.sha256(
                        canonical_json(manifest)
                    ).hexdigest()
                    if payload != manifest_reference(artifact_digest, manifest):
                        raise ValueError(
                            "relay manifest reference differs from its artifact"
                        )
                if record.get("kind") == "checkpoint":
                    run_id = record.get("run_id")
                    prefix = (
                        "run/"
                        if run_id == descriptor.run_id
                        else f"history/{run_id}/run/"
                    )
                    payload = record.get("payload", {})
                    path = (
                        prefix
                        + "checkpoints/"
                        + str(payload.get("relative_path"))
                        + "/manifest.json"
                    )
                    member = inventory.get(path)
                    if member is None:
                        raise ValueError(
                            "relay outbox checkpoint lacks its immutable closure"
                        )
                    checkpoint_manifest = (
                        Path(directory) / f"checkpoint-{member.sha256}.json"
                    )
                    if not checkpoint_manifest.exists():
                        relay.get_verified(member, checkpoint_manifest)
                    header = _strict_json_loads(checkpoint_manifest.read_bytes())
                    if not isinstance(header, dict):
                        raise ValueError("relay checkpoint manifest must be an object")
                    if header.get("sha256") != payload.get("digest"):
                        raise ValueError(
                            "relay checkpoint semantic digest differs from its generation"
                        )
                cursor += 1
            if item.last_sequence != cursor:
                raise ValueError("relay outbox cursor mismatch")
            cursor_by_origin[origin] = cursor
            if not verify_only:
                controller.store.import_records(records)


def verify_commit_assignment(
    controller: Any, attempt: dict[str, Any], worker: Any, descriptor: Any
) -> None:
    from .models import AttemptReceipt
    from .relay import assigned_worker

    original = assigned_worker(controller.root, attempt["attempt_id"])
    if original.worker_id != attempt["worker_id"]:
        raise ValueError("relay assignment worker differs from durable attempt")
    spec = controller._model("ExperimentSpec", attempt["spec"])
    instance = (
        original.colab.instance_id
        if original.colab is not None
        else original.instance_id
    )
    if instance is None:
        raise ValueError("assigned relay worker lacks an authenticated instance nonce")
    expected = {
        "attempt_id": attempt["attempt_id"],
        "run_id": attempt["run_id"],
        "experiment_id": attempt["experiment_id"],
        "worker_id": attempt["worker_id"],
        "spec_digest": spec.digest(),
        "bundle_digest": spec.dispatch_bundle_digest,
        "source_digest": spec.source_identity_sha256,
        "instance_id": instance,
    }
    if any(getattr(descriptor, field) != value for field, value in expected.items()):
        raise ValueError("relay descriptor differs from assigned identity")
    receipt = AttemptReceipt.model_validate(descriptor.receipt)
    if any(
        getattr(receipt, field) != expected[field]
        for field in (
            "attempt_id",
            "run_id",
            "experiment_id",
            "worker_id",
            "spec_digest",
            "bundle_digest",
        )
    ):
        raise ValueError("relay original receipt differs from assigned identity")


def collect_relay(
    controller: Any,
    run_id: str,
    profile: RelayProfile | RelayBinding,
    *,
    recover: bool = False,
    confirm_worker_lost: bool = False,
    flush: bool = True,
) -> dict[str, Any]:
    from .artifacts import ingest_attempt_artifacts
    from .models import AttemptReceipt
    from .transport import call_worker

    attempt = controller.store.attempt_by_run(run_id)
    if attempt is None:
        raise KeyError(f"unknown run: {run_id}")
    if confirm_worker_lost and not recover:
        raise ValueError("--confirm-worker-lost requires --recover")
    from .relay import assigned_worker

    registered = controller._worker_for_attempt(attempt)
    worker = assigned_worker(controller.root, attempt["attempt_id"])
    binding = profile.binding() if isinstance(profile, RelayProfile) else profile
    if (
        worker.relay is None
        or worker.relay.namespace != binding.namespace
        or worker.relay.worker != binding.worker
    ):
        raise ValueError("relay profile differs from assigned worker binding")
    key = read_attempt_key(
        controller.root / ".relay" / attempt["attempt_id"] / "relay-key.bin"
    )
    relay = RelayStore(binding, scratch_root=controller.root / ".relay" / "scratch")
    # Collection may repair only a finalized transfer, never issue optimizer execution.
    if (
        flush
        and registered is not None
        and registered == worker
        and attempt.get("terminal_receipt") is not None
        and attempt.get("recovery_observation") is None
    ):
        try:
            call_worker(
                worker,
                "relay_flush",
                {"attempt_id": attempt["attempt_id"]},
                timeout=controller.transfer_timeout,
            )
        except OSError, TimeoutError, ValueError, RuntimeError:
            pass  # The immutable relay remains usable when the VM is unreachable.
    chain = relay.verified_chain(attempt["attempt_id"], key=key)
    if not chain:
        controller.store.mark_unknown_if_nonterminal(
            attempt["attempt_id"], "NO_DURABLE_CHECKPOINT"
        )
        raise ValueError("NO_DURABLE_CHECKPOINT: no authenticated relay commit")
    spec = controller._model("ExperimentSpec", attempt["spec"])
    instance = (
        worker.colab.instance_id if worker.colab is not None else worker.instance_id
    )
    for commit in chain:
        verify_commit_assignment(controller, attempt, worker, commit.descriptor)
    selected = chain[-1]
    descriptor = selected.descriptor
    receipt = AttemptReceipt.model_validate(descriptor.receipt)
    terminal = receipt.state in {"COMPLETE", "FAILED", "INTERRUPTED", "UNKNOWN"}
    recovery = attempt.get("recovery_observation")
    _, bundle = controller._dispatch_bundle(spec)
    recovery_source = None
    if recovery is not None:
        selected = next(
            (commit for commit in chain if commit.digest == recovery["commit_sha256"]),
            None,
        )
        if selected is None:
            raise ValueError("confirmed recovery commit is unavailable")
        if chain[-1].digest != selected.digest:
            controller.store.retain_recovery_conflict(
                attempt["attempt_id"], chain[-1].descriptor.model_dump(mode="json")
            )
            controller.store.mark_ingestion_error(
                attempt["attempt_id"],
                ValueError("CONFLICT: new relay evidence after confirmed recovery"),
            )
            raise ValueError("CONFLICT: new relay evidence after confirmed recovery")
        descriptor = selected.descriptor
        receipt = AttemptReceipt.model_validate(descriptor.receipt)
        terminal = False
    elif not terminal:
        if not recover or not confirm_worker_lost:
            raise ValueError(
                "nonterminal relay requires --recover --confirm-worker-lost; worker may still be running"
            )
        selected = next(
            (
                commit
                for commit in reversed(chain)
                if commit.descriptor.checkpoint is not None
            ),
            None,
        )
        if selected is None:
            controller.store.mark_unknown_if_nonterminal(
                attempt["attempt_id"], "NO_DURABLE_CHECKPOINT"
            )
            raise ValueError("NO_DURABLE_CHECKPOINT: no full verified generation")
        descriptor = selected.descriptor
        receipt = AttemptReceipt.model_validate(descriptor.receipt)
        import_commit_records(controller, relay, descriptor, verify_only=True)
        recovery_source = _cold_recovery_source(
            controller, relay, worker, receipt, descriptor, spec, bundle
        )
        controller._reconcile_receipt(attempt, receipt.model_dump(mode="json"))
        recovery = {
            "reason": "HOSTED_WORKER_LOST",
            "commit_sha256": selected.digest,
            "confirmed_at": datetime.now(UTC).isoformat(),
            "instance_id": instance,
        }
        controller.store.record_recovery(
            attempt["attempt_id"],
            recovery,
            expected_worker_id=worker.worker_id,
            expected_spec_digest=spec.digest(),
            expected_bundle_digest=spec.dispatch_bundle_digest,
        )
    elif recover:
        raise ValueError(
            "terminal receipt requires ordinary collection, not fabricated loss recovery"
        )
    if terminal:
        controller._reconcile_receipt(attempt, receipt.model_dump(mode="json"))
    import_commit_records(controller, relay, descriptor)
    inventory = [
        ArtifactIdentity(item.relative_path, item.sha256, item.size_bytes)
        for item in relay.flattened_inventory(descriptor)
    ]
    run_items = [item for item in inventory if item.relative_path.startswith("run/")]
    if not run_items:
        if recovery is not None:
            raise ValueError("NO_DURABLE_CHECKPOINT: missing run closure")
        controller.store.mark_ingestion_not_required(attempt["attempt_id"])
    else:
        ingest_attempt_artifacts(
            worker,
            receipt,
            controller.root,
            spec=spec,
            bundle=bundle,
            records=controller.store.metrics,
            timeout=controller.transfer_timeout,
            artifact_source=recovery_source or RelayArtifactSource(relay),
            recovery_frontier=descriptor.checkpoint.model_dump(mode="json")
            if recovery is not None and descriptor.checkpoint is not None
            else None,
            verified_inventory=inventory,
        )
        controller.store.mark_ingestion_complete(attempt["attempt_id"])
        if recovery_source is not None:
            shutil.rmtree(recovery_source.root)
    result = controller.store.attempt_by_run(run_id)
    return {
        **result,
        "relay_commit_sha256": selected.digest,
        "durable_checkpoint": None
        if descriptor.checkpoint is None
        else descriptor.checkpoint.model_dump(mode="json"),
        "split_brain_risk": "Operator confirmation fences dispatch; it does not prove provider termination."
        if recovery
        else None,
    }
