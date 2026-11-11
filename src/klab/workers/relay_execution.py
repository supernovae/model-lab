"""Synchronous worker recovery boundaries; optimizer execution never owns a queue."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from klab.canonical import canonical_json
from klab.training.manifest import sha256_file
from klab.training.metrics import ExperimentStore
from klab.workers.relay import RelayStore, read_attempt_key
from klab.workers.relay_models import (
    RelayArtifactIdentity,
    RelayBinding,
    RelayCommitDescriptor,
)


def worker_binding(root: Path) -> RelayBinding | None:
    from .execution import _strict_json

    path = root / "relay-binding.json"
    if not path.exists():
        return None
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise PermissionError("relay binding must be a private real file")
    return RelayBinding.model_validate(_strict_json(path))


def configure_relay(definition: Any, payload: dict[str, Any]) -> dict[str, str]:
    from .execution import (
        _admission_lock,
        _atomic_json,
        _definition_value,
        _strict_json,
    )

    binding = RelayBinding.model_validate(payload["binding"])
    root = Path(_definition_value(definition, "root"))
    challenge = RelayArtifactIdentity.model_validate(payload["challenge"])
    with _admission_lock(definition):
        path = root / "relay-binding.json"
        if path.exists() and (
            path.is_symlink() or _strict_json(path) != binding.model_dump(mode="json")
        ):
            raise ValueError("conflicting relay binding")
        attempts = root / "attempts"
        for receipt_path in attempts.glob("*/receipt.json"):
            if _strict_json(receipt_path)["state"] in {"PREPARED", "RUNNING"}:
                raise ValueError("cannot configure relay with active attempts")
        with tempfile.TemporaryDirectory(prefix="relay-check-", dir=root) as temporary:
            relay = RelayStore(binding, role="worker", scratch_root=root / "scratch")
            relay.get_verified(challenge, Path(temporary) / "challenge")
        if not path.exists():
            _atomic_json(path, binding.model_dump(mode="json"))
            path.chmod(0o600)
        import secrets

        from .hosted_status import instance_identity
        from .leases import boot_identity

        if instance_identity(root) is None:
            _atomic_json(
                root / "hosted-instance.json",
                {
                    "instance_id": secrets.token_hex(32),
                    "boot_id": boot_identity(),
                },
            )
            (root / "hosted-instance.json").chmod(0o600)
    return {
        "binding_sha256": binding.digest(),
        "verified_challenge_sha256": challenge.sha256,
    }


def _identity(path: Path, relative: str) -> RelayArtifactIdentity:
    if path.is_symlink() or not path.is_file():
        raise ValueError("relay publication source is not a real file")
    return RelayArtifactIdentity(
        relative_path=relative, sha256=sha256_file(path), size_bytes=path.stat().st_size
    )


def publish_boundary(
    definition: Any, attempt_id: str, *, checkpoint: Any = None, terminal: bool = False
) -> dict[str, Any] | None:
    from .bundles import _cache_paths
    from .execution import (
        _atomic_json,
        _attempt_dir,
        _definition_value,
        _final_artifacts,
        _load_receipt,
        _strict_json,
    )
    from .hosted_status import instance_identity, write_hosted_status

    root = Path(_definition_value(definition, "root"))
    binding = worker_binding(root)
    if binding is None:
        return None
    directory = _attempt_dir(definition, attempt_id)
    key = read_attempt_key(directory / "relay-key.bin")
    receipt = _load_receipt(definition, attempt_id)
    relay = RelayStore(binding, role="worker", scratch_root=root / "scratch")
    state_path = directory / "relay-commit.json"
    remote_previous = relay.newest_verified_commit(attempt_id, key=key)
    previous = remote_previous.descriptor if remote_previous is not None else None
    if state_path.exists():
        local_previous = RelayCommitDescriptor.model_validate(_strict_json(state_path))
        if previous is None or local_previous.sequence > previous.sequence:
            raise ValueError("local relay frontier lacks a committed remote descriptor")
    if terminal and previous is not None and previous.receipt == receipt:
        write_hosted_status(definition, receipt)
        return {
            "commit_sha256": previous.descriptor_digest(),
            "sequence": previous.sequence,
        }
    inventory = (
        {item.relative_path: item for item in relay.flattened_inventory(previous)}
        if previous is not None
        else {}
    )
    # Immutable checkpoint objects stay in the inherited closure after local retention.
    live = receipt["artifacts"] if terminal else _final_artifacts(definition, receipt)
    sources: dict[str, Path] = {}
    historical = sorted((root / "attempts").glob("*/relay-commit.json"))
    if len(historical) > 4096:
        raise ValueError("historical relay closure exceeds attempt bound")
    for old_state in historical:
        if old_state.parent == directory:
            continue
        old_key = read_attempt_key(old_state.parent / "relay-key.bin")
        old_commit = relay.newest_verified_commit(old_state.parent.name, key=old_key)
        if old_commit is None or old_commit.descriptor.receipt["state"] not in {
            "COMPLETE",
            "FAILED",
            "INTERRUPTED",
            "UNKNOWN",
        }:
            raise ValueError(
                "historical relay transfer is incomplete; collect the previous attempt first"
            )
        prefix = f"history/{old_commit.descriptor.run_id}/"
        for item in relay.flattened_inventory(old_commit):
            if not item.relative_path.startswith("history/"):
                inherited = item.model_copy(
                    update={"relative_path": prefix + item.relative_path}
                )
                inventory[inherited.relative_path] = inherited
        metadata = directory / "relay-history" / f"{old_state.parent.name}.json"
        _atomic_json(metadata, old_commit.descriptor.model_dump(mode="json"))
        sources[prefix + "commit.json"] = metadata
    for raw in live:
        item = RelayArtifactIdentity.model_validate(raw)
        if item.relative_path.startswith("run/"):
            sources[item.relative_path] = (
                root
                / "runs"
                / receipt["run_id"]
                / item.relative_path.removeprefix("run/")
            )
        elif item.relative_path.startswith("logs/"):
            sources[item.relative_path] = directory / item.relative_path.removeprefix(
                "logs/"
            )
        else:
            raise ValueError("relay inventory source outside owned run")
        inventory[item.relative_path] = item
    sources["dispatch/spec.json"] = directory / "spec.json"
    _, bundle_root = _cache_paths(root, receipt["bundle_digest"])
    if (bundle_root / "bundle.json").is_file():
        sources["dispatch/bundle.json"] = bundle_root / "bundle.json"
    for relative, source in sources.items():
        item = _identity(source, relative)
        if (
            terminal
            and relative.startswith(("run/", "logs/"))
            and item != inventory[relative]
        ):
            raise ValueError("finalized relay inventory changed after terminal receipt")
        inventory[relative] = item
        relay.put_verified(source, item)
    records = ExperimentStore(root / "runs")
    from typing import cast

    pages = []
    cursor = 0
    with tempfile.TemporaryDirectory(
        prefix="relay-records-", dir=directory
    ) as temporary:
        while True:
            exported = records.export_records(cursor, 1000, 4 * 1024 * 1024)
            next_cursor = cast(int, exported["next_sequence"])
            for envelope in cast(list[dict], exported["records"]):
                if envelope["kind"] != "checkpoint":
                    continue
                prefix = (
                    "run/"
                    if envelope["run_id"] == receipt["run_id"]
                    else f"history/{envelope['run_id']}/run/"
                )
                path = (
                    prefix
                    + "checkpoints/"
                    + envelope["payload"]["relative_path"]
                    + "/manifest.json"
                )
                member = inventory.get(path)
                if member is None:
                    raise ValueError(
                        "outbox checkpoint lacks its committed immutable closure"
                    )
                checkpoint_manifest = (
                    Path(temporary) / f"checkpoint-{member.sha256}.json"
                )
                if not checkpoint_manifest.exists():
                    relay.get_verified(member, checkpoint_manifest)
                header = _strict_json(checkpoint_manifest)
                if header.get("sha256") != envelope["payload"]["digest"]:
                    raise ValueError(
                        "outbox checkpoint semantic digest differs from its generation"
                    )
            if exported["records"]:
                path = Path(temporary) / f"{cursor}.json"
                path.write_bytes(
                    canonical_json(
                        {
                            "origin_id": exported["origin_id"],
                            "records": exported["records"],
                        }
                    )
                )
                item = _identity(
                    path,
                    f"outbox/{exported['origin_id']}/{cursor + 1}-{exported['next_sequence']}.json",
                )
                relay.put_verified(path, item)
                pages.append(
                    {
                        "artifact": item.model_dump(mode="json"),
                        "origin_id": exported["origin_id"],
                        "first_sequence": cursor + 1,
                        "last_sequence": next_cursor,
                    }
                )
            if not exported["has_more"]:
                break
            if next_cursor <= cursor:
                raise ValueError("outbox export did not advance")
            cursor = next_cursor
    frontier = previous.checkpoint if previous is not None else None
    if terminal and checkpoint is None:
        from klab.training.checkpoints import CheckpointManager

        run = root / "runs" / receipt["run_id"]
        if run.exists():
            checkpoint = CheckpointManager(run).recovery_report().record
    if checkpoint is not None:
        from klab.training.checkpoints import CheckpointManager

        run = root / "runs" / receipt["run_id"]
        generation = run / "checkpoints" / checkpoint.relative_path
        verified = CheckpointManager(run).verify(
            generation, require_training_state=True
        )
        if not verified.valid:
            raise ValueError(f"relay checkpoint verification failed: {verified.errors}")
        frontier = {
            name: getattr(checkpoint, name)
            for name in (
                "generation_id",
                "relative_path",
                "manifest_sha256",
                "step",
                "tokens_seen",
            )
        }
        frontier["generation_id"] = str(checkpoint.generation_id)
    spec = _strict_json(directory / "spec.json")
    instance = instance_identity(root)
    if instance is None:
        raise ValueError("relay worker lacks a boot-bound instance nonce")
    descriptor = relay.build_commit(
        key=key,
        sequence=0 if previous is None else previous.sequence + 1,
        previous_commit_sha256=None
        if previous is None
        else previous.descriptor_digest(),
        worker_id=receipt["worker_id"],
        attempt_id=attempt_id,
        run_id=receipt["run_id"],
        experiment_id=receipt["experiment_id"],
        spec_digest=receipt["spec_digest"],
        bundle_digest=receipt["bundle_digest"],
        source_digest=spec["source_identity_sha256"],
        instance_id=instance,
        receipt=receipt,
        checkpoint=frontier,
        inventory=list(inventory.values()),
        outbox_pages=pages,
    )
    relay.publish_commit(descriptor, key=key)
    _atomic_json(state_path, descriptor.model_dump(mode="json"))
    write_hosted_status(definition, receipt)
    return {
        "commit_sha256": descriptor.descriptor_digest(),
        "sequence": descriptor.sequence,
    }


def flush_terminal(definition: Any, attempt_id: str) -> dict[str, Any]:
    from .execution import _attempt_dir, _claim_file, _load_receipt

    receipt = _load_receipt(definition, attempt_id)
    if receipt["state"] not in {"COMPLETE", "FAILED", "INTERRUPTED", "UNKNOWN"}:
        raise ValueError("cannot flush an active attempt")
    claim = _claim_file(_attempt_dir(definition, attempt_id))
    if claim is None:
        raise ValueError("cannot flush a mutating attempt")
    try:
        result = publish_boundary(definition, attempt_id, terminal=True)
        if result is None:
            raise ValueError("attempt has no relay binding")
        return result
    finally:
        claim.close()
