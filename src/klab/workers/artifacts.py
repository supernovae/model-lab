"""Known-inventory artifact access and atomic controller-side ingestion."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import RunConfig
from klab.training.checkpoints import (
    CheckpointManager,
    _fsync_directory,
    _safe_member,
)
from klab.training.manifest import (
    ArtifactIdentity,
    config_sha256,
    read_manifest,
    sha256_file,
)
from klab.training.metrics import ExperimentStore, _strict_json_loads
from klab.workers.models import AttemptReceipt, BundleManifest, ExperimentSpec
from klab.workers.transport import (
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS,
    MAX_TOTAL_ATTACHMENT_BYTES,
)

_MAX_CHUNK = 4 * 1024 * 1024


class ArtifactSource(Protocol):
    def fetch(
        self, item: ArtifactIdentity, destination: Path, *, deadline: float | None
    ) -> None: ...


class RPCArtifactSource:
    def __init__(
        self, worker: Any, receipt: AttemptReceipt, receive_root: Path, timeout: float
    ) -> None:
        self.worker = worker
        self.receipt = receipt
        self.receive_root = receive_root
        self.timeout = timeout

    def fetch(
        self, item: ArtifactIdentity, destination: Path, *, deadline: float | None
    ) -> None:
        _download(
            self.worker,
            self.receipt,
            item,
            destination,
            receive_root=self.receive_root,
            timeout=self.timeout,
            deadline=deadline,
        )


def _field(value: Any, name: str) -> Any:
    return getattr(value, name) if hasattr(value, name) else value[name]


_STREAM = 1024 * 1024


def _remaining_timeout(timeout: float, deadline: float | None) -> float:
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("controller tick deadline exhausted")
    return min(timeout, remaining)


def _receipt_items(receipt: Any) -> list[ArtifactIdentity]:
    items: list[ArtifactIdentity] = []
    names: set[str] = set()
    total_bytes = 0
    for value in _field(receipt, "artifacts"):
        item = (
            value
            if isinstance(value, ArtifactIdentity)
            else ArtifactIdentity(
                **(value.model_dump() if hasattr(value, "model_dump") else value)
            )
        )
        if (
            type(item.size_bytes) is not int
            or not 0 <= item.size_bytes <= MAX_ATTACHMENT_BYTES
        ):
            raise ValueError("receipt artifact size exceeds transfer bounds")
        total_bytes += item.size_bytes
        if len(items) >= MAX_ATTACHMENTS or total_bytes > MAX_TOTAL_ATTACHMENT_BYTES:
            raise ValueError("receipt inventory exceeds aggregate transfer bounds")
        if item.relative_path in names:
            raise ValueError("duplicate receipt artifact")
        names.add(item.relative_path)
        items.append(item)
    return items


def _source_for(worker_root: Path, receipt: Any, relative_path: str) -> Path:
    if relative_path.startswith("run/"):
        member = relative_path.removeprefix("run/")
        root = worker_root / "runs" / _field(receipt, "run_id")
    elif relative_path in {"logs/stdout.log", "logs/stderr.log"}:
        member = relative_path.removeprefix("logs/")
        root = worker_root / "attempts" / _field(receipt, "attempt_id")
    else:
        raise ValueError("artifact is outside the receipt inventory convention")
    path = _safe_member(root, member)
    if path is None or not path.is_file():
        raise ValueError("artifact source is unsafe or missing")
    return path


def artifact_inventory(worker_root: Path, receipt: Any) -> list[ArtifactIdentity]:
    """Verify every receipt-authorized final artifact before exposing it."""
    result = _receipt_items(receipt)
    for item in result:
        path = _source_for(worker_root.resolve(), receipt, item.relative_path)
        if path.stat().st_size != item.size_bytes or sha256_file(path) != item.sha256:
            raise ValueError(
                f"receipt artifact is incomplete or corrupt: {item.relative_path}"
            )
        if "/checkpoints/." in item.relative_path or item.relative_path.endswith(
            ".tmp"
        ):
            raise ValueError("temporary checkpoint files cannot be advertised")
    return result


def read_artifact_chunk(
    worker_root: Path,
    receipt: Any,
    relative_path: str,
    offset: int,
    max_bytes: int,
    destination: Path,
) -> tuple[dict[str, object], Path]:
    """Read a bounded verified segment from only a receipt-authorized file."""
    if (
        type(offset) is not int
        or offset < 0
        or type(max_bytes) is not int
        or not 1 <= max_bytes <= _MAX_CHUNK
    ):
        raise ValueError("invalid artifact offset or max_bytes")
    item = next(
        (
            value
            for value in _receipt_items(receipt)
            if value.relative_path == relative_path
        ),
        None,
    )
    if item is None:
        raise ValueError("artifact is not in receipt inventory")
    if offset > item.size_bytes:
        raise ValueError("artifact offset is beyond EOF")
    source = _source_for(worker_root.resolve(), receipt, relative_path)
    if source.stat().st_size != item.size_bytes:
        raise ValueError("artifact changed or truncated before reading")
    if destination.exists() and destination.is_dir():
        destination = destination / "artifact.chunk"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("artifact chunk destination exists")
    remaining = min(max_bytes, item.size_bytes - offset)
    digest = hashlib.sha256()
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        incoming.seek(offset)
        while remaining:
            block = incoming.read(min(_STREAM, remaining))
            if not block:
                raise ValueError("artifact changed or truncated while reading")
            outgoing.write(block)
            digest.update(block)
            remaining -= len(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    length = destination.stat().st_size
    return (
        {
            "file_sha256": item.sha256,
            "total_length": item.size_bytes,
            "offset": offset,
            "chunk_sha256": digest.hexdigest(),
            "next_offset": offset + length,
            "eof": offset + length == item.size_bytes,
        },
        destination,
    )


def _target(root: Path, relative: str) -> Path:
    if not relative.startswith("run/"):
        raise ValueError("controller only publishes run-owned artifacts")
    name = relative.removeprefix("run/")
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe controller artifact path")
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact target escapes run root")
    return target


def _verify_complete_run(
    root: Path,
    receipt: AttemptReceipt,
    items: list[ArtifactIdentity],
    worker: Any,
    spec: ExperimentSpec,
    bundle: BundleManifest,
    records: ExperimentStore,
    *,
    recovery_frontier: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[tuple[object, ...]]]:
    manifest = read_manifest(root / "manifest.json")
    manifest_digest = canonical_digest(manifest)
    for name in ("run_id", "attempt_id", "experiment_id"):
        if manifest.get(name) != _field(receipt, name):
            raise ValueError(f"ingested manifest {name} differs from receipt")
    requested = spec.config.model_dump(mode="json")
    if (
        manifest["requested_config"] != requested
        or manifest["requested_config_sha256"] != config_sha256(requested)
        or manifest["worker_id"] != receipt.worker_id
        or receipt.worker_id != worker.worker_id
        or receipt.spec_digest != spec.digest()
        or receipt.bundle_digest != bundle.digest()
        or bundle.digest() != spec.dispatch_bundle_digest
        or bundle.config.model_dump(mode="json") != requested
    ):
        raise ValueError("ingested run differs from dispatched specification")
    from klab.workers.execution import _effective_config

    expected_effective = _effective_config(worker, spec.config).model_dump(mode="json")
    if expected_effective["runtime"]["precision"] == "auto":
        expected_effective["runtime"]["precision"] = "fp32"
    if manifest["effective_config_sha256"] != config_sha256(expected_effective):
        raise ValueError("ingested effective configuration changes the experiment")
    if (
        not spec.continuation.allow_runtime_drift
        and manifest["source_identity"]["sha256"] != spec.source_identity_sha256
    ):
        raise ValueError("ingested source identity differs without authorized drift")
    for name in ("kind", "parent_run_id", "checkpoint_sha256"):
        key = "continuation_kind" if name == "kind" else name
        if manifest.get(key) != getattr(spec.continuation, name):
            raise ValueError("ingested continuation differs from dispatched lineage")
    dispatch = [
        decision
        for decision in manifest.get("resource_decisions", [])
        if isinstance(decision, dict) and decision.get("kind") == "worker_dispatch"
    ]
    if len(dispatch) != 1 or any(
        dispatch[0].get(name) != _field(receipt, name)
        for name in ("spec_digest", "bundle_digest")
    ):
        raise ValueError("ingested manifest dispatch identity differs from receipt")
    expected_matrix = (
        None if spec.matrix is None else spec.matrix.model_dump(mode="json")
    )
    if dispatch[0].get("matrix") != expected_matrix:
        raise ValueError("ingested matrix coordinate differs from dispatch")
    expected_plan = None if spec.plan is None else spec.plan.model_dump(mode="json")
    if dispatch[0].get("plan") != expected_plan:
        raise ValueError("ingested resolved plan identity differs from dispatch")
    resolved = root / "resolved_config.yaml"
    try:
        effective = _strict_json_loads(resolved.read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError(
            f"ingested resolved configuration is invalid: {error}"
        ) from error
    if (
        effective != manifest["effective_config"]
        or config_sha256(effective) != manifest["effective_config_sha256"]
    ):
        raise ValueError("ingested resolved configuration differs from manifest")
    effective_config = RunConfig.model_validate(effective)
    expected = {
        item.relative_path.removeprefix("run/"): item
        for item in items
        if item.relative_path.startswith("run/")
    }
    manifest_assets = manifest.get("artifacts")
    if not isinstance(manifest_assets, list):
        raise TypeError("ingested manifest asset inventory is invalid")
    for value in manifest_assets:
        declared = ArtifactIdentity(**value)
        received = expected.get(declared.relative_path)
        if (
            received is None
            or received.sha256 != declared.sha256
            or received.size_bytes != declared.size_bytes
        ):
            raise ValueError("receipt omits declared run manifest asset")
    for name, item in expected.items():
        target = _safe_member(root, name)
        if (
            target is None
            or not target.is_file()
            or target.stat().st_size != item.size_bytes
            or sha256_file(target) != item.sha256
        ):
            raise ValueError(f"ingested artifact integrity failure: {name}")
    for item in bundle.files:
        if not item.relative_path.startswith("assets/"):
            continue
        name = item.relative_path.removeprefix("assets/")
        if name not in {
            "tokenizer.json",
            "tokenizer_manifest.json",
            "portable_package",
        } and not name.startswith(("data/", "portable_package/", "corpus/")):
            continue
        received = expected.get(name)
        if received is None or (received.sha256, received.size_bytes) != (
            item.sha256,
            item.size_bytes,
        ):
            raise ValueError(f"ingested input differs from sealed dispatch: {name}")
    verified_checkpoints = []
    checkpoints = root / "checkpoints"
    if checkpoints.exists():
        manager = CheckpointManager(root, manifest_sha256=manifest_digest)
        for generation in checkpoints.glob("step_*_gen_*"):
            if not generation.is_dir() or generation.is_symlink():
                raise ValueError("unsafe checkpoint generation")
            report = manager.verify(
                generation,
                expected_manifest=manifest_digest,
                require_training_state=True,
                expected_config=effective_config,
            )
            if not report.valid:
                raise ValueError(
                    f"incomplete or invalid ingested checkpoint: {report.errors}"
                )
            raw = _strict_json_loads((generation / "manifest.json").read_bytes())
            record = manager._record_from_manifest(generation, raw)
            verified_checkpoints.append(
                (
                    str(record.generation_id),
                    record.relative_path,
                    record.manifest_sha256,
                    record.step,
                    record.tokens_seen,
                    record.created_at,
                    record.bytes,
                    record.validation_loss,
                    "verified",
                    report.resume_level,
                    record.backend,
                )
            )
        if (
            verified_checkpoints
            and receipt.state != "UNKNOWN"
            and recovery_frontier is None
            and not (checkpoints / "latest.json").is_file()
        ):
            raise ValueError("ingested checkpoints omit the latest pointer")
        for name in ("latest.json", "best.json"):
            pointer = checkpoints / name
            if pointer.exists():
                report = manager.verify(
                    pointer,
                    expected_manifest=manifest_digest,
                    require_training_state=True,
                    expected_config=effective_config,
                )
                if not report.valid:
                    raise ValueError(
                        f"invalid ingested checkpoint pointer: {report.errors}"
                    )
    if receipt.state in {"COMPLETE", "INTERRUPTED"}:
        progress = _strict_json_loads((root / "progress.json").read_bytes())
        pointer = _strict_json_loads((checkpoints / "latest.json").read_bytes())
        latest = next(
            (row for row in verified_checkpoints if row[1] == pointer["relative_path"]),
            None,
        )
        expected_status = "completed" if receipt.state == "COMPLETE" else "interrupted"
        if (
            latest is None
            or not isinstance(progress, dict)
            or progress.get("manifest_sha256") != manifest_digest
            or progress.get("status") != expected_status
            or (progress.get("step"), progress.get("tokens_seen")) != latest[3:5]
            or latest[3] > spec.config.training.max_steps
            or latest[4] > spec.config.training.max_tokens
        ):
            raise ValueError(
                "terminal receipt differs from committed training frontier"
            )
        if receipt.state == "COMPLETE" and (
            latest[3] < spec.config.training.max_steps
            and latest[4] < spec.config.training.max_tokens
        ):
            raise ValueError("completed receipt has not exhausted its declared budget")
    with records._connect() as con:
        projected_manifest = con.execute(
            "SELECT digest,json FROM manifests WHERE run_id=?", (receipt.run_id,)
        ).fetchall()
        projected_checkpoints = con.execute(
            "SELECT checkpoint_id,relative_path,digest,step,tokens_seen,created_at,"
            "size_bytes,validation_loss,verification_status,resume_level,backend "
            "FROM checkpoints WHERE run_id=?",
            (receipt.run_id,),
        ).fetchall()
    from klab.training.metrics import manifest_projection_matches

    if projected_manifest:
        if (
            len(projected_manifest) != 1
            or projected_manifest[0][0] != manifest_digest
            or not manifest_projection_matches(
                _strict_json_loads(projected_manifest[0][1]), manifest_digest, manifest
            )
        ):
            raise ValueError("replicated manifest differs from verified run artifacts")
    elif receipt.state != "UNKNOWN" and recovery_frontier is None:
        raise ValueError("replicated manifest is absent")
    expected_checkpoints = {row[0]: row for row in verified_checkpoints}
    if any(
        expected_checkpoints.get(row[0]) != row for row in projected_checkpoints
    ) or (
        receipt.state != "UNKNOWN"
        and recovery_frontier is None
        and len(projected_checkpoints) != len(verified_checkpoints)
    ):
        raise ValueError(
            "replicated checkpoint metadata differs from verified generations"
        )
    if recovery_frontier is not None:
        selected = next(
            (
                row
                for row in verified_checkpoints
                if row[0] == recovery_frontier["generation_id"]
            ),
            None,
        )
        if selected is None or (selected[1], selected[2], selected[3], selected[4]) != (
            recovery_frontier["relative_path"],
            recovery_frontier["manifest_sha256"],
            recovery_frontier["step"],
            recovery_frontier["tokens_seen"],
        ):
            raise ValueError("recovery frontier differs from full verified checkpoint")
        if (
            selected[3] > spec.config.training.max_steps
            or selected[4] > spec.config.training.max_tokens
        ):
            raise ValueError("recovery counters exceed declared budget")
    return manifest, verified_checkpoints


def _restore_crash_projection(
    records: ExperimentStore,
    receipt: AttemptReceipt,
    verified: tuple[dict[str, Any], list[tuple[object, ...]]],
    *,
    recovery: bool = False,
) -> None:
    """Resolve manifest references after publication; recover absent crash metadata."""
    manifest, checkpoints = verified
    encoded = canonical_json(manifest)
    digest = hashlib.sha256(encoded).hexdigest()
    with records._connect() as con:
        present = con.execute(
            "SELECT 1 FROM manifests WHERE run_id=?", (receipt.run_id,)
        ).fetchone()
    if present is not None:
        records.materialize_manifest(receipt.run_id, digest, manifest)
    if receipt.state != "UNKNOWN" and not recovery:
        return
    with records._connect() as con:
        con.execute(
            "INSERT OR IGNORE INTO manifests(run_id,digest,json) VALUES(?,?,?)",
            (
                receipt.run_id,
                hashlib.sha256(encoded).hexdigest(),
                encoded.decode("utf-8"),
            ),
        )
        for row in checkpoints:
            con.execute(
                "INSERT OR IGNORE INTO checkpoints("
                "run_id,checkpoint_id,relative_path,digest,step,tokens_seen,created_at,"
                "size_bytes,validation_loss,verification_status,resume_level,backend,"
                "verified_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))",
                (receipt.run_id, *row),
            )


def _download(
    worker: Any,
    receipt: Any,
    item: ArtifactIdentity,
    destination: Path,
    *,
    receive_root: Path,
    timeout: float,
    deadline: float | None = None,
) -> None:
    from klab.workers.transport import call_worker

    offset = 0
    digest = hashlib.sha256()
    with destination.open("xb") as outgoing:
        while offset < item.size_bytes:
            with tempfile.TemporaryDirectory(
                prefix=".artifact-transfer.", dir=receive_root
            ) as receive_dir:
                reply = call_worker(
                    worker,
                    "artifact",
                    {
                        "attempt_id": _field(receipt, "attempt_id"),
                        "relative_path": item.relative_path,
                        "offset": offset,
                        "max_bytes": _MAX_CHUNK,
                    },
                    receive_dir=Path(receive_dir),
                    timeout=_remaining_timeout(timeout, deadline),
                )
                result = reply.result
                attachment = next(iter(reply.attachments.values()), None)
                if attachment is None:
                    raise ValueError("artifact response omitted chunk attachment")
                if (
                    result.get("file_sha256") != item.sha256
                    or result.get("total_length") != item.size_bytes
                    or result.get("offset") != offset
                    or result.get("next_offset") is None
                    or not isinstance(result.get("chunk_sha256"), str)
                    or attachment.stat().st_size != result["next_offset"] - offset
                    or sha256_file(attachment) != result["chunk_sha256"]
                ):
                    raise ValueError("artifact chunk metadata or digest mismatch")
                with attachment.open("rb") as incoming:
                    while block := incoming.read(_STREAM):
                        outgoing.write(block)
                        digest.update(block)
                next_offset = result["next_offset"]
            if (
                type(next_offset) is not int
                or next_offset <= offset
                or next_offset > item.size_bytes
            ):
                raise ValueError("invalid artifact chunk progress")
            offset = next_offset
            if bool(result.get("eof")) != (offset == item.size_bytes):
                raise ValueError("artifact EOF marker mismatch")
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if (
        digest.hexdigest() != item.sha256
        or destination.stat().st_size != item.size_bytes
    ):
        raise ValueError("whole artifact digest mismatch")


def ingest_attempt_artifacts(
    worker: Any,
    receipt: AttemptReceipt,
    controller_root: Path,
    *,
    spec: ExperimentSpec,
    bundle: BundleManifest,
    records: ExperimentStore,
    timeout: float = 1800,
    deadline: float | None = None,
    artifact_source: ArtifactSource | None = None,
    recovery_frontier: dict[str, Any] | None = None,
    verified_inventory: list[ArtifactIdentity] | None = None,
) -> dict[str, object]:
    """Fetch and atomically publish an entire verified run inventory.

    The existing publication is accepted only when every requested byte is identical.
    No half-downloaded checkpoint directory ever becomes visible under ``run_id``.
    """
    items = _receipt_items(
        receipt if verified_inventory is None else {"artifacts": verified_inventory}
    )
    receipt = AttemptReceipt.model_validate(receipt)
    if recovery_frontier is not None:
        with records._connect() as con:
            recovered = con.execute(
                "SELECT recovery_observation_json FROM attempts WHERE attempt_id=?",
                (receipt.attempt_id,),
            ).fetchone()
        if recovered is None or recovered[0] is None:
            raise ValueError(
                "recovery ingestion requires a confirmed durable observation"
            )
    elif verified_inventory is not None:
        if receipt.state not in {"COMPLETE", "FAILED", "INTERRUPTED", "UNKNOWN"}:
            raise ValueError(
                "inventory override requires authenticated terminal or recovery evidence"
            )
        inventory_by_path = {item.relative_path: item for item in items}
        if any(
            inventory_by_path.get(item.relative_path) != item
            for item in _receipt_items(receipt)
        ):
            raise ValueError("verified relay inventory differs from terminal receipt")
    run_items = [item for item in items if item.relative_path.startswith("run/")]
    if not run_items:
        raise ValueError("receipt has no run-owned artifacts to ingest")
    controller_root = controller_root.resolve()
    run_id = _field(receipt, "run_id")
    target = controller_root / run_id
    if target.exists():
        verified = _verify_complete_run(
            target,
            receipt,
            run_items,
            worker,
            spec,
            bundle,
            records,
            recovery_frontier=recovery_frontier,
        )
        _restore_crash_projection(
            records, receipt, verified, recovery=recovery_frontier is not None
        )
        return {
            "run_id": run_id,
            "published": False,
            "artifacts": len(run_items),
            "status": "identical",
        }
    required_bytes = sum(item.size_bytes for item in run_items)
    controller_root.mkdir(parents=True, exist_ok=True)
    disk = os.statvfs(controller_root)
    if disk.f_bavail * disk.f_frsize < required_bytes:
        raise OSError("insufficient free disk for atomic artifact publication")
    with tempfile.TemporaryDirectory(
        prefix=f".{run_id}.", dir=controller_root
    ) as temporary:
        staging = Path(temporary)
        source = artifact_source or RPCArtifactSource(
            worker, receipt, controller_root, timeout
        )
        for item in run_items:
            path = _target(staging, item.relative_path)
            source.fetch(item, path, deadline=deadline)
        verified = _verify_complete_run(
            staging,
            receipt,
            run_items,
            worker,
            spec,
            bundle,
            records,
            recovery_frontier=recovery_frontier,
        )
        _fsync_directory(staging)
        try:
            os.rename(staging, target)
        except FileExistsError:
            verified = _verify_complete_run(
                target,
                receipt,
                run_items,
                worker,
                spec,
                bundle,
                records,
                recovery_frontier=recovery_frontier,
            )
            _restore_crash_projection(
                records, receipt, verified, recovery=recovery_frontier is not None
            )
            return {
                "run_id": run_id,
                "published": False,
                "artifacts": len(run_items),
                "status": "identical",
            }
        _fsync_directory(controller_root)
    _restore_crash_projection(
        records, receipt, verified, recovery=recovery_frontier is not None
    )
    return {
        "run_id": run_id,
        "published": True,
        "artifacts": len(run_items),
        "status": "published",
    }
