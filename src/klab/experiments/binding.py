"""Select and bind one verified checkpoint generation before child dispatch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from klab.canonical import canonical_digest, canonical_json
from klab.training.checkpoints import CheckpointManager

if TYPE_CHECKING:
    from klab.experiments.lock import ResolvedCell, ResolvedExperimentPlan
    from klab.runtime_profile import RuntimeProfile
    from klab.workers.controller import Controller


def _runtime_payload(
    lock: ResolvedExperimentPlan,
    cell: ResolvedCell,
    *,
    profile: RuntimeProfile | None = None,
    worker: str | None = None,
    controller: Controller | None = None,
) -> dict[str, Any]:
    from klab.experiments.source_compatibility import source_identities_compatible
    from klab.runtime import validate_runtime
    from klab.runtime_identity_probe import (
        source_identity as runtime_source_identity,
    )
    from klab.runtime_profile import authorize_profile
    from klab.training.manifest import config_sha256, source_identity

    if (profile is None) == (worker is None):
        raise ValueError("exactly one runtime profile or registered worker is required")
    if cell not in lock.cells:
        raise ValueError("runtime binding cell is not in the selected lock")
    if config_sha256(cell.config.model_dump(mode="json")) != cell.config_sha256:
        raise ValueError("runtime binding cell config identity changed")
    execution_source_sha256 = str(source_identity()["sha256"])
    if not source_identities_compatible(
        str(lock.source_identity["sha256"]), execution_source_sha256
    ):
        raise ValueError("locked source identity differs from current package")
    if profile is not None:
        sealed = authorize_profile(profile, cell.config)
        authorization = sealed.as_dict()
        descriptor = profile.model_dump(mode="json")
        kind = "profile"
        tested_runtime = authorization.get("tested_runtime")
        if cell.config.runtime.precision != "bf16" or tested_runtime is None:
            tested_runtime = validate_runtime(
                cell.config, authorization=sealed
            ).as_dict()
        memory = cell.config.runtime.memory
        required_features = {
            name
            for name, selected in (
                ("activation_checkpointing", memory.activation_checkpointing.enabled),
                ("activation_offload", memory.activation_offload.enabled),
            )
            if selected
        }
        if cell.config.runtime.precision not in tested_runtime[
            "tested_precisions"
        ] or not required_features.issubset(tested_runtime["tested_features"]):
            raise ValueError("requested runtime precision/features were not tested")
    else:
        if controller is None:
            raise ValueError("registered worker runtime binding requires a controller")
        definition, tested, authorization = controller.validate_registered_worker(
            worker, cell.config, source_sha256=execution_source_sha256
        )
        descriptor = definition.model_dump(mode="json")
        kind = "worker"
        tested_runtime = tested.runtime
    probe = authorization["probe"]
    if probe.get("source_sha256") != runtime_source_identity()["source_sha256"]:
        raise ValueError(
            "runtime interpreter source identity differs from current package"
        )
    return {
        "format": "klab-runtime-binding-v1",
        "plan_sha256": lock.plan_sha256,
        "scientific_sha256": lock.scientific_sha256,
        "cell_id": cell.id,
        "config_sha256": cell.config_sha256,
        "source_kind": kind,
        "descriptor": descriptor,
        "runtime_authorization": authorization,
        "tested_runtime": tested_runtime,
    }


def _runtime_digest(payload: dict[str, Any]) -> str:
    # Evidence timestamps and free-memory readings are not operational identity.
    def stable(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: stable(item)
                for key, item in value.items()
                if key
                not in {
                    "measured_at",
                    "validated_at",
                    "last_seen_at",
                    "receipt_sha256",
                    "system_available_bytes",
                    "device_free_bytes",
                    "device_driver_allocated_bytes",
                }
            }
        if isinstance(value, list):
            return [stable(item) for item in value]
        return value

    return canonical_digest(stable(payload))


def bind_runtime(
    lock: ResolvedExperimentPlan,
    cell: ResolvedCell,
    workspace: Path,
    *,
    profile: RuntimeProfile | None = None,
    worker: str | None = None,
    controller: Controller | None = None,
) -> Path:
    """Publish one immutable operational receipt after fresh selected-runtime validation."""
    payload = _runtime_payload(
        lock, cell, profile=profile, worker=worker, controller=controller
    )
    digest = _runtime_digest(payload)
    payload["binding_sha256"] = digest
    payload["receipt_sha256"] = canonical_digest(payload)
    path = (
        Path(workspace)
        / "runtime-bindings"
        / lock.plan_sha256
        / cell.id
        / f"{digest}.json"
    )
    encoded = canonical_json(payload) + b"\n"
    if path.is_symlink():
        raise ValueError("runtime binding receipt must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
    except FileExistsError:
        inspect_runtime_binding(path, lock, cell)
    return path


def inspect_runtime_binding(
    path: Path, lock: ResolvedExperimentPlan, cell: ResolvedCell
) -> dict[str, Any]:
    """Authenticate receipt bytes and locked cell before selecting an interpreter.

    Inspection does not authorize runtime execution; callers must reopen freshly.
    """
    from klab.training.manifest import config_sha256

    if cell not in lock.cells or cell.config_sha256 != config_sha256(
        cell.config.model_dump(mode="json")
    ):
        raise ValueError("runtime receipt does not belong to the verified lock cell")
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("runtime binding receipt is missing or symlinked")
    encoded = path.read_bytes()
    payload = json.loads(encoded)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("binding_sha256"), str
    ):
        raise ValueError("invalid runtime binding receipt payload")  # noqa: TRY004 - malformed serialized evidence
    digest = payload.pop("binding_sha256")
    receipt_hash = payload.pop("receipt_sha256", None)
    if (
        payload.get("format") != "klab-runtime-binding-v1"
        or digest != _runtime_digest(payload)
        or receipt_hash
        != hashlib.sha256(
            canonical_json({**payload, "binding_sha256": digest})
        ).hexdigest()
        or path.name != f"{digest}.json"
        or path.parent.name != cell.id
        or path.parent.parent.name != lock.plan_sha256
        or encoded
        != canonical_json(
            {**payload, "binding_sha256": digest, "receipt_sha256": receipt_hash}
        )
        + b"\n"
    ):
        raise ValueError("runtime binding receipt identity or bytes changed")
    if (
        payload.get("plan_sha256") != lock.plan_sha256
        or payload.get("scientific_sha256") != lock.scientific_sha256
        or payload.get("cell_id") != cell.id
        or payload.get("config_sha256") != cell.config_sha256
    ):
        raise ValueError("runtime binding receipt belongs to a different lock or cell")
    return {**payload, "binding_sha256": digest, "receipt_sha256": receipt_hash}


def open_runtime_binding(
    path: Path,
    lock: ResolvedExperimentPlan,
    cell: ResolvedCell,
    *,
    controller: Controller | None = None,
) -> dict[str, Any]:
    """Authenticate and freshly rederive capability; saved JSON is not authority."""
    from klab.runtime_profile import RuntimeProfile

    receipt = inspect_runtime_binding(path, lock, cell)
    payload = {
        key: value
        for key, value in receipt.items()
        if key not in {"binding_sha256", "receipt_sha256"}
    }
    digest = receipt["binding_sha256"]
    kind = payload.get("source_kind")
    descriptor = payload.get("descriptor")
    if kind == "profile":
        profile = RuntimeProfile.model_validate(descriptor)
        fresh = _runtime_payload(lock, cell, profile=profile)
    elif kind == "worker":
        if not isinstance(descriptor, dict):
            raise ValueError("runtime worker definition is missing")
        fresh = _runtime_payload(
            lock, cell, worker=descriptor["name"], controller=controller
        )
    else:
        raise ValueError("unknown runtime binding source")
    if _runtime_digest(fresh) != digest:
        raise ValueError(
            "bound runtime identity differs from freshly validated runtime"
        )
    return receipt


def select_generation(
    run: Path, *, selector: str, at_step: int | None, full_state: bool
) -> dict[str, Any]:
    """No mutable latest/best alias participates in a child identity."""
    if selector not in {"terminal", "best_validation"}:
        raise ValueError(f"unsupported parent selector: {selector}")
    manager = CheckpointManager(run)
    records, rejected = manager._verified_records(require_training_state=full_state)
    if rejected:
        raise ValueError(f"parent run contains invalid generations: {rejected}")
    if selector == "terminal":
        if at_step is None:
            raise ValueError("terminal parent needs a prespecified update")
        candidates = [item for item in records if item.step == at_step]
        selected = max(candidates, key=lambda item: item.generation_id, default=None)
    else:
        candidates = [item for item in records if item.validation_loss is not None]
        selected = min(
            candidates,
            key=lambda item: (float(item.validation_loss), item.generation_id),
            default=None,
        )
    if selected is None:
        raise ValueError(
            f"no verified {selector} checkpoint generation at step {at_step}"
        )
    path = run / "checkpoints" / selected.relative_path
    report = manager.verify(path, require_training_state=full_state)
    if not report.valid or path.is_symlink():
        raise ValueError(f"selected checkpoint generation is unverified: {path}")
    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": selected.manifest_sha256,
        "step": selected.step,
        "validation_loss": selected.validation_loss,
        "resume_level": report.resume_level,
        "run_id": run.name,
    }


def bind_generation(
    workspace: Path,
    *,
    plan_sha256: str,
    cell_id: str,
    parent_cell_id: str,
    parent_run: Path,
    selector: str,
    at_step: int | None,
    full_state: bool,
    read_only: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Exclusive immutable execution binding; replay verifies the original bytes."""
    selected = select_generation(
        parent_run, selector=selector, at_step=at_step, full_state=full_state
    )
    payload = {
        "format": "klab-plan-execution-binding-v1",
        "plan_sha256": plan_sha256,
        "cell_id": cell_id,
        "parent_cell_id": parent_cell_id,
        "selector": selector,
        "at_step": at_step,
        **selected,
    }
    encoded = canonical_json(payload) + b"\n"
    digest = canonical_digest(payload)
    binding = workspace / "bindings" / f"{digest}.json"
    if binding.is_symlink():
        raise ValueError(f"execution binding must not be a symlink: {binding}")
    if not read_only:
        binding.parent.mkdir(parents=True, exist_ok=True)
    if binding.exists():
        if binding.read_bytes() != encoded:
            raise ValueError(f"published execution binding changed: {binding}")
    elif read_only:
        raise ValueError(f"required execution binding is missing: {binding}")
    else:
        with binding.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
    return binding, payload


def read_binding(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "klab-plan-execution-binding-v1":
        raise ValueError(f"invalid phase binding: {path}")
    path = Path(payload["checkpoint_path"])
    if path.name in {"latest.json", "best.json"} or path.is_symlink():
        raise ValueError("execution binding must pin an immutable generation directory")
    report = CheckpointManager(path.parent.parent).verify(
        path, require_training_state=payload["resume_level"] == "full"
    )
    if not report.valid:
        raise ValueError("bound checkpoint generation no longer verifies")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("sha256") != payload["checkpoint_sha256"]
        or manifest.get("step") != payload["step"]
    ):
        raise ValueError("checkpoint digest or update differs from execution binding")
    return payload
