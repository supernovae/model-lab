"""Immutable manifest and artifact identity support."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from klab.address_hash import memory_addressing_identity, memory_schemes
from klab.canonical import canonical_digest, canonical_json

if TYPE_CHECKING:
    from klab.runtime import RuntimeInfo

from klab.canonical import reject_duplicate_keys

MANIFEST_VERSION = 1
IDENTITY_VERSION = "run-identity-v2"
ARCHITECTURE_VERSION = "klab-decoder-v1"
_MACHINE_LOCAL_PATH_KEYS = frozenset(
    {
        "path",
        "cache_dir",
        "root_dir",
        "output_dir",
        "memory_package_path",
        "train_path",
        "validation_path",
        "prepared_path",
        "corpus_release_path",
        "corpus_export_path",
    }
)


def _without_machine_local_paths(value: object) -> object:
    if isinstance(value, Mapping):
        # An explicit tokenizer origin binds declaration bytes, not their location.
        if "mixture_declaration_sha256" in value:
            value = {
                key: item
                for key, item in value.items()
                if key != "mixture_declaration_path"
            }
        return {
            str(key): _without_machine_local_paths(item)
            for key, item in value.items()
            if key not in _MACHINE_LOCAL_PATH_KEYS
        }
    if isinstance(value, list):
        return [_without_machine_local_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_without_machine_local_paths(item) for item in value]
    return value


def _without_legacy_final_injection(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    model = normalized.get("model")
    if isinstance(model, Mapping) and model.get("memory_injection") == "final":
        model_identity = dict(model)
        model_identity.pop("memory_injection", None)
        normalized["model"] = model_identity
    return normalized


def architecture_sha256(
    config: Mapping[str, Any], *, memory_addressing: bool = True
) -> str:
    """Digest model semantics, excluding the separately inventoried package location.

    Memory tables whose address scheme changed (``klab.address_hash``) also
    bind that scheme, so a checkpoint records the addressing it was trained
    with. ``memory_addressing=False`` reproduces the pre-fix digest.
    """
    model = config.get("model", config)
    if not isinstance(model, Mapping):
        raise TypeError("architecture identity requires a mapping-valued model")
    attention = config.get("attention", {})
    if not isinstance(attention, Mapping):
        raise TypeError("architecture identity requires a mapping-valued attention")
    model_identity = dict(model)
    model_identity.pop("memory_package_path", None)
    if model_identity.get("memory_injection") == "final":
        model_identity.pop("memory_injection", None)
    identity: dict[str, object] = {
        "identity_version": IDENTITY_VERSION,
        "architecture_version": config.get(
            "architecture_version", ARCHITECTURE_VERSION
        ),
        "model": model_identity,
        "attention": dict(attention),
    }
    addressing = memory_addressing_identity(model_identity)
    if memory_addressing and addressing is not None:
        identity["memory_addressing"] = addressing
    return canonical_digest(identity)


def trained_memory_addressing(
    effective_config: Mapping[str, Any], recorded_architecture_sha256: object
) -> dict[str, object]:
    """Address schemes a run was trained with, from its recorded architecture digest.

    The digest was written at training time, so it tells current code from
    pre-fix code at affected table sizes. Anything else is unknown and refused.
    """
    model = effective_config.get("model", effective_config)
    if not isinstance(model, Mapping):
        raise TypeError("architecture identity requires a mapping-valued model")
    if recorded_architecture_sha256 == architecture_sha256(effective_config):
        return memory_schemes(model)
    if recorded_architecture_sha256 == architecture_sha256(
        effective_config, memory_addressing=False
    ):
        return memory_schemes(model, legacy=True)
    raise ValueError("recorded architecture digest names no known memory addressing")


def config_sha256(config: Mapping[str, Any]) -> str:
    """Digest scientific configuration while excluding machine-local locations."""
    # The new generic source binds content through the authenticated snapshot.
    # Preserve historical local_stories identities, which included this field.
    dataset = config.get("dataset")
    if isinstance(dataset, Mapping) and dataset.get("source") == "snapshot":
        config = {
            **config,
            "dataset": {
                key: value
                for key, value in dataset.items()
                if key != "source_manifest_path"
            },
        }
    tokenizer = config.get("tokenizer")
    if (
        isinstance(tokenizer, Mapping)
        and isinstance(tokenizer.get("origin"), Mapping)
        and isinstance(tokenizer["origin"].get("dataset"), Mapping)
    ):
        origin = tokenizer["origin"]
        config = {
            **config,
            "tokenizer": {
                **tokenizer,
                "origin": {
                    **origin,
                    "dataset": {
                        key: value
                        for key, value in origin["dataset"].items()
                        if key != "source_manifest_path"
                    },
                },
            },
        }
    return canonical_digest(
        {
            "identity_version": IDENTITY_VERSION,
            "config": _without_machine_local_paths(
                _without_legacy_final_injection(config)
            ),
        }
    )


def checkpoint_selection(path: Path) -> Path:
    """Accept a run directory, or anything ``train --promote`` accepts."""
    pointer = path / "checkpoints" / "latest.json"
    return pointer if pointer.is_file() else path


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
    *,
    progress_phase: str | None = None,
) -> str:
    from klab.training.pilot_progress import current_pilot_progress

    emitter = current_pilot_progress()
    if progress_phase is None and emitter is not None:
        progress_phase = emitter.current_phase
    subject = None
    if (
        emitter is not None
        and progress_phase is not None
        and path.stat().st_size >= 64 * 1024 * 1024
    ):
        import secrets

        subject = secrets.token_hex(8)
    total = path.stat().st_size if subject is not None else 0
    digest = hashlib.sha256()
    counted = 0
    reported = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            if subject is not None:
                counted += len(chunk)
                if counted - reported >= 64 * 1024 * 1024 or counted == total:
                    from klab.training.pilot_progress import emit_pilot_progress

                    emit_pilot_progress(
                        "progress",
                        progress_phase,
                        counter="bytes",
                        value=counted,
                        total=total,
                        subject=subject,
                    )
                    reported = counted
    return digest.hexdigest()


def source_identity() -> dict[str, object]:
    """Hash package implementation, excluding its self-referential lifecycle record."""
    root = Path(__file__).parents[1]
    excluded = {"research/resources/lifecycle.json"}
    inventory: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not (
            path.is_file()
            and path.suffix in {".py", ".json", ".yaml", ".yml", ".md"}
            and "__pycache__" not in path.parts
        ):
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        inventory.append((relative, sha256_file(path)))
    return {
        "algorithm": "package-path-sha256-v2",
        "files": inventory,
        "sha256": canonical_digest(inventory),
    }


@dataclass(frozen=True)
class ArtifactIdentity:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    name: str
    runtime: RuntimeInfo
    requested_config: dict[str, object]
    effective_config: dict[str, object]
    architecture_sha256: str
    source_identity: dict[str, object]
    worker_id: str
    continuation_kind: str = "FRESH"
    experiment_id: str | None = None
    attempt_id: str | None = None
    parent_run_id: str | None = None
    checkpoint_sha256: str | None = None
    purpose: str = "training"
    manifest_version: int = MANIFEST_VERSION
    architecture_version: str = ARCHITECTURE_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_commit: str | None = None
    git_dirty: bool | None = None
    package_version: str | None = None
    python_version: str = field(default_factory=platform.python_version)
    artifacts: tuple[ArtifactIdentity, ...] = ()
    resource_decisions: tuple[dict[str, object], ...] = ()
    stage_bundle_sha256: str | None = None
    pilot_reports: tuple[dict[str, object], ...] = ()

    def payload(self) -> dict[str, object]:
        if self.manifest_version != MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest version: {self.manifest_version}")
        value = asdict(self)
        value["identity_version"] = IDENTITY_VERSION
        value["requested_config_sha256"] = config_sha256(self.requested_config)
        value["effective_config_sha256"] = config_sha256(self.effective_config)
        return value

    def digest(self) -> str:
        return canonical_digest(self.payload())


def write_manifest(path: Path, manifest: RunManifest) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = manifest.payload()
    data["sha256"] = canonical_digest(data)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json(data) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(data["sha256"])


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_current_identity(data: dict[str, Any]) -> None:
    if data.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("unsupported manifest architecture version")
    requested, effective = data.get("requested_config"), data.get("effective_config")
    if not isinstance(requested, Mapping) or not isinstance(effective, Mapping):
        raise TypeError(
            "manifest identity requires requested and effective config mappings"
        )
    if data.get("requested_config_sha256") != config_sha256(requested):
        raise ValueError("manifest requested config hash mismatch")
    if data.get("effective_config_sha256") != config_sha256(effective):
        raise ValueError("manifest effective config hash mismatch")
    expected_architecture = architecture_sha256(effective)
    if data.get("architecture_sha256") != expected_architecture:
        if data.get("architecture_sha256") == architecture_sha256(
            effective, memory_addressing=False
        ):
            raise ValueError(
                "run was trained with pre-fix degenerate memory addressing "
                "(table size shares a factor with 257); retrain it"
            )
        raise ValueError("manifest architecture hash mismatch")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("manifest artifacts must be an array")
    names: set[str] = set()
    for artifact in artifacts:
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("relative_path"), str)
            or not artifact["relative_path"]
            or Path(artifact["relative_path"]).is_absolute()
            or ".." in Path(artifact["relative_path"]).parts
            or artifact["relative_path"] in names
            or not _is_sha256(artifact.get("sha256"))
            or isinstance(artifact.get("size_bytes"), bool)
            or not isinstance(artifact.get("size_bytes"), int)
            or artifact["size_bytes"] < 0
        ):
            raise ValueError("manifest artifact inventory is invalid")
        names.add(artifact["relative_path"])


def read_manifest(path: Path) -> dict[str, Any]:
    """Read a supported, hash-verified manifest without reinterpreting v1 digests."""
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid manifest {path}: {error}") from error
    if not isinstance(data, dict):
        raise TypeError("manifest must be a JSON object")
    version = data.get("manifest_version")
    if type(version) is not int or version != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version: {version!r}")
    actual = data.pop("sha256", None)
    if not isinstance(actual, str) or canonical_digest(data) != actual:
        raise ValueError("manifest hash mismatch")
    identity_version = data.get("identity_version")
    if identity_version is not None and identity_version != IDENTITY_VERSION:
        raise ValueError(f"unsupported manifest identity version: {identity_version!r}")
    if identity_version == IDENTITY_VERSION:
        _validate_current_identity(data)
    return data
