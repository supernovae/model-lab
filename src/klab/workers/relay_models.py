"""Strict operational contracts for the hosted artifact relay.

Relay configuration is deliberately outside scientific run configuration.  It names
storage endpoints, never credentials or provider SDK settings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from klab.canonical import canonical_digest, reject_duplicate_keys
from klab.config.models import StrictModel

_RELAY_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _exact_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _exact_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")  # noqa: TRY004 — Pydantic validation boundary
    return value


class RelayArtifactIdentity(StrictModel):
    """JSON-native artifact identity, kept independent of worker model imports."""

    relative_path: str
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        _exact_string(value, "relative_path")
        path = Path(value)
        if (
            not value
            or value == "."
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or "\x00" in value
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("relay artifact path is unsafe")
        return value

    @field_validator("sha256")
    @classmethod
    def valid_digest(cls, value: str) -> str:
        _exact_string(value, "sha256")
        if not _SHA256.fullmatch(value):
            raise ValueError("relay artifact digest must be a lowercase SHA-256")
        return value

    @field_validator("size_bytes")
    @classmethod
    def exact_size(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("relay artifact size must be an integer")
        return value


class RelayLocation(StrictModel):
    """One side of a relay; rclone configuration remains host-local."""

    kind: Literal["file", "rclone"]
    root: str
    config: Path | None = None

    @field_validator("root")
    @classmethod
    def valid_root(cls, value: str) -> str:
        if not value or "\x00" in value:
            raise ValueError("relay root must be a nonempty string")
        return value

    @field_validator("config")
    @classmethod
    def absolute_config(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("rclone config must be an absolute path")
        return value

    @model_validator(mode="after")
    def valid_combination(self) -> RelayLocation:
        if self.kind == "file":
            if self.config is not None:
                raise ValueError("file relay locations cannot have a config")
            if not Path(self.root).is_absolute():
                raise ValueError("file relay root must be absolute")
        elif self.config is None:
            # rclone also supports its normal config lookup, so config is optional.
            pass
        return self


class RelayProfile(StrictModel):
    relay_version: Literal[1] = _RELAY_VERSION
    namespace: str
    controller: RelayLocation
    worker: RelayLocation
    transfer_timeout_seconds: float = Field(default=1800, gt=0, le=7200)

    @field_validator("namespace")
    @classmethod
    def valid_namespace(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("relay namespace must be a safe identifier")
        return value

    @field_validator("relay_version", mode="before")
    @classmethod
    def exact_version(cls, value: Any) -> int:
        return _exact_int(value, "relay_version")

    @field_validator("transfer_timeout_seconds", mode="before")
    @classmethod
    def exact_timeout(cls, value: Any) -> float:
        if type(value) not in (int, float) or isinstance(value, bool):
            raise ValueError("transfer_timeout_seconds must be numeric")
        return float(value)

    def binding(self) -> RelayBinding:
        return RelayBinding.from_profile(self)


class RelayBinding(StrictModel):
    """Credential-free binding; controller location is omitted on a worker."""

    relay_version: Literal[1] = _RELAY_VERSION
    namespace: str
    worker: RelayLocation
    controller: RelayLocation | None = None
    transfer_timeout_seconds: float = Field(default=1800, gt=0, le=7200)

    @field_validator("namespace")
    @classmethod
    def valid_namespace(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("relay namespace must be a safe identifier")
        return value

    @field_validator("relay_version", mode="before")
    @classmethod
    def exact_version(cls, value: Any) -> int:
        return _exact_int(value, "relay_version")

    @field_validator("transfer_timeout_seconds", mode="before")
    @classmethod
    def exact_timeout(cls, value: Any) -> float:
        if type(value) not in (int, float) or isinstance(value, bool):
            raise ValueError("transfer_timeout_seconds must be numeric")
        return float(value)

    @classmethod
    def from_profile(cls, profile: RelayProfile) -> RelayBinding:
        return cls(
            namespace=profile.namespace,
            controller=profile.controller,
            worker=profile.worker,
            transfer_timeout_seconds=profile.transfer_timeout_seconds,
        )

    def worker_binding(self) -> RelayBinding:
        """The subset persisted under the worker root."""
        return self.model_copy(update={"controller": None})

    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))


class CheckpointFrontier(StrictModel):
    generation_id: str
    relative_path: str
    manifest_sha256: str
    step: int = Field(ge=0)
    tokens_seen: int = Field(ge=0)

    @field_validator("step", "tokens_seen", mode="before")
    @classmethod
    def exact_counters(cls, value: Any, info: Any) -> int:
        return _exact_int(value, info.field_name)

    @field_validator("generation_id")
    @classmethod
    def generation(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("checkpoint generation_id is unsafe")
        return value

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        return RelayArtifactIdentity(
            relative_path=value, sha256="0" * 64, size_bytes=0
        ).relative_path

    @field_validator("manifest_sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("checkpoint manifest_sha256 must be SHA-256")
        return value


class RelayOutboxPage(StrictModel):
    """One immutable exported outbox page and its contiguous sequence range."""

    artifact: RelayArtifactIdentity
    origin_id: str
    first_sequence: int = Field(ge=1, le=2**63 - 1)
    last_sequence: int = Field(ge=1, le=2**63 - 1)

    @field_validator("origin_id")
    @classmethod
    def origin(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("outbox origin_id is unsafe")
        return value

    @field_validator("first_sequence", "last_sequence", mode="before")
    @classmethod
    def exact_sequences(cls, value: Any, info: Any) -> int:
        return _exact_int(value, info.field_name)

    @model_validator(mode="after")
    def contiguous_range(self) -> RelayOutboxPage:
        if not 0 <= self.last_sequence - self.first_sequence < 1000:
            raise ValueError(
                "outbox page range is inverted or exceeds native export limit"
            )
        return self


class RelayCommitDescriptor(StrictModel):
    """Authenticated immutable recovery boundary.

    The descriptor is intentionally compact; ``inventory`` names every object
    needed by this boundary, including immutable manifests and record pages.
    """

    relay_commit_version: Literal[1] = _RELAY_VERSION
    sequence: int = Field(ge=0, le=2**63 - 1)
    previous_commit_sha256: str | None = None
    worker_id: str
    attempt_id: str
    run_id: str
    experiment_id: str
    spec_digest: str
    bundle_digest: str
    source_digest: str
    instance_id: str
    receipt: dict[str, Any]
    checkpoint: CheckpointFrontier | None = None
    inventory: tuple[RelayArtifactIdentity, ...] = ()
    inventory_pages: tuple[RelayArtifactIdentity, ...] = ()
    outbox_pages: tuple[RelayOutboxPage, ...] = ()
    hmac_sha256: str

    @field_validator("relay_commit_version", "sequence", mode="before")
    @classmethod
    def exact_commit_integers(cls, value: Any, info: Any) -> int:
        return _exact_int(value, info.field_name)

    @field_validator(
        "spec_digest",
        "bundle_digest",
        "source_digest",
        "hmac_sha256",
        "previous_commit_sha256",
    )
    @classmethod
    def digest_fields(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("relay digest must be a lowercase SHA-256")
        return value

    @field_validator(
        "worker_id", "attempt_id", "run_id", "experiment_id", "instance_id"
    )
    @classmethod
    def identifiers(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("relay identifier is unsafe")
        return value

    @field_validator("receipt", mode="before")
    @classmethod
    def captured_receipt(cls, value: Any) -> dict[str, Any]:
        # Delayed import prevents WorkerDefinition -> RelayBinding import cycles.
        from klab.workers.models import AttemptReceipt

        return AttemptReceipt.model_validate(value).model_dump(mode="json")

    @model_validator(mode="after")
    def inventory_bounds(self) -> RelayCommitDescriptor:
        items = (
            *self.inventory,
            *self.inventory_pages,
            *(page.artifact for page in self.outbox_pages),
        )
        if len(items) > 4096:
            raise ValueError("relay inventory exceeds item limit")
        if sum(item.size_bytes for item in items) > 1024**4:
            raise ValueError("relay inventory exceeds aggregate byte limit")
        if len({item.relative_path for item in items}) != len(items):
            raise ValueError("relay inventory has duplicate paths")
        pages_by_origin: dict[str, list[RelayOutboxPage]] = {}
        for page in self.outbox_pages:
            pages_by_origin.setdefault(page.origin_id, []).append(page)
        for pages in pages_by_origin.values():
            expected_sequence = 1
            for page in sorted(pages, key=lambda item: item.first_sequence):
                if page.first_sequence != expected_sequence:
                    raise ValueError("outbox page ranges are not contiguous")
                expected_sequence = page.last_sequence + 1
        expected = {
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "worker_id": self.worker_id,
            "spec_digest": self.spec_digest,
            "bundle_digest": self.bundle_digest,
        }
        if any(self.receipt.get(field) != value for field, value in expected.items()):
            raise ValueError("relay descriptor receipt identity substitution")
        return self

    def unsigned(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"hmac_sha256"})

    def descriptor_digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))


def load_relay_profile(path: Path) -> RelayProfile:
    """Load a strictly typed YAML or JSON operational profile without links."""
    if path.is_symlink():
        raise ValueError("relay profile must not be a symlink")
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - project ships YAML support
            raise RuntimeError("YAML relay profiles require PyYAML") from error

        class UniqueKeyLoader(yaml.SafeLoader):
            pass

        def mapping(loader: Any, node: Any, deep: bool = False) -> dict[str, Any]:
            loader.flatten_mapping(node)
            return reject_duplicate_keys(
                [
                    (
                        loader.construct_object(key, deep=deep),
                        loader.construct_object(value, deep=deep),
                    )
                    for key, value in node.value
                ]
            )

        UniqueKeyLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping
        )
        value = yaml.load(raw, Loader=UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError("relay profile must be an object")  # noqa: TRY004 — invalid profile data
    return RelayProfile.model_validate(value)
