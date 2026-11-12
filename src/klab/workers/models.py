"""Versioned, JSON-native contracts for independent workers."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import RunConfig, StrictModel
from klab.data.encoding import (
    TOKENIZER_BATCH_DOCUMENTS,
    TOKENIZER_BATCH_SOURCE_BYTES,
    validate_tokenizer_batch_limits,
)
from klab.resource_envelope import ResourceEnvelope
from klab.runtime import RuntimeInfo
from klab.workers.relay_models import RelayBinding

PROTOCOL_VERSION = 1
SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_VERSION_KEYS = frozenset(
    {
        "protocol",
        "config",
        "manifest",
        "checkpoint",
        "bundle",
        "architecture",
        "state_codecs",
    }
)
MAX_RELATIVE_PATH_BYTES = 4096


def current_required_versions() -> dict[str, object]:
    """Return a fresh canonical worker compatibility requirement envelope."""
    from klab.training.mlx_checkpoints import CODEC_VERSION as MLX_CODEC_VERSION

    return {
        "protocol": [PROTOCOL_VERSION],
        "config": [2],
        "manifest": [1],
        "checkpoint": [2],
        "bundle": [1],
        "architecture": ["klab-decoder-v1"],
        "state_codecs": {"pytorch_native": [2], "mlx_native": [MLX_CODEC_VERSION]},
    }


def _closed_required_versions(required: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(required, Mapping) or set(required) != _REQUIRED_VERSION_KEYS:
        raise ValueError("required_versions has unsupported or missing keys")
    value = dict(required)
    for key in ("protocol", "config", "manifest", "checkpoint", "bundle"):
        versions = value[key]
        if (
            not isinstance(versions, (list, tuple))
            or not versions
            or any(type(item) is not int or item < 1 for item in versions)
        ):
            raise ValueError(
                f"required_versions.{key} must be nonempty integer versions"
            )
    architecture = value["architecture"]
    if (
        not isinstance(architecture, (list, tuple))
        or not architecture
        or any(not isinstance(item, str) or not item for item in architecture)
    ):
        raise ValueError("required_versions.architecture must be nonempty strings")
    codecs = value["state_codecs"]
    if not isinstance(codecs, Mapping) or set(codecs) != {
        "pytorch_native",
        "mlx_native",
    }:
        raise ValueError("required_versions.state_codecs must name supported codecs")
    for name, versions in codecs.items():
        if (
            not isinstance(versions, (list, tuple))
            or not versions
            or any(type(item) is not int or item < 1 for item in versions)
        ):
            raise ValueError(
                f"required_versions.state_codecs.{name} must be nonempty integer versions"
            )
    normalized = {
        key: list(value[key])
        for key in (
            "protocol",
            "config",
            "manifest",
            "checkpoint",
            "bundle",
            "architecture",
        )
    }
    normalized["state_codecs"] = {
        name: list(codecs[name]) for name in ("pytorch_native", "mlx_native")
    }
    return normalized


def validate_required_versions(
    required: Mapping[str, object],
    capabilities: WorkerCapabilities | Mapping[str, object] | None = None,
) -> None:
    """Reject unsupported persisted schemas, optionally against a worker advertisement."""
    normalized = _closed_required_versions(required)
    if normalized != current_required_versions():
        raise ValueError("required_versions is not supported by this worker protocol")
    if capabilities is None:
        return
    advertised = (
        capabilities
        if isinstance(capabilities, WorkerCapabilities)
        else WorkerCapabilities.model_validate(capabilities)
    )
    advertised_versions: dict[str, object] = {
        "protocol": list(advertised.protocol_versions),
        "config": list(advertised.config_versions),
        "manifest": list(advertised.manifest_versions),
        "checkpoint": list(advertised.checkpoint_versions),
        "bundle": list(advertised.bundle_versions),
        "architecture": list(advertised.architecture_versions),
        "state_codecs": {
            name: list(versions) for name, versions in advertised.state_codecs.items()
        },
    }
    for key in (
        "protocol",
        "config",
        "manifest",
        "checkpoint",
        "bundle",
        "architecture",
    ):
        if not set(normalized[key]).issubset(advertised_versions[key]):
            raise ValueError(f"worker does not support required {key} versions")
    for codec, versions in normalized["state_codecs"].items():
        available = advertised_versions["state_codecs"].get(codec, [])
        if not set(versions).issubset(available):
            raise ValueError(f"worker does not support required {codec} codec versions")


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} must be a safe identifier")
    return value


def _sha256(value: str, field: str = "sha256") -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def validate_relative_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > MAX_RELATIVE_PATH_BYTES
        or "\\" in value
        or any(unicodedata.category(char).startswith("C") for char in value)
        or value.startswith("./")
        or "//" in value
    ):
        raise ValueError("path must be a nonempty normalized relative POSIX path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("path must be a normalized relative path")
    return value


class WorkerModel(StrictModel):
    """Strict transport data; booleans never stand in for numeric fields."""

    @model_validator(mode="before")
    @classmethod
    def reject_boolean_numbers(cls, value: Any) -> Any:
        numeric_fields = {
            "schema_version",
            "size_bytes",
            "device_index",
            "max_concurrent_runs",
            "pid",
            "length",
            "min_device_memory_gb",
            "min_system_memory_gb",
        }
        version_fields = {
            "protocol_versions",
            "config_versions",
            "manifest_versions",
            "checkpoint_versions",
            "bundle_versions",
        }
        pending = [value]
        while pending:
            node = pending.pop()
            if isinstance(node, (list, tuple)):
                pending.extend(node)
                continue
            if not isinstance(node, Mapping):
                continue
            for name, item in node.items():
                if isinstance(item, (Mapping, list, tuple)):
                    pending.append(item)
                if name in numeric_fields and item is not None:
                    number_types = (
                        (int, float)
                        if name in {"min_device_memory_gb", "min_system_memory_gb"}
                        else (int,)
                    )
                    if type(item) not in number_types:
                        raise ValueError(
                            f"{name} must be a JSON number of the declared type"
                        )
                if (
                    name
                    in {
                        "allow_runtime_drift",
                        "cancellation_requested",
                        "cancellation_acknowledged",
                    }
                    and type(item) is not bool
                ):
                    raise ValueError(f"{name} must be a JSON boolean")
                if name in version_fields and (
                    not isinstance(item, (list, tuple))
                    or any(type(version) is not int for version in item)
                ):
                    raise ValueError(f"{name} must contain integer versions")
                if (
                    name == "state_codecs"
                    and isinstance(item, Mapping)
                    and any(
                        not isinstance(versions, (list, tuple))
                        or any(type(version) is not int for version in versions)
                        for versions in item.values()
                    )
                ):
                    raise ValueError("state_codecs must contain integer versions")
        return value

    @field_validator("*")
    @classmethod
    def finite_values(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value

    @model_validator(mode="after")
    def json_native(self) -> WorkerModel:
        try:
            canonical_json(self.model_dump(mode="json"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "worker contract must contain finite JSON-native values"
            ) from error
        return self


class ArtifactIdentity(WorkerModel):
    relative_path: str
    sha256: str

    size_bytes: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("size_bytes")
    @classmethod
    def exact_size(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("size_bytes must be an integer")
        return value


class ColabEndpoint(WorkerModel):
    session: str
    config_path: Path | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    auth: Literal["oauth2", "adc"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    instance_id: str
    job_timeout_seconds: float = Field(default=3600, gt=0)

    @field_validator("session", "instance_id")
    @classmethod
    def safe_endpoint_id(cls, value: str, info: Any) -> str:
        return _identifier(value, info.field_name)

    @field_validator("config_path")
    @classmethod
    def absolute_config(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("Colab config path must be absolute")
        return value


class WorkerPricing(WorkerModel):
    """What a worker costs to keep, and how much it may spend in this store."""

    hourly_rate: float = Field(gt=0, allow_inf_nan=False)
    currency: str = "USD"
    max_spend: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @field_validator("currency")
    @classmethod
    def currency_code(cls, value: str) -> str:
        if len(value) != 3 or not value.isalpha() or not value.isupper():
            raise ValueError("currency must be a three-letter code such as USD")
        return value


class WorkerDefinition(WorkerModel):
    schema_version: int = SCHEMA_VERSION
    worker_id: str
    name: str
    transport: Literal["local", "ssh", "colab"]
    host: str | None = None
    colab: ColabEndpoint | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    relay: RelayBinding | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    instance_id: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    pricing: WorkerPricing | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    python: Path
    root: Path
    engine: Literal["pytorch", "mlx"]
    backend: Literal["cpu", "mps", "cuda", "rocm", "xpu", "metal"]
    device_index: int = Field(ge=0)

    @field_validator("schema_version")
    @classmethod
    def worker_version(cls, value: int) -> int:
        if type(value) is not int or value != SCHEMA_VERSION:
            raise ValueError("unsupported worker schema version")
        return value

    @field_validator("worker_id", "name")
    @classmethod
    def safe_id(cls, value: str, info: Any) -> str:
        return _identifier(value, info.field_name)

    @field_validator("instance_id")
    @classmethod
    def safe_instance_id(cls, value: str | None) -> str | None:
        return None if value is None else _identifier(value, "instance_id")

    @field_validator("python", "root")
    @classmethod
    def absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("worker paths must be absolute")
        return value

    @field_validator("device_index")
    @classmethod
    def exact_index(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("device_index must be an integer")
        return value

    @field_validator("host")
    @classmethod
    def safe_host(cls, value: str | None) -> str | None:
        if value is not None and (
            not value
            or value.startswith("-")
            or any(unicodedata.category(char).startswith("C") for char in value)
        ):
            raise ValueError("unsafe SSH host")
        return value

    @model_validator(mode="after")
    def transport_fields(self) -> WorkerDefinition:
        if self.transport == "ssh" and self.host is None:
            raise ValueError("SSH workers require host")
        if self.transport != "ssh" and self.host is not None:
            raise ValueError("only SSH workers can declare host")
        if self.transport == "colab":
            if self.colab is None or self.relay is None:
                raise ValueError("Colab workers require endpoint and verified relay")
        elif self.colab is not None:
            raise ValueError("only Colab workers can declare a Colab endpoint")
        return self


class WorkerCapabilities(WorkerDefinition):
    runtime: dict[str, Any]
    supported_precisions: tuple[str, ...] = ()
    supported_features: tuple[str, ...] = ()
    max_concurrent_runs: int = 1
    validation_status: Literal["unverified", "passed", "failed"] = "unverified"
    validated_at: str | None = None
    status: Literal["idle", "busy", "offline", "unknown"] = "unknown"
    last_seen_at: str | None = None
    protocol_versions: tuple[int, ...] = (PROTOCOL_VERSION,)
    klab_version: str
    source_identity_sha256: str
    architecture_versions: tuple[str, ...]
    config_versions: tuple[int, ...]
    manifest_versions: tuple[int, ...]
    checkpoint_versions: tuple[int, ...]
    bundle_versions: tuple[int, ...]
    state_codecs: dict[str, tuple[int, ...]] = Field(default_factory=dict)

    @field_validator("runtime")
    @classmethod
    def runtime_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        RuntimeInfo.from_dict(value)
        return value

    @field_validator("source_identity_sha256")
    @classmethod
    def source_digest(cls, value: str) -> str:
        return _sha256(value, "source_identity_sha256")

    @field_validator("max_concurrent_runs")
    @classmethod
    def single_slot(cls, value: int) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("max_concurrent_runs must be exactly 1")
        return value

    @field_validator(
        "protocol_versions",
        "config_versions",
        "manifest_versions",
        "checkpoint_versions",
        "bundle_versions",
    )
    @classmethod
    def versions(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(type(item) is not int or item < 1 for item in value):
            raise ValueError("versions must be nonempty positive integers")
        return value

    @field_validator("state_codecs")
    @classmethod
    def codec_versions(
        cls, value: dict[str, tuple[int, ...]]
    ) -> dict[str, tuple[int, ...]]:
        if not isinstance(value, dict) or any(
            not isinstance(name, str)
            or not name
            or not versions
            or any(type(version) is not int or version < 1 for version in versions)
            for name, versions in value.items()
        ):
            raise ValueError("state codecs must map names to nonempty integer versions")
        return value


class SchedulingRequirements(WorkerModel):
    backend: tuple[Literal["cpu", "mps", "cuda", "rocm", "xpu", "metal"], ...] = ()
    min_device_memory_gb: float | None = Field(default=None, gt=0)
    min_system_memory_gb: float | None = Field(default=None, gt=0)
    precision: Literal["fp32", "bf16", "fp16"] | None = None


class ContinuationSpec(WorkerModel):
    kind: Literal["FRESH", "RESUMED", "PROMOTED"] = "FRESH"
    parent_run_id: str | None = None
    checkpoint_sha256: str | None = None
    artifact_identity: ArtifactIdentity | None = None
    allow_runtime_drift: bool = False
    budget_extension: bool = Field(default=False, exclude_if=lambda value: not value)

    @field_validator("parent_run_id")
    @classmethod
    def parent_id(cls, value: str | None) -> str | None:
        return None if value is None else _identifier(value, "parent_run_id")

    @field_validator("checkpoint_sha256")
    @classmethod
    def checkpoint_digest(cls, value: str | None) -> str | None:
        return None if value is None else _sha256(value, "checkpoint_sha256")

    @model_validator(mode="after")
    def continuation_shape(self) -> ContinuationSpec:
        values = (self.parent_run_id, self.checkpoint_sha256, self.artifact_identity)
        if self.kind == "FRESH":
            if any(value is not None for value in values) or self.allow_runtime_drift:
                raise ValueError(
                    "fresh continuation cannot include parent state or drift"
                )
        elif any(value is None for value in values):
            raise ValueError(
                "continuation requires parent run, checkpoint, and artifact identity"
            )
        elif self.kind != "RESUMED" and self.allow_runtime_drift:
            raise ValueError("runtime drift only applies to full resume")
        if self.budget_extension and (
            self.kind != "RESUMED" or self.allow_runtime_drift
        ):
            raise ValueError(
                "budget extension requires full-state resume without runtime drift"
            )
        return self


class MatrixMetadata(WorkerModel):
    matrix_sha256: str
    coordinate: dict[str, str]

    @field_validator("matrix_sha256")
    @classmethod
    def matrix_digest(cls, value: str) -> str:
        return _sha256(value, "matrix_sha256")

    @field_validator("coordinate")
    @classmethod
    def coordinate_labels(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or any(
            not isinstance(key, str)
            or not key
            or not isinstance(label, str)
            or not label
            for key, label in value.items()
        ):
            raise ValueError("matrix coordinate must map nonempty strings to labels")
        return value


class PlanMetadata(WorkerModel):
    """Lock identity preserved in the worker spec and run dispatch receipt."""

    plan_id: str
    plan_sha256: str
    scientific_sha256: str
    cell_id: str
    phase_id: str
    coordinate: dict[str, str]
    config_sha256: str
    parent_checkpoint_sha256: str | None = None
    execution_binding_sha256: str | None = None
    runtime_binding_sha256: str | None = None

    @field_validator(
        "plan_sha256",
        "scientific_sha256",
        "config_sha256",
        "parent_checkpoint_sha256",
        "execution_binding_sha256",
        "runtime_binding_sha256",
    )
    @classmethod
    def valid_digest(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _sha256(value, info.field_name)


class ExperimentSpec(WorkerModel):
    schema_version: int = SCHEMA_VERSION
    experiment_id: str
    config: RunConfig
    config_sha256: str
    bound_worker: str | None = None
    preferred_worker: str | None = None
    requirements: SchedulingRequirements = Field(default_factory=SchedulingRequirements)
    continuation: ContinuationSpec = Field(default_factory=ContinuationSpec)
    resource_envelope: ResourceEnvelope | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    tokenizer_batch_documents: int = Field(
        default=TOKENIZER_BATCH_DOCUMENTS,
        exclude_if=lambda value: value == TOKENIZER_BATCH_DOCUMENTS,
    )
    tokenizer_batch_source_bytes: int = Field(
        default=TOKENIZER_BATCH_SOURCE_BYTES,
        exclude_if=lambda value: value == TOKENIZER_BATCH_SOURCE_BYTES,
    )
    required_versions: dict[str, object]
    source_identity_sha256: str
    dispatch_bundle_digest: str
    matrix: MatrixMetadata | None = None
    plan: PlanMetadata | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    declaration_provenance: dict[str, Any] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    storage_checks: list[dict[str, str]] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )

    @field_validator(
        "tokenizer_batch_documents", "tokenizer_batch_source_bytes", mode="before"
    )
    @classmethod
    def valid_tokenizer_batch(cls, value: object, info: Any) -> int:
        if info.field_name == "tokenizer_batch_documents":
            return validate_tokenizer_batch_limits(value, TOKENIZER_BATCH_SOURCE_BYTES)[
                0
            ]
        return validate_tokenizer_batch_limits(TOKENIZER_BATCH_DOCUMENTS, value)[1]

    @field_validator("schema_version")
    @classmethod
    def version(cls, value: int) -> int:
        if type(value) is not int or value != SCHEMA_VERSION:
            raise ValueError("unsupported experiment spec version")
        return value

    @field_validator("experiment_id", "bound_worker", "preferred_worker")
    @classmethod
    def ids(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _identifier(value, info.field_name)

    @field_validator(
        "config_sha256", "source_identity_sha256", "dispatch_bundle_digest"
    )
    @classmethod
    def digests(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("required_versions")
    @classmethod
    def required_schema_versions(cls, value: dict[str, object]) -> dict[str, object]:
        validate_required_versions(value)
        return _closed_required_versions(value)

    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))


class BundleManifest(WorkerModel):
    schema_version: int = SCHEMA_VERSION
    config: RunConfig
    config_sha256: str
    required_versions: dict[str, object]
    source_identity_sha256: str
    continuation: ContinuationSpec = Field(default_factory=ContinuationSpec)
    files: tuple[ArtifactIdentity, ...]
    stage_evidence: dict[str, Any] | None = None

    @field_validator("schema_version")
    @classmethod
    def version(cls, value: int) -> int:
        if type(value) is not int or value != SCHEMA_VERSION:
            raise ValueError("unsupported bundle manifest version")
        return value

    @field_validator("config_sha256", "source_identity_sha256")
    @classmethod
    def digests(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("required_versions")
    @classmethod
    def required_schema_versions(cls, value: dict[str, object]) -> dict[str, object]:
        validate_required_versions(value)
        return _closed_required_versions(value)

    @model_validator(mode="after")
    def distinct_files(self) -> BundleManifest:
        paths = [item.relative_path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("bundle manifest has duplicate paths")
        return self

    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))


class AttemptReceipt(WorkerModel):
    schema_version: int = SCHEMA_VERSION
    attempt_id: str
    run_id: str
    experiment_id: str
    spec_digest: str
    bundle_digest: str
    worker_id: str
    state: Literal[
        "PREPARED", "RUNNING", "COMPLETE", "FAILED", "INTERRUPTED", "UNKNOWN"
    ]
    phase: str
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    boot_id: str | None = None
    pid: int | None = None
    process_start: str | None = None
    start_token: str | None = None
    heartbeat_at: str | None = None
    cancellation_requested: bool = False
    cancellation_acknowledged: bool = False
    error: dict[str, Any] | None = None
    artifacts: tuple[ArtifactIdentity, ...] = ()
    training_started_at: str | None = None

    @field_validator("schema_version")
    @classmethod
    def version(cls, value: int) -> int:
        if type(value) is not int or value != SCHEMA_VERSION:
            raise ValueError("unsupported attempt receipt version")
        return value

    @field_validator("attempt_id", "run_id", "experiment_id", "worker_id")
    @classmethod
    def receipt_ids(cls, value: str, info: Any) -> str:
        return _identifier(value, info.field_name)

    @field_validator("spec_digest", "bundle_digest")
    @classmethod
    def receipt_digests(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("pid")
    @classmethod
    def receipt_pid(cls, value: int | None) -> int | None:
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("pid must be a nonnegative integer or null")
        return value


class Submission(WorkerModel):
    experiment_id: str
    attempt_id: str
    run_id: str
    status: Literal["QUEUED"] = "QUEUED"

    @field_validator("experiment_id", "attempt_id", "run_id")
    @classmethod
    def submission_ids(cls, value: str, info: Any) -> str:
        return _identifier(value, info.field_name)


class AttachmentHeader(WorkerModel):
    name: str
    length: int = Field(ge=0)
    sha256: str

    @field_validator("name")
    @classmethod
    def attachment_name(cls, value: str) -> str:
        return validate_relative_path(value)

    @field_validator("length")
    @classmethod
    def attachment_length(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("attachment length must be an integer")
        return value

    @field_validator("sha256")
    @classmethod
    def attachment_digest(cls, value: str) -> str:
        return _sha256(value)


REQUEST_OPERATIONS = frozenset(
    {
        "discover",
        "validate",
        "install_bundle",
        "launch",
        "prepare",
        "status",
        "records",
        "artifact",
        "cancel",
        "attention_probe",
        "configure_relay",
        "relay_flush",
    }
)


def validate_operation_attachments(
    op: str,
    payload: Mapping[str, object] | None,
    attachments: tuple[AttachmentHeader, ...] | list[AttachmentHeader],
    *,
    response: bool = False,
) -> None:
    """Enforce operation-specific body names and metadata limits before reads."""
    if op not in REQUEST_OPERATIONS:
        raise ValueError(f"unsupported protocol operation: {op}")
    names = {item.name for item in attachments}
    if response:
        expected = {"records": {"records.json"}, "artifact": {"content"}}.get(op, set())
        if names != expected:
            raise ValueError(f"invalid {op} response attachments")
        if any(item.length > 4 * 1024 * 1024 for item in attachments):
            raise ValueError("response attachment exceeds 4 MiB")
        return
    payload = validate_operation(op, payload or {})
    if op in {"launch", "prepare"}:
        if names not in ({"spec.json"}, {"spec.json", "relay-key.bin"}):
            raise ValueError(f"{op} requires spec.json and optional relay-key.bin")
        if (
            next(item for item in attachments if item.name == "spec.json").length
            > 32 * 1024 * 1024
        ):
            raise ValueError("launch metadata exceeds 32 MiB")
        if (
            "relay-key.bin" in names
            and next(
                item for item in attachments if item.name == "relay-key.bin"
            ).length
            != 32
        ):
            raise ValueError("relay key must contain exactly 32 bytes")
        return
    if op == "install_bundle":
        if "bundle.json" not in names:
            raise ValueError("install_bundle requires bundle.json")
        if (
            next(item for item in attachments if item.name == "bundle.json").length
            > 32 * 1024 * 1024
        ):
            raise ValueError("bundle metadata exceeds 32 MiB")
        assets = names - {"bundle.json"}
        if payload["mode"] == "check" and assets:
            raise ValueError("install_bundle check cannot include assets")
        if any(
            not name.startswith("assets/")
            or not _SHA256.fullmatch(name.removeprefix("assets/"))
            for name in assets
        ):
            raise ValueError("install_bundle assets must be assets/<sha256>")
        return
    if names:
        raise ValueError(f"{op} does not accept attachments")


def validate_operation(
    op: str, payload: Mapping[str, Any], *, response: bool = False
) -> dict[str, Any]:
    """Validate closed protocol-v1 request payloads before attachment consumption."""
    if op not in REQUEST_OPERATIONS:
        raise ValueError(f"unsupported protocol operation: {op}")
    if not isinstance(payload, Mapping):
        raise TypeError("protocol payload must be an object")
    value = dict(payload)
    required: dict[str, set[str]] = {
        "discover": set(),
        "validate": {"config", "bundle_digest"},
        "install_bundle": {"manifest_digest", "mode"},
        "launch": {
            "attempt_id",
            "run_id",
            "experiment_id",
            "spec_digest",
            "bundle_digest",
        },
        "status": {"attempt_id"},
        "records": {"origin_id", "after_sequence", "limit", "max_bytes"},
        "artifact": {"attempt_id", "relative_path", "offset", "max_bytes"},
        "cancel": {"attempt_id", "reason"},
        "attention_probe": {"config"},
        "configure_relay": {"binding", "challenge"},
        "relay_flush": {"attempt_id"},
    }
    required["prepare"] = required["launch"]
    optional = (
        {"cold_verify"} if op in {"install_bundle", "launch", "prepare"} else set()
    )
    if not response and (
        not required[op] <= set(value) or set(value) - required[op] - optional
    ):
        raise ValueError(f"invalid {op} payload fields")
    if not response:
        if "cold_verify" in value and type(value["cold_verify"]) is not bool:
            raise ValueError("cold_verify must be a boolean")
        if op == "attention_probe":
            RunConfig.model_validate(value["config"])
        elif op == "configure_relay":
            binding = RelayBinding.model_validate(value["binding"])
            if binding.controller is not None:
                raise ValueError("configure_relay accepts only the worker binding")
            ArtifactIdentity.model_validate(value["challenge"])
        elif op == "relay_flush":
            _identifier(value["attempt_id"], "attempt_id")
        if op == "validate":
            RunConfig.model_validate(value["config"])
            if value["bundle_digest"] is not None:
                _sha256(value["bundle_digest"], "bundle_digest")
        elif op == "install_bundle":
            _sha256(value["manifest_digest"], "manifest_digest")
            if value["mode"] not in {"check", "install"}:
                raise ValueError("install_bundle mode must be check or install")
        elif op in {"launch", "prepare"}:
            for name in ("attempt_id", "run_id", "experiment_id"):
                _identifier(value[name], name)
            for name in ("spec_digest", "bundle_digest"):
                _sha256(value[name], name)
        elif op == "status" and value["attempt_id"] is not None:
            _identifier(value["attempt_id"], "attempt_id")
        elif op == "records":
            _identifier(value["origin_id"], "origin_id")
            for name, low, high in (
                ("after_sequence", 0, None),
                ("limit", 1, 1000),
                ("max_bytes", 65536, 4 * 1024 * 1024),
            ):
                item = value[name]
                if (
                    type(item) is not int
                    or item < low
                    or (high is not None and item > high)
                ):
                    raise ValueError(f"invalid records {name}")
        elif op == "artifact":
            _identifier(value["attempt_id"], "attempt_id")
            validate_relative_path(value["relative_path"])
            if type(value["offset"]) is not int or value["offset"] < 0:
                raise ValueError("invalid artifact offset")
            if (
                type(value["max_bytes"]) is not int
                or not 0 < value["max_bytes"] <= 4 * 1024 * 1024
            ):
                raise ValueError("invalid artifact max_bytes")
        elif op == "cancel":
            _identifier(value["attempt_id"], "attempt_id")
            if value["reason"] != "user":
                raise ValueError("cancel reason must be user")
    return value


def validate_operation_result(op: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the closed result envelope for a successful protocol operation."""
    if op not in REQUEST_OPERATIONS or not isinstance(result, Mapping):
        raise ValueError("unsupported operation result")
    value = dict(result)
    if op == "configure_relay":
        if set(value) != {"binding_sha256", "verified_challenge_sha256"}:
            raise ValueError("invalid configure_relay result")
        for digest in value.values():
            _sha256(digest)
        return value
    if op == "relay_flush":
        if set(value) != {"commit_sha256", "sequence"}:
            raise ValueError("invalid relay_flush result")
        _sha256(value["commit_sha256"])
        if type(value["sequence"]) is not int or value["sequence"] < 0:
            raise ValueError("invalid relay commit sequence")
        return value
    if op == "attention_probe":
        from .operational_models import AttentionProbeResult

        return AttentionProbeResult.model_validate(value).model_dump(mode="json")
    if op == "discover":
        return WorkerCapabilities.model_validate(value).model_dump(mode="json")
    if op in {"launch", "prepare"}:
        return AttemptReceipt.model_validate(value).model_dump(mode="json")
    if op == "validate":
        if set(value) != {
            "runtime",
            "validation_status",
            "reason",
            "runtime_authorization",
        }:
            raise ValueError("invalid validate result fields")
        RuntimeInfo.from_dict(value["runtime"])
        if value["validation_status"] not in {"passed", "failed"} or (
            value["reason"] is not None and not isinstance(value["reason"], str)
        ):
            raise ValueError("invalid validate result")
        evidence = value["runtime_authorization"]
        if evidence is not None and (
            not isinstance(evidence, dict)
            or evidence.get("authorization_version") != 1
            or evidence.get("kind") != "worker"
            or not isinstance(evidence.get("descriptor"), dict)
            or not isinstance(evidence.get("probe"), dict)
        ):
            raise ValueError("invalid worker runtime authorization evidence")
        if value["validation_status"] == "passed" and evidence is None:
            raise ValueError(
                "passed validation requires runtime authorization evidence"
            )
        return value
    if op == "install_bundle":
        if set(value) != {"bundle_digest", "missing_asset_digests", "installed"}:
            raise ValueError("invalid install_bundle result fields")
        _sha256(value["bundle_digest"], "bundle_digest")
        if type(value["installed"]) is not bool or not isinstance(
            value["missing_asset_digests"], list
        ):
            raise ValueError("invalid install_bundle result")
        for digest in value["missing_asset_digests"]:
            _sha256(digest, "missing asset digest")
        return value
    if op == "status":
        if set(value) != {"capabilities", "receipt", "receipts", "origin_id"}:
            raise ValueError("invalid status result fields")
        WorkerCapabilities.model_validate(value["capabilities"])
        if value["receipt"] is not None:
            AttemptReceipt.model_validate(value["receipt"])
        if not isinstance(value["receipts"], list) or len(value["receipts"]) > 100:
            raise ValueError("invalid status receipts")
        for receipt in value["receipts"]:
            AttemptReceipt.model_validate(receipt)
        _identifier(value["origin_id"], "origin_id")
        return value
    if op == "records":
        if set(value) != {"origin_id", "count", "next_sequence", "has_more"}:
            raise ValueError("invalid records result fields")
        if (
            not isinstance(value["origin_id"], str)
            or type(value["count"]) is not int
            or value["count"] < 0
            or type(value["next_sequence"]) is not int
            or value["next_sequence"] < 0
            or type(value["has_more"]) is not bool
        ):
            raise ValueError("invalid records result")
        return value
    if op == "artifact":
        fields = {
            "file_sha256",
            "total_length",
            "offset",
            "chunk_sha256",
            "next_offset",
            "eof",
        }
        if set(value) != fields:
            raise ValueError("invalid artifact result fields")
        _sha256(value["file_sha256"], "file_sha256")
        _sha256(value["chunk_sha256"], "chunk_sha256")
        for name in ("total_length", "offset", "next_offset"):
            if type(value[name]) is not int or value[name] < 0:
                raise ValueError(f"invalid artifact {name}")
        if type(value["eof"]) is not bool:
            raise ValueError("invalid artifact eof")
        return value
    if set(value) != {"receipt", "acknowledged"}:
        raise ValueError("invalid cancel result fields")
    if value["receipt"] is None:
        if value["acknowledged"] is not False:
            raise ValueError("cancellation without a receipt is not acknowledged")
    else:
        AttemptReceipt.model_validate(value["receipt"])
    if type(value["acknowledged"]) is not bool:
        raise ValueError("invalid cancel acknowledgement")
    return value
