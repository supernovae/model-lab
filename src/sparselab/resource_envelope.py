"""Operational resource limits and side-effect-free workspace measurements."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import psutil
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from sparselab import host_capacity
from sparselab.workspace_preflight import check_storage


class LabTryLimits(BaseModel):
    """Operational limits for each try arm, including its data preparation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_wall_seconds_per_arm: float | None = Field(
        default=None, strict=True, gt=0, allow_inf_nan=False
    )
    tokenizer_batch_documents: int | None = Field(
        default=None, strict=True, ge=1, le=256
    )
    tokenizer_batch_source_bytes: int | None = Field(
        default=None, strict=True, ge=1, le=8388608
    )


class ResourceEnvelope(BaseModel):
    """Versioned operational limits, separate from scientific run configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lab_try: LabTryLimits | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    resource_envelope_version: Literal[1]
    max_rss_bytes: int | None = Field(default=None, strict=True, gt=0)
    max_host_memory_fraction: float | None = Field(
        default=None, strict=True, gt=0, le=1
    )
    min_available_ram_bytes: int | None = Field(default=None, strict=True, ge=0)
    min_swap_bytes: int | None = Field(default=None, strict=True, ge=0)
    min_disk_bytes: int | None = Field(default=None, strict=True, ge=0)
    min_inodes: int | None = Field(default=None, strict=True, ge=0)
    max_workers: int | None = Field(default=None, strict=True, gt=0)
    preparation_shard_workers: int = Field(default=1, strict=True, ge=1, le=32)
    max_queue_depth: int | None = Field(default=None, strict=True, gt=0)
    spill_to_disk: bool = Field(default=True, strict=True)

    @field_validator("resource_envelope_version", mode="before")
    @classmethod
    def require_exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("resource_envelope_version must be integer 1")
        return value


def resolve_lab_try_limits(
    envelope: ResourceEnvelope | None,
    *,
    max_wall_seconds: float | None,
    tokenizer_batch_documents: int | None,
    tokenizer_batch_source_bytes: int | None,
) -> tuple[float | None, int, int]:
    """Use declared limits while preserving tighter explicit API limits."""
    from sparselab.data.encoding import (
        TOKENIZER_BATCH_DOCUMENTS,
        TOKENIZER_BATCH_SOURCE_BYTES,
        validate_tokenizer_batch_limits,
    )

    if max_wall_seconds is not None and (
        isinstance(max_wall_seconds, bool)
        or not isinstance(max_wall_seconds, (int, float))
        or not math.isfinite(max_wall_seconds)
        or max_wall_seconds <= 0
    ):
        raise ValueError("max_wall_seconds must be positive and finite")
    validate_tokenizer_batch_limits(
        TOKENIZER_BATCH_DOCUMENTS
        if tokenizer_batch_documents is None
        else tokenizer_batch_documents,
        TOKENIZER_BATCH_SOURCE_BYTES
        if tokenizer_batch_source_bytes is None
        else tokenizer_batch_source_bytes,
    )
    limits = envelope.lab_try if envelope is not None else None

    def tighter(explicit, declared, default=None):
        if explicit is not None and declared is not None:
            return min(explicit, declared)
        return (
            explicit
            if explicit is not None
            else (declared if declared is not None else default)
        )

    return (
        tighter(max_wall_seconds, limits.max_wall_seconds_per_arm if limits else None),
        tighter(
            tokenizer_batch_documents,
            limits.tokenizer_batch_documents if limits else None,
            TOKENIZER_BATCH_DOCUMENTS,
        ),
        tighter(
            tokenizer_batch_source_bytes,
            limits.tokenizer_batch_source_bytes if limits else None,
            TOKENIZER_BATCH_SOURCE_BYTES,
        ),
    )


def load_resource_envelope(path: Path) -> ResourceEnvelope:
    """Read a strict YAML envelope without inspecting or creating output paths."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(
            f"cannot read resource envelope {path}: {error.strerror or error}"
        ) from error
    except yaml.YAMLError as error:
        raise ValueError(
            f"invalid YAML in resource envelope {path}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise ValueError(f"resource envelope {path} must contain a YAML mapping")  # noqa: TRY004 - invalid serialized schema
    try:
        return ResourceEnvelope.model_validate(raw)
    except ValidationError as error:
        raise ValueError(f"invalid resource envelope {path}: {error}") from error


def require_envelope_applicability(
    envelope: ResourceEnvelope | None, *, operation: str
) -> None:
    """Reject scoped limits before a consumer can silently ignore them."""
    if envelope is not None and envelope.lab_try is not None and operation != "try":
        raise ValueError(
            f"resource envelope lab_try limits apply only to try, not {operation}; "
            "remove lab_try for this operation or select sparselab try"
        )


def current_process_rss_bytes() -> int | None:
    """Return this process's resident set size, or None when unavailable."""
    try:
        return int(psutil.Process().memory_info().rss)
    except psutil.Error, OSError, AttributeError, TypeError, ValueError:
        return None


def check_envelope(
    envelope: ResourceEnvelope | None,
    *,
    workspace: Path,
    rss_bytes: int | None,
    pending_workers: int = 0,
    queue_depth: int = 0,
) -> dict[str, int | float | None]:
    """Reject unconsumed scoped limits, then measure common host/storage bounds.

    Missing observations remain ``None``; they fail closed only when a limit
    requires them. Storage is sampled at the existing workspace ancestor, not
    by creating the destination directory. Callers supply their process RSS
    (or ``None`` if it is unavailable) and current worker/queue occupancy.
    """
    require_envelope_applicability(envelope, operation="native resource checks")
    if rss_bytes is not None and (type(rss_bytes) is not int or rss_bytes < 0):
        raise ValueError("rss_bytes must be a nonnegative integer or None")
    for name, count in (
        ("pending_workers", pending_workers),
        ("queue_depth", queue_depth),
    ):
        if type(count) is not int or count < 0:
            raise ValueError(f"{name} must be a nonnegative integer")

    try:
        memory = host_capacity.measure_memory()
        available_ram = int(memory.available)
        total_ram = int(memory.total)
        if available_ram < 0 or total_ram <= 0:
            available_ram = None
            total_ram = None
    except psutil.Error, OSError, AttributeError, TypeError, ValueError:
        available_ram = None
        total_ram = None
    try:
        swap = int(psutil.swap_memory().free)
        if swap < 0:
            swap = None
    except psutil.Error, OSError, AttributeError, TypeError, ValueError:
        swap = None
    try:
        storage = check_storage(
            workspace,
            projected_bytes=0,
            projected_inodes=0,
            reserve_bytes=0,
            reserve_inodes=0,
        )
        disk = storage.available_bytes
        inodes = storage.available_inodes
    except OSError, ValueError:
        disk = None
        inodes = None

    fraction = rss_bytes / total_ram if rss_bytes is not None and total_ram else None
    measurements: dict[str, int | float | None] = {
        "rss_bytes": rss_bytes,
        "host_total_ram_bytes": total_ram,
        "host_available_ram_bytes": available_ram,
        "host_memory_fraction": fraction,
        "swap_free_bytes": swap,
        "disk_free_bytes": disk,
        "disk_free_inodes": inodes,
        "pending_workers": pending_workers,
        "queue_depth": queue_depth,
    }
    if envelope is not None:
        checks = (
            ("max_rss_bytes", rss_bytes, envelope.max_rss_bytes, "max"),
            (
                "max_host_memory_fraction",
                fraction,
                envelope.max_host_memory_fraction,
                "max",
            ),
            (
                "min_available_ram_bytes",
                available_ram,
                envelope.min_available_ram_bytes,
                "min",
            ),
            ("min_swap_bytes", swap, envelope.min_swap_bytes, "min"),
            ("min_disk_bytes", disk, envelope.min_disk_bytes, "min"),
            ("min_inodes", inodes, envelope.min_inodes, "min"),
            ("max_workers", pending_workers, envelope.max_workers, "max"),
            ("max_queue_depth", queue_depth, envelope.max_queue_depth, "max"),
        )
        for name, observed, limit, direction in checks:
            if limit is None:
                continue
            if observed is None:
                raise ValueError(f"resource envelope {name}: measurement unavailable")
            if observed > limit if direction == "max" else observed < limit:
                raise ValueError(
                    f"resource envelope {name} exceeded: measured {observed}, limit {limit}"
                )
    return measurements
