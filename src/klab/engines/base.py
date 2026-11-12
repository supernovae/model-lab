"""Framework-neutral boundaries for the single-host trainer and checkpoint codecs."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np


class EngineError(RuntimeError):
    """An update could not complete; no committed result may be fabricated."""


class EngineCapabilityError(EngineError, ValueError):
    """The selected engine cannot execute the requested configuration."""


class EngineNonFiniteError(EngineError, FloatingPointError):
    """Nonfinite loss or unscaled gradients prevent an optimizer commit."""


class EngineOutOfMemory(EngineError, MemoryError):
    """The actual execution device could not satisfy an allocation."""


@dataclass(frozen=True)
class Microbatch:
    """Already-collated int64 NumPy arrays; engines must not mutate these."""

    inputs: np.ndarray
    targets: np.ndarray
    byte_addresses: np.ndarray | None = None
    position_mask: np.ndarray | None = None


@dataclass(frozen=True)
class CanonicalTensor:
    """One contiguous CPU array, kept stable while its shard is being written."""

    name: str
    array: np.ndarray
    trainable: bool


class WeightSource(Protocol):
    """Yield unique canonical tensors without cloning a complete model to CPU."""

    @property
    def aliases(self) -> Mapping[str, str]: ...

    def tensors(self) -> Iterator[CanonicalTensor]: ...


@dataclass(frozen=True)
class EngineState:
    """In-memory native state; explicit codecs, never pickle of a live engine."""

    optimizer: dict[str, object]
    rng: dict[str, object]
    scaler: dict[str, object] | None = None
    optimizer_parameter_names: dict[int, str] | list[list[str]] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class UpdateResult:
    """One optimizer attempt; only APPLIED may advance the trainer's cursor."""

    outcome: Literal["APPLIED", "OVERFLOW"]
    loss_sum: float | None
    valid_targets: int
    committed_targets: int
    elapsed_seconds: float
    metrics: dict[str, float] = field(default_factory=dict)
    format_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if type(self.valid_targets) is not int or self.valid_targets <= 0:
            raise ValueError("an update requires a positive integer target count")
        if type(self.committed_targets) is not int:
            raise TypeError("committed targets must be an integer")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("update duration must be finite and nonnegative")
        if self.outcome == "APPLIED":
            if self.committed_targets != self.valid_targets:
                raise ValueError(
                    "an applied update must commit its entire target window"
                )
            if self.loss_sum is None or not math.isfinite(self.loss_sum):
                raise ValueError("an applied update requires a finite loss sum")
        elif self.outcome == "OVERFLOW":
            if self.committed_targets != 0 or self.loss_sum is not None:
                raise ValueError("overflow cannot commit targets or successful loss")
        else:
            raise ValueError("unknown optimizer update outcome")


@dataclass(frozen=True)
class EvaluationResult:
    """Sums preserve exact supervised-target weighting across microbatches."""

    loss_sum: float
    valid_targets: int
    batches: int

    def to_report(self) -> dict[str, float | int | str | None]:
        if self.valid_targets <= 0 or self.batches <= 0:
            raise ValueError("evaluation contains no scored targets or batches")
        if not math.isfinite(self.loss_sum):
            raise EngineNonFiniteError("nonfinite validation loss")
        loss = self.loss_sum / self.valid_targets
        exceeds_range = loss > math.log(float.fromhex("0x1.fffffffffffffp+1023"))
        return {
            "loss": loss,
            "perplexity": None if exceeds_range else math.exp(loss),
            "perplexity_unavailable_reason": "loss exceeds finite exp range"
            if exceeds_range
            else None,
            "valid_targets": self.valid_targets,
            "batches": self.batches,
        }
