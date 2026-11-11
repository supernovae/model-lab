"""Closed operational probe results, separate from scientific declarations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from klab.runtime import RuntimeInfo
from klab.workers.models import WorkerModel

Backend = Literal["MATH", "EFFICIENT_ATTENTION", "FLASH_ATTENTION", "CUDNN_ATTENTION"]


class OperationalModel(WorkerModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AttentionDeclaration(OperationalModel):
    implementation: Literal["reference", "sdpa"]
    micro_batch_size: int = Field(gt=0)
    sequence_length: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    precision: Literal["auto", "fp32", "bf16", "fp16"]
    deterministic: bool


class AttentionCandidate(OperationalModel):
    status: Literal["SUPPORTED", "UNAVAILABLE"]
    error: str | None

    @model_validator(mode="after")
    def coherent_error(self) -> AttentionCandidate:
        if (self.status == "SUPPORTED") != (self.error is None):
            raise ValueError("candidate support and error disagree")
        return self


class AttentionRecommendation(OperationalModel):
    status: Literal["RECOMMENDED", "OBSERVED", "UNAVAILABLE", "NOT_INTEGRATED"]
    message: str


class AttentionProbeResult(OperationalModel):
    attention_probe_version: Literal[1]
    runtime: dict[str, Any]
    runtime_authorization: dict[str, Any] | None
    declared: AttentionDeclaration
    observed_operators: list[str]
    selected_backend: Backend | None
    candidates: dict[Backend, AttentionCandidate]
    recommendations: list[AttentionRecommendation]

    @field_validator("attention_probe_version", mode="before")
    @classmethod
    def exact_version(cls, value: Any) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("unsupported attention probe version")
        return value

    @model_validator(mode="after")
    def complete_candidates(self) -> AttentionProbeResult:
        RuntimeInfo.from_dict(self.runtime)
        if set(self.candidates) != {
            "MATH",
            "EFFICIENT_ATTENTION",
            "FLASH_ATTENTION",
            "CUDNN_ATTENTION",
        }:
            raise ValueError("attention probe must report all backend candidates")
        return self
