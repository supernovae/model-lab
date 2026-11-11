"""Append-only experiment stage records and validated transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import ClassVar


class ExperimentStage(StrEnum):
    CONFIGURED = "CONFIGURED"
    INSPECTED = "INSPECTED"
    VALIDATED = "VALIDATED"
    SMOKE_TEST = "SMOKE_TEST"
    WARMUP = "WARMUP"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    CHECKPOINTED = "CHECKPOINTED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


@dataclass(frozen=True)
class StageRecord:
    stage: ExperimentStage
    status: str
    started_at: str
    finished_at: str | None = None
    step: int = 0
    tokens_seen: int = 0
    reason: str | None = None
    payload: dict[str, object] | None = None


class StageHistory:
    """Completed intervals are append-only; the active interval is separate."""

    _next: ClassVar[dict[ExperimentStage | None, set[ExperimentStage]]] = {
        None: {ExperimentStage.CONFIGURED},
        ExperimentStage.CONFIGURED: {ExperimentStage.INSPECTED},
        ExperimentStage.INSPECTED: {ExperimentStage.VALIDATED},
        ExperimentStage.VALIDATED: {
            ExperimentStage.SMOKE_TEST,
            ExperimentStage.TRAINING,
        },
        ExperimentStage.SMOKE_TEST: {ExperimentStage.WARMUP, ExperimentStage.TRAINING},
        ExperimentStage.WARMUP: {ExperimentStage.TRAINING},
        ExperimentStage.TRAINING: {
            ExperimentStage.EVALUATING,
            ExperimentStage.CHECKPOINTED,
        },
        ExperimentStage.EVALUATING: {
            ExperimentStage.CHECKPOINTED,
            ExperimentStage.TRAINING,
        },
        ExperimentStage.CHECKPOINTED: {
            ExperimentStage.TRAINING,
            ExperimentStage.COMPLETE,
            ExperimentStage.INTERRUPTED,
        },
    }

    def __init__(self) -> None:
        self.records: list[StageRecord] = []
        self.current: StageRecord | None = None

    def start(
        self,
        stage: ExperimentStage,
        *,
        step: int = 0,
        tokens_seen: int = 0,
        payload: dict[str, object] | None = None,
    ) -> StageRecord:
        if self.current is not None:
            raise ValueError(f"stage is still active: {self.current.stage}")
        previous = self.records[-1].stage if self.records else None
        allowed = self._next.get(previous, set())
        if previous in self._next and previous is not None:
            allowed = allowed | {ExperimentStage.FAILED, ExperimentStage.INTERRUPTED}
        if self.records:
            last = self.records[-1]
            if step < last.step or tokens_seen < last.tokens_seen:
                raise ValueError("stage counters cannot move backward")
            if last.status == "failed":
                allowed &= {ExperimentStage.FAILED}
            elif last.status == "interrupted":
                allowed &= {ExperimentStage.INTERRUPTED}
        if stage not in allowed:
            raise ValueError(f"illegal stage transition: {previous} -> {stage}")
        if step < 0 or tokens_seen < 0:
            raise ValueError("stage counters must be nonnegative")
        self.current = StageRecord(
            stage,
            "running",
            datetime.now(UTC).isoformat(),
            step=step,
            tokens_seen=tokens_seen,
            payload=payload,
        )
        return self.current

    def finish(
        self,
        status: str = "complete",
        *,
        reason: str | None = None,
        step: int | None = None,
        tokens_seen: int | None = None,
        payload: dict[str, object] | None = None,
    ) -> StageRecord:
        if self.current is None:
            raise ValueError("no active stage")
        if status not in {"complete", "failed", "interrupted"}:
            raise ValueError(f"unsupported stage status: {status}")
        last = self.current
        final_step = last.step if step is None else step
        final_tokens = last.tokens_seen if tokens_seen is None else tokens_seen
        if final_step < last.step or final_tokens < last.tokens_seen:
            raise ValueError("stage counters cannot move backwards")
        record = StageRecord(
            last.stage,
            status,
            last.started_at,
            datetime.now(UTC).isoformat(),
            final_step,
            final_tokens,
            reason,
            payload if payload is not None else last.payload,
        )
        self.records.append(record)
        self.current = None
        return record
