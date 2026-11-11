"""Opt-in operational checkpoint cadence, separate from scientific RunConfig."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from klab.workspace_preflight import check_storage


class SpotSafetyPolicy(BaseModel):
    """Explicit observations and operator bounds; no inferred historical readings."""

    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, frozen=True
    )

    format: Literal["klab-spot-safety-v1"] = "klab-spot-safety-v1"
    observation_reference: str = Field(min_length=1)
    checkpoint_write_seconds: float = Field(gt=0)
    boundary_seconds: float = Field(gt=0)
    restart_seconds: float = Field(gt=0)
    interruption_notice_seconds: float = Field(ge=0)
    max_recovery_seconds: float = Field(gt=0)
    checkpoint_bytes: int = Field(gt=0)
    checkpoint_inodes: int = Field(gt=0)
    reserve_bytes: int = Field(ge=0)
    reserve_inodes: int = Field(ge=0)

    @model_validator(mode="after")
    def feasible(self) -> SpotSafetyPolicy:
        if not self.observation_reference.strip():
            raise ValueError(
                "observation_reference must identify retained measurements"
            )
        if self.interruption_notice_seconds < (
            self.checkpoint_write_seconds + self.boundary_seconds
        ):
            raise ValueError(
                "interruption notice cannot cover boundary and checkpoint write"
            )
        if self.every_seconds <= 0:
            raise ValueError("recovery budget cannot cover restart, write and boundary")
        return self

    @property
    def every_seconds(self) -> float:
        return (
            self.max_recovery_seconds
            - self.restart_seconds
            - self.checkpoint_write_seconds
            - self.boundary_seconds
        )

    def due(self, elapsed: float, last_checkpoint_elapsed: float) -> bool:
        return elapsed - last_checkpoint_elapsed >= self.every_seconds


def load_spot_policy(path: Path) -> SpotSafetyPolicy:
    return SpotSafetyPolicy.model_validate(json.loads(path.read_text(encoding="utf-8")))


def check_spot_capacity(policy: SpotSafetyPolicy, workspace: Path) -> dict[str, object]:
    """Recheck native capacity; retain existing generations until commit/retention.

    Two additional generation footprints conservatively cover staging plus a
    committed generation. Existing retained generations already consume free space.
    """
    storage = check_storage(
        workspace,
        projected_bytes=2 * policy.checkpoint_bytes,
        projected_inodes=2 * policy.checkpoint_inodes,
        reserve_bytes=policy.reserve_bytes,
        reserve_inodes=policy.reserve_inodes,
    )
    if storage.available_inodes is None:
        raise ValueError("spot-safety requires available inode observations")
    if storage.status != "adequate":
        raise ValueError("insufficient spot-checkpoint capacity")
    return {
        "format": "klab-spot-safety-decision-v1",
        "policy": policy.model_dump(mode="json"),
        "every_seconds": policy.every_seconds,
        "available_bytes": storage.available_bytes,
        "available_inodes": storage.available_inodes,
        "required_free_bytes": storage.projected_bytes + storage.reserve_bytes,
        "required_free_inodes": storage.projected_inodes + storage.reserve_inodes,
    }
