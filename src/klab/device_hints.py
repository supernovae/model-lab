"""Measured performance hints for a device family.

The shipped files in ``device_hints/`` are suggestions with the measurements
behind them. Nothing here changes a config.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import Field

from klab.config.models import StrictModel

HINT_DIR = Path(__file__).with_name("device_hints")


class HintMatch(StrictModel):
    backend: str
    device_name_regex: str


class HintEvidence(StrictModel):
    name: str
    tokens_per_second: float
    peak_gib: float


class Hint(StrictModel):
    lever: str
    value: str | int | float | bool
    summary: str
    evidence: tuple[HintEvidence, ...] = ()


class HintMeasured(StrictModel):
    device: str
    torch: str
    config_shape: str
    source: str


class DeviceHints(StrictModel):
    family: str
    description: str
    match: HintMatch
    measured: HintMeasured
    hints: tuple[Hint, ...] = Field(min_length=1)


def load_device_hints(directory: Path = HINT_DIR) -> tuple[DeviceHints, ...]:
    return tuple(
        DeviceHints.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*.yaml"))
    )


def hints_for_device(
    backend: str, device_name: str | None, directory: Path = HINT_DIR
) -> DeviceHints | None:
    """The family whose match covers this device, if any."""
    if not device_name:
        return None
    return next(
        (
            family
            for family in load_device_hints(directory)
            if family.match.backend == backend
            and re.search(family.match.device_name_regex, device_name)
        ),
        None,
    )
