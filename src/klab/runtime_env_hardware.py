"""Passive host hardware observations; never evidence of an executable runtime."""

from __future__ import annotations

import csv
import json
import platform
import re
import subprocess
from typing import Any

from klab.runtime_env_subprocess import run_bounded

_TIMEOUT = 5
_OUTPUT_LIMIT = 32 * 1024


def _query(command: list[str]) -> str | None:
    """Read bounded tool output without buffering an unlimited pipe in the parent."""
    try:
        result = run_bounded(command, timeout=_TIMEOUT, output_limit=_OUTPUT_LIMIT)
        data, error = result.stdout, result.stderr
    except OSError, subprocess.TimeoutExpired, ValueError:
        return None
    if result.returncode or len(data) > _OUTPUT_LIMIT or len(error) > _OUTPUT_LIMIT:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _record(backend: str, name: str | None, source: str) -> dict[str, str | None]:
    return {
        "backend": backend,
        "device_name": name,
        "observation_source": source,
        "status": "OBSERVED" if name else "UNKNOWN",
    }


def _rocm_names(text: str) -> list[str]:
    names: list[str] = []
    # rocminfo includes a CPU agent. Inspect each agent independently so a CPU
    # marketing name can never be paired with a different agent's gfx ISA.
    for agent in re.split(r"(?m)^\s*Agent\s+\d+\s*$", text)[1:]:
        fields = {
            key.strip().lower(): value.strip()
            for key, value in re.findall(r"(?m)^\s*([^:\n]+):\s*([^\n]*)$", agent)
        }
        kind = fields.get("device type", fields.get("type", "")).upper()
        identifier = fields.get("name", "")
        if kind == "CPU" or (
            kind != "GPU" and not re.fullmatch(r"gfx[0-9a-z]+", identifier)
        ):
            continue
        name = fields.get("marketing name", "")
        if not name or name.lower() in {"unknown", "n/a"}:
            name = identifier
        if name:
            names.append(name)
    return names


def _nvidia_names(text: str) -> list[str]:
    names: list[str] = []
    for row in csv.reader(text.splitlines()):
        if len(row) != 1:
            continue
        name = row[0].strip()
        if name and name.lower() not in {"name", "n/a", "unknown"}:
            names.append(name)
    return names


def _metal_names(text: str) -> list[str]:
    try:
        displays = json.loads(text).get("SPDisplaysDataType", [])
    except ValueError, AttributeError:
        return []
    if not isinstance(displays, list):
        return []
    names: list[str] = []
    for display in displays:
        if not isinstance(display, dict):
            continue
        metal = display.get("spdisplays_metal", "")
        if not isinstance(metal, str) or not (
            metal.lower().startswith("supported")
            or metal.lower() == "spdisplays_metal_supported"
        ):
            continue
        name: Any = display.get("sppci_model") or display.get("_name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _intel_names(text: str) -> list[str]:
    """Observe Intel PCI display devices, excluding bridges, CPUs and audio."""
    return [
        match[1].strip()
        for match in re.finditer(
            r"(?m)^\S+\s+[^\n]*\[03(?:00|02|80)\]:\s+([^\n]+?)\s+\[8086:[0-9a-fA-F]{4}\]",
            text,
        )
    ]


def hardware_inventory() -> list[dict[str, str | None]]:
    """Observe host devices, without importing frameworks or authorizing a runtime."""
    system = platform.system()
    probes = (
        (
            (
                "rocm",
                "rocminfo",
                ["rocminfo"],
                _rocm_names,
            ),
            (
                "cuda",
                "nvidia-smi",
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                _nvidia_names,
            ),
            (
                "xpu",
                "lspci",
                ["lspci", "-Dnn"],
                _intel_names,
            ),
        )
        if system == "Linux"
        else (
            (
                "metal",
                "system_profiler",
                ["system_profiler", "SPDisplaysDataType", "-json"],
                _metal_names,
            ),
        )
        if system == "Darwin"
        else ()
    )
    observations: list[dict[str, str | None]] = []
    for backend, source, command, parse in probes:
        text = _query(command)
        names = parse(text) if text is not None else []
        observations.extend(_record(backend, name, source) for name in names)
        if not names:
            observations.append(_record(backend, None, source))
    return observations
