"""Versioned vendor recipe and read-only local runtime storage preflight."""

from __future__ import annotations

import os
import platform
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from sparselab.runtime_environments import resolve_runtime_dir

_MIN_BYTES = 20 * 1024**3
_MIN_INODES = 100_000
_LOCAL_FILESYSTEMS = frozenset(
    {"ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "zfs", "overlay"}
)
_REQUIREMENTS_DIR = Path(__file__).resolve().parents[2] / "requirements"


@dataclass(frozen=True)
class Recipe:
    id: str
    version: int
    requirements_file: Path
    python_version: tuple[int, int]
    backend: str
    indexes: tuple[str, ...]
    requirements: Mapping[str, bool | str]
    test_precision: Literal["fp32", "bf16", "fp16"]


_ROCM_GFX1100 = Recipe(
    id="rocm-gfx1100-v1",
    version=1,
    requirements_file=_REQUIREMENTS_DIR / "rocm-gfx1100.txt",
    python_version=(3, 14),
    backend="rocm",
    indexes=("https://download.pytorch.org/whl/rocm10.0",),
    requirements=MappingProxyType(
        {
            "torch_hip": True,
            "bf16": True,
            "device_name_regex": r"Radeon.*7900 XTX",
        }
    ),
    test_precision="bf16",
)

_CUDA_CU126 = Recipe(
    id="cuda-cu126-v1",
    version=1,
    requirements_file=_REQUIREMENTS_DIR / "cuda-cu126.txt",
    python_version=(3, 14),
    backend="cuda",
    indexes=("https://download.pytorch.org/whl/cu126",),
    requirements=MappingProxyType({"torch_hip": False}),
    test_precision="fp16",
)

_XPU_ARC = Recipe(
    id="xpu-arc-v1",
    version=1,
    requirements_file=_REQUIREMENTS_DIR / "xpu-arc.txt",
    python_version=(3, 14),
    backend="xpu",
    indexes=("https://download.pytorch.org/whl/xpu",),
    requirements=MappingProxyType(
        {
            "torch_hip": False,
            "device_name_regex": r"(?i)Intel.*Arc.*\b(?:A\d{3}|B\d{2,3})\b",
        }
    ),
    test_precision="fp32",
)


def get_recipe(name: str) -> Recipe:
    """Select an explicit vendor recipe; never infer one from host hardware."""
    recipes = {item.id: item for item in (_ROCM_GFX1100, _CUDA_CU126, _XPU_ARC)}
    try:
        return recipes[name]
    except KeyError as error:
        raise ValueError(f"unknown runtime recipe: {name}") from error


def _mount_path(raw: str) -> Path:
    """Decode the kernel's octal-escaped mountpoint, not shell escape syntax."""
    return Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), raw))


def _filesystem_for(path: Path) -> tuple[str, Path]:
    """Find the deepest mount containing a resolved existing path."""
    best: tuple[str, Path] | None = None
    try:
        with Path("/proc/self/mountinfo").open(encoding="utf-8") as mounts:
            for line in mounts:
                fields = line.split()
                if "-" not in fields or len(fields) < 7:
                    continue
                separator = fields.index("-")
                if separator + 1 >= len(fields) or len(fields) < 5:
                    continue
                mount = _mount_path(fields[4])
                if path.is_relative_to(mount) and (
                    best is None or len(mount.parts) > len(best[1].parts)
                ):
                    best = (fields[separator + 1], mount)
    except OSError as exc:
        raise ValueError(f"cannot inspect local filesystem mounts: {exc}") from exc
    if best is None:
        raise ValueError(
            f"no mounted filesystem contains runtime root ancestor: {path}"
        )
    return best


def preflight_runtime_root(root: Path) -> dict:
    """Check storage without creating any directories or writing probe files."""
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "x86_64",
        "amd64",
    }:
        raise ValueError("runtime recipe requires Linux x86_64")
    selected = resolve_runtime_dir(root)
    ancestor = selected
    while not ancestor.exists():
        if ancestor == ancestor.parent:
            raise ValueError(f"no existing ancestor for runtime root: {selected}")
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise ValueError(f"runtime root ancestor is not a directory: {ancestor}")
    if not os.access(ancestor, os.W_OK | os.X_OK, effective_ids=True):
        raise ValueError(f"runtime root ancestor is not writable: {ancestor}")
    filesystem, mount = _filesystem_for(ancestor)
    if filesystem not in _LOCAL_FILESYSTEMS:
        raise ValueError(
            f"runtime root requires local filesystem, found {filesystem} at {mount}"
        )
    try:
        capacity = os.statvfs(ancestor)
    except OSError as exc:
        raise ValueError(f"cannot inspect runtime root capacity: {exc}") from exc
    if capacity.f_flag & os.ST_RDONLY:
        raise ValueError(f"runtime root filesystem is read-only at {mount}")
    available_bytes = capacity.f_bavail * capacity.f_frsize
    available_inodes = capacity.f_favail
    if available_bytes < _MIN_BYTES:
        raise ValueError(
            f"runtime root needs at least {_MIN_BYTES} available bytes; found {available_bytes}"
        )
    if available_inodes < _MIN_INODES:
        raise ValueError(
            f"runtime root needs at least {_MIN_INODES} available inodes; found {available_inodes}"
        )
    return {
        "root": str(selected),
        "existing_ancestor": str(ancestor),
        "filesystem": filesystem,
        "mount_point": str(mount),
        "available_bytes": available_bytes,
        "available_inodes": available_inodes,
    }
