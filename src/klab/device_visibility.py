"""Confine a process to the one accelerator it trains on.

On a host with several Intel Arc cards, a process that can see more than one
card also holds its device allocations in host memory: 8 GiB allocated on one
B70 took 8.2 GiB of host RAM with both cards visible and 0.2 GiB with only that
card visible. The visible set is fixed when the device runtime first
enumerates devices, so it has to be chosen before then.

Lab chooses it for XPU: :func:`confine` sets ``ZE_AFFINITY_MASK`` to the
configured physical device before anything in the process enumerates XPU
devices, and :func:`visible_index` maps the configured index to the one the
process sees. A mask set by the user is left alone; the configured index then
counts within the cards the user made visible, as it always has. Each
secondary training device runs in its own process, confined the same way
(:func:`child_environment`). Other backends are not confined.

Imports no Torch.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping

VISIBILITY_VARIABLE = {"xpu": "ZE_AFFINITY_MASK"}
# Set by Lab beside the mask: "backend:physical_index".
MARKER = "KLAB_CONFINED_DEVICE"

_enumerated: set[str] = set()


def note_enumerated(backend: str) -> None:
    """Record that this process asked the device runtime about ``backend``."""
    _enumerated.add(backend)


def _marker() -> tuple[str, int] | None:
    value = os.environ.get(MARKER)
    if not value:
        return None
    backend, _, index = value.partition(":")
    return backend, int(index)


def confine(backend: str, physical_index: int) -> None:
    """Make ``physical_index`` the only ``backend`` device this process sees."""
    variable = VISIBILITY_VARIABLE.get(backend)
    if variable is None:
        return
    marker = _marker()
    if marker is not None:
        if marker != (backend, physical_index):
            raise ValueError(
                f"this process is confined to {marker[0]} device {marker[1]}; "
                f"{backend} device {physical_index} needs its own process"
            )
        return
    if variable in os.environ:
        return
    if backend in _enumerated:
        warnings.warn(
            f"{backend} devices were enumerated before Lab could confine this "
            f"process to device {physical_index}; it sees every card, and its "
            "device memory is also held in host RAM on hosts with several Arc "
            f"cards. Set {variable}={physical_index} before starting it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    os.environ[variable] = str(physical_index)
    os.environ[MARKER] = f"{backend}:{physical_index}"


def visible_index(backend: str, index: int) -> int:
    """The index this process uses for configured ``backend`` device ``index``."""
    marker = _marker()
    if marker is None or marker[0] != backend:
        return index
    if index != marker[1]:
        raise ValueError(
            f"this process is confined to {backend} device {marker[1]} and "
            f"cannot use device {index}"
        )
    return 0


def physical_index(backend: str, visible: int) -> int:
    """The configured index of the ``backend`` device this process sees as ``visible``."""
    marker = _marker()
    if marker is None or marker[0] != backend:
        return visible
    if visible != 0:
        raise ValueError(f"a confined process sees one {backend} device, index 0")
    return marker[1]


def unconfined_environment() -> dict[str, str]:
    """This process's environment without the confinement Lab applied to it."""
    environment = dict(os.environ)
    marker = environment.pop(MARKER, None)
    if marker is not None:
        variable = VISIBILITY_VARIABLE.get(marker.partition(":")[0])
        if variable is not None:
            environment.pop(variable, None)
    return environment


def child_environment(
    backend: str, physical_index: int, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The environment for a new process that trains on one device."""
    environment = dict(os.environ if base is None else base)
    variable = VISIBILITY_VARIABLE.get(backend)
    if variable is None or (variable in environment and MARKER not in environment):
        # Not confined by Lab: the child counts devices the way the parent does.
        return environment
    environment[variable] = str(physical_index)
    environment[MARKER] = f"{backend}:{physical_index}"
    return environment
