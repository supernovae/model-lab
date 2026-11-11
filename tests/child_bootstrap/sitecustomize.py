"""Test-suite bootstrap for child interpreters launched by tests.

``tests/conftest.py`` puts this directory first on ``PYTHONPATH`` and sets
``KLAB_TEST_HOST_RAM=total:available`` for every test, so a fresh
``python -c``/``-m klab`` child plans against the same fixed host as the
parent process. Python imports ``sitecustomize`` at startup only from
``sys.path``; this directory is never there outside pytest, so shipped code
always measures real RAM (``host_capacity.measure_memory`` has no override).

The pin is applied lazily, when a child first imports
``klab.host_capacity``, so children that never touch Model Lab pay no
startup cost.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys

_MODULE = "klab.host_capacity"
_PINNED = os.environ.get("KLAB_TEST_HOST_RAM")


def _pin(module: object) -> None:
    total, _, available = str(_PINNED).partition(":")
    measure = module.measure_memory  # type: ignore[attr-defined]

    def pinned_measure_memory() -> object:
        return measure()._replace(total=int(total), available=int(available))

    module.measure_memory = pinned_measure_memory  # type: ignore[attr-defined]


class _PinningLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader) -> None:
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module) -> None:
        self._loader.exec_module(module)
        _pin(module)


class _PinningFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _PinningLoader(spec.loader)
        return spec


if _PINNED:
    sys.meta_path.insert(0, _PinningFinder())
