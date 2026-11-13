"""Registry of the attention, feed-forward and memory kinds a decoder is built from.

A decoder block looks its attention and feed-forward modules up here by kind, and
the model looks its memory up the same way. Lab's own kinds are built by
:mod:`klab.component_builders`.
Another installed package adds a kind by declaring an entry point in the
``klab.components`` group whose name is ``SLOT.KIND`` and whose value is a
:class:`Component`::

    [project.entry-points."klab.components"]
    "attention.cosine" = "my_package.cosine:COMPONENT"

This module does not import Torch. Builders import their modules when called, and
entry points are loaded only for a kind Lab does not ship, so configuration
commands stay Torch-free for built-in kinds.

What a module may offer beyond ``forward`` is described in ``docs/components.md``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from klab.component_builders import BUILDERS

if TYPE_CHECKING:
    from torch import nn

    from klab.config.models import AttentionConfig, ModelConfig

ENTRY_POINT_GROUP = "klab.components"
Slot = Literal["attention", "ffn", "memory"]
SLOTS: tuple[Slot, ...] = ("attention", "ffn", "memory")
_KIND = re.compile(r"[a-z][a-z0-9_]{0,63}")

Builder = Callable[["ModelConfig", "AttentionConfig", BaseModel | None], "nn.Module"]


@dataclass(frozen=True)
class Component:
    """One buildable kind for one slot of the decoder.

    ``build`` receives the layer's model and attention settings and the validated
    ``params`` instance (``None`` when the kind declares no ``params`` model).
    """

    slot: Slot
    kind: str
    build: Builder
    params: type[BaseModel] | None = None
    source: str = "klab"

    def __post_init__(self) -> None:
        if self.slot not in SLOTS:
            raise ValueError(f"unknown component slot: {self.slot!r}")
        if not isinstance(self.kind, str) or not _KIND.fullmatch(self.kind):
            raise ValueError(
                "component kind must be lowercase letters, digits and underscores"
            )
        if not callable(self.build):
            raise TypeError("component build must be callable")
        if self.params is not None and not (
            isinstance(self.params, type) and issubclass(self.params, BaseModel)
        ):
            raise TypeError("component params must be a pydantic model class")


BUILTIN: Mapping[tuple[str, str], Component] = {
    (slot, kind): Component(slot, kind, builder)  # type: ignore[arg-type]
    for (slot, kind), builder in BUILDERS.items()
}
_EXTERNAL: dict[tuple[str, str], Component] = {}


def builtin_kinds(slot: Slot) -> tuple[str, ...]:
    return tuple(kind for owner, kind in BUILTIN if owner == slot)


def is_builtin(slot: Slot, kind: str) -> bool:
    return (slot, kind) in BUILTIN or (slot == "memory" and kind == "none")


def _load_external(slot: Slot, kind: str) -> Component | None:
    name = f"{slot}.{kind}"
    matches = [
        entry for entry in entry_points(group=ENTRY_POINT_GROUP) if entry.name == name
    ]
    if not matches:
        return None
    if len(matches) > 1:
        owners = sorted(
            entry.dist.name if entry.dist is not None else entry.value
            for entry in matches
        )
        raise ValueError(
            f"{slot} kind {kind!r} is registered by more than one package: "
            + ", ".join(owners)
        )
    entry = matches[0]
    loaded = entry.load()
    if not isinstance(loaded, Component):
        raise TypeError(
            f"entry point {name!r} ({entry.value}) must name a klab.components.Component"
        )
    if (loaded.slot, loaded.kind) != (slot, kind):
        raise ValueError(
            f"entry point {name!r} names a component for "
            f"{loaded.slot}.{loaded.kind}; the two must match"
        )
    source = (
        f"{entry.dist.name} {entry.dist.version}"
        if entry.dist is not None
        else entry.value
    )
    return replace(loaded, source=source)


def component(slot: Slot, kind: str) -> Component:
    """Return the component for ``kind``, loading its entry point when needed."""
    key = (slot, kind)
    found = BUILTIN.get(key) or _EXTERNAL.get(key)
    if found is None and isinstance(kind, str) and _KIND.fullmatch(kind):
        found = _load_external(slot, kind)
        if found is not None:
            _EXTERNAL[key] = found
    if found is None:
        raise ValueError(
            f"unknown {slot} kind {kind!r}: Lab ships "
            f"{', '.join(builtin_kinds(slot))}; another kind needs an installed "
            f"package with a {ENTRY_POINT_GROUP!r} entry point named "
            f"'{slot}.{kind}'"
        )
    return found


def normalized_params(slot: Slot, kind: str, params: Any) -> dict[str, Any]:
    """Validate ``params`` for ``kind`` and return them with defaults filled in.

    Built-in kinds are configured through their typed fields and accept none.
    Storing the filled-in values keeps a config's identity stable when a package
    later changes a default.
    """
    params = {} if params is None else params
    if not isinstance(params, Mapping):
        raise ValueError(f"{slot} params must be a mapping")  # noqa: TRY004
    if is_builtin(slot, kind):
        if params:
            raise ValueError(
                f"built-in {slot} kind {kind!r} is configured through its own "
                "fields and accepts no params"
            )
        return {}
    found = component(slot, kind)
    if found.params is None:
        if params:
            raise ValueError(f"{slot} kind {kind!r} accepts no params")
        return {}
    return found.params.model_validate(dict(params)).model_dump(mode="json")


def build(
    slot: Slot,
    kind: str,
    model: ModelConfig,
    attention: AttentionConfig,
    params: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Construct the module for one slot of one layer."""
    found = component(slot, kind)
    parsed = (
        None
        if found.params is None
        else found.params.model_validate(dict(params or {}))
    )
    return found.build(model, attention, parsed)


def sources(kinds: Mapping[Slot, set[str] | frozenset[str]]) -> dict[str, str]:
    """Name the package and version behind each external kind in use."""
    return {
        f"{slot}.{kind}": component(slot, kind).source
        for slot in SLOTS
        for kind in sorted(kinds.get(slot, ()))
        if not is_builtin(slot, kind)
    }
