"""Registry of training objectives: what an optimizer update minimizes.

Every run trains under one objective, named by ``training.objective.kind``. Lab
ships three:

``next_token``
    Cross-entropy on the supervised targets. Pretraining and instruction tuning.
``distill``
    Cross-entropy blended with a KL term toward a frozen teacher's distribution.
``preference``
    Direct preference optimization (DPO) on chosen/rejected reply pairs, against
    the frozen parent checkpoint as the reference policy.

Another installed package adds an objective by declaring an entry point in the
``klab.objectives`` group whose name is the kind and whose value is an
:class:`Objective`::

    [project.entry-points."klab.objectives"]
    "calibrated" = "my_package.calibrated:OBJECTIVE"

This module does not import Torch. Builders import their code when called, and
entry points are loaded only for a kind Lab does not ship. What a built
objective receives is described in ``klab.training.objectives`` and
``docs/post-training.md``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib.metadata import entry_points
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

ENTRY_POINT_GROUP = "klab.objectives"
DEFAULT_KIND = "next_token"
_KIND = re.compile(r"[a-z][a-z0-9_]{0,63}")


@dataclass(frozen=True)
class Objective:
    """One objective kind and what it needs from a run.

    ``build`` receives the validated ``params`` instance (``None`` when the kind
    declares no ``params`` model) and returns a
    :class:`klab.training.objectives.TrainingObjective`.

    ``teacher`` asks for ``training.objective.teacher_checkpoint``. ``reference``
    asks for ``training.parent_checkpoint``, loaded a second time and kept
    frozen. ``pairs`` trains on a ``local_preference`` dataset, whose rows
    arrive as ``[pairs, 2, tokens]`` with the chosen reply first.
    """

    kind: str
    build: Callable[[BaseModel | None], Any]
    params: type[BaseModel] | None = None
    teacher: bool = False
    reference: bool = False
    pairs: bool = False
    source: str = "klab"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not _KIND.fullmatch(self.kind):
            raise ValueError(
                "objective kind must be lowercase letters, digits and underscores"
            )
        if not callable(self.build):
            raise TypeError("objective build must be callable")
        if self.params is not None and not (
            isinstance(self.params, type) and issubclass(self.params, BaseModel)
        ):
            raise TypeError("objective params must be a pydantic model class")


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DistillParams(_Params):
    # Share of the loss taken by the teacher term; the rest is cross-entropy.
    weight: float = Field(default=0.5, gt=0, le=1)
    # Both distributions are softened by this before they are compared.
    temperature: float = Field(default=2.0, gt=0)


class PreferenceParams(_Params):
    # How strongly the policy is held to the reference; smaller moves further.
    beta: float = Field(default=0.1, gt=0)


def _next_token(params):
    from klab.training.objectives import NextToken

    return NextToken()


def _distill(params):
    from klab.training.objectives import Distill

    return Distill(params.weight, params.temperature)


def _preference(params):
    from klab.training.objectives import Preference

    return Preference(params.beta)


BUILTIN: Mapping[str, Objective] = {
    item.kind: item
    for item in (
        Objective(DEFAULT_KIND, _next_token),
        Objective("distill", _distill, DistillParams, teacher=True),
        Objective(
            "preference", _preference, PreferenceParams, reference=True, pairs=True
        ),
    )
}
_EXTERNAL: dict[str, Objective] = {}


def is_builtin(kind: str) -> bool:
    return kind in BUILTIN


def _load_external(kind: str) -> Objective | None:
    matches = [
        entry for entry in entry_points(group=ENTRY_POINT_GROUP) if entry.name == kind
    ]
    if not matches:
        return None
    if len(matches) > 1:
        owners = sorted(
            entry.dist.name if entry.dist is not None else entry.value
            for entry in matches
        )
        raise ValueError(
            f"objective kind {kind!r} is registered by more than one package: "
            + ", ".join(owners)
        )
    entry = matches[0]
    loaded = entry.load()
    if not isinstance(loaded, Objective):
        raise TypeError(
            f"entry point {kind!r} ({entry.value}) must name a klab.objectives.Objective"
        )
    if loaded.kind != kind:
        raise ValueError(
            f"entry point {kind!r} names objective {loaded.kind!r}; the two must match"
        )
    source = (
        f"{entry.dist.name} {entry.dist.version}"
        if entry.dist is not None
        else entry.value
    )
    return replace(loaded, source=source)


def objective(kind: str) -> Objective:
    """Return the objective for ``kind``, loading its entry point when needed."""
    found = BUILTIN.get(kind) or _EXTERNAL.get(kind)
    if found is None and isinstance(kind, str) and _KIND.fullmatch(kind):
        found = _load_external(kind)
        if found is not None:
            _EXTERNAL[kind] = found
    if found is None:
        raise ValueError(
            f"unknown objective kind {kind!r}: Lab ships {', '.join(BUILTIN)}; "
            f"another kind needs an installed package with a "
            f"{ENTRY_POINT_GROUP!r} entry point named '{kind}'"
        )
    return found


def normalized_params(kind: str, params: Any) -> dict[str, Any]:
    """Validate ``params`` for ``kind`` and return them with defaults filled in.

    Storing the filled-in values keeps a config's identity stable when a default
    later changes.
    """
    params = {} if params is None else params
    if not isinstance(params, Mapping):
        raise ValueError("objective params must be a mapping")  # noqa: TRY004
    found = objective(kind)
    if found.params is None:
        if params:
            raise ValueError(f"objective kind {kind!r} accepts no params")
        return {}
    return found.params.model_validate(dict(params)).model_dump(mode="json")


def build(kind: str, params: Mapping[str, Any] | None = None) -> Any:
    """Construct the objective a run trains under."""
    found = objective(kind)
    parsed = (
        None
        if found.params is None
        else found.params.model_validate(dict(params or {}))
    )
    return found.build(parsed)
