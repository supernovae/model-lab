"""Build and load a decoder from an exported directory, without Lab installed.

``klab weights export --format hf_remote_code`` copies this module into the
export beside the decoder modules it imports. The export's ``config.json``
holds the architecture as plain JSON under ``klab_architecture``: the model
and attention settings and, when ``model.layers`` overrides some blocks, each
block's resolved settings. This module turns that JSON back into the objects
:class:`DenseLM` reads, so the exported model is built by the same code Lab
trains and generates with. It needs only Torch and safetensors.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from klab.model.transformer import DenseLM

WEIGHTS_FILE = "model.safetensors"
WEIGHT_PREFIX = "model."


def _frozen(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


class Settings:
    """Read-only attribute view of one exported settings mapping."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        for name, value in values.items():
            if name == "memory_token_mapping" and isinstance(value, Mapping):
                value = Settings(value)
            object.__setattr__(self, name, _frozen(value))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("exported settings are read-only")


class ModelSettings(Settings):
    """Model settings plus the per-block settings ``DenseLM`` asks for."""

    def __init__(
        self,
        values: Mapping[str, Any],
        layers: tuple[tuple[Settings, Settings], ...] | None = None,
    ) -> None:
        super().__init__(values)
        object.__setattr__(self, "_layers", layers)

    def layer_specs(self, attention: Settings) -> tuple[tuple[Any, Any], ...]:
        return self._layers or ((self, attention),) * self.num_layers


def build_model(architecture: Mapping[str, Any]) -> DenseLM:
    """Construct an untrained decoder from a ``klab_architecture`` mapping."""
    attention = Settings(architecture["attention"])
    layers = tuple(
        (ModelSettings(layer["model"]), Settings(layer["attention"]))
        for layer in architecture.get("layers", ())
    )
    model = ModelSettings(architecture["model"], layers or None)
    return DenseLM(model, attention)  # type: ignore[arg-type]


def load_model(directory: str | Path, device: str = "cpu") -> DenseLM:
    """Load the exported decoder in evaluation mode."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model = build_model(config["klab_architecture"])
    weights = {
        name.removeprefix(WEIGHT_PREFIX): tensor
        for name, tensor in load_file(directory / WEIGHTS_FILE).items()
    }
    if model.config.tie_embeddings:
        weights["output.weight"] = weights["embedding.weight"]
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()
