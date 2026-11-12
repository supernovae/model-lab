"""Named architecture presets and scaling one to a parameter target.

A preset holds only ``model`` and ``attention`` at a reference size. Scaling
keeps its proportions (head width, feed-forward ratio, key/value head ratio,
latent ratio and width-to-depth ratio) and searches width and depth for the
total parameter count closest to the target. Dataset, tokenizer, training and
runtime settings come from a base run config, never from the preset.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from klab.config.models import AttentionConfig, ModelConfig, StrictModel

PRESET_DIR = Path(__file__).with_name("architecture_presets")
_COUNT = re.compile(r"(\d+(?:\.\d+)?)\s*([kmb]?)", re.IGNORECASE)
_MULTIPLIER = {"": 1, "k": 10**3, "m": 10**6, "b": 10**9}
# Width is searched up to this many heads and depth within this factor of the
# preset's width-to-depth ratio.
_MAX_HEADS = 512
_DEPTH_RANGE = 2.0
# A candidate this close to the target competes on proportions, not on count.
_TOLERANCE = 0.02
# Further than this from the target is reported as unreachable, not returned.
_MAX_ERROR = 0.10


class ArchitecturePreset(StrictModel):
    """Model and attention settings at a reference size."""

    description: str
    model: ModelConfig
    attention: AttentionConfig = AttentionConfig()


@dataclass(frozen=True)
class ScaledArchitecture:
    preset: str
    model: ModelConfig
    attention: AttentionConfig
    parameters: int
    target: int

    @property
    def relative_error(self) -> float:
        return (self.parameters - self.target) / self.target


def preset_names() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in PRESET_DIR.glob("*.yaml")))


def load_preset(name_or_path: str) -> tuple[str, ArchitecturePreset]:
    """Load a shipped preset by name, or any preset file by path."""
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        if name_or_path not in preset_names():
            raise ValueError(
                f"unknown architecture preset {name_or_path!r}: choose one of "
                f"{', '.join(preset_names())} or pass a preset YAML path"
            )
        path = PRESET_DIR / f"{name_or_path}.yaml"
    if not path.is_file():
        raise ValueError(f"architecture preset file does not exist: {path}")
    return path.stem, ArchitecturePreset.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def parse_parameter_count(text: str) -> int:
    """Read ``300M``, ``1.2B``, ``50k`` or a plain integer."""
    match = _COUNT.fullmatch(text.strip().replace("_", "").replace(",", ""))
    if match is None:
        raise ValueError(
            f"invalid parameter count {text!r}: use an integer or a value such "
            "as 300M or 1B"
        )
    count = round(float(match.group(1)) * _MULTIPLIER[match.group(2).lower()])
    if count <= 0:
        raise ValueError("parameter count must be positive")
    return count


def count_parameters(model: ModelConfig, attention: AttentionConfig) -> int:
    """Stored parameters of the configured decoder, counted from shapes."""
    from klab.model.inspection import named_tensor_inventory

    return sum(
        spec.numel
        for spec in named_tensor_inventory(model, attention).values()
        if spec.alias_of is None
    )


def _resized(
    preset: ArchitecturePreset,
    heads: int,
    layers: int,
    vocab_size: int,
    max_seq_len: int,
) -> tuple[ModelConfig, AttentionConfig]:
    reference = preset.model
    head_dim = reference.hidden_dim // reference.num_heads
    hidden = heads * head_dim
    model: dict[str, Any] = reference.model_dump(mode="json")
    model.update(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        hidden_dim=hidden,
        num_heads=heads,
        num_layers=layers,
        # Feed-forward width keeps the preset's ratio, rounded to a multiple of 8.
        ffn_dim=max(
            8, 8 * round(reference.ffn_dim * hidden / reference.hidden_dim / 8)
        ),
    )
    if reference.num_kv_heads is not None:
        model["num_kv_heads"] = heads * reference.num_kv_heads // reference.num_heads
    depth = layers / reference.num_layers
    if reference.memory_block_index is not None:
        model["memory_block_index"] = min(
            layers - 1, int(reference.memory_block_index * depth)
        )
    if reference.memory_block_indices:
        model["memory_block_indices"] = sorted(
            {min(layers - 1, int(i * depth)) for i in reference.memory_block_indices}
        )
    attention: dict[str, Any] = preset.attention.model_dump(mode="json")
    if preset.attention.latent_dim is not None:
        per_head = max(
            2, 2 * round(preset.attention.latent_dim / reference.num_heads / 2)
        )
        attention["latent_dim"] = heads * per_head
    return ModelConfig.model_validate(model), AttentionConfig.model_validate(attention)


def scale_preset(
    name: str,
    preset: ArchitecturePreset,
    target: int,
    *,
    vocab_size: int | None = None,
    max_seq_len: int | None = None,
    num_layers: int | None = None,
) -> ScaledArchitecture:
    """Return the preset resized to the parameter count closest to ``target``."""
    reference = preset.model
    vocab_size = vocab_size or reference.vocab_size
    max_seq_len = max_seq_len or reference.max_seq_len
    # Heads move in steps that keep the key/value head count whole.
    step = (
        1
        if reference.num_kv_heads is None
        else reference.num_heads // reference.num_kv_heads
    )
    aspect = reference.num_heads / reference.num_layers
    best: tuple[tuple[float, int, float, float], ScaledArchitecture] | None = None
    failure: Exception | None = None
    for heads in range(step, _MAX_HEADS + 1, step):
        natural = max(1, round(heads / aspect))
        if num_layers is not None:
            depths = {num_layers}
        else:
            low = max(1, math.floor(natural / _DEPTH_RANGE))
            high = max(low + 1, math.ceil(natural * _DEPTH_RANGE))
            try:
                first = count_parameters(
                    *_resized(preset, heads, low, vocab_size, max_seq_len)
                )
                second = count_parameters(
                    *_resized(preset, heads, low + 1, vocab_size, max_seq_len)
                )
            except ValueError as error:
                failure = error
                continue
            if first > target * (1 + _TOLERANCE) and best is not None:
                # Wider models only grow; nothing further can be closer.
                break
            per_layer = max(1, second - first)
            solved = low + (target - first) / per_layer
            depths = {
                min(high, max(low, candidate))
                for candidate in (math.floor(solved), math.ceil(solved))
            }
        for layers in sorted(depths):
            try:
                model, attention = _resized(
                    preset, heads, layers, vocab_size, max_seq_len
                )
                parameters = count_parameters(model, attention)
            except ValueError as error:
                failure = error
                continue
            error = abs(parameters - target) / target
            shape = abs(math.log(layers / natural))
            # Among sizes close enough to the target, prefer an even head count
            # (a conventional width), then the preset's width-to-depth ratio.
            rank = (max(error, _TOLERANCE), heads % 2, shape, error)
            if best is None or rank < best[0]:
                best = (
                    rank,
                    ScaledArchitecture(name, model, attention, parameters, target),
                )
    if best is None:
        raise ValueError(
            f"preset {name!r} cannot be scaled to {target} parameters"
            + (f": {failure}" if failure is not None else "")
        )
    if abs(best[1].relative_error) > _MAX_ERROR:
        raise ValueError(
            f"preset {name!r} cannot reach {target} parameters with a "
            f"{vocab_size}-token vocabulary: the closest size keeping its "
            f"proportions has {best[1].parameters}"
        )
    return best[1]


def describe(scaled: ScaledArchitecture) -> dict[str, Any]:
    model = scaled.model
    return {
        "preset": scaled.preset,
        "target_parameters": scaled.target,
        "parameters": scaled.parameters,
        "relative_error": round(scaled.relative_error, 6),
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "ffn_dim": model.ffn_dim,
        "vocab_size": model.vocab_size,
        "max_seq_len": model.max_seq_len,
    }


def derive_run_config(
    preset: str,
    target: int,
    base: Path,
    output: Path,
    *,
    vocab_size: int | None = None,
    max_seq_len: int | None = None,
    num_layers: int | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Write ``base`` with its model and attention replaced by the scaled preset.

    The vocabulary and context default to the base config's, because its
    tokenizer and sequence length are what the run will actually use.
    """
    from klab.config.loading import load_config
    from klab.derivation import derive_config

    preset_name, loaded = load_preset(preset)
    base_config = load_config(Path(base))
    scaled = scale_preset(
        preset_name,
        loaded,
        target,
        vocab_size=vocab_size or base_config.model.vocab_size,
        max_seq_len=max_seq_len or base_config.model.max_seq_len,
        num_layers=num_layers,
    )
    settings: dict[str, Any] = {
        "model": scaled.model.model_dump(mode="json"),
        "attention": scaled.attention.model_dump(mode="json"),
    }
    if name is not None:
        settings["name"] = name
    receipt = derive_config(Path(base), Path(output), settings)
    return {
        **describe(scaled),
        "output": receipt["output"]["path"],
        "config_sha256": receipt["config_sha256"],
    }
