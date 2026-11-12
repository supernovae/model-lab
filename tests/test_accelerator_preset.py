"""The accelerator preset loads, plans and declares no per-step dead time."""

from __future__ import annotations

from pathlib import Path

from klab.config.loading import load_config
from klab.model.inspection import inspection_report

PRESET = Path("configs/accelerator_scale.yaml")


def test_accelerator_preset_plans_and_uses_token_cadences() -> None:
    config = load_config(PRESET)
    assert inspection_report(config)["total"] == 187_720_704
    assert config.runtime.compile_transformer_blocks
    assert not config.runtime.memory.activation_checkpointing.enabled
    # Validation and checkpoints follow trained tokens, never a step count,
    # and an improving validation loss alone does not write a checkpoint.
    assert config.evaluation.every_steps is None
    assert config.checkpoint.every_steps is None
    assert config.evaluation.every_tokens and config.checkpoint.every_tokens
    assert not config.checkpoint.save_on_new_best
    assert config.checkpoint.background
