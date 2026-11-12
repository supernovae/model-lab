from __future__ import annotations

from pathlib import Path

from klab.config.loading import load_config


def test_instruction_reference_config_declares_reproducible_training_contract() -> None:
    config = load_config(Path("configs/instruction_100m.yaml"))

    assert config.dataset.source == "synthetic"
    assert config.dataset.synthetic_task == "instruction_reference"
    assert config.model.vocab_size == 8192
    assert config.model.hidden_dim == 896
    assert config.training.max_tokens == 20_000_000
    assert (
        config.training.micro_batch_size * config.training.gradient_accumulation == 16
    )
