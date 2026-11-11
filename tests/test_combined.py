from __future__ import annotations

from pathlib import Path

import torch

from klab.config.loading import load_config
from klab.model.transformer import DenseLM


def test_combined_mla_moe_byte_memory_forward() -> None:
    config = load_config(Path("configs/smoke_combined_cpu.yaml"))
    model = DenseLM(config.model, config.attention)
    logits = model(
        torch.randint(0, config.model.vocab_size, (2, 8)),
        byte_addresses=torch.randint(0, config.model.memory_table_size, (2, 8)),
    )
    logits.mean().backward()
    assert logits.shape == (2, 8, config.model.vocab_size)
