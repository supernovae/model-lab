from __future__ import annotations

import pytest
import torch

from klab.config.models import AttentionConfig, ModelConfig
from klab.evaluation.byte_memory_transfer import transfer_byte_memory
from klab.model.transformer import DenseLM


def test_byte_memory_requires_prepared_addresses() -> None:
    config = ModelConfig(
        vocab_size=512,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        max_seq_len=16,
        memory="byte",
        memory_table_size=31,
        memory_ngram_size=3,
        memory_dim=7,
    )
    model = DenseLM(config, AttentionConfig())
    ids = torch.randint(0, 512, (2, 8))
    with pytest.raises(ValueError, match="prepared byte addresses"):
        model(ids)
    logits = model(ids, byte_addresses=torch.randint(0, 31, (2, 8)))
    logits.mean().backward()
    assert logits.shape == (2, 8, 512)


def test_byte_memory_transfer_rejects_non_adapter_state() -> None:
    config = ModelConfig(
        vocab_size=512,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        max_seq_len=16,
        memory="byte",
        memory_table_size=31,
        memory_ngram_size=3,
        memory_dim=7,
    )
    source = DenseLM(config, AttentionConfig())
    target = DenseLM(config, AttentionConfig())
    transfer_byte_memory(source.state_dict(), target)
    assert torch.equal(source.memory.table.weight, target.memory.table.weight)
    with pytest.raises(ValueError, match="exact byte-memory"):
        transfer_byte_memory({}, target)
