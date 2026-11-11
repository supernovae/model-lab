from __future__ import annotations

import pytest
import torch

from klab.config.models import AttentionConfig, ModelConfig
from klab.model.portable_engram import (
    PortableEngramAdapter,
    export_portable_engram,
    load_portable_engram,
)
from klab.model.transformer import DenseLM


def test_portable_package_freezes_table_and_trains_adapter(tmp_path) -> None:
    table = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    path = tmp_path / "memory.engram"
    manifest = export_portable_engram(table, path, ngram_size=3)
    package = load_portable_engram(path)
    assert package.manifest == manifest
    adapter = PortableEngramAdapter(package, hidden_dim=8)
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    output = adapter(hidden, torch.tensor([[0, 1, 2], [3, 4, 5]]))
    output.square().mean().backward()
    assert adapter.embedding.weight.requires_grad is False
    assert adapter.output.weight.grad is not None
    assert adapter.last_diagnostics is not None
    assert int(adapter.last_diagnostics.lookup_count) == 6


def test_portable_memory_is_selectable_and_frozen(tmp_path) -> None:
    path = tmp_path / "memory.engram"
    export_portable_engram(torch.zeros(17, 5), path, ngram_size=3)
    config = ModelConfig(
        vocab_size=512,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        max_seq_len=8,
        memory="portable",
        memory_table_size=17,
        memory_ngram_size=3,
        memory_dim=5,
        memory_package_path=path,
    )
    model = DenseLM(config, AttentionConfig())
    logits = model(
        torch.randint(0, 512, (2, 4)),
        byte_addresses=torch.randint(0, 17, (2, 4)),
    )
    logits.mean().backward()
    assert model.memory is not None
    assert model.memory.embedding.weight.requires_grad is False
    assert model.memory.output.weight.grad is not None


def test_portable_package_path_requires_portable_memory(tmp_path) -> None:
    path = tmp_path / "memory.engram"
    export_portable_engram(torch.zeros(4, 2), path, ngram_size=2)
    settings = {
        "vocab_size": 512,
        "hidden_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "ffn_dim": 16,
        "max_seq_len": 8,
        "memory_package_path": path,
    }
    with pytest.raises(ValueError):
        ModelConfig(**settings)


def test_portable_package_rejects_conflicting_rewrite(tmp_path) -> None:
    path = tmp_path / "memory.engram"
    export_portable_engram(torch.zeros(4, 2), path, ngram_size=2)
    with pytest.raises(FileExistsError, match="conflicting"):
        export_portable_engram(torch.ones(4, 2), path, ngram_size=2)
