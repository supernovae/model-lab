from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models

from klab.address_hash import engram_ngram_address, engram_table_sizes
from klab.config.models import AttentionConfig, ModelConfig
from klab.memory_tokenizer import build_memory_token_map
from klab.model.transformer import DenseLM
from klab.training.manifest import architecture_sha256


def model_config(tmp_path: Path, mode: str, **changes) -> ModelConfig:
    vocabulary = {f"token{i}": i for i in range(260)}
    del vocabulary["token4"], vocabulary["token5"], vocabulary["token6"]
    vocabulary.update({"Value": 4, "value": 5, "café": 6})
    path = tmp_path / "tokenizer.json"
    Tokenizer(models.WordLevel(vocabulary)).save(str(path))
    mapping = build_memory_token_map(path, mode=mode)
    return ModelConfig(
        **(
            {
                "vocab_size": 260,
                "hidden_dim": 16,
                "num_layers": 2,
                "num_heads": 2,
                "ffn_dim": 32,
                "max_seq_len": 16,
                "memory": "ngram",
                "memory_table_size": 31,
                "memory_table_sizes": engram_table_sizes(31, 4),
                "memory_ngram_size": 3,
                "memory_dim": 4,
                "memory_ngram_orders": (2, 3),
                "memory_hash_heads": 2,
                "memory_addressing": "prime_xor_v1",
                "memory_token_mapping": mapping.to_dict(),
                "memory_injection": "after_block",
                "memory_block_index": 0,
                "memory_variant": "context_key",
                "memory_conv_kernel_size": 4,
                "memory_independent_init": True,
                "memory_padding_token_id": 2,
            }
            | changes
        )
    )


@pytest.mark.parametrize("mode", ["identity", "deepseek_v1", "deepseek_safe_v1"])
@pytest.mark.parametrize("transform", ["identity", "signed_sqrt"])
def test_prime_mapping_cached_chunks_and_prefix_are_causal(tmp_path, mode, transform):
    config = model_config(tmp_path, mode, memory_gate_transform=transform)
    model = DenseLM(config, AttentionConfig()).eval()
    ids = torch.tensor([[4, 6, 5, 0, 2, 3], [8, 4, 5, 6, 7, 9]])
    full = model(ids)
    torch.testing.assert_close(model(ids[:, :3]), full[:, :3])
    first, cache = model.forward_cached(ids[:, :1], cache_capacity=8)
    second, cache = model.forward_cached(ids[:, 1:4], cache=cache)
    last, _ = model.forward_cached(ids[:, 4:], cache=cache)
    torch.testing.assert_close(torch.cat([first, second, last], dim=1), full)
    fresh, _ = model.forward_cached(ids[:, 2:], cache_capacity=8)
    torch.testing.assert_close(fresh, model(ids[:, 2:]))
    memory = model.memory
    mapped = [config.memory_token_mapping.token_ids[i] for i in ids[0].tolist()]
    pad = config.memory_token_mapping.token_ids[2]
    for order_index, order in enumerate(config.memory_ngram_orders):
        for head in range(2):
            size = config.memory_table_sizes[order_index * 2 + head]
            expected = [
                engram_ngram_address(
                    [mapped[t - k] if t >= k else pad for k in range(order)],
                    size,
                    memory.multipliers[:order],
                )
                for t in range(len(mapped))
            ]
            assert memory.addresses(ids, order, head)[0].tolist() == expected
    full.sum().backward()
    assert torch.isfinite(memory.table.weight.grad).all()
    assert memory.table.weight.grad.abs().sum() > 0


def test_mapping_ablation_keeps_hash_coefficients_and_rejects_identity_changes(
    tmp_path,
):
    identity = model_config(tmp_path, "identity")
    compressed = model_config(tmp_path, "deepseek_v1")
    left = DenseLM(identity, AttentionConfig())
    right = DenseLM(compressed, AttentionConfig())
    assert left.memory.multipliers == right.memory.multipliers
    assert (
        compressed.memory_token_mapping.token_ids[4]
        == compressed.memory_token_mapping.token_ids[5]
    )
    assert (
        identity.memory_token_mapping.token_ids[4]
        != identity.memory_token_mapping.token_ids[5]
    )

    # Architecture identity binds the full map, scheme, seed and ordered primes.
    def digest(config):
        return architecture_sha256(
            {
                "model": config.model_dump(mode="json"),
                "attention": AttentionConfig().model_dump(mode="json"),
            }
        )

    original = digest(identity)
    assert digest(compressed) != original
    assert digest(identity.model_copy(update={"memory_hash_seed": 17})) != original
    assert (
        digest(identity.model_copy(update={"memory_table_sizes": (31, 37, 41, 47)}))
        != original
    )
    with pytest.raises(ValueError, match="prime|distinct"):
        model_config(tmp_path, "identity", memory_table_sizes=(31, 31, 41, 43))


@pytest.mark.parametrize("transform", ["identity", "signed_sqrt"])
def test_multiport_histories_and_gradients_are_independent(tmp_path, transform):
    config = model_config(
        tmp_path,
        "deepseek_v1",
        memory_block_index=None,
        memory_block_indices=(0, 1),
        memory_table_sizes=engram_table_sizes(31, 8),
        memory_gate_transform=transform,
    )
    model = DenseLM(config, AttentionConfig()).eval()
    ids = torch.tensor([[4, 6, 5, 2, 8, 9]])
    full = model(ids)
    before, cache = model.forward_cached(ids[:, :3], cache_capacity=8)
    after, cache = model.forward_cached(ids[:, 3:], cache=cache)
    torch.testing.assert_close(torch.cat([before, after], dim=1), full)
    assert cache.memory_histories["1"] is not cache.memory_history
    assert model.memory.multipliers != model.extra_memories["1"].multipliers
    assert model.memory.table_sizes != model.extra_memories["1"].table_sizes
    full.sum().backward()
    assert model.memory.table.weight.grad.abs().sum() > 0
    assert model.extra_memories["1"].table.weight.grad.abs().sum() > 0
    metrics = model.architecture_metric_tensors()
    assert "engram/port/0/gate_mean" in metrics and "engram/port/1/gate_mean" in metrics
    cache.memory_histories.clear()
    with pytest.raises(ValueError, match="extra memory histories"):
        model.forward_cached(ids[:, :1], cache=cache)


def test_memory_map_rejects_different_supplied_tokenizer_object(tmp_path):
    from tokenizers import normalizers

    from klab.config.loading import load_config
    from klab.data.packing import prepare_data
    from klab.data.tokenizer import load_tokenizer

    model = model_config(tmp_path, "deepseek_v1")
    base = load_config(Path("configs/runtime_smoke_cpu.yaml"))
    path = tmp_path / "tokenizer.json"
    config = base.model_copy(
        update={
            "model": model,
            "tokenizer": base.tokenizer.model_copy(update={"path": path}),
        }
    )
    tokenizer = load_tokenizer(path)
    tokenizer.normalizer = normalizers.Lowercase()
    with pytest.raises(ValueError, match="tokenizer object"):
        prepare_data(config, tokenizer)
