from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from klab.config.migrate import migrate_v1
from klab.config.models import AttentionConfig, ModelConfig, RunConfig
from klab.data.byte_hash import table_address, token_bytes
from klab.data.packing import prepare_data
from klab.evaluation.chat import chat_turn
from klab.evaluation.generation import (
    _prompt_byte_addresses,
    generate,
    generate_with_token_ids,
)
from klab.model.attention.dense import DenseAttention
from klab.model.portable_engram import export_portable_engram
from klab.model.transformer import DenseLM


def _tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train_from_iterator(
        ["hello world\nhi\nUser: next", "é"],
        BpeTrainer(
            vocab_size=260,
            initial_alphabet=ByteLevel.alphabet(),
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        ),
    )
    return tokenizer


class _ScriptedModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer: Tokenizer,
        scripts: list[dict[int, float]],
        *,
        memory: str = "none",
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            memory=memory, memory_table_size=97, memory_ngram_size=3
        )
        self.vocab_size = tokenizer.get_vocab_size()
        self.scripts = scripts
        self.calls = 0
        self.byte_addresses: list[torch.Tensor | None] = []

    def forward(
        self, input_ids: torch.Tensor, *, byte_addresses: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.byte_addresses.append(
            None if byte_addresses is None else byte_addresses.cpu()
        )
        scores = torch.full((1, input_ids.shape[1], self.vocab_size), -100.0)
        for token_id, score in self.scripts[
            min(self.calls, len(self.scripts) - 1)
        ].items():
            scores[0, -1, token_id] = score
        self.calls += 1
        return scores.to(input_ids.device)


def _ids(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def test_generation_rejects_invalid_sampling_ranges_before_model_execution() -> None:
    tokenizer = _tokenizer()
    model = _ScriptedModel(tokenizer, [{0: 1}])

    with pytest.raises(ValueError, match="max_seq_len"):
        generate(model, tokenizer, "hello", 0, 1, torch.device("cpu"))
    with pytest.raises(ValueError, match="top_k"):
        generate(model, tokenizer, "hello", 8, 1, torch.device("cpu"), top_k=-1)
    with pytest.raises(ValueError, match="requested completion"):
        generate(
            model,
            tokenizer,
            "hello",
            len(_ids(tokenizer, "hello")),
            1,
            torch.device("cpu"),
            strict_context=True,
        )
    assert model.calls == 0


def test_generation_defaults_to_greedy_suppresses_special_outputs_and_restores_mode() -> (
    None
):
    tokenizer = _tokenizer()
    pad = tokenizer.token_to_id("<pad>")
    eos = tokenizer.token_to_id("<eos>")
    token = _ids(tokenizer, "h")[0]
    assert pad is not None and eos is not None
    model = _ScriptedModel(tokenizer, [{pad: 20, token: 10}, {eos: 20}])
    model.train()

    completion = generate(model, tokenizer, "prompt", 16, 2, torch.device("cpu"))

    assert completion == "prompth"
    assert model.training


def test_sampling_is_seeded_locally_without_mutating_global_rng() -> None:
    tokenizer = _tokenizer()
    first, second = _ids(tokenizer, "hi")
    model = _ScriptedModel(tokenizer, [{first: 0, second: 0}])
    global_before = torch.get_rng_state()

    first_completion = generate(
        model, tokenizer, "", 16, 1, torch.device("cpu"), temperature=1.0, seed=43
    )
    second_completion = generate(
        model, tokenizer, "", 16, 1, torch.device("cpu"), temperature=1.0, seed=43
    )

    assert first_completion == second_completion
    assert torch.equal(torch.get_rng_state(), global_before)


def test_generation_stops_on_eos_without_echoing_new_prompt_text() -> None:
    tokenizer = _tokenizer()
    eos = tokenizer.token_to_id("<eos>")
    assert eos is not None
    model = _ScriptedModel(tokenizer, [{eos: 10}])

    assert (
        generate(
            model, tokenizer, "User: hello\n\nAssistant:", 32, 4, torch.device("cpu")
        )
        == "User: hello\n\nAssistant:"
    )


@pytest.mark.parametrize("temperature,top_k", [(0.0, 0), (1.0, 2)])
def test_generation_returns_actual_tokens_without_changing_legacy_text(
    temperature: float, top_k: int
) -> None:
    tokenizer = _tokenizer()
    first, second = _ids(tokenizer, "hi")
    eos = tokenizer.token_to_id("<eos>")
    assert eos is not None
    script = [{first: 3.0, second: 2.0}, {eos: 10.0}]
    options = {"temperature": temperature, "top_k": top_k, "seed": 42}
    text, generated = generate_with_token_ids(
        _ScriptedModel(tokenizer, script),
        tokenizer,
        "hello",
        16,
        4,
        torch.device("cpu"),
        **options,
    )

    assert generated in ([first], [second])
    assert text == "hello" + tokenizer.decode(generated, skip_special_tokens=True)
    assert text == generate(
        _ScriptedModel(tokenizer, script),
        tokenizer,
        "hello",
        16,
        4,
        torch.device("cpu"),
        **options,
    )
    assert generate_with_token_ids(
        _ScriptedModel(tokenizer, [{eos: 10.0}]),
        tokenizer,
        "hello",
        16,
        4,
        torch.device("cpu"),
    ) == ("hello", [])


def test_generation_ids_retain_tokens_hidden_by_text_stop_sequence() -> None:
    tokenizer = _tokenizer()
    token = _ids(tokenizer, "h")[0]
    text, generated = generate_with_token_ids(
        _ScriptedModel(tokenizer, [{token: 10.0}]),
        tokenizer,
        "hello",
        16,
        4,
        torch.device("cpu"),
        stop_sequences=("h",),
    )

    assert text == "hello"
    assert generated == [token]
    assert generate_with_token_ids(
        _ScriptedModel(tokenizer, [{token: 10.0}]),
        tokenizer,
        "hello",
        16,
        0,
        torch.device("cpu"),
    ) == ("hello", [])


def test_byte_memory_uses_raw_leading_space_and_unicode_bytes_for_causal_addresses() -> (
    None
):
    tokenizer = _tokenizer()
    prompt_ids = _ids(tokenizer, " é")
    assert len(prompt_ids) == 3  # Space plus UTF-8's two BPE byte symbols.
    output = _ids(tokenizer, "!")[0]
    eos = tokenizer.token_to_id("<eos>")
    assert eos is not None
    model = _ScriptedModel(tokenizer, [{output: 10}, {eos: 10}], memory="byte")

    generate(model, tokenizer, " é", 3, 2, torch.device("cpu"))

    expected_prompt = [
        table_address(b" ", 97),
        table_address(b" \xc3", 97),
        table_address(b" \xc3\xa9", 97),
    ]
    assert model.byte_addresses[0].tolist() == [expected_prompt]
    assert model.byte_addresses[1].tolist() == [
        [expected_prompt[1], expected_prompt[2], table_address(b"\xc3\xa9!", 97)]
    ]


def test_prepare_data_unicode_addresses_match_generation_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = _tokenizer()
    document = " leading é"
    config = RunConfig.model_validate(
        migrate_v1(
            {
                "schema_version": 1,
                "name": "byte-parity",
                "seed": 1,
                "device": "cpu",
                "model": {
                    "vocab_size": tokenizer.get_vocab_size(),
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 2,
                    "ffn_dim": 32,
                    "max_seq_len": 16,
                    "memory": "byte",
                    "memory_table_size": 97,
                    "memory_ngram_size": 3,
                    "memory_dim": 8,
                },
                "tokenizer": {"path": tmp_path / "tokenizer.json"},
                "dataset": {
                    "source": "synthetic",
                    "cache_dir": tmp_path / "cache",
                    "train_max_documents": 1,
                    "validation_max_documents": 1,
                    "train_max_tokens": 64,
                    "validation_max_tokens": 64,
                },
                "training": {
                    "batch_size": 1,
                    "seq_len": 8,
                    "max_steps": 1,
                    "max_tokens": 8,
                },
                "optimizer": {
                    "learning_rate": 0.001,
                    "min_learning_rate": 0.0001,
                    "warmup_steps": 0,
                },
                "logging": {"root_dir": tmp_path / "runs", "checkpoint_every_steps": 1},
            }
        )
    )
    monkeypatch.setattr(
        "klab.data.packing.iter_documents",
        lambda _config, _split: iter([document]),
    )

    prepared = prepare_data(config, tokenizer)
    eos = tokenizer.token_to_id("<eos>")
    assert eos is not None and prepared.train_byte_addresses is not None
    end = list(prepared.train).index(eos)
    ids = prepared.train[:end].tolist()
    assert prepared.train_byte_addresses[:end].tolist() == _prompt_byte_addresses(
        tokenizer, document, ids, 97, 3
    )


def test_chat_stops_generation_when_another_role_boundary_is_decoded() -> None:
    tokenizer = _tokenizer()
    reply_ids = _ids(tokenizer, " hi\nUser:")
    eos = tokenizer.token_to_id("<eos>")
    assert eos is not None
    model = _ScriptedModel(
        tokenizer, [{token_id: 10} for token_id in reply_ids] + [{eos: 10}]
    )

    prompt, reply = chat_turn(
        model, tokenizer, [], "hello", 64, len(reply_ids) + 1, torch.device("cpu")
    )

    assert prompt == "User: hello\n\nAssistant:"
    assert reply == "hi"
    assert model.calls == len(reply_ids)


def test_real_tiny_model_forward_is_usable_for_generation() -> None:
    tokenizer = _tokenizer()
    model = DenseLM(
        ModelConfig(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=16,
        ),
        AttentionConfig(),
    )
    model.train()

    completion = generate(model, tokenizer, "hello", 16, 1, torch.device("cpu"))

    assert completion.startswith("hello")
    assert model.training


def test_grouped_query_attention_matches_explicit_kv_repeat_reference() -> None:
    torch.manual_seed(23)
    attention = DenseAttention(16, 4, 8, 10_000.0, num_kv_heads=2).eval()
    inputs = torch.randn(2, 5, 16)

    actual = attention(inputs)
    query = attention.rope(attention._heads(attention.q_proj, inputs, 4))
    key = attention.rope(attention._heads(attention.k_proj, inputs, 2))
    value = attention._heads(attention.v_proj, inputs, 2)
    repeated_key = key.repeat_interleave(2, dim=1)
    repeated_value = value.repeat_interleave(2, dim=1)
    scores = query @ repeated_key.transpose(-2, -1) / (attention.head_dim**0.5)
    scores.masked_fill_(attention.causal_mask[:5, :5], float("-inf"))
    output = torch.softmax(scores.float(), dim=-1).to(value.dtype) @ repeated_value
    expected = attention.out_proj(output.transpose(1, 2).contiguous().view(2, 5, 16))

    torch.testing.assert_close(actual, expected)

    actual.square().mean().backward()
    actual_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in attention.named_parameters()
    }
    attention.zero_grad(set_to_none=True)
    expected.square().mean().backward()
    for name, parameter in attention.named_parameters():
        torch.testing.assert_close(actual_gradients[name], parameter.grad)


@pytest.mark.parametrize(
    "attention",
    [AttentionConfig(), AttentionConfig(kind="sliding_window", window_size=2)],
)
def test_grouped_query_incremental_cache_matches_full_prefix_logits(
    attention: AttentionConfig,
) -> None:
    torch.manual_seed(29)
    model = DenseLM(
        ModelConfig(
            vocab_size=260,
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            ffn_dim=32,
            max_seq_len=8,
        ),
        attention,
    ).eval()
    prefix = torch.tensor([[3, 7, 11], [5, 9, 15]])
    appended = torch.tensor([[13], [17]])

    full_prefix = model(prefix)
    cached_prefix, cache = model.forward_cached(prefix, cache_capacity=8)
    torch.testing.assert_close(cached_prefix, full_prefix)
    cached_next, _ = model.forward_cached(appended, cache=cache)
    full_next = model(torch.cat((prefix, appended), dim=1))[:, -1]
    torch.testing.assert_close(cached_next[:, -1], full_next)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize(
    "attention",
    [
        AttentionConfig(),
        AttentionConfig(kind="sliding_window", window_size=2),
        AttentionConfig(kind="mla", latent_dim=8),
    ],
)
def test_incremental_cache_matches_full_prefix_logits(
    attention: AttentionConfig,
    autocast: bool,
) -> None:
    torch.manual_seed(7)
    model = DenseLM(
        ModelConfig(
            vocab_size=260,
            hidden_dim=16,
            num_layers=2,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=6,
        ),
        attention,
    ).eval()
    prefix = torch.tensor([[3, 7, 11], [5, 9, 15]])
    appended = torch.tensor([[13], [17]])

    tolerance = {"atol": 0.005, "rtol": 0.01} if autocast else {}
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        full_prefix = model(prefix)
        cached_prefix, cache = model.forward_cached(prefix, cache_capacity=6)
        torch.testing.assert_close(cached_prefix, full_prefix, **tolerance)

        full_next = model(torch.cat((prefix, appended), dim=1))[:, -1]
        cached_next, _ = model.forward_cached(appended, cache=cache)
        torch.testing.assert_close(cached_next[:, -1], full_next, **tolerance)


def test_incremental_cache_is_bounded_owned_and_graph_free() -> None:
    config = ModelConfig(
        vocab_size=260,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        max_seq_len=6,
    )
    model = DenseLM(config, AttentionConfig()).eval()
    other = DenseLM(config, AttentionConfig()).eval()
    prefix = torch.tensor([[3, 7]])
    with pytest.raises(ValueError, match="explicit cache_capacity"):
        model.forward_cached(prefix)

    _, cache = model.forward_cached(prefix, cache_capacity=3)
    key_storage = cache.layers[0].key.untyped_storage().data_ptr()
    value_storage = cache.layers[0].value.untyped_storage().data_ptr()

    logits, cache = model.forward_cached(torch.tensor([[11]]), cache=cache)

    assert cache.length == 3
    assert cache.capacity == 3
    assert cache.allocated_bytes == 3 * (8 + 2 * config.hidden_dim * 4)
    assert cache.layers[0].key.untyped_storage().data_ptr() == key_storage
    assert cache.layers[0].value.untyped_storage().data_ptr() == value_storage
    assert not logits.requires_grad
    with pytest.raises(ValueError, match="different model"):
        other.forward_cached(torch.tensor([[13]]), cache=cache)
    with pytest.raises(ValueError, match="capacity"):
        model.forward_cached(torch.tensor([[13]]), cache=cache)


def test_cached_generation_matches_reference_across_context_rollover() -> None:
    tokenizer = _tokenizer()
    model = DenseLM(
        ModelConfig(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=3,
        ),
        AttentionConfig(),
    ).eval()

    cached = generate(
        model, tokenizer, "hello", 3, 5, torch.device("cpu"), use_cache=True
    )
    reference = generate(
        model, tokenizer, "hello", 3, 5, torch.device("cpu"), use_cache=False
    )

    assert cached == reference


def test_block_sparse_generation_reports_prefix_recompute_capability() -> None:
    model = DenseLM(
        ModelConfig(
            vocab_size=260,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=8,
        ),
        AttentionConfig(kind="block_sparse", block_size=2, selected_blocks=1),
    )

    capability = model.incremental_cache_capability

    assert not capability.supported
    assert capability.reason is not None
    assert "prefix" in capability.reason


@pytest.mark.parametrize("injection", ["final", "embedding"])
def test_incremental_cache_preserves_unicode_byte_memory_addresses(
    injection: str,
) -> None:
    tokenizer = _tokenizer()
    prompt = " é"
    prefix = _ids(tokenizer, prompt)
    appended = _ids(tokenizer, "!")[0]
    model = DenseLM(
        ModelConfig(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=8,
            memory="byte",
            memory_table_size=97,
            memory_ngram_size=3,
            memory_dim=8,
            memory_injection=injection,
        ),
        AttentionConfig(),
    ).eval()
    addresses = _prompt_byte_addresses(tokenizer, prompt, prefix, 97, 3)
    prefix_ids = torch.tensor([prefix])
    prefix_addresses = torch.tensor([addresses])

    cached_prefix, cache = model.forward_cached(
        prefix_ids, cache_capacity=8, byte_addresses=prefix_addresses
    )
    full_prefix = model(prefix_ids, byte_addresses=prefix_addresses)
    torch.testing.assert_close(cached_prefix, full_prefix)

    source = prompt.encode("utf-8") + token_bytes(tokenizer, appended)
    next_address = table_address(source[-3:], 97)
    cached_next, _ = model.forward_cached(
        torch.tensor([[appended]]),
        cache=cache,
        byte_addresses=torch.tensor([[next_address]]),
    )
    full_ids = torch.tensor([prefix + [appended]])
    full_addresses = torch.tensor([addresses + [next_address]])
    torch.testing.assert_close(
        cached_next[:, -1], model(full_ids, byte_addresses=full_addresses)[:, -1]
    )


@pytest.mark.parametrize("injection", ["final", "embedding", "after_block"])
@pytest.mark.parametrize(
    "variant,kernel",
    [("hidden_gate", 0), ("context_key", 0), ("context_key", 1), ("context_key", 4)],
)
@pytest.mark.parametrize("block_index", [0, 1])
def test_incremental_cache_preserves_ngram_memory_history(
    injection: str, variant: str, kernel: int, block_index: int
) -> None:
    model = DenseLM(
        ModelConfig(
            vocab_size=260,
            hidden_dim=16,
            num_layers=2,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=8,
            memory="ngram",
            memory_table_size=97,
            memory_ngram_size=3,
            memory_dim=8,
            memory_ngram_orders=(2, 3),
            memory_hash_heads=2,
            memory_injection=injection,
            memory_block_index=block_index if injection == "after_block" else None,
            memory_variant=variant,
            memory_conv_kernel_size=kernel,
        ),
        AttentionConfig(),
    ).eval()
    prefix = torch.tensor([[3, 7, 11, 13], [5, 9, 15, 19]])
    appended = torch.tensor([[17, 23], [21, 25]])

    _, cache = model.forward_cached(prefix, cache_capacity=8)
    cached_next, _ = model.forward_cached(appended, cache=cache)
    full_next = model(torch.cat((prefix, appended), dim=1))[:, -2:]

    torch.testing.assert_close(cached_next, full_next)
    # Independent request and rollover must not inherit the preceding memory tail.
    rolled = torch.cat((prefix, appended), dim=1)[:, 2:]
    cached_roll, other = model.forward_cached(rolled, cache_capacity=8)
    torch.testing.assert_close(cached_roll, model(rolled))
    assert other is not cache
    if kernel:
        expected = prefix.shape[0] * (kernel - 1) * 3 * 16 * 4
        assert (
            cache.memory_history.numel() * cache.memory_history.element_size()
            == expected
        )
        assert cache.memory_history.untyped_storage().nbytes() == expected
        assert (
            cache.allocated_bytes
            == cache.input_ids.numel() * 8
            + sum(layer.allocated_bytes for layer in cache.layers)
            + expected
        )
        cache.memory_history = None
        with pytest.raises(ValueError, match="missing memory convolution history"):
            model.forward_cached(appended[:, :1], cache=cache)


@pytest.mark.parametrize("injection", ["final", "embedding"])
def test_incremental_cache_preserves_portable_memory_chunks(
    injection: str, tmp_path: Path
) -> None:
    package = tmp_path / "memory.engram"
    export_portable_engram(
        torch.arange(85, dtype=torch.float32).reshape(17, 5), package, ngram_size=3
    )
    model = DenseLM(
        ModelConfig(
            vocab_size=260,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=8,
            memory="portable",
            memory_table_size=17,
            memory_ngram_size=3,
            memory_dim=5,
            memory_package_path=package,
            memory_injection=injection,
        ),
        AttentionConfig(),
    ).eval()
    prefix = torch.tensor([[3, 7, 11], [5, 9, 15]])
    addresses = torch.tensor([[1, 4, 7], [2, 5, 8]])
    cached, cache = model.forward_cached(
        prefix, cache_capacity=8, byte_addresses=addresses
    )
    torch.testing.assert_close(cached, model(prefix, byte_addresses=addresses))
    for chunk, chunk_addresses in (
        (torch.tensor([[17], [19]]), torch.tensor([[10], [11]])),
        (torch.tensor([[21, 23], [25, 27]]), torch.tensor([[12, 13], [14, 15]])),
    ):
        cached, cache = model.forward_cached(
            chunk, cache=cache, byte_addresses=chunk_addresses
        )
        prefix = torch.cat((prefix, chunk), dim=1)
        addresses = torch.cat((addresses, chunk_addresses), dim=1)
        torch.testing.assert_close(
            cached, model(prefix, byte_addresses=addresses)[:, -chunk.shape[1] :]
        )


@pytest.mark.parametrize("memory_kind", ["ngram", "byte"])
def test_early_memory_cached_generation_matches_context_rollover(
    memory_kind: str,
) -> None:
    tokenizer = _tokenizer()
    model = DenseLM(
        ModelConfig(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=3,
            memory=memory_kind,
            memory_table_size=97,
            memory_ngram_size=3,
            memory_dim=8,
            memory_injection="embedding",
        ),
        AttentionConfig(),
    ).eval()

    cached = generate(
        model, tokenizer, "hello", 3, 5, torch.device("cpu"), use_cache=True
    )
    reference = generate(
        model, tokenizer, "hello", 3, 5, torch.device("cpu"), use_cache=False
    )

    assert cached == reference


def test_long_completion_respects_token_budget_and_strict_context():
    tokenizer = _tokenizer()
    token = _ids(tokenizer, "h")[0]
    model = _ScriptedModel(tokenizer, [{token: 10}])
    text, generated = generate_with_token_ids(
        model, tokenizer, "hello", 1024, 512, torch.device("cpu"), strict_context=True
    )
    assert len(generated) == 512
    assert text == "hello" + tokenizer.decode([token] * 512)
    with pytest.raises(ValueError, match="requested completion"):
        generate(
            model,
            tokenizer,
            "h" * 600,
            1024,
            512,
            torch.device("cpu"),
            strict_context=True,
        )


def test_context_memory_cache_bfloat16_keeps_fp32_residual_history() -> None:
    from klab.config.loading import load_config

    base = load_config(Path("configs/smoke_memory_cpu.yaml"))
    configured = base.model.model_copy(
        update={
            "memory_variant": "context_key",
            "memory_conv_kernel_size": 4,
            "memory_injection": "after_block",
            "memory_block_index": 0,
        }
    )
    model = DenseLM(configured, base.attention).eval()
    ids = torch.tensor([[3, 7, 11, 13, 17, 19]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, cache = model.forward_cached(ids[:, :4], cache_capacity=8)
        actual, cache = model.forward_cached(ids[:, 4:], cache=cache)
        expected = model(ids)[:, -2:]
        torch.testing.assert_close(actual, expected)
        assert cache.layers[0].key.dtype == torch.bfloat16
        assert cache.memory_history.dtype == torch.float32
        cache.memory_history = cache.memory_history.to(torch.bfloat16)
        with pytest.raises(ValueError, match="memory convolution history is invalid"):
            model.forward_cached(ids[:, :1], cache=cache)
        assert cache.length == 6


@pytest.mark.parametrize(
    ("model_options", "attention", "max_seq_len"),
    [
        ({}, AttentionConfig(), 256),
        ({}, AttentionConfig(), 40),
        ({}, AttentionConfig(kind="sliding_window", window_size=4), 256),
        ({}, AttentionConfig(kind="mla", latent_dim=8), 256),
        (
            {
                "memory": "ngram",
                "memory_table_size": 97,
                "memory_ngram_size": 3,
                "memory_dim": 8,
                "memory_ngram_orders": (2, 3),
                "memory_injection": "after_block",
                "memory_block_index": 0,
                "memory_variant": "context_key",
                "memory_conv_kernel_size": 4,
            },
            AttentionConfig(),
            40,
        ),
    ],
)
def test_speculation_keeps_greedy_tokens_when_drafts_are_partly_wrong(
    model_options: dict[str, object], attention: AttentionConfig, max_seq_len: int
) -> None:
    from klab.evaluation.speculation import PromptLookupSpeculation

    torch.manual_seed(0)
    tokenizer = _tokenizer()
    model = DenseLM(
        ModelConfig(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_dim=16,
            num_layers=2,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=max_seq_len,
            **model_options,
        ),
        attention,
    ).eval()
    prompt = "hello world\nhi"
    prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
    _, reference = generate_with_token_ids(
        model, tokenizer, prompt, max_seq_len, 60, torch.device("cpu")
    )

    class PartlyWrong(PromptLookupSpeculation):
        """Draft the true continuation with one wrong token at a moving position."""

        calls = 0

        def draft(self, ids, limit):
            done = len(ids) - prompt_length
            proposal = list(reference[done : done + min(limit, 8)])
            self.calls += 1
            wrong = self.calls % (len(proposal) + 1)
            if wrong < len(proposal):
                proposal[wrong] = (proposal[wrong] + 1) % 250 + 5
            return proposal

    speculation = PartlyWrong()
    _, speculative = generate_with_token_ids(
        model,
        tokenizer,
        prompt,
        max_seq_len,
        60,
        torch.device("cpu"),
        speculation=speculation,
    )

    assert speculative == reference
    assert len(reference) == 60
    assert 0 < speculation.accepted_tokens < speculation.drafted_tokens
    assert speculation.model_calls < len(reference)


def test_prompt_lookup_drafts_what_followed_the_latest_repeat() -> None:
    from klab.evaluation.speculation import PromptLookupSpeculation

    speculation = PromptLookupSpeculation(ngram=2, max_draft_tokens=3)
    assert speculation.draft([1, 2, 3], 8) == []
    assert speculation.draft([1, 2, 7, 8, 9, 1, 2, 5, 6, 1, 2], 8) == [5, 6, 1]
    assert speculation.draft([1, 2, 7, 8, 9, 1, 2, 5, 6, 1, 2], 1) == [5]
    speculation.begin()
    assert speculation.draft([4, 1, 2], 8) == []


def test_speculation_refuses_sampling() -> None:
    from klab.evaluation.speculation import PromptLookupSpeculation

    tokenizer = _tokenizer()
    model = DenseLM(
        ModelConfig(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=16,
        ),
        AttentionConfig(),
    ).eval()
    with pytest.raises(ValueError, match="speculation requires greedy"):
        generate(
            model,
            tokenizer,
            "hello",
            16,
            4,
            torch.device("cpu"),
            temperature=0.8,
            speculation=PromptLookupSpeculation(),
        )
