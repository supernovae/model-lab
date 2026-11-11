from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

mx = pytest.importorskip("mlx.core")
from mlx import nn, optimizers
from mlx.utils import tree_flatten

from klab.config.models import AttentionConfig, ModelConfig, RunConfig
from klab.engines.base import EngineNonFiniteError, EngineState, Microbatch
from klab.engines.mlx import MLXEngine
from klab.evaluation.generation import generate
from klab.model.mlx_dense import MLXDenseLM
from klab.runtime_profile import RuntimeProfile, authorize_profile

pytestmark = pytest.mark.mlx


def _model(
    *, tied: bool, recompute: bool = False, attention: AttentionConfig | None = None
) -> MLXDenseLM:
    mx.random.seed(17)
    return MLXDenseLM(
        ModelConfig(
            vocab_size=260,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=16,
            tie_embeddings=tied,
        ),
        attention or AttentionConfig(),
        recompute_blocks=recompute,
    )


def test_mlx_dense_rejects_grouped_query_attention() -> None:
    with pytest.raises(ValueError, match="does not support grouped-query attention"):
        MLXDenseLM(
            ModelConfig(
                vocab_size=260,
                hidden_dim=16,
                num_layers=1,
                num_heads=2,
                num_kv_heads=1,
                ffn_dim=32,
                max_seq_len=16,
            )
        )


def _config(*, tied: bool) -> RunConfig:
    return RunConfig.model_validate(
        {
            "schema_version": 2,
            "name": "mlx-engine-regression",
            "seed": 7,
            "model": {
                "vocab_size": 260,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "ffn_dim": 32,
                "max_seq_len": 16,
                "tie_embeddings": tied,
            },
            "tokenizer": {"path": "unused-tokenizer.json"},
            "dataset": {
                "source": "synthetic",
                "cache_dir": "unused-cache",
                "train_max_documents": 8,
                "validation_max_documents": 2,
                "train_max_tokens": 128,
                "validation_max_tokens": 64,
                "synthetic_seed": 7,
            },
            "training": {
                "seq_len": 4,
                "micro_batch_size": 1,
                "max_steps": 2,
                "max_tokens": 8,
            },
            "optimizer": {
                "name": "adamw",
                "peak": 0.003,
                "floor": 0.0003,
                "warmup_steps": 1,
            },
            "logging": {"root_dir": "unused-runs"},
            "runtime": {"engine": "mlx", "backend": "metal"},
        }
    )


def _engine(*, tied: bool) -> MLXEngine:
    config = _config(tied=tied)
    profile = RuntimeProfile(
        runtime_profile_version=1,
        id="mlx-engine-regression",
        python=Path(sys.executable),
        engine="mlx",
        backend="metal",
        device_index=0,
    )
    engine = MLXEngine()
    engine.authorization = authorize_profile(profile, config)
    engine.initialize(config)
    return engine


def _weights(engine: MLXEngine) -> dict[str, np.ndarray]:
    return {tensor.name: tensor.array for tensor in engine.export_weights().tensors()}


def _batch(tokens: np.ndarray, targets: np.ndarray) -> Microbatch:
    return Microbatch(tokens.astype(np.int64), targets.astype(np.int64))


def _tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train_from_iterator(
        ["hello world", "native MLX sampling"],
        BpeTrainer(
            vocab_size=260,
            initial_alphabet=ByteLevel.alphabet(),
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        ),
    )
    return tokenizer


def test_mlx_generation_uses_local_key_and_restores_mode_and_rng() -> None:
    engine = _engine(tied=False)
    tokenizer = _tokenizer()
    assert tokenizer.get_vocab_size() == engine.config.model.vocab_size
    assert engine.model is not None
    engine.model.train()
    before = engine._rng()

    first = generate(
        engine.model,
        tokenizer,
        "hello",
        16,
        3,
        "metal",
        temperature=1.0,
        seed=43,
        engine=engine,
    )
    second = generate(
        engine.model,
        tokenizer,
        "hello",
        16,
        3,
        "metal",
        temperature=1.0,
        seed=43,
        engine=engine,
    )

    after = engine._rng()
    assert first == second
    assert engine.model.training
    assert before["python"] == after["python"]
    assert before["numpy_kind"] == after["numpy_kind"]
    assert before["numpy_pos"] == after["numpy_pos"]
    np.testing.assert_array_equal(before["numpy_keys"], after["numpy_keys"])
    np.testing.assert_array_equal(before["mlx"], after["mlx"])


@pytest.mark.parametrize("tied", [False, True])
def test_mlx_canonical_roundtrip_preserves_logits_after_optimizer_update(
    tied: bool,
) -> None:
    engine = _engine(tied=tied)
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
    targets = np.array([[2, -100, 4, 5]], dtype=np.int64)
    result = engine.train_update([_batch(tokens, targets)], 1, 3)
    assert result.outcome == "APPLIED"
    expected = np.asarray(engine.logits(mx.array(tokens, dtype=mx.int32)))
    restored = _engine(tied=tied)
    restored.load_canonical_weights(_weights(engine), engine.export_weights().aliases)
    np.testing.assert_allclose(
        np.asarray(restored.logits(mx.array(tokens, dtype=mx.int32))),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_mlx_step_zero_state_restores_the_same_next_update() -> None:
    untouched = _engine(tied=False)
    restored = _engine(tied=False)
    restored.restore_training_state(untouched.export_training_state())
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
    batch = _batch(tokens, tokens)
    assert untouched.train_update([batch], 1, 4).outcome == "APPLIED"
    assert restored.train_update([batch], 1, 4).outcome == "APPLIED"
    for name, value in _weights(untouched).items():
        np.testing.assert_allclose(
            value, _weights(restored)[name], rtol=1e-6, atol=1e-6
        )


def test_mlx_update_emits_common_synchronized_memory_telemetry() -> None:
    engine = _engine(tied=False)
    result = engine.train_update(
        [_batch(np.array([[1, 2, 3, 4]]), np.array([[2, 3, 4, 5]]))],
        1,
        4,
    )
    assert result.outcome == "APPLIED"
    assert {
        "train/loss",
        "optimizer/learning_rate",
        "optimizer/grad_norm",
        "performance/step_seconds",
        "performance/tokens_per_second",
        "memory/process_rss_bytes",
        "memory/device_allocated_bytes",
        "memory/device_peak_allocated_bytes",
    } <= result.metrics.keys()
    assert result.metrics["memory/process_rss_bytes"] > 0
    assert result.metrics["memory/device_allocated_bytes"] > 0
    assert result.metrics["memory/device_peak_allocated_bytes"] > 0


def test_mlx_update_skips_zero_target_microbatches() -> None:
    engine = _engine(tied=False)
    clean = _engine(tied=False)
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
    ignored = _batch(tokens, np.full_like(tokens, -100))
    scored = _batch(tokens, tokens)
    assert engine.train_update([ignored, scored], 1, 4).outcome == "APPLIED"
    assert clean.train_update([scored], 1, 4).outcome == "APPLIED"
    for name, value in _weights(clean).items():
        np.testing.assert_allclose(value, _weights(engine)[name], rtol=1e-6, atol=1e-6)


def test_mlx_native_state_restores_two_adamw_groups_and_exact_rng_envelope() -> None:
    engine = _engine(tied=True)
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
    targets = np.array([[2, 3, 4, 5]], dtype=np.int64)
    assert engine.train_update([_batch(tokens, targets)], 1, 4).outcome == "APPLIED"
    state = engine.export_training_state()
    assert set(state.rng) == {
        "python",
        "numpy_kind",
        "numpy_keys",
        "numpy_pos",
        "numpy_has_gauss",
        "numpy_cached_gaussian",
        "mlx",
    }
    assert len(state.optimizer_parameter_names) == 2
    assert all(name.endswith(".weight") for name in state.optimizer_parameter_names[0])
    resumed = _engine(tied=True)
    resumed.load_canonical_weights(_weights(engine), engine.export_weights().aliases)
    resumed.restore_training_state(state)
    assert engine.train_update([_batch(tokens, targets)], 2, 4).outcome == "APPLIED"
    assert resumed.train_update([_batch(tokens, targets)], 2, 4).outcome == "APPLIED"
    for name, value in _weights(engine).items():
        np.testing.assert_allclose(value, _weights(resumed)[name], rtol=1e-6, atol=1e-6)


def _next_rng_draws(
    engine: MLXEngine, state: EngineState
) -> tuple[float, float, np.ndarray]:
    engine._restore_rng(state.rng)
    mlx_draw = mx.random.uniform(shape=(2,))
    mx.eval(mlx_draw)
    return random.random(), float(np.random.random()), np.asarray(mlx_draw)


def test_mlx_invalid_restore_preserves_next_rng_draws() -> None:
    engine = _engine(tied=False)
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
    assert engine.train_update([_batch(tokens, tokens)], 1, 4).outcome == "APPLIED"
    state = engine.export_training_state()
    expected = _next_rng_draws(engine, state)
    engine._restore_rng(state.rng)
    invalid_rng = {**state.rng, "mlx": np.zeros(3, dtype=np.uint32)}
    invalid = EngineState(
        state.optimizer,
        invalid_rng,
        state.scaler,
        state.optimizer_parameter_names,
    )
    with pytest.raises(ValueError, match="RNG"):
        engine.restore_training_state(invalid)
    actual = _next_rng_draws(engine, state)
    assert actual[:2] == expected[:2]
    np.testing.assert_array_equal(actual[2], expected[2])


@pytest.mark.parametrize(
    "attention",
    [
        AttentionConfig(),
        AttentionConfig(kind="block_sparse", block_size=2, selected_blocks=1),
    ],
    ids=["dense", "native_sparse"],
)
def test_mlx_recomputation_preserves_all_parameter_gradients_and_optimizer_update(
    attention: AttentionConfig,
) -> None:
    plain = _model(tied=False, attention=attention)
    recomputed = _model(tied=False, recompute=True, attention=attention)
    recomputed.update(plain.parameters())
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    targets = mx.array([[2, 3, 4, 5]], dtype=mx.int32)

    def gradients(model: MLXDenseLM) -> tuple[object, dict[str, object]]:
        def loss(ids: object, labels: object) -> object:
            return nn.losses.cross_entropy(model(ids), labels, reduction="mean")

        value, result = nn.value_and_grad(model, loss)(tokens, targets)
        mx.eval(value, result)
        return value, result

    expected_loss, expected_gradients = gradients(plain)
    actual_loss, actual_gradients = gradients(recomputed)
    np.testing.assert_allclose(
        np.asarray(actual_loss), np.asarray(expected_loss), rtol=1e-5, atol=1e-6
    )
    expected_flat, actual_flat = (
        dict(tree_flatten(expected_gradients)),
        dict(tree_flatten(actual_gradients)),
    )
    assert actual_flat.keys() == expected_flat.keys()
    for name, expected in expected_flat.items():
        np.testing.assert_allclose(
            np.asarray(actual_flat[name]),
            np.asarray(expected),
            rtol=1e-4,
            atol=1e-5,
            err_msg=name,
        )
    optimizers.AdamW(learning_rate=0.003).update(plain, expected_gradients)
    optimizers.AdamW(learning_rate=0.003).update(recomputed, actual_gradients)
    np.testing.assert_allclose(
        np.asarray(recomputed(tokens)), np.asarray(plain(tokens)), rtol=1e-4, atol=1e-5
    )


def test_mlx_nonfinite_loss_recovers_to_the_clean_next_update() -> None:
    engine = _engine(tied=False)
    clean = _engine(tied=False)
    assert engine.model is not None and engine.model.output is not None
    original = np.asarray(engine.model.output.weight).copy()
    engine.model.output.weight = mx.full(engine.model.output.weight.shape, float("inf"))
    before = _weights(engine)
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
    with pytest.raises(EngineNonFiniteError):
        engine.train_update([_batch(tokens, tokens)], 1, 4)
    for name, value in _weights(engine).items():
        np.testing.assert_array_equal(value, before[name], err_msg=name)
    engine.model.output.weight = mx.array(original)
    applied = engine.train_update([_batch(tokens, tokens)], 1, 4)
    clean_applied = clean.train_update([_batch(tokens, tokens)], 1, 4)
    assert applied.outcome == clean_applied.outcome == "APPLIED"
    assert applied.committed_targets == clean_applied.committed_targets == 4
    for name, value in _weights(clean).items():
        np.testing.assert_allclose(value, _weights(engine)[name], rtol=1e-6, atol=1e-6)
