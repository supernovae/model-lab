from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from klab.config.models import RunConfig, TokenizerTrainConfig
from klab.data.tokenizer import train_tokenizer
from klab.engines.base import EngineState, Microbatch
from klab.engines.mlx import MLXEngine
from klab.runtime_profile import (
    RuntimeAuthorization,
    RuntimeProfile,
    authorize_profile,
)
from klab.training.checkpoints import CheckpointManager
from klab.training.trainer import train

pytestmark = pytest.mark.mlx


def mlx_config(root: Path) -> RunConfig:
    tokenizer = train_tokenizer(
        TokenizerTrainConfig.model_validate(
            {
                "schema_version": 1,
                "vocab_size": 260,
                "min_frequency": 1,
                "max_documents": 20,
                "output_dir": root / "tokenizer",
                "dataset": {
                    "source": "synthetic",
                    "cache_dir": root / "cache",
                    "train_max_documents": 20,
                    "validation_max_documents": 4,
                    "train_max_tokens": 512,
                    "validation_max_tokens": 128,
                    "synthetic_seed": 7,
                },
            }
        )
    )
    return RunConfig.model_validate(
        {
            "schema_version": 2,
            "name": "mlx-smoke",
            "seed": 7,
            "model": {
                "vocab_size": 260,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "ffn_dim": 32,
                "max_seq_len": 16,
            },
            "tokenizer": {"path": tokenizer},
            "dataset": {
                "source": "synthetic",
                "cache_dir": root / "cache",
                "train_max_documents": 20,
                "validation_max_documents": 4,
                "train_max_tokens": 512,
                "validation_max_tokens": 128,
                "synthetic_seed": 7,
            },
            "training": {
                "seq_len": 16,
                "max_steps": 3,
                "max_tokens": 45,
                "micro_batch_size": 1,
                "gradient_accumulation": 2,
            },
            "optimizer": {
                "name": "adamw",
                "peak": 0.003,
                "floor": 0.0003,
                "warmup_steps": 1,
            },
            "logging": {"root_dir": root / "runs"},
            "runtime": {
                "engine": "mlx",
                "backend": "metal",
                "memory": {"activation_checkpointing": {"enabled": True}},
            },
            "checkpoint": {"every_steps": 1},
            "evaluation": {"every_steps": 1, "max_batches": 1},
        }
    )


def _authorization(config: RunConfig) -> RuntimeAuthorization:
    return authorize_profile(
        RuntimeProfile(
            runtime_profile_version=1,
            id="mlx-training-regression",
            python=Path(sys.executable),
            engine="mlx",
            backend="metal",
            device_index=0,
        ),
        config,
    )


def test_mlx_train_masks_final_window_to_exact_target_budget(tmp_path: Path) -> None:
    config = mlx_config(tmp_path)
    run_id = train(config, run_id="mlx", authorization=_authorization(config))
    latest = config.logging.root_dir / run_id / "checkpoints" / "latest.json"
    snapshot = CheckpointManager(config.logging.root_dir / run_id).load(latest)
    assert snapshot.tokens_seen == 45
    assert snapshot.step == 2
    changed = config.model_copy(
        update={"training": config.training.model_copy(update={"max_steps": 4})}
    )
    manager = CheckpointManager(config.logging.root_dir / run_id)
    assert not manager.verify(
        latest, require_training_state=True, expected_config=changed
    ).valid
    assert manager.verify(
        latest, require_training_state=False, expected_config=changed
    ).valid


def test_mlx_full_resume_preserves_next_update_and_rng(tmp_path: Path) -> None:
    config = mlx_config(tmp_path)
    authorization = _authorization(config)
    full = train(config, run_id="full", authorization=authorization)
    parent = train(
        config, run_id="parent", stop_after_step=1, authorization=authorization
    )
    checkpoint = config.logging.root_dir / parent / "checkpoints" / "latest.json"
    child = train(
        config, run_id="child", resume=checkpoint, authorization=authorization
    )

    def next_update(run_id: str):
        run = config.logging.root_dir / run_id
        snapshot = CheckpointManager(run).load(run / "checkpoints" / "latest.json")
        engine = MLXEngine()
        engine.authorization = authorization
        try:
            engine.initialize(config, initial_weights=snapshot.model)
            engine.restore_training_state(
                EngineState(
                    optimizer=snapshot.optimizer,
                    rng=snapshot.rng,
                    scaler=snapshot.scaler,
                    optimizer_parameter_names=snapshot.optimizer_parameter_names,
                )
            )
            draws = (
                random.random(),
                np.random.standard_normal(8),
                np.asarray(mx.random.normal(shape=(8,))),
            )
            batch = Microbatch(
                np.array([[1, 2, 3, 4]], dtype=np.int64),
                np.array([[2, 3, 4, 5]], dtype=np.int64),
            )
            result = engine.train_update([batch], snapshot.step + 1, 4)
            assert result.outcome == "APPLIED"
            weights = {
                tensor.name: tensor.array
                for tensor in engine.export_weights().tensors()
            }
            return draws, weights
        finally:
            engine.close()

    expected_draws, expected_weights = next_update(full)
    actual_draws, actual_weights = next_update(child)
    assert actual_draws[0] == expected_draws[0]
    np.testing.assert_array_equal(actual_draws[1], expected_draws[1])
    np.testing.assert_array_equal(actual_draws[2], expected_draws[2])
    assert actual_weights.keys() == expected_weights.keys()
    for name, expected in expected_weights.items():
        np.testing.assert_array_equal(actual_weights[name], expected, err_msg=name)
