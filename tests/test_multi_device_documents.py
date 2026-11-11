"""CPU replica fixtures cover padded document updates without accelerator work."""

import numpy as np
import pytest
import torch
from test_document_batching import configured, rows_for

from klab.data.document_batching import collate_document_update
from klab.engines.pytorch import PyTorchEngine


@pytest.mark.parametrize("memory", [False, True])
def test_padded_replica_updates_preserve_gradients_and_diagnostics(tmp_path, memory):
    config = configured(tmp_path, memory)
    rows = rows_for([31, 1, 3, 13, 32, 7, 19])
    rows[2][1][:] = -100
    total = sum(int((row[1] != -100).sum()) for row in rows)
    batches = collate_document_update(
        rows, width=1, eos=3, max_length=64, padding_multiple=16
    )
    engines = []
    try:
        for secondary in ((), (0,)):
            cfg = config.model_copy(
                update={
                    "runtime": config.runtime.model_copy(
                        update={"secondary_device_indices": secondary}
                    )
                }
            )
            engine = PyTorchEngine()
            engine.validate(cfg)
            torch.manual_seed(10)
            engine.initialize(cfg)
            engines.append(engine)
        expected, actual = [
            engine.train_update(batches, 1, total) for engine in engines
        ]
        assert actual.committed_targets == expected.committed_targets == total
        assert actual.loss_sum == pytest.approx(expected.loss_sum, rel=2e-6)
        for a, b in zip(
            engines[0].model.parameters(), engines[1].model.parameters(), strict=True
        ):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=3e-6)
            torch.testing.assert_close(a, b, rtol=3e-5, atol=5e-6)
        for name, value in actual.metrics.items():
            if name.startswith(("engram/", "batch/")):
                assert value == pytest.approx(
                    expected.metrics[name], rel=3e-5, abs=3e-6
                ), name
    finally:
        for engine in engines:
            engine.close()


@pytest.mark.parametrize("invalid_index", [0, 1])
def test_primary_and_secondary_reject_malformed_position_masks(tmp_path, invalid_index):
    cfg = configured(tmp_path, True)
    cfg = cfg.model_copy(
        update={
            "runtime": cfg.runtime.model_copy(update={"secondary_device_indices": (0,)})
        }
    )
    batches = collate_document_update(
        rows_for([7, 7]), width=1, eos=3, max_length=64, padding_multiple=16
    )
    batches[invalid_index].position_mask[0, 1] = False
    total = sum(int(np.count_nonzero(batch.targets != -100)) for batch in batches)
    engine = PyTorchEngine()
    try:
        engine.validate(cfg)
        engine.initialize(cfg)
        with pytest.raises(ValueError, match="right-padded prefixes"):
            engine.train_update(batches, 1, total)
    finally:
        engine.close()


def test_secondary_dynamic_compilation_matches_primary(tmp_path, monkeypatch):
    cfg = configured(tmp_path, True)
    cfg = cfg.model_copy(
        update={
            "runtime": cfg.runtime.model_copy(
                update={
                    "secondary_device_indices": (0,),
                    "compile_dynamic_shapes": True,
                    "compile_transformer_blocks": True,
                    "compile_output_head_loss": True,
                }
            )
        }
    )
    calls = []

    def compile_stub(function, **kwargs):
        calls.append(kwargs)
        return function

    monkeypatch.setattr(torch, "compile", compile_stub)
    engine = PyTorchEngine()
    try:
        engine.validate(cfg)
        engine.initialize(cfg)
        assert len(calls) == 2 * (cfg.model.num_layers + 1)
        assert all(call == {"dynamic": True, "fullgraph": True} for call in calls)
    finally:
        engine.close()
