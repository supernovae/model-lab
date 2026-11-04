"""Whole-document CPU paths preserve boundaries, supervision and memory history."""

import numpy as np
import pytest
import torch

from sparselab.config.models import AttentionConfig, ModelConfig, RunConfig
from sparselab.data.story_packing import MatchedStoryDataset
from sparselab.model.transformer import DenseLM


def test_unpadded_ngram_has_no_cross_document_attention_or_history():
    torch.manual_seed(7)
    model = DenseLM(
        ModelConfig(
            vocab_size=260,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=16,
            memory="ngram",
            memory_table_size=32,
            memory_ngram_size=3,
            memory_dim=8,
        ),
        AttentionConfig(),
    )
    model.eval()

    def observe(prefix):
        ids = np.array([*prefix, 3, 9, 10, 11, 3], dtype=np.int32)
        plan = MatchedStoryDataset(
            ids, 16, 3, layout="story", targets_per_update=1, unpadded_documents=True
        )
        row = plan.numpy_microblock(1)
        inputs = torch.from_numpy(row[0]).unsqueeze(0)
        return row, model(inputs), model.memory.addresses(inputs)

    a, la, ha = observe([4, 5])
    b, lb, hb = observe([6, 7, 8])
    np.testing.assert_array_equal(a[0], [3, 9, 10, 11])
    np.testing.assert_array_equal(a[1], [9, 10, 11, 3])
    np.testing.assert_array_equal(a[0], b[0])
    torch.testing.assert_close(la, lb, rtol=0, atol=0)
    torch.testing.assert_close(ha, hb, rtol=0, atol=0)
    # Causality remains intact inside the document as well.
    altered = torch.tensor([[3, 9, 22, 23]])
    torch.testing.assert_close(la[:, :2], model(altered)[:, :2], rtol=0, atol=0)


def test_unpadded_targets_resume_and_overlength_refusal():
    ids = np.array([4, 3, 5, 6, 3, 7, 3], dtype=np.int32)
    plan = MatchedStoryDataset(
        ids, 4, 3, layout="story", targets_per_update=3, unpadded_documents=True
    )
    rows = [row for i in range(len(plan)) for row in plan.update_rows(i, len(ids))]
    np.testing.assert_array_equal(np.concatenate([r[1] for r in rows]), ids)
    assert sum(r[0].size for r in rows) == len(ids)
    assert all(r[0][0] == 3 for r in rows)
    assert plan.layout_costs()["story"]["padded_slots_per_pass"] == 0
    with pytest.raises(ValueError, match="document-aligned"):
        next(plan.update_rows(1, 1))
    with pytest.raises(ValueError, match="document-aligned"):
        plan.horizon(1)
    with pytest.raises(ValueError, match="whole stories"):
        MatchedStoryDataset(ids, 2, 3, layout="story", unpadded_documents=True)


@pytest.mark.parametrize("multiport", [False, True])
def test_unpadded_native_train_resume(tmp_path, multiport):
    from test_training import config, equal

    from sparselab.data.packing import prepare_data
    from sparselab.data.tokenizer import load_tokenizer
    from sparselab.training.checkpoints import CheckpointManager
    from sparselab.training.trainer import train

    base = config(tmp_path)
    tokenizer = load_tokenizer(base.tokenizer.path)
    data = prepare_data(base, tokenizer)
    ends = np.flatnonzero(data.train == 3) + 1
    # Existing fixture has complete synthetic records; use full documents.
    length = max(int(np.diff(np.r_[0, ends]).max()), 256)
    plan = MatchedStoryDataset(
        data.train,
        length,
        3,
        layout="story",
        targets_per_update=128,
        unpadded_documents=True,
    )
    raw = base.model_dump(mode="json")
    raw["model"].update(
        max_seq_len=length,
        memory="ngram",
        memory_table_size=32,
        memory_ngram_size=3,
        memory_dim=8,
    )
    if multiport:
        from sparselab.address_hash import engram_table_sizes
        from sparselab.memory_tokenizer import build_memory_token_map

        raw["model"].update(
            num_layers=2,
            memory_variant="context_key",
            memory_injection="after_block",
            memory_block_indices=[0, 1],
            memory_conv_kernel_size=4,
            memory_ngram_orders=[2, 3],
            memory_hash_heads=2,
            memory_table_size=31,
            memory_table_sizes=list(engram_table_sizes(31, 8)),
            memory_addressing="prime_xor_v1",
            memory_token_mapping=build_memory_token_map(
                base.tokenizer.path, mode="identity"
            ).to_dict(),
        )
    budget = plan.targets_for_updates(2)
    raw["training"].update(
        seq_len=length,
        micro_batch_size=1,
        max_steps=2,
        max_tokens=budget,
        matched_stories={
            "layout": "story",
            "targets_per_update": 128,
            "unpadded_documents": True,
        },
    )
    raw["optimizer"]["warmup_steps"] = 0
    raw["evaluation"].update(max_batches=2, batch_size=1)
    measured = RunConfig.model_validate(raw)
    train(measured, run_id="whole")
    train(measured, run_id="part", stop_after_step=1)
    train(
        measured,
        run_id="resumed",
        resume=measured.logging.root_dir / "part/checkpoints/latest.json",
    )
    states = [
        CheckpointManager(measured.logging.root_dir / n).load(
            measured.logging.root_dir / n / "checkpoints/latest.json"
        )
        for n in ["whole", "resumed"]
    ]
    assert states[0].tokens_seen == states[1].tokens_seen == budget
    equal(states[0].model, states[1].model)
    equal(states[0].optimizer, states[1].optimizer)
    equal(states[0].rng, states[1].rng)
    raw["training"]["micro_batch_size"] = 2
    with pytest.raises(ValueError, match="batch size one"):
        RunConfig.model_validate(raw)
