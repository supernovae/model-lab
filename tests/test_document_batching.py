"""Semantic equivalence of independent rows, not bitwise reduction equivalence."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from test_engram_integration import model_config
from test_training import config as training_config

from klab.address_hash import engram_table_sizes
from klab.config.models import AttentionConfig, ModelConfig, RunConfig
from klab.data.document_batching import collate_document_update
from klab.data.story_packing import MatchedStoryDataset
from klab.engines.pytorch import PyTorchEngine
from klab.model.transformer import DenseLM
from klab.training.continuation import runtime_optimization_transition


def rows_for(lengths, masked=True):
    rng = np.random.default_rng(17)
    rows = []
    for length in lengths:
        x = rng.integers(4, 24, size=length, dtype=np.int64)
        x[0] = 3
        y = np.concatenate((x[1:], np.array([3], dtype=np.int64)))
        if masked and length > 2:
            y[1::3] = -100
        rows.append((x, y, None, None, None, None))
    return rows


def tiny_model(tmp_path, memory, max_length=257):
    if memory:
        return model_config(
            tmp_path,
            "deepseek_safe_v1",
            max_seq_len=max_length,
            memory_block_index=None,
            memory_block_indices=(0, 1),
            memory_table_sizes=engram_table_sizes(31, 8),
        )
    return ModelConfig(
        vocab_size=260,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        ffn_dim=32,
        max_seq_len=max_length,
    )


@pytest.mark.parametrize("memory", [False, True])
@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.parametrize("implementation", ["reference", "sdpa"])
def test_full_model_logits_loss_gradients_and_adamw(
    tmp_path, memory, width, implementation
):
    torch.manual_seed(7)
    reference = DenseLM(
        tiny_model(tmp_path, memory), AttentionConfig(implementation=implementation)
    )
    candidate = deepcopy(reference)
    rows = rows_for(
        [257, 1, 3, 13, 32, 7, 129]
    )  # partial mb4/8; masked prompts and EOS
    total = sum(int((r[1] != -100).sum()) for r in rows)
    expected = {}
    loss_a = torch.zeros(())
    for x, y, *_ in rows:
        logits, aux = reference.forward_with_aux(
            torch.from_numpy(x)[None],
            valid_target_mask=torch.from_numpy(y != -100)[None],
        )
        assert float(aux) == 0  # dense FFN is the admitted path
        expected[tuple(x)] = logits.detach()[0]
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            torch.from_numpy(y),
            ignore_index=-100,
            reduction="sum",
        )
        (loss / total).backward()
        loss_a += loss.detach()
    loss_b = torch.zeros(())
    for batch in collate_document_update(
        rows, width=width, eos=3, max_length=257, padding_multiple=32
    ):
        logits, aux = candidate.forward_with_aux(
            torch.from_numpy(batch.inputs),
            position_mask=torch.from_numpy(batch.position_mask),
            valid_target_mask=torch.from_numpy(batch.targets != -100),
        )
        for i, mask in enumerate(batch.position_mask):
            length = int(mask.sum())
            torch.testing.assert_close(
                logits[i, :length],
                expected[tuple(batch.inputs[i, :length])],
                rtol=3e-5,
                atol=3e-6,
            )
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            torch.from_numpy(batch.targets).flatten(),
            ignore_index=-100,
            reduction="sum",
        )
        ((loss + aux * int((batch.targets != -100).sum())) / total).backward()
        loss_b += loss.detach()
    torch.testing.assert_close(loss_a, loss_b, rtol=2e-6, atol=2e-4)
    for (name, a), (_, b) in zip(
        reference.named_parameters(), candidate.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=3e-6, msg=name)
    for model in (reference, candidate):
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    for (name, a), (_, b) in zip(
        reference.named_parameters(), candidate.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("memory", [False, True])
def test_padding_other_rows_and_prior_calls_cannot_leak(tmp_path, memory):
    model = DenseLM(tiny_model(tmp_path, memory), AttentionConfig()).eval()
    batch = collate_document_update(
        rows_for([9, 31, 64, 17]), width=4, eos=3, max_length=257
    )[0]
    x = torch.from_numpy(batch.inputs)
    with torch.no_grad():
        original = model(x)
        changed = x.clone()
        changed[0, 9:] = 77  # right padding including ngram and convolution history
        changed[1:] = 123  # independent documents
        actual = model(changed)
        torch.testing.assert_close(actual[0, :9], original[0, :9], rtol=0, atol=0)
        model(torch.full((4, 128), 99))
        torch.testing.assert_close(model(x), original, rtol=0, atol=0)


def configured(tmp_path, memory=True, width=4):
    d = training_config(tmp_path).model_dump(mode="json")
    d["model"] = tiny_model(tmp_path, memory, 64).model_dump(mode="json")
    d["training"].update(
        micro_batch_size=width,
        gradient_accumulation=1,
        seq_len=64,
        matched_stories={
            "layout": "story",
            "targets_per_update": 100,
            "unpadded_documents": True,
            "document_batching": "padded",
            "padding_multiple": 16,
        },
    )
    d["evaluation"]["batch_size"] = 1
    return RunConfig.model_validate(d)


@pytest.mark.parametrize("memory", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_native_engine_step_and_padding_aware_diagnostics(tmp_path, memory, bounded):
    cfg = configured(tmp_path, memory)
    rows = rows_for([61, 1, 3, 13, 32, 7, 19])
    rows[2][1][:] = -100  # all-masked real context still counts in diagnostics
    total = sum(int((r[1] != -100).sum()) for r in rows)
    engines = []
    try:
        for width in (1, 4, 8):
            engine = PyTorchEngine()
            engine.validate(cfg)
            torch.manual_seed(10)
            engine.initialize(cfg)
            engines.append(engine)
        updates = [
            engine.train_update(
                collate_document_update(
                    rows,
                    width=w,
                    eos=3,
                    max_length=64,
                    padding_multiple=16,
                    max_padded_slots=64 if bounded else None,
                    max_attention_cells=4096 if bounded else None,
                ),
                1,
                total,
            )
            for engine, w in zip(engines, (1, 4, 8), strict=True)
        ]
        for engine, update in zip(engines[1:], updates[1:], strict=True):
            assert update.committed_targets == total
            assert (
                update.metrics["optimizer/learning_rate"]
                == updates[0].metrics["optimizer/learning_rate"]
            )
            assert update.loss_sum == pytest.approx(updates[0].loss_sum, rel=2e-6)
            for (name, a), (_, b) in zip(
                engines[0].model.named_parameters(),
                engine.model.named_parameters(),
                strict=True,
            ):
                torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=3e-6)
                # Near-zero gradients amplify FP32 reduction differences through
                # Adam's epsilon division (observed max 3.60e-6 at lr=0.0015).
                torch.testing.assert_close(a, b, rtol=3e-5, atol=5e-6)
                state_a = engines[0].optimizer.state[a]
                state_b = engine.optimizer.state[b]
                assert state_a.keys() == state_b.keys()
                for field in state_a:
                    torch.testing.assert_close(
                        state_a[field], state_b[field], rtol=2e-4, atol=3e-6
                    )
            for key, value in update.metrics.items():
                if key.startswith("engram/"):
                    assert value == pytest.approx(
                        updates[0].metrics[key], rel=3e-5, abs=3e-6
                    ), key
        if memory:
            assert (
                updates[0].metrics["engram/lookup_count"]
                == sum(len(r[0]) for r in rows) * 8
            )
    finally:
        for engine in engines:
            engine.close()
    assert not torch.xpu.is_initialized()


def test_groups_masks_bounds_and_continuation_contract(tmp_path):
    rows = rows_for([1, 13, 7, 64, 3, 31, 8], masked=False)
    ids = np.concatenate([row[1] for row in rows]).astype(np.int32)
    dataset = MatchedStoryDataset(
        ids,
        64,
        3,
        layout="story",
        targets_per_update=30,
        unpadded_documents=True,
        frozen_update_ends=(21,),
    )
    for group in range(len(dataset)):
        original = list(dataset.update_rows(group, len(ids)))
        for width in (1, 4, 8):
            batches = collate_document_update(
                original, width=width, eos=3, max_length=64, padding_multiple=16
            )
            recovered = [
                (tuple(b.inputs[i, mask]), tuple(b.targets[i, mask]))
                for b in batches
                for i, mask in enumerate(b.position_mask)
            ]
            assert sorted(recovered) == sorted(
                (tuple(x), tuple(y)) for x, y, *_ in original
            )
            assert all(
                b.inputs.shape[0] <= width and b.inputs.shape[1] <= 64 for b in batches
            )
    target = configured(tmp_path)
    d = target.model_dump(mode="json")
    d["training"]["micro_batch_size"] = 1
    d["training"]["matched_stories"].pop("document_batching")
    source = RunConfig.model_validate(d)
    decision = runtime_optimization_transition(source, target, "immutable-parent")
    assert decision["effective_batch_contract"] == "original-document-update-groups"
    assert decision["parent_checkpoint_sha256"] == "immutable-parent"
    assert decision["resume_level"] == "best_effort"
    for section, field, value in [
        ("matched_stories", "targets_per_update", 101),
        ("matched_stories", "frozen_update_ends", [13]),
        ("training", "max_tokens", target.training.max_tokens + 1),
    ]:
        bad = target.model_dump(mode="json")
        (
            bad["training"]["matched_stories"]
            if section == "matched_stories"
            else bad["training"]
        )[field] = value
        with pytest.raises(ValueError, match="scientific or unsupported"):
            runtime_optimization_transition(
                source, RunConfig.model_validate(bad), "immutable-parent"
            )
    bad = source.model_dump(mode="json")
    bad["training"]["micro_batch_size"] = 4
    with pytest.raises(ValueError, match="batch size one"):
        RunConfig.model_validate(bad)


@pytest.mark.parametrize("lengths", [[4096, 1, 7, 33, 2]])
def test_context_limit_full_model_prefix(tmp_path, lengths):
    model = DenseLM(
        tiny_model(tmp_path, True, 4096), AttentionConfig(implementation="sdpa")
    ).eval()
    rows = rows_for(lengths)
    with torch.no_grad():
        batch = collate_document_update(rows, width=8, eos=3, max_length=4096)[-1]
        actual = model(torch.from_numpy(batch.inputs))
        for i, mask in enumerate(batch.position_mask):
            n = int(mask.sum())
            expected = model(torch.from_numpy(batch.inputs[i : i + 1, :n]))
            torch.testing.assert_close(actual[i, :n], expected[0], rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize(
    "change", ["moe", "implicit_eval", "accumulation", "not_whole"]
)
def test_unsupported_batching_contracts_refuse(tmp_path, change):
    raw = configured(tmp_path).model_dump(mode="json")
    if change == "moe":
        raw["model"].update(ffn="moe", num_experts=4, experts_per_token=2)
    elif change == "implicit_eval":
        raw["evaluation"]["batch_size"] = None
    elif change == "accumulation":
        raw["training"]["gradient_accumulation"] = 2
    else:
        raw["training"]["matched_stories"]["unpadded_documents"] = False
    with pytest.raises(ValueError):
        RunConfig.model_validate(raw)


def test_collator_refuses_truncation_and_sidecars():
    rows = rows_for([9])
    with pytest.raises(ValueError, match="bounded"):
        collate_document_update(rows, width=4, eos=3, max_length=8)
    x, y, *_ = rows[0]
    with pytest.raises(ValueError, match="sidecars"):
        collate_document_update(
            [(x, y, np.zeros_like(x), None, None, None)], width=4, eos=3, max_length=16
        )
