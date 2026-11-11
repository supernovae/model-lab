"""CPU-only suffix migration, bounded collation and full-state replay."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_training import config, equal

from sparselab.config.models import RunConfig
from sparselab.data.document_batching import collate_document_update
from sparselab.data.packing import prepare_data
from sparselab.data.story_packing import MatchedStoryDataset
from sparselab.data.tokenizer import load_tokenizer
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.continuation import (
    target_group_transition,
    verify_regroup_cursor,
)
from sparselab.training.manifest import read_manifest
from sparselab.training.trainer import train


def test_suffix_masks_order_boundaries_and_envelope():
    lengths = [7, 29, 3, 17, 32, 2, 27, 11, 31, 4, 23, 9]
    ids = np.concatenate(
        [np.array([i + 4] * (n - 1) + [3], np.int32) for i, n in enumerate(lengths)]
    )
    old = MatchedStoryDataset(
        ids,
        32,
        3,
        layout="story",
        targets_per_update=30,
        unpadded_documents=True,
        frozen_update_ends=(sum(lengths[:8]),),
    )
    cut = int(old.group_offsets[2])
    new = MatchedStoryDataset(
        ids,
        32,
        3,
        layout="story",
        targets_per_update=80,
        unpadded_documents=True,
        frozen_update_ends=old.frozen_update_ends,
        target_group_transition={
            "after_tokens": cut,
            "previous_targets_per_update": 30,
        },
    )
    assert new.group_offsets[:3].tolist() == old.group_offsets[:3].tolist()
    assert sum(lengths[:8]) in new.group_offsets
    seen = []
    for group in range(2, len(new)):
        rows = list(new.update_rows(group, len(ids)))
        seen.extend(np.concatenate([r[1] for r in rows]).tolist())
        # Masked prompt positions remain real context while supervision stays exact.
        for row in rows:
            row[1][1::3] = -100
        expected = sorted(tuple(row[1]) for row in rows)
        batches = collate_document_update(
            rows,
            width=4,
            eos=3,
            max_length=32,
            padding_multiple=8,
            max_padded_slots=64,
            max_attention_cells=2048,
        )
        actual = []
        for batch in batches:
            b, l = batch.inputs.shape
            assert b <= 4 and b * l <= 64 and b * l * l <= 2048
            actual.extend(
                tuple(y[m])
                for y, m in zip(batch.targets, batch.position_mask, strict=True)
            )
        assert sorted(actual) == expected
        assert sum(np.count_nonzero(b.targets != -100) for b in batches) == sum(
            np.count_nonzero(r[1] != -100) for r in rows
        )
    assert seen == ids[cut:].tolist()
    with pytest.raises(ValueError, match="single source-order pass"):
        new.horizon(len(ids) + 1)


def test_native_regroup_save_reload_adam_rng_cursor_and_floor(tmp_path):
    base = config(tmp_path)
    tokenizer = load_tokenizer(base.tokenizer.path)
    texts = [
        "one two.",
        "alpha.",
        "a longer tiny document.",
        "beta.",
        "some more words here.",
        "gamma.",
    ] * 4
    path = tmp_path / "documents.jsonl"
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    valid = tmp_path / "validation.jsonl"
    valid.write_text(json.dumps({"text": "held out text."}) + "\n")
    tokens = sum(
        len(tokenizer.encode(t, add_special_tokens=False).ids) + 1 for t in texts
    )
    raw = base.model_dump(mode="json")
    raw["model"].update(max_seq_len=64)
    raw["dataset"].update(
        source="local_text",
        revision="regroup-fixture",
        license="CC0",
        train_path=str(path),
        validation_path=str(valid),
        train_max_documents=len(texts),
        validation_max_documents=1,
        train_max_tokens=tokens,
        validation_max_tokens=64,
    )
    raw["training"].update(
        seq_len=64,
        micro_batch_size=4,
        max_steps=len(texts),
        max_tokens=tokens,
        matched_stories={
            "layout": "story",
            "targets_per_update": 1,
            "unpadded_documents": True,
            "document_batching": "padded",
            "padding_multiple": 8,
        },
    )
    raw["optimizer"].update(warmup_steps=0, decay_steps=2)
    raw["evaluation"].update(every_steps=100, max_batches=1, batch_size=1)
    raw["checkpoint"].update(every_steps=100)
    source = RunConfig.model_validate(raw)
    train(source, run_id="old", stop_after_step=2)
    root = source.logging.root_dir / "old"
    snap = CheckpointManager(root).load(root / "checkpoints/latest.json")
    pointer = json.loads((root / "checkpoints/latest.json").read_text())
    generation = root / "checkpoints" / pointer["relative_path"]
    policy = {
        "after_tokens": snap.tokens_seen,
        "previous_targets_per_update": 1,
        "parent_checkpoint_sha256": snap.checkpoint_sha256,
    }
    raw["training"]["matched_stories"].update(
        targets_per_update=40,
        target_group_transition=policy,
        max_padded_slots=128,
        max_attention_cells=8192,
    )
    data = prepare_data(source, tokenizer)
    plan = MatchedStoryDataset(
        data.train,
        64,
        3,
        layout="story",
        targets_per_update=40,
        unpadded_documents=True,
        target_group_transition=policy,
    )
    raw["training"]["max_steps"] = plan.horizon(tokens)
    target = RunConfig.model_validate(raw)
    target_group_transition(source, target, snap)
    verify_regroup_cursor(plan, snap)
    for section, field, value in [
        ("optimizer", "peak", source.optimizer.peak * 2),
        ("runtime", "compile_transformer_blocks", True),
        ("dataset", "revision", "changed-stream"),
    ]:
        changed = target.model_dump(mode="json")
        changed[section][field] = value
        with pytest.raises(ValueError, match="preserve"):
            target_group_transition(source, RunConfig.model_validate(changed), snap)
    with pytest.raises(ValueError, match="checkpoint"):
        target_group_transition(
            source, target, replace(snap, tokens_seen=snap.tokens_seen + 1)
        )
    train(target, resume=generation, run_id="part", stop_after_step=3)
    part = target.logging.root_dir / "part"
    saved = CheckpointManager(part).load(part / "checkpoints/latest.json")
    assert saved.cursor == (0, 3) and saved.tokens_seen == int(plan.group_offsets[3])
    assert all(float(v["step"]) == 3 for v in saved.optimizer["state"].values())
    assert all(
        v["lr"] == target.optimizer.floor * v.get("lr_multiplier", 1)
        for v in saved.optimizer["param_groups"]
    )
    assert any(
        d["kind"] == "target_group_transition"
        for d in read_manifest(part / "manifest.json")["resource_decisions"]
    )
    train(
        target,
        resume=part / "checkpoints/latest.json",
        run_id="resumed",
        stop_after_step=4,
    )
    train(target, resume=generation, run_id="whole", stop_after_step=4)
    states = [
        CheckpointManager(target.logging.root_dir / n).load(
            target.logging.root_dir / n / "checkpoints/latest.json"
        )
        for n in ["resumed", "whole"]
    ]
    for field in ["model", "optimizer", "rng", "cursor", "schedule", "tokens_seen"]:
        equal(getattr(states[0], field), getattr(states[1], field))
