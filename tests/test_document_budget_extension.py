"""Full-state variable-group extension and append-only stream boundaries on CPU."""

import json
from pathlib import Path

import pytest
from test_training import config, equal

from sparselab.config.models import RunConfig
from sparselab.data.packing import prepare_data
from sparselab.data.story_packing import MatchedStoryDataset
from sparselab.data.tokenizer import load_tokenizer
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import read_manifest, sha256_file
from sparselab.training.trainer import train


@pytest.fixture
def document_parent(tmp_path):
    base = config(tmp_path)
    tokenizer = load_tokenizer(base.tokenizer.path)
    texts = ["alpha.", "beta."]
    validation = "A different held out account."
    parent_file = tmp_path / "parent.jsonl"
    valid = tmp_path / "validation.jsonl"
    parent_file.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    valid.write_text(json.dumps({"text": validation}) + "\n")
    tokens = sum(
        len(tokenizer.encode(t, add_special_tokens=False).ids) + 1 for t in texts
    )
    raw = base.model_dump(mode="json")
    raw["model"].update(
        max_seq_len=64,
        memory="ngram",
        memory_table_size=31,
        memory_ngram_size=3,
        memory_dim=4,
    )
    raw["dataset"].update(
        source="local_text",
        revision="parent-whole-document-fixture",
        license="CC0-1.0",
        train_path=str(parent_file),
        validation_path=str(valid),
        train_max_documents=2,
        validation_max_documents=1,
        train_max_tokens=tokens,
        validation_max_tokens=len(
            tokenizer.encode(validation, add_special_tokens=False).ids
        )
        + 1,
    )
    raw["training"].update(
        seq_len=64,
        micro_batch_size=1,
        max_steps=1,
        max_tokens=tokens,
        matched_stories={
            "layout": "story",
            "targets_per_update": tokens + 1,
            "unpadded_documents": True,
        },
    )
    raw["optimizer"].update(
        warmup_steps=0, memory_table_lr_multiplier=5.0, memory_table_weight_decay=0.0
    )
    raw["checkpoint"].update(every_steps=1)
    raw["evaluation"].update(every_steps=1, max_batches=1, batch_size=1)
    source = RunConfig.model_validate(raw)
    train(source, run_id="parent")
    root = source.logging.root_dir / "parent"
    generation = next((root / "checkpoints").glob("step_00000001_gen_*"))
    return source, root, generation, tokenizer


def append_config(document_parent, tmp_path):
    source, _root, _generation, tokenizer = document_parent
    raw = source.model_dump(mode="json")
    old_file = Path(raw["dataset"]["train_path"])
    additions = [
        "one two three four five six seven eight.",
        "several complete words describing another process.",
        "this final document contains further distinct details.",
    ]
    path = tmp_path / "expanded.jsonl"
    path.write_bytes(
        old_file.read_bytes()
        + b"".join((json.dumps({"text": t}) + "\n").encode() for t in additions)
    )
    total = source.training.max_tokens + sum(
        len(tokenizer.encode(t, add_special_tokens=False).ids) + 1 for t in additions
    )
    raw["dataset"].update(
        train_path=str(path),
        revision="expanded-whole-document-fixture",
        train_max_documents=5,
        train_max_tokens=total,
    )
    raw["training"].update(max_tokens=total, max_steps=5)
    raw["training"]["matched_stories"]["frozen_update_ends"] = [
        source.training.max_tokens
    ]
    raw["optimizer"]["decay_steps"] = source.training.max_steps
    measured = RunConfig.model_validate(raw)
    data = prepare_data(measured, tokenizer)
    plan = MatchedStoryDataset(
        data.train,
        64,
        3,
        layout="story",
        targets_per_update=source.training.matched_stories.targets_per_update,
        unpadded_documents=True,
        frozen_update_ends=(source.training.max_tokens,),
    )
    raw["training"]["max_steps"] = plan.horizon(total)
    return RunConfig.model_validate(raw)


def test_native_append_preserves_optimizer_cursor_and_exact_resume(
    document_parent, tmp_path
):
    source, _root, generation, _ = document_parent
    target = append_config(document_parent, tmp_path)
    before = {p.name: sha256_file(p) for p in generation.iterdir()}
    assert target.training.max_steps >= 3
    train(
        target,
        extend_budget=generation,
        append_data=True,
        run_id="part",
        stop_after_step=2,
    )
    part = target.logging.root_dir / "part"
    snapshot = CheckpointManager(part).load(part / "checkpoints/latest.json")
    assert snapshot.step == 2 and snapshot.cursor == (0, 2)
    assert all(float(v["step"]) == 2 for v in snapshot.optimizer["state"].values())
    decision = next(
        d
        for d in read_manifest(part / "manifest.json")["resource_decisions"]
        if d["kind"] == "budget_extension"
    )
    assert (
        decision["append_data"]
        and decision["parent_prefix_tokens"] == source.training.max_tokens
    )
    assert decision["decay_steps"] == 1 and decision["variable_document_groups"]
    train(
        target,
        resume=part / "checkpoints/latest.json",
        run_id="resumed",
        stop_after_step=3,
    )
    train(
        target,
        extend_budget=generation,
        append_data=True,
        run_id="whole",
        stop_after_step=3,
    )
    states = [
        CheckpointManager(target.logging.root_dir / n).load(
            target.logging.root_dir / n / "checkpoints/latest.json"
        )
        for n in ["resumed", "whole"]
    ]
    equal(states[0].model, states[1].model)
    equal(states[0].optimizer, states[1].optimizer)
    equal(states[0].rng, states[1].rng)
    assert states[0].tokens_seen == states[1].tokens_seen and states[
        0
    ].cursor == states[1].cursor == (0, 3)
    assert before == {p.name: sha256_file(p) for p in generation.iterdir()}


@pytest.mark.parametrize("damage", ["prefix", "heldout", "boundary", "group_settings"])
def test_append_refuses_changed_history_or_grouping(document_parent, tmp_path, damage):
    source, _root, generation, _ = document_parent
    target = append_config(document_parent, tmp_path)
    raw = target.model_dump(mode="json")
    if damage == "prefix":
        path = Path(raw["dataset"]["train_path"])
        s = path.read_text()
        path.write_text(s.replace("alpha", "wrong", 1))
    elif damage == "heldout":
        path = tmp_path / "wrong-validation.jsonl"
        path.write_text('{"text":"Different evaluation."}\n')
        raw["dataset"].update(validation_path=str(path), validation_max_tokens=100)
    elif damage == "boundary":
        raw["training"]["matched_stories"]["frozen_update_ends"] = []
    else:
        raw["training"]["matched_stories"]["targets_per_update"] += 1
    with pytest.raises(ValueError):
        train(
            RunConfig.model_validate(raw),
            extend_budget=generation,
            append_data=True,
            run_id="refused",
        )
    assert not (source.logging.root_dir / "refused").exists()


def test_variable_budget_extension_can_chain_without_restarting_schedule(
    document_parent,
):
    source, _root, generation, _ = document_parent
    raw = source.model_dump(mode="json")
    raw["training"].update(max_steps=2, max_tokens=source.training.max_tokens * 2)
    raw["optimizer"]["decay_steps"] = 1
    second = RunConfig.model_validate(raw)
    train(second, extend_budget=generation, run_id="second")
    run = source.logging.root_dir / "second"
    snap = CheckpointManager(run).load(run / "checkpoints/latest.json")
    assert snap.step == 2 and snap.tokens_seen == source.training.max_tokens * 2
    raw["training"].update(max_steps=3, max_tokens=source.training.max_tokens * 3)
    third = RunConfig.model_validate(raw)
    next_generation = next((run / "checkpoints").glob("step_00000002_gen_*"))
    train(third, extend_budget=next_generation, run_id="third")
    final = CheckpointManager(source.logging.root_dir / "third").load(
        source.logging.root_dir / "third/checkpoints/latest.json"
    )
    assert final.step == 3 and final.tokens_seen == source.training.max_tokens * 3
    assert final.schedule["decay_steps"] == 1
