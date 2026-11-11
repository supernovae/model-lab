"""Full-state variable-group extension and append-only stream boundaries on CPU."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_training import config, equal

from klab.config.models import RunConfig
from klab.data.packing import prepare_data
from klab.data.story_packing import MatchedStoryDataset
from klab.data.tokenizer import load_tokenizer
from klab.training.checkpoints import CheckpointManager
from klab.training.manifest import read_manifest, sha256_file
from klab.training.trainer import train


@pytest.fixture
def document_parent(tmp_path, request):
    base = config(tmp_path)
    tokenizer = load_tokenizer(base.tokenizer.path)
    texts = ["alpha.", "beta."]
    validation = "A different held out account."
    validation_documents = getattr(request, "param", 1)
    parent_file = tmp_path / "parent.jsonl"
    valid = tmp_path / "validation.jsonl"
    parent_file.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    valid.write_text((json.dumps({"text": validation}) + "\n") * validation_documents)
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
        validation_max_documents=validation_documents,
        train_max_tokens=tokens,
        validation_max_tokens=(
            len(tokenizer.encode(validation, add_special_tokens=False).ids) + 1
        )
        * validation_documents,
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


@pytest.fixture
def checkpoint_phase(document_parent, tmp_path):
    _source, _root, generation, tokenizer = document_parent
    planned = append_config(document_parent, tmp_path)
    train(
        planned,
        extend_budget=generation,
        append_data=True,
        run_id="planned",
        stop_after_step=2,
    )
    root = planned.logging.root_dir / "planned"
    parent = next((root / "checkpoints").glob("step_00000002_gen_*"))
    snapshot = CheckpointManager(root).load(parent)
    ids = np.load(root / "data/train.npy", mmap_mode="r")
    document_count = int(np.count_nonzero(ids[: snapshot.tokens_seen] == 3))
    prefix = planned.dataset.train_path.read_text().splitlines()[:document_count]
    path = tmp_path / "checkpoint-phase.jsonl"
    path.write_text("\n".join(prefix) + '\n{"text":"new."}\n')
    raw = planned.model_dump(mode="json")
    total = (
        snapshot.tokens_seen
        + len(tokenizer.encode("new.", add_special_tokens=False).ids)
        + 1
    )
    raw["dataset"].update(
        train_path=str(path),
        revision="explicit-checkpoint-phase",
        train_max_documents=document_count + 1,
        train_max_tokens=total,
    )
    ends = [*planned.training.matched_stories.frozen_update_ends, snapshot.tokens_seen]
    raw["training"]["matched_stories"]["frozen_update_ends"] = ends
    raw["training"]["max_tokens"] = total
    candidate = RunConfig.model_validate(raw)
    data = prepare_data(candidate, tokenizer)
    plan = MatchedStoryDataset(
        data.train,
        64,
        3,
        layout="story",
        targets_per_update=planned.training.matched_stories.targets_per_update,
        unpadded_documents=True,
        frozen_update_ends=tuple(ends),
    )
    raw["training"]["max_steps"] = plan.horizon(total)
    return planned, RunConfig.model_validate(raw), root, parent, snapshot


def test_explicit_checkpoint_phase_retains_full_state_with_smaller_planned_cap(
    checkpoint_phase,
):
    source, target, root, parent, prior = checkpoint_phase
    assert prior.step == 2 < source.training.max_steps
    assert target.training.max_steps < source.training.max_steps
    assert target.training.max_tokens < source.training.max_tokens
    before = {p.name: sha256_file(p) for p in parent.iterdir()}
    with pytest.raises(ValueError, match="terminal"):
        train(target, extend_budget=parent, append_data=True, run_id="not_explicit")
    train(
        target,
        extend_budget=parent,
        append_data=True,
        append_from_checkpoint=True,
        run_id="phase",
    )
    child = source.logging.root_dir / "phase"
    final = CheckpointManager(child).load(child / "checkpoints/latest.json")
    assert (
        final.step == prior.step + 1 and final.tokens_seen == target.training.max_tokens
    )
    assert all(
        float(v["step"]) == final.step for v in final.optimizer["state"].values()
    )
    decision = next(
        d
        for d in read_manifest(child / "manifest.json")["resource_decisions"]
        if d["kind"] == "budget_extension"
    )
    assert (
        decision["append_from_checkpoint"]
        and decision["parent_consumed_tokens"] == prior.tokens_seen
    )
    assert (
        decision["unconsumed_source_tokens_deferred"] > 0
        and decision["decay_steps"] == 1
    )
    assert before == {p.name: sha256_file(p) for p in parent.iterdir()}
    assert (root / "data/train.npy").is_file()


def test_checkpoint_phase_rejects_changed_trained_history(checkpoint_phase):
    source, target, _root, parent, _prior = checkpoint_phase
    path = target.dataset.train_path
    path.write_text(path.read_text().replace("alpha", "wrong", 1))
    with pytest.raises(ValueError, match="trained token prefix"):
        train(
            target,
            extend_budget=parent,
            append_data=True,
            append_from_checkpoint=True,
            run_id="damaged_prefix",
        )
    assert not (source.logging.root_dir / "damaged_prefix").exists()


def test_append_composes_explicit_runtime_transition_without_reset(
    document_parent, tmp_path
):
    source, _root, parent, _tokenizer = document_parent
    raw = append_config(document_parent, tmp_path).model_dump(mode="json")
    raw["attention"]["implementation"] = "sdpa"
    raw["runtime"]["memory"]["estimate_policy"] = "observe"
    target = RunConfig.model_validate(raw)
    with pytest.raises(ValueError, match="architecture semantics"):
        train(target, extend_budget=parent, append_data=True, run_id="not_optimized")
    train(
        target,
        extend_budget=parent,
        append_data=True,
        allow_runtime_optimization=True,
        run_id="optimized",
        stop_after_step=2,
    )
    child = source.logging.root_dir / "optimized"
    snapshot = CheckpointManager(child).load(child / "checkpoints/latest.json")
    assert snapshot.step == 2 and all(
        float(v["step"]) == 2 for v in snapshot.optimizer["state"].values()
    )
    decisions = read_manifest(child / "manifest.json")["resource_decisions"]
    assert any(d["kind"] == "budget_extension" and d["append_data"] for d in decisions)
    optimization = next(
        d for d in decisions if d["kind"] == "runtime_optimization_transition"
    )
    assert optimization["changes"] == [
        {
            "field": "attention.implementation",
            "previous": "reference",
            "requested": "sdpa",
        },
        {
            "field": "runtime.memory.estimate_policy",
            "previous": "enforce",
            "requested": "observe",
        },
    ]


@pytest.mark.parametrize(
    "damage", ["prefix", "optimizer", "memory_budget", "memory_fraction"]
)
def test_runtime_flag_cannot_bypass_data_or_scientific_rails(
    document_parent, tmp_path, damage
):
    source, _root, parent, _tokenizer = document_parent
    raw = append_config(document_parent, tmp_path).model_dump(mode="json")
    raw["attention"]["implementation"] = "sdpa"
    raw["runtime"]["memory"]["estimate_policy"] = "observe"
    if damage == "prefix":
        path = Path(raw["dataset"]["train_path"])
        path.write_text(path.read_text().replace("alpha", "wrong", 1))
    elif damage == "optimizer":
        raw["optimizer"]["peak"] *= 2
    elif damage == "memory_budget":
        raw["runtime"]["memory"]["budget_bytes"] = 1024**3
    else:
        raw["runtime"]["memory"]["max_device_memory_fraction"] = 0.5
    with pytest.raises(ValueError):
        train(
            RunConfig.model_validate(raw),
            extend_budget=parent,
            append_data=True,
            allow_runtime_optimization=True,
            run_id="refused_optimization",
        )
    assert not (source.logging.root_dir / "refused_optimization").exists()


@pytest.mark.parametrize("document_parent", [8], indirect=True)
def test_try_bracket_append_uses_same_parent_and_target_groups(
    document_parent, tmp_path
):
    import yaml

    from klab.lab_mode import run_try
    from klab.try_config import TryContinuation

    source, parent_root, generation, _ = document_parent
    target = append_config(document_parent, tmp_path)
    baseline = tmp_path / "try-baseline.yaml"
    baseline.write_text(yaml.safe_dump(target.model_dump(mode="json")))
    delta = tmp_path / "runtime-delta.yaml"
    delta.write_text(
        yaml.safe_dump({"set": {"training.matched_stories.padding_multiple": 64}})
    )
    record, _ = run_try(
        delta,
        baseline,
        work_dir=tmp_path / "work",
        repeat_baseline=True,
        continuation=TryContinuation(
            checkpoint=generation, append_data=True, allow_runtime_optimization=True
        ),
        probe_tier=None,
    )
    assert record["status"] == "completed", record.get("failure")
    parent = CheckpointManager(parent_root).load(generation)
    for row in record["arms"].values():
        assert (
            row["training_targets_applied"]
            == target.training.max_tokens - source.training.max_tokens
        )
        run = Path(row["run_dir"])
        snapshot = CheckpointManager(run).load(run / "checkpoints/latest.json")
        assert snapshot.parent_checkpoint_sha256 == parent.checkpoint_sha256
        assert snapshot.tokens_seen == target.training.max_tokens
        assert snapshot.step == target.training.max_steps
        assert {int(v["step"]) for v in snapshot.optimizer["state"].values()} == {
            target.training.max_steps
        }


@pytest.mark.parametrize("document_parent", [8], indirect=True)
def test_try_nonterminal_checkpoint_bracket_preserves_consumed_prefix(
    checkpoint_phase, tmp_path
):
    import yaml

    from klab.lab_mode import run_try
    from klab.try_config import TryContinuation

    source, target, root, parent, prior = checkpoint_phase
    baseline = tmp_path / "bounded-baseline.yaml"
    baseline.write_text(yaml.safe_dump(target.model_dump(mode="json")))
    delta = tmp_path / "bounded-delta.yaml"
    delta.write_text(
        yaml.safe_dump({"set": {"training.matched_stories.padding_multiple": 64}})
    )
    record, _ = run_try(
        delta,
        baseline,
        work_dir=tmp_path / "bracket-work",
        repeat_baseline=True,
        continuation=TryContinuation(
            checkpoint=parent,
            append_data=True,
            append_from_checkpoint=True,
            allow_runtime_optimization=True,
        ),
        probe_tier=None,
    )
    assert record["status"] == "completed", record.get("failure")
    assert prior.step < source.training.max_steps
    for row in record["arms"].values():
        run = Path(row["run_dir"])
        child = CheckpointManager(run).load(run / "checkpoints/latest.json")
        assert child.parent_checkpoint_sha256 == prior.checkpoint_sha256
        assert child.step == prior.step + 1
        assert (
            row["training_targets_applied"]
            == target.training.max_tokens - prior.tokens_seen
        )
        assert all(
            float(v["step"]) == child.step for v in child.optimizer["state"].values()
        )
        decision = next(
            d
            for d in read_manifest(run / "manifest.json")["resource_decisions"]
            if d["kind"] == "budget_extension"
        )
        assert decision["parent_consumed_tokens"] == prior.tokens_seen
        assert decision["unconsumed_source_preserved_at"] == str(
            root / "data/train.npy"
        )


def test_append_extension_with_new_retention_boundary_preserves_full_state(
    document_parent, tmp_path
):
    source, root, generation, _ = document_parent
    target = append_config(document_parent, tmp_path)
    raw = target.model_dump(mode="json")
    raw["checkpoint"].update(
        keep_periodic=False,
        retention={
            "milestones": [],
            "terminal_step": target.training.max_steps,
            "terminal_targets": target.training.max_tokens,
            "max_generations": 4,
            "max_generation_bytes": 10485760,
            "max_retained_bytes": 41943040,
            "max_peak_bytes": 52428800,
            "reserve_free_bytes": 0,
        },
    )
    target = RunConfig.model_validate(raw)
    train(
        target, extend_budget=generation, append_data=True, run_id="retained-extension"
    )
    child_root = source.logging.root_dir / "retained-extension"
    child = CheckpointManager(child_root).load(child_root / "checkpoints/latest.json")
    parent = CheckpointManager(root).load(generation)
    assert child.parent_checkpoint_sha256 == parent.checkpoint_sha256
    assert child.step == target.training.max_steps
    assert child.tokens_seen == target.training.max_tokens
    assert child.config["checkpoint"]["retention"]["terminal_step"] == child.step
    assert {int(v["step"]) for v in child.optimizer["state"].values()} == {child.step}
    assert (
        CheckpointManager(child_root)
        .verify(child_root / "checkpoints/latest.json")
        .valid
    )
