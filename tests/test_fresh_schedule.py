"""Fresh schedules use actual EOS group boundaries, never nominal batch division."""

import json

import numpy as np
import pytest
import yaml
from lab_fixtures import write_smoke_inputs

from klab.config import RunConfig, load_config
from klab.config.models import EvaluationConfig, FixedValidationPanel
from klab.data.story_packing import MatchedStoryDataset
from klab.data.tokenizer import load_tokenizer
from klab.staging import materialize_prepared_inputs, verify_prepared_inputs
from klab.training.schedule import (
    derive_schedule,
    register_parser,
    schedule_settings,
)


def test_schedule_uses_exact_groups_and_preserves_warmup(tmp_path):
    source = write_smoke_inputs(tmp_path, max_steps=20, train_tokenizer_now=False)
    raw = load_config(source).model_dump(mode="json")
    raw["training"]["max_tokens"] = 24
    raw["checkpoint"].update(
        keep_periodic=False,
        retention={
            "milestones": [{"name": "first", "step": 1, "committed_targets": 1}],
            "terminal_step": 20,
            "terminal_targets": 24,
            "max_generations": 5,
            "max_generation_bytes": 10,
            "max_retained_bytes": 50,
            "max_peak_bytes": 60,
            "reserve_free_bytes": 0,
        },
    )
    config = RunConfig.model_validate(raw)
    # Four six-token documents form two 12-target groups, not ceil(24/10)=3.
    ids = np.asarray([1, 1, 1, 1, 1, 0] * 4, dtype=np.int32)
    plan = MatchedStoryDataset(ids, 8, 0, layout="story", targets_per_update=10)
    settings = schedule_settings(config, plan)
    assert settings["training.max_steps"] == settings["optimizer.decay_steps"] == 2
    assert settings["checkpoint.retention"]["terminal_targets"] == 24
    assert settings["checkpoint.retention"]["milestones"][0]["committed_targets"] == 12
    assert "optimizer.warmup_steps" not in settings
    raw["checkpoint"]["retention"]["milestones"][0]["step"] = 3
    with pytest.raises(ValueError, match="milestone exceeds"):
        schedule_settings(RunConfig.model_validate(raw), plan)


def test_schedule_cold_bundle_derivation_and_tamper_refusal(tmp_path):
    source = write_smoke_inputs(tmp_path, max_steps=20)
    raw = load_config(source).model_dump(mode="json")
    tokenizer = load_tokenizer(load_config(source).tokenizer.path)
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    texts = ["alpha beta.", "gamma delta."] * 3
    train.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))
    valid.write_text(json.dumps({"text": "A separate validation account."}) + "\n")
    tokens = sum(
        len(tokenizer.encode(text, add_special_tokens=False).ids) + 1 for text in texts
    )
    raw["dataset"].update(
        source="local_text",
        revision="schedule-fixture",
        license="CC0-1.0",
        train_path=str(train),
        validation_path=str(valid),
        train_max_documents=len(texts),
        validation_max_documents=1,
        train_max_tokens=tokens,
        validation_max_tokens=128,
    )
    raw["training"].update(
        micro_batch_size=1,
        gradient_accumulation=1,
        max_tokens=tokens,
        matched_stories={
            "layout": "story",
            "targets_per_update": 1,
            "unpadded_documents": True,
        },
    )
    raw["optimizer"]["warmup_steps"] = 0
    source.write_text(yaml.safe_dump(raw))
    config = load_config(source)
    bundle = tmp_path / "prepared"
    materialize_prepared_inputs(config, bundle)
    output = tmp_path / "scheduled.yaml"
    result = derive_schedule(source, bundle, output)
    derived = load_config(output)
    assert derived.training.max_steps == len(texts)
    assert derived.optimizer.decay_steps == len(texts)
    assert derived.training.max_tokens == tokens
    assert derived.dataset == config.dataset
    assert result["groups_per_pass"] == len(texts)
    assert result["committed_targets"] == tokens
    assert result["final_group_clipped_targets"] == 0
    assert result["final_group_targets"] == result["final_group_complete_targets"]
    original_bytes = source.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        derive_schedule(source, bundle, source)
    assert source.read_bytes() == original_bytes
    assert (tmp_path / "scheduled.yaml.derivation.json").exists()
    final_bundle = tmp_path / "final-prepared"
    materialize_prepared_inputs(derived, final_bundle)
    verify_prepared_inputs(final_bundle, derived)
    # The original bundle remains bound to the original config, never rewritten.
    verify_prepared_inputs(bundle, config)
    with pytest.raises(ValueError, match="configuration differs"):
        verify_prepared_inputs(bundle, derived)
    data = bundle / "assets/data/train.npy"
    ids = np.load(data, allow_pickle=False)
    ids[0] += 1
    data.chmod(0o600)
    np.save(data, ids)
    with pytest.raises((ValueError, OSError)):
        derive_schedule(source, bundle, tmp_path / "refused.yaml")
    assert not (tmp_path / "refused.yaml").exists()


@pytest.mark.parametrize("count", [0, 2])
def test_arbitrary_named_panel_counts(count):
    panels = [
        FixedValidationPanel(
            name=f"pillar_{index}",
            prepared_path="fixture",
            prepared_manifest_sha256="0" * 64,
            tokenizer_file_sha256="1" * 64,
            tokenizer_canonical_sha256="2" * 64,
            split="validation",
            sampling="first_eligible_blocks",
            sequence_length=8,
            blocks=3,
            supervised_targets=24,
            batch_size=2,
            max_forward_positions=24,
        )
        for index in range(count)
    ]
    config = EvaluationConfig(panels=tuple(panels))
    assert config.batch_limit == (2 * count if count else config.max_batches)
    if panels:
        with pytest.raises(ValueError, match="distinct named"):
            EvaluationConfig(panels=(*panels, panels[0]))


def test_schedule_cli_dispatch(tmp_path, capsys):
    import argparse

    parser = argparse.ArgumentParser()
    register_parser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(
        [
            "schedule",
            "packing.yaml",
            "--prepared-inputs",
            "prepared",
            "--output",
            "final.yaml",
        ]
    )
    assert args.config.name == "packing.yaml"
    assert args.prepared_inputs.name == "prepared"
    assert args.output.name == "final.yaml"
    assert callable(args.handler)


@pytest.mark.parametrize("budget, expected", [(12, 1), (18, 2), (24, 2)])
def test_schedule_exact_partial_pass(tmp_path, budget, expected):
    plan = MatchedStoryDataset(
        np.asarray([1, 1, 1, 1, 1, 0] * 4, dtype=np.int32),
        8,
        0,
        layout="story",
        targets_per_update=10,
        unpadded_documents=True,
    )
    source = write_smoke_inputs(tmp_path, max_steps=20, train_tokenizer_now=False)
    config = load_config(source)
    raw = config.model_dump(mode="json")
    raw["training"]["max_tokens"] = budget
    raw["optimizer"]["warmup_steps"] = 0
    assert (
        schedule_settings(RunConfig.model_validate(raw), plan)["training.max_steps"]
        == expected
    )
    with pytest.raises(ValueError, match="document-aligned"):
        plan.horizon(7)


def test_fresh_schedule_refuses_repeated_exposure_and_checkpoint_transitions(tmp_path):
    source = write_smoke_inputs(tmp_path, max_steps=20, train_tokenizer_now=False)
    raw = load_config(source).model_dump(mode="json")
    raw["training"]["max_tokens"] = 48
    plan = MatchedStoryDataset(
        np.asarray([1, 1, 1, 1, 1, 0] * 4, dtype=np.int32),
        8,
        0,
        layout="story",
        targets_per_update=10,
    )
    with pytest.raises(ValueError, match="exceeds prepared supply"):
        schedule_settings(RunConfig.model_validate(raw), plan)
    raw["training"]["max_tokens"] = 24
    plan.target_group_transition = {"after_tokens": 12}
    with pytest.raises(ValueError, match="checkpoint-bound"):
        schedule_settings(RunConfig.model_validate(raw), plan)


def test_schedule_does_not_shorten_warmup_to_fit(tmp_path):
    from klab.derivation import derive_config

    source = write_smoke_inputs(tmp_path, max_steps=20, train_tokenizer_now=False)
    raw = load_config(source).model_dump(mode="json")
    raw["training"]["max_tokens"] = 24
    raw["optimizer"]["warmup_steps"] = 2
    source.write_text(yaml.safe_dump(raw))
    plan = MatchedStoryDataset(
        np.asarray([1, 1, 1, 1, 1, 0] * 4, dtype=np.int32),
        8,
        0,
        layout="story",
        targets_per_update=10,
    )
    settings = schedule_settings(load_config(source), plan)
    output = tmp_path / "refused.yaml"
    with pytest.raises(ValueError, match="warmup_steps must be smaller"):
        derive_config(source, output, settings)
    assert not output.exists()
