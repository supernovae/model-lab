from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from klab.config.loading import load_config
from klab.config.models import AttentionConfig
from klab.data.tokenizer import SPECIAL_TOKENS, load_tokenizer
from klab.evaluation.generation import _prompt_byte_addresses
from klab.research.probe import probe_model
from klab.research.scaffold import scaffold_lesson


def _write_byte_tokenizer(path: Path) -> None:
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token in sorted(ByteLevel.alphabet()):
        vocabulary.setdefault(token, len(vocabulary))
    tokenizer = Tokenizer(BPE(vocab=vocabulary, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _rows(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return cast(list[dict[str, object]], value)


def _shape(value: object) -> list[int]:
    assert isinstance(value, list)
    return cast(list[int], value)


def _weight_rows(value: object) -> list[list[float]]:
    assert isinstance(value, list)
    assert all(isinstance(row, list) for row in value)
    return cast(list[list[float]], value)


def test_initialized_probe_bounds_and_reports_mechanisms(tmp_path: Path) -> None:
    lesson = scaffold_lesson("dense", tmp_path / "lesson", scale="smoke")
    config = load_config(lesson / "model.yaml")
    config = config.model_copy(
        update={"model": config.model.model_copy(update={"vocab_size": 260})}
    )
    tokenizer_path = config.tokenizer.path
    _write_byte_tokenizer(tokenizer_path)
    prompt = "a " * 12
    rng_state = torch.random.get_rng_state().clone()
    dense = probe_model(config, prompt)
    assert dense["execution"] == "cpu-fp32-reference"
    assert dense["logits_shape"] == [1, 24, config.model.vocab_size]
    assert set(_mapping(dense["source_identity"])) == {"algorithm", "sha256"}

    moe_model = config.model.model_copy(
        update={
            "ffn": "moe",
            "num_experts": 4,
            "experts_per_token": 2,
            "router_aux_loss_coefficient": 0.01,
        }
    )
    moe = probe_model(config.model_copy(update={"model": moe_model}), prompt)
    route = _mapping(_rows(moe["routing_samples"])[0])
    experts = _mapping(route["selected_experts"])
    weights = _mapping(route["selected_weights"])
    assert _shape(experts["sample_shape"]) == [8, 2]
    assert _shape(weights["sample_shape"]) == [8, 2]
    assert all(abs(sum(row) - 1.0) < 1e-6 for row in _weight_rows(weights["values"]))

    sparse_config = config.model_copy(
        update={
            "attention": AttentionConfig(
                kind="block_sparse", block_size=4, selected_blocks=2
            )
        }
    )
    sparse_result = _rows(probe_model(sparse_config, prompt)["sparse_samples"])[0]
    sparse = _mapping(sparse_result["selected_blocks"])
    assert _shape(sparse["sample_shape"]) == [8]
    assert sparse["truncated"] is True

    mla = probe_model(
        config.model_copy(
            update={"attention": AttentionConfig(kind="mla", latent_dim=32)}
        ),
        prompt,
    )
    kv_widths = []
    for layer in _rows(mla["layers"]):
        name = layer["name"]
        if isinstance(name, str) and name.endswith("kv_down"):
            kv_widths.append(_shape(layer["output_shape"])[-1])
    assert kv_widths == [32, 32]

    byte_lesson = scaffold_lesson(
        "byte-engram", tmp_path / "byte-lesson", scale="smoke"
    )
    byte_config = load_config(byte_lesson / "model.yaml")
    assert byte_config.model.memory_table_size == 263
    _write_byte_tokenizer(byte_config.tokenizer.path)
    byte_config = byte_config.model_copy(
        update={"model": byte_config.model.model_copy(update={"vocab_size": 260})}
    )
    unicode_prompt = "café"
    byte_result = probe_model(byte_config, unicode_prompt)
    tokenizer = load_tokenizer(byte_config.tokenizer.path)
    expected = _prompt_byte_addresses(
        tokenizer,
        unicode_prompt,
        tokenizer.encode(unicode_prompt, add_special_tokens=False).ids,
        byte_config.model.memory_table_size,
        byte_config.model.memory_ngram_size,
    )
    assert len(set(expected)) > 1
    addresses = _mapping(
        _mapping(byte_result["memory_lookup_addresses"])["memory.table"]
    )
    assert addresses["values"] == expected[:8]
    input_addresses = _mapping(
        _mapping(byte_result["memory_lookup_addresses"])["input.byte_addresses"]
    )
    assert input_addresses["values"] == [expected[:8]]
    assert torch.equal(rng_state, torch.random.get_rng_state())
    json.dumps(byte_result)

    with pytest.raises(ValueError, match="128-token bound"):
        probe_model(config, "a" * 129)
    assert torch.equal(rng_state, torch.random.get_rng_state())


def _scaffolded_lessons() -> list[str]:
    from klab.research.catalog import list_lessons

    return [lesson.id for lesson in list_lessons()]


@pytest.mark.parametrize("identifier", _scaffolded_lessons())
def test_scaffolded_readme_shows_only_existing_commands(
    identifier: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_docs import command_problems

    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    lesson = scaffold_lesson(identifier, tmp_path / identifier)
    readme = (lesson / "README.md").read_text(encoding="utf-8")
    assert command_problems(identifier, readme) == []


def test_every_lesson_cites_existing_source() -> None:
    from klab.research.catalog import list_lessons

    root = Path(__file__).resolve().parents[1]
    missing = [
        (lesson.id, step.source_path)
        for lesson in list_lessons()
        for step in lesson.steps
        if not (root / step.source_path).is_file()
    ]
    assert missing == []
