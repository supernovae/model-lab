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
from klab.engram.packs import compile_pack, verify_pack
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


def test_engrampack_lesson_emits_compile_ready_jsonl(tmp_path: Path) -> None:
    lesson = scaffold_lesson("engrampack", tmp_path / "engrampack")
    source = lesson / "records.jsonl"
    records = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    assert {record["id"] for record in records} == {
        "tutorial-atlas-maps-to-beacon",
        "tutorial-beacon-maps-to-amber",
        "tutorial-cipher-maps-to-delta",
    }

    pack = tmp_path / "compiled.enpack"
    manifest = compile_pack(
        source,
        pack,
        name="tutorial-map",
        namespace="tutorial",
        default_license="CC0-1.0",
        source_name="original-tutorial-records",
        created_at="2026-09-23T00:00:00Z",
    )
    assert manifest.record_count == 3
    assert verify_pack(pack).valid


def test_semantic_retrieval_lesson_builds_verified_pack_and_runnable_workspace(
    tmp_path: Path,
) -> None:
    lesson = scaffold_lesson("semantic-retrieval", tmp_path / "semantic-lesson")
    payload = json.loads((lesson / "lesson.json").read_text(encoding="utf-8"))
    pack_info = payload["semantic_pack"]
    assert pack_info["path"] == "semantic-pack"
    assert (lesson / "demo.py").is_file()
    assert "python demo.py" in (lesson / "README.md").read_text(encoding="utf-8")
    assert {
        "records.jsonl",
        "semantic_keys.safetensors",
        "semantic_values.safetensors",
        "semantic.json",
        "demo.py",
    } <= {item["path"] for item in payload["inputs"]}

    from klab.engram.packs import load_pack
    from klab.engram.semantic import SemanticRetriever

    pack = load_pack(lesson / pack_info["path"], expected_pack_id=pack_info["pack_id"])
    assert pack.manifest.semantic is not None
    assert pack.manifest.semantic.key_dim == 6
    assert pack.manifest.semantic.memory_dim == 7
    retriever = SemanticRetriever.from_pack(
        lesson / pack_info["path"], expected_pack_id=pack_info["pack_id"]
    )
    hit = retriever.retrieve(
        torch.tensor([1.0, 0, 0, 0, 0, 0]),
        key_encoder=retriever.key_encoder,
        min_score=0.9,
    )
    assert hit.status == "hit"
    assert hit.hits[0].record_id == "lesson-atlas-maps-to-beacon"
    conflict = retriever.retrieve(
        torch.tensor([0.0, 0, 0, 0, 0, 1]),
        key_encoder=retriever.key_encoder,
        top_k=2,
        min_score=0.9,
    )
    assert conflict.status == "conflict"
    assert tuple(item.record_id for item in conflict.hits) == (
        "lesson-fork-left",
        "lesson-fork-right",
    )


