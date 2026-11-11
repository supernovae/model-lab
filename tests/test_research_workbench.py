from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
import torch
import yaml
from pydantic import ValidationError
from research_fixtures import make_study_resources
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from klab.canonical import canonical_digest
from klab.config.loading import load_config
from klab.config.models import AttentionConfig
from klab.data.tokenizer import SPECIAL_TOKENS, load_tokenizer
from klab.engram.packs import compile_pack, verify_pack
from klab.evaluation.generation import _prompt_byte_addresses
from klab.experiments.reporting import _validate_research
from klab.experiments.study import plan_study
from klab.research.catalog import (
    ResearchEntry,
    list_research,
    load_datasets,
    load_recipe,
    load_research,
)
from klab.research.probe import probe_model
from klab.research.scaffold import scaffold_lesson, scaffold_research


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


def test_current_catalog_and_neutral_factorial_strict_versions(tmp_path: Path) -> None:
    # Personal no-recipe cards await their separate protocol retirement.
    assert {entry.id for entry in list_research()} == {
        "dense-lm-v1",
        "learned-engram-portability-v1",
    }
    root = make_study_resources(tmp_path)
    entry = load_research("neutral-study", root=root)
    recipe = load_recipe(entry, local_root=root)
    assert [
        factor.axis
        for factor in recipe.designs["default"]["smoke"].factorial_designs[0].factors
    ] == ["width", "depth"]
    invalid = entry.model_dump(mode="json")
    invalid["version"] = True
    with pytest.raises(ValidationError, match="version must be integer 1"):
        ResearchEntry.model_validate(invalid)


def test_learned_runner_catalog_rejects_generic_recipe_and_scaffold(
    tmp_path: Path,
) -> None:
    entry = load_research("learned-engram-portability-v1")
    guidance = (
        "learned-engram-portability-v1 uses research portability build "
        "--experiment learned-engram-portability-v1, not research scaffold"
    )
    with pytest.raises(ValueError, match=guidance):
        load_recipe(entry)

    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match=guidance):
        scaffold_research("learned-engram-portability-v1", output)
    assert not output.exists()


def test_catalog_rejects_duplicate_json_and_lists_unknown_ids(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"format":"klab-research-entry","version":1,"version":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_research(duplicate)

    with pytest.raises(ValueError, match="available IDs") as error:
        load_research("not-a-research-entry")
    assert "dense-lm-v1" in str(error.value)


def test_research_scaffold_binds_configs_without_preparing_or_overwriting(
    tmp_path: Path,
) -> None:
    root = make_study_resources(tmp_path)
    output = scaffold_research(
        "neutral-study", tmp_path / "study", scale="smoke", data="offline", root=root
    )
    plan = plan_study(output / "study.yaml")
    metadata = json.loads((output / "research.json").read_text(encoding="utf-8"))

    assert len(plan.expanded) == 8
    assert len(plan.pairs) == 4
    assert {item.config.seed for item in plan.expanded} == {5, 9}
    assert metadata["factorial_designs"][0]["id"] == "width-depth"
    assert metadata["factorial_designs"][0]["version"] == 1
    assert "1 versioned 2×2 factorial designs" in (output / "README.md").read_text(
        encoding="utf-8"
    )
    assert len({item["config_sha256"] for item in metadata["coordinates"]}) == 8
    for item in metadata["inputs"]:
        content = (output / item["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == item["sha256"]
    unsigned = {
        key: value for key, value in metadata.items() if key != "research_sha256"
    }
    assert canonical_digest(unsigned) == metadata["research_sha256"]
    assert not (output / "artifacts").exists()
    assert not (output / "runs").exists()

    marker = output / "keep.txt"
    marker.write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        scaffold_research("neutral-study", output, scale="smoke", root=root)
    assert marker.read_text(encoding="utf-8") == "user data"


def test_canonical_research_scaffold_uses_campaign_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_study_resources(tmp_path)
    work = tmp_path / "klab-work"
    monkeypatch.setenv("KLAB_WORK_DIR", str(work))
    output = scaffold_research(
        "neutral-study",
        tmp_path / "experiments/research/new-campaign",
        scale="smoke",
        data="offline",
        root=root,
    )
    config = load_config(output / "base.yaml")
    assert config.logging.root_dir == work / "experiments/new-campaign/runs"
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "klab-work/experiments/new-campaign" in readme
    assert "checkpoint verify" in readme
    assert "| `$WORK/runs/`, `$WORK/receipt.json` |" in readme


def test_fineweb_edu_scaffold_is_pinned_and_does_not_prepare_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_study_resources(tmp_path)

    def forbid_download(*_: object, **__: object) -> None:
        raise AssertionError("scaffolding must not access remote datasets")

    monkeypatch.setattr("klab.data.datasets.load_dataset", forbid_download)
    profile = load_datasets().datasets["fineweb_edu"]
    output = scaffold_research(
        "neutral-study",
        tmp_path / "fineweb-study",
        scale="micro",
        data="fineweb_edu",
        root=root,
    )
    plan = plan_study(output / "study.yaml")
    bound_research = _validate_research(output / "research.json", output / "study.yaml")
    assert _mapping(bound_research["dataset_profile"])["name"] == "fineweb_edu"
    tokenizer = _mapping(yaml.safe_load((output / "tokenizer.yaml").read_text()))
    tokenizer_dataset = _mapping(tokenizer["dataset"])

    assert profile.dataset_config == "sample-10BT"
    assert profile.revision == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    assert profile.train_max_documents == 20_000
    assert profile.validation_max_documents == 500
    assert profile.train_max_tokens == 2_000_000
    assert profile.validation_max_tokens == 65_536
    assert profile.license == "ODC-BY 1.0; attribution required."
    assert "web-crawled" in " ".join(profile.notes)
    assert len(plan.expanded) == 8
    assert {item.config.seed for item in plan.expanded} == {5, 9}
    assert {item.config.dataset.source for item in plan.expanded} == {"fineweb_edu"}
    assert {
        (
            item.config.dataset.train_max_documents,
            item.config.dataset.validation_max_documents,
            item.config.dataset.train_max_tokens,
            item.config.dataset.validation_max_tokens,
        )
        for item in plan.expanded
    } == {(20_000, 500, 2_000_000, 65_536)}
    assert {item.config.dataset.dataset_config for item in plan.expanded} == {
        "sample-10BT"
    }
    assert {item.config.dataset.revision for item in plan.expanded} == {
        "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    }
    assert tokenizer_dataset["dataset_config"] == "sample-10BT"
    assert tokenizer_dataset["train_max_documents"] == 20_000
    assert tokenizer_dataset["validation_max_documents"] == 500
    assert not (output / "artifacts").exists()
    assert not (output / "runs").exists()


def test_tinystories_scaffold_sets_runnable_tokenizer_prefix(
    tmp_path: Path,
) -> None:
    root = make_study_resources(tmp_path)
    profile = load_datasets().datasets["tinystories"]
    output = scaffold_research(
        "neutral-study",
        tmp_path / "tinystories-study",
        scale="smoke",
        data="tinystories",
        root=root,
    )
    tokenizer = _mapping(yaml.safe_load((output / "tokenizer.yaml").read_text()))
    tokenizer_dataset = _mapping(tokenizer["dataset"])

    assert profile.tokenizer_train_max_tokens == 5_000_000
    assert tokenizer["vocab_size"] == profile.vocab_size
    assert tokenizer_dataset["train_max_tokens"] == 5_000_000
    assert tokenizer_dataset["revision"] == profile.revision
    assert not (output / "artifacts").exists()


def test_research_scaffold_rebases_paths_from_symlinked_destination(
    tmp_path: Path,
) -> None:
    root = make_study_resources(tmp_path)
    alias = tmp_path.parent / f"{tmp_path.name}-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    output = scaffold_research(
        "neutral-study", alias / "study", scale="smoke", data="offline", root=root
    )
    config_path = next((output / "configs").glob("*.yaml"))
    config = load_config(config_path)
    root = output.resolve()

    assert config.tokenizer.path == root / "artifacts/tokenizer/tokenizer.json"
    assert config.dataset.cache_dir == root / "artifacts/data"
    from klab.experiments.study import study_workspace

    assert config.logging.root_dir == study_workspace("neutral-study") / "runs"
    readme = (output / "README.md").read_text()
    assert 'WORK="' in readme
    assert '--store "$WORK/runs"' in readme
    assert '--receipt "$WORK/receipt.json"' in readme
    assert '--runs-dir "$WORK/runs"' in readme
    assert '--output "$WORK/local-reports"' in readme


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


@pytest.mark.parametrize("external", [False, True])
def test_recipe_less_cards_refuse_scaffold_before_writing(
    tmp_path: Path, external: bool
):
    root = make_study_resources(tmp_path) if external else None
    entry_id = "neutral-study" if external else "dense-lm-v1"
    if root is not None:
        entry_path = root / "catalog/neutral-study.json"
        entry = json.loads(entry_path.read_text())
        entry["classification"] = "learning_reference"
        entry["recipe"] = None
        entry_path.write_text(json.dumps(entry), encoding="utf-8")
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="has no scaffold recipe.*--resource-root"):
        scaffold_research(entry_id, output, root=root)
    assert not output.exists()
