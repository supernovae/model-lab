from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from klab.config.loading import load_config
from klab.config.models import ModelConfig, RunConfig
from klab.derivation import derive_config


def _run() -> dict[str, object]:
    return {
        "schema_version": 2,
        "name": "path-test",
        "seed": 7,
        "model": {
            "vocab_size": 512,
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "ffn_dim": 32,
            "max_seq_len": 16,
        },
        "tokenizer": {"path": "assets/tokenizer.json"},
        "dataset": {
            "source": "synthetic",
            "cache_dir": "cache",
            "train_max_documents": 2,
            "validation_max_documents": 2,
            "train_max_tokens": 64,
            "validation_max_tokens": 64,
        },
        "training": {
            "micro_batch_size": 1,
            "seq_len": 8,
            "max_steps": 20,
            "max_tokens": 80,
        },
        "optimizer": {"name": "adamw", "peak": 0.003, "floor": 0.0003},
        "logging": {"root_dir": "runs"},
        "checkpoint": {"every_steps": 10},
    }


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("publication", ["load", "derive"])
def test_mixture_paths_preserve_source_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    absolute: bool,
    publication: str,
) -> None:
    source = tmp_path / "source" / "nested" / "run.yaml"
    source.parent.mkdir(parents=True)
    declaration = source.parent / "assets" / "mixture.yaml"
    declaration.parent.mkdir()
    declaration.write_text("synthetic declaration\n", encoding="utf-8")
    mixture_output = source.parent / "assets" / "mixture"
    mixture_output.mkdir()
    (mixture_output / "receipt.json").write_text("{}\n", encoding="utf-8")
    raw = _run()
    raw["dataset"].update(
        source="local_token_mixture",
        license="local synthetic fixture",
        train_path="assets/mixture/train.tokens.ids",
        validation_path="assets/validation.jsonl",
        mixture_declaration_path=str(declaration)
        if absolute
        else "assets/mixture.yaml",
        mixture_output_path=str(mixture_output) if absolute else "assets/mixture",
    )
    source.write_text(yaml.safe_dump(raw), encoding="utf-8")
    output = tmp_path / "published" / "elsewhere" / "run.yaml"
    output.parent.mkdir(parents=True)
    if publication == "derive":
        derive_config(source, output, {"name": "derived-mixture"})
    else:
        output = source

    for cwd in (tmp_path / "cwd-a", tmp_path / "unrelated" / "cwd-b"):
        cwd.mkdir(parents=True)
        monkeypatch.chdir(cwd)
        loaded = load_config(output)
        assert loaded.dataset.mixture_declaration_path == declaration
        assert loaded.dataset.mixture_output_path == mixture_output
        assert (
            loaded.dataset.mixture_declaration_path.read_text()
            == "synthetic declaration\n"
        )
        assert (
            loaded.dataset.mixture_output_path / "receipt.json"
        ).read_text() == "{}\n"


def test_memory_injection_default_is_omitted_from_serialized_configs() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "runtime_smoke_cpu.yaml"
    )
    explicit_final = config.model_copy(
        update={"model": config.model.model_copy(update={"memory_injection": "final"})}
    )

    for current in (config, explicit_final):
        assert "memory_injection" not in current.model_dump(mode="python")["model"]
        assert "memory_injection" not in current.model_dump(mode="json")["model"]
        assert "memory_hash_heads" in current.model_dump(mode="json")["model"]
        assert "memory_injection" not in current.model_dump_json()
        assert (
            RunConfig.model_validate(
                current.model_dump(mode="json")
            ).model.memory_injection
            == "final"
        )

    assert config.model_dump(mode="python") == explicit_final.model_dump(mode="python")
    assert config.model_dump(mode="json") == explicit_final.model_dump(mode="json")
    assert config.model_dump_json() == explicit_final.model_dump_json()

    enabled = {
        **config.model.model_dump(mode="python"),
        "memory": "ngram",
        "memory_table_size": 31,
        "memory_ngram_size": 3,
        "memory_dim": 8,
        "memory_injection": "embedding",
    }
    embedded = config.model_copy(update={"model": ModelConfig(**enabled)})
    assert (
        embedded.model_dump(mode="python")["model"]["memory_injection"] == "embedding"
    )
    assert embedded.model_dump(mode="json")["model"]["memory_injection"] == "embedding"
    assert (
        RunConfig.model_validate_json(embedded.model_dump_json()).model.memory_injection
        == "embedding"
    )


@pytest.mark.parametrize("value", [None, "middle", 2, [1]])
def test_memory_injection_rejects_unsupported_values(value: object) -> None:
    settings = {
        "vocab_size": 512,
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 2,
        "ffn_dim": 32,
        "max_seq_len": 16,
        "memory": "ngram",
        "memory_table_size": 31,
        "memory_ngram_size": 3,
        "memory_dim": 8,
        "memory_injection": value,
    }
    with pytest.raises(ValueError):
        ModelConfig(**settings)


def test_embedding_memory_injection_requires_enabled_memory() -> None:
    with pytest.raises(ValueError, match="requires enabled memory"):
        ModelConfig(
            vocab_size=512,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=16,
            memory_injection="embedding",
        )


def test_grouped_query_attention_rejects_mlx_run_configs() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "runtime_smoke_cpu.yaml"
    )
    values = config.model_dump(mode="json")
    values["model"]["num_kv_heads"] = 1
    values["runtime"]["engine"] = "mlx"
    values["runtime"]["backend"] = "metal"

    with pytest.raises(ValueError, match="grouped-query attention"):
        RunConfig.model_validate(values)


def test_adamw_decay_horizon_is_omitted_until_set() -> None:
    base = load_config(Path("configs/smoke_cpu.yaml"))
    serialized = base.model_dump(mode="json")
    for dump in (base.model_dump(mode="python"), serialized):
        assert "decay_steps" not in dump["optimizer"]
    assert "decay_steps" not in base.model_dump_json()
    assert RunConfig.model_validate(serialized).model_dump(mode="json") == serialized

    serialized["optimizer"]["decay_steps"] = 8
    extended = RunConfig.model_validate(serialized)
    assert extended.optimizer.decay_steps == 8
    assert extended.model_dump(mode="json")["optimizer"]["decay_steps"] == 8


@pytest.mark.parametrize("decay_steps", [0, 4, 41])
def test_adamw_decay_horizon_requires_valid_bounds(decay_steps: int) -> None:
    values = load_config(Path("configs/smoke_cpu.yaml")).model_dump(mode="json")
    values["optimizer"]["decay_steps"] = decay_steps
    with pytest.raises(ValueError, match="decay_steps"):
        RunConfig.model_validate(values)


def test_adamw_decay_horizon_rejects_mlx() -> None:
    values = load_config(Path("configs/smoke_cpu.yaml")).model_dump(mode="json")
    values["runtime"].update(engine="mlx", backend="metal")
    values["optimizer"]["decay_steps"] = 8
    with pytest.raises(ValueError, match="decay_steps is unsupported for MLX"):
        RunConfig.model_validate(values)


def test_trainable_parameters_are_opt_in_and_unique() -> None:
    base = load_config(Path("configs/runtime_smoke_cpu.yaml"))
    plain = base.model_dump(mode="json")
    assert "trainable_parameters" not in plain["training"]
    assert RunConfig.model_validate(plain).model_dump(mode="json") == plain

    selected = {
        **plain,
        "training": {
            **plain["training"],
            "trainable_parameters": ["blocks.0.attention.q_proj.weight"],
        },
    }
    configured = RunConfig.model_validate(selected)
    assert configured.training.trainable_parameters == (
        "blocks.0.attention.q_proj.weight",
    )

    duplicate = {
        **selected,
        "training": {
            **selected["training"],
            "trainable_parameters": ["embedding.weight", "embedding.weight"],
        },
    }
    with pytest.raises(ValueError, match="unique canonical names"):
        RunConfig.model_validate(duplicate)


@pytest.fixture
def origin_bound_config():
    base = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    )
    raw = base.model_dump(mode="json")
    snapshot = {
        **raw["dataset"],
        "source": "snapshot",
        "revision": "a" * 40,
        "license": "MIT",
        "dataset_config": "default",
        "source_manifest_path": "origin/manifest.json",
        "train_path": "origin/train.jsonl",
        "validation_path": "origin/validation.jsonl",
    }
    raw["dataset"] = {**snapshot, "cache_dir": "consumer/cache"}
    raw["tokenizer"]["origin"] = {
        "dataset": snapshot,
        "sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
    }
    return raw


def test_origin_binding_resolves_paths_and_preserves_default_identity(
    tmp_path, origin_bound_config
):
    from klab.training.manifest import config_sha256

    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(origin_bound_config))
    config = load_config(path)
    assert config.tokenizer.origin.dataset.train_path == tmp_path / "origin/train.jsonl"
    raw = config.model_dump(mode="json")
    before = config_sha256(raw)
    raw["tokenizer"]["origin"]["dataset"]["source_manifest_path"] = (
        "/moved/manifest.json"
    )
    assert config_sha256(raw) == before
    raw["tokenizer"]["origin"]["manifest_sha256"] = "d" * 64
    assert config_sha256(raw) != before
    del raw["tokenizer"]["origin"]
    assert RunConfig.model_validate(raw).model_dump(mode="json") == raw


@pytest.mark.parametrize(
    "change", ["missing_digest", "bad_digest", "unfrozen_origin", "unfrozen_consumer"]
)
def test_origin_binding_requires_pins_and_frozen_sources(origin_bound_config, change):
    raw = origin_bound_config
    if change == "missing_digest":
        del raw["tokenizer"]["origin"]["manifest_sha256"]
    elif change == "bad_digest":
        raw["tokenizer"]["origin"]["sha256"] = "bad"
    elif change == "unfrozen_origin":
        raw["tokenizer"]["origin"]["dataset"]["source"] = "synthetic"
    else:
        raw["dataset"]["source"] = "synthetic"
    with pytest.raises(ValueError):
        RunConfig.model_validate(raw)


@pytest.mark.parametrize(
    "change",
    [
        {"memory_injection": "after_block"},
        {"memory_injection": "after_block", "memory_block_index": 2},
        {"memory_injection": "after_block", "memory_block_index": True},
        {"memory_block_index": 0},
        {"memory_conv_kernel_size": 4},
    ],
)
def test_context_memory_rejects_invalid_placement_and_refinement(change):
    base = {
        "vocab_size": 260,
        "hidden_dim": 8,
        "num_layers": 2,
        "num_heads": 2,
        "ffn_dim": 16,
        "max_seq_len": 16,
        "memory": "ngram",
        "memory_table_size": 31,
        "memory_ngram_size": 3,
        "memory_dim": 4,
    }
    with pytest.raises(ValueError):
        ModelConfig(**(base | change))


def test_context_memory_roundtrip_preserves_distinct_architecture():
    base = ModelConfig(
        vocab_size=260,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        ffn_dim=16,
        max_seq_len=16,
        memory="ngram",
        memory_table_size=31,
        memory_ngram_size=3,
        memory_dim=4,
    )
    assert "memory_variant" not in base.model_dump()
    assert "memory_block_index" not in base.model_dump()
    assert "memory_conv_kernel_size" not in base.model_dump()
    contextual = ModelConfig.model_validate(
        base.model_dump()
        | {
            "memory_variant": "context_key",
            "memory_injection": "after_block",
            "memory_block_index": 0,
            "memory_conv_kernel_size": 4,
        }
    )
    assert ModelConfig.model_validate_json(contextual.model_dump_json()) == contextual
    assert contextual.model_dump() != base.model_dump()


def test_mixture_tokenizer_origin_requires_one_pinned_declaration(
    tmp_path, origin_bound_config
):
    from klab.training.manifest import config_sha256

    raw = origin_bound_config
    raw["tokenizer"]["origin"] = {
        "mixture_declaration_path": "origin/mixture.json",
        "mixture_declaration_sha256": "a" * 64,
        "sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert (
        config.tokenizer.origin.mixture_declaration_path
        == tmp_path / "origin/mixture.json"
    )
    before = config_sha256(raw)
    raw["tokenizer"]["origin"]["mixture_declaration_path"] = "/moved/mixture.json"
    assert config_sha256(raw) == before
    raw["tokenizer"]["origin"]["mixture_declaration_sha256"] = "d" * 64
    assert config_sha256(raw) != before
    del raw["tokenizer"]["origin"]["mixture_declaration_sha256"]
    with pytest.raises(ValueError, match="requires its digest"):
        RunConfig.model_validate(raw)
