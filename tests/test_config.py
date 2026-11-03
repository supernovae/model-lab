from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sparselab.config.loading import load_config
from sparselab.config.migrate import migrate_file
from sparselab.config.models import ModelConfig, RunConfig
from sparselab.derivation import derive_config


def _v1() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "migration-test",
        "seed": 7,
        "device": "cpu",
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
        "training": {"batch_size": 1, "seq_len": 8, "max_steps": 20, "max_tokens": 80},
        "logging": {"root_dir": "runs", "checkpoint_every_steps": 10},
    }


def test_v1_loading_gives_migration_guidance(tmp_path: Path) -> None:
    source = tmp_path / "legacy.yaml"
    source.write_text(yaml.safe_dump(_v1()), encoding="utf-8")
    with pytest.raises(ValueError, match="sparselab config migrate"):
        load_config(source)


def test_migration_rebases_relative_paths_for_moved_output(tmp_path: Path) -> None:
    source = tmp_path / "input" / "legacy.yaml"
    source.parent.mkdir()
    source.write_text(yaml.safe_dump(_v1(), sort_keys=False), encoding="utf-8")
    output = tmp_path / "published" / "nested" / "config.yaml"

    migrate_file(source, output)
    config = load_config(output)

    assert config.tokenizer.path == source.parent / "assets/tokenizer.json"
    assert config.dataset.cache_dir == source.parent / "cache"
    assert config.logging.root_dir == source.parent / "runs"


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("kind", ["allocation", "snapshot", "corpus", "portability"])
def test_migration_preserves_manifest_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, absolute: bool, kind: str
) -> None:
    source = tmp_path / "source" / "nested" / "legacy.yaml"
    source.parent.mkdir(parents=True)
    raw = _v1()
    dataset = raw["dataset"]
    section = dataset
    if kind == "portability":
        section = raw["training"]
        section["trainable_parameters"] = ["lm_head.weight"]
        fields = ["portability_manifest_path"]
    else:
        dataset.update(
            source="snapshot" if kind == "snapshot" else "local_chat",
            train_path="train.jsonl",
            validation_path="validation.jsonl",
            license="CC0-1.0",
            revision="a" * 64,
        )
        fields = {
            "allocation": ["allocation_manifest_path"],
            "snapshot": ["source_manifest_path"],
            "corpus": ["corpus_release_path", "corpus_export_path"],
        }[kind]
    targets = {}
    for field in fields:
        target = source.parent / f"{field}.json"
        target.write_text(f"{field} fixture\n", encoding="utf-8")
        targets[field] = target
        section[field] = str(target) if absolute else target.name
    original = yaml.safe_dump(raw, sort_keys=False)
    source.write_text(original, encoding="utf-8")
    output = tmp_path / "destination" / "nested" / "migrated.yaml"
    cwd = tmp_path / "unrelated"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    migrate_file(source, output)
    published = yaml.safe_load(output.read_text(encoding="utf-8"))
    for reload_cwd in (cwd, source.parent):
        monkeypatch.chdir(reload_cwd)
        loaded = load_config(output)
        loaded_section = loaded.training if kind == "portability" else loaded.dataset
        for field, target in targets.items():
            assert getattr(loaded_section, field) == target
            assert getattr(loaded_section, field).read_bytes() == target.read_bytes()
            value = published["training" if kind == "portability" else "dataset"][field]
            assert isinstance(value, str)
            assert (output.parent / value).resolve() == target
    assert source.read_text(encoding="utf-8") == original
    assert published["model"] == raw["model"]
    assert published["seed"] == raw["seed"]
    assert published["dataset"]["train_max_tokens"] == dataset["train_max_tokens"]


def test_invalid_migration_does_not_publish_output(tmp_path: Path) -> None:
    source = tmp_path / "invalid.yaml"
    invalid = _v1()
    invalid["training"] = {
        "batch_size": 1,
        "seq_len": 32,
        "max_steps": 20,
        "max_tokens": 80,
    }
    source.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    output = tmp_path / "output.yaml"

    with pytest.raises(ValueError, match="migrated config is invalid"):
        migrate_file(source, output)
    assert not output.exists()


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("publication", ["load", "derive", "migrate"])
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
    raw = _v1()
    raw["dataset"].update(
        source="local_token_mixture",
        license="local synthetic fixture",
        train_path="assets/mixture/train.tokens.jsonl",
        validation_path="assets/validation.jsonl",
        mixture_declaration_path=str(declaration)
        if absolute
        else "assets/mixture.yaml",
        mixture_output_path=str(mixture_output) if absolute else "assets/mixture",
    )
    if publication != "migrate":
        from sparselab.config.migrate import migrate_v1

        raw = migrate_v1(raw)
    source.write_text(yaml.safe_dump(raw), encoding="utf-8")
    output = tmp_path / "published" / "elsewhere" / "run.yaml"
    output.parent.mkdir(parents=True)
    if publication == "derive":
        derive_config(source, output, {"name": "derived-mixture"})
    elif publication == "migrate":
        migrate_file(source, output)
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


def test_memory_injection_serialization_preserves_legacy_final_config() -> None:
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


def test_adamw_decay_horizon_preserves_legacy_serialization() -> None:
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


def test_portability_config_is_opt_in_and_requires_exact_controls() -> None:
    base = load_config(Path("configs/runtime_smoke_cpu.yaml"))
    legacy = base.model_dump(mode="json")
    assert "trainable_parameters" not in legacy["training"]
    assert "portability_manifest_path" not in legacy["training"]
    assert RunConfig.model_validate(legacy).model_dump(mode="json") == legacy

    portability = {
        **legacy,
        "training": {
            **legacy["training"],
            "trainable_parameters": ["blocks.0.attention.q_proj.weight"],
            "portability_manifest_path": "run.json",
        },
    }
    configured = RunConfig.model_validate(portability)
    assert configured.training.trainable_parameters == (
        "blocks.0.attention.q_proj.weight",
    )
    assert configured.training.portability_manifest_path == Path("run.json")

    duplicate = {
        **portability,
        "training": {
            **portability["training"],
            "trainable_parameters": ["embedding.weight", "embedding.weight"],
        },
    }
    with pytest.raises(ValueError, match="unique canonical names"):
        RunConfig.model_validate(duplicate)

    missing_controls = {
        **legacy,
        "training": {
            **legacy["training"],
            "portability_manifest_path": "run.json",
        },
    }
    with pytest.raises(ValueError, match="explicit trainable_parameters"):
        RunConfig.model_validate(missing_controls)

    mlx = {
        **portability,
        "runtime": {"engine": "mlx", "backend": "metal"},
    }
    with pytest.raises(ValueError, match="require the PyTorch engine"):
        RunConfig.model_validate(mlx)


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
    from sparselab.training.manifest import config_sha256

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
