from __future__ import annotations

import json
from pathlib import Path

import pytest

from klab.config.models import DatasetConfig, TokenizerTrainConfig
from klab.data import tokenizer as tokenizer_module


def test_data_prepare_rejects_missing_tokenizer_before_packing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from klab.cli import main as cli
    from klab.config.loading import load_config

    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs/runtime_smoke_cpu.yaml")
    config = base.model_copy(
        update={
            "tokenizer": base.tokenizer.model_copy(
                update={"path": tmp_path / "missing.json"}
            )
        }
    )
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(
        "klab.data.packing.prepare_data",
        lambda *_args, **_kwargs: pytest.fail("packing started without a tokenizer"),
    )
    args = cli.build_parser().parse_args(
        ["data", "prepare", str(tmp_path / "unread-config.yaml")]
    )

    with pytest.raises(FileNotFoundError, match="complete tokenizer artifact"):
        args.handler(args)


def test_tokenizer_acquisition_does_not_request_past_document_bound(
    monkeypatch, tmp_path: Path
) -> None:
    requested: list[str] = []

    def documents(_config: DatasetConfig, _split: str):
        for value in ("one", "must not be requested"):
            requested.append(value)
            yield value

    monkeypatch.setattr(tokenizer_module, "iter_documents", documents)
    config = TokenizerTrainConfig(
        schema_version=1,
        vocab_size=260,
        min_frequency=1,
        max_documents=1,
        output_dir=tmp_path / "tokenizer",
        dataset=DatasetConfig(
            source="synthetic",
            cache_dir=tmp_path / "cache",
            train_max_documents=1,
            validation_max_documents=1,
            train_max_tokens=16,
            validation_max_tokens=16,
        ),
    )

    tokenizer_module.train_tokenizer(config)
    assert requested and set(requested) == {"one"}


def test_tokenizer_training_respects_utf8_byte_budget(
    monkeypatch, tmp_path: Path
) -> None:
    requested: list[str] = []

    def documents(_config: DatasetConfig, _split: str):
        for value in ("abc", "defgh", "must not be requested"):
            requested.append(value)
            yield value

    monkeypatch.setattr(tokenizer_module, "iter_documents", documents)
    config = TokenizerTrainConfig(
        schema_version=1,
        vocab_size=260,
        min_frequency=1,
        max_documents=10,
        output_dir=tmp_path / "tokenizer",
        dataset=DatasetConfig(
            source="synthetic",
            cache_dir=tmp_path / "cache",
            train_max_documents=10,
            validation_max_documents=1,
            train_max_tokens=6,
            validation_max_tokens=16,
        ),
    )

    path = tokenizer_module.train_tokenizer(config)
    manifest = json.loads(
        path.with_name("tokenizer_manifest.json").read_text(encoding="utf-8")
    )

    assert set(requested) == {"abc", "defgh"}
    assert manifest["selected_documents"] == 2
    assert manifest["selected_input_bytes_utf8"] == 6
    assert manifest["training_contract"]["input_byte_budget_utf8"] == 6

    verified = tokenizer_module.verify_tokenizer_artifact(
        path,
        source="synthetic",
        revision=None,
        vocab_size=260,
    )
    assert verified["selected_documents"] == 2


def test_tokenizer_artifact_prerequisite_fails_on_digest_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tokenizer_module,
        "iter_documents",
        lambda _config, _split: iter(("tiny corpus",)),
    )
    config = TokenizerTrainConfig(
        schema_version=1,
        vocab_size=260,
        min_frequency=1,
        max_documents=1,
        output_dir=tmp_path / "tokenizer",
        dataset=DatasetConfig(
            source="synthetic",
            cache_dir=tmp_path / "cache",
            train_max_documents=1,
            validation_max_documents=1,
            train_max_tokens=20,
            validation_max_tokens=16,
        ),
    )
    path = tokenizer_module.train_tokenizer(config)
    path.write_text("incomplete tokenizer")
    with pytest.raises(ValueError, match="provenance or digest mismatch"):
        tokenizer_module.verify_tokenizer_artifact(
            path, source="synthetic", revision=None, vocab_size=260
        )


def test_tokenizer_byte_budget_does_not_split_utf8_codepoint(
    monkeypatch, tmp_path: Path
) -> None:
    requested: list[str] = []

    def documents(_config: DatasetConfig, _split: str):
        for value in ("ok", "éx", "must not be requested"):
            requested.append(value)
            yield value

    monkeypatch.setattr(tokenizer_module, "iter_documents", documents)
    config = TokenizerTrainConfig(
        schema_version=1,
        vocab_size=260,
        min_frequency=1,
        max_documents=10,
        output_dir=tmp_path / "tokenizer",
        dataset=DatasetConfig(
            source="synthetic",
            cache_dir=tmp_path / "cache",
            train_max_documents=10,
            validation_max_documents=1,
            train_max_tokens=3,
            validation_max_tokens=16,
        ),
    )

    path = tokenizer_module.train_tokenizer(config)
    manifest = json.loads(
        path.with_name("tokenizer_manifest.json").read_text(encoding="utf-8")
    )

    assert set(requested) == {"ok", "éx"}
    assert manifest["selected_documents"] == 1
    assert manifest["selected_input_bytes_utf8"] == 2


def test_mixture_origin_rejects_changed_declaration_before_verification(tmp_path):
    from pathlib import Path

    from klab.config.loading import load_config
    from klab.config.models import RunConfig
    from klab.data.tokenizer import verify_run_tokenizer
    from klab.training.manifest import sha256_file

    raw = load_config(
        Path(__file__).resolve().parents[1] / "configs/smoke_cpu.yaml"
    ).model_dump(mode="json")
    token = tmp_path / "tokenizer.json"
    token.write_text("fixture")
    manifest = tmp_path / "tokenizer_manifest.json"
    manifest.write_text("{}")
    declaration = tmp_path / "mixture.json"
    declaration.write_text("{}")
    raw["dataset"].update(
        source="local_text",
        license="CC0-1.0",
        revision="fixture",
        train_path=str(tmp_path / "train.jsonl"),
        validation_path=str(tmp_path / "validation.jsonl"),
    )
    raw["tokenizer"] = {
        "path": str(token),
        "origin": {
            "mixture_declaration_path": str(declaration),
            "mixture_declaration_sha256": sha256_file(declaration),
            "sha256": sha256_file(token),
            "manifest_sha256": sha256_file(manifest),
        },
    }
    config = RunConfig.model_validate(raw)
    declaration.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="declaration digest mismatch"):
        verify_run_tokenizer(config)
