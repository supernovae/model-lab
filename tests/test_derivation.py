from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from klab.cli.main import main
from klab.config.loading import load_config
from klab.derivation import derive_config, parse_assignments
from klab.training.manifest import sha256_file


@pytest.fixture
def source_config(tmp_path: Path) -> Path:
    """A relocatable v2 config whose path values exercise source anchoring."""
    source = tmp_path / "source" / "run.yaml"
    source.parent.mkdir()
    raw = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1] / "configs" / "runtime_smoke_cpu.yaml"
        ).read_text(encoding="utf-8")
    )
    assert isinstance(raw, dict)
    raw["tokenizer"] = {"path": "assets/tokenizer.json"}
    raw["dataset"]["cache_dir"] = "cache"
    raw["logging"]["root_dir"] = "runs"
    raw["training"].update(
        trainable_parameters=["blocks.0.attention.q_proj.weight"],
        portability_manifest_path="manifests/portable.json",
    )
    source.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return source


def _sidecar(output: Path) -> Path:
    return output.with_name(output.name + ".derivation.json")


def _run_config_derive(monkeypatch, capsys, argv: list[str]) -> dict[str, object]:
    monkeypatch.setattr(sys, "argv", ["klab", "config", "derive", *argv])
    main()
    return json.loads(capsys.readouterr().out)


def test_config_derivation_relocates_paths_and_records_exact_bytes(
    source_config: Path, tmp_path: Path
) -> None:
    output = tmp_path / "published" / "derived.yaml"
    output.parent.mkdir()

    receipt = derive_config(
        source_config,
        output,
        {
            "optimizer.peak": 0.001,
            "optimizer.decay_steps": 12,
            "tokenizer": {"path": "replacement/tokenizer.json"},
        },
    )

    loaded = load_config(output)
    assert loaded.optimizer.peak == 0.001
    assert loaded.optimizer.decay_steps == 12
    assert (
        loaded.tokenizer.path
        == (source_config.parent / "replacement/tokenizer.json").resolve()
    )
    assert loaded.dataset.cache_dir == (source_config.parent / "cache").resolve()
    assert loaded.logging.root_dir == (source_config.parent / "runs").resolve()
    assert (
        loaded.training.portability_manifest_path
        == (source_config.parent / "manifests/portable.json").resolve()
    )
    assert receipt["format"] == "klab-derivation-v1"
    assert receipt["kind"] == "run_config"
    assert receipt["origin"]["path"] == str(source_config.resolve())
    assert receipt["origin"]["file_sha256"] == sha256_file(source_config)
    assert (
        receipt["output"]["file_sha256"]
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    assert receipt["declared_delta"]["optimizer.decay_steps"] == 12
    assert receipt["validation"] == {
        "scope": "typed_config",
        "executable": False,
        "runtime_verified": False,
    }
    assert _sidecar(output).exists()
    assert json.loads(_sidecar(output).read_text(encoding="utf-8")) == receipt


def test_config_derivation_noop_is_not_an_observed_scientific_delta(
    source_config: Path, tmp_path: Path
) -> None:
    output = tmp_path / "derived.yaml"
    receipt = derive_config(source_config, output, {"optimizer.peak": 0.003})

    assert "optimizer.peak" not in receipt["delta"]
    assert receipt["declared_delta"] == {"optimizer.peak": 0.003}


@pytest.mark.parametrize(
    "values",
    [
        [],
        ["optimizer.peak"],
        ["=1"],
        ["optimizer.peak=not-json"],
        ["optimizer.peak=NaN"],
        ["optimizer.peak=1e999"],
        ["optimizer.peak=1", "optimizer.peak=2"],
        ["optimizer.peak=Infinity"],
        ['tokenizer={"path":"a","path":"b"}'],
    ],
)
def test_assignment_parser_rejects_ambiguous_or_nonfinite_values(
    values: list[str],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_assignments(values)


def test_config_derivation_rejects_invalid_schema_without_publication(
    source_config: Path, tmp_path: Path
) -> None:
    output = tmp_path / "invalid.yaml"
    with pytest.raises(ValueError, match="seq_len"):
        derive_config(source_config, output, {"training.seq_len": 999999})
    assert not output.exists()
    assert not _sidecar(output).exists()


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("- not-a-mapping\n", "mapping"),
        ("schema_version: 1\n", "migrated"),
        (
            "schema_version: 2\nname: first\nname: second\n",
            "duplicate",
        ),
    ],
)
def test_config_derivation_rejects_invalid_source_documents(
    tmp_path: Path, content: str, match: str
) -> None:
    source = tmp_path / "invalid.yaml"
    output = tmp_path / "derived.yaml"
    source.write_text(content, encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=match):
        derive_config(source, output, {"optimizer.peak": 0.001})
    assert not output.exists()
    assert not _sidecar(output).exists()


def test_config_derivation_rejects_linked_destination_components(
    source_config: Path, tmp_path: Path
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(target, target_is_directory=True)
    output = linked_parent / "derived.yaml"

    with pytest.raises((OSError, ValueError), match="symlink"):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert not (target / "derived.yaml").exists()


@pytest.mark.parametrize(
    "settings",
    [
        {"unknown": 1},
        {"optimizer.unknown": 1},
        {"optimizer": {}, "optimizer.peak": 0.001},
        {},
        {"optimizer.peak": float("inf")},
        {"optimizer.peak": float("nan")},
    ],
)
def test_config_derivation_rejects_invalid_settings_without_output(
    source_config: Path, tmp_path: Path, settings: dict[str, object]
) -> None:
    output = tmp_path / "invalid.yaml"
    with pytest.raises(ValueError):
        derive_config(source_config, output, settings)
    assert not output.exists()
    assert not _sidecar(output).exists()


def test_config_derivation_publication_is_exclusive(
    source_config: Path, tmp_path: Path
) -> None:
    output = tmp_path / "derived.yaml"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises((FileExistsError, ValueError)):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert output.read_text(encoding="utf-8") == "existing"

    output.unlink()
    _sidecar(output).write_text("existing receipt", encoding="utf-8")
    with pytest.raises((FileExistsError, ValueError)):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert not output.exists()
    assert _sidecar(output).read_text(encoding="utf-8") == "existing receipt"

    _sidecar(output).unlink()
    output.symlink_to(tmp_path / "missing-target.yaml")
    with pytest.raises((FileExistsError, OSError, ValueError)):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert output.is_symlink()


def test_config_derivation_removes_only_its_receipt_after_payload_failure(
    source_config: Path, tmp_path: Path, monkeypatch
) -> None:
    from klab import derivation

    output = tmp_path / "derived.yaml"
    original = derivation._exclusive_bytes
    calls = 0

    def fail_payload(path: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected payload publication failure")
        original(path, content)

    monkeypatch.setattr(derivation, "_exclusive_bytes", fail_payload)
    with pytest.raises(OSError, match="injected"):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert not output.exists()
    assert not _sidecar(output).exists()


def test_config_derivation_rehashes_source_before_publication(
    source_config: Path, tmp_path: Path, monkeypatch
) -> None:
    from klab import derivation

    output = tmp_path / "derived.yaml"
    original = derivation.sha256_file
    source_reads = 0

    def mutate_between_hashes(path: Path) -> str:
        nonlocal source_reads
        digest = original(path)
        if path.resolve() == source_config.resolve():
            source_reads += 1
            if source_reads == 1:
                source_config.write_text(
                    source_config.read_text(encoding="utf-8").replace(
                        "seed: 7", "seed: 8"
                    ),
                    encoding="utf-8",
                )
        return digest

    monkeypatch.setattr(derivation, "sha256_file", mutate_between_hashes)
    with pytest.raises(ValueError, match="changed"):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert not output.exists()
    assert not _sidecar(output).exists()


def test_config_derive_cli_emits_receipt_and_expected_errors(
    source_config: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "cli.yaml"
    response = _run_config_derive(
        monkeypatch,
        capsys,
        [
            str(source_config),
            "--set",
            "optimizer.peak=0.001",
            "--output",
            str(output),
            "--json",
        ],
    )
    assert response["status"] == "ok"
    assert response["receipt"] == str(_sidecar(output))
    assert response["validation"]["scope"] == "typed_config"

    failed = tmp_path / "failed.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "klab",
            "config",
            "derive",
            str(source_config),
            "--set",
            "training.seq_len=999999",
            "--output",
            str(failed),
            "--json",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert isinstance(payload["error"], str)


@pytest.mark.parametrize(
    "dataset",
    [
        {
            "source": "snapshot",
            "revision": "a" * 64,
            "license": "test",
            "train_path": "data/train.jsonl",
            "validation_path": "data/validation.jsonl",
            "source_manifest_path": "data/source.json",
        },
        {
            "source": "local_chat",
            "revision": "a" * 64,
            "license": "test",
            "train_path": "data/train.jsonl",
            "validation_path": "data/validation.jsonl",
            "allocation_manifest_path": "data/allocation.json",
            "corpus_release_path": "forge/release",
            "corpus_export_path": "forge/export",
        },
    ],
)
def test_config_relocation_preserves_all_typed_input_targets(
    source_config, tmp_path, dataset
):
    raw = yaml.safe_load(source_config.read_text())
    raw["dataset"].update(dataset)
    raw["model"].update(
        memory="portable",
        memory_table_size=64,
        memory_ngram_size=2,
        memory_dim=8,
        memory_package_path="memory/package",
    )
    source_config.write_text(yaml.safe_dump(raw))
    output = tmp_path / "relocated.yaml"
    receipt = derive_config(source_config, output, {"optimizer.peak": 0.001})
    loaded = load_config(output)
    for field, authored in dataset.items():
        if field.endswith("_path"):
            assert (
                getattr(loaded.dataset, field)
                == (source_config.parent / authored).resolve()
            )
    assert (
        loaded.model.memory_package_path
        == (source_config.parent / "memory/package").resolve()
    )
    assert receipt["delta"] == {"optimizer.peak": {"base": 0.003, "variant": 0.001}}
    rebound = {row["path"]: row for row in receipt["path_rebindings"]}
    assert rebound["model.memory_package_path"]["source_value"] == "memory/package"
    assert rebound["model.memory_package_path"]["resolved_target"] == str(
        loaded.model.memory_package_path
    )


@pytest.mark.parametrize("destination", ["invalid.json", "missing/derived.yaml"])
def test_config_requires_yaml_and_existing_parent(source_config, tmp_path, destination):
    output = tmp_path / destination
    with pytest.raises(ValueError):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert not output.exists()
    assert not output.parent.joinpath(output.name + ".derivation.json").exists()


def test_competing_publication_never_overwrites_competitor(
    source_config, tmp_path, monkeypatch
):
    from klab import derivation

    output = tmp_path / "derived.yaml"
    original = derivation._exclusive_bytes

    def race(path, content):
        if path == output:
            output.write_bytes(b"competing writer")
        original(path, content)

    monkeypatch.setattr(derivation, "_exclusive_bytes", race)
    with pytest.raises(FileExistsError):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert output.read_bytes() == b"competing writer"
    assert not _sidecar(output).exists()


def test_dangling_receipt_is_not_replaced(source_config, tmp_path):
    output = tmp_path / "derived.yaml"
    _sidecar(output).symlink_to(tmp_path / "missing.json")
    with pytest.raises(FileExistsError):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert _sidecar(output).is_symlink()
    assert not output.exists()


def test_config_cli_text_reports_published_handoff(
    source_config, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "text.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "klab",
            "config",
            "derive",
            str(source_config),
            "--set",
            "optimizer.peak=0.001",
            "--output",
            str(output),
        ],
    )
    main()
    captured = capsys.readouterr()
    assert str(output) in captured.out
    assert str(_sidecar(output)) in captured.out
    assert "typed_config" in captured.out
    assert load_config(output).optimizer.peak == 0.001
    assert json.loads(_sidecar(output).read_text())["delta"]["optimizer.peak"] == {
        "base": 0.003,
        "variant": 0.001,
    }
