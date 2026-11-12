from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from klab.canonical import canonical_digest, canonical_json
from klab.config.loading import load_config
from klab.runtime import RuntimeInfo
from klab.training import manifest
from klab.training.manifest import (
    ArtifactIdentity,
    RunManifest,
    architecture_sha256,
    config_sha256,
    read_manifest,
    write_manifest,
)


def test_source_identity_excludes_only_self_referential_lifecycle_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "klab"
    lifecycle = package_root / "research" / "resources" / "lifecycle.json"
    implementation = package_root / "training" / "implementation.py"
    lifecycle.parent.mkdir(parents=True)
    implementation.parent.mkdir(parents=True)
    lifecycle.write_text('{"findings":[]}\n', encoding="utf-8")
    implementation.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        manifest, "__file__", str(implementation.parent / "manifest.py")
    )

    initial = manifest.source_identity()
    lifecycle.write_text('{"findings":[{"id":"later"}]}\n', encoding="utf-8")

    assert manifest.source_identity() == initial
    implementation.write_text("value = 2\n", encoding="utf-8")

    changed = manifest.source_identity()
    assert changed["algorithm"] == "package-path-sha256-v2"
    assert changed["sha256"] != initial["sha256"]


def _runtime() -> RuntimeInfo:
    return RuntimeInfo(
        "pytorch",
        "cpu",
        "cpu",
        0,
        "CPU",
        None,
        "test",
        None,
        None,
        "test",
        None,
        None,
        None,
        None,
        None,
        "test",
        "test",
        (),
    )


def _config(root: Path) -> dict[str, object]:
    return {
        "schema_version": 2,
        "model": {
            "vocab_size": 512,
            "hidden_dim": 16,
            "memory_package_path": str(root / "package"),
        },
        "attention": {"kind": "dense", "rope_base": 10000.0},
        "tokenizer": {"path": str(root / "tokenizer.json")},
        "dataset": {"cache_dir": str(root / "cache")},
        "logging": {"root_dir": str(root / "runs")},
    }


def test_identity_ignores_machine_paths_but_binds_architecture() -> None:
    left, right = _config(Path("/machine-a")), _config(Path("/machine-b"))
    assert config_sha256(left) == config_sha256(right)
    assert architecture_sha256(left) == architecture_sha256(right)

    changed = _config(Path("/machine-a"))
    changed_model = cast(dict[str, object], changed["model"])
    changed["model"] = {**changed_model, "hidden_dim": 32}
    assert architecture_sha256(left) != architecture_sha256(changed)
    assert config_sha256(left) != config_sha256(changed)


def test_final_memory_injection_preserves_hashes_and_embedding_binds() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "runtime_smoke_cpu.yaml": (
            "0a2c6cff8756684c12f1ce2743eb392e40a161ce64caa14dbd039aeaa56a4787",
            "4f14421750a237f8fbc9bfd4081931525c9185d9cf657cd92fbdc2a46b608ca4",
        ),
    }
    for filename, digests in expected.items():
        config = load_config(root / "configs" / filename).model_dump(mode="json")
        assert (config_sha256(config), architecture_sha256(config)) == digests

    legacy = load_config(root / "configs" / "runtime_smoke_cpu.yaml").model_dump(
        mode="json"
    )
    legacy["model"].update(
        memory="ngram", memory_table_size=31, memory_ngram_size=2, memory_dim=8
    )
    explicit_final = {
        **legacy,
        "model": {**legacy["model"], "memory_injection": "final"},
    }
    assert config_sha256(explicit_final) == config_sha256(legacy)
    assert architecture_sha256(explicit_final) == architecture_sha256(legacy)
    assert architecture_sha256(explicit_final["model"]) == architecture_sha256(
        legacy["model"]
    )
    assert explicit_final["model"]["memory_injection"] == "final"

    embedded = {
        **legacy,
        "model": {**legacy["model"], "memory_injection": "embedding"},
    }
    assert config_sha256(embedded) != config_sha256(legacy)
    assert architecture_sha256(embedded) != architecture_sha256(legacy)


def test_canonical_json_rejects_arbitrary_objects() -> None:
    with pytest.raises(TypeError):
        canonical_json(Path("not-json"))


def test_manifest_rejects_duplicate_nonfinite_and_unknown_versions(
    tmp_path: Path,
) -> None:
    artifact = ArtifactIdentity("assets/tokenizer.json", "a" * 64, 1)
    manifest = RunManifest(
        "run",
        "name",
        _runtime(),
        _config(tmp_path),
        _config(tmp_path),
        architecture_sha256(_config(tmp_path)),
        {"sha256": "source"},
        "worker",
        artifacts=(artifact,),
    )
    path = tmp_path / "manifest.json"
    digest = write_manifest(path, manifest)
    loaded = read_manifest(path)
    assert digest == canonical_digest(loaded)
    assert loaded["artifacts"] == [
        {
            "relative_path": artifact.relative_path,
            "sha256": artifact.sha256,
            "size_bytes": 1,
        }
    ]

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"manifest_version":1,"manifest_version":1}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"manifest_version":1,"x":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        read_manifest(nonfinite)

    unknown = dict(loaded)
    unknown["manifest_version"] = 99
    unknown["sha256"] = canonical_digest(unknown)
    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported manifest version"):
        read_manifest(unknown_path)


def test_resigned_architecture_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest = RunManifest(
        "run",
        "name",
        _runtime(),
        _config(tmp_path),
        _config(tmp_path),
        architecture_sha256(_config(tmp_path)),
        {"sha256": "source"},
        "worker",
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    resigned = json.loads(path.read_text(encoding="utf-8"))
    resigned["architecture_sha256"] = "0" * 64
    resigned["sha256"] = hashlib.sha256(
        canonical_json(
            {key: value for key, value in resigned.items() if key != "sha256"}
        )
    ).hexdigest()
    path.write_bytes(canonical_json(resigned) + b"\n")

    with pytest.raises(ValueError, match="architecture hash mismatch"):
        read_manifest(path)


@pytest.mark.parametrize(
    "artifacts",
    [
        [
            {"relative_path": "same.json", "sha256": "a" * 64, "size_bytes": 1},
            {"relative_path": "same.json", "sha256": "b" * 64, "size_bytes": 2},
        ],
        [{"relative_path": "", "sha256": "a" * 64, "size_bytes": True}],
    ],
)
def test_resigned_invalid_artifact_inventory_is_rejected(
    tmp_path: Path, artifacts: list[dict[str, object]]
) -> None:
    manifest = RunManifest(
        "run",
        "name",
        _runtime(),
        _config(tmp_path),
        _config(tmp_path),
        architecture_sha256(_config(tmp_path)),
        {"sha256": "source"},
        "worker",
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    resigned = json.loads(path.read_text(encoding="utf-8"))
    resigned["artifacts"] = artifacts
    resigned["sha256"] = hashlib.sha256(
        canonical_json(
            {key: value for key, value in resigned.items() if key != "sha256"}
        )
    ).hexdigest()
    path.write_bytes(canonical_json(resigned) + b"\n")

    with pytest.raises(ValueError, match="artifact inventory"):
        read_manifest(path)


def test_legacy_v1_manifest_digest_is_verified_without_reinterpretation(
    tmp_path: Path,
) -> None:
    legacy = {
        "manifest_version": 1,
        "run_id": "historic",
        "requested_config_sha256": "historical-digest",
    }
    legacy["sha256"] = canonical_digest(legacy)
    path = tmp_path / "legacy.json"
    path.write_bytes(canonical_json(legacy) + b"\n")

    assert read_manifest(path) == {
        "manifest_version": 1,
        "run_id": "historic",
        "requested_config_sha256": "historical-digest",
    }
