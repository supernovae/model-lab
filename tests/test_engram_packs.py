from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file, save_file

from klab.canonical import canonical_digest, canonical_json
from klab.cli.main import main
from klab.engram import packs
from klab.engram.packs import (
    PublishedPackDurabilityError,
    compile_pack,
    inspect_pack,
    load_pack,
    verify_pack,
)
from klab.model.portable_engram import (
    PortableEngramAdapter,
    export_portable_engram,
    load_portable_engram,
)

CREATED_AT = "2026-09-23T00:00:00Z"


def _rows() -> list[dict[str, object]]:
    return [
        {
            "id": "r1",
            "subject": "Blorvia",
            "relation": "capital",
            "value": "Zanther",
            "aliases": ["Zanther City"],
            "hard_negative_ids": ["r2"],
            "license": "CC0-1.0",
        },
        {
            "id": "r2",
            "subject": "Kelmar",
            "relation": "primary fruit",
            "value": "Tupin",
            "aliases": ["Tupin fruit"],
            "license": "CC0-1.0",
            "source": "catalog.other",
            "source_revision": "snapshot-2",
        },
        {
            "id": "r3",
            "text": "A time-bounded free-text knowledge entry",
            "license": "CC-BY-4.0",
            "valid_from": "2025-01-01",
            "valid_until": "2025-12-31T23:59:59Z",
        },
    ]


def _write_rows(path, *, cosmetic: bool = False) -> None:
    rows = _rows()
    if cosmetic:
        rows = [dict(reversed(list(row.items()))) for row in rows]
        path.write_text(
            "\n".join(
                json.dumps(row, ensure_ascii=False, separators=(", ", ": "))
                for row in rows
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )


def _compile(source, output, **kwargs):
    return compile_pack(
        source,
        output,
        name="synthetic-demo",
        namespace="synthetic",
        created_at=CREATED_AT,
        source_name="catalog.default",
        source_revision="snapshot-1",
        default_license="CC0-1.0",
        **kwargs,
    )


def _semantic_assets(root, *, ids=None, keys=None, values=None, dtype=np.float32):
    ids = ids or ["r3", "r1"]
    keys = np.asarray(
        keys if keys is not None else [[0.0, 1.0], [1.0, 0.0]], dtype=dtype
    )
    values = np.asarray(
        values if values is not None else [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
    )
    keys_path = root / "keys.safetensors"
    values_path = root / "values.safetensors"
    metadata_path = root / "semantic-input.json"
    save_file({"keys": keys}, keys_path)
    save_file({"values": values}, values_path)
    metadata_path.write_text(
        json.dumps(
            {
                "format": "klab-semantic-assets",
                "format_version": 1,
                "record_ids": ids,
                "key_encoder": {
                    "name": "shared-key-encoder",
                    "revision": "frozen-v1",
                    "sha256": "a" * 64,
                },
                "value_encoder": {
                    "name": "shared-value-encoder",
                    "revision": "frozen-v1",
                    "sha256": "b" * 64,
                },
                "key_normalization": "l2",
            }
        ),
        encoding="utf-8",
    )
    return keys_path, values_path, metadata_path


def test_deterministic_relocation_and_metadata_only_inspection(tmp_path) -> None:
    source_a = tmp_path / "first.jsonl"
    source_b = tmp_path / "second.jsonl"
    _write_rows(source_a)
    _write_rows(source_b, cosmetic=True)
    manifest_a = _compile(source_a, tmp_path / "a.enpack")
    manifest_b = _compile(source_b, tmp_path / "b.enpack")
    assert manifest_a.pack_id == manifest_b.pack_id
    assert sorted(path.name for path in (tmp_path / "a.enpack").iterdir()) == [
        "manifest.json",
        "records.jsonl",
    ]
    for name in ("manifest.json", "records.jsonl"):
        assert (tmp_path / "a.enpack" / name).read_bytes() == (
            tmp_path / "b.enpack" / name
        ).read_bytes()

    moved = tmp_path / "relocated.enpack"
    shutil.copytree(tmp_path / "a.enpack", moved)
    source_a.unlink()
    assert verify_pack(moved, expected_pack_id=manifest_a.pack_id).valid
    assert not verify_pack(moved, expected_pack_id="0" * 64).valid
    assert inspect_pack(moved)["verification_status"] == "not_verified"

    corrupt = tmp_path / "corrupt.enpack"
    shutil.copytree(moved, corrupt)
    (corrupt / "records.jsonl").write_bytes(b"damaged\n")
    assert inspect_pack(corrupt)["verification_status"] == "not_verified"
    report = verify_pack(corrupt)
    assert not report.valid
    assert any(error.field == "records.jsonl" for error in report.errors)


def test_semantic_arrays_bind_ordered_subset_and_dimensions(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(source)
    keys, values, metadata = _semantic_assets(tmp_path)
    manifest = _compile(
        source,
        tmp_path / "semantic.enpack",
        semantic_keys=keys,
        semantic_values=values,
        semantic_metadata=metadata,
    )
    loaded = load_pack(tmp_path / "semantic.enpack", expected_pack_id=manifest.pack_id)
    assert loaded.manifest.semantic is not None
    assert loaded.manifest.semantic.entry_count == 2
    assert loaded.manifest.semantic.key_dim == 2
    assert loaded.manifest.semantic.memory_dim == 3
    assert loaded.semantic_metadata is not None
    assert loaded.semantic_metadata.record_ids == ("r3", "r1")
    assert np.array_equal(
        load_file(loaded.root / "semantic_keys.safetensors")["keys"],
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )
    assert np.array_equal(
        load_file(loaded.root / "semantic_values.safetensors")["values"],
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    )
    changed_keys = np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    save_file({"keys": changed_keys}, keys)
    changed_vectors = _compile(
        source,
        tmp_path / "changed-vectors.enpack",
        semantic_keys=keys,
        semantic_values=values,
        semantic_metadata=metadata,
    )
    assert changed_vectors.pack_id != manifest.pack_id
    assert changed_vectors.semantic is not None
    assert manifest.semantic.space_id == changed_vectors.semantic.space_id
    (tmp_path / "semantic.enpack/semantic_keys.safetensors").write_bytes(
        (tmp_path / "semantic.enpack/semantic_keys.safetensors").read_bytes()
        + b"tamper"
    )
    assert not verify_pack(tmp_path / "semantic.enpack").valid


def test_semantic_inputs_require_complete_valid_group(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(source)
    keys, _, _ = _semantic_assets(tmp_path)
    with pytest.raises(ValueError, match="supplied together"):
        _compile(source, tmp_path / "partial.enpack", semantic_keys=keys)

    zero_keys, zero_values, zero_metadata = _semantic_assets(
        tmp_path,
        keys=[[0.0, 0.0], [1.0, 0.0]],
    )
    with pytest.raises(ValueError, match="zero vector"):
        _compile(
            source,
            tmp_path / "zero.enpack",
            semantic_keys=zero_keys,
            semantic_values=zero_values,
            semantic_metadata=zero_metadata,
        )

    missing_keys, missing_values, missing_metadata = _semantic_assets(
        tmp_path, ids=["r3", "unknown"]
    )
    with pytest.raises(ValueError, match="unknown record IDs"):
        _compile(
            source,
            tmp_path / "missing.enpack",
            semantic_keys=missing_keys,
            semantic_values=missing_values,
            semantic_metadata=missing_metadata,
        )
    assert not (tmp_path / "partial.enpack").exists()
    assert not (tmp_path / "zero.enpack").exists()
    assert not (tmp_path / "missing.enpack").exists()


def test_rehashed_semantic_row_map_still_fails_verification(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(source)
    keys, values, metadata = _semantic_assets(tmp_path)
    _compile(
        source,
        tmp_path / "semantic.enpack",
        semantic_keys=keys,
        semantic_values=values,
        semantic_metadata=metadata,
    )
    pack = tmp_path / "semantic.enpack"
    metadata_path = pack / "semantic.json"
    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_metadata["record_ids"][0] = "absent"
    metadata_bytes = canonical_json(raw_metadata) + b"\n"
    metadata_path.write_bytes(metadata_bytes)

    raw_manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    identity = next(
        item
        for item in raw_manifest["files"]
        if item["relative_path"] == "semantic.json"
    )
    identity["sha256"] = hashlib.sha256(metadata_bytes).hexdigest()
    identity["size_bytes"] = len(metadata_bytes)
    payload = dict(raw_manifest)
    payload.pop("pack_id")
    raw_manifest["pack_id"] = canonical_digest(payload)
    (pack / "manifest.json").write_bytes(canonical_json(raw_manifest) + b"\n")

    report = verify_pack(pack)
    assert not report.valid
    assert any("unknown record IDs" in error.reason for error in report.errors)


def test_legacy_component_is_copied_and_loaded_by_unchanged_api(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(source)
    legacy = tmp_path / "legacy.engram"
    table = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    export_portable_engram(table, legacy, ngram_size=3)
    legacy_bytes = legacy.read_bytes()

    manifest = _compile(
        source,
        tmp_path / "with-legacy.enpack",
        lexical_package=legacy,
    )
    assert manifest.lexical is not None
    assert manifest.lexical.table_size == 6
    copied = tmp_path / "with-legacy.enpack" / "lexical.engram"
    assert copied.read_bytes() == legacy_bytes
    assert (
        load_portable_engram(copied).manifest.table_sha256
        == manifest.lexical.table_sha256
    )

    package = load_portable_engram(copied)
    first = PortableEngramAdapter(package, hidden_dim=5)
    second = PortableEngramAdapter(package, hidden_dim=9)
    assert torch.equal(first.embedding.weight, second.embedding.weight)
    assert not first.embedding.weight.requires_grad
    assert first.output.weight.shape == (5, 4)
    assert second.output.weight.shape == (9, 4)

    corrupted = tmp_path / "with-legacy.enpack/lexical.engram"
    corrupted.write_bytes(corrupted.read_bytes() + b"tamper")
    assert not verify_pack(tmp_path / "with-legacy.enpack").valid


def test_rejects_unexpected_symlink_and_manifest_traversal(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(source)
    _compile(source, tmp_path / "valid.enpack")

    extra = tmp_path / "extra.enpack"
    shutil.copytree(tmp_path / "valid.enpack", extra)
    (extra / "other.txt").write_text("extra", encoding="utf-8")
    assert not verify_pack(extra).valid

    special = tmp_path / "special.enpack"
    shutil.copytree(tmp_path / "valid.enpack", special)
    os.mkfifo(special / "pipe")
    assert not verify_pack(special).valid

    outside = tmp_path / "outside.txt"
    outside.write_text("do not read", encoding="utf-8")
    linked = tmp_path / "linked.enpack"
    shutil.copytree(tmp_path / "valid.enpack", linked)
    (linked / "records.jsonl").unlink()
    (linked / "records.jsonl").symlink_to(outside)
    assert not verify_pack(linked).valid
    assert outside.read_text(encoding="utf-8") == "do not read"

    link_root = tmp_path / "root-link.enpack"
    link_root.symlink_to(tmp_path / "valid.enpack", target_is_directory=True)
    assert not verify_pack(link_root).valid

    manifest_path = tmp_path / "valid.enpack/manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["files"][0]["relative_path"] = "../outside.txt"
    identity = dict(raw)
    identity.pop("pack_id")
    raw["pack_id"] = canonical_digest(identity)
    manifest_path.write_bytes(canonical_json(raw) + b"\n")
    with pytest.raises(ValueError, match="unsafe or unsupported"):
        inspect_pack(tmp_path / "valid.enpack")


def test_no_replace_publication_preserves_racing_destination(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(source)
    destination = tmp_path / "race.enpack"
    real_rename = packs._rename_noreplace

    def race(source_path, output_path):
        output_path.mkdir()
        (output_path / "winner").write_text("pre-existing", encoding="utf-8")
        real_rename(source_path, output_path)

    monkeypatch.setattr(packs, "_rename_noreplace", race)
    with pytest.raises(FileExistsError):
        _compile(source, destination)
    assert (destination / "winner").read_text(encoding="utf-8") == "pre-existing"
    assert not list(tmp_path.glob(".race.enpack.tmp-*"))


def test_post_commit_parent_fsync_failure_keeps_published_pack(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(source)
    destination = tmp_path / "committed.enpack"

    def fail_fsync(_parent):
        raise OSError("simulated parent fsync failure")

    monkeypatch.setattr(packs, "_fsync_parent_if_supported", fail_fsync)
    with pytest.raises(PublishedPackDurabilityError, match="published at"):
        _compile(source, destination)
    assert destination.is_dir()
    assert verify_pack(destination).valid


def test_failed_compile_does_not_publish_or_modify_source(tmp_path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text('{"id":"bad","unknown":1}\n', encoding="utf-8")
    original = source.read_bytes()
    destination = tmp_path / "bad.enpack"
    with pytest.raises(ValueError, match="unknown record fields"):
        _compile(source, destination)
    assert source.read_bytes() == original
    assert not destination.exists()
    assert not list(tmp_path.glob(".bad.enpack.tmp-*"))
    linked_source = tmp_path / "linked.jsonl"
    linked_source.symlink_to(source)
    with pytest.raises(ValueError, match="regular nonsymlink"):
        _compile(linked_source, tmp_path / "linked.enpack")
    assert not (tmp_path / "linked.enpack").exists()


def _run_cli(monkeypatch, capsys, argv: list[str]) -> dict[str, object]:
    monkeypatch.setattr(sys, "argv", ["klab", *argv])
    main()
    return json.loads(capsys.readouterr().out)


def test_pack_cli_dispatch_json_and_exit_statuses(
    tmp_path, monkeypatch, capsys
) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(source)
    destination = tmp_path / "cli.enpack"
    compiled = _run_cli(
        monkeypatch,
        capsys,
        [
            "engram",
            "pack",
            "compile",
            str(source),
            "--output",
            str(destination),
            "--name",
            "cli",
            "--namespace",
            "synthetic",
            "--created-at",
            CREATED_AT,
        ],
    )
    assert compiled["record_count"] == 3
    assert compiled["lexical_count"] == 0
    assert compiled["semantic_count"] == 0
    inspected = _run_cli(
        monkeypatch, capsys, ["engram", "pack", "inspect", str(destination)]
    )
    assert inspected["verification_status"] == "not_verified"
    verified = _run_cli(
        monkeypatch, capsys, ["engram", "pack", "verify", str(destination)]
    )
    assert verified["valid"] is True

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "klab",
            "engram",
            "pack",
            "verify",
            str(destination),
            "--expected-pack-id",
            "0" * 64,
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False


def test_cli_schema_errors_are_concise(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "invalid.jsonl"
    invalid_confidence = int("9" * 400)
    source.write_text(
        json.dumps(_rows()[0] | {"confidence": invalid_confidence}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "klab",
            "engram",
            "pack",
            "compile",
            str(source),
            "--output",
            str(tmp_path / "invalid.enpack"),
            "--name",
            "invalid",
            "--namespace",
            "synthetic",
            "--created-at",
            CREATED_AT,
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()

    payload = json.loads(capsys.readouterr().out)
    reason = payload["errors"][0]["reason"]
    assert error.value.code == 1
    assert payload["valid"] is False
    assert len(reason) < 300
    assert str(invalid_confidence) not in reason


def test_identity_changes_with_records_time_and_name(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(source)
    base = _compile(source, tmp_path / "base.enpack")
    changed_name = compile_pack(
        source,
        tmp_path / "name.enpack",
        name="different-name",
        namespace="synthetic",
        created_at=CREATED_AT,
        source_name="catalog.default",
        source_revision="snapshot-1",
        default_license="CC0-1.0",
    )
    changed_time = compile_pack(
        source,
        tmp_path / "time.enpack",
        name="synthetic-demo",
        namespace="synthetic",
        created_at="2026-09-24T00:00:00Z",
        source_name="catalog.default",
        source_revision="snapshot-1",
        default_license="CC0-1.0",
    )
    changed_source = tmp_path / "changed.jsonl"
    changed_rows = _rows()
    changed_rows[0]["value"] = "Another capital"
    changed_source.write_text(
        "\n".join(json.dumps(row) for row in changed_rows) + "\n",
        encoding="utf-8",
    )
    changed_record = _compile(changed_source, tmp_path / "record.enpack")
    assert (
        len(
            {
                base.pack_id,
                changed_name.pack_id,
                changed_time.pack_id,
                changed_record.pack_id,
            }
        )
        == 4
    )


def test_semantic_rejects_bad_row_maps_shapes_dtypes_and_norms(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(source)
    cases = [
        ("duplicate", {"ids": ["r3", "r3"]}, "unique"),
        ("short-values", {"values": [[1.0, 2.0, 3.0]]}, "row counts"),
        ("wrong-dtype", {"dtype": np.float64}, "FP32"),
        ("nonfinite-key", {"keys": [[float("nan"), 0.0], [1.0, 0.0]]}, "nonfinite"),
        ("false-l2", {"keys": [[2.0, 0.0], [1.0, 0.0]]}, "norms"),
        (
            "nonfinite-value",
            {"values": [[float("nan"), 2.0, 3.0], [4.0, 5.0, 6.0]]},
            "nonfinite",
        ),
    ]
    for name, options, message in cases:
        keys, values, metadata = _semantic_assets(tmp_path, **options)
        with pytest.raises((TypeError, ValueError), match=message):
            _compile(
                source,
                tmp_path / f"{name}.enpack",
                semantic_keys=keys,
                semantic_values=values,
                semantic_metadata=metadata,
            )
        assert not (tmp_path / f"{name}.enpack").exists()


def test_existing_output_is_never_replaced(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(source)
    empty_directory = tmp_path / "empty.enpack"
    empty_directory.mkdir()
    with pytest.raises(FileExistsError):
        _compile(source, empty_directory)
    assert list(empty_directory.iterdir()) == []

    existing_file = tmp_path / "file.enpack"
    existing_file.write_bytes(b"prior bytes")
    with pytest.raises(FileExistsError):
        _compile(source, existing_file)
    assert existing_file.read_bytes() == b"prior bytes"


def test_copy_failure_cleans_only_private_temporary_artifact(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(source)
    lexical = tmp_path / "lexical.engram"
    export_portable_engram(
        torch.zeros((2, 3), dtype=torch.float32), lexical, ngram_size=2
    )
    input_before = source.read_bytes()
    lexical_before = lexical.read_bytes()
    destination = tmp_path / "copy-failure.enpack"

    def fail_copy(_source, _destination):
        raise OSError("injected copy failure")

    monkeypatch.setattr(packs, "_copy_regular", fail_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        _compile(source, destination, lexical_package=lexical)
    assert not destination.exists()
    assert source.read_bytes() == input_before
    assert lexical.read_bytes() == lexical_before
    assert not list(tmp_path.glob(".copy-failure.enpack.tmp-*"))


def test_legacy_engram_inspect_leaf_remains_unchanged(
    tmp_path, monkeypatch, capsys
) -> None:
    package_path = tmp_path / "legacy.engram"
    expected = export_portable_engram(
        torch.zeros((3, 2), dtype=torch.float32),
        package_path,
        ngram_size=2,
    )
    monkeypatch.setattr(sys, "argv", ["klab", "engram", "inspect", str(package_path)])
    main()
    assert json.loads(capsys.readouterr().out) == expected.as_dict()
