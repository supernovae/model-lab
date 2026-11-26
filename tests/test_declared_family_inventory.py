"""Family inventory for releases built with declared source-family splits."""

from __future__ import annotations

import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml

from klab.canonical import canonical_json
from klab.corpus import declared_family_inventory, split_freeze
from klab.corpus.mixture_index import MixtureIndex
from klab.corpus.release import ReleaseReader
from klab.corpus.release_receipt import fingerprint

TRAIN_SOURCES = {"ref_a": 400, "ref_b": 40, "ref_c": 10}
STRATA = {"ref_a": "reference", "ref_b": "reference", "ref_c": "reference"}


def _release(tmp_path: Path, monkeypatch, split) -> tuple[Path, Path]:
    release = tmp_path / "release"
    release.mkdir()
    sources = {**TRAIN_SOURCES, "ref_held": 20}
    docs = [
        {
            "document_id": f"{source}-{i}",
            "source_id": source,
            "source_family": source,
            "split": "validation" if source == "ref_held" else "train",
            "domains": ["reference"],
            "content_sha256": f"{source}-{i}".encode().hex().ljust(64, "0"),
            "drop_reason": "duplicate" if i == 0 else None,
            "rights": {"training_eligibility": "eligible"},
        }
        for source, count in sources.items()
        for i in range(count)
    ]
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in docs)
    )
    (release / "train.lineage.jsonl").write_bytes(
        b"".join(
            canonical_json({"record_id": row["document_id"]}) + b"\n"
            for row in docs
            if row["split"] == "train" and row["drop_reason"] is None
        )
    )
    manifest = {
        "release_id": "fixture",
        "build_identity": {"split": split},
        "files": {"documents.jsonl": {}},
        "snapshots": [],
    }
    (release / "manifest.json").write_bytes(canonical_json(manifest))
    monkeypatch.setattr(
        declared_family_inventory,
        "open_release",
        lambda root: nullcontext(ReleaseReader(root, manifest, fingerprint(root))),
    )
    monkeypatch.setattr(split_freeze, "verify_release", lambda _: manifest)
    declaration = tmp_path / "mixture.yaml"
    declaration.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "release_path": str(release),
                "tokenizer_config": str(tmp_path / "tokenizer.yaml"),
                "family_inventory": str(tmp_path / "out" / "family-inventory.jsonl"),
                "source_strata": {**STRATA, "ref_held": "reference"},
                "target_quotas": {"reference": 100},
                "seed": 42,
                "max_exposures": 1,
            }
        )
    )
    return release, declaration


def _split(unit="source_document_family", **extra):
    return {
        "schema_version": 1,
        "unit": unit,
        "family_key": "source_family",
        "assignments": {
            **{source: "train" for source in TRAIN_SOURCES},
            "ref_held": "validation",
        },
        **extra,
    }


def test_derived_inventory_fills_a_multi_source_stratum_without_dropping_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, declaration = _release(tmp_path, monkeypatch, _split())
    result = declared_family_inventory.derive_family_inventory(declaration)
    output = Path(result["output"])
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["kept_documents"] == len(rows) == 466
    assert {row["family_id"] for row in rows} == {*TRAIN_SOURCES, "ref_held"}
    assert {row["split"] for row in rows if row["family_id"] == "ref_held"} == {
        "validation"
    }

    scratch = tmp_path / "index"
    scratch.mkdir()
    with MixtureIndex(scratch, release / "documents.jsonl") as index:
        index.inventory(
            output,
            STRATA | {"ref_held": "reference"},
            42,
        )
        index.select(release / "train.lineage.jsonl")
        taken = Counter()
        for position, (doc, family) in enumerate(index.candidates("reference")):
            assert family == doc["source_id"]
            taken[family] += 1
            if position + 1 == 100:
                break
    # A 100-document quota out of 447 train documents takes every source in
    # proportion to its size; it never exhausts one source and skips another.
    assert set(taken) == set(TRAIN_SOURCES)
    assert taken["ref_a"] > taken["ref_b"] > 0
    assert taken["ref_c"] >= 1
    assert taken["ref_a"] < 400 and sum(taken.values()) == 100


def test_derived_inventory_refuses_reviewed_or_unmapped_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, declaration = _release(
        tmp_path, monkeypatch, _split(unit="document", assignment_file={"rows": 1})
    )
    with pytest.raises(ValueError, match="not built with declared source-family"):
        declared_family_inventory.derive_family_inventory(declaration)
    assert not (tmp_path / "out" / "family-inventory.jsonl").exists()

    declared = {"release_id": "fixture", "build_identity": {"split": _split()}}
    monkeypatch.setattr(split_freeze, "verify_release", lambda _: declared)
    with pytest.raises(ValueError, match="declared source-family splits"):
        split_freeze.finalize_family_inventory(
            tmp_path / "release", tmp_path / "splits.yaml", tmp_path / "x.jsonl"
        )


def test_derived_inventory_requires_stratum_in_domains_and_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, declaration = _release(tmp_path, monkeypatch, _split())
    rows = [
        json.loads(line)
        for line in (release / "documents.jsonl").read_text().splitlines()
    ]
    rows[1]["domains"] = ["code"]
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in rows)
    )
    with pytest.raises(ValueError, match="do not include its mixture stratum"):
        declared_family_inventory.derive_family_inventory(declaration)
    rows[1]["domains"] = ["reference"]
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in rows)
    )
    declared_family_inventory.derive_family_inventory(declaration)
    with pytest.raises(ValueError, match="already exists"):
        declared_family_inventory.derive_family_inventory(declaration)


def test_sharded_inventory_is_byte_identical_to_the_single_process_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus import shards
    from klab.corpus.release_verify import verification_settings

    _, declaration = _release(tmp_path, monkeypatch, _split())
    output = Path(yaml.safe_load(declaration.read_text())["family_inventory"]).resolve()
    with verification_settings(workers=1):
        single = declared_family_inventory.derive_family_inventory(declaration)
    expected = output.read_bytes()
    output.unlink()
    monkeypatch.setattr(shards, "PARALLEL_MIN_BYTES", 0)
    monkeypatch.setattr(shards, "CHUNK_BYTES", 3000)
    scratch = tmp_path / "scratch"
    with verification_settings(workers=3, scratch_dir=scratch):
        sharded = declared_family_inventory.derive_family_inventory(declaration)
    assert output.read_bytes() == expected
    assert sharded == single


def test_inventory_resumes_with_verified_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus import shards
    from klab.corpus.release_verify import verification_settings

    _, declaration = _release(tmp_path, monkeypatch, _split())
    output = Path(yaml.safe_load(declaration.read_text())["family_inventory"])
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(shards, "CHUNK_BYTES", 3000)
    with verification_settings(workers=1):
        expected = declared_family_inventory.derive_family_inventory(declaration)
    expected_bytes = output.read_bytes()
    output.unlink()

    monkeypatch.setenv("KLAB_FAULT", "family inventory:2")
    with (
        verification_settings(workers=1, scratch_dir=scratch),
        pytest.raises(RuntimeError, match="KLAB_FAULT"),
    ):
        declared_family_inventory.derive_family_inventory(declaration)
    assert not output.exists()
    good, damaged = sorted(
        scratch.glob("shards/files/shard/*/inventory.jsonl"),
        key=lambda path: json.loads(path.with_name("receipt.json").read_bytes())[
            "inputs"
        ]["task"][1],
    )
    inode = good.stat().st_ino
    damaged.write_bytes(b"bad data")
    monkeypatch.delenv("KLAB_FAULT")

    run = declared_family_inventory.run_sharded_files

    def unassembled_only(plan, *args, **options):
        assert plan.tasks and plan.tasks[0][1] > 0
        return run(plan, *args, **options)

    monkeypatch.setattr(
        declared_family_inventory, "run_sharded_files", unassembled_only
    )
    with verification_settings(workers=1, scratch_dir=scratch):
        resumed = declared_family_inventory.derive_family_inventory(declaration)
    assert good.stat().st_ino == inode
    assert damaged.read_bytes() != b"bad data"
    assert output.read_bytes() == expected_bytes
    assert resumed == expected
