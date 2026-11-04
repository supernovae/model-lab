"""Offline checks for admission-aware deterministic family split freezing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sparselab.canonical import canonical_json
from sparselab.corpus import split_freeze
from sparselab.corpus.mixture import _rank
from sparselab.corpus.mixture_index import MixtureIndex
from sparselab.corpus.project import SplitDeclaration
from sparselab.training.manifest import sha256_file


def _assignments(path):
    declaration = yaml.safe_load(path.read_text())
    reference = declaration["assignment_file"]
    companion = path.parent / ("assignments-" + reference["sha256"] + ".jsonl")
    assert sha256_file(companion) == reference["sha256"]
    rows = [json.loads(line) for line in companion.read_text().splitlines()]
    assert len(rows) == reference["rows"]
    return {row["document_id"]: row["split"] for row in rows}


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family_count=10):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    sources = [SimpleNamespace(id=source) for source in ("books", "pages", "incidents")]
    project_root = tmp_path / "project"
    project_root.mkdir()
    admission = {"schema_version": 2, "sources": []}
    admission_bytes = canonical_json(admission)
    (project_root / "admission.json").write_bytes(admission_bytes)
    project = SimpleNamespace(
        config=SimpleNamespace(id="freeze-fixture"),
        root=project_root,
        release=SimpleNamespace(
            record_admission=SimpleNamespace(
                path="admission.json",
                sha256=hashlib.sha256(admission_bytes).hexdigest(),
            )
        ),
        sources=sources,
    )
    inventory = tmp_path / "inventory.jsonl"
    rows = []
    decisions = {}
    snapshots = {}
    for source in sources:
        snapshot = tmp_path / source.id
        snapshot.mkdir()
        (snapshot / "manifest.json").write_text("{}")
        snapshots[source.id] = {
            "snapshot_path": str(snapshot),
            "snapshot_sha256": "a" * 64,
        }
        for index in range(family_count):
            path = (
                f"docs/{index}.md"
                if source.id == "incidents"
                else f"{source.id}.sample.jsonl"
            )
            rows.append(
                {
                    "document_id": f"{source.id}-{index}",
                    "source_id": source.id,
                    "source_location": path
                    + (
                        "#lines=1-2"
                        if source.id == "incidents"
                        else f"#row={index + 1}"
                    ),
                    "source_row_index": None if source.id == "incidents" else index,
                    "family_hint": f"{source.id}-family:{index}",
                    "content_sha256": hashlib.sha256(
                        f"{source.id}-{index}".encode()
                    ).hexdigest(),
                }
            )
            decisions[source.id, path] = (
                {"decision": "qualify"}
                if source.id == "incidents"
                else {
                    "decisions": {
                        i: {"decision": "qualify"} for i in range(family_count)
                    }
                }
            )
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    clusters_path = tmp_path / "clusters.json"
    clusters = {
        "schema_version": 1,
        "inventory_sha256": sha256_file(inventory),
        "seed": "reviewed-seed-v1",
        "reviewer": "Fixture reviewer",
        "reviewed_on": "2026-10-07",
        "source_strata": {
            "books": "general_prose",
            "pages": "explanatory_prose",
            "incidents": "incident_response_docs",
        },
        "merges": [],
    }
    clusters_path.write_bytes(canonical_json(clusters))
    monkeypatch.setattr(
        split_freeze,
        "write_split_inventory",
        lambda *_, completion_cache: {"sha256": sha256_file(inventory)},
    )
    monkeypatch.setattr(
        split_freeze,
        "verify_acquisition",
        lambda *_, **__: {"sources": snapshots},
    )
    monkeypatch.setattr(split_freeze, "source_declaration_payload", lambda _: {})
    monkeypatch.setattr(
        "sparselab.corpus.split_inventory_cache.source_declaration_payload",
        lambda _: {},
    )
    monkeypatch.setattr(split_freeze, "verify_record_admission", lambda *_: decisions)
    return project, inventory, clusters_path, rows, decisions


def test_freeze_splits_is_deterministic_and_validates_integer_family_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters, rows, _ = _fixture(tmp_path, monkeypatch)
    first = tmp_path / "splits-first.yaml"
    second = tmp_path / "splits-second.yaml"
    report = split_freeze.freeze_splits(project, tmp_path, inventory, clusters, first)
    split_freeze.freeze_splits(project, tmp_path, inventory, clusters, second)
    assert first.read_bytes() == second.read_bytes()
    declaration = yaml.safe_load(first.read_text())
    SplitDeclaration.model_validate(declaration)
    assert set(_assignments(first)) == {row["document_id"] for row in rows}
    assert report["eligible_documents"] == 30
    assert all(
        counts
        == {
            "families": 10,
            "train": 8,
            "validation": 1,
            "test": 1,
            "fractions": {
                "train": "8/10",
                "validation": "1/10",
                "test": "1/10",
            },
        }
        for counts in report["family_counts"].values()
    )
    candidates = [
        json.loads(line)
        for line in Path(report["family_candidates"]).read_text().splitlines()
    ]
    assert len(candidates) == 30
    assert set(json.loads(Path(report["receipt"]).read_text())) == {
        "schema_version",
        "split_sha256",
        "family_candidate_sha256",
        "inventory_sha256",
        "admission_sha256",
        "cluster_sha256",
        "family_counts",
    }
    assert all(
        set(row) == {"document_id", "family_id", "split", "stratum", "content_sha256"}
        for row in candidates
    )


def test_freeze_splits_reads_embedded_unicode_separators_as_one_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters, rows, _ = _fixture(tmp_path, monkeypatch)
    rows[0]["family_hint"] += "\u0085\u2028\u2029"
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    declaration = json.loads(clusters.read_text())
    declaration["inventory_sha256"] = sha256_file(inventory)
    clusters.write_bytes(canonical_json(declaration))
    output = tmp_path / "splits-unicode.yaml"
    report = split_freeze.freeze_splits(project, tmp_path, inventory, clusters, output)
    assert report["eligible_documents"] == 30
    assert len(_assignments(output)) == 30


def test_freeze_splits_preserves_authenticated_prior_families_with_new_document_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters_path, rows, _ = _fixture(tmp_path, monkeypatch)
    release = tmp_path / ("a" * 64)
    release.mkdir()
    previous = []
    old_docs = []
    strata = {
        "books": "general_prose",
        "pages": "explanatory_prose",
        "incidents": "incident_response_docs",
    }
    expected = {}
    for row in rows:
        index = int(row["document_id"].rsplit("-", 1)[1])
        if index >= 3:
            continue
        split = ("test", "validation", "train")[index]
        old_id = "old-" + row["document_id"]
        old_docs.append(
            {
                "document_id": old_id,
                "drop_reason": None,
                "split": split,
                "content_sha256": row["content_sha256"],
                "domains": [strata[row["source_id"]]],
            }
        )
        previous.append(
            {
                "document_id": old_id,
                "family_id": row["family_hint"],
                "split": split,
                "stratum": strata[row["source_id"]],
                "content_sha256": row["content_sha256"],
            }
        )
        expected[row["document_id"]] = split
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in old_docs)
    )
    prior_path = tmp_path / "prior-families.jsonl"
    prior_path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in previous))
    clusters = json.loads(clusters_path.read_text())
    clusters["prior_family_inventory"] = {
        "path": str(prior_path),
        "sha256": sha256_file(prior_path),
        "release_path": str(release),
    }
    clusters_path.write_bytes(canonical_json(clusters))
    monkeypatch.setattr(
        split_freeze,
        "verify_release",
        lambda *a, **kw: {"release_id": release.name},
    )
    output = tmp_path / "preserved.yaml"
    report = split_freeze.freeze_splits(
        project, tmp_path, inventory, clusters_path, output
    )
    actual = _assignments(output)
    assert {key: actual[key] for key in expected} == expected
    assert json.loads(Path(report["receipt"]).read_text())[
        "prior_family_inventory_sha256"
    ] == sha256_file(prior_path)


def test_prior_family_inventory_changed_bytes_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters_path, _, _ = _fixture(tmp_path, monkeypatch)
    prior = tmp_path / "prior.jsonl"
    prior.write_text("{}\n")
    clusters = json.loads(clusters_path.read_text())
    clusters["prior_family_inventory"] = {
        "path": str(prior),
        "sha256": "0" * 64,
        "release_path": str(tmp_path / ("a" * 64)),
    }
    clusters_path.write_bytes(canonical_json(clusters))
    with pytest.raises(ValueError, match="digest mismatch"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters_path, tmp_path / "rejected.yaml"
        )


def test_prior_content_cannot_be_renamed_into_another_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters_path, rows, _ = _fixture(tmp_path, monkeypatch)
    source = rows[0]
    release = tmp_path / ("b" * 64)
    release.mkdir()
    old_id = "old-" + source["document_id"]
    (release / "documents.jsonl").write_bytes(
        canonical_json(
            {
                "document_id": old_id,
                "drop_reason": None,
                "split": "test",
                "content_sha256": source["content_sha256"],
                "domains": ["general_prose"],
            }
        )
        + b"\n"
    )
    prior = tmp_path / "old-families.jsonl"
    prior.write_bytes(
        canonical_json(
            {
                "document_id": old_id,
                "family_id": source["family_hint"],
                "split": "test",
                "stratum": "general_prose",
                "content_sha256": source["content_sha256"],
            }
        )
        + b"\n"
    )
    clusters = json.loads(clusters_path.read_text())
    clusters["prior_family_inventory"] = {
        "path": str(prior),
        "sha256": sha256_file(prior),
        "release_path": str(release),
    }
    clusters["merges"] = [
        {
            "family_id": "renamed-family",
            "member_hints": [source["family_hint"], rows[1]["family_hint"]],
        }
    ]
    clusters_path.write_bytes(canonical_json(clusters))
    monkeypatch.setattr(
        split_freeze,
        "verify_release",
        lambda *a, **kw: {"release_id": release.name},
    )
    with pytest.raises(ValueError, match="changed reviewed family"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters_path, tmp_path / "rejected.yaml"
        )


def test_exact_content_alias_uses_accepted_family_id_and_held_out_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters_path, rows, decisions = _fixture(tmp_path, monkeypatch)
    accepted, alias = rows[0], rows[1]
    alias["content_sha256"] = accepted["content_sha256"]
    extra = dict(rows[2])
    extra.update(
        document_id="books-10",
        source_row_index=10,
        family_hint="books-family:10",
        content_sha256=hashlib.sha256(b"books-10").hexdigest(),
    )
    rows.append(extra)
    decisions["books", "books.sample.jsonl"]["decisions"][10] = {"decision": "qualify"}
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    release = tmp_path / ("c" * 64)
    release.mkdir()
    old_id = "old-" + accepted["document_id"]
    (release / "documents.jsonl").write_bytes(
        canonical_json(
            {
                "document_id": old_id,
                "drop_reason": None,
                "split": "test",
                "content_sha256": accepted["content_sha256"],
                "domains": ["general_prose"],
            }
        )
        + b"\n"
    )
    prior = tmp_path / "accepted-families.jsonl"
    prior.write_bytes(
        canonical_json(
            {
                "document_id": old_id,
                "family_id": accepted["family_hint"],
                "split": "test",
                "stratum": "general_prose",
                "content_sha256": accepted["content_sha256"],
            }
        )
        + b"\n"
    )
    clusters = json.loads(clusters_path.read_text())
    clusters["inventory_sha256"] = sha256_file(inventory)
    clusters["prior_family_inventory"] = {
        "path": str(prior),
        "sha256": sha256_file(prior),
        "release_path": str(release),
    }
    clusters["merges"] = [
        {
            "family_id": accepted["family_hint"],
            "member_hints": [accepted["family_hint"], alias["family_hint"]],
        }
    ]
    clusters_path.write_bytes(canonical_json(clusters))
    monkeypatch.setattr(
        split_freeze,
        "verify_release",
        lambda *a, **kw: {"release_id": release.name},
    )
    output = tmp_path / "aliases-preserved.yaml"
    report = split_freeze.freeze_splits(
        project, tmp_path, inventory, clusters_path, output
    )
    assignments = _assignments(output)
    assert assignments[accepted["document_id"]] == "test"
    assert assignments[alias["document_id"]] == "test"
    candidates = {
        row["document_id"]: row
        for row in (
            json.loads(line)
            for line in Path(report["family_candidates"]).read_text().splitlines()
        )
    }
    assert candidates[accepted["document_id"]]["family_id"] == accepted["family_hint"]
    assert candidates[alias["document_id"]]["family_id"] == accepted["family_hint"]


def test_finalized_family_inventory_matches_kept_release_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters, _, _ = _fixture(tmp_path, monkeypatch)
    split_path = tmp_path / "splits.yaml"
    report = split_freeze.freeze_splits(
        project, tmp_path, inventory, clusters, split_path
    )
    release = tmp_path / "release"
    release.mkdir()
    candidates = [
        json.loads(line)
        for line in Path(report["family_candidates"]).read_text().splitlines()
    ]
    source_by_stratum = {
        "general_prose": "books",
        "explanatory_prose": "pages",
        "incident_response_docs": "incidents",
    }
    documents = {
        row["document_id"]: {
            "document_id": row["document_id"],
            "source_id": source_by_stratum[row["stratum"]],
            "domains": [row["stratum"]],
            "split": row["split"],
            "content_sha256": row["content_sha256"],
            "drop_reason": "duplicate" if index == 0 else None,
        }
        for index, row in enumerate(candidates)
    }
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in documents.values())
    )
    monkeypatch.setattr(
        split_freeze,
        "verify_release",
        lambda _: {
            "release_id": "fixture-release",
            "build_identity": {"split": yaml.safe_load(split_path.read_text())},
        },
    )
    final = tmp_path / "final-family-inventory.jsonl"
    result = split_freeze.finalize_family_inventory(release, split_path, final)
    assert result["kept_documents"] == 29
    kept = {key: doc for key, doc in documents.items() if doc["drop_reason"] is None}
    scratch = tmp_path / "mixture-index"
    scratch.mkdir()
    with MixtureIndex(scratch, release / "documents.jsonl") as index:
        index.inventory(
            final,
            {v: k for k, v in source_by_stratum.items()},
            lambda row: _rank(42, row["stratum"], row["family_id"], row["document_id"]),
        )
        assert {
            row[0]
            for row in index.db.execute("SELECT id FROM docs WHERE inventoried=1")
        } == set(kept)


def test_freeze_splits_stops_on_missing_decision_unresolved_family_and_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters_path, rows, decisions = _fixture(tmp_path, monkeypatch)
    decisions.pop(("incidents", "docs/0.md"))
    with pytest.raises(ValueError, match="lacks file admission"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters_path, tmp_path / "missing.yaml"
        )
    decisions["incidents", "docs/0.md"] = {"decision": "qualify"}
    rows[0]["family_hint"] = "unresolved-family:books-0"
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    clusters = json.loads(clusters_path.read_text())
    clusters["inventory_sha256"] = sha256_file(inventory)
    clusters_path.write_bytes(canonical_json(clusters))
    with pytest.raises(ValueError, match="unresolved family"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters_path, tmp_path / "unresolved.yaml"
        )


def test_cross_split_duplicate_qualifying_content_fails_but_quarantine_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters_path, rows, decisions = _fixture(tmp_path, monkeypatch)
    first = tmp_path / "baseline.yaml"
    split_freeze.freeze_splits(project, tmp_path, inventory, clusters_path, first)
    assignments = _assignments(first)
    train = next(row for row in rows if assignments[row["document_id"]] == "train")
    heldout = next(
        row
        for row in rows
        if row["source_id"] == "books"
        and assignments[row["document_id"]] == "validation"
    )
    heldout["content_sha256"] = train["content_sha256"]
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    clusters = json.loads(clusters_path.read_text())
    clusters["inventory_sha256"] = sha256_file(inventory)
    clusters_path.write_bytes(canonical_json(clusters))
    with pytest.raises(ValueError, match="cross-split exact content duplicate"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters_path, tmp_path / "duplicate.yaml"
        )
    source_id = heldout["source_id"]
    path = heldout["source_location"].split("#", 1)[0]
    sibling = {
        **heldout,
        "document_id": "books-sibling",
        "source_location": path + "#row=11",
        "source_row_index": 10,
        "content_sha256": hashlib.sha256(b"sibling").hexdigest(),
    }
    rows.append(sibling)
    decisions[source_id, path]["decisions"][10] = {"decision": "qualify"}
    decisions[source_id, path]["decisions"][heldout["source_row_index"]] = {
        "decision": "quarantine"
    }
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    clusters["inventory_sha256"] = sha256_file(inventory)
    clusters_path.write_bytes(canonical_json(clusters))
    split_freeze.freeze_splits(
        project, tmp_path, inventory, clusters_path, tmp_path / "quarantined.yaml"
    )
