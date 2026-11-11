"""Corpus normalization, ancestry, deduplication and immutable release checks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from corpus_fixtures import neutral_corpus

from klab.corpus.acquisition import acquire
from klab.corpus.pipeline import _records_for_file, _split, build
from klab.corpus.project import SourceDeclaration, load_project
from klab.corpus.release import freeze, lineage, review, sample, verify_release
from klab.corpus.rights import resolve_file_rights

PROJECT = (
    Path(__file__).resolve().parents[1] / "tests/fixtures/corpus-sample/corpus.yaml"
)


def _project():
    return load_project(PROJECT)


def test_markdown_ancestry_fences_and_raw_evidence() -> None:
    raw = b"Introduction.\n# Alpha\n- preserve list\n```md\n# not a heading\n```\n## Beta\nkey = value\n"
    source = SourceDeclaration.model_validate(
        {
            "schema_version": 1,
            "id": "docs",
            "kind": "local",
            "canonical_uri": "fixture:docs",
            "revision": "v1",
            "license": "MIT",
            "redistribution": "redistributable",
            "domains": ["technical_docs"],
            "document_kinds": ["markdown"],
            "source_family": "docs",
            "acquisition": {
                "files": [{"path": "any.md", "name": "any.md"}],
                "max_bytes": 1000,
            },
        }
    )
    docs = _records_for_file(raw, "any.md", source, "a" * 64)
    assert [row[0]["section_path"] for row in docs] == [
        [],
        ["Alpha"],
        ["Alpha", "Beta"],
    ]
    assert (
        docs[1][0]["text"] == "# Alpha\n- preserve list\n```md\n# not a heading\n```\n"
    )
    assert docs[1][1]["line_start"] == 2
    assert docs[1][1]["line_end"] == 6
    assert (
        raw[docs[1][1]["byte_start"] : docs[1][1]["byte_end"]]
        == docs[1][0]["text"].encode()
    )
    assert docs[2][0]["content_sha256"] != docs[1][0]["content_sha256"]


def test_raw_spans_count_physical_lf_lines_not_unicode_separators() -> None:
    source = next(item for item in _project().sources if item.id == "sample_docs")
    raw = "first\fspace\vtab\u2028paragraph\u2029next\nsecond\n".encode()
    document, span = _records_for_file(raw, "document.txt", source, "a" * 64)[0]
    assert document["source_location"] == "document.txt#lines=1-2"
    assert (span["line_start"], span["line_end"]) == (1, 2)
    assert (span["byte_start"], span["byte_end"]) == (0, len(raw))

    lone_cr = b"first\rsecond"
    normalized, physical = _records_for_file(lone_cr, "document.txt", source, "c" * 64)[
        0
    ]
    assert normalized["text"] == "first\nsecond"
    assert (physical["line_start"], physical["line_end"]) == (1, 1)
    assert (physical["byte_start"], physical["byte_end"]) == (0, len(lone_cr))

    markdown = b"# Heading\nline \f and \v stay inside one raw line\n## Next\nbody\n"
    sections = _records_for_file(markdown, "document.md", source, "b" * 64)
    assert [item[0]["section_path"] for item in sections] == [
        ["Heading"],
        ["Heading", "Next"],
    ]
    assert [(item[1]["line_start"], item[1]["line_end"]) for item in sections] == [
        (1, 2),
        (3, 4),
    ]
    assert sections[-1][1]["byte_end"] == len(markdown)


def test_normalized_duplicates_preserve_distinct_raw_origins() -> None:
    source = SourceDeclaration.model_validate(
        {
            "schema_version": 1,
            "id": "docs",
            "kind": "local",
            "canonical_uri": "fixture:docs",
            "revision": "v1",
            "license": "MIT",
            "redistribution": "redistributable",
            "domains": ["technical_docs"],
            "document_kinds": ["plain_text"],
            "source_family": "docs",
            "acquisition": {
                "files": [{"path": "any.txt", "name": "any.txt"}],
                "max_bytes": 1000,
            },
        }
    )
    composed, _ = _records_for_file("café\r\n".encode(), "any.txt", source, "a" * 64)[0]
    decomposed, _ = _records_for_file(
        "cafe\u0301\n".encode(), "any.txt", source, "b" * 64
    )[0]
    assert composed["text"] == decomposed["text"] == "café\n"
    assert composed["content_sha256"] == decomposed["content_sha256"]
    assert composed["raw_content_sha256"] != decomposed["raw_content_sha256"]
    assert composed["document_id"] != decomposed["document_id"]


def test_prospective_hf_rows_keep_provenance_and_exclude_private_material(
    tmp_path: Path,
) -> None:
    source = neutral_corpus(tmp_path).sources[0]
    safe = {
        "text": "A public tutorial explains operating system scheduling and reliable "
        "deployments with concrete examples across several paragraphs. " * 2,
        "url": "https://example.org/tutorial",
        "id": "warc-record-1",
        "_klab_source": {
            "source_shard_path": "sample/10BT/000_00000.parquet",
            "source_row_index": 882,
            "source_row_sha256": "a" * 64,
        },
    }
    secret = {
        **safe,
        "text": safe["text"] + " AKIA" + "A" * 16,
        "id": "warc-record-2",
    }
    excluded_platform = {
        **safe,
        "url": "https://stackoverflow.com/questions/123",
        "id": "warc-record-3",
    }
    rejected: list[dict[str, object]] = []
    raw = b"".join(
        json.dumps(row).encode() + b"\n" for row in (safe, secret, excluded_platform)
    )
    name = "sample/10BT/000_00000.parquet.sample.jsonl"
    pairs = _records_for_file(
        raw,
        name,
        source,
        "b" * 64,
        file_rights=resolve_file_rights(
            source.rights, name, raw, prospective_private_research=True
        ),
        rejected_records=rejected,
    )
    assert len(pairs) == 1
    document, evidence = pairs[0]
    assert document["metadata"]["url"] == safe["url"]
    assert document["metadata"]["id"] == safe["id"]
    assert document["metadata"]["source_row_index"] == 882
    assert document["source_location"].endswith("#row=1")
    assert evidence["raw_sha256"] != document["content_sha256"]
    assert [row["reason"] for row in rejected] == [
        "secret_or_private_identifier",
        "excluded_platform_terms_or_private_paste",
    ]


def test_scholarly_abstracts_and_full_papers_keep_distinct_kinds_and_corpus_ids(
    tmp_path: Path,
) -> None:
    project = neutral_corpus(tmp_path)
    for source_id, upstream, expected_kind in (
        ("pes2o_fixture_abstract", "s2ag/train", "abstract"),
        ("pes2o_fixture_paper", "s2orc/train", "paper"),
    ):
        source = next(item for item in project.sources if item.id == source_id)
        raw = b"".join(
            json.dumps(
                {
                    "id": "paper-123",
                    "source": origin,
                    "text": "A scholarly discussion of operating systems and rigorous evidence. "
                    * 3,
                    "_klab_source": {"source_row_index": index},
                }
            ).encode()
            + b"\n"
            for index, origin in enumerate((upstream, "unreviewed/external"), 1)
        )
        name = source.acquisition.bounded_shards[0].path + ".sample.jsonl"
        rejected: list[dict[str, object]] = []
        rows = _records_for_file(
            raw,
            name,
            source,
            "d" * 64,
            file_rights=resolve_file_rights(
                source.rights, name, raw, prospective_private_research=True
            ),
            rejected_records=rejected,
        )
        assert len(rows) == 1
        assert rows[0][0]["document_kind"] == expected_kind
        assert rows[0][0]["metadata"]["id"] == "paper-123"
        assert rows[0][0]["metadata"]["source"] == upstream
        assert [item["reason"] for item in rejected] == ["unreviewed_academic_origin"]


def test_wikimedia_rows_keep_page_revision_not_jsonl_envelope(tmp_path: Path) -> None:
    project = neutral_corpus(tmp_path)
    source = next(item for item in project.sources if item.id == "wiki_fixture")
    raw = json.dumps(
        {
            "text": "Compiler construction begins with tokens and grammar definitions.",
            "_klab_source": {
                "page_title": "Compiler Construction",
                "page_id": "1234",
                "revision_id": "9876",
                "revision_timestamp": "2026-09-01T00:00:00Z",
                "page_uri": "https://en.wikibooks.org/?curid=1234",
            },
        }
    ).encode()
    name = "enwikibooks-20260901-pages-articles-multistream.xml.bz2.sample.jsonl"
    documents = _records_for_file(
        raw + b"\n",
        name,
        source,
        "c" * 64,
        file_rights=resolve_file_rights(
            source.rights, name, raw + b"\n", prospective_private_research=True
        ),
    )
    assert len(documents) == 1
    item, evidence = documents[0]
    assert item["title"] == "Compiler Construction"
    assert item["metadata"]["revision_id"] == "9876"
    assert item["metadata"]["page_uri"].endswith("curid=1234")
    assert item["source_location"].endswith("#row=1")
    assert evidence["line_start"] == 1


def test_transform_parameter_changes_produce_distinct_builds(tmp_path: Path) -> None:
    original = _project()
    acquire(original, tmp_path)
    old = build(original, tmp_path, offline=True)
    changed = tuple(
        stage.model_copy(update={"parameters": {"recognizer_revision": "other"}})
        if stage.kind == "lexical_candidates"
        else stage
        for stage in original.transforms
    )
    revised = original.model_copy(update={"transforms": changed})
    acquire(revised, tmp_path)
    new = build(revised, tmp_path, offline=True)
    assert new != old
    assert (old / "build.json").exists()
    assert build(revised, tmp_path, offline=True) == new


def test_split_requires_complete_consistent_family_assignments() -> None:
    policy = {
        "unit": "source_document_family",
        "assignments": {"docs": "train", "code": "test", "world": "validation"},
    }
    assert _split({"source_family_ids": ["docs"]}, policy) == "train"
    assert _split({"scenario_family_id": "world"}, policy) == "validation"
    with pytest.raises(ValueError, match="conflicting"):
        _split({"source_family_ids": ["docs", "code"]}, policy)
    with pytest.raises(ValueError, match="missing"):
        _split({"source_family_ids": ["unassigned"]}, policy)


def test_build_freeze_replay_spans_semantics_and_tamper(tmp_path: Path) -> None:
    project = _project()
    acquire(project, tmp_path)
    first = build(project, tmp_path, offline=True)
    assert build(project, tmp_path, offline=True) == first
    published = freeze(first, tmp_path)
    assert freeze(first, tmp_path) == published
    manifest = verify_release(published)
    docs = [
        json.loads(line)
        for line in (published / "documents.jsonl").read_text().splitlines()
    ]
    assert any(row["section_path"] for row in docs)
    item = next(row for row in docs if "technical_docs" in row["domains"])
    trace = lineage(published, item["document_id"])
    assert trace["parents"][0]["source"]["license"] == "MIT"
    assert trace["parents"][0]["span"]["raw_sha256"]
    chat = next(
        row
        for row in review(published, kind="chat", limit=100)["records"]
        if row["verification"]["status"] == "oracle_verified"
    )
    assert lineage(published, chat["record_id"])["scenario"]["oracle_answer"] == ".py"
    assert sample(published, domain="systems_scenarios", limit=1)["records"][0][
        "domains"
    ] == ["systems_scenarios"]
    semantic = [
        json.loads(line)
        for line in (published / "semantic/candidates.jsonl").read_text().splitlines()
    ]
    assert semantic
    assert all(
        next(d for d in docs if d["document_id"] == row["evidence_document_id"])[
            "text"
        ][slice(*row["evidence_span"])]
        == row["evidence_passage"]
        for row in semantic
    )
    lexical = [
        json.loads(line)
        for line in (published / "lexical/candidates.jsonl").read_text().splitlines()
    ]
    assert any(
        row["occurrences"] >= 1 and row["document_ids"] and row["domain_counts"]
        for row in lexical
    )
    candidate = lexical[0]
    assert {
        parent["document"]["document_id"]
        for parent in lineage(published, candidate["record_id"])["parents"]
    } == set(candidate["document_ids"])
    assert (
        lineage(published, semantic[0]["record_id"])["parents"][0]["document"][
            "document_id"
        ]
        == semantic[0]["evidence_document_id"]
    )
    artifact = published / "semantic/candidates.jsonl"
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="tampered"):
        verify_release(published)
    assert manifest["release_id"] == published.name


def test_cross_split_identical_text_is_rejected(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe"
    shutil.copytree(PROJECT.parent, recipe)
    duplicate = yaml.safe_load((recipe / "sources/docs.yaml").read_text())
    duplicate["id"] = "copied_docs"
    duplicate["canonical_uri"] = "fixture:copied_docs"
    duplicate["source_family"] = "sample_code_validation"
    (recipe / "sources/copied.yaml").write_text(yaml.safe_dump(duplicate))
    corpus_path = recipe / "corpus.yaml"
    corpus = yaml.safe_load(corpus_path.read_text())
    corpus["sources"] = sorted([*corpus["sources"], "sources/copied.yaml"])
    corpus_path.write_text(yaml.safe_dump(corpus))
    project = load_project(corpus_path)
    acquire(project, tmp_path / "work")
    with pytest.raises(ValueError, match="cross-split exact text overlap"):
        build(project, tmp_path / "work", offline=True)
    diagnostics = list(
        (tmp_path / "work/corpora/devmind-sample-v0/builds/diagnostics").glob("*.json")
    )
    assert len(diagnostics) == 1
    audit = json.loads(diagnostics[0].read_text())
    assert any(
        {"train", "validation"} == set(group["splits"]) and len(group["origins"]) > 1
        for group in audit["duplicates"]
    )


def test_empty_selected_training_view_cannot_replace_previous_release(
    tmp_path: Path,
) -> None:
    project = _project()
    acquire(project, tmp_path)
    published = freeze(build(project, tmp_path, offline=True), tmp_path)
    original_digest = (published / "manifest.json").read_bytes()
    revised = project.model_copy(
        update={
            "transforms": tuple(
                stage
                for stage in project.transforms
                if stage.kind not in {"chat_sft", "tool_episode"}
            )
        }
    )
    acquire(revised, tmp_path)
    candidate = build(revised, tmp_path, offline=True)
    with pytest.raises(ValueError, match="selected chat/train training view is empty"):
        freeze(candidate, tmp_path)
    assert verify_release(published)["release_id"] == published.name
    assert (published / "manifest.json").read_bytes() == original_digest
