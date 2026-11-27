"""Corpus normalization, ancestry, deduplication and immutable release checks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from corpus_fixtures import (
    PLAIN_OPTIONS,
    PRIVATE_RESEARCH_POLICY,
    STRICT_POLICY,
    fixture_rights,
    neutral_corpus,
)

from klab.corpus.acquisition import acquire
from klab.corpus.document_normalization import records_for_file
from klab.corpus.pipeline import (
    build,
    uses_disk_lm_builder,
)
from klab.corpus.project import DocumentPolicy, SourceDeclaration, load_project
from klab.corpus.release import freeze, lineage, review, sample, verify_release
from klab.corpus.rights import resolve_file_rights
from klab.corpus.split_assignments import assign_split

STRICT_DOCS = DocumentPolicy.model_validate(STRICT_POLICY["documents"])
PRIVATE_DOCS = DocumentPolicy.model_validate(PRIVATE_RESEARCH_POLICY["documents"])
PROJECT = (
    Path(__file__).resolve().parents[1] / "tests/fixtures/corpus-sample/corpus.yaml"
)


def _project():
    return load_project(PROJECT)


def test_markdown_ancestry_fences_and_raw_evidence() -> None:
    raw = b"Introduction.\n# Alpha\n- preserve list\n```md\n# not a heading\n```\n## Beta\nkey = value\n"
    source = SourceDeclaration.model_validate(
        {
            "id": "docs",
            "kind": "local",
            "canonical_uri": "fixture:docs",
            "revision": "v1",
            "license": "MIT",
            "rights": fixture_rights(),
            "domains": ["technical_docs"],
            "document_kinds": ["markdown"],
            "source_family": "docs",
            "acquisition": {
                "files": [{"path": "any.md", "name": "any.md"}],
                "max_bytes": 1000,
            },
        }
    )
    docs = records_for_file(
        raw,
        "any.md",
        source,
        "a" * 64,
        normalizer_options=PLAIN_OPTIONS,
        document_policy=STRICT_DOCS,
    )
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
    # Declared whole-file Markdown is one document; the default leaves the
    # source declaration's serialized identity unchanged.
    assert "markdown_documents" not in source.model_dump(mode="json")
    whole = SourceDeclaration.model_validate(
        {**source.model_dump(mode="json"), "markdown_documents": "files"}
    )
    [(document, span)] = records_for_file(
        raw,
        "any.md",
        whole,
        "a" * 64,
        normalizer_options=PLAIN_OPTIONS,
        document_policy=STRICT_DOCS,
    )
    assert document["text"] == raw.decode() and document["section_path"] == []
    assert (span["line_start"], span["line_end"]) == (1, 8)


def test_raw_spans_count_physical_lf_lines_not_unicode_separators() -> None:
    source = next(item for item in _project().sources if item.id == "sample_docs")
    raw = "first\fspace\vtab\u2028paragraph\u2029next\nsecond\n".encode()
    document, span = records_for_file(
        raw,
        "document.txt",
        source,
        "a" * 64,
        normalizer_options=PLAIN_OPTIONS,
        document_policy=STRICT_DOCS,
    )[0]
    assert document["source_location"] == "document.txt#lines=1-2"
    assert (span["line_start"], span["line_end"]) == (1, 2)
    assert (span["byte_start"], span["byte_end"]) == (0, len(raw))

    lone_cr = b"first\rsecond"
    normalized, physical = records_for_file(
        lone_cr,
        "document.txt",
        source,
        "c" * 64,
        normalizer_options=PLAIN_OPTIONS,
        document_policy=STRICT_DOCS,
    )[0]
    assert normalized["text"] == "first\nsecond"
    assert (physical["line_start"], physical["line_end"]) == (1, 1)
    assert (physical["byte_start"], physical["byte_end"]) == (0, len(lone_cr))

    markdown = b"# Heading\nline \f and \v stay inside one raw line\n## Next\nbody\n"
    sections = records_for_file(
        markdown,
        "document.md",
        source,
        "b" * 64,
        normalizer_options=PLAIN_OPTIONS,
        document_policy=STRICT_DOCS,
    )
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
            "id": "docs",
            "kind": "local",
            "canonical_uri": "fixture:docs",
            "revision": "v1",
            "license": "MIT",
            "rights": fixture_rights(),
            "domains": ["technical_docs"],
            "document_kinds": ["plain_text"],
            "source_family": "docs",
            "acquisition": {
                "files": [{"path": "any.txt", "name": "any.txt"}],
                "max_bytes": 1000,
            },
        }
    )
    composed, _ = records_for_file(
        "café\r\n".encode(),
        "any.txt",
        source,
        "a" * 64,
        normalizer_options=PLAIN_OPTIONS,
        document_policy=STRICT_DOCS,
    )[0]
    decomposed, _ = records_for_file(
        "cafe\u0301\n".encode(),
        "any.txt",
        source,
        "b" * 64,
        normalizer_options=PLAIN_OPTIONS,
        document_policy=STRICT_DOCS,
    )[0]
    assert composed["text"] == decomposed["text"] == "café\n"
    assert composed["content_sha256"] == decomposed["content_sha256"]
    assert composed["raw_content_sha256"] != decomposed["raw_content_sha256"]
    assert composed["document_id"] != decomposed["document_id"]


def test_prospective_hf_rows_keep_provenance_and_exclude_private_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from recipe_fixtures import Lab

    from klab.corpus.jsonl_records import iter_rows
    from klab.corpus.prepared_shards import normalized_store_root, prepare_file
    from klab.corpus.progress import BuildProgress
    from klab.training.manifest import sha256_file

    lab = Lab(tmp_path, monkeypatch, capsys)
    source = neutral_corpus(lab.work).sources[0]
    safe = {
        "text": "A public tutorial explains operating system scheduling and reliable "
        "deployments with concrete examples across several paragraphs. "
        * 2
        + "\n\nReviewers explain which changes affect this build.\n\n"
        "It mentions @internal.example and the unformatted number 2025550100.",
        "url": "https://example.org/tutorial",
        "id": "warc-record-1",
        "score": 0.8,
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
    empty = {**safe, "text": ""}
    email = {**safe, "text": safe["text"] + " Email reader@example.test.", "score": 0.2}
    phone = {
        **safe,
        "text": safe["text"] + " Telephone (202) 555-0100.",
        "score": "unknown",
    }
    international = {
        "text": "The office can help with this form and you could call +44 20 7946 0958. You can find details about the form with the application instructions.",
        "url": "https://example.org/contact",
    }
    prose = [
        ("Les enfants sont dans cette maison avec leur mère depuis une heure.", "fr"),
        (
            "Los vecinos han salido desde esta plaza para hablar con ellos cuando llegue el tren.",
            "es",
        ),
        ("Das ist der Weg für die Kinder, die sich bei dem alten Haus treffen.", "de"),
        (
            "Uma criança ainda não sabe como essa história dos avós também pode mudar.",
            "pt",
        ),
        ("Ciao, grazie per il viaggio di domani.", "it"),
    ]
    raw = b"".join(
        json.dumps(row).encode() + b"\n"
        for row in (
            safe,
            secret,
            empty,
            excluded_platform,
            email,
            phone,
            safe,
            international,
            *(
                {"text": " ".join([text] * 3), "url": f"https://example.org/{language}"}
                for text, language in prose
            ),
        )
    )
    raw_path = lab.work / "sample.jsonl"
    raw_path.write_bytes(raw)
    name = "sample/10BT/000_00000.parquet.sample.jsonl"
    normalized = normalized_store_root(lab.work)
    partition = prepare_file(
        prepared_root=normalized,
        source=source,
        file={"path": name, "sha256": sha256_file(raw_path), "size": len(raw)},
        snapshot_sha="b" * 64,
        source_path=raw_path,
        decision=resolve_file_rights(
            source.rights, name, raw, prospective_private_research=True
        ),
        progress=BuildProgress(lab.work / "progress.jsonl", "privacy-fixture"),
        normalizer_options=PLAIN_OPTIONS,
        document_policy=PRIVATE_DOCS,
    )
    docs, spans, rejected = (
        partition / member for member in ("docs.jsonl", "spans.jsonl", "rejected.jsonl")
    )
    document, email_doc, phone_doc, *other_docs = iter_rows(docs)
    evidence, *_ = iter_rows(spans)
    assert document["metadata"]["url"] == safe["url"]
    assert document["metadata"]["id"] == safe["id"]
    assert document["metadata"]["source_row_index"] == 882
    assert document["source_location"].endswith("#row=1")
    assert evidence["raw_sha256"] != document["content_sha256"]
    assert [row["reason"] for row in iter_rows(rejected)] == [
        "secret_or_private_identifier",
        "empty_or_nontext",
        "excluded_platform_terms_or_private_paste",
    ]
    annotate = (
        "corpus",
        "annotate-normalized",
        partition.relative_to(normalized).as_posix(),
        "--kind",
        "privacy-exclusions",
        "--limit",
        "1",
    )
    open_file = Path.open
    forbidden = {raw_path, spans, rejected}

    def check_open(path, *args, **kwargs):
        if path in forbidden:
            pytest.fail("privacy inspection reopened a body file")
        return open_file(path, *args, **kwargs)

    candidates = (
        "corpus",
        "annotate-normalized",
        partition.relative_to(normalized).as_posix(),
        "--kind",
        "privacy-candidates",
        "--limit",
        "2",
    )
    with monkeypatch.context() as reading:
        reading.setattr(Path, "open", check_open)
        flags = lab.json(*candidates)
        forbidden.add(docs)
        next_page = lab.json(
            *candidates[:-1], "1", "--offset", str(flags["next_offset"])
        )
    assert flags["totals"] == {"any": 2, "email_like": 1, "phone_like": 1}
    assert flags["payload"]["bytes"] == 33 * (3 + len(other_docs))
    assert flags["rows"] == [
        {"document_id": document["document_id"], "candidates": []},
        {"document_id": email_doc["document_id"], "candidates": ["email_like"]},
    ]
    assert next_page["reused"] and next_page["payload"] == flags["payload"]
    assert next_page["rows"] == [
        {"document_id": phone_doc["document_id"], "candidates": ["phone_like"]}
    ]

    parent = partition.relative_to(normalized).as_posix()
    detected = ("corpus", "annotate-normalized", parent, "--kind", "detected-language")
    without_memo = lab.json(*detected)
    with_memo = lab.json(*detected, "--memoize-content")
    assert with_memo["memoized_documents"] == 1
    assert with_memo["payload"]["sha256"] == without_memo["payload"]["sha256"]
    labeled_docs = [document, email_doc, phone_doc, *other_docs]

    def labels(kind, expected, **options):
        return {
            "kind": kind,
            "partition": parent,
            "rows": [
                {"document_id": doc["document_id"], "expected": value}
                for doc, value in zip(labeled_docs, expected, strict=True)
            ],
            **options,
        }

    label_file = tmp_path / "labels.json"
    samples = [
        labels("detected-language", ["en"] * 5 + [language for _, language in prose]),
        labels(
            "privacy-candidates",
            [
                [],
                ["email_like"],
                ["phone_like"],
                [],
                ["phone_like"],
                *([] for _ in prose),
            ],
        ),
        labels("template-lines", [True] * 4 + [False] * (1 + len(prose))),
        labels(
            "source-quality",
            [True, True, False, True, False, *([False] * len(prose))],
            quality_rule={"field": "score", "minimum": 0.5, "missing": "drop"},
        ),
    ]
    label_file.write_text(json.dumps({"annotations": samples}))
    audit = lab.json("corpus", "audit-evidence", str(label_file))
    language, privacy, template, quality = [
        item["metrics"] for item in audit["annotations"]
    ]
    assert language["coverage"] == 0.9 and language["accuracy_when_detected"] == 1
    assert (
        privacy["email_like"]["precision"] == 1
        and privacy["phone_like"]["recall"] == 0.5
    )
    assert template["precision"] == template["recall"] == 1
    assert quality["recall"] == 2 / 3 and quality["field_states"]["invalid"] == 1
    template_args = (
        "corpus",
        "annotate-normalized",
        parent,
        "--kind",
        "template-lines",
    )
    with monkeypatch.context() as reading:
        reading.setattr(Path, "open", check_open)
        template_page = lab.json(*template_args, "--limit", "1")
    assert template_page["reused"] and template_page["rows"][0]["template_lines"] == 1
    alternate = lab.json(*template_args, "--template-min-support", "5")
    assert alternate["computation_key"] != template_page["computation_key"]
    assert alternate["totals"]["marked_documents"] == 0
    samples[0]["rows"][0]["document_id"] = "0" * 64
    label_file.write_text(json.dumps({"annotations": samples}))
    with pytest.raises(ValueError, match="labels do not match verified document IDs"):
        lab.run("corpus", "audit-evidence", str(label_file))

    forbidden.remove(rejected)
    with monkeypatch.context() as reading:
        reading.setattr(Path, "open", check_open)
        first = lab.json(*annotate)
        forbidden.add(rejected)
        second = lab.json(*annotate, "--offset", str(first["next_offset"]))
    assert first["input"]["exclusions"] == "research_privacy"
    assert first["input"]["rejected_records"] == 3
    assert first["findings"] == 2 and first["payload"]["bytes"] == 34
    assert first["rows"] == [
        {
            "rejection_ordinal": 1,
            "snapshot_row": 2,
            "reason": "secret_or_private_identifier",
        }
    ]
    assert second["reused"] and second["payload"] == first["payload"]
    assert second["next_offset"] is None
    assert second["rows"] == [
        {
            "rejection_ordinal": 3,
            "snapshot_row": 4,
            "reason": "excluded_platform_terms_or_private_paste",
        }
    ]
    rejected.write_bytes(
        rejected.read_bytes().replace(
            b"secret_or_private_identifier", b"x" * len("secret_or_private_identifier")
        )
    )
    with pytest.raises(ValueError, match="JSONL source SHA-256 mismatch"):
        lab.run(*annotate)


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
        rows = records_for_file(
            raw,
            name,
            source,
            "d" * 64,
            file_rights=resolve_file_rights(
                source.rights, name, raw, prospective_private_research=True
            ),
            rejected_records=rejected,
            normalizer_options=PLAIN_OPTIONS,
            document_policy=PRIVATE_DOCS,
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
    documents = records_for_file(
        raw + b"\n",
        name,
        source,
        "c" * 64,
        file_rights=resolve_file_rights(
            source.rights, name, raw + b"\n", prospective_private_research=True
        ),
        normalizer_options=PLAIN_OPTIONS,
        document_policy=PRIVATE_DOCS,
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
    assert assign_split({"source_family_ids": ["docs"]}, policy) == "train"
    assert assign_split({"scenario_family_id": "world"}, policy) == "validation"
    with pytest.raises(ValueError, match="conflicting"):
        assign_split({"source_family_ids": ["docs", "code"]}, policy)
    with pytest.raises(ValueError, match="missing"):
        assign_split({"source_family_ids": ["unassigned"]}, policy)


def test_build_freeze_replay_spans_semantics_and_tamper(tmp_path: Path) -> None:
    project = _project()
    acquire(project, tmp_path)
    first = build(project, tmp_path, offline=True)
    assert build(project, tmp_path, offline=True) == first
    published = freeze(first, tmp_path, share_stage_artifacts=True)
    assert freeze(first, tmp_path, share_stage_artifacts=True) == published
    manifest = verify_release(published)
    assert all(
        entry.get("storage") == "shared_stage"
        for name, entry in manifest["files"].items()
        if name.startswith("stages/")
    )
    declared = project.release.normalizer.model_dump(mode="json")
    report = json.loads((published / "report.json").read_text())
    assert report["normalizer"] == declared
    ledger = {row["source_id"]: row for row in report["source_ledger"]}
    assert set(ledger) == {source.id for source in project.sources}
    assert {k: v["retained_documents"] for k, v in ledger.items()} == {
        k: report["source_counts"].get(k, 0) for k in ledger
    }
    assert "## Sources" in (published / "report.md").read_text()
    build_record = json.loads((first / "build.json").read_text())
    assert build_record["identity"]["release"]["normalizer"] == declared
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
        (tmp_path / "work/corpora/sample-corpus-v0/builds/diagnostics").glob("*.json")
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


def _disk_builder_project(strategy="auto", **release_overrides):
    from types import SimpleNamespace as N

    release = N(
        fraction=None,
        lm=N(selected=True),
        chat=N(selected=False),
        policy=N(build=N(strategy=strategy)),
    )
    for name, value in release_overrides.items():
        setattr(release, name, value)
    sources = [N(kind="huggingface_dataset"), N(kind="git")]
    return N(release=release, sources=sources), [N(kind="lm_text")]


def test_an_lm_only_release_uses_the_disk_builder_without_reviewed_admission():
    # Large corpora must stay memory-bounded without a reviewed admission.
    assert uses_disk_lm_builder(*_disk_builder_project())


def test_a_fractional_release_keeps_the_in_memory_builder_under_auto():
    assert not uses_disk_lm_builder(*_disk_builder_project(fraction=0.5))


def test_the_build_strategy_can_force_either_builder():
    assert not uses_disk_lm_builder(*_disk_builder_project(strategy="memory"))
    with pytest.raises(ValueError, match="build.strategy is disk"):
        uses_disk_lm_builder(*_disk_builder_project(strategy="disk", fraction=0.5))


def test_source_ledger_names_a_source_whose_files_were_all_set_aside() -> None:
    from types import SimpleNamespace

    from klab.corpus.build_outputs import Tally, source_ledger, source_ledger_markdown

    project = SimpleNamespace(
        sources=[SimpleNamespace(id="docs"), SimpleNamespace(id="code_rust")]
    )
    tally = Tally()
    for index in range(3):
        tally.add(
            {
                "text": "hello",
                "split": "train",
                "source_id": "docs",
                "document_kind": "prose",
                "domains": [],
            }
        )
    tally.add_dropped({"source_id": "docs", "drop_reason": "duplicate"})
    rights_files = [
        {"role": "document", "source_id": "docs"},
        {"role": "document", "source_id": "code_rust"},
        {"role": "document", "source_id": "code_rust"},
        {"role": "transform_input", "source_id": "code_rust"},
    ]
    rejected = [
        {"source_id": "code_rust", "path": "a.rs", "reason": "rights ineligible: x"},
        {"source_id": "code_rust", "path": "b.rs", "reason": "rights ineligible: x"},
        {"source_id": "docs", "path": "d.md", "row": 4, "reason": "empty_or_nontext"},
    ]
    ledger = {
        row["source_id"]: row
        for row in source_ledger(project, rights_files, rejected, tally)
    }
    assert ledger["code_rust"]["retained_documents"] == 0
    assert ledger["code_rust"]["files_seen"] == 2
    assert ledger["code_rust"]["file_set_aside_share"] == 1.0
    assert ledger["code_rust"]["set_aside_reasons"] == [
        {"level": "files", "reason": "rights ineligible: x", "count": 2}
    ]
    assert ledger["docs"]["rows_set_aside"] == 1
    assert ledger["docs"]["dedup_dropped"] == {"duplicate": 1}
    assert ledger["docs"]["documents_built"] == 4
    table = source_ledger_markdown(list(ledger.values()))
    assert (
        "WARNING: 1 declared source(s) have no retained documents: code_rust" in table
    )
    assert "NONE RETAINED" in table
