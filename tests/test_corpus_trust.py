"""Corpus trust gates: cross-split overlap, tokenizer selection, data coverage."""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from corpus_fixtures import write_prose_project
from lab_fixtures import write_smoke_inputs

from klab.canonical import canonical_digest
from klab.cli.main import main
from klab.config.loading import load_config
from klab.corpus.acquisition import acquire
from klab.corpus.pipeline import build
from klab.corpus.project import load_project
from klab.corpus.release import freeze
from klab.corpus.tokenizer_bakeoff import (
    VOCABS,
    bakeoff,
    load_declaration,
    verify_selection,
)
from klab.data.packing import prepare_data
from klab.data.tokenizer import load_tokenizer
from klab.training.manifest import sha256_file

SAMPLE = Path(__file__).resolve().parent / "fixtures" / "corpus-sample"
_PLANTED_PASSAGE = (
    "Record identifiers bind each retained answer to its original document and split"
)


def _klab(monkeypatch: pytest.MonkeyPatch, capsys, *argv: str) -> dict:
    monkeypatch.setattr(sys, "argv", ["klab", *argv])
    main()
    return json.loads(capsys.readouterr().out)


def _freeze(project: Path, work: Path) -> Path:
    loaded = load_project(project / "corpus.yaml")
    acquire(loaded, work)
    return freeze(build(loaded, work, offline=True), work)


# Near-duplicates: a lexical near-copy across train/validation must be reported.


@pytest.fixture
def planted_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The sample corpus with its validation file replaced by a near-copy of a
    training section (one word changed)."""
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    project = tmp_path / "project"
    shutil.copytree(SAMPLE, project)
    guide = (project / "sources/fixtures/guide.md").read_text(encoding="utf-8")
    section = guide[guide.index("## Inspecting lineage") :]
    assert section.rstrip().endswith("generated answer.")
    section = section.rstrip() + f"\n\n{_PLANTED_PASSAGE}.\n"
    (project / "sources/fixtures/guide.md").write_text(
        guide[: guide.index("## Inspecting lineage")] + section, encoding="utf-8"
    )
    (project / "sources/fixtures/checks.py").write_text(
        section.replace("generated answer.", "generated output."), encoding="utf-8"
    )
    return _freeze(project, tmp_path / "work")


def test_near_duplicates_reports_a_planted_cross_split_pair(
    planted_release: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    report = _klab(
        monkeypatch, capsys, "corpus", "near-duplicates", str(planted_release)
    )
    documents = [
        json.loads(line)
        for line in (planted_release / "documents.jsonl").read_text().splitlines()
    ]
    planted = {
        doc["document_id"]
        for doc in documents
        if doc["source_id"] == "sample_code"
        or doc["text"].startswith("## Inspecting lineage")
    }
    assert report["release_id"] == planted_release.name
    assert report["document_count"] == len(documents) > len(planted) == 2
    # Every document is accounted for, by split and source.
    covered = {(row["split"], row["source_id"]): row for row in report["coverage"]}
    for split in {doc["split"] for doc in documents}:
        in_split = [doc for doc in documents if doc["split"] == split]
        assert sum(
            row["documents"] for key, row in covered.items() if key[0] == split
        ) == len(in_split)
        assert sum(
            row["text_bytes"] for key, row in covered.items() if key[0] == split
        ) == sum(len(doc["text"].encode()) for doc in in_split)
    # Exactly the planted pair: unrelated documents in other splits are not flagged.
    [pair] = report["candidates"]
    assert {pair["left_document_id"], pair["right_document_id"]} == planted
    assert {pair["left_split"], pair["right_split"]} == {"train", "validation"}
    assert pair["identical_content_sha256"] is False
    assert pair["jaccard"] >= report["parameters"]["min_jaccard"]
    assert report["candidate_count"] == 1
    assert report["candidates_truncated"] is False


def test_template_aware_near_duplicates_keeps_the_planted_pair_on_the_cli(
    planted_release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    strict = _klab(
        monkeypatch, capsys, "corpus", "near-duplicates", str(planted_release)
    )
    aware = _klab(
        monkeypatch,
        capsys,
        "corpus",
        "near-duplicates",
        str(planted_release),
        "--template-aware",
        "--retain-evidence",
        "--scratch-dir",
        str(tmp_path / "scratch"),
        "--workers",
        "2",
        "--shard-bytes",
        "2048",
    )
    assert aware["parameters"]["template_aware"] is True
    [pair] = aware["candidates"]
    assert pair["class"] == "leak" and pair["content_jaccard"] >= 0.65
    assert aware["class_counts"]["template_only"] == 0
    assert any((tmp_path / "scratch" / "near-duplicates").rglob("evidence.bin"))
    [same] = strict["candidates"]
    assert (pair["left_document_id"], pair["right_document_id"]) == (
        same["left_document_id"],
        same["right_document_id"],
    )
    passage = _klab(
        monkeypatch,
        capsys,
        "corpus",
        "near-duplicates",
        str(planted_release),
        "--passage-words",
        "32",
        "--template-aware",
        "--list-template-pairs",
        "--workers",
        "1",
    )
    assert (
        passage["unit"] == "passage"
        and passage["passage_count"] > aware["document_count"]
    )
    texts = {
        doc["document_id"]: doc["text"]
        for line in (planted_release / "documents.jsonl").read_bytes().splitlines()
        for doc in [json.loads(line)]
    }
    duplicated = next(
        item
        for item in passage["candidates"]
        if item["identical_content_sha256"]
        and {item[side]["document_id"] for side in ("left_passage", "right_passage")}
        == {pair["left_document_id"], pair["right_document_id"]}
    )
    left, right = [
        texts[span["document_id"]][span["start"] : span["end"]]
        for side in ("left_passage", "right_passage")
        for span in [duplicated[side]]
    ]
    assert left == right == _PLANTED_PASSAGE

    from types import SimpleNamespace

    from klab.preflight import FAIL, check_near_duplicates

    saved = tmp_path / "passage-report.json"
    saved.write_text(json.dumps(passage))
    expected_ids = []
    for parent in (pair["left_document_id"], pair["right_document_id"]):
        start = texts[parent].index(_PLANTED_PASSAGE)
        expected_ids.append(
            canonical_digest([parent, start, start + len(_PLANTED_PASSAGE)])
        )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "screens": [
                    {
                        "report": saved.name,
                        "sha256": sha256_file(saved),
                        "pairs": [
                            {"units": expected_ids, "expected": "candidate"},
                        ],
                    }
                ]
            }
        )
    )
    audited = _klab(monkeypatch, capsys, "corpus", "audit-evidence", str(labels))
    assert audited["screens"][0]["metrics"]["candidate"]["recall"] == 1
    check = check_near_duplicates(
        SimpleNamespace(near_duplicates=saved, near_duplicate_decisions=saved)
    )
    assert check.status == FAIL


def test_decision_record_must_decide_every_pair_of_this_report() -> None:
    from klab.corpus.near_duplicates import check_decisions

    report = {
        "release_id": "r1",
        "candidates": [
            {"left_document_id": "a", "right_document_id": "b"},
            {"left_document_id": "c", "right_document_id": "d"},
        ],
    }
    record = {
        "schema_version": 1,
        "report_sha256": "h",
        "release_id": "r1",
        "accepted_by": "reviewer",
        "accepted_on": "2026-01-01",
        "decisions": [
            {"documents": ["b", "a"], "decision": "dropped"},
            {"documents": ["c", "d"], "decision": "kept"},
        ],
    }
    result = check_decisions(report, "h", record)
    assert result["undecided_count"] == 0
    assert result["problems"] == ["pair c / d is kept without a reason"]
    record["decisions"].pop()
    record["report_sha256"] = "other"
    result = check_decisions(report, "h", record)
    assert result["undecided"] == [["c", "d"]]
    assert "written for a different report" in result["problems"][0]


def test_near_duplicates_refuses_a_heldout_side_over_its_bound(
    planted_release: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "klab",
            "corpus",
            "near-duplicates",
            str(planted_release),
            "--max-heldout-shingles",
            "1",
        ],
    )
    with pytest.raises(ValueError):
        main()
    assert capsys.readouterr().out == ""


# Tokenizer bakeoff: a selection is reproducible from the release and tamper-evident.


def _words(seed: int, count: int, lexicon_size: int) -> str:
    rng = random.Random(seed)
    lexicon = [
        "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=rng.randint(3, 9)))
        for _ in range(lexicon_size)
    ]
    return " ".join(rng.choices(lexicon, k=count)) + "\n"


@pytest.fixture(scope="module")
def bakeoff_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A prose-only v2 release with enough distinct text for every candidate size."""
    root = tmp_path_factory.mktemp("bakeoff")
    # Train must fill the largest candidate vocabulary; held-out families are small.
    project = write_prose_project(
        root,
        {
            "prose_train": [_words(1, 120_000, 40_000)],
            "prose_validation": [_words(2, 400, 200)],
            "prose_test": [_words(3, 400, 200)],
        },
        corpus_id="bakeoff-fixture",
    )
    declaration = root / "bakeoff.yaml"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("KLAB_WORK_DIR", str(root / "work"))
        release = _freeze(project, root / "work")
        declaration.write_text(
            yaml.safe_dump(
                {
                    "release_path": f"bakeoff-fixture@{release.name}",
                    "vocab_sizes": list(VOCABS),
                    "max_fit_bytes": 268435456,
                    "eval_split": "validation",
                    "eval_max_docs_per_group": 200,
                    "groups": ["prose"],
                    "near_best_ratio": 0.98,
                    "train_token_scope": "all_selected_sources",
                }
            )
        )
        return bakeoff(declaration, root / "bakeoff")


def test_bakeoff_selection_verifies_against_its_release(
    bakeoff_report: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus import tokenizer_bakeoff as bakeoff_module

    report = json.loads(bakeoff_report.read_text(encoding="utf-8"))
    scores = {
        item["vocab_size"]: item["weighted_bytes_per_token"]
        for item in report["candidates"]
    }
    assert sorted(scores) == list(VOCABS)
    chosen = report["selected_vocab_size"]
    # The smallest vocabulary within near_best_ratio of the best held-out score.
    assert chosen == min(
        v for v, s in scores.items() if s >= 0.98 * max(scores.values())
    )
    tokenizer = bakeoff_report.parent / "candidates" / str(chosen) / "tokenizer.json"
    scans = []
    read_rows = bakeoff_module.ReleaseReader.rows

    def track_rows(reader, member):
        scans.append(member)
        return read_rows(reader, member)

    def hash_control(path):
        assert (
            path.name not in {"train.jsonl", "validation.jsonl"}
            or path.parent.name != "lm"
        )
        return sha256_file(path)

    monkeypatch.setattr(bakeoff_module.ReleaseReader, "rows", track_rows)
    monkeypatch.setattr(bakeoff_module, "sha256_file", hash_control)
    assert verify_selection(bakeoff_report) == (
        sha256_file(tokenizer),
        f"{report['identity']['release_id']}:{chosen}",
    )
    assert scans.count("documents.jsonl") == 1


def test_bakeoff_selection_refuses_a_forged_choice(bakeoff_report: Path) -> None:
    original = bakeoff_report.read_bytes()
    report = json.loads(original)
    other = next(v for v in VOCABS if v != report["selected_vocab_size"])
    report["selected_vocab_size"] = other
    report["selected_tokenizer"] = str(
        bakeoff_report.parent / "candidates" / str(other) / "tokenizer.json"
    )
    try:
        bakeoff_report.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(ValueError):
            verify_selection(bakeoff_report)
    finally:
        bakeoff_report.write_bytes(original)


def _declaration(tmp_path: Path, **changes: object) -> Path:
    body = {
        "release_path": "release",
        "vocab_sizes": list(VOCABS),
        "max_fit_bytes": 268435456,
        "eval_split": "validation",
        "eval_max_docs_per_group": 200,
        "groups": ["prose"],
        "near_best_ratio": 0.98,
        "train_token_scope": "grouped_documents",
    } | changes
    path = tmp_path / "declaration.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in body.items() if v is not None}))
    return path


def test_bakeoff_declaration_records_options_and_has_no_version(
    tmp_path: Path, bakeoff_report: Path
) -> None:
    assert load_declaration(_declaration(tmp_path)).train_token_scope == (
        "grouped_documents"
    )
    recorded = json.loads(bakeoff_report.read_text(encoding="utf-8"))["identity"]
    assert recorded["declaration"]["train_token_scope"] == "all_selected_sources"
    assert "schema_version" not in recorded
    assert "schema_version" not in recorded["declaration"]


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"train_token_scope": None},
        {"train_token_scope": "everything"},
    ],
)
def test_bakeoff_declaration_refuses_a_version_or_unnamed_option(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        load_declaration(_declaration(tmp_path, **changes))


# Data coverage and budget: only verified prepared data yields a budget.


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    config_path = write_smoke_inputs(tmp_path)
    config = load_config(config_path)
    data = prepare_data(config, load_tokenizer(config.tokenizer.path))
    return config_path, data.root


def test_coverage_counts_verified_blocks_and_budget_requires_proven_coverage(
    prepared: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    config_path, root = prepared
    seq_len = load_config(config_path).training.seq_len
    report = _klab(
        monkeypatch,
        capsys,
        "data",
        "coverage",
        "--prepared-root",
        str(root),
        "--config",
        str(config_path),
        "--json",
    )
    assert report["config_verified"] is True
    for row in report["splits"].values():
        assert row["blocks"]["complete_blocks"] == (row["packed_tokens"] - 1) // seq_len
        # A bounded synthetic source never certifies full coverage.
        assert row["source_coverage"]["full"] is False

    output = tmp_path / "budget.yaml"
    budget = ["data", "budget", "--prepared-root", str(root), "--config"]
    budget += [str(config_path), "--passes", "2", "--output", str(output), "--json"]
    monkeypatch.setattr(sys, "argv", ["klab", *budget])
    with pytest.raises(SystemExit):
        main()
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    assert not output.exists()

    result = _klab(monkeypatch, capsys, *budget, "--no-require-full")
    train_blocks = report["splits"]["train"]["blocks"]
    assert result["max_tokens"] == 2 * train_blocks["supervised_targets"]
    derived = load_config(output).training
    assert (derived.max_steps, derived.max_tokens) == (
        result["max_steps"],
        result["max_tokens"],
    )


def test_coverage_refuses_a_config_that_did_not_prepare_the_data(
    prepared: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    config_path, root = prepared
    other = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    other["dataset"]["synthetic_seed"] += 1
    other_path = tmp_path / "other.yaml"
    other_path.write_text(yaml.safe_dump(other))
    monkeypatch.setattr(
        sys,
        "argv",
        ["klab", "data", "coverage", "--prepared-root", str(root)]
        + ["--config", str(other_path), "--json"],
    )
    with pytest.raises(SystemExit):
        main()
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_a_default_decision_accepts_a_class_and_exceptions_override_it() -> None:
    from klab.corpus.near_duplicates import check_decisions

    def pair(left: str, right: str) -> dict:
        return {"left_document_id": left, "right_document_id": right}

    report = {
        "release_id": "r",
        "candidates": [pair("a", "b"), pair("c", "d"), pair("e", "f")],
    }
    base = {
        "schema_version": 1,
        "release_id": "r",
        "report_sha256": "h",
        "accepted_by": "reviewer",
        "accepted_on": "2026-11-16",
    }
    # Without a default, unlisted pairs are undecided.
    assert check_decisions(report, "h", base)["undecided_count"] == 3
    default = {"decision": "kept", "reason": "templated boilerplate"}
    result = check_decisions(
        report,
        "h",
        {
            **base,
            "default": default,
            "decisions": [{"documents": ["c", "d"], "decision": "dropped"}],
        },
    )
    assert result["problems"] == []
    assert (result["reported_pairs"], result["decided_pairs"]) == (3, 3)
    assert (result["defaulted_pairs"], result["undecided_count"]) == (2, 0)
    # A default must be `kept` with a reason, and does not hide bad entries.
    for bad in (
        {"decision": "dropped", "reason": "x"},
        {"decision": "kept"},
        {"decision": "kept", "reason": "  "},
    ):
        result = check_decisions(report, "h", {**base, "default": bad})
        assert "default decision must be `kept` with a reason" in result["problems"]
        assert result["undecided_count"] == 3


def test_a_killed_near_duplicate_screen_resumes_to_the_same_report(
    planted_release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus import near_duplicates

    calls = {"count": 0, "kill_at": None}
    real = near_duplicates._band_keys

    def counted(shingles):
        calls["count"] += 1
        if calls["count"] == calls["kill_at"]:
            raise KeyboardInterrupt
        return real(shingles)

    monkeypatch.setattr(near_duplicates, "_band_keys", counted)
    whole = near_duplicates.audit_release(planted_release)
    trained = sum(
        row["documents"] for row in whole["coverage"] if row["split"] == "train"
    )
    held = calls["count"] - trained
    assert trained >= 3

    checkpoints = tmp_path / "checkpoints"
    calls.update(count=0, kill_at=held + 2)
    with pytest.raises(KeyboardInterrupt):
        near_duplicates.audit_release(
            planted_release, checkpoint_dir=checkpoints, checkpoint_every=1
        )
    (saved,) = checkpoints.glob("near-duplicates-*.json")

    calls.update(count=0, kill_at=None)
    resumed = near_duplicates.audit_release(
        planted_release, checkpoint_dir=checkpoints, checkpoint_every=1
    )
    assert resumed == whole
    # The held-out side is indexed again, but only the unfinished training
    # documents are screened.
    assert calls["count"] == held + trained - 1
    assert not saved.exists()

    # A checkpoint that does not verify is ignored, not trusted.
    calls.update(count=0, kill_at=held + 2)
    with pytest.raises(KeyboardInterrupt):
        near_duplicates.audit_release(
            planted_release, checkpoint_dir=checkpoints, checkpoint_every=1
        )
    (saved,) = checkpoints.glob("near-duplicates-*.json")
    saved.write_bytes(saved.read_bytes().replace(b'"trained":1', b'"trained":2'))
    calls.update(count=0, kill_at=None)
    assert (
        near_duplicates.audit_release(
            planted_release, checkpoint_dir=checkpoints, checkpoint_every=1
        )
        == whole
    )
    assert calls["count"] == held + trained
