"""The near-duplicate screen on synthetic documents: templates, shards, resume."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from klab.corpus import near_duplicates
from klab.corpus.near_duplicates import screen_documents

TEMPLATE = (
    "Thank you for opening this pull request against the project. Please describe "
    "the change and link the issue it fixes. Tick every box that applies: tests "
    "added, documentation updated, changelog entry written, breaking change noted. "
    "Maintainers review in the order opened and may ask for a rebase before merge; "
    "by submitting you agree to the contributor license and the code of conduct."
)
SHORT = "thanks for the report everyone"


def _words(rng: random.Random, count: int) -> str:
    return " ".join(f"w{rng.randrange(10**9)}" for _ in range(count))


def _row(name: str, split: str, source: str, text: str) -> dict:
    return {
        "document_id": name,
        "split": split,
        "source_id": source,
        "source_location": f"{name}.txt",
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
    }


@pytest.fixture
def corpus() -> dict[str, str]:
    """Held-out threads that all use one template, and training documents that
    share the template, copy one thread, repeat one, or repeat a short reply."""
    rng = random.Random(7)
    rows = []
    content = {}
    for i in range(25):
        content[i] = _words(rng, 8)
        rows.append(
            _row(
                f"h{i:02d}", "validation", "threads_heldout", f"{TEMPLATE} {content[i]}"
            )
        )
        rows.append(_row(f"r{i:02d}", "validation", "replies_heldout", SHORT))
    rows += [
        # The template with unrelated content: similar only through boilerplate.
        _row("t_template", "train", "threads", f"{TEMPLATE} {_words(rng, 8)}"),
        # A real near-copy of one held-out thread, template around it.
        _row("t_copy", "train", "threads", f"{TEMPLATE} {content[3]} extra"),
        # Identical text, however common its words are.
        _row("t_same", "train", "threads", f"{TEMPLATE} {content[7]}"),
        # A short reply repeated by many held-out documents is not template.
        _row("t_short", "train", "replies", SHORT.upper() + "!"),
        _row("t_other", "train", "threads", _words(rng, 40)),
    ]
    return {"rows": rows}


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _pairs(report: dict) -> set[frozenset[str]]:
    return {
        frozenset((p["left_document_id"], p["right_document_id"]))
        for p in report["candidates"]
    }


def test_template_only_pairs_are_counted_and_real_copies_stay_candidates(
    tmp_path: Path, corpus
) -> None:
    documents = _write(tmp_path / "documents.jsonl", corpus["rows"])
    strict = screen_documents(documents, identity="x")
    aware = screen_documents(
        documents, identity="x", template_aware=True, list_template_pairs=True
    )

    assert strict["parameters"]["template_aware"] is False
    assert "template_only" not in strict
    # Strict mode flags the template on every held-out thread.
    assert len(_pairs(strict)) > 40
    # Template-aware mode keeps the near-copy, the identical text and the
    # repeated short reply (every shingle of it is boilerplate: nothing is left
    # to tell it from a copy), and counts the rest as template_only.
    assert _pairs(aware) == {
        frozenset(("h03", "t_copy")),
        frozenset(("h07", "t_same")),
        *(frozenset((f"r{i:02d}", "t_short")) for i in range(25)),
    }
    copy = next(p for p in aware["candidates"] if "t_copy" in p.values())
    assert copy["content_jaccard"] > 0.65 and copy["jaccard"] > 0.65
    assert copy["class"] == "leak"
    short = next(p for p in aware["candidates"] if "t_short" in p.values())
    assert short["class"] == "content_too_short" and short["content_jaccard"] is None
    assert aware["class_counts"] == {
        "leak": 2,
        "content_too_short": 25,
        "partly_template": 0,
        "template_only": aware["template_only_count"],
    }
    same = next(p for p in aware["candidates"] if "t_same" in p.values())
    assert same["identical_content_sha256"] and aware["identical_text_count"] == 1
    # Nothing is hidden: strict candidates are candidates plus template_only pairs.
    assert (
        aware["candidate_count"] + aware["template_only_count"]
        == strict["candidate_count"]
    )
    summary = {e["document_id"]: e for e in aware["template_only"]}
    assert set(summary) == {f"h{i:02d}" for i in range(25)}
    assert summary["h00"]["pairs"] >= 2  # t_template, t_other-like pairs and more
    assert summary["h00"]["max_jaccard"] >= 0.65
    assert summary["h00"]["min_boilerplate_share"] > 0.5
    assert summary["h00"]["closest_content_jaccard"] < 0.65
    # The pair that is a candidate is not also counted as template_only.
    assert aware["template_only_document_count"] == 25
    assert aware["boilerplate"]["heldout_shingles"] > 0

    from klab.corpus.evidence_audit import audit_evidence
    from klab.training.manifest import sha256_file

    report = tmp_path / "template-report.json"
    report.write_text(json.dumps(aware))
    labels = tmp_path / "labels.json"
    cohort = {
        "report": report.name,
        "sha256": sha256_file(report),
        "pairs": [
            {"units": ["h00", "t_template"], "expected": "template_only"},
            {"units": ["h03", "t_copy"], "expected": "candidate"},
            {"units": ["h07", "t_same"], "expected": "candidate"},
            {"units": ["h00", "t_short"], "expected": "absent"},
        ],
    }
    labels.write_text(json.dumps({"screens": [cohort]}))
    [measured] = audit_evidence(tmp_path, labels)["screens"]
    assert measured["predictions_by_expected_class"] == {
        "template_only": {"template_only": 1},
        "candidate": {"candidate": 2},
        "absent": {"absent": 1},
    }
    assert measured["metrics"]["candidate"]["recall"] == 1
    assert measured["metrics"]["template_only"]["precision"] == 1
    assert measured["unlisted_pairs"] == 0
    truncated = screen_documents(
        documents,
        identity="x",
        template_aware=True,
        list_template_pairs=True,
        max_candidates=1,
    )
    report.write_text(json.dumps(truncated))
    with pytest.raises(ValueError, match="screen digest differs"):
        audit_evidence(tmp_path, labels)
    cohort["sha256"] = sha256_file(report)
    labels.write_text(json.dumps({"screens": [cohort]}))
    [measured] = audit_evidence(tmp_path, labels)["screens"]
    assert measured["unlisted_pairs"] > 0


def test_the_training_sample_finds_a_template_the_heldout_side_barely_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = random.Random(3)
    rows = [_row("h0", "validation", "threads_heldout", f"{TEMPLATE} {_words(rng, 8)}")]
    rows += [
        _row(f"t{i}", "train", "threads", f"{TEMPLATE} {_words(rng, 8)}")
        for i in range(12)
    ]
    documents = _write(tmp_path / "documents.jsonl", rows)
    strict = screen_documents(documents, identity="x")
    retained = {
        "template_aware": True,
        "checkpoint_dir": tmp_path / "checkpoints",
        "retain_evidence": True,
        "shard_bytes": 2048,
    }
    sampled = screen_documents(
        documents, identity="x", boilerplate_sample_stride=1, **retained
    )
    assert strict["candidate_count"] == 12
    assert sampled["candidate_count"] == 0 and sampled["template_only_count"] == 12

    def rescan(*args):
        pytest.fail("a classification change reread a retained evidence range")

    new_policy = {
        "identity": "other release using the same documents",
        "boilerplate_sample_stride": 1,
        "boilerplate_sample_documents": 13,
        "max_candidates": 5,
    }
    fresh = screen_documents(documents, template_aware=True, **new_policy)
    with monkeypatch.context() as patch:
        patch.setattr(near_duplicates, "_scan_file", rescan)
        assert screen_documents(documents, **retained, **new_policy) == fresh
    assert fresh["candidate_count"] == 12 and fresh["candidates_truncated"]

    unsampled = screen_documents(
        documents, identity="x", boilerplate_sample_stride=10**9, **retained
    )
    # With too few held-out documents and no sample the template is unknown,
    # so every pair stays a candidate: the failure is toward review.
    assert unsampled["candidate_count"] == 12 and unsampled["template_only_count"] == 0

    rows.append(_row("h1", "test", "threads_heldout", f"{TEMPLATE} {_words(rng, 8)}"))
    _write(documents, rows)
    fresh = screen_documents(documents, template_aware=True, **new_policy)
    assert fresh["document_count"] == len(rows)
    assert screen_documents(documents, **retained, **new_policy) == fresh


@pytest.mark.parametrize(
    ("template_aware", "passage_words"), [(False, None), (True, None), (True, 32)]
)
def test_shards_workers_and_a_killed_run_all_give_the_serial_report(
    tmp_path: Path,
    corpus,
    monkeypatch: pytest.MonkeyPatch,
    template_aware: bool,
    passage_words: int | None,
) -> None:
    from klab.corpus import shards

    documents = _write(tmp_path / "documents.jsonl", corpus["rows"])
    size = documents.stat().st_size
    options = {
        "identity": "x",
        "template_aware": template_aware,
        "passage_words": passage_words,
        "list_template_pairs": template_aware,
    }
    whole = screen_documents(documents, shard_bytes=size, **options)
    assert near_duplicates._line_ranges(documents, 4096)[1:]
    monkeypatch.setattr(shards, "PARALLEL_MIN_BYTES", 0)
    sharded = screen_documents(
        documents,
        shard_bytes=4096,
        workers=3,
        checkpoint_dir=tmp_path / "parallel",
        **options,
    )
    assert sharded == whole

    calls = {"count": 0, "kill_at": None}
    real = near_duplicates._band_keys

    def counted(shingles):
        calls["count"] += 1
        if calls["count"] == calls["kill_at"]:
            raise KeyboardInterrupt
        return real(shingles)

    monkeypatch.setattr(near_duplicates, "_band_keys", counted)
    checkpoints = tmp_path / "checkpoints"
    resumable = {
        "checkpoint_dir": checkpoints,
        "checkpoint_every": 1,
        "shard_bytes": 4096,
    }
    screen_documents(documents, **options, **resumable)
    full_calls = calls["count"]
    assert not list(checkpoints.glob("near-duplicates-*.json"))

    calls.update(count=0, kill_at=full_calls - 3)
    with pytest.raises(KeyboardInterrupt):
        screen_documents(documents, **options, **resumable)
    saved = sorted(checkpoints.glob("near-duplicates-*.json"))
    assert len(saved) > 1  # shards finished before the kill are kept
    evidence = sorted(
        checkpoints.glob("near-duplicates-*.evidence/files/shard/*/evidence.bin")
    )
    assert len(evidence) > 1
    damaged = evidence[0]
    task = json.loads(damaged.with_name("receipt.json").read_bytes())["inputs"]["task"]
    damaged.write_bytes(b"damaged held-out evidence")
    scans = []
    scan = near_duplicates._scan_file

    def count_scan(staging, path, start, end, *options):
        scans.append((start, end))
        return scan(staging, path, start, end, *options)

    monkeypatch.setattr(near_duplicates, "_scan_file", count_scan)
    calls.update(count=0, kill_at=None)
    assert screen_documents(documents, **options, **resumable) == whole
    assert scans == [tuple(task[1:3])]
    assert calls["count"] < full_calls
    assert not list(checkpoints.glob("near-duplicates-*.json"))
    assert not list(checkpoints.glob("near-duplicates-*.evidence"))
