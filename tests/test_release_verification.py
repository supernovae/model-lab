"""Release verification stays strict while it runs in parallel and reports progress."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from corpus_fixtures import edit_declared_source as _sources
from corpus_fixtures import forge_release as forge
from corpus_fixtures import write_prose_project

from klab.canonical import canonical_json
from klab.corpus import release as release_module
from klab.corpus import release_verify
from klab.corpus.acquisition import acquire
from klab.corpus.pipeline import build
from klab.corpus.project import load_project
from klab.corpus.release import audit, freeze, verify_release
from klab.corpus.release_verify import verification_settings

TEXTS = {
    "prose_train": [f"training document number {n} with words\n" for n in range(6)],
    "prose_validation": ["validation document\n", "another held-out document\n"],
    "prose_test": ["test document\n"],
}


@pytest.fixture(scope="module")
def release(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("verify")
    loaded = load_project(write_prose_project(root, TEXTS) / "corpus.yaml")
    acquire(loaded, root / "work")
    return freeze(build(loaded, root / "work", offline=True), root / "work")


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))


def _documents(edit):
    def apply(root: Path) -> None:
        rows = _lines(root / "documents.jsonl")
        edit(rows)
        _write(root / "documents.jsonl", rows)

    return apply


def _lineage(edit):
    def apply(root: Path) -> None:
        rows = _lines(root / "lineage.jsonl")
        edit(rows)
        _write(root / "lineage.jsonl", rows)

    return apply


def _spans(edit):
    def apply(root: Path) -> None:
        rows = _lines(root / "spans.jsonl")
        edit(rows)
        _write(root / "spans.jsonl", rows)

    return apply


def _drop_first_train_view_row(root: Path) -> None:
    for name in ("lm/train.jsonl", "lm/train.lineage.jsonl"):
        lines = (root / name).read_bytes().splitlines(keepends=True)
        (root / name).write_bytes(b"".join(lines[1:]))


def _drop_last_link(root: Path) -> None:
    lines = (root / "lm/train.lineage.jsonl").read_bytes().splitlines(keepends=True)
    (root / "lm/train.lineage.jsonl").write_bytes(b"".join(lines[:-1]))


def _edit_stage_output(root: Path) -> None:
    path = root / "stages/lm.jsonl"
    rows = _lines(path)
    rows[0]["text"] += "edited"
    _write(path, rows)


def _relicense(row):
    row["license"] = "Apache-2.0"


def _new_license_url(row):
    row["license_url"] = "https://example.org/other-license"


def _looser_rights(row):
    row["rights_policy"]["training_eligibility"] = "eligible_with_obligations"


def _claim_no_restriction(row):
    row["explicit_training_restriction"] = "none_found"


def _stale_text(rows):
    rows[0]["text"] += "edited"


def _wrong_split(rows):
    rows[0]["split"] = "train" if rows[0]["split"] != "train" else "validation"


def _repeat_row(rows):
    rows.append(rows[0])


def _raw_digest(rows):
    rows[0]["raw_sha256"] = "0" * 64


def _self_duplicate(rows):
    rows[0]["drop_reason"] = "duplicate"


def _foreign_license(rows):
    rows[0]["license"] = "Apache-2.0"


FORGERIES = {
    "document text no longer matches its digest": (
        _documents(_stale_text),
        "document content digest mismatch",
    ),
    "document rights differ from the pinned file": (
        _documents(_foreign_license),
        "document rights attribution mismatch",
    ),
    "source license edited after the build": (
        _sources(_relicense),
        "differs from its build identity declaration",
    ),
    "source license URL edited after the build": (
        _sources(_new_license_url),
        "differs from its build identity declaration",
    ),
    "source rights edited after the build": (
        _sources(_looser_rights),
        "differs from its build identity declaration",
    ),
    "source restriction state added after the build": (
        _sources(_claim_no_restriction),
        "differs from its build identity declaration",
    ),
    "source rights re-declared in the carried build identity": (
        _sources(_looser_rights, identity=True),
        "release build identity differs from its build",
    ),
    "document claims to duplicate itself": (
        _documents(_self_duplicate),
        "invalid duplicate representative identity",
    ),
    "document identity repeated": (
        _documents(_repeat_row),
        "duplicate document identity",
    ),
    "stage output differs from its receipt": (
        _edit_stage_output,
        "stage receipt does not match output",
    ),
    "span cites different raw bytes": (
        _spans(_raw_digest),
        "document snapshot span mismatch",
    ),
    "lineage disagrees with its document": (
        _lineage(_wrong_split),
        "document lineage mismatch",
    ),
    "lineage identity repeated": (_lineage(_repeat_row), "duplicate lineage ID"),
    "view omits a selected document": (
        _drop_first_train_view_row,
        "LM view omitted selected documents",
    ),
    "view and lineage lengths differ": (_drop_last_link, "LM lineage count mismatch"),
}


@pytest.fixture(params=["inline", "parallel"])
def execution(request, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run once in-process and once through worker processes over tiny chunks."""
    if request.param == "parallel":
        monkeypatch.setattr(release_verify, "PARALLEL_MIN_BYTES", 0)
        monkeypatch.setattr(release_verify, "CHUNK_BYTES", 400)
        monkeypatch.setattr(release_module, "VIEW_LINES", 2)
    return request.param


def test_valid_release_verifies_in_either_execution(
    release: Path, execution: str
) -> None:
    with verification_settings(workers=3):
        assert verify_release(release)["release_id"] == release.name


@pytest.mark.parametrize("name", FORGERIES)
def test_resealed_forgery_is_refused_with_its_cause(
    release: Path, execution: str, name: str
) -> None:
    edit, message = FORGERIES[name]
    forged = forge(release, edit)
    with verification_settings(workers=3), pytest.raises(ValueError, match=message):
        verify_release(forged)


def test_scratch_and_progress_live_where_the_operator_points(
    release: Path, tmp_path: Path
) -> None:
    before = sorted(path.name for path in release.parent.iterdir())
    scratch = tmp_path / "fast-disk"
    with verification_settings(scratch_dir=scratch, workers=2):
        verify_release(release)
    assert sorted(path.name for path in release.parent.iterdir()) == before
    [journal] = scratch.glob("verify-*.progress.jsonl")
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    phases = {event["phase"] for event in events}
    assert {"hash_files", "documents", "raw_replay", "lineage"} <= phases
    assert events[-1]["event"] == "complete"
    assert events[-1]["documents_processed"] > 0
    assert not list(scratch.glob(".verify-corpus-*"))


def test_receipt_is_reused_until_the_release_changes(
    release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.verification_proofs import verification_options

    work = release.parents[3]
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    cold_runs = []
    original = release_module._verify_release_cold
    monkeypatch.setattr(
        release_module,
        "_verify_release_cold",
        lambda *args, **kwargs: (cold_runs.append(args[0]), original(*args, **kwargs))[
            1
        ],
    )
    work.chmod(0o700)
    with verification_settings(**verification_options(work)):
        assert audit(release) == audit(release)
    assert len(cold_runs) == 1, "the second audit must reuse the signed receipt"
    with verification_settings(**verification_options(work, cold=True)):
        audit(release)
    assert len(cold_runs) == 2, "--cold ignores receipts"
    victim = release / "report.md"
    original_bytes = victim.read_bytes()
    try:
        victim.write_bytes(original_bytes + b"x")
        with (
            verification_settings(**verification_options(work)),
            pytest.raises(ValueError, match="tampered"),
        ):
            audit(release)
    finally:
        victim.write_bytes(original_bytes)


def test_audit_command_reports_progress_and_honours_scratch(
    release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from klab.cli.main import main

    scratch = tmp_path / "fast-disk"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "klab",
            "--work-dir",
            str(release.parents[3]),
            "corpus",
            "audit",
            str(release),
            "--cold",
            "--workers",
            "2",
            "--scratch-dir",
            str(scratch),
        ],
    )
    main()
    output = capsys.readouterr()
    assert json.loads(output.out) == json.loads((release / "audit.json").read_text())
    assert "[raw_replay]" in output.err and "documents/s" in output.err
    assert list(scratch.glob("verify-*.progress.jsonl"))


def test_altered_snapshot_bytes_are_refused(release: Path) -> None:
    snapshot_file = next((release.parents[1] / "snapshots").glob("*/*/files/*"))
    original = snapshot_file.read_bytes()
    try:
        snapshot_file.write_bytes(original + b"edited\n")
        with (
            verification_settings(workers=2),
            pytest.raises(ValueError, match="snapshot byte mismatch"),
        ):
            verify_release(release)
    finally:
        snapshot_file.write_bytes(original)


def test_each_snapshot_is_hashed_once_per_verification(
    release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus import acquisition

    seen = []
    original = acquisition._verify_snapshot_cold
    monkeypatch.setattr(
        acquisition,
        "_verify_snapshot_cold",
        lambda path, **kwargs: (seen.append(path), original(path, **kwargs))[1],
    )
    verify_release(release)
    assert len(seen) == len(set(seen)) == len(TEXTS)


def test_duplicate_keys_follow_the_declared_policy() -> None:
    # Two pages with the same URL are distinct documents unless the release
    # declares canonical_origin duplicates; verification must not assume they merge.
    doc = {
        "raw_content_sha256": "r" * 64,
        "content_sha256": "n" * 64,
        "source_id": "web",
        "metadata": {"url": "https://example.org/page"},
    }
    strict = release_module._declared_keys(doc, ("raw", "normalized"))
    assert strict == ["raw:" + "r" * 64, "normalized:" + "n" * 64]
    full = release_module._declared_keys(doc, ("raw", "normalized", "canonical_origin"))
    assert full[:2] == strict
    assert full[2:] == ["origin:url:example.org/page"]
    assert release_module._declared_keys(doc, ("canonical_origin",)) == full[2:]
