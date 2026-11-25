"""Release verification stays strict while it runs in parallel and reports progress."""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from corpus_fixtures import edit_declared_source as _sources
from corpus_fixtures import forge_release as forge
from corpus_fixtures import write_prose_project

from klab.canonical import canonical_json
from klab.corpus import release as release_module
from klab.corpus import shards
from klab.corpus.acquisition import acquire
from klab.corpus.pipeline import build
from klab.corpus.project import load_project
from klab.corpus.release import (
    audit,
    describe,
    freeze,
    sample,
    sources,
    verify_release,
)
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
    from klab.training.manifest import sha256_file

    path = root / "stages/lm.jsonl"
    rows = _lines(path)
    rows[0]["text"] += "edited"
    _write(path, rows)
    # Re-seal the raw bytes while leaving the ordered canonical-row identity stale.
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    next(stage for stage in manifest["stages"] if stage["id"] == "lm")[
        "output_sha256"
    ] = sha256_file(path)
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")


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
        monkeypatch.setattr(shards, "PARALLEL_MIN_BYTES", 0)
        monkeypatch.setattr(shards, "CHUNK_BYTES", 400)
        monkeypatch.setattr(release_module, "VIEW_LINES", 2)
    return request.param


def test_valid_release_verifies_in_either_execution(
    release: Path, execution: str
) -> None:
    with verification_settings(workers=3):
        assert verify_release(release)["release_id"] == release.name


def test_raw_replay_authenticates_bytes_outside_document_spans(
    release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(release.parent.parent, corpus)
    victim = next((corpus / "snapshots/prose_train").glob("*/files/*.txt"))
    verify_rights = release_module._verify_rights_files

    def append_after_snapshot_check(*args, **kwargs):
        files = verify_rights(*args, **kwargs)
        with victim.open("ab") as stream:
            stream.write(b"appended outside every document span\n")
        return files

    monkeypatch.setattr(
        release_module, "_verify_rights_files", append_after_snapshot_check
    )
    with (
        verification_settings(workers=1),
        pytest.raises(ValueError, match="source replay SHA-256 mismatch"),
    ):
        verify_release(corpus / "releases" / release.name)


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
    phase_start = {
        event["phase"]: event["input_bytes_processed"]
        for event in events
        if event["event"] == "phase"
    }
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    stage_bytes = sum(
        manifest["files"][f"stages/{stage['id']}.jsonl"]["size"]
        for stage in manifest["stages"]
    )
    inventory_bytes = sum(entry["size"] for entry in manifest["files"].values())
    assert phase_start["stage_receipts"] == (
        inventory_bytes + manifest_path.stat().st_size - stage_bytes
    )
    assert phase_start["snapshots"] - phase_start["stage_receipts"] == 2 * stage_bytes
    assert not list(scratch.glob(".verify-corpus-*"))


@pytest.fixture
def cold_runs(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every full verification of a release, so a test can count what was re-read."""
    runs: list[Path] = []
    original = release_module._verify_release_cold
    monkeypatch.setattr(
        release_module,
        "_verify_release_cold",
        lambda *args, **kwargs: (runs.append(args[0]), original(*args, **kwargs))[1],
    )
    return runs


@pytest.fixture
def receipts(
    release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """The release's work root with a private proof store, and its receipt files."""
    work = release.parents[3]
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    work.chmod(0o700)
    yield work
    shutil.rmtree(work / "cache", ignore_errors=True)


def _release_receipts(work: Path) -> list[Path]:
    return sorted((work / "cache/verification-v1").glob("release-*.json"))


def test_one_verification_serves_every_later_command(
    release: Path,
    receipts: Path,
    cold_runs: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from klab.verification_proofs import verification_options

    monkeypatch.setattr(release_module, "INVENTORY_RECEIPT_MIN_BYTES", 0)
    verify_stage = release_module._verify_stage_output
    stage_reads = []

    def observe_stage(path, *args):
        stage_reads.append(path)
        verify_stage(path, *args)

    def stop_before_snapshots(*args, **kwargs):
        raise RuntimeError("stopped before snapshot checks")

    monkeypatch.setattr(release_module, "_verify_stage_output", observe_stage)
    with verification_settings(echo=sys.stderr, **verification_options(receipts)):
        with monkeypatch.context() as fault:
            fault.setattr(release_module, "_verified_snapshot", stop_before_snapshots)
            with pytest.raises(RuntimeError, match="stopped before snapshot checks"):
                audit(release)
        assert not _release_receipts(receipts)
        finished = stage_reads.copy()
        assert finished

        first = audit(release)
        assert stage_reads == finished
        assert len(cold_runs) == 2 and len(_release_receipts(receipts)) == 1
        capsys.readouterr()
        assert sources(release) and sample(release) and describe(release)
        assert audit(release) == first
    assert len(cold_runs) == 2, "later commands must verify by receipt"
    assert capsys.readouterr().err.count("verified by receipt") == 4
    with verification_settings(**verification_options(receipts, cold=True)):
        audit(release)
    assert stage_reads == finished + finished
    assert len(cold_runs) == 3, "--cold ignores receipts"


@pytest.mark.parametrize("cause", ["rewritten", "forged", "verifier changed"])
def test_receipt_is_refused_when_its_cause_no_longer_holds(
    release: Path,
    receipts: Path,
    cold_runs: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    cause: str,
) -> None:
    from klab.corpus import release_receipt
    from klab.verification_proofs import verification_options

    with verification_settings(**verification_options(receipts)):
        audit(release)
        victim = release / "report.md"
        if cause == "rewritten":
            # Same bytes, same size: only the inode metadata says it was touched.
            victim.write_bytes(victim.read_bytes())
        elif cause == "forged":
            path = _release_receipts(receipts)[0]
            path.write_bytes(path.read_bytes().replace(b'"id"', b'"ID"'))
        else:
            monkeypatch.setattr(
                release_receipt, "verifier_digest", lambda kind: "0" * 64
            )
        audit(release)
    assert len(cold_runs) == 2


def test_tampered_file_is_found_through_a_receipt(
    release: Path,
    receipts: Path,
    cold_runs: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from klab.verification_proofs import verification_options

    monkeypatch.setattr(release_module, "INVENTORY_RECEIPT_MIN_BYTES", 0)
    victim = release / "stages/lm.jsonl"
    original = victim.read_bytes()
    edited = original.replace(b"document", b"documenT", 1)
    assert edited != original and len(edited) == len(original)
    with verification_settings(**verification_options(receipts)):
        audit(release)
        try:
            victim.write_bytes(edited)
            with pytest.raises(ValueError, match="stage receipt does not match output"):
                audit(release)
        finally:
            victim.write_bytes(original)
        # Restoring the bytes is a change too: verified again, never trusted blind.
        audit(release)
    assert len(cold_runs) == 3


def test_files_a_receipt_cannot_vouch_for_are_verified_every_time(
    release: Path, receipts: Path, cold_runs: list[Path], capsys
) -> None:
    from klab.verification_proofs import verification_options

    link = release / "linked.tmp"
    os.link(release / "report.md", link)
    try:
        with (
            verification_settings(echo=sys.stderr, **verification_options(receipts)),
            pytest.raises(ValueError),
        ):
            audit(release)
    finally:
        link.unlink()
    assert not _release_receipts(receipts)
    assert "another hard link" in capsys.readouterr().err


def test_receipt_survives_changes_to_code_that_does_not_verify() -> None:
    from klab.verifier_authority import NOT_VERIFICATION, local_import_closure

    closure = local_import_closure(
        ("corpus.release",), within="corpus", exclude=NOT_VERIFICATION
    )
    assert "corpus.release_verify" in closure and "corpus.acquisition" in closure
    for unrelated in ("near_duplicates", "export", "cli", "mixture", "measurement"):
        assert f"corpus.{unrelated}" not in closure


def test_audit_command_reports_progress_and_honours_scratch(
    release: Path,
    receipts: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from klab.cli.main import main
    from klab.corpus import release_verify
    from klab.training import manifest

    scratch = tmp_path / "fast-disk"
    args = [
        "klab",
        "--work-dir",
        str(receipts),
        "corpus",
        "audit",
        str(release),
        "--workers",
        "1",
        "--scratch-dir",
        str(scratch),
    ]
    monkeypatch.setattr("sys.argv", args)
    monkeypatch.setattr(release_verify, "INVENTORY_RECEIPT_MIN_BYTES", 0)
    hash_file = manifest.sha256_file
    hashed = []

    def observe_inventory(path, *args, hash_chunk=None, **options):
        if hash_chunk is not None and path.is_relative_to(release):
            hashed.append(path)
        return hash_file(path, *args, hash_chunk=hash_chunk, **options)

    def stop_after_first(path, *args, hash_chunk=None, **options):
        if hashed and hash_chunk is not None and path.is_relative_to(release):
            raise RuntimeError("stopped during inventory")
        return observe_inventory(path, *args, hash_chunk=hash_chunk, **options)

    with monkeypatch.context() as fault:
        fault.setattr(manifest, "sha256_file", stop_after_first)
        with pytest.raises(RuntimeError, match="stopped during inventory"):
            main()
    [finished] = hashed
    assert not _release_receipts(receipts)
    hashed.clear()
    capsys.readouterr()
    monkeypatch.setattr(manifest, "sha256_file", observe_inventory)
    main()
    output = capsys.readouterr()
    assert json.loads(output.out) == json.loads((release / "audit.json").read_text())
    assert finished not in hashed and hashed
    assert "inventory hashes: reused 1/" in output.err
    assert "[raw_replay]" in output.err and "documents/s" in output.err
    [journal] = scratch.glob("verify-*.progress.jsonl")
    assert any(
        json.loads(line)["input_bytes_reused"] == finished.stat().st_size
        for line in journal.read_text().splitlines()
    )

    hashed.clear()
    monkeypatch.setattr("sys.argv", [*args, "--cold"])
    main()
    assert finished in hashed

    monkeypatch.setattr("sys.argv", args)
    documents = release / "documents.jsonl"
    original, stamp = documents.read_bytes(), documents.stat()
    try:
        documents.write_bytes(b" " + original[1:])
        os.utime(documents, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with pytest.raises(ValueError, match="tampered artifact: documents.jsonl"):
            main()
    finally:
        documents.write_bytes(original)


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
