"""Freezing renames the verified build into the release; it never copies the payload."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from corpus_fixtures import forge_release, write_prose_project

from klab.canonical import canonical_digest, canonical_json
from klab.corpus import release as release_module
from klab.corpus.acquisition import acquire
from klab.corpus.pipeline import build
from klab.corpus.project import load_project
from klab.corpus.release import consumed_release, freeze, verify_build, verify_release
from klab.training.manifest import sha256_file
from klab.verification_proofs import verification_options

TEXTS = {
    "prose_train": [f"training document {n} with some words\n" for n in range(4)],
    "prose_validation": ["validation document\n"],
    "prose_test": ["test document\n"],
}


def _corpus_json(work, monkeypatch, capsys, *arguments, exit_code=0):
    from klab.cli.main import main

    monkeypatch.setattr(
        sys, "argv", ["klab", "--work-dir", str(work), "corpus", *arguments, "--json"]
    )
    if exit_code:
        with pytest.raises(SystemExit) as stopped:
            main()
        assert stopped.value.code == exit_code
    else:
        main()
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def prepared(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    original_umask = os.umask(0o077)
    try:
        loaded = load_project(write_prose_project(tmp_path, TEXTS) / "corpus.yaml")
        work = tmp_path / "work"
        acquire(loaded, work)
        yield build(loaded, work, offline=True), work
    finally:
        os.umask(original_umask)


def _payload(root: Path) -> dict[str, tuple[str, int]]:
    """Name, digest and inode of every file except the two identity files."""
    return {
        str(path.relative_to(root)): (sha256_file(path), path.stat().st_ino)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"build.json", "manifest.json"}
    }


def test_frozen_release_is_the_build_directory_renamed(
    prepared: tuple[Path, Path],
) -> None:
    build_dir, work = prepared
    before = _payload(build_dir)
    release = freeze(build_dir, work)
    assert verify_release(release)["release_id"] == release.name
    assert _payload(release) == before, "same bytes, same inodes: nothing was copied"
    assert not (release / "build.json").exists()
    assert sorted(path.name for path in build_dir.iterdir()) == [
        "build.json",
        "consumed.json",
    ]
    assert consumed_release(build_dir) == release.name
    assert freeze(build_dir, work) == release
    with pytest.raises(ValueError, match="frozen in place"):
        verify_build(build_dir)
    assert not list(release.parent.glob(".freezing-*"))


def test_normalized_prune_retries_a_retired_partition(tmp_path, monkeypatch, capsys):
    from klab.corpus import artifact_inventory
    from klab.corpus.normalized_retention import (
        clear_normalized_use,
        record_normalized_use,
    )
    from klab.corpus.prepared_shards import normalized_store_root
    from klab.corpus.release_members import normalized_partition_lease

    work = tmp_path / "work"
    project = work / "corpora" / "example"
    project.mkdir(parents=True, mode=0o700)
    identity = {"artifact_type": "normalized_source_partition", "source_id": "example"}
    key = canonical_digest(identity)
    partition = normalized_store_root(work) / "example" / key
    partition.mkdir(parents=True)
    files = {}
    for name in ("docs.jsonl", "spans.jsonl", "rejected.jsonl"):
        file = partition / name
        file.write_bytes(b"{}\n")
        files[name] = {"sha256": sha256_file(file), "size": file.stat().st_size}
    receipt = {"schema_version": 1, "identity": identity, "key": key, "files": files}
    (partition / "receipt.json").write_bytes(canonical_json(receipt) + b"\n")
    size = sum(path.stat().st_size for path in partition.iterdir())
    artifact_id = f"example/{key}"
    command = ("release-prune", "--kind", "normalized", "--partition", artifact_id)

    def prune(*args, blocked=False):
        return _corpus_json(
            work, monkeypatch, capsys, *command, *args, exit_code=int(blocked)
        )

    assert prune()["payloads"][0]["action"] == "prune"
    assert (partition / "receipt.json").is_file()
    unexpected = partition / "unrelated.txt"
    unexpected.write_text("outside the normalized partition contract")
    assert prune("--apply", blocked=True)["status"] == "blocked"
    assert unexpected.is_file()
    unexpected.unlink()
    build_id = "a" * 64
    record_normalized_use(project, build_id, ["example"])
    retained = prune()["payloads"][0]
    assert retained["action"] == "protected" and retained["resumable_builds"] == 1
    clear_normalized_use(project, build_id)

    legacy = project / "builds" / "stages" / f".index-{'b' * 64}.groups"
    legacy.mkdir(parents=True)
    unknown = prune(blocked=True)
    assert any(
        row["reason"] == "unregistered_native_checkpoint" for row in unknown["problems"]
    )
    legacy.rmdir()

    with normalized_partition_lease(partition):
        assert prune("--apply", blocked=True)["status"] == "blocked"

    retired = partition.with_name(f".pruned-{key}")

    def interrupt(_namespace, _name):
        raise InterruptedError("after retiring partition")

    with monkeypatch.context() as fault:
        fault.setattr(artifact_inventory, "_remove_retired_partition", interrupt)
        assert prune("--apply", blocked=True)["status"] == "partial"
    assert not partition.exists() and retired.is_dir()
    assert prune()["payloads"][0]["interrupted_prune"] is True
    removed = prune("--apply")
    assert removed["pruned"] == [artifact_id] and removed["pruned_bytes"] == size
    assert not retired.exists()
    assert prune("--apply")["payloads"][0]["action"] == "missing"


def test_verification_checks_frozen_document_assignments(
    prepared: tuple[Path, Path],
) -> None:
    build_dir, work = prepared
    rows = sorted(
        (
            json.loads(line)
            for line in (build_dir / "documents.jsonl").read_bytes().splitlines()
        ),
        key=lambda row: row["document_id"],
    )
    assignments = [
        {"document_id": row["document_id"], "split": row["split"]} for row in rows
    ]
    # Split preparation may assign more documents than a release includes.
    assignments.append({"document_id": "~unused", "split": "train"})

    def write_assignments(root):
        temporary = root / "assignments.jsonl"
        temporary.write_bytes(
            b"".join(canonical_json(row) + b"\n" for row in assignments)
        )
        digest = sha256_file(temporary)
        name = f"assignments-{digest}.jsonl"
        temporary.rename(root / name)
        return name, {"sha256": digest, "rows": len(assignments)}

    project = work.parent / "project"
    name, reference = write_assignments(project)
    (project / "splits.yaml").write_bytes(
        canonical_json(
            {"schema_version": 1, "unit": "document", "assignment_file": reference}
        )
    )
    loaded = load_project(project / "corpus.yaml")
    release = freeze(build(loaded, work, offline=True), work)
    assert (
        verify_release(release)["build_identity"]["split"]["assignment_file"]
        == reference
    )

    def replace_assignment(root):
        assignments[0]["split"] = (
            "test" if assignments[0]["split"] == "train" else "train"
        )
        changed_name, changed_reference = write_assignments(root)
        (root / name).unlink()
        path = root / "manifest.json"
        manifest = json.loads(path.read_bytes())
        manifest["build_identity"]["split"]["assignment_file"] = changed_reference
        manifest["build_id"] = canonical_digest(manifest["build_identity"])
        manifest["files"][changed_name] = manifest["files"].pop(name)
        path.write_bytes(canonical_json(manifest) + b"\n")

    with pytest.raises(
        ValueError, match="document differs from frozen split assignment"
    ):
        verify_release(forge_release(release, replace_assignment))


def test_build_changed_after_verification_is_never_published(
    prepared: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    build_dir, work = prepared
    verified = release_module.verify_build

    def verify_then_alter(*args, **kwargs):
        result = verified(*args, **kwargs)
        with (build_dir / "lm/train.jsonl").open("ab") as stream:
            stream.write(b'{"text": "late edit"}\n')
        return result

    monkeypatch.setattr(release_module, "verify_build", verify_then_alter)
    with pytest.raises(ValueError, match="changed while it was being verified"):
        freeze(build_dir, work)
    releases = work / "corpora/prose-fixture/releases"
    assert not list(releases.glob("*")), "nothing published or staged"
    assert consumed_release(build_dir) is None, "the build was not consumed"


def test_shared_view_changed_after_build_rename_is_never_placed(
    prepared: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus.release_members import SHARED_LM_VIEW, shared_store

    build_dir, work = prepared
    digest = sha256_file(build_dir / "lm/train.jsonl")
    publish = release_module.publish_shared_members

    def edit_staging(staging, *args, **kwargs):
        view = staging / "lm/train.jsonl"
        stamp = view.stat()
        data = view.read_bytes()
        changed = data.replace(b"training", b"traininx", 1)
        assert changed != data
        view.write_bytes(changed)
        os.utime(view, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        return publish(staging, *args, **kwargs)

    monkeypatch.setattr(release_module, "publish_shared_members", edit_staging)
    with pytest.raises(ValueError, match="shared staging payload changed"):
        freeze(build_dir, work, share_lm_views=True)
    store = shared_store(work, SHARED_LM_VIEW)
    assert not (store / digest[:2] / digest).exists()


# Each entry makes the n-th durable write or the n-th publishing rename fail.
CRASHES = {
    "after the first rename": ("write", 1),
    "before the consumed marker": ("write", 2),
    "before the manifest": ("write", 3),
    "before the publishing rename": ("rename", 2),
}


@pytest.mark.parametrize("crash", CRASHES)
def test_interrupted_freeze_resumes_to_the_identical_release(
    prepared: tuple[Path, Path],
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    crash: str,
) -> None:
    from klab import safe_io

    build_dir, work = prepared
    reference_root = tmp_path_factory.mktemp("reference")
    reference = load_project(write_prose_project(reference_root, TEXTS) / "corpus.yaml")
    acquire(reference, reference_root / "work")
    expected = freeze(
        build(reference, reference_root / "work", offline=True), reference_root / "work"
    )
    kind, number = CRASHES[crash]
    calls = {"write": 0, "rename": 0}
    writer, renamer = release_module._write_durably, safe_io.rename_noreplace

    def flaky_write(path, content):
        calls["write"] += 1
        if kind == "write" and calls["write"] == number:
            raise OSError("killed")
        return writer(path, content)

    def flaky_rename(source, target):
        calls["rename"] += 1
        if kind == "rename" and calls["rename"] == number:
            raise OSError("killed")
        return renamer(source, target)

    monkeypatch.setattr(release_module, "_write_durably", flaky_write)
    monkeypatch.setattr(safe_io, "rename_noreplace", flaky_rename)
    with pytest.raises(OSError, match="killed"):
        freeze(build_dir, work)
    monkeypatch.undo()
    releases = work / "corpora/prose-fixture/releases"
    assert [path.name for path in releases.iterdir()] == [f".freezing-{build_dir.name}"]

    release = freeze(build_dir, work)
    assert release.name == expected.name
    assert (release / "manifest.json").read_bytes() == (
        expected / "manifest.json"
    ).read_bytes()
    assert {name: digest for name, (digest, _) in _payload(release).items()} == {
        name: digest for name, (digest, _) in _payload(expected).items()
    }
    assert verify_release(release)["release_id"] == expected.name
    assert consumed_release(build_dir) == release.name


def test_resumed_freeze_reuses_verified_staging_after_another_stop(
    prepared: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab import safe_io
    from klab.corpus import freeze_planning
    from klab.corpus.release_members import ReleasePlacement
    from klab.corpus.release_verify import verification_settings

    build_dir, work = prepared
    monkeypatch.setenv("XDG_CONFIG_HOME", str(work.parent / "settings"))
    options = verification_options(work)
    scratch = work / "alternate-scratch"
    inspect_options = verification_options(work, read_only=True)
    original_rename = safe_io.rename_noreplace
    verify = release_module._verify_release_cold
    checks = []

    def observe(path, **kwargs):
        checks.append(path)
        return verify(path, **kwargs)

    def stop_before_publish(source, target):
        if source.name.startswith(".freezing-"):
            raise OSError("stopped before publication")
        return original_rename(source, target)

    def forecast():
        with verification_settings(scratch_dir=scratch, **inspect_options):
            return freeze_planning.plan_freeze(
                build_dir,
                work,
                placement=ReleasePlacement(),
                inspect_verification=True,
            )

    monkeypatch.setattr(release_module, "_verify_release_cold", observe)
    with monkeypatch.context() as fault:
        fault.setattr(safe_io, "rename_noreplace", stop_before_publish)
        with pytest.raises(OSError, match="stopped before publication"):
            freeze(build_dir, work, **options)
        assert not checks

        def no_dependency_walk(*args):
            pytest.fail("missing receipt scanned snapshot dependencies")

        with monkeypatch.context() as missing:
            missing.setattr(
                freeze_planning, "snapshot_dependencies", no_dependency_walk
            )
            pending = forecast()
        assert (
            pending["verification_inspection"]["steps"][0]["method"] == "verify_bytes"
        )
        assert pending["verification_scratch_bytes"] is None
        with pytest.raises(OSError, match="stopped before publication"):
            freeze(build_dir, work, **options)
        [staged] = checks
        assert list(staged.parent.iterdir()) == [staged]
        warm = forecast()
        inspection = warm["verification_inspection"]
        assert inspection["steps"][0]["method"] == "receipt"
        assert warm["verification_scratch_bytes"] == 0
        assert inspection["scratch_path"] == str(scratch)
        assert inspection["scratch_available_bytes"] > 0
        assert not scratch.exists()
        with pytest.raises(OSError, match="stopped before publication"):
            freeze(build_dir, work, **verification_options(work, cold=True))
        assert checks == [staged, staged], "cold freeze must recheck the staged files"

    sealed = (staged / "manifest.json").read_bytes()
    release = freeze(build_dir, work, **options)
    assert verify_release(release, **options)["release_id"] == release.name
    assert checks == [staged, staged], (
        "both the staged and final receipts should be reusable"
    )
    assert (release / "manifest.json").read_bytes() == sealed
    assert (
        release.name == consumed_release(build_dir) == json.loads(sealed)["release_id"]
    )
    assert list(release.parent.iterdir()) == [release]


def test_resumed_freeze_refuses_an_edit_after_staged_verification(
    prepared: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab import result_publication
    from klab.verification_proofs import verification_options

    build_dir, work = prepared
    monkeypatch.setenv("XDG_CONFIG_HOME", str(work.parent / "settings"))
    options = verification_options(work)
    with monkeypatch.context() as fault:
        fault.setattr(
            release_module,
            "_write_durably",
            lambda *args: (_ for _ in ()).throw(OSError("killed")),
        )
        with pytest.raises(OSError, match="killed"):
            freeze(build_dir, work, **options)
    staged = work / "corpora/prose-fixture/releases" / f".freezing-{build_dir.name}"
    sync = result_publication.sync_directory

    def edit_staged_file(path):
        sync(path)
        if path == staged:
            with (staged / "lm/train.jsonl").open("ab") as stream:
                stream.write(b'{"text": "late edit"}\n')

    with monkeypatch.context() as fault:
        fault.setattr(result_publication, "sync_directory", edit_staged_file)
        with pytest.raises(ValueError, match="staged release changed"):
            freeze(build_dir, work, **options)
    with pytest.raises(ValueError, match="tampered artifact: lm/train.jsonl"):
        freeze(build_dir, work, **options)
    assert list(staged.parent.iterdir()) == [staged]


def test_freeze_authenticates_each_snapshot_once(
    prepared: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus import acquisition

    seen = []
    original = acquisition._verify_snapshot_cold
    monkeypatch.setattr(
        acquisition,
        "_verify_snapshot_cold",
        lambda path, **kwargs: (seen.append(path), original(path, **kwargs))[1],
    )
    freeze(*prepared)
    assert len(seen) == len(set(seen)) == len(TEXTS)


def test_release_manifest_names_its_files_and_not_the_build_file(
    prepared: tuple[Path, Path],
) -> None:
    release = freeze(*prepared)
    manifest = json.loads((release / "manifest.json").read_text())
    assert "build.json" not in manifest["files"]
    assert set(manifest["files"]) == {
        str(path.relative_to(release))
        for path in release.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }


def test_shared_releases_reuse_members_and_reject_drift(
    prepared: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from klab.corpus import artifact_inventory, release_members
    from klab.corpus.release_members import LM_VIEW_MEMBERS
    from klab.training import manifest as file_manifest

    build_dir, work = prepared
    monkeypatch.setenv("XDG_CONFIG_HOME", str(work.parent / "settings"))
    monkeypatch.setattr(release_members, "INVENTORY_RECEIPT_MIN_BYTES", 0)
    options = {
        **verification_options(work),
        "share_stage_artifacts": True,
        "share_document_artifacts": True,
        "share_lm_views": True,
    }
    stage_name = "stages/lm.jsonl"
    document_names = ("documents.jsonl", "spans.jsonl")
    original = {
        name: (build_dir / name).stat()
        for name in (stage_name, *document_names, *LM_VIEW_MEMBERS)
    }
    shared_bytes = sum(info.st_size for info in original.values())
    publish = release_module.publish_shared_members

    def query(*arguments):
        return _corpus_json(work, monkeypatch, capsys, *arguments)

    def inventory(*arguments):
        return query("artifacts", "--kind", "release-stages", *arguments)

    def plan(path, *options):
        return query("plan", "--build", str(path), *options)

    share = (
        "--share-stage-artifacts",
        "--share-document-artifacts",
        "--share-lm-views",
    )
    opened_file = Path.open

    def forbid_payload(path, *args, **kwargs):
        if path.is_relative_to(work) and (
            path.suffix == ".jsonl" or path.is_relative_to(work / "artifacts")
        ):
            pytest.fail(f"freeze planning opened payload bytes: {path}")
        return opened_file(path, *args, **kwargs)

    with monkeypatch.context() as metadata_only:
        metadata_only.setattr(Path, "open", forbid_payload)
        first_plan = plan(build_dir, *share)
    assert first_plan["state"] == "build"
    assert first_plan["new_shared_bytes"] == shared_bytes
    assert first_plan["publication_payload_copy_bytes"] == 0
    assert first_plan["verification_scratch_bytes"] is None

    def stop_after_shared_move(staging, files, **kwargs):
        publish(
            staging,
            {name: row for name, row in files.items() if name != "spans.jsonl"},
            **kwargs,
        )
        raise InterruptedError("after shared move")

    spans_digest = sha256_file(build_dir / "spans.jsonl")
    with monkeypatch.context() as fault:
        fault.setattr(release_module, "publish_shared_members", stop_after_shared_move)
        with pytest.raises(InterruptedError, match="after shared move"):
            freeze(build_dir, work, **options)
    pending = inventory()
    assert pending["scan_status"] == "complete"
    assert pending["unreferenced_payloads"] == 0
    assert pending["payloads"][0]["staging_releases"] == 1
    pending_documents = query("artifacts", "--kind", "release-documents")
    assert pending_documents["shared_payloads"] == 1
    assert all(row["staging_releases"] == 1 for row in pending_documents["payloads"])
    pending_views = query("artifacts", "--kind", "release-lm-views")
    assert pending_views["shared_payloads"] == len(LM_VIEW_MEMBERS)
    assert all(row["staging_releases"] == 1 for row in pending_views["payloads"])
    resumed = plan(build_dir, *share)
    assert resumed["state"] == "staging"
    assert {row["member"]: row["action"] for row in resumed["shared_members"]} == {
        "documents.jsonl": "verify_staged_reference",
        "spans.jsonl": "move_to_shared_store",
        stage_name: "verify_staged_reference",
        **dict.fromkeys(LM_VIEW_MEMBERS, "verify_staged_reference"),
    }
    assert resumed["stages"][-1]["inputs"] == ["verify_staged_release"]
    assert plan(build_dir, share[0])["status"] == "blocked"

    with pytest.raises(ValueError, match="same freeze options"):
        freeze(build_dir, work, share_stage_artifacts=True)
    original_hash = file_manifest.sha256_file
    span_hashes = []

    def observe_span_hash(path, **kwargs):
        if path.name in {"spans.jsonl", spans_digest}:
            span_hashes.append(path.name)
        return original_hash(path, **kwargs)

    with monkeypatch.context() as measure:
        measure.setattr(file_manifest, "sha256_file", observe_span_hash)
        first = freeze(build_dir, work, **(options | {"verification_mode": "cold"}))
    assert span_hashes == ["spans.jsonl"]
    assert first_plan["release_id"] == resumed["release_id"] == first.name
    assert plan(build_dir)["state"] == "published"
    assert plan(build_dir, "--inspect-verification")["verification_scratch_bytes"] == 0
    cold_plan = plan(build_dir, "--cold")
    assert cold_plan["verification_inspection"]["steps"][0]["method"] == "verify_bytes"
    assert cold_plan["verification_scratch_bytes"] is None
    proof_options = verification_options(work)
    with release_module.open_release(first, **proof_options) as opened:
        paths = {name: opened.member(name) for name in original}
        for name, path in paths.items():
            assert not (first / name).exists()
            assert path.stat().st_ino == original[name].st_ino
            assert path.name == sha256_file(path)

    declaration = work.parent / "project" / "corpus.yaml"
    project = yaml.safe_load(declaration.read_bytes())
    project["id"] = "second-prose"
    declaration.write_text(yaml.safe_dump(project))
    loaded = load_project(declaration)
    acquire(loaded, work)
    next_build = build(loaded, work, offline=True)
    reuse = plan(next_build, *share)
    assert (
        reuse["shared_candidate_bytes"]
        == reuse["duplicate_staging_bytes"]
        == shared_bytes
    )
    assert reuse["new_shared_bytes"] == 0
    with monkeypatch.context() as metadata_only:
        metadata_only.setattr(Path, "open", forbid_payload)
        inspected = plan(next_build, *share, "--inspect-verification")
    assert inspected["shared_receipt_candidate_bytes"] == shared_bytes
    assert all(
        row["retained_authentication"] == "receipt"
        for row in inspected["shared_members"]
    )
    assert plan(next_build, *share, "--cold")["shared_receipt_candidate_bytes"] == 0
    documents = paths["documents.jsonl"]
    original_documents = documents.read_bytes()
    changed = original_documents.replace(b"training", b"traininx", 1)
    assert changed != original_documents and len(changed) == len(original_documents)
    record = release_module._record_frozen

    def change_before_receipt(*args, **kwargs):
        stamp = documents.stat()
        documents.write_bytes(changed)
        os.utime(documents, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        return record(*args, **kwargs)

    def forbid_retained_hash(path, **kwargs):
        if path in paths.values():
            pytest.fail(f"freeze rehashed a placed member: {path}")
        return original_hash(path, **kwargs)

    with monkeypatch.context() as drift:
        drift.setattr(release_module, "_record_frozen", change_before_receipt)
        drift.setattr(file_manifest, "sha256_file", forbid_retained_hash)
        second = freeze(next_build, work, **options)
    assert reuse["release_id"] == second.name
    assert second != first
    for released in (first, second):
        with pytest.raises(ValueError, match=r"tampered artifact: documents\.jsonl"):
            verify_release(released, **proof_options)
    documents.write_bytes(original_documents)
    with release_module.open_release(second, **proof_options) as opened:
        assert {name: opened.member(name) for name in paths} == paths
        for name in paths:
            assert not (second / name).exists()

    from klab.corpus.near_duplicates import audit_release

    assert audit_release(first, workers=1)["release_id"] == first.name
    assert verify_release(first, **proof_options)["release_id"] == first.name
    document_inventory = query("artifacts", "--kind", "release-documents")
    assert document_inventory["retained_bytes"] == sum(
        original[name].st_size for name in document_names
    )
    assert all(row["published_releases"] == 2 for row in document_inventory["payloads"])
    assert (
        query("artifacts", "--kind", "release-documents", "--unreferenced")["payloads"]
        == []
    )
    lm_view = paths["lm/train.jsonl"]
    view_inventory = query("artifacts", "--kind", "release-lm-views")
    assert all(row["published_releases"] == 2 for row in view_inventory["payloads"])
    protected = _corpus_json(
        work, monkeypatch, capsys,
        "release-prune", "--kind", "release-lm-views", "--digest", lm_view.name,
        "--apply", exit_code=1,
    )  # fmt: skip
    assert protected["payloads"][0]["action"] == "protected" and lm_view.exists()

    payload = paths[stage_name]
    unused = b"an unreferenced stage payload\n"
    digest = hashlib.sha256(unused).hexdigest()
    orphan = payload.parents[1] / digest[:2] / digest
    orphan.parent.mkdir(exist_ok=True)
    orphan.write_bytes(unused)
    head = inventory("--limit", "1")
    tail = inventory("--after", head["next_after"], "--limit", "1")
    observed = {row["artifact_id"]: row for row in head["payloads"] + tail["payloads"]}
    assert set(observed) == {payload.name, digest}
    assert head["retained_bytes"] == original[stage_name].st_size + len(unused)
    assert observed[payload.name]["published_releases"] == 2
    candidates = inventory("--unreferenced")
    assert candidates["payloads"] == [observed[digest]]
    assert candidates["unreferenced_bytes"] == len(unused)

    racing = first.parent / (".freezing-" + "0" * 64)
    payloads = artifact_inventory._shared_payloads

    def start_freeze_during_scan(*args):
        racing.mkdir()
        return payloads(*args)

    with monkeypatch.context() as race:
        race.setattr(artifact_inventory, "_shared_payloads", start_freeze_during_scan)
        incomplete = inventory("--unreferenced")
    assert incomplete["scan_status"] == "incomplete"
    assert incomplete["unreferenced_bytes"] is None and not incomplete["payloads"]
    assert any(row["reason"] == "changed_during_scan" for row in incomplete["problems"])
    assert inventory("--unreferenced")["scan_status"] == "incomplete"
    assert orphan.read_bytes() == unused
    racing.rmdir()

    with monkeypatch.context() as warm:

        def no_replay(*args, **kwargs):
            pytest.fail("verified release replayed unchanged shared members")

        warm.setattr(release_module, "_verify_release_cold", no_replay)
        assert verify_release(first, **proof_options)["release_id"] == first.name

    original_view = lm_view.read_bytes()
    tampered_view = original_view.replace(b"training", b"traininx", 1)
    assert tampered_view != original_view
    stamp = lm_view.stat()
    try:
        lm_view.write_bytes(tampered_view)
        os.utime(lm_view, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        for released in (first, second):
            with pytest.raises(ValueError, match=r"tampered artifact: lm/train\.jsonl"):
                verify_release(released, **proof_options)
    finally:
        lm_view.write_bytes(original_view)

    data = payload.read_bytes()
    changed = data.replace(b"training", b"traininx", 1)
    assert changed != data and len(changed) == len(data)
    stamp = payload.stat()
    payload.write_bytes(changed)
    os.utime(payload, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    for released in (first, second):
        with pytest.raises(ValueError, match="stage output|stage receipt"):
            verify_release(released, **proof_options)


def test_prune_protects_active_and_published_release_members(
    prepared: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from klab.corpus.release_members import (
        SHARED_DOCUMENT,
        shared_payload_lease,
        shared_store,
    )

    build_dir, work = prepared
    body = b"Unreferenced fixture payload.\n"
    digest = hashlib.sha256(body).hexdigest()
    store = shared_store(work, SHARED_DOCUMENT)
    orphan = store / digest[:2] / digest
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(body)
    live = sha256_file(build_dir / "documents.jsonl")
    command = ("release-prune", "--kind", "release-documents", "--digest", digest)

    def prune(*args, blocked=False):
        return _corpus_json(
            work, monkeypatch, capsys, *command, *args, exit_code=int(blocked)
        )

    preview = prune()
    assert preview["status"] == "planned"
    assert preview["payloads"][0]["action"] == "prune"
    assert not (store.parent / ".release-retention.lock").exists()
    assert orphan.exists()

    publish = release_module.publish_shared_members

    def stop_after_move(*args, **kwargs):
        publish(*args, **kwargs)
        busy = prune("--apply", blocked=True)
        assert busy["status"] == "blocked" and not busy["pruned"]
        raise InterruptedError("stopped after shared moves")

    with monkeypatch.context() as fault:
        fault.setattr(release_module, "publish_shared_members", stop_after_move)
        with pytest.raises(InterruptedError, match="stopped after shared moves"):
            freeze(build_dir, work, share_document_artifacts=True)
    pending = prune("--digest", live, "--apply", blocked=True)
    protected = next(row for row in pending["payloads"] if row["artifact_id"] == live)
    assert protected["staging_releases"] == 1
    assert protected["action"] == "protected"
    assert orphan.exists()

    release = freeze(build_dir, work, share_document_artifacts=True)
    published = prune("--digest", live, "--apply", blocked=True)
    protected = next(row for row in published["payloads"] if row["artifact_id"] == live)
    assert protected["published_releases"] == 1
    assert orphan.exists()

    # An older publisher may hold only the per-digest lock.
    with shared_payload_lease(orphan):
        assert prune("--apply", blocked=True)["status"] == "blocked"

    unknown = release.parent / (".freezing-" + "e" * 64)
    from klab.corpus import artifact_inventory

    references = artifact_inventory._release_references

    def change_after_mark(*args, **kwargs):
        result = references(*args, **kwargs)
        unknown.mkdir()
        return result

    try:
        with monkeypatch.context() as race:
            race.setattr(artifact_inventory, "_release_references", change_after_mark)
            changing = prune("--apply", blocked=True)
        assert not changing["payloads"]
        assert any(
            problem["reason"] == "changed_during_scan"
            for problem in changing["problems"]
        )
        assert orphan.exists()
    finally:
        if unknown.exists():
            unknown.rmdir()

    removed = prune("--apply")
    assert removed["pruned"] == [digest]
    assert removed["pruned_bytes"] == len(body) and not orphan.exists()
    assert prune("--apply")["payloads"][0]["action"] == "missing"
    assert verify_release(release)["release_id"] == release.name
