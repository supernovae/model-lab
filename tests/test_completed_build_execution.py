"""Tiny real-file exercise of retained build recovery after a closed job."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.admission_inspection_cache import capture
from klab.corpus.completed_release import freeze_retained, verify_retained_release
from klab.corpus.project import ProjectConfig
from klab.corpus.split_runs import SplitStages
from klab.training.manifest import sha256_file


def test_retained_build_recovers_from_closed_execution(tmp_path, monkeypatch):
    root = tmp_path / "work"
    root.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    key_dir = tmp_path / "config" / "klab"
    key_dir.mkdir(parents=True, mode=0o700)
    (tmp_path / "config").chmod(0o700)
    key = key_dir / "verification-key-v1"
    key.write_bytes(bytes(range(32)))
    key.chmod(0o600)
    source_checkout = Path(__file__).resolve().parents[1]
    (root / "sources").mkdir(mode=0o777)
    (root / "sources").chmod(0o777)
    (root / "transforms").mkdir()
    scientific = [
        root / "sources/input.yaml",
        root / "transforms/lm.yaml",
        root / "splits.yaml",
        root / "release.yaml",
    ]
    for file in scientific:
        file.write_text("{}\n")
        file.chmod(0o600)
    project = root / "project.yaml"
    config_data = {
        "schema_version": 1,
        "id": "tiny",
        "sources": ["sources/input.yaml"],
        "transforms": ["transforms/lm.yaml"],
        "splits": "splits.yaml",
        "release": "release.yaml",
    }
    project.write_text(yaml.safe_dump(config_data))
    project.chmod(0o600)
    config = ProjectConfig.model_validate(config_data).model_dump(mode="json")
    producer_names = (
        "completed_preparation.py",
        "preparation_inputs.py",
        "streaming_build_rows.py",
        "split_runs.py",
        "large_build.py",
        "pipeline.py",
        "rights.py",
        "project.py",
        "jsonl_records.py",
    )
    producer = {
        name: sha256_file(source_checkout / "src/klab/corpus" / name)
        for name in producer_names
    }
    request = {
        "producer": producer,
        "project": {"root": str(root), "config": config},
        "inputs": capture([project, *scientific]),
    }
    scope = {
        "policy": "completed_local_preparation",
        "request": request,
        "source_bindings": capture([scientific[0]]),
    }
    identity = {"project_id": "tiny", "completed_preparation": scope}
    build_id = canonical_digest(identity)
    build = root / "corpora/tiny/builds" / build_id
    build.mkdir(parents=True)
    data = build / "documents.jsonl"
    data.write_text('{"text":"example"}\n')
    build_json = build / "build.json"
    manifest = {
        "build_id": build_id,
        "identity": identity,
        "stages": [],
        "snapshots": [],
        "files": {
            data.name: {"size": data.stat().st_size, "sha256": sha256_file(data)}
        },
    }
    build_json.write_bytes(canonical_json(manifest) + b"\n")
    stages = SplitStages(root / "corpora/tiny/build-stages", request, work_root=root)
    stages.phase(
        "native-build-publication",
        lambda _: {
            "files": [
                {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for p in (data, build_json)
            ],
            "build_path": str(build),
            "build_id": build_id,
        },
    )
    with pytest.raises(ValueError):
        freeze_retained(build, root)

    evidence = root / "evidence"
    evidence.mkdir()

    def save(name, obj):
        path = evidence / name
        path.write_bytes(obj if isinstance(obj, bytes) else canonical_json(obj) + b"\n")
        return {"path": str(path), "sha256": sha256_file(path)}

    rev, runtime, image = "a" * 40, "b" * 64, "fixture@sha256:" + "c" * 64
    argv = [
        "/opt/venv/bin/python",
        "-B",
        "-m",
        "klab",
        "--work-dir",
        str(root),
        "corpus",
        "build",
        str(project),
        "--offline",
        "--reuse-completed-preparation",
    ]
    source = {"path": str(source_checkout), "commit": rev}
    runtime_spec = {"executable_sha256": runtime}
    spec = {
        "job_id": "fixture",
        "source": source,
        "cwd": str(source_checkout / "src"),
        "argv": argv,
        "runtime": runtime_spec,
        "required_files": {str(p): sha256_file(p) for p in (project, *scientific)},
    }
    refs = {
        "job_spec": save("job-spec.json", spec),
        "stdout": save("stdout", (str(build) + "\n").encode()),
        "stderr": save("stderr", b""),
        "events": save("events", b"{}\n"),
    }
    session_spec = {
        "session_id": "fixture",
        "slots": [{"id": "build", "root": "A", "argv": argv}],
        "roots": {"A": str(root)},
        "execution_policy": {"image": image},
        "runtime_sha256": runtime,
    }
    refs["session_spec"] = save("session-spec.json", session_spec)
    job = {
        "job_id": "fixture",
        "launched": True,
        "reason": "exit",
        "returncode": 0,
        "process_success": True,
        "cleanup_verified": True,
        "evidence_complete": True,
        "logs_complete": True,
        "errors": [],
        "interruption_signal": None,
        "spec_sha256": refs["job_spec"]["sha256"],
        "source": source,
        "runtime": runtime_spec,
        "runtime_identity": {"invocation": argv[0], "executable_sha256": runtime},
    }
    refs["job_report"] = save("job-report.json", job)
    item = {
        "job_id": "fixture",
        "slot": "build",
        "root": "A",
        "launched": True,
        "returncode": 0,
        **{
            field + "_sha256": refs[role]["sha256"]
            for field, role in (
                ("spec", "job_spec"),
                ("report", "job_report"),
                ("stdout", "stdout"),
                ("stderr", "stderr"),
            )
        },
    }
    session = {
        "session_id": "fixture",
        "reason": "complete",
        "cleanup_verified": True,
        "evidence_complete": True,
        "pending_failure": None,
        "incomplete_jobs": [],
        "slots_completed": 1,
        "slots_total": 1,
        "progress": {"events_sha256": refs["events"]["sha256"], "jobs": [item]},
    }
    refs["session_report"] = save("session-report.json", session)
    refs["container_report"] = save(
        "container-report.json",
        {
            "reason": "exit",
            "engine_exit": 0,
            "container_exit": 0,
            "oom_killed": False,
            "cleanup_verified": True,
            "logs_complete": True,
            "errors": [],
            "image": image,
            "session_spec_sha256": refs["session_spec"]["sha256"],
        },
    )
    declaration = root / "corpora/tiny/build-executions" / (build_id + ".json")
    declaration.parent.mkdir()
    declaration.write_bytes(
        canonical_json(
            {
                "kind": "completed_local_build_execution",
                "producer": {
                    "revision": rev,
                    "runtime_sha256": runtime,
                    "image": image,
                },
                "project": {"path": str(project), "sha256": sha256_file(project)},
                "build": {
                    "path": str(build),
                    "build_id": build_id,
                    "manifest_sha256": sha256_file(build_json),
                },
                "evidence": refs,
            }
        )
        + b"\n"
    )
    # A closed but failed job must not become a release merely because its
    # diagnostic files and execution references are self-consistent.
    original = declaration.read_bytes()
    job["returncode"] = 1
    job["process_success"] = False
    refs["job_report"] = save("job-report.json", job)
    item["report_sha256"] = refs["job_report"]["sha256"]
    refs["session_report"] = save("session-report.json", session)
    rejected = json.loads(original)
    rejected["evidence"] = refs
    declaration.write_bytes(canonical_json(rejected) + b"\n")
    with pytest.raises(ValueError):
        freeze_retained(build, root)
    job["returncode"] = 0
    job["process_success"] = True
    save("job-report.json", job)
    item["report_sha256"] = json.loads(original)["evidence"]["job_report"]["sha256"]
    save("session-report.json", session)
    declaration.write_bytes(original)
    before = data.stat()
    assert freeze_retained(build, root) == build
    assert verify_retained_release(build)["release_id"]
    assert data.stat() == before
    assert (
        json.loads((build / "release-publication.json").read_text())["method"]
        == "in_place_adopted_executor_build"
    )
    publication = json.loads((build / "release-publication.json").read_text())
    assert publication["declared_revision"] == rev
    assert "producer_revision" not in publication

    # A migration is a new observation: preserve the original execution and
    # exercise actual files, a clean verifier checkout, and changed inodes.
    from klab.corpus.release import verify_release
    from klab.corpus.release_migration import adopt_target, attest_source

    (root / "sources").chmod(0o700)
    (root / "transforms").chmod(0o700)
    verifier = tmp_path / "verifier"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--no-hardlinks",
            str(source_checkout),
            str(verifier),
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(verifier), "rev-parse", "HEAD"], text=True
    ).strip()
    attestation = attest_source(
        build, verifier_checkout=verifier, verifier_revision=revision
    )
    source_progress = attestation.with_name(
        publication["release_id"] + ".source-progress.json"
    )

    def progress_stamp(path):
        info = path.stat()
        return info.st_ino, info.st_mtime_ns, info.st_size

    source_before = progress_stamp(source_progress)
    attestation.unlink()  # Simulate interruption before final source publication.
    assert (
        attest_source(build, verifier_checkout=verifier, verifier_revision=revision)
        == attestation
    )
    assert progress_stamp(source_progress) == source_before
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    moved_build = moved / build.relative_to(root)
    moved_attestation = moved / attestation.relative_to(root)
    from klab.verification_proofs import ProofStore

    original_attestation = moved_attestation.read_bytes()
    incomplete = json.loads(original_attestation)
    incomplete["observation"]["inputs"].pop(
        next(iter(incomplete["observation"]["inputs"]))
    )
    incomplete["hmac"] = ProofStore(moved).migration_signature(
        incomplete["observation"]
    )
    moved_attestation.write_bytes(canonical_json(incomplete) + b"\n")
    with pytest.raises(ValueError, match="input mapping differs"):
        adopt_target(moved_build, moved_attestation)
    moved_attestation.write_bytes(original_attestation)
    target_receipt = adopt_target(moved_build, moved_attestation)
    source_progress = attestation.with_name(
        publication["release_id"] + ".source-progress.json"
    )
    target_progress = target_receipt.with_name(
        publication["release_id"] + ".target-progress.json"
    )
    saved = (progress_stamp(source_progress), progress_stamp(target_progress))
    target_receipt.unlink()  # Simulate interruption before final target publication.
    assert (
        attest_source(build, verifier_checkout=verifier, verifier_revision=revision)
        == attestation
    )
    assert adopt_target(moved_build, moved_attestation) == target_receipt
    assert (progress_stamp(source_progress), progress_stamp(target_progress)) == saved
    assert verify_release(moved_build)["release_id"] == publication["release_id"]
    # A later compatible Lab update must not force rereading the source host.
    # The real files and signed receipt remain unchanged by this authority change.
    import klab.corpus.release_migration as migration

    with monkeypatch.context() as later_version:
        later_version.setattr(migration, "_authority", lambda: "later-code-version")
        assert verify_release(moved_build)["release_id"] == publication["release_id"]
    assert (
        verify_release(
            moved_build,
            proof_store=ProofStore(moved),
            verification_mode="verified_reuse",
        )["release_id"]
        == publication["release_id"]
    )
    (moved_build / "documents.jsonl").write_text('{"text":"tampered"}\n')
    with pytest.raises(ValueError):
        verify_release(moved_build)
    assert verify_retained_release(build)["release_id"] == publication["release_id"]

    scientific[0].write_text("changed\n")
    with pytest.raises(ValueError):
        verify_retained_release(build)


def test_migration_uses_bound_snapshot_digest(tmp_path):
    from klab.corpus.release_migration import _snapshot_digests

    root = tmp_path / "work"
    release = root / "corpora" / "tiny" / "builds" / "build"
    release.mkdir(parents=True)
    content = b"small retained snapshot\n"
    import hashlib

    digest = hashlib.sha256(content).hexdigest()
    files = [{"path": "part.txt", "size": len(content), "sha256": digest}]
    identity = {"declaration_sha256": "a" * 64, "adapter": "fixture", "files": files}
    snapshot_id = canonical_digest(identity)
    snapshot = root / "corpora" / "tiny" / "snapshots" / "source" / snapshot_id
    (snapshot / "files").mkdir(parents=True)
    (snapshot / "files" / "part.txt").write_bytes(content)
    manifest = {
        **identity,
        "source_id": "source",
        "snapshot_sha256": snapshot_id,
        "declaration": {"kind": "local_files", "acquisition": {}},
    }
    (snapshot / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    manifest_name = (snapshot / "manifest.json").relative_to(root).as_posix()
    data_name = (snapshot / "files" / "part.txt").relative_to(root).as_posix()
    inputs = {
        manifest_name: (snapshot / "manifest.json").stat().st_size,
        data_name: len(content),
    }
    build = {"snapshots": [{"source_id": "source", "sha256": snapshot_id}]}
    assert _snapshot_digests(build, release, inputs) == {data_name: digest}
    manifest["files"][0]["sha256"] = "0" * 64
    (snapshot / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    with pytest.raises(ValueError, match="snapshot manifest identity changed"):
        _snapshot_digests(build, release, inputs)
