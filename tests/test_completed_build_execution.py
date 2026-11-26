"""Tiny real-file exercise of retained build recovery after a closed job."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.admission_inspection_cache import capture
from klab.corpus.completed_release import freeze_retained, verify_retained_release
from klab.corpus.project import ProjectConfig, SplitDeclaration
from klab.corpus.split_freeze import finalize_family_inventory
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
    splits = scientific[2]
    split = SplitDeclaration(
        schema_version=1, unit="document", assignments={"doc": "test"}
    )
    splits.write_text(yaml.safe_dump(split.model_dump(mode="json")))
    split_receipt = splits.with_suffix(".receipt.json")
    split_receipt.write_bytes(
        canonical_json(
            {"split_sha256": sha256_file(splits), "family_candidate_sha256": "a" * 64}
        )
    )
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
        "inputs": capture([project, *scientific, split_receipt]),
    }
    scope = {
        "policy": "completed_local_preparation",
        "request": request,
        "source_bindings": capture([scientific[0]]),
    }
    identity = {
        "project_id": "tiny",
        "completed_preparation": scope,
        "split": split.model_dump(mode="json"),
    }
    build_id = canonical_digest(identity)
    build = root / "corpora/tiny/builds" / build_id
    build.mkdir(parents=True)
    data = build / "documents.jsonl"
    data.write_text('{"text":"example"}\n')
    family = build / "family-inventory.jsonl"
    family.write_bytes(
        canonical_json(
            {
                "document_id": "doc",
                "family_id": "family",
                "split": "test",
                "stratum": "reference",
                "content_sha256": "b" * 64,
            }
        )
        + b"\n"
    )
    report = build / "report.json"
    report.write_bytes(canonical_json({"kept_document_count": 1, "document_count": 1}))
    outputs = (data, family, report)
    build_json = build / "build.json"
    manifest = {
        "build_id": build_id,
        "identity": identity,
        "stages": [],
        "snapshots": [],
        "files": {
            path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in outputs
        },
    }
    build_json.write_bytes(canonical_json(manifest) + b"\n")
    stages = SplitStages(root / "corpora/tiny/build-stages", request, work_root=root)
    stages.phase(
        "native-build-publication",
        lambda _: {
            "files": [
                {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for p in (*outputs, build_json)
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

    finalized = root / "exported-family-inventory.jsonl"
    result = finalize_family_inventory(build, splits, finalized)
    assert result["kept_documents"] == 1
    assert finalized.read_bytes() == family.read_bytes()
    assert sha256_file(finalized) == result["sha256"]

    copy = shutil.copyfileobj

    def change_receipt_after_copy(source, destination):
        copy(source, destination)
        split_receipt.write_bytes(split_receipt.read_bytes())

    interrupted = root / "changed-family-inventory.jsonl"
    with monkeypatch.context() as fault:
        fault.setattr(shutil, "copyfileobj", change_receipt_after_copy)
        with pytest.raises(ValueError, match="inputs changed"):
            finalize_family_inventory(build, splits, interrupted)
    assert not interrupted.exists()

    scientific[0].write_text("changed\n")
    with pytest.raises(ValueError):
        verify_retained_release(build)
