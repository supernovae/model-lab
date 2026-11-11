"""Focused native stage fallback and migration checks."""

import json
from pathlib import Path

import pytest

from sparselab.corpus.admission_inspection_cache import capture
from sparselab.corpus.completed_preparation import _producer
from sparselab.corpus.retained_mixture import _phase
from sparselab.corpus.split_runs import SplitStages, write_rows


def test_owned_native_stage_fallback_refuses_changed_or_present_bad_proof(
    tmp_path, monkeypatch
):
    root = tmp_path / "work"
    root.mkdir(mode=0o700)
    config = tmp_path / "config"
    config.mkdir(mode=0o700)
    key_dir = config / "sparselab"
    key_dir.mkdir(mode=0o700)
    key = key_dir / "verification-key-v1"
    key.write_bytes(bytes(range(32)))
    key.chmod(0o600)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    input_path = root / "input.json"
    input_path.write_text("{}\n")
    input_path.chmod(0o600)
    request = {"producer": _producer(), "inputs": capture([input_path])}
    stages = SplitStages(root / "stages", request, work_root=root)
    payload = stages.phase(
        "build-frozen-family-bindings",
        lambda d: {"files": [write_rows(d / "references.jsonl", [{"record": "tiny"}])]},
    )
    event_path = next(stages.root.glob("completion-*.json"))
    event = json.loads(event_path.read_bytes())
    request_id = {"request": stages.identity, "completion": event["sha256"]}
    cache_path = stages.cache.entry_path("split-phase", request_id)
    raw = cache_path.read_bytes()
    cache_value = json.loads(raw)
    binding = stages.cache.entry_binding(
        cache_path, raw, request_id, capture(cache_value["dependencies"])
    )
    proof_path = stages.cache.proofs.receipt_path(binding)
    assert proof_path.is_file()
    proof_path.unlink()  # Tiny fixture only: simulate an optional receipt not minted.
    result, evidence = _phase(
        root,
        stages.root,
        "build-frozen-family-bindings",
        expected_producer=request["producer"],
        allow_native=True,
    )
    assert result == payload
    assert evidence["method"] == "owned_native_completion"
    assert "proof_path" not in evidence
    with pytest.raises(ValueError):
        _phase(
            root,
            stages.root,
            "build-frozen-family-bindings",
            expected_producer=request["producer"],
        )
    proof_path.write_text("{}\n")
    with pytest.raises((KeyError, ValueError)):
        _phase(
            root,
            stages.root,
            "build-frozen-family-bindings",
            expected_producer=request["producer"],
            allow_native=True,
        )
    proof_path.unlink()
    Path(payload["files"][0]["path"]).write_text('{"record":"tampered"}\n')
    with pytest.raises(ValueError, match="native completion unavailable"):
        _phase(
            root,
            stages.root,
            "build-frozen-family-bindings",
            expected_producer=request["producer"],
            allow_native=True,
        )


def test_stage_migration_reuses_real_files_and_refuses_tampering(tmp_path, monkeypatch):
    import hashlib
    import shutil

    from sparselab.canonical import canonical_digest, canonical_json
    from sparselab.corpus.release_migration import (
        _SOURCE,
        _TARGET,
        _authority,
        _sign,
        adopt_mixture_stages,
        attest_mixture_stages,
        verify_mixture_stage_migration,
    )
    from sparselab.verification_proofs import ProofStore

    root = tmp_path / "source" / "work"
    root.mkdir(parents=True, mode=0o700)
    config_root = tmp_path / "private-config"
    config_root.mkdir(mode=0o700)
    key_dir = config_root / "sparselab"
    key_dir.mkdir(mode=0o700)
    key = key_dir / "verification-key-v1"
    key.write_bytes(bytes(range(32)))
    key.chmod(0o600)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_root))
    input_path = root / "input.json"
    input_path.write_text("{}\n")
    input_path.chmod(0o600)
    request = {"producer": _producer(), "inputs": capture([input_path])}
    workspace = root / "corpora" / "tiny"
    render = SplitStages(
        workspace / "source-build-stages" / "source", request, work_root=root
    )
    snapshot = "a" * 64

    def make_render(directory):
        prepared = directory / "prepared"
        prepared.mkdir()
        docs = write_rows(
            prepared / "docs.jsonl", [{"document_id": "one", "text": "tiny"}]
        )
        receipt = write_rows(
            prepared / "receipt.json",
            [
                {
                    "identity": {"source_id": "source", "snapshot_sha256": snapshot},
                    "provenance": {"build_id": render.identity},
                }
            ],
        )
        return {"prepared": str(prepared), "files": [docs, receipt]}

    render.phase("native-shard-render", make_render)
    stages = SplitStages(workspace / "build-stages", request, work_root=root)
    stages.phase(
        "build-frozen-family-bindings",
        lambda d: {
            "references": str(d / "references.jsonl"),
            "documents": 1,
            "files": [write_rows(d / "references.jsonl", [["one"]])],
        },
    )
    stages.phase(
        "build-duplicate-keys",
        lambda d: {
            "origins": str(d / "origins.jsonl"),
            "files": [write_rows(d / "origins.jsonl", [["one"]])],
        },
    )
    stages.phase(
        "build-deduplicate",
        lambda d: {
            "parents": str(d / "parents.jsonl"),
            "files": [write_rows(d / "parents.jsonl", [[0]])],
        },
    )
    # Simulate missing optional global-stage proofs in the tiny fixture only.
    for event_path in stages.root.glob("completion-*.json"):
        event = json.loads(event_path.read_bytes())
        stage_request = {"request": stages.identity, "completion": event["sha256"]}
        cache_path = stages.cache.entry_path("split-phase", stage_request)
        raw = cache_path.read_bytes()
        value = json.loads(raw)
        binding = stages.cache.entry_binding(
            cache_path, raw, stage_request, capture(value["dependencies"])
        )
        proof = stages.cache.proofs.receipt_path(binding)
        assert proof.is_file()
        proof.unlink()

    scope = {
        "policy": "completed_local_preparation",
        "request": request,
        "source_bindings": capture([input_path]),
        "source_stage_ids": {"source": render.identity},
    }
    identity = {
        "project_id": "tiny",
        "completed_preparation": scope,
        "lock": {"source": {"snapshot_sha256": snapshot}},
    }
    build_id = canonical_digest(identity)
    release = workspace / "builds" / build_id
    release.mkdir(parents=True)
    output = release / "documents.jsonl"
    output.write_bytes(b"{}\n")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    build = {
        "build_id": build_id,
        "identity": identity,
        "files": {
            "documents.jsonl": {"size": output.stat().st_size, "sha256": digest(output)}
        },
    }
    (release / "build.json").write_bytes(canonical_json(build) + b"\n")
    payload = {"corpus_id": "tiny", "build_id": build_id, "build_identity": identity}
    release_id = canonical_digest(payload)
    manifest = {**payload, "release_id": release_id}
    (release / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    (release / "release-publication.json").write_bytes(b"{}\n")
    migration = root / "release-migrations"
    migration.mkdir(mode=0o700)
    store = ProofStore(root)
    source = {
        "kind": _SOURCE,
        "release_id": release_id,
        "source_root": str(root),
        "release_relative_path": release.relative_to(root).as_posix(),
        "source_verification": "native_retained_release_verified",
        "build_sha256": digest(release / "build.json"),
        "manifest_sha256": digest(release / "manifest.json"),
        "publication_sha256": digest(release / "release-publication.json"),
        "migration_authority": _authority(),
        "historical_evidence": {},
        "inputs": {
            "input.json": {
                "size": input_path.stat().st_size,
                "sha256": digest(input_path),
                "basis": "source_sha256",
            }
        },
    }
    source_path = migration / (release_id + ".source.json")
    source_path.write_bytes(canonical_json(_sign(store, source)) + b"\n")
    source_path.chmod(0o600)
    stages_source = attest_mixture_stages(release)
    assert (
        json.loads(stages_source.read_bytes())["observation"]["cohort"]["evidence"][
            "references"
        ]["method"]
        == "owned_native_completion"
    )

    moved = tmp_path / "target" / "work"
    moved.parent.mkdir()
    shutil.copytree(root, moved)
    moved_release = moved / release.relative_to(root)
    target_store = ProofStore(moved)
    paths = [
        moved / "input.json",
        moved_release / "documents.jsonl",
        moved_release / "build.json",
        moved_release / "manifest.json",
        moved_release / "release-publication.json",
    ]
    target = {
        "kind": _TARGET,
        "release_id": release_id,
        "target_root": str(moved),
        "release_relative_path": moved_release.relative_to(moved).as_posix(),
        "source_observation": json.loads(
            (moved / source_path.relative_to(root)).read_bytes()
        ),
        "target_fingerprints": capture(paths),
        "migration_authority": _authority(),
    }
    target_path = moved / "release-migrations" / (release_id + ".target.json")
    target_path.write_bytes(canonical_json(_sign(target_store, target)) + b"\n")
    target_path.chmod(0o600)
    stages_target = adopt_mixture_stages(
        moved_release, moved / stages_source.relative_to(root)
    )
    cohort = verify_mixture_stage_migration(moved_release, manifest)
    assert cohort["source_ids"] == ["source"]
    assert Path(cohort["sources"][0]).is_file()
    progress = stages_target.with_name(release_id + ".stages.target-progress.json")
    before = (
        progress.stat().st_ino,
        progress.stat().st_mtime_ns,
        progress.stat().st_size,
    )
    stages_target.unlink()  # Tiny fixture: simulate interruption before receipt publication.
    assert (
        adopt_mixture_stages(moved_release, moved / stages_source.relative_to(root))
        == stages_target
    )
    assert (
        progress.stat().st_ino,
        progress.stat().st_mtime_ns,
        progress.stat().st_size,
    ) == before
    Path(cohort["sources"][0]).write_text('{"document_id":"one","text":"altered"}\n')
    with pytest.raises(ValueError):
        verify_mixture_stage_migration(moved_release, manifest)
