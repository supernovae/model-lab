"""Tiny real-file exercise of retained build recovery after a closed job."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from corpus_fixtures import PLAIN_NORMALIZER, STRICT_POLICY, neutral_corpus
from recipe_fixtures import Lab, MockHub, Shard

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.admission_inspection_cache import capture
from klab.corpus.completed_release import freeze_retained, verify_retained_release
from klab.corpus.project import ProjectConfig, SplitDeclaration
from klab.corpus.release import consumed_release, freeze
from klab.corpus.split_freeze import finalize_family_inventory
from klab.corpus.split_runs import SplitStages
from klab.training.manifest import sha256_file


def test_retained_train_selection_reuses_rows_and_limits_ranked_offsets(
    tmp_path, monkeypatch, capsys
):
    from corpus_fixtures import forge_release

    from klab.corpus import split_inventory_adoption
    from klab.corpus.acquisition import acquire
    from klab.corpus.admission_draft import draft_admission_manifest
    from klab.corpus.pipeline import build
    from klab.corpus.project import declaration_payload, load_project
    from klab.corpus.release import verify_release
    from klab.corpus.retained_mixture import RetainedMixtureIndex, prepared_release
    from klab.corpus.split_freeze import freeze_splits
    from klab.corpus.split_inventory import write_split_inventory
    from klab.corpus.split_runs import rows, write_rows

    root = Lab(tmp_path, monkeypatch, capsys).work
    source = declaration_payload(neutral_corpus(root).sources[0])
    source.pop("explicit_training_restriction")
    source_id = source["id"]
    stratum = source["domains"][0]
    shard = Shard(
        "fixture/train.jsonl.gz",
        [
            {
                "text": f"Chapter {number}\n"
                + "A café has a quiet garden. " * number
                + (
                    "Contact fixture@example.org."
                    if number % 2
                    else "Call 415-555-0134."
                ),
                "url": f"https://example.org/chapter/{number}",
                "language": f"label-{number}",
                "score": number,
            }
            for number in range(1, 17)
        ],
    )
    source["acquisition"].update(
        bounded_shards=[
            {
                "path": shard.path,
                "expected_sha256": shard.sha256,
                "max_shard_bytes": len(shard.body),
                "max_scanned_rows": 16,
                "hash_modulus": 1,
                "hash_remainders": [0],
            }
        ],
        qualification={"profile": "web_text", "family_fields": ["url"]},
    )
    hub = MockHub()
    hub.register("fixture/neutral", source["revision"], shard, "fixture")
    hub.install(monkeypatch)

    def write(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        return path

    write("sources/fixture.yaml", source)
    transform = {
        "id": "lm",
        "version": "1",
        "kind": "lm_text",
        "inputs": [],
        "parameters": {},
    }
    write("transforms/lm.yaml", transform)
    write(
        "splits.yaml",
        {"schema_version": 1, "unit": "document", "assignments": {"document": "train"}},
    )
    release = {
        "publication_mode": "metadata_reconstruction_only",
        "normalizer": PLAIN_NORMALIZER,
        "policy": STRICT_POLICY,
        "mixture": {stratum: 1.0},
        "lm": {"selected": True, "training_splits": ["train", "validation"]},
        "chat": {"selected": False, "training_splits": []},
    }
    write("inventory-release.yaml", release)
    config = {
        "schema_version": 1,
        "id": "selected-fixture",
        "sources": ["sources/fixture.yaml"],
        "transforms": ["transforms/lm.yaml"],
        "splits": "splits.yaml",
        "release": "inventory-release.yaml",
    }
    producer = write("inventory.yaml", config)
    project = load_project(producer)
    acquire(project, root)
    policy = root / "policy.md"
    policy.write_text("Reviewed fixture source policy.\n")
    application = root / "application.json"
    application.write_bytes(
        canonical_json(
            {
                "policy_id": "fixture",
                "policy_sha256": sha256_file(policy),
                "sources": [
                    {
                        "source_id": source_id,
                        "license_label": "MIT",
                        "rights": source["rights"],
                    }
                ],
            }
        )
    )
    admission = root / "admission.json"
    draft = draft_admission_manifest(project, root, application, policy, admission)
    assert draft["counts"] == {"qualify": 16, "exclude": 0, "quarantine": 0}
    inventory = root / "inventory.jsonl"
    report = write_split_inventory(project, root, inventory)
    write(
        "release.yaml",
        {
            **release,
            "record_admission": {
                "path": admission.name,
                "sha256": sha256_file(admission),
            },
        },
    )
    reviewed = write("reviewed.yaml", {**config, "release": "release.yaml"})
    # The historical executor handoff is tested separately. Use a real
    # inventory, admission, split, build and signed publication here.
    completion_file = root / "completed-inventory.json"
    completion_file.write_bytes(
        canonical_json(
            {
                "project": {"path": str(producer)},
                "inventory": {"path": str(inventory)},
            }
        )
    )
    completion = {"path": str(completion_file), "sha256": sha256_file(completion_file)}
    monkeypatch.setattr(
        split_inventory_adoption,
        "adopt_completed_inventory",
        lambda *_: SimpleNamespace(
            provenance={"declaration": completion},
            dependencies=capture([inventory]),
            report=report,
            check=lambda *_: None,
        ),
    )
    clusters = root / "clusters.json"
    clusters.write_bytes(
        canonical_json(
            {
                "schema_version": 1,
                "inventory_sha256": sha256_file(inventory),
                "inventory_completion": completion,
                "seed": "fixture",
                "source_strata": {source_id: stratum},
                "split_weights": {"train": 2, "validation": 1, "test": 1},
                "merges": [],
                "reviewer": "Fixture reviewer",
                "reviewed_on": "2026-11-12",
            }
        )
    )
    frozen = root / "frozen.yaml"
    freeze_splits(
        load_project(reviewed),
        root,
        inventory,
        clusters,
        frozen,
        reuse_completed_preparation=True,
    )
    recipe = write(
        "build.yaml", {**config, "release": "release.yaml", "splits": frozen.name}
    )
    baseline = build(load_project(recipe), root, reuse_completed_preparation=True)

    def width(doc):
        return len(doc["text"].encode("utf-8"))

    shortest, disallowed, low_score, *qualified = sorted(
        (doc for doc in rows(baseline / "documents.jsonl") if doc["split"] == "train"),
        key=width,
    )
    assert width(shortest) < width(disallowed)
    marker, category = (
        ("fixture@example.org", "email_like")
        if "fixture@example.org" in qualified[0]["text"]
        else ("415-555-0134", "phone_like")
    )
    kept = [doc for doc in qualified if marker not in doc["text"]]
    assert 0 < len(kept) < len(qualified)
    selection = {
        "allowed": sorted(
            doc["metadata"]["language"] for doc in [low_score, *qualified]
        ),
        "missing": "drop",
    }
    score_floor = qualified[0]["metadata"]["score"]
    transform["parameters"].update(
        minimum_train_utf8_bytes=width(shortest) + 1,
        train_row_language=selection,
        train_source_quality={
            source_id: {"field": "score", "minimum": score_floor, "missing": "drop"}
        },
        train_privacy_candidates=[category],
    )
    expected = {doc["document_id"] for doc in kept}
    write("transforms/lm.yaml", transform)
    ordinary = freeze(build(load_project(recipe), root, offline=True), root)
    assert {
        row["record_id"] for row in rows(ordinary / "lm/train.lineage.jsonl")
    } == expected
    audited = json.loads((ordinary / "report.json").read_bytes())["train_selection"]
    assert audited["excluded_by_reason"] == {
        "below_minimum_utf8_bytes": 1,
        "row_language_not_allowed": 1,
        "row_language_missing": 0,
        "below_source_quality_minimum": 1,
        "source_quality_unavailable": 0,
        "privacy_candidate": len(qualified) - len(kept),
    }
    assert (
        audited["language_sidecars"]
        and audited["quality_sidecars"]
        and audited["privacy_sidecars"]
    )

    def edit_row_label(path):
        def edited():
            for doc in rows(path / "documents.jsonl"):
                if doc["document_id"] == disallowed["document_id"]:
                    doc["metadata"]["language"] = selection["allowed"][0]
                yield doc

        changed = path / "label-edited.jsonl"
        write_rows(changed, edited())
        changed.replace(path / "documents.jsonl")

    with pytest.raises(
        ValueError, match="row language decision differs from pinned source"
    ):
        verify_release(
            forge_release(ordinary, edit_row_label), verification_mode="cold"
        )

    del transform["parameters"]["minimum_train_utf8_bytes"]
    write("transforms/lm.yaml", transform)
    selected = build(load_project(recipe), root, reuse_completed_preparation=True)
    assert freeze(selected, root, reuse_completed_preparation=True) == selected
    assert (selected / "documents.jsonl").read_bytes() == (
        baseline / "documents.jsonl"
    ).read_bytes()
    for split in ("validation", "test"):
        name = f"lm/{split}.jsonl"
        assert (selected / name).read_bytes() == (baseline / name).read_bytes()
    report = json.loads((selected / "report.json").read_bytes())
    assert report["train_selection"]["excluded_by_reason"] == {
        "row_language_not_allowed": 2,
        "row_language_missing": 0,
        "below_source_quality_minimum": 1,
        "source_quality_unavailable": 0,
        "privacy_candidate": len(qualified) - len(kept),
    }
    assert report["mixture_supply"]["strata"][stratum]["train_utf8_bytes"] == sum(
        map(width, kept)
    )
    progress = (
        selected.parent.parent / "progress" / f"{selected.name}.jsonl"
    ).read_text()
    for event in (
        "reused_completed_source_render",
        "reused_build_join_splits",
        "reused_build_deduplicate",
    ):
        assert event in progress

    cohort = prepared_release(root, selected)
    assert {
        row["record_id"] for row in rows(selected / "lm/train.lineage.jsonl")
    } == expected
    scratch = root / "mixture-scratch"
    scratch.mkdir()
    with RetainedMixtureIndex(
        scratch, root, [cohort], 17, {source_id: stratum}
    ) as index:
        assert {row[2] for row in rows(index.ranked)} == expected
    with (selected / "lm/train.lineage.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="JSONL source SHA-256 mismatch"):
        RetainedMixtureIndex(scratch, root, [cohort], 17, {source_id: stratum})

    selection["allowed"] = ["und"]
    write("transforms/lm.yaml", transform)
    empty = build(load_project(recipe), root, reuse_completed_preparation=True)
    with pytest.raises(ValueError, match="declared mixture cannot be supplied"):
        freeze(empty, root, reuse_completed_preparation=True)
    assert not (empty / "manifest.json").exists()


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
    for option in (
        "share_stage_artifacts",
        "share_document_artifacts",
        "share_lm_views",
    ):
        with pytest.raises(ValueError, match="require an ordinary Corpus build"):
            freeze(build, root, **{option: True})
    assert consumed_release(build) is None
    assert data.stat() == before
    assert freeze_retained(build, root) == build
    release_id = verify_retained_release(build)["release_id"]
    from klab.corpus.release_catalog import release_catalog, release_profile

    catalog = release_catalog(root, corpus_id=build.parent.parent.name)
    assert str(build.relative_to(root)) in {
        row["location"] for row in catalog["releases"]
    }
    assert release_profile(root, str(build))["release_id"] == release_id
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
