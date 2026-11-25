"""The disk builder resumes from its last verified stage after any failure."""

from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from corpus_fixtures import fixture_rights, neutral_corpus
from recipe_fixtures import Lab, write_recipe

from klab.canonical import canonical_digest, canonical_json
from klab.corpus import (
    admission_inspection_cache,
    build_outputs,
    build_stages,
    completed_build_inputs,
    completed_preparation,
    large_build,
    measurement,
    native_dedup,
    native_index,
    pipeline,
    retained_mixture,
)
from klab.corpus.dedup_policy import STRICT_CHAIN
from klab.corpus.progress import BuildProgress
from klab.corpus.project import SplitDeclaration, source_build_declarations
from klab.corpus.split_runs import SplitStages, write_rows
from klab.corpus.streaming_build_rows import StreamBuildRows
from klab.corpus.view_checkpoints import ViewCheckpoints
from klab.training.manifest import sha256_file
from klab.verification_proofs import ProofStore


def tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class Calls:
    """Count the stage bodies a build runs and fail one on request."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.counts = dict.fromkeys(("index", "dedup", "emit"), 0)
        self.fail: str | None = None
        for stage, module, name in (
            ("index", native_index, "index_prepared"),
            ("dedup", native_dedup, "deduplicate"),
            ("emit", large_build, "_emit_build"),
        ):
            monkeypatch.setattr(
                module, name, self.counted(stage, getattr(module, name))
            )

    def counted(self, stage: str, original):
        def run(*args, **kwargs):
            self.counts[stage] += 1
            if self.fail == stage:
                raise OSError(f"injected failure in {stage}")
            return original(*args, **kwargs)

        return run

    def reset(self) -> dict[str, int]:
        seen, self.counts = self.counts, dict.fromkeys(self.counts, 0)
        return seen


def stage_dirs(work: Path) -> list[str]:
    return sorted(
        f"{path.parent.name}/{path.name}"
        for path in work.glob("corpora/*/builds/stages/*/*")
        if not path.name.startswith(".") and not path.parent.name.startswith(".")
    )


def test_a_build_killed_at_any_stage_resumes_without_redoing_earlier_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    lab = Lab(tmp_path, monkeypatch, capsys)
    recipe = write_recipe(tmp_path / "recipe")
    recipe.hub.install(monkeypatch)
    project = str(recipe.project)
    lab.run("corpus", "budget-init", project)
    lab.run("corpus", "acquire", project)
    calls = Calls(monkeypatch)

    reference = Path(lab.run("corpus", "build", project, "--offline"))
    expected = tree(reference)
    assert calls.reset() == {"index": 1, "dedup": 1, "emit": 1}
    # A successful build prunes its checkpoints.
    assert stage_dirs(lab.work) == []

    shutil.rmtree(reference)
    run_stage = native_index.run_stage
    index_rows = native_index._index_rows
    scanned = []

    def observe_index_rows(connection, group, *args, **kwargs):
        scanned.extend(number for number, _ in group)
        return index_rows(connection, group, *args, **kwargs)

    def stop_after_group(store, stage, *args, **kwargs):
        output = run_stage(store, stage, *args, **kwargs)
        if stage == "index-group":
            raise OSError("interrupted after index group")
        return output

    with monkeypatch.context() as recovery:
        recovery.setattr(native_index, "_INDEX_GROUP_BYTES", 1)
        recovery.setattr(native_index, "_index_rows", observe_index_rows)
        with monkeypatch.context() as fault:
            fault.setattr(native_index, "run_stage", stop_after_group)
            with pytest.raises(OSError, match="interrupted after index group"):
                lab.run("corpus", "build", project, "--offline")
        [committed] = scanned
        assert not reference.exists()
        scanned.clear()
        reference = Path(lab.run("corpus", "build", project, "--offline"))
        assert scanned and committed not in scanned
        assert tree(reference) == expected
        assert not list(lab.work.glob("corpora/*/builds/stages/.index-*.groups"))
    calls.reset()

    shutil.rmtree(reference)
    append_part = native_index._append_index_part
    assembly_operations = []
    stop_on_append = False

    def stop_after_assembly_append(
        connection, part, offsets, *, verify_only, verify_rows
    ):
        output = append_part(
            connection, part, offsets, verify_only=verify_only, verify_rows=verify_rows
        )
        assembly_operations.append((verify_only, verify_rows))
        if stop_on_append and not verify_only:
            raise OSError("interrupted compact index assembly")
        return output

    attest = ProofStore.publish

    def stop_after_assembly_proof(store, receipt, payload, **kwargs):
        saved = attest(store, receipt, payload, **kwargs)
        if payload.get("kind") == "native-index-assembly":
            assert saved and store.load(receipt) == payload
            raise RuntimeError("interrupted signed index assembly")
        return saved

    with monkeypatch.context() as recovery:
        recovery.setattr(native_index, "_INDEX_GROUP_BYTES", 1)
        with monkeypatch.context() as fault:
            fault.setattr(
                native_index, "_append_index_part", stop_after_assembly_append
            )
            with monkeypatch.context() as first:
                first.setattr(ProofStore, "publish", stop_after_assembly_proof)
                with pytest.raises(
                    RuntimeError, match="interrupted signed index assembly"
                ):
                    lab.run("corpus", "build", project, "--offline")
            assert assembly_operations == [(False, True)]
            assert not reference.exists()
            assembly_operations.clear()
            stop_on_append = True
            with pytest.raises(OSError, match="interrupted compact index assembly"):
                lab.run("corpus", "build", project, "--offline")
            assert assembly_operations == [(True, False), (False, False)]
            assert not reference.exists()
        assembly = next(
            lab.work.glob("corpora/*/builds/stages/.index-*.groups/assembly")
        )
        with closing(sqlite3.connect(assembly / "index.sqlite")) as connection:
            connection.execute(
                "UPDATE documents SET span_offset=span_offset+1 WHERE rowid=1"
            )
            connection.commit()
        publish = build_stages.StageStore._publish

        def stop_before_index_publish(store, staging, stage, key, inputs):
            if stage == "index":
                assert (staging / "receipt.json").is_file()
                raise OSError("interrupted index publication")
            publish(store, staging, stage, key, inputs)

        with monkeypatch.context() as fault:
            fault.setattr(
                build_stages.StageStore, "_publish", stop_before_index_publish
            )
            with pytest.raises(OSError, match="interrupted index publication"):
                lab.run("corpus", "build", project, "--offline")
        assert not reference.exists()
        calls.reset()
        reference = Path(lab.run("corpus", "build", project, "--offline"))
        assert calls.reset()["index"] == 0
        assert tree(reference) == expected
        assert not assembly.exists()
    calls.reset()

    shutil.rmtree(reference)
    ranges = []
    committed_range = None

    def observe_range(connection, group, *args, start, max_bytes):
        if max_bytes is not None:
            ranges.append((group[0][0], start))
        return index_rows(connection, group, *args, start=start, max_bytes=max_bytes)

    def stop_inside_prepared_file(store, stage, inputs, produce, **kwargs):
        nonlocal committed_range
        part, result = run_stage(store, stage, inputs, produce, **kwargs)
        if stage == "index-group" and "start" in inputs:
            with closing(sqlite3.connect(part / "index.sqlite")) as connection:
                end = connection.execute(
                    "SELECT document_offset+document_length, span_offset+span_length "
                    "FROM documents ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
            document, span = inputs["generations"][0]
            if end[0] < document[2] and end[1] < span[2]:
                committed_range = (inputs["shards"][0], inputs["start"], end)
                raise OSError("interrupted inside prepared file")
        return part, result

    with monkeypatch.context() as recovery:
        recovery.setattr(native_index, "_INDEX_GROUP_BYTES", 1)
        recovery.setattr(native_index, "_INDEX_RANGE_BYTES", 1)
        recovery.setattr(native_index, "_index_rows", observe_range)
        with monkeypatch.context() as fault:
            fault.setattr(native_index, "run_stage", stop_inside_prepared_file)
            with pytest.raises(OSError, match="interrupted inside prepared file"):
                lab.run("corpus", "build", project, "--offline")
        assert not reference.exists()
        shard, start, end = committed_range
        ranges.clear()
        reference = Path(lab.run("corpus", "build", project, "--offline"))
        assert (shard, start) not in ranges
        assert (shard, end) in ranges
        assert tree(reference) == expected
        assert not list(lab.work.glob("corpora/*/builds/stages/.index-*.groups"))
    calls.reset()

    shutil.rmtree(reference)
    scan_keys = native_dedup._duplicate_keys
    dedup_scans = []
    committed_windows = []
    committed_size = 0
    commit = ViewCheckpoints.commit

    def observe_dedup_scan(connection, method, after, end):
        dedup_scans.append((method, after))
        return scan_keys(connection, method, after, end)

    def stop_after_dedup_windows(checkpoints, state):
        nonlocal committed_size
        commit(checkpoints, state)
        if checkpoints.root.name.startswith(".dedup-"):
            report_size = (checkpoints.root / "build" / "groups.jsonl").stat().st_size
            if report_size > committed_size:
                committed_size = report_size
                committed_windows.append(
                    (dedup_scans[-1], max(checkpoints.root.glob("commit-*.json")))
                )
                if len(committed_windows) == 2:
                    raise OSError("interrupted after dedup windows")

    with monkeypatch.context() as recovery:
        recovery.setattr(native_dedup, "_DEDUP_WINDOW_KEYS", 2)
        recovery.setattr(native_dedup, "_duplicate_keys", observe_dedup_scan)
        with monkeypatch.context() as fault:
            fault.setattr(ViewCheckpoints, "commit", stop_after_dedup_windows)
            with pytest.raises(OSError, match="interrupted after dedup windows"):
                lab.run("corpus", "build", project, "--offline")
        assert not reference.exists()
        intact, (damaged, cursor) = committed_windows
        with cursor.open("r+b") as stream:
            stream.write(b"x")
        dedup_scans.clear()
        reference = Path(lab.run("corpus", "build", project, "--offline"))
        assert intact[0] not in dedup_scans
        assert damaged in dedup_scans
        assert tree(reference) == expected
        assert not list(lab.work.glob("corpora/*/builds/stages/.dedup-*.groups"))
    calls.reset()

    shutil.rmtree(reference)
    final_windows = []

    def stop_in_final_decisions(checkpoints, state):
        commit(checkpoints, state)
        if checkpoints.root.name == "chain-decisions":
            final_windows.append(state["last"])
            raise OSError("interrupted final decision window")

    def stop_after_final_decisions(checkpoints, state):
        commit(checkpoints, state)
        if checkpoints.root.name == "chain-decisions":
            final_windows.append(state["last"])
            if state["complete"]:
                raise OSError("interrupted after final decisions")

    with monkeypatch.context() as recovery:
        recovery.setattr(native_dedup, "_DEDUP_DECISION_ROWS", 2)
        with monkeypatch.context() as fault:
            fault.setattr(ViewCheckpoints, "commit", stop_in_final_decisions)
            with pytest.raises(OSError, match="interrupted final decision window"):
                lab.run("corpus", "build", project, "--offline")
        assert not reference.exists()
        with monkeypatch.context() as fault:
            fault.setattr(ViewCheckpoints, "commit", stop_after_final_decisions)
            with pytest.raises(OSError, match="interrupted after final decisions"):
                lab.run("corpus", "build", project, "--offline")
        assert not reference.exists()
        assert final_windows[0] == 2
        assert final_windows == sorted(set(final_windows))

        def unexpected_group_replay(*args, **kwargs):
            pytest.fail("replayed groups with final decisions already committed")

        recovery.setattr(
            native_dedup, "_resolve_duplicate_groups", unexpected_group_replay
        )
        reference = Path(lab.run("corpus", "build", project, "--offline"))
        assert tree(reference) == expected
        assert not list(lab.work.glob("corpora/*/builds/stages/.dedup-*.groups"))
    calls.reset()

    for failing, redone, kept in (
        ("index", {"index": 1, "dedup": 1, "emit": 1}, set()),
        ("dedup", {"index": 0, "dedup": 1, "emit": 1}, {"index"}),
        ("emit", {"index": 0, "dedup": 0, "emit": 1}, {"index", "dedup"}),
    ):
        shutil.rmtree(reference)
        calls.fail = failing
        with pytest.raises(OSError, match="injected failure"):
            lab.run("corpus", "build", project, "--offline")
        assert {item.split("/")[0] for item in stage_dirs(lab.work)} == kept
        calls.fail = None
        calls.reset()
        reference = Path(lab.run("corpus", "build", project, "--offline"))
        assert calls.reset() == redone, failing
        assert tree(reference) == expected, failing
        assert stage_dirs(lab.work) == []


def test_a_stage_that_does_not_verify_is_rebuilt_and_leftovers_are_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    lab = Lab(tmp_path, monkeypatch, capsys)
    recipe = write_recipe(tmp_path / "recipe")
    recipe.hub.install(monkeypatch)
    project = str(recipe.project)
    lab.run("corpus", "budget-init", project)
    lab.run("corpus", "acquire", project)
    calls = Calls(monkeypatch)
    reference = Path(lab.run("corpus", "build", project, "--offline"))
    expected = tree(reference)
    shutil.rmtree(reference)
    calls.reset()

    calls.fail = "emit"
    with pytest.raises(OSError, match="injected failure"):
        lab.run("corpus", "build", project, "--offline")
    dedup = next(lab.work.glob("corpora/*/builds/stages/dedup/*/dedup.sqlite"))
    stages = dedup.parents[2]
    # A same-size flipped byte, an incomplete directory and an unreceipted
    # directory under an arbitrary name must not become inputs.
    data = bytearray(dedup.read_bytes())
    data[len(data) // 2] ^= 1
    dedup.chmod(0o600)
    dedup.write_bytes(bytes(data))
    (stages / "dedup" / ".incomplete-killed").mkdir()
    (stages / "index" / ("0" * 64)).mkdir()
    calls.fail = None
    calls.reset()
    resumed = Path(lab.run("corpus", "build", project, "--offline"))
    assert calls.reset() == {"index": 0, "dedup": 1, "emit": 1}
    assert tree(resumed) == expected

    shutil.rmtree(resumed)
    lab.run("corpus", "build", project, "--offline", "--keep-stages", "1")
    assert len(stage_dirs(lab.work)) == 2
    calls.reset()

    def shard_receipts():
        return set(lab.work.glob("cache/corpus/normalized/*/*/receipt.json"))

    receipts = shard_receipts()
    assert receipts
    changed_producer = ""

    def bumped_digest(path, *args, **kwargs):
        digest = sha256_file(path, *args, **kwargs)
        if Path(path).name == changed_producer:
            return canonical_digest(digest)
        return digest

    monkeypatch.setattr(pipeline, "sha256_file", bumped_digest)
    monkeypatch.setattr(build_stages, "sha256_file", bumped_digest)
    builds = {resumed}
    for changed_producer, rebuilt_index, rebuilt_dedup, new_shards in (
        ("large_build.py", 0, 0, 0),
        ("dedup_policy.py", 0, 1, 0),
        ("native_index.py", 1, 1, 0),
        ("document_normalization.py", 1, 1, len(receipts)),
    ):
        updated = Path(
            lab.run("corpus", "build", project, "--offline", "--keep-stages", "1")
        )
        assert updated not in builds
        builds.add(updated)
        assert calls.reset() == {
            "index": rebuilt_index,
            "dedup": rebuilt_dedup,
            "emit": 1,
        }
        assert len(shard_receipts() - receipts) == new_shards
        for name in ("documents.jsonl", "spans.jsonl", "lineage.jsonl", "audit.json"):
            assert sha256_file(updated / name) == expected[name]


def test_view_resume_discards_uncommitted_tail_and_reuses_rendered_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    lab = Lab(tmp_path, monkeypatch, capsys)
    recipe = write_recipe(tmp_path / "recipe")
    recipe.hub.install(monkeypatch)
    project = str(recipe.project)
    lab.run("corpus", "budget-init", project)
    lab.run("corpus", "acquire", project)
    monkeypatch.setattr(large_build, "_VIEW_CHECKPOINT_ROWS", 2)

    hash_file = build_outputs.sha256_file

    def hash_untracked(path):
        if path.name in {"documents.jsonl", "spans.jsonl", "lineage.jsonl"} or (
            path.suffix == ".jsonl" and path.parent.name in {"lm", "stages"}
        ):
            pytest.fail(f"manifest reread a checkpointed view: {path}")
        return hash_file(path)

    monkeypatch.setattr(build_outputs, "sha256_file", hash_untracked)
    reference = Path(lab.run("corpus", "build", project, "--offline"))
    expected = tree(reference)
    manifest = json.loads((reference / "build.json").read_bytes())
    assert {name: entry["sha256"] for name, entry in manifest["files"].items()} == {
        name: digest for name, digest in expected.items() if name != "build.json"
    }
    for stage in manifest["stages"]:
        assert stage["output_sha256"] == expected[f"stages/{stage['id']}.jsonl"]
    shutil.rmtree(reference)

    commit = ViewCheckpoints.commit

    def interrupt_after_commit(checkpoints, state):
        commit(checkpoints, state)
        if state.get("count") == 2:
            with (checkpoints.root / "build" / "documents.jsonl").open("ab") as stream:
                stream.write(b"uncommitted tail")
            raise OSError("interrupted view emission")

    with monkeypatch.context() as fault:
        fault.setattr(ViewCheckpoints, "commit", interrupt_after_commit)
        with pytest.raises(OSError, match="interrupted view emission"):
            lab.run("corpus", "build", project, "--offline")
    assert not reference.exists()
    staging = next(lab.work.glob("corpora/*/.views-*/build"))
    assert (staging / "documents.jsonl").read_bytes().endswith(b"uncommitted tail")

    rendered = []
    lineage = large_build._document_lineage

    def count_lineage(document, *args):
        rendered.append(document["document_id"])
        return lineage(document, *args)

    monkeypatch.setattr(large_build, "_document_lineage", count_lineage)

    measured = []
    measure = measurement._measure_shard

    def count_measure(*args):
        measured.append(args[:5])
        return measure(*args)

    monkeypatch.setattr(measurement, "_measure_shard", count_measure)
    with monkeypatch.context() as fault:
        fault.setenv("KLAB_FAULT", f"{measurement.VIEW_STAGE}:1")
        with pytest.raises(RuntimeError, match="KLAB_FAULT after shard 1"):
            lab.run("corpus", "build", project, "--offline")
    total_documents = len((staging / "documents.jsonl").read_bytes().splitlines())
    assert len(rendered) == total_documents - 2
    (first_run,) = measured
    rendered.clear()
    measured.clear()

    def interrupt_publication(staging, target):
        assert (staging / "build.json").exists()
        raise OSError("interrupted publication")

    with monkeypatch.context() as fault:
        fault.setattr(large_build, "publish_staging", interrupt_publication)
        with pytest.raises(OSError, match="interrupted publication"):
            lab.run("corpus", "build", project, "--offline")
    assert not reference.exists()
    assert not rendered
    assert first_run not in measured

    # A same-size edit after view verification must not publish a stale hash.
    # The following attempt must retain the valid prefix and rebuild that group.
    write_manifest = large_build.write_release_files

    def mutate_before_manifest(staging, *args, **kwargs):
        with (staging / "lineage.jsonl").open("r+b") as stream:
            stream.seek(-2, 2)
            stream.write(b"x")
        return write_manifest(staging, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(large_build, "write_release_files", mutate_before_manifest)
        with pytest.raises(ValueError, match="corpus view changed"):
            lab.run("corpus", "build", project, "--offline")
    assert not reference.exists()
    assert not rendered

    # A repaired frontier with identical output can reuse a verified census.
    def unexpected_census(*args, **kwargs):
        pytest.fail("reran completed native census")

    monkeypatch.setattr(measurement, "summarize_release", unexpected_census)
    resumed = Path(lab.run("corpus", "build", project, "--offline"))
    assert len(rendered) == (total_documents % 2 or 2)
    assert tree(resumed) == expected
    assert not list(lab.work.glob("corpora/*/.views-*"))


def test_completed_source_reuses_admission_and_render_with_scoped_proofs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    lab = Lab(tmp_path, monkeypatch, capsys)
    project = neutral_corpus(lab.work)
    source = project.sources[0]
    declarations = source_build_declarations([source])
    source_path = lab.work / "source.jsonl"
    raw_rows = write_rows(
        source_path,
        [
            {
                "text": "A public tutorial explains reliable deployments. " * 4,
                "url": "https://example.org/tutorial",
                "_klab_source": {"source_row_index": 0},
            }
        ],
    )
    admission = lab.work / "admission.json"
    admission.write_bytes(
        canonical_json(
            {
                "schema_version": 2,
                "policy_id": "fixture",
                "policy_sha256": "a" * 64,
                "sources": [
                    {
                        "source_id": source.id,
                        "source_revision": source.revision,
                        "snapshot_sha256": "b" * 64,
                        "sample_path": "source.jsonl",
                        "sample_sha256": raw_rows["sha256"],
                        "rights": source.rights.model_dump(mode="json"),
                        "license_label": source.license,
                        "records": [
                            {
                                "source_row_index": 0,
                                "source_row_sha256": raw_rows["sha256"],
                                "decision": "qualify",
                            }
                        ],
                    }
                ],
            }
        )
        + b"\n"
    )
    manifest_sha = sha256_file(admission)

    def materialize():
        return completed_build_inputs.materialize_admission(
            lab.work / "build-input-stages",
            lab.work,
            admission,
            manifest_sha,
            declarations,
        )["sources"][source.id]

    def render(header, progress):
        stages = completed_build_inputs.source_render_stage(
            lab.work / "source-build-stages",
            lab.work,
            declarations[source.id],
            header,
            source_path,
            project.release,
        )
        return completed_build_inputs.render_source(
            stages,
            source,
            {**header, "admission_sha256": manifest_sha},
            source_path,
            progress,
            project.release,
        )

    initial = materialize()
    fresh = BuildProgress(lab.work / "fresh.jsonl", "fresh")
    rendered = render(initial, fresh)
    assert fresh.input_bytes == raw_rows["bytes"]
    assert rendered["counts"]["documents"] == 1
    stage_root = next((lab.work / "source-build-stages").iterdir())
    adopted, _ = retained_mixture._phase(
        lab.work, stage_root, "native-shard-render", expected_producer=None
    )
    assert adopted == rendered

    def unexpected_snapshot_verification(*args):
        pytest.fail("an input phase depends on the downstream corpus verifier")

    monkeypatch.setattr(
        admission_inspection_cache,
        "verifier_authority",
        unexpected_snapshot_verification,
    )
    resumed = BuildProgress(lab.work / "resumed.jsonl", "retry")
    assert materialize() == initial
    assert render(initial, resumed) == rendered
    assert resumed.input_bytes == 0
    assert "reused_completed_source_render" in resumed.journal.read_text()

    decision_table = Path(initial["decisions"]["data"])
    with decision_table.open("ab") as stream:
        stream.write(b"\n")
    recovered = materialize()
    assert recovered["decisions"]["data"] != str(decision_table)
    assert render(recovered, resumed) == rendered
    assert resumed.input_bytes == 0

    with source_path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="admitted source shard bytes differ"):
        render(recovered, resumed)


def test_completed_preparation_recovers_views_and_signed_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    lab = Lab(tmp_path, monkeypatch, capsys)
    project = neutral_corpus(lab.work)
    policy = project.release.policy.model_copy(update={"duplicates": STRICT_CHAIN})
    project = project.model_copy(
        update={"release": project.release.model_copy(update={"policy": policy})}
    )
    source = project.sources[0]
    prepared = []
    for first in (1, 4):
        shard = lab.work / f"shard-{first}"
        shard.mkdir()
        documents = []
        for index in range(first, first + 3):
            identifier = f"{index:064x}"
            documents.append(
                {
                    "document_id": identifier,
                    "raw_content_sha256": identifier,
                    "content_sha256": identifier,
                    "source_id": source.id,
                    "source_family": source.source_family,
                    "document_kind": "prose",
                    "modality": "text",
                    "domains": source.domains,
                    "text": f"example {index}",
                    "license": "MIT",
                    "redistribution": "allowed",
                    "rights": {
                        **fixture_rights(),
                        "notices": [],
                        "training_restriction": "none_found",
                    },
                }
            )
        write_rows(shard / "docs.jsonl", documents)
        write_rows(
            shard / "spans.jsonl",
            ({"record_id": doc["document_id"]} for doc in documents),
        )
        write_rows(shard / "rejected.jsonl", ())
        prepared.append(shard)
    workspace = lab.work / "corpora" / project.config.id
    request = {"producer": completed_preparation._producer(), "inputs": {}}

    def stages(current=request):
        return SplitStages(workspace / "build-stages", current, work_root=lab.work)

    def declare_assignments(first_split):
        candidates = [
            {
                "document_id": f"{index:064x}",
                "content_sha256": f"{index:064x}",
                "family_id": f"fixture-{index}",
                "stratum": source.domains[0],
                "split": first_split if index == 1 else "train",
            }
            for index in range(1, 7)
        ]
        assignment = write_rows(
            lab.work / f"assignments-{first_split}.jsonl",
            (
                {"document_id": row["document_id"], "split": row["split"]}
                for row in candidates
            ),
        )
        Path(assignment["path"]).rename(
            lab.work / f"assignments-{assignment['sha256']}.jsonl"
        )
        families = write_rows(lab.work / f"families-{first_split}.jsonl", candidates)
        splits = SplitDeclaration.model_validate(
            {
                "schema_version": 1,
                "unit": "document",
                "assignment_file": {key: assignment[key] for key in ("sha256", "rows")},
            }
        )
        return project.model_copy(update={"splits": splits}), families

    project, family_reference = declare_assignments("train")
    original_stages = stages()
    ordered = StreamBuildRows.create(
        original_stages, prepared, project, family_reference=family_reference
    )
    inventory = ordered.final_family_stage(original_stages)

    view_edit = {
        **request,
        "producer": {**request["producer"], "corpus.large_build": "f" * 64},
    }
    view_stages = stages(view_edit)
    same_rows = StreamBuildRows.create(
        view_stages, prepared, project, family_reference=family_reference
    )
    assert same_rows.final_family_stage(view_stages) == inventory
    assert set(view_stages.reused) == {
        "build-document-references",
        "build-join-splits",
        "build-frozen-family-bindings",
        "build-duplicate-keys",
        "build-deduplicate",
        "build-final-family-inventory",
    }
    for name, expected in (
        ("build-frozen-family-bindings", same_rows.assigned),
        ("build-duplicate-keys", view_stages.completed["build-duplicate-keys"]),
        ("build-deduplicate", same_rows.deduplicated),
    ):
        recorded, _ = retained_mixture._phase(
            lab.work, view_stages.root, name, expected_producer=view_edit["producer"]
        )
        assert recorded == expected

    with monkeypatch.context() as edited:

        def resolver_digest(path, *args, **kwargs):
            digest = sha256_file(path, *args, **kwargs)
            return (
                canonical_digest(digest)
                if Path(path).name == "completed_row_dedup.py"
                else digest
            )

        edited.setattr(build_stages, "sha256_file", resolver_digest)
        resolver_stages = stages(
            {
                **request,
                "producer": {
                    **request["producer"],
                    "corpus.completed_row_dedup": "f" * 64,
                },
            }
        )
        new_resolver = StreamBuildRows.create(
            resolver_stages, prepared, project, family_reference=family_reference
        )
        assert resolver_stages.reused == [
            "build-document-references",
            "build-join-splits",
            "build-frozen-family-bindings",
            "build-duplicate-keys",
        ]
        assert (
            new_resolver.final_family_stage(resolver_stages)["files"][0]["sha256"]
            == inventory["files"][0]["sha256"]
        )

    resplit, resplit_families = declare_assignments("validation")
    split_stages = stages({**request, "split": resplit.splits.model_dump(mode="json")})
    changed_rows = StreamBuildRows.create(
        split_stages, prepared, resplit, family_reference=resplit_families
    )
    assert split_stages.reused == ["build-document-references"]
    assert next(changed_rows.family_inventory())["split"] == "validation"

    inputs = [
        *[path for shard in prepared for path in shard.glob("*.jsonl")],
        *[
            item["path"]
            for item in ordered.assigned["files"] + ordered.deduplicated["files"]
        ],
    ]
    identity = {
        "implementation_sha256": "a" * 64,
        "completed_preparation": {
            "policy": "completed_local_preparation",
            "request": request,
            "source_bindings": {},
        },
    }
    build_id = canonical_digest(identity)
    target = workspace / "builds" / build_id
    progress = BuildProgress(workspace / "progress.jsonl", build_id)
    monkeypatch.setattr(large_build, "_VIEW_CHECKPOINT_ROWS", 2)

    def emit(staging, checkpoints):
        large_build._emit_build(
            connection=None,
            staging=staging,
            project=project,
            lock={"sources": {}},
            identity=identity,
            build_id=build_id,
            transform=project.transforms[0],
            prepared=prepared,
            rights_files=[],
            source_events=prepared,
            groups_path=Path(ordered.deduplicated["groups"]),
            group_count=ordered.deduplicated["group_count"],
            dropped=ordered.deduplicated["dropped"],
            progress=progress,
            ordered_rows=ordered,
            descriptive_census=False,
            checkpoints=checkpoints,
        )

    # Fix the build identity and compare to an uninterrupted renderer.
    reference = lab.work / "reference"
    (reference / "build").mkdir(parents=True)
    emit(reference / "build", ViewCheckpoints(reference, "reference"))
    expected = tree(reference / "build")

    def run():
        completed_preparation._publish_completed_views(
            stages(), target, inputs, emit, progress
        )

    commit = ViewCheckpoints.commit

    def stop_after_documents(checkpoints, state):
        commit(checkpoints, state)
        if state["count"] == 2:
            raise OSError("stopped after document checkpoint")

    with monkeypatch.context() as fault:
        fault.setattr(ViewCheckpoints, "commit", stop_after_documents)
        with pytest.raises(OSError, match="stopped after document checkpoint"):
            run()
    assert not target.exists()

    documents = ordered.documents
    read_ids = []

    def observe_document_reads(after):
        for row in documents(after):
            read_ids.append(row[4])
            yield row

    monkeypatch.setattr(ordered, "documents", observe_document_reads)
    publish = completed_preparation.publish_staging

    def stop_before_rename(staging, target, *, before_rename):
        def stop_after_seal():
            before_rename()
            raise OSError("stopped before rename")

        publish(staging, target, before_rename=stop_after_seal)

    with monkeypatch.context() as fault:
        fault.setattr(completed_preparation, "publish_staging", stop_before_rename)
        with pytest.raises(OSError, match="stopped before rename"):
            run()
    assert read_ids == [f"{index:064x}" for index in range(3, 7)]
    assert not target.exists()
    read_ids.clear()

    def stop_after_rename(*args, **kwargs):
        publish(*args, **kwargs)
        raise OSError("stopped after rename")

    with monkeypatch.context() as fault:
        fault.setattr(completed_preparation, "publish_staging", stop_after_rename)
        with pytest.raises(OSError, match="stopped after rename"):
            run()
    assert not read_ids
    with pytest.raises(ValueError, match="completion|build evidence"):
        completed_preparation.verify_completed_build(target)

    # A replacement containing the same bytes still does not prove execution.
    authentic = target.with_name("authentic")
    target.rename(authentic)
    shutil.copytree(authentic, target)
    with pytest.raises(ValueError, match="unsealed completed build"):
        run()
    shutil.rmtree(target)
    authentic.rename(target)

    run()
    assert not read_ids
    assert tree(target) == expected
    assert completed_preparation.verify_completed_build(target)["build_id"] == build_id
    assert not (stages().root / "view-emission").exists()


def test_stage_keys_follow_exactly_their_inputs(tmp_path: Path) -> None:
    project = neutral_corpus(tmp_path)
    base = large_build._stage_inputs(project, [])
    policy = project.release.policy
    flipped = "error" if policy.duplicates.rights_conflict == "allow" else "allow"
    changed = project.model_copy(
        update={
            "release": project.release.model_copy(
                update={
                    "policy": policy.model_copy(
                        update={
                            "duplicates": policy.duplicates.model_copy(
                                update={"rights_conflict": flipped}
                            )
                        }
                    )
                }
            )
        }
    )
    after = large_build._stage_inputs(changed, [])
    # Relaxing a duplicate-only option keeps the index and redoes deduplication.
    assert after["index"] == base["index"]
    assert after["dedup"] != base["dedup"]
    store = build_stages.StageStore(tmp_path / "stages")
    assert store.key(base["index"]) != store.key(base["dedup"])
