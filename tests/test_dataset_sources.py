"""Pinned source acquisition, bounded journals, and offline snapshot contracts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from klab.config.models import DatasetConfig
from klab.data import snapshot_writer, sources


def test_published_source_schema_is_current():
    path = Path(__file__).resolve().parents[1] / "schemas/dataset-source-v1.schema.json"
    assert json.loads(path.read_text()) == sources.DatasetSource.model_json_schema()


def declaration(**updates):
    return {
        "repo_id": "fixture/data",
        "revision": "a" * 40,
        "config": "default",
        "splits": {"train": "training", "validation": "heldout"},
        "text_field": "text",
        "attribution": "fixture",
        "license": "MIT",
        "selection": {"mode": "exhaustion"},
        "resources": {
            "max_source_records": 20,
            "max_text_bytes": 100000,
            "max_record_bytes": 10000,
            "max_work_bytes": 10000000,
            "min_free_bytes": 0,
        },
        **updates,
    }


def inventory(repo_id, revision, config, splits, **kwargs):
    return {
        "repo_id": repo_id,
        "revision": revision,
        "loader": "parquet",
        "files": {
            split: [{"path": f"data/{split}.parquet", "size": 42, "blob_id": "b" * 40}]
            for split in splits
        },
    }


@pytest.fixture
def offline(monkeypatch):
    rows = {
        "train": [{"text": "train one"}, {"text": "train two"}],
        "validation": [{"text": "validation one"}],
    }
    monkeypatch.setattr(sources, "hub_identity", inventory)
    monkeypatch.setattr(
        sources, "stream_source", lambda source, split, cache, **kw: iter(rows[split])
    )
    return rows


def acquire(root, payload=None, *, resume=False):
    root.mkdir(parents=True, exist_ok=True)
    config = root / "source.yaml"
    lock = root / "lock.json"
    if not resume:
        config.write_text(yaml.safe_dump(payload or declaration()))
        sources.lock_source(config, lock, root / "cache")
    return sources.snapshot_source(
        lock, root / "snapshot", root / "cache", resume=resume
    )


def runtime(manifest):
    return DatasetConfig(
        source="snapshot",
        revision="a" * 40,
        license="MIT",
        dataset_config="default",
        source_manifest_path=manifest,
        train_path=manifest.parent / "train.jsonl",
        validation_path=manifest.parent / "validation.jsonl",
        cache_dir=manifest.parent.parent / "prepared",
        train_max_documents=20,
        validation_max_documents=20,
        train_max_tokens=1000,
        validation_max_tokens=1000,
    )


def test_exhaustion_and_relocation(tmp_path, offline):
    manifest = acquire(tmp_path)
    verified = sources.verify_snapshot(runtime(manifest))
    assert verified["source_exhausted"] is True
    assert list(sources.iter_snapshot(runtime(manifest), "train")) == [
        "train one",
        "train two",
    ]
    shutil.copytree(manifest.parent, tmp_path / "moved")
    moved = tmp_path / "moved/manifest.json"
    assert sources.verify_snapshot(runtime(moved)) == verified


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "c" * 40),
        ("dataset_config", "other"),
        ("train_path", "other.jsonl"),
    ],
)
def test_config_binding_rejects_changes(tmp_path, offline, field, value):
    manifest = acquire(tmp_path)
    if field.endswith("path"):
        path = tmp_path / value
        path.write_bytes((manifest.parent / "train.jsonl").read_bytes())
        value = path
    bad = runtime(manifest).model_copy(update={field: value})
    with pytest.raises(ValueError, match="mismatch"):
        sources.verify_snapshot(bad)


@pytest.mark.parametrize("member", ["train.jsonl", "events.jsonl", "excluded.jsonl"])
def test_changed_inventory_fails(tmp_path, offline, member):
    manifest = acquire(tmp_path)
    target = manifest.parent / member
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        sources.verify_snapshot(manifest)


def test_extra_split_is_verified(tmp_path, offline):
    offline["test"] = [{"text": "third split"}]
    payload = declaration(
        splits={"train": "training", "validation": "heldout", "test": "test"},
        dedup={"priority": ["validation", "train", "test"]},
    )
    manifest = acquire(tmp_path, payload)
    assert sources.verify_snapshot(runtime(manifest))["splits"]["test"]["count"] == 1
    (manifest.parent / "test.jsonl").unlink()
    with pytest.raises(ValueError, match="inventory"):
        sources.verify_snapshot(manifest)


def test_reserved_split_names_rejected():
    with pytest.raises(ValueError, match="split mapping"):
        sources.DatasetSource.model_validate(
            declaration(splits={"events": "upstream"}, dedup={"priority": ["events"]})
        )


@pytest.mark.parametrize(
    "suffix",
    [
        "license: other\n",
        "true: value\n",
    ],
)
def test_strict_yaml_rejects_ambiguous_keys(tmp_path, suffix):
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(declaration()) + suffix)
    with pytest.raises((ValueError, TypeError)):
        sources.load_source(path)


@pytest.mark.parametrize(
    ("artifact", "extra"), [("lock", "duplicate"), ("snapshot", "nonfinite")]
)
def test_locked_documents_use_strict_json(tmp_path, offline, artifact, extra):
    manifest = acquire(tmp_path)
    path = tmp_path / "lock.json" if artifact == "lock" else manifest
    raw = path.read_text().rstrip()
    if extra == "duplicate":
        field = ',"format":' + json.dumps(json.loads(raw)["format"])
    else:
        field = ',"bad":NaN'
    path.write_text(raw[:-1] + field + "}")
    with pytest.raises(ValueError, match="duplicate JSON|non-finite"):
        (sources.verify_lock if artifact == "lock" else sources.verify_snapshot)(path)


@pytest.mark.parametrize("policy", ["exclude", "error"])
def test_overlap_keep_does_not_bypass_within_split(tmp_path, offline, policy):
    offline["validation"] = [{"text": "shared"}]
    offline["train"] = [{"text": "shared"}, {"text": "shared"}]
    payload = declaration(dedup={"overlap": "keep", "within_split": policy})
    if policy == "error":
        with pytest.raises(ValueError, match="forbidden duplicate"):
            acquire(tmp_path, payload)
        assert not (tmp_path / "snapshot").exists()
    else:
        manifest = acquire(tmp_path, payload)
        result = sources.verify_snapshot(manifest)
        assert result["splits"]["train"]["count"] == 1
        assert result["splits"]["train"]["duplicate"] == 1


@pytest.mark.parametrize(
    "limit,value",
    [
        ("max_source_records", 1),
        ("max_text_bytes", 5),
        ("max_record_bytes", 5),
        ("max_work_bytes", 1024),
    ],
)
def test_resource_caps_never_publish(tmp_path, offline, limit, value):
    payload = declaration()
    payload["resources"][limit] = value
    with pytest.raises(ValueError, match="resource bound"):
        acquire(tmp_path, payload)
    assert not (tmp_path / "snapshot").exists()
    assert (tmp_path / ".snapshot.work").is_dir()
    with pytest.raises(ValueError, match="resource bound"):
        acquire(tmp_path, resume=True)
    assert not (tmp_path / "snapshot").exists()


def test_exact_record_cap_can_establish_eof(tmp_path, offline):
    payload = declaration()
    payload["resources"]["max_source_records"] = 2
    assert (
        sources.verify_snapshot(acquire(tmp_path, payload))["source_exhausted"] is True
    )


def test_document_target_does_not_claim_exhaustion(tmp_path, offline):
    payload = declaration(
        selection={"mode": "bounded", "documents": {"train": 2, "validation": 1}}
    )
    result = sources.verify_snapshot(acquire(tmp_path, payload))
    assert result["source_exhausted"] is False
    assert result["splits"]["train"]["stop_reason"] == "document_target"


@pytest.mark.parametrize("drift", [False, True])
def test_interruption_replays_committed_prefix(tmp_path, offline, monkeypatch, drift):
    monkeypatch.setattr(snapshot_writer, "_SNAPSHOT_BATCH_RECORDS", 1)

    def interrupted(source, split, cache, **kw):
        yield offline[split][0]
        raise InterruptedError("fixture interruption")

    monkeypatch.setattr(sources, "stream_source", interrupted)
    with pytest.raises(InterruptedError):
        acquire(tmp_path)
    assert not (tmp_path / "snapshot").exists()
    if drift:
        offline["validation"][0] = {"text": "changed upstream"}
    monkeypatch.setattr(
        sources,
        "stream_source",
        lambda source, split, cache, **kw: iter(offline[split]),
    )
    if drift:
        with pytest.raises(ValueError, match="prefix mismatch"):
            acquire(tmp_path, resume=True)
    else:
        manifest = acquire(tmp_path, resume=True)
        assert sources.verify_snapshot(manifest)["splits"]["train"]["count"] == 2


def test_interrupted_export_replays_without_duplicate_output(
    tmp_path, offline, monkeypatch
):
    original = snapshot_writer._rename_noreplace

    def fail_publication(source, destination):
        if destination.name == "snapshot":
            raise InterruptedError("export interrupted")
        return original(source, destination)

    monkeypatch.setattr(snapshot_writer, "_rename_noreplace", fail_publication)
    with pytest.raises(InterruptedError):
        acquire(tmp_path)
    monkeypatch.setattr(snapshot_writer, "_rename_noreplace", original)
    manifest = acquire(tmp_path, resume=True)
    assert sources.verify_snapshot(manifest)["splits"]["train"]["count"] == 2


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/data.parquet",
        "hf://datasets/fixture/data@main/a.parquet",
        "hf://datasets/other/data@" + "a" * 40 + "/a.parquet",
        "https://huggingface.co/datasets/fixture/data/resolve/"
        + "a" * 40
        + "/../a.parquet",
    ],
)
def test_inventory_rejects_external_or_unpinned_files(url):
    source = sources.DatasetSource.model_validate(declaration())
    with pytest.raises(ValueError):
        sources._hub_file(source.repo_id, source.revision, url)


def test_hub_lock_resolves_file_inventory_and_stream_uses_only_it(
    tmp_path, monkeypatch
):
    source = sources.DatasetSource.model_validate(declaration())
    files = {
        name: [f"hf://datasets/{source.repo_id}@{source.revision}/data/{name}.parquet"]
        for name in source.splits.values()
    }
    monkeypatch.setattr(
        "huggingface_hub.HfApi.dataset_info",
        lambda *a, **kw: SimpleNamespace(
            sha=source.revision,
            siblings=[
                SimpleNamespace(
                    rfilename=f"data/{name}.parquet",
                    size=42,
                    blob_id="b" * 40,
                    lfs=SimpleNamespace(sha256="c" * 64),
                )
                for name in source.splits.values()
            ],
        ),
    )
    monkeypatch.setattr(
        "datasets.load_dataset_builder",
        lambda *a, **kw: SimpleNamespace(
            info=SimpleNamespace(builder_name="parquet"),
            config=SimpleNamespace(data_files=files),
        ),
    )
    identity = sources.hub_identity(
        source.repo_id, source.revision, source.config, source.splits
    )
    assert identity["files"]["train"][0]["path"] == "data/training.parquet"
    assert identity["files"]["train"][0]["lfs_sha256"] == "c" * 64

    def load(files, options, repo_id, cache, auth):
        assert repo_id == source.repo_id
        assert cache == tmp_path
        assert files == [
            f"https://huggingface.co/datasets/{source.repo_id}/resolve/{source.revision}/data/training.parquet"
        ]
        return iter([{"text": "locked data"}])

    monkeypatch.setattr(sources, "_stream_parquet", load)
    monkeypatch.setattr(
        "datasets.load_dataset", lambda *a, **kw: pytest.fail("threaded parquet loader")
    )
    assert list(
        sources.stream_source(source, "train", tmp_path, inventory=identity)
    ) == [{"text": "locked data"}]


@pytest.mark.parametrize("bounded", [False, True])
def test_parquet_reads_synchronously_and_closes_on_bounded_stop(
    tmp_path, monkeypatch, bounded
):
    import io
    import threading

    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "fixture.parquet"
    pq.write_table(
        pa.table({"text": [f"row {i}" for i in range(12)]}),
        parquet_path,
        row_group_size=3,
    )
    owner = threading.get_ident()
    handles = []

    class CheckedFile(io.BytesIO):
        def read(self, *args):
            assert threading.get_ident() == owner, "asynchronous Arrow read"
            return super().read(*args)

        def readinto(self, *args):
            assert threading.get_ident() == owner, "asynchronous Arrow readinto"
            return super().readinto(*args)

        def seek(self, *args):
            assert threading.get_ident() == owner, "asynchronous Arrow seek"
            return super().seek(*args)

    def open_file(path, mode, **kwargs):
        handle = CheckedFile(parquet_path.read_bytes())
        handles.append(handle)
        return handle

    monkeypatch.setattr("datasets.utils.file_utils.xopen", open_file)
    source = sources.DatasetSource.model_validate(declaration())
    stream = sources._stream_parquet(
        ["first", "second"], {"batch_size": 2}, source.repo_id, tmp_path, {}
    )
    if bounded:
        assert next(stream) == {"text": "row 0"}
        stream.close()
        assert len(handles) == 1
    else:
        assert list(stream) == [{"text": f"row {i}"} for i in range(12)] * 2
        assert len(handles) == 2
    assert all(handle.closed for handle in handles)


def test_parquet_preserves_filter_projection_and_feature_cast(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Features, Value

    path = tmp_path / "fixture.parquet"
    pq.write_table(
        pa.table({"text": [1, 2, 3, 4], "selected": [False, True, True, False]}),
        path,
        row_group_size=2,
    )
    source = sources.DatasetSource.model_validate(declaration())
    options = {
        "batch_size": 1,
        "columns": ["text"],
        "filters": [["selected", "=", True]],
        "features": Features({"text": Value("string")}).to_dict(),
    }
    assert list(
        sources._stream_parquet([str(path)], options, source.repo_id, tmp_path, {})
    ) == [{"text": "2"}, {"text": "3"}]


@pytest.mark.parametrize("declared_batch_size", [None, 1000000])
def test_parquet_caps_decoder_batches(tmp_path, monkeypatch, declared_batch_size):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "fixture.parquet"
    pq.write_table(pa.table({"text": ["row"] * 3000}), path, row_group_size=3000)
    original = pq.ParquetFile.iter_batches
    observed = []

    def batches(self, **kwargs):
        observed.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", batches)
    stream = sources._stream_parquet(
        [str(path)],
        {"batch_size": declared_batch_size},
        sources.DatasetSource.model_validate(declaration()).repo_id,
        tmp_path,
        {},
    )
    assert next(stream) == {"text": "row"}
    stream.close()
    assert observed == [
        {
            "batch_size": min(1024, declared_batch_size or 1024),
            "columns": None,
            "use_threads": False,
        }
    ]


def test_bounded_parquet_subprocess_exits(tmp_path):
    import subprocess
    import sys

    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "fixture.parquet"
    pq.write_table(pa.table({"text": ["row"] * 5000}), path, row_group_size=5000)
    program = """
import io, json, sys
from pathlib import Path
from unittest.mock import patch
from klab.data.sources import DatasetSource, _stream_parquet
path = Path(sys.argv[1])
source = DatasetSource.model_validate(json.loads(sys.argv[2]))
with patch('datasets.utils.file_utils.xopen',
           side_effect=lambda *a, **k: io.BytesIO(path.read_bytes())):
    stream = _stream_parquet([str(path)], {}, source.repo_id, path.parent, {})
    assert next(stream) == {'text': 'row'}
    stream.close()
print('complete', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(path), json.dumps(declaration())],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert result.stdout.strip() == "complete"


def test_forge_locations_do_not_change_scientific_lock(tmp_path, offline, monkeypatch):
    monkeypatch.setattr(
        sources,
        "forge_bindings",
        lambda source: {split: "c" * 64 for split in source.splits},
    )
    payload = declaration(
        kind="forge_files",
        forge_files={
            split: {"snapshot_path": f"first/{split}", "files": ["data.jsonl"]}
            for split in ("train", "validation")
        },
    )
    first = acquire(tmp_path / "first", payload)
    for binding in payload["forge_files"].values():
        binding["snapshot_path"] = "relocated"
    second = acquire(tmp_path / "second", payload)
    assert first.read_bytes() == second.read_bytes()
    assert (tmp_path / "first/lock.json").read_bytes() == (
        tmp_path / "second/lock.json"
    ).read_bytes()
    assert b"snapshot_path" not in first.read_bytes()


def test_existing_snapshot_work_requires_explicit_resume_without_rewriting(
    tmp_path, offline, monkeypatch
):
    import errno
    import fcntl

    def interrupted(source, split, cache, **kw):
        yield offline[split][0]
        raise InterruptedError("fixture interruption")

    monkeypatch.setattr(sources, "stream_source", interrupted)
    with pytest.raises(InterruptedError):
        acquire(tmp_path)
    work = tmp_path / ".snapshot.work"
    before = {
        p.relative_to(work): p.read_bytes() for p in work.rglob("*") if p.is_file()
    }
    lock = tmp_path / "lock.json"
    lock_bytes = lock.read_bytes()
    with pytest.raises(FileExistsError, match="--resume.*original lock") as refused:
        sources.snapshot_source(lock, tmp_path / "snapshot", tmp_path / "cache")
    assert refused.value.errno == errno.EEXIST
    assert refused.value.filename == str(work)
    assert isinstance(refused.value.__cause__, FileExistsError)
    assert not (tmp_path / "snapshot").exists()
    assert {
        p.relative_to(work): p.read_bytes() for p in work.rglob("*") if p.is_file()
    } == before
    assert lock.read_bytes() == lock_bytes
    # Explicit resume remains subject to the existing live journal lock.
    with (work / "guard").open("r+b") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            acquire(tmp_path, resume=True)
    # An altered declaration cannot use the old journal identity.
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(declaration(revision="b" * 40)))
    changed_lock = tmp_path / "changed-lock.json"
    sources.lock_source(changed, changed_lock, tmp_path / "cache")
    with pytest.raises(ValueError, match="resume lock mismatch"):
        sources.snapshot_source(
            changed_lock, tmp_path / "snapshot", tmp_path / "cache", resume=True
        )
    assert lock.read_bytes() == lock_bytes
    assert {
        p.relative_to(work): p.read_bytes() for p in work.rglob("*") if p.is_file()
    } == before
    monkeypatch.setattr(
        sources,
        "stream_source",
        lambda source, split, cache, **kw: iter(offline[split]),
    )
    manifest = acquire(tmp_path, resume=True)
    assert sources.verify_snapshot(manifest)["splits"]["train"]["count"] == 2


def test_guard_only_snapshot_work_does_not_promise_resume_or_cleanup(
    tmp_path, offline, monkeypatch
):
    import errno
    import fcntl

    def refuse(*args, **kwargs):
        raise ValueError("fixture admission refusal")

    monkeypatch.setattr(snapshot_writer, "_admit", refuse)
    with pytest.raises(ValueError, match="fixture admission refusal"):
        acquire(tmp_path)
    work = tmp_path / ".snapshot.work"
    assert not (work / "journal.sqlite").exists()
    guard = work / "guard"
    before = {p.name: (p.stat().st_ino, p.read_bytes()) for p in work.iterdir()}
    assert set(before) == {"guard"}
    with guard.open("r+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            FileExistsError, match="no regular resume journal"
        ) as caught:
            sources.snapshot_source(
                tmp_path / "lock.json", tmp_path / "snapshot", tmp_path / "cache"
            )
        assert "Another initializer may still be active" in str(caught.value)
        assert "does not authorize deleting" in str(caught.value)
        assert "then use --resume" not in str(caught.value)
        assert caught.value.errno == errno.EEXIST
        assert caught.value.filename == str(work)
        assert isinstance(caught.value.__cause__, FileExistsError)
        with pytest.raises(BlockingIOError):
            sources.snapshot_source(
                tmp_path / "lock.json",
                tmp_path / "snapshot",
                tmp_path / "cache",
                resume=True,
            )
    assert {p.name: (p.stat().st_ino, p.read_bytes()) for p in work.iterdir()} == before
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize(
    ("bound", "crash"),
    [("records", False), ("bytes", False), ("time", False), ("records", True)],
)
def test_batched_snapshot_recovery_matches_uninterrupted(
    tmp_path, offline, monkeypatch, crash, bound
):
    import itertools
    import multiprocessing
    import os
    import sqlite3

    offline["validation"] = [{"text": f"heldout {i}"} for i in range(8)]
    offline["train"] = [
        {"text": "heldout 0"},
        {"text": "unique"},
        {"text": "unique"},
        {"text": None},
        {"text": ""},
    ]
    expected = acquire(tmp_path / "expected")
    if bound == "records":
        monkeypatch.setattr(snapshot_writer, "_SNAPSHOT_BATCH_RECORDS", 2)
    elif bound == "bytes":
        monkeypatch.setattr(snapshot_writer, "_SNAPSHOT_BATCH_BYTES", 70000)
    else:
        ticks = itertools.count()
        monkeypatch.setattr(snapshot_writer, "monotonic", lambda: next(ticks) * 0.1)
    root = tmp_path / "resumed"

    def interrupted(source, split, cache, **kw):
        yield from offline[split][:5]
        if crash:
            os._exit(17)
        raise InterruptedError("fixture interruption")

    monkeypatch.setattr(sources, "stream_source", interrupted)
    if crash:
        process = multiprocessing.get_context("fork").Process(
            target=acquire, args=(root,)
        )
        process.start()
        process.join(20)
        assert process.exitcode == 17
    else:
        with pytest.raises(InterruptedError):
            acquire(root)
    assert not (root / "snapshot").exists()
    with sqlite3.connect(root / ".snapshot.work/journal.sqlite") as db:
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        saved = db.execute("SELECT id FROM events ORDER BY id").fetchall()
        assert 0 < len(saved) < 5
        assert saved == [(i,) for i in range(len(saved))]
    monkeypatch.setattr(
        sources,
        "stream_source",
        lambda source, split, cache, **kw: iter(offline[split]),
    )
    actual = acquire(root, resume=True)
    assert sources.verify_snapshot(actual) == sources.verify_snapshot(expected)
    assert {p.name: p.read_bytes() for p in actual.parent.iterdir()} == {
        p.name: p.read_bytes() for p in expected.parent.iterdir()
    }


def test_record_larger_than_batch_is_admitted_alone(tmp_path, offline, monkeypatch):
    monkeypatch.setattr(snapshot_writer, "_SNAPSHOT_BATCH_BYTES", 1)
    offline["train"] = [{"text": "x" * 9000}]
    result = sources.verify_snapshot(acquire(tmp_path))
    assert result["splits"]["train"]["text_bytes"] == 9000


@pytest.mark.parametrize("phase", ["commit", "export"])
def test_capacity_loss_retains_recoverable_journal(
    tmp_path, offline, monkeypatch, phase
):
    import sqlite3

    admit = snapshot_writer._admit

    def lose_capacity(work, cache, source, *, growth=0):
        # A second connection can only observe committed events.
        journal = work / "journal.sqlite"
        if journal.exists():
            with sqlite3.connect(journal) as db:
                saved = db.execute("SELECT count(*) FROM events").fetchone()[0]
            if (phase == "commit" and growth == 0) or (
                phase == "export" and (work / "publication").exists()
            ):
                assert saved == (0 if phase == "commit" else 3)
                raise ValueError("resource bound: insufficient free bytes")
        return admit(work, cache, source, growth=growth)

    monkeypatch.setattr(snapshot_writer, "_admit", lose_capacity)
    with pytest.raises(ValueError, match="insufficient free bytes"):
        acquire(tmp_path)
    assert not (tmp_path / "snapshot").exists()
    monkeypatch.setattr(snapshot_writer, "_admit", admit)
    assert sources.verify_snapshot(acquire(tmp_path, resume=True))["source_exhausted"]


def test_batch_allowance_shrinks_to_available_work_budget(tmp_path, offline):
    payload = declaration()
    payload["resources"]["max_work_bytes"] = 65536
    result = sources.verify_snapshot(acquire(tmp_path, payload))
    assert result["splits"]["train"]["count"] == 2
    assert (
        sum(
            p.stat().st_size
            for p in (tmp_path / ".snapshot.work").rglob("*")
            if p.is_file()
        )
        <= 65536
    )


@pytest.fixture
def shared_tokenizer(tmp_path, offline):
    from klab.config.loading import load_config
    from klab.config.models import RunConfig, TokenizerTrainConfig
    from klab.data.tokenizer import train_tokenizer
    from klab.training.manifest import sha256_file

    origin = runtime(acquire(tmp_path / "origin"))
    path = train_tokenizer(
        TokenizerTrainConfig(
            schema_version=1,
            vocab_size=270,
            min_frequency=1,
            max_documents=2,
            output_dir=tmp_path / "tokenizer",
            dataset=origin,
        )
    )
    offline["train"] = [{"text": "a different training story"}]
    offline["validation"] = [{"text": "a different heldout story"}]
    consumer = runtime(acquire(tmp_path / "consumer"))
    base = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    )
    raw = base.model_dump(mode="json")
    raw["model"]["vocab_size"] = 270
    raw["dataset"] = consumer.model_dump(mode="json")
    raw["tokenizer"] = {
        "path": str(path),
        "origin": {
            "dataset": origin.model_dump(mode="json"),
            "sha256": sha256_file(path),
            "manifest_sha256": sha256_file(path.with_name("tokenizer_manifest.json")),
        },
    }
    config = RunConfig.model_validate(raw)
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    return config, config_path


def test_shared_tokenizer_verification_and_public_preparation(
    shared_tokenizer, capsys, monkeypatch, tmp_path
):
    import sys

    from klab.cli.main import main
    from klab.data.tokenizer_cli import verify_configured_tokenizer
    from klab.training.manifest import sha256_file

    config, path = shared_tokenizer
    before = sha256_file(config.tokenizer.path)
    result = verify_configured_tokenizer(path)
    assert result["origin"]["source_manifest_sha256"] == sha256_file(
        config.tokenizer.origin.dataset.source_manifest_path
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "klab",
            "--work-dir",
            str(tmp_path / "work"),
            "data",
            "prepare",
            str(path),
        ],
    )
    main()
    packed = Path(capsys.readouterr().out.strip().splitlines()[-1])
    manifest = json.loads((packed / "manifest.json").read_text())
    from klab.training.manifest import canonical_digest

    assert manifest["cache_identity"]["snapshot_source_sha256"] == canonical_digest(
        sources.verify_snapshot(config.dataset)
    )
    assert sha256_file(config.tokenizer.path) == before
    # A cache hit must reauthenticate the origin, not trust old packed arrays.
    config.tokenizer.origin.dataset.train_path.write_text("tampered")
    with pytest.raises(ValueError):
        main()


@pytest.mark.parametrize("target", ["tokenizer", "manifest", "origin", "consumer"])
def test_shared_tokenizer_rejects_tampered_inputs(shared_tokenizer, target):
    from klab.data.packing import prepare_data
    from klab.data.tokenizer import load_tokenizer

    config, _ = shared_tokenizer
    tokenizer = load_tokenizer(config.tokenizer.path)
    path = {
        "tokenizer": config.tokenizer.path,
        "manifest": config.tokenizer.path.with_name("tokenizer_manifest.json"),
        "origin": config.tokenizer.origin.dataset.train_path,
        "consumer": config.dataset.train_path,
    }[target]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        prepare_data(config, tokenizer)


def test_repeat_token_uses_a_tokenizer_fitted_on_a_snapshot(shared_tokenizer):
    from klab.config.models import RunConfig
    from klab.data.packing import prepare_data
    from klab.data.tokenizer import load_tokenizer

    config, _ = shared_tokenizer
    raw = config.model_dump(mode="json")
    raw["dataset"] = {
        "source": "synthetic",
        "synthetic_task": "repeat_token",
        "synthetic_repeat_length": 4,
        "cache_dir": raw["dataset"]["cache_dir"],
        "train_max_documents": 8,
        "validation_max_documents": 4,
        "train_max_tokens": 64,
        "validation_max_tokens": 32,
    }
    spam = RunConfig.model_validate(raw)
    prepared = prepare_data(spam, load_tokenizer(spam.tokenizer.path))
    assert prepared.manifest["repeat_token_symbols"] >= 2
    # The origin is still checked: a tampered origin snapshot is refused.
    spam.tokenizer.origin.dataset.train_path.write_text("tampered")
    with pytest.raises(ValueError):
        prepare_data(spam, load_tokenizer(spam.tokenizer.path))


def test_shared_tokenizer_requires_correct_explicit_origin(shared_tokenizer):
    from klab.config.models import RunConfig
    from klab.data.tokenizer import verify_run_tokenizer

    config, _ = shared_tokenizer
    raw = config.model_dump(mode="json")
    raw["tokenizer"].pop("origin")
    with pytest.raises(ValueError, match="snapshot identity mismatch"):
        verify_run_tokenizer(RunConfig.model_validate(raw))
    raw["dataset"] = config.tokenizer.origin.dataset.model_dump(mode="json")
    assert (
        verify_run_tokenizer(RunConfig.model_validate(raw))["sha256"]
        == config.tokenizer.origin.sha256
    )
    assert "origin" not in RunConfig.model_validate(raw).model_dump()["tokenizer"]
    raw = config.model_dump(mode="json")
    raw["tokenizer"]["origin"]["dataset"] = config.dataset.model_dump(mode="json")
    with pytest.raises(ValueError, match="snapshot identity mismatch"):
        verify_run_tokenizer(RunConfig.model_validate(raw))
