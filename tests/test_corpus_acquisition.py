"""Acquisition identity, strict recipes, and offline byte verification."""

from __future__ import annotations

import bz2
import hashlib
import io
import json
import os
import select
import signal
import stat
import subprocess
import sys
import threading
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from corpus_fixtures import serve_pinned_acquisition as _serve_pinned
from corpus_fixtures import write_acquisition_recipe as _fixture

from klab.canonical import canonical_digest
from klab.corpus.acquisition import (
    acquire,
    declaration_sha256,
    verify_acquisition,
    verify_snapshot,
)
from klab.corpus.project import acquisition_identity_payload, load_project
from klab.corpus.transport_budget import TransportBudget


def _yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_local_offline_change_and_tamper(tmp_path: Path) -> None:
    recipe = _fixture(tmp_path)
    file = recipe.parent / "sources/data.md"
    file.write_text("# Before\n", encoding="utf-8")
    project = load_project(recipe)
    first = acquire(project, tmp_path / "work")
    original = first["sources"]["one"]
    assert acquire(project, tmp_path / "work", offline=True) == first
    file.write_text("# After\n", encoding="utf-8")
    assert acquire(project, tmp_path / "work", offline=True) == first
    second = acquire(project, tmp_path / "work")
    assert second["sources"]["one"]["snapshot_sha256"] != original["snapshot_sha256"]
    frozen = Path(original["snapshot_path"])
    assert (
        verify_snapshot(frozen)["files"][0]["sha256"]
        == hashlib.sha256(b"# Before\n").hexdigest()
    )
    active = Path(second["sources"]["one"]["snapshot_path"])
    (active / "files/data.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        verify_acquisition(project, tmp_path / "work")
    with pytest.raises(ValueError, match="mismatch"):
        acquire(project, tmp_path / "work", offline=True)


def test_acquisition_reclaims_only_abandoned_disposable_work(tmp_path: Path) -> None:
    recipe = _fixture(tmp_path)
    document = recipe.parent / "sources/data.md"
    document.write_text("# Before\n", encoding="utf-8")
    project = load_project(recipe)
    work = tmp_path / "work"
    first = acquire(project, work)
    parent = Path(first["sources"]["one"]["snapshot_path"]).parent
    legacy = parent / ".acquire-leftover"
    legacy.mkdir()
    keep = legacy / "original.txt"
    keep.write_bytes(b"unmarked payload")
    document.write_text("# After\n", encoding="utf-8")
    script = """
import sys
from pathlib import Path
from klab.corpus import acquisition
from klab.corpus.project import load_project

def pause_before_verification(path, *, _staged=False, **kwargs):
    assert _staged
    print(path, flush=True)
    sys.stdin.buffer.read(1)
    raise SystemExit(1)

acquisition.verify_snapshot = pause_before_verification
acquisition.acquire(load_project(Path(sys.argv[1])), Path(sys.argv[2]))
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(recipe), str(work)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as running:
        assert running.stdout is not None
        try:
            assert select.select([running.stdout], [], [], 20)[0]
            line = running.stdout.readline().strip()
            assert line
            staging = Path(line)
            assert (staging / "files/data.md").read_bytes() == b"# After\n"
            with pytest.raises(ValueError, match="another acquisition"):
                acquire(project, work)
            assert acquire(project, work, offline=True) == first
            assert (staging / "files/data.md").exists()
        finally:
            running.kill()
            running.wait(timeout=20)
    assert running.returncode == -signal.SIGKILL
    assert staging.is_dir()

    resumed = acquire(project, work)
    clean = acquire(project, tmp_path / "clean")
    assert (
        resumed["sources"]["one"]["snapshot_sha256"]
        == clean["sources"]["one"]["snapshot_sha256"]
    )
    assert verify_acquisition(project, work) == resumed
    assert not staging.exists()
    assert keep.read_bytes() == b"unmarked payload"


def test_git_pinned_revision_and_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Test")
    (repo / "docs").mkdir()
    document = repo / "docs/guide.md"
    document.write_text("first\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "first")
    first_revision = _git(repo, "rev-parse", "HEAD")
    recipe = _fixture(
        tmp_path,
        kind="git",
        acquisition={"include": ["docs/*.md"], "max_bytes": 1024},
        revision=first_revision,
        uri=str(repo),
    )
    project = load_project(recipe)
    first = acquire(project, tmp_path / "work")
    assert first["sources"]["one"]["snapshot_sha256"]
    document.write_text("second\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "second")
    latest = _git(repo, "rev-parse", "HEAD")
    source = yaml.safe_load((recipe.parent / "sources/one.yaml").read_text())
    source["revision"] = latest
    _yaml(recipe.parent / "sources/one.yaml", source)
    second = acquire(load_project(recipe), tmp_path / "work")
    assert (
        second["sources"]["one"]["snapshot_sha256"]
        != first["sources"]["one"]["snapshot_sha256"]
    )
    (repo / "docs/link.md").symlink_to("guide.md")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "symlink")
    source["revision"] = _git(repo, "rev-parse", "HEAD")
    _yaml(recipe.parent / "sources/one.yaml", source)
    with pytest.raises(ValueError, match="symlink"):
        acquire(load_project(recipe), tmp_path / "work")
    assert (
        json.loads((tmp_path / "work/corpora/example/acquisition.json").read_text())[
            "sources"
        ]["one"]
        == second["sources"]["one"]
    )


def test_verified_pinned_git_snapshot_survives_adapter_module_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus import acquisition

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Test")
    (repo / "guide.md").write_text("A pinned source document.\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "source")
    recipe = _fixture(
        tmp_path,
        kind="git",
        acquisition={"include": ["guide.md"], "max_bytes": 1024},
        revision=_git(repo, "rev-parse", "HEAD"),
        uri=str(repo),
    )
    project = load_project(recipe)
    work = tmp_path / "work"
    original = acquire(project, work)
    original_adapter = acquisition._adapter
    monkeypatch.setattr(
        acquisition,
        "_adapter",
        lambda source: {**original_adapter(source), "module_sha256": "0" * 64},
    )
    monkeypatch.setattr(
        acquisition,
        "_acquire_git",
        lambda *_args, **_kwargs: pytest.fail("pinned snapshot was refetched"),
    )
    assert acquire(project, work) == original
    (work / "corpora" / project.config.id / "acquisition.json").unlink()
    assert acquire(project, work) == original


def test_git_v2_acquires_pinned_nested_license_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Test")
    (repo / "docs").mkdir()
    (repo / "docs/guide.md").write_text("# SPDX-License-Identifier: MIT\nSource text\n")
    metadata = {"files": [{"path": "docs/guide.md", "license": "MIT"}]}
    (repo / "license-metadata.json").write_text(json.dumps(metadata))
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "pinned rights")
    recipe = _fixture(
        tmp_path,
        kind="git",
        acquisition={"include": ["docs/*.md"], "max_bytes": 4096},
        revision=_git(repo, "rev-parse", "HEAD"),
        uri=str(repo),
    )
    source_path = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(source_path.read_text())
    source["license_url"] = "https://example.org/upstream/LICENSE"
    source["rights"] = {
        "training_eligibility": "eligible",
        "redistribution_mode": "metadata_reconstruction_only",
        "spdx_expression": "MIT",
        "license_references": ["https://example.org/upstream/LICENSE"],
        "nested_metadata_path": "license-metadata.json",
    }
    _yaml(source_path, source)
    release_path = recipe.parent / "release.yaml"
    release = yaml.safe_load(release_path.read_text())
    _yaml(release_path, release)
    lock = acquire(load_project(recipe), tmp_path / "work")
    source = load_project(recipe).sources[0]
    full = source.model_dump(mode="json")
    identity = {
        key: full[key] for key in ("id", "kind", "canonical_uri", "revision")
    } | {
        "acquisition": full["acquisition"],
        "nested_metadata_path": "license-metadata.json",
    }
    identity["acquisition"].pop("tree_oid")
    identity["acquisition"].pop("bounded_blobs")
    assert acquisition_identity_payload(source) == identity
    snapshot = verify_snapshot(Path(lock["sources"]["one"]["snapshot_path"]))
    assert {item["path"] for item in snapshot["files"]} == {
        "docs/guide.md",
        "license-metadata.json",
    }
    assert (
        Path(lock["sources"]["one"]["snapshot_path"]) / "files/license-metadata.json"
    ).read_text() == json.dumps(metadata)
    assert acquire(load_project(recipe), tmp_path / "work", offline=True) == lock


def _identity_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Test")
    (repo / "docs").mkdir()
    (repo / "docs/guide.md").write_text("# Guide\nText.\n")
    (repo / "license-metadata.json").write_text("{}")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "pinned")
    recipe = _fixture(
        tmp_path,
        kind="git",
        acquisition={"include": ["docs/*.md"], "max_bytes": 4096},
        revision=_git(repo, "rev-parse", "HEAD"),
        uri=str(repo),
    )
    return repo, recipe


def _declare(recipe: Path, edit) -> None:
    path = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(path.read_text())
    edit(source)
    _yaml(path, source)


_BUILD_ONLY_EDITS = {
    "license": lambda s: (
        s.update(license="Apache-2.0")
        or s["rights"].update(spdx_expression="Apache-2.0")
    ),
    "license_url": lambda s: s.update(license_url="https://example.org/other"),
    "notes": lambda s: s.update(notes="reviewed by hand"),
    "rights": lambda s: s["rights"].update(
        training_eligibility="eligible_with_obligations",
        redistribution_mode="metadata_reconstruction_only",
        allowed_boundary_paths=["docs/guide.md"],
        license_references=["https://example.org/license", "https://example.org/2"],
    ),
    "domains": lambda s: s.update(domains=["technical_docs", "reference"]),
    "document_kinds": lambda s: s.update(document_kinds=["markdown", "text"]),
    "source_family": lambda s: s.update(source_family="other_docs"),
    "origin": lambda s: s.update(origin="human_authored"),
    "markdown_documents": lambda s: s.update(markdown_documents="files"),
}


@pytest.mark.parametrize("field", sorted(_BUILD_ONLY_EDITS) + ["restriction"])
def test_build_only_declaration_edits_keep_the_acquisition_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    from klab.corpus import acquisition

    _, recipe = _identity_repo(tmp_path)
    work = tmp_path / "work"
    first = acquire(load_project(recipe), work)
    before = declaration_sha256(load_project(recipe).sources[0])
    if field == "restriction":
        _declare(recipe, lambda s: s.update(explicit_training_restriction="none_found"))
        release = yaml.safe_load((recipe.parent / "release.yaml").read_text())
        release["policy"]["rights"]["training_use"] = (
            "allowed_unless_explicitly_prohibited"
        )
        _yaml(recipe.parent / "release.yaml", release)
    else:
        _declare(recipe, _BUILD_ONLY_EDITS[field])
    project = load_project(recipe)
    assert declaration_sha256(project.sources[0]) == before
    monkeypatch.setattr(
        acquisition,
        "_acquire_git",
        lambda *_a, **_k: pytest.fail("a build-only edit re-acquired the source"),
    )
    second = acquire(project, work)
    assert second == first
    verify_acquisition(project, work)
    snapshots = (work / "corpora/example/snapshots/one").iterdir()
    assert {path.name for path in snapshots if not path.name.startswith(".")} == {
        first["sources"]["one"]["snapshot_sha256"]
    }
    assert (
        "rights"
        not in verify_snapshot(first["sources"]["one"]["snapshot_path"])["declaration"]
    )


_IDENTITY_EDITS = {
    "revision": lambda s: s.update(revision="b" * 40),
    "canonical_uri": lambda s: s.update(canonical_uri=s["canonical_uri"] + "-mirror"),
    "max_bytes": lambda s: s["acquisition"].update(max_bytes=4097),
    "include": lambda s: s["acquisition"].update(include=["docs/*.md", "*.txt"]),
    "exclude": lambda s: s["acquisition"].update(exclude=["docs/x.md"]),
    "nested_metadata_path": lambda s: s["rights"].update(
        nested_metadata_path="license-metadata.json"
    ),
}


@pytest.mark.parametrize("field", sorted(_IDENTITY_EDITS))
def test_edits_that_change_acquired_bytes_change_the_acquisition_identity(
    tmp_path: Path, field: str
) -> None:
    _, recipe = _identity_repo(tmp_path)
    before = declaration_sha256(load_project(recipe).sources[0])
    _declare(recipe, _IDENTITY_EDITS[field])
    assert declaration_sha256(load_project(recipe).sources[0]) != before


def test_http_hash_bound_and_offline(tmp_path: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        body = b"# bounded document\n"
        calls = 0

        def do_GET(self) -> None:
            Handler.calls += 1
            self.send_response(200)
            self.send_header("ETag", "test")
            self.end_headers()
            self.wfile.write(Handler.body)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        uri = f"http://127.0.0.1:{server.server_port}/manual.md"
        recipe = _fixture(
            tmp_path,
            kind="http_document",
            acquisition={
                "expected_sha256": hashlib.sha256(Handler.body).hexdigest(),
                "max_bytes": 100,
            },
            uri=uri,
        )
        project = load_project(recipe)
        first = acquire(project, tmp_path / "work")
        before = Handler.calls
        assert acquire(project, tmp_path / "work", offline=True) == first
        assert Handler.calls == before
        Handler.body = b"# changed document\n"
        with pytest.raises(ValueError, match="SHA-256"):
            acquire(project, tmp_path / "work")
        assert verify_acquisition(project, tmp_path / "work") == first
        source = yaml.safe_load((recipe.parent / "sources/one.yaml").read_text())
        source["acquisition"]["expected_sha256"] = hashlib.sha256(
            Handler.body
        ).hexdigest()
        _yaml(recipe.parent / "sources/one.yaml", source)
        second = acquire(load_project(recipe), tmp_path / "work")
        assert (
            second["sources"]["one"]["snapshot_sha256"]
            != first["sources"]["one"]["snapshot_sha256"]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_hf_pinned_selection_and_unsupported_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    revision = "a" * 40
    location = tmp_path / revision
    (location / "default/train").mkdir(parents=True)
    selected = location / "default/train/data.jsonl"
    selected.write_text('{"text":"hello"}\n', encoding="utf-8")
    calls = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(location)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    recipe = _fixture(
        tmp_path,
        kind="huggingface_dataset",
        acquisition={
            "config": "default",
            "split": "train",
            "include": ["default/train/*"],
            "text_field": "text",
            "max_rows": 10,
            "max_bytes": 1024,
        },
        revision=revision,
        uri="org/dataset",
    )
    project = load_project(recipe)
    first = acquire(project, tmp_path / "work")
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["revision"] == revision
    assert acquire(project, tmp_path / "work", offline=True) == first
    assert len(calls) == 1
    (location / "default/train/unsupported.bin").write_bytes(b"binary")
    source = yaml.safe_load((recipe.parent / "sources/one.yaml").read_text())
    source["revision"] = "b" * 40
    _yaml(recipe.parent / "sources/one.yaml", source)
    location.rename(tmp_path / ("b" * 40))
    location = tmp_path / ("b" * 40)
    with pytest.raises(ValueError, match="unsupported HF file format"):
        acquire(load_project(recipe), tmp_path / "work")
    assert verify_acquisition(project, tmp_path / "work") == first


def test_hf_parquet_rows_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    revision = "c" * 40
    snapshot = tmp_path / revision
    folder = snapshot / "default/train"
    folder.mkdir(parents=True)
    pq.write_table(pa.table({"text": ["one", "two"]}), folder / "data.parquet")
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **_kwargs: str(snapshot)
    )
    recipe = _fixture(
        tmp_path,
        kind="huggingface_dataset",
        acquisition={
            "config": "default",
            "split": "train",
            "include": ["default/train/*.parquet"],
            "text_field": "text",
            "max_rows": 1,
            "max_bytes": 4096,
        },
        revision=revision,
        uri="org/parquet",
    )
    with pytest.raises(ValueError, match="max_rows"):
        acquire(load_project(recipe), tmp_path / "work")
    assert not (tmp_path / "work/corpora/example/acquisition.json").exists()
    source = yaml.safe_load((recipe.parent / "sources/one.yaml").read_text())
    source["acquisition"]["max_rows"] = 2
    _yaml(recipe.parent / "sources/one.yaml", source)
    lock = acquire(load_project(recipe), tmp_path / "work")
    assert (
        verify_snapshot(lock["sources"]["one"]["snapshot_path"])["retrieval"]["rows"]
        == 2
    )


def _bounded_hf_fixture(
    tmp_path: Path, data: bytes, path: str, *, rows: int = 3
) -> Path:
    return _fixture(
        tmp_path,
        kind="huggingface_dataset",
        acquisition={
            "config": "default",
            "split": "train",
            "bounded_shards": [
                {
                    "path": path,
                    "expected_sha256": hashlib.sha256(data).hexdigest(),
                    "max_shard_bytes": len(data),
                    "max_scanned_rows": rows,
                    "hash_modulus": 3,
                    "hash_remainders": [0, 2],
                }
            ],
            "text_field": "text",
            "max_rows": 5,
            "max_bytes": 4096,
        },
        revision="d" * 40,
        uri="org/dataset",
    )


def _mock_hf_stream(monkeypatch: pytest.MonkeyPatch, content: bytes) -> list[str]:
    import huggingface_hub

    calls: list[str] = []

    def no_snapshot(**_kwargs: object) -> None:
        raise AssertionError("bounded HF mode must not download a snapshot")

    class Response(io.BytesIO):
        url = "https://cdn.example.test/pinned-shard"

        def __init__(self, body: bytes):
            super().__init__(body)
            self.headers = {"Content-Length": str(len(body))}

    class Opener:
        def open(self, request: urllib.request.Request, timeout: int) -> Response:
            calls.append(request.full_url)
            return Response(content)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_snapshot)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: Opener())
    return calls


@pytest.mark.parametrize("memory_exhausted", [False, True])
def test_reacquire_reads_advisory_lock_once_and_can_fall_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    memory_exhausted: bool,
) -> None:
    from klab.corpus import acquisition

    content = b'{"text":"one"}\n'
    recipe = _bounded_hf_fixture(tmp_path, content, "default/train/data.jsonl", rows=1)
    corpus = yaml.safe_load(recipe.read_text())
    corpus["sources"].append("sources/two.yaml")
    _yaml(recipe, corpus)
    source = yaml.safe_load((recipe.parent / "sources/one.yaml").read_text())
    _yaml(recipe.parent / "sources/two.yaml", {**source, "id": "two"})
    calls = _mock_hf_stream(monkeypatch, content)
    project = load_project(recipe)
    work = tmp_path / "work"
    first = acquire(project, work)
    calls.clear()

    lock_path = work / "corpora/example/acquisition.json"
    reads = []
    real_read = acquisition.read_metadata

    def read(path: Path):
        if path == lock_path:
            reads.append(path)
            if len(reads) == 1 and memory_exhausted:
                raise MemoryError
        return real_read(path)

    monkeypatch.setattr(acquisition, "read_metadata", read)
    assert acquire(project, work, require_existing=True) == first
    assert reads == [lock_path, lock_path]  # Advisory lookup, then final verification.
    assert not calls
    if memory_exhausted:
        assert "MemoryError" in capsys.readouterr().err


def test_hf_declaration_hash_and_receipt_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b'{"text":"one"}\n'
    recipe = _bounded_hf_fixture(tmp_path, content, "default/train/data.jsonl", rows=1)
    project = load_project(recipe)
    source = project.sources[0]
    full = source.model_dump(mode="json")
    expected = {
        key: full[key]
        for key in ("id", "kind", "canonical_uri", "revision", "acquisition")
    }
    expected["acquisition"].pop("include")
    expected["acquisition"].pop("max_decompressed_bytes")
    for shard in expected["acquisition"]["bounded_shards"]:
        shard.pop("declared_config")
        shard.pop("declared_split")
    assert acquisition_identity_payload(source) == expected
    assert (
        declaration_sha256(source)
        == hashlib.sha256(
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    calls = _mock_hf_stream(monkeypatch, content)
    first = acquire(project, tmp_path / "work")
    manifest = verify_snapshot(first["sources"]["one"]["snapshot_path"])
    assert manifest["declaration_sha256"] == declaration_sha256(source)
    assert "transport_budget" not in manifest["retrieval"]
    assert "row_encoding" not in manifest["retrieval"]
    assert acquire(project, tmp_path / "work") == first
    assert len(calls) == 1


@pytest.mark.parametrize("damage_selection", [False, True])
def test_bounded_hf_source_step_resumes_without_recharging_verified_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage_selection: bool
) -> None:
    from klab.corpus import acquisition as native

    monkeypatch.delenv("KLAB_TRANSPORT_CACHE", raising=False)
    content = b'{"text":"One verified document."}\n'
    first = "default/train/first.jsonl"
    second = "default/train/second.jsonl"
    recipe = _bounded_hf_fixture(tmp_path, content, first, rows=1)
    source_path = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(source_path.read_text())
    (part,) = source["acquisition"]["bounded_shards"]
    part["hash_modulus"] = 1
    part["hash_remainders"] = [0]
    source["acquisition"]["bounded_shards"].append({**part, "path": second})
    _yaml(source_path, source)
    config = yaml.safe_load(recipe.read_text())
    transfers = 3 if damage_selection else 2
    config["transport_budget"] = {
        "attempt_id": "hf_resume",
        "max_source_body_bytes": len(content) * transfers,
        "max_metadata_body_bytes": 4096,
        "max_transfers": transfers,
        "max_retries_per_shard": int(damage_selection),
        "max_wall_seconds": 120,
        "max_disk_bytes": 2 * 1024**3,
    }
    _yaml(recipe, config)
    project = load_project(recipe)
    root = tmp_path / "work"
    budget = TransportBudget.initialize(
        root / "corpora/example/transport-budget.sqlite", project
    )
    requests = _mock_hf_stream(monkeypatch, content)
    monkeypatch.setattr(native, "_hf_pinned_metadata", lambda *_args: {})
    scans = []
    read_rows = native._bounded_hf_rows
    interrupted = True

    def rows(path, **kwargs):
        scans.append(path.name)
        if path.name == "second.jsonl" and interrupted:
            raise OSError("interrupted after verified transfer")
        yield from read_rows(path, **kwargs)

    monkeypatch.setattr(native, "_bounded_hf_rows", rows)
    step_options = {
        "source_id": "one",
        "max_new_bytes": 2 * 1024**3,
        "min_free_bytes": 0,
    }
    with pytest.raises(OSError, match="interrupted after verified transfer"):
        acquire(project, root, **step_options)
    snapshots = root / "corpora/example/snapshots/one"
    assert not any(not path.name.startswith(".") for path in snapshots.iterdir())
    assert budget.receipt()["source_actual"] == 2 * len(content)
    assert len(requests) == 2
    if damage_selection:
        selection = next(snapshots.glob(".hf-parts-*/**/first.jsonl.sample.jsonl"))
        selection.write_bytes(selection.read_bytes().replace(b"verified", b"changed!"))

    interrupted = False
    scans.clear()
    plan = acquire(project, root, plan_only=True, **step_options)
    expected = {second: "body"}
    if not damage_selection:
        expected[first] = "selection"
    assert plan["admission"]["hf_checkpoint_reuse"] == expected
    if not damage_selection:
        staged_body = next(snapshots.glob(".hf-parts-*/**/hf-input/**/second.jsonl"))
        staging = native.acquisition_staging

        @contextmanager
        def change_after_admission(parent, parts):
            with staging(parent, parts) as path:
                staged_body.write_bytes(b"changed")
                yield path

        with monkeypatch.context() as drift:
            drift.setattr(native, "acquisition_staging", change_after_admission)
            with pytest.raises(ValueError, match="checkpoint changed after admission"):
                acquire(project, root, **step_options)
        assert len(requests) == 2
        staged_body.write_bytes(content)

    resumed = acquire(project, root, **step_options)
    assert scans == (["first.jsonl"] if damage_selection else []) + ["second.jsonl"]
    assert len(requests) == transfers
    assert budget.receipt()["source_actual"] == transfers * len(content)
    assert not list(snapshots.glob(".hf-parts-*"))
    assert (
        stat.S_IMODE(Path(resumed["source"]["snapshot_path"]).stat().st_mode) == 0o700
    )

    # Compare with a fresh run of the same declaration and producer.
    control = tmp_path / "control"
    TransportBudget.initialize(
        control / "corpora/example/transport-budget.sqlite", project
    )
    reference = acquire(project, control, **step_options)["source"]
    assert resumed["source"]["snapshot_sha256"] == reference["snapshot_sha256"]
    assert (
        verify_snapshot(reference["snapshot_path"])["files"]
        == verify_snapshot(resumed["source"]["snapshot_path"])["files"]
    )


_GIT_BLOB = "docs/one.md"


def _bounded_git_fixture(
    tmp_path: Path,
    content: bytes,
    *,
    paths: tuple[str, ...] = (_GIT_BLOB,),
    retry_slots: int = 1,
) -> tuple[Path, str]:
    oid = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    recipe = _fixture(
        tmp_path,
        kind="git",
        uri="https://github.com/org/repo",
        revision="a" * 40,
        acquisition={
            "max_bytes": len(content) * len(paths),
            "tree_oid": "b" * 40,
            "bounded_blobs": [
                {"path": path, "git_blob_oid": oid, "max_bytes": len(content)}
                for path in paths
            ],
        },
    )
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    config["transport_budget"] = {
        "attempt_id": "git_fixture",
        "max_source_body_bytes": len(content) * (len(paths) + retry_slots),
        "max_metadata_body_bytes": 10000,
        "max_transfers": len(paths) + retry_slots,
        "max_retries_per_shard": 1,
        "max_wall_seconds": 120,
        "max_disk_bytes": 1000000,
    }
    _yaml(recipe, config)
    return recipe, oid


def _mock_bounded_git_http(
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
    oid: str,
    *,
    paths: tuple[str, ...] = (_GIT_BLOB,),
    wrong_tree: bool = False,
    corrupt: bool = False,
    interrupt: str | None = None,
    extra_body: bool = False,
) -> list[str]:
    requests = []
    raw_prefix = "https://raw.githubusercontent.com/org/repo/" + "a" * 40 + "/"

    class Response(io.BytesIO):
        def __init__(self, body: bytes, url: str):
            super().__init__(body)
            self.url = url
            self.status = 200
            self.headers = {"Content-Length": str(len(body))}

        def read(self, size: int = -1) -> bytes:
            if interrupt is not None and self.url == raw_prefix + interrupt:
                if self.tell():
                    raise OSError("interrupted bounded Git transfer")
                return super().read(min(2, size))
            return super().read(size)

    class Opener:
        def open(self, request: urllib.request.Request, timeout: float) -> Response:
            url = request.full_url
            if "/git/commits/" in url:
                body = {"sha": "a" * 40, "tree": {"sha": "b" * 40}}
            elif "/git/trees/" in url:
                body = {
                    "sha": "b" * 40,
                    "truncated": False,
                    "tree": [
                        {
                            "path": "docs/wrong.md" if wrong_tree else path,
                            "type": "blob",
                            "mode": "100644",
                            "sha": oid,
                            "size": len(content),
                        }
                        for path in paths
                    ],
                }
            else:
                assert url.startswith(raw_prefix)
                path = url.removeprefix(raw_prefix)
                assert path in paths
                requests.append(path)
                return Response(
                    content + b"x"
                    if extra_body
                    else bytes([content[0] ^ 1]) + content[1:]
                    if corrupt
                    else content,
                    url,
                )
            return Response(json.dumps(body).encode(), url)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: Opener())
    return requests


def test_bounded_git_blob_metadata_checksum_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KLAB_TRANSPORT_CACHE", raising=False)
    content = b"# Pinned text\n"
    recipe, oid = _bounded_git_fixture(tmp_path, content)
    project = load_project(recipe)
    root = tmp_path / "work"
    ledger_path = root / "corpora/example/transport-budget.sqlite"
    with pytest.raises(ValueError, match="ledger missing"):
        acquire(project, root)
    TransportBudget.initialize(ledger_path, project)
    _mock_bounded_git_http(monkeypatch, content, oid, wrong_tree=True)
    with pytest.raises(ValueError, match="missing or ambiguous"):
        acquire(project, root)
    assert TransportBudget(ledger_path, project).receipt()["source_charged"] == 0
    _mock_bounded_git_http(monkeypatch, content, oid, interrupt=_GIT_BLOB)
    with pytest.raises(OSError, match="interrupted"):
        acquire(project, root)
    _mock_bounded_git_http(monkeypatch, content, oid)
    lock = acquire(project, root)
    assert (
        verify_snapshot(lock["sources"]["one"]["snapshot_path"])["files"][0][
            "git_blob_id"
        ]
        == oid
    )
    state = TransportBudget(ledger_path, project).receipt()
    assert state["source_charged"] == 2 * len(content)
    assert [item["status"] for item in state["transfers"]] == [
        "interrupted",
        "complete",
    ]
    assert acquire(project, root) == lock

    # A later file in the same bounded Git source can interrupt after this
    # one completed; a second invocation must re-fetch verified earlier blobs
    # within the same shared byte and per-blob retry allowance.
    replay_root = tmp_path / "replay-work"
    replay_ledger_path = replay_root / "corpora/example/transport-budget.sqlite"
    replay = TransportBudget.initialize(replay_ledger_path, project)
    first = replay.reserve_transfer("one", "docs/one.md", len(content))
    replay.charge_transfer_actual(first, len(content))
    replay.finish_transfer(first, success=True)
    second = replay.reserve_transfer(
        "one", "docs/one.md", len(content), allow_verified_retry=True
    )
    replay.charge_transfer_actual(second, len(content))
    replay.finish_transfer(second, success=True)
    with pytest.raises(ValueError, match="retry allowance exhausted"):
        replay.reserve_transfer(
            "one", "docs/one.md", len(content), allow_verified_retry=True
        )

    bad_root = tmp_path / "bad-work"
    TransportBudget.initialize(
        bad_root / "corpora/example/transport-budget.sqlite", project
    )
    _mock_bounded_git_http(monkeypatch, content, oid, corrupt=True)
    with pytest.raises(ValueError, match="SHA-1 mismatch"):
        acquire(project, bad_root)

    oversized_root = tmp_path / "oversized-work"
    TransportBudget.initialize(
        oversized_root / "corpora/example/transport-budget.sqlite", project
    )
    _mock_bounded_git_http(monkeypatch, content, oid, extra_body=True)
    with pytest.raises(ValueError, match="Content-Length mismatch"):
        acquire(project, oversized_root)

    exhausted_root = tmp_path / "exhausted-work"
    exhausted_ledger = exhausted_root / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(exhausted_ledger, project)
    _mock_bounded_git_http(monkeypatch, content, oid, interrupt=_GIT_BLOB)
    for _ in range(2):
        with pytest.raises(OSError, match="interrupted"):
            acquire(project, exhausted_root)
    with pytest.raises(ValueError, match="retry allowance exhausted"):
        acquire(project, exhausted_root)
    assert TransportBudget(exhausted_ledger, project).receipt()["source_charged"] == (
        2 * len(content)
    )


@pytest.mark.parametrize("failure", ["later_blob", "damaged_blob", "unsettled_ledger"])
def test_bounded_git_acquisition_reuses_only_verified_blobs_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, failure: str
) -> None:
    monkeypatch.delenv("KLAB_TRANSPORT_CACHE", raising=False)
    content = b"# Pinned text\n"
    second = "docs/two.md"
    paths = (_GIT_BLOB, second)
    damage_staged_blob = failure == "damaged_blob"
    unsettled_ledger = failure == "unsettled_ledger"
    recipe, oid = _bounded_git_fixture(
        tmp_path, content, paths=paths, retry_slots=1 + damage_staged_blob
    )
    project = load_project(recipe)
    root = tmp_path / "work"
    ledger_path = root / "corpora/example/transport-budget.sqlite"
    budget = TransportBudget.initialize(ledger_path, project)
    staged_reads = []
    open_path = Path.open

    def record_staged_read(path, mode="r", *args, **kwargs):
        if mode == "rb" and any(part.startswith(".git-parts-") for part in path.parts):
            staged_reads.append(path)
        return open_path(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record_staged_read)
    requests = _mock_bounded_git_http(
        monkeypatch,
        content,
        oid,
        paths=paths,
        interrupt=None if unsettled_ledger else second,
    )
    if unsettled_ledger:

        def stop_before_ledger_commit(*args, **kwargs):
            raise SystemExit("interrupted before committing Git transfer")

        with monkeypatch.context() as fault:
            fault.setattr(TransportBudget, "finish_transfer", stop_before_ledger_commit)
            with pytest.raises(SystemExit, match="interrupted before committing"):
                acquire(project, root)
        assert requests == [_GIT_BLOB]
        assert budget.receipt()["transfers"][0]["status"] == "started"
    else:
        with pytest.raises(OSError, match="interrupted bounded Git transfer"):
            acquire(project, root)
        assert requests == list(paths)
    # The interrupted acquisition never reaches snapshot verification.
    # Hashing a freshly downloaded blob should not reread its staged file.
    assert not staged_reads
    snapshots = root / "corpora/example/snapshots/one"
    assert not any(not path.name.startswith(".") for path in snapshots.iterdir())
    assert not (root / "corpora/example/acquisition.json").exists()
    staged = next(snapshots.glob(".git-parts-*/snapshot/files/docs/one.md"))
    inode = staged.stat().st_ino
    if damage_staged_blob:
        staged.write_bytes(b"!" + content[1:])

    abandoned = None
    if failure == "later_blob":
        abandoned = snapshots / (".git-parts-" + "0" * 64)
        abandoned.mkdir()

    initial_requests = len(requests)
    requests = _mock_bounded_git_http(monkeypatch, content, oid, paths=paths)
    resumed = acquire(project, root)
    assert requests == (list(paths) if damage_staged_blob else [second])
    ledger = budget.receipt()
    assert ledger["source_charged"] == (initial_requests + len(requests)) * len(content)
    assert all(item["status"] != "started" for item in ledger["transfers"])
    assert list(snapshots.glob(".git-parts-*")) == ([abandoned] if abandoned else [])
    if abandoned is not None:
        assert str(abandoned) in capsys.readouterr().err
    output = Path(resumed["sources"]["one"]["snapshot_path"])
    if not damage_staged_blob:
        assert (output / "files" / _GIT_BLOB).stat().st_ino == inode

    control = tmp_path / "control"
    TransportBudget.initialize(
        control / "corpora/example/transport-budget.sqlite", project
    )
    expected = acquire(project, control)
    resumed_manifest, resumed_files = _snapshot_payload(resumed)
    expected_manifest, expected_files = _snapshot_payload(expected)
    assert resumed_manifest["snapshot_sha256"] == expected_manifest["snapshot_sha256"]
    assert resumed_files == expected_files


def test_bounded_git_rejects_ambiguous_declaration(tmp_path: Path) -> None:
    recipe, _ = _bounded_git_fixture(tmp_path, b"# one\n")
    source_file = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(source_file.read_text())
    source["acquisition"]["include"] = ["docs/*.md"]
    _yaml(source_file, source)
    with pytest.raises(ValueError, match="bounded Git requires exact blobs"):
        load_project(recipe)


def test_admission_reuses_verified_snapshot_and_rejects_changed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus import acquisition
    from klab.corpus.admission_draft import draft_admission_manifest

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Test")
    (repo / "guide.md").write_text("A small source document.\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "source")
    recipe = _fixture(
        tmp_path,
        kind="git",
        acquisition={"include": ["guide.md"], "max_bytes": 1024},
        revision=_git(repo, "rev-parse", "HEAD"),
        uri=str(repo),
    )
    declaration_path = recipe.parent / "sources/one.yaml"
    declaration = yaml.safe_load(declaration_path.read_text())
    declaration["license_url"] = "https://example.test/LICENSE"
    declaration["rights"] = {
        "training_eligibility": "eligible",
        "redistribution_mode": "metadata_reconstruction_only",
        "spdx_expression": "MIT",
        "license_references": ["https://example.test/LICENSE"],
    }
    _yaml(declaration_path, declaration)
    release_path = recipe.parent / "release.yaml"
    release = yaml.safe_load(release_path.read_text())
    _yaml(release_path, release)
    project = load_project(recipe)
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    lock = acquire(project, work)
    document = tmp_path / "policy.md"
    document.write_text("Use source terms.\n", encoding="utf-8")
    template = tmp_path / "policy.json"
    template.write_text(
        json.dumps(
            {
                "policy_id": "fixture",
                "policy_sha256": hashlib.sha256(document.read_bytes()).hexdigest(),
                "sources": [
                    {
                        "source_id": "one",
                        "license_label": "MIT",
                        "rights": {
                            "training_eligibility": "eligible",
                            "redistribution_mode": "metadata_reconstruction_only",
                            "spdx_expression": "MIT",
                            "license_references": ["https://example.test/LICENSE"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    first = draft_admission_manifest(
        project, work, template, document, work / "first.json"
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            acquisition,
            "_verify_snapshot_cold",
            lambda *args, **kwargs: pytest.fail("verified snapshot was read again"),
        )
        second = draft_admission_manifest(
            project, work, template, document, work / "second.json"
        )
    assert first["sha256"] == second["sha256"]
    snapshot = Path(lock["sources"]["one"]["snapshot_path"])
    (snapshot / "files/guide.md").write_text("Changed source.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        draft_admission_manifest(project, work, template, document, work / "third.json")


_RFC_URL = "https://www.rfc-editor.org/in-notes/tar/RFC-all.tar.gz"


def _tar_gz(members: list[tuple[str, bytes | None, str | None]]) -> bytes:
    """Members are (name, regular-file bytes, or None with a symlink target)."""
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data, link in members:
            info = tarfile.TarInfo(name)
            if data is None:
                info.type, info.linkname = tarfile.SYMTYPE, link or ""
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


_ARCHIVE_RIGHTS = {
    "training_eligibility": "eligible_with_obligations",
    "redistribution_mode": "metadata_reconstruction_only",
    "spdx_expression": "BSD-3-Clause",
    "license_references": ["https://trustee.ietf.org/license-info"],
    "notices": ["Retain the IETF Trust notice"],
}


def _archive_recipe(tmp_path: Path, body: bytes, *, budget: bool = False) -> Path:
    recipe = _fixture(
        tmp_path,
        kind="archive",
        uri=_RFC_URL,
        revision="2026-11-01",
        license="BSD-3-Clause",
        rights=_ARCHIVE_RIGHTS,
        acquisition={
            "expected_sha256": hashlib.sha256(body).hexdigest(),
            "format": "tar.gz",
            "include": ["rfc*.txt"],
            "exclude": ["rfc3.txt"],
            "max_archive_bytes": len(body),
            "max_expanded_bytes": 1 << 20,
            "max_members": 10,
            "max_member_bytes": 1024,
            "max_bytes": 4096,
        },
    )
    if budget:
        config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
        config["transport_budget"] = {
            "attempt_id": "rfc_fixture",
            "max_source_body_bytes": 2 * len(body),
            "max_metadata_body_bytes": 10000,
            "max_transfers": 2,
            "max_retries_per_shard": 1,
            "max_wall_seconds": 120,
            "max_disk_bytes": 1 << 30,
        }
        _yaml(recipe, config)
    return recipe


def test_archive_plan_step_snapshot_and_reviewed_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus.admission_draft import draft_admission_manifest
    from klab.corpus.pipeline import build
    from klab.corpus.release import freeze, verify_release
    from klab.corpus.split_freeze import freeze_splits
    from klab.corpus.split_inventory import write_split_inventory

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    included = {
        f"rfc{number}.txt": (
            f"Request for Comments {number} defines protocol behaviour {number}. "
            "Implementations follow each numbered step and check the result.\n"
        ).encode()
        for number in (1, 2, 4, 5)
    }
    body = _tar_gz(
        [(name, data, None) for name, data in included.items()]
        + [
            ("rfc3.txt", b"excluded by pattern\n", None),
            ("index.html", b"<html></html>", None),
            ("latest", None, "rfc2.txt"),  # never selected, never materialized
        ]
    )
    recipe = _archive_recipe(tmp_path, body, budget=True)
    project = load_project(recipe)
    root = tmp_path  # the recipe, and so its frozen splits, lie in the work root
    ledger = root / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(ledger, project)
    calls = _serve_pinned(monkeypatch, {_RFC_URL: body})
    step = {"source_id": "one", "max_new_bytes": 1 << 30, "min_free_bytes": 0}
    planned = acquire(project, root, plan_only=True, **step)
    assert planned["status"] == "planned" and calls == []
    assert acquire(project, root, **step)["status"] == "source_verified"
    lock = acquire(project, root)
    assert calls == [_RFC_URL]
    manifest = verify_snapshot(lock["sources"]["one"]["snapshot_path"])
    assert manifest["files"] == [
        {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        for name, data in included.items()
    ]
    assert manifest["retrieval"]["archive_sha256"] == hashlib.sha256(body).hexdigest()
    assert "rights" not in manifest["declaration"]
    assert TransportBudget(ledger, project).receipt()["source_charged"] == len(body)
    assert acquire(project, root, offline=True) == lock

    # Reviewed path: one admission decision and one family per archive member.
    policy = recipe.parent / "policy.md"
    policy.write_text("Reviewed RFC fixture policy.\n", encoding="utf-8")
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "policy_id": "rfc-fixture",
                "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                "sources": [
                    {
                        "source_id": "one",
                        "license_label": "BSD-3-Clause local research",
                        "rights": _ARCHIVE_RIGHTS,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    draft = draft_admission_manifest(
        project, root, template, policy, root / "admission.json"
    )
    assert draft["counts"] == {"qualify": 4, "exclude": 0, "quarantine": 0}
    admission = recipe.parent / "admission.json"
    admission.write_bytes((root / "admission.json").read_bytes())
    release = yaml.safe_load((recipe.parent / "release.yaml").read_text())
    release["record_admission"] = {
        "path": admission.name,
        "sha256": hashlib.sha256(admission.read_bytes()).hexdigest(),
    }
    _yaml(recipe.parent / "release.yaml", release)
    project = load_project(recipe)
    inventory = root / "inventory.jsonl"
    write_split_inventory(project, root, inventory)
    hints = [json.loads(line)["family_hint"] for line in inventory.open()]
    assert sorted(hints) == [f"archive-member:one:{name}" for name in included]
    clusters = root / "clusters.json"
    clusters.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
                "seed": "rfc-fixture",
                "source_strata": {"one": "technical_docs"},
                "split_weights": {"train": 2, "validation": 1, "test": 1},
                "merges": [],
                "reviewer": "Fixture reviewer",
                "reviewed_on": "2026-11-12",
            }
        ),
        encoding="utf-8",
    )
    freeze_splits(project, root, inventory, clusters, recipe.parent / "frozen.yaml")
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    config.update(splits="frozen.yaml", transforms=["transforms/lm.yaml"])
    _yaml(recipe, config)
    _yaml(
        recipe.parent / "transforms/lm.yaml",
        {
            "id": "lm",
            "version": "1",
            "kind": "lm_text",
            "parameters": {},
            "inputs": ["one"],
        },
    )
    published = freeze(build(load_project(recipe), root, offline=True), root)
    assert verify_release(published)["release_id"] == published.name


@pytest.mark.parametrize("damage", ["none", "derivation", "both"])
def test_archive_recovery_checks_staged_body_and_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    from klab.corpus import acquisition, acquisition_parts

    text = b"pinned source text\n"
    body = _tar_gz([("rfc1.txt", text, None)])
    recipe = _archive_recipe(tmp_path, body, budget=True)
    project = load_project(recipe)
    work = tmp_path / "work"
    ledger = work / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(ledger, project)
    calls = _serve_pinned(monkeypatch, {_RFC_URL: body})
    step = {"source_id": "one", "max_new_bytes": 1 << 30, "min_free_bytes": 0}
    real_verify = acquisition.verify_snapshot
    real_derive = acquisition._acquire_archive
    extractions = []

    def interrupt(path, **options):
        if options.get("_staged"):
            raise OSError("stop before staged verification")
        return real_verify(path, **options)

    def count(*args, **kwargs):
        extractions.append(1)
        return real_derive(*args, **kwargs)

    monkeypatch.setattr(acquisition, "_acquire_archive", count)
    with monkeypatch.context() as fault:
        fault.setattr(acquisition, "verify_snapshot", interrupt)
        with pytest.raises(OSError, match="stop before staged verification"):
            acquire(project, work, **step)
    assert not (ledger.parent / "acquisition.json").exists()
    parent = ledger.parent / "snapshots/one"
    (checkpoint,) = parent.glob(".archive-parts-*")
    if damage != "none":
        (checkpoint / "snapshot/files/rfc1.txt").write_bytes(b"damaged")
    if damage == "both":
        (checkpoint / "input/RFC-all.tar.gz").write_bytes(b"damaged")

    plan = acquire(project, work, plan_only=True, **step)["admission"]
    assert (
        plan["pinned_checkpoint_reuse"]
        == {"none": "derivation", "derivation": "body", "both": None}[damage]
    )
    output_hashes = []
    body_hashes = []
    real_hash = acquisition.sha256_file

    def record_hash(path: Path) -> str:
        if path.name == "rfc1.txt":
            output_hashes.append(path)
        elif path.name == "RFC-all.tar.gz":
            body_hashes.append(path)
        return real_hash(path)

    monkeypatch.setattr(acquisition, "sha256_file", record_hash)
    monkeypatch.setattr(acquisition_parts, "sha256_file", record_hash)
    assert acquire(project, work, **step)["status"] == "source_verified"
    if damage != "both":
        assert len(output_hashes) == 1
        assert len(body_hashes) == (0 if damage == "none" else 1)
    lock = acquire(project, work, require_existing=True)
    assert not checkpoint.exists()
    assert _snapshot_payload(lock)[1] == {"rfc1.txt": text}
    assert len(extractions) == (1 if damage == "none" else 2)
    assert len(calls) == (2 if damage == "both" else 1)
    assert TransportBudget(ledger, project).receipt()["source_actual"] == len(
        calls
    ) * len(body)


def test_archive_retry_reclaims_body_left_after_snapshot_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus import acquisition

    body = _tar_gz([("rfc1.txt", b"pinned source text\n", None)])
    project = load_project(_archive_recipe(tmp_path, body, budget=True))
    work = tmp_path / "work"
    base = work / "corpora/example"
    TransportBudget.initialize(base / "transport-budget.sqlite", project)
    calls = _serve_pinned(monkeypatch, {_RFC_URL: body})
    step = {"source_id": "one", "max_new_bytes": 1 << 30, "min_free_bytes": 0}
    rename = acquisition.rename_noreplace

    def stop_after_rename(source: Path, target: Path) -> None:
        rename(source, target)
        raise OSError("stop just after rename")

    with monkeypatch.context() as fault:
        fault.setattr(acquisition, "rename_noreplace", stop_after_rename)
        with pytest.raises(OSError, match="stop just after rename"):
            acquire(project, work, **step)
    parent = base / "snapshots/one"
    (orphan,) = parent.glob(".archive-parts-*")
    assert (orphan / "input/RFC-all.tar.gz").stat().st_size == len(body)
    sibling = parent / (".archive-parts-" + "0" * 64)
    (sibling / "snapshot").mkdir(parents=True, mode=0o700)
    adapter = acquisition._adapter
    monkeypatch.setattr(
        acquisition,
        "_adapter",
        lambda source: {**adapter(source), "module_sha256": "b" * 64},
    )
    plan = acquire(project, work, plan_only=True, **step)["admission"]
    assert plan["existing_snapshot"] is not None
    assert orphan.exists()
    resumed = acquire(project, work, **step)
    assert resumed["source"]["snapshot_path"] == plan["existing_snapshot"]
    assert not orphan.exists()
    assert (sibling / "snapshot").is_dir()
    assert calls == [_RFC_URL]


@pytest.mark.parametrize("mutation", ["admitted", "renamed"])
def test_archive_rejects_changed_bytes_after_checkpoint_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    from klab.corpus import acquisition

    body = _tar_gz([("rfc1.txt", b"initial\n", None)])
    project = load_project(_archive_recipe(tmp_path, body, budget=True))
    work = tmp_path / "work"
    base = work / "corpora/example"
    TransportBudget.initialize(base / "transport-budget.sqlite", project)
    calls = _serve_pinned(monkeypatch, {_RFC_URL: body})
    step = {"source_id": "one", "max_new_bytes": 1 << 30, "min_free_bytes": 0}

    def replace_bytes(path: Path) -> None:
        info = path.stat()
        path.write_bytes(b"mutated\n")
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))

    if mutation == "admitted":

        def stop(path, **options):
            raise OSError("stop before publish")

        with monkeypatch.context() as fault:
            fault.setattr(acquisition, "verify_snapshot", stop)
            with pytest.raises(OSError, match="stop before publish"):
                acquire(project, work, **step)
        stage = acquisition.acquisition_staging

        @contextmanager
        def mutate_before_lock(parent, parts):
            replace_bytes(parts.staging / "files/rfc1.txt")
            with stage(parent, parts) as staging:
                yield staging

        monkeypatch.setattr(acquisition, "acquisition_staging", mutate_before_lock)
        reason = "derivation changed after admission"
    else:
        rename = acquisition.rename_noreplace

        def mutate_after_rename(source, target):
            rename(source, target)
            replace_bytes(target / "files/rfc1.txt")

        monkeypatch.setattr(acquisition, "rename_noreplace", mutate_after_rename)
        reason = "snapshot byte mismatch"
    with pytest.raises(ValueError, match=reason):
        acquire(project, work, **step)
    assert calls == [_RFC_URL]
    assert not (base / "acquisition.json").exists()


@pytest.mark.parametrize(
    ("members", "tamper", "message"),
    [
        ([("../rfc9.txt", b"escape\n", None)], False, "unsafe path"),
        ([("rfc9.txt", None, "/etc/passwd")], False, "not a plain regular file"),
        ([("rfc9.txt", b"text\n", None)], True, "SHA-256 mismatch"),
    ],
)
def test_archive_refuses_unsafe_members_and_unpinned_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    members: list,
    tamper: bool,
    message: str,
) -> None:
    body = _tar_gz([("rfc1.txt", b"RFC 1 text\n", None), *members])
    project = load_project(_archive_recipe(tmp_path, body))
    served = body[:-1] + bytes([body[-1] ^ 1]) if tamper else body
    _serve_pinned(monkeypatch, {_RFC_URL: served})
    with pytest.raises(ValueError, match=message):
        acquire(project, tmp_path / "work")
    assert not (tmp_path / "work/corpora/example/acquisition.json").exists()
    snapshots = tmp_path / "work/corpora/example/snapshots/one"
    assert not any(not path.name.startswith(".") for path in snapshots.iterdir())
    assert not list(snapshots.glob(".archive-parts-*/snapshot/manifest.json"))


def _pdf(text: str) -> bytes:
    """A minimal one-page PDF whose content stream shows ``text`` in Helvetica."""
    stream = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    out += b"startxref\n%d\n%%%%EOF\n" % xref
    return bytes(out)


def _pdf_recipe(
    tmp_path: Path, texts: tuple[str, ...]
) -> tuple[Path, dict[str, bytes]]:
    pypdf = pytest.importorskip("pypdf")
    base = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications"
    bodies = {
        f"{base}/NIST.SP.800-{index}.pdf": _pdf(text)
        for index, text in enumerate(texts)
    }
    recipe = _fixture(
        tmp_path,
        kind="pdf_documents",
        uri="https://csrc.nist.gov/publications/sp",
        revision="2026-11-01",
        license="LicenseRef-US-Government-Work",
        acquisition={
            "documents": [
                {
                    "url": url,
                    "expected_sha256": hashlib.sha256(body).hexdigest(),
                    "max_bytes": len(body),
                    "name": Path(url).stem + ".txt",
                }
                for url, body in bodies.items()
            ],
            "extraction": {"engine": "pypdf", "version": pypdf.__version__},
            "max_pages": 5,
            "max_text_bytes": 4096,
        },
    )
    return recipe, bodies


def test_pdf_text_is_reproducible_from_pinned_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from klab.corpus import acquisition

    recipe, bodies = _pdf_recipe(tmp_path, ("Pinned safety guidance text",))
    pypdf = pytest.importorskip("pypdf")
    (body,) = bodies.values()
    project = load_project(recipe)
    _serve_pinned(monkeypatch, bodies)
    lock = acquire(project, tmp_path / "work")
    snapshot = Path(lock["sources"]["one"]["snapshot_path"])
    text = (snapshot / "files/NIST.SP.800-0.txt").read_bytes()
    assert text == b"Pinned safety guidance text\n"
    document = verify_snapshot(snapshot)["retrieval"]["documents"][0]
    assert document["text_sha256"] == hashlib.sha256(text).hexdigest()
    assert (snapshot / "pinned/NIST.SP.800-0.pdf").read_bytes() == body
    with monkeypatch.context() as patch:
        patch.setattr(pypdf, "__version__", "0.0.1")
        with pytest.raises(ValueError, match="pinned to pypdf"):
            verify_snapshot(snapshot)
    with monkeypatch.context() as patch:
        patch.setattr(acquisition, "_pdf_text", lambda *_args: (b"different\n", 1))
        with pytest.raises(ValueError, match="not reproducible"):
            verify_snapshot(snapshot)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "pypdf", None)
        with pytest.raises(ValueError, match="`pdf` extra"):
            verify_snapshot(snapshot)


@pytest.mark.parametrize("damage_text", [False, True])
def test_pdf_source_step_reuses_pinned_bodies_and_verified_text_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage_text: bool
) -> None:
    from klab.corpus import acquisition as native

    monkeypatch.delenv("KLAB_TRANSPORT_CACHE", raising=False)
    recipe, bodies = _pdf_recipe(tmp_path, ("First pinned PDF", "Second pinned PDF"))
    names = ("NIST.SP.800-0.txt", "NIST.SP.800-1.txt")
    total_body_bytes = sum(map(len, bodies.values()))
    config = yaml.safe_load(recipe.read_text())
    config["transport_budget"] = {
        "attempt_id": "pdf_resume",
        "max_source_body_bytes": total_body_bytes,
        "max_metadata_body_bytes": 4096,
        "max_transfers": 2,
        "max_retries_per_shard": 0,
        "max_wall_seconds": 120,
        "max_disk_bytes": 2 * 1024**3,
    }
    _yaml(recipe, config)
    project = load_project(recipe)
    root = tmp_path / "work"
    budget = TransportBudget.initialize(
        root / "corpora/example/transport-budget.sqlite", project
    )
    requests = _serve_pinned(monkeypatch, bodies)
    extract = native._pdf_text
    extracted = []
    interrupted = True

    def extract_with_fault(path, spec, name):
        if not (path.parent.parent / "manifest.json").exists():
            extracted.append(name)
            if name == names[1] and interrupted:
                raise OSError("interrupted before PDF extraction")
        return extract(path, spec, name)

    monkeypatch.setattr(native, "_pdf_text", extract_with_fault)
    step = {"source_id": "one", "max_new_bytes": 1 << 30, "min_free_bytes": 0}
    with pytest.raises(OSError, match="interrupted before PDF extraction"):
        acquire(project, root, **step)
    snapshots = root / "corpora/example/snapshots/one"
    assert not any(not path.name.startswith(".") for path in snapshots.iterdir())
    assert requests == list(bodies)
    assert budget.receipt()["source_actual"] == total_body_bytes
    if damage_text:
        staged_text = next(snapshots.glob(".pdf-parts-*/snapshot/files/" + names[0]))
        staged_text.write_bytes(staged_text.read_bytes().replace(b"First", b"Other"))

    interrupted = False
    extracted.clear()
    plan = acquire(project, root, plan_only=True, **step)["admission"]
    assert plan["pdf_checkpoint_reuse"] == {
        names[0]: "body" if damage_text else "text",
        names[1]: "body",
    }
    resumed = acquire(project, root, **step)["source"]
    assert extracted == (list(names) if damage_text else [names[1]])
    assert requests == list(bodies)
    assert budget.receipt()["source_actual"] == total_body_bytes
    assert not list(snapshots.glob(".pdf-parts-*"))

    control = tmp_path / "control"
    TransportBudget.initialize(
        control / "corpora/example/transport-budget.sqlite", project
    )
    expected = acquire(project, control, **step)["source"]
    assert resumed["snapshot_sha256"] == expected["snapshot_sha256"]
    assert verify_snapshot(resumed["snapshot_path"]) == verify_snapshot(
        expected["snapshot_path"]
    )


@pytest.mark.parametrize(
    ("language", "article"),
    [
        ("en", "Probe, sense, respond. '''Complex'''."),
        # Non-English sites keep their own script and label documents with it.
        ("ru", "Исследуй, распознай, реагируй."),
    ],
)
def test_mediawiki_dump_of_any_site_yields_one_row_per_article(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, language: str, article: str
) -> None:
    import gzip

    from klab.corpus.pipeline import _records_for_file

    url = "https://cynefin.example.org/dumps/cynefinwiki-20261101.xml.gz"
    export = "http://www.mediawiki.org/xml/export-0.10/"

    def page(page_id: int, ns: int, title: str, text: str, extra: str = "") -> str:
        return (
            f"<page><title>{title}</title><ns>{ns}</ns><id>{page_id}</id>{extra}"
            f"<revision><id>{page_id + 100}</id>"
            "<timestamp>2026-10-01T00:00:00Z</timestamp>"
            f"<text>{text}</text></revision></page>"
        )

    body = gzip.compress(
        (
            f'<mediawiki xmlns="{export}" xml:lang="{language}"><siteinfo>'
            "<sitename>Cynefin</sitename><dbname>cynefinwiki</dbname></siteinfo>"
            + page(1, 0, "Complex domain", article)
            + page(2, 4, "Cynefin:About", "Project page outside the namespaces.")
            + page(3, 0, "Chaos", "#REDIRECT [[Chaotic domain]]", "<redirect />")
            + "</mediawiki>"
        ).encode()
    )
    recipe = _fixture(
        tmp_path,
        kind="mediawiki_dump",
        uri=url,
        revision="20261101",
        license="CC-BY-SA-4.0",
        acquisition={
            "expected_sha256": hashlib.sha256(body).hexdigest(),
            "compression": "gzip",
            "dbname": "cynefinwiki",
            "language": language,
            "site_url": "https://cynefin.example.org/index.php",
            "max_compressed_bytes": len(body),
            "max_decompressed_bytes": 1 << 20,
            "max_scanned_pages": 10,
            "max_selected_pages": 10,
            "max_emitted_bytes": 1 << 20,
        },
    )
    _serve_pinned(monkeypatch, {url: body})
    project = load_project(recipe)
    lock = acquire(project, tmp_path / "work")
    snapshot = Path(lock["sources"]["one"]["snapshot_path"])
    manifest = verify_snapshot(snapshot)
    name = manifest["files"][0]["path"]
    raw = (snapshot / "files" / name).read_bytes()
    assert [
        (row["_klab_source"]["page_title"], row["_klab_source"]["page_uri"])
        for row in map(json.loads, raw.splitlines())
    ] == [("Complex domain", "https://cynefin.example.org/index.php?curid=1")]
    assert manifest["retrieval"]["scanned_pages"] == 3
    documents = _records_for_file(
        raw,
        name,
        project.sources[0],
        snapshot.name,
        normalizer_options=project.release.normalizer,
        document_policy=project.release.policy.documents,
    )
    assert [document["language"] for document, _ in documents] == [language]


def test_a_transport_budget_admits_a_git_source_selected_by_pattern(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Test")
    (repo / "docs").mkdir()
    (repo / "docs/guide.md").write_text("# Guide\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "first")
    body = _tar_gz([("rfc1.txt", b"Request for Comments 1 defines behaviour.\n", None)])
    recipe = _archive_recipe(tmp_path, body, budget=True)
    sources = recipe.parent / "sources"
    two = yaml.safe_load((sources / "one.yaml").read_text(encoding="utf-8"))
    two.update(
        id="two",
        kind="git",
        canonical_uri=str(repo),
        revision=_git(repo, "rev-parse", "HEAD"),
        acquisition={"include": ["docs/*.md"], "max_bytes": 1024},
    )
    _yaml(sources / "two.yaml", two)
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    config["sources"].append("sources/two.yaml")
    _yaml(recipe, config)
    project = load_project(recipe)
    ledger = tmp_path / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(ledger, project)
    _serve_pinned(monkeypatch, {_RFC_URL: body})
    lock = acquire(project, tmp_path)
    assert set(lock["sources"]) == {"one", "two"}
    assert TransportBudget(ledger, project).receipt()["source_charged"] == len(body)


# -- operator-local transport cache -------------------------------------------


def _edit_source(recipe: Path, **fields: object) -> None:
    path = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(path.read_text())
    source.update(fields)
    _yaml(path, source)


def _widen_cap(recipe: Path) -> None:
    """An acquisition edit that changes the identity but not one requested byte."""
    path = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(path.read_text())
    source["acquisition"]["max_bytes"] += 1
    _yaml(path, source)


def _snapshot_payload(lock: dict) -> tuple[dict, dict[str, bytes]]:
    path = Path(lock["sources"]["one"]["snapshot_path"])
    manifest = verify_snapshot(path)
    return manifest, {
        item["path"]: (path / "files" / item["path"]).read_bytes()
        for item in manifest["files"]
    }


def _cache_files(cache: Path) -> list[Path]:
    return sorted(p for p in (cache / "bodies").rglob("*") if p.is_file())


def test_cache_serves_a_declaration_edit_without_network_and_charges_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"# Pinned text\n"
    recipe, oid = _bounded_git_fixture(tmp_path, content)
    cache = tmp_path / "cold"
    root = tmp_path / "work"
    ledger = root / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(ledger, load_project(recipe))
    _mock_bounded_git_http(monkeypatch, content, oid)
    first = acquire(load_project(recipe), root, transport_cache=cache)
    charged = TransportBudget(ledger, load_project(recipe)).receipt()
    assert charged["source_actual"] == len(content)
    assert first["sources"]["one"]["transport_cache"][0]["served_from"] == "network"

    # A wider byte cap changes the identity but not what is asked of the network.
    _widen_cap(recipe)
    project = load_project(recipe)
    assert (
        declaration_sha256(project.sources[0])
        != (first["sources"]["one"]["declaration_sha256"])
    )

    def no_network(*_args: object) -> None:
        raise AssertionError("a cache hit must make no request")

    monkeypatch.setattr(urllib.request, "build_opener", no_network)
    second = acquire(project, root, transport_cache=cache)
    after = TransportBudget(ledger, project).receipt()
    assert (after["source_actual"], after["metadata_actual"], after["transfers"]) == (
        charged["source_actual"],
        charged["metadata_actual"],
        charged["transfers"],
    )
    (hit,) = second["sources"]["one"]["transport_cache"]
    assert hit["served_from"] == "cache"
    assert hit["sha256"] == hashlib.sha256(content).hexdigest()
    assert (
        hit["cache_key"] == first["sources"]["one"]["transport_cache"][0]["cache_key"]
    )

    manifest_one, files_one = _snapshot_payload(first)
    manifest_two, files_two = _snapshot_payload(second)
    assert files_one == files_two
    assert manifest_one["files"] == manifest_two["files"]
    assert manifest_one["retrieval"]["commit"] == manifest_two["retrieval"]["commit"]
    assert manifest_one["declaration_sha256"] != manifest_two["declaration_sha256"]

    # A different work root with a fresh ledger is served too, and charged nothing.
    other = tmp_path / "other-work"
    TransportBudget.initialize(
        other / "corpora/example/transport-budget.sqlite", project
    )
    acquire(project, other, transport_cache=cache)
    fresh = TransportBudget(other / "corpora/example/transport-budget.sqlite", project)
    assert fresh.receipt()["source_charged"] == 0
    assert fresh.receipt()["metadata_charged"] == 0


def test_cache_hit_needs_the_cache_host_setting_and_never_a_recipe_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b'{"text":"one"}\n'
    recipe = _bounded_hf_fixture(tmp_path, content, "default/train/data.jsonl", rows=1)
    cache = tmp_path / "cold"
    monkeypatch.setenv("KLAB_TRANSPORT_CACHE", str(cache))
    calls = _mock_hf_stream(monkeypatch, content)
    first = acquire(load_project(recipe), tmp_path / "work")
    _widen_cap(recipe)
    second = acquire(load_project(recipe), tmp_path / "work")
    assert len(calls) == 1
    assert _snapshot_payload(first)[1] == _snapshot_payload(second)[1]
    assert second["sources"]["one"]["transport_cache"][0]["served_from"] == "cache"
    monkeypatch.delenv("KLAB_TRANSPORT_CACHE")
    _widen_cap(recipe)
    acquire(load_project(recipe), tmp_path / "work")
    assert len(calls) == 2  # no setting, no cache
    with pytest.raises(ValueError, match="extra|Extra|permitted"):
        _edit_source(recipe, transport_cache=str(cache))
        load_project(recipe)


@pytest.mark.parametrize("damage", ["truncate", "flip", "wrong_entry"])
def test_damaged_cache_entry_is_detected_and_refetched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    damage: str,
) -> None:
    from klab.corpus.transport_cache import TransportCache

    content = b'{"text":"one"}\n{"text":"two"}\n'
    recipe = _bounded_hf_fixture(tmp_path, content, "default/train/data.jsonl", rows=2)
    cache = tmp_path / "cold"
    calls = _mock_hf_stream(monkeypatch, content)
    acquire(load_project(recipe), tmp_path / "work", transport_cache=cache)
    (body,) = _cache_files(cache)
    body.chmod(0o644)
    if damage == "truncate":
        body.write_bytes(content[:-3])
    elif damage == "flip":
        body.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    else:
        (key,) = (cache / "keys").glob("*.json")
        entry = json.loads(key.read_text())
        entry["key"]["url"] += "x"  # an entry that answers a different request
        key.write_text(json.dumps(entry))
    capsys.readouterr()
    _widen_cap(recipe)
    again = acquire(load_project(recipe), tmp_path / "work", transport_cache=cache)
    assert len(calls) == 2
    assert "rejected" in capsys.readouterr().err
    assert _snapshot_payload(again)[1]["default/train/data.jsonl.sample.jsonl"]
    assert body.read_bytes() == content
    assert {r["status"] for r in TransportCache(cache).entries(verify=True)} == {"ok"}


def test_server_bytes_that_miss_the_pin_are_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned = b'{"text":"one"}\n'
    recipe = _bounded_hf_fixture(tmp_path, pinned, "default/train/data.jsonl", rows=1)
    cache = tmp_path / "cold"
    _mock_hf_stream(monkeypatch, b'{"text":"two"}\n')
    with pytest.raises(ValueError, match="SHA-256 mismatch|Content-Length|exceeds"):
        acquire(load_project(recipe), tmp_path / "work", transport_cache=cache)
    assert not _cache_files(cache)
    assert not list((cache / "keys").glob("*.json"))


def test_pinned_downloads_and_http_documents_are_cached_by_what_was_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    body = _tar_gz([("rfc1.txt", b"Request for Comments 1 " * 3, None)])
    recipe = _archive_recipe(tmp_path, body, budget=True)
    cache = tmp_path / "cold"
    calls = _serve_pinned(monkeypatch, {_RFC_URL: body})
    project = load_project(recipe)
    TransportBudget.initialize(
        tmp_path / "w1/corpora/example/transport-budget.sqlite", project
    )
    acquire(project, tmp_path / "w1", transport_cache=cache)
    assert len(calls) == 1
    # A new work root and ledger, and an edited declaration: still no request.
    _widen_cap(recipe)
    project = load_project(recipe)
    ledger = tmp_path / "w2/corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(ledger, project)
    leftover = ledger.parent / "snapshots/one/.acquire-leftover"
    leftover.mkdir(parents=True)
    (leftover / "payload").write_bytes(body)
    second = acquire(project, tmp_path / "w2", transport_cache=cache)
    assert len(calls) == 1
    assert TransportBudget(ledger, project).receipt()["source_charged"] == 0
    assert second["sources"]["one"]["transport_cache"][0]["served_from"] == "cache"
    assert str(leftover) in capsys.readouterr().err
    assert (leftover / "payload").read_bytes() == body


def _wikimedia_fixture(
    root: Path,
    body: bytes,
    *,
    sha1: str,
    sha256: str | None = None,
    last_page: int = 1,
) -> tuple[Path, dict[str, bytes]]:
    date = "20261101"
    prefix = f"https://dumps.wikimedia.org/enwiki/{date}"
    name = f"enwiki-{date}-pages-articles-multistream1.xml-p1p{last_page}.bz2"
    uri = f"{prefix}/{name}"
    checksum_uri = f"{prefix}/enwiki-{date}-sha1sums.txt"
    recipe = _fixture(
        root,
        kind="wikimedia_dump",
        uri=uri,
        revision=date,
        license="CC-BY-SA-4.0",
        acquisition={
            "expected_sha1": sha1,
            "expected_sha256": sha256,
            "checksum_uri": checksum_uri,
            "max_compressed_bytes": len(body),
            "max_decompressed_bytes": 1024,
            "max_scanned_pages": 1,
            "max_selected_pages": 1,
            "max_emitted_bytes": 1024,
        },
    )
    return recipe, {checksum_uri: f"{sha1}  {name}\n".encode(), uri: body}


def _multistream_wikimedia_fixture(
    root: Path,
    *,
    fault: str | None = None,
    shared_members: bool = False,
    one_page_per_member: bool = False,
):
    header = (
        b'<?xml version="1.0"?>\n<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" '
        b'xml:lang="en"><siteinfo><dbname>enwiki</dbname></siteinfo>\n'
    )

    def page(number: int, title: str, extra: str = "") -> bytes:
        return (
            f"<page><title>{title}</title><ns>0</ns><id>{number}</id>{extra}"
            f"<revision><id>{number + 100}</id><timestamp>2026-10-01T00:00:00Z</timestamp>"
            f"<text>Article {number} contains usable text.</text></revision></page>\n"
        ).encode()

    first = page(1, "Topic:A")
    second = page(2, "Topic B", '<redirect title="Topic:A"/>')
    third, fourth = page(3, "Topic C"), page(4, "Topic D")
    last = third + fourth
    footer = b"</mediawiki>\n"
    if one_page_per_member:
        pieces = (header + first, second, third, fourth + footer)
    elif fault == "split_page":
        pieces = (header, first[:22], first[22:] + second, last, footer)
    elif shared_members:
        pieces = (header + first + second, last + footer)
    else:
        pieces = (header, first + second, last, footer)
    members = [bz2.compress(piece) for piece in pieces]
    body = b"".join(members)
    recipe, responses = _wikimedia_fixture(
        root, body, sha1=hashlib.sha1(body).hexdigest(), last_page=4
    )
    first_offset = 0 if shared_members else len(members[0])
    third_offset = sum(map(len, members[: 1 if shared_members else -2]))
    index_text = (
        f"{first_offset}:1:Topic:A\n"
        f"{first_offset}:{9 if fault == 'page_id' else 2}:Topic B\n"
        f"{third_offset}:3:Topic C\n"
        f"{third_offset}:{9 if fault == 'late_page_id' else 4}:Topic D\n"
    ).encode()
    if one_page_per_member:
        index_text = b"".join(
            f"{sum(map(len, members[:number]))}:{number + 1}:{title}\n".encode()
            for number, title in enumerate(("Topic:A", "Topic B", "Topic C", "Topic D"))
        )
    index_uri = next(reversed(responses)).replace(
        "multistream1.xml", "multistream-index1.txt"
    )
    responses[index_uri] = bz2.compress(index_text)
    return recipe, responses, pieces, members, index_text


def _fixture_index_options(
    responses: dict[str, bytes], pieces: tuple[bytes, ...], index_text: bytes
) -> dict:
    uri = next(reversed(responses))
    index = responses[uri]
    return {
        "uri": uri,
        "expected_sha256": hashlib.sha256(index).hexdigest(),
        "max_bytes": len(index),
        "max_expanded_bytes": len(index_text),
        "max_members": len(pieces),
        "max_member_expanded_bytes": max(map(len, pieces)),
    }


@pytest.mark.parametrize("fault", [None, "page_id", "split_page", "expanded_bound"])
def test_wikimedia_range_plan_verifies_index_and_physical_members(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fault: str | None
) -> None:
    from klab.cli.main import build_parser

    recipe, responses, pieces, members, index_text = _multistream_wikimedia_fixture(
        tmp_path, fault=fault
    )
    _, body, compressed_index = responses.values()
    first_offset = len(members[0])
    third_offset = len(body) - len(members[-2]) - len(members[-1])
    dump = tmp_path / "dump.bz2"
    index = tmp_path / "index.bz2"
    dump.write_bytes(body)
    index.write_bytes(compressed_index)
    work = tmp_path / "work"
    index_sha256 = hashlib.sha256(compressed_index).hexdigest()
    args = build_parser().parse_args(
        [
            "--work-dir",
            str(work),
            "corpus",
            "plan-wikimedia-ranges",
            str(recipe),
            "--source-id",
            "one",
            "--dump",
            str(dump),
            "--index",
            str(index),
            "--index-sha256",
            index_sha256,
            "--max-index-bytes",
            str(len(compressed_index)),
            "--max-index-expanded-bytes",
            str(len(index_text) - (fault == "expanded_bound")),
            "--max-members",
            str(len(members)),
        ]
    )
    if fault is not None:
        with pytest.raises(
            ValueError,
            match={
                "page_id": "page IDs or offset differ",
                "split_page": "XML crosses a compressed member boundary",
                "expanded_bound": "index exceeds its expanded byte bound",
            }[fault],
        ):
            args.handler(args)
        assert not work.exists()
        return

    args.handler(args)
    plan = json.loads(capsys.readouterr().out)
    assert not work.exists()
    assert plan["dump_sha256"] == hashlib.sha256(body).hexdigest()
    assert plan["index_sha256"] == index_sha256
    assert (plan["dbname"], plan["language"], plan["indexed_pages"]) == (
        "enwiki",
        "en",
        4,
    )
    assert plan["expanded_bytes"] == sum(map(len, pieces))
    ranges = plan["ranges"]
    assert [item["pages"] for item in ranges] == [0, 2, 2, 0]
    assert [item["start"] for item in ranges] == [
        0,
        first_offset,
        third_offset,
        len(body) - len(members[-1]),
    ]
    assert [item["end"] for item in ranges] == [
        *(item["start"] for item in ranges[1:]),
        len(body),
    ]
    assert ranges[0]["opens_document"] and ranges[-1]["closes_document"]
    assert ranges[1]["page_ids_sha256"] == hashlib.sha256(b"1\n2\n").hexdigest()


@pytest.mark.parametrize(
    "case", ["resume", "damaged", "invalid_suffix", "changed_dump", "prefix"]
)
def test_indexed_wikimedia_resumes_range_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    from klab.corpus import acquisition, acquisition_parts, wikimedia_ranges

    monkeypatch.delenv("KLAB_TRANSPORT_CACHE", raising=False)
    recipe, responses, pieces, compressed, index_text = _multistream_wikimedia_fixture(
        tmp_path, fault="late_page_id" if case == "invalid_suffix" else None
    )
    index = _fixture_index_options(responses, pieces, index_text)
    source_path = recipe.parent / "sources/one.yaml"
    declaration = yaml.safe_load(source_path.read_text())
    declaration["acquisition"]["multistream_index"] = index
    _yaml(source_path, declaration)
    project = load_project(recipe)
    calls = _serve_pinned(monkeypatch, responses)
    cache = tmp_path / "cache" if case == "resume" else None
    monkeypatch.setattr(wikimedia_ranges, "PLAN_GROUP_MEMBERS", 2)
    run_stage = wikimedia_ranges.run_stage
    run_acquisition_stage = acquisition.run_stage
    scan = wikimedia_ranges._scan_members
    scanned = []
    decoded = []
    read_member = wikimedia_ranges.read_wikimedia_member
    hashed = []
    real_hash = wikimedia_ranges.sha256_file
    dump_name = Path(project.sources[0].canonical_uri).name

    def record_hash(path: Path, **kwargs):
        if path.name in {dump_name, "dump.bz2"}:
            hashed.append(path)
        return real_hash(path, **kwargs)

    def record_scan(*args, **kwargs):
        for member in scan(*args, **kwargs):
            scanned.append(member.start)
            yield member

    def record_decode(path, member, **kwargs):
        decoded.append(member.start)
        return read_member(path, member, **kwargs)

    def interrupt_group(store, stage, inputs, produce, **kwargs):
        run_stage(store, stage, inputs, produce, **kwargs)
        raise RuntimeError("stop after verified range group")

    def interrupt_member(store, stage, inputs, produce):
        result = run_acquisition_stage(store, stage, inputs, produce)
        if stage == "wikimedia-member":
            raise OSError("stop after verified prefix extraction")
        return result

    monkeypatch.setattr(wikimedia_ranges, "_scan_members", record_scan)
    monkeypatch.setattr(wikimedia_ranges, "read_wikimedia_member", record_decode)
    monkeypatch.setattr(wikimedia_ranges, "sha256_file", record_hash)
    monkeypatch.setattr(acquisition_parts, "sha256_file", record_hash)
    work = tmp_path / "work"
    base = work / "corpora/example"
    with monkeypatch.context() as fault:
        if case == "prefix":
            fault.setattr(acquisition, "run_stage", interrupt_member)
            expectation = pytest.raises(
                OSError, match="stop after verified prefix extraction"
            )
        else:
            fault.setattr(wikimedia_ranges, "run_stage", interrupt_group)
            expectation = pytest.raises(
                RuntimeError, match="stop after verified range group"
            )
        with expectation:
            acquire(project, work, transport_cache=cache)
    assert scanned == [0, len(compressed[0])]
    assert not (base / "acquisition.json").exists()
    assert calls == list(responses)
    assert not hashed  # The download was checked as it was copied.
    assert decoded == ([len(compressed[0])] if case == "prefix" else [])
    calls.clear()
    scanned.clear()
    (parts,) = (base / "snapshots/one").glob(".wikimedia_dump-parts-*")
    dump_path = parts / "input" / dump_name
    if case == "changed_dump":
        plan = wikimedia_ranges.plan_wikimedia_ranges

        def change_after_verification(source, dump, index_path, **kwargs):
            original = dump.stat()
            dump.write_bytes(b"!" + dump.read_bytes()[1:])
            os.utime(dump, ns=(original.st_atime_ns, original.st_mtime_ns))
            return plan(source, dump, index_path, **kwargs)

        monkeypatch.setattr(
            wikimedia_ranges, "plan_wikimedia_ranges", change_after_verification
        )
        with pytest.raises(ValueError, match="dump changed after pin verification"):
            acquire(project, work)
        assert hashed == [dump_path]
        assert not calls and not scanned
        assert not (base / "acquisition.json").exists()
        return
    if case == "invalid_suffix":
        with pytest.raises(ValueError, match="page IDs or offset differ"):
            acquire(project, work)
        assert not calls and not scanned
        assert hashed == [dump_path]
        assert not (base / "acquisition.json").exists()
        assert decoded == [len(compressed[0])]
        assert not list((base / "snapshots/one").glob("*/manifest.json"))
        return
    if case == "damaged":
        (group,) = (parts / "ranges/wikimedia-plan-group").glob("*/ranges.json")
        group.write_bytes(b"damaged")

    recovered = acquire(project, work, transport_cache=cache)
    manifest, _ = _snapshot_payload(recovered)
    starts = [sum(map(len, compressed[:i])) for i in range(len(compressed))]
    assert scanned == (starts if case == "damaged" else starts[2:])
    assert not calls and not parts.exists()
    assert decoded == [len(compressed[0])]
    assert hashed == [dump_path]
    if cache is not None:
        control = acquire(project, tmp_path / "cache-copy", transport_cache=cache)
        assert _snapshot_payload(control) == _snapshot_payload(recovered)
        assert not calls and hashed == [dump_path]

    local_dump = tmp_path / "dump.bz2"
    index_path = tmp_path / "index.bz2"
    local_dump.write_bytes(b"".join(compressed))
    index_path.write_bytes(responses[index["uri"]])
    control = wikimedia_ranges.plan_wikimedia_ranges(
        project.sources[0],
        local_dump,
        index_path,
        index_sha256=index["expected_sha256"],
        max_index_bytes=index["max_bytes"],
        max_index_expanded_bytes=index["max_expanded_bytes"],
        max_members=index["max_members"],
    )
    assert hashed == [dump_path, local_dump]
    assert manifest["retrieval"]["multistream_index"][
        "plan_sha256"
    ] == canonical_digest(asdict(control))


@pytest.mark.parametrize(
    "case", ["scan_boundary", "selected_inside", "damaged", "byte_cap"]
)
def test_wikimedia_indexed_acquisition_matches_serial_and_reuses_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    from klab.corpus import acquisition, wikimedia_ranges

    monkeypatch.delenv("KLAB_TRANSPORT_CACHE", raising=False)
    recipe, responses, pieces, _, index_text = _multistream_wikimedia_fixture(
        tmp_path, shared_members=case != "scan_boundary"
    )
    source_path = recipe.parent / "sources/one.yaml"
    declaration = yaml.safe_load(source_path.read_text())
    spec = declaration["acquisition"]
    spec.update(
        max_scanned_pages=2 if case == "scan_boundary" else 4,
        max_selected_pages=2 if case == "selected_inside" else 4,
        max_emitted_bytes=4096,
    )
    _yaml(source_path, declaration)
    calls = _serve_pinned(monkeypatch, responses)
    control = acquire(load_project(recipe), tmp_path / "serial")
    serial_manifest, serial_output = _snapshot_payload(control)
    assert len(calls) == 2
    if case == "byte_cap":
        spec["max_emitted_bytes"] = sum(map(len, serial_output.values())) - 1
        _yaml(source_path, declaration)
        with pytest.raises(ValueError, match="emitted JSONL exceeds limit"):
            acquire(load_project(recipe), tmp_path / "serial-cap")
    calls.clear()
    index = _fixture_index_options(responses, pieces, index_text)
    spec["multistream_index"] = index
    _yaml(source_path, declaration)
    indexed = load_project(recipe)
    work = tmp_path / "indexed"
    if case == "byte_cap":
        with pytest.raises(ValueError, match="emitted JSONL exceeds limit"):
            acquire(
                indexed,
                work,
                wikimedia_prefetch_bytes=index["max_member_expanded_bytes"],
            )
        assert not (work / "corpora/example/acquisition.json").exists()
        return

    decoded = []
    plans = []
    read_member = wikimedia_ranges.read_wikimedia_member
    read_plan = wikimedia_ranges.plan_wikimedia_ranges
    run_stage = acquisition.run_stage

    def record_member(path, member, **options):
        decoded.append(member.start)
        return read_member(path, member, **options)

    def record_plan(*args, **options):
        plans.append(1)
        return read_plan(*args, **options)

    def interrupt(store, stage, inputs, produce):
        result = run_stage(store, stage, inputs, produce)
        if stage == "wikimedia-member":
            raise OSError("stop after member checkpoint")
        return result

    monkeypatch.setattr(wikimedia_ranges, "read_wikimedia_member", record_member)
    monkeypatch.setattr(wikimedia_ranges, "plan_wikimedia_ranges", record_plan)
    with monkeypatch.context() as fault:
        fault.setattr(acquisition, "run_stage", interrupt)
        with pytest.raises(OSError, match="stop after member checkpoint"):
            acquire(indexed, work)
    base = work / "corpora/example"
    assert not (base / "acquisition.json").exists()
    assert calls == list(responses)
    calls.clear()
    (staged,) = (base / "snapshots/one").glob(".wikimedia_dump-parts-*")
    if case == "damaged":
        (rows,) = (staged / "ranges/wikimedia-member").glob("*/rows.jsonl")
        rows.write_bytes(b"damaged")

    lock = acquire(
        indexed, work, wikimedia_prefetch_bytes=index["max_member_expanded_bytes"]
    )
    manifest, output = _snapshot_payload(lock)
    assert output == serial_output
    assert (
        manifest["retrieval"]["scanned_pages"]
        == serial_manifest["retrieval"]["scanned_pages"]
    )
    assert (
        manifest["retrieval"]["selected_pages"]
        == serial_manifest["retrieval"]["selected_pages"]
    )
    assert (
        manifest["retrieval"]["multistream_index"]["index_sha256"]
        == index["expected_sha256"]
    )
    first, *remaining = decoded
    assert remaining.count(first) == (case == "damaged")
    assert len(set(decoded)) == (1 if case == "scan_boundary" else 2)
    assert len(plans) == 1
    assert not calls and not staged.exists()


@pytest.mark.parametrize("case", ["selected", "scanned", "fallback", "byte_cap"])
def test_wikimedia_process_parsing_preserves_ordered_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    from klab.corpus import acquisition, shards, wikimedia_ranges

    monkeypatch.delenv("KLAB_TRANSPORT_CACHE", raising=False)
    recipe, responses, pieces, _, index_text = _multistream_wikimedia_fixture(
        tmp_path, shared_members=True
    )
    source = recipe.parent / "sources/one.yaml"
    declaration = yaml.safe_load(source.read_text())
    index = _fixture_index_options(responses, pieces, index_text)
    spec = declaration["acquisition"]
    spec.update(
        multistream_index=index,
        max_scanned_pages=3 if case == "scanned" else 4,
        max_selected_pages=4 if case == "scanned" else 2,
        max_emitted_bytes=4096,
    )
    _yaml(source, declaration)
    calls = _serve_pinned(monkeypatch, responses)
    control = acquire(load_project(recipe), tmp_path / "control")
    if case == "byte_cap":
        spec["max_emitted_bytes"] = (
            sum(map(len, _snapshot_payload(control)[1].values())) - 1
        )
        _yaml(source, declaration)
    project = load_project(recipe)
    work = tmp_path / "process"
    worker_pool = shards._worker_pool
    pools = []

    def record_pool(workers, *, threaded):
        pools.append((workers, threaded))
        return worker_pool(workers, threaded=threaded)

    monkeypatch.setattr(shards, "_worker_pool", record_pool)
    for module in (acquisition, shards):
        monkeypatch.setattr(module, "worker_count", lambda: 2)
    monkeypatch.setattr(
        shards,
        "plan_host_workers",
        lambda *_, operator_cap, **__: SimpleNamespace(workers=operator_cap),
    )
    options = {
        "wikimedia_prefetch_bytes": index["max_member_expanded_bytes"],
        "wikimedia_parse_bytes": 2 if case == "fallback" else 8192,
    }
    if case == "byte_cap":
        with pytest.raises(ValueError, match="emitted JSONL exceeds limit"):
            acquire(project, work, **options)
        assert pools == [(2, False)]
        assert not (work / "corpora/example/acquisition.json").exists()
        return
    decoded_by_coordinator = []
    read_member = wikimedia_ranges.read_wikimedia_member

    def record_member(path, member, **kwargs):
        decoded_by_coordinator.append(member.start)
        return read_member(path, member, **kwargs)

    monkeypatch.setattr(wikimedia_ranges, "read_wikimedia_member", record_member)
    run_stage = acquisition.run_stage

    def stop_at_member(store, stage, inputs, produce):
        result = run_stage(store, stage, inputs, produce)
        if stage == "wikimedia-member":
            raise OSError("stop after process member checkpoint")
        return result

    if case == "selected":
        with monkeypatch.context() as fault:
            fault.setattr(acquisition, "run_stage", stop_at_member)
            with pytest.raises(OSError, match="stop after process member checkpoint"):
                acquire(project, work, **options)
        calls.clear()
        (parts,) = (work / "corpora/example/snapshots/one").glob(
            ".wikimedia_dump-parts-*"
        )
        assert (
            len(list((parts / "ranges/wikimedia-member").glob("*/receipt.json"))) == 1
        )
        threaded = acquire(project, work)
        assert not calls and not parts.exists()
        assert len(decoded_by_coordinator) == 1 and decoded_by_coordinator[0] > 0
        direct = acquire(project, tmp_path / "process-cold", **options)
        assert _snapshot_payload(direct) == _snapshot_payload(control)
        assert len(decoded_by_coordinator) == 1
    else:
        threaded = acquire(project, work, **options)
        assert len(decoded_by_coordinator) == (2 if case == "fallback" else 0)
    assert pools == [(2, False)] * (2 if case == "selected" else 1)
    assert _snapshot_payload(threaded) == _snapshot_payload(control)


def test_wikimedia_rebuilds_damaged_group_without_replaying_valid_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus import acquisition, wikimedia_ranges

    monkeypatch.delenv("KLAB_TRANSPORT_CACHE", raising=False)
    recipe, responses, pieces, compressed, index_text = _multistream_wikimedia_fixture(
        tmp_path, one_page_per_member=True
    )
    source_path = recipe.parent / "sources/one.yaml"
    declaration = yaml.safe_load(source_path.read_text())
    declaration["acquisition"].update(
        multistream_index=_fixture_index_options(responses, pieces, index_text),
        max_scanned_pages=4,
        max_selected_pages=2,
        max_emitted_bytes=4096,
    )
    _yaml(source_path, declaration)
    project = load_project(recipe)
    calls = _serve_pinned(monkeypatch, responses)
    control = acquire(project, tmp_path / "control")
    work = tmp_path / "work"
    visited = []
    commits = []
    run_stage = acquisition.run_stage
    read = wikimedia_ranges.read_wikimedia_member

    def record_decode(path, member, **options):
        visited.append(member.start)
        return read(path, member, **options)

    def interrupt(store, stage, inputs, produce):
        result = run_stage(store, stage, inputs, produce)
        if stage == "wikimedia-member":
            commits.append(result[0])
            if len(commits) == 2:
                raise OSError("stop after group commit")
        return result

    monkeypatch.setattr(wikimedia_ranges, "read_wikimedia_member", record_decode)
    with monkeypatch.context() as fault:
        fault.setattr(acquisition, "run_stage", interrupt)
        with pytest.raises(OSError, match="stop after group commit"):
            acquire(project, work, wikimedia_checkpoint_members=2)
    expected = [0, len(compressed[0]), len(compressed[0]) + len(compressed[1])]
    assert visited == expected
    assert not (work / "corpora/example/acquisition.json").exists()
    calls.clear()
    visited.clear()
    first_rows = commits[0] / "rows.jsonl"
    first_rows.write_bytes(b"!" + first_rows.read_bytes()[1:])
    recovered = acquire(project, work, wikimedia_checkpoint_members=2)
    assert visited == expected[:2]
    assert not calls
    assert _snapshot_payload(recovered) == _snapshot_payload(control)


def test_wikimedia_prefetch_overlaps_decode_without_changing_selected_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from klab.cli.main import build_parser
    from klab.corpus import shards, wikimedia_ranges
    from klab.corpus.release_verify import current_settings, verification_settings

    monkeypatch.delenv("KLAB_TRANSPORT_CACHE", raising=False)
    recipe, responses, pieces, _, index_text = _multistream_wikimedia_fixture(
        tmp_path, shared_members=True
    )
    source_path = recipe.parent / "sources/one.yaml"
    declaration = yaml.safe_load(source_path.read_text())
    index = _fixture_index_options(responses, pieces, index_text)
    declaration["acquisition"].update(
        multistream_index=index, max_scanned_pages=4, max_selected_pages=1
    )
    _yaml(source_path, declaration)
    _serve_pinned(monkeypatch, responses)
    project = load_project(recipe)
    original = wikimedia_ranges.read_wikimedia_member
    observed = []
    admissions = []

    def serial_decode(path, member, **options):
        observed.append(member.expanded_bytes)
        return original(path, member, **options)

    def host_plan(stage, *, operator_cap, **_options):
        if stage == "wikimedia member decode":
            admissions.append(operator_cap)
        return SimpleNamespace(workers=operator_cap)

    monkeypatch.setattr(shards, "worker_count", lambda: current_settings().workers)
    monkeypatch.setattr(shards, "plan_host_workers", host_plan)
    monkeypatch.setattr(wikimedia_ranges, "read_wikimedia_member", serial_decode)
    with verification_settings(workers=2):
        serial = acquire(
            project,
            tmp_path / "serial",
            wikimedia_prefetch_bytes=max(map(len, pieces)) - 1,
        )
    assert len(observed) == 1
    assert not admissions

    concurrent = threading.Barrier(2, timeout=10)
    threads = set()

    def parallel_decode(path, member, **options):
        threads.add(threading.get_ident())
        concurrent.wait()
        if member.start != 0:
            raise ValueError("speculative member must not affect the prefix")
        return original(path, member, **options)

    monkeypatch.setattr(wikimedia_ranges, "read_wikimedia_member", parallel_decode)
    args = build_parser().parse_args(
        [
            "--work-dir",
            str(tmp_path / "threaded"),
            "corpus",
            "acquire",
            str(recipe),
            "--cold",
            "--workers",
            "2",
            "--wikimedia-prefetch-bytes",
            str(max(map(len, pieces))),
        ]
    )
    args.handler(args)
    threaded = json.loads(capsys.readouterr().out)
    assert admissions == [2]
    assert len(threads) == 2
    assert (
        threaded["sources"]["one"]["snapshot_sha256"]
        == serial["sources"]["one"]["snapshot_sha256"]
    )
    assert _snapshot_payload(threaded) == _snapshot_payload(serial)


def test_wikimedia_download_checks_sha1_even_with_matching_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"corrupt compressed body"
    recipe, responses = _wikimedia_fixture(
        tmp_path, body, sha1="0" * 40, sha256=hashlib.sha256(body).hexdigest()
    )
    _serve_pinned(monkeypatch, responses)
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="Wikimedia compressed SHA-1 mismatch"):
        acquire(load_project(recipe), work)
    base = work / "corpora/example"
    assert not (base / "acquisition.json").exists()
    assert not list(
        (base / "snapshots/one").glob(".wikimedia_dump-parts-*/input/*.bz2")
    )


@pytest.mark.parametrize(
    "stop",
    ["before_promote", "partial_download", "extraction", "damaged_body", "sealed"],
)
def test_wikimedia_retry_verifies_and_reuses_completed_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    from klab.corpus import acquisition, acquisition_parts

    monkeypatch.delenv("KLAB_TRANSPORT_CACHE", raising=False)
    xml = (
        '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" xml:lang="en">'
        "<siteinfo><dbname>enwiki</dbname></siteinfo><page><title>Article</title>"
        "<ns>0</ns><id>1</id><revision><id>2</id>"
        "<timestamp>2026-10-01T00:00:00Z</timestamp>"
        "<text>A pinned example.</text></revision></page></mediawiki>"
    )
    body = bz2.compress(xml.encode())
    sha1 = hashlib.sha1(body).hexdigest()
    recipe, responses = _wikimedia_fixture(tmp_path, body, sha1=sha1)
    calls = _serve_pinned(monkeypatch, responses)
    project = load_project(recipe)
    control = acquire(project, tmp_path / "control")
    assert calls == list(responses)
    calls.clear()
    work = tmp_path / "work"
    real_verify = acquisition.verify_snapshot
    real_extract = acquisition._wiki_jsonl
    real_rename = acquisition_parts.rename_noreplace
    extractions = []

    def stop_before_verification(path, **options):
        if options.get("_staged"):
            raise OSError("interrupted Wikimedia acquisition")
        return real_verify(path, **options)

    def observe_extraction(source, stream, output, resources):
        extractions.append(output)
        if stop in {"extraction", "damaged_body"} and len(extractions) == 1:
            output.write_bytes(b"incomplete output")
            raise OSError("interrupted Wikimedia acquisition")
        return real_extract(source, stream, output, resources)

    def stop_before_promote(source, target):
        if source.name == ".download":
            raise OSError("interrupted Wikimedia acquisition")
        return real_rename(source, target)

    monkeypatch.setattr(acquisition, "_wiki_jsonl", observe_extraction)
    with monkeypatch.context() as fault:
        if stop == "sealed":
            fault.setattr(acquisition, "verify_snapshot", stop_before_verification)
        elif stop in {"before_promote", "partial_download"}:
            fault.setattr(acquisition_parts, "rename_noreplace", stop_before_promote)
        with pytest.raises(OSError, match="interrupted Wikimedia acquisition"):
            acquire(project, work)
    base = work / "corpora/example"
    assert not (base / "acquisition.json").exists()
    assert calls == list(responses)
    calls.clear()
    (parts,) = (base / "snapshots/one").glob(".wikimedia_dump-parts-*")
    (pinned,) = (parts / "input").iterdir()
    assert pinned.read_bytes() == body
    if stop == "damaged_body":
        pinned.write_bytes(b"!" + body[1:])
    elif stop == "partial_download":
        pinned.write_bytes(body[: len(body) // 2])

    resumed = acquire(project, work)
    assert calls == (
        list(responses) if stop in {"damaged_body", "partial_download"} else []
    )
    assert len(extractions) == (2 if stop in {"extraction", "damaged_body"} else 1)
    assert not parts.exists()
    actual = _snapshot_payload(resumed)
    assert actual == _snapshot_payload(control)
    assert actual[0]["retrieval"]["source_sha256"] == hashlib.sha256(body).hexdigest()


def test_http_document_cache_replays_the_recorded_retrieval(
    tmp_path: Path,
) -> None:
    class Handler(BaseHTTPRequestHandler):
        calls = 0

        def do_GET(self) -> None:
            Handler.calls += 1
            self.send_response(200)
            self.send_header("ETag", "tag-1")
            self.end_headers()
            self.wfile.write(b"# bounded document\n")

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        recipe = _fixture(
            tmp_path,
            kind="http_document",
            acquisition={
                "expected_sha256": hashlib.sha256(b"# bounded document\n").hexdigest(),
                "max_bytes": 100,
            },
            uri=f"http://127.0.0.1:{server.server_port}/manual.md",
        )
        cache = tmp_path / "cold"
        first = acquire(load_project(recipe), tmp_path / "work", transport_cache=cache)
        _widen_cap(recipe)
        second = acquire(load_project(recipe), tmp_path / "work", transport_cache=cache)
        assert Handler.calls == 1
        assert _snapshot_payload(first)[1] == _snapshot_payload(second)[1]
        assert second["sources"]["one"]["receipt"]["retrieval"]["etag"] == "tag-1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_pattern_selected_git_repository_is_reused_and_rebuilt_when_damaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus import acquisition

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Test")
    (repo / "docs").mkdir()
    (repo / "docs/guide.md").write_text("first\n", encoding="utf-8")
    (repo / "docs/empty.md").write_bytes(b"")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "first")
    recipe = _fixture(
        tmp_path,
        kind="git",
        acquisition={"include": ["docs/*.md"], "max_bytes": 1024},
        revision=_git(repo, "rev-parse", "HEAD"),
        uri=str(repo),
    )
    cache = tmp_path / "cold"
    fetches = []
    blob_reads = []
    real = acquisition._git
    popen = subprocess.Popen

    def watch(args: list[str], **options: object) -> bytes:
        if "fetch" in args:
            fetches.append(args)
        return real(args, **options)

    def watch_process(args, *options, **kwargs):
        if isinstance(args, list) and "cat-file" in args:
            blob_reads.append(args[args.index("cat-file") + 1 :])
        return popen(args, *options, **kwargs)

    monkeypatch.setattr(acquisition, "_git", watch)
    monkeypatch.setattr(subprocess, "Popen", watch_process)
    first = acquire(load_project(recipe), tmp_path / "w1", transport_cache=cache)
    assert len(fetches) == 1
    assert blob_reads == [["--batch"]]
    assert _snapshot_payload(first)[1] == {
        "docs/empty.md": b"",
        "docs/guide.md": b"first\n",
    }
    _widen_cap(recipe)
    second = acquire(load_project(recipe), tmp_path / "w2", transport_cache=cache)
    assert len(fetches) == 1
    assert blob_reads == [["--batch"], ["--batch"]]
    assert second["sources"]["one"]["transport_cache"][0]["served_from"] == "cache"
    assert _snapshot_payload(first)[1] == _snapshot_payload(second)[1]

    (repository,) = (cache / "git").iterdir()
    for obj in repository.glob("objects/??/*"):
        obj.chmod(0o644)
        obj.write_bytes(b"garbage")
    _widen_cap(recipe)
    third = acquire(load_project(recipe), tmp_path / "w3", transport_cache=cache)
    assert len(fetches) == 2
    assert third["sources"]["one"]["transport_cache"][0]["served_from"] == "network"
    assert _snapshot_payload(first)[1] == _snapshot_payload(third)[1]


def test_concurrent_writers_and_readers_leave_one_valid_entry(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from klab.corpus.transport_cache import BodyKey, TransportCache

    body = b"x" * 3_000_000
    source = tmp_path / "download"
    source.write_bytes(body)
    key = BodyKey(
        "archive",
        "https://example.test/a.tar.gz",
        sha256=hashlib.sha256(body).hexdigest(),
    )
    root = tmp_path / "cold"

    def work(index: int) -> str:
        cache = TransportCache(root)
        if index % 2:
            cache.store(key, source)
            return "stored"
        served = cache.lookup(key, tmp_path / f"copy-{index}")
        if served is not None:
            assert (tmp_path / f"copy-{index}").read_bytes() == body
        return "hit" if served else "miss"  # never a torn read

    with ThreadPoolExecutor(8) as pool:
        outcomes = list(pool.map(work, range(16)))
    assert outcomes.count("stored") == 8
    cache = TransportCache(root)
    assert [r["status"] for r in cache.entries(verify=True)] == ["ok"]
    assert len(_cache_files(root)) == 1
    assert not list((root / "tmp").iterdir())


def test_cache_fill_downloads_bodies_but_makes_no_snapshot_then_acquire_is_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from klab.cli.main import build_parser
    from klab.corpus.cli import _handle

    content = b'{"text":"one"}\n'
    recipe = _bounded_hf_fixture(tmp_path, content, "default/train/data.jsonl", rows=1)
    cache, root = tmp_path / "cold", tmp_path / "work"
    configured_cache = str(tmp_path / "environment-cache")
    monkeypatch.setenv("KLAB_TRANSPORT_CACHE", configured_cache)
    calls = _mock_hf_stream(monkeypatch, content)

    def run(*arguments: str) -> object:
        _handle(
            build_parser().parse_args(
                [
                    "--work-dir",
                    str(root),
                    "corpus",
                    *arguments,
                    "--transport-cache",
                    str(cache),
                ]
            )
        )
        return json.loads(capsys.readouterr().out)

    filled = run("cache-fill", str(recipe))
    assert filled["sources"][0]["bodies"][0]["served_from"] == "network"
    assert os.environ["KLAB_TRANSPORT_CACHE"] == configured_cache
    assert len(calls) == 1
    assert not list((root / "corpora/example/snapshots").rglob("manifest.json"))
    assert not (root / "corpora/example/acquisition.json").exists()
    assert (
        run("cache-fill", str(recipe))["sources"][0]["bodies"][0]["served_from"]
        == "cache"
    )
    assert len(calls) == 1
    (entry,) = run("cache-list", "--verify")
    assert entry["status"] == "ok" and entry["kind"] == "huggingface_shard"

    acquire(load_project(recipe), root, transport_cache=cache)
    assert len(calls) == 1

    # Prune removes what is unusable and keeps every valid entry.
    (body,) = _cache_files(cache)
    orphan = body.parent / ("0" * 64)
    orphan.write_bytes(b"orphan")
    old = 1.0
    os.utime(orphan, (old, old))
    (cache / "keys" / ("1" * 64 + ".json")).write_text("not json")
    removed = run("cache-prune")
    assert removed["bodies"] == ["0" * 64]
    assert removed["entries"] == ["1" * 64]
    assert len(run("cache-list")) == 1
    assert body.exists()
