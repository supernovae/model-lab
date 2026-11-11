"""Acquisition identity, strict recipes, and offline byte verification."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from klab.corpus.acquisition import (
    acquire,
    declaration_sha256,
    verify_acquisition,
    verify_snapshot,
)
from klab.corpus.project import load_project, source_declaration_payload
from klab.corpus.transport_budget import TransportBudget


def _yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def _fixture(
    tmp_path: Path,
    *,
    kind: str = "local",
    acquisition: dict | None = None,
    revision: str = "v1",
    uri: str = "fixture://source",
    license: str = "MIT",
) -> Path:
    root = tmp_path / "recipe"
    root.mkdir(exist_ok=True)
    _yaml(
        root / "corpus.yaml",
        {
            "schema_version": 1,
            "id": "example",
            "sources": ["sources/one.yaml"],
            "transforms": [],
            "splits": "splits.yaml",
            "release": "release.yaml",
        },
    )
    _yaml(
        root / "splits.yaml",
        {"schema_version": 1, "unit": "document", "assignments": {"document": "train"}},
    )
    _yaml(
        root / "release.yaml",
        {
            "schema_version": 1,
            "mixture": {"technical_docs": 1.0},
            "lm": {"selected": True, "training_splits": ["train", "validation"]},
            "chat": {"selected": False, "training_splits": []},
        },
    )
    _yaml(
        root / "sources/one.yaml",
        {
            "schema_version": 1,
            "id": "one",
            "kind": kind,
            "canonical_uri": uri,
            "revision": revision,
            "license": license,
            "redistribution": "redistributable",
            "domains": ["technical_docs"],
            "document_kinds": ["markdown"],
            "source_family": "docs",
            "acquisition": acquisition
            or {
                "files": [{"path": "sources/data.md", "name": "data.md"}],
                "max_bytes": 1024,
            },
        },
    )
    return root / "corpus.yaml"


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
    source["schema_version"] = 2
    source.pop("redistribution")
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
    release.update(schema_version=2, publication_mode="metadata_reconstruction_only")
    _yaml(release_path, release)
    lock = acquire(load_project(recipe), tmp_path / "work")
    source = load_project(recipe).sources[0]
    legacy_payload = source.model_dump(mode="json")
    legacy_payload.pop("explicit_training_restriction")
    legacy_payload["acquisition"].pop("tree_oid")
    legacy_payload["acquisition"].pop("bounded_blobs")
    assert source_declaration_payload(source) == legacy_payload
    snapshot = verify_snapshot(Path(lock["sources"]["one"]["snapshot_path"]))
    assert {item["path"] for item in snapshot["files"]} == {
        "docs/guide.md",
        "license-metadata.json",
    }
    assert (
        Path(lock["sources"]["one"]["snapshot_path"]) / "files/license-metadata.json"
    ).read_text() == json.dumps(metadata)
    assert acquire(load_project(recipe), tmp_path / "work", offline=True) == lock


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


def test_hf_legacy_declaration_hash_and_receipt_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b'{"text":"one"}\n'
    recipe = _bounded_hf_fixture(tmp_path, content, "default/train/data.jsonl", rows=1)
    project = load_project(recipe)
    source = project.sources[0]
    expected = source.model_dump(mode="json")
    expected.pop("rights")
    expected.pop("explicit_training_restriction")
    expected["acquisition"].pop("include")
    expected["acquisition"].pop("max_decompressed_bytes")
    for shard in expected["acquisition"]["bounded_shards"]:
        shard.pop("declared_config")
        shard.pop("declared_split")
    assert source_declaration_payload(source) == expected
    assert (
        declaration_sha256(source)
        == hashlib.sha256(
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert (
        declaration_sha256(source)
        == "feae644fd3f3070beab148748c16f77ccae4654af7e322abedb00c39e8926b25"
    )
    calls = _mock_hf_stream(monkeypatch, content)
    first = acquire(project, tmp_path / "work")
    manifest = verify_snapshot(first["sources"]["one"]["snapshot_path"])
    assert manifest["declaration_sha256"] == declaration_sha256(source)
    assert "transport_budget" not in manifest["retrieval"]
    assert "row_encoding" not in manifest["retrieval"]
    assert acquire(project, tmp_path / "work") == first
    assert len(calls) == 1


def _bounded_git_fixture(tmp_path: Path, content: bytes) -> tuple[Path, str]:
    oid = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    recipe = _fixture(
        tmp_path,
        kind="git",
        uri="https://github.com/org/repo",
        revision="a" * 40,
        acquisition={
            "max_bytes": len(content),
            "tree_oid": "b" * 40,
            "bounded_blobs": [
                {"path": "docs/one.md", "git_blob_oid": oid, "max_bytes": len(content)}
            ],
        },
    )
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    config["transport_budget"] = {
        "attempt_id": "git_fixture",
        "max_source_body_bytes": len(content) * 2,
        "max_metadata_body_bytes": 10000,
        "max_transfers": 2,
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
    wrong_tree: bool = False,
    corrupt: bool = False,
    interrupt: bool = False,
    extra_body: bool = False,
) -> None:
    class Response(io.BytesIO):
        def __init__(self, body: bytes, url: str):
            super().__init__(body)
            self.url = url
            self.status = 200
            self.headers = {"Content-Length": str(len(body))}

        def read(self, size: int = -1) -> bytes:
            if interrupt and "raw.githubusercontent.com" in self.url:
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
                            "path": "docs/wrong.md" if wrong_tree else "docs/one.md",
                            "type": "blob",
                            "mode": "100644",
                            "sha": oid,
                            "size": len(content),
                        }
                    ],
                }
            else:
                assert (
                    url
                    == "https://raw.githubusercontent.com/org/repo/"
                    + "a" * 40
                    + "/docs/one.md"
                )
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


def test_bounded_git_blob_metadata_checksum_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    _mock_bounded_git_http(monkeypatch, content, oid, interrupt=True)
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
    _mock_bounded_git_http(monkeypatch, content, oid, interrupt=True)
    for _ in range(2):
        with pytest.raises(OSError, match="interrupted"):
            acquire(project, exhausted_root)
    with pytest.raises(ValueError, match="retry allowance exhausted"):
        acquire(project, exhausted_root)
    assert TransportBudget(exhausted_ledger, project).receipt()["source_charged"] == (
        2 * len(content)
    )


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
    declaration["schema_version"] = 2
    declaration.pop("redistribution")
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
    release.update(schema_version=2, publication_mode="metadata_reconstruction_only")
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
