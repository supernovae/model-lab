"""Authenticated Hub acquisition does not put credentials into corpus identities."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import huggingface_hub
import pytest
import yaml
from huggingface_hub.errors import HfHubHTTPError
from requests import Response

from klab import hf_auth
from klab.cli.main import build_parser, main
from klab.config.models import DatasetConfig
from klab.corpus.acquisition import (
    _acquire_http,
    _PrivateHubRedirect,
    acquire,
    verify_snapshot,
)
from klab.corpus.project import SourceDeclaration, load_project
from klab.data import datasets


def test_hub_token_authorizes_without_changing_frozen_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "a" * 40
    upstream = tmp_path / revision
    source_file = upstream / "default/train/data.jsonl"
    source_file.parent.mkdir(parents=True)
    source_file.write_text('{"text":"A pinned source."}\n', encoding="utf-8")
    project_dir = tmp_path / "recipe"
    (project_dir / "sources").mkdir(parents=True)
    declarations = {
        "corpus.yaml": {
            "schema_version": 1,
            "id": "auth-fixture",
            "sources": ["sources/hub.yaml"],
            "transforms": [],
            "splits": "splits.yaml",
            "release": "release.yaml",
        },
        "splits.yaml": {
            "schema_version": 1,
            "unit": "document",
            "assignments": {"fixture": "train"},
        },
        "release.yaml": {
            "schema_version": 1,
            "mixture": {"technical_docs": 1},
            "lm": {"selected": False, "training_splits": []},
            "chat": {"selected": False, "training_splits": []},
        },
        "sources/hub.yaml": {
            "schema_version": 1,
            "id": "hub",
            "kind": "huggingface_dataset",
            "canonical_uri": "org/example",
            "revision": revision,
            "license": "MIT",
            "redistribution": "redistributable",
            "domains": ["technical_docs"],
            "document_kinds": ["text"],
            "source_family": "fixture",
            "acquisition": {
                "config": "default",
                "split": "train",
                "include": ["default/train/*"],
                "text_field": "text",
                "max_rows": 1,
                "max_bytes": 1024,
            },
        },
    }
    for name, data in declarations.items():
        (project_dir / name).write_text(yaml.safe_dump(data), encoding="utf-8")
    project = load_project(project_dir / "corpus.yaml")
    supplied: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        supplied.append(kwargs)
        return str(upstream)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(hf_auth, "get_token", lambda: "private-test-token")
    authenticated = acquire(project, tmp_path / "authenticated")
    snapshot = authenticated["sources"]["hub"]
    assert supplied[-1]["token"] == "private-test-token"
    assert "private-test-token" not in json.dumps(authenticated)
    assert "private-test-token" not in json.dumps(
        verify_snapshot(Path(snapshot["snapshot_path"]))
    )
    monkeypatch.setenv("HF_TOKEN", " \n")
    assert acquire(project, tmp_path / "authenticated", offline=True) == authenticated
    assert len(supplied) == 1
    monkeypatch.delenv("HF_TOKEN")

    monkeypatch.setattr(hf_auth, "get_token", lambda: None)
    anonymous = acquire(project, tmp_path / "anonymous")
    assert "token" not in supplied[-1]
    assert anonymous["sources"]["hub"]["snapshot_sha256"] == snapshot["snapshot_sha256"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "klab",
            "--work-dir",
            str(tmp_path / "cli"),
            "--hf-token-file",
            str(tmp_path / "missing-token"),
            "corpus",
            "acquire",
            str(project_dir / "corpus.yaml"),
        ],
    )
    with pytest.raises(SystemExit, match="Hugging Face token file is unreadable"):
        main()
    assert len(supplied) == 2
    assert not (tmp_path / "cli/corpora/auth-fixture/acquisition.json").exists()


def test_http_hub_token_never_leaks_to_other_origins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = "private-test-token"
    monkeypatch.setattr(hf_auth, "get_token", lambda: token)
    body = b"# Pinned text\n"
    calls: list[urllib.request.Request] = []

    class Response(io.BytesIO):
        url = "https://huggingface.co/datasets/org/example/resolve/main/note.md"
        headers: dict[str, str]

        def __init__(self, content: bytes) -> None:
            super().__init__(content)
            self.headers = {}

    class Opener:
        def open(self, request: urllib.request.Request, *, timeout: int) -> Response:
            assert timeout == 30
            calls.append(request)
            return Response(body)

    monkeypatch.setattr(urllib.request, "build_opener", lambda redirect: Opener())
    declaration = {
        "schema_version": 1,
        "id": "document",
        "kind": "http_document",
        "canonical_uri": Response.url,
        "revision": "pinned-source-v1",
        "license": "MIT",
        "redistribution": "redistributable",
        "domains": ["technical_docs"],
        "document_kinds": ["markdown"],
        "source_family": "notes",
        "acquisition": {
            "expected_sha256": hashlib.sha256(body).hexdigest(),
            "max_bytes": 100,
        },
    }
    staging = tmp_path / "staging"
    (staging / "files").mkdir(parents=True)
    inventory, retrieval = _acquire_http(
        SourceDeclaration.model_validate(declaration), staging, False
    )
    assert inventory[0]["sha256"] == hashlib.sha256(body).hexdigest()
    assert retrieval["final_url"] == Response.url
    assert calls[-1].get_header("Authorization") == f"Bearer {token}"

    redirect = _PrivateHubRedirect()
    moved = redirect.redirect_request(
        calls[-1], None, 302, "Found", {}, "https://cdn.example.net/document.md"
    )
    assert moved is not None and moved.get_header("Authorization") is None
    same_origin = redirect.redirect_request(
        calls[-1], None, 302, "Found", {}, Response.url + "?download=1"
    )
    assert same_origin is not None
    assert same_origin.get_header("Authorization") == f"Bearer {token}"

    declaration["canonical_uri"] = "https://example.net/document.md"
    other = tmp_path / "other"
    (other / "files").mkdir(parents=True)
    _acquire_http(SourceDeclaration.model_validate(declaration), other, False)
    assert calls[-1].get_header("Authorization") is None


@pytest.fixture(autouse=True)
def isolated_hf_credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    hf_auth.set_token_file(None)
    yield
    hf_auth.set_token_file(None)


def test_credential_precedence_and_strict_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hf_auth, "get_token", lambda: "stored-login")
    assert hf_auth.hub_auth_kwargs() == {"token": "stored-login"}
    monkeypatch.setenv("HF_TOKEN", " env-token \n")
    assert hf_auth.hub_auth_kwargs() == {"token": "env-token"}
    token_file = tmp_path / "secret"
    token_file.write_text(" file-token\n", encoding="utf-8")
    hf_auth.set_token_file(token_file)
    assert hf_auth.hub_auth_kwargs() == {"token": "file-token"}
    assert (
        build_parser()
        .parse_args(
            ["--hf-token-file", str(token_file), "corpus", "acquire", "recipe.yaml"]
        )
        .hf_token_file
        == token_file
    )
    token_file.write_text(" \n", encoding="utf-8")
    with pytest.raises(ValueError, match="token file") as error:
        hf_auth.hub_auth_kwargs()
    assert "env-token" not in str(error.value)
    token_file.write_text("a\nb", encoding="utf-8")
    with pytest.raises(ValueError, match="token file"):
        hf_auth.hub_auth_kwargs()
    token_file.unlink()
    with pytest.raises(ValueError, match="unreadable"):
        hf_auth.hub_auth_kwargs()
    hf_auth.set_token_file(None)
    monkeypatch.setenv("HF_TOKEN", " \n")
    with pytest.raises(ValueError, match="HF_TOKEN"):
        hf_auth.hub_auth_kwargs()
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setattr(hf_auth, "get_token", lambda: None)
    assert hf_auth.hub_auth_kwargs() == {}


class Denied(HfHubHTTPError):
    def __init__(self, status: int, secret: str) -> None:
        response = Response()
        response.status_code = status
        super().__init__(f"request Authorization: Bearer {secret}", response=response)


@pytest.mark.parametrize("status", [401, 403])
def test_hub_snapshot_denial_redacts_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    token = "DO-NOT-LOG-hub-token"
    monkeypatch.setenv("HF_TOKEN", token)
    source = SourceDeclaration.model_validate(
        {
            "schema_version": 1,
            "id": "hub",
            "kind": "huggingface_dataset",
            "canonical_uri": "org/example",
            "revision": "a" * 40,
            "license": "MIT",
            "redistribution": "redistributable",
            "domains": ["technical_docs"],
            "document_kinds": ["text"],
            "source_family": "example",
            "acquisition": {
                "config": "default",
                "split": "train",
                "include": ["default/train/*"],
                "text_field": "text",
                "max_rows": 1,
                "max_bytes": 100,
            },
        }
    )

    def reject(**kwargs: object) -> str:
        assert kwargs["token"] == token
        raise Denied(status, token)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", reject)
    from klab.corpus.acquisition import _acquire_hf

    with pytest.raises(RuntimeError, match="Hugging Face access denied") as error:
        _acquire_hf(source, tmp_path, tmp_path / "cache", False)
    assert token not in str(error.value)
    assert error.value.__cause__ is None
    assert not list(tmp_path.rglob("manifest.json"))


@pytest.mark.parametrize("status", [401, 403])
def test_http_hub_denial_redacts_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    token = "DO-NOT-LOG-http-token"
    monkeypatch.setenv("HF_TOKEN", token)
    source = SourceDeclaration.model_validate(
        {
            "schema_version": 1,
            "id": "document",
            "kind": "http_document",
            "canonical_uri": "https://huggingface.co/datasets/org/private/resolve/main/note.md",
            "revision": "pinned-v1",
            "license": "MIT",
            "redistribution": "redistributable",
            "domains": ["technical_docs"],
            "document_kinds": ["markdown"],
            "source_family": "notes",
            "acquisition": {"expected_sha256": "0" * 64, "max_bytes": 100},
        }
    )

    class Opener:
        def open(self, request: urllib.request.Request, *, timeout: int) -> object:
            assert request.get_header("Authorization") == f"Bearer {token}"
            raise urllib.error.HTTPError(request.full_url, status, token, {}, None)

    monkeypatch.setattr(urllib.request, "build_opener", lambda _: Opener())
    (tmp_path / "files").mkdir()
    with pytest.raises(RuntimeError, match="Hugging Face access denied") as error:
        _acquire_http(source, tmp_path, False)
    assert token not in str(error.value)


@pytest.mark.parametrize("status", [401, 403])
def test_streaming_denial_at_iteration_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    token = "DO-NOT-LOG-stream-token"
    monkeypatch.setenv("HF_TOKEN", token)

    def reject(*args: object, **kwargs: object) -> Iterator[dict[str, str]]:
        assert kwargs["token"] == token
        yield from ()
        raise Denied(status, token)

    monkeypatch.setattr(datasets, "load_dataset", reject)
    config = DatasetConfig(
        source="huggingface",
        dataset_id="example/corpus",
        license="CC0-1.0",
        revision="a" * 40,
        cache_dir=tmp_path,
        train_max_documents=1,
        validation_max_documents=1,
        train_max_tokens=10,
        validation_max_tokens=10,
    )
    with pytest.raises(RuntimeError, match="Hugging Face access denied") as error:
        list(datasets.iter_documents(config, "train"))
    assert token not in str(error.value)
