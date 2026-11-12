"""Native completion flow, exact identities, local API boundaries and safe refresh."""

import io
import json
from dataclasses import replace
from threading import Thread
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from streamlit.testing.v1 import AppTest

from klab.dashboard import generation, snapshot_sampler
from klab.dashboard.snapshot_catalog import SavedSnapshot
from klab.dashboard.snapshot_controls import validate_controls


def _page(root):
    from pathlib import Path

    from klab.dashboard.generation import generation_page

    generation_page(None, Path(root), [])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = {"url": "http://127.0.0.1:8502", "store": str(tmp_path)}
    path = tmp_path / "service.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("KLAB_SNAPSHOT_SERVICE_CONFIG", str(path))
    item = SavedSnapshot(
        "a", "checkpoints/step_0001_gen_0001", "a" * 64, 1, "2026-01-01", 100
    )
    monkeypatch.setattr(generation, "catalog", lambda root: [item])
    calls = []

    def request(config, endpoint, payload=None):
        if endpoint == "/health":
            return {"status": "idle", "identity": "No model loaded"}
        calls.append(payload)
        return {
            "prompt": payload["prompt"],
            "completion": " continuation",
            "tokenizer_sha256": "tok",
            "backend": "cpu",
            "settings": {
                key: value for key, value in payload.items() if key != "prompt"
            },
        }

    monkeypatch.setattr(generation, "service_request", request)
    return tmp_path, config, item, calls


def test_service_is_loopback_only(setup, monkeypatch):
    root, config, _, _ = setup
    assert generation.service_config() == config
    for url in (
        "https://example.org",
        "http://127.0.0.1:8502/other",
        "http://user@127.0.0.1:8502",
    ):
        (root / "service.json").write_text(json.dumps(dict(config, url=url)))
        with pytest.raises(ValueError):
            generation.service_config()


def test_empty_and_missing_snapshot_fail_without_substitution(setup, monkeypatch):
    root, _, item, calls = setup
    monkeypatch.setattr(generation, "catalog", lambda root: [])
    app = AppTest.from_function(_page, args=(str(root),)).run()
    assert not app.exception and app.info and not app.text_area and not calls
    monkeypatch.setattr(generation, "catalog", lambda root: [item])
    app.query_params.update(run="missing", checkpoint="checkpoints/missing")
    app.button(key="use_newest").click().run()
    app.query_params.update(run="missing", checkpoint="checkpoints/missing")
    app.run()
    assert app.error and not app.text_area and not calls
    app.button(key="use_newest").click().run()
    assert app.text_area and not app.error


def test_native_controls_pin_refresh_and_clear_output(setup, monkeypatch):
    root, _, item, calls = setup
    app = AppTest.from_function(_page, args=(str(root),)).run()
    assert not app.exception and not app.get("iframe")
    assert app.text_area[0].value == "" and not calls
    assert [i.value for i in app.number_input] == [0.7, 40, 42]
    newer = replace(
        item, checkpoint="checkpoints/step_0002_gen_0002", digest="b" * 64, step=2
    )
    monkeypatch.setattr(generation, "catalog", lambda root: [newer, item])
    app.text_area[0].set_value("neutral prefix")
    for control, value in zip(app.number_input, (1.1, 20, 99)):
        control.set_value(value)
    app.slider[0].set_value(512)
    next(b for b in app.button if b.label == "Generate completion").click().run()
    assert not app.exception
    assert calls[-1]["checkpoint_sha256"] == item.digest
    assert [
        calls[-1][key] for key in ("temperature", "top_k", "seed", "max_new_tokens")
    ] == [1.1, 20, 99, 512]
    assert app.session_state["completion_result"]["completion"] == " continuation"
    app.button(key="reset_sampling").click().run()
    assert [i.value for i in app.number_input] == [0.7, 40, 42]
    assert app.text_area[0].value == "neutral prefix"
    assert "completion_result" not in app.session_state and len(calls) == 1
    next(b for b in app.button if b.label == "Refresh snapshots").click().run()
    assert app.selectbox[0].value == newer.key
    app.selectbox[0].set_value(item.key).run()
    assert app.query_params["run"] == [item.run]
    next(b for b in app.button if b.label == "Refresh snapshots").click().run()
    assert app.selectbox[0].value == item.key
    app.button(key="use_newest").click().run()
    assert app.selectbox[0].value == newer.key
    # Empty input and service failures never leave a previous response visible.
    app.text_area[0].set_value("")
    next(b for b in app.button if b.label == "Generate completion").click().run()
    assert app.error and "completion_result" not in app.session_state
    app.text_area[0].set_value("hello")

    def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(generation, "service_request", offline)
    next(b for b in app.button if b.label == "Generate completion").click().run()
    assert app.error and "completion_result" not in app.session_state


def test_invalid_response_identity_is_not_displayed(setup, monkeypatch):
    root, _, _, _ = setup
    monkeypatch.setattr(
        generation,
        "service_request",
        lambda *a, **kw: {"settings": {}, "status": "idle", "identity": "none"},
    )
    app = AppTest.from_function(_page, args=(str(root),)).run()
    app.text_area[0].set_value("hello")
    next(b for b in app.button if b.label == "Generate completion").click().run()
    assert app.error and "completion_result" not in app.session_state


@pytest.mark.parametrize(
    "args",
    [
        ("", 32, 0.0, 0, 0),
        ("x", 513, 0.0, 0, 0),
        ("x", 32, float("nan"), 0, 0),
        ("x", True, 0.7, 40, 42),
        ("x", 32, True, 40, 42),
    ],
)
def test_sampling_bounds(args):
    validate_controls("hello", 512, 0.7, 40, 42)
    with pytest.raises(ValueError):
        validate_controls(*args)


def test_sampler_native_generation_reuse_and_context_limit(setup, monkeypatch):
    from klab.evaluation import generation as native

    root, _, item, _ = setup
    sampler = snapshot_sampler.Sampler(root)
    monkeypatch.setattr(snapshot_sampler, "available_catalog", lambda root: [item])
    loads = []
    loaded = SimpleNamespace(
        model=None,
        tokenizer=SimpleNamespace(encode=lambda *a, **kw: SimpleNamespace(ids=[1, 2])),
        device="cpu",
        engine=None,
        identity={
            "run_id": item.run,
            "checkpoint_sha256": item.digest,
            "tokenizer_sha256": "tok",
        },
    )

    def loader(value):
        loads.append(value)
        return loaded

    monkeypatch.setattr(sampler, "load_selected", loader)
    requests = []

    def generate(*args, **kwargs):
        requests.append(kwargs)
        return args[2] + " output"

    monkeypatch.setattr(native, "generate", generate)
    payload = {
        "store": str(root),
        "run": item.run,
        "checkpoint": item.checkpoint,
        "checkpoint_sha256": item.digest,
        "prompt": "hello",
        "max_new_tokens": 32,
        "temperature": 0.7,
        "top_k": 40,
        "seed": 42,
    }
    assert sampler.complete(payload)["completion"] == " output"
    assert sampler.complete(payload)["settings"] == {
        k: v for k, v in payload.items() if k != "prompt"
    }
    assert len(loads) == 1 and requests[-1]["strict_context"] is True
    with pytest.raises(ValueError, match="no substitute"):
        sampler.complete(dict(payload, checkpoint_sha256="b" * 64))
    loaded.tokenizer.encode = lambda *a, **kw: SimpleNamespace(ids=list(range(1000)))
    with pytest.raises(ValueError, match="1,024"):
        sampler.complete(payload)
    assert len(requests) == 2
    with pytest.raises(ValueError, match="configured"):
        sampler.complete(dict(payload, store="/other"))


def test_local_http_transport_and_browser_refusal():
    from http.server import ThreadingHTTPServer

    calls = []
    sampler = SimpleNamespace(
        owner=SimpleNamespace(status=("idle", "none")),
        complete=lambda body: calls.append(body) or {"completion": "ok"},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), snapshot_sampler.handler(sampler))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/health") as response:
            assert json.load(response)["backend"] == "cpu"
        request = Request(
            base + "/complete", data=b"{}", headers={"Content-Type": "application/json"}
        )
        with urlopen(request) as response:
            assert json.load(response)["completion"] == "ok"
        for headers in (
            {"Origin": "https://other.test"},
            {"Host": "attacker.test"},
            {"Content-Type": "text/plain"},
        ):
            req = Request(base + "/complete", data=b"{}", headers=headers)
            with pytest.raises(HTTPError):
                urlopen(req)
        assert calls == [{}]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_client_surfaces_busy_message(monkeypatch):
    def busy(*args, **kwargs):
        raise HTTPError(
            "http://127.0.0.1:8502/complete",
            409,
            "busy",
            {},
            io.BytesIO(b'{"error":"Sampler busy"}'),
        )

    monkeypatch.setattr(generation, "urlopen", busy)
    with pytest.raises(ValueError, match="Sampler busy"):
        generation.service_request({"url": "http://127.0.0.1:8502"}, "/complete", {})


def _summary(root, selected_ids, events):
    from pathlib import Path
    from types import SimpleNamespace

    from klab.dashboard.generation import checkpoint_summary

    checkpoint_summary(Path(root), selected_ids, SimpleNamespace(events=events))


def test_summary_separates_store_selected_saves_and_retention(setup, monkeypatch):
    root, _, item, _ = setup
    newer = replace(item, checkpoint="checkpoints/step_0002_gen_0002", step=2)
    other = replace(item, run="b")
    monkeypatch.setattr(generation, "catalog", lambda root: [newer, item, other])
    event = {
        "kind": "checkpoint_retention",
        "run_id": "a",
        "payload_json": json.dumps(
            {
                "status": "verified",
                "retained": [
                    {"step": 2, "roles": ["latest"]},
                    {"step": 1, "roles": ["previous"]},
                ],
                "prune": ["older"],
            }
        ),
    }
    app = AppTest.from_function(_summary, args=(str(root), ["a"], [event])).run()
    assert not app.exception
    assert [item.value for item in app.metric] == ["3", "2", "2"]
    assert app.dataframe[0].value["Step"].tolist() == [2, 1]
    assert any(
        "2 retained, 1 older save(s) pruned" in item.value for item in app.caption
    )
