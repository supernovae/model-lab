import json
from dataclasses import replace

from streamlit.testing.v1 import AppTest

from sparselab.dashboard.generation import service_config
from sparselab.dashboard.queries import snapshot
from sparselab.training.metrics import ExperimentStore


def _page(root, run):
    from pathlib import Path

    from sparselab.dashboard.generation import generation_page
    from sparselab.dashboard.queries import snapshot

    generation_page(snapshot(Path(root)), Path(root), [run])


def test_service_is_loopback_only(tmp_path, monkeypatch):
    config = {
        "url": "http://127.0.0.1:8502",
        "store": str(tmp_path),
        "run": "a",
        "checkpoint": "checkpoints/step_1",
        "checkpoint_sha256": "digest",
    }
    path = tmp_path / "service.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("SPARSELAB_SNAPSHOT_SERVICE_CONFIG", str(path))
    assert service_config() == config
    import pytest

    for url in (
        "https://example.org",
        "http://127.0.0.1:8502/other",
        "http://user@127.0.0.1:8502",
    ):
        path.write_text(json.dumps(dict(config, url=url)))
        with pytest.raises(ValueError):
            service_config()


def test_empty_snapshot_metadata_does_not_load_or_embed(tmp_path, monkeypatch):
    config = {
        "url": "http://127.0.0.1:8502",
        "store": str(tmp_path),
        "run": "a",
        "checkpoint": "checkpoints/step_1",
        "checkpoint_sha256": "digest",
    }
    path = tmp_path / "service.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("SPARSELAB_SNAPSHOT_SERVICE_CONFIG", str(path))
    ExperimentStore(tmp_path).create_run(
        "a", {"name": "generic"}, {"purpose": "training"}
    )
    app = AppTest.from_function(_page, args=(str(tmp_path), "a")).run()
    assert not app.exception
    assert not app.get("iframe")
    assert any("No saved checkpoint metadata" in item.value for item in app.info)


def test_unverified_metadata_refuses_embedding(tmp_path, monkeypatch):
    config = {
        "url": "http://127.0.0.1:8502",
        "store": str(tmp_path),
        "run": "a",
        "checkpoint": "checkpoints/step_1",
        "checkpoint_sha256": "expected",
    }
    path = tmp_path / "service.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("SPARSELAB_SNAPSHOT_SERVICE_CONFIG", str(path))
    ExperimentStore(tmp_path).create_run(
        "a", {"name": "generic"}, {"purpose": "training"}
    )
    view = replace(
        snapshot(tmp_path),
        checkpoints=(
            {"run_id": "a", "relative_path": "step_1", "step": 1, "digest": "wrong"},
        ),
    )
    from sparselab.dashboard import queries

    monkeypatch.setattr(queries, "snapshot", lambda root: view)
    app = AppTest.from_function(_page, args=(str(tmp_path), "a")).run()
    assert not app.exception
    assert app.info
    assert not app.get("iframe")
    app.query_params.update(
        store=str(tmp_path), run="a", checkpoint="checkpoints/missing"
    )
    app.run()
    assert app.info
    assert not app.get("iframe")


def test_sampling_controls_reject_invalid_values(tmp_path, monkeypatch):
    import importlib

    import pytest

    config = {
        "url": "http://127.0.0.1:8502",
        "store": str(tmp_path),
        "run": "a",
        "checkpoint": "checkpoints/step_1",
        "checkpoint_sha256": "expected",
    }
    path = tmp_path / "service.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("SPARSELAB_SNAPSHOT_SERVICE_CONFIG", str(path))
    sampler = importlib.import_module("sparselab.dashboard.snapshot_sampler")
    sampler.validate_controls("hello", 32, 0.7, 40, 12)
    sampler.validate_controls("hello", 512, 0.7, 40, 42)
    for args in [
        ("", 32, 0.0, 0, 0),
        ("x", 513, 0.0, 0, 0),
        ("x", 32, float("nan"), 0, 0),
        ("x", 32, 3.0, 0, 0),
        ("x", 32, 0.0, -1, 0),
        ("x", 32, 0.0, 0, -1),
    ]:
        with pytest.raises(ValueError):
            sampler.validate_controls(*args)


def test_supported_sampler_health_and_run_changes(tmp_path, monkeypatch):
    import io

    from sparselab.dashboard import generation, queries

    config = {
        "url": "http://127.0.0.1:8502",
        "store": str(tmp_path),
        "run": "a",
        "checkpoint": "checkpoints/step_1",
        "checkpoint_sha256": "digest",
    }
    path = tmp_path / "service.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("SPARSELAB_SNAPSHOT_SERVICE_CONFIG", str(path))
    store = ExperimentStore(tmp_path)
    for name in ("a", "b"):
        store.create_run(name, {"name": name}, {"purpose": "training"})
    view = replace(
        snapshot(tmp_path),
        checkpoints=tuple(
            {
                "run_id": name,
                "relative_path": "step_1",
                "step": 1,
                "digest": "digest",
                "verification_status": "verified",
                "created_at": "today",
            }
            for name in ("a", "b")
        ),
    )
    monkeypatch.setattr(queries, "snapshot", lambda root: view)
    monkeypatch.setattr(
        generation, "urlopen", lambda *args, **kwargs: io.BytesIO(b"ok")
    )
    app = AppTest.from_function(_page, args=(str(tmp_path), "a")).run()
    assert not app.exception
    assert len(app.get("iframe")) == 1
    other = AppTest.from_function(_page, args=(str(tmp_path), "b")).run()
    assert not other.exception
    assert len(other.get("iframe")) == 1

    def unavailable(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(generation, "urlopen", unavailable)
    app.run()
    assert app.error and not app.get("iframe")


def _sampler_page():
    from sparselab.dashboard.snapshot_sampler import render

    render()


def test_sampler_passes_controls_and_clears_old_output(tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace

    # No model load: verify UI calls the native generator with exactly these controls.
    config = {
        "store": str(tmp_path),
        "run": "a",
        "checkpoint": "checkpoints/step_1",
        "checkpoint_sha256": "digest",
    }
    path = tmp_path / "sampler.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("SPARSELAB_SNAPSHOT_SERVICE_CONFIG", str(path))
    sampler = importlib.import_module("sparselab.dashboard.snapshot_sampler")
    monkeypatch.setattr(sampler, "RUNS", tmp_path)
    monkeypatch.setattr(sampler, "CONFIG", config)
    generation = tmp_path / "a/checkpoints/step_1"
    generation.mkdir(parents=True)
    from sparselab.dashboard.snapshot_catalog import SavedSnapshot, SnapshotOwner

    selected = SavedSnapshot("a", "checkpoints/step_1", "digest", 1, "today", 100)
    monkeypatch.setattr(sampler, "catalog", lambda root: [selected])
    loaded = SimpleNamespace(
        model=None,
        device="cpu",
        engine=None,
        tokenizer=SimpleNamespace(
            encode=lambda *args, **kwargs: SimpleNamespace(ids=[1, 2])
        ),
        identity={
            "run_id": "a",
            "checkpoint_sha256": "digest",
            "tokenizer_sha256": "tok",
        },
    )
    owner = SnapshotOwner()
    monkeypatch.setattr(sampler, "checkpoint", lambda: owner)
    monkeypatch.setattr(sampler, "load_selected", lambda selected: loaded)
    calls = []

    def generate(*args, **kwargs):
        calls.append(dict(kwargs, max_new_tokens=args[4]))
        return args[2] + " continuation"

    monkeypatch.setattr(sampler, "generate", generate)
    app = AppTest.from_function(_sampler_page).run()
    assert not app.exception
    assert app.text_area[0].value == ""
    assert [item.value for item in app.number_input] == [0.7, 40, 42]
    assert app.slider[0].value == 32
    # A newly published checkpoint must not change the model on form submission.
    newer = replace(selected, checkpoint="checkpoints/step_2", digest="newer", step=2)
    monkeypatch.setattr(sampler, "catalog", lambda root: [newer, selected])
    app.text_area[0].set_value("neutral prefix")
    app.number_input[0].set_value(1.1)
    app.number_input[1].set_value(20)
    app.number_input[2].set_value(99)
    app.slider[0].set_value(512)
    next(
        button for button in app.button if button.label == "Generate completion"
    ).click().run()
    assert not app.exception
    assert calls[-1]["temperature"] == 1.1 and calls[-1]["top_k"] == 20
    assert calls[-1]["seed"] == 99
    assert calls[-1]["max_new_tokens"] == 512
    assert app.session_state["completion_result"][3]["checkpoint_sha256"] == "digest"
    assert app.session_state["completion_result"][1] == " continuation"
    count = len(calls)
    app.button(key="reset_sampling").click().run()
    assert [item.value for item in app.number_input] == [0.7, 40, 42]
    assert app.text_area[0].value == "neutral prefix"
    assert "completion_result" not in app.session_state
    assert len(calls) == count
    assert app.slider[0].value == 32
    app.text_area[0].set_value("")
    next(
        button for button in app.button if button.label == "Generate completion"
    ).click().run()
    assert app.warning
    assert "completion_result" not in app.session_state
    app.query_params["run"] = "different"
    app.run()
    assert app.error and not app.text_area
