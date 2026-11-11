"""Descriptive panels retain every single attempt and authenticate their bindings."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_training import config

from klab.canonical import canonical_digest, canonical_json
from klab.evaluation import panel
from klab.evaluation.suite import run_suite
from klab.training.trainer import train


@pytest.fixture(scope="module")
def panel_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("generation-panel")
    cfg = config(root)
    cfg = cfg.model_copy(
        update={
            "training": cfg.training.model_copy(update={"max_steps": 1}),
            "optimizer": cfg.optimizer.model_copy(update={"warmup_steps": 0}),
        }
    )
    train(cfg, run_id="panel-run")
    suite = root / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "evaluation_suite_version": 1,
                "id": "tiny",
                "evaluations": [{"id": "loss", "role": "gate", "kind": "heldout_lm"}],
            }
        )
    )
    index = run_suite(
        suite, "panel-run", "latest.json", cfg.logging.root_dir, backend="cpu"
    )
    return root, index


def declaration(prompts=None):
    return {
        "generation_panel_version": 1,
        "id": "descriptive",
        "role": "descriptive_not_quality_gate",
        "prompts": ["hello"] if prompts is None else prompts,
        "decoder": {"temperature": 0, "top_k": 0, "max_new_tokens": 3, "seed": 42},
        "checkpoint_selection": "identical to heldout evaluation binding",
    }


def write_panel(root, name, value):
    path = root / name
    path.write_text(json.dumps(value))
    return path


def test_current_panel_declaration_is_supported(tmp_path):
    source = write_panel(tmp_path, "panel.json", declaration(["one", "two", "three"]))
    parsed = panel.load_panel(source)
    assert parsed.decoder.seed == 42
    assert len(parsed.prompts) == 3


@pytest.mark.parametrize(
    "key,value",
    [
        ("seed", True),
        ("seed", -1),
        ("seed", 2**64),
        ("temperature", float("nan")),
        ("temperature", -1),
        ("top_k", 1.5),
        ("max_new_tokens", -1),
    ],
)
def test_decoder_rejects_invalid_values(tmp_path, key, value):
    payload = declaration()
    payload["decoder"][key] = value
    with pytest.raises(ValueError):
        panel.load_panel(write_panel(tmp_path, "panel.json", payload))


@pytest.mark.parametrize(
    "change",
    [
        {"role": "gate"},
        {"rerolls": 2},
        {"prompts": []},
        {"decoder": {"temperature": 0, "top_k": 0, "max_new_tokens": 3}},
    ],
)
def test_no_gates_rerolls_or_implicit_decoder(tmp_path, change):
    with pytest.raises(ValidationError):
        panel.load_panel(write_panel(tmp_path, "panel.json", declaration() | change))


def test_native_generation_and_tamper_detection(panel_run):
    root, index = panel_run
    source = write_panel(root, "native.json", declaration())
    output = panel.run_panel(source, index, backend="cpu")
    record = panel.verify_panel_result(output)
    assert record["checkpoint"] != "latest.json"
    assert record["identity"]["runtime"]["engine"] == "pytorch"
    assert record["rows"][0]["status"] == "COMPLETED"
    assert len(record["rows"][0]["token_ids"]) <= 3
    original = output.read_bytes()
    value = json.loads(original)
    value["rows"][0]["completion"] = "tampered"
    output.write_text(json.dumps(value))
    try:
        with pytest.raises(ValueError):
            panel.verify_panel_result(output)
    finally:
        output.write_bytes(original)
    before = source.read_bytes()
    source.write_text(json.dumps(declaration(["different"])))
    try:
        with pytest.raises(ValueError, match="declaration changed"):
            panel.verify_panel_result(output)
    finally:
        source.write_bytes(before)


def test_empty_repetitive_failed_observations_no_retry(panel_run, monkeypatch):
    root, index = panel_run
    source = write_panel(
        root, "outcomes.json", declaration(["", "repeat", "fail", "last"])
    )
    calls = []
    original_loader = panel.load_run
    load_calls = []

    def loader(*args, **kwargs):
        load_calls.append(kwargs["authorization"])
        return original_loader(*args, **kwargs)

    def generate(model, tokenizer, prompt, context, count, device, **kwargs):
        calls.append((prompt, kwargs))
        if prompt == "fail":
            raise RuntimeError("retained failure")
        if prompt == "repeat":
            return prompt + "aaa", [7, 7, 7]
        return prompt, []

    monkeypatch.setattr(panel, "load_run", loader)
    monkeypatch.setattr(panel, "generate_with_token_ids", generate)
    binding = {"runtime_acceptance_sha256": "a" * 64, "runtime_binding_sha256": None}
    output = panel.run_panel(source, index, backend="cpu", runtime_binding=binding)
    record = panel.verify_panel_result(output)
    assert [row["status"] for row in record["rows"]] == [
        "COMPLETED",
        "COMPLETED",
        "FAILED",
        "COMPLETED",
    ]
    assert record["rows"][0]["text"] == ""
    assert record["rows"][0]["token_ids"] == []
    assert record["rows"][1]["token_ids"] == [7, 7, 7]
    assert record["rows"][2]["token_ids"] is None
    assert record["rows"][2]["error"] == {
        "type": "RuntimeError",
        "message": "retained failure",
    }
    assert record["runtime_binding"] == binding
    assert all(
        kwargs["seed"] == 42 and kwargs["temperature"] == 0 for _, kwargs in calls
    )
    assert (
        panel.run_panel(source, index, backend="cpu", runtime_binding=binding) == output
    )
    assert len(calls) == 4
    assert load_calls == [None, None]


def test_index_authenticated_before_generation(panel_run, monkeypatch):
    root, index = panel_run
    source = write_panel(root, "bad-index-panel.json", declaration())
    bad_index = root / "forged-index.json"
    bad_index.write_bytes(index.read_bytes())
    monkeypatch.setattr(
        panel, "load_run", lambda *a, **kw: pytest.fail("unauthenticated index loaded")
    )
    with pytest.raises(ValueError):
        panel.run_panel(source, bad_index, backend="cpu")


def test_runtime_mismatch_rejected_before_generation(panel_run, monkeypatch):
    from dataclasses import replace

    root, index = panel_run
    source = write_panel(root, "runtime-mismatch.json", declaration(["runtime"]))
    original = panel.load_run

    def wrong_runtime(*args, **kwargs):
        loaded = original(*args, **kwargs)
        identity = loaded.identity | {
            "runtime": loaded.identity["runtime"] | {"backend": "metal"}
        }
        return replace(loaded, identity=identity)

    monkeypatch.setattr(panel, "load_run", wrong_runtime)
    monkeypatch.setattr(
        panel,
        "generate_with_token_ids",
        lambda *a, **kw: pytest.fail("mismatched runtime generated"),
    )
    with pytest.raises(ValueError, match="runtime differs"):
        panel.run_panel(source, index, backend="cpu")


def test_native_engine_and_authorization_forwarded(panel_run, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    root, index = panel_run
    source = write_panel(root, "native-dispatch.json", declaration(["dispatch"]))
    original = panel.load_run
    engine = object()
    authorization = SimpleNamespace(as_dict=lambda: {"verified": "test-authorization"})
    calls = []

    def loader(*args, **kwargs):
        assert kwargs.pop("authorization") is authorization
        loaded = original(*args, **kwargs)
        return replace(loaded, engine=engine)

    def generate(*args, **kwargs):
        assert kwargs["engine"] is engine
        calls.append(kwargs)
        return "dispatch", []

    monkeypatch.setattr(panel, "load_run", loader)
    monkeypatch.setattr(panel, "generate_with_token_ids", generate)
    output = panel.run_panel(source, index, backend="cpu", authorization=authorization)
    record = panel.verify_panel_result(output)
    assert len(calls) == 1
    assert record["runtime_authorization"] == {"verified": "test-authorization"}


def test_structural_coverage_rejects_rehashed_row_loss(panel_run):
    root, index = panel_run
    source = write_panel(root, "coverage.json", declaration(["coverage"]))
    output = panel.run_panel(source, index, backend="cpu")
    original = output.read_bytes()
    record = json.loads(original)
    record["rows"] = []
    record["record_sha256"] = canonical_digest(
        {k: v for k, v in record.items() if k != "record_sha256"}
    )
    output.write_bytes(canonical_json(record) + b"\n")
    try:
        with pytest.raises(ValueError, match="coverage mismatch"):
            panel.verify_panel_result(output)
    finally:
        output.write_bytes(original)


def test_interruption_preserves_previous_negative_and_active_attempt(
    panel_run, monkeypatch
):
    root, index = panel_run
    source = write_panel(
        root, "interrupt.json", declaration(["negative", "interrupted", "remaining"])
    )
    calls = []

    def generate(model, tokenizer, prompt, *args, **kwargs):
        calls.append(prompt)
        if prompt == "interrupted":
            raise KeyboardInterrupt("process interrupted")
        return prompt, []

    monkeypatch.setattr(panel, "generate_with_token_ids", generate)
    with pytest.raises(KeyboardInterrupt):
        panel.run_panel(source, index, backend="cpu")
    output = panel.run_panel(source, index, backend="cpu")
    record = panel.verify_panel_result(output)
    assert calls == ["negative", "interrupted", "remaining"]
    assert record["rows"][0]["completion"] == ""
    assert record["rows"][1]["error"]["type"] == "InterruptedAttempt"
    assert record["rows"][1]["token_ids"] is None
    assert record["rows"][2]["status"] == "COMPLETED"


def test_crash_after_started_marker_never_starts_generator(panel_run, monkeypatch):
    root, index = panel_run
    source = write_panel(root, "marker-crash.json", declaration(["crash-before-call"]))
    publish = panel.publish_immutable

    def crash_after_marker(path, payload):
        result = publish(path, payload)
        if path.name.endswith(".started.json"):
            raise KeyboardInterrupt("crash after fsync")
        return result

    monkeypatch.setattr(panel, "publish_immutable", crash_after_marker)
    monkeypatch.setattr(
        panel,
        "generate_with_token_ids",
        lambda *a, **kw: pytest.fail("started attempt rerolled"),
    )
    with pytest.raises(KeyboardInterrupt):
        panel.run_panel(source, index, backend="cpu")
    monkeypatch.setattr(panel, "publish_immutable", publish)
    output = panel.run_panel(source, index, backend="cpu")
    row = panel.verify_panel_result(output)["rows"][0]
    assert row["status"] == "FAILED"
    assert row["error"]["type"] == "InterruptedAttempt"


def test_concurrent_panel_calls_execute_each_prompt_once(panel_run, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    root, index = panel_run
    source = write_panel(root, "concurrent.json", declaration(["concurrent"]))
    record = panel.verify_evaluation_index(index)
    loaded = panel.load_run(
        record["run_id"], Path(record["run"]).parent, record["checkpoint"], "cpu"
    )
    first_generating = Event()
    second_loaded = Event()
    release = Event()
    loads = []
    calls = []

    def loader(*args, **kwargs):
        loads.append(True)
        if len(loads) == 2:
            second_loaded.set()
        return loaded

    def generate(*args, **kwargs):
        calls.append(True)
        first_generating.set()
        assert release.wait(timeout=10)
        return "concurrent", []

    monkeypatch.setattr(panel, "load_run", loader)
    monkeypatch.setattr(panel, "generate_with_token_ids", generate)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(panel.run_panel, source, index, backend="cpu")
        try:
            assert first_generating.wait(timeout=10)
            second = executor.submit(panel.run_panel, source, index, backend="cpu")
            assert second_loaded.wait(timeout=10)
        finally:
            release.set()
        assert first.result(timeout=10) == second.result(timeout=10)
    assert len(calls) == 1


def test_volatile_native_runtime_observations_do_not_reroll(panel_run, monkeypatch):
    from dataclasses import replace

    root, index = panel_run
    source = write_panel(
        root, "volatile-runtime.json", declaration(["volatile-runtime"])
    )
    original_loader = panel.load_run
    loads = []
    calls = []

    def loader(*args, **kwargs):
        loaded = original_loader(*args, **kwargs)
        loads.append(True)
        runtime = loaded.identity["runtime"] | {
            "measured_at": f"observation-{len(loads)}",
            "device_free_bytes": 1000 * len(loads),
            "system_available_bytes": 2000 * len(loads),
        }
        return replace(loaded, identity=loaded.identity | {"runtime": runtime})

    def generate(*args, **kwargs):
        calls.append(True)
        return "volatile-runtime", []

    monkeypatch.setattr(panel, "load_run", loader)
    monkeypatch.setattr(panel, "generate_with_token_ids", generate)
    first = panel.run_panel(source, index, backend="cpu")
    assert panel.run_panel(source, index, backend="cpu") == first
    assert len(calls) == 1
    assert (
        panel.verify_panel_result(first)["identity"]["runtime"]["measured_at"]
        == "observation-1"
    )


@pytest.mark.parametrize(
    "field",
    ["device_name", "framework_version", "os", "runtime_version", "physical_device_id"],
)
def test_stable_runtime_identity_changes_rejected(panel_run, monkeypatch, field):
    from dataclasses import replace

    root, index = panel_run
    source = write_panel(root, f"stable-runtime-{field}.json", declaration([field]))
    original = panel.load_run

    def loader(*args, **kwargs):
        loaded = original(*args, **kwargs)
        return replace(
            loaded,
            identity=loaded.identity
            | {
                "runtime": loaded.identity["runtime"] | {field: "changed"},
            },
        )

    monkeypatch.setattr(panel, "load_run", loader)
    monkeypatch.setattr(
        panel,
        "generate_with_token_ids",
        lambda *a, **kw: pytest.fail("changed runtime generated"),
    )
    with pytest.raises(ValueError, match="runtime differs"):
        panel.run_panel(source, index, backend="cpu")
