"""Retained mixture stages refuse a missing or invalid historical proof."""

import json

import pytest

from klab.corpus.admission_inspection_cache import capture
from klab.corpus.completed_preparation import _producer
from klab.corpus.retained_mixture import _phase
from klab.corpus.split_runs import SplitStages, write_rows


def test_retained_stage_refuses_missing_or_bad_proof(tmp_path, monkeypatch):
    root = tmp_path / "work"
    root.mkdir(mode=0o700)
    config = tmp_path / "config"
    config.mkdir(mode=0o700)
    key_dir = config / "klab"
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
    result, _ = _phase(
        root,
        stages.root,
        "build-frozen-family-bindings",
        expected_producer=request["producer"],
    )
    assert result == payload
    proof_path.unlink()  # Tiny fixture only: simulate an optional receipt not minted.
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
        )
