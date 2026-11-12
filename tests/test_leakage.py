"""Held-out leakage: probe items planted in training data stop a try's verdict."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from lab_fixtures import write_smoke_inputs

from klab.lab_mode import run_try
from klab.probes.contamination import find_windows
from klab.probes.suite import fact_items


def _write_texts(path: Path, texts: list[str]) -> None:
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))


def test_try_flags_a_planted_heldout_item_and_keeps_the_clean_arm_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    baseline = write_smoke_inputs(tmp_path)
    stories = [f"story {n}: the fox jumps over dog {n % 7} twice" for n in range(240)]
    item = fact_items("heldout")[0]
    planted = f"Once more. {item['question']} {item['answer']}. The end."
    _write_texts(tmp_path / "clean.jsonl", stories[:200])
    _write_texts(tmp_path / "leaky.jsonl", [planted, *stories[:199]])
    _write_texts(tmp_path / "validation.jsonl", stories[200:])
    config = yaml.safe_load(baseline.read_text())
    config["dataset"].update(
        source="local_text",
        train_path=str(tmp_path / "clean.jsonl"),
        validation_path=str(tmp_path / "validation.jsonl"),
        license="CC0-1.0",
        train_max_documents=200,
        validation_max_documents=40,
    )
    config["dataset"].pop("synthetic_seed")
    baseline.write_text(yaml.safe_dump(config, sort_keys=False))
    delta = tmp_path / "leaky.yaml"
    delta.write_text(
        yaml.safe_dump({"set": {"dataset.train_path": str(tmp_path / "leaky.jsonl")}})
    )

    record, _ = run_try(delta, baseline, work_dir=tmp_path / "work")

    assert record["status"] == "completed", record.get("failure")
    assert record["comparison"]["comparable"] is True
    leakage = record["leakage"]
    assert leakage["status"] == "CONTAMINATED"
    clean, leaky = leakage["arms"]["baseline"], leakage["arms"]["candidate"]
    assert clean["status"] == "CLEAN"
    assert clean["contaminated"] == {"heldout": 0, "final": 0}
    assert leaky["status"] == "CONTAMINATED"
    assert leaky["contaminated"] == {"heldout": 1, "final": 0}
    assert leaky["items"] == ["fact_recall/0"]
    assert clean["checked"] == leaky["checked"]
    verdict = record["probe"]["verdict"]
    assert verdict["action"] == "rerun"
    assert [gap["id"] for gap in verdict["missing"]] == ["leakage"]


def test_scan_finds_an_exact_sequence_and_not_a_one_token_near_miss() -> None:
    rng = np.random.default_rng(0)
    train = rng.integers(0, 500, 50_000, dtype=np.int32)
    planted = tuple(int(token) for token in rng.integers(500, 600, 13))
    train[20_000:20_013] = planted
    near = (*planted[:6], planted[6] + 1, *planted[7:])
    assert find_windows(train, {planted, near}) == {planted}
