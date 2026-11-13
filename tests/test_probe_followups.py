"""Probe follow-ups: closed-book recall, text-level item groups, reliability
diagrams, lm-eval overrides and fact recall as a comparable metric."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from klab.data.withheld_facts import (
    diagnostic_manifest,
    split_facts,
    training_documents,
)
from klab.lab_records import write_sealed
from klab.probes import compare as compare_mod
from klab.probes import lm_eval_adapter, metrics, runner, scoring
from klab.probes import verdict as verdict_mod
from klab.probes.points import collect_points, metric_points
from klab.probes.suite import BY_ID, parametric_items


class _Words:
    """Whitespace tokenizer; ``offset`` shifts every id (a different tokenizer)."""

    def __init__(self, offset: int = 0) -> None:
        self.offset = offset
        self.vocab: dict[str, int] = {"<eos>": 0}

    def encode(self, text: str, add_special_tokens: bool = False) -> Any:
        ids = [
            self.vocab.setdefault(w, len(self.vocab)) + self.offset
            for w in text.split()
        ]
        return type("Encoding", (), {"ids": ids})()

    def token_to_id(self, token: str) -> int | None:
        return self.vocab.get(token)


class _Loaded:
    def __init__(
        self,
        tokenizer: Any,
        max_seq_len: int = 256,
        *,
        source: str | None = "withheld_facts",
        seed: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        dataset = (
            type(
                "Dataset",
                (),
                {
                    "source": "synthetic" if source == "withheld_facts" else source,
                    "synthetic_task": source if source == "withheld_facts" else None,
                    "synthetic_seed": seed,
                },
            )()
            if source is not None
            else None
        )
        self.config = type(
            "Config",
            (),
            {
                "model": type("Model", (), {"max_seq_len": max_seq_len})(),
                "dataset": dataset,
            },
        )()


@pytest.fixture
def uniform(monkeypatch: pytest.MonkeyPatch) -> None:
    vocab = 5000
    monkeypatch.setattr(
        scoring,
        "log_probs",
        lambda loaded, ids, last=None: np.full(
            (last or len(ids), vocab), -math.log(vocab)
        ),
    )


def _measure(probe: str, loaded: Any) -> dict[str, Any]:
    return runner._measure(BY_ID[probe], loaded, runner.Arm(load=lambda: loaded), {})


# --- Reliability diagram ------------------------------------------------------


def test_reliability_bins_are_the_data_behind_ece() -> None:
    conf = [0.9, 0.9, 0.1, 0.1]
    hit = [True, False, False, False]
    bins = metrics.reliability_bins(conf, hit, bins=10)
    assert [(b["low"], b["high"]) for b in bins] == [(0.0, 0.1), (0.8, 0.9)]
    assert [b["accuracy"] for b in bins] == [0.0, 0.5]
    assert [b["share"] for b in bins] == [0.5, 0.5]
    assert metrics.expected_calibration_error(conf, hit, 10) == pytest.approx(
        sum(b["share"] * abs(b["accuracy"] - b["confidence"]) for b in bins)
    )


def test_calibration_row_keeps_both_reliability_diagrams() -> None:
    spec = BY_ID["calibration"]
    t = {"value": 0.1, "bins": 15, "positions": 4, "reliability": [{"share": 1.0}]}
    b = {**t, "reliability": [{"share": 0.5}, {"share": 0.5}]}
    row = runner._judge(spec, t, b, {"validation": True, "tokenizer": True}, {})
    assert row["details"]["reliability"] == t["reliability"]
    assert row["details"]["baseline_reliability"] == b["reliability"]


# --- Closed-book (parametric) fact recall -------------------------------------


def test_parametric_items_follow_the_withheld_facts_manifest() -> None:
    seed = 42
    manifest = diagnostic_manifest(seed)
    trained = parametric_items("heldout", seed=seed)
    control = parametric_items("heldout", seed=seed, control=True)
    # Verdict items: the manifest's training facts, asked closed-book with its
    # canonical prompt (the fact itself is never in the context).
    assert [f"{i['question']} {i['answer']}." for i in trained] == manifest[
        "training_statements"
    ]
    assert [f"{i['question']} {i['answer']}." for i in trained] == list(
        training_documents(seed)
    )
    assert all(i["answer"] not in i["question"] for i in trained)
    # Control: exactly the never-trained held-out cases.
    assert [(i["question"], i["answer"]) for i in control] == [
        (c["prompt"], c["expected_value"]) for c in manifest["held_out_cases"]
    ]
    assert {i["question"] for i in trained}.isdisjoint(i["question"] for i in control)
    dev = parametric_items("dev", seed=seed)
    assert [i["answer"] for i in dev] == [i["answer"] for i in trained]
    assert {i["question"] for i in dev}.isdisjoint(i["question"] for i in trained)
    assert all(len(i["candidates"]) == 4 for i in trained + control)


def test_parametric_recall_binds_the_runs_nonzero_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run trained on withheld_facts with synthetic_seed=42 is scored on
    seed 42's facts: its trained facts are never counted as the control."""
    seed = 42
    assert split_facts(seed) != split_facts(0)
    trained_text = set(training_documents(seed))

    def knows(loaded: Any, items: list[dict[str, Any]]) -> dict[str, Any]:
        # A model that memorized exactly its training documents.
        hits = [f"{i['prefix']} {i['answer']}." in trained_text for i in items]
        return {
            "credits": [1.0 if hit else 0.0 for hit in hits],
            "picked": [
                i["answer"] if hit else "?" for i, hit in zip(items, hits, strict=True)
            ],
        }

    monkeypatch.setattr(runner, "_ranked", knows)
    monkeypatch.setattr(scoring, "ranking_credit", lambda loaded, items: [])
    out = _measure("parametric_recall", _Loaded(_Words(), seed=seed))
    assert out["value"] == 1.0  # every trained fact recalled
    assert out["control_accuracy"] == 0.0  # never-trained facts stay unknown
    assert out["manifest_seed"] == seed
    assert out["manifest_sha256"] == diagnostic_manifest(seed)["sha256"]
    assert out["manifest_sha256"] != diagnostic_manifest(0)["sha256"]
    # Another seed's facts are other items: never paired with seed 0.
    zero = _measure("parametric_recall", _Loaded(_Words(), seed=0))
    assert zero["item_group"] != out["item_group"]


def test_parametric_recall_is_inapplicable_or_missing_without_provenance(
    tmp_path: Path,
) -> None:
    spec = BY_ID["parametric_recall"]

    def measured(loaded: Any, *, reference: bool = False) -> dict[str, Any]:
        arm = runner.Arm(load=lambda: loaded, reference=reference)
        return runner._measure_safely(spec, loaded, arm, {})

    # Not trained on withheld_facts (or a public reference): does not apply.
    other = measured(_Loaded(_Words(), source="local_text"))
    assert other["status"] == "skipped" and "inapplicable" in other["note"]
    assert "local_text" in other["note"]
    ref = measured(_Loaded(_Words()), reference=True)
    assert ref["status"] == "skipped" and "inapplicable" in ref["note"]
    # No dataset provenance: missing evidence, never a guessed manifest.
    bare = measured(_Loaded(_Words(), source=None))
    assert bare["status"] == "unavailable" and "provenance" in bare["note"]
    # Prepared data that disagrees with the config: missing evidence.
    loaded = _Loaded(_Words(), seed=42)
    loaded.run = tmp_path
    missing = measured(loaded)
    assert (
        missing["status"] == "unavailable" and "data/manifest.json" in missing["note"]
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "manifest.json").write_text(
        json.dumps(
            {
                "cache_identity": {
                    "dataset": {
                        "source": "synthetic",
                        "synthetic_task": "withheld_facts",
                        "synthetic_seed": 7,
                    }
                }
            }
        )
    )
    loaded = _Loaded(_Words(), seed=42)
    loaded.run = tmp_path
    mismatch = measured(loaded)
    assert mismatch["status"] == "unavailable" and "does not match" in mismatch["note"]
    row = runner._judge(spec, mismatch, None, {"validation": True}, {})
    assert row["status"] == "unavailable" and row["value"] is None
    assert verdict_mod.missing_evidence([row])[0]["id"] == "parametric_recall"
    assert not verdict_mod.missing_evidence(
        [runner._judge(spec, other, None, {"validation": True}, {})]
    )


@pytest.mark.usefixtures("uniform")
def test_parametric_recall_of_a_uniform_model_is_chance_with_a_chance_control() -> None:
    out = _measure("parametric_recall", _Loaded(_Words()))
    assert out["value"] == pytest.approx(0.25)
    assert out["control_accuracy"] == pytest.approx(0.25)
    assert out["chance"] == pytest.approx(0.25)
    assert out["manifest_sha256"] == diagnostic_manifest(0)["sha256"]
    assert out["manifest_seed"] == 0
    assert len(out["items"]) == len(out["credits"]) == len(out["picked"]) == 6


# --- Text-level item groups -----------------------------------------------------


@pytest.mark.usefixtures("uniform")
def test_text_probe_item_groups_are_tokenizer_independent() -> None:
    small = _Loaded(_Words(), max_seq_len=64)
    other = _Loaded(_Words(offset=100), max_seq_len=256)  # another tokenizer
    for probe in ("fact_recall", "parametric_recall", "needle"):
        assert (
            _measure(probe, small)["item_group"] == _measure(probe, other)["item_group"]
        )


@pytest.mark.usefixtures("uniform")
def test_needle_items_have_fixed_character_lengths_for_any_tokenizer() -> None:
    lengths = BY_ID["needle"].params["char_lengths"]
    items = runner.needle_items("heldout", lengths)
    assert sorted({len(i["prefix"]) for i in items}) == sorted(lengths)
    assert all(len(i["prefix"]) == i["chars"] for i in items)
    # The code word opens the context; the question closes it.
    assert all(i["prefix"].startswith(f"Code: {i['answer']}.") for i in items)
    assert all(i["prefix"].endswith(" Code:") for i in items)
    # Model token counts are recorded per arm but never enter the item group.
    small = _measure("needle", _Loaded(_Words(), max_seq_len=64))
    other = _measure("needle", _Loaded(_Words(offset=7), max_seq_len=64))
    assert small["chars"] == [i["chars"] for i in items]
    assert small["item_group"] == other["item_group"]
    assert runner.REFERENCE_PROBES >= {"needle", "fact_recall"}


def _side(credits: list[float], picked: list[str], group: str = "g" * 64) -> dict:
    return {
        "value": float(np.mean(credits)),
        "credits": credits,
        "picked": picked,
        "dev_credits": [],
        "fractions": [None] * len(credits),
        "tokens": [None] * len(credits),
        "chance": 0.25,
        "item_group": group,
        "items": [{"prompt": f"q{i}", "answer": "red"} for i in range(len(credits))],
    }


def test_ranking_rows_keep_each_items_pick_and_refuse_other_items() -> None:
    spec = BY_ID["fact_recall"]
    ok = {"validation": True, "tokenizer": False}  # text-level: tokenizers may differ
    t = _side([1.0, 0.0, 1.0], ["red", "blue", "red"])
    b = _side([0.0, 0.0, 1.0], ["gold", "blue", "red"])
    row = runner._judge(spec, t, b, ok, {})
    assert row["status"] != "not_comparable" and row["delta"] == pytest.approx(1 / 3)
    first = row["details"]["items"][0]
    assert first == {
        "prompt": "q0",
        "answer": "red",
        "picked": "red",
        "credit": 1.0,
        "baseline_picked": "gold",
        "baseline_credit": 0.0,
    }
    other = runner._judge(
        spec, t, _side([1.0] * 3, ["red"] * 3, group="h" * 64), ok, {}
    )
    assert other["status"] == "not_comparable" and "different items" in other["note"]


# --- lm-eval overrides ------------------------------------------------------------


def test_lm_eval_overrides_make_a_new_benchmark_never_a_new_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runner.lm_eval_spec() is BY_ID["lm_eval"]
    spec = runner.lm_eval_spec(["piqa", "winogrande"], 200)
    assert spec.params["tasks"] == ["piqa", "winogrande"]
    assert spec.params["limit"] == 200
    assert spec.params["chance"] == {"piqa": 0.5, "winogrande": None}
    with pytest.raises(ValueError):
        runner.lm_eval_spec(["piqa", "piqa"])
    with pytest.raises(ValueError):
        runner.lm_eval_spec(limit=0)
    seen: dict[str, Any] = {}

    def fake(loaded: Any, tasks: list[str], limit: int) -> dict[str, Any]:
        seen.update(tasks=tasks, limit=limit)
        raise lm_eval_adapter.LmEvalUnavailable("not installed")

    monkeypatch.setattr(lm_eval_adapter, "run_lm_eval", fake)
    out = runner._measure_safely(spec, object(), runner.Arm(load=object), {})
    assert seen == {"tasks": ["piqa", "winogrande"], "limit": 200}
    assert out["status"] == "unavailable"


def test_probe_cli_rejects_bad_lm_eval_options_before_any_work(tmp_path: Path) -> None:
    from klab.probes.cli import run_probe

    with pytest.raises(ValueError, match="at least 1"):
        run_probe("missing-run", None, lab_dir=tmp_path, lm_eval_limit=0)
    assert not (tmp_path / "probes").exists()


# --- Fact recall as a comparable metric (points, compare) ------------------------


def _record(lab: Path, probe_id: str, row: dict[str, Any], sha: str) -> None:
    target = {
        "run_id": probe_id,
        "checkpoint_sha256": sha,
        "step": 1,
        "tokens_seen": 10,
        "parameters": 100,
        "active_parameters": 50,
        "eval_group": None,
    }
    record = {
        "format": "klab-probe-v1",
        "probe_id": probe_id,
        "created_at": "2026-10-10T12:00:00+00:00",
        "target": target,
        "baseline": None,
        "probes": [row],
        "verdict": {"status": "info"},
    }
    path = lab / "probes" / probe_id / "probe.json"
    path.parent.mkdir(parents=True)
    write_sealed(path, record)


def test_fact_recall_is_a_point_metric_paired_only_within_one_item_group(
    tmp_path: Path,
) -> None:
    spec = BY_ID["fact_recall"]
    ok = {"validation": True, "tokenizer": True}
    good = runner._judge(spec, _side([1.0, 1.0, 1.0, 0.0], ["red"] * 4), None, ok, {})
    weak = runner._judge(spec, _side([0.0, 1.0, 0.0, 0.0], ["red"] * 4), None, ok, {})
    alien = runner._judge(
        spec, _side([1.0, 0.0, 0.0, 0.0], ["red"] * 4, group="h" * 64), None, ok, {}
    )
    _record(tmp_path, "probe-good", good, "a" * 64)
    _record(tmp_path, "probe-weak", weak, "b" * 64)
    _record(tmp_path, "probe-alien", alien, "c" * 64)
    points = metric_points(collect_points(tmp_path, references=False), "fact_recall")
    assert {p["source"]: p["value"] for p in points} == {
        "probe-good": 0.75,
        "probe-weak": 0.25,
        "probe-alien": 0.25,
    }
    report = compare_mod.compare(
        "probe-good", ["probe-weak", "probe-alien"], lab_dir=tmp_path
    )
    pairs = {
        c["other"]: next(p for p in c["pairs"] if p["metric"] == "fact_recall")
        for c in report["comparisons"]
    }
    weak_pair = pairs["probe-weak@1"]
    assert weak_pair["delta"] == pytest.approx(0.5)
    assert weak_pair["se"] is not None  # paired over the same items
    assert pairs["probe-alien@1"]["status"] == "not_comparable"
    assert "fact recall" in compare_mod.render(report)
