"""Probe battery: metrics, verdicts, fast-fail ordering, schema, CLI and `try`."""

from __future__ import annotations

import json
import math
import os
import signal
import sys
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import yaml
from lab_fixtures import write_smoke_inputs

from klab.lab_context import LabContext
from klab.lab_records import eval_group
from klab.probes import lm_eval_adapter, metrics, runner, scoring
from klab.probes.render import render
from klab.probes.suite import (
    BY_ID,
    PROBES,
    SUITE_VERSION,
    TIERS,
    fact_items,
    needle_split,
    ordered,
    prompts,
    suite_identity,
)
from klab.probes.verdict import (
    ACTIONS,
    MISSING_STATUSES,
    decide,
    judge,
    overfit_guard,
)

# Any change to probe declarations or items must bump SUITE_VERSION and this pin,
# so historical results are never silently compared across different suites.
PINNED_SUITE = {
    1: "68a1bba35e4dca50ba4cf45d49fd466c0a7c764ef2a1f02227753f930c4545f3",
    # v2: fractional tie credit, paired/clustered uncertainty declarations.
    2: "ec0544e78b2303deec540f317c237f28bfb4e61cb71bb372f864906cfb1478a0",
    # v3: closed-book fact recall from the run's own withheld-facts manifest
    # (bound to dataset.synthetic_seed; unreleased before PR #63 merged).
    3: "bc974a71841375e8788b612186f67a69d7d26598cd1325499b64ffbb5b484a50",
    # v4: held-back final split, text-level needle (fixed character lengths),
    # within-run eval-noise wording on held-out loss.
    4: "b24f01ec34293994ad0df3a3e45f1e602c54cef21a769f17c24ddcf084c58289",
    # v5: battery and scoring protocol identities renamed with the project
    # (sparselab-* -> klab-*); probe declarations and items unchanged.
    5: "b7f7b01dc04b9531f5680b398afbf4ffb0e7333e267fe79540e73eda84c20ec9",
}

RESULT_KEYS = {
    "format",
    "created_at",
    "suite",
    "tier",
    "tiers_run",
    "stop",
    "target",
    "baseline",
    "comparable",
    "protocol",
    "probes",
    "guard",
    "verdict",
    "seconds",
}
ROW_KEYS = {
    "id",
    "title",
    "tier",
    "cost",
    "metric",
    "higher_is_better",
    "hard",
    "thresholds",
    "uncertainty",
    "suggests",
    "explains",
    "reference",
    "status",
    "value",
    "baseline_value",
    "delta",
    "regression",
    "delta_se",
    "within_noise",
    "improved",
    "note",
    "details",
    "seconds",
}
IDENTITY_KEYS = {
    "run_id",
    "run_dir",
    "checkpoint",
    "checkpoint_sha256",
    "step",
    "tokens_seen",
    "parameters",
    "parameter_bytes",
    "active_parameters",
    "active_parameter_bytes",
    "tokenizer_sha256",
    "validation_sha256",
    "max_seq_len",
    "eval_group",
}
VERDICT_KEYS = {"status", "action", "next_tier", "reasons", "missing", "suggestion"}
STOP_KEYS = {"stopped", "kind", "at", "reason"}


# --- Metrics (known answers) -------------------------------------------------


def test_distinct_n_and_seq_rep_known_answers() -> None:
    assert metrics.distinct_n([[1, 2, 3, 4]], 1) == 1.0
    assert metrics.distinct_n([[1, 1, 1, 1]], 1) == 0.25
    # Pooled across sequences: bigrams (1,2),(2,1),(1,2) + (1,2) -> 2 unique of 4.
    assert metrics.distinct_n([[1, 2, 1, 2], [1, 2]], 2) == 0.5
    assert metrics.distinct_n([[1]], 2) is None
    assert metrics.seq_rep_n([1, 2, 3, 4, 5], 4) == 0.0
    # A pure loop of period 1: 7 four-grams, one unique.
    assert metrics.seq_rep_n([7] * 10, 4) == pytest.approx(1 - 1 / 7)
    assert metrics.seq_rep_n([1, 2, 3], 4) is None
    assert metrics.mean_seq_rep_n([[1, 2, 3, 4, 5], [7] * 10, [1]], 4) == (
        pytest.approx((0 + 6 / 7) / 2)
    )
    with pytest.raises(ValueError):
        metrics.ngrams([1, 2], 0)


def test_ece_known_answers() -> None:
    assert metrics.expected_calibration_error([1.0, 1.0], [True, True]) == 0.0
    # Two bins of two: confident and right half the time.
    value = metrics.expected_calibration_error(
        [0.9, 0.9, 0.1, 0.1], [True, False, False, False], bins=10
    )
    assert value == pytest.approx(0.5 * abs(0.5 - 0.9) + 0.5 * abs(0.0 - 0.1))
    # Bin edges are right-closed: 0.2 joins (0.1, 0.2] with 10 bins.
    edge = metrics.expected_calibration_error([0.2, 0.15], [False, False], bins=10)
    assert edge == pytest.approx(0.175)
    with pytest.raises(ValueError):
        metrics.expected_calibration_error([1.5], [True])
    with pytest.raises(ValueError):
        metrics.expected_calibration_error([], [])


def test_divergences_and_agreement_known_answers() -> None:
    p = np.log(np.array([[0.5, 0.5], [0.9, 0.1]]))
    q = np.log(np.array([[0.5, 0.5], [0.1, 0.9]]))
    kl = metrics.kl_divergence(p, q)
    assert kl[0] == pytest.approx(0.0)
    assert kl[1] == pytest.approx(0.9 * math.log(9) + 0.1 * math.log(1 / 9))
    js = metrics.js_divergence(p, q)
    assert js[0] == pytest.approx(0.0)
    assert 0 < js[1] < math.log(2)
    disjoint = metrics.js_divergence(
        np.log(np.array([1.0, 1e-300])), np.log(np.array([1e-300, 1.0]))
    )
    assert disjoint == pytest.approx(math.log(2))
    assert metrics.top1_agreement(p[1:], q[1:]) == 0.0
    assert metrics.top1_agreement(p[1:], p[1:]) == 1.0
    logits = np.array([[1.0, 2.0, 3.0]])
    assert np.exp(metrics.log_softmax(logits)).sum() == pytest.approx(1.0)


def test_standard_errors_known_answers() -> None:
    mean, se = metrics.paired_mean_and_se([1.0, 3.0])
    assert mean == 2.0 and se == pytest.approx(1.0)
    assert metrics.paired_mean_and_se([5.0]) == (5.0, None)
    # Identical windows on both sides: the clustered SE of the difference is 0.
    assert metrics.ratio_difference_se([1, 2], [1, 1], [1, 2], [1, 1]) == 0.0
    assert metrics.ratio_difference_se([1.0], [1], [1.0], [1]) is None
    assert metrics.ratio_difference_se([1, 2], [1, 1], [1], [1]) is None


def test_ratio_se_is_the_clustered_delta_method_and_token_weighted() -> None:
    rng = np.random.default_rng(0)
    counts_a = rng.integers(5, 40, size=30)
    counts_b = counts_a.copy()
    sums_a = counts_a * rng.normal(4.0, 0.3, size=30)
    sums_b = counts_b * rng.normal(4.1, 0.3, size=30)
    se = metrics.ratio_difference_se(sums_a, counts_a, sums_b, counts_b)
    la, lb = sums_a.sum() / counts_a.sum(), sums_b.sum() / counts_b.sum()
    z = (sums_a - la * counts_a) / counts_a.sum() - (sums_b - lb * counts_b) / (
        counts_b.sum()
    )
    assert se == pytest.approx(math.sqrt(30 / 29 * float((z**2).sum())))
    # Windows carry weight by tokens: a window of 1 token barely moves the SE.
    tiny = metrics.ratio_difference_se(
        [*sums_a, 50.0], [*counts_a, 1], [*sums_b, 0.0], [*counts_b, 1]
    )
    assert tiny is not None and tiny < 2 * se


def test_accuracy_delta_se_is_paired_on_items() -> None:
    # Same items right and wrong on both arms: no difference, zero SE.
    same = runner._paired([1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0])
    assert same == 0.0
    flipped = runner._paired([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    assert flipped == pytest.approx(metrics.paired_mean_and_se([1, 0, 0, 0])[1])
    assert runner._paired([1.0], [0.0]) is None


# --- Recall/needle credit ----------------------------------------------------


def test_tie_credit_is_fractional_and_answer_independent() -> None:
    assert scoring.tie_credit({"a": -1.0, "b": -2.0}, "a") == 1.0
    assert scoring.tie_credit({"a": -1.0, "b": -2.0}, "b") == 0.0
    tied = {"a": -1.0, "b": -1.0, "c": -1.0, "d": -3.0}
    assert scoring.tie_credit(tied, "a") == pytest.approx(1 / 3)
    assert scoring.tie_credit(tied, "d") == 0.0
    # The answer's position in the candidate list never matters.
    assert scoring.tie_credit({"x": 0.0, "y": 0.0}, "x") == scoring.tie_credit(
        {"y": 0.0, "x": 0.0}, "x"
    )


class _TinyTokenizer:
    """Whitespace tokenizer for ranking tests (ids from a growing vocabulary)."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<eos>": 0}

    def encode(self, text: str, add_special_tokens: bool = False) -> Any:
        ids = [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]
        return type("Encoding", (), {"ids": ids})()

    def token_to_id(self, token: str) -> int | None:
        return self.vocab.get(token)


class _Loaded:
    def __init__(self, tokenizer: Any, max_seq_len: int = 64) -> None:
        self.tokenizer = tokenizer
        self.config = type(
            "Config", (), {"model": type("Model", (), {"max_seq_len": max_seq_len})()}
        )()


@pytest.mark.parametrize("probe", ["fact_recall", "needle"])
def test_uniform_scores_give_chance_accuracy(
    probe: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: ties used to break toward the answer (uniform model = 100%)."""
    vocab = 5000
    monkeypatch.setattr(
        scoring,
        "log_probs",
        lambda loaded, ids: np.full((len(ids), vocab), -math.log(vocab)),
    )
    loaded = _Loaded(_TinyTokenizer(), max_seq_len=256)
    if probe == "fact_recall":
        items = runner.recall_items("heldout")
    else:
        items = runner.needle_items("heldout", [192])
    credits = scoring.ranking_credit(loaded, items)
    chance = 1 / len(items[0]["candidates"])
    assert credits == pytest.approx([chance] * len(items))


# --- Suite identity and splits ----------------------------------------------


def test_suite_identity_is_pinned_and_versioned() -> None:
    identity = suite_identity()
    assert identity["name"] == "klab-probe-battery"
    assert identity["version"] == SUITE_VERSION
    assert identity["sha256"] == PINNED_SUITE[SUITE_VERSION], (
        "probe declarations or items changed: bump SUITE_VERSION and the pin"
    )
    assert identity["verdict_split"] == "heldout"
    assert set(identity["splits"]) == {"dev", "heldout", "final"}


def test_heldout_items_are_disjoint_from_dev_items() -> None:
    dev, held = fact_items("dev"), fact_items("heldout")
    assert {i["question"] for i in dev}.isdisjoint({i["question"] for i in held})
    assert {i["context"] for i in dev}.isdisjoint({i["context"] for i in held})
    assert {i["answer"] for i in dev}.isdisjoint({i["answer"] for i in held})
    assert set(needle_split("dev")["needles"]).isdisjoint(
        needle_split("heldout")["needles"]
    )
    assert set(prompts("dev")).isdisjoint(prompts("heldout"))
    for item in dev + held:
        assert item["answer"] in item["candidates"]
    for item in held:
        # Reworded: held-out questions never repeat the context's phrasing.
        assert item["context"].split(" is ")[0] not in item["question"]


def test_every_probe_declares_tier_cost_thresholds_and_hint() -> None:
    assert {p.tier for p in PROBES} == set(TIERS)
    for spec in PROBES:
        assert spec.cost >= 1 and spec.suggests and spec.explains and spec.reference
        assert spec.mode in {"delta_rel", "delta_abs", "value_min"}
        assert spec.fail is None or spec.fail >= spec.warn or spec.mode == "value_min"


def test_ordering_is_cheapest_first_and_tiered() -> None:
    fast = [s.id for s in ordered("fast")]
    assert fast[0] == "heldout_loss" and {BY_ID[i].tier for i in fast} == {"fast"}
    full = ordered("full")
    keys = [(TIERS.index(s.tier), s.cost) for s in full]
    assert keys == sorted(keys)
    assert full[-1].id == "lm_eval"
    with pytest.raises(ValueError):
        ordered("huge")


# --- Verdict logic ------------------------------------------------------------


def test_judge_relative_loss_thresholds_and_noise() -> None:
    spec = BY_ID["heldout_loss"]
    assert judge(spec, 4.0, 4.0)["status"] == "pass"
    worse = judge(spec, 4.06, 4.0)
    assert worse["regression"] == pytest.approx(0.015) and worse["status"] == "warn"
    assert judge(spec, 4.2, 4.0)["status"] == "fail"
    noisy = judge(spec, 4.2, 4.0, se=0.15)
    assert noisy["status"] == "pass" and noisy["within_noise"] is True
    better = judge(spec, 3.8, 4.0, se=0.01)
    assert better["improved"] is True and better["status"] == "pass"
    assert judge(spec, float("nan"), 4.0)["status"] == "fail"
    assert judge(spec, 4.0, None)["status"] == "info"


def test_judge_higher_is_better_and_value_min() -> None:
    recall = BY_ID["fact_recall"]
    assert judge(recall, 0.5, 0.7)["status"] == "fail"
    assert judge(recall, 0.6, 0.7)["status"] == "warn"
    assert judge(recall, 0.9, 0.7)["improved"] is True
    agreement = BY_ID["token_agreement"]
    assert judge(agreement, 0.9, None)["status"] == "pass"
    assert judge(agreement, 0.2, None)["status"] == "warn"
    loops = judge(BY_ID["repetition"], 0.99, None)
    assert loops["status"] == "warn"


def _result_row(probe_id: str, status: str, **extra: Any) -> dict[str, Any]:
    return runner._row(BY_ID[probe_id], status=status, **extra)


def test_decide_actions_cover_the_documented_table() -> None:
    def verdict(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        options = {
            "has_baseline": True,
            "tiers_run": ["fast"],
            "requested_tier": "fast",
            "guard": None,
            "specs": BY_ID,
            **kw,
        }
        out = decide(rows, **options)
        assert set(out) == VERDICT_KEYS and out["action"] in ACTIONS
        return out

    improved = _result_row("heldout_loss", "pass", improved=True)
    assert verdict([improved])["action"] == "escalate"
    assert verdict([improved])["next_tier"] == "standard"
    assert (
        verdict([improved], tiers_run=["fast", "standard", "full"])["action"]
        == "longer_run"
    )
    flat = _result_row("heldout_loss", "pass")
    assert verdict([flat])["action"] == "tweak"
    hard = verdict([_result_row("heldout_loss", "fail")])
    assert hard["action"] == "abandon" and hard["status"] == "fail"
    soft = verdict([improved, _result_row("calibration", "fail")])
    assert soft["action"] == "tweak" and soft["status"] == "fail"
    warned = verdict([improved, _result_row("calibration", "warn")])
    assert warned["status"] == "warn" and warned["action"] == "escalate"
    guarded = verdict([improved], guard={"overfit_suspected": True})
    assert guarded["action"] == "tweak" and "held-out" in guarded["suggestion"]
    assert verdict([improved], has_baseline=False)["action"] == "compare"
    assert hard["missing"] == [] and verdict([improved])["missing"] == []


@pytest.mark.parametrize("has_baseline", [True, False])
def test_missing_evidence_is_never_success(has_baseline: bool) -> None:
    """An unavailable tier or a probe error never reads as PASS or escalation."""
    # Every MISSING_STATUSES member takes the same branch; this one guards
    # NOT_COMPARABLE evidence from ever being promoted.
    status = "not_comparable"
    assert status in MISSING_STATUSES
    improved = _result_row("heldout_loss", "pass", improved=True)
    rows = [improved, _result_row("lm_eval", status, note="lm-eval not installed")]
    for tiers in (["fast"], ["fast", "standard", "full"]):
        out = decide(
            rows,
            has_baseline=has_baseline,
            tiers_run=tiers,
            requested_tier="full",
            guard=None,
            specs=BY_ID,
        )
        assert out["status"] == "incomplete" and out["action"] == "rerun"
        assert out["action"] not in {"escalate", "longer_run"}
        assert out["next_tier"] is None
        assert out["missing"] == [
            {"id": "lm_eval", "status": status, "note": "lm-eval not installed"}
        ]
        assert "Every tier holds up" not in out["suggestion"]
        assert "lm_eval" in out["suggestion"]
        baseline_suggestion = (
            "Not enough evidence for a verdict. Fix and re-run: "
            f"lm_eval ({status}: lm-eval not installed)."
        )
        if has_baseline:
            assert out["suggestion"] == baseline_suggestion
        else:
            assert out["suggestion"].startswith(baseline_suggestion)
            assert "Add --vs BASELINE for pass/warn/fail" in out["suggestion"]
    # A real failure still decides the verdict.
    failed = decide(
        [*rows, _result_row("calibration", "fail")],
        has_baseline=True,
        tiers_run=["fast"],
        requested_tier="fast",
        guard=None,
        specs=BY_ID,
    )
    assert failed["action"] == "tweak" and failed["missing"]


@pytest.mark.parametrize("has_baseline", [True, False])
def test_stopped_battery_names_the_unrun_probes(has_baseline: bool) -> None:
    kind = "resources"  # every INCOMPLETE_STOPS kind takes the same branch
    improved = _result_row("heldout_loss", "pass", improved=True)
    out = decide(
        [improved, _result_row("calibration", "skipped")],
        has_baseline=has_baseline,
        tiers_run=["fast"],
        requested_tier="fast",
        guard=None,
        specs=BY_ID,
        stop={"kind": kind, "reason": "synthetic stop"},
    )
    assert out["status"] == "incomplete" and out["action"] == "rerun"
    assert out["next_tier"] is None
    assert "calibration" in out["suggestion"]
    assert ("--vs BASELINE" in out["suggestion"]) is (not has_baseline)
    assert out["missing"] == [
        {"id": "calibration", "status": f"stopped:{kind}", "note": "synthetic stop"}
    ]


@pytest.mark.parametrize("has_baseline", [True, False])
@pytest.mark.parametrize("missing", [True, False])
def test_missing_evidence_preserves_failure_precedence(
    has_baseline: bool, missing: bool
) -> None:
    options = {
        "has_baseline": has_baseline,
        "tiers_run": ["fast"],
        "requested_tier": "fast",
        "guard": None,
        "specs": BY_ID,
    }
    gaps = [_result_row("lm_eval", "unavailable")] if missing else []
    numerical = decide(
        [
            _result_row("heldout_loss", "fail", details={"numerical_failure": "nan"}),
            *gaps,
        ],
        **options,
    )
    assert numerical["status"] == "fail" and numerical["action"] == "abandon"
    assert bool(numerical["missing"]) == missing
    if has_baseline:
        hard = decide([_result_row("repetition", "fail"), *gaps], **options)
        assert hard["status"] == "fail" and hard["action"] == "abandon"
        assert bool(hard["missing"]) == missing
    soft = decide([_result_row("calibration", "fail"), *gaps], **options)
    expected = (
        ("fail", "tweak")
        if has_baseline
        else ("incomplete", "rerun")
        if missing
        else ("info", "compare")
    )
    assert (soft["status"], soft["action"]) == expected


def test_overfit_guard_flags_dev_only_gains_beyond_noise() -> None:
    held = {"fact_recall": {"value": 0.5, "baseline_value": 0.5}}
    big = overfit_guard({"fact_recall": {"value": 1.0, "baseline_value": 0.5}}, held)
    assert big["overfit_suspected"] is True
    shared = overfit_guard(
        {"fact_recall": {"value": 1.0, "baseline_value": 0.5}},
        {"fact_recall": {"value": 0.9, "baseline_value": 0.5}},
    )
    assert shared["overfit_suspected"] is False
    # One flipped item out of four is noise (paired SE 0.25), not overfitting.
    tiny = overfit_guard(
        {"fact_recall": {"value": 0.5, "baseline_value": 0.25, "se": 0.25}}, held
    )
    assert tiny["overfit_suspected"] is False
    assert tiny["verdict_split"] == "heldout"


# --- Battery control flow (scripted probes, fake arms, no model) ---------------


class _FakeModel:
    training = False

    def eval(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.training = mode


class _FakeRun:
    """A loaded arm; ``alive`` counts how many are in memory at once."""

    alive: ClassVar[list[str]] = []

    def __init__(self, name: str) -> None:
        assert not _FakeRun.alive, f"{name} loaded while {_FakeRun.alive} alive"
        _FakeRun.alive.append(name)
        self.name = name
        self.model = _FakeModel()
        self.engine = None
        weakref.finalize(self, _FakeRun.alive.remove, name)


class _Script:
    def __init__(self) -> None:
        self.status: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []
        self.loads: list[str] = []
        self.hooks: dict[str, Callable[[], None]] = {}

    def arm(self, name: str) -> runner.Arm:
        def load() -> _FakeRun:
            self.loads.append(name)
            return _FakeRun(name)

        return runner.Arm(load=load)

    def candidate_calls(self) -> list[str]:
        return [probe for arm, probe in self.calls if arm == "cand"]


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> _Script:
    script = _Script()
    _FakeRun.alive.clear()

    def describe(loaded: Any, arm: runner.Arm) -> None:
        arm.cache.setdefault(
            "identity",
            {key: None for key in IDENTITY_KEYS - {"eval_group"}}
            | {"run_id": loaded.name, "checkpoint_sha256": loaded.name},
        )
        arm.cache.setdefault("validation_identity", {"v": 1})
        arm.cache.setdefault("tokenizer_digest", "t")

    def measure(spec, loaded, arm, protocol):
        script.calls.append((loaded.name, spec.id))
        hook = script.hooks.get(f"{loaded.name}:{spec.id}")
        if hook is not None:
            hook()
        return {"value": 1.0}

    def fake_judge(spec, t, b, comparable, dev):
        for name, side in (("candidate", t), ("baseline", b)):
            if side is not None and "status" in side:
                note = f"{name}: {side['note']}"
                return runner._row(spec, status=side["status"], note=note)
        status = script.status.get(spec.id, "pass")
        return runner._row(spec, status=status, value=1.0, baseline_value=1.0)

    monkeypatch.setattr(runner, "_describe", describe)
    monkeypatch.setattr(runner, "_measure", measure)
    monkeypatch.setattr(runner, "_judge", fake_judge)
    return script


PROTOCOL = {"seq_len": 32, "batch_size": 4, "max_batches": 1}


def _battery(script: _Script, **kw: Any) -> dict[str, Any]:
    options = {"tier": "fast", "protocol": PROTOCOL, **kw}
    return runner.run_battery(script.arm("cand"), script.arm("base"), **options)


def test_battery_runs_cheapest_first_one_arm_at_a_time(scripted: _Script) -> None:
    result = _battery(scripted, tier="standard")
    assert scripted.candidate_calls() == [s.id for s in ordered("standard")]
    # Per tier: the baseline is loaded, measured and released, then the
    # candidate. _FakeRun asserts no two arms are ever alive together.
    assert scripted.loads == ["base", "cand", "base", "cand"]
    assert result["tiers_run"] == ["fast", "standard"]
    assert result["stop"] == {
        "stopped": False,
        "kind": None,
        "at": None,
        "reason": None,
    }
    assert set(result) == RESULT_KEYS
    assert set(result["stop"]) == STOP_KEYS
    assert result["target"]["eval_group"] == result["baseline"]["eval_group"]


def test_hard_fail_stops_the_battery(scripted: _Script) -> None:
    scripted.status["heldout_loss"] = "fail"
    result = _battery(scripted, tier="full")
    assert scripted.candidate_calls() == ["heldout_loss"]
    assert result["stop"] == {
        "stopped": True,
        "kind": "fast_fail",
        "at": "heldout_loss",
        "reason": "fast-fail: hard failure in heldout_loss",
    }
    assert all(r["status"] == "skipped" for r in result["probes"][1:])
    assert result["verdict"]["action"] == "abandon"


def test_soft_fail_finishes_the_tier_but_does_not_escalate(scripted: _Script) -> None:
    scripted.status["calibration"] = "fail"
    result = _battery(scripted, tier="standard")
    assert scripted.candidate_calls() == [s.id for s in ordered("fast")]
    assert result["tiers_run"] == ["fast"]
    assert result["stop"]["kind"] == "not_promising"
    assert "not escalating" in result["stop"]["reason"]
    scripted.calls.clear()
    off = _battery(scripted, tier="standard", fast_fail=False)
    assert off["tiers_run"] == ["fast", "standard"]


def test_probe_exception_is_missing_evidence_not_a_crash_or_pass(
    scripted: _Script,
) -> None:
    def boom() -> None:
        raise RuntimeError("kaput")

    scripted.hooks["cand:calibration"] = boom
    states: list[str] = []
    result = _battery(
        scripted, tier="standard", progress=lambda s: states.append(s["state"])
    )
    row = next(r for r in result["probes"] if r["id"] == "calibration")
    assert row["status"] == "error" and row["note"] == "candidate: RuntimeError: kaput"
    # Missing evidence: no escalation to the next tier, and no pass.
    assert result["tiers_run"] == ["fast"]
    assert result["stop"]["kind"] == "missing_evidence"
    verdict = result["verdict"]
    assert verdict["status"] == "incomplete" and verdict["action"] == "rerun"
    assert verdict["missing"][0]["id"] == "calibration"
    assert states[-1] == "done" and "running" in states


def test_unavailable_optional_tier_is_incomplete(scripted: _Script) -> None:
    def missing() -> None:
        raise runner.ProbeUnsupported("lm-eval is not installed")

    scripted.hooks["base:lm_eval"] = missing
    scripted.status["heldout_loss"] = "pass"
    result = _battery(scripted, tier="full")
    row = next(r for r in result["probes"] if r["id"] == "lm_eval")
    assert row["status"] == "unavailable"
    assert row["note"] == "baseline: lm-eval is not installed"
    assert result["verdict"]["status"] == "incomplete"
    assert result["verdict"]["action"] == "rerun"


@pytest.mark.parametrize("failure", ["error", "interrupted", None])
def test_no_baseline_battery_guidance(scripted: _Script, failure: str | None) -> None:
    def stop() -> None:
        if failure == "unavailable":
            raise runner.ProbeUnsupported("synthetic missing dependency")
        if failure == "error":
            raise RuntimeError("synthetic probe error")
        raise KeyboardInterrupt

    if failure:
        scripted.hooks["cand:calibration"] = stop
    result = runner.run_battery(
        scripted.arm("cand"), None, tier="fast", protocol=PROTOCOL
    )
    verdict = result["verdict"]
    assert result["baseline"] is None
    if failure:
        assert verdict["status"] == "incomplete" and verdict["action"] == "rerun"
        assert "calibration" in verdict["suggestion"]
        assert any("calibration" in gap["id"] for gap in verdict["missing"])
    else:
        assert verdict["status"] == "info" and verdict["action"] == "compare"
        assert verdict["missing"] == []
    assert verdict["next_tier"] is None


def test_cancel_sentinel_stops_probing_at_the_next_safe_point(
    scripted: _Script, tmp_path: Path
) -> None:
    context = LabContext(tmp_path / "CANCEL")
    scripted.hooks["cand:heldout_loss"] = lambda: context.cancel_path.touch()
    states: list[str] = []
    result = _battery(
        scripted,
        tier="standard",
        context=context,
        progress=lambda s: states.append(s["state"]),
    )
    assert scripted.candidate_calls() == ["heldout_loss"]
    assert result["stop"]["kind"] == "cancelled"
    assert result["stop"]["at"] == "heldout_loss"
    assert result["verdict"]["action"] == "rerun"
    assert result["probes"][0]["status"] == "pass"  # completed work is kept
    assert all(r["status"] == "skipped" for r in result["probes"][1:])
    assert states[-1] == "stopped"


def test_cancel_before_probing_loads_nothing(scripted: _Script, tmp_path: Path) -> None:
    context = LabContext(tmp_path / "CANCEL")
    context.cancel_path.touch()
    result = _battery(scripted, context=context)
    assert scripted.loads == [] and result["stop"]["kind"] == "cancelled"
    assert result["target"] is None and result["verdict"]["action"] == "rerun"


@pytest.mark.parametrize("error", [MemoryError, "torch"])
def test_out_of_memory_stops_further_probe_work(scripted: _Script, error: Any) -> None:
    import torch

    def oom() -> None:
        if error == "torch":
            raise torch.OutOfMemoryError("CUDA out of memory")
        raise MemoryError("host")

    scripted.hooks["base:calibration"] = oom
    result = _battery(scripted, tier="full")
    assert scripted.calls == [("base", "heldout_loss"), ("base", "calibration")]
    assert scripted.loads == ["base"]
    assert result["stop"]["kind"] == "oom" and result["stop"]["at"] == "calibration"
    assert not any(r["status"] == "error" for r in result["probes"])
    assert result["verdict"]["status"] == "incomplete"
    assert _FakeRun.alive == []  # the arm was released after the OOM


def test_resource_envelope_violation_stops_probing(
    scripted: _Script, tmp_path: Path
) -> None:
    from klab.resource_envelope import ResourceEnvelope

    envelope = ResourceEnvelope(
        resource_envelope_version=1, max_rss_bytes=1
    )  # any live process exceeds this
    context = LabContext(tmp_path / "CANCEL", resource_envelope=envelope)
    result = _battery(scripted, context=context)
    assert scripted.loads == []
    assert result["stop"]["kind"] == "resources"
    assert result["verdict"]["action"] == "rerun"


def test_validation_is_computed_only_on_a_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from klab import lab_mode

    forwards: list[int] = []

    class _Run:
        def evaluate(self, observer: Any = None) -> dict[str, Any]:
            forwards.append(1)
            return {"loss": 4.0, "valid_targets": 8}

    loaded = _Loaded(_TinyTokenizer(), max_seq_len=64)
    monkeypatch.setattr(lab_mode, "_with_eval_protocol", lambda loaded, p: _Run())
    arm = runner.Arm(load=lambda: loaded)
    first = runner.validation(loaded, arm, PROTOCOL)
    again = runner.validation(loaded, arm, PROTOCOL)
    assert forwards == [1] and again is first
    runner.validation(loaded, arm, {**PROTOCOL, "max_batches": 2})
    assert forwards == [1, 1]  # a different protocol is a miss
    with pytest.raises(runner._NotComparable):
        runner.validation(loaded, arm, {**PROTOCOL, "seq_len": 65})


# --- Rendering ---------------------------------------------------------------


def _fake_result(**verdict: Any) -> dict[str, Any]:
    loss = runner._row(BY_ID["heldout_loss"])
    loss.update(
        judge(BY_ID["heldout_loss"], 3.9, 4.0, se=0.01),
        details={"perplexity": math.exp(3.9)},
    )
    calibration = runner._row(BY_ID["calibration"])
    calibration.update(judge(BY_ID["calibration"], 0.2, 0.1))
    skipped = runner._row(BY_ID["needle"], note="skipped: fast-fail")
    who = {key: None for key in IDENTITY_KEYS} | {
        "run_id": "cand",
        "parameters": 43_200,
        "tokens_seen": 1536,
    }
    return {
        "format": runner.RESULT_FORMAT,
        "suite": suite_identity(),
        "tier": "standard",
        "tiers_run": ["fast"],
        "stop": {"stopped": False, "kind": None, "at": None, "reason": None},
        "target": who,
        "baseline": who | {"run_id": "base"},
        "probes": [loss, calibration, skipped],
        "guard": overfit_guard({}, {}),
        "verdict": {
            "status": "fail",
            "action": "tweak",
            "next_tier": None,
            "reasons": [],
            "missing": [],
            "suggestion": "Try again.",
            **verdict,
        },
        "seconds": 0.5,
    }


def test_render_names_missing_evidence() -> None:
    gap = {"id": "lm_eval", "status": "unavailable", "note": "lm-eval not installed"}
    result = _fake_result(status="incomplete", action="rerun", missing=[gap])
    text = render(result, color=False)
    assert "next → RERUN (missing evidence)" in text
    assert "missing lm_eval (unavailable): lm-eval not installed" in text


# --- Scoring and lm-eval boundaries (real tokenizer, fake model) -------------


def _bigram(vocab: int) -> Callable[[Any, list[int]], np.ndarray]:
    """Deterministic bigram model: log P(next | previous token) only."""
    table = np.log(
        np.random.default_rng(7).dirichlet(np.ones(vocab), size=vocab)
    ).astype(np.float64)

    def log_probs(loaded: Any, ids: list[int]) -> np.ndarray:
        assert 0 < len(ids) <= loaded.config.model.max_seq_len
        return table[np.asarray(ids)]

    log_probs.table = table  # type: ignore[attr-defined]
    return log_probs


@pytest.fixture(scope="module")
def smoke_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Any:
    from tokenizers import Tokenizer

    root = tmp_path_factory.mktemp("tok")
    write_smoke_inputs(root, max_steps=8)
    return Tokenizer.from_file(str(root / "tokenizer" / "tokenizer.json"))


def test_long_continuations_are_scored_completely(
    smoke_tokenizer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: leading answer tokens were dropped beyond max_seq_len."""
    bigram = _bigram(smoke_tokenizer.get_vocab_size())
    monkeypatch.setattr(scoring, "log_probs", bigram)
    loaded = _Loaded(smoke_tokenizer, max_seq_len=8)
    context = "The cat sat on the mat."
    continuation = " " + " ".join(["and then it ran far away from home"] * 4)
    ctx, cont = scoring.split_pair(loaded, context, continuation)
    assert len(cont) > 2 * loaded.config.model.max_seq_len
    tokens = [*ctx, *cont]
    brute = sum(
        bigram.table[tokens[i - 1], tokens[i]] for i in range(len(ctx), len(tokens))
    )
    total, count, _ = scoring.continuation_score(loaded, context, continuation)
    assert count == len(cont)
    assert total == pytest.approx(brute)
    rolling = scoring.rolling_score(loaded, continuation)
    ids = [scoring.eot(loaded), *scoring.encode(loaded, continuation)]
    assert rolling == pytest.approx(
        sum(bigram.table[ids[i - 1], ids[i]] for i in range(1, len(ids)))
    )


def test_score_tokens_windows_every_target_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[int]] = []

    def uniform(loaded: Any, ids: list[int]) -> np.ndarray:
        calls.append(list(ids))
        assert len(ids) <= loaded.config.model.max_seq_len
        return np.full((len(ids), 10), -math.log(10))

    monkeypatch.setattr(scoring, "log_probs", uniform)
    loaded = _Loaded(_TinyTokenizer(), max_seq_len=4)
    values, _ = scoring.score_tokens(loaded, list(range(10)), 1)
    assert len(values) == 9 and values.sum() == pytest.approx(9 * -math.log(10))
    assert calls[0] == [0, 1, 2, 3]  # windows never exceed max_seq_len
    with pytest.raises(ValueError):
        scoring.score_tokens(loaded, [1, 2], 0)


def test_continuation_greedy_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    def peaked(loaded: Any, ids: list[int]) -> np.ndarray:
        rows = np.full((len(ids), 50), -20.0)
        for i, token in enumerate(ids):
            rows[i, (token + 1) % 50] = 0.0  # always predicts token + 1
        return rows

    monkeypatch.setattr(scoring, "log_probs", peaked)
    tokenizer = _TinyTokenizer()
    loaded = _Loaded(tokenizer)
    for word in ("a", "b", "c", "d"):
        tokenizer.encode(word)  # ids 1..4 in order
    assert scoring.continuation_score(loaded, "a b", " c d") == (0.0, 2, True)
    total, _, greedy = scoring.continuation_score(loaded, "a b", " d")
    assert total == pytest.approx(-20.0) and greedy is False


def test_split_pair_matches_the_harness_boundary_rules(smoke_tokenizer: Any) -> None:
    loaded = _Loaded(smoke_tokenizer)

    def enc(text: str) -> list[int]:
        return scoring.encode(loaded, text)

    # Trailing context whitespace moves into the continuation.
    ctx, cont = scoring.split_pair(loaded, "Question: 2+2=  ", "4")
    assert ctx == enc("Question: 2+2=")
    assert cont == enc("Question: 2+2=  4")[len(ctx) :]
    # Context and continuation are encoded together, then split by length.
    for context, continuation in [
        ("The quick brown", " fox"),
        ("The quick bro", "wn fox"),
        ("hello", "world"),
    ]:
        ctx, cont = scoring.split_pair(loaded, context, continuation)
        whole = enc(context + continuation)
        assert ctx == enc(context) and cont == whole[len(ctx) :]
    # An empty context is a single end-of-text token.
    ctx, cont = scoring.split_pair(loaded, "", "hello")
    assert ctx == [scoring.eot(loaded)] and cont == enc("hello")


def test_split_pair_agrees_with_lm_eval_when_installed(smoke_tokenizer: Any) -> None:
    pytest.importorskip("lm_eval")
    from lm_eval.api.model import TemplateLM

    loaded = _Loaded(smoke_tokenizer)

    class _Harness(TemplateLM):
        eot_token_id = scoring.eot(loaded)

        def tok_encode(self, string: str, **kwargs: Any) -> list[int]:
            return scoring.encode(loaded, string)

        def _loglikelihood_tokens(self, requests, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def loglikelihood_rolling(self, requests, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def generate_until(self, requests, **kwargs):  # pragma: no cover
            raise NotImplementedError

    harness = _Harness()
    for context, continuation in [
        ("Question: what is it?  ", "a cat"),
        ("The quick bro", "wn fox"),
        ("Answer:", " yes"),
    ]:
        assert tuple(scoring.split_pair(loaded, context, continuation)) == tuple(
            harness._encode_pair(context, continuation)
        )


def test_lm_eval_summary_prefers_acc_then_acc_norm() -> None:
    summary = lm_eval_adapter.summarize_results(
        {
            "results": {
                "piqa": {"acc,none": 0.6, "acc_norm,none": 0.55},
                "hellaswag": {"acc_norm,none": 0.3},
                "lambada_openai": {"acc,none": 0.0, "perplexity,none": 900.0},
            }
        },
        ["piqa", "hellaswag", "lambada_openai", "arc_easy"],
    )
    assert summary["tasks"]["piqa"]["acc"] == 0.6
    assert summary["tasks"]["hellaswag"]["acc"] == 0.3
    assert summary["tasks"]["arc_easy"]["acc"] is None
    assert summary["mean_accuracy"] == pytest.approx(0.3)


# --- End to end: try runs the fast tier; `probe` and `report` CLIs -----------


@pytest.fixture(scope="module")
def tried(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    from klab.lab_mode import run_try

    root = tmp_path_factory.mktemp("probe-lab")
    baseline = write_smoke_inputs(root, max_steps=8)
    delta = root / "wider.yaml"
    delta.write_text(
        yaml.safe_dump({"question": "wider?", "set": {"model.ffn_dim": 128}})
    )
    from klab.evaluation.inference import InferenceRun

    forwards: list[str] = []
    native = InferenceRun.evaluate

    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
        forwards.append("evaluate")
        return native(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(InferenceRun, "evaluate", counted)
        record, _ = run_try(delta, baseline, work_dir=root / "work")
    # One held-out scoring pass per arm; the probes reuse it (no extra forward).
    assert forwards == ["evaluate", "evaluate"]
    return root, record


def _cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    root: Path,
    *args: str,
) -> str:
    from klab.cli.main import main

    monkeypatch.setenv("KLAB_WORK_DIR", str(root / "work"))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys, "argv", ["klab", *args])
    capsys.readouterr()
    main()
    return capsys.readouterr().out


def test_try_runs_the_fast_tier_and_records_it(
    tried: tuple[Path, dict[str, Any]],
) -> None:
    root, record = tried
    probe = record["probe"]
    assert probe["format"] == runner.RESULT_FORMAT
    assert probe["tier"] == "fast" and probe["tiers_run"] == ["fast"]
    assert set(probe) == RESULT_KEYS
    assert [r["id"] for r in probe["probes"]] == [s.id for s in ordered("fast")]
    for row in probe["probes"]:
        assert set(row) == ROW_KEYS
        assert row["status"] in {"pass", "warn", "fail", "skipped"}
    assert set(probe["target"]) == IDENTITY_KEYS
    assert probe["target"]["run_id"] == record["arms"]["candidate"]["run_id"]
    assert probe["baseline"]["run_id"] == record["arms"]["baseline"]["run_id"]
    assert set(probe["verdict"]) == VERDICT_KEYS
    assert probe["verdict"]["missing"] == [] and probe["stop"]["stopped"] is False
    assert probe["target"]["eval_group"] == probe["baseline"]["eval_group"]
    loss = probe["probes"][0]
    # The probe's loss mirrors the lab try's native held-out loss exactly.
    assert loss["value"] == pytest.approx(
        record["arms"]["candidate"]["heldout"]["loss"], rel=1e-9
    )
    for key, value in probe["protocol"].items():
        assert record["eval_protocol"][key] == value
    progress = json.loads(
        (root / "work/lab/tries" / record["try_id"] / "probe-progress.json").read_text()
    )
    assert progress["state"] == "done"


@pytest.mark.parametrize("policy", ["standard", "final"])
def test_probe_cli_json_and_report(
    policy: str,
    tried: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, record = tried
    # The same sealed record handle used by report selects the exact candidate.
    candidate = record["try_id"]
    baseline = record["arms"]["baseline"]["run_id"]
    out = _cli(
        monkeypatch,
        capsys,
        root,
        "probe",
        candidate,
        "--vs",
        baseline,
        "--config",
        policy,
        "--json",
    )
    result = json.loads(out)
    assert (
        result["target"]["checkpoint_sha256"]
        == record["arms"]["candidate"]["checkpoint_sha256"]
    )
    assert set(result) >= RESULT_KEYS | {"probe_id", "record_sha256", "record"}
    assert result["request"]["tier"] == "standard"
    assert result["request"]["fast_fail"] is True
    assert result["request"]["baseline"] == baseline
    assert result["request"]["final"] is (policy == "final")
    assert result.get("final", False) is (policy == "final")
    assert result["tiers_run"][0] == "fast"
    ids = [r["id"] for r in result["probes"]]
    assert ids == [s.id for s in ordered("standard")]
    assert Path(result["record"]).is_file()

    text = _cli(monkeypatch, capsys, root, "report", result["probe_id"])
    assert text.startswith("PROBE BATTERY")
    assert "verdict " in text and "next → " in text and "\x1b[" not in text
    full = json.loads(
        _cli(monkeypatch, capsys, root, "report", result["probe_id"], "--json")
    )
    assert full["request"] == result["request"]
    assert full["record_sha256"] == result["record_sha256"]

    summary = _cli(monkeypatch, capsys, root, "report", record["try_id"])
    assert "PROBE BATTERY" in summary and "tier fast" in summary


def test_probe_cli_without_baseline_reports_absolute_values(
    tried: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, record = tried
    run_dir = root / "work/lab/runs" / record["arms"]["candidate"]["run_id"]
    text = _cli(monkeypatch, capsys, root, "probe", str(run_dir))
    assert "next → COMPARE" in text
    assert "--vs BASELINE" in text


def test_standalone_probe_scores_validation_once_per_arm(
    tried: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from klab.evaluation.inference import InferenceRun

    root, record = tried
    forwards: list[int] = []
    native = InferenceRun.evaluate

    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
        forwards.append(1)
        return native(self, *args, **kwargs)

    monkeypatch.setattr(InferenceRun, "evaluate", counted)
    out = _cli(
        monkeypatch,
        capsys,
        root,
        "probe",
        record["arms"]["candidate"]["run_id"],
        "--vs",
        record["arms"]["baseline"]["run_id"],
        "--json",
    )
    assert json.loads(out)["probes"][0]["status"] != "error"
    # held-out loss and calibration share one validation pass per arm
    assert len(forwards) == 2


def test_report_rejects_an_edited_probe_record(
    tried: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    root, record = tried
    source = root / "work/lab/tries" / record["try_id"] / "try.json"
    edited = json.loads(source.read_text())
    edited["probe"]["verdict"]["action"] = "longer_run"
    path = tmp_path / "try.json"
    path.write_text(json.dumps(edited))
    with pytest.raises(SystemExit):
        _cli(monkeypatch, capsys, root, "report", str(path))


def _stop_on_candidate_calibration(
    monkeypatch: pytest.MonkeyPatch, stop: Callable[[], None]
) -> None:
    """Stop once the candidate's held-out loss row is complete.

    Arms run baseline first, so the second calibration measurement is the
    candidate's, after its held-out loss was judged.
    """
    measure = runner._measure
    seen: list[str] = []

    def measure_then_stop(spec, loaded, arm, protocol):
        if spec.id == "calibration":
            seen.append(spec.id)
            if len(seen) == 2:
                stop()
        return measure(spec, loaded, arm, protocol)

    monkeypatch.setattr(runner, "_measure", measure_then_stop)


def _assert_partial(probe: dict[str, Any], kind: str) -> None:
    rows = {r["id"]: r for r in probe["probes"]}
    assert rows["heldout_loss"]["status"] in {"pass", "warn"}  # kept
    assert rows["heldout_loss"]["value"] is not None
    assert rows["repetition"]["status"] == "skipped"
    assert probe["stop"]["stopped"] is True and probe["stop"]["kind"] == kind
    verdict = probe["verdict"]
    assert verdict["status"] == "incomplete" and verdict["action"] == "rerun"
    assert any("repetition" in m["id"] for m in verdict["missing"])


@pytest.mark.parametrize("how", ["cancel", "signal"])
def test_try_stopped_during_probing_keeps_partial_battery_and_comparison(
    how: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.lab_mode import read_record, run_try

    baseline = write_smoke_inputs(tmp_path, max_steps=8)
    delta = tmp_path / "wider.yaml"
    delta.write_text(
        yaml.safe_dump({"question": "wider?", "set": {"model.ffn_dim": 128}})
    )
    work = tmp_path / "work"

    def stop() -> None:
        if how == "signal":
            os.kill(os.getpid(), signal.SIGTERM)
        else:
            (next((work / "lab/tries").glob("*/")) / "CANCEL").touch()

    _stop_on_candidate_calibration(monkeypatch, stop)
    record, code = run_try(delta, baseline, work_dir=work)
    assert record["status"] == "interrupted" and code != 0
    assert record["interruption"]["phase"] == "probing"
    if how == "signal":
        assert record["interruption"]["reason"] == "SIGTERM"
    # The sealed record on disk carries the finalized partial battery.
    (path,) = (work / "lab/tries").glob("*/try.json")
    sealed = read_record(path)
    _assert_partial(sealed["probe"], "interrupted" if how == "signal" else "cancelled")
    # The completed training comparison survives the stopped battery.
    assert sealed["comparison"]["heldout_loss_delta"] is not None
    assert sealed["arms"]["candidate"]["heldout"]["loss"] is not None
    progress = json.loads((path.parent / "probe-progress.json").read_text())
    assert progress["state"] == "stopped"


def test_standalone_probe_signal_publishes_the_partial_battery(
    tried: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from klab.lab_records import read_lab_record

    root, record = tried
    before = set((root / "work/lab/probes").glob("*/"))
    _stop_on_candidate_calibration(
        monkeypatch, lambda: os.kill(os.getpid(), signal.SIGTERM)
    )
    with pytest.raises(SystemExit) as exited:
        _cli(
            monkeypatch,
            capsys,
            root,
            "probe",
            record["arms"]["candidate"]["run_id"],
            "--vs",
            record["arms"]["baseline"]["run_id"],
        )
    assert exited.value.code == 130
    (folder,) = set((root / "work/lab/probes").glob("*/")) - before
    _, sealed = read_lab_record(folder / "probe.json", "probe")
    _assert_partial(sealed, "interrupted")
    assert sealed["request"]["tier"] == "fast"
    assert sealed["request"]["baseline"] == record["arms"]["baseline"]["run_id"]
    assert sealed["stop"]["reason"] == "interrupted: SIGTERM"
    progress = json.loads((folder / "progress.json").read_text())
    assert progress["state"] == "stopped"
    assert signal.getsignal(signal.SIGTERM) is not None  # handlers restored


def test_nonfinite_native_loss_is_a_numerical_hard_failure(
    tried: tuple[Path, dict[str, Any]],
) -> None:
    """A NaN loss from the real evaluator stops the battery: abandon, not rerun."""
    import torch

    from klab.probes.cli import load_target

    root, record = tried
    lab = root / "work/lab"

    def arm(name: str, poison: bool) -> runner.Arm:
        def load() -> Any:
            loaded = load_target(
                record["arms"][name]["run_id"],
                lab_dir=lab,
                runs_dir=None,
                backend=None,
                authorization=None,
            )
            if poison:
                with torch.no_grad():
                    for parameter in loaded.model.parameters():
                        parameter.fill_(float("nan"))
            return loaded

        return runner.Arm(load=load)

    result = runner.run_battery(
        arm("candidate", True), arm("baseline", False), tier="standard"
    )
    loss = result["probes"][0]
    assert loss["id"] == "heldout_loss" and loss["status"] == "fail"
    assert loss["note"] == "numerical failure: nonfinite validation loss"
    assert loss["details"]["numerical_failure"] == "nonfinite validation loss"
    assert result["stop"]["kind"] == "fast_fail"
    assert result["stop"]["reason"] == "fast-fail: numerical failure in heldout_loss"
    assert all(r["status"] == "skipped" for r in result["probes"][1:])
    verdict = result["verdict"]
    assert verdict["status"] == "fail" and verdict["action"] == "abandon"
    assert verdict["reasons"] == [
        "hard fail: heldout_loss (numerical failure: nonfinite validation loss)"
    ]
    assert "Numerical failure" in verdict["suggestion"]
    assert verdict["missing"] == []
    text = render(result, color=False)
    assert "numerical failure" in text and "ABANDON" in text

    # A broken baseline is not the candidate's fault: missing evidence instead.
    flipped = runner.run_battery(
        arm("candidate", False), arm("baseline", True), tier="fast"
    )
    assert flipped["probes"][0]["status"] == "error"
    assert flipped["verdict"]["action"] == "rerun"


def test_dashboard_history_reads_verified_records(
    tried: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    import shutil

    from klab.dashboard.probe_data import (
        history_rows,
        live_batteries,
        load_history,
    )
    from klab.probes.points import (
        metric_points,
        pareto_frontier,
        points_from_record,
    )

    root, record = tried
    entries, rejected = load_history(root / "work/lab")
    assert any(e.source == "try" and e.key == record["try_id"] for e in entries)
    assert rejected == []
    rows = history_rows(entries)
    assert all(r["verdict"] for r in rows)
    assert live_batteries(root / "work/lab") == []  # finished batteries are not live
    assert pareto_frontier([(1, 3), (2, 2), (3, 3), (0.5, 4)]) == [3, 0, 1]

    # The dashboard rejects an edited record exactly like `report` does.
    lab = tmp_path / "lab"
    shutil.copytree(root / "work/lab/tries", lab / "tries")
    edited = lab / "tries" / record["try_id"] / "try.json"
    body = json.loads(edited.read_text())
    body["probe"]["verdict"]["status"] = "pass"
    body["probe"]["verdict"]["action"] = "longer_run"
    edited.write_text(json.dumps(body))
    kept, dropped = load_history(lab)
    assert record["try_id"] not in {e.key for e in kept}
    assert [path for path, _ in dropped] == [edited]

    # Pareto points: one per checkpoint, grouped by eval protocol identity.
    # The try arms and the probe battery of the same checkpoints agree on the
    # eval group, so they collapse into one point each.
    path = root / "work/lab/tries" / record["try_id"] / "try.json"
    points = metric_points(points_from_record("try", record, path), "heldout_loss")
    checkpoints = [p["checkpoint_sha256"] for p in points]
    assert len(checkpoints) == len(set(checkpoints))
    probe = record["probe"]
    assert {
        probe["target"]["checkpoint_sha256"],
        probe["baseline"]["checkpoint_sha256"],
    } == set(checkpoints)
    assert {p["group"] for p in points} == {probe["target"]["eval_group"]}
    for arm in record["arms"].values():
        assert arm["eval_group"] == probe["target"]["eval_group"]


def test_pareto_points_are_unique_per_group_and_checkpoint() -> None:
    from klab.probes.points import metric_points, probe_points

    validation = {"validation_sha256": "v", "tokenizer_sha256": "t"}
    short = eval_group(validation, {"seq_len": 32, "batch_size": 4})
    long = eval_group(validation, {"seq_len": 64, "batch_size": 4})
    assert short != long

    def entry(key: str, group: str, loss: float) -> list[dict[str, Any]]:
        who = {
            "run_id": "cand",
            "step": 12,
            "checkpoint_sha256": "same-checkpoint",
            "eval_group": group,
            "parameters": 1000,
        }
        row = runner._row(BY_ID["heldout_loss"], status="pass", value=loss)
        result = {"target": who, "baseline": None, "probes": [row], "verdict": {}}
        return probe_points(result, source=key, path=None, packaged=False)

    # Newest first: the same checkpoint under two protocols, and a stale repeat.
    points = metric_points(
        [
            *entry("new-long", long, 3.0),
            *entry("short", short, 4.0),
            *entry("old-long", long, 9.0),
        ],
        "heldout_loss",
    )
    by_group = {p["group"]: p for p in points}
    assert len(points) == 2 and set(by_group) == {short, long}
    assert by_group[long]["value"] == 3.0 and by_group[long]["source"] == "new-long"
    assert by_group[short]["value"] == 4.0
    assert {p["checkpoint_sha256"] for p in points} == {"same-checkpoint"}


def test_explore_cli_reads_a_lab_run_through_the_verified_loader(
    tried: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CI-safe explorer smoke: a real (tiny) lab run, explored and cached."""
    from klab.explorer import EXPLORER_FORMAT, cached_explorations, explore

    root, record = tried
    run_id = record["arms"]["candidate"]["run_id"]
    lab = root / "work" / "lab"
    request = root / "explore.yaml"
    request.write_text("text: Once upon a\n")
    out = _cli(
        monkeypatch, capsys, root, "explore", record["try_id"], "--config", str(request)
    )
    assert out.startswith(f"EXPLORE  {run_id}")
    assert "cache:" in out
    (cached,) = cached_explorations(lab)
    assert cached["format"] == EXPLORER_FORMAT
    assert (
        cached["target"]["checkpoint_sha256"]
        == (record["arms"]["candidate"]["checkpoint_sha256"])
    )
    assert cached["attention"]["layers"] and cached["tokens"][1]["loss"] > 0
    # Same checkpoint and text: served from the sealed cache.
    again, path = explore(run_id, lab_dir=lab, text="Once upon a")
    assert again["created_at"] == cached["created_at"] and path is not None


def test_explore_cli_uses_the_shared_runtime_preparation_and_envelope(
    tried: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """explore goes through _prepare_runtime_command like probe/try: the
    device is authorized there and the envelope reaches every stage check."""
    import klab.cli.main as cli_main
    import klab.resource_envelope as envelopes

    root, record = tried
    run_id = record["arms"]["candidate"]["run_id"]
    seen: list[str] = []
    native_require = cli_main.require_authorization

    def require(config: Any, authorization: Any) -> None:
        seen.append(config.runtime.backend)
        native_require(config, authorization)

    monkeypatch.setattr(cli_main, "require_authorization", require)
    envelope = root / "envelope.yaml"
    envelope.write_text(
        yaml.safe_dump({"resource_envelope_version": 1, "max_rss_bytes": 1 << 50})
    )
    phases: list[str] = []
    native_measure = LabContext.measure

    def measure(self: LabContext) -> Any:
        assert self.resource_envelope is not None
        phases.append(self.phase)
        return native_measure(self)

    monkeypatch.setattr(LabContext, "measure", measure)
    request = root / "explore.yaml"
    request.write_text("text: Once upon\nrefresh: true\n")
    args = ("explore", run_id, "--config", str(request))
    out = _cli(monkeypatch, capsys, root, *args, "--resource-envelope", str(envelope))
    assert out.startswith(f"EXPLORE  {run_id}")
    assert seen == ["cpu"]
    assert phases[:2] == ["load", "weights"] and len(phases) >= 3
    # A violation at a stage check stops the exploration cleanly.
    monkeypatch.setattr(
        envelopes,
        "check_envelope",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("rss over the limit")),
    )
    with pytest.raises(SystemExit) as stopped:
        _cli(monkeypatch, capsys, root, *args, "--resource-envelope", str(envelope))
    assert "stopped" in str(stopped.value) and "rss over the limit" in str(
        stopped.value
    )


def test_explore_compares_pinned_generations(
    tried: tuple[Path, dict[str, Any]],
) -> None:
    """Explore two immutable generations of one real lab run and compare activations."""
    from klab.explorer import (
        ACTIVATION_SITES,
        EXPLORER_FORMAT,
        compare_explorations,
        explore,
    )

    root, record = tried
    lab = root / "work" / "lab"
    run_dir = lab / "runs" / record["arms"]["candidate"]["run_id"]
    gen_4_dir = next((run_dir / "checkpoints").glob("step_00000004_gen_*"))
    gen_8_dir = max((run_dir / "checkpoints").glob("step_00000008_gen_*"))
    gen_4, _ = explore(str(gen_4_dir), lab_dir=lab, text="Once upon a")
    gen_8, _ = explore(str(gen_8_dir), lab_dir=lab, text="Once upon a")

    assert gen_4["format"] == gen_8["format"] == EXPLORER_FORMAT
    assert gen_4["target"]["step"] == 4 and gen_8["target"]["step"] == 8
    assert gen_4["target"]["checkpoint_sha256"] != gen_8["target"]["checkpoint_sha256"]
    assert (
        gen_4["target"]["tokenizer_sha256"]
        == gen_8["target"]["tokenizer_sha256"]
        == record["probe"]["target"]["tokenizer_sha256"]
    )
    assert [t["id"] for t in gen_4["tokens"]] == [t["id"] for t in gen_8["tokens"]]

    comparison = compare_explorations(gen_4, gen_8)
    assert comparison["architecture_matches"] is True
    assert comparison["common_layers"] == [0, 1]
    assert comparison["only_left"] == [] and comparison["only_right"] == []
    assert comparison["delta"] is not None
    for layer_key in ("0", "1"):
        for site in ACTIVATION_SITES:
            deltas = comparison["delta"][layer_key][site]
            assert len(deltas) == len(gen_4["tokens"])
            assert any(abs(v) > 0 for v in deltas)
