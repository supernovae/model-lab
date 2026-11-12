"""Evaluation integrity: the held-back final split, NOT_COMPARABLE precedence,
eval-noise labelling and the recall-pick investigation (PR3)."""

from __future__ import annotations

import ast
import random
from pathlib import Path
from typing import Any

import pytest

from klab.probes import compare as compare_mod
from klab.probes import runner, scoring
from klab.probes import suite as suite_mod
from klab.probes.suite import (
    BY_ID,
    FinalSplitLocked,
    fact_items,
    final_verdict_access,
    needle_split,
    parametric_items,
    prompts,
)
from klab.probes.verdict import (
    SEED_CAVEAT,
    decide,
    respect_try_comparison,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "klab"
READ_FILTERS = {"compare", "collect_points", "resolve_point"}


# --- Held-back final split -----------------------------------------------------


# Every split reader goes through suite.require_split: one case for the shared
# _pick path and one for the reader that calls require_split directly.
@pytest.mark.parametrize(
    "read",
    [
        lambda: fact_items("final"),
        lambda: parametric_items("final", seed=0),
    ],
    ids=["facts", "parametric"],
)
def test_final_split_is_locked_outside_a_final_verdict(read: Any) -> None:
    with pytest.raises(FinalSplitLocked):
        read()
    with final_verdict_access():
        assert read()
    with pytest.raises(FinalSplitLocked):
        read()  # closed again


def test_final_split_is_deterministic_and_disjoint_from_selection_splits() -> None:
    with final_verdict_access():
        final = {
            "facts": fact_items("final"),
            "needles": needle_split("final"),
            "prompts": prompts("final"),
        }
        assert final["facts"] == fact_items("final")
    for split in suite_mod.SELECTION_SPLITS:
        facts = fact_items(split)
        assert {i["answer"] for i in final["facts"]}.isdisjoint(
            i["answer"] for i in facts
        )
        assert {i["question"] for i in final["facts"]}.isdisjoint(
            i["question"] for i in facts
        )
        assert set(final["needles"]["needles"]).isdisjoint(
            needle_split(split)["needles"]
        )
        assert set(final["needles"]["filler"]).isdisjoint(needle_split(split)["filler"])
        assert set(final["prompts"]).isdisjoint(prompts(split))
    for item in final["facts"]:
        assert item["answer"] in item["candidates"]


def test_selection_paths_never_open_the_final_split() -> None:
    """Only the probe CLI's final-declaration path may open the held-back split.

    `final_verdict_access` is entered only in suite.py (digest) and the
    run_battery wrapper; `final=` is passed only by the probe CLI. try/lab
    mode, iteration and the dashboard never reference either.
    """
    openers, passers = set(), set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = str(path.relative_to(SRC))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "final_verdict_access":
                openers.add(rel)
            if isinstance(node, ast.Call):
                callee = getattr(node.func, "attr", getattr(node.func, "id", ""))
                for keyword in node.keywords:
                    if keyword.arg == "final":
                        passers.add((rel, callee))
    assert openers == {"probes/suite.py", "probes/runner.py"}
    # `final=` opens the split only through run_battery (the probe CLI); the
    # compare/points `final=` is a read filter for explicit final reporting.
    battery = {rel for rel, callee in passers if callee not in READ_FILTERS}
    assert battery == {"probes/cli.py", "cli/main.py"}
    readers = {rel for rel, callee in passers if callee in READ_FILTERS}
    assert readers == {"probes/compare.py", "cli/main.py"}


def test_try_probe_battery_is_a_selection_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from klab import lab_mode

    seen: dict[str, Any] = {}

    def fake_battery(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        assert suite_mod.verdict_split() == "heldout"
        return {"status": "ok"}

    monkeypatch.setattr(runner, "run_battery", fake_battery)
    record = {
        "arms": {
            arm: {"run_id": f"run-{arm}", "checkpoint_sha256": "x"}
            for arm in ("baseline", "candidate")
        },
        "comparison": {"verdict": "CANDIDATE_LOWER_LOSS"},
    }
    from klab.lab_context import LabContext

    lab_mode._probe_arms(
        record,
        tmp_path,
        {},
        "standard",
        tmp_path,
        None,
        LabContext(tmp_path / "CANCEL"),
        {},
    )
    assert "final" not in seen
    assert seen["comparison"] == record["comparison"]


def test_final_verdict_scores_the_held_back_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scoring, "continuation_score", lambda loaded, prefix, cont: (0.0, 1, False)
    )
    loaded = None  # scoring is stubbed; fact recall needs no model here
    spec = BY_ID["fact_recall"]
    arm = runner.Arm(load=lambda: loaded)
    heldout = runner._measure(spec, loaded, arm, {})
    with final_verdict_access():
        final = runner._measure(spec, loaded, arm, {})
    assert heldout["item_group"] != final["item_group"]
    with final_verdict_access():
        expected = {i["answer"] for i in fact_items("final")}
    assert {i["answer"] for i in final["items"]} == expected
    with pytest.raises(ValueError, match="standalone"):
        runner.run_battery(arm, final=True, comparison={"verdict": "TIE"})


# --- NOT_COMPARABLE takes precedence -------------------------------------------


def _loss_row(**extra: Any) -> dict[str, Any]:
    return {"id": "heldout_loss", "status": "pass", "improved": True, **extra}


NOT_COMPARABLE = {"verdict": "NOT_COMPARABLE", "failed": ["same_training_data"]}


def test_not_comparable_try_never_escalates() -> None:
    kwargs = {
        "has_baseline": True,
        "tiers_run": ["fast"],
        "requested_tier": "fast",
        "guard": None,
        "specs": BY_ID,
    }
    promising = decide([_loss_row()], **kwargs)
    assert promising["action"] == "escalate"
    blocked = decide([_loss_row()], comparison=NOT_COMPARABLE, **kwargs)
    assert blocked["status"] == "incomplete" and blocked["action"] == "rerun"
    assert "Promising" not in blocked["suggestion"]
    assert "same_training_data" in blocked["reasons"][0]
    # A comparable try is unaffected.
    fine = decide(
        [_loss_row()], comparison={"verdict": "CANDIDATE_LOWER_LOSS"}, **kwargs
    )
    assert fine == promising
    # Older records: the shown verdict follows the try comparison too.
    assert respect_try_comparison(promising, NOT_COMPARABLE)["action"] == "rerun"
    assert respect_try_comparison(promising, None) is promising


def test_dashboard_activity_shows_not_comparable_tries_as_not_escalating(
    tmp_path: Path,
) -> None:
    from klab.dashboard import lab_data

    record = {
        "try_id": "t1",
        "created_at": "2026-10-10T00:00:00Z",
        "comparison": NOT_COMPARABLE,
        "probe": {"verdict": {"status": "pass", "action": "escalate"}},
    }
    snap = lab_data.LabSnapshot(
        lab_dir=tmp_path,
        entries=[],
        tries=[(tmp_path / "try.json", record)],
        probes=[],
        rejected=[],
        points=[],
    )
    (row,) = lab_data.activity(snap)
    assert row["action"] == "rerun" and row["verdict"] == "incomplete"


# --- Error bars: within-run eval noise, paired seeds ---------------------------


def test_single_seed_win_recommends_paired_seeds() -> None:
    out = decide(
        [_loss_row()],
        has_baseline=True,
        tiers_run=["fast", "standard", "full"],
        requested_tier="full",
        guard=None,
        specs=BY_ID,
    )
    assert out["action"] == "longer_run"
    assert out["suggestion"].endswith(SEED_CAVEAT)
    assert "within-run eval noise" in " ".join(out["reasons"])
    assert "paired seeds" in SEED_CAVEAT
    assert "within-run eval noise" in BY_ID["heldout_loss"].explains


def test_rendered_loss_error_bar_is_labelled_eval_noise() -> None:
    from klab.probes.render import _delta

    row = {"id": "heldout_loss", "delta": -0.01, "delta_se": 0.002, "thresholds": {}}
    assert "(eval noise, 1 seed)" in _delta(row)
    assert "eval noise" not in _delta({**row, "id": "fact_recall"})


# --- Recall picks: prior, not candidate order ----------------------------------


def _scored_by(monkeypatch: pytest.MonkeyPatch, score: Any) -> None:
    monkeypatch.setattr(
        scoring,
        "continuation_score",
        lambda loaded, prefix, cont: (score(prefix, cont.strip()), 1, False),
    )


def _shuffled(items: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for item in items:
        candidates = list(item["candidates"])
        rng.shuffle(candidates)
        out.append({**item, "candidates": candidates})
    return out


def test_ties_are_shown_as_ties_and_earn_chance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flat model is never reported as picking the first candidate."""
    _scored_by(monkeypatch, lambda prefix, candidate: -1.0)
    items = runner.recall_items("heldout")
    for order in (items, _shuffled(items, 1)):
        scores = scoring.ranking_scores(None, order)
        picks = [scoring.top_candidate(s) for s in scores]
        assert all(p.startswith("tie: ") for p in picks)
        assert scoring.ranking_credit(None, order) == pytest.approx(
            [1 / len(items[0]["candidates"])] * len(items)
        )


def test_context_free_prior_gives_constant_picks_at_chance_whatever_the_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PR #63 demo finding, reproduced: an untrained model's near-uniform
    output still has tiny prior preferences, so it picks the same candidate
    for every question in a candidate set (not the first one). Accuracy is
    then exactly chance, and candidate order changes nothing."""
    prior = {"gold": -6.90, "falcon": -6.11}  # e.g. the demo baseline's argmax
    _scored_by(monkeypatch, lambda prefix, candidate: prior.get(candidate, -7.0))
    items = runner.recall_items("heldout")
    base = [scoring.top_candidate(s) for s in scoring.ranking_scores(None, items)]
    assert set(base) == {"gold", "falcon"}
    assert scoring.constant_pick(items, base)
    for seed in range(3):
        shuffled = _shuffled(items, seed)
        picks = [
            scoring.top_candidate(s) for s in scoring.ranking_scores(None, shuffled)
        ]
        assert picks == base
    credits = scoring.ranking_credit(None, items)
    assert sum(credits) / len(credits) == pytest.approx(1 / 4)


def test_a_model_that_reads_the_context_is_not_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scored_by(
        monkeypatch, lambda prefix, candidate: 0.0 if candidate in prefix else -1
    )
    items = runner.recall_items("heldout")
    picks = [scoring.top_candidate(s) for s in scoring.ranking_scores(None, items)]
    assert picks == [i["answer"] for i in items]
    assert not scoring.constant_pick(items, picks)


# --- One reference resolver ----------------------------------------------------


def test_resolve_point_is_the_only_reference_resolver(tmp_path: Path) -> None:
    assert not hasattr(compare_mod, "_reference_point")
    with pytest.raises(ValueError, match="no result for ref:SmolLM2-135M"):
        compare_mod.resolve_point("ref:SmolLM2-135M", tmp_path, [])


# --- Final evidence never reaches selection (PR #72 review) --------------------

SHA = {name: name[0] * 64 for name in ("aaa", "bbb", "ccc")}
EVAL_GROUP = "e" * 64
FACT_GROUP = "f" * 64  # matching final item groups: pairable if allowed through


def _arm_identity(name: str) -> dict[str, Any]:
    return {
        "run_id": f"run-{name}",
        "checkpoint_sha256": SHA[name],
        "step": 60,
        "tokens_seen": 7680,
        "eval_group": EVAL_GROUP,
    }


def _final_probe(target: str, baseline: str, score: float) -> dict[str, Any]:
    items = [
        {"credit": score, "baseline_credit": 0.0},
        {"credit": score, "baseline_credit": 0.0},
    ]
    row = {
        "status": "pass",
        "baseline_value": 0.0,
        "details": {"item_group": FACT_GROUP, "items": items, "chance": 0.25},
    }
    return {
        "format": "klab-probe-v1",
        "probe_id": f"probe-final-{target}",
        "created_at": f"2026-10-10T20:00:0{len(target) % 10}+00:00",
        "suite": {"sha256": suite_mod.suite_identity()["sha256"]},
        "tier": "standard",
        "target": _arm_identity(target),
        "baseline": _arm_identity(baseline),
        "verdict": {"status": "pass", "action": "report", "next_tier": None},
        "final": True,
        "probes": [
            {"id": "fact_recall", "value": score, **row},
            {"id": "needle", "value": score, **row},
        ],
    }


@pytest.fixture
def final_lab(tmp_path: Path) -> Path:
    """Ordinary fast tries A and B, then `probe --config final` records for both."""
    from klab.lab_records import write_sealed

    for name, loss in (("aaa", 3.2), ("bbb", 3.3)):
        path = tmp_path / f"tries/try-{name}/try.json"
        path.parent.mkdir(parents=True)
        write_sealed(
            path,
            {
                "format": "klab-lab-try-v1",
                "try_id": f"try-{name}",
                "created_at": "2026-10-10T18:00:00+00:00",
                "arms": {
                    "baseline": {**_arm_identity("ccc"), "heldout": {"loss": 3.4}},
                    "candidate": {**_arm_identity(name), "heldout": {"loss": loss}},
                },
                "comparison": {"verdict": "CANDIDATE_LOWER_LOSS"},
            },
        )
    for target, score in (("aaa", 1.0), ("bbb", 0.5)):
        path = tmp_path / f"probes/probe-final-{target}/probe.json"
        path.parent.mkdir(parents=True)
        write_sealed(path, _final_probe(target, "ccc", score))
    return tmp_path


def test_ordinary_compare_never_acquires_final_evidence(final_lab: Path) -> None:
    from klab.probes.points import collect_points

    report = compare_mod.compare("try-aaa", ["try-bbb"], lab_dir=final_lab)
    assert report["final"] is False
    for point in report["points"]:
        assert set(point["metrics"]) == {"heldout_loss"}
        assert not point["final"]
    pairs = {p["metric"]: p for p in report["comparisons"][0]["pairs"]}
    assert pairs["heldout_loss"]["status"] == "lower"
    for metric in ("fact_recall", "needle"):
        assert pairs[metric]["status"] == "missing_evidence"
        assert pairs[metric]["value"] is None
    assert compare_mod.as_json(report)["final"] is False
    # Final records are refused as ordinary subjects or others.
    with pytest.raises(ValueError, match="final verdict"):
        compare_mod.compare("probe-final-aaa", ["try-bbb"], lab_dir=final_lab)
    with pytest.raises(ValueError, match="final verdict"):
        compare_mod.compare("try-aaa", ["probe-final-bbb"], lab_dir=final_lab)
    assert not any(p["final"] for p in collect_points(final_lab))


def test_final_results_are_reported_only_explicitly(final_lab: Path) -> None:
    report = compare_mod.compare(
        "probe-final-aaa", ["probe-final-bbb"], lab_dir=final_lab, final=True
    )
    assert report["final"] and all(p["final"] for p in report["points"])
    pairs = {p["metric"]: p for p in report["comparisons"][0]["pairs"]}
    assert pairs["fact_recall"]["delta"] == pytest.approx(0.5)
    assert "FINAL" in compare_mod.render(report).splitlines()[0]
    with pytest.raises(ValueError, match="not a final verdict"):
        compare_mod.compare(
            "try-aaa", ["probe-final-bbb"], lab_dir=final_lab, final=True
        )
    with pytest.raises(ValueError, match="references"):
        compare_mod.compare(
            "probe-final-aaa", [], lab_dir=final_lab, final=True, references=True
        )


def test_dashboard_never_ranks_final_results(final_lab: Path) -> None:
    from klab.dashboard import lab_data

    snap = lab_data.snapshot(final_lab)
    assert snap.points and not any(p["final"] for p in snap.points)
    assert not any(e.source == "probe" for e in snap.entries)
    rows = {r["id"]: r for r in lab_data.activity(snap)}
    final_row = rows["probe-final-aaa"]
    assert final_row["kind"] == "final" and final_row["loss_delta"] is None
    assert "FINAL" in final_row["what"]
    catalog = lab_data.checkpoint_catalog(snap.points)
    assert all(
        r.get("fact_recall") is None for r in catalog if r["kind"] != "reference"
    )


class _Model:
    training = False

    def eval(self) -> None: ...

    def train(self, mode: bool = True) -> None: ...


class _LoadedStub:
    engine = None
    model = _Model()


def _primed_arm(name: str) -> runner.Arm:
    """An Arm whose identity is already described (no run directory needed)."""
    arm = runner.Arm(load=_LoadedStub)
    arm.cache.update(
        identity=_arm_identity(name),
        validation_identity={"validation_sha256": "v" * 64},
        tokenizer_digest="t" * 64,
        config=None,
    )
    return arm


@pytest.mark.parametrize(
    "final_first", [False, True], ids=["ordinary-first", "final-first"]
)
def test_reused_arms_score_each_split_afresh(
    monkeypatch: pytest.MonkeyPatch, final_first: bool
) -> None:
    """Two full battery calls on the same Arms never share split-scored results."""
    _scored_by(monkeypatch, lambda prefix, answer: float(len(answer)))
    measured: list[tuple[str, str]] = []
    real_measure = runner._measure

    def counting(spec: Any, loaded: Any, arm: Any, protocol: Any) -> Any:
        measured.append((spec.id, suite_mod.verdict_split()))
        return real_measure(spec, loaded, arm, protocol)

    monkeypatch.setattr(runner, "_measure", counting)
    target, baseline = _primed_arm("aaa"), _primed_arm("ccc")
    options = {"tier": "standard", "fast_fail": False, "protocol": {"seq_len": 32}}

    def battery(final: bool) -> dict[str, Any]:
        return runner.run_battery(target, baseline, final=final, **options)

    order = [True, False] if final_first else [False, True]
    results = {final: battery(final) for final in order}

    def fact(result: dict[str, Any]) -> dict[str, Any]:
        return next(r for r in result["probes"] if r["id"] == "fact_recall")

    ordinary, final = fact(results[False]), fact(results[True])
    assert ordinary["details"]["item_group"] != final["details"]["item_group"]
    with final_verdict_access():
        final_answers = {i["answer"] for i in fact_items("final")}
    assert {i["answer"] for i in final["details"]["items"]} == final_answers
    assert {i["answer"] for i in ordinary["details"]["items"]}.isdisjoint(final_answers)
    # Each arm scored fact recall once per split, never reused across them.
    recall = [split for probe, split in measured if probe == "fact_recall"]
    assert sorted(recall) == ["final", "final", "heldout", "heldout"]
    assert results[True]["final"] and "final" not in results[False]
    assert results[True]["verdict"]["next_tier"] is None
    assert results[True]["verdict"]["action"] in {"report", "rerun"}
    # Same split again: the arm's cached measurement is reused.
    battery(False)
    assert [s for p, s in measured if p == "fact_recall"] == recall


# --- Completed final verdicts are terminal -------------------------------------


@pytest.mark.parametrize("improved", [True, False], ids=["improved", "unchanged"])
def test_completed_final_verdict_is_report_only(improved: bool) -> None:
    from klab.probes.verdict import final_verdict

    ordinary = decide(
        [_loss_row(improved=improved)],
        has_baseline=True,
        tiers_run=["fast", "standard"],
        requested_tier="standard",
        guard=None,
        specs=BY_ID,
    )
    assert ordinary["action"] == ("escalate" if improved else "tweak")
    out = final_verdict(ordinary)
    assert out["action"] == "report" and out["next_tier"] is None
    assert out["status"] == ordinary["status"]
    assert out["reasons"][: len(ordinary["reasons"])] == ordinary["reasons"]
    for selecting in ("Promising", "--config", "bolder", "longer run", "paired seeds"):
        assert selecting not in out["suggestion"]
    assert "report" in out["suggestion"] and "terminal" in out["suggestion"]


def test_incomplete_final_verdict_reruns_unchanged() -> None:
    from klab.probes.verdict import final_verdict

    out = final_verdict(
        {
            "status": "incomplete",
            "action": "rerun",
            "next_tier": "standard",
            "reasons": ["missing evidence"],
            "missing": [{"id": "needle"}],
            "suggestion": "fix it and retry",
        }
    )
    assert out["action"] == "rerun" and out["next_tier"] is None
    assert "needle" in out["suggestion"] and "unchanged" in out["suggestion"]


# --- Final verdicts with missing evidence stay retryable -----------------------

LM_EVAL_MISSING = {
    "id": "lm_eval",
    "status": "unavailable",
    "note": "lm-eval not installed",
}


def _decide_final(results: list[dict[str, Any]], *, has_baseline: bool) -> Any:
    from klab.probes.verdict import final_verdict

    ordinary = decide(
        results,
        has_baseline=has_baseline,
        tiers_run=["fast", "standard", "full"],
        requested_tier="full",
        guard=None,
        specs=BY_ID,
    )
    return ordinary, final_verdict(ordinary)


def test_no_baseline_final_with_missing_evidence_is_rerun() -> None:
    ordinary, out = _decide_final([_loss_row(), LM_EVAL_MISSING], has_baseline=False)
    assert ordinary["status"] == "incomplete" and ordinary["action"] == "rerun"
    assert ordinary["missing"] == [LM_EVAL_MISSING]
    assert out["status"] == "incomplete" and out["action"] == "rerun"
    assert out["next_tier"] is None
    assert "lm_eval" in out["suggestion"] and "unchanged" in out["suggestion"]
    assert "--vs BASELINE" not in out["suggestion"]
    assert "missing evidence: lm_eval (unavailable)" in out["reasons"]
    # Complete and without a baseline: terminal.
    _, done = _decide_final([_loss_row()], has_baseline=False)
    assert done["status"] == "info" and done["action"] == "report"


def test_soft_fail_with_missing_evidence_is_rerun_but_decisive_failures_report() -> (
    None
):
    soft = {"id": "calibration", "status": "fail"}
    _, out = _decide_final([_loss_row(), soft, LM_EVAL_MISSING], has_baseline=True)
    assert out["action"] == "rerun" and out["status"] == "incomplete"
    hard = {"id": "heldout_loss", "status": "fail"}
    assert BY_ID["heldout_loss"].hard
    _, out = _decide_final([hard, LM_EVAL_MISSING], has_baseline=True)
    assert out["action"] == "report" and out["status"] == "fail"
    nan = _loss_row(status="fail", details={"numerical_failure": "nan"}, value=None)
    for has_baseline in (True, False):
        _, out = _decide_final([nan, LM_EVAL_MISSING], has_baseline=has_baseline)
        assert out["action"] == "report" and out["status"] == "fail"


@pytest.mark.parametrize("nan_loss", [False, True], ids=["lm-eval-missing", "nan-loss"])
def test_no_baseline_final_battery_without_lm_eval(
    monkeypatch: pytest.MonkeyPatch, nan_loss: bool
) -> None:
    """a full final declaration with fast_fail: false without --vs or lm-eval."""
    real_judge = runner._judge

    def measure(spec: Any, loaded: Any, arm: Any, protocol: Any) -> dict[str, Any]:
        if spec.id == "lm_eval":
            raise runner.ProbeUnsupported("lm-eval not installed")
        if spec.id == "heldout_loss" and nan_loss:
            return {"value": float("nan"), "numerical_failure": "nan"}
        return {"value": 0.5}

    def judge(spec: Any, t: Any, b: Any, comparable: Any, dev: Any) -> Any:
        if "status" in t or "numerical_failure" in t:
            return real_judge(spec, t, b, comparable, dev)
        return {**runner._row(spec), "status": "info", "value": t["value"]}

    monkeypatch.setattr(runner, "_measure", measure)
    monkeypatch.setattr(runner, "_judge", judge)
    result = runner.run_battery(
        _primed_arm("aaa"),
        None,
        final=True,
        tier="full",
        fast_fail=False,
        protocol={"seq_len": 32},
    )
    verdict = result["verdict"]
    assert [m["id"] for m in verdict["missing"]] == ["lm_eval"]
    assert verdict["next_tier"] is None
    if nan_loss:  # a numerical failure stays decisive despite the gap
        assert verdict["status"] == "fail" and verdict["action"] == "report"
    else:
        assert verdict["status"] == "incomplete" and verdict["action"] == "rerun"
        assert "re-run --final" not in verdict["suggestion"]
