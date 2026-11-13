"""Lab dashboard: data views, explorer figures and page renders (CPU, no training)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from test_explorer import MOE_NGRAM, TEXT, _loaded

from klab import explorer
from klab.dashboard import explorer_view as xv
from klab.dashboard import lab_data
from klab.lab_records import jsonable, seal, write_json_atomic, write_sealed
from klab.probes.suite import suite_identity

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = {
    "run_id": "lab-try-1",
    "checkpoint_sha256": "a" * 64,
    "step": 60,
    "tokens_seen": 7680,
    "parameters": 98_720,
    "active_parameters": 43_424,
    "eval_group": "g" * 64,
}
BASELINE = {**CANDIDATE, "run_id": "lab-base-1", "checkpoint_sha256": "b" * 64}
ITEM_GROUP = "i" * 64


def _row(probe: str, value: float, baseline: float | None, **details: Any) -> dict:
    return {
        "id": probe,
        "title": probe.replace("_", " "),
        "tier": "standard",
        "cost": 1,
        "hard": False,
        "status": "pass",
        "value": value,
        "baseline_value": baseline,
        "delta": None if baseline is None else value - baseline,
        "higher_is_better": probe != "heldout_loss",
        "thresholds": {"warn": 0.05, "fail": 0.15},
        "details": details,
    }


def _probe_result() -> dict[str, Any]:
    items = [
        {"prompt": "Q: Where does Mira live? A:", "answer": " Oslo", "picked": " Oslo", "credit": 1.0, "baseline_picked": " Lima", "baseline_credit": 0.0},
        {"prompt": "Q: What does Tam keep? A:", "answer": " a fox", "picked": " a cat", "credit": 0.0, "baseline_picked": " a cat", "baseline_credit": 0.0},
    ]  # fmt: skip
    bins = [
        {"low": 0.0, "high": 0.1, "confidence": 0.05, "accuracy": 0.04, "share": 0.6},
        {"low": 0.1, "high": 0.2, "confidence": 0.15, "accuracy": 0.2, "share": 0.4},
    ]
    return {
        "format": "klab-probe-v1",
        "created_at": "2026-10-10T19:00:00+00:00",
        "suite": {"sha256": suite_identity()["sha256"]},
        "tier": "standard",
        "tiers_run": ["fast", "standard"],
        "seconds": 3.2,
        "target": CANDIDATE,
        "baseline": BASELINE,
        "guard": {"overfit_suspected": False},
        "verdict": {
            "status": "warn",
            "action": "tweak",
            "next_tier": None,
            "missing": [],
            "suggestion": "Calibration slipped: tweak and retry.",
        },
        "probes": [
            _row(
                "heldout_loss",
                3.247,
                3.278,
                window_sums=[32.0, 33.0],
                window_counts=[10, 10],
            ),
            _row(
                "calibration", 0.296, 0.263, reliability=bins, baseline_reliability=bins
            ),
            _row(
                "repetition",
                0.69,
                0.966,
                distinct_1=0.2,
                distinct_2=0.03,
                samples=[{"prompt": "One day", "continuation": " the the bird"}],
                baseline_samples=[{"prompt": "One day", "continuation": " 1 1 1 1"}],
            ),
            _row(
                "fact_recall",
                0.5,
                0.0,
                chance=0.25,
                n=2,
                items=items,
                item_group=ITEM_GROUP,
            ),
            _row(
                "parametric_recall",
                0.5,
                0.25,
                chance=0.25,
                n=2,
                items=items,
                control_accuracy=0.25,
            ),
            _row(
                "needle",
                0.5,
                0.25,
                chance=1 / 6,
                n=6,
                item_group="n" * 64,
                by_length={
                    "0.5": {"tokens": 27, "accuracy": 0.5, "baseline_accuracy": 0.25}
                },
            ),
        ],
    }


@pytest.fixture(scope="module")
def lab(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dash-lab")
    probe = {**_probe_result(), "probe_id": "probe-1"}
    write_sealed(_mk(root / "probes/probe-1/probe.json"), probe)
    write_sealed(
        _mk(root / "tries/try-0/try.json"),
        {
            "format": "klab-lab-try-v1",
            "try_id": "try-0",
            "created_at": "2026-10-10T18:00:00+00:00",
            "question": "Do 4 experts beat the dense FFN?",
            "delta": {"model.ffn": {"base": "dense", "variant": "moe"}},
            "arms": {
                "baseline": {**BASELINE, "heldout": {"loss": 3.278}},
                "candidate": {**CANDIDATE, "heldout": {"loss": 3.247}},
            },
            "comparison": {
                "verdict": "CANDIDATE_LOWER_LOSS",
                "metric": "loss",
                "heldout_delta": -0.031,
            },
        },
    )
    exploration = explorer.explore_loaded(_loaded(**MOE_NGRAM), TEXT)
    path = explorer.cache_path(root, "c" * 64, TEXT, exploration["runtime"])
    path.parent.mkdir(parents=True)
    write_json_atomic(path, seal(jsonable(exploration)))
    run = root / "runs" / "plain-run"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text("{}")
    (run / "resolved_config.yaml").write_text("{}")
    return root


def _mk(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# --- Data views ----------------------------------------------------------------


def test_activity_and_next_steps_are_verdict_first(lab: Path) -> None:
    snap = lab_data.snapshot(lab)
    rows = lab_data.activity(snap)
    assert [r["id"] for r in rows] == ["probe-1", "try-0"]  # newest first
    assert rows[0]["verdict"] == "warn" and rows[0]["action"] == "tweak"
    assert rows[1]["outcome"] == "candidate has lower held-out loss"
    steps = lab_data.next_steps(snap)
    # "tweak" names no command (the verdict banner says it); the rest do.
    assert not any(s["title"].startswith("Latest verdict") for s in steps)
    assert all(s["why"] and s["command"].startswith("klab ") for s in steps)
    assert steps[-1]["command"] == "klab explore lab-try-1"
    with_runs = lab_data.next_steps(snap, [lab / "runs"])
    assert any(
        s["command"] == f"klab probe {lab / 'runs/plain-run'}" for s in with_runs
    )
    empty = lab_data.next_steps(lab_data.snapshot(lab / "nowhere"))
    assert empty[0]["command"].startswith("klab try")


def test_catalog_unscored_runs_and_behaviors(lab: Path) -> None:
    snap = lab_data.snapshot(lab)
    catalog = {r["checkpoint"]: r for r in lab_data.checkpoint_catalog(snap.points)}
    mine = catalog["a" * 12]
    assert mine["heldout_loss"] == pytest.approx(3.247)
    assert mine["fact_recall"] == pytest.approx(0.5)
    assert any(r["kind"] == "reference" for r in catalog.values())
    unscored = lab_data.unscored_runs([lab / "runs"], snap.points)
    assert [u["run_id"] for u in unscored] == ["plain-run"]
    assert unscored[0]["command"] == f"klab probe {lab / 'runs/plain-run'}"
    entry = snap.entries[0]
    assert lab_data.generations(entry) == [
        {"prompt": "One day", "candidate": " the the bird", "baseline": " 1 1 1 1"}
    ]
    misses = [i for i in lab_data.recall_items(entry, "fact_recall") if i["credit"] < 1]
    assert [m["answer"] for m in misses] == [" a fox"]
    curve = lab_data.needle_curves(snap.entries)
    assert curve[0]["tokens"] == 27 and curve[0]["accuracy"] == 0.5
    arms = {r["arm"] for r in lab_data.calibration_curves(entry)}
    assert arms == {"candidate", "baseline"}


def test_explorer_figures_follow_the_exploration(lab: Path) -> None:
    (exploration,) = lab_data.snapshot(lab).explorations
    arch = exploration["architecture"]
    page = xv.architecture_html(arch)
    assert "Mixture of experts" in page and "Memory · ngram" in page
    assert "shared KV head" in page  # grouped-query attention is called out
    layer = exploration["routing"]["0"]
    experts = arch["layers"][0]["ffn"]["experts"]
    figure = xv.routing_figure(layer, exploration["tokens"], experts, "r")
    gates = figure.data[0].z
    assert len(gates) == experts and len(gates[0]) == len(exploration["tokens"])
    assert sum(map(sum, gates)) == pytest.approx(
        sum(map(sum, layer["weights"])), rel=1e-6
    )
    assert sum(xv.expert_load(layer)) == pytest.approx(1.0)
    # head_summary on a pure previous-token head
    previous = [[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]]
    summary = xv.head_summary(previous)
    assert summary["mean_distance"] == pytest.approx(1.0)
    assert summary["mean_entropy"] == pytest.approx(0.0, abs=1e-9)
    rows = xv.memory_rows(exploration["memory"], exploration["tokens"])
    assert len(rows) == len(exploration["tokens"]) and "gate" in rows[0]
    assert set(xv.memory_reuse(exploration["memory"])) == {
        s["table"] for s in exploration["memory"]["streams"]
    }
    strip = xv.token_strip_html(exploration["tokens"])
    assert strip.count("lab-tok") == len(exploration["tokens"])


# --- Page renders ----------------------------------------------------------------


def _page(name: str, lab: str) -> None:
    from pathlib import Path

    from klab.dashboard import lab_pages

    root = Path(lab)
    runs = [root / "runs"]
    {
        "home": lambda: lab_pages.home_page(root, runs, {}),
        "experiments": lambda: lab_pages.experiments_page(root),
        "models": lambda: lab_pages.models_page(root, runs),
        "behaviors": lambda: lab_pages.behaviors_page(root),
        "explorer": lambda: lab_pages.explorer_page(root, runs),
    }[name]()


def _render(name: str, lab: Path) -> AppTest:
    test = AppTest.from_function(_page, args=(name, str(lab)), default_timeout=60)
    test.run()
    assert not test.exception, test.exception
    return test


def _text(test: AppTest) -> str:
    return "\n".join(
        [m.value for m in test.markdown]
        + [c.value for c in test.code]
        + [c.value for c in test.caption]
        + [i.value for i in test.info]
    )


@pytest.mark.parametrize(
    ("page", "command"),
    [
        ("home", "klab try DELTA.yaml --vs BASELINE.yaml"),
        ("experiments", "klab try DELTA.yaml --vs BASELINE.yaml"),
        ("models", "klab try DELTA.yaml --vs BASELINE.yaml"),
        ("behaviors", "klab probe RUN --config standard"),
        ("explorer", "klab explore RUN"),
    ],
)
def test_empty_pages_name_the_command_to_run(
    page: str, command: str, tmp_path: Path
) -> None:
    assert command in _text(_render(page, tmp_path))


def test_experiments_page_shows_results_trends_and_compare(lab: Path) -> None:
    test = _render("experiments", lab)
    assert [t.label for t in test.tabs] == [
        "Results",
        "History & trends",
        "Compare",
        "What the probes mean",
    ]
    assert len(test.get("plotly_chart")) >= 2  # comparison bars + trend
    text = _text(test)
    assert "fact recall" in text.lower()
    # Compare against the references: held-out loss is refused, never mixed.
    assert "NOT COMPARABLE" in text


def test_models_page_lists_catalog_references_and_unscored_runs(lab: Path) -> None:
    test = _render("models", lab)
    assert len(test.dataframe) >= 2  # catalog + references
    text = _text(test)
    assert f"klab probe {lab / 'runs/plain-run'}" in text


def test_behaviors_page_shows_generations_misses_and_reliability(lab: Path) -> None:
    test = _render("behaviors", lab)
    text = _text(test)
    assert "the the bird" in text and "1 1 1 1" in text
    assert "a fox" in text and "MISS" in text
    assert "never-trained" in text.lower() or any(
        "Never-trained" in m.label for m in test.metric
    )
    assert len(test.get("plotly_chart")) >= 2  # needle + reliability


@pytest.mark.parametrize(
    "runtime, reason",
    [
        (None, "missing actual-runtime provenance"),
        ({}, "malformed"),
        (
            {
                "engine": "pytorch",
                "backend": "rocm",
                "device_index": 0,
                "precision": "fp32",
            },
            "sealed payload/cache key mismatch",
        ),
    ],
)
def test_explorer_dashboard_labels_unverified_cache(
    lab: Path, runtime: Any, reason: str
) -> None:
    from klab.lab_records import seal, write_json_atomic

    entry = lab_data.snapshot(lab).explorations[0]
    path = Path(entry["path"])
    original = path.read_bytes()
    stored = explorer.read_cached(path)
    stored["runtime"] = runtime
    write_json_atomic(path, seal(stored))
    modified = path.read_bytes()
    try:
        test = _render("explorer", lab)
        assert any(reason in warning.value for warning in test.warning)
        assert "⚠ unverified" in test.selectbox(key="explorer_cached").options[0]
        assert path.read_bytes() == modified
    finally:
        path.write_bytes(original)


def test_explorer_picker_distinguishes_runtime_and_legacy_entries(
    lab: Path, tmp_path: Path
) -> None:
    from klab.lab_records import seal, write_json_atomic

    entry = lab_data.snapshot(lab).explorations[0]
    stored = explorer.read_cached(Path(entry["path"]))
    sha, text = stored["target"]["checkpoint_sha256"], stored["text"]
    # Isolate additional caches from the module-scoped dashboard fixture.
    lab = tmp_path
    cpu_path = explorer.cache_path(lab, sha, text, stored["runtime"])
    cpu_path.parent.mkdir(parents=True)
    write_json_atomic(cpu_path, stored)
    rocm = {**stored["runtime"], "backend": "rocm"}
    write_json_atomic(
        explorer.cache_path(lab, sha, text, rocm), seal({**stored, "runtime": rocm})
    )
    legacy = {k: v for k, v in stored.items() if k != "runtime"}
    write_json_atomic(explorer.cache_path(lab, sha, text), seal(legacy))
    test = _render("explorer", lab)
    options = test.selectbox(key="explorer_cached").options
    assert len(options) == 3
    assert any(
        "pytorch / cpu:0 / fp32" in label and "unverified" not in label
        for label in options
    )
    assert any(
        "pytorch / rocm:0 / fp32" in label and "unverified" not in label
        for label in options
    )
    assert any(
        "missing actual-runtime provenance" in label and "⚠ unverified" in label
        for label in options
    )
    assert any(
        "runtime shown per exploration" in caption.value for caption in test.caption
    )
    assert not any("computed on CPU" in caption.value for caption in test.caption)


def test_explorer_page_names_the_size_limit(lab: Path) -> None:
    exploration = lab_data.snapshot(lab).explorations[0]

    def app(payload: dict) -> None:
        from klab.dashboard.lab_pages import render_exploration

        render_exploration(payload)

    too_big = {
        "format": exploration["format"],
        "target": {"run_id": "big"},
        "architecture": exploration["architecture"],
        "unavailable": "120,000,000 parameters is above the explorer's limit",
    }
    test = AppTest.from_function(app, args=(too_big,), default_timeout=60)
    test.run()
    assert not test.exception
    assert "Too big to explore" in test.warning[0].value


def test_full_app_opens_on_lab_home(lab: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real entry point: navigation builds and the default page is Home."""
    monkeypatch.setattr(
        sys, "argv", ["app", "--runs-dir", str(lab / "runs"), "--lab-dir", str(lab)]
    )
    test = AppTest.from_file(
        str(ROOT / "src/klab/dashboard/app.py"), default_timeout=60
    )
    test.run()
    assert not test.exception
    assert [h.value for h in test.header] == ["Lab home"]
    assert any(str(lab) in c.value for c in test.sidebar.caption)


def _lm_result(probe_id: str, when: str, value: float, group: str) -> dict:
    tasks = {"piqa": {"acc": value, "items": [1.0, 0.0]}}
    lm = _row("lm_eval", value, None, tasks=tasks, benchmark_group=group)
    return {
        **_probe_result(),
        "probe_id": probe_id,
        "created_at": when,
        "probes": [_row("heldout_loss", 3.2, None), lm],
    }


def _values(data: Any) -> list[float]:
    """Plotly JSON arrays may be base64 typed arrays."""
    import base64

    import numpy as np

    if isinstance(data, dict):
        raw = base64.b64decode(data["bdata"])
        return [round(float(v), 6) for v in np.frombuffer(raw, dtype=data["dtype"])]
    return [round(float(v), 6) for v in data]


def test_trends_never_join_results_from_different_benchmark_groups(
    tmp_path: Path,
) -> None:
    """lm-eval with other tasks/limit is another benchmark group: its own trend."""
    import json

    default, custom = "d" * 64, "w" * 64
    for probe_id, when, value, group in (
        ("probe-11", "2026-10-10T10:00:00+00:00", 0.40, default),
        ("probe-12", "2026-10-10T11:00:00+00:00", 0.42, default),
        ("probe-13", "2026-10-10T12:00:00+00:00", 0.90, custom),
    ):
        path = _mk(tmp_path / f"probes/{probe_id}/probe.json")
        write_sealed(path, _lm_result(probe_id, when, value, group))
    snap = lab_data.snapshot(tmp_path)
    rows = [r for r in lab_data.probe_trends(snap.entries) if r["probe"] == "lm_eval"]
    by_id = {r["id"]: r["trend_group"] for r in rows}
    assert len(set(by_id.values())) == 2
    assert by_id["probe-11"] == by_id["probe-12"] != by_id["probe-13"]
    # Held-out loss of the same results shares the eval group.
    loss = [
        r for r in lab_data.probe_trends(snap.entries) if r["probe"] == "heldout_loss"
    ]
    assert len({r["trend_group"] for r in loss}) == 1

    def trend_values(test: AppTest) -> list[list[float]]:
        for chart in test.get("plotly_chart"):
            figure = json.loads(chart.proto.spec)
            title = figure["layout"].get("title", {}).get("text", "")
            if "per result" in title:
                return [
                    _values(trace["y"])
                    for trace in figure["data"]
                    if trace.get("name") == "candidate"
                ]
        raise AssertionError("no trend chart")

    test = _render("experiments", tmp_path)
    test.selectbox(key="experiments_trend_probe").set_value("lm_eval").run()
    assert not test.exception
    # Newest result's group by default: only the custom-task point is drawn.
    assert trend_values(test) == [[0.9]]
    assert "1 other comparison group(s) hidden" in _text(test)
    group = test.selectbox(key="experiments_trend_group_lm_eval")
    assert len(group.options) == 2
    assert group.index == 0
    group.select_index(1).run()
    assert trend_values(test) == [[0.4, 0.42]]


def _needle_result(
    probe_id: str, when: str, sha: str, group: str, tokens: int, acc: float, chance: Any
) -> dict:
    needle = _row(
        "needle",
        acc,
        None,
        chance=chance,
        item_group=group,
        by_length={
            "0.5": {"tokens": tokens, "accuracy": acc},
            "0.95": {"tokens": tokens * 2, "accuracy": acc / 2},
        },
    )
    return {
        **_probe_result(),
        "probe_id": probe_id,
        "created_at": when,
        "target": {**CANDIDATE, "checkpoint_sha256": sha},
        "probes": [_row("heldout_loss", 3.2, None), needle],
    }


def test_needle_chart_never_overlays_item_groups(tmp_path: Path) -> None:
    """Needle lengths are per tokenizer/context: one item group per chart, one
    curve per checkpoint (newest wins), chance from that group only."""
    import json

    small, big = "s" * 64, "b" * 64
    for args in (
        ("probe-21", "2026-10-10T10:00:00+00:00", "1" * 64, small, 27, 0.10, 1 / 6),
        # Same checkpoint and group, newer: replaces probe-21's curve.
        ("probe-22", "2026-10-10T11:00:00+00:00", "1" * 64, small, 27, 0.50, 1 / 6),
        ("probe-23", "2026-10-10T11:30:00+00:00", "2" * 64, small, 27, 0.25, 1 / 6),
        # Another tokenizer/context: other lengths, another item group.
        ("probe-24", "2026-10-10T12:00:00+00:00", "3" * 64, big, 120, 0.75, 0.2),
    ):
        path = _mk(tmp_path / f"probes/{args[0]}/probe.json")
        write_sealed(path, _needle_result(*args))
    snap = lab_data.snapshot(tmp_path)
    rows = lab_data.needle_curves(snap.entries)
    per = {(r["checkpoint_sha256"], r["group"]) for r in rows}
    assert per == {("1" * 64, small), ("2" * 64, small), ("3" * 64, big)}
    one = [r for r in rows if r["checkpoint_sha256"] == "1" * 64]
    assert [r["accuracy"] for r in one] == [0.5, 0.25]  # newest wins, 2 lengths

    def needle_figure(test: AppTest) -> dict:
        for chart in test.get("plotly_chart"):
            figure = json.loads(chart.proto.spec)
            if "Needle" in figure["layout"].get("title", {}).get("text", ""):
                return figure
        raise AssertionError("no needle chart")

    test = _render("behaviors", tmp_path)
    selector = test.selectbox(key="behaviors_needle_group")
    assert len(selector.options) == 2 and selector.index == 0
    # Newest result's group first: only the big-context checkpoint is drawn.
    figure = needle_figure(test)
    assert len(figure["data"]) == 1
    assert _values(figure["data"][0]["x"]) == [120, 240]
    (line,) = figure["layout"]["shapes"]
    assert line["y0"] == pytest.approx(0.2)
    assert "1 other item group(s) hidden" in _text(test)
    selector.select_index(1).run()
    figure = needle_figure(test)
    assert len(figure["data"]) == 2  # one curve per checkpoint
    assert all(_values(t["x"]) == [27, 54] for t in figure["data"])
    (line,) = figure["layout"]["shapes"]
    assert line["y0"] == pytest.approx(1 / 6)


def test_needle_chance_is_omitted_when_the_group_disagrees(tmp_path: Path) -> None:
    import json

    from klab.probes.points import consistent_chance

    assert consistent_chance([0.25, 0.25]) == pytest.approx(0.25)
    assert consistent_chance([0.25, 0.2]) is None
    assert consistent_chance([0.25, None]) is None
    assert consistent_chance([float("nan")]) is None
    assert consistent_chance([]) is None
    group = "s" * 64
    for args in (
        ("probe-31", "2026-10-10T10:00:00+00:00", "1" * 64, group, 27, 0.5, 0.25),
        ("probe-32", "2026-10-10T11:00:00+00:00", "2" * 64, group, 27, 0.5, None),
    ):
        write_sealed(
            _mk(tmp_path / f"probes/{args[0]}/probe.json"), _needle_result(*args)
        )
    test = _render("behaviors", tmp_path)
    for chart in test.get("plotly_chart"):
        figure = json.loads(chart.proto.spec)
        if "Needle" in figure["layout"].get("title", {}).get("text", ""):
            assert not figure["layout"].get("shapes")
            break
    else:
        raise AssertionError("no needle chart")


def test_setup_probes_do_not_replace_scored_dashboard_history(tmp_path: Path) -> None:
    from test_lab_results import _run

    from klab.dashboard.probe_data import load_history
    from klab.lab_records import is_setup_record, read_lab_record
    from klab.lab_results import ResultLookupError
    from klab.probes.cli import run_probe
    from klab.probes.render import render

    # Real failed lookup produces the setup shape; started/interrupted are
    # retained fixture states of the same attempt, without executing a model.
    run = _run(tmp_path)
    with pytest.raises(ResultLookupError):
        run_probe(str(run), "missing", lab_dir=tmp_path)
    (setup_path,) = (tmp_path / "probes").glob("*/probe.json")
    setup = read_lab_record(setup_path)[1]
    status = setup["status"] = "interrupted"
    setup_path.write_text(json.dumps(seal(setup)))
    original = setup_path.read_bytes()
    assert is_setup_record(setup)
    assert status in render(setup)

    scored = {**_probe_result(), "probe_id": "scored"}
    write_sealed(_mk(tmp_path / "probes/scored/probe.json"), scored)
    # A genuine incomplete battery has a suite and scoring evidence: preserve
    # it even though its status is interrupted.
    partial = {
        **_probe_result(),
        "probe_id": "partial",
        "status": "interrupted",
        "created_at": "2026-10-10T18:00:00+00:00",
        "stop": {"kind": "interrupted"},
    }
    write_sealed(_mk(tmp_path / "probes/partial/probe.json"), partial)
    assert not is_setup_record(partial)
    entries, rejected = load_history(tmp_path)
    assert [entry.key for entry in entries] == ["scored", "partial"]
    assert not rejected
    snap = lab_data.snapshot(tmp_path)
    assert [entry.key for entry in snap.entries] == ["scored", "partial"]
    assert [row["id"] for row in lab_data.activity(snap)] == ["scored", "partial"]
    assert snap.entries[0].result["verdict"] == scored["verdict"]
    assert all("None" not in step["command"] for step in lab_data.next_steps(snap))
    home = _text(_render("home", tmp_path))
    assert "WARN" in home and "Calibration slipped" in home
    experiments = _render("experiments", tmp_path)
    assert "fact recall" in _text(experiments).lower()
    assert experiments.selectbox[0].value == 0
    assert "scored" in experiments.selectbox[0].options[0]
    assert len(experiments.selectbox[0].options) == 2
    assert setup_path.read_bytes() == original


@pytest.mark.parametrize("page", ["home", "experiments"])
def test_setup_only_dashboard_keeps_empty_result_guidance(
    tmp_path: Path, page: str
) -> None:
    write_sealed(
        _mk(tmp_path / "probes/setup/probe.json"),
        {"format": "klab-probe-v1", "probe_id": "setup", "status": "started"},
    )
    snap = lab_data.snapshot(tmp_path)
    assert snap.empty and not snap.entries and not lab_data.activity(snap)
    assert "klab try DELTA.yaml --vs BASELINE.yaml" in _text(_render(page, tmp_path))


@pytest.mark.parametrize("action", ["rerun", "escalate"])
def test_followup_plan_preserves_sealed_policy_and_ui_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    import copy

    from klab.dashboard.probe_hints import followup_plan, render_followup
    from klab.probes import cli as probe_cli
    from klab.probes import config as probe_config

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("rendering must not load current declarations or execute probes")

    monkeypatch.setattr(probe_config, "load_probe_config", forbidden)
    monkeypatch.setattr(probe_cli, "run_probe", forbidden)
    request = {
        "probe_config_version": 1,
        "tier": "standard" if action == "escalate" else "full",
        "final": False,
        "fast_fail": False,
        "lm_eval_tasks": None if action == "escalate" else ["piqa", "winogrande"],
        "lm_eval_limit": None if action == "escalate" else 37,
        "lab_dir": "/original/lab",
        "runs_dir": "/original/runs",
        "backend_override": "cpu",
        "runtime_id": "saved-runtime",
        "runtime_profile_path": "/original/runtime.yaml",
        "resource_envelope_path": "/original/bounds.yaml",
        "runtime_profile": {"id": "saved-runtime", "backend": "cpu"},
        "resource_envelope": {"max_rss_bytes": 123456, "max_wall_seconds": 60},
        "baseline": "../old-base",
    }
    result = {
        **_probe_result(),
        "probe_id": "policy-probe",
        "tier": "fast",
        "request": request,
        "target": {**CANDIDATE, "checkpoint": "checkpoints/step-60"},
        "baseline": {**BASELINE, "checkpoint": "checkpoints/step-60"},
        "baseline_provenance": {
            "selector": "../old-base",
            "working_directory": "/original/cwd",
            "status": "resolved",
            "resolved": {"path": "/original/old-base"},
        },
        "verdict": {
            "status": "pass",
            "action": action,
            "next_tier": "full",
            "suggestion": "Promising: run `--config full`"
            if action == "escalate"
            else "Rerun the battery",
        },
    }
    if action == "rerun":
        result["probes"].append(
            _row(
                "lm_eval",
                0.5,
                None,
                benchmark_group="custom-benchmark",
                tasks={
                    name: {"acc": 0.5, "items": [{"acc": 1.0}]}
                    for name in request["lm_eval_tasks"]
                },
                benchmark={
                    "tasks": request["lm_eval_tasks"],
                    "items_sha256": {"piqa": "item-digest"},
                },
                limit=request["lm_eval_limit"],
            )
        )
    # Escalation must use the sealed declaration tier, not a display/result tier.
    path = _mk(tmp_path / "lab/probes/policy-probe/probe.json")
    sealed = write_sealed(path, result)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    reader = tmp_path / "moved/reader"
    reader.mkdir(parents=True)
    monkeypatch.chdir(reader)
    snap = lab_data.snapshot(tmp_path / "lab")
    original = copy.deepcopy(snap.entries[0].result)
    plan = followup_plan(snap.entries[0])
    assert plan["sealed_request"] == request
    assert plan["recorded_tier"] == request["tier"] != result["tier"]
    if action == "rerun":
        assert plan["benchmark_groups"] == [
            {
                "benchmark_group": "custom-benchmark",
                "tasks": ["piqa", "winogrande"],
                "benchmark": {
                    "tasks": ["piqa", "winogrande"],
                    "items_sha256": {"piqa": "item-digest"},
                },
                "limit": 37,
            }
        ]
    assert plan["checkpoint_bindings"]["target"]["checkpoint_sha256"] == "a" * 64
    assert plan["baseline_provenance"] == result["baseline_provenance"]
    assert plan["record_sha256"] == sealed["record_sha256"]
    assert plan["intended_changes"] == (
        {"tier": {"from": "standard", "to": "full"}} if action == "escalate" else {}
    )
    assert not plan["missing_fields"]
    assert snap.entries[0].result == original
    step = lab_data.next_steps(snap)[0]
    assert step["command"] == ""
    assert step["plan"] == render_followup(plan)
    assert step["why"].startswith("Recorded verdict text:")
    assert "bare --config tier does not retain" in step["why"]
    app = _render("home", tmp_path / "lab")
    assert not app.exception
    assert step["plan"] in [code.value for code in app.code]
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_historical_followup_is_explicit_about_missing_policy(tmp_path: Path) -> None:
    from klab.dashboard.probe_data import ProbeEntry
    from klab.dashboard.probe_hints import followup_plan

    result = _probe_result()
    result["verdict"]["action"] = "rerun"
    entry = ProbeEntry(
        "old", "probe", "old", tmp_path / "probe.json", result, None, {}, None
    )
    plan = followup_plan(entry)
    assert "fast_fail" in plan["missing_fields"]
    assert "runtime_profile" in plan["missing_fields"]
    assert "target.checkpoint" in plan["missing_fields"]
    assert any("historical" in field for field in plan["missing_fields"])
    assert plan["sealed_request"] == {}
    assert "current defaults are not historical evidence" in plan["preparation"]
    assert plan["intended_changes"] == {}


def test_embedded_try_followup_uses_owner_policy(tmp_path: Path) -> None:
    from klab.dashboard.probe_data import ProbeEntry
    from klab.dashboard.probe_hints import followup_plan

    result = _probe_result()
    result["verdict"].update(action="escalate", next_tier="full")
    owner = seal(
        {
            "request": {
                "probe_tier": "standard",
                "runtime_profile": {"id": "try-runtime"},
                "resource_envelope": {"max_rss_bytes": 999},
            },
            "baseline_provenance": {"status": "resolved", "selector": "base.yaml"},
        }
    )
    entry = ProbeEntry(
        "try", "try", "now", tmp_path / "try.json", result, None, {}, None
    )
    plan = followup_plan(entry, owner)
    assert plan["sealed_request"] == owner["request"]
    assert plan["record_sha256"] == owner["record_sha256"]
    assert plan["intended_changes"] == {"tier": {"from": "standard", "to": "full"}}
    assert "fast_fail" not in plan["missing_fields"]
    assert "fast_fail" in plan["not_in_try_schema"]
    assert "review embedded battery settings separately" in plan["preparation"]


def test_not_comparable_try_next_step_requires_repaired_new_try(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.dashboard.probe_hints import render_followup
    from klab.try_config import TryConfig

    probe = _probe_result()
    probe["verdict"].update(action="escalate", next_tier="full")
    failed = ["validation", "tokenizer"]
    request = {
        **TryConfig().model_dump(mode="json"),
        "baseline": "../baseline.yaml",
        "backend_override": "cpu",
        "runtime_id": None,
        "runtime_profile_path": None,
        "resource_envelope_path": None,
    }
    record = {
        "format": "klab-lab-try-v1",
        "try_id": "incomparable",
        "created_at": probe["created_at"],
        "probe": probe,
        "request": request,
        "inputs": {
            "baseline": {"path": "../baseline.yaml"},
            "candidate": {"path": "delta.yaml"},
        },
        "delta": {"dataset.synthetic_seed": {"base": 0, "variant": 1}},
        "arms": {"baseline": BASELINE, "candidate": CANDIDATE},
        "comparison": {"verdict": "NOT_COMPARABLE", "failed": failed},
        "baseline_provenance": {
            "selector": "../baseline.yaml",
            "working_directory": "/original/cwd",
            "status": "resolved",
        },
    }
    path = _mk(tmp_path / "lab/tries/incomparable/try.json")
    sealed = write_sealed(path, record)
    before = path.read_bytes()
    monkeypatch.chdir(tmp_path)
    snap = lab_data.snapshot(tmp_path / "lab")
    step = lab_data.next_steps(snap)[0]
    plan = json.loads(step["plan"])
    assert "rerun" in step["title"] and "Fix the cause" in step["why"]
    assert plan["operation"] == "try"
    assert plan["comparison_failed"] == failed
    assert plan["sealed_request"] == request
    assert (
        plan["try_inputs"] == record["inputs"] and plan["try_delta"] == record["delta"]
    )
    assert plan["record_sha256"] == sealed["record_sha256"]
    assert plan["baseline_provenance"] == record["baseline_provenance"]
    assert plan["checkpoint_bindings"]["target"]["checkpoint_sha256"] == "a" * 64
    assert plan["intended_changes"]["comparison"]
    assert set(plan["not_in_try_schema"]) == {
        "final",
        "fast_fail",
        "lm_eval_tasks",
        "lm_eval_limit",
        "runs_dir",
    }
    assert not set(plan["not_in_try_schema"]) & set(plan["missing_fields"])
    assert "klab try" in plan["preparation"]
    assert (
        "ProbeConfig" not in plan["preparation"]
        and "klab probe" not in plan["preparation"]
    )
    assert "rerun changes nothing" not in plan["preparation"]
    assert "unchanged rerun cannot fix comparability" in plan["preparation"]
    assert step["command"] == "" and step["plan"] == render_followup(plan)
    app = _render("home", tmp_path / "lab")
    assert not app.exception and step["plan"] in [code.value for code in app.code]
    assert path.read_bytes() == before


def test_explorer_cached_tabs_and_compare_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    from klab.dashboard import lab_pages

    lab_dir = tmp_path / "lab"
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "moe-run"
    gen_a = run_dir / "checkpoints" / "step_00000004_gen_aaaa"
    gen_b = run_dir / "checkpoints" / "step_00000008_gen_bbbb"
    gen_a.mkdir(parents=True)
    gen_b.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}")
    (run_dir / "resolved_config.yaml").write_text("{}")
    (gen_a / "manifest.json").write_text(json.dumps({"step": 4}))
    (gen_b / "manifest.json").write_text(json.dumps({"step": 8}))
    # A mutable latest.json pointer in checkpoints must never appear as a selector.
    (run_dir / "checkpoints" / "latest.json").write_text(
        json.dumps({"relative_path": "step_00000008_gen_bbbb"})
    )

    # Write a cached exploration for the first generation.
    loaded_a = _loaded(**MOE_NGRAM)
    loaded_a.identity.update(
        run_id="moe-run",
        checkpoint_relative_path="checkpoints/step_00000004_gen_aaaa",
        checkpoint_sha256="1" * 64,
        step=4,
    )
    v2_a = jsonable(explorer.explore_loaded(loaded_a, TEXT))
    cached_path = explorer.cache_path(lab_dir, "1" * 64, TEXT, v2_a["runtime"])
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(cached_path, seal(v2_a))

    loaded_b = _loaded(**MOE_NGRAM)
    loaded_b.identity.update(
        run_id="moe-run",
        checkpoint_relative_path="checkpoints/step_00000008_gen_bbbb",
        checkpoint_sha256="2" * 64,
        step=8,
    )
    with torch.no_grad():
        for param in loaded_b.model.parameters():
            param.mul_(1.2)
    v2_b = jsonable(explorer.explore_loaded(loaded_b, TEXT))

    by_spec = {
        str(gen_a.resolve()): seal(v2_a),
        str(gen_b.resolve()): seal(v2_b),
    }

    calls = []

    def fake_explore(
        target: str, *, lab_dir: Path, text: str = TEXT, **kwargs: Any
    ) -> tuple[dict[str, Any], Path | None]:
        calls.append(target)
        resolved = str(Path(target).resolve())
        if resolved not in by_spec:
            raise FileNotFoundError(f"missing generation at {target}")
        rec = seal({**by_spec[resolved], "text": text})
        return rec, None

    monkeypatch.setattr(explorer, "explore", fake_explore)

    def run_explorer(lab_root: str, runs_root: str) -> None:
        from pathlib import Path

        from klab.dashboard import lab_pages

        lab_pages.explorer_page(Path(lab_root), [Path(runs_root)])

    # 1. The cached record is listed and opens with every diagnostic tab.
    test = AppTest.from_function(
        run_explorer, args=(str(lab_dir), str(runs_dir)), default_timeout=60
    )
    test.run()
    assert not test.exception
    assert len(test.selectbox(key="explorer_cached").options) == 1
    assert [t.label for t in test.tabs] == [
        "Architecture",
        "Tokens",
        "Attention",
        "Weights",
        "Activations",
        "Expert routing",
        "Memory lookups",
    ]

    # Verify selectors list both generations and never latest.json.
    snap = lab_data.snapshot(lab_dir)
    choices = lab_pages._explorable_runs(snap, [runs_dir])
    assert [c["spec"] for c in choices] == [
        str(gen_a.resolve()),
        str(gen_b.resolve()),
    ]
    assert all("latest" not in c["label"] for c in choices)
    assert "step 4" in choices[0]["label"] and "step 8" in choices[1]["label"]

    # Single results must disappear when either provenance selector changes.
    test.radio(key="explorer_source").set_value("Explore a run").run()
    test.text_input(key="explorer_text").set_value(TEXT).run()
    test.selectbox(key="explorer_run").select_index(0).run()
    test.button(key="explorer_go").click().run()
    assert not test.exception and test.get("plotly_chart")
    assert "Activations" in [tab.label for tab in test.tabs]
    test.selectbox(key="explorer_run").select_index(1).run()
    assert not test.exception and not test.get("plotly_chart")
    test.button(key="explorer_go").click().run()
    assert not test.exception and test.get("plotly_chart")
    test.text_input(key="explorer_text").set_value("once upon").run()
    assert not test.exception and not test.get("plotly_chart")
    # An old unbound session payload must not bypass those guards.
    test.session_state["explorer_last"] = seal(v2_a)
    test.run()
    assert not test.exception and not test.get("plotly_chart")

    # 2. Switch to "Compare checkpoints", select the two generations, and click Compare.
    test.radio(key="explorer_source").set_value("Compare checkpoints").run()
    assert not test.exception
    left_sel = test.selectbox(key="explorer_compare_left")
    right_sel = test.selectbox(key="explorer_compare_right")
    assert len(left_sel.options) == 2
    left_sel.select_index(0).run()
    right_sel.select_index(1).run()
    test.text_input(key="explorer_compare_text").set_value(TEXT).run()
    test.button(key="explorer_compare_go").click().run()
    assert not test.exception, test.exception

    page_text = _text(test)
    assert "111111111111" in page_text and "222222222222" in page_text
    assert "step `4`" in page_text and "step `8`" in page_text
    assert "Diagnostics runtime: pytorch / cpu:0 / fp32" in page_text
    assert "Matched layers: [0, 1]" in page_text
    charts = [json.loads(c.proto.spec) for c in test.get("plotly_chart")]
    titles = [c["layout"].get("title", {}).get("text", "") for c in charts]
    assert any("Left: moe-run · step 4" in t for t in titles)
    assert any("Right: moe-run · step 8" in t for t in titles)
    assert any("Signed Δ RMS (right − left)" in t for t in titles)
    assert any("Left · Layer 0 routing" in t for t in titles)
    assert any("Right · Layer 0 routing" in t for t in titles)
    # Verify shared RMS scale on paired heatmaps and centered zero on delta heatmap.
    left_heat = next(
        c
        for c in charts
        if "Left: moe-run · step 4" in c["layout"].get("title", {}).get("text", "")
    )
    right_heat = next(
        c
        for c in charts
        if "Right: moe-run · step 8" in c["layout"].get("title", {}).get("text", "")
    )
    delta_heat = next(
        c
        for c in charts
        if "Signed Δ RMS" in c["layout"].get("title", {}).get("text", "")
    )
    assert left_heat["data"][0]["zmin"] == right_heat["data"][0]["zmin"] == 0.0
    assert left_heat["data"][0]["zmax"] == pytest.approx(right_heat["data"][0]["zmax"])
    assert delta_heat["data"][0]["zmid"] == 0.0
    assert delta_heat["data"][0]["zmin"] == pytest.approx(
        -delta_heat["data"][0]["zmax"]
    )

    # 3. Changing a selector or text clears the stale comparison immediately.
    test.text_input(key="explorer_compare_text").set_value("once upon").run()
    assert not test.exception
    assert "Signed Δ RMS" not in "\n".join(
        json.loads(c.proto.spec)["layout"].get("title", {}).get("text", "")
        for c in test.get("plotly_chart")
    )
    assert "Pick two distinct pinned checkpoints" in _text(test)

    # Same selector refuses before either forward pass.
    test.selectbox(key="explorer_compare_right").select_index(0).run()
    count = len(calls)
    test.button(key="explorer_compare_go").click().run()
    assert not test.exception
    assert any(
        "Pick two distinct checkpoints" in warning.value for warning in test.warning
    )
    assert len(calls) == count

    # Reestablish a successful comparison immediately before a failed click.
    test.selectbox(key="explorer_compare_right").select_index(1).run()
    test.text_input(key="explorer_compare_text").set_value(TEXT).run()
    test.button(key="explorer_compare_go").click().run()
    assert not test.exception and test.get("plotly_chart")
    # 4. A generation becomes unavailable while its selector is still displayed.
    del by_spec[str(gen_b.resolve())]
    test.button(key="explorer_compare_go").click().run()
    assert not test.exception
    assert any("missing generation" in e.value for e in test.error)
    assert len(test.get("plotly_chart")) == 0
    test.run()
    assert not test.exception and not test.get("plotly_chart")


def test_explorer_selectors_include_sealed_records_and_deduplicate_roots(tmp_path):
    from klab.dashboard.lab_pages import _explorable_runs

    lab = tmp_path / "lab"
    root = lab / "runs"
    generation = root / "local" / "checkpoints" / "step_00000003_gen_000001"
    generation.mkdir(parents=True)
    (generation / "manifest.json").write_text('{"step":3}')
    missing = generation.parent / "step_00000004_gen_000002"
    missing.mkdir()
    (generation.parent / "latest.json").write_text("{}")
    expected = [
        {
            "spec": str(generation.resolve()),
            "label": "local · step_00000003_gen_000001 · step 3",
        }
    ]
    for kind, folder, fmt, target_key in (
        ("try", "tries", "klab-lab-try-v1", "arms"),
        ("probe", "probes", "klab-probe-v1", "target"),
    ):
        for suffix, extra in (
            ("local", {}),
            ("reference", {"reference": True}),
            ("ref-id", {"run_id": "ref:fixture"}),
        ):
            identifier = f"{kind}-{suffix}"
            target = {
                "run_id": "local",
                "step": 3,
                "checkpoint": str(generation),
                **extra,
            }
            body = {
                "format": fmt,
                f"{kind}_id": identifier,
                "created_at": "2026-10-16T00:00:00Z",
                target_key: {"candidate": target} if kind == "try" else target,
            }
            path = lab / folder / identifier / f"{kind}.json"
            path.parent.mkdir(parents=True)
            write_sealed(path, body)
            if suffix == "local":
                expected.append(
                    {
                        "spec": str(path.resolve()),
                        "label": f"local · {kind} {identifier} ({generation.name}) · step 3",
                    }
                )
    snap = lab_data.snapshot(lab)
    assert not snap.rejected
    # LAB/runs alone must work; the same root twice must not duplicate selectors.
    assert _explorable_runs(snap, []) == expected
    assert _explorable_runs(snap, [root, root]) == expected
