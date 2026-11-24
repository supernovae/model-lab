from __future__ import annotations

from datetime import UTC, datetime, timedelta

from klab.dashboard import ui
from klab.dashboard.lineage import lineages, stitch
from klab.dashboard.queries import metric_series, run_facts, runs
from klab.training.metrics import ExperimentStore

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)


def _train(store: ExperimentStore, run_id: str, first: int, last: int, rate: float):
    for step in range(first, last + 1):
        store.log_metrics(
            run_id,
            step,
            step * 100,
            float(step),
            {"train/loss": 5.0 - step / 100, "performance/tokens_per_second": rate},
        )


def _heartbeat(store: ExperimentStore, run_id: str, step: int, age: timedelta) -> None:
    store.upsert_runtime_progress_snapshot(
        run_id,
        step,
        step * 100,
        float(step),
        {"event": "heartbeat", "observed_at_utc": (NOW - age).isoformat()},
    )


def test_runs_are_grouped_into_a_lineage_with_a_verdict_per_attempt(tmp_path) -> None:
    store = ExperimentStore(tmp_path)
    config = {"name": "job", "training": {"max_tokens": 10_000}}
    store.create_run("job-a0", config, {})
    _train(store, "job-a0", 0, 20, 1000.0)
    store.log_event("job-a0", 20, 2000, 20.0, "run_interrupted", {"reason": "signal"})
    store.log_event("job-a0", 20, 2000, 20.0, "resumable", {"checkpoint": "c20"})
    store.finish_run("job-a0", "interrupted")

    store.create_run("job-a1", config, {}, parent_run_id="job-a0")
    _train(store, "job-a1", 20, 40, 600.0)
    reason = "device memory spill: host available memory fell"
    store.log_event("job-a1", 40, 4000, 40.0, "run_failed", {"reason": reason})
    store.finish_run("job-a1", "failed")

    # Recovered from a checkpoint at step 30: steps 31 to 40 of a1 are redone.
    store.create_run("job-a2", config, {}, parent_run_id="job-a1")
    _train(store, "job-a2", 30, 50, 1000.0)
    _heartbeat(store, "job-a2", 50, timedelta(hours=3))
    store.create_run("other", {"name": "other"}, {})

    groups = lineages(runs(tmp_path), run_facts(tmp_path), NOW)
    job = next(group for group in groups if group.root_id == "job-a0")

    assert {group.root_id for group in groups} == {"job-a0", "other"}
    assert [(a.run_id, a.verdict.level, a.verdict.label) for a in job.attempts] == [
        ("job-a0", "pass", "Paused cleanly"),
        ("job-a1", "fail", "Memory spill"),
        # A stored `running` status without a fresh heartbeat is never called live.
        ("job-a2", "warn", "Not verified"),
    ]
    assert not job.live
    assert any(note.startswith("Slower") for note in job.attempts[1].notes)
    assert job.tip.run_id == "job-a2" and job.progress == 0.5

    rows = stitch(
        metric_series(tmp_path, ("job-a0", "job-a1", "job-a2"), ("train/loss",)),
        job.main_line,
    )
    steps = [(row["run_id"], row["step"]) for row in rows]
    # a0 gives 0-20, a1 gives 20-30, a2 gives 30-50.
    assert len(steps) == 21 + 11 + 21
    assert ("job-a1", 31) not in steps and ("job-a2", 31) in steps

    _heartbeat(store, "job-a2", 50, timedelta(seconds=20))
    live = lineages(runs(tmp_path), run_facts(tmp_path), NOW)
    assert live[0].root_id == "job-a0" and live[0].live
    assert live[0].tip.verdict.label == "Training"


def _run_page_script(root):
    from pathlib import Path

    from klab.dashboard.app import _run_page, _selection_controls, run_tabs

    _selection_controls(Path(root))
    _run_page(Path(root), run_tabs())


def test_run_page_opens_the_linked_tab_with_the_attempt_outcome(tmp_path) -> None:
    from streamlit.testing.v1 import AppTest

    store = ExperimentStore(tmp_path)
    store.create_run("job-a0", {"name": "job"}, {})
    _train(store, "job-a0", 0, 8, 1000.0)
    store.log_event("job-a0", 8, 800, 8.0, "run_failed", {"reason": "nonfinite loss"})
    store.finish_run("job-a0", "failed")
    store.create_run("job-a1", {"name": "job"}, {}, parent_run_id="job-a0")

    app = AppTest.from_function(_run_page_script, args=(str(tmp_path),))
    app.query_params["run"] = "job-a0"
    app.query_params["tab"] = "stages"
    app.run()

    assert not app.exception
    assert [h.value for h in app.header] == ["Stages"]
    header = next(m.value for m in app.markdown if "rh-run" in m.value)
    assert ui.STATUS_COLOR["fail"] in header  # the failed attempt's outcome chip
    assert "run=job-a1" in header and "tab=stages" in header

    app.button_group[0].set_value("checkpoints").run()
    assert [h.value for h in app.header] == ["Checkpoints"]
    assert app.query_params["tab"] == ["checkpoints"]


def test_engram_tables_follow_the_config_and_readings_never_invent_values() -> None:
    from klab.dashboard.engram_view import readings, scopes, tables

    model = {
        "memory": "ngram",
        "memory_dim": 8,
        "memory_injection": "after_block",
        "memory_block_indices": [1, 14],
        "memory_ngram_orders": [2, 3],
        "memory_hash_heads": 2,
        "memory_table_sizes": [11, 13, 17, 19, 23, 29, 31, 37],
    }
    layout = tables({"model": model})
    assert [(t.site, t.order, t.head, t.rows) for t in layout][4:6] == [
        ("after block 14", 2, 0, 23),
        ("after block 14", 2, 1, 29),
    ]
    assert layout[0].parameters == 88
    # Sizes that do not match sites x orders x heads are listed, never labelled.
    short = tables({"model": {**model, "memory_table_sizes": [11, 13, 17]}})
    assert [(t.site, t.order, t.rows) for t in short] == [
        ("unlabelled", None, 11),
        ("unlabelled", None, 13),
        ("unlabelled", None, 17),
    ]
    assert tables({"model": {"memory": None}}) == []

    series = {
        "engram/gate_mean": [(1, 0.9)],
        "engram/port/1/gate_mean": [(1, 0.2), (5, 0.1)],
        "engram/port/1/value_norm": [(1, 4.0), (5, 5.0)],
        "engram/port/1/hidden_norm": [(1, 100.0), (5, 50.0)],
        "engram/port/14/gate_mean": [(5, 0.05)],
        "engram/port/14/value_norm": [(5, 3.0)],
        "engram/port/14/hidden_norm": [(3, 200.0), (5, None)],
    }
    assert [prefix for _, prefix in scopes(set(series))] == [
        "engram/port/1/",
        "engram/port/14/",
    ]
    site = readings(series, "engram/port/1/")
    assert (site["gate_mean"].value, site["gate_mean"].first) == (0.1, 0.2)
    assert site["push"].value == 0.1 * 5.0 / 50.0
    # Site 14 has no finite hidden norm at step 5: its push is unavailable, not
    # computed from the older value and not zero.
    assert readings(series, "engram/port/14/")["push"].value is None
