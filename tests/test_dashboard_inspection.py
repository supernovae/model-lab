from dataclasses import replace

from streamlit.testing.v1 import AppTest

from sparselab.dashboard.inspection import latest_metrics
from sparselab.dashboard.queries import snapshot
from sparselab.training.metric_registry import metric_spec
from sparselab.training.metrics import ExperimentStore


def _store(root):
    store = ExperimentStore(root)
    for run in ("memory", "dense"):
        store.create_run(run, {"name": run}, {"purpose": "training"})
        store.log_metrics(run, 1, 10, 1.0, {"train/loss": 3.0})
    store.log_metrics("memory", 2, 20, 2.0, {"engram/port/1/gate_mean": 0.5})
    store.log_metrics("memory", 4, 40, 4.0, {"engram/port/1/gate_mean": 0.6})
    return store


def test_selected_snapshot_and_observation_deltas(tmp_path):
    _store(tmp_path)
    view = snapshot(tmp_path, ("memory",))
    assert {row["run_id"] for row in view.metrics} == {"memory"}
    assert {run.run_id for run in view.runs} == {"memory", "dense"}
    assert snapshot(tmp_path, ()).metrics == ()
    rows = latest_metrics(view, ["memory"])
    gate = next(row for row in rows if row["metric"] == "engram/port/1/gate_mean")
    assert gate["step"] == 4 and gate["previous_step"] == 2
    assert round(gate["change"], 2) == 0.1
    assert gate["unit"] == metric_spec("engram/gate_mean").unit
    assert next(row for row in rows if row["metric"] == "train/loss")["step"] == 1
    bad = replace(view, metrics=({**view.metrics[-1], "value": float("inf")},))
    assert latest_metrics(bad, ["memory"])[0]["value"] is None
    assert latest_metrics(bad, ["memory"])[0]["availability"] != "recorded"


def _inspector(root):
    from pathlib import Path

    from sparselab.dashboard.inspection import engram_summary, snapshot_inspector
    from sparselab.dashboard.queries import snapshot

    view = snapshot(Path(root))
    engram_summary(view, ["memory", "dense"])
    snapshot_inspector(view, ["memory", "dense"])


def test_inspector_missing_values_filter_navigation_and_refresh(tmp_path):
    store = _store(tmp_path)
    app = AppTest.from_function(_inspector, args=(str(tmp_path),)).run()
    assert not app.exception
    assert any("no persisted Engram" in item.value for item in app.info)
    app.text_input(key="inspect_metric_filter").set_value("port/1").run()
    assert len(app.dataframe[-1].value) == 1
    store.log_metrics("memory", 5, 50, 5.0, {"engram/port/1/gate_mean": 0.7})
    app.run()
    assert app.dataframe[-1].value.iloc[0]["step"] == 5
    app.radio(key="inspect_section").set_value("Checkpoints").run()
    assert not app.exception
    assert any("No checkpoint metadata" in item.value for item in app.info)
    app.radio(key="inspect_section").set_value("Configuration").run()
    assert not app.exception
    app.radio(key="inspect_section").set_value("Recent events").run()
    assert not app.exception


def test_failed_refresh_retains_only_same_selection(tmp_path, monkeypatch):
    import sqlite3
    from types import SimpleNamespace

    from sparselab.dashboard import app

    _store(tmp_path)
    state = {"selected_runs": ["memory"]}
    monkeypatch.setattr(app, "st", SimpleNamespace(session_state=state))
    good, observed, error = app._read_snapshot(tmp_path)
    assert error is None

    def unavailable(*args):
        raise sqlite3.OperationalError("locked")

    monkeypatch.setattr(app, "snapshot", unavailable)
    stale, stale_time, error = app._read_snapshot(tmp_path)
    assert stale is good and stale_time == observed and error == "locked"
    state["selected_runs"] = ["dense"]
    missing, _, error = app._read_snapshot(tmp_path)
    assert missing is None and error == "locked"


def _telemetry(root):
    from pathlib import Path

    import streamlit as st

    from sparselab.dashboard.app import _run_page, _selection_controls

    _selection_controls(Path(root))
    st.selectbox("Refresh", [10, 0], key="refresh_seconds")
    page = st.radio("Page", ["engram", "snapshots"])
    _run_page(Path(root), page)


def test_telemetry_manual_refresh_and_selection_survive_page_change(tmp_path):
    store = _store(tmp_path)
    app = AppTest.from_function(_telemetry, args=(str(tmp_path),)).run()
    app.multiselect(key="selected_runs").set_value(["memory"]).run()
    app.selectbox(key="refresh_seconds").set_value(0).run()
    assert not app.exception
    assert any("paused" in item.value for item in app.caption)
    store.log_metrics("memory", 6, 60, 6.0, {"engram/port/1/gate_mean": 0.8})
    app.button(key="dashboard_refresh").click().run()
    assert not app.exception
    assert 6 in app.dataframe[0].value["step"].values
    app.radio[0].set_value("snapshots").run()
    assert not app.exception
    assert app.multiselect(key="selected_runs").value == ["memory"]
    assert app.selectbox(key="refresh_seconds").value == 0
