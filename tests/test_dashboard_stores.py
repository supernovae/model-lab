import argparse

import pytest
from streamlit.testing.v1 import AppTest

from sparselab.dashboard.stores import PRIMARY_STORE, configured_stores, parse_run_store
from sparselab.training.metrics import ExperimentStore


def _stores_page(primary, pilot):
    from pathlib import Path

    from sparselab.dashboard.app import _run_page, _selection_controls, _store_controls

    root = _store_controls(Path(primary), [("Pilot", pilot)])
    _selection_controls(root)
    _run_page(root, "snapshots")


def test_switching_stores_isolates_duplicate_run_ids_and_restores_selection(tmp_path):
    primary, pilot = tmp_path / "history", tmp_path / "pilot"
    for root, loss in [(primary, 3.0), (pilot, 1.0)]:
        store = ExperimentStore(root)
        store.create_run("same-id", {"name": "same-id"}, {"purpose": "training"})
        store.log_metrics("same-id", 2, 20, 2.0, {"train/loss": loss})
    app = AppTest.from_function(_stores_page, args=(str(primary), str(pilot))).run()
    assert not app.exception
    assert app.dataframe[-1].value.iloc[0]["value"] == 3.0
    app.selectbox(key="dashboard_run_store").set_value("Pilot").run()
    assert not app.exception
    assert app.dataframe[-1].value.iloc[0]["value"] == 1.0
    app.multiselect(key="selected_runs").set_value([]).run()
    app.selectbox(key="dashboard_run_store").set_value(PRIMARY_STORE).run()
    assert app.multiselect(key="selected_runs").value == ["same-id"]
    assert app.dataframe[-1].value.iloc[0]["value"] == 3.0
    app.selectbox(key="dashboard_run_store").set_value("Pilot").run()
    assert app.multiselect(key="selected_runs").value == []
    assert not app.dataframe
    assert not app.exception


def test_missing_store_does_not_show_previous_stores_data(tmp_path):
    primary, missing = tmp_path / "history", tmp_path / "missing"
    store = ExperimentStore(primary)
    store.create_run("run", {"name": "run"}, {"purpose": "training"})
    store.log_metrics("run", 1, 10, 1.0, {"train/loss": 2.0})
    app = AppTest.from_function(_stores_page, args=(str(primary), str(missing))).run()
    assert app.dataframe
    app.selectbox(key="dashboard_run_store").set_value("Pilot").run()
    assert not app.exception
    assert not app.dataframe
    assert not missing.exists()


def test_single_store_and_invalid_store_configuration(tmp_path):
    assert configured_stores(tmp_path, []) == {PRIMARY_STORE: tmp_path}
    assert parse_run_store("Pilot=/data/path=version") == (
        "Pilot",
        "/data/path=version",
    )
    for value in (
        "missing-separator",
        "=empty-label",
        "Pilot=",
        f"{PRIMARY_STORE}=/data",
    ):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_run_store(value)
    with pytest.raises(ValueError, match="Duplicate"):
        configured_stores(tmp_path, [("Pilot", "/a"), ("Pilot", "/b")])
    with pytest.raises(ValueError, match="repeats"):
        configured_stores(tmp_path, [("Pilot", str(tmp_path))])


def test_native_dashboard_forwards_named_stores_without_opening_them(
    tmp_path, monkeypatch
):
    from sparselab.cli import main as cli

    parser = cli.build_parser(work_dir=tmp_path)
    primary, pilot = tmp_path / "history", tmp_path / "pilot"
    args = parser.parse_args(
        [
            "dashboard",
            "--runs-dir",
            str(primary),
            "--run-store",
            f"Pilot={pilot}",
            "--lab-dir",
            str(tmp_path / "lab"),
        ]
    )
    captured = []
    monkeypatch.setattr(
        cli.os, "execv", lambda executable, command: captured.append(command)
    )
    cli._dashboard(args)
    command = captured[0]
    assert command[command.index("--runs-dir") + 1] == str(primary)
    assert command[command.index("--run-store") + 1] == f"Pilot={pilot}"
    assert not primary.exists() and not pilot.exists()
    single = parser.parse_args(["dashboard"])
    assert single.run_store == []


def test_deep_link_selects_exact_store_and_run_and_missing_link_selects_nothing(
    tmp_path,
):
    primary, pilot = tmp_path / "history", tmp_path / "pilot"
    for root in (primary, pilot):
        store = ExperimentStore(root)
        store.create_run("a", {"name": "Model A"}, {"purpose": "training"})
        store.create_run("b", {"name": "Model B"}, {"purpose": "training"})
    app = AppTest.from_function(_stores_page, args=(str(primary), str(pilot)))
    app.query_params.update(store=str(pilot), run="b")
    app.run()
    assert not app.exception
    assert app.selectbox(key="dashboard_run_store").value == "Pilot"
    assert app.multiselect(key="selected_runs").value == ["b"]
    app.multiselect(key="selected_runs").set_value(["a"]).run()
    assert app.multiselect(key="selected_runs").value == ["a"]
    app.query_params["run"] = "missing"
    app.run()
    assert app.multiselect(key="selected_runs").value == []
    assert not app.dataframe
    app.query_params["store"] = str(tmp_path / "not-configured")
    app.run()
    assert app.error
    assert not app.dataframe
    assert not app.exception


def test_empty_and_completed_store_do_not_offer_active_focus(tmp_path):
    import sqlite3

    store = ExperimentStore(tmp_path)
    store.create_run("done", {"name": "Done"}, {"purpose": "training"})
    with sqlite3.connect(tmp_path / "experiments.sqlite3") as db:
        db.execute("UPDATE runs SET status='completed'")
    app = AppTest.from_function(
        _stores_page, args=(str(tmp_path), str(tmp_path / "empty"))
    ).run()
    assert not any(
        button.label == "Focus recorded running runs" for button in app.button
    )
    app.selectbox(key="dashboard_run_store").set_value("Pilot").run()
    assert not app.exception
    assert app.multiselect(key="selected_runs").value == []


def test_active_focus_is_explicit_and_preserves_manual_selection(tmp_path):
    import sqlite3

    store = ExperimentStore(tmp_path)
    for run in ("old", "active"):
        store.create_run(run, {"name": run}, {"purpose": "training"})
    with sqlite3.connect(tmp_path / "experiments.sqlite3") as db:
        db.execute("UPDATE runs SET status='completed' WHERE run_id='old'")
        db.execute("UPDATE runs SET status='running' WHERE run_id='active'")
    app = AppTest.from_function(
        _stores_page, args=(str(tmp_path), str(tmp_path / "other"))
    ).run()
    assert app.multiselect(key="selected_runs").value == ["active"]
    app.multiselect(key="selected_runs").set_value(["old"]).run()
    assert app.multiselect(key="selected_runs").value == ["old"]
    next(
        button for button in app.button if button.label == "Focus recorded running runs"
    ).click().run()
    assert app.multiselect(key="selected_runs").value == ["active"]
    assert not app.exception
