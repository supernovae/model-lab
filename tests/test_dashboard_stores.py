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
