"""klab sweep: one baseline, several values, each run as an ordinary try."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from lab_fixtures import write_smoke_inputs

from klab.lab_records import read_lab_record
from klab.lab_results import ResultLookupError, checkpoint_from_result, resolve_result
from klab.sweep import load_sweep, points, run_sweep, summarize_point


def _declaration(root: Path, **fields: object) -> Path:
    body = {"baseline": "baseline.yaml", "vary": {"model.ffn_dim": [48, 96]}}
    body.update(fields)
    path = root / "sweep.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False))
    return path


def _fake_try(calls: list[dict[str, object]], deltas: dict[int, float]):
    def run(delta: Path, baseline: Path, **kwargs: object):
        settings = yaml.safe_load(delta.read_text())["set"]
        calls.append({"set": settings, **kwargs})
        width = settings["model.ffn_dim"]
        seed = kwargs["seed"] or 0
        loss = 3.0 + deltas[width] + 0.001 * seed
        return (
            {
                "try_id": f"try-{width}-{seed}",
                "status": "completed",
                "exit_status": 0,
                "arms": {
                    "baseline": {"heldout": {"loss": 3.0}, "reused": len(calls) > 1},
                    "candidate": {"heldout": {"loss": loss}, "parameters": width},
                },
                "comparison": {
                    "verdict": "CANDIDATE_LOWER_LOSS",
                    "metric": "loss",
                    "heldout_delta": loss - 3.0,
                },
                "resources": {"wall_seconds": 1.5},
            },
            delta.parent / "try.json",
        )

    return run


def test_grid_is_every_combination_in_declaration_order(tmp_path: Path) -> None:
    (tmp_path / "baseline.yaml").write_text("{}")
    config = load_sweep(_declaration(tmp_path, vary={"a.x": [1, 2], "b.y": ["p", "q"]}))
    assert points(config) == [
        {"a.x": 1, "b.y": "p"},
        {"a.x": 1, "b.y": "q"},
        {"a.x": 2, "b.y": "p"},
        {"a.x": 2, "b.y": "q"},
    ]
    assert config.baseline == (tmp_path / "baseline.yaml").resolve()
    # Explicit points set several fields together, in declaration order.
    paired = [{"tokenizer.path": "a.json", "model.vocab_size": 512}, {"a.x": 2}]
    config = load_sweep(_declaration(tmp_path, vary=None, points=paired))
    assert points(config) == paired


@pytest.mark.parametrize(
    "fields",
    [
        {"vary": {}},
        {"vary": {"model.ffn_dim": []}},
        {"vary": {"model.ffn_dim": [48, 48]}},
        {"seeds": [1, 1]},
        {"points": [{"model.ffn_dim": 48}]},  # with vary: one of the two
        {"vary": None, "points": [{"a.x": 1}, {"a.x": 1}]},
        {"seeds": [-1]},
        {"unknown": True},
    ],
)
def test_invalid_declarations_are_refused(tmp_path: Path, fields: dict) -> None:
    with pytest.raises(ValueError, match="invalid sweep"):
        load_sweep(_declaration(tmp_path, **fields))


def test_each_point_and_seed_runs_a_try_and_is_summarized(tmp_path: Path) -> None:
    (tmp_path / "baseline.yaml").write_text("{}")
    declaration = _declaration(tmp_path, seeds=[1, 2, 3])
    calls: list[dict[str, object]] = []
    record, path = run_sweep(
        declaration,
        work_dir=tmp_path / "work",
        try_fn=_fake_try(calls, {48: -0.2, 96: 0.0}),
    )
    assert [(c["set"], c["seed"]) for c in calls] == [
        ({"model.ffn_dim": 48}, 1),
        ({"model.ffn_dim": 96}, 1),
        ({"model.ffn_dim": 48}, 2),
        ({"model.ffn_dim": 96}, 2),
        ({"model.ffn_dim": 48}, 3),
        ({"model.ffn_dim": 96}, 3),
    ]
    assert record["status"] == "completed" and record["exit_status"] == 0
    assert read_lab_record(path, "sweep")[1] == record
    first, second = record["summary"]
    assert first["comparable"] == 3
    assert first["verdict"] == "LOWER_LOSS"
    assert first["heldout_delta_mean"] == pytest.approx(-0.198)
    assert second["verdict"] == "HIGHER_LOSS"
    with (path.parent / "series.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["verdict"] for row in rows] == ["LOWER_LOSS", "HIGHER_LOSS"]


def test_one_seed_never_claims_a_direction() -> None:
    entry = {"heldout_delta": -0.5, "wall_seconds": 1.0}
    assert summarize_point(0, {"x": 1}, [entry])["verdict"] == "SINGLE_SEED_LOWER"
    spread = [{"heldout_delta": d} for d in (-0.5, 0.4, -0.1)]
    assert summarize_point(0, {"x": 1}, spread)["verdict"] == "WITHIN_SEED_NOISE"
    assert summarize_point(0, {"x": 1}, [])["verdict"] == "NO_EVIDENCE"


def test_an_interrupted_try_stops_the_sweep(tmp_path: Path) -> None:
    (tmp_path / "baseline.yaml").write_text("{}")
    calls: list[dict[str, object]] = []
    base = _fake_try(calls, {48: -0.1, 96: 0.0})

    def interrupted(delta: Path, baseline: Path, **kwargs: object):
        result, path = base(delta, baseline, **kwargs)
        return {**result, "status": "interrupted", "exit_status": 130}, path

    record, _ = run_sweep(
        _declaration(tmp_path, seeds=[1, 2]),
        work_dir=tmp_path / "work",
        try_fn=interrupted,
    )
    assert len(calls) == 1
    assert record["status"] == "interrupted" and record["exit_status"] == 130


def test_a_sweep_is_not_a_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "baseline.yaml").write_text("{}")
    _, path = run_sweep(
        _declaration(tmp_path),
        work_dir=tmp_path / "work",
        try_fn=_fake_try([], {48: 0.0, 96: 0.0}),
    )
    lab = tmp_path / "work" / "lab"
    result = resolve_result(path.parent.name, lab_dir=lab)
    assert result.kind == "sweep"
    with pytest.raises(ResultLookupError, match="several tries"):
        checkpoint_from_result(path.parent.name, result)


def test_a_sweep_may_vary_a_path_field(tmp_path: Path) -> None:
    write_smoke_inputs(tmp_path, train_tokenizer_now=False)
    declaration = _declaration(tmp_path)
    body = yaml.safe_load(declaration.read_text())
    del body["vary"]
    body["points"] = [{"tokenizer.path": "other/tokenizer.json"}]
    declaration.write_text(yaml.safe_dump(body))
    calls: list[object] = []

    def run(delta: Path, baseline: Path, **kwargs: object):
        calls.append(yaml.safe_load(delta.read_text())["set"])
        return {"status": "failed", "exit_status": 1}, delta.parent / "try.json"

    run_sweep(declaration, work_dir=tmp_path / "work", try_fn=run)
    # Not the baseline's own path, so the point is trained rather than crashing.
    assert calls == [{"tokenizer.path": "other/tokenizer.json"}]


def test_sweep_trains_the_baseline_once_per_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    from klab.cli.main import main

    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    write_smoke_inputs(tmp_path)
    declaration = _declaration(
        tmp_path, vary={"model.ffn_dim": [48, 96, 128]}, config="no-probes"
    )
    monkeypatch.setattr(sys, "argv", ["klab", "sweep", str(declaration), "--json"])
    main()
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "completed" and record["exit_status"] == 0
    trained = [entry for entry in record["tries"] if entry["status"] != "baseline"]
    # 96 is the baseline's own width: it is reported, not trained again.
    assert [entry["set"] for entry in trained] == [
        {"model.ffn_dim": 48},
        {"model.ffn_dim": 128},
    ]
    assert [entry["baseline_reused"] for entry in trained] == [False, True]
    assert all(entry["verdict"] != "NOT_COMPARABLE" for entry in trained)
    verdicts = [point["verdict"] for point in record["summary"]]
    assert verdicts[1] == "BASELINE"
    assert {verdicts[0], verdicts[2]} <= {
        "SINGLE_SEED_LOWER",
        "SINGLE_SEED_HIGHER",
        "TIE",
    }
    monkeypatch.setattr(sys, "argv", ["klab", "report", record["sweep_id"]])
    main()
    assert record["sweep_id"] in capsys.readouterr().out
