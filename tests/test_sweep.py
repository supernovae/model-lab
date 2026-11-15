"""klab sweep: one baseline, several values, each run as an ordinary try."""

from __future__ import annotations

import csv
import json
import threading
import time
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


def _unexpected(where: str, calls: list[object], tmp_path: Path):
    base = _fake_try(calls, {48: -0.1, 96: 0.0})

    def run(delta: Path, baseline: Path, **kwargs: object):
        if where == "try" and len(calls) == 1:
            raise RuntimeError("boom")
        return base(delta, baseline, **kwargs)

    return run


@pytest.mark.parametrize("where", ["try", "summary"])
def test_an_unexpected_error_seals_the_sweep_failed_and_is_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    (tmp_path / "baseline.yaml").write_text("{}")
    if where == "summary":

        def broken(*args: object, **kwargs: object):
            raise ZeroDivisionError("bad summary")

        monkeypatch.setattr("klab.sweep.summarize_point", broken)
    message = "boom" if where == "try" else "bad summary"
    with pytest.raises(Exception, match=message):
        run_sweep(
            _declaration(tmp_path),
            work_dir=tmp_path / "work",
            try_fn=_unexpected(where, [], tmp_path),
        )
    (path,) = (tmp_path / "work" / "lab" / "sweeps").glob("*/sweep.json")
    _, record = read_lab_record(path, "sweep")
    assert record["status"] == "failed" and record["exit_status"] == 1
    assert message in record["failure"]
    # The tries that finished are kept; nothing claims a summary.
    assert len(record["tries"]) == (1 if where == "try" else 2)
    assert "summary" not in record
    from klab.sweep import summarize_sweep

    shown = summarize_sweep(record, path)
    assert "status: failed" in shown and message in shown


def _pool_inputs(root: Path, backend: str = "xpu") -> None:
    write_smoke_inputs(root, train_tokenizer_now=False)
    baseline = root / "baseline.yaml"
    body = yaml.safe_load(baseline.read_text())
    body["runtime"] = {**body.get("runtime", {}), "backend": backend}
    baseline.write_text(yaml.safe_dump(body, sort_keys=False))


class _Cards:
    """Fake cards: a try holds its device and trains a seed's baseline once."""

    def __init__(self, deltas: dict[int, float], pioneers: int = 0) -> None:
        self.deltas = deltas
        self.lock = threading.Lock()
        self.busy: set[int] = set()
        self.training: set[object] = set()
        self.trained: list[object] = []
        self.overlaps: list[int] = []
        self.peak = 0
        self.rendezvous = threading.Barrier(pioneers, timeout=20) if pioneers else None

    def __call__(self, delta: Path, baseline: Path, **kwargs: object):
        device = kwargs.get("device")
        seed = kwargs["seed"]
        with self.lock:
            if device in self.busy:
                self.overlaps.append(device)
            self.busy.add(device)
            self.peak = max(self.peak, len(self.busy))
            first = seed not in self.trained
            if first:
                assert seed not in self.training, "baseline trained twice at once"
                self.training.add(seed)
        if first and self.rendezvous is not None:
            self.rendezvous.wait()  # every seed's first try runs at the same time
        time.sleep(0.02)
        with self.lock:
            if first:
                self.training.discard(seed)
                self.trained.append(seed)
            self.busy.discard(device)
        return _fake_try([], self.deltas)(delta, baseline, **kwargs)


def _comparable(record: dict) -> tuple[list, list]:
    tries = [
        {k: v for k, v in t.items() if k not in {"device", "record"}}
        for t in record["tries"]
    ]
    return tries, record["summary"]


def test_devices_pool_seeds_in_one_record_like_a_serial_sweep(tmp_path: Path) -> None:
    _pool_inputs(tmp_path)
    deltas = {48: -0.2, 128: 0.1}
    declaration = _declaration(
        tmp_path, vary={"model.ffn_dim": [48, 128]}, seeds=[1, 2, 3]
    )
    serial, _ = run_sweep(
        declaration, work_dir=tmp_path / "serial", try_fn=_Cards(deltas)
    )
    cards = _Cards(deltas, pioneers=3)
    pooled, path = run_sweep(
        declaration,
        work_dir=tmp_path / "pooled",
        try_fn=cards,
        devices=[0, 1, 2],
    )
    # Three seeds' baselines train at once, each on its own card, once per seed.
    assert cards.peak == 3 and cards.overlaps == []
    assert sorted(cards.trained) == [1, 2, 3]
    assert pooled["devices"] == [0, 1, 2]
    assert {t["device"] for t in pooled["tries"] if "device" in t} == {0, 1, 2}
    assert _comparable(pooled) == _comparable(serial)
    assert [p["verdict"] for p in pooled["summary"]] == ["LOWER_LOSS", "HIGHER_LOSS"]
    assert read_lab_record(path, "sweep")[1] == pooled


def test_a_seed_waits_for_its_baseline_before_other_cards_use_it(
    tmp_path: Path,
) -> None:
    _pool_inputs(tmp_path)
    cards = _Cards({48: -0.1, 96: 0.0, 128: 0.2, 160: 0.3})
    record, _ = run_sweep(
        _declaration(tmp_path, vary={"model.ffn_dim": [48, 96, 128, 160]}),
        work_dir=tmp_path / "work",
        try_fn=cards,
        devices=[0, 1, 2, 3],
    )
    assert cards.trained == [None] and cards.overlaps == []
    assert [t["point"] for t in record["tries"]] == [0, 1, 2, 3]


def test_a_worker_error_seals_the_pooled_sweep_failed(tmp_path: Path) -> None:
    _pool_inputs(tmp_path)
    cards = _Cards({48: -0.1, 160: 0.0, 128: 0.2})

    def run(delta: Path, baseline: Path, **kwargs: object):
        if yaml.safe_load(delta.read_text())["set"] == {"model.ffn_dim": 128}:
            raise RuntimeError("device lost")
        return cards(delta, baseline, **kwargs)

    with pytest.raises(RuntimeError, match="device lost"):
        run_sweep(
            _declaration(tmp_path, vary={"model.ffn_dim": [48, 160, 128]}),
            work_dir=tmp_path / "work",
            try_fn=run,
            devices=[0, 1],
        )
    (path,) = (tmp_path / "work" / "lab" / "sweeps").glob("*/sweep.json")
    record = read_lab_record(path, "sweep")[1]
    assert record["status"] == "failed" and "device lost" in record["failure"]
    assert 1 <= len(record["tries"]) <= 2
    assert all(t["status"] == "completed" for t in record["tries"])


def test_ctrl_c_during_a_pooled_sweep_records_it_as_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pool_inputs(tmp_path)
    cards = _Cards({48: -0.1, 128: 0.0})

    class Impatient(threading.Thread):
        signalled = False

        def join(self, timeout: float | None = None) -> None:
            if not Impatient.signalled:
                Impatient.signalled = True
                raise KeyboardInterrupt
            super().join(timeout)

    monkeypatch.setattr("klab.sweep.threading.Thread", Impatient)
    record, _ = run_sweep(
        _declaration(tmp_path, vary={"model.ffn_dim": [48, 128]}, seeds=[1, 2, 3]),
        work_dir=tmp_path / "work",
        try_fn=cards,
        devices=[0, 1],
    )
    assert record["status"] == "interrupted" and record["exit_status"] == 130
    assert len(record["tries"]) < 4


@pytest.mark.parametrize(
    ("edit", "devices", "message"),
    [
        (lambda body: body["runtime"].update(backend="cpu"), [0, 1], "xpu, cuda"),
        (lambda body: body["runtime"].update(device_index=1), [0, 1], "device 0"),
        (
            lambda body: body["runtime"].update(secondary_device_indices=[1]),
            [0, 1],
            "device 0",
        ),
        (lambda body: None, [0, 0], "distinct"),
    ],
)
def test_devices_refuse_a_baseline_or_list_they_cannot_serve(
    tmp_path: Path, edit, devices: list[int], message: str
) -> None:
    _pool_inputs(tmp_path)
    baseline = tmp_path / "baseline.yaml"
    body = yaml.safe_load(baseline.read_text())
    edit(body)
    baseline.write_text(yaml.safe_dump(body, sort_keys=False))
    with pytest.raises(ValueError, match=message):
        run_sweep(
            _declaration(tmp_path),
            work_dir=tmp_path / "work",
            try_fn=_Cards({48: 0.0, 96: 0.0}),
            devices=devices,
        )


def test_each_device_try_is_a_klab_try_confined_to_its_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    _pool_inputs(tmp_path)
    monkeypatch.setenv("KLAB_CONFINED_DEVICE", "xpu:0")
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0")
    seen: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command: list[str], env: dict[str, str], **kwargs: object):
        seen.append((command, env))
        if "48" in Path(command[command.index("try") + 1]).read_text():
            record = {"try_id": "t", "status": "completed", "exit_status": 0}
            record["record"] = str(tmp_path / "try.json")
            return types.SimpleNamespace(stdout=json.dumps(record), returncode=0)
        kwargs["stderr"].write(b"klab try: no such thing\n")
        return types.SimpleNamespace(stdout="", returncode=1)

    monkeypatch.setattr("klab.sweep.subprocess.run", fake_run)
    record, _ = run_sweep(
        _declaration(tmp_path, vary={"model.ffn_dim": [48, 128]}, seeds=[7]),
        work_dir=tmp_path / "work",
        devices=[2],
        runtime_id="arc",
    )
    assert [t["status"] for t in record["tries"]] == ["completed", "failed"]
    assert "no such thing" in record["tries"][1]["failure"]
    command, env = seen[0]
    assert env["ZE_AFFINITY_MASK"] == "2" and "KLAB_CONFINED_DEVICE" not in env
    assert command[command.index("--runtime") + 1] == "arc"
    assert command[command.index("--sweep-point") + 1] == "0"
    policy = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
    assert policy["seed"] == 7


def test_a_try_started_by_a_sweep_records_which_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    from klab.cli.main import main

    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    baseline = write_smoke_inputs(tmp_path, train_tokenizer_now=False)
    delta = tmp_path / "delta.yaml"
    delta.write_text(yaml.safe_dump({"lab_delta_version": 1, "set": {"seed": 5}}))
    seen: dict[str, object] = {}

    def run(*args: object, request: dict, **kwargs: object):
        seen.update(request)
        return {"exit_status": 0}, tmp_path / "try.json"

    monkeypatch.setattr("klab.lab_mode.run_try", run)
    monkeypatch.setattr("klab.lab_mode.summarize", lambda *a, **k: "")
    argv = ["klab", "try", str(delta), "--vs", str(baseline)]
    for extra in ([], ["--sweep-id", "sweep-x", "--sweep-point", "3"]):
        monkeypatch.setattr(sys, "argv", [*argv, *extra])
        main()
    assert seen["sweep_id"] == "sweep-x" and seen["point"] == 3
