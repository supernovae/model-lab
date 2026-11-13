"""Reference points, cross-record comparison and the Pareto data path.

Runs in CI without transformers/lm-eval and without downloads: the reference
battery uses a tiny in-memory model through ``ReferenceRun``, and comparisons
use the sealed reference results packaged with Model Lab.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from klab import reference_models
from klab.lab_records import comparison_group, read_lab_record, write_sealed
from klab.probes import compare as compare_mod
from klab.probes import lm_eval_adapter, metrics, runner
from klab.probes.points import (
    METRICS,
    collect_points,
    metric_points,
    packaged_reference_records,
    pareto_frontier,
    points_from_record,
    probe_points,
)
from klab.probes.suite import BY_ID
from klab.reference_models import (
    REFERENCES,
    ReferenceRun,
    ReferenceUnavailable,
    reference_for,
)

LM_TASKS = BY_ID["lm_eval"].params["tasks"]
# References with a packaged lm-eval result (Qwen2.5-Coder ships a code
# benchmark result instead).
LM_REFERENCES = set(REFERENCES) - {"Qwen2.5-Coder-0.5B"}


# --- Registry and snapshot safety ---------------------------------------------


def test_registry_pins_immutable_commits_and_declared_facts() -> None:
    assert set(REFERENCES) == {
        "pythia-70m-deduped",
        "pythia-160m-deduped",
        "SmolLM2-135M",
        "SmolLM2-360M",
        "Qwen2.5-Coder-0.5B",
    }
    for ref in REFERENCES.values():
        assert len(ref.revision) == 40 and int(ref.revision, 16) >= 0
        assert ref.repo_id.split("/")[0] in {"EleutherAI", "HuggingFaceTB", "Qwen"}
        assert ref.training_tokens > 0 and ref.training_tokens_source
    assert reference_for("ref:SmolLM2-135M") is REFERENCES["SmolLM2-135M"]
    with pytest.raises(ValueError, match="known: ref:pythia-70m-deduped"):
        reference_for("ref:gpt2")


def _snapshot(tmp_path: Path, **config: Any) -> Path:
    snap = tmp_path / "snap"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(
        json.dumps(
            {"model_type": "llama", "architectures": ["LlamaForCausalLM"], **config}
        )
    )
    (snap / "tokenizer.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"x")
    return snap


def test_reference_snapshots_refuse_pickles_remote_code_and_quantization(
    tmp_path: Path,
) -> None:
    snap = _snapshot(tmp_path)
    assert {p.name for p in reference_models.validate_snapshot(snap)} == {
        "config.json",
        "tokenizer.json",
        "model.safetensors",
    }
    reference_models.validate_model_metadata(
        snap, model_type="llama", architecture="LlamaForCausalLM"
    )
    (snap / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(ValueError, match="disallowed file: pytorch_model.bin"):
        reference_models.validate_snapshot(snap, "SmolLM2")
    remote = _snapshot(tmp_path / "r", auto_map={"AutoModel": "x.Y"})
    with pytest.raises(ValueError, match="forbids remote code or quantization"):
        reference_models.validate_model_metadata(
            remote, model_type="llama", architecture="LlamaForCausalLM"
        )
    with pytest.raises(ValueError, match="model_type=gpt_neox"):
        reference_models.validate_model_metadata(
            snap, model_type="gpt_neox", architecture="GPTNeoXForCausalLM"
        )


def test_missing_optional_stack_fails_early_with_install_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import importlib.util

    from klab.probes.cli import run_probe

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: (
            None if name in {"transformers", "lm_eval"} else real(name, *a)
        ),
    )
    with pytest.raises(ReferenceUnavailable, match="uv sync --extra reference"):
        run_probe("ref:SmolLM2-135M", None, lab_dir=tmp_path, tier="full")
    assert not (tmp_path / "probes").exists()  # nothing half-written


def test_probe_cli_refuses_reference_misuse(tmp_path: Path) -> None:
    from klab.probes.cli import run_probe

    with pytest.raises(ValueError, match="use --config full"):
        run_probe("ref:SmolLM2-135M", None, lab_dir=tmp_path, tier="fast")
    with pytest.raises(ValueError, match="not a probe baseline"):
        run_probe("some-run", "ref:SmolLM2-135M", lab_dir=tmp_path, tier="full")
    with pytest.raises(ValueError, match="unknown reference"):
        run_probe("ref:nope", None, lab_dir=tmp_path, tier="full")


# --- Reference arm through the battery (tiny fake model, no downloads) ------


def _fake_reference() -> ReferenceRun:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    tokenizer = Tokenizer(WordLevel({"a": 0, "b": 1, "<unk>": 2}, unk_token="<unk>"))
    model = torch.nn.Sequential(torch.nn.Embedding(3, 4), torch.nn.Linear(4, 3))
    ref = REFERENCES["SmolLM2-135M"]
    identity = {
        "run_id": "ref:SmolLM2-135M",
        "checkpoint_relative_path": f"{ref.repo_id}@{ref.revision}",
        "checkpoint_sha256": "w" * 64,
        "step": None,
        "tokens_seen": ref.training_tokens,
        "tokenizer_sha256": "t" * 64,
        "data_sha256": {},
        "parameter_inventory": {"total": 27, "active_per_token": 19},
    }
    return ReferenceRun(
        ref,
        model,
        tokenizer,
        max_seq_len=8,
        eot_token_id=0,
        identity=identity,
        device=torch.device("cpu"),
    )


def _summary(
    items: dict[str, list[float]], doc_hash: str = "d", prompt: str = "Q"
) -> dict[str, Any]:
    """An lm-eval summary built by the real summarizer from harness-shaped output."""
    results = {
        "results": {t: {"acc,none": float(np.mean(v))} for t, v in items.items()},
        "versions": dict.fromkeys(items, 1.0),
        "n-shot": dict.fromkeys(items, 0),
        "samples": {
            t: [
                {
                    "doc_id": i,
                    "doc_hash": f"{doc_hash}{t}{i}",
                    "arguments": [[f"{prompt}{i}:", " yes"], [f"{prompt}{i}:", " no"]],
                    "target": 0,
                    "acc": x,
                }
                for i, x in reversed(list(enumerate(v)))  # harness order is not ours
            ]
            for t, v in items.items()
        },
    }
    summary = lm_eval_adapter.summarize_results(results, list(items))
    summary["benchmark"]["limit"] = len(next(iter(items.values())))
    summary["benchmark_group"] = lm_eval_adapter.benchmark_group(summary)
    summary["version"] = "test"
    return summary


def test_reference_arm_runs_text_level_probes_and_lm_eval_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from klab.probes.scoring import eot

    loaded = _fake_reference()
    assert eot(loaded) == 0
    items = {t: [1.0, 0.0, 1.0, 1.0] * 12 + [1.0, 1.0] for t in LM_TASKS}  # limit 50
    monkeypatch.setattr(lm_eval_adapter, "run_lm_eval", lambda *a, **k: _summary(items))
    result = runner.run_battery(
        runner.Arm(load=lambda: loaded, reference=True), tier="full"
    )
    # Tokenizer-independent probes only: text-level recall and lm-eval.
    assert [r["id"] for r in result["probes"]] == [
        "parametric_recall",
        "fact_recall",
        "needle",
        "lm_eval",
    ]
    assert result["tiers_run"] == ["standard", "full"] and result["protocol"] == {}
    # Closed-book recall needs the run's own facts manifest: a public model
    # was never trained on one, so it is inapplicable (not missing evidence).
    parametric, recall = result["probes"][:2]
    assert parametric["status"] == "skipped" and parametric["value"] is None
    assert "inapplicable" in parametric["note"]
    assert recall["status"] == "info" and recall["details"]["item_group"]
    assert len(recall["details"]["items"]) == recall["details"]["n"]
    needle = result["probes"][2]
    assert needle["status"] == "info" and needle["details"]["item_group"]
    row = result["probes"][3]
    assert row["status"] == "info" and row["value"] == pytest.approx(38 / 50)
    assert result["verdict"]["status"] == "info"
    assert result["verdict"]["action"] == "compare"
    assert "klab compare" in row["note"]
    assert row["details"]["benchmark_group"]
    assert "klab compare RESULT --references" in result["verdict"]["suggestion"]
    target = result["target"]
    assert target["reference"]["revision"] == REFERENCES["SmolLM2-135M"].revision
    assert target["parameters"] == 27 and target["active_parameters"] == 19
    assert target["active_parameter_bytes"] == round(27 * 4 * 19 / 27)
    assert target["eval_group"]  # never equal to a lab run's (see below)
    with pytest.raises(ValueError, match="not a probe baseline"):
        runner.run_battery(
            runner.Arm(load=lambda: loaded),
            runner.Arm(load=lambda: loaded, reference=True),
            tier="full",
        )


def test_reference_validation_identity_never_matches_lab_data() -> None:
    identity = runner._validation_identity(_fake_reference())
    assert set(identity) == {"reference", "tokenizer_sha256"}
    assert "validation_sha256" not in identity


# --- lm-eval benchmark group and paired SE ------------------------------------


def test_benchmark_group_pins_tasks_versions_shots_and_items() -> None:
    items = {"piqa": [1.0, 0.0, 1.0], "arc_easy": [0.0, 0.0, 1.0]}
    base = _summary(items)
    assert base["tasks"]["piqa"]["items"] == [1.0, 0.0, 1.0]  # ordered by doc_id
    assert base["mean_accuracy"] == pytest.approx((2 / 3 + 1 / 3) / 2)
    same = _summary({"piqa": [0.0, 0.0, 0.0], "arc_easy": [1.0, 1.0, 1.0]})
    assert same["benchmark_group"] == base["benchmark_group"]  # scores don't matter
    other_items = _summary(items, doc_hash="other")
    assert other_items["benchmark_group"] != base["benchmark_group"]
    # The rendered prompts/targets and the scoring protocol are part of it.
    reworded = _summary(items, prompt="Question ")
    assert reworded["benchmark_group"] != base["benchmark_group"]
    assert base["benchmark"]["scoring_protocol"] == lm_eval_adapter.SCORING_PROTOCOL
    fewer = _summary({"piqa": [1.0, 0.0, 1.0]})
    assert fewer["benchmark_group"] != base["benchmark_group"]
    assert base["benchmark_group"] == comparison_group(base["benchmark"])


def test_task_mean_difference_combines_per_task_paired_se() -> None:
    a = {"x": [1, 1, 0, 1], "y": [0, 1, 1, 1]}
    b = {"x": [0, 1, 0, 0], "y": [0, 1, 0, 1]}
    delta, se = metrics.task_mean_difference(a, b)
    se_x = metrics.paired_mean_and_se(np.array([1, 0, 0, 1]))[1]
    se_y = metrics.paired_mean_and_se(np.array([0, 0, 1, 0]))[1]
    assert delta == pytest.approx((0.5 + 0.25) / 2)
    assert se == pytest.approx(math.sqrt(se_x**2 + se_y**2) / 2)
    assert metrics.task_mean_difference(a, {"x": [1, 1, 1, 1]}) is None
    assert metrics.task_mean_difference(a, {"x": [1], "y": [1]}) is None


def test_lm_eval_judge_refuses_other_benchmark_groups() -> None:
    spec = BY_ID["lm_eval"]
    t = {**_summary({"piqa": [1.0, 1.0, 0.0, 1.0]}), "value": 0.75}
    b = {**_summary({"piqa": [1.0, 0.0, 0.0, 1.0]}), "value": 0.5}
    comparable = {"validation": True, "validation_reason": None, "tokenizer": True}
    row = runner._judge(spec, t, b, comparable, {})
    assert row["delta"] == pytest.approx(0.25)
    assert row["delta_se"] == pytest.approx(0.25)
    other = {**_summary({"piqa": [1.0, 0.0, 0.0, 1.0]}, doc_hash="z"), "value": 0.5}
    row = runner._judge(spec, t, other, comparable, {})
    assert row["status"] == "not_comparable"
    assert "different benchmark group" in row["note"]


# --- Packaged reference results -----------------------------------------------


def test_packaged_reference_results_are_sealed_and_match_the_registry() -> None:
    records = packaged_reference_records()
    # Text-level recall is scored on the same items for every model, so one
    # item group spans all references (and any lab run).
    item_groups = {
        "fact_recall": runner.item_group("fact_recall", runner.recall_items("heldout"))
    }
    names = set()
    groups = set()
    for path, record in records:
        assert read_lab_record(path, "probe")[1] == record
        target = record["target"]
        ref = REFERENCES[target["reference"]["name"]]
        names.add(ref.name)
        assert path.stem == ref.name
        assert target["checkpoint"] == f"{ref.repo_id}@{ref.revision}"
        assert target["reference"]["revision"] == ref.revision
        assert target["tokens_seen"] == ref.training_tokens
        assert len(target["checkpoint_sha256"]) == 64
        assert 0 < target["active_parameters"] <= target["parameters"]
        assert [r["id"] for r in record["probes"]] == [
            "parametric_recall",
            "fact_recall",
            "lm_eval",
        ]
        # Sealed under suite v3 (before the text-level needle); still
        # comparable on fact recall and lm-eval because comparisons pair by
        # item/benchmark group, checked below. Re-sealing adds needle (PR4).
        assert record["suite"]["version"] == 3
        parametric = record["probes"][0]
        assert parametric["status"] == "skipped" and parametric["value"] is None
        assert "inapplicable" in parametric["note"]
        recall = record["probes"][1]
        assert recall["details"]["item_group"] == item_groups["fact_recall"]
        row = record["probes"][2]
        assert 0 < row["value"] < 1
        details = row["details"]
        assert details["benchmark"]["limit"] == BY_ID["lm_eval"].params["limit"]
        assert set(details["tasks"]) == set(LM_TASKS)
        for task in details["tasks"].values():
            assert len(task["items"]) == BY_ID["lm_eval"].params["limit"]
            assert task["acc"] == pytest.approx(np.mean(task["items"]))
        groups.add(details["benchmark_group"])
    assert names == LM_REFERENCES
    assert len(groups) == 1  # every reference sits on one comparable curve


# --- Points, compare and the Pareto data path --------------------------------


def _ref_lm(name: str) -> dict[str, Any]:
    for _, record in packaged_reference_records():
        if record["target"]["reference"]["name"] == name:
            return next(r for r in record["probes"] if r["id"] == "lm_eval")
    raise AssertionError(name)


def _try_record(
    lab: Path, try_id: str, *, eval_group: str | None, shas: str = "ab"
) -> dict[str, Any]:
    def arm(sha: str, loss: float) -> dict[str, Any]:
        return {
            "run_id": f"lab-{try_id}-{sha}",
            "checkpoint_sha256": sha * 64,
            "step": 20,
            "tokens_seen": 40_960,
            "parameters": 3_000_000,
            "active_parameters": 1_000_000,
            "parameter_bytes": 12_000_000,
            "active_parameter_bytes": 4_000_000,
            "eval_group": eval_group,
            "heldout": {
                "loss": loss,
                "ms_per_token": 0.5,
                "window_sums": [loss * 10, loss * 10 + 1],
                "window_counts": [10, 10],
            },
        }

    record = {
        "format": "klab-lab-try-v1",
        "try_id": try_id,
        "created_at": "2026-10-10T12:00:00+00:00",
        "question": "does it help?",
        "arms": {"baseline": arm(shas[0], 5.0), "candidate": arm(shas[1], 4.0)},
        "comparison": {"verdict": "BETTER"},
    }
    path = lab / "tries" / try_id / "try.json"
    path.parent.mkdir(parents=True)
    return write_sealed(path, record)


def _probe_record(
    lab: Path,
    probe_id: str,
    *,
    lm_row: dict[str, Any] | None,
    group: str,
    checkpoint: str = "c" * 64,
    created_at: str = "2026-10-10T13:00:00+00:00",
    baseline: dict[str, Any] | None = None,
    ms_per_token: float | None = None,
) -> dict[str, Any]:
    loss = runner._row(BY_ID["heldout_loss"], status="info", value=4.5)
    loss["details"] = {
        "window_sums": [45.0, 46.0],
        "window_counts": [10, 10],
        "ms_per_token": ms_per_token,
    }
    if baseline is not None:
        loss["baseline_value"] = 4.6
    target = {
        "run_id": "lab-run",
        "checkpoint_sha256": checkpoint,
        "step": 50,
        "tokens_seen": 100_000,
        "parameters": 5_000_000,
        "active_parameters": 2_000_000,
        "eval_group": group,
    }
    record = {
        "format": "klab-probe-v1",
        "probe_id": probe_id,
        "created_at": created_at,
        "target": target,
        "baseline": baseline,
        "probes": [loss, *([lm_row] if lm_row else [])],
        "verdict": {"status": "info"},
    }
    path = lab / "probes" / probe_id / "probe.json"
    path.parent.mkdir(parents=True)
    return write_sealed(path, record)


def test_try_arms_without_probes_are_pareto_points_with_active_and_resident(
    tmp_path: Path,
) -> None:
    record = _try_record(tmp_path, "try-1", eval_group="g" * 64)
    points = points_from_record("try", record, tmp_path / "try.json")
    assert {p["role"] for p in points} == {"baseline", "candidate"}
    flat = metric_points(points, "heldout_loss")
    assert {p["value"] for p in flat} == {5.0, 4.0}
    assert all(p["parameters"] > p["active_parameters"] for p in flat)
    assert pareto_frontier([(1, 4.0), (1, 5.0)]) == [0]
    assert pareto_frontier([(1, 0.4), (2, 0.5), (3, 0.45)], maximize=True) == [0, 1]
    everything = collect_points(tmp_path)
    assert {p["kind"] for p in everything} == {"try", "reference"}
    lm = metric_points(everything, "lm_eval")
    assert {p["kind"] for p in lm} == {"reference"} and len(lm) == len(LM_REFERENCES)


def test_compare_never_crosses_groups_and_names_missing_evidence(
    tmp_path: Path,
) -> None:
    _try_record(tmp_path, "try-1", eval_group="g" * 64)
    _try_record(tmp_path, "try-old", eval_group=None, shas="de")
    report = compare_mod.compare(
        "try-1", ["try-old"], lab_dir=tmp_path, references=True
    )
    by_other = {
        c["other"]: {p["metric"]: p for p in c["pairs"]} for c in report["comparisons"]
    }
    old = by_other["try-old:candidate"]["heldout_loss"]
    assert old["status"] == "not_comparable" and "older record" in old["note"]
    for name in REFERENCES:
        loss = by_other[name]["heldout_loss"]
        assert loss["status"] == "not_comparable" and "own tokenizer" in loss["note"]
        assert loss["delta"] is None
        lm = by_other[name]["lm_eval"]
        assert lm["status"] == "missing_evidence"
        assert (
            f"klab probe {tmp_path / 'tries/try-1/try.json'} --config full"
            in lm["note"]
        )
    text = compare_mod.render(report)
    assert "NOT COMPARABLE" in text and "MISSING EVIDENCE" in text
    assert "reference curve" in text and "--config full" in text
    json.dumps(compare_mod.as_json(report))


def test_compare_pairs_lm_eval_with_references_in_the_same_benchmark_group(
    tmp_path: Path,
) -> None:
    ref = _ref_lm("SmolLM2-135M")
    flipped = json.loads(json.dumps(ref))
    for task in flipped["details"]["tasks"].values():
        task["items"] = [1.0 - x for x in task["items"]]
        task["acc"] = float(np.mean(task["items"]))
    flipped["value"] = float(
        np.mean([t["acc"] for t in flipped["details"]["tasks"].values()])
    )
    _probe_record(tmp_path, "probe-1", lm_row=flipped, group="h" * 64)
    report = compare_mod.compare(
        "probe-1", ["ref:SmolLM2-135M"], lab_dir=tmp_path, references=False
    )
    (comparison,) = report["comparisons"]
    pairs = {p["metric"]: p for p in comparison["pairs"]}
    assert pairs["lm_eval"]["delta"] == pytest.approx(flipped["value"] - ref["value"])
    assert pairs["lm_eval"]["se"] is not None and pairs["lm_eval"]["se"] > 0
    assert pairs["lm_eval"]["status"] in {"higher", "lower", "within_noise"}
    assert pairs["heldout_loss"]["status"] == "not_comparable"

    # Same tasks, different items: refused, never compared.
    other = json.loads(json.dumps(flipped))
    other["details"]["benchmark_group"] = "x" * 64
    _probe_record(
        tmp_path, "probe-2", lm_row=other, group="h" * 64, checkpoint="z" * 64
    )
    report = compare_mod.compare("probe-2", [], lab_dir=tmp_path, references=True)
    statuses = {
        c["other"]: c["pairs"][1]["status"]
        for c in report["comparisons"]
        if c["other"] in LM_REFERENCES
    }
    assert set(statuses.values()) == {"not_comparable"}
    assert report["curve"]["excluded"] == len(LM_REFERENCES)


def test_compare_cli_lists_references_and_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from klab.cli.main import main

    _try_record(tmp_path / "lab", "try-1", eval_group="g" * 64)
    monkeypatch.setenv("NO_COLOR", "1")
    for argv in (
        ["compare", "--list-references"],
        ["compare", "try-1", "--references"],
        ["compare", "try-1", "--references", "--json"],
    ):
        monkeypatch.setattr(sys, "argv", ["klab", "--work-dir", str(tmp_path), *argv])
        main()
    out = capsys.readouterr().out
    assert "REFERENCE MODELS" in out and "ref:SmolLM2-360M" in out
    assert "COMPARE" in out and "MISSING EVIDENCE" in out
    assert '"comparisons"' in out
    monkeypatch.setattr(sys, "argv", ["klab", "--work-dir", str(tmp_path), "compare"])
    with pytest.raises(SystemExit, match="name a result"):
        main()


@pytest.mark.parametrize("populated", [False, True], ids=["empty", "populated"])
def test_compare_cli_reference_listing_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    populated: bool,
) -> None:
    from klab.cli.main import main
    from klab.probes import points

    names = list(REFERENCES)
    pool = []
    if populated:
        # Zero accuracy is still evidence; recall alone and non-reference
        # points are not lm-eval availability. Duplicate evidence adds no rows.
        pool = [
            {
                "kind": "reference",
                "reference": {"name": names[0]},
                "metrics": {"lm_eval": {"value": 0.0}},
            },
            {
                "kind": "reference",
                "reference": {"name": names[0]},
                "metrics": {"lm_eval": {"value": 0.5}},
            },
            {
                "kind": "reference",
                "reference": {"name": names[1]},
                "metrics": {"fact_recall": {"value": 1.0}},
            },
            {"kind": "reference", "reference": {"name": names[2]}, "metrics": {}},
            {
                "kind": "probe",
                "reference": {"name": names[3]},
                "metrics": {"lm_eval": {"value": 1.0}},
            },
        ]

    def collect(lab_dir: Path) -> list[dict[str, Any]]:
        assert lab_dir == tmp_path / "lab"
        return pool

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("reference listing must not compare, score or download")

    monkeypatch.setattr(points, "collect_points", collect)
    monkeypatch.setattr(compare_mod, "compare", forbidden)
    monkeypatch.setattr(compare_mod, "as_json", forbidden)
    monkeypatch.setattr(reference_models, "load_reference", forbidden)
    monkeypatch.setattr(reference_models, "safe_snapshot", forbidden)
    monkeypatch.setenv("NO_COLOR", "1")
    argv = ["klab", "--work-dir", str(tmp_path), "compare", "--list-references"]
    monkeypatch.setattr(sys, "argv", [*argv, "--json"])
    main()
    captured = capsys.readouterr()
    assert captured.err == ""
    result = json.loads(captured.out)
    assert result == {
        "schema_version": 1,
        "references": [
            {
                "target": f"ref:{ref.name}",
                "reference": asdict(ref),
                "lm_eval_available": populated and ref.name == names[0],
            }
            for ref in REFERENCES.values()
        ],
    }
    assert all(len(row["reference"]["revision"]) == 40 for row in result["references"])
    monkeypatch.setattr(sys, "argv", argv)
    main()
    text = capsys.readouterr().out
    assert text == compare_mod.render_references(pool) + "\n"
    for name in names:
        mark = "✔" if populated and name == names[0] else "○"
        assert f"{mark} ref:{name}" in text
    assert not (tmp_path / "lab").exists()


def test_dashboard_pareto_renders_references_and_try_points(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    _try_record(tmp_path, "try-1", eval_group="g" * 64)

    def app(lab: str) -> None:
        from pathlib import Path

        from klab.dashboard.probes import _pareto
        from klab.probes.points import collect_points

        _pareto(collect_points(Path(lab)), None)

    test = AppTest.from_function(app, args=(str(tmp_path),), default_timeout=60)
    test.run()
    assert not test.exception
    assert len(test.get("plotly_chart")) == 1
    test.radio(key="probe_pareto_metric").set_value("lm_eval").run()
    assert not test.exception
    assert len(test.get("plotly_chart")) == 1
    frame = test.dataframe[0].value
    assert set(frame["source"]) == {"reference"} and len(frame) == len(LM_REFERENCES)


# --- Review fixes (PR #62): each test failed before its fix. -----------------


def _comparable_lm_row(name: str = "SmolLM2-135M") -> dict[str, Any]:
    """A complete lm-eval row in the packaged references' benchmark group."""
    row = json.loads(json.dumps(_ref_lm(name)))
    for task in row["details"]["tasks"].values():
        task["items"] = [1.0 - x for x in task["items"]]
        task["acc"] = float(np.mean(task["items"]))
    row["value"] = float(np.mean([t["acc"] for t in row["details"]["tasks"].values()]))
    return row


def test_compare_picks_up_a_later_probe_of_the_same_checkpoint(tmp_path: Path) -> None:
    _try_record(tmp_path, "try-1", eval_group="g" * 64)

    def lm_pairs() -> tuple[dict[str, Any], dict[str, Any]]:
        report = compare_mod.compare("try-1", [], lab_dir=tmp_path, references=True)
        lm = {
            c["other"]: c["pairs"][1]
            for c in report["comparisons"]
            if c["other"] in LM_REFERENCES
        }
        return report, lm

    report, lm = lm_pairs()
    assert {p["status"] for p in lm.values()} == {"missing_evidence"}
    assert (
        f"klab probe {tmp_path / 'tries/try-1/try.json'} --config full"
        in lm["SmolLM2-135M"]["note"]
    )

    # The suggested `probe RECORD --config full` measured the try's candidate
    # checkpoint: its sealed record lands in the shared pool.
    _probe_record(
        tmp_path,
        "probe-later",
        lm_row=_comparable_lm_row(),
        group="g" * 64,
        checkpoint="b" * 64,
        created_at="2026-10-10T14:00:00+00:00",
    )
    report, lm = lm_pairs()
    assert {p["status"] for p in lm.values()} <= {"higher", "lower", "within_noise"}
    assert lm["SmolLM2-135M"]["se"] is not None
    evidence = report["subject"]["metrics"]["lm_eval"]["evidence"]
    assert evidence["value"] == "probe-later"
    # The try's own held-out loss still wins for its own eval group.
    assert report["subject"]["metrics"]["heldout_loss"]["value"] == 4.0
    assert report["subject"]["metrics"]["heldout_loss"]["evidence"]["value"] == "try-1"
    text = compare_mod.render(report)
    assert "lm-eval accuracy from probe-later (same checkpoint bbbbbbbbbbbb)" in text


def test_pareto_merge_keeps_latency_from_an_older_candidate_record(
    tmp_path: Path,
) -> None:
    group = "h" * 64
    _probe_record(
        tmp_path,
        "probe-old",
        lm_row=None,
        group=group,
        checkpoint="x" * 64,
        created_at="2026-10-10T13:00:00+00:00",
        ms_per_token=0.42,
    )
    # Newer: x is only the baseline (baselines are not timed).
    base = {
        "run_id": "lab-x",
        "checkpoint_sha256": "x" * 64,
        "step": 50,
        "tokens_seen": 100_000,
        "parameters": 5_000_000,
        "eval_group": group,
    }
    _probe_record(
        tmp_path,
        "probe-new",
        lm_row=None,
        group=group,
        checkpoint="y" * 64,
        created_at="2026-10-10T15:00:00+00:00",
        baseline=base,
    )
    points = metric_points(collect_points(tmp_path), "heldout_loss")
    (x,) = [p for p in points if p["checkpoint_sha256"] == "x" * 64]
    assert x["value"] == 4.6 and x["evidence"]["value"] == "probe-new"
    assert x["ms_per_token"] == 0.42 and x["evidence"]["ms_per_token"] == "probe-old"
    assert x["active_parameters"] == 2_000_000  # also only in the older record


def _reference_battery(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> dict:
    def fake(*_: Any, **__: Any) -> dict[str, Any]:
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(lm_eval_adapter, "run_lm_eval", fake)
    loaded = _fake_reference()
    return runner.run_battery(
        runner.Arm(load=lambda: loaded, reference=True), tier="full"
    )


def _assert_reference_rerun(result: dict[str, Any], tasks: list[str]) -> None:
    verdict = result["verdict"]
    assert (verdict["status"], verdict["action"]) == ("incomplete", "rerun")
    assert "klab compare" not in verdict["suggestion"]
    assert "unscored tasks: " + ", ".join(sorted(tasks)) in verdict["reasons"]
    (row,) = [r for r in result["probes"] if r["id"] == "lm_eval"]
    assert row["status"] in {"error", "skipped"}
    assert "klab compare" not in (row.get("note") or "")
    # Never a point on the reference curve.
    points = probe_points(result, source="p", path=None, packaged=False)
    assert "lm_eval" not in points[0]["metrics"]


def test_failed_reference_task_is_incomplete_not_a_reference_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("lm-eval returned no accuracy for: piqa")
    result = _reference_battery(monkeypatch, error)
    _assert_reference_rerun(result, list(LM_TASKS))
    assert "piqa" in result["probes"][-1]["note"]

    # Every task returned an accuracy but piqa scored only part of its items.
    items = {t: [1.0, 0.0] * 25 for t in LM_TASKS}
    partial = _summary(items)
    partial["tasks"]["piqa"]["items"] = partial["tasks"]["piqa"]["items"][:10]
    result = _reference_battery(monkeypatch, partial)
    _assert_reference_rerun(result, ["piqa"])
    assert "incomplete benchmark" in result["probes"][-1]["note"]


def test_reference_oom_is_incomplete_not_a_reference_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    result = _reference_battery(monkeypatch, torch.OutOfMemoryError("CUDA OOM"))
    assert result["stop"]["kind"] == "oom"
    _assert_reference_rerun(result, list(LM_TASKS))


def test_compare_uses_the_checkpoints_measurement_in_the_shared_group(
    tmp_path: Path,
) -> None:
    stale = _comparable_lm_row()
    stale["details"]["benchmark_group"] = "s" * 64  # e.g. an older protocol
    _probe_record(tmp_path, "probe-stale", lm_row=stale, group="h" * 64)
    report = compare_mod.compare("probe-stale", [], lab_dir=tmp_path, references=True)
    assert {
        c["pairs"][1]["status"]
        for c in report["comparisons"]
        if c["other"] in LM_REFERENCES
    } == {"not_comparable"}
    _probe_record(
        tmp_path,
        "probe-fresh",
        lm_row=_comparable_lm_row(),
        group="h" * 64,
        created_at="2026-10-10T14:00:00+00:00",
    )
    report = compare_mod.compare("probe-stale", [], lab_dir=tmp_path, references=True)
    pairs = [
        c["pairs"][1] for c in report["comparisons"] if c["other"] in LM_REFERENCES
    ]
    assert {p["status"] for p in pairs} <= {"higher", "lower", "within_noise"}
    assert {p["evidence"] for p in pairs} == {"probe-fresh"}
    assert "subject measured in probe-fresh" in compare_mod.render(report)
    assert report["curve"]["excluded"] == 0


def test_auto_added_references_resolve_like_named_ones(tmp_path: Path) -> None:
    # An older local result of the same reference checkpoint, in a stale group,
    # must not hide the packaged result in the subject's group.
    packaged = dict(packaged_reference_records())
    path = next(p for p in packaged if p.name == "SmolLM2-135M.json")
    old = json.loads(json.dumps(packaged[path]))
    old.pop("record_sha256", None)
    old["probe_id"] = "probe-old-ref"
    old["created_at"] = "2026-10-10T15:00:00+00:00"  # newer than the packaged one
    for row in old["probes"]:
        if row["id"] == "lm_eval":
            row["details"]["benchmark_group"] = "0" * 64
    local = tmp_path / "probes" / "probe-old-ref" / "probe.json"
    local.parent.mkdir(parents=True)
    write_sealed(local, old)
    _probe_record(tmp_path, "probe-1", lm_row=_comparable_lm_row(), group="h" * 64)

    auto = compare_mod.compare("probe-1", [], lab_dir=tmp_path, references=True)
    named = compare_mod.compare("probe-1", ["ref:SmolLM2-135M"], lab_dir=tmp_path)
    by_other = {c["other"]: c["pairs"] for c in auto["comparisons"]}
    assert by_other["SmolLM2-135M"] == named["comparisons"][0]["pairs"]
    lm = by_other["SmolLM2-135M"][1]
    assert lm["status"] != "not_comparable"
    assert lm["group"] == _ref_lm("SmolLM2-135M")["details"]["benchmark_group"]


def test_custom_task_without_chance_compares_with_references_and_no_chance_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """winogrande has no known chance baseline: compare --references and the
    Pareto figure omit the chance line (never a mean over the known tasks)."""
    import pandas as pd

    from klab.cli.main import main
    from klab.dashboard.probes import _pareto_figure

    spec = runner.lm_eval_spec(["piqa", "winogrande"], 20)
    assert spec.params["chance"] == {"piqa": 0.5, "winogrande": None}

    def harness(loaded: Any, tasks: list[str], limit: int) -> dict[str, Any]:
        rows = {
            task: {"acc": 0.55, "acc_metric": "acc", "items": [1.0, 0.0] * 10}
            for task in tasks
        }
        benchmark = {"tasks": tasks, "limit": limit}
        return {
            "tasks": rows,
            "version": "0.4.13",
            "benchmark": benchmark,
            "benchmark_group": comparison_group({"benchmark": benchmark}),
            "mean_accuracy": 0.55,
        }

    monkeypatch.setattr(lm_eval_adapter, "run_lm_eval", harness)
    measured = runner._measure_safely(spec, object(), runner.Arm(load=object), {})
    row = runner._judge(spec, measured, None, {"validation": True}, {})
    assert row["status"] == "info" and row["value"] == pytest.approx(0.55)
    lab = tmp_path / "lab"
    _probe_record(lab, "probe-wino", lm_row=row, group="w" * 64)

    report = compare_mod.compare("probe-wino", [], lab_dir=lab, references=True)
    assert report["curve"]["chance"] is None
    assert report["curve"]["excluded"] == len(LM_REFERENCES)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "klab",
            "--work-dir",
            str(tmp_path),
            "compare",
            "probe-wino",
            "--references",
        ],
    )
    capsys.readouterr()
    main()
    out = capsys.readouterr().out
    assert "reference curve" in out and "chance" not in out

    lm = pd.DataFrame(metric_points(collect_points(lab), "lm_eval"))
    mine = lm[lm["group"] == row["details"]["benchmark_group"]].reset_index(drop=True)
    mine["frontier"] = True
    figure = _pareto_figure(mine, "active_parameters", METRICS["lm_eval"], False)
    assert not figure.layout.shapes
    # Shared rule: unknown anywhere means no line; known everywhere is the mean.
    from klab.probes.points import mean_chance

    assert mean_chance({"piqa": 0.5, "winogrande": None}) is None
    assert mean_chance({"piqa": 0.5, "arc_easy": 0.25}) == pytest.approx(0.375)
    assert mean_chance({}) is None
