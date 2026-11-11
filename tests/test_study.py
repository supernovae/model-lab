from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest
import yaml
from context_fixtures import write_composition_fixture
from tokenizers import Tokenizer

from klab.evaluation.capabilities import load_capability_card
from klab.experiments.study import (
    ArchitectureStudy,
    StudyComparison,
    StudyPair,
    _comparison_report,
    plan_study,
    study_plan_payload,
)
from klab.model.inspection import parameter_inventory

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "runtime_smoke_cpu.yaml"


def _write_study(
    tmp_path: Path,
    variants: list[dict[str, object]],
    comparisons: list[dict[str, object]],
    *,
    seeds: bool = True,
    axis_name: str = "architecture",
) -> Path:
    axes: dict[str, object] = {}
    if seeds:
        axes["seed"] = [
            {"label": "seed-7", "set": {"seed": 7}},
            {"label": "seed-41", "set": {"seed": 41}},
        ]
    axes[axis_name] = variants
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "matrix_version": 1,
                "base_config": str(BASE),
                "axes": axes,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    study_path = tmp_path / "study.yaml"
    study_path.write_text(
        yaml.safe_dump(
            {
                "study_version": 1,
                "name": "tiny-architecture-study",
                "matrix": matrix_path.name,
                "cards": ["chat-alias-recall-v1"],
                "comparisons": comparisons,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return study_path


def _variants() -> list[dict[str, object]]:
    return [
        {"label": "dense", "set": {}},
        {
            "label": "ngram",
            "set": {
                "model.memory": "ngram",
                "model.memory_table_size": 31,
                "model.memory_ngram_size": 2,
                "model.memory_dim": 8,
            },
        },
        {
            "label": "moe",
            "set": {"model.ffn": "moe", "model.num_experts": 2},
        },
    ]


def test_plan_pairs_architectures_across_seeds_and_reports_size(tmp_path: Path) -> None:
    path = _write_study(
        tmp_path,
        _variants(),
        [
            {
                "id": "ngram-effect",
                "vary": "memory",
                "baseline": {"architecture": "dense"},
                "variant": {"architecture": "ngram"},
            },
            {
                "id": "moe-effect",
                "vary": "ffn",
                "baseline": {"architecture": "dense"},
                "variant": {"architecture": "moe"},
            },
        ],
    )

    plan = plan_study(path)
    payload = study_plan_payload(plan)

    assert len(plan.expanded) == 6
    assert len(plan.pairs) == 4
    comparisons = payload["comparisons"]
    assert isinstance(comparisons, list)
    assert [item["id"] for item in comparisons] == [
        "ngram-effect",
        "moe-effect",
    ]
    run_rows = payload["runs"]
    assert isinstance(run_rows, list)
    dense, ngram, moe = run_rows[:3]
    assert isinstance(dense, dict) and isinstance(ngram, dict) and isinstance(moe, dict)
    assert ngram["parameter_inventory"]["total"] > dense["parameter_inventory"]["total"]
    assert moe["parameter_inventory"]["routed_expert"] > 0
    assert all(len(item["pairs"]) == 2 for item in comparisons)
    interpretation = payload["interpretation"]
    assert isinstance(interpretation, str) and interpretation.startswith("Plan only")


def test_plan_supports_explicit_custom_architecture_fields(tmp_path: Path) -> None:
    path = _write_study(
        tmp_path,
        [
            {"label": "base", "set": {}},
            {"label": "wider-ffn", "set": {"model.ffn_dim": 48}},
        ],
        [
            {
                "id": "ffn-width",
                "vary": "custom",
                "vary_fields": ["model.ffn_dim"],
                "baseline": {"size": "base"},
                "variant": {"size": "wider-ffn"},
            }
        ],
        seeds=False,
        axis_name="size",
    )

    plan = plan_study(path)

    assert len(plan.pairs) == 1
    assert set(plan.pairs[0].differences) == {"model.ffn_dim"}


def test_plan_rejects_unmatched_or_misdeclared_architecture_comparisons(
    tmp_path: Path,
) -> None:
    path = _write_study(
        tmp_path,
        _variants(),
        [
            {
                "id": "wrong-axis",
                "vary": "ffn",
                "baseline": {"architecture": "dense"},
                "variant": {"architecture": "ngram"},
            }
        ],
    )

    with pytest.raises(ValueError, match="not controlled"):
        plan_study(path)


def test_duplicate_study_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "study.yaml"
    path.write_text("study_version: 1\nstudy_version: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate study YAML key"):
        plan_study(path)


def test_validation_loss_requires_matched_run_identity(tmp_path: Path) -> None:
    study = ArchitectureStudy(
        path=tmp_path / "study.yaml",
        name="identity-mismatch",
        matrix_path=tmp_path / "matrix.yaml",
        matrix_sha256="matrix",
        cards=(),
        expanded=(),
        comparisons=(),
        pairs=(),
        study_sha256="study",
    )
    comparison = StudyComparison(
        "ffn-width",
        "custom",
        {"architecture": "baseline"},
        {"architecture": "variant"},
        ("model.ffn_dim",),
    )
    pair = StudyPair(
        comparison,
        0,
        1,
        {},
        {"model.ffn_dim": {"base": 32, "variant": 48}},
    )
    identity = {
        "step": 1,
        "tokens_seen": 32,
        "tokenizer_sha256": "tokenizer",
        "data_sha256": {"train": "train", "validation": "validation"},
        "source_identity_sha256": "source",
        "runtime": {"engine": "pytorch"},
        "training_runtime": {"engine": "pytorch"},
    }
    baseline = {
        "coordinate": {"architecture": "baseline"},
        "run_id": "baseline-run",
        "identity": identity,
        "validation": {"loss": 4.0},
        "capabilities": {},
    }
    variant = {
        "coordinate": {"architecture": "variant"},
        "run_id": "variant-run",
        "identity": {**identity, "tokens_seen": 31},
        "validation": {"loss": 3.5},
        "capabilities": {},
    }

    report = _comparison_report(study, pair, [baseline, variant])
    validation = report["validation_loss"]

    assert isinstance(validation, dict)
    assert validation["status"] == "inconclusive"
    assert validation["reason"] == "run identity differs at tokens_seen"
    assert "delta_variant_minus_baseline" not in validation


def test_neutral_memory_placement_plan_and_composition_card(tmp_path: Path) -> None:
    memory = {
        "model.memory": "ngram",
        "model.memory_table_size": 31,
        "model.memory_ngram_size": 2,
        "model.memory_dim": 8,
    }
    path = _write_study(
        tmp_path,
        [
            {"label": "none", "set": {}},
            {"label": "final", "set": {**memory, "model.memory_injection": "final"}},
            {
                "label": "embedding",
                "set": {**memory, "model.memory_injection": "embedding"},
            },
        ],
        [
            {
                "id": f"{variant}-minus-none",
                "vary": "memory",
                "baseline": {"architecture": "none"},
                "variant": {"architecture": variant},
            }
            for variant in ("final", "embedding")
        ]
        + [
            {
                "id": "embedding-minus-final",
                "vary": "custom",
                "vary_fields": ["model.memory_injection"],
                "baseline": {"architecture": "final"},
                "variant": {"architecture": "embedding"},
            }
        ],
    )
    matrix_path = tmp_path / "matrix.yaml"
    matrix = yaml.safe_load(matrix_path.read_text())
    matrix["axes"]["seed"] = [
        {"label": f"s{seed}", "set": {"seed": seed}} for seed in (7, 11, 13)
    ]
    matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False))
    study = plan_study(path)

    assert len(study.expanded) == 9
    assert len(study.pairs) == 9
    assert [item.coordinate["seed"] for item in study.expanded] == [
        seed for seed in ("s7", "s11", "s13") for _ in ("none", "final", "embedding")
    ]
    placement_pairs = [
        pair
        for pair in study.pairs
        if pair.comparison.identifier == "embedding-minus-final"
    ]
    assert len(placement_pairs) == 3
    for pair in placement_pairs:
        assert set(pair.differences) == {"model.memory_injection"}
        baseline = study.expanded[pair.baseline_index].config
        variant = study.expanded[pair.variant_index].config
        assert parameter_inventory(baseline) == parameter_inventory(variant)

    card_path, tokenizer = write_composition_fixture(tmp_path)
    card = load_capability_card(card_path)
    assert [case.identifier for case in card.cases] == [
        f"two-hop-{pair}{side}" for pair in range(1, 5) for side in ("a", "b")
    ]
    assert Counter(case.expected for case in card.cases) == Counter(
        {
            "amber": 1,
            "ivory": 1,
            "onyx": 1,
            "pearl": 1,
            "cobalt": 1,
            "sienna": 1,
            "topaz": 1,
            "violet": 1,
        }
    )
    for pair_index in range(0, len(card.cases), 2):
        left, right = card.cases[pair_index : pair_index + 2]
        left_edges = re.findall(r"(\w+) maps to (\w+)", left.prompt)
        right_edges = re.findall(r"(\w+) maps to (\w+)", right.prompt)
        assert left_edges == right_edges
        left_start = re.search(r"Follow two links from (\w+)", left.prompt)
        right_start = re.search(r"Follow two links from (\w+)", right.prompt)
        assert left_start is not None and right_start is not None
        assert left_start.group(1) != right_start.group(1)
        assert left.expected != right.expected
        graph = dict(left_edges)
        assert graph[left_start.group(1)] != left.expected
        assert graph[graph[left_start.group(1)]] == left.expected
        assert graph[right_start.group(1)] != right.expected
        assert graph[graph[right_start.group(1)]] == right.expected
    lengths = [
        len(tokenizer.encode(case.prompt, add_special_tokens=False).ids)
        for case in card.cases
    ]
    assert lengths == [26] * 8
    assert max(lengths) + card.generation["max_new_tokens"] <= 40


@pytest.mark.parametrize("external", [False, True])
def test_three_seed_cli_submission_uses_one_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    external: bool,
) -> None:
    """Submission seals inputs and queues coordinates; it never starts a worker."""
    import json
    import tempfile

    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel

    from klab.cli.main import main
    from klab.config.loading import load_config
    from klab.workers.controller import Controller

    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "klab-work"))
    tokenizer_path = tmp_path / "tokenizer.json"
    vocabulary = {
        token: index
        for index, token in enumerate(
            ["<unk>", "<pad>", "<bos>", "<eos>", *sorted(ByteLevel.alphabet())]
        )
    }
    tokenizer = Tokenizer(BPE(vocab=vocabulary, merges=[], unk_token="<unk>"))
    tokenizer.save(str(tokenizer_path))
    config = load_config(BASE)
    config = config.model_copy(
        update={
            "model": config.model.model_copy(update={"vocab_size": len(vocabulary)}),
            "tokenizer": config.tokenizer.model_copy(update={"path": tokenizer_path}),
            "dataset": config.dataset.model_copy(
                update={"cache_dir": tmp_path / "cache"}
            ),
        }
    )
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")))
    (tmp_path / "matrix.yaml").write_text(
        yaml.safe_dump(
            {
                "matrix_version": 1,
                "base_config": "base.yaml",
                "axes": {
                    "seed": [
                        {"label": f"s{seed}", "set": {"seed": seed}}
                        for seed in (17, 42, 73)
                    ]
                },
            }
        )
    )
    (tmp_path / "study.yaml").write_text(
        yaml.safe_dump(
            {
                "study_version": 1,
                "name": "dense-lm-v1",
                "matrix": "matrix.yaml",
                "cards": ["chat-alias-recall-v1"],
                "comparisons": [],
            }
        )
    )
    work = tmp_path / (
        "external-disk/dense-lm-v1" if external else "klab-work/experiments/dense-lm-v1"
    )
    args = ["study", "submit", "study.yaml"]
    if external:
        args.extend(["--store", str(work / "runs")])
    monkeypatch.setattr("sys.argv", ["klab", *args])
    main()
    output = json.loads(capsys.readouterr().out)
    receipt = json.loads((work / "receipt.json").read_text())
    queued = Controller(work / "runs").list_experiments()
    assert len(queued) == len(receipt["runs"]) == 3
    assert {item["coordinate"]["seed"] for item in receipt["runs"]} == {
        "s17",
        "s42",
        "s73",
    }
    assert {item["run_id"] for item in queued} == {
        item["run_id"] for item in receipt["runs"]
    }
    assert all(item["status"] == "QUEUED" for item in queued)
    assert Path(output["store"]) == work / "runs"
    assert not list(tmp_path.glob("runs*"))
    assert {path.name for path in work.glob("receipt*.json")} == {
        "receipt.json",
        "receipt.json.intent.json",
    }

    # Exercise the controller's resume routing without fabricating model weights
    # or running training. Only checkpoint bundle preparation is stubbed.
    controller = Controller(work / "runs")
    parent_id = receipt["runs"][0]["run_id"]
    parent = controller.store.attempt_by_run(parent_id)
    controller.store.metrics.create_run(parent_id, parent["spec"]["config"], {})
    checkpoint = work / "runs" / parent_id / "checkpoints" / "test-generation"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text(json.dumps({"sha256": "test-digest"}))
    controller.store.metrics.record_checkpoint(
        parent_id,
        {
            "relative_path": "test-generation",
            "digest": "test-digest",
            "step": 1,
            "tokens_seen": 1,
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    with controller.store.transaction() as connection:
        connection.execute(
            "UPDATE attempts SET status='INTERRUPTED', ingestion_status='COMPLETE' WHERE run_id=?",
            (parent_id,),
        )
    prepare = controller._prepare_entry
    requests = []

    def prepare_without_weights(request):
        requests.append(request)
        return prepare(
            {key: value for key, value in request.items() if key != "resume"}
        )

    monkeypatch.setattr(controller, "_prepare_entry", prepare_without_weights)
    child = controller.resume(parent_id)
    assert requests[0]["resume"] == checkpoint
    assert requests[0]["matrix"].model_dump(mode="json") == parent["spec"]["matrix"]
    assert controller.store.attempt_by_run(child.run_id) is not None
    assert len(controller.list_experiments()) == 4
    assert not list(tmp_path.glob("runs*"))


def test_study_paths_preserve_explicit_overrides(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from klab.experiments.study import study_execution_paths

    study = SimpleNamespace(name="any-study")
    receipt = tmp_path / "custom" / "submission.json"
    store = tmp_path / "other-disk" / "custom-store"
    assert study_execution_paths(study, receipt=receipt) == (
        receipt.parent / "runs",
        receipt,
    )
    assert study_execution_paths(study, store=store, receipt=receipt) == (
        store,
        receipt,
    )
