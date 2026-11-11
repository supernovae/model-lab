from __future__ import annotations

import json
from pathlib import Path

import pytest

from klab.config.loading import load_config
from klab.engines.base import EngineCapabilityError
from klab.experiments.compiler import compare_configs
from klab.experiments.lock import _runtime, resolve_plan
from klab.experiments.plan import load_plan, schema


@pytest.mark.parametrize(
    ("suffix", "document", "error"),
    [
        ("yaml", "plan_version: 1\nplan_version: 1\n", "duplicate YAML key"),
        ("json", '{"plan_version":1,"plan_version":1}', "duplicate JSON key"),
        (
            "yaml",
            "plan_version: !!python/object/apply:os.system ['true']",
            "invalid experiment document",
        ),
        ("json", '{"plan_version":NaN}', "non-finite JSON number"),
    ],
)
def test_authored_plan_rejects_ambiguous_or_executable_data(
    tmp_path: Path, suffix: str, document: str, error: str
) -> None:
    path = tmp_path / f"plan.{suffix}"
    path.write_text(document)
    with pytest.raises(ValueError, match=error):
        load_plan(path)


def test_authored_plan_rejects_unknown_fields_and_duplicate_axis_labels(
    tmp_path: Path,
) -> None:
    base = {
        "plan_version": 1,
        "id": "strict",
        "base_run": "run.yaml",
        "axes": [
            {
                "name": "seed",
                "choices": [
                    {"label": "same", "set": {"seed": 7}},
                    {"label": "same", "set": {"seed": 17}},
                ],
            }
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="duplicate choice labels"):
        load_plan(path)
    base.pop("axes")
    base["training_code"] = "import os"
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="extra_forbidden"):
        load_plan(path)


def test_comparison_rejects_hidden_artifact_change_and_unchanged_intervention() -> None:
    baseline = {"training": {"max_steps": 20}, "model": {"hidden_dim": 16}}
    variant = {"training": {"max_steps": 25}, "model": {"hidden_dim": 16}}
    with pytest.raises(ValueError, match="undeclared changed fields"):
        compare_configs(baseline, variant, expected=("model.hidden_dim",))
    with pytest.raises(ValueError, match="declared interventions unchanged"):
        compare_configs(baseline, baseline, expected=("model.hidden_dim",))
    with pytest.raises(ValueError, match="undeclared changed fields"):
        compare_configs(
            baseline,
            baseline,
            expected=(),
            artifacts={
                "base": {"tokenizer": {"sha256": "a" * 64}},
                "variant": {"tokenizer": {"sha256": "b" * 64}},
            },
        )


def test_published_plan_schema_is_current() -> None:
    published = (
        Path(__file__).resolve().parents[1] / "schemas/experiment-plan-v1.schema.json"
    )
    assert json.loads(published.read_text()) == schema()


@pytest.mark.parametrize(
    ("placement", "patch_path"),
    [
        ("inline", None),
        ("file", None),
        ("axis", "tokenizer.origin"),
        ("axis", "tokenizer"),
        ("phase", "tokenizer.origin"),
        ("phase", "tokenizer"),
        ("phase", "tokenizer.origin.sha256"),
    ],
)
def test_plan_refuses_tokenizer_origin_before_artifact_verification(
    tmp_path: Path, placement: str, patch_path: str | None
) -> None:
    config = load_config(
        Path(__file__).parent / "fixtures/snapshot-comparison/run.yaml"
    ).model_dump(mode="json")
    origin = {
        "dataset": config["dataset"],
        "sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
    }
    plan = {"plan_version": 1, "id": "origin-refusal", "base_run": config}
    if placement in {"inline", "file"}:
        config["tokenizer"]["origin"] = origin
        if placement == "file":
            (tmp_path / "run.json").write_text(json.dumps(config))
            plan["base_run"] = "run.json"
    else:
        value = (
            {**config["tokenizer"], "origin": origin}
            if patch_path == "tokenizer"
            else origin["sha256"]
            if patch_path == "tokenizer.origin.sha256"
            else origin
        )
        patch = {patch_path: value}
        if placement == "axis":
            plan["axes"] = [
                {"name": "source", "choices": [{"label": "shared", "set": patch}]}
            ]
        else:
            plan["phases"] = [{"id": "main", "transition": "fresh", "set": patch}]
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    with pytest.raises(
        ValueError, match="ExperimentPlan does not support tokenizer.origin"
    ):
        resolve_plan(load_plan(source), source)


@pytest.mark.parametrize("patch_path", ["tokenizer.origin", "tokenizer"])
def test_plan_without_tokenizer_origin_reaches_input_verification(
    tmp_path: Path, patch_path: str
) -> None:
    config = load_config(
        Path(__file__).parent / "fixtures/snapshot-comparison/run.yaml"
    ).model_dump(mode="json")
    value = None if patch_path == "tokenizer.origin" else config["tokenizer"]
    source = tmp_path / "plan.json"
    source.write_text(
        json.dumps(
            {
                "plan_version": 1,
                "id": "ordinary-tokenizer",
                "base_run": config,
                "axes": [
                    {
                        "name": "source",
                        "choices": [{"label": "ordinary", "set": {patch_path: value}}],
                    }
                ],
            }
        )
    )
    with pytest.raises(
        ValueError, match="no verified external dataset/tokenizer inputs"
    ):
        resolve_plan(load_plan(source), source)


def test_accelerator_science_resolves_without_local_hardware() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    )
    request = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={"backend": "rocm", "device_index": 0, "precision": "bf16"}
            )
        }
    )
    resolved, requested = _runtime(request, "rocm")
    assert resolved.runtime.backend == "rocm"
    assert resolved.runtime.precision == "bf16"
    assert requested["backend"] == "rocm"
    assert requested["precision"] == "bf16"
    cpu_bf16, _ = _runtime(
        config.model_copy(
            update={
                "runtime": config.runtime.model_copy(
                    update={"backend": "cpu", "precision": "bf16"}
                )
            }
        ),
        "cpu",
    )
    assert cpu_bf16.runtime.precision == "bf16"


@pytest.mark.parametrize(
    ("backend", "engine", "precision", "index", "selected", "error"),
    [
        ("rocm", "pytorch", "bf16", 1, "cuda", "disagrees"),
        ("auto", "pytorch", "bf16", 0, None, "explicit execution.backend=cpu"),
        ("rocm", "pytorch", "auto", 0, "rocm", "precision=auto"),
        ("cpu", "pytorch", "fp16", 0, "cpu", "does not support fp16"),
        ("mps", "pytorch", "fp16", 0, "mps", "does not support fp16"),
        ("cpu", "pytorch", "bf16", 1, "cpu", "device_index=0"),
        ("mps", "pytorch", "fp32", 1, "mps", "device_index=0"),
        ("metal", "pytorch", "fp32", 0, "metal", "unsupported runtime"),
        ("rocm", "mlx", "fp32", 0, "rocm", "unsupported runtime"),
    ],
)
def test_scientific_runtime_rejects_ambiguous_or_unsupported_requests(
    backend: str,
    engine: str,
    precision: str,
    index: int,
    selected: str | None,
    error: str,
) -> None:
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    )
    config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "engine": engine,
                    "backend": backend,
                    "precision": precision,
                    "device_index": index,
                }
            )
        }
    )
    with pytest.raises(ValueError, match=error):
        _runtime(config, selected)


def test_mlx_scientific_validation_requires_no_mlx_sdk() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    )
    config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={"engine": "mlx", "backend": "metal", "precision": "fp32"}
            )
        }
    )
    resolved, _ = _runtime(config, "metal")
    assert resolved.runtime.backend == "metal"
    invalid = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"precision": "bf16"})}
    )
    with pytest.raises(EngineCapabilityError, match="fp32 only"):
        _runtime(invalid, "metal")
