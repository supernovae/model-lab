"""Real checkpoint-bound evaluation and immutable index behavior."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_surface_review import cells
from test_training import config

from sparselab.campaign.state import publish_immutable
from sparselab.canonical import canonical_digest, canonical_json
from sparselab.evaluation import inference
from sparselab.evaluation.capabilities import describe_capability_card
from sparselab.evaluation.suite import (
    EvaluationSuite,
    run_suite,
    verify_evaluation_index,
)
from sparselab.evaluation.surface_review import (
    complete_surface_review,
    create_surface_bundle,
    open_surface_bundle,
    record_surface_judgment,
    reveal_surface_review,
)
from sparselab.runtime_profile import _seal
from sparselab.training.trainer import train


@pytest.fixture(scope="module")
def evaluated_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("evaluation-suite")
    base = config(root)
    small = base.model_copy(
        update={
            "training": base.training.model_copy(update={"max_steps": 2}),
            "optimizer": base.optimizer.model_copy(update={"warmup_steps": 0}),
        }
    )
    train(small, run_id="suite-run")
    source = root / "suite.json"
    source.write_text(
        json.dumps(
            {
                "evaluation_suite_version": 1,
                "id": "tiny",
                "evaluations": [
                    {"id": "loss", "role": "gate", "kind": "heldout_lm"},
                    {
                        "id": "surface",
                        "role": "blinded_surface",
                        "kind": "surface_review",
                        "source": "surface.json",
                    },
                    {
                        "id": "external",
                        "role": "descriptive",
                        "kind": "evidence_reference",
                        "source": "external.json",
                    },
                ],
            }
        )
    )
    return source, small.logging.root_dir


@pytest.fixture(scope="module")
def fixed_panel_run(tmp_path_factory):
    from sparselab.config.models import RunConfig
    from sparselab.data.packing import prepare_data
    from sparselab.data.tokenizer import load_tokenizer
    from sparselab.training.manifest import sha256_file

    root = tmp_path_factory.mktemp("fixed-panel-evidence")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SPARSELAB_WORK_DIR", str(root / "work"))
        base = config(root)
        values = base.model_dump(mode="json")
        values["training"].update(max_steps=1, max_tokens=32)
        values["optimizer"]["warmup_steps"] = 0
        values["evaluation"].update(every_steps=1, max_batches=1)
        values["checkpoint"]["every_steps"] = 1
        base = RunConfig.model_validate(values)
        tokenizer = load_tokenizer(base.tokenizer.path)
        panels = []
        for seed, name in ((8, "first"), (9, "second")):
            source = base.model_copy(
                update={
                    "dataset": base.dataset.model_copy(
                        update={
                            "synthetic_seed": seed,
                            "train_max_documents": 8,
                            "validation_max_documents": 4,
                            "train_max_tokens": 256,
                            "validation_max_tokens": 256,
                        }
                    )
                }
            )
            prepared = prepare_data(source, tokenizer)
            panels.append(
                {
                    "name": name,
                    "prepared_path": str(prepared.root),
                    "prepared_manifest_sha256": prepared.manifest["manifest_sha256"],
                    "tokenizer_file_sha256": sha256_file(base.tokenizer.path),
                    "tokenizer_canonical_sha256": prepared.manifest["tokenizer_sha256"],
                    "split": "validation",
                    "sampling": "first_eligible_blocks",
                    "sequence_length": 16,
                    "blocks": 3,
                    "supervised_targets": 48,
                    "batch_size": 2,
                    "max_forward_positions": 48,
                }
            )
        values["evaluation"]["panels"] = panels
        measured = RunConfig.model_validate(values)
        train(measured, run_id="fixed-panels")
    return measured.logging.root_dir / "fixed-panels"


def _panel_run_config(fixed_panel_run, root):
    from sparselab.config.models import RunConfig
    from sparselab.training.manifest import read_manifest

    values = read_manifest(fixed_panel_run / "manifest.json")["effective_config"]
    values["logging"]["root_dir"] = str(root / "runs")
    values["dataset"]["cache_dir"] = str(root / "cache")
    return RunConfig.model_validate(values)


@pytest.mark.parametrize("field", ["valid_targets", "batches"])
def test_active_fixed_panels_refuse_incomplete_evaluation(
    fixed_panel_run, tmp_path, monkeypatch, field
):
    from sparselab.engines.pytorch import PyTorchEngine

    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path))
    measured = _panel_run_config(fixed_panel_run, tmp_path)
    original = PyTorchEngine.evaluate
    observed = []

    def incomplete(self, batches):
        result = original(self, batches)
        observed.append(result)
        return replace(result, **{field: getattr(result, field) - 1})

    monkeypatch.setattr(PyTorchEngine, "evaluate", incomplete)
    with pytest.raises(
        ValueError, match="active evaluation did not complete panel coverage"
    ):
        train(measured, run_id="incomplete")
    assert len(observed) == 1
    assert observed[0].valid_targets == 48 and observed[0].batches == 2
    assert not list(
        (measured.logging.root_dir / "incomplete" / "evaluations").glob(
            "validation_step_*.json"
        )
    )


@pytest.fixture
def aliased_panel_inputs(fixed_panel_run, tmp_path, monkeypatch):
    from sparselab.config.models import RunConfig
    from sparselab.data.packing import prepare_data
    from sparselab.data.tokenizer import load_tokenizer

    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path))
    measured = _panel_run_config(fixed_panel_run, tmp_path)

    def local_config(values, train_name, train_text, validation_name, validation_text):
        for name, text in (
            (train_name, train_text),
            (validation_name, validation_text),
        ):
            (tmp_path / name).write_bytes(canonical_json({"text": text}) + b"\n")
        values["dataset"].update(
            source="local_text",
            train_path=str(tmp_path / train_name),
            validation_path=str(tmp_path / validation_name),
            license="CC0-1.0",
            train_max_documents=1,
            validation_max_documents=1,
            train_max_tokens=256,
            validation_max_tokens=256,
        )
        return RunConfig.model_validate(values)

    shared_text = "A neutral fixture sentence about X. " * 4
    panel_config = local_config(
        measured.model_dump(mode="json"),
        "panel-train.jsonl",
        "Other text for Y. " * 8,
        "panel-validation.jsonl",
        shared_text,
    )
    tokenizer = load_tokenizer(measured.tokenizer.path)
    prepared = prepare_data(panel_config, tokenizer)
    values = measured.model_dump(mode="json")
    values["evaluation"]["panels"][0].update(
        prepared_path=str(prepared.root),
        prepared_manifest_sha256=prepared.manifest["manifest_sha256"],
        tokenizer_canonical_sha256=prepared.manifest["tokenizer_sha256"],
    )
    measured = local_config(
        values,
        "run-train.jsonl",
        shared_text,
        "run-validation.jsonl",
        "Third unrelated text Z. " * 8,
    )
    training = prepare_data(measured, tokenizer)
    # Each bundle has disjoint splits; the forbidden alias crosses bundles.
    assert (
        training.manifest["train"]["sha256"]
        == prepared.manifest["validation"]["sha256"]
    )

    return measured, training


@pytest.mark.parametrize("consumer", ["train", "stage"])
def test_fixed_panel_training_alias_refused_at_public_consumers(
    aliased_panel_inputs, tmp_path, monkeypatch, consumer
):
    from sparselab.engines.pytorch import PyTorchEngine
    from sparselab.staging import materialize_prepared_inputs

    measured, _ = aliased_panel_inputs

    def no_model(*args, **kwargs):
        pytest.fail("aliased inputs must be refused before model initialization")

    monkeypatch.setattr(PyTorchEngine, "initialize", no_model)
    destination = tmp_path / "stage"
    with pytest.raises(
        ValueError, match="fixed panel validation aliases the training array"
    ):
        if consumer == "train":
            train(measured, run_id="aliased")
        else:
            materialize_prepared_inputs(measured, destination)
    assert not destination.exists()
    assert not (measured.logging.root_dir / "aliased").exists()


def test_artifact_copy_refuses_training_alias_before_copying_panels(
    aliased_panel_inputs, fixed_panel_run, tmp_path
):
    from sparselab.training.trainer import _copy_artifacts

    measured, training = aliased_panel_inputs
    # The original synthetic panels do not alias this local-text training array.
    # First prove the real copier accepts the same training bundle with them.
    allowed = measured.model_copy(
        update={"evaluation": _panel_run_config(fixed_panel_run, tmp_path).evaluation}
    )
    control = tmp_path / "accepted-copy"
    control.mkdir()
    artifacts = _copy_artifacts(control, allowed, training)
    for panel in allowed.evaluation.panels:
        name = f"panels/{panel.name}/validation.npy"
        assert any(artifact.relative_path == name for artifact in artifacts)
        assert (control / name).is_file()

    refused = tmp_path / "refused-copy"
    refused.mkdir()
    with pytest.raises(
        ValueError, match="fixed panel validation aliases the training array"
    ):
        _copy_artifacts(refused, measured, training)
    assert not (refused / "panels").exists()


def test_run_input_panel_check_refuses_alias_after_earlier_checks_are_bypassed(
    aliased_panel_inputs, fixed_panel_run, tmp_path, monkeypatch
):
    from sparselab.training import trainer

    measured, _ = aliased_panel_inputs
    original_load = trainer.load_fixed_panels
    original_copy = trainer.copy_fixed_panels

    # Fault injection: omit alias validation at the two earlier boundaries,
    # retaining native preparation, panel authentication and artifact copying.
    # Only the final run-owned input loader can refuse this alias now.
    def load_before_run(config, **kwargs):
        if kwargs.get("inputs_root") is None:
            kwargs.pop("training_manifest", None)
        return original_load(config, **kwargs)

    def copy_without_alias_check(config, destination, **kwargs):
        kwargs.pop("training_manifest", None)
        return original_copy(config, destination, **kwargs)

    monkeypatch.setattr(trainer, "load_fixed_panels", load_before_run)
    monkeypatch.setattr(trainer, "copy_fixed_panels", copy_without_alias_check)
    allowed = measured.model_copy(
        update={"evaluation": _panel_run_config(fixed_panel_run, tmp_path).evaluation}
    )
    train(allowed, run_id="post-copy-control")
    with pytest.raises(
        ValueError, match="fixed panel validation aliases the training array"
    ):
        train(measured, run_id="post-copy-alias")
    refused = measured.logging.root_dir / "post-copy-alias"
    assert (refused / "panels/first/validation.npy").is_file()
    assert not list((refused / "checkpoints").glob("step_*"))
    assert not list((refused / "evaluations").glob("validation_step_*.json"))


def test_staged_panel_training_and_resume_use_owned_inputs(
    fixed_panel_run, tmp_path, monkeypatch
):
    from sparselab.config.models import RunConfig
    from sparselab.evaluation.evidence import experiment_evidence
    from sparselab.staging import stage
    from sparselab.training.manifest import read_manifest

    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))
    values = _panel_run_config(fixed_panel_run, tmp_path).model_dump(mode="json")
    values["training"].update(max_steps=2, max_tokens=64)
    caller_inputs = tmp_path / "caller-inputs"
    caller_inputs.mkdir()
    shutil.copy2(values["tokenizer"]["path"], caller_inputs / "tokenizer.json")
    values["tokenizer"]["path"] = str(caller_inputs / "tokenizer.json")
    for panel in values["evaluation"]["panels"]:
        target = caller_inputs / panel["name"]
        shutil.copytree(panel["prepared_path"], target)
        panel["prepared_path"] = str(target)
    measured = RunConfig.model_validate(values)
    bundle = stage(measured, tmp_path / "stage", through="validate")
    caller_inputs.rename(tmp_path / "removed-caller-inputs")
    measured.dataset.cache_dir.rename(tmp_path / "removed-cache")

    train(measured, run_id="staged-panels", stage_bundle=bundle, stop_after_step=1)
    parent = measured.logging.root_dir / "staged-panels"
    bundle.rename(tmp_path / "removed-stage")
    train(
        measured,
        run_id="resumed-panels",
        resume=parent / "checkpoints/latest.json",
    )
    for name in ("staged-panels", "resumed-panels"):
        run = measured.logging.root_dir / name
        evidence = experiment_evidence(run)
        assert evidence["evidence_level"] == "checkpointed_held_out"
        assert not evidence["rejected_reports"]
        manifest = read_manifest(run / "manifest.json")
        owned = {item["relative_path"] for item in manifest["artifacts"]}
        for panel in measured.evaluation.panels:
            path = f"panels/{panel.name}/validation.npy"
            assert path in owned and (run / path).is_file()
        reports = list((run / "evaluations").glob("validation_step_*_gen_*.json"))
        assert reports
        for path in reports:
            report = json.loads(path.read_text())
            assert report["valid_targets"] == 96 and report["batches"] == 4


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("protocol", "fixed-panel report protocol mismatch"),
        ("ownership", "fixed-panel inputs are not owned run artifacts"),
        ("artifact", "fixed-panel artifact identity mismatch"),
        ("data", "fixed-panel data identity mismatch"),
    ],
)
def test_fixed_panel_evidence_refuses_resealed_metadata(
    fixed_panel_run, tmp_path, change, reason
):
    from sparselab.evaluation.evidence import (
        experiment_evidence,
        validate_held_out_report,
    )
    from sparselab.evaluation.post_train_triage import read_triage
    from sparselab.training.manifest import read_manifest

    run = fixed_panel_run
    evidence = experiment_evidence(run)
    assert evidence["evidence_level"] == "checkpointed_held_out"
    assert not evidence["rejected_reports"]
    manifest = read_manifest(run / "manifest.json")
    artifacts = {
        item["relative_path"]: item["sha256"] for item in manifest["artifacts"]
    }
    checkpoints = {
        item["path"]: item for item in evidence["checkpoints"] if item["verified"]
    }
    original = next((run / "evaluations").glob("validation_step_*_gen_*.json"))
    accepted, error = validate_held_out_report(
        original, checkpoints, artifacts, manifest, 96, 4
    )
    assert error is None
    assert accepted["valid_targets"] == 96 and accepted["batches"] == 4
    triage = read_triage(run.name, run.parent)
    assert triage["core"]["evidence"]["expected_validation_targets"] == 96

    payload = json.loads(original.read_text())
    if change == "protocol":
        payload["panels"]["first"]["protocol"]["blocks"] = 2
    elif change == "ownership":
        del artifacts["panels/first/manifest.json"]
    elif change == "artifact":
        payload["identities"]["panels/first/manifest.json"] = "0" * 64
    else:
        payload["panels"]["first"]["data_identity"]["validation_sha256"] = "0" * 64
    payload.pop("sha256")
    payload["sha256"] = canonical_digest(payload)
    report = tmp_path / "resealed.json"
    report.write_bytes(canonical_json(payload))
    rejected, error = validate_held_out_report(
        report, checkpoints, artifacts, manifest, 96, 4
    )
    assert rejected is None
    assert error == reason


def test_ordinary_run_refuses_resealed_unexpected_panel_results(
    evaluated_run, tmp_path
):
    from sparselab.evaluation.evidence import experiment_evidence

    _, runs = evaluated_run
    run = tmp_path / "suite-run"
    shutil.copytree(runs / "suite-run", run)
    assert experiment_evidence(run)["evidence_level"] == "checkpointed_held_out"
    report = next((run / "evaluations").glob("validation_step_*_gen_*.json"))
    payload = json.loads(report.read_text())
    payload["panels"] = {"undeclared": {}}
    payload.pop("sha256")
    payload["sha256"] = canonical_digest(payload)
    report.write_bytes(canonical_json(payload))
    result = experiment_evidence(run)
    assert {"path": report.name, "reason": "unexpected fixed-panel results"} in result[
        "rejected_reports"
    ]
    assert payload["checkpoint"] in result["missing_reports"]
    assert all(
        item["checkpoint"] != payload["checkpoint"]
        for item in result["quality_observations"]
    )


def test_run_evidence_signed_reuse_across_process_and_restored_mtime(
    evaluated_run, tmp_path, monkeypatch
):
    from sparselab.evaluation.evidence import experiment_evidence
    from sparselab.verification_proofs import ProofStore

    _, runs = evaluated_run
    root = tmp_path / "trusted"
    root.mkdir()
    run = root / "runs" / "suite-run"
    shutil.copytree(runs / "suite-run", run)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    store = ProofStore(root)
    cold = experiment_evidence(
        run, proof_store=store, verification_mode="verified_reuse"
    )
    script = """
import json, sys
from pathlib import Path
from sparselab.evaluation.evidence import experiment_evidence
from sparselab.data.verification import manifest_module
from sparselab.verification_proofs import ProofStore
calls = []
original = manifest_module.sha256_file
def counted(path):
    if Path(path).is_relative_to(run):
        calls.append(str(path))
    return original(path)
manifest_module.sha256_file = counted
run = Path(sys.argv[1])
result = experiment_evidence(run, proof_store=ProofStore(Path(sys.argv[2])),
                             verification_mode="verified_reuse")
print(json.dumps({"run_id": result["run_id"], "payload_sha_calls": calls}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(run), str(root)],
        capture_output=True,
        text=True,
        check=True,
    )
    warmed = json.loads(result.stdout)
    assert warmed["run_id"] == cold["run_id"]
    assert warmed["payload_sha_calls"] == []
    target = run / "data" / "train.npy"
    original = target.read_bytes()
    timestamp = target.stat().st_mtime_ns
    target.write_bytes(original[:100] + bytes([original[100] ^ 1]) + original[101:])
    import os

    os.utime(target, ns=(timestamp, timestamp))
    with pytest.raises(ValueError, match="digest mismatch"):
        experiment_evidence(
            run, proof_store=ProofStore(root), verification_mode="verified_reuse"
        )


@pytest.mark.parametrize(
    "change",
    [
        {"evaluations": [{"id": "loss", "role": "gate", "kind": "heldout_lm"}] * 2},
        {
            "evaluations": [
                {
                    "id": "x",
                    "role": "gate",
                    "kind": "evidence_reference",
                    "source": "x.json",
                }
            ]
        },
        {"evaluations": [{"id": "x", "role": "blinded_surface", "kind": "heldout_lm"}]},
        {
            "evaluations": [
                {
                    "id": "x",
                    "role": "diagnostic",
                    "kind": "capability_card",
                    "source": "../escape",
                }
            ]
        },
        {
            "evaluations": [
                {"id": "x", "role": "gate", "kind": "heldout_lm", "unexpected": True}
            ]
        },
    ],
)
def test_invalid_suite(change):
    with pytest.raises(ValueError):
        EvaluationSuite.model_validate(
            {"evaluation_suite_version": 1, "id": "tiny", **change}
        )


def test_evaluation_index_verifies_generation_and_excludes_untrusted_evidence(
    evaluated_run,
):
    source, runs = evaluated_run
    run = runs / "suite-run"
    generation = min((run / "checkpoints").glob("step_*")).name
    index_path = run_suite(source, "suite-run", generation, runs, backend="cpu")
    result = verify_evaluation_index(index_path)
    assert [item["status"] for item in result["evaluations"]] == [
        "COMPLETED",
        "SKIPPED_REVIEW",
        "UNAVAILABLE",
    ]
    assert result["checkpoint"] == f"checkpoints/{generation}"
    assert result["evaluations"][0]["result"]["loss"] >= 0
    timestamp = result["created_at_utc"]
    assert run_suite(source, "suite-run", generation, runs, backend="cpu") == index_path
    assert verify_evaluation_index(index_path)["created_at_utc"] == timestamp
    assert result["evaluation_runtime"] == {
        "engine": "pytorch",
        "backend": "cpu",
        "precision": "fp32",
        "device_index": 0,
        "observed": json.loads(Path(result["evaluations"][0]["path"]).read_text())[
            "identity"
        ]["runtime"],
    }
    pointer = run / "checkpoints" / "latest.json"
    old = pointer.read_bytes()
    try:
        pointer.write_text(
            json.dumps(
                {
                    "relative_path": generation,
                    "manifest_sha256": result["checkpoint_sha256"],
                }
            )
        )
        assert (
            verify_evaluation_index(index_path)["checkpoint_sha256"]
            == result["checkpoint_sha256"]
        )
    finally:
        pointer.write_bytes(old)


def test_missing_or_moved_result_fails_verification(evaluated_run, tmp_path):
    source, runs = evaluated_run
    generation = min((runs / "suite-run/checkpoints").glob("step_*")).name
    index_path = run_suite(source, "suite-run", generation, runs, backend="cpu")
    result = verify_evaluation_index(index_path)
    path = result["evaluations"][0]["path"]
    moved = tmp_path / "lost.json"
    shutil.move(path, moved)
    try:
        with pytest.raises(ValueError, match="missing or changed"):
            verify_evaluation_index(index_path)
    finally:
        shutil.move(moved, path)


def test_capability_card_results_reopen_and_detect_changed_card(
    evaluated_run, tmp_path
):
    _, runs = evaluated_run
    card = describe_capability_card("chat-alias-recall-v1")
    card["cases"] = [{**card["cases"][0], "prompt": "hello", "expected": "hello"}]
    card["generation"]["max_new_tokens"] = 1
    card["digest"] = hashlib.sha256(
        canonical_json(
            {
                key: value
                for key, value in card.items()
                if key not in {"digest", "format"}
            }
        )
    ).hexdigest()
    card_file = tmp_path / "card.json"
    card_file.write_text(json.dumps(card))
    source = tmp_path / "capability-suite.json"
    source.write_text(
        json.dumps(
            {
                "evaluation_suite_version": 1,
                "id": "capability",
                "evaluations": [
                    {
                        "id": "card",
                        "role": "gate",
                        "kind": "capability_card",
                        "source": "card.json",
                    }
                ],
            }
        )
    )
    generation = min((runs / "suite-run/checkpoints").glob("step_*")).name
    index = run_suite(source, "suite-run", generation, runs, backend="cpu")
    assert verify_evaluation_index(index)["evaluations"][0]["status"] == "COMPLETED"
    old = card_file.read_bytes()
    try:
        card_file.write_text(old.decode().replace("hello", "changed"))
        with pytest.raises(ValueError, match="evaluation source changed"):
            verify_evaluation_index(index)
    finally:
        card_file.write_bytes(old)


def test_unavailable_backend_retains_verified_generation(evaluated_run):
    source, runs = evaluated_run
    generation = min((runs / "suite-run/checkpoints").glob("step_*")).name
    index = run_suite(source, "suite-run", generation, runs, backend="missing-backend")
    rows = verify_evaluation_index(index)["evaluations"]
    assert rows[0]["status"] == "UNAVAILABLE"
    assert rows[1]["status"] == "SKIPPED_REVIEW"
    pointer = run_suite(
        source, "suite-run", "latest.json", runs, backend="missing-backend"
    )
    selected = verify_evaluation_index(pointer)
    assert selected["evaluations"][0]["status"] == "UNAVAILABLE"


def test_checkpoint_runtime_binding_and_cpu_override(evaluated_run, monkeypatch):
    source, runs = evaluated_run
    generation = (
        "checkpoints/" + min((runs / "suite-run/checkpoints").glob("step_*")).name
    )
    original = inference._verified_checkpoint

    def rocm_checkpoint(run, checkpoint, **verification):
        config, manifest, selected, metadata, manager = original(
            run, checkpoint, **verification
        )
        runtime = config.runtime.model_copy(
            update={"backend": "rocm", "precision": "bf16", "device_index": 2}
        )
        config = config.model_copy(update={"runtime": runtime})
        manifest = {
            **manifest,
            "runtime": {**manifest["runtime"], "backend": "rocm", "device_index": 2},
        }
        return (
            config,
            manifest,
            selected,
            {**metadata, "backend": "rocm"},
            manager,
        )

    monkeypatch.setattr(inference, "_verified_checkpoint", rocm_checkpoint)
    ephemeral = inference.evaluation_config("suite-run", runs, generation)
    assert (
        ephemeral.runtime.backend,
        ephemeral.runtime.precision,
        ephemeral.runtime.device_index,
    ) == ("rocm", "fp32", 0)
    assert (
        inference.evaluation_config(
            "suite-run", runs, generation, "cpu"
        ).runtime.backend
        == "cpu"
    )
    unavailable = verify_evaluation_index(
        run_suite(source, "suite-run", generation, runs)
    )
    assert unavailable["evaluations"][0]["status"] == "UNAVAILABLE"
    assert unavailable["evaluation_runtime"]["backend"] == "rocm"
    assert unavailable["evaluation_runtime"]["observed"] is None

    cpu = verify_evaluation_index(
        run_suite(source, "suite-run", generation, runs, backend="cpu")
    )
    assert cpu["evaluations"][0]["status"] == "COMPLETED"
    assert cpu["evaluation_runtime"]["observed"]["backend"] == "cpu"
    result = json.loads(Path(cpu["evaluations"][0]["path"]).read_text())
    assert result["identity"]["training_runtime"]["backend"] == "rocm"
    assert result["identity"]["training_runtime"]["device_index"] == 2
    assert result["identity"]["config"]["runtime"]["device_index"] == 2
    assert result["identity"]["runtime"]["device_index"] == 0
    assert (
        json.loads((runs / "suite-run/resolved_config.yaml").read_text())["runtime"][
            "backend"
        ]
        == "cpu"
    )

    probe = {
        "profile_id": "mock-rocm",
        "engine": "pytorch",
        "backend": "rocm",
        "device_index": 0,
        "available": True,
        "device_count": 1,
        "torch_hip": "6.0",
        "package_root": "/mock/source",
        "source_sha256": "f" * 64,
    }
    authorization = _seal("profile", {"python": "/mock/interpreter"}, probe)
    monkeypatch.setattr("sparselab.runtime_profile._check_current", lambda *_: None)
    monkeypatch.setattr(inference, "select_device", lambda *_: torch.device("cuda"))
    monkeypatch.setattr(inference, "torch_device_for", lambda *_: torch.device("cpu"))
    monkeypatch.setattr(inference.torch.version, "hip", "6.0")
    discover = inference.discover_runtimes
    monkeypatch.setattr(
        inference,
        "discover_runtimes",
        lambda: [
            replace(info, backend="rocm")
            for info in discover()
            if info.backend == "cpu"
        ],
    )
    rocm = verify_evaluation_index(
        run_suite(source, "suite-run", generation, runs, authorization=authorization)
    )
    assert rocm["evaluations"][0]["status"] == "COMPLETED"
    assert rocm["evaluation_runtime"]["observed"]["backend"] == "rocm"
    assert rocm["evaluation_runtime"]["observed"]["device_index"] == 0
    with pytest.raises(ValueError, match="runtime selection"):
        run_suite(
            source,
            "suite-run",
            generation,
            runs,
            backend="cpu",
            authorization=authorization,
        )


def test_authenticated_reference_reopens_underlying_evaluation(evaluated_run, tmp_path):
    source, runs = evaluated_run
    generation = min((runs / "suite-run/checkpoints").glob("step_*")).name
    original = run_suite(source, "suite-run", generation, runs, backend="cpu")
    verified = verify_evaluation_index(original)
    reference = tmp_path / "reference.json"
    publish_immutable(
        reference,
        {
            "format": "scientific-evidence-reference-v1",
            "kind": "evaluation_index",
            "sha256": verified["index_sha256"],
            "external_location": str(original),
        },
    )
    derived_suite = tmp_path / "referencing-suite.json"
    derived_suite.write_text(
        json.dumps(
            {
                "evaluation_suite_version": 1,
                "id": "reference",
                "evaluations": [
                    {
                        "id": "evidence",
                        "role": "descriptive",
                        "kind": "evidence_reference",
                        "source": "reference.json",
                    }
                ],
            }
        )
    )
    index = run_suite(derived_suite, "suite-run", generation, runs, backend="cpu")
    assert verify_evaluation_index(index)["evaluations"][0]["status"] == "COMPLETED"
    before = reference.read_bytes()
    try:
        reference.write_bytes(before.replace(b"evaluation_index", b"tokenizer"))
        with pytest.raises(ValueError, match="evaluation source changed"):
            verify_evaluation_index(index)
    finally:
        reference.write_bytes(before)


def test_sealed_single_reviewer_surface_is_verified_not_automated(
    evaluated_run, tmp_path
):
    source, runs = evaluated_run
    generation = min((runs / "suite-run/checkpoints").glob("step_*")).name
    selected = verify_evaluation_index(
        run_suite(source, "suite-run", generation, runs, backend="cpu")
    )
    bundle = tmp_path / "sealed"
    create_surface_bundle(
        cells(1),
        profile="full",
        selection_seed=23,
        presentation_seed=42,
        output_dir=bundle,
        source_artifacts=[{"checkpoint_sha256": selected["checkpoint_sha256"]}],
    )
    case = open_surface_bundle(bundle)["blind"]["cases"][0]
    record_surface_judgment(
        bundle,
        case["blind_case_id"],
        {dimension: "A" for dimension in case["dimensions"]},
        [],
        "2026-09-28T12:00:00Z",
    )
    complete_surface_review(bundle)
    reveal_surface_review(bundle)
    derived_suite = tmp_path / "review-suite.json"
    derived_suite.write_text(
        json.dumps(
            {
                "evaluation_suite_version": 1,
                "id": "review",
                "evaluations": [
                    {
                        "id": "human",
                        "role": "blinded_surface",
                        "kind": "surface_review",
                        "source": "sealed",
                    }
                ],
            }
        )
    )
    index = run_suite(derived_suite, "suite-run", generation, runs, backend="cpu")
    assert verify_evaluation_index(index)["evaluations"][0]["status"] == "COMPLETED"
    wrong = max((runs / "suite-run/checkpoints").glob("step_*")).name
    if wrong != generation:
        with pytest.raises(ValueError, match="checkpoint binding"):
            run_suite(derived_suite, "suite-run", wrong, runs, backend="cpu")


@pytest.mark.parametrize("completed_review", [False, True])
def test_pending_surface_review_preserves_completed_numeric_gate(
    evaluated_run, tmp_path, completed_review
):
    _, runs = evaluated_run
    generation = min((runs / "suite-run/checkpoints").glob("step_*"))
    checkpoint_sha = json.loads((generation / "manifest.json").read_text())["sha256"]
    bundle = tmp_path / "pending"
    create_surface_bundle(
        cells(1),
        profile="full",
        selection_seed=23,
        presentation_seed=42,
        output_dir=bundle,
        source_artifacts=[{"checkpoint_sha256": checkpoint_sha}],
    )
    if completed_review:
        case = open_surface_bundle(bundle)["blind"]["cases"][0]
        record_surface_judgment(
            bundle,
            case["blind_case_id"],
            {dimension: "A" for dimension in case["dimensions"]},
            [],
            "2026-09-28T12:00:00Z",
        )
        complete_surface_review(bundle)
    source = tmp_path / "suite.json"
    source.write_text(
        json.dumps(
            {
                "evaluation_suite_version": 1,
                "id": "pending",
                "evaluations": [
                    {"id": "loss", "role": "gate", "kind": "heldout_lm"},
                    {
                        "id": "review",
                        "role": "blinded_surface",
                        "kind": "surface_review",
                        "source": "pending",
                    },
                ],
            }
        )
    )
    index = verify_evaluation_index(
        run_suite(source, "suite-run", generation.name, runs, backend="cpu")
    )
    assert [row["status"] for row in index["evaluations"]] == [
        "COMPLETED",
        "SKIPPED_REVIEW",
    ]
    if completed_review:
        (bundle / "review.json").write_text("{}")
        with pytest.raises(ValueError):
            run_suite(source, "suite-run", generation.name, runs, backend="cpu")


def test_suite_signed_replay_reuses_payload_and_preserves_canonical_index(
    evaluated_run, tmp_path, monkeypatch
):
    from sparselab.data import verification
    from sparselab.verification_proofs import ProofStore

    source, runs = evaluated_run
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    options = {
        "proof_store": ProofStore(source.parent),
        "verification_mode": "verified_reuse",
    }
    index_path = run_suite(source, "suite-run", "latest.json", runs, backend="cpu")
    original_index = index_path.read_bytes()
    run_suite(source, "suite-run", "latest.json", runs, backend="cpu", **options)
    hashed_payloads = []
    original_hasher = verification.manifest_module.sha256_file

    def counted(path, *args, **kwargs):
        if Path(path).suffix in {".npy", ".safetensors"}:
            hashed_payloads.append(str(path))
        return original_hasher(path, *args, **kwargs)

    monkeypatch.setattr(verification.manifest_module, "sha256_file", counted)
    replay = run_suite(
        source, "suite-run", "latest.json", runs, backend="cpu", **options
    )
    verify_evaluation_index(replay, **options)
    assert hashed_payloads == [], options["proof_store"].diagnostics()
    assert replay.read_bytes() == original_index
    from sparselab.training.checkpoints import CheckpointManager

    checkpoint_root = runs / "suite-run" / "checkpoints"
    rejected = CheckpointManager(checkpoint_root).verify(
        checkpoint_root / "latest.json",
        expected_manifest="0" * 64,
        **options,
    )
    assert rejected.valid is False
    assert {"field": "manifest_sha256", "reason": "mismatch"} in rejected.errors


def test_group_write_is_trusted_only_for_a_user_private_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ubuntu's umask 002 must not silently disable signed caches."""
    import grp
    import os

    from sparselab import verification_proofs

    directory = tmp_path / "work"
    directory.mkdir()
    real = grp.getgrgid(os.getgid())
    for members, mode, trusted in (
        ([], 0o775, True),
        (["someone-else"], 0o775, False),
        ([], 0o777, False),
    ):
        verification_proofs._private_group.cache_clear()
        monkeypatch.setattr(
            grp,
            "getgrgid",
            lambda gid, members=members: grp.struct_group(
                (real.gr_name, real.gr_passwd, real.gr_gid, members)
            ),
        )
        monkeypatch.setattr(
            verification_proofs.pwd,
            "getpwuid",
            lambda uid: type("User", (), {"pw_name": real.gr_name}),
        )
        directory.chmod(mode)
        assert verification_proofs._trusted(tmp_path, directory) is trusted
    verification_proofs._private_group.cache_clear()


def test_context_memory_fixed_panels_preserve_coverage(
    fixed_panel_run, tmp_path, monkeypatch
):
    from sparselab.config.models import RunConfig
    from sparselab.evaluation.evidence import experiment_evidence

    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))
    values = _panel_run_config(fixed_panel_run, tmp_path).model_dump(mode="json")
    values["model"].update(
        memory="ngram",
        memory_variant="context_key",
        memory_injection="after_block",
        memory_block_index=0,
        memory_table_size=31,
        memory_ngram_size=3,
        memory_dim=4,
        memory_ngram_orders=[2, 3],
        memory_hash_heads=2,
        memory_conv_kernel_size=4,
    )
    measured = RunConfig.model_validate(values)
    train(measured, run_id="context-memory-panels")
    run = measured.logging.root_dir / "context-memory-panels"
    evidence = experiment_evidence(run)
    assert evidence["evidence_level"] == "checkpointed_held_out"
    assert not evidence["rejected_reports"]
    reports = list((run / "evaluations").glob("validation_step_*_gen_*.json"))
    assert reports
    for path in reports:
        report = json.loads(path.read_text())
        assert report["valid_targets"] == 96 and report["batches"] == 4
    # A sidecar-addressed mode still cannot consume token-only fixed panels.
    values["model"].update(
        memory="byte",
        memory_variant="hidden_gate",
        memory_injection="final",
        memory_block_index=None,
        memory_hash_heads=1,
        memory_ngram_orders=[],
        memory_conv_kernel_size=0,
    )
    with pytest.raises(ValueError, match="fixed panels require"):
        RunConfig.model_validate(values)
