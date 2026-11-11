"""A real tiny CPU run carries verified frozen corpus evidence end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from klab.config.loading import load_config, load_tokenizer_config
from klab.corpus.export import export_release, verify_release_export
from klab.corpus.measurement import capability_matrix
from klab.data.packing import prepare_data
from klab.data.tokenizer import (
    load_tokenizer,
    train_tokenizer,
    verify_tokenizer_artifact,
)
from klab.evaluation.capabilities import (
    capability_card,
    evaluate_capability,
    write_capability_result,
)
from klab.evaluation.evidence import experiment_evidence
from klab.evaluation.inference import load_run
from klab.training.manifest import read_manifest
from klab.training.trainer import train

pytest_plugins = ("test_corpus_exports",)


def test_export_mismatch_fails_at_fit_verify_prepare_and_train(
    frozen_sample: tuple[Path, Path], tmp_path: Path
) -> None:
    release, root = frozen_sample
    first = export_release(
        release, "lm", Path("configs/runtime_smoke_cpu.yaml"), 300, root
    )
    second = export_release(
        release, "lm", Path("configs/runtime_smoke_cpu.yaml"), 301, root
    )
    assert first != second
    tokenizer_config = load_tokenizer_config(first / "tokenizer.yaml")
    artifact = train_tokenizer(tokenizer_config)
    config = load_config(first / "run.yaml")
    binding = verify_release_export(config.dataset)
    assert binding["train_sha256"] != binding["validation_sha256"]
    prepared = prepare_data(config, load_tokenizer(artifact))
    assert prepared.manifest["cache_identity"]["corpus_export"] == binding
    swapped = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(update={"corpus_export_path": second})
        }
    )
    with pytest.raises(ValueError, match="corpus export"):
        verify_tokenizer_artifact(
            artifact,
            source="local_text",
            revision=release.name,
            vocab_size=300,
            dataset=swapped.dataset,
        )
    with pytest.raises(ValueError, match="corpus export"):
        prepare_data(swapped, load_tokenizer(artifact))
    with pytest.raises(ValueError, match="corpus export"):
        train(swapped, run_id="bad-export")
    assert not (swapped.logging.root_dir / "bad-export").exists()
    altered_path = tmp_path / "train.jsonl"
    altered_path.write_bytes(config.dataset.train_path.read_bytes())
    changed = config.dataset.model_copy(update={"train_path": altered_path})
    with pytest.raises(ValueError, match="corpus export"):
        verify_release_export(changed)
    saved = (first / "run.yaml").read_bytes()
    try:
        (first / "run.yaml").write_bytes(saved + b"\n")
        with pytest.raises(ValueError, match="corpus export config changed"):
            verify_release_export(config.dataset)
        with pytest.raises(ValueError, match="existing corpus export has changed"):
            export_release(
                release, "lm", Path("configs/runtime_smoke_cpu.yaml"), 300, root
            )
    finally:
        (first / "run.yaml").write_bytes(saved)
    assert verify_release_export(config.dataset) == binding


def test_cpu_update_inventories_corpus_release(
    frozen_sample: tuple[Path, Path], tmp_path: Path
) -> None:
    release, root = frozen_sample
    exported = export_release(
        release, "lm", Path("configs/runtime_smoke_cpu.yaml"), 300, root
    )
    tokenizer_config = load_tokenizer_config(exported / "tokenizer.yaml")
    artifact = train_tokenizer(tokenizer_config)
    base = load_config(exported / "run.yaml")
    config = base.model_copy(
        update={
            "model": base.model.model_copy(update={"max_seq_len": 256}),
            "logging": base.logging.model_copy(update={"root_dir": tmp_path / "runs"}),
            "training": base.training.model_copy(
                update={"max_steps": 2, "max_tokens": 64, "gradient_accumulation": 1}
            ),
            "optimizer": base.optimizer.model_copy(
                update={"warmup_steps": 0, "decay_steps": None}
            ),
            "checkpoint": base.checkpoint.model_copy(update={"every_steps": 1}),
            "evaluation": base.evaluation.model_copy(
                update={"every_steps": 1, "max_batches": 1}
            ),
        }
    )
    run_id = train(config, run_id="forge-cpu", stop_after_step=1)
    run = config.logging.root_dir / run_id
    manifest = read_manifest(run / "manifest.json")
    names = {item["relative_path"] for item in manifest["artifacts"]}
    assert {
        "corpus/manifest.json",
        "corpus/report.json",
        "corpus/license-report.json",
        "corpus/audit.json",
        "corpus/export.json",
    } <= names
    assert manifest["effective_config"]["dataset"]["revision"] == release.name
    assert (
        json.loads((run / "data/manifest.json").read_text())["corpus_export"][
            "release_id"
        ]
        == release.name
    )
    evidence = experiment_evidence(run)
    assert evidence["corpus_release_sha256"] == release.name
    assert evidence["corpus_report_path"] == "corpus/report.json"
    assert evidence["verified_checkpoints"]
    assert artifact.is_file()
    assert capability_matrix(release, run) is None
    loaded = load_run(run_id, config.logging.root_dir)
    card = capability_card("engram-recall-v1")
    card_result = evaluate_capability(
        card,
        loaded.model,
        loaded.tokenizer,
        loaded.config.model.max_seq_len,
        loaded.device,
    )
    assert card_result["valid"] is True
    card_result["identity"] = loaded.identity
    write_capability_result(run, card_result)
    matrix = capability_matrix(release, run)
    assert matrix is not None
    assert matrix["release_id"] == release.name
    assert matrix["capabilities"][0]["card_sha256"] == card.digest
    assert matrix["transform_usefulness"][0]["state"] == "UNKNOWN"
    result_path = next((run / "evaluations").glob("capability-*.json"))
    saved_result = result_path.read_bytes()
    try:
        damaged = json.loads(saved_result)
        damaged["results"][0]["response"] += " fabricated"
        result_path.write_text(json.dumps(damaged))
        with pytest.raises(ValueError, match="unverified capability result"):
            capability_matrix(release, run)
    finally:
        result_path.write_bytes(saved_result)
    report = run / "corpus/report.json"
    saved = report.read_bytes()
    try:
        report.write_bytes(saved + b"\n")
        with pytest.raises(ValueError):
            experiment_evidence(run)
    finally:
        report.write_bytes(saved)
