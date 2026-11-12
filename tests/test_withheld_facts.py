from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from klab.config.models import DatasetConfig
from klab.data.datasets import iter_documents
from klab.data.withheld_facts import (
    audit_manifest,
    evaluation_cases,
    split_facts,
    training_documents,
    verify_manifest,
    write_manifest,
)


def test_withheld_facts_are_disjoint_by_identity_and_value() -> None:
    train, held_out = split_facts(3)
    assert len(train) == 6
    assert len(held_out) == 2
    assert not {fact.key for fact in train} & {fact.key for fact in held_out}
    assert not {fact.value for fact in held_out} & set(
        " ".join(training_documents(3)).split()
    )


def test_evaluation_prompts_exclude_held_out_values() -> None:
    for prompt, value in evaluation_cases(5):
        assert value not in prompt
        assert prompt.endswith(" is")


def test_manifest_is_canonical_and_conflict_safe(tmp_path: Path) -> None:
    path = tmp_path / "facts.json"
    write_manifest(path, seed=2)
    manifest = json.loads(path.read_text())
    digest = manifest.pop("sha256")
    canonical = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    assert digest == hashlib.sha256(canonical).hexdigest()
    assert manifest["training_statements"] == list(training_documents(2))
    assert [
        (case["prompt"], case["expected_value"]) for case in manifest["held_out_cases"]
    ] == list(evaluation_cases(2))
    write_manifest(path, seed=2)
    with pytest.raises(FileExistsError, match="conflicting"):
        write_manifest(path, seed=3)


def test_manifest_verifier_rejects_corruption_and_substitution(tmp_path: Path) -> None:
    path = tmp_path / "facts.json"
    write_manifest(path, seed=2)
    assert verify_manifest(path)["seed"] == 2
    manifest = json.loads(path.read_text())
    manifest["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_manifest(path)

    manifest = json.loads(path.read_text())
    manifest["training_statements"][0] = "Altered fact."
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    manifest["sha256"] = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match fixture"):
        verify_manifest(path)


def test_manifest_audit_reports_verified_split_evidence(tmp_path: Path) -> None:
    path = tmp_path / "facts.json"
    write_manifest(path, seed=6)
    audit = audit_manifest(path)
    assert audit == {
        "format_version": 1,
        "held_out_case_count": 2,
        "held_out_values_absent_from_training": True,
        "seed": 6,
        "sha256": verify_manifest(path)["sha256"],
        "training_statement_count": 6,
        "valid": True,
    }


def test_withheld_fact_source_never_yields_evaluation_values(tmp_path: Path) -> None:
    config = DatasetConfig(
        source="synthetic",
        synthetic_task="withheld_facts",
        cache_dir=tmp_path,
        train_max_documents=6,
        validation_max_documents=6,
        train_max_tokens=256,
        validation_max_tokens=256,
        synthetic_seed=4,
    )
    documents = tuple(iter_documents(config, "train"))
    assert documents == tuple(iter_documents(config, "validation"))
    assert all(value not in " ".join(documents) for _, value in evaluation_cases(4))
