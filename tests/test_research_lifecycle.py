from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from klab.research.lifecycle import (
    Baseline,
    LifecycleRegistry,
    _baseline_scoped_diagnostics,
    _validate_chat_transcript_capture,
    _validate_generation_capture,
    next_experiments,
    prior_evidence_warnings,
    validate_lifecycle,
)
from klab.training.manifest import sha256_file

_ENTRY = "catalog/neutral-study.json"


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _rows(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    return [_mapping(item) for item in value]


def _root(**updates: object) -> dict[str, object]:
    return {
        "format": "klab-research-lifecycle",
        "version": 1,
        "evidence": [],
        "baselines": [],
        "entries": [],
        "findings": [],
        "promotions": [],
        **updates,
    }


def _evidence(path: Path, evidence_id: str = "history") -> dict[str, object]:
    return {
        "id": evidence_id,
        "kind": "document",
        "artifact": {
            "relative_path": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        },
    }


def _finding(evidence_id: str, *, disposition: str = "learning") -> dict[str, object]:
    return {
        "id": "prior-finding",
        "entry": _ENTRY,
        "recorded_at": "2026-09-23",
        "evidence_ids": [evidence_id],
        "tested_conditions": ["CPU, seed 17, held-out aliases"],
        "observation": "The bounded report contains the declared outcomes.",
        "supported_claims": ["The recorded outcome applies to this task and seed."],
        "unsupported_claims": ["This does not establish general lexical transfer."],
        "disposition": disposition,
        "decided_by": "research reviewer",
        "rationale": "Retain the bounded evidence without scaling the claim.",
        "next_action": "Run an independent replication before increasing compute.",
        "reopen_conditions": [
            "A new independent seed or task population is available."
        ],
        "design_identity": {"study_sha256": "a" * 64, "protocol_sha256": None},
    }


def _baseline() -> Baseline:
    artifact = {
        "relative_path": "configs/reference.yaml",
        "sha256": "b" * 64,
        "size_bytes": 10,
    }
    return Baseline.model_validate(
        {
            "id": "dense-small-v1",
            "title": "Dense reference",
            "purpose": "Bounded CPU integration capture.",
            "status": "candidate",
            "configuration": artifact,
            "tokenizer_configuration": {
                **artifact,
                "relative_path": "configs/tokenizer.yaml",
            },
            "profile": {
                "scale": "smoke",
                "data": "offline",
                "note": "Explicit bounded integration profile.",
            },
            "seeds": [42],
            "budget": {"max_steps": 1536, "max_tokens": 100_000},
            "source_revision": None,
            "required_evidence": {},
            "capability_expectations": [
                {
                    "card": "chat-alias-retention-v1",
                    "role": "acquisition",
                    "expectation": "At least half of 24 cases pass.",
                }
            ],
            "limitations": ["One seed does not establish replication."],
            "integration_study": {
                **artifact,
                "relative_path": "configs/study.yaml",
            },
            "reproduction": "configs/study.yaml",
        }
    )


def test_baseline_diagnostic_scope_keeps_unrelated_registry_errors_global() -> None:
    baseline_data = _baseline().model_dump(mode="json")
    baseline_data["required_evidence"] = {"training": ["dense-run-evidence"]}
    registry = LifecycleRegistry.model_validate(_root(baselines=[baseline_data]))
    diagnostics = [
        {
            "severity": "error",
            "code": "candidate_error",
            "message": "Candidate-specific failure.",
            "reference_ids": ["dense-small-v1", "dense-run-evidence"],
        },
        {
            "severity": "error",
            "code": "unrelated_error",
            "message": "Unrelated historical finding.",
            "reference_ids": ["learned-engram-portability-v1"],
        },
    ]

    baseline, scoped = _baseline_scoped_diagnostics(
        registry, "dense-small-v1", diagnostics
    )

    assert baseline.id == "dense-small-v1"
    assert [item["code"] for item in scoped] == ["candidate_error"]
    assert [item["code"] for item in diagnostics] == [
        "candidate_error",
        "unrelated_error",
    ]


def test_learning_reference_requires_fixed_32_token_generation_panel() -> None:
    baseline_data = _baseline().model_dump(mode="json")
    baseline_data["purpose_classification"] = "learning_reference"
    baseline_data["capability_expectations"] = []
    baseline = Baseline.model_validate(baseline_data)
    report_identity = {
        "checkpoint_sha256": "c" * 64,
        "tokenizer_sha256": "d" * 64,
        "runtime": {"backend": "rocm"},
        "config": {"model": {"max_seq_len": 128}},
    }
    capture = {
        "format": "klab-generation-capture",
        "version": 1,
        "identity": {
            "run_id": "run-42",
            "checkpoint_sha256": "c" * 64,
            "tokenizer_sha256": "d" * 64,
            "backend": "rocm",
        },
        "options": {
            "max_new_tokens": 32,
            "temperature": 0.0,
            "top_k": 0,
            "seed": 42042,
        },
        "outputs": [{"prompt": "Prompt:", "output": "Prompt: fixed output"}],
    }
    diagnostics: list[dict[str, object]] = []

    _validate_generation_capture(
        capture, baseline, report_identity, "run-42", diagnostics
    )

    assert diagnostics == []
    capture["options"]["seed"] = 0
    _validate_generation_capture(
        capture, baseline, report_identity, "run-42", diagnostics
    )
    assert [item["code"] for item in diagnostics] == [
        "baseline_generation_options_invalid"
    ]


def test_generation_capture_binds_checkpoint_and_caps_fixed_greedy_options() -> None:
    baseline = _baseline()
    report_identity = {
        "checkpoint_sha256": "c" * 64,
        "tokenizer_sha256": "d" * 64,
        "source_identity_sha256": "f" * 64,
        "data_sha256": {"train": "a" * 64, "validation": "b" * 64},
        "runtime": {"backend": "cpu"},
        "config": {"model": {"max_seq_len": 128}},
    }
    capture = {
        "format": "klab-generation-capture",
        "version": 1,
        "identity": {
            "run_id": "run-42",
            "checkpoint_sha256": "c" * 64,
            "tokenizer_sha256": "d" * 64,
            "backend": "cpu",
        },
        "options": {
            "max_new_tokens": 16,
            "temperature": 0.0,
            "top_k": 0,
            "seed": 0,
        },
        "outputs": [{"prompt": "Recall:", "output": "Recall: a bounded answer"}],
    }
    diagnostics: list[dict[str, object]] = []
    capture_identity = cast(dict[str, object], capture["identity"])
    capture_options = cast(dict[str, object], capture["options"])

    _validate_generation_capture(
        capture, baseline, report_identity, "run-42", diagnostics
    )

    transcript = {
        "format": "chat_transcript_v1",
        "identity": {
            "run_id": "run-42",
            "checkpoint_sha256": "c" * 64,
            "tokenizer_sha256": "d" * 64,
            "source_identity_sha256": "f" * 64,
            "data_sha256": {"train": "a" * 64, "validation": "b" * 64},
            "config": {"model": {"max_seq_len": 128}},
            "training_runtime": {"backend": "cpu"},
        },
        "generation": capture_options,
        "turns": [
            {
                "user": "Recall?",
                "prompt": "User: Recall?\\nAssistant:",
                "response": " answer",
                "prompt_tokens": 64,
            }
        ],
    }
    _validate_chat_transcript_capture(
        transcript, report_identity, "run-42", capture_options, baseline, diagnostics
    )
    assert diagnostics == []

    invalid_transcript = {
        **transcript,
        "turns": [
            {
                **cast(list[dict[str, object]], transcript["turns"])[0],
                "prompt_tokens": 121,
            }
        ],
    }
    _validate_chat_transcript_capture(
        invalid_transcript,
        report_identity,
        "run-42",
        capture_options,
        baseline,
        diagnostics,
    )
    assert [item["code"] for item in diagnostics] == [
        "baseline_chat_transcript_turns_invalid"
    ]
    diagnostics.clear()

    invalid = {
        **capture,
        "identity": {**capture_identity, "checkpoint_sha256": "e" * 64},
        "options": {**capture_options, "max_new_tokens": 33, "seed": 1},
        "outputs": [{"prompt": "Recall:", "output": "wrong prompt binding"}],
    }
    _validate_generation_capture(
        invalid, baseline, report_identity, "run-42", diagnostics
    )

    assert {item["code"] for item in diagnostics} == {
        "baseline_generation_identity_mismatch",
        "baseline_generation_options_invalid",
        "baseline_generation_outputs_invalid",
    }


def test_lifecycle_json_rejects_duplicate_nonfinite_and_non_integer_versions(
    tmp_path: Path,
) -> None:
    from klab.research.lifecycle import load_lifecycle

    cases = (
        (
            '{"format":"klab-research-lifecycle","version":1,"version":1}',
            "duplicate JSON key",
        ),
        (
            '{"format":"klab-research-lifecycle","version":true,"evidence":[],"baselines":[],"entries":[],"findings":[],"promotions":[]}',
            "version must be integer 1",
        ),
        (
            '{"format":"klab-research-lifecycle","version":1,"evidence":[],"baselines":[],"entries":[],"findings":[],"promotions":[],"extra":NaN}',
            "nonfinite JSON number",
        ),
    )
    for index, (content, message) in enumerate(cases):
        path = tmp_path / f"lifecycle-{index}.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_lifecycle(path)


def test_evidence_status_distinguishes_verified_missing_tampered_and_symlink(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    artifact = evidence_root / "history.md"
    artifact.write_text("bounded historical observation\n", encoding="utf-8")
    verified = LifecycleRegistry.model_validate(_root(evidence=[_evidence(artifact)]))
    report = validate_lifecycle(verified, evidence_root=evidence_root)
    assert _mapping(_mapping(report["availability"])["history"])["status"] == "verified"

    artifact.write_text("changed after identity capture\n", encoding="utf-8")
    tampered = validate_lifecycle(verified, evidence_root=evidence_root)
    assert (
        _mapping(_mapping(tampered["availability"])["history"])["status"] == "invalid"
    )
    assert any(
        item["code"] == "evidence_invalid" for item in _rows(tampered["diagnostics"])
    )

    artifact.unlink()
    missing = validate_lifecycle(verified, evidence_root=evidence_root)
    assert (
        _mapping(_mapping(missing["availability"])["history"])["status"]
        == "unavailable"
    )

    outside = tmp_path / "outside.md"
    outside.write_text("never dereference a linked artifact\n", encoding="utf-8")
    (evidence_root / "linked.md").symlink_to(outside)
    linked = LifecycleRegistry.model_validate(
        _root(
            evidence=[
                _evidence(outside, "linked")
                | {
                    "artifact": {
                        "relative_path": "linked.md",
                        "sha256": sha256_file(outside),
                        "size_bytes": outside.stat().st_size,
                    }
                }
            ]
        )
    )
    rejected = validate_lifecycle(linked, evidence_root=evidence_root)
    assert _mapping(_mapping(rejected["availability"])["linked"])["status"] == "invalid"
    assert any(
        item["code"] == "evidence_path_invalid"
        for item in _rows(rejected["diagnostics"])
    )


def test_default_registry_is_empty_and_grants_no_scientific_clearance(tmp_path: Path):
    from klab.research.lifecycle import load_lifecycle, research_status

    registry = load_lifecycle()
    assert not registry.entries
    assert not registry.evidence
    assert not registry.baselines
    assert not registry.findings
    assert not registry.promotions
    status = research_status(registry, evidence_root=tmp_path)
    assert status["findings"] == status["promotions"] == status["baselines"] == []
    assert all(not row["assessed"] for row in status["entries"])
    assert next_experiments(registry, evidence_root=tmp_path)["items"] == []


def test_explicit_registry_evidence_is_selected_and_missing_registry_refuses(
    tmp_path: Path,
):
    from klab.research.lifecycle import load_lifecycle

    artifact = tmp_path / "note.txt"
    artifact.write_text("Synthetic historical evidence, not a current claim.\n")
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(_root(evidence=[_evidence(artifact)])))
    registry = load_lifecycle(path)
    status = validate_lifecycle(registry, evidence_root=tmp_path)
    assert status["availability"]["history"]["status"] == "verified"
    artifact.write_text("changed")
    assert (
        validate_lifecycle(registry, evidence_root=tmp_path)["availability"]["history"][
            "status"
        ]
        == "invalid"
    )
    path.unlink()
    with pytest.raises(ValueError, match="unavailable"):
        load_lifecycle(path)


def test_next_replicate_gate_does_not_treat_documentation_as_measurement(
    tmp_path: Path,
) -> None:
    from research_fixtures import make_resources

    evidence_root = make_resources(tmp_path)
    evidence = evidence_root / "history.md"
    evidence.write_text("A source note, not measured behavior.\n", encoding="utf-8")
    finding = _finding("history")
    entry_path = evidence_root / _ENTRY

    entry = {
        "entry": _ENTRY,
        "entry_sha256": sha256_file(entry_path),
        "baseline_id": "reference-not-yet-established",
        "stage": "replicate",
        "maturity": {},
        "finding_ids": ["prior-finding"],
        "next_test": {
            "action": "Run the same frozen card on an independent seed.",
            "stage": "replicate",
            "cost_class": "short",
            "prerequisites": [],
            "blocker_ids": [],
            "eligible_scales": ["nano"],
            "reopens": [],
            "design_change": None,
        },
        "blockers": [],
    }
    registry = LifecycleRegistry.model_validate(
        _root(
            evidence=[_evidence(evidence)],
            findings=[finding],
            entries=[entry],
        )
    )

    view = next_experiments(registry, evidence_root=evidence_root)

    items = _rows(view["items"])
    assert len(items) == 1
    item = items[0]
    assert item["status"] == "blocked"
    assert any(
        "measured finding" in reason for reason in cast(list[str], item["reasons"])
    )

    acceptance_path = evidence_root / "acceptance.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "format": "astra-scientific-acceptance-v1",
                "accepted": True,
                "source_identity_sha256": "a" * 64,
                "endpoints": [
                    {
                        "run_id": "run-1",
                        "checkpoint_sha256": "b" * 64,
                        "step": 10,
                        "targets": 100,
                        "files_verified": 2,
                    },
                    {
                        "run_id": "run-2",
                        "checkpoint_sha256": "c" * 64,
                        "step": 10,
                        "targets": 100,
                        "files_verified": 2,
                    },
                ],
                "card_reports": [
                    {
                        "run_id": run_id,
                        "card": "held-out",
                        "cases": 2,
                        "passed": 1,
                        "path": f"runs/{run_id}/card.json",
                        "result_digest": "d" * 64,
                        "sha256": "e" * 64,
                    }
                    for run_id in ("run-1", "run-2")
                ],
                "paired_deltas_verified": [
                    {"left": "run-1", "right": "run-2", "verified": True}
                ],
            }
        ),
        encoding="utf-8",
    )
    measured_registry = LifecycleRegistry.model_validate(
        _root(
            evidence=[_evidence(acceptance_path) | {"kind": "acceptance"}],
            findings=[finding],
            entries=[entry],
        )
    )
    measured_items = _rows(
        next_experiments(measured_registry, evidence_root=evidence_root)["items"]
    )
    assert measured_items[0]["status"] == "ready"
    assert measured_items[0]["reasons"] == []


def test_prior_design_warning_matches_exact_identity_and_includes_positive_findings() -> (
    None
):
    findings = [
        _finding("report", disposition="promote"),
        {
            **_finding("report", disposition="learning"),
            "id": "older-finding",
            "recorded_at": "2026-09-20",
        },
    ]
    registry = LifecycleRegistry.model_validate(_root(findings=findings))

    warnings = prior_evidence_warnings(registry, study_sha256="a" * 64)

    assert [warning["finding_id"] for warning in warnings] == [
        "older-finding",
        "prior-finding",
    ]
    assert prior_evidence_warnings(registry, protocol_sha256="b" * 64) == []
    with pytest.raises(ValueError, match="provide study_sha256 or protocol_sha256"):
        prior_evidence_warnings(
            registry, study_sha256="a" * 64, protocol_sha256="b" * 64
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        prior_evidence_warnings(registry, study_sha256="not-a-digest")


def test_lessons_do_not_depend_on_personal_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from klab.research import lifecycle
    from klab.research.catalog import list_lessons
    from klab.research.scaffold import scaffold_lesson

    empty_resources = tmp_path / "empty-resources"
    empty_resources.mkdir()
    monkeypatch.setattr(lifecycle, "_RESOURCE", empty_resources)
    with pytest.raises(ValueError, match="lifecycle metadata is unavailable"):
        lifecycle.load_lifecycle()
    assert len(list_lessons()) == 11
    output = scaffold_lesson("dense", tmp_path / "lesson", scale="smoke")
    assert (output / "model.yaml").is_file()
    assert not (output / "artifacts").exists()
