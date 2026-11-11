"""Tracked tiny fixtures exercise offline lint without external scientific bytes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from klab.research.lint import lint_research


def _write(root: Path, name: str, content: str | dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content) if isinstance(content, dict) else content)
    return path


def _track(root: Path) -> None:
    subprocess.run(
        ["git", "-C", str(root), "add", "-A"], check=True, capture_output=True
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    campaign = "experiments/research/demo"
    protocol = _write(
        tmp_path, f"{campaign}/protocol.md", "Frozen question and acceptance gates.\n"
    )
    config = _write(tmp_path, "configs/demo.yaml", "seed: 42\n")
    _write(
        tmp_path,
        f"{campaign}/preregistration.json",
        {
            "frozen_sha256": {
                f"{campaign}/protocol.md": _digest(protocol),
                "configs/demo.yaml": _digest(config),
            },
        },
    )
    _write(
        tmp_path,
        f"{campaign}/checkpoint-reference.json",
        {
            "format": "scientific-evidence-reference-v1",
            "kind": "checkpoint",
            "sha256": "a" * 64,
            "identifier": "step_1",
            "source_commit": "b" * 40,
            "declaration_hashes": [
                {"path": f"{campaign}/protocol.md", "sha256": _digest(protocol)}
            ],
            "external_location": "/unavailable/checkpoint",
            "verification_scope": "verified_artifact",
            "verified_at": "2026-10-05T00:00:00Z",
        },
    )
    _track(tmp_path)
    return tmp_path


def _errors(root: Path) -> str:
    report = lint_research(root)
    assert report["valid"] is False
    return json.dumps(report["errors"])


def test_empty_research_checkout_is_valid(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    assert lint_research(tmp_path)["valid"]


def test_valid_offline_research_records_and_untracked_outputs(repository: Path) -> None:
    _write(repository, "experiments/research/demo/runs/untracked.log", "not checked in")
    report = lint_research(repository)
    assert report["valid"] is True
    assert report["checked_files"] == 3
    assert "external payloads not verified" in report["scope"]


@pytest.mark.parametrize("target", ["protocol.md", "../../../configs/demo.yaml"])
def test_missing_or_changed_frozen_references(repository: Path, target: str) -> None:
    path = repository / "experiments/research/demo" / target
    path.write_text("Changed after freeze")
    assert "digest mismatch" in _errors(repository)
    path.unlink()
    assert "missing tracked reference" in _errors(repository)


def test_untracked_reference_is_missing(repository: Path) -> None:
    path = _write(repository, "experiments/research/demo/untracked.md", "review")
    _write(
        repository,
        "experiments/research/demo/preregistration.json",
        {
            "frozen_sha256": {"untracked.md": _digest(path)},
        },
    )
    assert "missing tracked reference" in _errors(repository)


@pytest.mark.parametrize(
    "name",
    [
        "checkpoints/state.json",
        "cache/key.json",
        "datasets/train.txt",
        "logs/train.txt",
        "runs/result.json",
        "outputs/stats.json",
        "state.pt",
        "state.safetensors",
        "tokens.npy",
        "train.parquet",
        "run.sqlite",
        "train.log",
        "metrics.jsonl",
        "events.out.tfevents.123",
        "nested/__pycache__/code.pyc",
    ],
)
def test_forbidden_tracked_output(repository: Path, name: str) -> None:
    _write(repository, f"experiments/research/demo/{name}", "{}")
    _track(repository)
    assert "forbidden mutable output" in _errors(repository)


@pytest.mark.parametrize(
    "name",
    [
        "../outside.md",
        "/unavailable/protocol.md",
        "a/../../secret",
        "a\\b",
        "./protocol.md",
    ],
)
def test_unsafe_binding(repository: Path, name: str) -> None:
    _write(
        repository,
        "experiments/research/demo/preregistration.json",
        {"frozen_sha256": {name: "a" * 64}},
    )
    assert "unsafe repository reference" in _errors(repository)


def test_symlink_reference_never_reads_external_target(
    repository: Path, tmp_path: Path
) -> None:
    target = repository / "configs/demo.yaml"
    target.unlink()
    target.symlink_to("/unavailable/private-config.yaml")
    assert "symlink" in _errors(repository)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", "bad"),
        ("declaration_hashes", []),
        ("source_commit", "main"),
        ("kind", "unknown"),
        ("verification_scope", "observation"),
        ("external_location", ""),
        ("format", "scientific-evidence-reference-v2"),
        ("format", None),
    ],
)
def test_malformed_evidence_reference(
    repository: Path, field: str, value: object
) -> None:
    path = repository / "experiments/research/demo/checkpoint-reference.json"
    raw = json.loads(path.read_text())
    raw[field] = value
    path.write_text(json.dumps(raw))
    assert _errors(repository)


def test_evidence_declaration_mismatch(repository: Path) -> None:
    path = repository / "experiments/research/demo/checkpoint-reference.json"
    raw = json.loads(path.read_text())
    raw["declaration_hashes"][0]["sha256"] = "c" * 64
    path.write_text(json.dumps(raw))
    assert "reference digest mismatch" in _errors(repository)


def test_explicit_scientific_bindings(repository: Path) -> None:
    path = repository / "experiments/research/demo/preregistration.json"
    path.write_text(
        json.dumps(
            {"bindings": {"protocol": {"path": "protocol.md", "sha256": "a" * 64}}}
        )
    )
    assert "reference digest mismatch" in _errors(repository)


def test_plan_schema_and_config_closure(repository: Path) -> None:
    path = _write(
        repository,
        "experiments/research/demo/plan.yaml",
        "plan_version: 1\nid: demo\nbase_run: missing.yaml\n",
    )
    _track(repository)
    assert "missing tracked reference" in _errors(repository)
    path.write_text("plan_version: 999\nid: demo\nbase_run: missing.yaml\n")
    assert "invalid experiment plan" in _errors(repository)


def test_historical_implementation_inventory_is_not_a_current_binding(
    repository: Path,
) -> None:
    _write(
        repository,
        "experiments/research/demo/historical.json",
        {
            "implementation": [{"path": "src/old.py", "sha256": "c" * 64}],
            "archive_inventory": [
                {"path": "declarations/old.json", "sha256": "d" * 64}
            ],
        },
    )
    _track(repository)
    assert lint_research(repository)["valid"]


@pytest.mark.parametrize("name", ["train.jsonl", "evidence/train.jsonl"])
def test_arbitrary_dataset_in_evidence_directory_rejected(
    repository: Path, name: str
) -> None:
    _write(
        repository,
        f"experiments/research/demo/{name}",
        '{"text":"training document"}\n',
    )
    _track(repository)
    assert "JSONL payloads belong outside active research records" in _errors(
        repository
    )


def test_large_record_rejected(repository: Path) -> None:
    _write(repository, "experiments/research/demo/huge.md", "x" * (1024 * 1024 + 1))
    _track(repository)
    assert "1 MiB" in _errors(repository)


def test_duplicate_json_keys_rejected(repository: Path) -> None:
    _write(
        repository,
        "experiments/research/demo/preregistration.json",
        '{"frozen_sha256":{},"frozen_sha256":{}}',
    )
    assert "duplicate" in _errors(repository)


def test_actual_tracked_research_records() -> None:
    root = Path(__file__).resolve().parents[1]
    report = lint_research(root)
    assert report["valid"], report["errors"]
    assert report["checked_files"] == 1  # Only the Workshop/history navigation README.


def test_cli_failure_exit_code(repository: Path) -> None:
    _write(repository, "experiments/research/demo/protocol.md", "changed")
    import argparse

    from klab.research.lint import _handle

    with pytest.raises(SystemExit, match="1"):
        _handle(argparse.Namespace(repository=str(repository), json=True))


@pytest.fixture
def archived_repository(repository: Path) -> Path:
    archive = "experiments/research/history/demo/preregistration.json"
    record = _write(
        repository,
        archive,
        {"frozen_sha256": {"old/missing/config.yaml": "a" * 64}},
    )
    _write(
        repository,
        "experiments/research/history/path-map.json",
        {
            "format": "research-history-path-map-v1",
            "source_commit": "b" * 40,
            "entries": [
                {
                    "original_path": "experiments/research/demo/old.json",
                    "archive_path": archive,
                    "action": "relocated_unchanged",
                    "sha256": _digest(record),
                    "bytes": record.stat().st_size,
                },
                {
                    "original_path": "experiments/research/demo/old-runner.py",
                    "archive_path": None,
                    "action": "deleted_obsolete_executable",
                    "sha256": "c" * 64,
                    "bytes": 123,
                },
            ],
        },
    )
    _write(repository, "experiments/research/history/README.md", "Historical evidence")
    _track(repository)
    return repository


def test_archived_records_preserve_bytes_without_replaying_old_references(
    archived_repository: Path,
) -> None:
    report = lint_research(archived_repository)
    assert report["valid"], report["errors"]


@pytest.mark.parametrize("change", ["changed", "missing", "symlink", "unindexed"])
def test_archived_records_fail_closed(archived_repository: Path, change: str) -> None:
    record = (
        archived_repository / "experiments/research/history/demo/preregistration.json"
    )
    if change == "changed":
        record.write_text(record.read_text().replace("config", "CONFIG"))
    elif change == "missing":
        record.unlink()
    elif change == "symlink":
        record.unlink()
        record.symlink_to("/unavailable/private.json")
    else:
        record.with_name("unindexed.json").write_text("{}")
        _track(archived_repository)
    assert _errors(archived_repository)
