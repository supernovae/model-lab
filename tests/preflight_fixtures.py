"""A consistent synthetic pretraining setup for `klab preflight` tests.

The heavy verifiers (release, mixture, prepared data, stage bundle) are
replaced by stand-ins that return the artifacts written here; what the
preflight checks is how those artifacts agree with each other.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from klab.canonical import canonical_digest
from klab.config.loading import load_config
from klab.training.manifest import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "web_a": "web",
    "web_a_heldout": "web",
    "code_a": "code",
    "code_a_heldout": "code",
}
QUOTAS = {"web": 600, "code": 400}
GIB = 2**30


def _source_row(source: str, retained: int = 10) -> dict:
    return {
        "source_id": source,
        "files_seen": 4,
        "files_set_aside": 0,
        "file_set_aside_share": 0.0,
        "rows_set_aside": 0,
        "documents_built": retained,
        "dedup_dropped": {},
        "retained_documents": retained,
        "set_aside_reasons": [],
    }


def build(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    release = tmp / "release"
    mixture = tmp / "mixture"
    stage = tmp / "stage"
    for path in (release, mixture, stage / "assets" / "data", tmp / "tokenizer"):
        path.mkdir(parents=True)
    manifest = {
        "release_id": "r" * 64,
        "build_identity": {"release": {"policy": {"duplicates": {"keys": ["raw"]}}}},
    }
    (release / "manifest.json").write_text(json.dumps(manifest))
    scale = {
        s: {
            "train": {"documents": 10},
            "validation": {"documents": 3, "utf8_bytes": 30, "whitespace_words": 5},
        }
        for s in SOURCES
    }
    (release / "report.json").write_text(
        json.dumps(
            {
                "document_count": 52,
                "split_counts": {"train": 40, "validation": 12, "test": 0},
                "source_ledger": [_source_row(s) for s in SOURCES],
                "source_scale": scale,
            }
        )
    )
    (release / "sources.json").write_text(json.dumps([{"id": s} for s in SOURCES]))
    (tmp / "tokenizer" / "tokenizer.json").write_text("{}")
    tok_sha = sha256_file(tmp / "tokenizer" / "tokenizer.json")
    group = {"roundtrip_failures": 0, "unknown_tokens": 0}
    (tmp / "tokenizer" / "tokenizer_manifest.json").write_text(
        json.dumps(
            {
                "sha256": tok_sha,
                "vocab_size": 512,
                "sampling_audit": {
                    "complete": True,
                    "selected_bytes": 9,
                    "strata": {"web": group, "code": group},
                },
            }
        )
    )
    receipt = {
        "release_id": manifest["release_id"],
        "tokenizer_sha256": tok_sha,
        "records": 7,
        "requested_target_quotas": QUOTAS,
        "actual_target_tokens": dict(QUOTAS),
        "available_unique_positions": {"web": 700, "code": 400},
        "quota_shortfalls": {"web": 0, "code": 0},
        "max_exposures": 1,
    }
    (mixture / "receipt.json").write_text(json.dumps(receipt))
    declaration = tmp / "mixture.yaml"
    declaration.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "release_path": str(release),
                "tokenizer_config": str(tmp / "tokenizer.yaml"),
                "family_inventory": str(tmp / "families.jsonl"),
                "source_strata": SOURCES,
                "target_quotas": QUOTAS,
                "seed": 1,
                "max_exposures": 1,
            }
        )
    )
    (stage / "assets" / "data" / "manifest.json").write_text(
        '{"manifest_sha256": "p1"}'
    )

    config = load_config(ROOT / "configs/smoke_memory_cpu.yaml")
    config = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(
                update={
                    "source": "local_token_mixture",
                    "mixture_declaration_path": declaration,
                    "mixture_output_path": mixture,
                    "train_path": mixture / "train.tokens.ids",
                    "validation_path": release / "lm" / "validation.jsonl",
                    "validation_max_documents": 12,
                    "license": "test fixture",
                }
            ),
            "tokenizer": config.tokenizer.model_copy(
                update={"path": tmp / "tokenizer" / "tokenizer.json"}
            ),
            "model": config.model.model_copy(update={"vocab_size": 512}),
        }
    )
    pilot = {
        "purpose": "warmup",
        "sha256": "w" * 64,
        "report": {
            "sha256": "w" * 64,
            "update_observations": [{"targets": 1000, "update_seconds": 1.0}] * 5,
            "observed_peak_bytes": 20 * GIB,
            "peak_method": "native",
            "runtime": {"device_total_bytes": 32 * GIB, "device_name": "test card"},
        },
    }
    staged = {"sha256": "s" * 64, "through": "warmup", "pilot_reports": [pilot]}
    prepared = SimpleNamespace(
        root=tmp / "prepared",
        manifest={"manifest_sha256": "p1", "token_mixture": receipt},
        train_tokens=1000,
        validation=np.zeros(10, dtype=np.int32),
    )
    from klab import staging
    from klab.corpus import mixture as mixture_module
    from klab.corpus import release as release_module
    from klab.data import packing, tokenizer

    # Import consumers before replacing their dependencies. Their from-imports
    # must keep the real helpers after the fixture restores these attributes.
    monkeypatch.setattr(release_module, "verify_release", lambda *a, **k: manifest)
    monkeypatch.setattr(
        mixture_module, "verify_mixture_dataset", lambda *a, **k: receipt
    )
    monkeypatch.setattr(packing, "prepare_data", lambda *a, **k: prepared)
    monkeypatch.setattr(tokenizer, "load_tokenizer", lambda path: None)
    monkeypatch.setattr(staging, "verify_stage_bundle", lambda *a, **k: staged)

    report_path = tmp / "near-duplicates.json"
    near = {
        "release_id": manifest["release_id"],
        "document_count": 52,
        "coverage": [
            {"split": "train", "source_id": "web_a", "documents": 40},
            {"split": "validation", "source_id": "web_a_heldout", "documents": 12},
        ],
        "candidate_count": 1,
        "identical_text_count": 0,
        "candidates_truncated": False,
        "candidates": [{"left_document_id": "a", "right_document_id": "b"}],
        "parameters": {"template_aware": True},
        "template_only_count": 7,
        "template_only_document_count": 2,
    }
    report_path.write_text(json.dumps(near))
    decisions = tmp / "decisions.yaml"
    decisions.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "release_id": manifest["release_id"],
                "report_sha256": sha256_file(report_path),
                "accepted_by": "reviewer",
                "accepted_on": "2026-01-01",
                "decisions": [{"documents": ["a", "b"], "decision": "dropped"}],
            }
        )
    )
    return SimpleNamespace(
        config=config,
        release=release,
        mixture=mixture,
        stage=stage,
        receipt=receipt,
        manifest=manifest,
        report_path=release / "report.json",
        near_path=report_path,
        near=near,
        decisions=decisions,
        staged=staged,
        pilot=pilot,
        tokenizer_manifest=tmp / "tokenizer" / "tokenizer_manifest.json",
        digest=canonical_digest,
    )
