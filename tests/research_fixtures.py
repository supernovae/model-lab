"""Neutral authoring declarations; metadata only, no scientific claims."""

import json
from pathlib import Path

import yaml


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_resources(tmp_path: Path) -> Path:
    root = tmp_path / "resources"
    root.mkdir()
    base = {
        "schema_version": 2,
        "name": "neutral-base",
        "seed": 5,
        "model": {
            "vocab_size": 260,
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "ffn_dim": 32,
            "max_seq_len": 16,
        },
        "tokenizer": {"path": "artifacts/tokenizer/tokenizer.json"},
        "dataset": {
            "source": "synthetic",
            "synthetic_task": "chat_recall",
            "cache_dir": "artifacts/data",
            "train_max_documents": 2,
            "validation_max_documents": 2,
            "train_max_tokens": 64,
            "validation_max_tokens": 64,
            "synthetic_seed": 3,
        },
        "training": {
            "seq_len": 8,
            "micro_batch_size": 1,
            "gradient_accumulation": 1,
            "max_steps": 8,
            "max_tokens": 64,
        },
        "optimizer": {"name": "adamw", "warmup_steps": 2},
        "logging": {"root_dir": "runs"},
    }
    from klab.config.models import RunConfig

    base = RunConfig.model_validate(base).model_dump(mode="json")
    (root / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    profile = {
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 2,
        "ffn_dim": 32,
        "seq_len": 8,
        "max_seq_len": 16,
        "micro_batch_size": 1,
        "gradient_accumulation": 1,
        "max_steps": 8,
        "max_tokens": 64,
        "validation_steps": 4,
        "memory_table_size": 17,
        "memory_dim": 4,
    }
    scales = ("smoke", "nano", "micro", "tiny")
    _json(
        root / "profiles.json",
        {
            "format": "klab-research-profiles",
            "version": 1,
            "scales": {scale: profile for scale in scales},
        },
    )
    dataset = {
        "source": "synthetic",
        "synthetic_task": "chat_recall",
        "revision": None,
        "dataset_seed": 3,
        "train_max_documents": 2,
        "validation_max_documents": 2,
        "train_max_tokens": 64,
        "validation_max_tokens": 64,
        "vocab_size": 260,
        "min_frequency": 1,
        "tokenizer_max_documents": 2,
        "tokenizer_train_max_tokens": 64,
        "tokenizer_validation_max_documents": 2,
        "tokenizer_validation_max_tokens": 64,
        "license": "CC0-1.0",
        "source_url": "https://example.org/neutral",
        "notes": [],
        "card_applicability": {},
    }
    _json(
        root / "datasets.json",
        {
            "format": "klab-research-datasets",
            "version": 1,
            "datasets": {"neutral": dataset},
        },
    )
    entry = {
        "format": "klab-research-entry",
        "version": 1,
        "id": "neutral-study",
        "title": "Neutral comparison",
        "question": "Does the plan bind two widths?",
        "mechanisms": [],
        "hypothesis": "Widths differ",
        "failure_interpretation": "Invalid plan",
        "controls": [],
        "independent_variables": [],
        "dependent_metrics": [],
        "confounders": [],
        "minimum_useful_scale": "smoke",
        "recommended_scale": "smoke",
        "hardware_notes": "Metadata only",
        "runtime_class": "fixture",
        "evaluation_policy": {
            "periodic_validation_steps": {},
            "endpoint_cards": [],
            "endpoint_policy": "first_configured_limit",
            "milestones_supported": False,
            "primary_thresholds": [],
        },
        "cards": [
            {
                "reference": "cards/neutral.json",
                "role": "development",
                "category": "fixture",
                "applicability": {},
            }
        ],
        "papers": [],
        "implementation_links": [],
        "docs": [],
        "can_establish": [],
        "cannot_establish": [],
        "prerequisites": [],
        "recipe": "recipes/neutral-study.json",
    }
    _json(root / "catalog/neutral-study.json", entry)
    recipe = {
        "base_set": {},
        "axes": {
            "seed": [{"label": "s5", "set": {"seed": 5}}],
            "width": [
                {"label": "small", "set": {"model.ffn_dim": 32}},
                {"label": "wide", "set": {"model.ffn_dim": 48}},
            ],
        },
        "comparisons": [
            {
                "id": "width-effect",
                "vary": "custom",
                "vary_fields": ["model.ffn_dim"],
                "baseline": {"width": "small"},
                "variant": {"width": "wide"},
            }
        ],
    }
    _json(
        root / "recipes/neutral-study.json",
        {
            "format": "klab-research-recipe",
            "version": 1,
            "id": "neutral-study",
            "designs": {"default": {scale: recipe for scale in scales}},
        },
    )
    # Capability implementation is generic Lab code, independent of research resources.
    from klab.evaluation.capabilities import (
        capability_card,
        capability_card_payload,
    )

    _json(
        root / "cards/neutral.json",
        capability_card_payload(capability_card("chat-alias-recall-v1")),
    )
    _json(
        root / "lifecycle.json",
        {
            "format": "klab-research-lifecycle",
            "version": 1,
            "entries": [],
            "evidence": [],
            "baselines": [],
            "findings": [],
            "promotions": [],
        },
    )
    return root


def make_study_resources(tmp_path: Path) -> Path:
    """Two neutral factors and two explicit seeds; retain dataset adapter coverage."""
    import shutil

    from klab.research import catalog

    root = make_resources(tmp_path)
    for name in ("datasets.json", "dataset_sources"):
        source = catalog.resource_root() / name
        if source.is_dir():
            shutil.copytree(source, root / name)
        else:
            shutil.copyfile(source, root / name)
    path = root / "recipes/neutral-study.json"
    value = json.loads(path.read_text())
    for recipe in value["designs"]["default"].values():
        recipe["base_set"]["optimizer.warmup_steps"] = 2
        recipe["axes"]["seed"] = [
            {"label": f"s{seed}", "set": {"seed": seed}} for seed in (5, 9)
        ]
        recipe["axes"]["depth"] = [
            {"label": "one", "set": {"model.num_layers": 1}},
            {"label": "two", "set": {"model.num_layers": 2}},
        ]
        recipe["factorial_designs"] = [
            {
                "format": "klab-factorial-design",
                "version": 1,
                "id": "width-depth",
                "factors": [
                    {"axis": "width", "control": "small", "treatment": "wide"},
                    {"axis": "depth", "control": "one", "treatment": "two"},
                ],
            }
        ]
    _json(path, value)
    return root
