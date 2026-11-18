from __future__ import annotations

from pathlib import Path

import pytest

from klab.config.loading import load_config
from klab.device_hints import hints_for_device
from klab.runtime_tune import LEVER_GRID, tune, variant_config


def test_tune_measures_each_variant_and_names_the_refused_ones() -> None:
    config = load_config(Path("configs/smoke_cpu.yaml"))
    grid = tuple(row for row in LEVER_GRID if row[0] in {"config", "fused_adamw"}) + (
        ("recompute_off+flex", LEVER_GRID[3][1]),
    )
    profile = tune(
        config, authorization=None, updates=1, warmup=1, microbatches=2, grid=grid
    )
    by_name = {row["name"]: row for row in profile["trials"]}
    assert by_name["config"]["status"] == "ok"
    assert by_name["config"]["tokens_per_second"] > 0
    # Fused AdamW and flex attention are not CPU levers or not valid for the smoke
    # config; a variant that cannot run is reported, never silently skipped.
    assert set(by_name) == {"config", "fused_adamw", "recompute_off+flex"}
    assert all(
        row["status"] in {"ok", "not_applicable", "failed"} for row in by_name.values()
    )
    assert profile["best_fitting"] is not None
    assert profile["device"]["backend"] == "cpu"


def test_flex_attention_cannot_be_recomputed() -> None:
    config = load_config(Path("configs/smoke_cpu.yaml"))
    flex = {"attention": {"implementation": "flex"}}
    with pytest.raises(ValueError, match="cannot be recomputed"):
        variant_config(
            config,
            (
                flex,
                {
                    "runtime": {
                        "memory": {"activation_checkpointing": {"enabled": True}}
                    }
                },
            ),
        )


def test_shipped_hints_match_their_device_only() -> None:
    family = hints_for_device("xpu", "Intel(R) Arc(TM) Pro B70 Graphics")
    assert family is not None and family.family == "xpu-intel-arc-pro-b70"
    assert hints_for_device("xpu", "Intel(R) Arc(TM) A770 Graphics") is None
    assert hints_for_device("cuda", "Intel(R) Arc(TM) Pro B70 Graphics") is None
