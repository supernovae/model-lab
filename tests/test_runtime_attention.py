from __future__ import annotations

from pathlib import Path

import pytest

from klab.config.loading import load_config
from klab.runtime_attention import attention_probe, retains_scores


def _measure(exponent: float) -> list[dict[str, float]]:
    return [
        {
            "sequence_length": length,
            "peak_extra_bytes": 1000 * (length / 1024) ** exponent,
        }
        for length in (1024, 2048, 4096)
    ]


def test_score_scaling_separates_fused_from_materialized_attention() -> None:
    assert not retains_scores(_measure(1.0))
    assert retains_scores(_measure(2.0))
    with pytest.raises(ValueError, match="at least two"):
        retains_scores(_measure(1.0)[:1])


def test_probe_reports_unmeasured_scores_without_a_device_memory_counter() -> None:
    probe = attention_probe(load_config(Path("configs/smoke_cpu.yaml")))
    assert probe["score_memory"]["status"] == "UNMEASURED"
    assert "peak device memory" in probe["score_memory"]["reason"]
