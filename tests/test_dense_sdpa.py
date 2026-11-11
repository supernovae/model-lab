from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
import torch

from klab.config.loading import load_config
from klab.config.models import AttentionConfig, RunConfig
from klab.model.attention.dense import DenseAttention


def _attention(implementation: Literal["reference", "sdpa"]) -> DenseAttention:
    return DenseAttention(16, 4, 8, 10_000.0, implementation=implementation)


def test_cpu_sdpa_matches_reference_forward_and_backward() -> None:
    torch.manual_seed(419)
    reference = _attention("reference")
    sdpa = _attention("sdpa")
    sdpa.load_state_dict(reference.state_dict())
    values = torch.randn(2, 6, 16)
    reference_input = values.clone().requires_grad_()
    sdpa_input = values.clone().requires_grad_()

    expected = reference(reference_input)
    actual = sdpa(sdpa_input)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(
        sdpa_input.grad, reference_input.grad, rtol=1e-5, atol=1e-6
    )
    for (_, actual_parameter), (_, expected_parameter) in zip(
        sdpa.named_parameters(), reference.named_parameters(), strict=True
    ):
        torch.testing.assert_close(
            actual_parameter.grad, expected_parameter.grad, rtol=1e-5, atol=1e-6
        )


def test_sdpa_is_causal_and_cached_appends_match_full_prefix() -> None:
    torch.manual_seed(420)
    attention = _attention("sdpa").eval()
    values = torch.randn(2, 6, 16)
    altered = values.clone()
    altered[:, 4:] = torch.randn_like(altered[:, 4:])
    torch.testing.assert_close(attention(values)[:, :4], attention(altered)[:, :4])

    with torch.no_grad():
        full = attention(values)
        cache = attention.create_cache(2, 8, device=values.device, dtype=values.dtype)
        first, cache = attention.forward_cached(values[:, :3], cache)
        second, _ = attention.forward_cached(values[:, 3:], cache)
    torch.testing.assert_close(
        torch.cat((first, second), dim=1), full, rtol=1e-5, atol=1e-6
    )


def test_sdpa_cached_context_overflow_fails() -> None:
    attention = DenseAttention(8, 2, 4, 10_000.0, implementation="sdpa")
    values = torch.randn(1, 4, 8)
    cache = attention.create_cache(1, 8, device=values.device, dtype=values.dtype)
    attention.forward_cached(values, cache)
    with pytest.raises(ValueError, match="configured attention context"):
        attention.forward_cached(values[:, :1], cache)


def test_sdpa_is_opt_in_and_default_serialization_is_unchanged() -> None:
    attention = AttentionConfig()
    assert attention.implementation == "reference"
    assert "implementation" not in attention.model_dump(mode="json")
    assert (
        AttentionConfig.model_validate(attention.model_dump(mode="json")) == attention
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kind": "sliding_window", "window_size": 2}, "requires attention.kind=dense"),
        ({"kind": "mla", "latent_dim": 4}, "requires attention.kind=dense"),
        (
            {"kind": "block_sparse", "block_size": 2, "selected_blocks": 1},
            "requires attention.kind=dense",
        ),
    ],
)
def test_sdpa_rejects_incompatible_attention_families(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        AttentionConfig.model_validate({"implementation": "sdpa", **kwargs})
    with pytest.raises(ValueError, match="equal Q/KV"):
        DenseAttention(8, 2, 4, 10_000.0, num_kv_heads=1, implementation="sdpa")


def test_sdpa_rejects_mlx_and_grouped_query_run_configs() -> None:
    base = load_config(Path("configs/runtime_smoke_cpu.yaml")).model_dump(mode="json")
    base["attention"]["implementation"] = "sdpa"
    mlx = {**base, "runtime": {**base["runtime"], "engine": "mlx", "backend": "metal"}}
    with pytest.raises(ValueError, match="SDPA requires PyTorch"):
        RunConfig.model_validate(mlx)
    grouped_model = {**base["model"], "num_kv_heads": 1}
    with pytest.raises(ValueError, match="equal Q/KV"):
        RunConfig.model_validate({**base, "model": grouped_model})


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or bool(torch.version.hip),
    reason="requires an actual CUDA device",
)
def test_cuda_fp16_sdpa_forward_backward_and_cache_match_reference() -> None:
    torch.manual_seed(421)
    reference = _attention("reference").eval()
    native = _attention("sdpa").cuda().half().eval()
    native.load_state_dict(reference.state_dict())
    values = torch.randn(2, 6, 16)
    reference_input = values.clone().requires_grad_()
    native_input = values.cuda().half().requires_grad_()
    expected = reference(reference_input)
    actual = native(native_input)
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=3e-2, atol=3e-3)
    expected.square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(
        native_input.grad.float().cpu(), reference_input.grad, rtol=3e-2, atol=3e-3
    )
    with torch.no_grad():
        cache = native.create_cache(
            2, 8, device=native_input.device, dtype=native_input.dtype
        )
        first, cache = native.forward_cached(native_input[:, :3], cache)
        second, _ = native.forward_cached(native_input[:, 3:], cache)
    torch.testing.assert_close(
        torch.cat((first, second), dim=1).float().cpu(),
        expected,
        rtol=3e-2,
        atol=3e-3,
    )
