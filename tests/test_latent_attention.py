from __future__ import annotations

import torch

from klab.model.attention.latent import LatentAttention


def test_latent_attention_is_causal_and_shape_preserving() -> None:
    torch.manual_seed(0)
    attention = LatentAttention(8, 2, 4, 6, 10000.0)
    first = torch.randn(1, 6, 8)
    second = first.clone()
    second[:, 5] += 10
    assert attention(first).shape == first.shape
    assert torch.allclose(attention(first)[:, :5], attention(second)[:, :5])
