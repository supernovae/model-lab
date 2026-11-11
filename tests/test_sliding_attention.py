from __future__ import annotations

import torch

from klab.model.attention.dense import DenseAttention


def test_sliding_window_mask_excludes_distant_past_and_future() -> None:
    attention = DenseAttention(8, 2, 6, 10000.0, window_size=3)
    expected = torch.tensor(
        [
            [False, True, True, True, True, True],
            [False, False, True, True, True, True],
            [False, False, False, True, True, True],
            [True, False, False, False, True, True],
            [True, True, False, False, False, True],
            [True, True, True, False, False, False],
        ]
    )
    assert torch.equal(attention.causal_mask, expected)
