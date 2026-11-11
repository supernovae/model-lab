from __future__ import annotations

import numpy as np
import pytest
import torch

from klab.data.packing import TokenBlockDataset
from klab.evaluation.language_model import evaluate


class UniformModel(torch.nn.Module):
    def forward(
        self, input_ids: torch.Tensor, *, byte_addresses: torch.Tensor | None = None
    ) -> torch.Tensor:
        return torch.zeros((*input_ids.shape, 3), device=input_ids.device)


class RaisingModel(torch.nn.Module):
    def forward(
        self, input_ids: torch.Tensor, *, byte_addresses: torch.Tensor | None = None
    ) -> torch.Tensor:
        torch.rand(1)
        raise RuntimeError("evaluation failure")


def test_evaluation_counts_only_limited_held_out_targets() -> None:
    dataset = TokenBlockDataset(np.array([0, 1, 2, 1, 2, 0, 1], dtype=np.int32), 2)
    result = evaluate(
        UniformModel(), dataset, batch_size=2, max_batches=1, device=torch.device("cpu")
    )
    assert result["batches"] == 1
    assert result["valid_targets"] == 4
    assert result["loss"] == pytest.approx(torch.log(torch.tensor(3.0)).item())


def test_evaluation_restores_mode_and_rng_after_failure() -> None:
    dataset = TokenBlockDataset(np.array([0, 1, 2], dtype=np.int32), 2)
    model = RaisingModel()
    model.train()
    torch.manual_seed(123)
    expected = torch.rand(1)
    torch.manual_seed(123)
    with pytest.raises(RuntimeError, match="evaluation failure"):
        evaluate(
            model, dataset, batch_size=1, max_batches=1, device=torch.device("cpu")
        )
    assert model.training
    assert torch.equal(torch.rand(1), expected)


def test_dataset_masks_targets_but_retains_prompt_context_and_skips_empty_blocks() -> (
    None
):
    ids = np.array([0, 1, 2, 0, 1, 2, 0], dtype=np.int32)
    supervision = np.array([False, True, False, False, False, False, True], dtype=bool)
    dataset = TokenBlockDataset(ids, 2, supervision=supervision)

    assert len(dataset) == 2
    inputs, targets, addresses = dataset.numpy_block(0)
    assert addresses is None
    assert inputs.tolist() == [0, 1]
    assert targets.tolist() == [1, -100]
    _, final_targets, _ = dataset.numpy_block(1)
    assert final_targets.tolist() == [-100, 0]


def test_evaluation_scores_only_supervision_masked_targets() -> None:
    dataset = TokenBlockDataset(
        np.array([0, 1, 2, 1, 2, 0, 1], dtype=np.int32),
        2,
        supervision=np.array(
            [False, True, False, False, True, False, False], dtype=bool
        ),
    )
    result = evaluate(
        UniformModel(), dataset, batch_size=2, max_batches=1, device=torch.device("cpu")
    )
    assert result["valid_targets"] == 2
