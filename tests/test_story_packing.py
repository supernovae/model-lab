from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from klab.data.story_packing import MatchedStoryDataset


def story_ids():
    # Unique non-EOS IDs expose dropped/duplicated targets, even within long stories.
    return np.array([1, 2, 0, *range(3, 24), 0, 24, 0, 25, 26, 0], dtype=np.int32)


@pytest.mark.parametrize("budget", [1, 3, 9, 25, 31, 32, 61, 100])
def test_layouts_match_every_update_and_clipped_suffix(budget):
    ids = story_ids()
    plans = [
        MatchedStoryDataset(ids, 4, 0, layout=layout, targets_per_update=8)
        for layout in ("story", "contiguous")
    ]
    np.testing.assert_array_equal(plans[0].group_offsets, plans[1].group_offsets)
    assert plans[0].group_manifest(budget, train_tokens_sha256="fixture") == plans[
        1
    ].group_manifest(budget, train_tokens_sha256="fixture")
    all_targets = []
    seen = 0
    for step in range(plans[0].horizon(budget)):
        targets = []
        for plan in plans:
            rows = list(plan.update_rows(step % len(plan), budget - seen))
            targets.append(np.concatenate([row[1][row[1] != -100] for row in rows]))
            for inputs, labels, *_ in rows:
                valid = np.count_nonzero(labels != -100)
                assert inputs.shape == labels.shape == (4,)
                np.testing.assert_array_equal(inputs[1:valid], labels[: valid - 1])
        np.testing.assert_array_equal(*targets)
        seen += len(targets[0])
        all_targets.extend(targets[0])
    assert seen == budget
    np.testing.assert_array_equal(all_targets, np.resize(ids, budget))
    assert plans[0].horizon(budget) == plans[1].horizon(budget)


def test_story_rows_keep_first_token_eos_and_long_story_end():
    plan = MatchedStoryDataset(story_ids(), 4, 0, layout="story", targets_per_update=8)
    rows = list(plan.update_rows(0, 100))
    assert rows[0][0][0] == 0
    np.testing.assert_array_equal(rows[0][1], [1, 2, 0, -100])
    assert rows[1][0][0] == 0
    assert rows[2][0][0] == 6  # long-story continuation carries preceding token
    assert rows[-1][1][1] == 0  # long story's ending is retained
    rebuilt = MatchedStoryDataset(
        story_ids(), 4, 0, layout="story", targets_per_update=8
    )
    for a, b in zip(plan.update_rows(1, 100), rebuilt.update_rows(1, 100), strict=True):
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


def test_padding_has_zero_loss_and_gradient():
    plan = MatchedStoryDataset(story_ids(), 4, 0, layout="story", targets_per_update=8)
    targets = torch.from_numpy(next(plan.update_rows(0, 100))[1])
    logits = torch.randn(4, 32, requires_grad=True)
    loss = F.cross_entropy(logits, targets, reduction="sum")
    expected = F.cross_entropy(logits[:3], targets[:3], reduction="sum")
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[3].count_nonzero() == 0


def test_refuses_unterminated_data():
    with pytest.raises(ValueError, match="terminal EOS"):
        MatchedStoryDataset(np.array([1, 2], dtype=np.int32), 4, 0, layout="story")


def test_manifest_layout_costs_match_materialized_full_pass():
    ids = story_ids()
    plan = MatchedStoryDataset(ids, 4, 0, layout="story", targets_per_update=8)
    for layout, cost in plan.layout_costs().items():
        other = MatchedStoryDataset(ids, 4, 0, layout=layout, targets_per_update=8)
        groups = [list(other.update_rows(i, len(ids))) for i in range(len(other))]
        counts = [len(rows) for rows in groups]
        padding = [
            sum(int(np.count_nonzero(row[1] == -100)) for row in rows)
            for rows in groups
        ]
        assert cost["rows_per_pass"] == sum(counts)
        assert cost["max_rows_per_update"] == max(counts)
        assert counts[cost["max_rows_group_index"]] == max(counts)
        assert cost["padded_slots_per_pass"] == sum(padding)
        assert cost["max_padded_slots_per_update"] == max(padding)


@pytest.mark.parametrize("updates", [1, 2, 3, 4, 7, 10])
def test_complete_group_budget_inverts_horizon_across_passes(updates):
    budgets = []
    for layout in ("contiguous", "story"):
        plan = MatchedStoryDataset(
            story_ids(), 4, 0, layout=layout, targets_per_update=8
        )
        budget = plan.targets_for_updates(updates)
        assert plan.horizon(budget) == updates
        observed = sum(
            int(np.count_nonzero(row[1] != -100))
            for step in range(updates)
            for row in plan.update_rows(step % len(plan), budget)
        )
        assert observed == budget
        budgets.append(budget)
    assert budgets[0] == budgets[1]
