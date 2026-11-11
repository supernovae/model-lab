"""Combined multiport Engram and whole-document history boundaries."""

import numpy as np
import torch
from test_engram_integration import model_config

from klab.address_hash import engram_table_sizes
from klab.config.models import AttentionConfig
from klab.data.story_packing import MatchedStoryDataset
from klab.model.transformer import DenseLM


def test_unpadded_multiport_convolution_and_cached_rows_reset_at_document_boundary(
    tmp_path,
):
    config = model_config(
        tmp_path,
        "deepseek_safe_v1",
        memory_block_index=None,
        memory_block_indices=(0, 1),
        memory_table_sizes=engram_table_sizes(31, 8),
    )
    model = DenseLM(config, AttentionConfig()).eval()
    plan = MatchedStoryDataset(
        np.array([4, 6, 5, 3, 8, 4, 5, 6, 3], dtype=np.int32),
        16,
        3,
        layout="story",
        targets_per_update=1,
        unpadded_documents=True,
    )
    outputs = []
    with torch.no_grad():
        for i in range(len(plan)):
            x, y, *_ = plan.numpy_microblock(i)
            assert x[0] == 3 and y[-1] == 3 and len(x) == len(y)
            ids = torch.from_numpy(x)[None]
            full = model(ids)
            first, cache = model.forward_cached(ids[:, :2], cache_capacity=16)
            last, cache = model.forward_cached(ids[:, 2:], cache=cache)
            torch.testing.assert_close(torch.cat([first, last], dim=1), full)
            assert cache.memory_histories["1"] is not cache.memory_history
            outputs.append(full)
        # Prior full and cached calls, including changed previous documents,
        # cannot affect a subsequent fresh document row.
        model(torch.tensor([[22, 23, 24, 25, 3]]))
        ids = torch.from_numpy(plan.numpy_microblock(1)[0])[None]
        torch.testing.assert_close(model(ids), outputs[1], rtol=0, atol=0)
        fresh, _ = model.forward_cached(ids, cache_capacity=16)
        torch.testing.assert_close(fresh, outputs[1])
