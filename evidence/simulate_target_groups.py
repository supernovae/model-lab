import json
from pathlib import Path

import numpy as np

from sparselab.data.document_batching import collate_document_update
from sparselab.data.story_packing import MatchedStoryDataset

root = Path(
    "/tank/sparselab/projects/devmind/experiments/reasoning-spine-next-20261104-v1/runs/devmind-engram-mb4-compiled-main-20261104-v1"
)
start = 61981272
ids = np.load(root / "data/train.npy", mmap_mode="r", allow_pickle=False)
# Fixed 8 MiB read, not a corpus scan or checkpoint tensor load.
sample = ids[start : start + 2097152].copy()
sample = sample[: int(np.flatnonzero(sample == 3)[-1]) + 1]
out = {
    "scope": "fixed contiguous 8 MiB maximum prepared-token sample after completed checkpoint step 12500; not full corpus or GPU benchmark",
    "start_token": start,
    "tokens": len(sample),
    "source_checkpoint": "step_00012500_gen_000005",
    "rows": [],
}
for budget in [4096, 32768, 65536, 131072]:
    ds = MatchedStoryDataset(
        sample,
        4096,
        3,
        layout="story",
        targets_per_update=budget,
        unpadded_documents=True,
    )
    targets = []
    slots = 0
    peaks = [0] * 5
    shapes = set()
    for group in range(len(ds)):
        records = list(ds.update_rows(group, len(sample)))
        targets.append(sum(len(r[1]) for r in records))
        batches = collate_document_update(
            records,
            width=4,
            eos=3,
            max_length=4096,
            padding_multiple=128,
            **(
                {"max_padded_slots": 8192, "max_attention_cells": 16777216}
                if budget > 4096
                else {}
            ),
        )
        for b in batches:
            B, L = b.inputs.shape
            slots += B * L
            shapes.add((B, L))
            peaks = [
                max(x, y) for x, y in zip(peaks, [B, L, B * L, B * L * L, len(batches)])
            ]
    out["rows"].append(
        {
            "target_goal": budget,
            "updates": len(ds),
            "targets_mean": float(np.mean(targets)),
            "targets_min": min(targets),
            "targets_p50": float(np.median(targets)),
            "targets_max": max(targets),
            "valid_slots": len(sample),
            "input_slots": slots,
            "padded_slots": slots - len(sample),
            "padding_fraction": 1 - len(sample) / slots,
            "B_peak": peaks[0],
            "L_peak": peaks[1],
            "BL_peak": peaks[2],
            "BL2_peak": peaks[3],
            "microbatches_per_update_peak": peaks[4],
            "shape_count": len(shapes),
        }
    )
for row in out["rows"]:
    row["update_count_reduction_factor"] = out["rows"][0]["updates"] / row["updates"]
Path("evidence/target-group-sample.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
