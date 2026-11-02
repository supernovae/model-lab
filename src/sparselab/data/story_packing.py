"""Matched EOS-delimited document groups without changing prepared data bytes.

Targets are every stored token, including the first token and final EOS. A
virtual preceding EOS supplies context for the first token in both layouts.
Groups retain source order and repeat deterministically at epoch boundaries.
"""

from __future__ import annotations

import numpy as np

from sparselab.canonical import canonical_digest


class MatchedStoryDataset:
    def __init__(
        self,
        ids: np.ndarray,
        seq_len: int,
        eos: int,
        *,
        layout: str,
        targets_per_update: int = 4096,
    ) -> None:
        if ids.ndim != 1 or ids.dtype != np.dtype(np.int32) or not len(ids):
            raise ValueError("story IDs must be nonempty one-dimensional int32")
        if (
            seq_len <= 0
            or targets_per_update <= 0
            or layout not in {"story", "contiguous"}
        ):
            raise ValueError("invalid matched story layout or bounds")
        if ids[-1] != eos:
            raise ValueError("matched stories require a terminal EOS")
        self.ids, self.seq_len, self.eos = ids, seq_len, eos
        self.layout, self.targets_per_update = layout, targets_per_update
        self.supervision = None
        # Scan in bounded slices: no corpus-sized temporary boolean array.
        ends = []
        for offset in range(0, len(ids), 1_048_576):
            ends.extend(
                (
                    np.flatnonzero(ids[offset : offset + 1_048_576] == eos) + offset + 1
                ).tolist()
            )
        self.document_ends = np.asarray(ends, dtype=np.int64)
        groups = [0]
        for end in ends:
            if end - groups[-1] >= targets_per_update:
                groups.append(end)
        if groups[-1] != len(ids):
            groups.append(len(ids))
        self.group_offsets = np.asarray(groups, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.group_offsets) - 1

    def horizon(self, budget: int) -> int:
        """Exact update count for a fresh run, including the clipped final group."""
        if budget <= 0:
            raise ValueError("budget must be positive")
        passes, tail = divmod(budget, len(self.ids))
        return passes * len(self) + (
            int(np.searchsorted(self.group_offsets, tail, side="left")) if tail else 0
        )

    def group_manifest(self, budget: int, *, train_tokens_sha256: str) -> dict:
        """Layout-independent declaration for freezing a matched pair."""
        return {
            "kind": "matched-eos-stories-v1",
            "train_tokens_sha256": train_tokens_sha256,
            "sequence_length": self.seq_len,
            "eos_token_id": self.eos,
            "initial_context": "virtual-eos",
            "order": "source",
            "targets_per_update": self.targets_per_update,
            "group_offsets_sha256": canonical_digest(self.group_offsets.tolist()),
            "groups_per_pass": len(self),
            "targets_per_pass": len(self.ids),
            "max_tokens": budget,
            "max_steps": self.horizon(budget),
        }

    def exposure_supply(self) -> dict[str, int]:
        return {
            "raw_packed_positions": len(self.ids),
            "full_blocks": len(self),
            "eligible_blocks": len(self),
            "zero_target_blocks_skipped": 0,
            "trailing_target_positions": 0,
            "usable_supervised_targets_per_pass": len(self.ids),
        }

    def update_rows(self, group: int, remaining: int):
        """Yield padded native microblock records for one common target interval."""
        if not 0 <= group < len(self) or remaining <= 0:
            raise ValueError("invalid group cursor or remaining budget")
        start, end = map(int, self.group_offsets[group : group + 2])
        end = min(end, start + remaining)
        if self.layout == "story":
            lo, hi = np.searchsorted(self.document_ends, [start, end], side="right")
            boundaries = [*self.document_ends[lo:hi], end]
        else:
            boundaries = [end]
        for boundary in boundaries:
            boundary = int(boundary)
            while start < boundary:
                stop = min(start + self.seq_len, boundary)
                count = stop - start
                inputs = np.full(self.seq_len, self.eos, dtype=np.int64)
                labels = np.full(self.seq_len, -100, dtype=np.int64)
                inputs[0] = self.eos if start == 0 else self.ids[start - 1]
                inputs[1:count] = self.ids[start : stop - 1]
                labels[:count] = self.ids[start:stop]
                yield inputs, labels, None, None, None, None
                start = stop
