"""Independent right-padded documents within one immutable optimizer group."""

from __future__ import annotations

import numpy as np

from sparselab.engines.base import Microbatch


def collate_document_update(
    records,
    *,
    width: int,
    eos: int,
    max_length: int,
    padding_multiple: int = 128,
    max_padded_slots: int | None = None,
    max_attention_cells: int | None = None,
) -> list[Microbatch]:
    """Stable length buckets; never pack, truncate or move a document to another update.

    Inputs retain virtual preceding EOS and positions restart at zero per row.
    Causal attention and convolution cannot read right padding. Position masks are
    separate from supervision masks: masked prompt tokens remain real context.
    Sequence shapes are rounded up and capped at max_length; final row count may
    be smaller than width. No dummy documents are introduced.
    """
    if min(width, max_length, padding_multiple) < 1 or not records:
        raise ValueError("positive batching bounds and a nonempty update required")
    for record in records:
        x, y, *sidecars = record
        if (
            x.ndim != 1
            or y.shape != x.shape
            or not 0 < len(x) <= max_length
            or x.dtype != np.int64
            or y.dtype != np.int64
        ):
            raise ValueError("documents require aligned nonempty bounded int64 vectors")
        if any(value is not None for value in sidecars):
            raise ValueError("document batching does not support sidecars")
    ordered = sorted(records, key=lambda row: len(row[0]))
    batches = []
    offset = 0
    while offset < len(ordered):
        rows = []
        length = 0
        for row in ordered[offset : offset + width]:
            candidate_length = min(
                max_length,
                ((len(row[0]) + padding_multiple - 1) // padding_multiple)
                * padding_multiple,
            )
            count = len(rows) + 1
            if (
                max_padded_slots is not None
                and count * candidate_length > max_padded_slots
            ) or (
                max_attention_cells is not None
                and count * candidate_length**2 > max_attention_cells
            ):
                if not rows:
                    raise ValueError(
                        "one document exceeds the microbatch shape envelope"
                    )
                break
            rows.append(row)
            length = candidate_length
        offset += len(rows)
        inputs = np.full((len(rows), length), eos, dtype=np.int64)
        targets = np.full_like(inputs, -100)
        positions = np.zeros(inputs.shape, dtype=np.bool_)
        for index, (x, y, *_) in enumerate(rows):
            inputs[index, : len(x)] = x
            targets[index, : len(y)] = y
            positions[index, : len(x)] = True
        batches.append(Microbatch(inputs, targets, position_mask=positions))
    return batches
