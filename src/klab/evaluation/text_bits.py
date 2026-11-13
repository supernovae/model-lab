"""Held-out bits per byte: a loss that does not depend on the tokenizer.

Loss per token falls when a tokenizer makes tokens longer, so it cannot compare
models with different vocabularies. Bits per byte can: the same held-out text,
the same bytes, scored by each model through its own tokenizer.

The documents come from the run's own prepared validation tokens, cut at each
``<eos>`` and decoded. Lab's ByteLevel BPE has no normalizer, so decoding
returns the source text byte for byte; a digest of the documents then proves
two runs with different tokenizers scored the same text. Each document is
scored on its own, after one ``<eos>`` token as context, every token exactly
once; its own end-of-text is not a target, so the score covers the text's bytes
and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from klab.data.packing import supervision_requires_mask

PROTOCOL = "klab-text-bpb-v1"


def heldout_documents(run: Path, tokenizer: Any, count: int) -> list[str]:
    """The first COUNT validation documents of RUN, decoded from its prepared tokens."""
    if count < 1:
        raise ValueError("text_documents must be positive")
    root = Path(run) / "data"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if supervision_requires_mask(manifest):
        raise ValueError(
            "bits per byte needs plain text validation; this run's validation "
            "targets are masked (chat or preference data)"
        )
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("tokenizer has no <eos> special token")
    tokens = np.load(root / "validation.npy", mmap_mode="r", allow_pickle=False)
    ends = np.flatnonzero(np.asarray(tokens) == eos)
    if len(ends) < count:
        raise ValueError(
            f"bits per byte needs {count} validation documents; "
            f"the run's validation split holds {len(ends)}"
        )
    documents = []
    start = 0
    for end in ends[:count]:
        documents.append(
            tokenizer.decode(
                [int(value) for value in tokens[start:end]],
                skip_special_tokens=False,
            )
        )
        start = int(end) + 1
    return documents


def text_digest(documents: list[str]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def score_documents(
    loaded: Any, documents: list[str], *, batch_size: int
) -> dict[str, Any]:
    """Bits per byte of DOCUMENTS under LOADED, each scored after ``<eos>``."""
    import torch
    from torch.nn import functional

    from klab.probes.scoring import encode, eot, score_tokens

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    window = loaded.config.model.max_seq_len
    start_token = eot(loaded)
    encoded = [[start_token, *encode(loaded, document)] for document in documents]
    # Byte-address memory needs each sequence's own addresses: score one by one.
    batched = loaded.config.model.memory not in {"byte", "portable"}
    total = 0.0
    targets = 0
    short = [i for i, ids in enumerate(encoded) if batched and len(ids) <= window + 1]
    long = sorted(set(range(len(encoded))) - set(short))
    # Similar lengths together waste less padding; the sum does not depend on order.
    short.sort(key=lambda i: (len(encoded[i]), i))
    with torch.inference_mode():
        for first in range(0, len(short), batch_size):
            rows = [encoded[i] for i in short[first : first + batch_size]]
            width = max(len(ids) for ids in rows) - 1
            if width < 1:
                continue
            inputs = np.zeros((len(rows), width), dtype=np.int64)
            labels = np.full((len(rows), width), -100, dtype=np.int64)
            for row, ids in enumerate(rows):
                inputs[row, : len(ids) - 1] = ids[:-1]
                labels[row, : len(ids) - 1] = ids[1:]
            x = torch.from_numpy(inputs).to(loaded.device)
            y = torch.from_numpy(labels).to(loaded.device)
            logits = loaded.model(x)
            loss = functional.cross_entropy(
                logits.float().flatten(0, 1),
                y.flatten(),
                ignore_index=-100,
                reduction="sum",
            )
            total += float(loss)
            targets += int((y != -100).sum())
        for index in long:
            ids = encoded[index]
            if len(ids) < 2:
                continue
            values, _ = score_tokens(loaded, ids, 1)
            total -= float(values.sum())
            targets += len(values)
    text_bytes = sum(len(document.encode("utf-8")) for document in documents)
    if not text_bytes:
        raise ValueError("bits per byte needs documents with text")
    if not math.isfinite(total):
        raise ValueError("nonfinite held-out text loss")
    return {
        "protocol": PROTOCOL,
        "documents": len(documents),
        "text_sha256": text_digest(documents),
        "bytes": text_bytes,
        "targets": targets,
        "nats": total,
        "bits_per_byte": total / (math.log(2) * text_bytes),
    }
