"""Bounded source document iterators."""

from __future__ import annotations

import json
from collections.abc import Iterator

from datasets import load_dataset

from klab.config.models import DatasetConfig
from klab.data.chat_recall import iter_chat_recall
from klab.data.conversations import iter_conversations
from klab.data.engram_recall import iter_engram_recall
from klab.data.instruction_reference import iter_instruction_reference
from klab.data.local_stories import iter_local_stories
from klab.data.synthetic import iter_synthetic
from klab.data.withheld_facts import training_documents
from klab.hf_auth import HUB_ACCESS_ERRORS, hub_auth_kwargs, raise_for_hub_auth
from klab.progress import progress_phase

TINYSTORIES_DATASET = "roneneldan/TinyStories"
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
FINEWEB_EDU_DATASET = "HuggingFaceFW/fineweb-edu"
COSMOPEDIA_DATASET = "HuggingFaceTB/cosmopedia"


def iter_documents(
    config: DatasetConfig,
    split: str,
    *,
    local_text_source_bytes: int | None = None,
) -> Iterator[str]:
    """Yield source documents in stable stream order, without implicit fallbacks."""
    if split not in {"train", "validation"}:
        raise ValueError(f"unknown split {split!r}")
    if config.source == "synthetic":
        yield from iter_synthetic(config.synthetic_seed, split)
        return
    if config.source == "instruction_reference":
        yield from iter_instruction_reference(config.synthetic_seed, split)
        return
    if config.source == "chat_recall":
        yield from iter_chat_recall(config.synthetic_seed, split)
        return
    if config.source == "local_chat":
        path = config.train_path if split == "train" else config.validation_path
        assert path is not None
        yield from iter_conversations(path)
        return
    if config.source in {"local_text", "local_token_mixture"}:
        if config.source == "local_token_mixture" and split == "train":
            raise ValueError(
                "token mixture training IDs must use the verified native packing path"
            )
        path = config.train_path if split == "train" else config.validation_path
        assert path is not None
        with path.open("rb") as handle:
            for number in range(1, 2**63):
                # Escaped JSON may need six raw bytes per decoded UTF-8 byte.
                limit = (
                    6 * local_text_source_bytes + 1024
                    if local_text_source_bytes is not None
                    else -1
                )
                line = handle.readline(limit + 1) if limit >= 0 else handle.readline()
                if not line:
                    break
                if limit >= 0 and len(line) > limit:
                    raise ValueError(
                        f"local_text raw JSON record at {path}:{number} exceeds "
                        f"operational raw-line cap ({limit} bytes)"
                    )
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid local_text JSON at {path}:{number}"
                    ) from error
                if (
                    not isinstance(row, dict)
                    or set(row) != {"text"}
                    or not isinstance(row["text"], str)
                    or not row["text"].strip()
                ):
                    raise ValueError(f"invalid local_text record at {path}:{number}")
                yield row["text"]
        return
    if config.source == "local_stories":
        yield from iter_local_stories(config, split)
        return
    if config.source == "snapshot":
        from klab.data.sources import iter_snapshot

        yield from iter_snapshot(config, split)
        return
    if config.source == "engram_recall":
        yield from iter_engram_recall(config.synthetic_seed, split)
        return
    if config.source == "withheld_facts":
        yield from training_documents(config.synthetic_seed)
        return
    dataset_name = {
        "tinystories": TINYSTORIES_DATASET,
        "fineweb_edu": FINEWEB_EDU_DATASET,
        "cosmopedia": COSMOPEDIA_DATASET,
    }[config.source]
    subset = "default" if config.source == "tinystories" else config.dataset_config
    source_split = split if config.source == "tinystories" else "train"
    with progress_phase(
        f"dataset_initialization_{config.source}_{split}",
        completed_work=0,
        total_work=1,
        unit="dataset",
        raw_counters={"source": config.source, "split": split},
    ) as progress:
        auth = hub_auth_kwargs()
        try:
            dataset = load_dataset(
                dataset_name,
                name=subset,
                split=source_split,
                revision=config.revision,
                streaming=True,
                cache_dir=str(config.cache_dir),
                **auth,
            )
        except HUB_ACCESS_ERRORS as error:
            raise_for_hub_auth(error, credential_supplied=bool(auth))
        if split == "validation" and source_split == "train":
            dataset = dataset.skip(config.train_max_documents)
        progress.update(
            completed_work=1,
            total_work=1,
            unit="dataset",
            raw_counters={
                "source": config.source,
                "split": split,
                "initialized_datasets": 1,
            },
        )
    try:
        for record in dataset:
            if "text" not in record or not isinstance(record["text"], str):
                raise ValueError(
                    f"{config.source} stream record has missing or non-string text"
                )
            yield record["text"]
    except HUB_ACCESS_ERRORS as error:
        raise_for_hub_auth(error, credential_supplied=bool(auth))
