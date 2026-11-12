"""Bounded source document iterators."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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
from klab.token_encode_worker import LineError, local_text_record

TINYSTORIES_DATASET = "roneneldan/TinyStories"
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
FINEWEB_EDU_DATASET = "HuggingFaceFW/fineweb-edu"
COSMOPEDIA_DATASET = "HuggingFaceTB/cosmopedia"


def local_text_error(
    path: Path, number: int, error: LineError, split: str = "train"
) -> ValueError:
    """The message for a refused local_text line, wherever it was parsed."""
    if error.kind == "json":
        return ValueError(f"invalid local_text JSON at {path}:{number}")
    if error.kind == "record":
        return ValueError(f"invalid local_text record at {path}:{number}")
    if error.kind == "line_cap":
        return ValueError(
            f"local_text raw JSON record at {path}:{number} exceeds "
            f"operational raw-line cap ({error.detail[0]} bytes)"
        )
    size, limit = error.detail
    return ValueError(
        f"{split} source document {number} exceeds "
        f"tokenizer_batch_source_bytes ({size} > {limit})"
    )


def iter_local_text(
    path: Path,
    *,
    source_bytes: int | None = None,
    offset: int = 0,
    number: int = 1,
) -> Iterator[tuple[str, int]]:
    """Yield ``(text, end_offset)`` for each ``{"text": ...}`` line of a JSONL file.

    ``offset`` and ``number`` resume after an earlier record: the byte offset
    that record ended at and the line number of the next one.
    """
    # Escaped JSON may need six raw bytes per decoded UTF-8 byte.
    limit = 6 * source_bytes + 1024 if source_bytes is not None else -1
    with path.open("rb") as handle:
        handle.seek(offset)
        for line_number in range(number, 2**63):
            line = handle.readline(limit + 1) if limit >= 0 else handle.readline()
            if not line:
                break
            if limit >= 0 and len(line) > limit:
                raise ValueError(
                    f"local_text raw JSON record at {path}:{line_number} exceeds "
                    f"operational raw-line cap ({limit} bytes)"
                )
            try:
                text = local_text_record(line)
            except LineError as error:
                raise local_text_error(path, line_number, error) from error
            offset += len(line)
            yield text, offset


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
        for text, _ in iter_local_text(path, source_bytes=local_text_source_bytes):
            yield text
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
