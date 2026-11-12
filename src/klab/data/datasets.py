"""Bounded source document iterators."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from datasets import load_dataset

from klab.config.models import DatasetConfig
from klab.data.chat_recall import iter_chat_recall
from klab.data.conversations import iter_conversations, iter_preference_pairs
from klab.data.engram_recall import iter_engram_recall
from klab.data.instruction_reference import iter_instruction_reference
from klab.data.synthetic import iter_synthetic
from klab.data.withheld_facts import training_documents
from klab.hf_auth import hub_access_errors, hub_auth_kwargs, raise_for_hub_auth
from klab.progress import progress_phase
from klab.token_encode_worker import LineError, local_text_record


def _withheld_facts(seed: int, split: str) -> Iterator[str]:
    return training_documents(seed)


# Built-in offline generators, selected by dataset.synthetic_task.
_SYNTHETIC = {
    "stories": iter_synthetic,
    "instruction_reference": iter_instruction_reference,
    "chat_recall": iter_chat_recall,
    "engram_recall": iter_engram_recall,
    "withheld_facts": _withheld_facts,
}


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
        yield from _SYNTHETIC[config.synthetic_task](config.synthetic_seed, split)
        return
    if config.source == "local_chat":
        path = config.train_path if split == "train" else config.validation_path
        assert path is not None
        yield from iter_conversations(path, config.template)
        return
    if config.source == "local_preference":
        path = config.train_path if split == "train" else config.validation_path
        assert path is not None
        # The text view of a pair is its prompt with the chosen reply.
        for pair in iter_preference_pairs(path, config.template):
            yield pair.prompt + pair.chosen
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
    if config.source == "snapshot":
        from klab.data.sources import iter_snapshot

        yield from iter_snapshot(config, split)
        return
    # Without a validation split, validation is the documents that follow
    # the training documents in the training split.
    source_split = (
        config.train_split
        if split == "train" or config.validation_split is None
        else config.validation_split
    )
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
                config.dataset_id,
                name=config.dataset_config,
                split=source_split,
                revision=config.revision,
                streaming=True,
                cache_dir=str(config.cache_dir),
                **auth,
            )
        except hub_access_errors() as error:
            raise_for_hub_auth(error, credential_supplied=bool(auth))
        if split == "validation" and config.validation_split is None:
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
            text = record.get(config.text_field)
            if config.text_field not in record or not isinstance(text, str):
                raise ValueError(
                    f"{config.dataset_id} stream record has missing or "
                    f"non-string {config.text_field}"
                )
            yield text
    except hub_access_errors() as error:
        raise_for_hub_auth(error, credential_supplied=bool(auth))
