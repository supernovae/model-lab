from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from klab.config.models import DatasetConfig
from klab.data import datasets
from klab.data.engram_recall import recall_case
from klab.data.instruction_reference import iter_instruction_reference


class Stream:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records
        self.skipped = 0

    def skip(self, count: int) -> Stream:
        self.skipped = count
        return self

    def __iter__(self) -> Iterator[dict[str, object]]:
        return iter(self.records[self.skipped :])


def remote_config(tmp_path: Path, source: str) -> DatasetConfig:
    return DatasetConfig(
        source=source,
        revision="pinned-revision",
        dataset_config="sample",
        cache_dir=tmp_path,
        train_max_documents=2,
        validation_max_documents=1,
        train_max_tokens=32,
        validation_max_tokens=16,
    )


@pytest.mark.parametrize("source", ("fineweb_edu", "cosmopedia"))
def test_train_only_remote_sources_use_disjoint_validation_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str
) -> None:
    calls: list[dict[str, object]] = []

    def load(dataset_name: str, **kwargs: object) -> Stream:
        calls.append({"dataset_name": dataset_name, **kwargs})
        return Stream([{"text": "zero"}, {"text": "one"}, {"text": "two"}])

    monkeypatch.setattr(datasets, "load_dataset", load)
    config = remote_config(tmp_path, source)
    assert list(datasets.iter_documents(config, "train")) == ["zero", "one", "two"]
    assert list(datasets.iter_documents(config, "validation")) == ["two"]
    assert all(call["split"] == "train" for call in calls)
    assert all(call["streaming"] is True for call in calls)


def test_instruction_reference_uses_plain_chat_transcripts() -> None:
    first_train = next(iter_instruction_reference(7, "train"))
    first_validation = next(iter_instruction_reference(7, "validation"))
    train_documents = {
        document
        for _, document in zip(range(12), iter_instruction_reference(7, "train"))
    }
    validation_documents = {
        document
        for _, document in zip(range(12), iter_instruction_reference(7, "validation"))
    }

    assert first_train.startswith(
        "System: You are a concise local assistant.\n\nUser: "
    )
    assert "\n\nAssistant: " in first_train
    assert first_train != first_validation
    assert train_documents.isdisjoint(validation_documents)


def test_instruction_reference_is_available_as_a_dataset_source(tmp_path: Path) -> None:
    config = DatasetConfig(
        source="instruction_reference",
        cache_dir=tmp_path,
        train_max_documents=1,
        validation_max_documents=1,
        train_max_tokens=64,
        validation_max_tokens=64,
    )

    assert next(datasets.iter_documents(config, "train")).startswith("System: ")


def test_chat_recall_has_finite_held_out_validation_and_cyclic_train(
    tmp_path: Path,
) -> None:
    config = DatasetConfig(
        source="chat_recall",
        cache_dir=tmp_path,
        train_max_documents=1,
        validation_max_documents=1,
        train_max_tokens=64,
        validation_max_tokens=64,
    )

    train = datasets.iter_documents(config, "train")
    validation = list(datasets.iter_documents(config, "validation"))

    assert next(train).endswith(
        (
            "lumen",
            "orbit",
            "quartz",
            "ripple",
            "saffron",
            "thistle",
            "velvet",
            "yarrow",
            "zephyr",
            "apricot",
            "bronze",
            "cinder",
            "dahlia",
            "fable",
            "garnet",
            "helium",
            "indigo",
            "jasper",
            "kelp",
            "lilac",
            "mango",
            "nectar",
            "onyx",
            "pearl",
        )
    )
    assert validation
    assert all(
        document.endswith(
            (
                "lumen",
                "quartz",
                "saffron",
                "velvet",
                "zephyr",
                "bronze",
                "dahlia",
                "garnet",
                "indigo",
                "kelp",
                "mango",
                "onyx",
            )
        )
        for document in validation
    )


def test_engram_recall_source_has_held_out_prompt_wording(tmp_path: Path) -> None:
    train_prompt, train_answer = recall_case(42, 0)
    held_out_prompt, held_out_answer = recall_case(42, 2_000_000)
    config = DatasetConfig(
        source="engram_recall",
        cache_dir=tmp_path,
        train_max_documents=1,
        validation_max_documents=1,
        train_max_tokens=64,
        validation_max_tokens=64,
    )

    assert train_answer == held_out_answer
    assert train_prompt != held_out_prompt
    assert next(datasets.iter_documents(config, "validation")).startswith(
        "Memory record 1000000."
    )


def test_remote_sources_require_subset_and_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="revision"):
        DatasetConfig(
            source="fineweb_edu",
            dataset_config="sample",
            cache_dir=tmp_path,
            train_max_documents=1,
            validation_max_documents=1,
            train_max_tokens=1,
            validation_max_tokens=1,
        )
