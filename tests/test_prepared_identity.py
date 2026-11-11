"""Prepared content reuse across publication code, model sizes and locations."""

from __future__ import annotations

import shutil

import pytest
from test_dataset_sources import (
    acquire,
    offline,  # noqa: F401 - shared offline source fixture
    runtime,
)
from test_training import config

from klab.config.models import TokenizerTrainConfig
from klab.data import packing
from klab.data.tokenizer import load_tokenizer, train_tokenizer


@pytest.fixture(autouse=True)
def isolated_workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))


@pytest.mark.usefixtures("offline")
@pytest.mark.parametrize("source", ["synthetic", "snapshot"])
def test_writer_changes_and_model_scaling_reuse_relocated_arrays(
    tmp_path, monkeypatch, source
):
    configured = config(tmp_path)
    if source == "snapshot":
        dataset = runtime(acquire(tmp_path / "snapshot-input"))
        tokenizer_path = train_tokenizer(
            TokenizerTrainConfig(
                schema_version=1,
                vocab_size=260,
                min_frequency=1,
                max_documents=20,
                output_dir=tmp_path / "snapshot-tokenizer",
                dataset=dataset,
            )
        )
        configured = configured.model_copy(
            update={
                "dataset": dataset,
                "training": configured.training.model_copy(update={"seq_len": 8}),
                "model": configured.model.model_copy(update={"vocab_size": 260}),
                "tokenizer": configured.tokenizer.model_copy(
                    update={"path": tokenizer_path}
                ),
            }
        )
    tokenizer = load_tokenizer(configured.tokenizer.path)
    prepared = packing.prepare_data(configured, tokenizer)
    original_hash = packing._sha256

    def changed_writer(path):
        return "f" * 64 if path.name == "snapshot_writer.py" else original_hash(path)

    monkeypatch.setattr(packing, "_sha256", changed_writer)

    def forbid_repacking(*args, **kwargs):
        pytest.fail("reusable prepared arrays must not iterate input documents")

    monkeypatch.setattr(packing, "iter_documents", forbid_repacking)
    destination = tmp_path / "replica"
    shutil.copytree(configured.dataset.cache_dir, destination)
    for width in (32, 64, 128):
        scaled = configured.model_copy(
            update={
                "model": configured.model.model_copy(update={"hidden_dim": width}),
                "dataset": configured.dataset.model_copy(
                    update={"cache_dir": destination}
                ),
            }
        )
        reused = packing.prepare_data(scaled, tokenizer, tokenizer_batch_documents=1)
        assert reused.root == destination / prepared.root.name
        assert reused.manifest == prepared.manifest
        assert reused.train.tobytes() == prepared.train.tobytes()
        assert reused.validation.tobytes() == prepared.validation.tobytes()


@pytest.mark.parametrize(
    "module",
    [
        "sources.py",
        "datasets.py",
        "conversations.py",
        "encoding.py",
        "packing.py",
        "tokenizer.py",
    ],
)
def test_semantic_source_changes_invalidate_preparation(tmp_path, monkeypatch, module):
    configured = config(tmp_path)
    tokenizer = load_tokenizer(configured.tokenizer.path)
    prepared = packing.prepare_data(configured, tokenizer)
    original_hash = packing._sha256
    monkeypatch.setattr(
        packing,
        "_sha256",
        lambda path: "e" * 64 if path.name == module else original_hash(path),
    )
    changed = packing.prepare_data(configured, tokenizer)
    assert changed.root != prepared.root


@pytest.mark.parametrize("change", ["data", "tokenizer", "packing"])
def test_content_inputs_invalidate_preparation(tmp_path, change):
    configured = config(tmp_path)
    tokenizer = load_tokenizer(configured.tokenizer.path)
    prepared = packing.prepare_data(configured, tokenizer)
    if change == "data":
        configured = configured.model_copy(
            update={
                "dataset": configured.dataset.model_copy(update={"synthetic_seed": 8})
            }
        )
    elif change == "tokenizer":
        tokenizer.add_tokens(["new-fixture-token"])
    else:
        configured = configured.model_copy(
            update={
                "model": configured.model.model_copy(update={"memory_ngram_size": 7})
            }
        )
    changed = packing.prepare_data(configured, tokenizer)
    assert changed.root != prepared.root


def test_unknown_data_implementation_is_fail_closed():
    inventory = {"config/models.py": "a", "data/packing.py": "b"}
    baseline = packing.preparation_source_identity(inventory)
    assert (
        packing.preparation_source_identity({**inventory, "data/new_transform.py": "c"})
        != baseline
    )
    assert (
        packing.preparation_source_identity(
            {**inventory, "data/snapshot_writer.py": "c"}
        )
        == baseline
    )
