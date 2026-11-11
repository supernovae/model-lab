"""The reusable token store: same prepared data, built once, shared and resumable."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_training import config

from klab.data import packing, token_store
from klab.data.packing import prepare_data, token_store_root
from klab.data.tokenizer import load_tokenizer


def _local_text(tmp_path: Path, train: int = 90, validation: int = 20):
    run = config(tmp_path)
    paths = {}
    for split, count in (("train", train), ("validation", validation)):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {"text": f"{split} {n} fox café 🦊\n```python\nx = {n} * 3\n```"},
                    ensure_ascii=False,
                )
                + "\n"
                for n in range(count)
            ),
            encoding="utf-8",
        )
        paths[f"{split}_path"] = path
    dataset = run.dataset.model_copy(
        update={
            "source": "local_text",
            "license": "fixture",
            "train_max_documents": 60,
            "validation_max_documents": 20,
            "train_max_tokens": 50_000,
            "validation_max_tokens": 50_000,
            **paths,
        }
    )
    return run.model_copy(update={"dataset": dataset})


def _train_store(run) -> dict[str, object]:
    """The store of the 90-document train file (the validation file has 20)."""
    stores = token_store.list_stores(token_store_root(run.dataset.cache_dir))
    return max(stores, key=lambda store: store["documents"])


def _published(prepared) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(prepared.root.iterdir())
        if path.suffix in {".npy", ".json"} and path.name != "staging.json"
    }


def test_store_path_publishes_the_same_prepared_data_as_the_chunked_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _local_text(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    chunked_run = run.model_copy(
        update={
            "dataset": run.dataset.model_copy(
                update={"cache_dir": tmp_path / "chunked-cache"}
            )
        }
    )
    monkeypatch.setattr(packing, "TOKEN_STORE_ENABLED", False)
    chunked = _published(prepare_data(chunked_run, tokenizer))
    monkeypatch.setattr(packing, "TOKEN_STORE_ENABLED", True)
    stored = _published(prepare_data(run, tokenizer))

    assert stored.keys() == chunked.keys()
    assert stored == chunked
    assert _train_store(run)["documents"] == 60
    # The store lives beside the cache directory, inside the test's own tree.
    assert token_store_root(run.dataset.cache_dir).is_relative_to(tmp_path)


def test_larger_budget_and_second_cache_reuse_stored_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _local_text(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    first = prepare_data(run, tokenizer)
    train = _train_store(run)
    assert train["documents"] == 60 and not train["complete"]

    encoded: list[int] = []
    real = token_store._encoded_batches

    def counting(*args, **kwargs):
        for batch in real(*args, **kwargs):
            encoded.append(len(batch[1]))
            yield batch

    monkeypatch.setattr(token_store, "_encoded_batches", counting)
    larger = run.model_copy(
        update={"dataset": run.dataset.model_copy(update={"train_max_documents": 85})}
    )
    second = prepare_data(larger, tokenizer)
    # Only documents after the 60 already stored were tokenized.
    assert 25 <= sum(encoded) <= 30
    assert _train_store(run)["documents"] == 85
    assert np.array_equal(second.train[: len(first.train)], first.train)

    # Another model with the same tokenizer and data tokenizes nothing, even
    # when its prepared cache has to be written again.
    encoded.clear()
    shutil.rmtree(second.root)
    third = prepare_data(larger, tokenizer)
    assert not encoded
    assert np.array_equal(third.train, second.train)


def test_failed_preparation_leaves_no_staging_and_resumes_from_sealed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _local_text(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)

    def failing(*args, **kwargs):
        raise OSError("disk went away")

    monkeypatch.setattr(packing, "_copy_from_token_store", failing)
    with pytest.raises(OSError, match="disk went away"):
        prepare_data(run, tokenizer)
    cache = run.dataset.cache_dir
    assert not [entry for entry in cache.iterdir() if entry.is_dir()]
    tokens = token_store_root(cache)
    assert not list(tokens.rglob("*.tmp"))
    assert token_store.list_stores(tokens)

    monkeypatch.undo()
    encoded: list[int] = []
    real = token_store._encoded_batches

    def counting(*args, **kwargs):
        for batch in real(*args, **kwargs):
            encoded.append(len(batch[1]))
            yield batch

    monkeypatch.setattr(token_store, "_encoded_batches", counting)
    prepared = prepare_data(run, tokenizer)
    assert not encoded
    assert len(prepared.train) > 0


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"train_max_tokens": 200}, "train token cap would truncate selected story"),
        ({"train_max_documents": 500}, "train snapshot has only 90 selected"),
    ],
)
def test_store_path_refuses_selections_in_the_chunked_paths_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, update: dict[str, int], match: str
) -> None:
    run = _local_text(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    refused = run.model_copy(update={"dataset": run.dataset.model_copy(update=update)})
    messages = []
    for enabled in (False, True):
        monkeypatch.setattr(packing, "TOKEN_STORE_ENABLED", enabled)
        with pytest.raises(ValueError, match=match) as error:
            prepare_data(refused, tokenizer)
        messages.append(str(error.value))
    assert messages[0] == messages[1]


def test_damaged_or_foreign_shard_is_not_copied(tmp_path: Path) -> None:
    run = _local_text(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    prepare_data(run, tokenizer)
    root = Path(_train_store(run)["path"])
    # The first shard holds the first documents, which every selection reads.
    shard = root / "shard-000000.ids"
    damaged = bytearray(shard.read_bytes())
    damaged[0] ^= 1
    shard.write_bytes(bytes(damaged))
    shutil.rmtree(next(p for p in run.dataset.cache_dir.iterdir() if p.is_dir()))
    with pytest.raises(ValueError, match="token shard changed"):
        prepare_data(run, tokenizer)
    assert not [entry for entry in run.dataset.cache_dir.iterdir() if entry.is_dir()]
