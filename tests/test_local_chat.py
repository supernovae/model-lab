import json

import numpy as np
import pytest
from test_training import config

from klab.data.packing import prepare_data
from klab.data.tokenizer import load_tokenizer


def conversation(question, answer):
    return (
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
        )
        + "\n"
    )


def test_local_chat_content_changes_cache_and_overlap_is_rejected(tmp_path):
    original = config(tmp_path)
    train_path, validation_path = (
        tmp_path / "train.jsonl",
        tmp_path / "validation.jsonl",
    )
    train_path.write_text(
        conversation("Tell me about the library.", "It has many books to read.") * 8
    )
    validation_path.write_text(
        conversation("What is in the garden?", "The garden has many flowers.") * 8
    )
    local = original.model_copy(
        update={
            "dataset": original.dataset.model_copy(
                update={
                    "source": "local_chat",
                    "train_path": train_path,
                    "validation_path": validation_path,
                    "license": "CC0-1.0",
                }
            )
        }
    )
    tokenizer = load_tokenizer(local.tokenizer.path)
    first = prepare_data(local, tokenizer)
    train_path.write_text(
        conversation("Tell me about the library.", "It is closed this afternoon.") * 8
    )
    second = prepare_data(local, tokenizer)
    assert first.root != second.root
    assert not np.array_equal(first.train, second.train)
    assert first.manifest["license"] == "CC0-1.0"
    validation_path.write_bytes(train_path.read_bytes())
    with pytest.raises(ValueError, match="overlapping conversation"):
        prepare_data(local, tokenizer)


def test_tampered_prepared_cache_cannot_be_reused(tmp_path):
    configured = config(tmp_path)
    tokenizer = load_tokenizer(configured.tokenizer.path)
    prepared = prepare_data(configured, tokenizer)
    path = prepared.root / "train.npy"
    values = np.load(path)
    values[0] = (values[0] + 1) % configured.model.vocab_size
    np.save(path, values)
    with pytest.raises(ValueError):
        prepare_data(configured, tokenizer)
