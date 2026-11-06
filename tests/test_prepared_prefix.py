"""Authenticated whole-document prefixes reuse IDs, never re-encode text."""

import json

import numpy as np
import pytest
from test_training import config

from sparselab.data.encoding import PreparationEncoder
from sparselab.data.packing import PreparedPrefix, load_prepared_data, prepare_data
from sparselab.data.tokenizer import load_tokenizer


@pytest.fixture
def prefix_fixture(tmp_path):
    run = config(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    texts = ["one small story", "another different tale", "third account", "last story"]
    validation = "A separate held out narrative."
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "valid.jsonl"
    train_path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    validation_path.write_text(json.dumps({"text": validation}) + "\n")
    lengths = [
        len(tokenizer.encode(t, add_special_tokens=False).ids) + 1 for t in texts
    ]
    dataset = run.dataset.model_copy(
        update={
            "source": "local_text",
            "license": "CC0-1.0",
            "revision": "fixture-v1",
            "train_path": train_path,
            "validation_path": validation_path,
            "train_max_documents": len(texts),
            "train_max_tokens": sum(lengths),
            "validation_max_documents": 1,
            "validation_max_tokens": len(
                tokenizer.encode(validation, add_special_tokens=False).ids
            )
            + 1,
        }
    )
    source = run.model_copy(update={"dataset": dataset})
    parent = prepare_data(source, tokenizer)
    target = source.model_copy(
        update={
            "dataset": dataset.model_copy(
                update={
                    "train_max_documents": 2,
                    "train_max_tokens": sum(lengths[:2]),
                    "cache_dir": tmp_path / "prefix-cache",
                }
            )
        }
    )
    return source, target, tokenizer, parent


def request(parent):
    return PreparedPrefix(parent.root, parent.manifest["manifest_sha256"])


def test_prefix_matches_fresh_native_preparation_without_encoding(
    prefix_fixture, monkeypatch
):
    source, target, tokenizer, parent = prefix_fixture
    fresh = target.model_copy(
        update={
            "dataset": target.dataset.model_copy(
                update={
                    "cache_dir": target.dataset.cache_dir.parent / "fresh-cache",
                }
            )
        }
    )
    expected = prepare_data(fresh, tokenizer)
    before = (parent.root / "manifest.json").read_bytes()

    def no_encoding(*_args, **_kwargs):
        pytest.fail("prefix derivation must not invoke tokenization")

    monkeypatch.setattr(PreparationEncoder, "encode", no_encoding)
    actual = prepare_data(target, tokenizer, prefix_from=request(parent))
    np.testing.assert_array_equal(actual.train, expected.train)
    np.testing.assert_array_equal(actual.validation, parent.validation)
    for split in ("train", "validation"):
        assert (actual.root / f"{split}.npy").read_bytes() == (
            expected.root / f"{split}.npy"
        ).read_bytes()
        assert actual.manifest[split] == expected.manifest[split]
    assert actual.manifest["cache_identity"] == expected.manifest["cache_identity"]
    assert (
        actual.manifest["prefix_derivation"]["parent_manifest_sha256"]
        == parent.manifest["manifest_sha256"]
    )
    assert (parent.root / "manifest.json").read_bytes() == before
    # Both ordinary preparation and a cold load recognize the native cache.
    assert prepare_data(target, tokenizer).root == actual.root
    assert (
        prepare_data(target, tokenizer, prefix_from=request(parent)).root == actual.root
    )
    np.testing.assert_array_equal(
        load_prepared_data(actual.root, byte_enabled=False).train, expected.train
    )
    assert source.dataset.train_max_documents == 4


@pytest.mark.parametrize(
    "change,match",
    [
        ({"train_max_tokens": 1}, "EOS boundary"),
        ({"train_max_documents": 3}, "document count/EOS"),
        ({"revision": "changed"}, "identity mismatch"),
        ({"validation_max_documents": 2}, "identity mismatch"),
        ({"train_max_documents": 4}, "reduce both"),
    ],
)
def test_rejects_incompatible_selection(prefix_fixture, change, match):
    _source, target, tokenizer, parent = prefix_fixture
    altered = target.model_copy(
        update={"dataset": target.dataset.model_copy(update=change)}
    )
    with pytest.raises(ValueError, match=match):
        prepare_data(altered, tokenizer, prefix_from=request(parent))
    assert not list(target.dataset.cache_dir.glob("*/manifest.json"))


def test_rejects_wrong_parent_pin(prefix_fixture):
    _source, target, tokenizer, parent = prefix_fixture
    with pytest.raises(ValueError, match="pin mismatch"):
        prepare_data(
            target, tokenizer, prefix_from=PreparedPrefix(parent.root, "0" * 64)
        )


def test_rejects_changed_source(prefix_fixture):
    source, target, tokenizer, parent = prefix_fixture
    with source.dataset.train_path.open("a") as handle:
        handle.write(json.dumps({"text": "an unexpected additional record"}) + "\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        prepare_data(target, tokenizer, prefix_from=request(parent))


def test_rejects_corrupt_parent_array(prefix_fixture):
    _source, target, tokenizer, parent = prefix_fixture
    with (parent.root / "train.npy").open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\xff")
    with pytest.raises(ValueError, match="digest mismatch"):
        prepare_data(target, tokenizer, prefix_from=request(parent))


def test_rejects_changed_tokenizer(prefix_fixture):
    _source, target, tokenizer, parent = prefix_fixture
    tokenizer.add_tokens(["unexpected-new-token"])
    with pytest.raises(ValueError, match="identity mismatch"):
        prepare_data(target, tokenizer, prefix_from=request(parent))


def test_prefix_preserves_failed_streaming_chunks(prefix_fixture):
    _source, target, tokenizer, parent = prefix_fixture
    # Recover the deterministic destination from an independently prepared cache.
    fresh = target.model_copy(
        update={
            "dataset": target.dataset.model_copy(
                update={
                    "cache_dir": target.dataset.cache_dir.parent / "fresh-cache",
                }
            )
        }
    )
    expected = prepare_data(fresh, tokenizer)
    failed = target.dataset.cache_dir / (expected.root.name + ".tmp")
    failed.mkdir(parents=True)
    sentinel = failed / "retained-failure-evidence"
    sentinel.write_text("do not erase")
    result = prepare_data(target, tokenizer, prefix_from=request(parent))
    assert result.root.name == expected.root.name
    assert sentinel.read_text() == "do not erase"


def test_embedded_eos_cannot_masquerade_as_document_boundary(prefix_fixture):
    source, target, tokenizer, _parent = prefix_fixture
    source.dataset.train_path.write_text(
        "".join(
            json.dumps({"text": t}) + "\n"
            for t in [
                "one <eos> nested",
                "second",
                "third",
                "fourth",
            ]
        )
    )
    source = source.model_copy(
        update={"dataset": source.dataset.model_copy(update={"train_max_tokens": 1000})}
    )
    parent = prepare_data(source, tokenizer)
    eos = tokenizer.token_to_id("<eos>")
    boundary = int(np.flatnonzero(parent.train == eos)[1]) + 1
    target = target.model_copy(
        update={
            "dataset": source.dataset.model_copy(
                update={
                    "train_max_documents": 2,
                    "train_max_tokens": boundary,
                    "cache_dir": target.dataset.cache_dir,
                }
            )
        }
    )
    with pytest.raises(ValueError, match="document count/EOS"):
        prepare_data(target, tokenizer, prefix_from=request(parent))


def test_pin_checked_again_after_parent_verification(prefix_fixture, monkeypatch):
    import hashlib

    from sparselab.canonical import canonical_json
    from sparselab.data import packing

    _source, target, tokenizer, parent = prefix_fixture
    original_load = packing.load_prepared_data

    def concurrent_manifest_change(root, **kwargs):
        changed = dict(parent.manifest)
        changed.pop("manifest_sha256")
        changed["concurrent_edit"] = True
        changed["manifest_sha256"] = hashlib.sha256(canonical_json(changed)).hexdigest()
        (parent.root / "manifest.json").write_bytes(canonical_json(changed))
        return original_load(root, **kwargs)

    monkeypatch.setattr(packing, "load_prepared_data", concurrent_manifest_change)
    with pytest.raises(ValueError, match="pin changed"):
        prepare_data(target, tokenizer, prefix_from=request(parent))


def test_parent_mutation_during_copy_refuses_publication(prefix_fixture, monkeypatch):
    from sparselab.data import packing

    _source, target, tokenizer, parent = prefix_fixture
    original_write = packing._atomic_array

    def mutate_parent(path, values):
        proof = original_write(path, values)
        if path.name == "train.npy":
            with (parent.root / "train.npy").open("r+b") as handle:
                handle.seek(-1, 2)
                handle.write(b"\xff")
        return proof

    monkeypatch.setattr(packing, "_atomic_array", mutate_parent)
    with pytest.raises(ValueError, match="changed|proof"):
        prepare_data(target, tokenizer, prefix_from=request(parent))
    assert not list(target.dataset.cache_dir.glob("*/manifest.json"))


def test_publication_failure_preserves_staging_and_parent(prefix_fixture, monkeypatch):
    from sparselab.data import packing

    _source, target, tokenizer, parent = prefix_fixture
    original = (parent.root / "train.npy").read_bytes()

    def refuse_publication(*_args):
        raise OSError("injected publication failure")

    monkeypatch.setattr(packing, "_publish_prepared_directory", refuse_publication)
    with pytest.raises(OSError, match="publication failure"):
        prepare_data(target, tokenizer, prefix_from=request(parent))
    assert (parent.root / "train.npy").read_bytes() == original
    assert len(list(target.dataset.cache_dir.glob("*.prefix-*/manifest.json"))) == 1
    assert not list(target.dataset.cache_dir.glob("????????????????/manifest.json"))


def test_byte_sidecars_are_outside_prefix_scope(prefix_fixture):
    _source, target, tokenizer, parent = prefix_fixture
    target = target.model_copy(
        update={"model": target.model.model_copy(update={"memory": "byte"})}
    )
    with pytest.raises(ValueError, match="plain all-token"):
        prepare_data(target, tokenizer, prefix_from=request(parent))


def test_historical_producer_cannot_populate_current_preparation_cache(
    prefix_fixture, monkeypatch
):
    from sparselab.data import packing

    source, target, tokenizer, _parent = prefix_fixture
    # A valid pinned cache from a different preparation producer is insufficient
    # evidence to claim fresh encoding equivalence under the current producer.
    with monkeypatch.context() as older:
        older.setattr(packing, "preparation_source_identity", lambda: "a" * 64)
        historical = prepare_data(source, tokenizer)
    with pytest.raises(ValueError, match="identity mismatch"):
        prepare_data(target, tokenizer, prefix_from=request(historical))
    assert not list(target.dataset.cache_dir.glob("*/manifest.json"))
    fresh = prepare_data(target, tokenizer)
    assert fresh.manifest["cache_identity"]["source_identity_sha256"] != "a" * 64
    assert "prefix_derivation" not in fresh.manifest


def test_short_prefix_preserves_native_next_token_block_requirement(prefix_fixture):
    _source, target, tokenizer, parent = prefix_fixture
    first_end = (
        int(np.flatnonzero(parent.train == tokenizer.token_to_id("<eos>"))[0]) + 1
    )
    assert first_end < target.training.seq_len + 1
    target = target.model_copy(
        update={
            "dataset": target.dataset.model_copy(
                update={
                    "train_max_documents": 1,
                    "train_max_tokens": first_end,
                }
            )
        }
    )
    with pytest.raises(ValueError, match="full next-token block"):
        prepare_data(target, tokenizer)
    with pytest.raises(ValueError, match="full next-token block"):
        prepare_data(target, tokenizer, prefix_from=request(parent))


def test_prefix_preserves_validation_next_token_block_requirement(prefix_fixture):
    source, target, tokenizer, _parent = prefix_fixture
    source.dataset.validation_path.write_text(json.dumps({"text": "x"}) + "\n")
    source = source.model_copy(
        update={
            "training": source.training.model_copy(update={"seq_len": 1}),
            "dataset": source.dataset.model_copy(
                update={
                    "validation_max_tokens": len(
                        tokenizer.encode("x", add_special_tokens=False).ids
                    )
                    + 1,
                }
            ),
        }
    )
    parent = prepare_data(source, tokenizer)
    target = target.model_copy(
        update={
            "dataset": target.dataset.model_copy(
                update={
                    "validation_max_tokens": source.dataset.validation_max_tokens,
                }
            )
        }
    )
    assert len(parent.validation) < target.training.seq_len + 1
    assert target.dataset.train_max_tokens >= target.training.seq_len + 1
    with pytest.raises(ValueError, match="full next-token block"):
        prepare_data(target, tokenizer)
    with pytest.raises(ValueError, match="full next-token block"):
        prepare_data(target, tokenizer, prefix_from=request(parent))
