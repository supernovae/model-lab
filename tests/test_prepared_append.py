"""Authenticated fresh tails preserve historical IDs and native full state."""

import json

import numpy as np
import pytest
import torch
from test_document_budget_extension import append_config
from test_document_budget_extension import (
    document_parent as document_parent,  # noqa: PLC0414
)
from test_prepared_prefix import prefix_fixture as prefix_fixture  # noqa: PLC0414
from test_training import equal

from klab.data.encoding import PreparationEncoder
from klab.data.packing import PreparedAppend, load_prepared_data, prepare_data
from klab.data.story_packing import MatchedStoryDataset
from klab.training.checkpoints import CheckpointManager
from klab.training.manifest import sha256_file
from klab.training.trainer import train


@pytest.fixture
def append_fixture(prefix_fixture, tmp_path):
    source, _smaller, tokenizer, parent = prefix_fixture
    additions = ["A fresh account of careful engineering.", "New independent words."]
    path = tmp_path / "expanded.jsonl"
    path.write_bytes(
        source.dataset.train_path.read_bytes()
        + b"".join((json.dumps({"text": t}) + "\n").encode() for t in additions)
    )
    tokens = len(parent.train) + sum(
        len(tokenizer.encode(t, add_special_tokens=False).ids) + 1 for t in additions
    )
    target = source.model_copy(
        update={
            "dataset": source.dataset.model_copy(
                update={
                    "revision": "expanded-v2",
                    "train_path": path,
                    "train_max_documents": 6,
                    "train_max_tokens": tokens,
                    "cache_dir": tmp_path / "append-cache",
                }
            )
        }
    )
    return target, tokenizer, parent, additions


def request(parent):
    return PreparedAppend(parent.root, parent.manifest["manifest_sha256"])


def test_append_matches_fresh_and_encodes_only_tail(
    append_fixture, tmp_path, monkeypatch
):
    target, tokenizer, parent, additions = append_fixture
    expected = prepare_data(
        target.model_copy(
            update={
                "dataset": target.dataset.model_copy(
                    update={"cache_dir": tmp_path / "fresh"}
                )
            }
        ),
        tokenizer,
    )
    encoded = []
    original = PreparationEncoder.encode

    def track(self, texts):
        encoded.extend(texts)
        return original(self, texts)

    monkeypatch.setattr(PreparationEncoder, "encode", track)
    before = sha256_file(parent.root / "manifest.json")
    actual = prepare_data(target, tokenizer, append_from=request(parent))
    assert encoded == additions
    for split in ("train", "validation"):
        assert (actual.root / f"{split}.npy").read_bytes() == (
            expected.root / f"{split}.npy"
        ).read_bytes()
    assert actual.manifest["cache_identity"] == expected.manifest["cache_identity"]
    proof = actual.manifest["append_derivation"]
    assert proof["parent_manifest_sha256"] == parent.manifest["manifest_sha256"]
    assert proof["parent_train_tokens"] == len(parent.train)
    assert proof["tail_documents"] == 2 and proof["tail_tokens"] == len(
        actual.train
    ) - len(parent.train)
    assert sha256_file(parent.root / "manifest.json") == before
    assert prepare_data(target, tokenizer).root == actual.root
    assert (
        prepare_data(target, tokenizer, append_from=request(parent)).root == actual.root
    )
    assert encoded == additions
    np.testing.assert_array_equal(
        load_prepared_data(actual.root, byte_enabled=False).train, expected.train
    )


@pytest.mark.parametrize(
    "damage",
    [
        "raw_prefix",
        "array",
        "pin",
        "validation",
        "producer",
        "embedded_eos",
        "short_cap",
        "long_cap",
        "extra_record",
        "short_source",
    ],
)
def test_append_refuses_changed_history_and_invalid_tail(
    append_fixture, tmp_path, monkeypatch, damage
):
    target, tokenizer, parent, _ = append_fixture
    selected = request(parent)
    if damage == "raw_prefix":
        path = target.dataset.train_path
        path.write_bytes(
            path.read_bytes().replace(b"one small story", b"bad small story", 1)
        )
    elif damage == "array":
        path = parent.root / "train.npy"
        with path.open("r+b") as handle:
            handle.seek(-4, 2)
            handle.write(b"\xff" * 4)
    elif damage == "pin":
        selected = PreparedAppend(parent.root, "0" * 64)
    elif damage == "validation":
        path = tmp_path / "different-validation.jsonl"
        path.write_text('{"text":"Changed heldout."}\n')
        target = target.model_copy(
            update={
                "dataset": target.dataset.model_copy(update={"validation_path": path})
            }
        )
    elif damage == "producer":
        monkeypatch.setattr(
            "klab.data.packing.preparation_source_identity", lambda: "f" * 64
        )
    elif damage == "embedded_eos":
        path = target.dataset.train_path
        path.write_text(
            path.read_text().replace("New independent words.", "New <eos> words.")
        )
    elif damage in {"short_cap", "long_cap"}:
        target = target.model_copy(
            update={
                "dataset": target.dataset.model_copy(
                    update={
                        "train_max_tokens": target.dataset.train_max_tokens
                        + (-1 if damage == "short_cap" else 1)
                    }
                )
            }
        )
    elif damage == "extra_record":
        with target.dataset.train_path.open("a") as handle:
            handle.write('{"text":"Unselected trailing record."}\n')
    else:
        path = target.dataset.train_path
        path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[:-1]))
    with pytest.raises(ValueError):
        prepare_data(target, tokenizer, append_from=selected)
    assert not list(target.dataset.cache_dir.glob("*/manifest.json"))


def test_publication_failure_leaves_no_reusable_cache(append_fixture, monkeypatch):
    target, tokenizer, parent, _ = append_fixture

    def fail(*_):
        raise OSError("fixture publication failure")

    monkeypatch.setattr("klab.data.packing._publish_prepared_directory", fail)
    with pytest.raises(OSError, match="publication failure"):
        prepare_data(target, tokenizer, append_from=request(parent))
    assert list(target.dataset.cache_dir.glob("*.append-*/manifest.json"))
    assert not [
        p
        for p in target.dataset.cache_dir.glob("*/manifest.json")
        if ".append-" not in p.parent.name
    ]


def test_append_preserves_groups_and_cold_full_state_resume(
    document_parent, tmp_path, monkeypatch
):
    source, root, generation, tokenizer = document_parent
    target = append_config(document_parent, tmp_path)
    fresh = prepare_data(target, tokenizer)
    target = target.model_copy(
        update={
            "dataset": target.dataset.model_copy(
                update={"cache_dir": tmp_path / "derived"}
            )
        }
    )
    parent = load_prepared_data(root / "data", byte_enabled=False)
    derived = prepare_data(target, tokenizer, append_from=request(parent))
    np.testing.assert_array_equal(derived.train, fresh.train)
    settings = {
        "layout": "story",
        "targets_per_update": source.training.matched_stories.targets_per_update,
        "unpadded_documents": True,
    }
    old_plan = MatchedStoryDataset(parent.train, 64, 3, **settings)
    new_plan = MatchedStoryDataset(
        derived.train, 64, 3, frozen_update_ends=(len(parent.train),), **settings
    )
    np.testing.assert_array_equal(
        old_plan.group_offsets, new_plan.group_offsets[: len(old_plan) + 1]
    )
    before = {p.name: sha256_file(p) for p in generation.iterdir()}

    def no_encoding(*_args, **_kwargs):
        pytest.fail("native continuation must reuse the authenticated derived cache")

    monkeypatch.setattr(PreparationEncoder, "encode", no_encoding)
    train(
        target,
        extend_budget=generation,
        append_data=True,
        run_id="append-part",
        stop_after_step=2,
    )
    part = target.logging.root_dir / "append-part"
    saved = CheckpointManager(part).load(
        part / "checkpoints/latest.json", verification_mode="cold"
    )
    assert saved.step == 2 and saved.cursor == (0, 2)
    assert all(float(v["step"]) == 2 for v in saved.optimizer["state"].values())
    train(
        target,
        resume=part / "checkpoints/latest.json",
        run_id="append-resumed",
        stop_after_step=3,
    )
    train(
        target,
        extend_budget=generation,
        append_data=True,
        run_id="append-whole",
        stop_after_step=3,
    )
    states = [
        CheckpointManager(target.logging.root_dir / n).load(
            target.logging.root_dir / n / "checkpoints/latest.json",
            verification_mode="cold",
        )
        for n in ("append-resumed", "append-whole")
    ]
    for field in ("model", "optimizer", "rng", "schedule", "cursor"):
        equal(getattr(states[0], field), getattr(states[1], field))
    assert states[0].cursor == (0, 3) and states[0].tokens_seen == states[1].tokens_seen
    assert states[0].schedule["decay_steps"] == source.training.max_steps
    assert before == {p.name: sha256_file(p) for p in generation.iterdir()}
    assert not torch.xpu.is_initialized() and not torch.cuda.is_initialized()


def test_cached_child_does_not_authorize_unrelated_parent(append_fixture, tmp_path):
    target, tokenizer, parent, _ = append_fixture
    prepare_data(target, tokenizer)
    old = tmp_path / "unrelated-parent.jsonl"
    rows = target.dataset.train_path.read_bytes().splitlines(keepends=True)[:4]
    old.write_bytes(b"".join(rows).replace(b"one small story", b"two small story", 1))
    unrelated_cfg = target.model_copy(
        update={
            "dataset": target.dataset.model_copy(
                update={
                    "train_path": old,
                    "train_max_documents": 4,
                    "train_max_tokens": len(parent.train) + 20,
                    "cache_dir": tmp_path / "other",
                }
            )
        }
    )
    unrelated = prepare_data(unrelated_cfg, tokenizer)
    with pytest.raises(ValueError, match="historical raw prefix digest mismatch"):
        prepare_data(target, tokenizer, append_from=request(unrelated))


def test_source_change_during_encoding_refuses_publication(append_fixture, monkeypatch):
    target, tokenizer, parent, _ = append_fixture
    original = PreparationEncoder.encode

    def change_source(self, texts):
        result = original(self, texts)
        path = target.dataset.train_path
        path.write_bytes(
            path.read_bytes().replace(b"one small story", b"bad small story", 1)
        )
        return result

    monkeypatch.setattr(PreparationEncoder, "encode", change_source)
    with pytest.raises(ValueError, match="source changed before publication"):
        prepare_data(target, tokenizer, append_from=request(parent))
    assert not list(target.dataset.cache_dir.glob("*/manifest.json"))


def test_append_requires_frozen_parent_boundary(document_parent, tmp_path):
    _source, root, _generation, tokenizer = document_parent
    target = append_config(document_parent, tmp_path)
    raw = target.model_dump(mode="json")
    raw["training"]["matched_stories"]["frozen_update_ends"] = []
    raw["dataset"]["cache_dir"] = str(tmp_path / "derived-unfrozen")
    with pytest.raises(ValueError, match="frozen parent update boundary"):
        prepare_data(
            type(target).model_validate(raw),
            tokenizer,
            append_from=request(load_prepared_data(root / "data", byte_enabled=False)),
        )


def test_append_signed_reuse_then_tamper_refusal(append_fixture, tmp_path):
    from klab.verification_proofs import ProofStore

    target, tokenizer, parent, _ = append_fixture
    derived = prepare_data(target, tokenizer, append_from=request(parent))
    store = ProofStore(tmp_path)
    assert store.key_available(create=True)
    for _ in range(2):
        load_prepared_data(
            derived.root,
            byte_enabled=False,
            proof_store=store,
            verification_mode="verified_reuse",
        )
    assert store.diagnostics()["hits"] > 0
    with (derived.root / "train.npy").open("r+b") as handle:
        handle.seek(-4, 2)
        handle.write(b"\xff" * 4)
    with pytest.raises(ValueError, match="digest mismatch"):
        load_prepared_data(
            derived.root,
            byte_enabled=False,
            proof_store=store,
            verification_mode="verified_reuse",
        )


@pytest.mark.parametrize("mode", ["byte", "compatibility", "both"])
def test_append_rejects_unsupported_producer_paths(append_fixture, monkeypatch, mode):
    from klab.data.packing import PreparedPrefix

    target, tokenizer, parent, _ = append_fixture
    options = {"append_from": request(parent)}
    if mode == "byte":
        target = target.model_copy(
            update={"model": target.model.model_copy(update={"memory": "byte"})}
        )
    elif mode == "compatibility":
        monkeypatch.setattr(
            "klab.experiments.source_compatibility.active_source_compatibility",
            dict,
        )
    else:
        options["prefix_from"] = PreparedPrefix(
            parent.root, parent.manifest["manifest_sha256"]
        )
    with pytest.raises(ValueError):
        prepare_data(target, tokenizer, **options)
    assert not list(target.dataset.cache_dir.glob("*/manifest.json"))


def test_new_tail_cannot_overlap_unchanged_validation(append_fixture):
    target, tokenizer, parent, _ = append_fixture
    path = target.dataset.train_path
    rows = path.read_bytes().splitlines(keepends=True)
    rows[-1] = target.dataset.validation_path.read_bytes()
    path.write_bytes(b"".join(rows))
    with pytest.raises(ValueError, match="overlapping conversation"):
        prepare_data(target, tokenizer, append_from=request(parent))
