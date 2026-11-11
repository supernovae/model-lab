from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_training import config

from klab.canonical import canonical_digest, canonical_json
from klab.config.loading import load_config
from klab.config.models import AttentionConfig, DatasetConfig, ModelConfig
from klab.data.allocation import build_allocation_manifest
from klab.data.allocation_tasks import _collect_provenance
from klab.data.packing import (
    canonical_tokenizer_sha256,
    load_prepared_data,
    prepare_data,
)
from klab.data.tokenizer import load_tokenizer
from klab.engines.base import Microbatch
from klab.engines.pytorch import PyTorchEngine
from klab.model.transformer import DenseLM
from klab.training.manifest import sha256_file, source_identity


def _allocation_config(tmp_path: Path, *, weight: float, memory: bool = True):
    base = config(tmp_path)
    tokenizer = load_tokenizer(base.tokenizer.path)
    manifest = build_allocation_manifest(
        tmp_path / "allocation.json",
        source_identity_sha256=source_identity()["sha256"],
        tokenizer_sha256=canonical_tokenizer_sha256(tokenizer),
        train_jsonl_sha256="0" * 64,
        validation_jsonl_sha256="1" * 64,
        train_owner=np.array([1], dtype=np.uint8),
        validation_owner=np.array([1], dtype=np.uint8),
    )
    model_update = (
        {"memory": "ngram", "memory_table_size": 17, "memory_dim": 8} if memory else {}
    )
    return base.model_copy(
        update={
            "model": base.model.model_copy(update=model_update),
            "dataset": base.dataset.model_copy(
                update={"allocation_manifest_path": manifest.path}
            ),
            "training": base.training.model_copy(update={"neural_loss_weight": weight}),
        }
    )


def _lexical_batch() -> Microbatch:
    return Microbatch(
        np.array([[1, 2, 3, 4]], dtype=np.int64),
        np.array([[2, 3, 4, 5]], dtype=np.int64),
        owner_ids=np.ones((1, 4), dtype=np.uint8),
    )


def test_non_owner_memory_residual_is_output_and_gradient_isolated() -> None:
    torch.manual_seed(7)
    lexical = DenseLM(
        ModelConfig(
            vocab_size=260,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            ffn_dim=16,
            max_seq_len=4,
            memory="ngram",
            memory_table_size=17,
            memory_dim=8,
            memory_ngram_size=3,
        ),
        AttentionConfig(),
    )
    baseline = DenseLM(
        ModelConfig(
            vocab_size=260,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            ffn_dim=16,
            max_seq_len=4,
        ),
        AttentionConfig(),
    )
    baseline.load_state_dict(
        {
            name: value
            for name, value in lexical.state_dict().items()
            if not name.startswith("memory.")
        }
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[2, 3, 4, 5]])
    masked_logits, _ = lexical.forward_with_aux(
        tokens, memory_mask=torch.zeros_like(tokens, dtype=torch.bool)
    )
    baseline_logits, _ = baseline.forward_with_aux(tokens)
    assert torch.equal(masked_logits, baseline_logits)
    torch.nn.functional.cross_entropy(
        masked_logits.flatten(0, 1), labels.flatten()
    ).backward()
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for parameter in lexical.memory.parameters()
    )


def test_lexical_owner_excludes_neural_adamw_updates_at_zero_weight(
    tmp_path: Path,
) -> None:
    engine = PyTorchEngine()
    engine.initialize(_allocation_config(tmp_path, weight=0.0))
    assert engine.model is not None
    before = copy.deepcopy(engine.model.state_dict())
    try:
        result = engine.train_update([_lexical_batch()], 1, 4)
        neural = {
            name: parameter
            for name, parameter in engine.model.named_parameters()
            if not name.startswith(("memory.", "semantic_memories."))
        }
        assert all(parameter.grad is None for parameter in neural.values())
        assert all(
            torch.equal(before[name], parameter) for name, parameter in neural.items()
        )
        assert any(
            not torch.equal(before[name], parameter)
            for name, parameter in engine.model.named_parameters()
            if name.startswith("memory.")
        )
    finally:
        engine.close()
    assert result.metrics["allocation/weighted_neural_supervision_mass"] == 0


def test_neural_owner_weight_one_matches_standard_neural_update(tmp_path: Path) -> None:
    allocated = PyTorchEngine()
    baseline = PyTorchEngine()
    allocated.initialize(
        _allocation_config(tmp_path / "allocated", weight=1.0, memory=False)
    )
    baseline.initialize(config(tmp_path / "baseline"))
    assert allocated.model is not None and baseline.model is not None
    baseline.model.load_state_dict(copy.deepcopy(allocated.model.state_dict()))
    neural_batch = Microbatch(
        np.array([[1, 2, 3, 4]], dtype=np.int64),
        np.array([[2, 3, 4, 5]], dtype=np.int64),
        owner_ids=np.zeros((1, 4), dtype=np.uint8),
    )
    try:
        allocated.train_update([neural_batch], 1, 4)
        baseline.train_update(
            [Microbatch(neural_batch.inputs, neural_batch.targets)], 1, 4
        )
        for (name, value), (other_name, other) in zip(
            allocated.model.state_dict().items(),
            baseline.model.state_dict().items(),
            strict=True,
        ):
            assert name == other_name
            assert torch.equal(value, other)
    finally:
        allocated.close()
        baseline.close()


def test_neural_loss_weights_scale_neural_gradients_from_one_to_zero(
    tmp_path: Path,
) -> None:
    batch = Microbatch(
        np.array([[1, 2, 3, 4]], dtype=np.int64),
        np.array([[2, 3, 4, 5]], dtype=np.int64),
        owner_ids=np.zeros((1, 4), dtype=np.uint8),
    )

    def configured(path: Path, weight: float):
        result = _allocation_config(path, weight=weight, memory=False)
        training = result.training.model_copy(update={"grad_clip_norm": 1e6})
        return result.model_copy(update={"training": training})

    reference = PyTorchEngine()
    reference.initialize(configured(tmp_path / "reference", 1.0))
    assert reference.model is not None
    initial_state = copy.deepcopy(reference.model.state_dict())
    try:
        reference.train_update([batch], 1, 4)
        reference_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in reference.model.named_parameters()
            if parameter.grad is not None
        }
    finally:
        reference.close()

    for weight in (1.0, 0.75, 0.5, 0.25, 0.0):
        engine = PyTorchEngine()
        engine.initialize(configured(tmp_path / f"weight-{weight}", weight))
        assert engine.model is not None
        engine.model.load_state_dict(initial_state)
        try:
            result = engine.train_update([batch], 1, 4)
            assert result.metrics["allocation/weighted_neural_supervision_mass"] == (
                4 * weight
            )
            parameters = dict(engine.model.named_parameters())
            if weight == 0:
                assert all(parameter.grad is None for parameter in parameters.values())
                assert all(
                    torch.equal(initial_state[name], parameter)
                    for name, parameter in parameters.items()
                )
            else:
                assert set(reference_gradients) == set(parameters)
                for name, parameter in parameters.items():
                    assert parameter.grad is not None
                    torch.testing.assert_close(
                        parameter.grad,
                        reference_gradients[name] * weight,
                        rtol=1e-5,
                        atol=1e-7,
                    )
        finally:
            engine.close()


def test_allocation_mass_counters_keep_raw_and_hybrid_distinct(tmp_path: Path) -> None:
    engine = PyTorchEngine()
    engine.initialize(_allocation_config(tmp_path, weight=0.5))
    batch = Microbatch(
        np.array([[1, 2, 3, 4]], dtype=np.int64),
        np.array([[2, -100, -100, -100]], dtype=np.int64),
        owner_ids=np.array([[1, 2, 3, 0]], dtype=np.uint8),
    )
    try:
        result = engine.train_update([batch], 1, 1)
    finally:
        engine.close()
    assert result.metrics["allocation/raw_tokens"] == 4
    assert result.metrics["allocation/valid_targets"] == 1
    assert result.metrics["allocation/owner/hybrid_raw_tokens"] == 1
    assert result.metrics["allocation/owner/hybrid_targets"] == 0
    assert result.metrics["allocation/weighted_neural_supervision_mass"] == 0


def _array_metadata(
    path: Path, *, dtype: np.dtype | None = None, dimensions: int = 1
) -> dict[str, object]:
    dtype = np.dtype(np.int32) if dtype is None else dtype
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    assert values.dtype == dtype and values.ndim == dimensions
    return {
        "dtype": values.dtype.name,
        "shape": list(values.shape),
        "tokens": int(values.shape[0]),
        "sha256": sha256_file(path),
    }


def _prepared_cache(root: Path) -> None:
    root.mkdir()
    arrays = {
        "train.npy": np.array([1, 2, 3, 4], dtype=np.int32),
        "validation.npy": np.array([4, 3, 2, 1], dtype=np.int32),
        "train_supervision.npy": np.ones(4, dtype=bool),
        "validation_supervision.npy": np.ones(4, dtype=bool),
        "train_owner_ids.npy": np.array([2, 2, 3, 3], dtype=np.uint8),
        "validation_owner_ids.npy": np.array([2, 2, 3, 3], dtype=np.uint8),
        "train_semantic_queries.npy": np.ones((4, 2), dtype=np.float32),
        "validation_semantic_queries.npy": np.ones((4, 2), dtype=np.float32),
        "train_semantic_mask.npy": np.array([True, True, True, False], dtype=bool),
        "validation_semantic_mask.npy": np.array([True, True, True, False], dtype=bool),
    }
    for name, values in arrays.items():
        np.save(root / name, values, allow_pickle=False)
    identity: dict[str, object] = {}
    allocation: dict[str, object] = {
        "format": "klab-prepared-allocation-v1",
        "manifest_sha256": "a" * 64,
        "owner_codes": {"neural": 0, "lexical": 1, "semantic": 2, "hybrid": 3},
        "semantic": {"verified": True},
    }
    for split in ("train", "validation"):
        allocation[split] = {
            "owner": _array_metadata(
                root / f"{split}_owner_ids.npy", dtype=np.dtype(np.uint8)
            ),
            "semantic_queries": _array_metadata(
                root / f"{split}_semantic_queries.npy",
                dtype=np.dtype(np.float32),
                dimensions=2,
            ),
            "semantic_mask": _array_metadata(
                root / f"{split}_semantic_mask.npy", dtype=np.dtype(bool)
            ),
        }
    manifest = {
        "packing_version": "contiguous-eos-v5",
        "cache_identity": identity,
        "settings_sha256": canonical_digest(identity),
        "train": _array_metadata(root / "train.npy"),
        "validation": _array_metadata(root / "validation.npy"),
        "supervision": {
            "kind": "token-loss-mask-v1",
            "train": _array_metadata(
                root / "train_supervision.npy", dtype=np.dtype(bool)
            ),
            "validation": _array_metadata(
                root / "validation_supervision.npy", dtype=np.dtype(bool)
            ),
        },
        "byte_addressing": None,
        "allocation": allocation,
    }
    manifest["manifest_sha256"] = canonical_digest(manifest)
    (root / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("train_semantic_queries.npy", np.zeros((4, 2), dtype=np.float32)),
        ("train_semantic_mask.npy", np.zeros(4, dtype=bool)),
    ],
)
def test_prepared_semantic_query_or_mask_tampering_fails_closed(
    tmp_path: Path, name: str, values: np.ndarray
) -> None:
    root = tmp_path / "prepared"
    _prepared_cache(root)
    assert (
        load_prepared_data(root, byte_enabled=False).train_semantic_queries is not None
    )
    np.save(root / name, values, allow_pickle=False)
    with pytest.raises(ValueError):
        load_prepared_data(root, byte_enabled=False)


def test_prepared_owner_tampering_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "prepared"
    _prepared_cache(root)
    np.save(
        root / "train_owner_ids.npy",
        np.array([2, 2, 3, 9], dtype=np.uint8),
        allow_pickle=False,
    )
    with pytest.raises(ValueError):
        load_prepared_data(root, byte_enabled=False)


def test_allocation_corpus_digest_mismatch_fails_before_preparation(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    record = (
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        )
        + "\n"
    )
    train.write_text(record * 20, encoding="utf-8")
    validation.write_text(record.replace("answer", "different") * 20, encoding="utf-8")
    base = config(tmp_path / "source")
    tokenizer = load_tokenizer(base.tokenizer.path)
    manifest = build_allocation_manifest(
        tmp_path / "bad-allocation.json",
        source_identity_sha256=source_identity()["sha256"],
        tokenizer_sha256=canonical_tokenizer_sha256(tokenizer),
        train_jsonl_sha256="0" * 64,
        validation_jsonl_sha256="1" * 64,
        train_owner=np.zeros(1, dtype=np.uint8),
        validation_owner=np.zeros(1, dtype=np.uint8),
    )
    local = base.dataset.model_copy(
        update={
            "source": "local_chat",
            "train_path": train,
            "validation_path": validation,
            "license": "CC0",
            "allocation_manifest_path": manifest.path,
        }
    )
    with pytest.raises(ValueError, match="corpus digests"):
        prepare_data(base.model_copy(update={"dataset": local}), tokenizer)


@pytest.mark.parametrize(
    ("audited_path", "audited_answer", "message"),
    [
        ("/tmp/file.py", "wrong.py", "source answer differs"),
        ("/tmp/other.py", "file.py", "source prompt differs"),
    ],
)
def test_provenance_requires_audited_path_and_answer(
    tmp_path: Path,
    audited_path: str,
    audited_answer: str,
    message: str,
) -> None:
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "format_version": 2,
                "loss_mode": "assistant_only",
                "messages": [
                    {
                        "role": "user",
                        "content": "Return the name for path /tmp/file.py.",
                    },
                    {"role": "assistant", "content": "file.py"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validation_corpus = tmp_path / "validation.jsonl"
    validation_corpus.write_text(corpus.read_text(encoding="utf-8"), encoding="utf-8")
    base = config(tmp_path / "tokenizer")
    tokenizer = load_tokenizer(base.tokenizer.path)
    dataset = DatasetConfig(
        source="local_chat",
        cache_dir=tmp_path / "cache",
        train_max_documents=1,
        validation_max_documents=1,
        train_max_tokens=256,
        validation_max_tokens=256,
        train_path=corpus,
        validation_path=validation_corpus,
        license="MIT",
    )
    record = {
        "record_id": "path-domain-train-01",
        "rendered_assistant_label": audited_answer,
        "semantic_key": {
            "path": audited_path,
            "operation": "name",
            "argument": None,
        },
    }
    with pytest.raises(ValueError, match=message):
        _collect_provenance(dataset, tokenizer, "train", [record])


def test_explicit_trainable_parameters_freeze_optimizer_and_update():
    config = load_config(Path("configs/runtime_smoke_cpu.yaml"))
    config = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "gradient_accumulation": 1,
                    "trainable_parameters": ("embedding.weight",),
                }
            )
        }
    )
    torch.manual_seed(config.seed)
    engine = PyTorchEngine()
    engine.initialize(config)
    assert engine.model is not None
    model = engine.model
    initial = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == {"embedding.weight"}

    result = engine.train_update(
        [
            Microbatch(
                np.array([[1, 2, 3, 4]], dtype=np.int64),
                np.array([[2, 3, 4, 5]], dtype=np.int64),
            )
        ],
        update_index=2,
        valid_targets=4,
    )

    assert result.outcome == "APPLIED"
    assert engine._last_gradient_parameter_names == ("embedding.weight",)
    for name, parameter in model.named_parameters():
        if name == "embedding.weight":
            assert not torch.equal(parameter.detach(), initial[name])
        else:
            assert torch.equal(parameter.detach(), initial[name])
