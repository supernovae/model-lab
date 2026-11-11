from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from klab.canonical import canonical_json
from klab.config.models import AttentionConfig, ModelConfig
from klab.engram.packs import compile_pack
from klab.engram.semantic import (
    SemanticMemoryAdapter,
    SemanticQueryBatch,
    SemanticRetriever,
)
from klab.evaluation.generation import generate
from klab.model.transformer import DenseLM

_CREATED_AT = "2026-09-23T00:00:00Z"


def _retriever(
    root,
    *,
    name: str,
    record_ids: tuple[str, ...],
    keys: list[list[float]],
    values: list[list[float]] | None = None,
    key_encoder_name: str | None = None,
    value_encoder_name: str = "test-value-encoder",
    key_normalization: str = "none",
    validity: dict[str, dict[str, str]] | None = None,
) -> SemanticRetriever:
    root.mkdir(parents=True)
    source = root / "records.jsonl"
    validity = validity or {}
    rows = [
        {
            "record_version": 1,
            "id": record_id,
            "namespace": "semantic-test",
            "subject": f"subject-{record_id}",
            "relation": "related-to",
            "value": f"value-{record_id}",
            "license": "CC0-1.0",
            "created_at": _CREATED_AT,
            **validity.get(record_id, {}),
        }
        for record_id in record_ids
    ]
    source.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    key_array = np.asarray(keys, dtype=np.float32)
    value_array = np.asarray(
        values
        if values is not None
        else [[float(index + 1)] * 5 for index in range(len(record_ids))],
        dtype=np.float32,
    )
    keys_path = root / "keys.safetensors"
    values_path = root / "values.safetensors"
    metadata_path = root / "semantic.json"
    save_file({"keys": key_array}, keys_path)
    save_file({"values": value_array}, values_path)

    def identity(label: str) -> dict[str, str]:
        return {
            "name": label,
            "revision": "frozen-v1",
            "sha256": hashlib.sha256(label.encode()).hexdigest(),
        }

    key_name = key_encoder_name or f"{name}-key-encoder"
    metadata_path.write_bytes(
        canonical_json(
            {
                "format": "klab-semantic-assets",
                "format_version": 1,
                "record_ids": record_ids,
                "key_encoder": identity(key_name),
                "value_encoder": identity(value_encoder_name),
                "key_normalization": key_normalization,
            }
        )
        + b"\n"
    )
    manifest = compile_pack(
        source,
        root / "pack.enpack",
        name=f"semantic-test-{name}",
        namespace="semantic-test",
        default_license="CC0-1.0",
        source_name="semantic-test-fixture",
        source_revision="frozen-v1",
        created_at=_CREATED_AT,
        semantic_keys=keys_path,
        semantic_values=values_path,
        semantic_metadata=metadata_path,
    )
    return SemanticRetriever.from_pack(
        root / "pack.enpack", expected_pack_id=manifest.pack_id
    )


def _query(vector: list[float]) -> torch.Tensor:
    return torch.tensor(vector, dtype=torch.float32)


def _small_model() -> DenseLM:
    config = ModelConfig(
        vocab_size=260,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        ffn_dim=32,
        max_seq_len=8,
    )
    return DenseLM(config, AttentionConfig())


def test_exact_dot_cosine_ties_thresholds_and_comparison_bounds(
    tmp_path, monkeypatch
) -> None:
    dot = _retriever(
        tmp_path / "dot",
        name="dot",
        record_ids=("z-record", "a-record", "other"),
        keys=[[1, 0, 0], [1, 0, 0], [0, 1, 0]],
    )
    outcome = dot.retrieve(
        _query([1, 0, 0]),
        key_encoder=dot.key_encoder,
        top_k=2,
        min_score=0.9,
    )
    assert outcome.status == "conflict"
    assert tuple(hit.record_id for hit in outcome.hits) == ("a-record", "z-record")
    assert outcome.trace.tie_count == 2
    assert outcome.trace.tied_record_ids == ("a-record", "z-record")
    assert outcome.trace.comparison_count == 3
    assert outcome.trace.candidate_count == 3
    first_value = outcome.hits[0].value_vector.clone()
    outcome.hits[0].value_vector.fill_(999)
    repeated = dot.retrieve(_query([1, 0, 0]), key_encoder=dot.key_encoder, top_k=1)
    torch.testing.assert_close(repeated.hits[0].value_vector, first_value)

    unknown = dot.retrieve(
        _query([0, 0, 1]), key_encoder=dot.key_encoder, min_score=0.5
    )
    assert unknown.status == "unknown"
    assert unknown.hits == ()
    assert unknown.trace.best_score == 0.0

    cosine = _retriever(
        tmp_path / "cosine",
        name="cosine",
        record_ids=("x", "y"),
        keys=[[1, 0, 0], [0, 1, 0]],
        key_normalization="l2",
    )
    cosine_result = cosine.retrieve(_query([5, 0, 0]), key_encoder=cosine.key_encoder)
    assert cosine_result.status == "hit"
    assert cosine_result.trace.metric == "cosine"
    assert cosine_result.hits[0].score == 1.0

    bounded = SemanticRetriever.from_pack(
        dot.pack.root, expected_pack_id=dot.pack_id, max_comparisons=3
    )
    with pytest.raises(ValueError, match="exact retrieval needs 6 comparisons"):
        bounded.retrieve_many(
            torch.tensor([[1.0, 0, 0], [0, 1, 0]]),
            key_encoder=bounded.key_encoder,
        )
    monkeypatch.setattr("klab.engram.semantic.MAX_SEMANTIC_RESULT_BYTES", 4)
    with pytest.raises(ValueError, match="semantic results need up to"):
        dot.retrieve(_query([1, 0, 0]), key_encoder=dot.key_encoder, top_k=1)


def test_temporal_filter_is_inclusive_and_distinguishes_miss_from_unknown(
    tmp_path,
) -> None:
    retriever = _retriever(
        tmp_path / "temporal",
        name="temporal",
        record_ids=("expired", "current", "boundary"),
        keys=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        validity={
            "expired": {"valid_until": "2020-12-31"},
            "boundary": {
                "valid_from": "2025-01-01",
                "valid_until": "2025-01-01",
            },
        },
    )
    expired = retriever.retrieve(
        _query([1, 0, 0]),
        key_encoder=retriever.key_encoder,
        min_score=0.9,
        as_of="2025-01-01",
    )
    assert expired.status == "temporal_miss"
    assert expired.trace.temporal_excluded_count == 1
    assert expired.trace.best_record_id == "expired"

    boundary = retriever.retrieve(
        _query([0, 0, 1]),
        key_encoder=retriever.key_encoder,
        min_score=0.9,
        as_of="2025-01-01",
    )
    assert boundary.status == "hit"
    assert boundary.hits[0].record_id == "boundary"

    unknown = retriever.retrieve(
        _query([1, 1, 0]),
        key_encoder=retriever.key_encoder,
        min_score=1.5,
        as_of="2025-01-01",
    )
    assert unknown.status == "unknown"
    assert unknown.trace.candidate_count == 2


def test_temporal_as_of_flows_through_batched_adapter_queries(tmp_path) -> None:
    retriever = _retriever(
        tmp_path / "query-batch-temporal",
        name="query-batch-temporal",
        record_ids=("expired", "current"),
        keys=[[1, 0, 0], [0, 1, 0]],
        validity={"expired": {"valid_until": "2020-12-31"}},
    )
    adapter = SemanticMemoryAdapter(retriever, hidden_dim=4, min_score=0.9)
    queries = SemanticQueryBatch(
        retriever.key_encoder,
        torch.tensor([[[1.0, 0, 0], [1.0, 0, 0]]]),
        mask=torch.ones((1, 2), dtype=torch.bool),
        as_of=("2025-01-01", "2019-01-01"),
    )

    adapter(torch.zeros((1, 2, 4)), queries)

    assert tuple(trace.status for trace in adapter.last_traces) == (
        "temporal_miss",
        "hit",
    )
    assert tuple(trace.as_of for trace in adapter.last_traces) == (
        "2025-01-01T00:00:00+00:00",
        "2019-01-01T00:00:00+00:00",
    )


def test_adapter_keeps_pack_assets_external_and_replacement_preserves_weights(
    tmp_path,
) -> None:
    original = _retriever(
        tmp_path / "original",
        name="original",
        record_ids=("fact",),
        keys=[[1, 0, 0]],
        key_encoder_name="shared-key",
        value_encoder_name="shared-value",
    )
    replacement = _retriever(
        tmp_path / "replacement",
        name="replacement",
        record_ids=("fact",),
        keys=[[1, 0, 0]],
        values=[[9, 8, 7, 6, 5]],
        key_encoder_name="shared-key",
        value_encoder_name="shared-value",
    )
    adapter = SemanticMemoryAdapter(original, hidden_dim=7)
    assert original.key_dim == 3
    assert original.memory_dim == 5
    assert adapter.output.weight.shape == (7, 5)
    assert set(adapter.state_dict()) == {"output.weight", "gate.weight"}
    assert not original._keys.requires_grad and not original._values.requires_grad

    with torch.no_grad():
        adapter.output.weight.fill_(0.1)
        adapter.gate.weight.fill_(0.1)
    hidden = torch.ones((1, 2, 7), requires_grad=True)
    query_vectors = torch.tensor([[1.0, 0, 0]], requires_grad=True)
    result = adapter(hidden, SemanticQueryBatch(original.key_encoder, query_vectors))
    result.sum().backward()
    assert adapter.output.weight.grad is not None
    assert adapter.gate.weight.grad is not None
    assert hidden.grad is not None
    assert query_vectors.grad is None
    assert len(adapter.last_traces) == 2
    assert adapter.last_traces[-1].best_record_id == "fact"

    parameters = {
        name: value.detach().clone() for name, value in adapter.named_parameters()
    }
    adapter.replace_retriever(replacement)
    assert all(
        torch.equal(parameters[name], value)
        for name, value in adapter.named_parameters()
    )
    with pytest.raises(ValueError, match="feature contract"):
        incompatible = _retriever(
            tmp_path / "incompatible",
            name="incompatible",
            record_ids=("fact",),
            keys=[[1, 0, 0]],
            key_encoder_name="different-key",
            value_encoder_name="shared-value",
        )
        adapter.replace_retriever(incompatible)
    adapter.freeze()
    assert all(not parameter.requires_grad for parameter in adapter.parameters())


def test_dense_lm_hybrid_sites_and_cached_forward_parity(tmp_path) -> None:
    sites: tuple[
        tuple[str, Literal["embedding", "after_block", "final"], int | None], ...
    ] = (
        ("at-embedding", "embedding", None),
        ("after-zero", "after_block", 0),
        ("after-one", "after_block", 1),
        ("at-final", "final", None),
    )
    model = _small_model()
    retrievers = {
        name: _retriever(
            tmp_path / name,
            name=name,
            record_ids=("fact",),
            keys=[[1, 0, 0]],
            key_encoder_name=f"key-{name}",
            value_encoder_name=f"value-{name}",
        )
        for name, _, _ in sites
    }
    query_batches = {}
    for name, site, block_index in sites:
        retriever = retrievers[name]
        model.add_semantic_memory(
            name, retriever, site=site, block_index=block_index, min_score=0.9
        )
        query_batches[retriever.key_encoder.sha256] = SemanticQueryBatch(
            retriever.key_encoder, torch.tensor([[1.0, 0, 0]])
        )
    model.eval()
    input_ids = torch.tensor([[3, 7, 5, 2]])
    with torch.inference_mode():
        full = model(input_ids, semantic_queries=query_batches)
        cached, _ = model.forward_cached(
            input_ids,
            cache_capacity=8,
            semantic_queries=query_batches,
        )
        full_trace_counts = {
            name: len(adapter.last_traces)
            for name, adapter in model.semantic_memories.items()
        }
        incremental_cache = None
        incremental_logits = []
        for token_index in range(input_ids.shape[1]):
            kwargs = (
                {"cache_capacity": 8}
                if incremental_cache is None
                else {"cache": incremental_cache}
            )
            token_logits, incremental_cache = model.forward_cached(
                input_ids[:, token_index : token_index + 1],
                semantic_queries=query_batches,
                **kwargs,
            )
            incremental_logits.append(token_logits)
        incremental = torch.cat(incremental_logits, dim=1)
    torch.testing.assert_close(cached, full, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-5)
    assert model.semantic_memories["after-zero"].block_index == 0
    assert model.semantic_memories["after-one"].block_index == 1
    assert all(count == 4 for count in full_trace_counts.values())
    assert all(
        len(adapter.last_traces) == 1 for adapter in model.semantic_memories.values()
    )
    metrics = model.architecture_metric_tensors()
    assert "semantic/at-embedding/injection/embedding" in metrics
    assert "semantic/after-zero/injection/after_block/0" in metrics
    assert "semantic/after-one/injection/after_block/1" in metrics
    assert "semantic/at-final/injection/final" in metrics

    with pytest.raises(ValueError, match="exactly the attached key encoders"):
        model(input_ids, semantic_queries={})
    with pytest.raises(ValueError, match="requires explicit query vectors"):
        model(input_ids)

    model.requires_grad_(False)
    for adapter in model.semantic_memories.values():
        adapter.freeze()
    with torch.inference_mode():
        frozen_logits = model(input_ids, semantic_queries=query_batches)
    assert frozen_logits.shape == (1, 4, model.config.vocab_size)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_frozen_moe_accepts_unseen_pack_without_changing_adapter_or_gradients(
    tmp_path,
) -> None:
    training_pack = _retriever(
        tmp_path / "training-pack",
        name="training",
        record_ids=("fact",),
        keys=[[1, 0, 0]],
        values=[[1, 2, 3, 4, 5]],
        key_encoder_name="shared-key",
        value_encoder_name="shared-value",
    )
    unseen_pack = _retriever(
        tmp_path / "unseen-pack",
        name="unseen",
        record_ids=("fact",),
        keys=[[1, 0, 0]],
        values=[[9, 8, 7, 6, 5]],
        key_encoder_name="shared-key",
        value_encoder_name="shared-value",
    )
    model = DenseLM(
        ModelConfig(
            vocab_size=260,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=8,
            ffn="moe",
            num_experts=3,
            experts_per_token=2,
            shared_expert=True,
            router_aux_loss_coefficient=0.01,
        ),
        AttentionConfig(),
    )
    model.requires_grad_(False)
    adapter = model.add_semantic_memory(
        "unseen-transfer",
        training_pack,
        site="after_block",
        block_index=0,
        min_score=0.9,
    )
    adapter.freeze()
    frozen_state = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    model.replace_semantic_memory("unseen-transfer", unseen_pack)
    assert all(
        torch.equal(frozen_state[name], parameter)
        for name, parameter in model.named_parameters()
    )
    router_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if ".ffn.router." in name
    ]
    assert router_parameters
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    model.eval()
    query = SemanticQueryBatch(
        unseen_pack.key_encoder,
        torch.tensor([[1.0, 0, 0]]),
    )
    with torch.inference_mode():
        logits = model(
            torch.tensor([[3, 7]]),
            semantic_queries={unseen_pack.key_encoder.sha256: query},
        )
    assert logits.shape == (1, 2, model.config.vocab_size)
    assert adapter.last_traces[0].pack_id == unseen_pack.pack_id
    assert adapter.last_traces[0].status == "hit"
    assert all(parameter.grad is None for parameter in model.parameters())


def test_semantic_query_generation_matches_cached_and_full_prefix(
    tmp_path,
) -> None:
    retriever = _retriever(
        tmp_path / "generation-pack",
        name="generation",
        record_ids=("answer",),
        keys=[[1, 0, 0]],
        values=[[1, 2, 3, 4, 5]],
    )
    vocab = {"<unk>": 0, "hello": 1, "answer": 2}
    vocab.update({f"unused-{index}": index + 3 for index in range(257)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    model = DenseLM(
        ModelConfig(
            vocab_size=260,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            ffn_dim=16,
            max_seq_len=8,
        ),
        AttentionConfig(),
    )
    model.add_semantic_memory("allocation", retriever, site="final")
    query = SemanticQueryBatch(
        retriever.key_encoder, torch.tensor([[1.0, 0, 0]], dtype=torch.float32)
    )
    cached = generate(
        model,
        tokenizer,
        "hello",
        8,
        3,
        torch.device("cpu"),
        semantic_queries=query,
        use_cache=True,
    )
    full_prefix = generate(
        model,
        tokenizer,
        "hello",
        8,
        3,
        torch.device("cpu"),
        semantic_queries=query,
        use_cache=False,
    )
    assert cached == full_prefix
    assert model.semantic_memories["allocation"].last_traces[0].status == "hit"

    long_prompt = " ".join(["hello", *("answer" for _ in range(8))])
    sequence_vectors = torch.zeros((1, 9, 3), dtype=torch.float32)
    sequence_vectors[0, -1] = torch.tensor([1.0, 0, 0])
    sequence_mask = torch.zeros((1, 9), dtype=torch.bool)
    sequence_mask[0, -1] = True
    sequence_query = SemanticQueryBatch(
        retriever.key_encoder,
        sequence_vectors,
        mask=sequence_mask,
        as_of=(_CREATED_AT,) * 9,
    )
    cached_sequence = generate(
        model,
        tokenizer,
        long_prompt,
        8,
        3,
        torch.device("cpu"),
        semantic_queries=sequence_query,
        use_cache=True,
    )
    full_prefix_sequence = generate(
        model,
        tokenizer,
        long_prompt,
        8,
        3,
        torch.device("cpu"),
        semantic_queries=sequence_query,
        use_cache=False,
    )
    assert cached_sequence == full_prefix_sequence
    assert model.semantic_memories["allocation"].last_traces[0].status == "hit"
