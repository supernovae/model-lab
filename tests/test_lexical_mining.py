from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from sparselab.canonical import canonical_digest
from sparselab.data import lexical_mining
from sparselab.data.lexical_mining import analyze_training_corpus
from sparselab.model.memory import TokenNgramMemory


def _write_conversations(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ]
                }
            )
            + "\n"
            for user, assistant in pairs
        ),
        encoding="utf-8",
    )


def _tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<unk>": 0,
                "User:": 1,
                "Assistant:": 2,
                "a": 3,
                "b": 4,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.save(str(path))


def _ngram_keys(ids: list[int], order: int) -> set[tuple[int, ...]]:
    return {
        tuple(
            ids[position - offset] if position >= offset else 0
            for offset in range(order)
        )
        for position in range(len(ids))
    }


def _address_for_key(key: tuple[int, ...], table_size: int, head: int) -> int:
    address = head + 1
    for token_id in key:
        address = (address * (257 + head * 2) + token_id) % table_size
    return address


def test_analyze_training_corpus_reports_train_only_counts_entropy_and_addresses(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    tokenizer = tmp_path / "tokenizer.json"
    _write_conversations(train, [("a", "b"), ("b", "a")])
    _tokenizer(tokenizer)

    analysis = analyze_training_corpus(
        train,
        tokenizer,
        table_size=17,
        memory_dim=3,
        ngram_orders=(1, 2),
        hash_heads=2,
    )

    first, second = [1, 3, 2, 4], [1, 4, 2, 3]
    assert analysis["documents"] == 2
    assert analysis["tokens"] == 8
    assert analysis["token_frequency"] == [
        {"token_id": token, "count": 2} for token in range(1, 5)
    ]
    assert analysis["document_frequency"] == [
        {"token_id": token, "count": 2} for token in range(1, 5)
    ]
    assert analysis["empirical_unigram_entropy_bits_per_token"] == pytest.approx(2.0)
    assert analysis["training_corpus"] == {
        "sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "bytes": train.stat().st_size,
    }
    assert analysis["tokenizer"] == {
        "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    }
    assert analysis["addressing"] == {
        "algorithm": "token-ngram-recurrence-v1",
        "recurrence": "address=(address*(257+hash_head*2)+shifted_token_id)%table_size",
        "initial_address": "hash_head+1",
        "document_boundary": "zero_padded",
        "collision_key": "distinct zero-padded token n-gram keys across corpus",
        "collision_definition": (
            "distinct keys minus occupied buckets; repeated lookups are separate"
        ),
        "distinct_key_limits": {
            "max_keys": lexical_mining.MAX_UNIQUE_NGRAMS,
            "max_key_components": lexical_mining.MAX_UNIQUE_NGRAM_COMPONENTS,
        },
        "table_size": 17,
        "ngram_orders": [1, 2],
        "hash_heads": 2,
    }
    for stream in analysis["streams"]:
        assert isinstance(stream, dict)
        order, head = stream["order"], stream["hash_head"]
        ngram_keys = _ngram_keys(first, order) | _ngram_keys(second, order)
        addresses = {_address_for_key(key, 17, head) for key in ngram_keys}
        collisions = len(ngram_keys) - len(addresses)
        repeated_accesses = 8 - len(ngram_keys)
        assert stream == {
            "order": order,
            "hash_head": head,
            "lookup_count": 8,
            "distinct_ngram_count": len(ngram_keys),
            "repeated_accesses": repeated_accesses,
            "unique_occupied_buckets": len(addresses),
            "collisions": collisions,
            "collision_rate": collisions / len(ngram_keys),
            "repeated_access_rate": repeated_accesses / 8,
            "utilization": len(addresses) / 17,
        }
    assert analysis["storage"] == {
        "table_fp32_bytes": 17 * 3 * 4 * 4,
        "table_count": 4,
        "memory_dim": 3,
        "excludes": ["output_projection_weights", "gate_weights"],
    }


def test_analyze_training_corpus_has_deterministic_content_identity_and_reads_train_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    held_out = tmp_path / "validation.jsonl"
    tokenizer = tmp_path / "tokenizer.json"
    _write_conversations(train, [("a", "b")])
    held_out.write_text("not valid JSON\n", encoding="utf-8")
    _tokenizer(tokenizer)
    original = lexical_mining.iter_conversations
    opened: list[Path] = []

    def train_only(path: Path):
        opened.append(path)
        assert path == train
        yield from original(path)

    monkeypatch.setattr(lexical_mining, "iter_conversations", train_only)
    first = analyze_training_corpus(
        train, tokenizer, table_size=19, memory_dim=2, ngram_orders=[2]
    )
    second = analyze_training_corpus(
        train, tokenizer, table_size=19, memory_dim=2, ngram_orders=[2]
    )

    assert opened == [train, train]
    assert held_out.name not in json.dumps(first)
    assert first == second
    body = dict(first)
    identity = body.pop("analysis_sha256")
    assert identity == canonical_digest(body)


@pytest.mark.parametrize(
    ("records", "kwargs", "message"),
    [
        ([], {}, "is empty"),
        ([("a", "b")], {"table_size": 0}, "table_size"),
        ([("a", "b")], {"ngram_orders": (2, 2)}, "duplicates"),
        ([("a", "b")], {"ngram_orders": ()}, "nonempty"),
    ],
)
def test_analyze_training_corpus_rejects_empty_or_invalid_configuration(
    tmp_path: Path,
    records: list[tuple[str, str]],
    kwargs: dict[str, object],
    message: str,
) -> None:
    train = tmp_path / "train.jsonl"
    tokenizer = tmp_path / "tokenizer.json"
    _write_conversations(train, records)
    _tokenizer(tokenizer)
    arguments: dict[str, object] = {
        "table_size": 17,
        "memory_dim": 2,
        "ngram_orders": (2,),
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        analyze_training_corpus(train, tokenizer, **arguments)  # type: ignore[arg-type]


def test_analyze_training_corpus_entropy_is_empirical_unigram_entropy(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    tokenizer = tmp_path / "tokenizer.json"
    _write_conversations(train, [("a", "a"), ("a", "b")])
    _tokenizer(tokenizer)

    analysis = analyze_training_corpus(
        train, tokenizer, table_size=31, memory_dim=1, ngram_orders=(2,)
    )

    counts = [2, 2, 3, 1]
    expected = -sum((count / 8) * math.log2(count / 8) for count in counts)
    assert analysis["empirical_unigram_entropy_bits_per_token"] == pytest.approx(
        expected
    )


def test_address_streams_match_token_ngram_memory_exactly(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_conversations(train, [("a b", "a"), ("b", "a b")])
    _tokenizer(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    memory = TokenNgramMemory(
        hidden_dim=4,
        table_size=17,
        ngram_size=2,
        value_dim=3,
        ngram_orders=(1, 2),
        hash_heads=2,
    )

    expected: dict[tuple[int, int], list[int]] = {
        (order, head): [] for order in (1, 2) for head in range(2)
    }
    distinct_keys = {order: set() for order in (1, 2)}
    for document in lexical_mining.iter_conversations(train):
        tokens = tokenizer.encode(document, add_special_tokens=False).ids
        input_ids = torch.tensor([tokens], dtype=torch.long)
        for order, keys in distinct_keys.items():
            keys.update(_ngram_keys(tokens, order))
        for (order, head), addresses in expected.items():
            addresses.extend(
                memory.addresses(input_ids, order=order, seed=head).flatten().tolist()
            )

    analysis = analyze_training_corpus(
        train,
        tokenizer_path,
        table_size=17,
        memory_dim=3,
        ngram_orders=(1, 2),
        hash_heads=2,
    )
    for stream in analysis["streams"]:
        addresses = expected[stream["order"], stream["hash_head"]]
        unique = len(set(addresses))
        distinct_count = len(distinct_keys[stream["order"]])
        collisions = distinct_count - unique
        repeated_accesses = len(addresses) - distinct_count
        assert stream["lookup_count"] == len(addresses)
        assert stream["distinct_ngram_count"] == distinct_count
        assert stream["repeated_accesses"] == repeated_accesses
        assert stream["unique_occupied_buckets"] == unique
        assert stream["collisions"] == collisions
        assert stream["collision_rate"] == collisions / distinct_count
        assert stream["repeated_access_rate"] == repeated_accesses / len(addresses)


def test_address_bitmap_limit_rejects_unbounded_table_state(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="collision-bitmap memory limit"):
        analyze_training_corpus(
            tmp_path / "unused-train.jsonl",
            tmp_path / "unused-tokenizer.json",
            table_size=lexical_mining.MAX_ADDRESS_BITMAP_BITS + 1,
            memory_dim=1,
            ngram_orders=(1,),
        )


def test_exact_unique_ngram_tracking_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    tokenizer = tmp_path / "tokenizer.json"
    _write_conversations(train, [("a", "b")])
    _tokenizer(tokenizer)
    monkeypatch.setattr(lexical_mining, "MAX_UNIQUE_NGRAMS", 1)

    with pytest.raises(ValueError, match="unique n-gram limit"):
        analyze_training_corpus(
            train, tokenizer, table_size=17, memory_dim=1, ngram_orders=(1,)
        )


@pytest.fixture
def packed_training(tmp_path):
    from test_training import config

    from sparselab.data.packing import prepare_data
    from sparselab.data.tokenizer import load_tokenizer

    configured = config(tmp_path)
    paths = [tmp_path / "packed-train.jsonl", tmp_path / "packed-validation.jsonl"]
    for split, path in enumerate(paths):
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "format_version": 2,
                        "loss_mode": "assistant_only",
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Question {split} {index} about the fox",
                            },
                            {
                                "role": "assistant",
                                "content": f"The fox answered {index} with a book",
                            },
                        ],
                    }
                )
                + "\n"
                for index in range(8)
            ),
            encoding="utf-8",
        )
    configured = configured.model_copy(
        update={
            "dataset": configured.dataset.model_copy(
                update={
                    "source": "local_chat",
                    "license": "MIT synthetic fixture",
                    "train_path": paths[0],
                    "validation_path": paths[1],
                }
            ),
        }
    )
    return prepare_data(configured, load_tokenizer(configured.tokenizer.path))


def test_prepared_addresses_match_native_blocks_and_masked_context(packed_training):
    from sparselab.data.packing import TokenBlockDataset

    prepared = packed_training
    dataset = TokenBlockDataset(
        prepared.train, 16, supervision=prepared.train_supervision
    )
    sampled = min(3, len(dataset))
    result = lexical_mining.analyze_prepared_training(
        prepared.root,
        sequence_length=16,
        max_blocks=3,
        table_size=17,
        memory_dim=4,
        ngram_orders=(2, 3),
        hash_heads=2,
    )
    assert result["tokens"] == sampled * 16
    assert (
        result["sampling"]["selected_block_indices"]
        == dataset.block_indices[:sampled].tolist()
    )
    assert result["sampling"]["schedule_weighted"] is False
    assert (
        result["tokenizer"]["canonical_sha256"] == prepared.manifest["tokenizer_sha256"]
    )
    blocks = [dataset.numpy_block(i)[0].tolist() for i in range(sampled)]
    assert any((dataset.numpy_block(i)[1] == -100).any() for i in range(sampled))
    assert any(3 in block for block in blocks)  # Native EOS retained within blocks.
    memory = TokenNgramMemory(8, 17, 3, 4, (2, 3), 2)
    for stream in result["streams"]:
        keys = set().union(*(_ngram_keys(block, stream["order"]) for block in blocks))
        addresses = memory.addresses(
            torch.tensor(blocks), stream["order"], stream["hash_head"]
        )
        occupied = len(set(addresses.flatten().tolist()))
        assert stream["lookup_count"] == sampled * 16
        assert stream["distinct_ngram_count"] == len(keys)
        assert stream["collisions"] == len(keys) - occupied
        assert stream["repeated_accesses"] == sampled * 16 - len(keys)
    body = dict(result)
    assert body.pop("analysis_sha256") == canonical_digest(body)


def test_prepared_analysis_uses_no_validation_tokens(packed_training, monkeypatch):
    from sparselab.data.packing import TokenBlockDataset

    prepared = packed_training
    original = TokenBlockDataset.numpy_block
    read = []

    def train_block(self, index):
        # Native verification opens validation for integrity, but mining never
        # constructs a dataset on its array or consumes its tokens.
        assert Path(self.ids.filename).name == "train.npy"
        read.append(index)
        return original(self, index)

    monkeypatch.setattr(TokenBlockDataset, "numpy_block", train_block)
    lexical_mining.analyze_prepared_training(
        prepared.root,
        sequence_length=16,
        max_blocks=1,
        table_size=17,
        memory_dim=4,
        ngram_orders=(2,),
    )
    assert read == [0]


def test_prepared_analysis_rejects_tampered_train(packed_training):
    path = packed_training.root / "train.npy"
    with path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError):
        lexical_mining.analyze_prepared_training(
            packed_training.root,
            sequence_length=16,
            max_blocks=1,
            table_size=17,
            memory_dim=4,
            ngram_orders=(2,),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sequence_length": 0},
        {"max_blocks": 0},
        {"sequence_length": 1000000, "max_blocks": 1000000},
    ],
)
def test_prepared_analysis_refuses_unbounded_sample_before_read(tmp_path, kwargs):
    arguments = {"sequence_length": 16, "max_blocks": 1, **kwargs}
    with pytest.raises(ValueError):
        lexical_mining.analyze_prepared_training(
            tmp_path / "missing",
            **arguments,
            table_size=17,
            memory_dim=4,
            ngram_orders=(2,),
        )


def test_packed_candidate_limit_precedes_dataset_allocation(
    packed_training, monkeypatch
):
    from sparselab.data import packing

    def forbidden(*args, **kwargs):
        pytest.fail("candidate enumeration must not start")

    monkeypatch.setattr(lexical_mining, "MAX_DOCUMENTS", 2)
    monkeypatch.setattr(packing, "TokenBlockDataset", forbidden)
    with pytest.raises(ValueError, match="candidate-block limit"):
        lexical_mining.analyze_prepared_training(
            packed_training.root,
            sequence_length=16,
            max_blocks=1,
            table_size=17,
            memory_dim=4,
            ngram_orders=(2,),
        )


def _configured_tokenizer(path: Path) -> None:
    vocabulary = {"<unk>": 0, "User:": 1, "Assistant:": 2, "a": 3, "b": 4, "A": 5}
    vocabulary.update({f"word{index}": index for index in range(6, 260)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.save(str(path))


def _prime_model(
    tokenizer_path: Path, *, mapped: bool = True, addressing: str = "prime_xor_v1"
):
    from sparselab.config.models import ModelConfig
    from sparselab.memory_tokenizer import build_memory_token_map

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    mapping = (
        build_memory_token_map(tokenizer_path, "deepseek_safe_v1") if mapped else None
    )
    return ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        ffn_dim=16,
        max_seq_len=16,
        memory="ngram",
        memory_ngram_size=3,
        memory_ngram_orders=(2, 3),
        memory_hash_heads=2,
        memory_table_size=17,
        memory_table_sizes=(17, 19, 23, 29) if addressing == "prime_xor_v1" else (),
        memory_dim=4,
        memory_addressing=addressing,
        memory_hash_seed=17 if addressing == "prime_xor_v1" else 0,
        memory_injection="after_block",
        memory_block_index=0,
        memory_token_mapping=mapping.to_dict() if mapping is not None else None,
        memory_padding_token_id=5,
    )


def _assert_model_analysis(analysis, sequences, model):
    from sparselab.model.memory import build_token_memory

    memory = build_token_memory(model)
    mapping = model.memory_token_mapping
    pad = model.memory_padding_token_id
    sizes = model.memory_table_sizes or (model.memory_table_size,) * (
        len(model.memory_ngram_orders) * model.memory_hash_heads
    )
    for stream in analysis["streams"]:
        order, head = stream["order"], stream["hash_head"]
        raw_keys = set()
        addresses = []
        for sequence in sequences:
            raw_keys.update(
                tuple(
                    sequence[position - offset] if position >= offset else pad
                    for offset in range(order)
                )
                for position in range(len(sequence))
            )
            addresses.extend(
                memory.addresses(torch.tensor([sequence]), order, head)
                .flatten()
                .tolist()
            )
        canonical_keys = {
            tuple(mapping.token_ids[value] for value in key)
            if mapping is not None
            else key
            for key in raw_keys
        }
        size = sizes[
            model.memory_ngram_orders.index(order) * model.memory_hash_heads + head
        ]
        occupied = len(set(addresses))
        assert stream["table_size"] == size
        assert stream["distinct_raw_ngram_count"] == len(raw_keys)
        assert stream["distinct_ngram_count"] == len(canonical_keys)
        assert stream["mapping_equivalent_key_count"] == len(raw_keys) - len(
            canonical_keys
        )
        assert stream["collisions"] == len(canonical_keys) - occupied
        assert stream["unique_occupied_buckets"] == occupied
        assert stream["utilization"] == occupied / size
        assert stream["repeated_accesses"] == len(addresses) - len(canonical_keys)
        assert stream["raw_repeated_accesses"] == len(addresses) - len(raw_keys)
    assert analysis["storage"]["table_fp32_bytes"] == sum(sizes) * model.memory_dim * 4
    assert analysis["addressing"].get("multipliers", []) == list(memory.multipliers)
    assert analysis["addressing"]["memory_padding_token_id"] == (
        mapping.token_ids[pad] if mapping is not None else pad
    )
    body = dict(analysis)
    assert body.pop("analysis_sha256") == canonical_digest(body)


@pytest.mark.parametrize("addressing", ["legacy", "prime_xor_v1"])
def test_model_configured_document_mining_separates_mapping_equivalence_from_hash_collision(
    tmp_path,
    addressing,
):
    tokenizer_path, train = tmp_path / "tokenizer.json", tmp_path / "train.jsonl"
    _configured_tokenizer(tokenizer_path)
    _write_conversations(train, [("a", "b"), ("A", "b"), ("A b", "a")])
    model = _prime_model(tokenizer_path, addressing=addressing)
    result = analyze_training_corpus(train, tokenizer_path, model_config=model)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sequences = [
        tokenizer.encode(document, add_special_tokens=False).ids
        for document in lexical_mining.iter_conversations(train)
    ]
    _assert_model_analysis(result, sequences, model)
    assert result["version"] == 3
    assert result["tokenizer"]["vocabulary_bound_checked"] is True
    assert any(
        stream["mapping_equivalent_key_count"] > 0 for stream in result["streams"]
    )
    assert (
        result["addressing"]["token_mapping"]["mapping_sha256"]
        == model.memory_token_mapping.mapping_sha256
    )
    # Raw frequencies still distinguish a/A, irrespective of their memory alias.
    assert {3, 5} <= {entry["token_id"] for entry in result["token_frequency"]}


@pytest.mark.parametrize("mapped", [False, True])
def test_model_configured_packed_mining_matches_prime_streams_and_mapped_padding(
    packed_training, tmp_path, mapped
):
    from sparselab.data.packing import TokenBlockDataset

    tokenizer_path = tmp_path / "tokenizer" / "tokenizer.json"
    model = _prime_model(tokenizer_path, mapped=mapped)
    prepared = packed_training
    result = lexical_mining.analyze_prepared_training(
        prepared.root,
        sequence_length=16,
        max_blocks=3,
        model_config=model,
        tokenizer_path=tokenizer_path,
    )
    dataset = TokenBlockDataset(
        prepared.train, 16, supervision=prepared.train_supervision
    )
    sequences = [
        dataset.numpy_block(index)[0].tolist() for index in range(min(3, len(dataset)))
    ]
    _assert_model_analysis(result, sequences, model)
    assert result["version"] == 3
    assert result["tokenizer"]["vocabulary_bound_checked"] is True
    assert (
        result["tokenizer"]["canonical_sha256"] == prepared.manifest["tokenizer_sha256"]
    )
    assert result["blocks"] == len(sequences)


def test_model_configured_mining_refuses_map_bound_to_different_tokenizer_bytes(
    tmp_path,
):
    tokenizer_path, train = tmp_path / "tokenizer.json", tmp_path / "train.jsonl"
    _configured_tokenizer(tokenizer_path)
    _write_conversations(train, [("a", "b")])
    model = _prime_model(tokenizer_path)
    tokenizer_path.write_bytes(tokenizer_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="tokenizer digest mismatch"):
        analyze_training_corpus(train, tokenizer_path, model_config=model)


def test_model_configured_packed_mining_refuses_different_prepared_tokenizer(
    packed_training, tmp_path
):
    from tokenizers.normalizers import Lowercase

    tokenizer_path = tmp_path / "tokenizer" / "tokenizer.json"
    model = _prime_model(tokenizer_path, mapped=False)
    changed = tmp_path / "changed-tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.normalizer = Lowercase()
    tokenizer.save(str(changed))
    with pytest.raises(ValueError, match="differs from the prepared tokenizer"):
        lexical_mining.analyze_prepared_training(
            packed_training.root,
            sequence_length=16,
            max_blocks=1,
            model_config=model,
            tokenizer_path=changed,
        )


def test_model_configured_mining_bounds_actual_prime_table_sum_before_inputs(
    tmp_path, monkeypatch
):
    tokenizer_path = tmp_path / "tokenizer.json"
    _configured_tokenizer(tokenizer_path)
    model = _prime_model(tokenizer_path)
    # First-table * streams is 68, actual declared prime sum is 88.
    monkeypatch.setattr(lexical_mining, "MAX_ADDRESS_BITMAP_BITS", 80)
    with pytest.raises(ValueError, match="collision-bitmap memory limit"):
        analyze_training_corpus(
            tmp_path / "missing", tokenizer_path, model_config=model
        )


def test_model_configured_mining_bounds_raw_and_canonical_key_storage(
    tmp_path, monkeypatch
):
    tokenizer_path, train = tmp_path / "tokenizer.json", tmp_path / "train.jsonl"
    _configured_tokenizer(tokenizer_path)
    _write_conversations(train, [("a A a A a A", "b")])
    model = _prime_model(tokenizer_path)
    monkeypatch.setattr(lexical_mining, "MAX_UNIQUE_NGRAMS", 4)
    with pytest.raises(ValueError, match="unique n-gram limit"):
        analyze_training_corpus(train, tokenizer_path, model_config=model)


def _multiport_model(tokenizer_path):
    model = _prime_model(tokenizer_path)
    payload = model.model_dump(mode="python")
    payload.update(
        num_layers=3,
        memory_block_index=None,
        memory_block_indices=(0, 2),
        memory_table_sizes=(17, 19, 23, 29, 31, 37, 41, 43),
    )
    return type(model).model_validate(payload)


@pytest.mark.parametrize("selected", [0, 2])
def test_multiport_mining_selects_declared_prime_slice_and_binds_full_model(
    tmp_path, selected
):
    tokenizer_path, train = tmp_path / "tokenizer.json", tmp_path / "train.jsonl"
    _configured_tokenizer(tokenizer_path)
    _write_conversations(train, [("a", "A b")])
    model = _multiport_model(tokenizer_path)
    result = analyze_training_corpus(
        train, tokenizer_path, model_config=model, memory_block_index=selected
    )
    offset = model.memory_block_indices.index(selected) * 4
    sizes = model.memory_table_sizes[offset : offset + 4]
    assert result["addressing"]["table_sizes"] == list(sizes)
    assert result["addressing"]["layer_id"] == selected
    assert result["model_configuration"] == {
        "source_model_config_sha256": canonical_digest(model.model_dump(mode="json")),
        "selected_memory_block_index": selected,
    }
    selected_payload = model.model_dump(mode="python")
    selected_payload.update(
        memory_block_indices=(),
        memory_block_index=selected,
        memory_table_size=sizes[0],
        memory_table_sizes=sizes,
    )
    selected_model = type(model).model_validate(selected_payload)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sequences = [
        tokenizer.encode(document, add_special_tokens=False).ids
        for document in lexical_mining.iter_conversations(train)
    ]
    _assert_model_analysis(result, sequences, selected_model)


def test_multiport_mining_requires_valid_explicit_selection_before_reading(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    _configured_tokenizer(tokenizer_path)
    model = _multiport_model(tokenizer_path)
    with pytest.raises(ValueError, match="multiport lexical analysis requires"):
        analyze_training_corpus(
            tmp_path / "missing", tokenizer_path, model_config=model
        )
    with pytest.raises(ValueError, match="not a configured memory port"):
        analyze_training_corpus(
            tmp_path / "missing",
            tokenizer_path,
            model_config=model,
            memory_block_index=1,
        )
    with pytest.raises(ValueError, match="requires model_config"):
        analyze_training_corpus(
            tmp_path / "missing", tokenizer_path, memory_block_index=0
        )


def test_multiport_packed_mining_keeps_selected_port_identity(
    packed_training, tmp_path
):
    tokenizer_path = tmp_path / "tokenizer" / "tokenizer.json"
    model = _multiport_model(tokenizer_path)
    result = lexical_mining.analyze_prepared_training(
        packed_training.root,
        sequence_length=16,
        max_blocks=1,
        model_config=model,
        tokenizer_path=tokenizer_path,
        memory_block_index=2,
    )
    assert result["model_configuration"]["selected_memory_block_index"] == 2
    assert result["model_configuration"][
        "source_model_config_sha256"
    ] == canonical_digest(model.model_dump(mode="json"))
    assert result["addressing"]["table_sizes"] == [31, 37, 41, 43]
