"""Corpus-only lexical statistics for token n-gram memory planning.

Consumes an explicit training JSONL or the TRAIN blocks of verified prepared data.
No model is constructed and no held-out tokens enter statistics. Native prepared
verification authenticates the entire closure before TRAIN-only analysis.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Final

from sparselab.address_hash import (
    ENGRAM_SCHEME_V1,
    TOKEN_SCHEME_V1,
    TOKEN_SCHEME_V2,
    engram_multipliers,
    engram_ngram_address,
    token_ngram_address,
    token_uses_legacy,
    validate_engram_table_sizes,
)
from sparselab.canonical import canonical_digest
from sparselab.data.conversations import iter_conversations
from sparselab.data.tokenizer import load_tokenizer
from sparselab.training.manifest import sha256_file

if TYPE_CHECKING:
    from tokenizers import Tokenizer

    from sparselab.config.models import ModelConfig


MAX_INPUT_BYTES: Final = 256 * 1024 * 1024
MAX_DOCUMENTS: Final = 1_000_000
MAX_TOKENS: Final = 100_000_000
MAX_TOKENS_PER_DOCUMENT: Final = 1_000_000
MAX_TABLE_SIZE: Final = 1_000_000_000
MAX_MEMORY_DIM: Final = 1_000_000
MAX_NGRAM_ORDER: Final = 64
MAX_HASH_HEADS: Final = 64
MAX_ADDRESS_STREAMS: Final = 64
MAX_ADDRESS_BITMAP_BITS: Final = 512 * 1024 * 1024
MAX_HASH_OPERATIONS: Final = 50_000_000
MAX_UNIQUE_NGRAMS: Final = 1_000_000
MAX_UNIQUE_NGRAM_COMPONENTS: Final = 8_000_000


_V1_RECURRENCE: Final = (
    "address=(address*(257+hash_head*2)+shifted_token_id)%table_size"
)
_V2_RECURRENCE: Final = (
    "per hash head: v1 recurrence when gcd(257+hash_head*2,table_size)==1, else "
    "address=(address*(1103515245+hash_head*2)+shifted_token_id+1)%(2**31-1) "
    "and a final %table_size"
)


def _positive_int(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or value < 1 or value > maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return value


def _validated_orders(ngram_orders: object) -> tuple[int, ...]:
    if not isinstance(ngram_orders, tuple | list) or not ngram_orders:
        raise ValueError("ngram_orders must be a nonempty tuple or list of integers")
    orders = tuple(
        _positive_int(order, "each ngram order", MAX_NGRAM_ORDER)
        for order in ngram_orders
    )
    if len(set(orders)) != len(orders):
        raise ValueError("ngram_orders must not contain duplicates")
    return orders


def _ngram_key(
    token_ids: list[int], position: int, order: int, padding: int = 0
) -> tuple[int, ...]:
    return tuple(
        token_ids[position - offset] if position >= offset else padding
        for offset in range(order)
    )


def _address_for_key(ngram_key: tuple[int, ...], *, table_size: int, head: int) -> int:
    return token_ngram_address(ngram_key, table_size, head)


def _file(path: Path, name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{name} must be an existing regular file: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"{name} is empty: {path}")
    if size > MAX_INPUT_BYTES:
        raise ValueError(f"{name} exceeds {MAX_INPUT_BYTES} byte limit: {path}")
    return path


def _settings(
    table_size: int,
    memory_dim: int,
    ngram_orders: tuple[int, ...] | list[int],
    hash_heads: int,
    table_sizes: tuple[int, ...] = (),
) -> tuple[int, int, int, tuple[int, ...]]:
    table_size = _positive_int(table_size, "table_size", MAX_TABLE_SIZE)
    memory_dim = _positive_int(memory_dim, "memory_dim", MAX_MEMORY_DIM)
    hash_heads = _positive_int(hash_heads, "hash_heads", MAX_HASH_HEADS)
    orders = _validated_orders(ngram_orders)
    stream_count = len(orders) * hash_heads
    if stream_count > MAX_ADDRESS_STREAMS:
        raise ValueError(f"order/head combinations exceed {MAX_ADDRESS_STREAMS}")
    if (
        sum(table_sizes) if table_sizes else table_size * stream_count
    ) > MAX_ADDRESS_BITMAP_BITS:
        raise ValueError(
            "configured tables exceed the exact collision-bitmap memory limit"
        )

    return table_size, memory_dim, hash_heads, orders


def _select_model_port(
    model: ModelConfig | None, memory_block_index: int | None
) -> tuple[ModelConfig | None, dict[str, object]]:
    if memory_block_index is not None and (
        type(memory_block_index) is not int or memory_block_index < 0
    ):
        raise ValueError("memory_block_index must be a nonnegative integer")
    if model is None:
        if memory_block_index is not None:
            raise ValueError("memory_block_index requires model_config")
        return None, {}
    source_identity = canonical_digest(model.model_dump(mode="json"))
    ports = model.memory_block_indices
    selected = memory_block_index
    if ports:
        if selected is None:
            if len(ports) > 1:
                raise ValueError(
                    "multiport lexical analysis requires memory_block_index"
                )
            selected = ports[0]
        if selected not in ports:
            raise ValueError("memory_block_index is not a configured memory port")
        stream_count = (len(model.memory_ngram_orders) or 1) * model.memory_hash_heads
        slot = ports.index(selected)
        sizes = model.memory_table_sizes[
            slot * stream_count : (slot + 1) * stream_count
        ]
        payload = model.model_dump(mode="python")
        payload.update(memory_block_indices=(), memory_block_index=selected)
        if model.memory_addressing == "prime_xor_v1":
            payload.update(memory_table_sizes=sizes, memory_table_size=sizes[0])
        model = type(model).model_validate(payload)
    elif selected is not None and selected != model.memory_block_index:
        raise ValueError("memory_block_index is not a configured memory port")
    else:
        selected = model.memory_block_index
    return model, {
        "model_configuration": {
            "source_model_config_sha256": source_identity,
            "selected_memory_block_index": selected,
        }
    }


def _model_settings(model: ModelConfig) -> tuple[int, int, int, tuple[int, ...]]:
    if model.memory != "ngram":
        raise ValueError("lexical analysis model_config requires token ngram memory")
    orders = model.memory_ngram_orders or (model.memory_ngram_size,)
    sizes = model.memory_table_sizes
    if model.memory_addressing == "prime_xor_v1":
        validate_engram_table_sizes(sizes, len(orders) * model.memory_hash_heads)
        for size in sizes:
            _positive_int(size, "each table_size", MAX_TABLE_SIZE)
    return _settings(
        model.memory_table_size,
        model.memory_dim,
        orders,
        model.memory_hash_heads,
        sizes,
    )


def _bind_model_tokenizer(
    model: ModelConfig, tokenizer_file: Path
) -> tuple[Tokenizer, dict[str, object]]:
    from sparselab.memory_tokenizer import validate_memory_token_map

    digest = sha256_file(tokenizer_file)
    tokenizer = load_tokenizer(tokenizer_file)
    if tokenizer.get_vocab_size(with_added_tokens=True) != model.vocab_size:
        raise ValueError("lexical model and tokenizer vocabulary sizes differ")
    if model.memory_token_mapping is not None:
        validate_memory_token_map(
            model.memory_token_mapping.model_dump(mode="json"),
            tokenizer_path=tokenizer_file,
        )
    if sha256_file(tokenizer_file) != digest:
        raise ValueError("tokenizer changed while binding lexical analysis")
    return tokenizer, {"sha256": digest, "vocabulary_bound_checked": True}


def analyze_training_corpus(
    train_jsonl: Path | str,
    tokenizer_path: Path | str,
    *,
    table_size: int = 0,
    memory_dim: int = 0,
    ngram_orders: tuple[int, ...] | list[int] = (),
    hash_heads: int = 1,
    model_config: ModelConfig | None = None,
    memory_block_index: int | None = None,
) -> dict[str, object]:
    """Analyze one caller-designated training JSONL; never open held-out files."""
    model_config, model_identity = _select_model_port(model_config, memory_block_index)
    settings = (
        _model_settings(model_config)
        if model_config is not None
        else _settings(table_size, memory_dim, ngram_orders, hash_heads)
    )
    train_path = _file(Path(train_jsonl), "train_jsonl")
    tokenizer_file = _file(Path(tokenizer_path), "tokenizer_path")
    if model_config is not None:
        tokenizer, tokenizer_identity = _bind_model_tokenizer(
            model_config, tokenizer_file
        )
    else:
        tokenizer_identity = {"sha256": sha256_file(tokenizer_file)}
        tokenizer = load_tokenizer(tokenizer_file)
    return _analyze_sequences(
        (
            tokenizer.encode(document, add_special_tokens=False).ids
            for document in iter_conversations(train_path)
        ),
        settings,
        {
            **model_identity,
            "input_scope": "one_explicit_training_jsonl",
            "training_corpus": {
                "sha256": sha256_file(train_path),
                "bytes": train_path.stat().st_size,
            },
            "tokenizer": tokenizer_identity,
        },
        model_config=model_config,
    )


def analyze_prepared_training(
    prepared_path: Path | str,
    *,
    sequence_length: int,
    max_blocks: int,
    table_size: int = 0,
    memory_dim: int = 0,
    ngram_orders: tuple[int, ...] | list[int] = (),
    hash_heads: int = 1,
    model_config: ModelConfig | None = None,
    tokenizer_path: Path | str | None = None,
    memory_block_index: int | None = None,
) -> dict[str, object]:
    """Measure a bounded prefix of eligible TRAIN x blocks, once per block.

    The native reader verifies all prepared files, including validation, but only
    TRAIN input tokens are mined. This describes packed addresses, not the
    shuffled/repeated training schedule or learned gate behavior.
    """
    from sparselab.data.packing import (
        TokenBlockDataset,
        canonical_tokenizer_sha256,
        load_prepared_data,
    )

    model_config, model_identity = _select_model_port(model_config, memory_block_index)
    settings = (
        _model_settings(model_config)
        if model_config is not None
        else _settings(table_size, memory_dim, ngram_orders, hash_heads)
    )
    sequence_length = _positive_int(
        sequence_length, "sequence_length", MAX_TOKENS_PER_DOCUMENT
    )
    max_blocks = _positive_int(max_blocks, "max_blocks", MAX_DOCUMENTS)
    if max_blocks * sequence_length > MAX_TOKENS:
        raise ValueError("packed sample exceeds token limit")
    if (
        max_blocks * sequence_length * sum(settings[3]) * settings[2]
        > MAX_HASH_OPERATIONS
    ):
        raise ValueError("packed sample exceeds address-step limit")
    if model_config is not None and tokenizer_path is None:
        raise ValueError("model-configured packed analysis requires tokenizer_path")
    if model_config is None and tokenizer_path is not None:
        raise ValueError("packed tokenizer_path requires model_config")
    tokenizer_identity: dict[str, object] = {"vocabulary_bound_checked": False}
    if model_config is not None:
        tokenizer_file = _file(Path(tokenizer_path), "tokenizer_path")
        tokenizer, tokenizer_identity = _bind_model_tokenizer(
            model_config, tokenizer_file
        )
        tokenizer_identity["canonical_sha256"] = canonical_tokenizer_sha256(tokenizer)
    prepared = load_prepared_data(Path(prepared_path), byte_enabled=False)
    if (
        model_config is not None
        and tokenizer_identity["canonical_sha256"]
        != prepared.manifest["tokenizer_sha256"]
    ):
        raise ValueError("packed tokenizer differs from the prepared tokenizer")
    tokenizer_identity["canonical_sha256"] = prepared.manifest["tokenizer_sha256"]
    if prepared.manifest.get("allocation") is not None:
        raise ValueError("packed lexical analysis does not support allocation sidecars")
    candidate_blocks = max(0, (len(prepared.train) - 1) // sequence_length)
    if candidate_blocks > MAX_DOCUMENTS:
        raise ValueError(
            "prepared training exceeds candidate-block limit; use a bounded prepared subset"
        )
    dataset = TokenBlockDataset(
        prepared.train, sequence_length, supervision=prepared.train_supervision
    )
    count = min(max_blocks, len(dataset))
    if not count:
        raise ValueError("prepared training has no eligible full blocks")

    def sequences() -> Iterable[list[int]]:
        for index in range(count):
            tokens, _, _ = dataset.numpy_block(index)
            if (tokens < 0).any():
                raise ValueError("packed training token IDs must be nonnegative")
            yield tokens.tolist()

    result = _analyze_sequences(
        sequences(),
        settings,
        {
            **model_identity,
            "input_scope": "verified_prepared_train_prefix",
            "training_corpus": {
                "prepared_manifest_sha256": prepared.manifest["manifest_sha256"],
                "train": prepared.manifest["train"],
                "supervision": prepared.manifest.get("supervision"),
                "source_identity_sha256": prepared.manifest.get(
                    "source_identity_sha256"
                ),
            },
            "tokenizer": tokenizer_identity,
        },
        model_config=model_config,
    )
    result.pop("analysis_sha256")
    result["version"] = 3 if model_config is not None else 2
    result["blocks"] = result.pop("documents")
    result["block_frequency"] = result.pop("document_frequency")
    result["addressing"].pop("document_boundary")
    result["addressing"]["block_boundary"] = (
        "configured_padding; document EOS preserved inside blocks"
        if model_config is not None
        else "zero_padded; document EOS preserved inside blocks"
    )
    result["sampling"] = {
        "kind": "first_eligible_training_blocks_once",
        "sequence_length": sequence_length,
        "max_blocks": max_blocks,
        "selected_blocks": count,
        "eligible_blocks": len(dataset),
        "selected_block_indices": dataset.block_indices[:count].tolist(),
        "complete_single_pass": count == len(dataset),
        "schedule_weighted": False,
        "includes_unsupervised_input_positions": True,
    }
    result["analysis_sha256"] = canonical_digest(result)
    return result


def _analyze_sequences(
    sequences: Iterable[list[int]],
    settings: tuple[int, int, int, tuple[int, ...]],
    identity: dict[str, object],
    *,
    model_config: ModelConfig | None = None,
) -> dict[str, object]:
    table_size, memory_dim, hash_heads, orders = settings
    prime_xor = (
        model_config is not None and model_config.memory_addressing == "prime_xor_v1"
    )
    sizes = (
        model_config.memory_table_sizes
        if prime_xor
        else (table_size,) * (len(orders) * hash_heads)
    )
    mapping = model_config.memory_token_mapping if model_config is not None else None
    token_mapping = mapping.token_ids if mapping is not None else None
    padding = model_config.memory_padding_token_id if model_config is not None else 0
    memory_padding = token_mapping[padding] if token_mapping is not None else padding
    multipliers = (
        engram_multipliers(
            model_config.memory_hash_seed,
            max(orders),
            model_config.vocab_size,
            layer_id=model_config.memory_block_index or 0,
        )
        if prime_xor
        else ()
    )
    stream_sizes = {
        (order, head): sizes[index * hash_heads + head]
        for index, order in enumerate(orders)
        for head in range(hash_heads)
    }
    track_raw = token_mapping is not None and any(
        index != value for index, value in enumerate(token_mapping)
    )
    raw_ngram_keys: dict[int, set[tuple[int, ...]]] = (
        {order: set() for order in orders} if track_raw else {}
    )
    stream_count = len(orders) * hash_heads
    hash_steps_per_token = sum(orders) * hash_heads
    bitmaps = {key: bytearray((size + 7) // 8) for key, size in stream_sizes.items()}
    unique_address_counts = {key: 0 for key in bitmaps}
    unique_ngram_keys: dict[int, set[tuple[int, ...]]] = {
        order: set() for order in orders
    }
    unique_ngram_counts = {order: 0 for order in orders}
    unique_ngram_total = 0
    unique_ngram_components = 0
    token_frequency: Counter[int] = Counter()
    document_frequency: Counter[int] = Counter()
    document_count = 0
    token_count = 0

    for token_ids in sequences:
        document_count += 1
        if document_count > MAX_DOCUMENTS:
            raise ValueError(
                f"training sequences exceeds {MAX_DOCUMENTS} document limit"
            )
        if len(token_ids) > MAX_TOKENS_PER_DOCUMENT:
            raise ValueError(
                f"training sequences document {document_count} exceeds "
                f"{MAX_TOKENS_PER_DOCUMENT} token limit"
            )
        token_count += len(token_ids)
        if token_count > MAX_TOKENS:
            raise ValueError(f"training sequences exceeds {MAX_TOKENS} token limit")
        if token_count * hash_steps_per_token > MAX_HASH_OPERATIONS:
            raise ValueError(
                f"training sequences exceeds {MAX_HASH_OPERATIONS} address-step limit"
            )
        if model_config is not None and any(
            type(token) is not int or not 0 <= token < model_config.vocab_size
            for token in token_ids
        ):
            raise ValueError("training token IDs exceed model vocabulary")
        token_frequency.update(token_ids)
        document_frequency.update(set(token_ids))
        memory_ids = (
            [token_mapping[token] for token in token_ids]
            if token_mapping is not None
            else token_ids
        )
        for position in range(len(token_ids)):
            for order in orders:
                if track_raw:
                    raw_key = _ngram_key(token_ids, position, order, padding)
                    if raw_key not in raw_ngram_keys[order]:
                        if unique_ngram_total >= MAX_UNIQUE_NGRAMS:
                            raise ValueError(
                                "training sequences exceeds the combined raw/canonical unique n-gram limit"
                            )
                        if (
                            unique_ngram_components + order
                            > MAX_UNIQUE_NGRAM_COMPONENTS
                        ):
                            raise ValueError(
                                "training sequences exceeds the combined raw/canonical unique n-gram component limit"
                            )
                        raw_ngram_keys[order].add(raw_key)
                        unique_ngram_total += 1
                        unique_ngram_components += order
                ngram_key = _ngram_key(memory_ids, position, order, memory_padding)
                known_keys = unique_ngram_keys[order]
                if ngram_key in known_keys:
                    continue
                if unique_ngram_total >= MAX_UNIQUE_NGRAMS:
                    raise ValueError(
                        f"training sequences exceeds {MAX_UNIQUE_NGRAMS} unique n-gram limit"
                    )
                if unique_ngram_components + order > MAX_UNIQUE_NGRAM_COMPONENTS:
                    raise ValueError(
                        "training sequences exceeds the unique n-gram component limit"
                    )
                known_keys.add(ngram_key)
                unique_ngram_counts[order] += 1
                unique_ngram_total += 1
                unique_ngram_components += order
                for head in range(hash_heads):
                    size = stream_sizes[order, head]
                    address = (
                        engram_ngram_address(ngram_key, size, multipliers)
                        if prime_xor
                        else _address_for_key(ngram_key, table_size=size, head=head)
                    )
                    bitmap = bitmaps[order, head]
                    byte_index, bit_index = divmod(address, 8)
                    mask = 1 << bit_index
                    if not bitmap[byte_index] & mask:
                        bitmap[byte_index] |= mask
                        unique_address_counts[order, head] += 1

    if document_count == 0:
        raise ValueError("training sequences contains no conversations")
    if token_count == 0:
        raise ValueError("training sequences tokenized to no tokens")

    entropy_bits = -sum(
        (count / token_count) * math.log2(count / token_count)
        for count in token_frequency.values()
    )
    streams: list[dict[str, object]] = []
    for order in orders:
        distinct_ngrams = unique_ngram_counts[order]
        repeated_accesses = token_count - distinct_ngrams
        for head in range(hash_heads):
            unique = unique_address_counts[order, head]
            collisions = distinct_ngrams - unique
            streams.append(
                {
                    "order": order,
                    "hash_head": head,
                    "lookup_count": token_count,
                    "distinct_ngram_count": distinct_ngrams,
                    "repeated_accesses": repeated_accesses,
                    "unique_occupied_buckets": unique,
                    "collisions": collisions,
                    "collision_rate": collisions / distinct_ngrams,
                    "repeated_access_rate": repeated_accesses / token_count,
                    "utilization": unique / stream_sizes[order, head],
                    **(
                        {
                            "table_size": stream_sizes[order, head],
                            "distinct_raw_ngram_count": len(raw_ngram_keys[order])
                            if track_raw
                            else distinct_ngrams,
                            "mapping_equivalent_key_count": (
                                len(raw_ngram_keys[order]) - distinct_ngrams
                            )
                            if track_raw
                            else 0,
                            "raw_repeated_accesses": token_count
                            - (
                                len(raw_ngram_keys[order])
                                if track_raw
                                else distinct_ngrams
                            ),
                        }
                        if model_config is not None
                        else {}
                    ),
                }
            )

    table_bytes = sum(sizes) * memory_dim * 4
    result: dict[str, object] = {
        "format": "sparselab-lexical-corpus-analysis",
        "version": 1,
        **identity,
        "documents": document_count,
        "tokens": token_count,
        "token_frequency": [
            {"token_id": token_id, "count": count}
            for token_id, count in sorted(token_frequency.items())
        ],
        "document_frequency": [
            {"token_id": token_id, "count": count}
            for token_id, count in sorted(document_frequency.items())
        ],
        "empirical_unigram_entropy_bits_per_token": entropy_bits,
        "addressing": {
            **(
                {
                    "algorithm": TOKEN_SCHEME_V1,
                    "recurrence": _V1_RECURRENCE,
                }
                if all(
                    token_uses_legacy(table_size, head) for head in range(hash_heads)
                )
                else {
                    "algorithm": TOKEN_SCHEME_V2,
                    "recurrence": _V2_RECURRENCE,
                }
            ),
            "initial_address": "hash_head+1",
            "document_boundary": "zero_padded",
            "collision_key": "distinct zero-padded token n-gram keys across corpus",
            "collision_definition": (
                "distinct keys minus occupied buckets; repeated lookups are separate"
            ),
            "distinct_key_limits": {
                "max_keys": MAX_UNIQUE_NGRAMS,
                "max_key_components": MAX_UNIQUE_NGRAM_COMPONENTS,
            },
            "table_size": table_size,
            "ngram_orders": list(orders),
            "hash_heads": hash_heads,
        },
        "streams": streams,
        "storage": {
            "table_fp32_bytes": table_bytes,
            "table_count": stream_count,
            "memory_dim": memory_dim,
            "excludes": ["output_projection_weights", "gate_weights"],
        },
    }
    if model_config is not None:
        result["version"] = 3
        addressing = result["addressing"]
        addressing.update(
            {
                "table_sizes": list(sizes),
                "raw_padding_token_id": padding,
                "memory_padding_token_id": memory_padding,
                "document_boundary": "configured_padding",
                "collision_key": "distinct canonical padded token n-gram keys across corpus",
                "collision_definition": "canonical distinct keys minus occupied buckets; raw-to-canonical equivalences and repeated accesses are separate",
                "token_mapping": (
                    {
                        key: value
                        for key, value in mapping.model_dump(mode="json").items()
                        if key != "token_ids"
                    }
                    if mapping is not None
                    else {
                        "algorithm": "identity",
                        "vocab_size": model_config.vocab_size,
                        "memory_vocab_size": model_config.vocab_size,
                    }
                ),
            }
        )
        if track_raw:
            addressing["distinct_key_limits"]["scope"] = (
                "combined raw and canonical key/component storage"
            )
        if prime_xor:
            addressing.update(
                {
                    "algorithm": ENGRAM_SCHEME_V1,
                    "recurrence": "signed_int64_XOR(token_id[position-offset]*multiplier[offset]) % stream_prime",
                    "initial_address": 0,
                    "multipliers": list(multipliers),
                    "hash_seed": model_config.memory_hash_seed,
                    "multiplier_vocab_size": model_config.vocab_size,
                    "layer_id": model_config.memory_block_index or 0,
                }
            )
        result["token_frequency_space"] = "original tokenizer IDs"
    result["analysis_sha256"] = canonical_digest(result)
    return result


__all__ = ["analyze_prepared_training", "analyze_training_corpus"]
