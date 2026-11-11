from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from tokenizers import Tokenizer

from klab.data.byte_hash import table_address, token_bytes
from klab.engines.mlx import MLXEngine, preserve_rng_state
from klab.engram.semantic import SemanticQueryBatch


def _addresses_from_ids(
    tokenizer: Tokenizer, ids: Sequence[int], table_size: int, ngram_size: int
) -> list[int]:
    source = bytearray()
    addresses: list[int] = []
    for token_id in ids:
        source.extend(token_bytes(tokenizer, token_id))
        addresses.append(table_address(bytes(source[-ngram_size:]), table_size))
    return addresses


def _prompt_byte_addresses(
    tokenizer: Tokenizer, prompt: str, ids: list[int], table_size: int, ngram_size: int
) -> list[int]:
    """Build causal addresses from raw tokenizer bytes, including split Unicode tokens."""
    if not prompt:
        return [0]
    encoding = tokenizer.encode(prompt, add_special_tokens=False)
    if encoding.ids != ids:
        raise ValueError("prompt IDs must come from the supplied tokenizer")
    addresses = _addresses_from_ids(tokenizer, ids, table_size, ngram_size)
    if b"".join(token_bytes(tokenizer, token_id) for token_id in ids) != prompt.encode(
        "utf-8"
    ):
        raise ValueError("tokenizer byte encoding does not reproduce the prompt")
    return addresses


def _validate_generation_options(
    max_seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    stop_sequences: Sequence[str],
) -> None:
    if (
        isinstance(max_seq_len, bool)
        or not isinstance(max_seq_len, int)
        or max_seq_len <= 0
    ):
        raise ValueError("max_seq_len must be a positive integer")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 0
    ):
        raise ValueError("max_new_tokens must be a non-negative integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be finite and non-negative")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if isinstance(stop_sequences, str) or any(
        not isinstance(stop, str) or not stop for stop in stop_sequences
    ):
        raise ValueError("stop_sequences must contain only non-empty strings")


def _next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    blocked_ids: set[int],
    generator: torch.Generator,
) -> int:
    if not torch.isfinite(logits).all():
        raise ValueError("generation logits must be finite")
    scores = logits.float().clone()
    for token_id in blocked_ids:
        if 0 <= token_id < scores.numel():
            scores[token_id] = -torch.inf
    if not torch.isfinite(scores).any():
        raise ValueError("generation has no permitted output tokens")
    if temperature == 0:
        return int(scores.argmax())
    scores.div_(temperature)
    if top_k:
        limit = min(top_k, scores.numel())
        cutoff = torch.topk(scores, limit).values[-1]
        scores[scores < cutoff] = -torch.inf
    probabilities = torch.softmax(scores, dim=0).cpu()
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def _cache_capability(model: torch.nn.Module) -> tuple[bool, str | None]:
    """Duck-type the cache API so existing inference-only modules still work."""
    capability = getattr(model, "incremental_cache_capability", None)
    cached_forward = getattr(model, "forward_cached", None)
    if not callable(cached_forward):
        return False, "model does not expose incremental KV-cache decoding"
    if capability is None:
        return True, None
    return bool(capability.supported), capability.reason


def _mlx_next_token(
    engine: MLXEngine,
    logits: Any,
    *,
    temperature: float,
    top_k: int,
    blocked_ids: set[int],
    key: Any,
) -> tuple[int, Any]:
    """Select one native MLX token without materializing a logits vector on CPU."""
    mx = engine._mx
    assert mx is not None
    finite = mx.all(mx.isfinite(logits))
    mx.eval(finite)
    if not bool(finite):
        raise ValueError("generation logits must be finite")
    scores = logits.astype(mx.float32)
    for token_id in blocked_ids:
        if 0 <= token_id < scores.shape[0]:
            scores = mx.where(
                mx.arange(scores.shape[0]) == token_id, -float("inf"), scores
            )
    permitted = mx.any(mx.isfinite(scores))
    mx.eval(permitted)
    if not bool(permitted):
        raise ValueError("generation has no permitted output tokens")
    if temperature == 0:
        token = mx.argmax(scores)
        mx.eval(token)
        return int(token), key
    scores = scores / temperature
    if top_k:
        limit = min(top_k, scores.shape[0])
        cutoff = mx.min(mx.topk(scores, k=limit))
        scores = mx.where(scores < cutoff, -float("inf"), scores)
    next_key, sample_key = mx.random.split(key)
    token = mx.random.categorical(scores[None, :], key=sample_key)[0]
    mx.eval(token)
    return int(token), next_key


def _semantic_queries_for_context(
    queries: Any,
    *,
    prompt_length: int,
    active_start: int,
    active_length: int,
) -> Any:
    if queries is None:
        return None
    if isinstance(queries, Mapping):
        return {
            key: _semantic_queries_for_context(
                batch,
                prompt_length=prompt_length,
                active_start=active_start,
                active_length=active_length,
            )
            for key, batch in queries.items()
        }
    if not isinstance(queries, SemanticQueryBatch) or queries.vectors.ndim == 2:
        return queries
    vectors = queries.vectors
    if vectors.shape[1] != prompt_length:
        raise ValueError(
            "sequence semantic queries must match the encoded prompt length"
        )
    prompt_start = min(active_start, prompt_length)
    prompt_end = min(active_start + active_length, prompt_length)
    generated_count = max(0, active_start + active_length - prompt_length)
    vector_parts = [vectors[:, prompt_start:prompt_end]]
    if generated_count:
        vector_parts.append(vectors[:, -1:, :].expand(-1, generated_count, -1))
    active_vectors = (
        vector_parts[0] if len(vector_parts) == 1 else torch.cat(vector_parts, dim=1)
    )
    active_mask = queries.mask
    if active_mask is not None and active_mask.ndim == 2:
        mask_parts = [active_mask[:, prompt_start:prompt_end]]
        if generated_count:
            mask_parts.append(active_mask[:, -1:].expand(-1, generated_count))
        active_mask = (
            mask_parts[0] if len(mask_parts) == 1 else torch.cat(mask_parts, dim=1)
        )
    as_of = queries.as_of
    if isinstance(as_of, tuple):
        batch_size = vectors.shape[0]
        as_of_rows = [
            as_of[index * prompt_length : (index + 1) * prompt_length]
            for index in range(batch_size)
        ]
        active_as_of: list[Any] = []
        for row in as_of_rows:
            active_as_of.extend(row[prompt_start:prompt_end])
            if generated_count:
                active_as_of.extend([row[-1]] * generated_count)
        as_of = tuple(active_as_of)
    return SemanticQueryBatch(
        encoder=queries.encoder,
        vectors=active_vectors,
        mask=active_mask,
        as_of=as_of,
    )


def generate(
    model: Any,
    tokenizer: Tokenizer,
    prompt: str,
    max_seq_len: int,
    max_new_tokens: int,
    device: torch.device | str,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int = 0,
    stop_sequences: Sequence[str] = (),
    strict_context: bool = False,
    use_cache: bool = True,
    engine: MLXEngine | None = None,
    semantic_queries: SemanticQueryBatch
    | Mapping[str, SemanticQueryBatch]
    | None = None,
) -> str:
    """Continue ``prompt`` using locally seeded sampled decoding."""
    text, _ = generate_with_token_ids(
        model,
        tokenizer,
        prompt,
        max_seq_len,
        max_new_tokens,
        device,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        stop_sequences=stop_sequences,
        strict_context=strict_context,
        use_cache=use_cache,
        engine=engine,
        semantic_queries=semantic_queries,
    )
    return text


def generate_with_token_ids(
    model: Any,
    tokenizer: Tokenizer,
    prompt: str,
    max_seq_len: int,
    max_new_tokens: int,
    device: torch.device | str,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int = 0,
    stop_sequences: Sequence[str] = (),
    strict_context: bool = False,
    use_cache: bool = True,
    engine: MLXEngine | None = None,
    semantic_queries: SemanticQueryBatch
    | Mapping[str, SemanticQueryBatch]
    | None = None,
) -> tuple[str, list[int]]:
    """Continue ``prompt`` using locally seeded sampled decoding.

    Semantic inputs are already-encoded, identity-checked query batches. Sequence
    queries track context truncation and repeat their final position across generated
    tokens; cached and full-prefix PyTorch paths preserve the same query trajectory.
    Native MLX does not implement semantic attachments.

    Returned IDs are sampled completion tokens (not prompt or EOS), before any
    text-only stop-sequence trimming. They preserve the exact decoding trajectory
    even if decoding cannot be reversed by encoding the displayed text.
    """
    _validate_generation_options(
        max_seq_len, max_new_tokens, temperature, top_k, seed, stop_sequences
    )
    if engine is not None:
        if device != "metal" or model is not engine.model:
            raise ValueError(
                "native MLX generation requires its model and device='metal'"
            )
        if semantic_queries is not None:
            raise ValueError("native MLX generation does not support semantic queries")
    elif not isinstance(device, torch.device):
        raise TypeError("PyTorch generation requires a torch.device")
    mx = engine._mx if engine is not None else None
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    byte_memory = engine is None and getattr(model.config, "memory", "none") in {
        "byte",
        "portable",
    }
    if not ids:
        bos = tokenizer.token_to_id("<bos>")
        if bos is None:
            raise ValueError("tokenizer has no <bos> token")
        ids = [bos]
    prompt_length = len(ids)
    if strict_context and len(ids) + max_new_tokens > max_seq_len:
        raise ValueError("prompt and requested completion exceed max_seq_len")
    if max_new_tokens == 0:
        return prompt, []

    source_bytes = bytearray(prompt.encode("utf-8")) if byte_memory else None
    addresses = (
        _prompt_byte_addresses(
            tokenizer,
            prompt,
            ids,
            model.config.memory_table_size,
            model.config.memory_ngram_size,
        )
        if byte_memory and prompt
        else [0]
        if byte_memory
        else None
    )
    blocked_ids = {
        token_id
        for name in ("<pad>", "<bos>", "<unk>")
        if (token_id := tokenizer.token_to_id(name)) is not None
    }
    eos = tokenizer.token_to_id("<eos>")
    generated: list[int] = []
    generator = None
    if engine is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    cache_enabled = engine is None and use_cache and _cache_capability(model)[0]
    cache: Any = None

    def full_prefix_logits() -> Any:
        active_ids = ids[-max_seq_len:]
        if engine is not None:
            assert mx is not None
            return engine.logits(mx.array([active_ids], dtype=mx.int32))[0, -1]
        input_ids = torch.tensor([active_ids], device=device)
        byte_addresses = (
            torch.tensor([addresses[-max_seq_len:]], device=device)
            if addresses is not None
            else None
        )
        options: dict[str, Any] = {"byte_addresses": byte_addresses}
        query_input = _semantic_queries_for_context(
            semantic_queries,
            prompt_length=prompt_length,
            active_start=len(ids) - len(active_ids),
            active_length=len(active_ids),
        )
        if query_input is not None:
            options["semantic_queries"] = query_input
        return model(input_ids, **options)[0, -1]

    def rebuild_cache(remaining_tokens: int) -> torch.Tensor:
        nonlocal cache
        active_ids = ids[-max_seq_len:]
        input_ids = torch.tensor([active_ids], device=device)
        byte_addresses = (
            torch.tensor([addresses[-max_seq_len:]], device=device)
            if addresses is not None
            else None
        )
        options: dict[str, Any] = {
            "cache_capacity": min(max_seq_len, len(active_ids) + remaining_tokens),
            "byte_addresses": byte_addresses,
        }
        query_input = _semantic_queries_for_context(
            semantic_queries,
            prompt_length=prompt_length,
            active_start=len(ids) - len(active_ids),
            active_length=len(active_ids),
        )
        if query_input is not None:
            options["semantic_queries"] = query_input
        logits, cache = model.forward_cached(  # type: ignore[attr-defined]
            input_ids, **options
        )
        return logits[0, -1]

    was_training = model.training
    model.eval()
    try:
        with preserve_rng_state() if engine is not None else torch.inference_mode():
            key = mx.random.key(seed) if mx is not None else None
            logits = (
                rebuild_cache(max_new_tokens) if cache_enabled else full_prefix_logits()
            )
            for _ in range(max_new_tokens):
                if engine is not None:
                    token, key = _mlx_next_token(
                        engine,
                        logits,
                        temperature=temperature,
                        top_k=top_k,
                        blocked_ids=blocked_ids,
                        key=key,
                    )
                else:
                    assert generator is not None
                    token = _next_token(
                        logits,
                        temperature=temperature,
                        top_k=top_k,
                        blocked_ids=blocked_ids,
                        generator=generator,
                    )
                if token == eos:
                    break
                generated.append(token)
                ids.append(token)
                if addresses is not None:
                    assert source_bytes is not None
                    source_bytes.extend(token_bytes(tokenizer, token))
                    addresses.append(
                        table_address(
                            bytes(source_bytes[-model.config.memory_ngram_size :]),
                            model.config.memory_table_size,
                        )
                    )
                if stop_sequences:
                    decoded = tokenizer.decode(generated, skip_special_tokens=True)
                    if any(stop in decoded for stop in stop_sequences):
                        break
                if len(generated) == max_new_tokens:
                    break
                if not cache_enabled:
                    logits = full_prefix_logits()
                elif cache.length >= cache.capacity:
                    logits = rebuild_cache(max_new_tokens - len(generated))
                else:
                    next_ids = torch.tensor([[token]], device=device)
                    next_addresses = (
                        torch.tensor([[addresses[-1]]], device=device)
                        if addresses is not None
                        else None
                    )
                    options: dict[str, Any] = {
                        "cache": cache,
                        "byte_addresses": next_addresses,
                    }
                    query_input = _semantic_queries_for_context(
                        semantic_queries,
                        prompt_length=prompt_length,
                        active_start=len(ids) - 1,
                        active_length=1,
                    )
                    if query_input is not None:
                        options["semantic_queries"] = query_input
                    next_logits, cache = model.forward_cached(  # type: ignore[attr-defined]
                        next_ids, **options
                    )
                    logits = next_logits[0, -1]
    finally:
        model.train(was_training)

    completion = tokenizer.decode(generated, skip_special_tokens=True)
    for stop in stop_sequences:
        completion = completion.split(stop, 1)[0]
    return prompt + completion, generated
