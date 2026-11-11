"""Request-local decoding with usage and cancellation for the local server.

The historical generation.py is byte-pinned by scientific preregistrations.
Keep that reference decoder unchanged; reuse its sampling, addressing, semantic
context and cache-capability helpers here, with parity tests guarding trajectory.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import Event
from typing import Any, Literal

import torch
from tokenizers import Tokenizer

from klab.data.byte_hash import table_address, token_bytes
from klab.engines.mlx import MLXEngine, preserve_rng_state
from klab.engram.semantic import SemanticQueryBatch
from klab.evaluation.generation import (
    _cache_capability,
    _mlx_next_token,
    _next_token,
    _prompt_byte_addresses,
    _semantic_queries_for_context,
    _validate_generation_options,
)
from klab.training.attempt_budget import AttemptBudget, AttemptBudgetError


class GenerationCancelled(RuntimeError):
    """A request was cancelled at a decoding boundary."""


@dataclass(frozen=True)
class GenerationResult:
    """Decoded text and actual sampling accounting, before text-only trimming.

    ``text`` includes the prompt. ``token_ids`` excludes prompt and EOS tokens;
    ``completion_tokens`` includes a sampled EOS and any trimmed string stop.
    An empty prompt consumes a synthetic BOS, counted in ``prompt_tokens``.
    """

    text: str
    token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length"]
    forward_input_positions: int = 0
    cache_used: bool | None = None


def _check_cancelled(cancellation: Event | Callable[[], bool] | None) -> None:
    if cancellation is not None:
        cancelled = (
            cancellation.is_set() if isinstance(cancellation, Event) else cancellation()
        )
        if cancelled:
            raise GenerationCancelled("generation cancelled")


def generate_result(
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
    cancellation: Event | Callable[[], bool] | None = None,
    engine: MLXEngine | None = None,
    semantic_queries: SemanticQueryBatch
    | Mapping[str, SemanticQueryBatch]
    | None = None,
    accounting_label: str | None = None,
) -> GenerationResult:
    """Continue ``prompt`` with sampling metadata and cooperative cancellation.

    Cancellation is checked at token boundaries; it cannot interrupt a running
    backend kernel. Request-local caches are discarded and model mode restored.

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
    _check_cancelled(cancellation)
    if max_new_tokens == 0:
        return GenerationResult(prompt, [], prompt_length, 0, "length")
    forward_allocation = AttemptBudget.forward_allocation_active_from_environment()
    if forward_allocation and accounting_label is None:
        raise AttemptBudgetError("contracted generation requires an accounting label")
    if (
        accounting_label is not None
        and os.environ.get("KLAB_ATTEMPT_BUDGET_LEDGER")
        and not forward_allocation
    ):
        raise AttemptBudgetError(
            "labelled generation requires a forward-input allocation in the active ledger"
        )
    if accounting_label is not None:
        if not accounting_label.strip():
            raise ValueError("generation accounting label must be nonempty")
        if forward_allocation:
            worst_case_inputs = sum(
                min(max_seq_len, prompt_length + generated)
                for generated in range(max_new_tokens)
            )
            AttemptBudget.reserve_generation_from_environment(
                accounting_label,
                requested_tokens=max_new_tokens,
                forward_positions=worst_case_inputs,
            )

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
    completion_tokens = 0
    finish_reason: Literal["stop", "length"] = "length"
    generator = None
    if engine is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    cache_enabled = engine is None and use_cache and _cache_capability(model)[0]
    cache: Any = None
    forward_input_positions = 0

    def full_prefix_logits() -> Any:
        nonlocal forward_input_positions
        active_ids = ids[-max_seq_len:]
        forward_input_positions += len(active_ids)
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
        nonlocal cache, forward_input_positions
        active_ids = ids[-max_seq_len:]
        forward_input_positions += len(active_ids)
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
            _check_cancelled(cancellation)
            logits = (
                rebuild_cache(max_new_tokens) if cache_enabled else full_prefix_logits()
            )
            for _ in range(max_new_tokens):
                _check_cancelled(cancellation)
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
                completion_tokens += 1
                if token == eos:
                    finish_reason = "stop"
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
                        finish_reason = "stop"
                        break
                if len(generated) == max_new_tokens:
                    break
                _check_cancelled(cancellation)
                if not cache_enabled:
                    logits = full_prefix_logits()
                elif cache.length >= cache.capacity:
                    logits = rebuild_cache(max_new_tokens - len(generated))
                else:
                    forward_input_positions += 1
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
        # Drop request-owned tensors even when a caller retains an exception traceback.
        cache = None
        options = {}
        logits = None
        next_logits = None
        model.train(was_training)

    completion = tokenizer.decode(generated, skip_special_tokens=True)
    for stop in stop_sequences:
        completion = completion.split(stop, 1)[0]
    return GenerationResult(
        prompt + completion,
        generated,
        prompt_length,
        completion_tokens,
        finish_reason,
        forward_input_positions,
        cache_enabled,
    )
