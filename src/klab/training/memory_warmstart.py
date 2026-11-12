"""Explicit, zero-impact dense-to-token-memory weight promotion.

This transforms verified in-memory weights only. Native continuation owns source
checkpoint verification and child-run creation; no source file is changed here.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from klab.config.models import RunConfig
from klab.model.inspection import named_tensor_inventory
from klab.model.memory import build_token_memory
from klab.model.transformer import initialize_module
from klab.training.manifest import architecture_sha256, config_sha256

_TOKEN_MEMORY_FIELDS = frozenset(
    {
        "memory",
        "memory_injection",
        "memory_block_index",
        "memory_block_indices",
        "memory_variant",
        "memory_conv_kernel_size",
        "memory_table_size",
        "memory_ngram_size",
        "memory_dim",
        "memory_ngram_orders",
        "memory_hash_heads",
        "memory_addressing",
        "memory_table_sizes",
        "memory_hash_seed",
        "memory_token_mapping",
        "memory_padding_token_id",
        "memory_independent_init",
        "memory_gate_transform",
    }
)


def migrate_dense_to_ngram(
    source: RunConfig,
    target: RunConfig,
    source_weights: Mapping[str, Tensor],
    *,
    source_tokenizer_sha256: str,
    target_tokenizer_sha256: str,
    source_checkpoint_sha256: str,
    extend_context: bool = False,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Preserve every trunk tensor and initialize only a zero-output memory branch.

    The caller must verify the source generation and tokenizer artifacts first.
    This is a new experiment with reset optimizer/schedule/RNG/counters, never a
    full-state resume of a different architecture.
    """
    for value in (
        source_tokenizer_sha256,
        target_tokenizer_sha256,
        source_checkpoint_sha256,
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("memory migration requires verified SHA-256 identities")
    if source_tokenizer_sha256 != target_tokenizer_sha256:
        raise ValueError("memory migration requires matching tokenizer identity")
    if source.runtime.engine != "pytorch" or target.runtime.engine != "pytorch":
        raise ValueError("memory migration requires PyTorch weights")
    if source.model.memory != "none" or target.model.memory != "ngram":
        raise ValueError("memory migration requires dense no-memory to token ngram")
    source_model = source.model.model_dump(mode="json")
    target_model = target.model.model_dump(mode="json")
    for field in _TOKEN_MEMORY_FIELDS:
        source_model.pop(field, None)
        target_model.pop(field, None)
    context_transfer = None
    if extend_context:
        if (
            source.attention_kinds != {"dense"}
            or target.model.max_seq_len <= source.model.max_seq_len
        ):
            raise ValueError("context transfer requires a larger dense RoPE context")
        context_transfer = {
            "source_max_seq_len": source.model.max_seq_len,
            "target_max_seq_len": target.model.max_seq_len,
            "rope_policy": "unchanged-base-unscaled-positions",
            "rope_base": source.attention.rope_base,
            "prefix_policy": "preserve-original-position-frequencies",
        }
        source_model.pop("max_seq_len")
        target_model.pop("max_seq_len")
    if source_model != target_model or source.attention != target.attention:
        raise ValueError(
            "memory migration must preserve backbone and attention semantics"
        )
    mapping = getattr(target.model, "memory_token_mapping", None)
    if mapping is not None:
        identity = mapping if isinstance(mapping, dict) else mapping.model_dump()
        if identity["tokenizer_sha256"] != target_tokenizer_sha256:
            raise ValueError("memory mapping differs from verified tokenizer identity")
    target_inventory = named_tensor_inventory(
        target.model,
        target.attention,
        trainable_parameters=target.training.trainable_parameters,
    )
    blocks = getattr(target.model, "memory_block_indices", ())
    ports = [
        ("memory", None),
        *((f"extra_memories.{block}", block) for block in blocks[1:]),
    ]
    output_names = [f"{prefix}.output.weight" for prefix, _ in ports]
    if not all(target_inventory[name].trainable for name in output_names):
        raise ValueError(
            "memory migration requires trainable memory output projections"
        )
    expected = named_tensor_inventory(source.model, source.attention)
    if set(source_weights) != set(expected):
        raise ValueError("source trunk tensor inventory differs from configuration")
    for name, spec in expected.items():
        value = source_weights[name]
        if (
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or not value.is_floating_point()
            or value.layout != torch.strided
            or tuple(value.shape) != spec.shape
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError(f"invalid source trunk tensor: {name}")
        if spec.alias_of is not None and not torch.equal(
            value, source_weights[spec.alias_of]
        ):
            raise ValueError(f"source tied tensor differs: {name}")
    # CPU-only initialization does not consume the training RNG or change trunk
    # initialization according to memory size. Constructors retain their normal
    # convolution initialization; only the value/output projection is zeroed.
    weights = dict(source_weights)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(target.seed)
        for prefix, block in ports:
            memory = build_token_memory(target.model, block_index=block)
            memory.apply(initialize_module)
            with torch.no_grad():
                memory.output.weight.zero_()
            weights.update(
                {
                    f"{prefix}.{name}": tensor
                    for name, tensor in memory.state_dict().items()
                }
            )
    decision: dict[str, object] = {
        "kind": "dense_to_ngram_warmstart",
        "version": 1,
        "parent_checkpoint_sha256": source_checkpoint_sha256,
        "source_config_sha256": config_sha256(source.model_dump(mode="json")),
        "target_config_sha256": config_sha256(target.model_dump(mode="json")),
        "source_architecture_sha256": architecture_sha256(
            source.model_dump(mode="json")
        ),
        "target_architecture_sha256": architecture_sha256(
            target.model_dump(mode="json")
        ),
        "tokenizer_sha256": source_tokenizer_sha256,
        "memory_initialization": "target-seed-normal-init-zero-value-projection-v1",
        "zeroed_tensors": output_names,
        "memory_initialization_seed": target.seed,
        "copied_trunk_tensors": sorted(source_weights),
        "optimizer_policy": "reset",
        "schedule_policy": "reset",
        "rng_policy": "target_seed",
        "counter_policy": "new_run_zero_with_parent_lineage",
        "resume_level": "weights_only",
        "reason": "explicit --add-ngram-memory with --promote; not exact resume",
    }
    if context_transfer is not None:
        decision["context_transfer"] = context_transfer
        decision["source_optimizer"] = source.optimizer.model_dump(mode="json")
        decision["target_optimizer"] = target.optimizer.model_dump(mode="json")
    return weights, decision
