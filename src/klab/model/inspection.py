"""Truthful parameter accounting and shape-only model inspection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache

from torch import Tensor, nn

from klab.config.models import AttentionConfig, ModelConfig, RunConfig
from klab.model.moe import TopKMoE
from klab.model.transformer import DenseLM


@dataclass(frozen=True)
class TensorSpec:
    """One canonical model tensor, represented without allocating it."""

    shape: tuple[int, ...]
    dtype: str = "float32"
    trainable: bool = True
    alias_of: str | None = None

    @property
    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


@dataclass(frozen=True)
class ParameterInventory:
    """Disjoint parameter categories for a configured decoder."""

    total: int
    trainable: int
    active_per_token: int
    embedding: int
    output_head: int
    attention: int
    dense_ffn: int
    routed_expert: int
    shared_expert: int
    router: int
    norm: int
    memory_table: int
    memory_adapter: int
    frozen: int


def _add(
    tensors: dict[str, TensorSpec],
    name: str,
    shape: tuple[int, ...],
    *,
    trainable: bool = True,
) -> None:
    tensors[name] = TensorSpec(shape, trainable=trainable)


def _memory_table_name(name: str) -> bool:
    if name.startswith("extra_memories."):
        name = "memory." + name.split(".", 2)[2]
    return (
        name.startswith(("memory.table.", "memory.extra_tables."))
        or name == "memory.embedding.weight"
    )


@lru_cache(maxsize=64)
def _built_shapes_cached(
    slot: str, kind: str, model_json: str, attention_json: str, params_json: str
) -> tuple[tuple[str, tuple[int, ...], bool], ...]:
    import json

    from klab.components import build

    module = build(
        slot,  # type: ignore[arg-type]
        kind,
        ModelConfig.model_validate_json(model_json),
        AttentionConfig.model_validate_json(attention_json),
        json.loads(params_json),
    )
    return tuple(
        (name, tuple(parameter.shape), parameter.requires_grad)
        for name, parameter in module.named_parameters()
    )


def _built_shapes(
    slot: str,
    kind: str,
    model: ModelConfig,
    attention: AttentionConfig,
    params: Mapping[str, object],
) -> tuple[tuple[str, tuple[int, ...], bool], ...]:
    """Parameter shapes of a kind from another package, read from one instance.

    Lab knows the shapes of its own kinds without building them. Another
    package's kind is built once per distinct setting on the CPU and its
    parameters are listed; the instance is then dropped.
    """
    from klab.training.manifest import canonical_json

    return _built_shapes_cached(
        slot,
        kind,
        model.model_dump_json(),
        attention.model_dump_json(),
        canonical_json(dict(params)).decode(),
    )


def _add_built(
    tensors: dict[str, TensorSpec],
    prefix: str,
    shapes: tuple[tuple[str, tuple[int, ...], bool], ...],
) -> None:
    for name, shape, trainable in shapes:
        _add(tensors, f"{prefix}.{name}", shape, trainable=trainable)


def _add_attention(
    tensors: dict[str, TensorSpec],
    attention_prefix: str,
    m: ModelConfig,
    attention: AttentionConfig,
) -> None:
    from klab.components import is_builtin

    d = m.hidden_dim
    if not is_builtin("attention", attention.kind):
        _add_built(
            tensors,
            attention_prefix,
            _built_shapes("attention", attention.kind, m, attention, attention.params),
        )
        return
    if attention.latent_dim is not None:
        latent = attention.latent_dim
        _add(tensors, f"{attention_prefix}.q_proj.weight", (d, d))
        _add(tensors, f"{attention_prefix}.kv_down.weight", (latent, d))
        _add(tensors, f"{attention_prefix}.k_up.weight", (d, latent))
        _add(tensors, f"{attention_prefix}.out_proj.weight", (d, latent))
    else:
        kv_dim = (m.num_kv_heads or m.num_heads) * (d // m.num_heads)
        _add(tensors, f"{attention_prefix}.q_proj.weight", (d, d))
        _add(tensors, f"{attention_prefix}.k_proj.weight", (kv_dim, d))
        _add(tensors, f"{attention_prefix}.v_proj.weight", (kv_dim, d))
        _add(tensors, f"{attention_prefix}.out_proj.weight", (d, d))
    if attention.selector == "learned":
        assert attention.block_size is not None
        _add(
            tensors,
            f"{attention_prefix}.pool_logits",
            (m.num_heads, attention.block_size),
        )
        _add(tensors, f"{attention_prefix}.coarse_gate", (m.num_heads,))


def _add_feed_forward(
    tensors: dict[str, TensorSpec],
    ffn_prefix: str,
    m: ModelConfig,
    attention: AttentionConfig,
) -> None:
    from klab.components import is_builtin

    d, f = m.hidden_dim, m.ffn_dim
    swiglu = (("gate", (f, d)), ("up", (f, d)), ("down", (d, f)))
    if not is_builtin("ffn", m.ffn):
        _add_built(
            tensors,
            ffn_prefix,
            _built_shapes("ffn", m.ffn, m, attention, m.ffn_params),
        )
    elif m.ffn == "dense":
        for projection, shape in swiglu:
            _add(tensors, f"{ffn_prefix}.{projection}.weight", shape)
    else:
        _add(tensors, f"{ffn_prefix}.router.weight", (m.num_experts, d))
        for expert in range(m.num_experts):
            for projection, shape in swiglu:
                _add(
                    tensors,
                    f"{ffn_prefix}.experts.{expert}.{projection}.weight",
                    shape,
                )
        if m.shared_expert:
            for projection, shape in swiglu:
                _add(
                    tensors,
                    f"{ffn_prefix}.shared_expert.{projection}.weight",
                    shape,
                )


def named_tensor_inventory(
    model_config: ModelConfig,
    attention_config: AttentionConfig | None = None,
    *,
    trainable_parameters: tuple[str, ...] | None = None,
) -> Mapping[str, TensorSpec]:
    """Return the canonical parameter schema without constructing a model.

    Alias entries have the same shape as their stored tensor and point at its
    canonical name.  Portable table dimensions come from the resolved config;
    package identity and contents are deliberately not opened by inspection.
    """
    attention = attention_config or AttentionConfig()
    m = model_config
    d, v = m.hidden_dim, m.vocab_size
    tensors: dict[str, TensorSpec] = {}
    _add(tensors, "embedding.weight", (v, d))

    for index, (layer, layer_attention) in enumerate(m.layer_specs(attention)):
        prefix = f"blocks.{index}"
        _add(tensors, f"{prefix}.norm1.weight", (d,))
        _add(tensors, f"{prefix}.norm2.weight", (d,))
        _add_attention(tensors, f"{prefix}.attention", layer, layer_attention)
        _add_feed_forward(tensors, f"{prefix}.ffn", layer, layer_attention)

    _add(tensors, "norm.weight", (d,))
    if m.external_memory:
        _add_built(
            tensors,
            "memory",
            _built_shapes("memory", m.memory, m, attention, m.memory_params),
        )
    elif m.memory != "none":
        streams = (
            (len(m.memory_ngram_orders) or 1) * m.memory_hash_heads
            if m.memory == "ngram"
            else 1
        )
        ports = max(1, len(m.memory_block_sites))
        for port in range(ports):
            prefix = (
                "memory"
                if port == 0
                else f"extra_memories.{m.memory_block_sites[port]}"
            )
            sizes = (m.memory_table_sizes or (m.memory_table_size,) * streams * ports)[
                port * streams : (port + 1) * streams
            ]
            frozen = m.memory == "portable"
            _add(
                tensors,
                f"{prefix}.table.weight"
                if m.memory != "portable"
                else f"{prefix}.embedding.weight",
                (sizes[0], m.memory_dim),
                trainable=not frozen,
            )
            if m.memory == "ngram":
                for stream in range(1, streams):
                    _add(
                        tensors,
                        f"{prefix}.extra_tables.{stream - 1}.weight",
                        (
                            sizes[stream],
                            m.memory_dim,
                        ),
                    )
            if m.memory_variant == "context_key":
                _add(tensors, f"{prefix}.output.weight", (d, streams * m.memory_dim))
                _add(tensors, f"{prefix}.key.weight", (d, streams * m.memory_dim))
                _add(tensors, f"{prefix}.gate.query_norm.weight", (d,))
                _add(tensors, f"{prefix}.gate.key_norm.weight", (d,))
                if m.memory_conv_kernel_size:
                    _add(
                        tensors,
                        f"{prefix}.conv.weight",
                        (d, 1, m.memory_conv_kernel_size),
                    )
                    _add(tensors, f"{prefix}.conv_norm.weight", (d,))
            else:
                _add(tensors, f"{prefix}.output.weight", (d, m.memory_dim))
                _add(tensors, f"{prefix}.gate.weight", (1, d))

    if m.tie_embeddings:
        tensors["output.weight"] = TensorSpec((v, d), alias_of="embedding.weight")
    else:
        _add(tensors, "output.weight", (v, d))
    if trainable_parameters is None:
        return tensors
    if (
        type(trainable_parameters) is not tuple
        or not trainable_parameters
        or len(trainable_parameters) != len(set(trainable_parameters))
    ):
        raise ValueError(
            "trainable_parameters must be a nonempty tuple of unique names"
        )
    unknown = set(trainable_parameters) - set(tensors)
    if unknown:
        raise ValueError(f"unknown trainable parameter names: {sorted(unknown)}")
    invalid = [
        name
        for name in trainable_parameters
        if tensors[name].alias_of is not None or not tensors[name].trainable
    ]
    if invalid:
        raise ValueError(
            "trainable parameter names must identify trainable canonical storages: "
            f"{sorted(invalid)}"
        )
    selected = set(trainable_parameters)
    return {
        name: TensorSpec(
            spec.shape,
            spec.dtype,
            trainable=(
                spec.alias_of in selected
                if spec.alias_of is not None
                else name in selected
            ),
            alias_of=spec.alias_of,
        )
        for name, spec in tensors.items()
    }


def _sum_specs(tensors: Mapping[str, TensorSpec], predicate: object) -> int:
    return sum(
        spec.numel
        for name, spec in tensors.items()
        if spec.alias_of is None and predicate(name, spec)  # type: ignore[operator]
    )


def parameter_inventory(config: RunConfig) -> ParameterInventory:
    """Aggregate the pure canonical tensor schema into disjoint categories."""
    m = config.model
    tensors = named_tensor_inventory(
        m,
        config.attention,
        trainable_parameters=config.training.trainable_parameters,
    )
    stored = lambda name, spec: spec.alias_of is None
    embedding = _sum_specs(tensors, lambda name, spec: name == "embedding.weight")
    output = _sum_specs(tensors, lambda name, spec: name == "output.weight")
    attention = _sum_specs(tensors, lambda name, spec: ".attention." in name)
    dense = _sum_specs(
        tensors,
        lambda name, spec: (
            ".ffn." in name
            and ".experts." not in name
            and ".shared_expert." not in name
            and ".router." not in name
        ),
    )
    routed = _sum_specs(tensors, lambda name, spec: ".ffn.experts." in name)
    shared = _sum_specs(tensors, lambda name, spec: ".ffn.shared_expert." in name)
    router = _sum_specs(tensors, lambda name, spec: ".ffn.router." in name)
    norm = _sum_specs(
        tensors, lambda name, spec: name == "norm.weight" or ".norm" in name
    )
    table = _sum_specs(
        tensors,
        lambda name, spec: _memory_table_name(name),
    )
    adapter = _sum_specs(
        tensors,
        lambda name, spec: (
            name.startswith(("memory.", "extra_memories.", "semantic_memories."))
            and not (_memory_table_name(name))
        ),
    )
    total = _sum_specs(tensors, stored)
    trainable = _sum_specs(tensors, lambda name, spec: spec.trainable)
    frozen = total - trainable
    active = total
    if not m.tie_embeddings:
        active -= embedding - m.hidden_dim
    for layer, _ in config.layer_specs:
        if layer.ffn == "moe":
            per_expert = 3 * layer.hidden_dim * layer.ffn_dim
            active -= (layer.num_experts - layer.experts_per_token) * per_expert
    if m.memory != "none" and not m.external_memory:
        streams = (
            (len(m.memory_ngram_orders) or 1) * m.memory_hash_heads
            if m.memory == "ngram"
            else 1
        )
        active -= table - streams * m.memory_dim * max(1, len(m.memory_block_sites))
    return ParameterInventory(
        total,
        trainable,
        active,
        embedding,
        output,
        attention,
        dense,
        routed,
        shared,
        router,
        norm,
        table,
        adapter,
        frozen,
    )


def inspection_report(config: RunConfig) -> dict[str, int | str]:
    """Return ``inspect_model``-compatible accounting from shapes alone."""
    from klab.memory import optimizer_state_bytes

    inventory = parameter_inventory(config)
    expert = inventory.routed_expert + inventory.shared_expert
    engram = inventory.memory_table + inventory.memory_adapter
    weight_bytes = inventory.total * 4
    optimizer_bytes = optimizer_state_bytes(config, inventory)
    return {
        "total": inventory.total,
        "trainable": inventory.trainable,
        "active_per_token": inventory.active_per_token,
        "embedding": inventory.embedding,
        "attention": inventory.attention,
        "ffn": inventory.dense_ffn,
        "norm": inventory.norm,
        "output_head": inventory.output_head,
        "expert": expert,
        "routed_expert": inventory.routed_expert,
        "shared_expert": inventory.shared_expert,
        "router": inventory.router,
        "engram": engram,
        "engram_table": inventory.memory_table,
        "engram_adapter": inventory.memory_adapter,
        "frozen": inventory.frozen,
        "memory_injection": "none"
        if config.model.memory == "none"
        else config.model.memory_injection,
        "memory_variant": config.model.memory_variant,
        "memory_block_index": config.model.memory_block_index,
        **(
            {"memory_block_indices": list(config.model.memory_block_indices)}
            if config.model.memory_block_indices
            else {}
        ),
        "expert_note": "not present in dense model"
        if "moe" not in config.ffn_kinds
        else "routed and shared expert storage; expert is a non-additive subtotal",
        "engram_note": "not present"
        if config.model.memory == "none"
        else "all tables and adapters counted in total; one row per table stream in active_per_token",
        "model_weight_bytes": weight_bytes,
        "optimizer_state_bytes": optimizer_bytes,
        "estimated_checkpoint_bytes": weight_bytes + optimizer_bytes,
    }


def _unique_numel(parameters: Iterable[nn.Parameter]) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            total += parameter.numel()
    return total


def inspect_model(model: nn.Module) -> dict[str, int | str]:
    """Count storage/direct use; optimizer fields assume FP32 AdamW.

    Use inspection_report for a configured optimizer rather than this model-only view.
    """
    if not isinstance(model, DenseLM):
        raise TypeError("inspection currently supports DenseLM")
    total = _unique_numel(model.parameters())
    trainable = _unique_numel(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    embedding = model.embedding.weight.numel()
    tied = model.output.weight is model.embedding.weight
    attention = sum(
        _unique_numel(block.attention.parameters()) for block in model.blocks
    )
    dense_ffn = routed_expert = shared_expert = router = active_expert = 0
    for block in model.blocks:
        if isinstance(block.ffn, TopKMoE):
            routed_expert += _unique_numel(block.ffn.experts.parameters())
            active_expert += _unique_numel(
                parameter
                for expert_module in block.ffn.experts[: block.ffn.experts_per_token]
                for parameter in expert_module.parameters()
            )
            router += _unique_numel(block.ffn.router.parameters())
            if block.ffn.shared_expert is not None:
                shared_expert += _unique_numel(block.ffn.shared_expert.parameters())
                active_expert += _unique_numel(block.ffn.shared_expert.parameters())
        else:
            dense_ffn += _unique_numel(block.ffn.parameters())
    norm = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name == "norm.weight" or ".norm" in name
    )
    output_head = 0 if tied else model.output.weight.numel()
    active = total if tied else total - embedding + model.config.hidden_dim
    active -= routed_expert + shared_expert
    active += active_expert
    memory_parameters = memory_tables = active_memory_rows = frozen = 0
    for memory in ([model.memory] if model.memory is not None else []) + list(
        model.extra_memories.values()
    ):
        memory_parameters += _unique_numel(memory.parameters())
        frozen += sum(
            parameter.numel()
            for parameter in memory.parameters()
            if not parameter.requires_grad
        )
        for module in memory.modules():
            if isinstance(module, nn.Embedding):
                memory_tables += module.weight.numel()
                active_memory_rows += module.embedding_dim
    active -= memory_tables - active_memory_rows
    weight_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    optimizer_state_bytes = trainable * 8 + 4 * sum(
        parameter.requires_grad for parameter in model.parameters()
    )
    expert = routed_expert + shared_expert
    return {
        "total": total,
        "trainable": trainable,
        "active_per_token": active,
        "embedding": embedding,
        "attention": attention,
        "ffn": dense_ffn,
        "norm": norm,
        "output_head": output_head,
        "expert": expert,
        "routed_expert": routed_expert,
        "shared_expert": shared_expert,
        "router": router,
        "engram": memory_parameters,
        "engram_table": memory_tables,
        "engram_adapter": memory_parameters - memory_tables,
        "memory_injection": "none"
        if model.memory is None
        else model.config.memory_injection,
        "memory_variant": model.config.memory_variant,
        "memory_block_index": model.config.memory_block_index,
        **(
            {"memory_block_indices": list(model.config.memory_block_indices)}
            if model.config.memory_block_indices
            else {}
        ),
        "frozen": frozen,
        "expert_note": (
            "not present in dense model"
            if not routed_expert
            else "routed and shared expert storage; expert is a non-additive subtotal"
        ),
        "engram_note": (
            "not present"
            if model.memory is None
            else "all tables and adapters counted in total; one row per table stream in active_per_token"
        ),
        "model_weight_bytes": weight_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
        "estimated_checkpoint_bytes": weight_bytes + optimizer_state_bytes,
    }


def architecture_metrics(model: DenseLM) -> dict[str, Tensor]:
    """Return detached device diagnostics from the most recent model forward.

    This compatibility wrapper intentionally leaves scalar materialization to
    the execution boundary; converting accelerator tensors here would serialize
    every accumulated microbatch.
    """
    return model.architecture_metric_tensors()
