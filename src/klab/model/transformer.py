"""Explicit pre-norm dense decoder language model."""

from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Literal

import torch
from torch import Tensor, nn

from klab.config.models import AttentionConfig, ModelConfig
from klab.engram.semantic import (
    SemanticMemoryAdapter,
    SemanticQueryBatch,
    SemanticRetriever,
)
from klab.model.attention.dense import DenseAttention
from klab.model.attention.latent import LatentAttention
from klab.model.attention.sparse import BlockSparseAttention
from klab.model.cache import (
    AttentionKVCache,
    GenerationCache,
    IncrementalCacheCapability,
)
from klab.model.ffn import SwiGLU
from klab.model.memory import (
    ByteAddressMemory,
    TokenNgramMemory,
    build_token_memory,
)
from klab.model.moe import (
    TopKMoE,
    diagnostic_writes_enabled,
    suppress_diagnostic_writes,
)
from klab.model.norm import RMSNorm
from klab.model.portable_engram import PortableEngramAdapter, load_portable_engram

SemanticQueryInput = SemanticQueryBatch | Mapping[str, SemanticQueryBatch] | None


class DecoderBlock(nn.Module):
    def __init__(self, model: ModelConfig, attention: AttentionConfig) -> None:
        super().__init__()
        if (
            model.num_kv_heads is not None
            and model.num_kv_heads != model.num_heads
            and attention.kind in {"mla", "block_sparse"}
        ):
            raise ValueError(
                "grouped-query attention is supported only for dense or sliding_window attention"
            )
        self.norm1 = RMSNorm(model.hidden_dim, model.rms_norm_eps)
        self.attention: nn.Module = (
            LatentAttention(
                model.hidden_dim,
                model.num_heads,
                attention.latent_dim,
                model.max_seq_len,
                attention.rope_base,
            )
            if attention.kind == "mla"
            else BlockSparseAttention(
                model.hidden_dim,
                model.num_heads,
                model.max_seq_len,
                attention.rope_base,
                attention.block_size,
                attention.selected_blocks,
            )
            if attention.kind == "block_sparse"
            else DenseAttention(
                model.hidden_dim,
                model.num_heads,
                model.max_seq_len,
                attention.rope_base,
                attention.window_size,
                model.num_kv_heads,
                implementation=attention.implementation,
            )
        )
        self.norm2 = RMSNorm(model.hidden_dim, model.rms_norm_eps)
        self.ffn: nn.Module = (
            SwiGLU(model.hidden_dim, model.ffn_dim)
            if model.ffn == "dense"
            else TopKMoE(
                model.hidden_dim,
                model.ffn_dim,
                model.num_experts,
                model.experts_per_token,
                model.shared_expert,
                model.router_aux_loss_coefficient,
            )
        )

    def forward_with_aux(
        self,
        x: Tensor,
        *,
        valid_target_mask: Tensor | None = None,
        diagnostics: Literal["scalar", "full"] = "scalar",
    ) -> tuple[Tensor, Tensor]:
        attention_input = self.norm1(x)
        if isinstance(self.attention, BlockSparseAttention):
            attention = self.attention(
                attention_input,
                valid_target_mask=valid_target_mask,
                diagnostics=diagnostics,
            )
        else:
            attention = self.attention(attention_input)
        x = x + attention
        if isinstance(self.ffn, TopKMoE):
            ffn_output, auxiliary_loss = self.ffn.forward_with_aux(
                self.norm2(x),
                valid_target_mask=valid_target_mask,
                diagnostics=diagnostics,
            )
            return x + ffn_output, auxiliary_loss
        return x + self.ffn(self.norm2(x)), x.new_zeros(())

    def forward_cached(
        self, x: Tensor, cache: AttentionKVCache
    ) -> tuple[Tensor, AttentionKVCache]:
        """Inference-only incremental block evaluation without auxiliary diagnostics."""
        attention_input = self.norm1(x)
        if not isinstance(self.attention, (DenseAttention, LatentAttention)):
            raise TypeError("block-sparse attention requires prefix recomputation")
        attention, next_cache = self.attention.forward_cached(attention_input, cache)
        x = x + attention
        return x + self.ffn(self.norm2(x)), next_cache

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_aux(x)[0]


def initialize_module(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)) and module.weight.requires_grad:
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)


class DenseLM(nn.Module):
    def __init__(self, model: ModelConfig, attention: AttentionConfig) -> None:
        super().__init__()
        self.config = model
        self.embedding = nn.Embedding(model.vocab_size, model.hidden_dim)
        self.blocks = nn.ModuleList(
            DecoderBlock(model, attention) for _ in range(model.num_layers)
        )
        self.norm = RMSNorm(model.hidden_dim, model.rms_norm_eps)
        independent_memory = model.memory == "ngram" and model.memory_independent_init
        memory_rng = torch.get_rng_state() if independent_memory else None
        self.memory: nn.Module | None = (
            None
            if model.memory == "none"
            else (
                build_token_memory(model)
                if model.memory == "ngram"
                else (
                    ByteAddressMemory(
                        model.hidden_dim, model.memory_table_size, model.memory_dim
                    )
                    if model.memory == "byte"
                    else PortableEngramAdapter(
                        load_portable_engram(
                            model.memory_package_path,
                            expected_shape=(model.memory_table_size, model.memory_dim),
                            expected_ngram_size=model.memory_ngram_size,
                        ),
                        model.hidden_dim,
                    )
                )
            )
        )
        self.extra_memories = nn.ModuleDict(
            {
                str(site): build_token_memory(model, site)
                for site in model.memory_block_sites[1:]
            }
        )
        if memory_rng is not None:
            torch.set_rng_state(memory_rng)
        self.semantic_memories = nn.ModuleDict()
        self.output = nn.Linear(model.hidden_dim, model.vocab_size, bias=False)
        if independent_memory:
            memory, extras = self.memory, self.extra_memories
            self.memory = None
            self.extra_memories = nn.ModuleDict()
            self.apply(self._initialize)
            with torch.random.fork_rng(devices=[]):
                memory.apply(self._initialize)
                extras.apply(self._initialize)
            self.memory, self.extra_memories = memory, extras
        else:
            self.apply(self._initialize)
        if model.tie_embeddings:
            self.output.weight = self.embedding.weight
        self._forward_block_calls = 0
        self._cache_owner = object()
        self._recomputed_block_calls = 0

    _initialize = staticmethod(initialize_module)

    def _apply_memory(
        self,
        hidden: Tensor,
        token_ids: Tensor,
        byte_addresses: Tensor | None,
        *,
        mask: Tensor | None = None,
        cache: GenerationCache | None = None,
        position_mask: Tensor | None = None,
        block_index: int | None = None,
    ) -> Tensor:
        extra_key = str(block_index)
        memory = self.memory
        if extra_key in self.extra_memories:
            memory = self.extra_memories[extra_key]
        if memory is None:
            return hidden
        if mask is not None and (
            mask.dtype != torch.bool or mask.shape != hidden.shape[:2]
        ):
            raise ValueError("memory mask must be boolean and align with hidden states")
        if isinstance(memory, TokenNgramMemory):
            if mask is not None and memory.conv is not None:
                raise ValueError("convolved token memory does not support memory_mask")
            if cache is not None:
                history = (
                    cache.memory_histories.get(extra_key)
                    if extra_key in self.extra_memories
                    else cache.memory_history
                )
                updated, next_history = memory.forward_last(hidden, token_ids, history)
                if extra_key in self.extra_memories:
                    if next_history is not None:
                        cache.memory_histories[extra_key] = next_history
                else:
                    cache.memory_history = next_history
            else:
                updated = memory(hidden, token_ids, position_mask=position_mask)
        elif isinstance(memory, (ByteAddressMemory, PortableEngramAdapter)):
            if byte_addresses is None:
                raise ValueError(
                    "byte-addressed memory requires prepared byte addresses"
                )
            updated = memory(hidden, byte_addresses)
        else:
            raise TypeError(f"unsupported memory module: {type(memory).__name__}")
        return (
            updated
            if mask is None
            else torch.where(mask.unsqueeze(-1), updated, hidden)
        )

    def add_semantic_memory(
        self,
        name: str,
        retriever: SemanticRetriever,
        *,
        site: Literal["embedding", "after_block", "final"],
        block_index: int | None = None,
        min_score: float | None = None,
    ) -> SemanticMemoryAdapter:
        """Attach one verified semantic pack with independent site ownership."""
        if site not in ("embedding", "after_block", "final"):
            raise ValueError(f"unsupported semantic injection site: {site!r}")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name):
            raise ValueError("semantic memory name must be lowercase kebab/snake case")
        if name in self.semantic_memories:
            raise ValueError(f"semantic memory {name!r} is already attached")
        if site == "after_block":
            if type(block_index) is not int or not 0 <= block_index < len(self.blocks):
                raise ValueError(
                    "after_block site needs a valid zero-based block_index"
                )
        elif block_index is not None:
            raise ValueError("block_index is valid only for after_block site")
        if not isinstance(retriever, SemanticRetriever):
            raise TypeError("semantic memory requires a verified SemanticRetriever")
        for attached in self.semantic_memories.values():
            if (
                attached.retriever.key_encoder.sha256 == retriever.key_encoder.sha256
                and attached.retriever.key_encoder != retriever.key_encoder
            ):
                raise ValueError("one encoder SHA-256 cannot name different identities")
        adapter = SemanticMemoryAdapter(
            retriever,
            self.config.hidden_dim,
            min_score=min_score,
            site=site,
            block_index=block_index,
        )
        adapter.apply(self._initialize)
        adapter.to(
            device=self.embedding.weight.device, dtype=self.embedding.weight.dtype
        )
        adapter.train(self.training)
        self.semantic_memories[name] = adapter
        return adapter

    def replace_semantic_memory(self, name: str, retriever: SemanticRetriever) -> None:
        """Swap verified pack assets while preserving a named adapter's weights."""
        try:
            adapter = self.semantic_memories[name]
        except KeyError as error:
            raise ValueError(f"semantic memory {name!r} is not attached") from error
        adapter.replace_retriever(retriever)

    def _semantic_query_map(
        self, semantic_queries: SemanticQueryInput
    ) -> dict[str, SemanticQueryBatch]:
        if not self.semantic_memories:
            if semantic_queries is not None:
                raise ValueError("semantic queries were supplied without an attachment")
            return {}
        if semantic_queries is None:
            raise ValueError("attached semantic memory requires explicit query vectors")
        if isinstance(semantic_queries, SemanticQueryBatch):
            batches = {semantic_queries.encoder.sha256: semantic_queries}
        elif isinstance(semantic_queries, Mapping):
            batches = {}
            for key, batch in semantic_queries.items():
                if not isinstance(batch, SemanticQueryBatch):
                    raise TypeError(
                        "semantic query mapping values must be SemanticQueryBatch"
                    )
                if key != batch.encoder.sha256 or key in batches:
                    raise ValueError(
                        "semantic query mapping keys must be unique encoder SHA-256 values"
                    )
                batches[key] = batch
        else:
            raise TypeError(
                "semantic_queries must be a SemanticQueryBatch or encoder mapping"
            )
        required = {
            module.retriever.key_encoder.sha256: module.retriever.key_encoder
            for module in self.semantic_memories.values()
        }
        if set(batches) != set(required):
            raise ValueError(
                "semantic query mapping must cover exactly the attached key encoders"
            )
        for digest, identity in required.items():
            if batches[digest].encoder != identity:
                raise ValueError(
                    "semantic query encoder identity does not match its attached pack"
                )
        return batches

    def _apply_semantic_site(
        self,
        hidden: Tensor,
        queries: Mapping[str, SemanticQueryBatch],
        *,
        site: Literal["embedding", "after_block", "final"],
        block_index: int | None = None,
    ) -> Tensor:
        for name in sorted(self.semantic_memories):
            adapter = self.semantic_memories[name]
            if adapter.site == site and adapter.block_index == block_index:
                batch = queries[adapter.retriever.key_encoder.sha256]
                hidden = adapter(hidden, batch)
        return hidden

    def forward_with_aux(
        self,
        input_ids: Tensor,
        *,
        byte_addresses: Tensor | None = None,
        semantic_queries: SemanticQueryInput = None,
        memory_mask: Tensor | None = None,
        valid_target_mask: Tensor | None = None,
        activation_checkpointing: bool = False,
        diagnostics: Literal["scalar", "full"] = "scalar",
        return_hidden: bool = False,
        position_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, non-empty sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence exceeds model.max_seq_len")
        if diagnostics not in ("scalar", "full"):
            raise ValueError(f"unknown diagnostics policy: {diagnostics}")
        semantic_query_map = self._semantic_query_map(semantic_queries)
        self._forward_block_calls = 0
        self._recomputed_block_calls = 0
        x = self.embedding(input_ids)
        if self.config.memory_injection == "embedding":
            x = self._apply_memory(
                x,
                input_ids,
                byte_addresses,
                mask=memory_mask,
                position_mask=position_mask,
            )
        x = self._apply_semantic_site(x, semantic_query_map, site="embedding")
        auxiliary_loss = x.new_zeros(())
        for block_index, block in enumerate(self.blocks):
            if activation_checkpointing and self.training and torch.is_grad_enabled():

                def run_block(
                    hidden: Tensor, current: DecoderBlock = block
                ) -> tuple[Tensor, Tensor]:
                    if diagnostic_writes_enabled():
                        self._forward_block_calls += 1
                    else:
                        self._recomputed_block_calls += 1
                    return current.forward_with_aux(
                        hidden,
                        valid_target_mask=valid_target_mask,
                        diagnostics=diagnostics,
                    )

                def checkpoint_context() -> tuple[object, object]:
                    return nullcontext(), suppress_diagnostic_writes()

                x, block_aux = torch.utils.checkpoint.checkpoint(
                    run_block,
                    x,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    context_fn=checkpoint_context,
                )
            else:
                self._forward_block_calls += 1
                x, block_aux = block.forward_with_aux(
                    x, valid_target_mask=valid_target_mask, diagnostics=diagnostics
                )
            if (
                self.config.memory_injection == "after_block"
                and block_index in self.config.memory_block_sites
            ):
                x = self._apply_memory(
                    x,
                    input_ids,
                    byte_addresses,
                    mask=memory_mask,
                    position_mask=position_mask,
                    block_index=block_index,
                )
            x = self._apply_semantic_site(
                x,
                semantic_query_map,
                site="after_block",
                block_index=block_index,
            )
            auxiliary_loss = auxiliary_loss + block_aux
        x = self.norm(x)
        if self.config.memory_injection == "final":
            x = self._apply_memory(
                x,
                input_ids,
                byte_addresses,
                mask=memory_mask,
                position_mask=position_mask,
            )
        x = self._apply_semantic_site(x, semantic_query_map, site="final")
        return (x if return_hidden else self.output(x)), auxiliary_loss

    def architecture_metric_tensors(self) -> dict[str, Tensor]:
        """Detached scalar diagnostics from the most recent forward on this device.

        The engine deliberately materializes these only at its update boundary so
        accelerator accumulation does not synchronize once per microbatch.
        """
        result: dict[str, Tensor] = {}
        if self.memory is not None and self.memory.last_diagnostics is not None:
            diagnostic = self.memory.last_diagnostics
            for name in (
                "lookup_count",
                "unique_addresses",
                "collision_count",
                "bucket_reuse_rate",
                "table_utilization",
                "maximum_address_fraction",
                "gate_mean",
                "value_norm",
                "hidden_norm",
            ):
                result[f"engram/{name}"] = getattr(diagnostic, name).detach()
            result[f"engram/injection/{self.config.memory_injection}"] = (
                diagnostic.gate_mean.new_ones(()).detach()
            )
        if self.config.memory_block_indices:
            ports = {
                str(self.config.memory_block_sites[0]): self.memory,
                **dict(self.extra_memories.items()),
            }
            for site, memory in ports.items():
                diagnostic = memory.last_diagnostics
                if diagnostic is not None:
                    for name in (
                        "lookup_count",
                        "unique_addresses",
                        "collision_count",
                        "bucket_reuse_rate",
                        "table_utilization",
                        "maximum_address_fraction",
                        "gate_mean",
                        "value_norm",
                        "hidden_norm",
                    ):
                        result[f"engram/port/{site}/{name}"] = getattr(
                            diagnostic, name
                        ).detach()
            diagnostics = [
                memory.last_diagnostics
                for memory in ports.values()
                if memory.last_diagnostics is not None
            ]
            if diagnostics:
                total_lookups = sum(item.lookup_count for item in diagnostics)
                unique = sum(item.unique_addresses for item in diagnostics)
                result["engram/lookup_count"] = total_lookups
                result["engram/unique_addresses"] = unique
                result["engram/collision_count"] = total_lookups - unique
                result["engram/bucket_reuse_rate"] = (
                    total_lookups - unique
                ).float() / total_lookups.clamp_min(1)
                if diagnostics[0].document_scoped:
                    result["engram/table_utilization"] = sum(
                        memory.last_diagnostics.table_utilization
                        * sum(memory.table_sizes)
                        for memory in ports.values()
                    ) / sum(sum(memory.table_sizes) for memory in ports.values())
                    result["engram/maximum_address_fraction"] = torch.stack(
                        [item.maximum_address_fraction for item in diagnostics]
                    ).mean()
                else:
                    result["engram/table_utilization"] = unique.float() / sum(
                        sum(memory.table_sizes) for memory in ports.values()
                    )
                    result["engram/maximum_address_fraction"] = torch.stack(
                        [
                            item.maximum_address_fraction
                            * item.lookup_count
                            / total_lookups.clamp_min(1)
                            for item in diagnostics
                        ]
                    ).max()
                for name in ("gate_mean", "value_norm", "hidden_norm"):
                    result[f"engram/{name}"] = torch.stack(
                        [getattr(item, name) for item in diagnostics]
                    ).mean()
        for name, adapter in sorted(self.semantic_memories.items()):
            diagnostic = adapter.last_diagnostics
            if diagnostic is None:
                continue
            for metric in (
                "lookup_count",
                "hit_count",
                "unknown_count",
                "conflict_count",
                "temporal_miss_count",
                "mean_score",
                "gate_mean",
                "value_norm",
                "hidden_norm",
            ):
                result[f"semantic/{name}/{metric}"] = getattr(
                    diagnostic, metric
                ).detach()
            location = (
                f"after_block/{adapter.block_index}"
                if adapter.site == "after_block"
                else adapter.site
            )
            result[f"semantic/{name}/injection/{location}"] = (
                diagnostic.gate_mean.new_ones(()).detach()
            )
        for index, block in enumerate(self.blocks):
            if (
                isinstance(block.ffn, TopKMoE)
                and block.ffn.last_diagnostics is not None
            ):
                diagnostic = block.ffn.last_diagnostics
                for metric, field in (
                    ("router_entropy", "entropy"),
                    ("maximum_expert_fraction", "maximum_fraction"),
                    ("mean_topk_probability", "mean_topk_probability"),
                ):
                    result[f"moe/layer_{index}/{metric}"] = getattr(
                        diagnostic, field
                    ).detach()
            diagnostic = getattr(block.attention, "last_diagnostics", None)
            if diagnostic is not None:
                for metric, field in (
                    ("available_tokens", "available_tokens"),
                    ("selected_tokens", "selected_tokens"),
                    ("selection_ratio", "selection_ratio"),
                    ("estimated_flops", "estimated_attention_flops"),
                    ("dense_teacher_mass", "dense_teacher_mass"),
                    ("dense_teacher_topk_recall", "dense_teacher_topk_recall"),
                ):
                    value = getattr(diagnostic, field)
                    if value is not None:
                        result[f"attention/layer_{index}/{metric}"] = value.detach()
        return result

    @property
    def incremental_cache_capability(self) -> IncrementalCacheCapability:
        """Describe whether this architecture can decode from projected K/V state."""
        if any(
            isinstance(block.attention, BlockSparseAttention) for block in self.blocks
        ):
            return IncrementalCacheCapability(
                False,
                "block_sparse selection is prefix-dependent; generation recomputes "
                "the active prefix for exact reference semantics",
            )
        mode = (
            "mla"
            if any(
                isinstance(block.attention, LatentAttention) for block in self.blocks
            )
            else "sliding"
            if any(
                isinstance(block.attention, DenseAttention)
                and block.attention.window_size is not None
                for block in self.blocks
            )
            else "dense"
        )
        return IncrementalCacheCapability(True, backend=mode)

    @torch.no_grad()
    def forward_cached(
        self,
        input_ids: Tensor,
        *,
        cache_capacity: int | None = None,
        cache: GenerationCache | None = None,
        byte_addresses: Tensor | None = None,
        semantic_queries: SemanticQueryInput = None,
    ) -> tuple[Tensor, GenerationCache]:
        """Return logits and bounded, owner-checked inference-only request state.

        A new request must state its total prompt-plus-generation capacity.
        Callers rebuild the cache at a context rollover so absolute RoPE positions
        retain the same reset semantics as full-prefix reference generation.
        """
        if self.training:
            raise RuntimeError(
                "forward_cached is inference-only; call model.eval() first"
            )
        capability = self.incremental_cache_capability
        if not capability.supported:
            raise RuntimeError(capability.reason)
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, non-empty sequence]")
        if input_ids.device != self.embedding.weight.device:
            raise ValueError("input_ids device does not match model parameters")
        projection_dtype = (
            torch.get_autocast_dtype(input_ids.device.type)
            if torch.is_autocast_enabled(input_ids.device.type)
            else self.embedding.weight.dtype
        )
        if cache is None:
            if cache_capacity is None:
                raise ValueError(
                    "new cached requests require an explicit cache_capacity"
                )
            if type(cache_capacity) is not int:
                raise TypeError("cache_capacity must be an integer")
            if not 0 < cache_capacity <= self.config.max_seq_len:
                raise ValueError("cache_capacity must be within model.max_seq_len")
            if input_ids.shape[1] > cache_capacity:
                raise ValueError("input_ids exceed requested cache_capacity")
            layers = [
                block.attention.create_cache(
                    input_ids.shape[0],
                    cache_capacity,
                    device=input_ids.device,
                    dtype=projection_dtype,
                )
                for block in self.blocks
                if isinstance(block.attention, (DenseAttention, LatentAttention))
            ]
            cache = GenerationCache(
                layers=layers,
                input_ids=torch.empty(
                    (input_ids.shape[0], cache_capacity),
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                ),
                length=0,
                capacity=cache_capacity,
                mode=capability.backend or "dense",
                owner=self._cache_owner,
                batch_size=input_ids.shape[0],
                device=input_ids.device,
                input_dtype=input_ids.dtype,
            )
        else:
            if cache_capacity is not None:
                raise ValueError("cache_capacity is set only when creating a cache")
            if cache.owner is not self._cache_owner:
                raise ValueError(
                    "generation cache belongs to a different model instance"
                )
            if (
                cache.mode != capability.backend
                or len(cache.layers) != len(self.blocks)
                or cache.length < 0
                or cache.length > cache.capacity
            ):
                raise ValueError("generation cache is incompatible with this model")
            if cache.length + input_ids.shape[1] > cache.capacity:
                raise ValueError(
                    "generation cache capacity exceeded; rebuild the active context"
                )
            if (
                cache.input_ids.shape != (cache.batch_size, cache.capacity)
                or cache.input_ids.device != cache.device
                or cache.input_ids.dtype != cache.input_dtype
            ):
                raise ValueError("generation cache input-ID storage is invalid")
            if any(
                layer.length != cache.length
                or layer.position != cache.length
                or layer.capacity != cache.capacity
                for layer in cache.layers
            ):
                raise ValueError("generation cache layers are inconsistent")
            for block, layer in zip(self.blocks, cache.layers, strict=True):
                attention = block.attention
                if not isinstance(attention, (DenseAttention, LatentAttention)):
                    raise TypeError("generation cache contains a non-incremental layer")
                value_dim = (
                    attention.head_dim
                    if isinstance(attention, DenseAttention)
                    else attention.value_dim
                )
                kv_heads = (
                    attention.num_kv_heads
                    if isinstance(attention, DenseAttention)
                    else attention.num_heads
                )
                if (
                    layer.key.shape
                    != (
                        cache.batch_size,
                        kv_heads,
                        cache.capacity,
                        attention.head_dim,
                    )
                    or layer.value.shape
                    != (
                        cache.batch_size,
                        kv_heads,
                        cache.capacity,
                        value_dim,
                    )
                    or layer.key.device != input_ids.device
                    or layer.value.device != input_ids.device
                    or layer.key.dtype != projection_dtype
                    or layer.value.dtype != projection_dtype
                    or layer.key.requires_grad
                    or layer.value.requires_grad
                    or layer.key.grad_fn is not None
                    or layer.value.grad_fn is not None
                ):
                    raise ValueError(
                        "generation cache layer shape, device, or dtype is invalid"
                    )
        if isinstance(self.memory, (ByteAddressMemory, PortableEngramAdapter)):
            if byte_addresses is None:
                raise ValueError(
                    "byte-addressed memory requires prepared byte addresses"
                )
            if (
                byte_addresses.shape != input_ids.shape
                or byte_addresses.device != input_ids.device
            ):
                raise ValueError(
                    "byte_addresses must match cached input IDs in shape and device"
                )
        if isinstance(self.memory, TokenNgramMemory) and self.memory.conv is not None:
            expected_history = (
                cache.batch_size,
                self.memory.conv_history_length,
                self.config.hidden_dim,
            )
            history = cache.memory_history
            if cache.length and history is None:
                raise ValueError(
                    "generation cache is missing memory convolution history"
                )
            if history is not None and (
                history.shape != expected_history
                or history.device != input_ids.device
                or history.dtype != self.embedding.weight.dtype
                or history.requires_grad
                or history.grad_fn is not None
            ):
                raise ValueError(
                    "generation cache memory convolution history is invalid"
                )
        elif cache.memory_history is not None:
            raise ValueError("generation cache contains unexpected memory history")
        expected_extra = {
            key
            for key, memory in self.extra_memories.items()
            if memory.conv is not None
        }
        if set(cache.memory_histories) - expected_extra or (
            cache.length and set(cache.memory_histories) != expected_extra
        ):
            raise ValueError("generation cache extra memory histories mismatch")
        for key, history in cache.memory_histories.items():
            memory = self.extra_memories[key]
            if (
                history.shape
                != (
                    cache.batch_size,
                    memory.conv_history_length,
                    self.config.hidden_dim,
                )
                or history.device != input_ids.device
                or history.dtype != self.embedding.weight.dtype
                or history.requires_grad
                or history.grad_fn is not None
            ):
                raise ValueError("generation cache extra memory history is invalid")
        semantic_query_map = self._semantic_query_map(semantic_queries)
        cache.append_input_ids(input_ids)
        full_ids = cache.input_ids[:, : cache.length]
        x = self.embedding(input_ids)
        if self.config.memory_injection == "embedding":
            x = self._apply_memory(x, full_ids, byte_addresses, cache=cache)
        x = self._apply_semantic_site(x, semantic_query_map, site="embedding")
        for block_index, (block, layer_cache) in enumerate(
            zip(self.blocks, cache.layers, strict=True)
        ):
            x, _ = block.forward_cached(x, layer_cache)
            if (
                self.config.memory_injection == "after_block"
                and block_index in self.config.memory_block_sites
            ):
                x = self._apply_memory(
                    x, full_ids, byte_addresses, cache=cache, block_index=block_index
                )
            x = self._apply_semantic_site(
                x,
                semantic_query_map,
                site="after_block",
                block_index=block_index,
            )
        x = self.norm(x)
        if self.config.memory_injection == "final":
            x = self._apply_memory(x, full_ids, byte_addresses, cache=cache)
        x = self._apply_semantic_site(x, semantic_query_map, site="final")
        return self.output(x), cache

    @property
    def recomputed_block_call_ratio(self) -> float:
        """Recomputed DecoderBlock calls divided by original forward block calls."""
        if not self._forward_block_calls:
            return 0.0
        return self._recomputed_block_calls / self._forward_block_calls

    def forward(
        self,
        input_ids: Tensor,
        *,
        byte_addresses: Tensor | None = None,
        semantic_queries: SemanticQueryInput = None,
        memory_mask: Tensor | None = None,
        diagnostics: Literal["scalar", "full"] = "scalar",
    ) -> Tensor:
        return self.forward_with_aux(
            input_ids,
            byte_addresses=byte_addresses,
            semantic_queries=semantic_queries,
            memory_mask=memory_mask,
            diagnostics=diagnostics,
        )[0]
