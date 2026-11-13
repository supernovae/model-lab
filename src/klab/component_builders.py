"""Builders for the attention, feed-forward and memory kinds Lab ships.

Each builder turns one layer's model and attention settings into a module.
:mod:`klab.components` registers them beside entry-point kinds; an exported
model directory carries this module as its whole registry, so it knows only
these kinds. Like the registry, it imports no Torch until a builder runs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def _rope_scaling(model, attention):
    from klab.model.rope import resolve_rope_scaling

    return resolve_rope_scaling(
        getattr(attention, "rope_scaling", None), model.max_seq_len
    )


def _dense_attention(model, attention, params):
    from klab.model.attention.dense import DenseAttention

    return DenseAttention(
        model.hidden_dim,
        model.num_heads,
        model.max_seq_len,
        attention.rope_base,
        attention.window_size,
        model.num_kv_heads,
        rope_scaling=_rope_scaling(model, attention),
        implementation=attention.implementation,
    )


def _latent_attention(model, attention, params):
    from klab.model.attention.latent import LatentAttention

    return LatentAttention(
        model.hidden_dim,
        model.num_heads,
        attention.latent_dim,
        model.max_seq_len,
        attention.rope_base,
        rope_scaling=_rope_scaling(model, attention),
        implementation=attention.implementation,
    )


def _block_sparse_attention(model, attention, params):
    from klab.model.attention.sparse import BlockSparseAttention

    return BlockSparseAttention(
        model.hidden_dim,
        model.num_heads,
        model.max_seq_len,
        attention.rope_base,
        attention.block_size,
        attention.selected_blocks,
        rope_scaling=_rope_scaling(model, attention),
    )


def _local_selected_attention(model, attention, params):
    from klab.model.attention.local_selected import LocalSelectedAttention

    return LocalSelectedAttention(
        model.hidden_dim,
        model.num_heads,
        model.max_seq_len,
        attention.rope_base,
        attention.block_size,
        attention.window_size,
        attention.selected_blocks,
        attention.latent_dim,
        rope_scaling=_rope_scaling(model, attention),
        selector=attention.selector,
    )


def _dense_ffn(model, attention, params):
    from klab.model.ffn import SwiGLU

    return SwiGLU(model.hidden_dim, model.ffn_dim)


def _moe_ffn(model, attention, params):
    from klab.model.moe import TopKMoE

    return TopKMoE(
        model.hidden_dim,
        model.ffn_dim,
        model.num_experts,
        model.experts_per_token,
        model.shared_expert,
        model.router_aux_loss_coefficient,
    )


def _ngram_memory(model, attention, params):
    from klab.model.memory import build_token_memory

    return build_token_memory(model)


def _byte_memory(model, attention, params):
    from klab.model.memory import ByteAddressMemory

    return ByteAddressMemory(
        model.hidden_dim, model.memory_table_size, model.memory_dim
    )


def _portable_memory(model, attention, params):
    from klab.model.portable_engram import PortableEngramAdapter, load_portable_engram

    return PortableEngramAdapter(
        load_portable_engram(
            model.memory_package_path,
            expected_shape=(model.memory_table_size, model.memory_dim),
            expected_ngram_size=model.memory_ngram_size,
        ),
        model.hidden_dim,
    )


Builder = Callable[[Any, Any, Any], Any]
BUILDERS: Mapping[tuple[str, str], Builder] = {
    ("attention", "dense"): _dense_attention,
    ("attention", "sliding_window"): _dense_attention,
    ("attention", "mla"): _latent_attention,
    ("attention", "block_sparse"): _block_sparse_attention,
    ("attention", "local_selected"): _local_selected_attention,
    ("ffn", "dense"): _dense_ffn,
    ("ffn", "moe"): _moe_ffn,
    ("memory", "ngram"): _ngram_memory,
    ("memory", "byte"): _byte_memory,
    ("memory", "portable"): _portable_memory,
}


def build(
    slot: str,
    kind: str,
    model: Any,
    attention: Any,
    params: Mapping[str, Any] | None = None,
) -> Any:
    """Construct a built-in kind; kinds from other packages are not known here."""
    builder = BUILDERS.get((slot, kind))
    if builder is None or params:
        raise ValueError(f"{slot} kind {kind!r} is not one Lab ships")
    return builder(model, attention, None)
