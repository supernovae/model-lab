"""Export a checkpoint as a Hugging Face Llama directory.

The inverse of ``weights import --format hf_llama_safetensors``: the same
tensor names, the same rotary-coordinate permutation and the same config
fields, so an exported directory imports back to identical weights and loads
with ``transformers.LlamaForCausalLM``. The tokenizer is copied beside the
weights with the run's chat template in the form ``apply_chat_template`` reads.

Only the architecture Llama can express exports: dense attention (grouped
key/value heads included), dense SwiGLU feed-forward and no memory.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from klab.config.models import RunConfig
from klab.data.tokenizer import SPECIAL_TOKENS, load_tokenizer
from klab.training.manifest import canonical_json, sha256_file
from klab.training.objectives import read_checkpoint

EXPORT_FORMAT = "hf_llama_safetensors"
_BLOCK_NAMES = {
    "norm1.weight": "input_layernorm.weight",
    "norm2.weight": "post_attention_layernorm.weight",
    "attention.q_proj.weight": "self_attn.q_proj.weight",
    "attention.k_proj.weight": "self_attn.k_proj.weight",
    "attention.v_proj.weight": "self_attn.v_proj.weight",
    "attention.out_proj.weight": "self_attn.o_proj.weight",
    "ffn.gate.weight": "mlp.gate_proj.weight",
    "ffn.up.weight": "mlp.up_proj.weight",
    "ffn.down.weight": "mlp.down_proj.weight",
}


def _require_llama_architecture(config: RunConfig) -> None:
    model, attention = config.model, config.attention
    if (
        config.attention_kinds != {"dense"}
        or config.ffn_kinds != {"dense"}
        or model.memory != "none"
        or attention.window_size is not None
    ):
        raise ValueError(
            "Llama export requires dense attention, dense SwiGLU and no memory; "
            "this run uses "
            f"attention {sorted(config.attention_kinds)}, "
            f"feed-forward {sorted(config.ffn_kinds)}, memory {model.memory}"
        )
    if (model.hidden_dim // model.num_heads) % 2:
        raise ValueError("Llama export requires an even attention head width")


def llama_config(config: RunConfig, stop_ids: tuple[int, ...]) -> dict[str, Any]:
    model = config.model
    eos = SPECIAL_TOKENS.index("<eos>")
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": model.hidden_dim,
        "intermediate_size": model.ffn_dim,
        "num_hidden_layers": model.num_layers,
        "num_attention_heads": model.num_heads,
        "num_key_value_heads": model.num_kv_heads or model.num_heads,
        "vocab_size": model.vocab_size,
        "max_position_embeddings": model.max_seq_len,
        "rms_norm_eps": model.rms_norm_eps,
        "rope_theta": float(config.attention.rope_base),
        "hidden_act": "silu",
        "tie_word_embeddings": model.tie_embeddings,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "use_cache": True,
        "torch_dtype": "float32",
        "pad_token_id": SPECIAL_TOKENS.index("<pad>"),
        "bos_token_id": SPECIAL_TOKENS.index("<bos>"),
        "eos_token_id": [eos, *stop_ids] if stop_ids else eos,
    }


def llama_tensors(
    config: RunConfig, weights: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Rename canonical tensors to Llama's and restore its rotary pairing."""
    model = config.model
    remaining = dict(weights)
    exported = {
        "model.embed_tokens.weight": remaining.pop("embedding.weight"),
        "model.norm.weight": remaining.pop("norm.weight"),
    }
    output = remaining.pop("output.weight")
    if not model.tie_embeddings:
        exported["lm_head.weight"] = output
    head_dim = model.hidden_dim // model.num_heads
    for index in range(model.num_layers):
        for ours, theirs in _BLOCK_NAMES.items():
            tensor = remaining.pop(f"blocks.{index}.{ours}")
            # Model Lab pairs adjacent rotary coordinates; Llama pairs the two
            # halves of a head. Reorder the query and key output rows.
            if ours in {"attention.q_proj.weight", "attention.k_proj.weight"}:
                heads = tensor.shape[0] // head_dim
                tensor = (
                    tensor.reshape(heads, head_dim // 2, 2, model.hidden_dim)
                    .transpose(1, 2)
                    .reshape(tensor.shape)
                )
            exported[f"model.layers.{index}.{theirs}"] = tensor
    if remaining:
        raise ValueError(
            f"checkpoint tensors have no Llama equivalent: {sorted(remaining)}"
        )
    return {
        name: tensor.detach().to(torch.float32).contiguous().clone()
        for name, tensor in exported.items()
    }


def export_weights(
    checkpoint: Path, output: Path, *, export_format: str = EXPORT_FORMAT
) -> dict[str, Any]:
    """Write ``checkpoint`` to a new directory ``output`` and return its receipt."""
    if export_format != EXPORT_FORMAT:
        raise ValueError(f"unsupported export format: {export_format}")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite export destination: {output}")
    config, snapshot, manifest = read_checkpoint(Path(checkpoint), role="export")
    _require_llama_architecture(config)
    run_root = Path(manifest["run_root"])
    tokenizer = load_tokenizer(run_root / "tokenizer.json")
    if [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] != [0, 1, 2, 3]:
        raise ValueError("tokenizer special token IDs are not the required 0..3")
    template = config.dataset.template
    tensors = llama_tensors(config, dict(snapshot.model))
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}.export.", dir=output.parent))
    try:
        save_file(tensors, staged / "model.safetensors", metadata={"format": "pt"})
        shutil.copyfile(run_root / "tokenizer.json", staged / "tokenizer.json")
        special = dict(
            zip(
                ("pad_token", "unk_token", "bos_token", "eos_token"),
                SPECIAL_TOKENS,
                strict=True,
            )
        )
        documents: dict[str, dict[str, Any]] = {
            "config.json": llama_config(config, template.stop_token_ids(tokenizer)),
            "generation_config.json": {
                key: value
                for key, value in llama_config(
                    config, template.stop_token_ids(tokenizer)
                ).items()
                if key.endswith("_token_id")
            },
            "tokenizer_config.json": {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": config.model.max_seq_len,
                "clean_up_tokenization_spaces": False,
                "chat_template": template.to_jinja(),
                **special,
            },
            "special_tokens_map.json": special,
        }
        for name, document in documents.items():
            (staged / name).write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        receipt = {
            "format": EXPORT_FORMAT,
            "run_id": manifest["run_id"],
            "checkpoint_sha256": snapshot.checkpoint_sha256,
            "chat_template": template.name,
            "parameters": sum(tensor.numel() for tensor in tensors.values()),
            "files": {
                path.name: sha256_file(path) for path in sorted(staged.iterdir())
            },
        }
        (staged / "klab_export.json").write_bytes(canonical_json(receipt) + b"\n")
        staged.replace(output)
    finally:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
    return {**receipt, "output": str(output)}
