"""Export a checkpoint as a Hugging Face directory.

``hf_llama_safetensors`` is the inverse of ``weights import --format
hf_llama_safetensors``: the same tensor names, the same rotary-coordinate
permutation and the same config fields, so an exported directory imports back
to identical weights and loads with ``transformers.LlamaForCausalLM``. Only the
architecture Llama can express exports this way: dense attention (grouped
key/value heads included), dense SwiGLU feed-forward and no memory.

``hf_remote_code`` exports any other built-in architecture with Lab's decoder
modules beside the weights (see :mod:`klab.training.remote_code`). Without a
format, a Llama-shaped run exports as Llama and any other as remote code.

Either way the tokenizer is copied beside the weights with the run's chat
template in the form ``apply_chat_template`` reads.
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
from klab.training.manifest import canonical_json, sha256_file, source_identity
from klab.training.objectives import read_checkpoint
from klab.training.remote_code import (
    EXPORT_FORMAT as REMOTE_CODE_FORMAT,
)
from klab.training.remote_code import (
    remote_code_config,
    remote_code_refusal,
    remote_code_tensors,
    vendor_modules,
)

EXPORT_FORMAT = "hf_llama_safetensors"
EXPORT_FORMATS = (EXPORT_FORMAT, REMOTE_CODE_FORMAT)
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


def _llama_refusal(config: RunConfig) -> str | None:
    model, attention = config.model, config.attention
    if (
        config.attention_kinds != {"dense"}
        or config.ffn_kinds != {"dense"}
        or model.memory != "none"
        or attention.window_size is not None
    ):
        return (
            "Llama export requires dense attention, dense SwiGLU and no memory; "
            "this run uses "
            f"attention {sorted(config.attention_kinds)}, "
            f"feed-forward {sorted(config.ffn_kinds)}, memory {model.memory}; "
            f"--format {REMOTE_CODE_FORMAT} exports it with Lab's model code"
        )
    if (model.hidden_dim // model.num_heads) % 2:
        return "Llama export requires an even attention head width"
    return None


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
    checkpoint: Path, output: Path, *, export_format: str | None = None
) -> dict[str, Any]:
    """Write ``checkpoint`` to a new directory ``output`` and return its receipt."""
    if export_format is not None and export_format not in EXPORT_FORMATS:
        raise ValueError(f"unsupported export format: {export_format}")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite export destination: {output}")
    config, snapshot, manifest = read_checkpoint(Path(checkpoint), role="export")
    llama_refusal = _llama_refusal(config)
    if export_format is None:
        export_format = REMOTE_CODE_FORMAT if llama_refusal else EXPORT_FORMAT
    refusal = (
        llama_refusal if export_format == EXPORT_FORMAT else remote_code_refusal(config)
    )
    if refusal:
        raise ValueError(f"cannot export run {manifest['run_id']}: {refusal}")
    run_root = Path(manifest["run_root"])
    tokenizer = load_tokenizer(run_root / "tokenizer.json")
    if [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] != [0, 1, 2, 3]:
        raise ValueError("tokenizer special token IDs are not the required 0..3")
    template = config.dataset.template
    stop_ids = template.stop_token_ids(tokenizer)
    code: dict[str, tuple[str, str]] = {}
    if export_format == EXPORT_FORMAT:
        tensors = llama_tensors(config, dict(snapshot.model))
        model_config = llama_config(config, stop_ids)
    else:
        source_sha256 = str(source_identity()["sha256"])
        tensors = remote_code_tensors(config, dict(snapshot.model))
        model_config = remote_code_config(
            config,
            stop_ids,
            {
                "lab_source_sha256": source_sha256,
                "run_id": manifest["run_id"],
                "checkpoint_sha256": snapshot.checkpoint_sha256,
                "tokenizer_sha256": sha256_file(run_root / "tokenizer.json"),
            },
        )
        code = vendor_modules(source_sha256)
    generation: dict[str, Any] = {
        key: value for key, value in model_config.items() if key.endswith("_token_id")
    }
    if code:
        # klab generate never emits these; suppressing them keeps greedy
        # decoding token-for-token the same.
        generation["suppress_tokens"] = sorted(
            SPECIAL_TOKENS.index(name) for name in ("<pad>", "<unk>", "<bos>")
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}.export.", dir=output.parent))
    try:
        save_file(tensors, staged / "model.safetensors", metadata={"format": "pt"})
        shutil.copyfile(run_root / "tokenizer.json", staged / "tokenizer.json")
        for name, (_, source) in code.items():
            (staged / name).write_text(source, encoding="utf-8")
        special = dict(
            zip(
                ("pad_token", "unk_token", "bos_token", "eos_token"),
                SPECIAL_TOKENS,
                strict=True,
            )
        )
        documents: dict[str, dict[str, Any]] = {
            "config.json": model_config,
            "generation_config.json": generation,
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
            "format": export_format,
            "run_id": manifest["run_id"],
            "checkpoint_sha256": snapshot.checkpoint_sha256,
            "chat_template": template.name,
            "parameters": sum(tensor.numel() for tensor in tensors.values()),
            "files": {
                path.name: sha256_file(path) for path in sorted(staged.iterdir())
            },
        }
        if code:
            receipt["code"] = {name: module for name, (module, _) in code.items()}
        (staged / "klab_export.json").write_bytes(canonical_json(receipt) + b"\n")
        staged.replace(output)
    finally:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
    return {**receipt, "output": str(output)}
