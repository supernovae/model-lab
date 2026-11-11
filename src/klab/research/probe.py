"""Bounded initialized-model shape and mechanism diagnostics."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from klab.config.models import RunConfig
from klab.data.tokenizer import load_tokenizer
from klab.evaluation.generation import _prompt_byte_addresses
from klab.model.attention.sparse import BlockSparseAttention
from klab.model.moe import TopKMoE
from klab.model.transformer import DenseLM
from klab.training.manifest import config_sha256, sha256_file, source_identity


def _tensor(value: object) -> Tensor | None:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if isinstance(item, Tensor)), None)
    return None


def _shape(tensor: Tensor | None) -> list[int] | None:
    return list(tensor.shape) if tensor is not None else None


def _sample(
    value: Tensor | None, limit: int = 8, *, position_axis: int = 0
) -> dict[str, object] | None:
    if value is None:
        return None
    detached = value.detach().cpu()
    if detached.ndim == 0:
        sampled = detached.reshape(1)[:limit]
        truncated = False
    else:
        axis = position_axis % detached.ndim
        slices = [slice(None)] * detached.ndim
        slices[axis] = slice(0, limit)
        sampled = detached[tuple(slices)]
        truncated = detached.shape[axis] > limit
    return {
        "shape": list(value.shape),
        "sample_shape": list(sampled.shape),
        "values": sampled.tolist(),
        "truncated": truncated,
    }


def _scalar(value: Tensor) -> int | float:
    item = value.detach().cpu().item()
    return int(item) if isinstance(item, int) else float(item)


def probe_model(config: RunConfig, prompt: str) -> dict[str, object]:
    """Run one CPU FP32 forward from fresh initialization, retaining no model state."""
    if config.runtime.engine == "mlx":
        raise ValueError(
            "learn probe does not convert MLX configurations; use the native MLX train/stage commands"
        )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("probe prompt must be a nonempty string")
    tokenizer_path = Path(config.tokenizer.path)
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if vocab_size != config.model.vocab_size:
        raise ValueError(
            f"tokenizer vocabulary {vocab_size} does not match model vocabulary {config.model.vocab_size}"
        )
    encoding = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = list(encoding.ids)
    if not input_ids:
        raise ValueError("probe prompt encodes to no tokens")
    if len(input_ids) > 128:
        raise ValueError(
            "probe prompt exceeds the 128-token bound; no truncation is performed"
        )
    if len(input_ids) > config.model.max_seq_len:
        raise ValueError("probe prompt exceeds model.max_seq_len")
    if any(
        token_id < 0 or token_id >= config.model.vocab_size for token_id in input_ids
    ):
        raise ValueError(
            "probe tokenizer emitted an ID outside the configured vocabulary"
        )

    memory_address_samples: dict[str, dict[str, object]] = {}
    byte_addresses: Tensor | None = None
    if config.model.memory in {"byte", "portable"}:
        addresses = _prompt_byte_addresses(
            tokenizer,
            prompt,
            input_ids,
            config.model.memory_table_size,
            config.model.memory_ngram_size,
        )
        byte_addresses = torch.tensor([addresses], dtype=torch.long, device="cpu")
        byte_samples = _sample(byte_addresses, position_axis=1)
        if byte_samples is not None:
            memory_address_samples["input.byte_addresses"] = byte_samples

    hooks: list[torch.utils.hooks.RemovableHandle] = []
    layers: list[dict[str, object]] = []
    model: DenseLM | None = None
    logits: Tensor | None = None
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            model = DenseLM(config.model, config.attention).to(
                device="cpu", dtype=torch.float32
            )
            model.eval()

            for name, module in model.named_modules():
                if not name or not isinstance(module, (nn.Embedding, nn.Linear)):
                    continue

                def record(
                    current: nn.Module,
                    inputs: tuple[object, ...],
                    output: object,
                    *,
                    layer_name: str = name,
                ) -> None:
                    input_tensors = [
                        value for value in inputs if isinstance(value, Tensor)
                    ]
                    result = _tensor(output)
                    parameter = getattr(current, "weight", None)
                    record_value: dict[str, object] = {
                        "name": layer_name,
                        "kind": type(current).__name__,
                        "input_shapes": [list(value.shape) for value in input_tensors],
                        "output_shape": _shape(result),
                        "input_dtypes": [str(value.dtype) for value in input_tensors],
                        "output_dtype": str(result.dtype)
                        if result is not None
                        else None,
                        "trainable": bool(
                            isinstance(parameter, Tensor) and parameter.requires_grad
                        ),
                    }
                    layers.append(record_value)
                    if layer_name.endswith(".table") or ".extra_tables." in layer_name:
                        source = input_tensors[0] if input_tensors else None
                        if source is not None:
                            memory_address_samples[layer_name] = {
                                "shape": list(source.shape),
                                "values": source.detach()
                                .cpu()
                                .reshape(-1)[:8]
                                .tolist(),
                                "truncated": source.numel() > 8,
                            }

                hooks.append(module.register_forward_hook(record))

            ids_tensor = torch.tensor([input_ids], dtype=torch.long, device="cpu")
            with torch.inference_mode():
                logits, _ = model.forward_with_aux(
                    ids_tensor,
                    byte_addresses=byte_addresses,
                    diagnostics="full",
                )
            diagnostic_values = {
                name: _scalar(value)
                for name, value in model.architecture_metric_tensors().items()
            }
            routing: list[dict[str, object]] = []
            sparse: list[dict[str, object]] = []
            for index, block in enumerate(model.blocks):
                if (
                    isinstance(block.ffn, TopKMoE)
                    and block.ffn.last_diagnostics is not None
                ):
                    diagnostic = block.ffn.last_diagnostics
                    routing.append(
                        {
                            "layer": index,
                            "selected_experts": _sample(diagnostic.selected_experts),
                            "selected_weights": _sample(diagnostic.selected_weights),
                            "experts_per_token": block.ffn.experts_per_token,
                        }
                    )
                if isinstance(block.attention, BlockSparseAttention):
                    diagnostic = block.attention.last_diagnostics
                    if diagnostic is not None:
                        sparse.append(
                            {
                                "layer": index,
                                "selected_blocks": _sample(diagnostic.selected_blocks),
                                "available_tokens": _scalar(
                                    diagnostic.available_tokens
                                ),
                                "selected_tokens": _scalar(diagnostic.selected_tokens),
                                "selection_ratio": _scalar(diagnostic.selection_ratio),
                            }
                        )
    finally:
        for hook in hooks:
            hook.remove()
        # No model, parameters, hook closures, or forward tensors escape this call.
        del model

    assert logits is not None
    cfg_payload = config.model_dump(mode="json")
    identity = source_identity()
    return {
        "format": "klab-mechanism-probe",
        "version": 1,
        "config_sha256": config_sha256(cfg_payload),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "source_identity": {
            "algorithm": identity["algorithm"],
            "sha256": identity["sha256"],
        },
        "initialization_seed": config.seed,
        "execution": "cpu-fp32-reference",
        "configured_runtime": {
            "engine": config.runtime.engine,
            "backend": config.runtime.backend,
            "precision": config.runtime.precision,
            "device_index": config.runtime.device_index,
        },
        "prompt_tokens": len(input_ids),
        "input_shape": [1, len(input_ids)],
        "logits_shape": list(logits.shape),
        "layers": layers,
        "diagnostics": diagnostic_values,
        "memory_lookup_addresses": memory_address_samples,
        "routing_samples": routing,
        "sparse_samples": sparse,
        "evidence_scope": "initialized_mechanism_only",
    }
