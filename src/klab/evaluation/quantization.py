"""Weight-only quantization of a verified checkpoint: int8 and 4-bit.

Schemes (both symmetric, no zero point, weights only; activations stay fp32):

* ``int8``: one scale per output channel, ``q = round(w / s)`` in [-127, 127];
* ``int4``: one scale per group of ``group_size`` input columns of each output
  row, ``q`` in [-7, 7], two values packed per byte.

Only the ``nn.Linear`` projections inside decoder blocks are quantized
(attention and feed-forward, including MoE experts). Token embeddings, the
output head (often tied to the embeddings), norms, MoE routers and memory
modules keep their checkpoint weights. A :class:`QuantLinear` stores the
integer weights and scales and dequantizes on the fly in ``forward`` (and when
``.weight`` is read), so the model runs unchanged code. This saves resident
weight memory; it is not faster than fp32.

An artifact (``klab weights quantize``) is a directory with
``quantized.safetensors`` and a sealed ``klab_quantized.json``: the source run
and checkpoint digest, the scheme, the quantized modules, the file digest and
the held-out loss of the source and the quantized model on the run's
validation split under the same evaluation settings, with their difference.
``load_run(..., quantized=DIR)`` applies it only to the checkpoint it was made
from (``generate``, ``chat`` and ``serve`` take ``--quantized DIR``).

No quantization library is in the locked environment (torchao, bitsandbytes,
optimum-quanto, llm-compressor); ``torch.ao`` dynamic quantization quantizes
activations too and has no 4-bit path, so this minimal implementation is used.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from klab.lab_records import is_sealed, seal
from klab.training.manifest import sha256_file

QUANTIZED_FORMAT = "klab-quantized-v1"
MANIFEST = "klab_quantized.json"
WEIGHTS = "quantized.safetensors"
SCOPE = (
    "nn.Linear inside decoder blocks (attention, feed-forward, experts); "
    "not embeddings, output head, norms, routers or memory modules"
)


def _scheme(bits: int, group_size: int | None) -> dict[str, Any]:
    if bits == 8:
        if group_size is not None:
            raise ValueError("int8 uses one scale per output channel: no group size")
        return {"name": "int8", "bits": 8, "granularity": "per_output_channel"}
    if bits == 4:
        size = 128 if group_size is None else group_size
        if size < 2 or size % 2:
            raise ValueError("int4 group size must be an even integer >= 2")
        return {
            "name": "int4",
            "bits": 4,
            "granularity": "group",
            "group_size": size,
        }
    raise ValueError("--bits must be 8 or 4")


def quantize_weight(weight: torch.Tensor, scheme: Mapping[str, Any]) -> dict:
    """Integer codes and fp32 scales of one 2-D weight under SCHEME."""
    w = weight.detach().to(torch.float32).cpu()
    out_features, in_features = w.shape
    if scheme["bits"] == 8:
        scale = w.abs().amax(dim=1) / 127.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
        return {"qweight": q, "scale": scale}
    size = scheme["group_size"]
    if in_features % size:
        raise ValueError(
            f"int4 group size {size} does not divide {in_features} input "
            "features; pass a --group-size that divides every projection"
        )
    groups = w.reshape(out_features, in_features // size, size)
    scale = groups.abs().amax(dim=2) / 7.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(groups / scale[..., None]).clamp(-7, 7).to(torch.int16) + 8
    q = q.reshape(out_features, in_features).to(torch.uint8)
    packed = q[:, 0::2] | (q[:, 1::2] << 4)
    return {"qweight": packed, "scale": scale}


def dequantize(
    qweight: torch.Tensor, scale: torch.Tensor, scheme: Mapping[str, Any]
) -> torch.Tensor:
    if scheme["bits"] == 8:
        return qweight.to(scale.dtype) * scale[:, None]
    low = (qweight & 0x0F).to(torch.int16) - 8
    high = (qweight >> 4).to(torch.int16) - 8
    q = torch.stack((low, high), dim=2).reshape(qweight.shape[0], -1)
    size = scheme["group_size"]
    groups = q.reshape(q.shape[0], -1, size).to(scale.dtype) * scale[..., None]
    return groups.reshape(q.shape[0], -1)


class QuantLinear(nn.Module):
    """A Linear whose weight is stored as integer codes plus scales."""

    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        scheme: Mapping[str, Any],
        bias: torch.Tensor | None,
        in_features: int,
    ) -> None:
        super().__init__()
        self.scheme = dict(scheme)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.out_features = int(qweight.shape[0])
        self.in_features = in_features

    @property
    def weight(self) -> torch.Tensor:  # type: ignore[override]
        return dequantize(self.qweight, self.scale, self.scheme)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight.to(x.dtype), self.bias)


def quantizable(model: nn.Module) -> dict[str, nn.Linear]:
    """The block projections the scheme covers (see SCOPE), by module name."""
    out = {}
    for name, module in model.named_modules():
        parts = name.split(".")
        if (
            isinstance(module, nn.Linear)
            and len(parts) >= 4
            and parts[0] == "blocks"
            and parts[2] in {"attention", "ffn"}
            and parts[-1] != "router"
        ):
            out[name] = module
    return out


def quantize_tensors(
    model: nn.Module, scheme: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    tensors = {}
    for name, module in quantizable(model).items():
        for key, value in quantize_weight(module.weight, scheme).items():
            tensors[f"{name}.{key}"] = value.contiguous()
    if not tensors:
        raise ValueError("this model has no block projections to quantize")
    return tensors


def _replace(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent_name), child, module)


def read_artifact(directory: Path) -> dict[str, Any]:
    """The sealed manifest of a quantized artifact, with its weight file verified."""
    path = Path(directory) / MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"not a quantized artifact: {directory} ({error})") from None
    if not isinstance(manifest, dict) or manifest.get("format") != QUANTIZED_FORMAT:
        raise ValueError(f"not a {QUANTIZED_FORMAT} artifact: {directory}")
    if not is_sealed(manifest):
        raise ValueError(f"quantized artifact manifest digest mismatch: {path}")
    if sha256_file(Path(directory) / WEIGHTS) != manifest["files"][WEIGHTS]:
        raise ValueError(f"quantized weights differ from their manifest: {directory}")
    return manifest


def apply_artifact(
    model: nn.Module, directory: Path, *, checkpoint_sha256: str
) -> dict[str, Any]:
    """Swap the artifact's QuantLinear modules into MODEL (refusing mismatches)."""
    from safetensors.torch import load_file

    manifest = read_artifact(directory)
    source = manifest["source"]
    if source["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(
            "quantized artifact was made from checkpoint "
            f"{source['checkpoint_sha256'][:12]} ({source['checkpoint']}), not "
            f"the selected {checkpoint_sha256[:12]}: pass --checkpoint "
            f"{source['checkpoint']}"
        )
    scheme = manifest["scheme"]
    tensors = load_file(str(Path(directory) / WEIGHTS))
    targets = quantizable(model)
    if sorted(targets) != manifest["modules"]:
        raise ValueError("quantized artifact modules differ from this model's")
    for name, linear in targets.items():
        qweight, scale = tensors[f"{name}.qweight"], tensors[f"{name}.scale"]
        restored = dequantize(qweight, scale, scheme)
        if restored.shape != linear.weight.shape:
            raise ValueError(f"quantized {name} has the wrong shape")
        bias = None if linear.bias is None else linear.bias.detach().clone()
        _replace(
            model,
            name,
            QuantLinear(qweight, scale, scheme, bias, linear.in_features),
        )
    return {
        "artifact": str(Path(directory).resolve()),
        "manifest_sha256": manifest["record_sha256"],
        "scheme": scheme,
        "modules": len(targets),
        "heldout": manifest["heldout"],
    }


def _weight_bytes(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in [*model.parameters(), *model.buffers()]:
        if tensor.data_ptr() in seen:
            continue
        seen.add(tensor.data_ptr())
        total += tensor.numel() * tensor.element_size()
    return total


def _heldout(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("loss", "perplexity", "tokens", "batches")
    return {k: result.get(k) for k in keys if k in result}


def quantize_run(
    run_id: str,
    runs_dir: Path,
    output: Path,
    *,
    bits: int,
    group_size: int | None = None,
    checkpoint: str | None = None,
    backend: str | None = "cpu",
    authorization: Any = None,
) -> dict[str, Any]:
    """Quantize a verified checkpoint and score both on its validation split."""
    from safetensors.torch import save_file

    from klab.evaluation.inference import load_run

    scheme = _scheme(bits, group_size)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite quantized output: {output}")
    loaded = load_run(
        run_id, runs_dir, checkpoint, backend, authorization=authorization
    )
    if loaded.engine is not None:
        raise ValueError("weight quantization needs the PyTorch engine (not MLX)")
    if getattr(loaded.model, "semantic_memories", None):
        raise ValueError("weight quantization does not support attached semantic packs")
    source_eval = loaded.evaluate()
    source_bytes = _weight_bytes(loaded.model)
    tensors = quantize_tensors(loaded.model, scheme)
    identity = loaded.identity
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}.quant.", dir=output.parent))
    try:
        save_file(tensors, staged / WEIGHTS, metadata={"format": QUANTIZED_FORMAT})
        settings = {
            "split": "validation",
            "validation_sha256": identity["data_sha256"]["validation"],
            "seq_len": loaded.config.training.seq_len,
            "batch_size": loaded.config.evaluation.batch_size
            or loaded.config.training.micro_batch_size,
            "max_batches": loaded.config.evaluation.max_batches,
            "runtime": identity["runtime"],
        }
        manifest: dict[str, Any] = {
            "format": QUANTIZED_FORMAT,
            "created_at": datetime.now(UTC).isoformat(),
            "source": {
                "run_id": identity["run_id"],
                "checkpoint": identity["checkpoint_relative_path"],
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "step": identity["step"],
                "tokenizer_sha256": identity["tokenizer_sha256"],
            },
            "scheme": scheme,
            "scope": SCOPE,
            "modules": sorted(quantizable(loaded.model)),
            "files": {WEIGHTS: sha256_file(staged / WEIGHTS)},
            # Filled after the quantized model is scored from these files.
            "heldout": None,
        }
        draft = staged / MANIFEST
        draft.write_text(json.dumps(seal(manifest), sort_keys=True), encoding="utf-8")
        # Score the quantized model through the same path inference uses.
        apply_artifact(
            loaded.model, staged, checkpoint_sha256=identity["checkpoint_sha256"]
        )
        loaded.model.to(loaded.device).eval()
        quant_eval = loaded.evaluate()
        source_loss, quant_loss = source_eval["loss"], quant_eval["loss"]
        manifest["heldout"] = {
            "settings": settings,
            "source": {**_heldout(source_eval), "precision": "fp32"},
            "quantized": {**_heldout(quant_eval), "precision": scheme["name"]},
            "loss_delta": None
            if source_loss is None or quant_loss is None
            else quant_loss - source_loss,
            "weight_bytes": {
                "source": source_bytes,
                "quantized": _weight_bytes(loaded.model),
            },
        }
        draft.write_text(
            json.dumps(seal(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged.replace(output)
    finally:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
    return {**read_artifact(output), "output": str(output)}
