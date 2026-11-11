"""One-shot PyTorch precision probe used by :mod:`klab.runtime`.

This module deliberately has no project configuration imports.  It is launched in a
separate interpreter so failed accelerator initialization and allocator state cannot
leak into the training process.
"""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from typing import Any

import numpy as np


def _device(backend: str, index: int):
    import torch

    if backend in {"cuda", "rocm"}:
        return torch.device("cuda", index)
    if backend == "xpu":
        return torch.device("xpu", index)
    return torch.device(backend)


def _autocast(torch: Any, device: Any, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _scaler(torch: Any, device: Any, precision: str):
    if precision != "fp16":
        return None
    # A scaler is required for fp16 execution.  CPU fp16 is intentionally rejected
    # by the parent before reaching this point.
    if device.type not in {"cuda", "xpu"}:
        raise RuntimeError(f"fp16 GradScaler is unsupported on {device.type}")
    return torch.amp.GradScaler(device.type, enabled=True)


def _mlx_probe(payload: dict[str, Any]) -> dict[str, Any]:
    """Exercise a fixed tiny MLX graph, never the requested experiment model."""

    import mlx
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx import nn

    from klab.runtime import (
        RuntimeInfo,
        _now,
        _os_identity,
        _ram,
        detect_host_environment,
    )

    required = {
        "format_version",
        "engine",
        "backend",
        "device_index",
        "precision",
        "optimizer",
        "checkpointing",
        "activation_offload",
        "attention",
    }
    if (
        set(payload) != required
        or payload["engine"] != "mlx"
        or payload["format_version"] != 1
    ):
        raise ValueError("unsupported MLX runtime probe request")
    if payload["backend"] != "metal" or payload["device_index"] != 0:
        raise ValueError("MLX requires Metal device_index=0")
    if payload["precision"] != "fp32" or payload["optimizer"] != "adamw":
        raise ValueError("MLX probe supports fp32 AdamW only")
    if payload["activation_offload"] is not False:
        raise ValueError("MLX activation offload is unsupported")
    if type(payload["checkpointing"]) is not bool or payload["attention"] not in {
        "dense",
        "block_sparse",
    }:
        raise ValueError("invalid MLX probe feature request")
    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal execution is unavailable")
    mx.set_default_device(mx.gpu)
    init_key, _next_key = mx.random.split(mx.random.key(0))
    mx.eval(init_key)
    words = np.asarray(init_key, dtype=np.uint32)
    mx.random.seed((int(words[0]) << 32) | int(words[1]))
    model = nn.Linear(4, 4, bias=False)
    forward = nn.utils.checkpoint(model) if payload["checkpointing"] else model
    value_and_grad = nn.value_and_grad(
        model, lambda inputs: mx.mean(forward(inputs) ** 2)
    )
    inputs = mx.full((1, 2, 4), 0.125, dtype=mx.float32)
    loss, gradients = value_and_grad(inputs)
    optimizer = optim.AdamW(learning_rate=0.01)
    optimizer.update(model, gradients)
    mx.eval(loss, gradients, model.parameters(), optimizer.state)
    if not bool(mx.all(mx.isfinite(loss))) or not bool(
        mx.all(mx.isfinite(model.weight))
    ):
        raise RuntimeError("MLX probe produced a non-finite update")
    features = ["forward_backward_optimizer", "optimizer:adamw"]
    if payload["attention"] == "block_sparse":
        # This fixed 8-wide graph validates the native sparse kernel and its
        # gradients without allocating the requested training model.
        from klab.model.attention.mlx_sparse import MLXBlockSparseAttention

        sparse = MLXBlockSparseAttention(8, 2, 8, 10_000.0, 2, 1)
        sparse_value_and_grad = nn.value_and_grad(
            sparse, lambda values: mx.mean(sparse(values) ** 2)
        )
        sparse_loss, sparse_gradients = sparse_value_and_grad(
            mx.full((1, 4, 8), 0.125, dtype=mx.float32)
        )
        optim.AdamW(learning_rate=0.01).update(sparse, sparse_gradients)
        mx.eval(sparse_loss, sparse_gradients, sparse.parameters())
        if not bool(mx.all(mx.isfinite(sparse_loss))):
            raise RuntimeError("MLX sparse probe produced a non-finite update")
        features.append("native_block_sparse_attention")
    if payload["checkpointing"]:
        features.append("activation_checkpointing")
    total, available = _ram()
    version = getattr(mlx, "__version__", None)
    info = RuntimeInfo(
        engine="mlx",
        backend="metal",
        torch_device=None,
        device_index=0,
        device_name="Apple Metal",
        physical_device_id=None,
        framework_version=version,
        runtime_version=version,
        driver_version=None,
        os=_os_identity(),
        system_total_bytes=total,
        system_available_bytes=available,
        device_total_bytes=None,
        device_free_bytes=None,
        device_recommended_bytes=None,
        measurement_source="mlx.get_active_memory/get_peak_memory/get_cache_memory",
        measured_at=_now(),
        precision_capabilities=("fp32",),
        limitations=(
            "Apple unified memory is one shared system-memory pool",
            "MLX does not expose a reserved-memory capacity counter",
            "Metal driver and physical-device identity are unavailable through MLX",
        ),
        device_driver_allocated_bytes=None,
        tested_precisions=("fp32",),
        tested_features=tuple(features),
        validated_at=_now(),
        **detect_host_environment(),
    )
    return {
        "format_version": 1,
        "ok": True,
        "tested_precision": "fp32",
        "tested_features": features,
        "runtime": info.as_dict(),
        "memory": {
            "active_bytes": int(mx.get_active_memory()),
            "peak_bytes": int(mx.get_peak_memory()),
            "cache_bytes": int(mx.get_cache_memory()),
        },
    }


def probe(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict) and payload.get("engine") == "mlx":
        return _mlx_probe(payload)
    from dataclasses import replace

    import torch
    from torch.utils.checkpoint import checkpoint

    from klab.runtime import _now, _safe_call, discover_runtimes

    if (
        not isinstance(payload, dict)
        or type(payload.get("format_version")) is not int
        or payload["format_version"] != 1
        or payload.get("engine") != "pytorch"
    ):
        raise ValueError("unsupported runtime probe request")
    expected = {
        "format_version",
        "engine",
        "backend",
        "device_index",
        "precision",
        "optimizer",
        "checkpointing",
        "activation_offload",
        "attention",
    }
    if set(payload) != expected or payload["attention"] not in {
        "dense",
        "block_sparse",
    }:
        raise ValueError("unsupported PyTorch runtime probe request")
    backend = payload["backend"]
    precision = payload["precision"]
    index = payload["device_index"]
    optimizer_name = payload["optimizer"]
    checkpointing = payload["checkpointing"]
    activation_offload = payload["activation_offload"]
    if type(activation_offload) is not bool:
        raise TypeError("invalid activation offload flag")
    if type(index) is not int or index < 0 or type(checkpointing) is not bool:
        raise ValueError("invalid probe device index or checkpointing flag")
    if optimizer_name not in {"adamw", "adafactor"}:
        raise ValueError("unsupported probe optimizer")
    device = _device(backend, index)
    parameter = torch.nn.Parameter(torch.full((4, 4), 0.125, device=device))
    before = parameter.detach().clone()
    optimizer_type = (
        torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adafactor
    )
    optimizer = optimizer_type([parameter], lr=0.01, foreach=False)
    scaler = _scaler(torch, device, precision)
    optimizer.zero_grad(set_to_none=True)
    with _autocast(torch, device, precision):
        result = (
            checkpoint(lambda value: value @ value, parameter, use_reentrant=False)
            if checkpointing
            else parameter @ parameter
        )
    expected_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[precision]
    if result.dtype != expected_dtype:
        raise RuntimeError("probe silently changed the requested compute precision")
    loss = result.float().square().mean()
    if not torch.isfinite(loss).item():
        raise RuntimeError("probe produced a nonfinite loss")
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    if (
        parameter.grad is None
        or parameter.grad.dtype != torch.float32
        or not torch.isfinite(parameter.grad).all().item()
    ):
        raise RuntimeError("probe did not produce finite fp32 master gradients")
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    if not torch.isfinite(parameter).all().item() or torch.equal(parameter, before):
        raise RuntimeError("probe did not complete a finite optimizer update")
    native_sparse = False
    if payload["attention"] == "block_sparse":
        from klab.model.attention.sparse import BlockSparseAttention

        sparse = BlockSparseAttention(8, 2, 8, 10_000.0, 2, 1).to(device)
        sparse_optimizer = optimizer_type(sparse.parameters(), lr=0.01, foreach=False)
        sparse_before = [
            parameter.detach().clone() for parameter in sparse.parameters()
        ]
        sparse_input = torch.full((1, 4, 8), 0.125, device=device, requires_grad=True)
        sparse_optimizer.zero_grad(set_to_none=True)
        with _autocast(torch, device, precision):
            sparse_output = (
                checkpoint(
                    lambda value: sparse(value, diagnostics="scalar"),
                    sparse_input,
                    use_reentrant=False,
                )
                if checkpointing
                else sparse(sparse_input, diagnostics="scalar")
            )
        sparse_loss = sparse_output.float().square().mean()
        sparse_loss.backward()
        if (
            not torch.isfinite(sparse_loss).item()
            or sparse_input.grad is None
            or not torch.isfinite(sparse_input.grad).all().item()
            or any(
                parameter.grad is None
                or not torch.isfinite(parameter.grad).all().item()
                for parameter in sparse.parameters()
            )
        ):
            raise RuntimeError("sparse attention probe produced nonfinite gradients")
        sparse_optimizer.step()
        if all(
            torch.equal(parameter, before)
            for parameter, before in zip(sparse.parameters(), sparse_before)
        ):
            raise RuntimeError("sparse attention probe did not update its parameters")
        if backend == "rocm":
            if not torch.version.hip or sparse.last_backend != "hip":
                raise RuntimeError(
                    "ROCm sparse probe did not execute the native HIP kernel"
                )
            native_sparse = True
    info = next(
        item
        for item in discover_runtimes(measurement_device=device)
        if item.engine == "pytorch" and item.backend == backend
    )
    features = ["forward_backward_optimizer", f"optimizer:{optimizer_name}"]
    if native_sparse:
        features.append("native_block_sparse_attention")
    if checkpointing:
        features.append("activation_checkpointing")
    if activation_offload:
        from klab.training.offload import probe_activation_offload

        with _autocast(torch, device, precision):
            offload_report = probe_activation_offload(device)
        if not offload_report.supported:
            raise RuntimeError(
                f"activation offload probe failed: {offload_report.reason}"
            )
        features.append("activation_offload")
    updates: dict[str, Any] = {
        "device_index": index,
        "torch_device": str(device),
        "tested_precisions": (precision,),
        "tested_features": tuple(features),
        "validated_at": _now(),
    }
    if backend in {"cuda", "rocm", "xpu"}:
        api = torch.cuda if backend in {"cuda", "rocm"} else torch.xpu
        properties, _ = _safe_call(lambda: api.get_device_properties(index))
        if properties is not None:
            updates["device_name"] = properties.name
            updates["device_total_bytes"] = int(properties.total_memory)
            uuid = getattr(properties, "uuid", None)
            updates["physical_device_id"] = str(uuid) if uuid else None
        if hasattr(api, "mem_get_info"):
            memory, _ = _safe_call(lambda: api.mem_get_info(index))
            if memory is not None:
                updates["device_free_bytes"] = int(memory[0])
                updates["measurement_source"] = f"torch.{device.type}.mem_get_info"
    return {
        "format_version": 1,
        "ok": True,
        "tested_precision": precision,
        "tested_features": features,
        "runtime": replace(info, **updates).as_dict(),
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read(16 * 1024).decode("utf-8"))
        result = probe(payload)
    except (
        RuntimeError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        MemoryError,
        AttributeError,
    ) as error:
        result = {
            "format_version": 1,
            "ok": False,
            "reason": f"{type(error).__name__}: {error}",
        }
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
