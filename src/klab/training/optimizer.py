from __future__ import annotations

import math
import re

import torch
from torch import nn

from klab.config.models import AdamWConfig, RunConfig


def learning_rate_for_step(
    step: int,
    max_steps: int,
    warmup_steps: int,
    peak: float,
    floor: float,
    *,
    decay_steps: int | None = None,
    stable_steps: int = 0,
) -> float:
    if decay_steps is not None:
        max_steps = decay_steps
        step = min(step, decay_steps)
    if warmup_steps and step <= warmup_steps:
        return peak * step / warmup_steps
    if stable_steps:
        decay_start = warmup_steps + stable_steps
        if not 0 <= warmup_steps < decay_start < max_steps:
            raise ValueError("stable phase must leave room for decay")
        if step <= decay_start:
            return peak
        fraction = (step - decay_start) / (max_steps - decay_start)
        return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * fraction))
    fraction = (
        ((step - warmup_steps) / (max_steps - warmup_steps))
        if warmup_steps
        else ((step - 1) / max(1, max_steps - 1))
    )
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * fraction))


def schedule_payload(config: RunConfig, step: int) -> dict[str, object]:
    """Bind an update count to the exact configured schedule, including its horizon."""
    optimizer = config.optimizer
    payload: dict[str, object] = {
        "kind": "warmup_cosine_v1",
        "completed_updates": step,
        "max_steps": config.training.max_steps,
        "warmup_steps": optimizer.warmup_steps,
        "peak": optimizer.peak,
        "floor": optimizer.floor,
    }
    if isinstance(optimizer, AdamWConfig) and optimizer.decay_steps is not None:
        payload["kind"] = "warmup_cosine_floor_v1"
        payload["decay_steps"] = optimizer.decay_steps
    if config.training.stable_steps:
        payload["kind"] = "warmup_stable_cosine_v1"
        payload["stable_steps"] = config.training.stable_steps
    return payload


def is_ngram_table_parameter(name: str) -> bool:
    """Exact canonical names of token-ngram tables, including additional ports."""
    return (
        re.fullmatch(
            r"(?:memory|extra_memories\.[0-9]+)\.(?:table|extra_tables\.[0-9]+)\.weight",
            name,
        )
        is not None
    )


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer, base_rate: float
) -> None:
    """Apply the schedule while retaining each declared group's relative rate."""
    for group in optimizer.param_groups:
        group["lr"] = base_rate * group.get("lr_multiplier", 1.0)


def make_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    *,
    memory_table_lr_multiplier: float = 1.0,
    memory_table_weight_decay: float | None = None,
) -> torch.optim.AdamW:
    from klab.model.memory import TokenNgramMemory

    if not math.isfinite(memory_table_lr_multiplier) or memory_table_lr_multiplier <= 0:
        raise ValueError(
            "memory table learning-rate multiplier must be finite and positive"
        )
    if memory_table_weight_decay is not None and (
        not math.isfinite(memory_table_weight_decay) or memory_table_weight_decay < 0
    ):
        raise ValueError("memory table weight decay must be finite and nonnegative")
    table_ids = {
        id(table.weight)
        for module in model.modules()
        if isinstance(module, TokenNgramMemory)
        for table in (module.table, *module.extra_tables)
    }
    custom_tables = (
        memory_table_lr_multiplier != 1.0 or memory_table_weight_decay is not None
    )
    if custom_tables and not table_ids:
        raise ValueError("memory table optimizer policy requires token-ngram tables")
    decay, no_decay, tables, seen = [], [], [], set()
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if custom_tables and id(parameter) in table_ids:
            tables.append(parameter)
        else:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if tables:
        groups.append(
            {
                "params": tables,
                "weight_decay": weight_decay
                if memory_table_weight_decay is None
                else memory_table_weight_decay,
                "lr_multiplier": memory_table_lr_multiplier,
                "lr": learning_rate * memory_table_lr_multiplier,
            }
        )
    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        foreach=False,
    )


# Parameter bytes whose optimizer state is on the device at once during an
# offloaded step. A larger tensor is streamed alone.
STATE_OFFLOAD_BUCKET_BYTES = 128 * 1024 * 1024


def offloaded_state_device_bytes(state_bytes: int, largest_parameter_bytes: int) -> int:
    """Upper bound on optimizer state resident on the device during one bucket.

    AdamW and Adafactor keep at most two FP32 slots the size of a parameter, so
    one bucket holds at most twice its parameter bytes.
    """
    return min(
        state_bytes, 2 * max(STATE_OFFLOAD_BUCKET_BYTES, largest_parameter_bytes)
    )


class OptimizerStateOffload:
    """Keep optimizer state in host memory and stream it per bucket for a step.

    Between steps every non-scalar state tensor lives in host memory (pinned on
    CUDA and XPU). ``step`` replaces the optimizer's own: for each bucket of
    parameters it copies their state to the device, runs the unchanged PyTorch
    update on that bucket alone and copies the state back into the same host
    buffers. Scalar step counters stay where PyTorch keeps them. The update is
    the optimizer's own single-tensor arithmetic, so it matches a resident-state
    run; only where the state waits between steps changes.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        if any(
            group.get("foreach") or group.get("fused")
            for group in optimizer.param_groups
        ):
            raise ValueError("optimizer state offload needs single-tensor updates")
        self.optimizer = optimizer
        self.device = device
        self.pinned = device.type in {"cuda", "xpu"}
        self._resident_step = optimizer.step
        # GradScaler and the engine both call ``optimizer.step``.
        optimizer.step = self.step  # type: ignore[method-assign]
        self.to_host()

    def _host(self, value: torch.Tensor) -> torch.Tensor:
        host = torch.empty(
            value.shape, dtype=value.dtype, device="cpu", pin_memory=self.pinned
        )
        host.copy_(value)
        return host

    def to_host(self) -> None:
        """Move any device-resident state, for example after a checkpoint load."""
        for values in self.optimizer.state.values():
            for key, value in values.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim
                    and (
                        value.device.type != "cpu"
                        or (self.pinned and not value.is_pinned())
                    )
                ):
                    values[key] = self._host(value)

    def _buckets(
        self, groups: list[list[torch.nn.Parameter]]
    ) -> list[tuple[int, list[torch.nn.Parameter]]]:
        buckets: list[tuple[int, list[torch.nn.Parameter]]] = []
        for index, parameters in enumerate(groups):
            bucket: list[torch.nn.Parameter] = []
            size = 0
            for parameter in parameters:
                nbytes = parameter.numel() * parameter.element_size()
                if bucket and size + nbytes > STATE_OFFLOAD_BUCKET_BYTES:
                    buckets.append((index, bucket))
                    bucket, size = [], 0
                bucket.append(parameter)
                size += nbytes
            if bucket:
                buckets.append((index, bucket))
        return buckets

    def step(self, closure: object = None) -> None:
        if closure is not None:
            raise ValueError("optimizer state offload does not take a closure")
        state = self.optimizer.state
        groups = self.optimizer.param_groups
        saved = [group["params"] for group in groups]
        try:
            for group in groups:
                group["params"] = []
            for index, bucket in self._buckets(saved):
                hosts: dict[tuple[int, str], torch.Tensor] = {}
                for parameter in bucket:
                    values = state.get(parameter, {})
                    for key, value in values.items():
                        if isinstance(value, torch.Tensor) and value.ndim:
                            hosts[(id(parameter), key)] = value
                            values[key] = value.to(
                                self.device, non_blocking=self.pinned
                            )
                groups[index]["params"] = bucket
                self._resident_step()
                groups[index]["params"] = []
                for parameter in bucket:
                    values = state[parameter]
                    for key, value in values.items():
                        if not isinstance(value, torch.Tensor) or not value.ndim:
                            continue
                        host = hosts.get((id(parameter), key))
                        if host is None:
                            # State created by this first step gets its buffer.
                            host = torch.empty(
                                value.shape,
                                dtype=value.dtype,
                                device="cpu",
                                pin_memory=self.pinned,
                            )
                        # Stream-ordered: the next read of ``host`` is either a
                        # copy on the same stream or follows a device sync.
                        host.copy_(value, non_blocking=self.pinned)
                        values[key] = host
        finally:
            for group, parameters in zip(groups, saved, strict=True):
                group["params"] = parameters


def make_adafactor(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    beta2_decay: float,
    eps: tuple[float | None, float],
    d: float,
) -> torch.optim.Adafactor:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    return torch.optim.Adafactor(
        parameters,
        lr=learning_rate,
        beta2_decay=beta2_decay,
        eps=eps,
        d=d,
        weight_decay=weight_decay,
        foreach=False,
    )
