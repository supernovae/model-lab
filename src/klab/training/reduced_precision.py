"""bf16 gradients and bf16 AdamW moments around FP32 master weights.

Parameters stay FP32. Every update is computed in FP32; only what waits in
memory between operations is narrowed, and it is narrowed by stochastic
rounding so the stored value is unbiased: a step smaller than bf16's spacing
still moves the stored value on average, where round-to-nearest would drop it
every time. The random bits come from a generator seeded by the run seed and
the update index, so a resumed run rounds exactly as an uninterrupted one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from klab.config.models import RunConfig


@dataclass(frozen=True)
class ReducedPrecision:
    """The run's narrowed storage: what the optimizer keeps and how it rounds."""

    state_dtype: torch.dtype
    seed: int

    @classmethod
    def from_config(cls, config: RunConfig) -> ReducedPrecision | None:
        """The reduced-precision settings, or ``None`` for the FP32 reference path."""
        state = config.optimizer_state_dtype
        if state == "fp32" and config.runtime.gradient_dtype == "fp32":
            return None
        return cls(torch.bfloat16 if state == "bf16" else torch.float32, config.seed)


def stochastic_round_bf16(
    value: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Round an FP32 tensor to bf16, up with probability equal to the remainder.

    A float32 is a bfloat16 followed by sixteen more mantissa bits. Adding a
    uniform sixteen-bit integer to the bit pattern and clearing those bits
    carries into the bfloat16 part with exactly the probability of the dropped
    fraction, for either sign (the sign is a separate bit).
    """
    bits = value.contiguous().view(torch.int32)
    noise = torch.randint(
        0,
        1 << 16,
        bits.shape,
        dtype=torch.int32,
        device=bits.device,
        generator=generator,
    )
    rounded = (bits + noise) & -(1 << 16)
    return rounded.view(torch.float32).to(torch.bfloat16)


def seeded_generator(
    device: torch.device, seed: int, index: int, stream: int = 0
) -> torch.Generator:
    """A generator fixed by the run seed, the update ``index`` and a ``stream``."""
    generator = torch.Generator(device)
    generator.manual_seed(((seed * 1_000_003 + index) * 1_000_033 + stream) % (1 << 63))
    return generator


class ReducedPrecisionAdamW(torch.optim.AdamW):
    """AdamW with the arithmetic of ``torch.optim.AdamW`` and narrower storage.

    ``state_dtype`` is the dtype the two moments are kept in between steps. The
    gradient may be bf16 or FP32. Each parameter is updated in FP32 from its
    moments and gradient, then the moments are written back, stochastically
    rounded when ``state_dtype`` is bf16. With FP32 state and FP32 gradients
    the result equals ``torch.optim.AdamW``.
    """

    def __init__(
        self,
        params: Iterable[object],
        *,
        state_dtype: torch.dtype,
        seed: int,
        **options: object,
    ) -> None:
        if state_dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError("optimizer state dtype must be float32 or bfloat16")
        super().__init__(params, foreach=False, **options)  # type: ignore[arg-type]
        self.state_dtype = state_dtype
        self.seed = seed
        # Each parameter rounds with its own stream, so the result does not
        # depend on the order or the buckets an offloaded step visits it in.
        self._stream = {
            parameter: stream
            for stream, parameter in enumerate(
                parameter
                for group in self.param_groups
                for parameter in group["params"]
            )
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:  # type: ignore[override]
        super().load_state_dict(state_dict)  # type: ignore[arg-type]
        # PyTorch casts floating state to the parameter's dtype on load.
        for values in self.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in values and values[key].dtype != self.state_dtype:
                    values[key] = values[key].to(self.state_dtype)

    @torch.no_grad()
    def step(self, closure: object = None) -> None:  # type: ignore[override]
        if closure is not None:
            raise ValueError("ReducedPrecisionAdamW does not take a closure")
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["step"] = torch.tensor(0.0)
                    for key in ("exp_avg", "exp_avg_sq"):
                        state[key] = torch.zeros_like(parameter, dtype=self.state_dtype)
                state["step"] += 1
                count = int(state["step"].item())
                gradient = parameter.grad.float()
                first, second = state["exp_avg"], state["exp_avg_sq"]
                narrow = self.state_dtype != torch.float32
                first32 = first.float() if narrow else first
                second32 = second.float() if narrow else second
                first32.lerp_(gradient, 1 - beta1)
                second32.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                denominator = (second32.sqrt() / math.sqrt(1 - beta2**count)).add_(
                    group["eps"]
                )
                parameter.addcdiv_(
                    first32, denominator, value=-group["lr"] / (1 - beta1**count)
                )
                if narrow:
                    generator = seeded_generator(
                        parameter.device, self.seed, count, self._stream[parameter]
                    )
                    first.copy_(stochastic_round_bf16(first32, generator))
                    second.copy_(stochastic_round_bf16(second32, generator))


class Bf16GradientBuffers:
    """Hold each parameter's gradient in bf16 between microbatches.

    Autograd still produces an FP32 gradient per parameter, but it lives only
    until a hook folds it into a persistent bf16 buffer with stochastic
    rounding and drops it, so at most one parameter's FP32 gradient exists at a
    time. ``publish`` hands the buffers to the optimizer as ``parameter.grad``.
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], seed: int) -> None:
        self.seed = seed
        self._buffers: dict[torch.nn.Parameter, torch.Tensor] = {}
        self._filled: set[torch.nn.Parameter] = set()
        self._generator: torch.Generator | None = None
        for parameter in parameters:
            if parameter.requires_grad:
                self._buffers[parameter] = torch.zeros_like(
                    parameter, dtype=torch.bfloat16
                )
                # The hook consumes autograd's FP32 gradient; the bf16 buffer is
                # published as ``grad`` afterwards, which grad_dtype must allow.
                parameter.grad_dtype = None
                parameter.register_post_accumulate_grad_hook(self._fold)
        self._device = next(iter(self._buffers.values())).device

    def begin_update(self, update_index: int) -> None:
        """Forget the previous update's gradients and seed this update's rounding."""
        self._filled.clear()
        self._generator = seeded_generator(self._device, self.seed, update_index)

    def _fold(self, parameter: torch.nn.Parameter) -> None:
        assert self._generator is not None, "begin_update must precede backward"
        gradient, parameter.grad = parameter.grad, None
        assert gradient is not None
        buffer = self._buffers[parameter]
        total = gradient.float()
        if parameter in self._filled:
            total = total + buffer.float()
        self._filled.add(parameter)
        buffer.copy_(stochastic_round_bf16(total, self._generator))

    def publish(self) -> None:
        """Expose the accumulated bf16 gradients as ``parameter.grad``."""
        for parameter in self._filled:
            parameter.grad = self._buffers[parameter]

    def clip_(self, max_norm: float, update_index: int) -> torch.Tensor:
        """Clip the published gradients to ``max_norm`` and return the FP32 norm.

        The norm is taken in FP32, and the scaling is rounded stochastically,
        so a bf16 gradient is clipped as accurately as it is accumulated.
        """
        # In parameter order: the rounding stream is consumed in this order.
        buffers = [
            buffer
            for parameter, buffer in self._buffers.items()
            if parameter in self._filled
        ]
        if not buffers:
            return torch.zeros(())
        norm = torch.linalg.vector_norm(
            torch.stack(
                [torch.linalg.vector_norm(b, dtype=torch.float32) for b in buffers]
            )
        )
        coefficient = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
        if float(coefficient) >= 1.0:
            return norm
        generator = seeded_generator(self._device, self.seed, ~update_index)
        for buffer in buffers:
            buffer.copy_(stochastic_round_bf16(buffer.float() * coefficient, generator))
        return norm
