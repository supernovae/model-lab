"""Disposable evidence for the configured PyTorch SDPA execution path."""

from typing import Any

import torch

from klab.config.models import RunConfig
from klab.model.attention.dense import DenseAttention
from klab.runtime import (
    capture_rng_state,
    precision_context,
    restore_rng_state,
    seed_everything,
    torch_device_for,
    validate_runtime,
)
from klab.runtime_profile import RuntimeAuthorization, require_authorization

_ATTENTION_PROBE_VERSION = 1
_CANDIDATES = (
    ("MATH", "MATH"),
    ("EFFICIENT_ATTENTION", "EFFICIENT_ATTENTION"),
    ("FLASH_ATTENTION", "FLASH_ATTENTION"),
    ("CUDNN_ATTENTION", "CUDNN_ATTENTION"),
)


def _error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:1024]


def _operators(events: Any) -> list[str]:
    names = sorted(
        {
            str(event.key)
            for event in events
            if "scaled_dot_product" in str(event.key).lower()
        }
    )
    return names


def _selected_backend(operators: list[str]) -> str | None:
    joined = " ".join(operators).lower()
    if "flash" in joined:
        return "FLASH_ATTENTION"
    if "efficient" in joined:
        return "EFFICIENT_ATTENTION"
    if "cudnn" in joined:
        return "CUDNN_ATTENTION"
    if "math" in joined:
        return "MATH"
    return None


def _run_backward(
    attention: DenseAttention,
    values: torch.Tensor,
    *,
    config: RunConfig,
    device: torch.device,
) -> None:
    with torch.enable_grad():
        attention.zero_grad(set_to_none=True)
        inputs = values.detach().clone().requires_grad_(True)
        context = precision_context(config, device)
        with context:
            output = attention(inputs)
            loss = output.float().square().mean()
        loss.backward()
        if inputs.grad is None or any(
            parameter.grad is None for parameter in attention.parameters()
        ):
            raise RuntimeError("attention probe backward did not produce all gradients")


def _candidate_context(name: str) -> Any:
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except (ImportError, AttributeError) as error:
        raise RuntimeError("PyTorch SDPA backend controls are unavailable") from error
    backend = getattr(SDPBackend, name, None)
    if backend is None:
        raise RuntimeError(f"PyTorch does not expose SDPBackend.{name}")
    return sdpa_kernel(backend)


def _recommendations(
    runtime: Any, candidates: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    name = runtime.device_name or ""
    result: list[dict[str, object]] = []
    if "T4" in name:
        result.append(
            {
                "status": "RECOMMENDED",
                "message": "T4 workloads should use FP16 with GradScaler.",
            }
        )
        result.append(
            {
                "status": (
                    "OBSERVED"
                    if candidates["EFFICIENT_ATTENTION"]["status"] == "SUPPORTED"
                    else "UNAVAILABLE"
                ),
                "message": (
                    "Memory-efficient SDPA is suitable only when this probe "
                    "observed its backward path."
                ),
            }
        )
    if runtime.backend == "cuda":
        capability = torch.cuda.get_device_capability(runtime.device_index)
        if capability[0] >= 8:
            result.append(
                {
                    "status": (
                        "OBSERVED"
                        if candidates["FLASH_ATTENTION"]["status"] == "SUPPORTED"
                        else "UNAVAILABLE"
                    ),
                    "message": (
                        "Native SDPA flash support is reported only from this "
                        "forced backward probe."
                    ),
                }
            )
    if "H100" in name or "H800" in name:
        result.append(
            {
                "status": "NOT_INTEGRATED",
                "message": (
                    "FlashAttention-3 requires H100/H800 and CUDA >=12.3 "
                    "(upstream recommends 12.8): "
                    "https://github.com/Dao-AILab/flash-attention"
                ),
            }
        )
    return result


def _training_state() -> dict[str, object]:
    cudnn = torch.backends.cudnn
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "num_threads": torch.get_num_threads(),
        "cudnn_benchmark": cudnn.benchmark,
        "cudnn_deterministic": cudnn.deterministic,
    }


def _set_determinism(seed: int, deterministic: bool) -> None:
    """Mirror the training declaration for the isolated probe graph."""
    seed_everything(seed, deterministic_cpu=deterministic)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False


def _restore_training_state(state: dict[str, object]) -> None:
    torch.set_num_threads(int(state["num_threads"]))
    torch.use_deterministic_algorithms(
        bool(state["deterministic_algorithms"]),
        warn_only=bool(state["deterministic_warn_only"]),
    )
    torch.backends.cudnn.benchmark = bool(state["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(state["cudnn_deterministic"])


def attention_probe(
    config: RunConfig, *, authorization: RuntimeAuthorization | None = None
) -> dict[str, object]:
    """Run one disposable configured SDPA graph and forced backend candidates.

    The result is evidence only: each successful candidate executed both forward
    and backward under a one-candidate SDPA context.  It does not certify the
    unrestricted dispatch selected for training.
    """
    if config.runtime.engine != "pytorch":
        raise ValueError("attention probe requires the PyTorch engine")
    if config.attention.kind != "dense":
        raise ValueError("attention probe requires attention.kind=dense")
    if (
        config.model.num_kv_heads is not None
        and config.model.num_kv_heads != config.model.num_heads
    ):
        raise ValueError("attention probe requires equal Q/KV head counts")
    rng_state = capture_rng_state()
    training_state = _training_state()
    try:
        require_authorization(config, authorization)
        runtime = validate_runtime(config, authorization=authorization)
        device = torch_device_for(runtime.backend, runtime.device_index)
        _set_determinism(config.seed, config.training.deterministic)
        attention = DenseAttention(
            config.model.hidden_dim,
            config.model.num_heads,
            config.model.max_seq_len,
            config.attention.rope_base,
            implementation=config.attention.implementation,
        ).to(device)
        diagnostic_attention = (
            attention
            if config.attention.implementation == "sdpa"
            else DenseAttention(
                config.model.hidden_dim,
                config.model.num_heads,
                config.model.max_seq_len,
                config.attention.rope_base,
                implementation="sdpa",
            ).to(device)
        )
        if diagnostic_attention is not attention:
            diagnostic_attention.load_state_dict(attention.state_dict())
        values = torch.randn(
            config.training.micro_batch_size,
            config.training.seq_len,
            config.model.hidden_dim,
            device=device,
        )
        profiler = getattr(torch, "profiler", None)
        if profiler is None:
            _run_backward(attention, values, config=config, device=device)
            observed_operators: list[str] = []
        else:
            activity = [profiler.ProfilerActivity.CPU] + (
                [profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []
            )
            with profiler.profile(activities=activity) as recording:
                _run_backward(attention, values, config=config, device=device)
            observed_operators = _operators(recording.key_averages())
        candidates: dict[str, dict[str, object]] = {}
        for label, backend_name in _CANDIDATES:
            try:
                with _candidate_context(backend_name):
                    _run_backward(
                        diagnostic_attention, values, config=config, device=device
                    )
            except (RuntimeError, TypeError, ValueError) as error:
                candidates[label] = {"status": "UNAVAILABLE", "error": _error(error)}
            else:
                candidates[label] = {"status": "SUPPORTED", "error": None}
        return {
            "attention_probe_version": _ATTENTION_PROBE_VERSION,
            "runtime": runtime.as_dict(),
            "runtime_authorization": authorization.as_dict()
            if authorization is not None
            else None,
            "declared": {
                "implementation": config.attention.implementation,
                "micro_batch_size": config.training.micro_batch_size,
                "sequence_length": config.training.seq_len,
                "num_heads": config.model.num_heads,
                "head_dim": config.model.hidden_dim // config.model.num_heads,
                "precision": config.runtime.precision,
                "deterministic": config.training.deterministic,
            },
            "observed_operators": observed_operators,
            "selected_backend": _selected_backend(observed_operators),
            "candidates": candidates,
            "recommendations": _recommendations(runtime, candidates),
        }
    finally:
        restore_rng_state(rng_state)
        _restore_training_state(training_state)
