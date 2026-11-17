"""Disposable evidence for the configured PyTorch SDPA execution path."""

import math
from typing import Any

import torch
from torch.nn import functional

from klab.config.models import RunConfig
from klab.memory import assumes_fused_scores
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

_ATTENTION_PROBE_VERSION = 2
_CANDIDATES = (
    ("MATH", "MATH"),
    ("OVERRIDEABLE", "OVERRIDEABLE"),
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
    if "overrideable" in joined:
        return "OVERRIDEABLE"
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


# The probe's own lengths, plus the configured one. Linear growth in tokens
# means the kernel keeps no [T, T] scores; quadratic growth means it does.
_SCORE_LENGTHS = (1024, 2048, 4096)
# Above this exponent between the shortest and longest length, extra memory is
# growing with T^2 (2.0) rather than T (1.0).
_QUADRATIC_EXPONENT = 1.5
# A score matrix is never allocated if it would take more than this share of
# the device's free memory.
_SCORE_FREE_SHARE = 0.25


def retains_scores(measurements: list[dict[str, float]]) -> bool:
    """Whether peak extra memory scales with T^2, from at least two lengths.

    ``peak_extra_bytes`` is the largest allocation above the inputs during one
    forward and backward. A kernel that keeps scores grows four-fold when the
    length doubles; a fused one doubles.
    """
    if len(measurements) < 2:
        raise ValueError("score scaling needs at least two sequence lengths")
    short, long = measurements[0], measurements[-1]
    exponent = math.log(
        long["peak_extra_bytes"] / short["peak_extra_bytes"]
    ) / math.log(long["sequence_length"] / short["sequence_length"])
    return exponent > _QUADRATIC_EXPONENT


def _score_memory(config: RunConfig, device: torch.device) -> dict[str, object]:
    """Measure whether SDPA keeps a [T, T] score matrix for this attention shape.

    Runs the same causal ``scaled_dot_product_attention`` call training makes,
    forward and backward, at several lengths and records the peak device memory
    above the q, k, v tensors. Lengths whose score matrix alone would not fit
    comfortably are skipped, not run.
    """
    api = getattr(torch, device.type, None)
    if device.type not in {"cuda", "xpu"} or not hasattr(api, "max_memory_allocated"):
        return {
            "status": "UNMEASURED",
            "reason": f"{device.type} exposes no peak device memory counter",
        }
    model, training = config.model, config.training
    heads, head_dim = model.num_heads, model.hidden_dim // model.num_heads
    precision = config.runtime.precision
    dtype = (
        torch.bfloat16
        if precision == "bf16"
        else (torch.float16 if precision == "fp16" else torch.float32)
    )
    itemsize = torch.empty((), dtype=dtype).element_size()
    free = api.mem_get_info(device)[0] if hasattr(api, "mem_get_info") else None
    lengths = sorted({*_SCORE_LENGTHS, training.seq_len})
    measurements: list[dict[str, float]] = []
    skipped: list[int] = []
    for length in lengths:
        score_bytes = training.micro_batch_size * heads * length * length * itemsize
        if free is not None and score_bytes > _SCORE_FREE_SHARE * free:
            skipped.append(length)
            continue
        shape = (training.micro_batch_size, heads, length, head_dim)
        query, key, value = (
            torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
            for _ in range(3)
        )
        api.synchronize(device)
        baseline = api.memory_allocated(device)
        api.reset_peak_memory_stats(device)
        output = functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=True
        )
        output.float().square().mean().backward()
        api.synchronize(device)
        peak = api.max_memory_allocated(device) - baseline
        del query, key, value, output
        measurements.append(
            {
                "sequence_length": length,
                "peak_extra_bytes": peak,
                "score_matrix_bytes": score_bytes,
                "extra_over_score": peak / score_bytes,
            }
        )
    if len(measurements) < 2:
        return {
            "status": "UNMEASURED",
            "reason": "fewer than two sequence lengths fit beside a score matrix",
            "skipped_sequence_lengths": skipped,
        }
    keeps = retains_scores(measurements)
    assumed = assumes_fused_scores(config, config.attention)
    return {
        "status": "MEASURED",
        "shape": {
            "micro_batch_size": training.micro_batch_size,
            "num_heads": heads,
            "head_dim": head_dim,
            "dtype": str(dtype).removeprefix("torch."),
        },
        "measurements": measurements,
        "skipped_sequence_lengths": skipped,
        "retains_score_matrix": keeps,
        # The estimate charges no [T, T] scores for fused SDPA on CUDA and XPU.
        "estimate_assumption": (
            "not_applicable"
            if not assumed
            else "contradicted"
            if keeps
            else "confirmed"
        ),
    }


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
    if config.attention_kinds != {"dense"}:
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
            "score_memory": _score_memory(config, device),
            "recommendations": _recommendations(runtime, candidates),
        }
    finally:
        restore_rng_state(rng_state)
        _restore_training_state(training_state)
