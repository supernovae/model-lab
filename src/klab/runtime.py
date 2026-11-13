"""Runtime discovery, device selection, measurements, and reproducibility helpers."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import uuid as _uuid_mod
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import psutil

from klab import host_capacity

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if TYPE_CHECKING:
    from klab.runtime_profile import RuntimeAuthorization


@dataclass(frozen=True)
class RuntimeInfo:
    engine: str
    backend: str
    torch_device: str | None
    device_index: int
    device_name: str | None
    physical_device_id: str | None
    framework_version: str | None
    runtime_version: str | None
    driver_version: str | None
    os: str
    system_total_bytes: int | None
    system_available_bytes: int | None
    device_total_bytes: int | None
    device_free_bytes: int | None
    device_recommended_bytes: int | None
    measurement_source: str | None
    measured_at: str
    precision_capabilities: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    device_driver_allocated_bytes: int | None = None
    # Capability declarations are passive inventory; these fields are evidence
    # from a disposable validation probe.
    tested_precisions: tuple[str, ...] = ()
    tested_features: tuple[str, ...] = ()
    validated_at: str | None = None
    format_version: int = 1
    # Optional additions preserve historical records without inferring their host
    # from the machine reading them.
    host_os: str | None = None
    host_environment: str | None = None
    host_architecture: str | None = None

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != 1:
            raise ValueError("unsupported RuntimeInfo format version")
        for name in ("host_os", "host_environment", "host_architecture"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"RuntimeInfo {name} must be a string or null")
        for name in (
            "precision_capabilities",
            "limitations",
            "tested_precisions",
            "tested_features",
        ):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or any(
                not isinstance(item, str) for item in values
            ):
                raise TypeError(f"RuntimeInfo {name} must contain strings")
            object.__setattr__(self, name, tuple(values))
        for name in (
            "device_index",
            "system_total_bytes",
            "system_available_bytes",
            "device_total_bytes",
            "device_free_bytes",
            "device_recommended_bytes",
            "device_driver_allocated_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(
                    f"RuntimeInfo {name} must be a nonnegative integer or null"
                )

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> RuntimeInfo:
        if not isinstance(value, dict) or "format_version" not in value:
            raise ValueError("RuntimeInfo requires a versioned object")
        return cls(**value)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _ram() -> tuple[int, int]:
    memory = host_capacity.measure_memory()
    return int(memory.total), int(memory.available)


def detect_host_environment() -> dict[str, str]:
    """Describe the host independently of installed or available accelerators.

    Distro/session variables identify WSL but do not establish its generation.
    A custom WSL kernel can lack the usual WSL2 release marker; retain that
    uncertainty instead of claiming native Linux or a particular WSL version.
    """
    system = platform.system().lower()
    environment = "native"
    if system == "linux":
        release = platform.release().lower()
        if "wsl2" in release or "microsoft-standard" in release:
            environment = "wsl2"
        elif (
            "microsoft" in release
            or os.environ.get("WSL_DISTRO_NAME")
            or os.environ.get("WSL_INTEROP")
        ):
            environment = "unknown-wsl"
    return {
        "host_os": "macos" if system == "darwin" else system,
        "host_environment": environment,
        "host_architecture": platform.machine(),
    }


def _os_identity() -> str:
    identity = platform.platform()
    if detect_host_environment()["host_environment"] != "native":
        return f"{identity} (WSL)"
    return identity


def _backend_available(backend: str) -> bool:
    if torch is None:
        return False
    if backend == "cpu":
        return True
    if backend == "mps":
        return bool(torch.backends.mps.is_available())
    if backend in {"cuda", "rocm"}:
        return bool(torch.cuda.is_available()) and (
            (backend == "rocm") == bool(torch.version.hip)
        )
    if backend == "xpu":
        xpu = getattr(torch, "xpu", None)
        return bool(xpu and xpu.is_available())
    return False


def _auto_backend() -> str:
    if _backend_available("mps"):
        return "mps"
    if torch.cuda.is_available():
        return "rocm" if torch.version.hip else "cuda"
    if _backend_available("xpu"):
        return "xpu"
    return "cpu"


def torch_device_for(backend: str, device_index: int = 0) -> torch.device:
    if backend in {"cuda", "rocm"}:
        return torch.device("cuda", device_index)
    if backend == "xpu":
        return torch.device("xpu", device_index)
    if backend == "mps":
        return torch.device("mps")
    if backend == "cpu":
        return torch.device("cpu")
    raise ValueError(f"backend has no PyTorch device: {backend}")


def _device_count(backend: str) -> int:
    if backend in {"cuda", "rocm"}:
        return int(torch.cuda.device_count())
    if backend == "xpu":
        xpu = getattr(torch, "xpu", None)
        return int(xpu.device_count()) if xpu and hasattr(xpu, "device_count") else 0
    return 1


def _declared_precisions(backend: str, available: bool) -> tuple[str, ...]:
    """Return static API declarations, never validation evidence."""
    if not available:
        return ()
    if backend in {"cpu", "mps"}:
        return ("fp32", "bf16")
    if backend in {"cuda", "rocm", "xpu"}:
        return ("fp32", "bf16", "fp16")
    return ()


def _safe_call(callable_: Any) -> tuple[Any | None, str | None]:
    try:
        return callable_(), None
    except (
        RuntimeError,
        OSError,
        AttributeError,
        ValueError,
        TypeError,
        AssertionError,
    ) as error:
        return None, f"{type(error).__name__}: {error}"


def _format_device_uuid(raw_uuid: object) -> str | None:
    if raw_uuid is None:
        return None
    if isinstance(raw_uuid, (list, tuple)) and all(
        isinstance(b, int) and 0 <= b <= 255 for b in raw_uuid
    ):
        raw_uuid = bytes(raw_uuid)
    if isinstance(raw_uuid, (bytes, bytearray)):
        if len(raw_uuid) == 16:
            return str(_uuid_mod.UUID(bytes=bytes(raw_uuid)))
        hex_str = bytes(raw_uuid).hex()
        return hex_str or None
    text = str(raw_uuid).strip()
    return text or None


def discover_runtimes(
    *, measurement_device: torch.device | None = None
) -> list[RuntimeInfo]:
    """Inventory without allocator initialization; probes may measure one explicit device."""
    total, available = _ram()
    host = detect_host_environment()
    infos: list[RuntimeInfo] = []
    for backend in ("cpu", "mps", "cuda", "rocm", "xpu"):
        supported = _backend_available(backend)
        measure = (
            torch is not None
            and measurement_device is not None
            and torch_device_for(backend).type == measurement_device.type
        )
        count_val = None
        if measure:
            indices: tuple[int, ...] = (measurement_device.index or 0,)
        elif supported and torch is not None and backend in {"cuda", "rocm", "xpu"}:
            api = (
                getattr(torch, "xpu", None)
                if backend == "xpu"
                else getattr(torch, "cuda", None)
            )
            if api is not None and hasattr(api, "device_count"):
                count_val, _ = _safe_call(api.device_count)
            indices = (
                tuple(range(int(count_val)))
                if isinstance(count_val, int) and count_val > 1
                else (0,)
            )
        else:
            indices = (0,)

        for index in indices:
            limitations: list[str] = []
            if (
                torch is not None
                and backend in {"cuda", "rocm"}
                and ((backend == "rocm") != bool(torch.version.hip))
            ):
                limitations.append("PyTorch build targets the other CUDA/HIP API")
            elif not supported:
                limitations.append("runtime unavailable")
            name = physical_device_id = device_total = device_free = recommended = (
                driver_allocated
            ) = None
            if supported and backend != "cpu":
                limitations.append(
                    "driver version unavailable through safe PyTorch APIs"
                )
            if (
                supported
                and backend != "cpu"
                and not (
                    getattr(torch.version, "xpu", None)
                    if backend == "xpu"
                    else (torch.version.hip or torch.version.cuda)
                )
            ):
                limitations.append(
                    "accelerator runtime version unavailable through safe PyTorch APIs"
                )
            source = None
            if supported and backend != "cpu" and not measure:
                name = "Apple Metal" if backend == "mps" else None
                if (
                    torch is not None
                    and backend in {"cuda", "rocm", "xpu"}
                    and isinstance(count_val, int)
                    and count_val > 0
                ):
                    api = (
                        getattr(torch, "xpu", None)
                        if backend == "xpu"
                        else getattr(torch, "cuda", None)
                    )
                    if api is not None and hasattr(api, "get_device_properties"):
                        props, _ = _safe_call(
                            lambda a=api, i=index: a.get_device_properties(i)
                        )
                        if props is not None:
                            prop_name = getattr(props, "name", None)
                            if prop_name:
                                name = str(prop_name)
                            physical_device_id = _format_device_uuid(
                                getattr(props, "uuid", None)
                            )
                if name is not None and backend != "mps":
                    limitations.append("device memory deferred to isolated validation")
                else:
                    limitations.append(
                        "device properties and memory deferred to isolated validation"
                    )
            elif supported and backend in {"cuda", "rocm"}:
                properties, reason = _safe_call(
                    lambda i=index: torch.cuda.get_device_properties(i)
                )
                if properties is None:
                    limitations.append(f"device properties unavailable: {reason}")
                    limitations.append(
                        "physical device identity unavailable through safe PyTorch APIs"
                    )
                else:
                    name, device_total = properties.name, int(properties.total_memory)
                    physical_device_id = _format_device_uuid(
                        getattr(properties, "uuid", None)
                    )
                    if physical_device_id is None:
                        limitations.append(
                            "physical device identity unavailable through safe PyTorch APIs"
                        )
                memory, reason = _safe_call(lambda i=index: torch.cuda.mem_get_info(i))
                if memory is None:
                    limitations.append(f"free-memory reading unavailable: {reason}")
                else:
                    device_free, source = int(memory[0]), "torch.cuda.mem_get_info"
            elif supported and backend == "mps":
                limitations.append(
                    "physical device identity unavailable through safe PyTorch APIs"
                )
                name, source = "Apple Metal", "torch.mps"
                mps = torch.mps
                if hasattr(mps, "recommended_max_memory"):
                    value, reason = _safe_call(mps.recommended_max_memory)
                    recommended = int(value) if value is not None else None
                    if reason:
                        limitations.append(
                            f"recommended-memory reading unavailable: {reason}"
                        )
                else:
                    limitations.append("recommended-memory API unavailable")
                if hasattr(mps, "driver_allocated_memory"):
                    value, reason = _safe_call(mps.driver_allocated_memory)
                    driver_allocated = int(value) if value is not None else None
                    if reason:
                        limitations.append(
                            f"driver-memory reading unavailable: {reason}"
                        )
                else:
                    limitations.append("driver-memory API unavailable")
            elif supported and backend == "xpu":
                xpu = torch.xpu
                if hasattr(xpu, "get_device_name"):
                    value, reason = _safe_call(
                        lambda api=xpu, i=index: api.get_device_name(i)
                    )
                    name = str(value) if value is not None else None
                    if reason:
                        limitations.append(f"device-name reading unavailable: {reason}")
                else:
                    limitations.append("device-name API unavailable")
                if hasattr(xpu, "get_device_properties"):
                    properties, reason = _safe_call(
                        lambda api=xpu, i=index: api.get_device_properties(i)
                    )
                    if properties is not None:
                        device_total = int(properties.total_memory)
                        physical_device_id = _format_device_uuid(
                            getattr(properties, "uuid", None)
                        )
                    if reason:
                        limitations.append(f"device properties unavailable: {reason}")
                else:
                    limitations.append("device-properties API unavailable")
                if physical_device_id is None:
                    limitations.append(
                        "physical device identity unavailable through safe PyTorch APIs"
                    )
                if hasattr(xpu, "mem_get_info"):
                    memory, reason = _safe_call(
                        lambda api=xpu, i=index: api.mem_get_info(i)
                    )
                    if memory is None:
                        limitations.append(f"free-memory reading unavailable: {reason}")
                    else:
                        device_free, source = int(memory[0]), "torch.xpu.mem_get_info"
                else:
                    limitations.append("free-memory API unavailable")
            infos.append(
                RuntimeInfo(
                    engine="pytorch",
                    backend=backend,
                    torch_device=str(torch_device_for(backend, index))
                    if supported
                    else None,
                    device_index=index,
                    device_name=name,
                    physical_device_id=physical_device_id,
                    framework_version=torch.__version__ if torch is not None else None,
                    runtime_version=(
                        getattr(torch.version, "xpu", None)
                        if backend == "xpu"
                        else (torch.version.hip or torch.version.cuda)
                    )
                    if torch is not None
                    else None,
                    driver_version=None,
                    os=_os_identity(),
                    system_total_bytes=total,
                    system_available_bytes=available,
                    device_total_bytes=device_total,
                    device_free_bytes=device_free,
                    device_recommended_bytes=recommended,
                    measurement_source=source,
                    measured_at=_now(),
                    precision_capabilities=_declared_precisions(backend, supported),
                    limitations=tuple(limitations),
                    device_driver_allocated_bytes=driver_allocated,
                    **host,
                )
            )
    mlx_spec = importlib.util.find_spec("mlx")
    if mlx_spec is None:
        infos.append(
            RuntimeInfo(
                "mlx",
                "metal",
                None,
                0,
                None,
                None,
                None,
                None,
                None,
                _os_identity(),
                total,
                available,
                None,
                None,
                None,
                None,
                _now(),
                (),
                ("MLX package unavailable",),
                **host,
            )
        )
    else:
        try:
            mlx_version = importlib.metadata.version("mlx")
        except importlib.metadata.PackageNotFoundError:
            mlx_version = None
        # Discovery intentionally uses package metadata only.  Importing mlx.core
        # may initialize Metal and is reserved for the disposable validation probe.
        infos.append(
            RuntimeInfo(
                engine="mlx",
                backend="metal",
                torch_device=None,
                device_index=0,
                device_name="Apple Metal",
                physical_device_id=None,
                framework_version=mlx_version,
                runtime_version=mlx_version,
                driver_version=None,
                os=_os_identity(),
                system_total_bytes=total,
                system_available_bytes=available,
                device_total_bytes=None,
                device_free_bytes=None,
                device_recommended_bytes=None,
                measurement_source="mlx package metadata",
                measured_at=_now(),
                precision_capabilities=("fp32",),
                limitations=(
                    "MLX execution has not been validated",
                    "Metal driver and physical-device identity are unavailable through MLX",
                    "Apple unified memory is one shared system-memory pool",
                    "Metal capacity is unavailable without an active probe",
                ),
                **host,
            )
        )
    return infos


_PROBE_TIMEOUT_SECONDS = 30
_MAX_PROBE_OUTPUT_BYTES = 16 * 1024


def _probe_runtime(
    *,
    engine: Literal["pytorch", "mlx"],
    backend: str,
    device_index: int,
    precision: Literal["fp32", "bf16", "fp16"],
    optimizer: str,
    checkpointing: bool,
    activation_offload: bool = False,
    attention: str = "dense",
) -> dict[str, Any]:
    """Run a versioned mutating probe outside this interpreter."""
    request = json.dumps(
        {
            "format_version": 1,
            "engine": engine,
            "backend": backend,
            "device_index": device_index,
            "precision": precision,
            "optimizer": optimizer,
            "checkpointing": checkpointing,
            "activation_offload": activation_offload,
            "attention": attention,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "klab.runtime_probe"],
            input=request,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError(
            f"runtime probe timed out after {_PROBE_TIMEOUT_SECONDS} seconds"
        ) from error
    if len(completed.stdout) > _MAX_PROBE_OUTPUT_BYTES:
        raise ValueError("runtime probe returned output larger than the safety limit")
    if completed.returncode:
        raise ValueError(f"runtime probe exited with status {completed.returncode}")
    try:
        result = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("runtime probe returned invalid JSON") from error
    if not isinstance(result, dict) or result.get("format_version") != 1:
        raise ValueError("runtime probe returned an unsupported format version")
    if result.get("ok") is not True:
        raise ValueError(
            f"runtime probe failed: {result.get('reason') or 'unknown failure'}"
        )
    if result.get("tested_precision") != precision:
        raise ValueError("runtime probe returned an invalid precision result")
    expected_features = ["forward_backward_optimizer", f"optimizer:{optimizer}"]
    if attention == "block_sparse" and (
        engine == "mlx" or (engine == "pytorch" and backend == "rocm")
    ):
        expected_features.append("native_block_sparse_attention")
    if checkpointing:
        expected_features.append("activation_checkpointing")
    if activation_offload:
        expected_features.append("activation_offload")
    if result.get("tested_features") != expected_features:
        raise ValueError("runtime probe returned invalid feature evidence")
    return result


def _validated_probe_info(
    result: Mapping[str, object],
    *,
    engine: str,
    backend: str,
    device_index: int,
    precision: str,
    torch_device: str | None,
) -> RuntimeInfo:
    data = result.get("runtime")
    if not isinstance(data, dict):
        raise TypeError("runtime probe omitted device measurements")
    try:
        info = RuntimeInfo.from_dict(data)
    except (KeyError, TypeError) as error:
        raise ValueError(
            "runtime probe returned invalid device measurements"
        ) from error
    if (
        info.engine != engine
        or info.backend != backend
        or info.device_index != device_index
        or info.torch_device != torch_device
        or info.tested_precisions != (precision,)
        or info.tested_features != tuple(result["tested_features"])
        or info.validated_at is None
    ):
        raise ValueError("runtime probe returned mismatched device evidence")
    return info


def _resolved_precision(requested: str) -> str:
    if requested == "auto":
        return "fp32"
    if requested not in {"fp32", "bf16", "fp16"}:
        raise ValueError(f"unsupported precision: {requested}")
    return requested


def _validate_mlx_runtime(config: Any) -> RuntimeInfo:
    """Validate semantic support in-process, then probe a fixed tiny native graph."""
    from klab.engines.mlx_policy import validate as validate_mlx

    validate_mlx(config)
    runtime = config.runtime
    precision = _resolved_precision(runtime.precision)
    result = _probe_runtime(
        engine="mlx",
        backend="metal",
        device_index=runtime.device_index,
        precision=precision,
        optimizer=config.optimizer.name,
        checkpointing=runtime.memory.activation_checkpointing.enabled,
        attention=(
            "block_sparse"
            if "block_sparse" in config.attention_kinds
            else config.attention.kind
        ),
    )
    return _validated_probe_info(
        result,
        engine="mlx",
        backend="metal",
        device_index=runtime.device_index,
        precision=precision,
        torch_device=None,
    )


def validate_runtime(
    config: Any, *, authorization: RuntimeAuthorization | None = None
) -> RuntimeInfo:
    """Probe the requested engine without changing parent allocator/RNG state."""
    from klab.runtime_profile import require_authorization

    require_authorization(config, authorization)
    runtime = config.runtime
    if runtime.engine == "mlx":
        return _validate_mlx_runtime(config)
    if runtime.engine != "pytorch":
        raise ValueError(f"unsupported runtime engine: {runtime.engine}")
    backend = _auto_backend() if runtime.backend == "auto" else runtime.backend
    if backend not in {"cpu", "mps", "cuda", "rocm", "xpu"}:
        raise ValueError(f"unsupported PyTorch backend: {backend}")
    if not _backend_available(backend):
        raise ValueError(f"requested backend unavailable: {backend}")
    if runtime.device_index >= _device_count(backend):
        raise ValueError(
            f"requested device_index {runtime.device_index} unavailable for {backend}"
        )
    if backend in {"cpu", "mps"} and runtime.device_index != 0:
        raise ValueError(f"{backend} supports only device_index=0")
    precision = _resolved_precision(runtime.precision)
    if precision == "fp16" and backend == "cpu":
        raise ValueError("fp16 is unsupported on CPU; use fp32 or bf16")
    if precision == "fp16" and backend not in {"cuda", "rocm", "xpu"}:
        raise ValueError(f"fp16 requires a supported GradScaler on {backend}")
    if runtime.memory.activation_offload.enabled and backend == "cpu":
        raise ValueError("activation offload is meaningless on CPU")
    result = _probe_runtime(
        engine="pytorch",
        backend=backend,
        device_index=runtime.device_index,
        precision=precision,
        optimizer=config.optimizer.name,
        checkpointing=runtime.memory.activation_checkpointing.enabled,
        activation_offload=runtime.memory.activation_offload.enabled,
    )
    return _validated_probe_info(
        result,
        engine="pytorch",
        backend=backend,
        device_index=runtime.device_index,
        precision=precision,
        torch_device=str(torch_device_for(backend, runtime.device_index)),
    )


def precision_context(config: Any, device: torch.device) -> Any:
    """Return the trainer autocast context; reductions remain explicitly fp32."""
    precision = _resolved_precision(config.runtime.precision)
    if precision == "fp32":
        return nullcontext()
    if precision == "fp16" and device.type not in {"cuda", "xpu"}:
        raise ValueError(f"fp16 requires a supported GradScaler on {device.type}")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_grad_scaler(config: Any, device: torch.device) -> Any | None:
    """Return an enabled fp16 scaler only for validated scaler-capable backends."""
    if _resolved_precision(config.runtime.precision) != "fp16":
        return None
    if device.type not in {"cuda", "xpu"}:
        raise ValueError(f"fp16 GradScaler is unsupported on {device.type}")
    return torch.amp.GradScaler(device.type, enabled=True)


def select_device(requested: str) -> torch.device:
    backend = _auto_backend() if requested == "auto" else requested
    if backend not in {"cpu", "mps", "cuda", "rocm", "xpu"} or not _backend_available(
        backend
    ):
        raise ValueError(f"requested device unavailable: {requested}")
    return torch_device_for(backend)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)


def allocated_memory_bytes(device: torch.device) -> int | None:
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return int(torch.mps.current_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    if device.type == "xpu" and hasattr(torch.xpu, "memory_allocated"):
        return int(torch.xpu.memory_allocated(device))
    return None


def process_rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def seed_everything(seed: int, *, deterministic_cpu: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cpu:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if _backend_available("xpu"):
        state["xpu"] = torch.xpu.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "xpu" in state and _backend_available("xpu"):
        torch.xpu.set_rng_state_all(state["xpu"])
    if (
        "mps" in state
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state["mps"])
