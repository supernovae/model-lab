"""Framework-free semantic and availability checks for native MLX execution."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

from klab.engines.base import EngineCapabilityError

if TYPE_CHECKING:
    from klab.config.models import RunConfig
    from klab.runtime import RuntimeInfo


def _mlx_available() -> bool:
    """Avoid ``find_spec`` raising when the optional parent package is absent."""
    try:
        return find_spec("mlx.core") is not None
    except ModuleNotFoundError:
        return False


def validate_config(config: RunConfig) -> None:
    """Check MLX semantics without importing or probing the optional SDK."""
    if config.runtime.engine != "mlx" or config.runtime.backend != "metal":
        raise EngineCapabilityError(
            "MLX engine requires runtime.engine=mlx and backend=metal"
        )
    if config.runtime.device_index != 0:
        raise EngineCapabilityError("MLX Metal supports only device_index=0")
    if config.model.layers:
        raise EngineCapabilityError("MLX does not support a per-layer schedule")
    if config.model.ffn != "dense":
        raise EngineCapabilityError("MLX supports the dense feed-forward only")
    if config.model.memory != "none":
        raise EngineCapabilityError("MLX does not support memory modules")
    if config.attention.kind not in {"dense", "block_sparse"}:
        raise EngineCapabilityError(
            "MLX supports dense and native block_sparse attention only"
        )
    if config.runtime.memory.activation_offload.enabled:
        raise EngineCapabilityError("activation offload is unavailable for MLX")
    if config.runtime.precision not in {"fp32", "auto"}:
        raise EngineCapabilityError("MLX precision is currently verified for fp32 only")
    if config.optimizer.name != "adamw":
        raise EngineCapabilityError("MLX supports AdamW only; Adafactor is unavailable")
    if config.optimizer.state_offload:
        raise EngineCapabilityError("optimizer state offload is unavailable for MLX")


def validate(config: RunConfig) -> RuntimeInfo:
    """Validate MLX semantics and availability on the selected machine."""
    validate_config(config)
    if not _mlx_available():
        raise EngineCapabilityError("MLX runtime is unavailable")
    from klab.runtime import discover_runtimes

    infos = [info for info in discover_runtimes() if info.engine == "mlx"]
    if not infos:
        raise EngineCapabilityError("MLX runtime discovery returned no Metal runtime")
    return infos[0]
