"""Synchronous saved-tensor activation offload for PyTorch accelerators.

Only autograd-saved activation storage is copied. Parameters, buffers, and their
views are excluded by storage identity; no hook mutates an input tensor.
"""

from __future__ import annotations

import gc
import time
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch.multiprocessing.reductions import StorageWeakRef


@dataclass(frozen=True)
class OffloadMetrics:
    """Per-update transfer and live-host-storage measurements."""

    bytes_to_cpu: int = 0
    bytes_to_device: int = 0
    live_host_bytes: int = 0
    peak_host_bytes: int = 0
    transfer_seconds: float = 0.0

    def as_metrics(self, update_seconds: float | None = None) -> dict[str, float]:
        result = {
            "offload/bytes_to_cpu": float(self.bytes_to_cpu),
            "offload/bytes_to_device": float(self.bytes_to_device),
            "offload/peak_host_bytes": float(self.peak_host_bytes),
            "offload/transfer_seconds": self.transfer_seconds,
        }
        if update_seconds is not None and update_seconds > 0:
            result["offload/transfer_fraction"] = self.transfer_seconds / update_seconds
        return result


@dataclass(frozen=True)
class OffloadProbe:
    """Result of a real disposable saved-tensor roundtrip/backward probe."""

    supported: bool
    reason: str | None
    device_type: str
    unified_memory: bool
    estimated_device_capacity_savings_bytes: int
    measurement: OffloadMetrics | None

    def as_dict(self) -> dict[str, Any]:
        measurement = self.measurement
        return {
            "supported": self.supported,
            "reason": self.reason,
            "device_type": self.device_type,
            "unified_memory": self.unified_memory,
            "estimated_device_capacity_savings_bytes": (
                self.estimated_device_capacity_savings_bytes
            ),
            "measurement": None
            if measurement is None
            else {
                "bytes_to_cpu": measurement.bytes_to_cpu,
                "bytes_to_device": measurement.bytes_to_device,
                "live_host_bytes": measurement.live_host_bytes,
                "peak_host_bytes": measurement.peak_host_bytes,
                "transfer_seconds": measurement.transfer_seconds,
            },
        }


@dataclass
class _StoredActivation:
    key: tuple[object, ...]
    host_storage: torch.Tensor
    nbytes: int
    references: int = 0
    restored_storage: torch.Tensor | None = None


class _PackedActivation:
    """Opaque Autograd handle releasing exactly one saved-tensor reference."""

    __slots__ = (
        "_owner",
        "dtype",
        "released",
        "shape",
        "storage_offset",
        "stored",
        "stride",
    )

    def __init__(
        self,
        owner: ActivationOffload,
        stored: _StoredActivation,
        tensor: torch.Tensor,
    ) -> None:
        self._owner = weakref.ref(owner)
        self.stored = stored
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = tensor.storage_offset()
        self.dtype = tensor.dtype
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        owner = self._owner()
        if owner is not None:
            owner._release(self.stored)

    def __del__(self) -> None:
        self.release()


class ActivationOffload:
    """Offload eligible saved storage for one model and one accelerator.

    ``model`` is mandatory: Autograd may save parameter or buffer views that are
    non-leaves. Storage identity replaces leaf filtering, retaining eligible
    non-parameter leaves while excluding every parameter/buffer alias.
    """

    _ACCELERATOR_TYPES = frozenset(("cuda", "mps", "xpu"))

    def __init__(self, device: torch.device, model: torch.nn.Module) -> None:
        self.device = torch.device(device)
        if self.device.type == "cpu":
            raise ValueError("activation offload is meaningless on CPU")
        if self.device.type not in self._ACCELERATOR_TYPES:
            raise ValueError(
                f"activation offload is unsupported for PyTorch device {self.device}"
            )
        self._excluded_storage_keys = _model_storage_keys(model)
        self._modules = weakref.WeakSet(model.modules())
        self._stored: dict[tuple[object, ...], _StoredActivation] = {}
        self._bytes_to_cpu = 0
        self._bytes_to_device = 0
        self._live_host_bytes = 0
        self._peak_host_bytes = 0
        self._transfer_seconds = 0.0

    @property
    def metrics(self) -> OffloadMetrics:
        return OffloadMetrics(
            bytes_to_cpu=self._bytes_to_cpu,
            bytes_to_device=self._bytes_to_device,
            live_host_bytes=self._live_host_bytes,
            peak_host_bytes=self._peak_host_bytes,
            transfer_seconds=self._transfer_seconds,
        )

    def reset(self) -> bool:
        """Reset update counters only when all packed handles are released.

        Returning ``False`` means a graph is still retained; counters deliberately
        remain intact rather than concealing a live saved-storage lifetime.
        """
        if self._stored:
            return False
        self._bytes_to_cpu = 0
        self._bytes_to_device = 0
        self._live_host_bytes = 0
        self._peak_host_bytes = 0
        self._transfer_seconds = 0.0
        return True

    def pack(self, tensor: torch.Tensor) -> torch.Tensor | _PackedActivation:
        """Copy each saved storage version once and retain its view metadata."""
        if not self._eligible(tensor):
            return tensor
        key = _storage_key(tensor)
        assert key is not None
        # A later in-place operation creates a different saved value even when
        # aliases still share the same backing allocation.
        key = (*key, tensor._version)
        stored = self._stored.get(key)
        if stored is None:
            source = _whole_storage_view(tensor)
            if source is None:
                return tensor
            host = self._copy_to_cpu(source)
            stored = _StoredActivation(key=key, host_storage=host, nbytes=host.nbytes)
            self._stored[key] = stored
            self._bytes_to_cpu += stored.nbytes
            self._live_host_bytes += stored.nbytes
            self._peak_host_bytes = max(self._peak_host_bytes, self._live_host_bytes)
        stored.references += 1
        return _PackedActivation(self, stored, tensor)

    def unpack(self, packed: torch.Tensor | _PackedActivation) -> torch.Tensor:
        """Restore a saved view while retaining storage through graph release."""
        if isinstance(packed, torch.Tensor):
            return packed
        stored = packed.stored
        if stored.restored_storage is None:
            stored.restored_storage = self._copy_to_device(stored.host_storage)
            self._bytes_to_device += stored.nbytes
        storage = (
            stored.restored_storage
            if stored.restored_storage.dtype == packed.dtype
            else stored.restored_storage.view(packed.dtype)
        )
        return storage.as_strided(packed.shape, packed.stride, packed.storage_offset)

    @contextmanager
    def hooks(self) -> Iterator[None]:
        # RoPE and other modules may register lazy buffers during the forward.
        # Capture those registrations without rescanning every parameter/buffer
        # for every saved activation.
        def exclude_buffer(module, name, buffer):
            if module in self._modules and isinstance(buffer, torch.Tensor):
                key = _storage_key(buffer)
                if key is not None:
                    self._excluded_storage_keys.add(key)

        with (
            torch.nn.modules.module.register_module_buffer_registration_hook(
                exclude_buffer
            ),
            torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack),
        ):
            yield

    def _eligible(self, tensor: torch.Tensor) -> bool:
        if tensor.layout != torch.strided or tensor.device.type != self.device.type:
            return False
        if self.device.index is not None and tensor.device.index != self.device.index:
            return False
        key = _storage_key(tensor)
        return key is not None and key not in self._excluded_storage_keys

    def _copy_to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        self._synchronize()
        started = time.perf_counter()
        host = tensor.detach().to(device="cpu", copy=True).contiguous()
        self._synchronize()
        self._transfer_seconds += time.perf_counter() - started
        return host

    def _copy_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        self._synchronize()
        started = time.perf_counter()
        device_storage = tensor.to(device=self.device, copy=True)
        self._synchronize()
        self._transfer_seconds += time.perf_counter() - started
        return device_storage

    def _release(self, stored: _StoredActivation) -> None:
        stored.references -= 1
        if stored.references < 0:
            raise RuntimeError("activation offload packed-storage reference underflow")
        if stored.references:
            return
        if self._stored.get(stored.key) is not stored:
            raise RuntimeError("activation offload lost packed-storage ownership")
        del self._stored[stored.key]
        self._live_host_bytes -= stored.nbytes
        if self._live_host_bytes < 0:
            raise RuntimeError("activation offload live-host accounting underflow")
        stored.restored_storage = None

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "xpu":
            torch.xpu.synchronize(self.device)


def _storage_key(tensor: torch.Tensor) -> tuple[object, ...] | None:
    """Keep storage identity unique without retaining its device allocation."""
    if tensor.layout != torch.strided:
        return None
    try:
        storage = tensor.untyped_storage()
        return (
            tensor.device.type,
            tensor.device.index,
            StorageWeakRef(storage),
            storage.nbytes(),
        )
    except AttributeError, RuntimeError:
        return None


def _whole_storage_view(tensor: torch.Tensor) -> torch.Tensor | None:
    """Return a flat view of precisely the storage backing ``tensor``."""
    try:
        storage_bytes = tensor.untyped_storage().nbytes()
        if storage_bytes % tensor.element_size():
            return None
        return tensor.detach().as_strided(
            (storage_bytes // tensor.element_size(),), (1,), storage_offset=0
        )
    except RuntimeError:
        return None


def _model_storage_keys(model: torch.nn.Module) -> set[tuple[object, ...]]:
    keys: set[tuple[object, ...]] = set()
    for tensor in (*model.parameters(), *model.buffers()):
        key = _storage_key(tensor)
        if key is not None:
            keys.add(key)
    return keys


def probe_activation_offload(device: torch.device) -> OffloadProbe:
    """Execute a disposable saved-tensor roundtrip/backward probe on ``device``.

    Passing this probe proves only hook execution on this runtime. It does not
    certify capacity relief; MPS is unified memory and always reports zero
    estimated capacity saving.
    """
    device = torch.device(device)
    unified_memory = device.type == "mps"
    if device.type == "cpu":
        return OffloadProbe(
            False,
            "activation offload is meaningless on CPU",
            device.type,
            False,
            0,
            None,
        )
    if device.type not in ActivationOffload._ACCELERATOR_TYPES:
        return OffloadProbe(
            False, f"unsupported PyTorch device {device}", device.type, False, 0, None
        )
    try:
        model = torch.nn.Linear(4, 4, bias=False).to(device)
        offload = ActivationOffload(device, model)
        x = torch.randn(3, 4, device=device, requires_grad=True)
        with offload.hooks():
            loss = model(x).square().sum()
        loss.backward()
        if x.grad is None or not torch.isfinite(x.grad).all():
            raise RuntimeError("probe backward produced no finite input gradient")
        del loss, x, model
        gc.collect()
        measurement = offload.metrics
        if measurement.bytes_to_cpu <= 0 or measurement.bytes_to_device <= 0:
            raise RuntimeError("probe did not exercise a saved-tensor roundtrip")
        if measurement.live_host_bytes:
            raise RuntimeError(
                "probe retained packed activation storage after backward"
            )
        return OffloadProbe(True, None, device.type, unified_memory, 0, measurement)
    except (RuntimeError, AttributeError, AssertionError) as exc:
        return OffloadProbe(
            False,
            f"{type(exc).__name__}: {exc}",
            device.type,
            unified_memory,
            0,
            None,
        )
