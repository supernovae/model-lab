"""Read-only executable capability inventory for a selected Python interpreter.

Run as ``python -I /absolute/path/to/runtime_env_probe.py``. This module uses
only the standard library and does not require Model Lab or an ML framework.
"""

# Vendor import/API failures must become independent diagnostic records.
# ruff: noqa: BLE001

from __future__ import annotations

import importlib
import json
import sys
from typing import Any

_MAX_NAMES = 32


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:512]


def _device() -> dict[str, Any]:
    return {"available": False, "count": 0, "names": []}


def _import(name: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(name), None
    except Exception as exc:
        return None, _error(exc)


def _sample_devices(api: Any) -> dict[str, Any]:
    """A broken driver/query must not hide unrelated backends."""
    result = _device()
    try:
        result["available"] = bool(api.is_available())
        if result["available"]:
            result["count"] = max(0, int(api.device_count()))
            for index in range(min(result["count"], _MAX_NAMES)):
                try:
                    result["names"].append(str(api.get_device_name(index))[:256])
                except Exception as exc:
                    result["names"].append(f"<device {index}: {_error(exc)}>")
    except Exception as exc:
        result["error"] = _error(exc)
        result["available"] = False
        result["count"] = 0
        result["names"] = []
    return result


def probe() -> dict[str, Any]:
    """Observe imports and devices without authorizing or initializing a runtime."""
    result: dict[str, Any] = {
        "python": sys.executable,
        "prefix": sys.prefix,
        "python_version": sys.version.split()[0],
        "klab_import": {"success": False, "error": None},
        "source_sha256": None,
        "torch": {
            "installed": False,
            "version": None,
            "path": None,
            "hip": None,
            "cuda": None,
        },
        "mlx": {"installed": False, "version": None, "available": False},
        "devices": {
            name: _device() for name in ("rocm", "cuda", "xpu", "mps", "metal")
        },
        "backends": [],
    }
    package, error = _import("klab")
    if package is None:
        result["klab_import"]["error"] = error
    else:
        result["klab_import"]["success"] = True
        try:
            from klab.runtime_identity_probe import source_identity

            result["source_sha256"] = source_identity()["source_sha256"]
        except Exception as exc:
            result["source_error"] = _error(exc)

    torch, error = _import("torch")
    if torch is None:
        if error and not error.startswith(
            "ModuleNotFoundError: No module named 'torch'"
        ):
            result["torch"]["installed"] = True
            result["torch"]["error"] = error
    else:
        details = result["torch"]
        details["installed"] = True
        details["version"] = str(getattr(torch, "__version__", "unknown"))[:256]
        details["path"] = getattr(torch, "__file__", None)
        try:
            details["hip"] = getattr(torch.version, "hip", None)
            details["cuda"] = getattr(torch.version, "cuda", None)
        except Exception as exc:
            details["error"] = _error(exc)
        result["backends"].append("cpu")
        if details["hip"] or details["cuda"]:
            try:
                cuda_devices = _sample_devices(torch.cuda)
                backend = "rocm" if details["hip"] else "cuda"
                result["devices"][backend] = cuda_devices
                if cuda_devices["available"] and cuda_devices["count"] > 0:
                    result["backends"].append(backend)
            except Exception as exc:
                result["devices"]["rocm" if details["hip"] else "cuda"]["error"] = (
                    _error(exc)
                )
        try:
            xpu_devices = _sample_devices(torch.xpu)
            result["devices"]["xpu"] = xpu_devices
            if xpu_devices["available"] and xpu_devices["count"] > 0:
                result["backends"].append("xpu")
        except AttributeError:
            pass
        except Exception as exc:
            result["devices"]["xpu"]["error"] = _error(exc)
        try:
            mps = torch.backends.mps
            mps_device = result["devices"]["mps"]
            mps_device["available"] = bool(mps.is_available())
            if mps_device["available"]:
                mps_device["count"] = 1
                mps_device["names"] = ["Apple MPS"]
                result["backends"].append("mps")
        except AttributeError:
            pass
        except Exception as exc:
            result["devices"]["mps"]["error"] = _error(exc)

    core, error = _import("mlx.core")
    if core is None:
        if error and not error.startswith("ModuleNotFoundError: No module named 'mlx'"):
            result["mlx"]["installed"] = True
            result["mlx"]["error"] = error
    else:
        details = result["mlx"]
        details["installed"] = True
        try:
            details["version"] = str(getattr(core, "__version__", "unknown"))[:256]
            metal = result["devices"]["metal"]
            metal["available"] = bool(core.metal.is_available())
            details["available"] = metal["available"]
            if metal["available"]:
                metal["count"] = 1
                metal["names"] = ["Apple Metal"]
                result["backends"].append("metal")
        except Exception as exc:
            details["error"] = _error(exc)
            result["devices"]["metal"]["error"] = _error(exc)
    return result


def main() -> None:
    sys.stdout.write(json.dumps(probe(), ensure_ascii=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
