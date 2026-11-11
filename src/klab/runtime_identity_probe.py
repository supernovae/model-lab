"""Read-only inventory from the exact interpreter named by a runtime profile."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


def source_identity() -> dict[str, str]:
    """Bind executable evidence to this entire installed Model Lab Python source tree."""
    import klab

    root = Path(klab.__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(root.rglob("*.py"))
    if len(files) > 4096:
        raise ValueError("Model Lab source exceeds identity file limit")
    total = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe Model Lab source file: {relative}")
        size = path.stat().st_size
        total += size
        if total > 64 * 1024 * 1024:
            raise ValueError("Model Lab source exceeds identity byte limit")
        digest.update(relative.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return {"package_root": str(root), "source_sha256": digest.hexdigest()}


def _optional(call: Any) -> Any | None:
    try:
        return call()
    except AttributeError, RuntimeError, OSError, ValueError:
        return None


def _host_environment() -> dict[str, str]:
    # Keep this entrypoint importable by a standalone MLX environment without Torch.
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


def probe(request: dict[str, object]) -> dict[str, object]:
    """Only query the selected backend; no model allocation or precision pilot."""
    if set(request) != {"id", "engine", "backend", "device_index"}:
        raise ValueError("invalid runtime identity probe request")
    engine, backend, index = (
        request[key] for key in ("engine", "backend", "device_index")
    )
    if type(index) is not int or index < 0:
        raise ValueError("invalid device index")
    torch: Any | None = None
    if engine == "pytorch":
        import torch

    names: list[str | None] = []
    available = False
    count = 0
    bf16: bool | None = None
    xpu_version: str | None = None
    framework_path: str | None
    framework_version: str | None
    if engine == "mlx" and backend == "metal":
        import mlx.core as mx

        framework_path = str(Path(mx.__file__).resolve())
        framework_version = getattr(mx, "__version__", None)
        available = bool(_optional(mx.metal.is_available))
        count = 1 if available else 0
        names = ["Apple Metal"] if available else []
        bf16 = None  # No passive API establishes a working BF16 optimizer update.
    elif engine == "pytorch" and backend in {"cpu", "cuda", "rocm", "mps", "xpu"}:
        assert torch is not None
        framework_path = str(Path(torch.__file__).resolve())
        framework_version = torch.__version__
        if backend == "cpu":
            available, count, names = True, 1, [platform.processor() or "CPU"]
            bf16 = None
        elif backend in {"cuda", "rocm"}:
            available = bool(_optional(torch.cuda.is_available)) and (
                bool(torch.version.hip) == (backend == "rocm")
            )
            count = int(_optional(torch.cuda.device_count) or 0) if available else 0
            names = [
                _optional(lambda i=i: torch.cuda.get_device_name(i))
                for i in range(count)
            ]
            # The no-index BF16 API queries the current device, not this index.
            # Only the separate mutating precision pilot proves this requirement.
            bf16 = None
        elif backend == "mps":
            available = bool(_optional(torch.backends.mps.is_available))
            count = 1 if available else 0
            names = ["Apple Metal"] if available else []
        else:
            xpu = getattr(torch, "xpu", None)
            available = bool(_optional(xpu.is_available)) if xpu else False
            count = int(_optional(xpu.device_count) or 0) if available else 0
            names = [
                _optional(lambda i=i: xpu.get_device_name(i))
                if hasattr(xpu, "get_device_name")
                else None
                for i in range(count)
            ]
            version = getattr(torch.version, "xpu", None)
            xpu_version = str(version) if version is not None else None
    else:
        raise ValueError("unsupported engine/backend pair")
    selected_name = names[index] if available and index < count else None
    return {
        "schema_version": 1,
        "profile_id": request["id"],
        "engine": engine,
        "backend": backend,
        "device_index": index,
        "python": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "sys_base_prefix": str(Path(sys.base_prefix).resolve()),
        "environment_root": str(Path(sys.prefix).resolve())
        if sys.prefix != sys.base_prefix
        else None,
        "torch_path": str(Path(torch.__file__).resolve()) if torch else None,
        "torch_version": torch.__version__ if torch else None,
        "torch_hip": torch.version.hip if torch else None,
        "torch_cuda": torch.version.cuda if torch else None,
        "xpu_runtime_version": xpu_version,
        "framework_path": framework_path,
        "framework_version": framework_version,
        "available": available,
        "device_count": count,
        "device_name": selected_name,
        "device_names": names,
        "bf16_supported": bf16,
        **source_identity(),
        **_host_environment(),
        "platform": platform.platform(),
    }


def main() -> None:
    try:
        request = json.loads(sys.stdin.buffer.read(4097))
        if not isinstance(request, dict):
            raise ValueError("request must be an object")  # noqa: TRY004 - invalid JSON request
        answer: dict[str, object] = probe(request)
    except Exception as error:  # noqa: BLE001 - subprocess protocol must report vendor errors as JSON
        answer = {
            "schema_version": 1,
            "error": f"{type(error).__name__}: {error}"[:1024],
        }
    sys.stdout.write(json.dumps(answer, ensure_ascii=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
