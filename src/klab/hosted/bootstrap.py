"""Fixed stdlib bootstrap programs for already user-provisioned hosts."""

from __future__ import annotations

import json
from pathlib import Path

from klab import runtime_env_subprocess
from klab.hosted import bootstrap_worker


def _program(action: str, payload: dict, timeout: float, *, before: str = "") -> str:
    # Both sources are versioned stdlib-only implementations, not user programs.
    # Install the worker first, then its helper without dynamic global lookup.
    bounded = Path(runtime_env_subprocess.__file__).read_text(encoding="utf-8")
    worker = Path(bootstrap_worker.__file__).read_text(encoding="utf-8")
    data = json.dumps({**payload, "timeout": timeout}, sort_keys=True, allow_nan=False)
    return (
        "import json\n_klab_bootstrap = {'__name__': 'klab_hosted_bootstrap'}\n"
        + f"exec(compile({worker!r}, '<klab-bootstrap>', 'exec'), _klab_bootstrap)\n"
        + f"exec(compile({bounded!r}, '<klab-bounded>', 'exec'), _klab_bootstrap)\n"
        + f"_klab_bootstrap['_deadline'] = _klab_bootstrap['time'].monotonic() + {timeout!r}\n"
        + "try:\n"
        + f" exec(compile({before!r}, '<klab-transfer>', 'exec'), {{'__name__': 'klab_transfer', '_klab_deadline': _klab_bootstrap['_deadline']}})\n"
        + f" _klab_payload = json.loads({data!r})\n"
        + " _klab_payload['timeout'] = _klab_bootstrap['remaining']()\n"
        + f" _klab_result = _klab_bootstrap['dispatch']({action!r}, _klab_payload)\n"
        + " print(json.dumps(_klab_result, sort_keys=True, allow_nan=False))\n"
        + "except Exception as _klab_error:\n"
        + " print(json.dumps({'hosted_error_version': 1, 'error_type': type(_klab_error).__name__, 'message': str(_klab_error)[:4096]}, sort_keys=True))\n"
        + " raise\n"
    )


def diagnostic_program(root: str, *, timeout: float = 60) -> str:
    return _program("inspect", {"root": root}, timeout)


def preflight_program(
    *,
    root: str,
    runtime_root: str,
    source_bytes: int,
    source_commit: str,
    source_sha256: str,
    recipe: str,
    timeout: float,
    instance_id: str | None = None,
) -> str:
    return _program(
        "preflight",
        {
            "root": root,
            "runtime_root": runtime_root,
            "source_bytes": source_bytes,
            "source_commit": source_commit,
            "source_sha256": source_sha256,
            "recipe": recipe,
            "instance_id": instance_id,
        },
        timeout,
    )


def setup_program(
    *,
    root: str,
    runtime_root: str,
    source_bundle: str,
    source_commit: str,
    source_sha256: str,
    recipe: str,
    instance_id: str,
    timeout: float,
    assembly: str = "",
) -> str:
    return _program(
        "setup",
        {
            "root": root,
            "runtime_root": runtime_root,
            "source_bundle": source_bundle,
            "source_commit": source_commit,
            "source_sha256": source_sha256,
            "recipe": recipe,
            "instance_id": instance_id,
        },
        timeout,
        before=assembly,
    )
