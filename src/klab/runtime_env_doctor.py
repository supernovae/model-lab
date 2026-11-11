"""Fresh, disposable runtime authorization in the selected interpreter."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

from klab.runtime_env_subprocess import run_bounded
from klab.runtime_identity_probe import source_identity
from klab.runtime_profile import RuntimeProfile, probe_runtime_profile

_OUTPUT_LIMIT = 32768


def tiny_config(profile: RuntimeProfile, *, precision: str | None = None):
    """A typed probe-only configuration; none of its paths are ever opened."""
    from klab.config.models import RunConfig

    return RunConfig.model_validate(
        {
            "schema_version": 2,
            "name": "runtime-environment-doctor",
            "seed": 0,
            "runtime": {
                "engine": profile.engine,
                "backend": profile.backend,
                "device_index": profile.device_index,
                "precision": precision
                if precision is not None
                else ("bf16" if profile.requirements.bf16 is True else "fp32"),
                "memory": {
                    "activation_checkpointing": {"enabled": False},
                    "activation_offload": {"enabled": False},
                },
            },
            "model": {
                "vocab_size": 260,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "ffn_dim": 32,
                "max_seq_len": 16,
            },
            "tokenizer": {"path": "/nonexistent/klab-doctor/tokenizer"},
            "dataset": {
                "source": "synthetic",
                "cache_dir": "/nonexistent/klab-doctor/cache",
                "train_max_documents": 1,
                "validation_max_documents": 1,
                "train_max_tokens": 16,
                "validation_max_tokens": 16,
            },
            "training": {
                "micro_batch_size": 1,
                "seq_len": 16,
                "max_steps": 1,
                "max_tokens": 16,
            },
            "optimizer": {"name": "adamw", "warmup_steps": 0, "state_offload": False},
            "checkpoint": {"every_steps": 1},
            "logging": {"root_dir": "/nonexistent/klab-doctor/logs"},
        }
    )


def source_repair_command(profile: RuntimeProfile | Path) -> str:
    """Return an explicit source-only repair; never execute it automatically."""
    python_path = profile.python if isinstance(profile, RuntimeProfile) else profile
    python = shlex.quote(str(python_path))
    checkout = shlex.quote(str(Path(__file__).resolve().parents[2]))
    return f"uv pip install --python {python} --no-deps --editable {checkout}"


def _envelope(profile: RuntimeProfile) -> dict[str, object]:
    return {
        "runtime_env_doctor_version": 1,
        "id": profile.id,
        "status": "ERROR",
        "profile": profile.model_dump(mode="json"),
        "probe": None,
        "tested_runtime": None,
        "reason": None,
    }


def _failure(result: dict[str, object], error: Exception | str) -> dict[str, object]:
    reason = str(error)[:2048]
    if (
        "package/source identity differs from this checkout" in reason
        or "source identity changed" in reason
    ):
        result["status"] = "SOURCE_MISMATCH"
        if "; repair:" not in reason:
            reason += f"; repair: {source_repair_command(Path(str(result['profile']['python'])))}"
    elif not Path(result["profile"]["python"]).is_file():
        result["status"] = "NOT_PROVISIONED"
    elif any(
        phrase in reason
        for phrase in (
            "No module named",
            "ModuleNotFoundError",
            "cannot import",
            "requested backend unavailable",
            "not available",
            "unavailable for",
            "is unavailable",
            "requires a matching CUDA/HIP Torch build",
            "requires a matching CUDA Torch build",
            "differs from authorized",
            "does not match selected device",
            "does not match Torch build",
        )
    ):
        result["status"] = "UNAVAILABLE"
    result["reason"] = reason
    return result


def _child(
    profile: RuntimeProfile, *, precision: str | None = None
) -> dict[str, object]:
    result = _envelope(profile)
    try:
        from klab.runtime import validate_runtime
        from klab.runtime_profile import authorize_profile

        config = tiny_config(profile, precision=precision)
        result["probe"] = probe_runtime_profile(profile)
        authorization = authorize_profile(profile, config)
        tested = validate_runtime(config, authorization=authorization)
        precision = config.runtime.precision
        if (
            tested.engine != profile.engine
            or tested.backend != profile.backend
            or tested.device_index != profile.device_index
            or precision not in tested.tested_precisions
            or "forward_backward_optimizer" not in tested.tested_features
        ):
            raise ValueError(
                "selected runtime did not verify the requested optimizer update"
            )
        result["tested_runtime"] = tested.as_dict()
        result["status"] = "READY"
    except Exception as error:  # noqa: BLE001 - child must return structured vendor errors
        _failure(result, f"{type(error).__name__}: {error}")
    return result


def doctor(
    profile: RuntimeProfile, *, precision: str | None = None
) -> dict[str, object]:
    """Probe local checkout identity and delegate all optimizer work to profile.python."""
    result = _envelope(profile)
    try:
        parent_probe = probe_runtime_profile(profile)
        result["probe"] = parent_probe
        payload = json.dumps(
            {"profile": profile.model_dump(mode="json"), "precision": precision}
        ).encode("utf-8")
        child = run_bounded(
            [str(profile.python), "-m", "klab.runtime_env_doctor", "--child"],
            input=payload,
            timeout=120,
            output_limit=_OUTPUT_LIMIT,
        )
        if child.returncode:
            raise ValueError(
                f"doctor subprocess status {child.returncode}: "
                f"{child.stderr.decode('utf-8', 'replace')[:1024]}"
            )
        answer = json.loads(child.stdout)
        if not isinstance(answer, dict) or set(answer) != set(result):
            raise ValueError("invalid doctor child response")
        if (
            answer["runtime_env_doctor_version"] != 1
            or answer["id"] != profile.id
            or answer["profile"] != result["profile"]
        ):
            raise ValueError("doctor child profile identity mismatch")
        if answer["probe"] != parent_probe:
            raise ValueError("doctor child probe identity changed since parent probe")
        if any(
            parent_probe.get(key) != value for key, value in source_identity().items()
        ):
            raise ValueError("Model Lab source identity changed during doctor")
        if answer["status"] == "READY":
            tested = answer["tested_runtime"]
            if not isinstance(tested, dict) or (
                tested.get("engine") != profile.engine
                or tested.get("backend") != profile.backend
                or tested.get("device_index") != profile.device_index
                or (
                    precision
                    if precision is not None
                    else ("bf16" if profile.requirements.bf16 is True else "fp32")
                )
                not in tested.get("tested_precisions", [])
                or "forward_backward_optimizer" not in tested.get("tested_features", [])
            ):
                raise ValueError(
                    "doctor child did not verify selected optimizer update"
                )
            if answer["reason"] is not None:
                raise ValueError("READY doctor response contains failure reason")
        else:
            return _failure(result, str(answer["reason"] or "doctor child failed"))
        return answer
    except (
        OSError,
        ValueError,
        TypeError,
        UnicodeError,
        subprocess.TimeoutExpired,
    ) as error:
        return _failure(result, f"{type(error).__name__}: {error}")


def main() -> None:
    """Internal bounded JSON subprocess protocol (not the user-facing CLI)."""
    if sys.argv[1:] != ["--child"]:
        raise SystemExit("runtime_env_doctor is an internal child entry point")
    try:
        payload = sys.stdin.buffer.read(8193)
        if len(payload) > 8192:
            raise ValueError("doctor profile request exceeds 8192 bytes")
        request = json.loads(payload)
        if not isinstance(request, dict) or set(request) != {"profile", "precision"}:
            raise ValueError("invalid doctor request")
        precision = request["precision"]
        if precision not in {None, "fp32", "bf16", "fp16"}:
            raise ValueError("unsupported doctor precision")
        profile = RuntimeProfile.model_validate(request["profile"])
        answer = _child(profile, precision=precision)
    except Exception as error:  # noqa: BLE001 - report malformed protocol request
        answer = {
            "runtime_env_doctor_version": 1,
            "error": f"{type(error).__name__}: {error}"[:1024],
        }
    sys.stdout.write(json.dumps(answer, ensure_ascii=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
