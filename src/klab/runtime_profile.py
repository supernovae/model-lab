"""Operational interpreter selection and non-portable accelerator authorization."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from klab.config.loading import _load
from klab.config.models import StrictModel
from klab.runtime_identity_probe import source_identity

_PROVISION = (
    "profile python cannot import compatible klab/torch; provision the vendor "
    "environment and install this source before retrying"
)
_TOKEN = object()
_TIMEOUT = 30
_OUTPUT_LIMIT = 32768
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class RuntimeAuthorizationRequired(ValueError):
    """Execution needs an explicitly authorized interpreter and device."""


class RuntimeRequirements(StrictModel):
    torch_hip: bool | None = None
    bf16: bool | None = None
    device_name_regex: str | None = None

    @field_validator("torch_hip", "bf16", mode="before")
    @classmethod
    def strict_bool(cls, value: bool | None) -> bool | None:
        if value is not None and type(value) is not bool:
            raise ValueError("requirement must be a boolean or null")
        return value

    @field_validator("device_name_regex")
    @classmethod
    def valid_regex(cls, value: str | None) -> str | None:
        if value is not None:
            if not value or len(value) > 1024:
                raise ValueError("device_name_regex must contain 1..1024 characters")
            try:
                re.compile(value)
            except re.error as error:
                raise ValueError(f"invalid device_name_regex: {error}") from error
        return value


class RuntimeProfile(StrictModel):
    runtime_profile_version: Literal[1]
    id: str
    python: Path
    engine: Literal["pytorch", "mlx"]
    backend: Literal["cpu", "rocm", "cuda", "mps", "xpu", "metal"]
    device_index: int = Field(ge=0, strict=True)
    requirements: RuntimeRequirements = RuntimeRequirements()

    @field_validator("runtime_profile_version", mode="before")
    @classmethod
    def exact_version(cls, value: int) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("unsupported runtime profile version")
        return value

    @field_validator("id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if not _IDENT.fullmatch(value):
            raise ValueError("profile id must be a safe identifier")
        return value

    @field_validator("python")
    @classmethod
    def executable_python(cls, value: Path) -> Path:
        if (
            not value.is_absolute()
            or not value.is_file()
            or not os.access(value, os.X_OK)
        ):
            raise ValueError("profile python must be an absolute executable file")
        return value

    @model_validator(mode="after")
    def compatible_pair(self) -> RuntimeProfile:
        if (self.engine == "mlx") != (self.backend == "metal"):
            raise ValueError("MLX requires metal; metal requires MLX")
        if self.backend in {"cpu", "mps", "metal"} and self.device_index != 0:
            raise ValueError(f"{self.backend} supports only device_index=0")
        if self.requirements.torch_hip is True and self.backend != "rocm":
            raise ValueError("torch_hip true requires ROCm backend")
        return self


def load_runtime_profile(path: Path) -> RuntimeProfile:
    """Strict YAML loading; interpreter path is deliberately never relativized."""
    return _load(path, RuntimeProfile)


def _limit_probe_output() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024, 64 * 1024))


def _probe(
    python: Path,
    *,
    identifier: str,
    engine: str,
    backend: str,
    index: int,
) -> dict[str, object]:
    from klab.device_visibility import unconfined_environment

    request = json.dumps(
        {"id": identifier, "engine": engine, "backend": backend, "device_index": index},
        separators=(",", ":"),
    ).encode()
    try:
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.run(
                [str(python), "-m", "klab.runtime_identity_probe"],
                input=request,
                # The devices as the host has them, not as a confined
                # training process sees them.
                env=unconfined_environment(),
                stdout=stdout,
                stderr=stderr,
                timeout=_TIMEOUT,
                check=False,
                preexec_fn=_limit_probe_output if os.name == "posix" else None,
            )
            stdout.seek(0)
            data = stdout.read(_OUTPUT_LIMIT + 1)
            stderr.seek(0)
            error_data = stderr.read(1025)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"{_PROVISION}: {error}") from error
    if process.returncode or len(data) > _OUTPUT_LIMIT or len(error_data) > 1024:
        raise ValueError(
            f"{_PROVISION}: subprocess status {process.returncode}; "
            f"{error_data[:1024].decode('utf-8', 'replace')}"
        )
    try:
        result = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{_PROVISION}: invalid JSON output") from error
    if (
        not isinstance(result, dict)
        or result.get("schema_version") != 1
        or "error" in result
    ):
        raise ValueError(
            f"{_PROVISION}: {str(result.get('error'))[:1024] if isinstance(result, dict) else 'invalid response'}"
        )
    expected = {
        "profile_id": identifier,
        "engine": engine,
        "backend": backend,
        "device_index": index,
        "python": str(python.resolve()),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f"runtime probe {key} mismatch: expected {value!r}, got {result.get(key)!r}"
            )
    local = source_identity()
    if any(result.get(key) != value for key, value in local.items()):
        raise ValueError(
            "profile python Model Lab package/source identity differs from this checkout"
        )
    return result


def probe_runtime_profile(profile: RuntimeProfile) -> dict[str, object]:
    return _probe(
        profile.python,
        identifier=profile.id,
        engine=profile.engine,
        backend=profile.backend,
        index=profile.device_index,
    )


def _backend(config: Any) -> tuple[str, str, int]:
    from klab.runtime import _auto_backend

    runtime = config.runtime
    backend = (
        "metal"
        if runtime.engine == "mlx"
        else (_auto_backend() if runtime.backend == "auto" else runtime.backend)
    )
    return runtime.engine, backend, runtime.device_index


def _match(
    config: Any, engine: str, backend: str, index: int, probe: dict[str, object]
) -> None:
    selection = _backend(config)
    observed = (probe.get("engine"), probe.get("backend"), probe.get("device_index"))
    if selection != (engine, backend, index) or observed != (engine, backend, index):
        raise ValueError(
            f"runtime selection {selection} differs from authorized {(engine, backend, index)} "
            f"or probed {observed}"
        )
    if (
        probe.get("available") is not True
        or type(probe.get("device_count")) is not int
        or index >= probe["device_count"]
    ):
        raise ValueError(
            f"selected {backend}:{index} is unavailable under the requested interpreter"
        )
    if backend in {"cuda", "rocm"} and bool(probe.get("torch_hip")) != (
        backend == "rocm"
    ):
        raise ValueError(f"{backend} requires a matching CUDA/HIP Torch build")
    if backend == "cuda" and not probe.get("torch_cuda"):
        raise ValueError("cuda requires a matching CUDA Torch build")


class RuntimeAuthorization:
    """In-process sealed value; its JSON rendering is evidence, never authority."""

    __slots__ = ("_evidence_json", "_identity", "_token")

    def __init__(
        self, token: object, evidence: dict[str, object], identity: dict[str, str]
    ) -> None:
        if token is not _TOKEN:
            raise TypeError(
                "RuntimeAuthorization may only be created by a verified probe"
            )
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_evidence_json", json.dumps(evidence, sort_keys=True))
        object.__setattr__(
            self, "_identity", (identity["package_root"], identity["source_sha256"])
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("RuntimeAuthorization is frozen")

    def as_dict(self) -> dict[str, object]:
        return json.loads(self._evidence_json)


def _seal(
    kind: str,
    descriptor: dict[str, object],
    probe: dict[str, object],
    *,
    tested_runtime: dict[str, object] | None = None,
) -> RuntimeAuthorization:
    evidence: dict[str, object] = {
        "authorization_version": 1,
        "kind": kind,
        "descriptor": descriptor,
        "probe": probe,
    }
    if tested_runtime is not None:
        evidence["tested_runtime"] = tested_runtime
    return RuntimeAuthorization(
        _TOKEN,
        evidence,
        {
            "package_root": str(probe["package_root"]),
            "source_sha256": str(probe["source_sha256"]),
        },
    )


def _check_current(probe: dict[str, object], python: Path) -> None:
    if Path(sys.executable).resolve() != python.resolve():
        raise ValueError(
            "profile/worker Python differs from running interpreter; execute under selected python"
        )
    from klab.runtime_identity_probe import DEVICE_FIELDS
    from klab.runtime_identity_probe import probe as local_probe

    # The devices were read by the subprocess probe; asking again here would
    # fix which devices this process sees before it is confined to its own.
    current = local_probe(
        {
            "id": probe["profile_id"],
            "engine": probe["engine"],
            "backend": probe["backend"],
            "device_index": probe["device_index"],
        },
        devices=False,
    )
    if {key: value for key, value in current.items() if key not in DEVICE_FIELDS} != {
        key: value for key, value in probe.items() if key not in DEVICE_FIELDS
    }:
        raise ValueError(
            "selected interpreter/Torch/device/source identity changed since probe"
        )
    # A fresh process without this one's confinement sees the devices as the
    # authorizing probe did.
    devices = _probe(
        python,
        identifier=str(probe["profile_id"]),
        engine=str(probe["engine"]),
        backend=str(probe["backend"]),
        index=int(probe["device_index"]),
    )
    if {key: devices.get(key) for key in DEVICE_FIELDS} != {
        key: probe.get(key) for key in DEVICE_FIELDS
    }:
        raise ValueError(
            "selected interpreter/Torch/device/source identity changed since probe"
        )


def authorize_profile(profile: RuntimeProfile, config: Any) -> RuntimeAuthorization:
    probe = probe_runtime_profile(profile)
    _match(config, profile.engine, profile.backend, profile.device_index, probe)
    requirements = profile.requirements
    if requirements.torch_hip is not None:
        hip = probe.get("torch_hip")
        if hip is not None and not isinstance(hip, str):
            raise ValueError("Torch HIP runtime version is malformed")
        if bool(hip) != requirements.torch_hip:
            raise ValueError(
                "runtime profile torch_hip requirement does not match Torch build"
            )
    if requirements.device_name_regex is not None:
        name = probe.get("device_name")
        if (
            not isinstance(name, str)
            or re.search(requirements.device_name_regex, name) is None
        ):
            raise ValueError(
                "runtime profile device name requirement does not match selected device"
            )
    _check_current(probe, profile.python)
    if requirements.bf16 is False and probe.get("bf16_supported") is not False:
        raise ValueError(
            "BF16 unsupported requirement cannot be proven by this runtime"
        )
    if requirements.bf16 is False and config.runtime.precision == "bf16":
        raise ValueError("requested BF16 conflicts with profile bf16=false requirement")
    authorization = _seal("profile", profile.model_dump(mode="json"), probe)
    if requirements.bf16 is True or config.runtime.precision == "bf16":
        if not hasattr(config, "model_copy"):
            raise ValueError(
                "BF16 requirement needs a complete RunConfig for disposable optimizer validation"
            )
        from klab.runtime import validate_runtime

        bf16_config = config.model_copy(
            update={"runtime": config.runtime.model_copy(update={"precision": "bf16"})}
        )
        tested = validate_runtime(bf16_config, authorization=authorization)
        if "bf16" not in tested.tested_precisions:
            raise ValueError(
                "BF16 optimizer update was not verified on selected runtime"
            )
        authorization = _seal(
            "profile",
            profile.model_dump(mode="json"),
            probe,
            tested_runtime=tested.as_dict(),
        )
    return authorization


def authorize_worker(definition: Any, config: Any) -> RuntimeAuthorization:
    if (
        not isinstance(definition.python, Path)
        or not definition.python.is_absolute()
        or not definition.python.is_file()
        or not os.access(definition.python, os.X_OK)
    ):
        raise ValueError(
            "registered worker requires explicit executable absolute Python"
        )
    if (definition.engine == "mlx") != (definition.backend == "metal"):
        raise ValueError("worker engine/backend mismatch")
    probe = _probe(
        definition.python,
        identifier=definition.worker_id,
        engine=definition.engine,
        backend=definition.backend,
        index=definition.device_index,
    )
    _match(
        config, definition.engine, definition.backend, definition.device_index, probe
    )
    _check_current(probe, definition.python)
    return _seal("worker", definition.model_dump(mode="json"), probe)


def rederive_authorization(
    evidence: dict[str, object], config: Any
) -> RuntimeAuthorization:
    """A persisted descriptor is only a request to perform a fresh probe."""
    if not isinstance(evidence, dict) or evidence.get("authorization_version") != 1:
        raise ValueError("invalid runtime authorization evidence")
    descriptor = evidence.get("descriptor")
    if not isinstance(descriptor, dict):
        raise ValueError("runtime authorization descriptor missing")  # noqa: TRY004 - malformed serialized evidence
    if evidence.get("kind") == "profile":
        profile = RuntimeProfile.model_validate(descriptor)
        authorization = authorize_profile(profile, config)
    elif evidence.get("kind") == "worker":
        from klab.workers.models import WorkerDefinition

        authorization = authorize_worker(
            WorkerDefinition.model_validate(descriptor), config
        )
    else:
        raise ValueError("unknown runtime authorization origin")
    if evidence.get("probe") != authorization.as_dict()["probe"]:
        raise ValueError(
            "saved runtime identity differs from freshly probed interpreter/device/source"
        )
    return authorization


def require_authorization(
    config: Any, authorization: RuntimeAuthorization | None
) -> None:
    engine, backend, index = _backend(config)
    if backend == "cpu" and engine == "pytorch" and authorization is None:
        return
    if (
        not isinstance(authorization, RuntimeAuthorization)
        or authorization._token is not _TOKEN
    ):
        raise RuntimeAuthorizationRequired(
            f"{backend} requires --runtime ID or an explicit profile "
            "(try/probe: declaration runtime_profile; other commands: "
            "--runtime-profile PATH), or a registered compatible worker"
        )
    evidence = authorization.as_dict()
    probe = evidence["probe"]
    if not isinstance(probe, dict):
        raise ValueError("invalid runtime authorization probe")  # noqa: TRY004 - malformed serialized evidence
    _match(config, engine, backend, index, probe)
    if (
        probe.get("package_root"),
        probe.get("source_sha256"),
    ) != authorization._identity:
        raise ValueError("Model Lab source authorization identity is invalid")
    descriptor = evidence["descriptor"]
    if not isinstance(descriptor, dict):
        raise ValueError("invalid runtime authorization descriptor")  # noqa: TRY004 - malformed serialized evidence
    _check_current(probe, Path(str(descriptor["python"])))
