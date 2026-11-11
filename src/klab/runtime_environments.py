"""Machine-local interpreter registry; never part of scientific identity."""

from __future__ import annotations

import fcntl
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from klab.config.models import StrictModel
from klab.runtime_profile import RuntimeProfile, RuntimeRequirements
from klab.workdir import resolve_work_dir, storage_checks

_LOGGER = logging.getLogger(__name__)

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def validate_id(identifier: str) -> str:
    if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
        raise ValueError("runtime id must be a safe identifier")
    return identifier


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not str(value).strip() or not path.is_absolute():
        raise ValueError(f"{label} must be an absolute nonblank path")
    return path


def resolve_runtime_dir(path: Path | None = None) -> Path:
    configured = path if path is not None else os.environ.get("KLAB_RUNTIME_DIR")
    if configured is not None:
        root = _absolute(configured, "runtime root")
    else:
        xdg = os.environ.get("XDG_DATA_HOME", "")
        base = (
            Path(xdg)
            if xdg.strip() and Path(xdg).is_absolute()
            else Path.home() / ".local/share"
        )
        root = base / "klab/runtimes"
    root = root.resolve()
    if storage_checks(root):
        raise ValueError("runtime root must not be inside a Git checkout")
    if root == resolve_work_dir():
        raise ValueError("runtime root must differ from scientific work root")
    return root


def resolve_registry_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    base = (
        Path(xdg)
        if xdg.strip() and Path(xdg).is_absolute()
        else Path.home() / ".config"
    )
    return (base / "klab/runtimes.yaml").resolve()


class RuntimeEntry(StrictModel):
    python: Path
    engine: Literal["pytorch", "mlx"]
    backend: Literal["cpu", "rocm", "cuda", "mps", "xpu", "metal"]
    device_index: int = Field(ge=0, strict=True)
    requirements: RuntimeRequirements = RuntimeRequirements()

    @field_validator("python", mode="before")
    @classmethod
    def absolute_python(cls, value: object) -> Path:
        if not isinstance(value, (str, Path)):
            raise ValueError("python must be an absolute path")  # noqa: TRY004 - Pydantic validation error
        return _absolute(value, "python")

    @model_validator(mode="after")
    def compatible_pair(self) -> RuntimeEntry:
        if (self.engine == "mlx") != (self.backend == "metal"):
            raise ValueError("MLX requires metal; metal requires MLX")
        if self.backend in {"cpu", "mps", "metal"} and self.device_index != 0:
            raise ValueError(f"{self.backend} supports only device_index=0")
        if self.requirements.torch_hip is True and self.backend != "rocm":
            raise ValueError("torch_hip true requires ROCm backend")
        return self


class RuntimeRegistry(StrictModel):
    runtime_registry_version: Literal[1] = 1
    runtimes: dict[str, RuntimeEntry] = Field(default_factory=dict)

    @field_validator("runtime_registry_version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("unsupported runtime registry version")
        return value

    @field_validator("runtimes", mode="before")
    @classmethod
    def safe_ids(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError("runtimes must be a mapping")  # noqa: TRY004 - Pydantic validation error
        for identifier in value:
            validate_id(identifier)
        return value


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: _UniqueLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ValueError("registry keys must be scalar") from error
        if duplicate:
            raise ValueError(f"duplicate registry key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def read_registry(path: Path | None = None) -> RuntimeRegistry:
    selected = resolve_registry_path() if path is None else path
    try:
        raw = yaml.load(selected.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except FileNotFoundError:
        return RuntimeRegistry()
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read runtime registry {selected}: {error}") from error
    if not isinstance(raw, dict) or set(raw) != {
        "runtime_registry_version",
        "runtimes",
    }:
        raise ValueError("registry must declare runtime_registry_version and runtimes")
    return RuntimeRegistry.model_validate(raw)


def profile_for_id(
    identifier: str, *, registry_path: Path | None = None
) -> RuntimeProfile:
    validate_id(identifier)
    entry = read_registry(registry_path).runtimes.get(identifier)
    if entry is None:
        raise ValueError(f"unknown runtime id: {identifier}")
    return RuntimeProfile.model_validate(
        {"runtime_profile_version": 1, "id": identifier, **entry.model_dump()}
    )


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(
        path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        os.fchmod(fd, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _LOGGER.warning(
                "Waiting for runtime registry lock %s; preserve the lock and retained "
                "state. Owner identity is unknown; normal blocking wait continues.",
                path.with_suffix(".lock"),
            )
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _publish(path: Path, registry: RuntimeRegistry) -> None:
    fd, name = tempfile.mkstemp(prefix=".runtimes-", suffix=".yaml", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            yaml.safe_dump(registry.model_dump(mode="json"), stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def register_runtime(
    identifier: str, entry: RuntimeEntry, *, registry_path: Path | None = None
) -> None:
    validate_id(identifier)
    path = resolve_registry_path() if registry_path is None else registry_path
    with _locked(path):
        registry = read_registry(path)
        if identifier in registry.runtimes:
            raise ValueError(f"runtime id already registered: {identifier}")
        _publish(
            path, RuntimeRegistry(runtimes={**registry.runtimes, identifier: entry})
        )


def unregister_runtime(identifier: str, *, registry_path: Path | None = None) -> None:
    validate_id(identifier)
    path = resolve_registry_path() if registry_path is None else registry_path
    with _locked(path):
        registry = read_registry(path)
        if identifier not in registry.runtimes:
            raise ValueError(f"unknown runtime id: {identifier}")
        _publish(
            path,
            RuntimeRegistry(
                runtimes={
                    key: value
                    for key, value in registry.runtimes.items()
                    if key != identifier
                }
            ),
        )
