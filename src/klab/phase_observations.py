"""Native, opt-in operational phase-observation envelopes.

This module deliberately sits outside all artifact and scientific identity domains.
It validates the report destination before command admission, then owns one observer
and one atomic report publication around an already-selected command handler.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from klab.bottleneck_observations import BottleneckObserver
from klab.canonical import canonical_json
from klab.config.models import RunConfig
from klab.result_publication import write_bytes_exclusive
from klab.training.manifest import sha256_file


def _path(value: Path | str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.absolute()


def _components_are_safe(path: Path) -> None:
    if ".." in path.parts or any(part == "" for part in path.parts):
        raise ValueError("unsafe observations output path")
    for component in (path.parent, *path.parent.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked observations output parent: {component}")


def _within(path: Path, root: Path) -> bool:
    return path.is_relative_to(root)


def _reject_protected(destination: Path, protected: set[Path]) -> None:
    for root in protected:
        root = root.resolve()
        if destination == root or _within(destination, root):
            raise ValueError(
                f"observations output must not be inside inventoried path: {root}"
            )


def _run_config_inventory(config: RunConfig) -> tuple[set[Path], set[Path]]:
    roots = {_path(config.dataset.cache_dir), _path(config.logging.root_dir)}
    files = {_path(config.tokenizer.path)}
    for field in ("train_path", "validation_path", "allocation_manifest_path"):
        value = getattr(config.dataset, field)
        if value is not None:
            files.add(_path(value))
    for field in ("source_manifest_path", "corpus_release_path", "corpus_export_path"):
        value = getattr(config.dataset, field)
        if value is not None:
            path = _path(value)
            files.add(path)
            roots.add(path if path.is_dir() else path.parent)
    return roots, files


def _config_inventory(config_path: Path) -> tuple[set[Path], set[Path]]:
    from klab.config import load_config

    roots, files = _run_config_inventory(load_config(config_path))
    files.add(config_path)
    return roots, files


def _config_paths(args: Any) -> tuple[set[Path], set[Path]]:
    """Return protected roots and exact files for direct native commands."""
    config_value = getattr(args, "config", None)
    if config_value is None:
        return set(), set()
    roots, files = _config_inventory(_path(config_value))
    prepared = getattr(args, "prepared_inputs", None)
    if prepared is not None:
        roots.add(_path(prepared))
    output = getattr(args, "output", None)
    if output is not None:
        roots.add(_path(output))
    return roots, files


def _operation(args: Any) -> str | None:
    if (
        getattr(args, "command", None) == "data"
        and getattr(args, "data_command", None) == "prepare"
    ):
        return "data.prepare"
    if getattr(args, "command", None) == "stage":
        return "stage"
    return None


def validate_observations_destination(args: Any) -> None:
    """Admit an observation destination without creating any runtime state.

    The main CLI calls this before runtime selection and work-root creation.  A
    successful validation records only resolved operational metadata on ``args``.
    """
    value = getattr(args, "observations_output", None)
    if value is None:
        return
    operation = _operation(args)
    if operation is None:
        raise ValueError("--observations-output is unsupported for this command")
    destination = _path(value)
    if destination.suffix != ".json":
        raise ValueError("observations output must have a .json suffix")
    _components_are_safe(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"observations output already exists: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("observations output parent must already exist")
    roots, files = _config_paths(args)
    _reject_protected(destination, roots)
    if destination in files:
        raise ValueError("observations output must not replace an input file")
    args.observations_output = destination
    args._phase_observation_operation = operation
    args._phase_observation_input = _path(args.config)


def set_observation_runtime(
    args: Any,
    *,
    requested: object,
    resolved: object | None,
    reason: str | None,
    observation_scope: str,
) -> None:
    """Attach already-resolved native runtime metadata to an active session.

    Callers must pass only runtime evidence they already selected or verified;
    this adapter never probes or infers a device from the observation host.
    """
    profile = getattr(args, "runtime_profile_loaded", None)
    selected: object | None = (
        profile.model_dump(mode="json")
        if profile is not None
        else {"id": args.runtime}
        if getattr(args, "runtime", None) is not None
        else None
    )
    args._phase_observation_runtime = {
        "requested": (
            {"configuration": requested, "selection": selected}
            if selected is not None
            else requested
        ),
        "resolved": resolved,
        "reason": reason,
        "observation_scope": observation_scope,
    }


def _runtime(args: Any) -> dict[str, object]:
    supplied = getattr(args, "_phase_observation_runtime", None)
    if supplied is not None:
        return supplied
    profile = getattr(args, "runtime_profile_loaded", None)
    command = getattr(args, "command", None)
    if command == "data":
        from klab.config import load_config

        config = load_config(_path(args.config))
        return {
            "requested": config.runtime.model_dump(mode="json"),
            "resolved": None,
            "reason": "preparation_executes_on_host",
            "observation_scope": "host_process_tree",
        }
    if profile is not None:
        return {
            "requested": profile.model_dump(mode="json"),
            "resolved": None,
            "reason": "execution_runtime_not_resolved",
            "observation_scope": "host_process_tree",
        }
    requested = getattr(args, "runtime", None)
    return {
        "requested": {"id": requested} if requested is not None else None,
        "resolved": None,
        "reason": "runtime_not_resolved",
        "observation_scope": "host_process_tree",
    }


def _unavailable_reasons(
    records: list[dict[str, object]], observer: BottleneckObserver
) -> dict[str, str]:
    reasons: dict[str, str] = {}
    if not records:
        reasons["records"] = "counter_unavailable_or_incomplete"
    for index, record in enumerate(records):
        for field in (
            "cpu_user_seconds",
            "cpu_system_seconds",
            "process_read_bytes",
            "process_write_bytes",
            "observed_tree_rss_max_bytes",
            "observed_tree_swap_max_bytes",
            "host_available_ram_bytes",
        ):
            if record.get(field) is None:
                reasons[f"records[{index}].{field}"] = (
                    "counter_unavailable_or_incomplete"
                )
        if record.get("accelerator_utilization_percent") is None:
            reasons[f"records[{index}].accelerator_utilization_percent"] = (
                "no_accelerator_probe"
                if observer._accelerator_probe is None
                else "counter_unavailable_or_incomplete"
            )
    return reasons


def _snapshot_statistics() -> dict[str, int | float] | None:
    try:
        from klab.data.sources import snapshot_verification_statistics

        return snapshot_verification_statistics()
    except ImportError:
        return None


@contextmanager
def observation_session(args: Any) -> Iterator[BottleneckObserver | None]:
    """Own optional observer lifetime, outer phase and best-effort publication.

    Exceptions from the wrapped command are never replaced by report failures.
    ``args.observer`` is set only while the opted-in session is active.
    """
    destination = getattr(args, "observations_output", None)
    if destination is None:
        yield None
        return
    try:
        observer = BottleneckObserver()
    except Exception as error:  # noqa: BLE001 -- diagnostics are optional after admission
        print(
            f"klab: warning: phase observer setup unavailable: {error}",
            file=sys.stderr,
        )
        observer = None
    args.observer = observer
    status = "completed"
    try:
        with (
            observer.phase(args._phase_observation_operation)
            if observer is not None
            else nullcontext()
        ):
            yield observer
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException:
        status = "failed"
        raise
    finally:
        try:
            source = args._phase_observation_input
            report = {
                "format": "klab-phase-observations-v1",
                "operation": args._phase_observation_operation,
                "input": {"path": str(source), "file_sha256": sha256_file(source)},
                "runtime": _runtime(args),
                "observation_host": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                    "pid": os.getpid(),
                },
                "status": status,
                "records": observer.records if observer is not None else [],
                "snapshot_verification": _snapshot_statistics(),
                "unavailable_reasons": (
                    _unavailable_reasons(observer.records, observer)
                    if observer is not None
                    else {"records": "counter_unavailable_or_incomplete"}
                ),
            }
            write_bytes_exclusive(destination, canonical_json(report) + b"\n")
        except BaseException as error:  # noqa: BLE001 -- diagnostic publication cannot replace the workflow result
            print(
                f"klab: warning: could not publish phase observations: {error}",
                file=sys.stderr,
            )
        finally:
            delattr(args, "observer")
