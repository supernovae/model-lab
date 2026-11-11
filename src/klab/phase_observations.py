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
from klab.experiments.lock import _exclusive_bytes
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


def _tokenizer_config_inventory(config_path: Path) -> tuple[set[Path], set[Path]]:
    from klab.config import load_tokenizer_config

    config = load_tokenizer_config(config_path)
    roots = {_path(config.dataset.cache_dir), _path(config.output_dir)}
    files = {config_path}
    for field in ("train_path", "validation_path", "source_manifest_path"):
        value = getattr(config.dataset, field)
        if value is not None:
            path = _path(value)
            files.add(path)
            if field == "source_manifest_path":
                roots.add(path if path.is_dir() else path.parent)
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


def _campaign_paths(args: Any) -> tuple[set[Path], set[Path]]:
    """Resolve Campaign paths through its typed declaration helpers."""
    from klab.campaign.plan import (
        ArtifactReference,
        CorpusRelease,
        DataPrepare,
        DatasetSnapshot,
        ExperimentPlanStage,
        TokenizerReference,
        TokenizerTrain,
        load_campaign,
        operational_path,
        safe_path,
    )
    from klab.campaign.state import CampaignStore
    from klab.workdir import resolve_work_dir

    source = _path(args.source)
    plan = load_campaign(source)
    base = source.parent
    roots = {CampaignStore(plan, resolve_work_dir(args.work_dir)).root.absolute()}
    files = {source}
    for stage in plan.stages:
        if isinstance(stage, DatasetSnapshot):
            roots.update(
                {
                    operational_path(base, stage.output).absolute(),
                    operational_path(base, stage.cache_dir).absolute(),
                }
            )
            files.add(operational_path(base, stage.lock).absolute())
        elif isinstance(stage, (ArtifactReference, TokenizerReference)):
            assert stage.artifact.path is not None
            artifact = operational_path(base, stage.artifact.path).absolute()
            files.add(artifact)
            roots.add(artifact if artifact.is_dir() else artifact.parent)
        elif isinstance(stage, CorpusRelease):
            project = safe_path(base, stage.project).absolute()
            files.add(project)
            roots.add(resolve_work_dir(args.work_dir) / "corpora")
        elif isinstance(stage, (TokenizerTrain, DataPrepare)):
            config = safe_path(base, stage.config).absolute()
            config_roots, config_files = (
                _tokenizer_config_inventory(config)
                if isinstance(stage, TokenizerTrain)
                else _config_inventory(config)
            )
            roots.update(config_roots)
            files.update(config_files)
        elif isinstance(stage, ExperimentPlanStage):
            from klab.experiments.plan import base_run_config, load_plan
            from klab.recovery.provenance import declaration_reference

            source_plan = safe_path(base, stage.source).absolute()
            files.add(source_plan)
            experiment = load_plan(source_plan)
            config_roots, config_files = _run_config_inventory(
                base_run_config(experiment, source_plan)
            )
            roots.update(config_roots)
            files.update(config_files)
            if isinstance(experiment.base_run, str):
                files.add(declaration_reference(source_plan, experiment.base_run))
            for artifact in experiment.artifacts.values():
                if artifact.path is not None:
                    path = operational_path(
                        source_plan.parent, artifact.path
                    ).absolute()
                    files.add(path)
                    roots.add(path if path.is_dir() else path.parent)
            if stage.lock is not None:
                lock = operational_path(base, stage.lock).absolute()
                files.add(lock)
                roots.add(lock if lock.is_dir() else lock.parent)
        else:
            for field in ("project", "suite", "panel", "policy"):
                value = getattr(stage, field, None)
                if value is not None:
                    files.add(safe_path(base, value).absolute())
            for field in ("measurement_receipt", "review"):
                value = getattr(stage, field, None)
                if value is not None:
                    path = operational_path(base, value).absolute()
                    files.add(path)
                    roots.add(path if path.is_dir() else path.parent)
    return roots, files


def _operation(args: Any) -> str | None:
    if (
        getattr(args, "command", None) == "data"
        and getattr(args, "data_command", None) == "prepare"
    ):
        return "data.prepare"
    if getattr(args, "command", None) == "stage":
        return "stage"
    if getattr(args, "command", None) == "campaign" and getattr(
        args, "campaign_command", None
    ) in {"plan", "status", "next", "explain", "apply", "resume"}:
        return f"campaign.{args.campaign_command}"
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
    roots, files = (
        _campaign_paths(args)
        if getattr(args, "command", None) == "campaign"
        else _config_paths(args)
    )
    _reject_protected(destination, roots)
    if destination in files:
        raise ValueError("observations output must not replace an input file")
    args.observations_output = destination
    args._phase_observation_operation = operation
    args._phase_observation_input = (
        _path(args.source)
        if getattr(args, "command", None) == "campaign"
        else _path(args.config)
    )


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
            "observation_scope": (
                "controller_host_process_tree"
                if command == "campaign"
                else "host_process_tree"
            ),
        }
    requested = getattr(args, "runtime", None)
    return {
        "requested": {"id": requested} if requested is not None else None,
        "resolved": None,
        "reason": "runtime_not_resolved",
        "observation_scope": (
            "controller_host_process_tree"
            if command == "campaign"
            else "host_process_tree"
        ),
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
            _exclusive_bytes(destination, canonical_json(report) + b"\n")
        except BaseException as error:  # noqa: BLE001 -- diagnostic publication cannot replace the workflow result
            print(
                f"klab: warning: could not publish phase observations: {error}",
                file=sys.stderr,
            )
        finally:
            delattr(args, "observer")
