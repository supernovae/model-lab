"""Authored experiment and immutable locked-plan CLI commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from klab.experiments.plan import base_run_config, load_plan
from klab.workdir import resolve_work_dir


def _verification(args: argparse.Namespace) -> dict[str, Any]:
    from klab.verification_proofs import verification_options

    options = getattr(args, "_verification_options", None)
    if options is None:
        options = verification_options(
            resolve_work_dir(), cold=getattr(args, "cold_verify", False)
        )
        args._verification_options = options
    return options


def _emit(args: argparse.Namespace, data: dict[str, Any]) -> None:
    envelope = {
        "format": "klab-experiment-command-v1",
        "command": args.experiment_command,
        **data,
    }
    options = getattr(args, "_verification_options", None)
    if options is not None:
        store = options["proof_store"]
        envelope["verification"] = {
            "mode": options["verification_mode"],
            "proof_hits": 0 if store is None else store.hits,
            "proof_misses": 0 if store is None else store.misses,
            "proof_records": 0 if store is None else store.recorded,
            **(
                {}
                if store is None or not hasattr(store, "diagnostics")
                else store.diagnostics()
            ),
        }
    if args.json:
        print(json.dumps(envelope, sort_keys=True, allow_nan=False, default=str))
    else:
        for key, value in envelope.items():
            print(f"{key}: {json.dumps(value, sort_keys=True, default=str)}")


def _workspace(plan_id: str) -> Path:
    return resolve_work_dir() / "experiments" / plan_id


def _declaration(args: argparse.Namespace) -> tuple[Any, Path]:
    source = Path(args.source).absolute()
    return load_plan(source), source


def _preparation(workspace: Path) -> dict[str, Any] | None:
    path = workspace / "preparation.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("format") != "experiment-preparation-v1"
    ):
        raise ValueError(f"invalid preparation record: {path}")
    return value


def _inspect_declaration(args: argparse.Namespace) -> None:
    from klab.experiments.compiler import expand_axes, flatten_config

    plan, source = _declaration(args)
    if args.experiment_command == "diff" and plan.corpus_variants:
        from klab.experiments.lock import resolve_plan

        prepared = _preparation(_workspace(plan.id))
        if prepared is None:
            raise ValueError("prepare corpus variants before resolving comparisons")
        locked = resolve_plan(
            plan,
            source,
            prepared=prepared,
            max_runs=args.max_runs,
            **_verification(args),
        )
        _emit(
            args,
            {
                "id": plan.id,
                "scientific_sha256": locked.scientific_sha256,
                "comparisons": [
                    comparison.model_dump(mode="json")
                    for comparison in locked.comparisons
                ],
                "executable": False,
            },
        )
        return
    config = base_run_config(plan, source)
    axes = plan.axes
    if any(
        key.startswith("inputs.")
        for axis in axes
        for choice in axis.choices
        for key in choice.set
    ):
        # Typed selections are resolved by the lock compiler, never patched as RunConfig fields.
        axes = tuple(
            axis.model_copy(
                update={
                    "choices": tuple(
                        choice.model_copy(
                            update={
                                "set": {
                                    key: value
                                    for key, value in choice.set.items()
                                    if not key.startswith("inputs.")
                                }
                            }
                        )
                        for choice in axis.choices
                    )
                }
            )
            for axis in axes
        )
    cells = expand_axes(config, axes, max_runs=args.max_runs, base_dir=source.parent)
    payload: dict[str, Any] = {
        "id": plan.id,
        "plan_version": plan.plan_version,
        "cells": [
            {
                "coordinate": coordinate,
                "effective": flatten_config(cell, include_derived=True),
            }
            for cell, coordinate in cells
        ],
        "inputs": sorted(plan.inputs),
        "unresolved": ["external artifact identities", "selected runtime"]
        if plan.inputs
        or config.runtime.backend == "auto"
        or config.runtime.precision == "auto"
        else [],
        "executable": False,
    }
    if args.experiment_command == "inspect":
        from klab.experiments.artifacts import verify_prepared_artifact
        from klab.experiments.lock import storage_preview

        prepared_receipt = None
        prepared_names = {
            name
            for name in plan.inputs.values()
            if plan.artifacts[name].kind == "prepared_data"
            and plan.artifacts[name].from_phase is None
        }
        typed_choices = any(
            key.startswith("inputs.")
            for axis in plan.axes
            for choice in axis.choices
            for key in choice.set
        )
        if len(prepared_names) == 1 and not typed_choices:
            prepared_receipt = verify_prepared_artifact(
                plan.artifacts[next(iter(prepared_names))],
                source,
                **_verification(args),
            )
        payload["estimates"] = [
            storage_preview(
                cell,
                plan.retention.model_dump(mode="json"),
                verified_prepared=prepared_receipt,
            )
            for cell, _ in cells
        ]
    if args.experiment_command == "diff":
        from klab.experiments.compiler import compare_configs

        comparisons = []
        for comparison in plan.comparisons:
            base = [
                (cell, coordinate)
                for cell, coordinate in cells
                if all(coordinate.get(k) == v for k, v in comparison.baseline.items())
            ]
            variant = [
                (cell, coordinate)
                for cell, coordinate in cells
                if all(coordinate.get(k) == v for k, v in comparison.variant.items())
            ]
            if len(base) != 1 or len(variant) != 1:
                raise ValueError(
                    f"comparison {comparison.id} selectors must identify one cell on each side"
                )
            differences = compare_configs(
                base[0][0],
                variant[0][0],
                expected=comparison.interventions,
                invariants=comparison.invariants,
                include_derived=True,
            )
            comparisons.append(
                {
                    "id": comparison.id,
                    "mode": comparison.mode,
                    "differences": differences,
                    "invariants": "passed",
                }
            )
        payload["comparisons"] = comparisons
    _emit(args, payload)


def _prepare(args: argparse.Namespace) -> None:
    from klab.canonical import canonical_json
    from klab.experiments.prepare import prepare_plan

    plan, source = _declaration(args)
    from klab.recovery.provenance import declaration_preflight
    from klab.workdir import ensure_work_dir

    if not plan.evaluations and not plan.evaluation_suite:
        raise ValueError(
            "EVALUATION_NOT_DECLARED: prepare requires an evaluation declaration"
        )
    provenance = declaration_preflight(
        source, "experiment", args.allow_uncommitted_declaration
    )
    ensure_work_dir(args.work_dir)
    workspace = _workspace(plan.id)
    resource_envelope = args.resource_envelope_value
    record = prepare_plan(
        plan,
        source,
        workspace,
        resource_envelope=resource_envelope,
        tokenizer_batch_documents=args.tokenizer_batch_documents,
        tokenizer_batch_source_bytes=args.tokenizer_batch_source_bytes,
        **_verification(args),
    )
    record["declaration_provenance"] = provenance
    record["storage_checks"] = getattr(args, "storage_checks", [])
    path = workspace / "preparation.json"
    encoded = canonical_json(record) + b"\n"
    if path.exists():
        from klab.documents import read_document

        previous = read_document(path)
        operational = {"declaration_provenance", "storage_checks"}
        if {
            key: value for key, value in previous.items() if key not in operational
        } != {key: value for key, value in record.items() if key not in operational}:
            raise ValueError(
                f"existing preparation record changed: {path}; use a new plan ID"
            )
    else:
        with path.open("xb") as handle:
            handle.write(encoded)
    _emit(
        args,
        {
            "id": plan.id,
            "preparation": str(path),
            "variants": [
                {key: value for key, value in variant.items() if key != "config"}
                for variant in record["variants"]
            ],
            "training_started": False,
        },
    )


def _lock(args: argparse.Namespace) -> None:
    from klab.experiments.lock import publish_lock, resolve_plan

    plan, source = _declaration(args)
    workspace = _workspace(plan.id)
    resolved = resolve_plan(
        plan,
        source,
        prepared=_preparation(workspace),
        max_runs=args.max_runs,
        **_verification(args),
    )
    path = publish_lock(resolved, workspace, **_verification(args))
    _emit(
        args,
        {
            "id": plan.id,
            "lock": str(path),
            "scientific_sha256": resolved.scientific_sha256,
            "plan_sha256": resolved.plan_sha256,
            "valid_science": True,
            "runtime_availability": "not_verified",
        },
    )


def _bind_inputs(args: argparse.Namespace) -> None:
    from klab.experiments.direct_inputs import bind_direct_inputs

    plan = bind_direct_inputs(
        Path(args.run_config),
        Path(args.template),
        args.prepared_root,
        args.output,
        **_verification(args),
    )
    _emit(args, {"id": plan.id, "plan": str(args.output.absolute())})


def _export_config(args: argparse.Namespace) -> None:
    from klab.experiments.export_config import export_effective_config

    digest = export_effective_config(
        Path(args.lock), args.cell, args.output, **_verification(args)
    )
    _emit(
        args,
        {
            "cell": args.cell,
            "config": str(args.output.absolute()),
            "config_sha256": digest,
        },
    )


def _derive(args: argparse.Namespace) -> None:
    """Publish a verified, derived experiment declaration."""
    from klab.derivation import derive_experiment, parse_assignments

    prepared = Path(args.prepared).absolute() if args.prepared is not None else None
    receipt = derive_experiment(
        Path(args.source).absolute(),
        Path(args.output).absolute(),
        parse_assignments(args.settings),
        prepared=prepared,
        max_runs=args.max_runs,
    )
    _emit(
        args,
        {
            "status": "ok",
            "receipt": str(
                Path(args.output)
                .absolute()
                .with_name(Path(args.output).name + ".derivation.json")
            ),
            "derivation": receipt,
        },
    )


def _require_executable_profile(locked: Any) -> None:
    if "workflow_profile" in locked.availability:
        raise ValueError(
            "WORKFLOW_EXECUTION_UNAVAILABLE: this profile supports native locking only; evaluation/context admission is not implemented"
        )


def locked_cell_request(
    locked: Any,
    cell: Any,
    worker: str,
    workspace: Path,
    controller: Any,
    *,
    runtime_binding_sha256: str | None = None,
    read_only: bool = False,
    workflow_context: Any = None,
) -> dict[str, Any]:
    """Construct the same lock-bound worker request for CLI and campaign runs."""
    if workflow_context is None:
        _require_executable_profile(locked)
    else:
        workflow_context.authorize_lock(locked)
    phase = next(phase for phase in locked.phases if phase.id == cell.phase)
    parent_path: Path | None = None
    parent_digest: str | None = None
    binding_digest: str | None = None
    if phase.parent is not None:
        from klab.experiments.binding import bind_generation
        from klab.training.manifest import sha256_file

        parent_id = f"{phase.parent}:{cell.id.split(':', 1)[1]}"
        matches = [
            row
            for row in controller.list_experiments()
            if (row["spec"].get("plan") or {}).get("plan_sha256") == locked.plan_sha256
            and (row["spec"].get("plan") or {}).get("cell_id") == parent_id
            and row["status"] == "COMPLETE"
            and row["ingestion_status"] == "COMPLETE"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"phase {cell.phase} needs one ingested parent {parent_id}"
            )
        parent_run_id = matches[0].get("run_id")
        if (
            not isinstance(parent_run_id, str)
            or not parent_run_id
            or Path(parent_run_id).name != parent_run_id
            or parent_run_id in {".", ".."}
        ):
            raise ValueError(f"ingested parent {parent_id} lacks a valid run_id")
        binding, selected_parent = bind_generation(
            workspace,
            plan_sha256=locked.plan_sha256,
            cell_id=cell.id,
            parent_cell_id=parent_id,
            parent_run=controller.root / parent_run_id,
            selector=phase.selector,
            at_step=phase.at_step,
            full_state=phase.transition != "promote",
            read_only=read_only,
        )
        parent_path = Path(selected_parent["checkpoint_path"])
        parent_digest = selected_parent["checkpoint_sha256"]
        binding_digest = sha256_file(binding)
    elif phase.checkpoint is not None:
        parent_path = Path(locked.availability["artifacts"][phase.checkpoint])
        parent_digest = locked.artifacts[phase.checkpoint]["sha256"]
    request = {
        "config": cell.config,
        "worker": worker,
        "plan": {
            "plan_id": locked.id,
            "plan_sha256": locked.plan_sha256,
            "scientific_sha256": locked.scientific_sha256,
            "cell_id": cell.id,
            "phase_id": cell.phase,
            "coordinate": cell.coordinate,
            "config_sha256": cell.config_sha256,
            "parent_checkpoint_sha256": parent_digest,
            "execution_binding_sha256": binding_digest,
            "runtime_binding_sha256": runtime_binding_sha256,
        },
    }
    if phase.transition in {"resume", "extend_budget"}:
        request[
            "extend_budget" if phase.transition == "extend_budget" else "resume"
        ] = parent_path
    elif phase.transition == "promote":
        request["promote"] = parent_path
    return request


def _bind(args: argparse.Namespace) -> None:
    from klab.experiments.binding import bind_runtime
    from klab.experiments.lock import open_lock
    from klab.workers.controller import Controller

    locked = open_lock(Path(args.lock), **_verification(args))
    selected = [
        cell for cell in locked.cells if args.cell is None or cell.id == args.cell
    ]
    if not selected:
        raise ValueError("no locked cell matches --cell")
    workspace = _workspace(locked.id)
    profile = getattr(args, "runtime_profile_loaded", None)
    controller = Controller(workspace / "controller") if args.worker else None
    receipts = [
        {
            "cell_id": cell.id,
            "binding": str(
                bind_runtime(
                    locked,
                    cell,
                    workspace,
                    profile=profile,
                    worker=args.worker,
                    controller=controller,
                )
            ),
        }
        for cell in selected
    ]
    for receipt in receipts:
        receipt["binding_sha256"] = Path(receipt["binding"]).stem
    _emit(args, {"plan_sha256": locked.plan_sha256, "bindings": receipts})


def _run_cells(locked: Any, args: argparse.Namespace) -> list[Any]:
    """Select exactly the same immutable cells for preflight and submission."""
    dependent = {phase.id for phase in locked.phases if phase.parent is not None}
    selected = [
        cell
        for cell in locked.cells
        if (args.cell is None or cell.id == args.cell)
        and (
            cell.phase == args.phase
            if args.phase is not None
            else cell.phase not in dependent
        )
    ]
    if not selected:
        raise ValueError("no locked cell matches --cell/--phase")
    if args.binding is not None and len(selected) != 1:
        raise ValueError("one --binding receipt can select only one locked cell")
    return selected


def _run(args: argparse.Namespace) -> None:
    from klab.experiments.binding import bind_runtime, open_runtime_binding
    from klab.experiments.lock import open_lock
    from klab.workers.controller import Controller
    from klab.workers.models import WorkerDefinition

    verification = _verification(args)
    locked = open_lock(Path(args.lock), **verification)
    _require_executable_profile(locked)
    selected = _run_cells(locked, args)
    from klab.recovery.provenance import declaration_preflight

    if not locked.evaluations and not locked.evaluation_suite:
        raise ValueError(
            "EVALUATION_NOT_DECLARED: run requires an evaluation declaration"
        )
    declaration_source = locked.availability.get("declaration_source")
    if declaration_source is None:
        raise ValueError("DECLARATION_SOURCE_UNAVAILABLE: re-lock the committed plan")
    provenance = declaration_preflight(
        Path(declaration_source), "experiment", args.allow_uncommitted_declaration
    )
    actual_hashes = {
        declaration["path"]: declaration["sha256"]
        for declaration in provenance["declarations"]
    }
    if actual_hashes != locked.availability.get("declaration_hashes"):
        raise ValueError(
            "DECLARATION_IDENTITY_CHANGED: authored inputs differ from the frozen lock"
        )
    workspace = _workspace(locked.id)
    controller = Controller(workspace / "controller", **verification)
    profile = getattr(args, "runtime_profile_loaded", None)
    source_worker = args.worker or (
        locked.execution.get("worker")
        if profile is None and args.binding is None
        else None
    )
    receipts: list[dict[str, Any]] = []
    if args.binding is not None:
        receipts = [
            open_runtime_binding(
                Path(args.binding), locked, selected[0], controller=controller
            )
        ]
        if receipts[0]["source_kind"] == "worker":
            source_worker = str(receipts[0]["descriptor"]["name"])
        elif receipts[0]["source_kind"] == "profile":
            from klab.runtime_profile import RuntimeProfile

            profile = RuntimeProfile.model_validate(receipts[0]["descriptor"])
        else:
            raise ValueError("unsupported runtime binding source")
    elif profile is not None:
        receipts = [
            {
                "binding_sha256": bind_runtime(
                    locked, cell, workspace, profile=profile
                ).stem
            }
            for cell in selected
        ]
    elif source_worker is not None:
        receipts = [
            {
                "binding_sha256": bind_runtime(
                    locked, cell, workspace, worker=source_worker, controller=controller
                ).stem
            }
            for cell in selected
        ]
    else:
        from klab.runtime_profile import require_authorization

        for cell in selected:
            runtime = cell.config.runtime
            if runtime.engine != "pytorch" or runtime.backend != "cpu":
                raise ValueError(
                    f"runtime binding required for non-CPU locked cell {cell.id}"
                )
            require_authorization(cell.config, None)
    worker = source_worker
    if worker is None:
        first = selected[0].config.runtime
        if any(
            (
                cell.config.runtime.engine,
                cell.config.runtime.backend,
                cell.config.runtime.device_index,
            )
            != (first.engine, first.backend, first.device_index)
            for cell in selected
        ):
            raise ValueError("one local worker cannot execute mixed locked runtimes")
        worker = f"plan-{locked.id}-{first.backend}"
        controller.register(
            WorkerDefinition(
                worker_id=worker,
                name=worker,
                transport="local",
                python=profile.python
                if profile is not None
                else Path(sys.executable).absolute(),
                root=workspace / "workers" / worker,
                engine=first.engine,
                backend=first.backend,
                device_index=first.device_index,
            )
        )
        if not receipts:
            receipts = [
                {
                    "binding_sha256": bind_runtime(
                        locked, cell, workspace, worker=worker, controller=controller
                    ).stem
                }
                for cell in selected
            ]
    requests = [
        locked_cell_request(
            locked,
            cell,
            worker,
            workspace,
            controller,
            runtime_binding_sha256=receipt["binding_sha256"],
        )
        for cell, receipt in zip(selected, receipts, strict=True)
    ]
    for request in requests:
        request["declaration_provenance"] = provenance
        request["storage_checks"] = getattr(args, "storage_checks", [])
    submissions = controller.submit_many(requests)
    _emit(
        args,
        {
            "plan_sha256": locked.plan_sha256,
            "submissions": [item.model_dump(mode="json") for item in submissions],
        },
    )


def _collect(args: argparse.Namespace) -> None:
    from klab.experiments.evidence import collect_evidence
    from klab.experiments.lock import open_lock

    verification = _verification(args)
    locked = open_lock(Path(args.lock), **verification)
    workspace = _workspace(locked.id)
    _emit(args, collect_evidence(locked, workspace, **verification))


def _explain(args: argparse.Namespace) -> None:
    from klab.experiments.lock import open_lock

    locked = open_lock(Path(args.lock), **_verification(args))
    _emit(
        args,
        {
            "plan_sha256": locked.plan_sha256,
            "scientific_sha256": locked.scientific_sha256,
            "cells": [
                {
                    "id": cell.id,
                    "coordinate": cell.coordinate,
                    "phase": cell.phase,
                    "config_sha256": cell.config_sha256,
                    "artifacts": cell.artifacts,
                }
                for cell in locked.cells
            ],
        },
    )


def _reconstruct(args: argparse.Namespace) -> None:
    from klab.experiments.lock import open_lock
    from klab.experiments.retrospective import retrospective_views

    locked = open_lock(Path(args.lock), **_verification(args))
    options = (
        {}
        if args.index is None
        else {"workspace": _workspace(locked.id), "index_path": Path(args.index)}
    )
    _emit(args, retrospective_views(locked, **options, **_verification(args)))


def _handle(args: argparse.Namespace) -> None:
    try:
        if args.experiment_command in {"validate", "inspect", "diff"}:
            _inspect_declaration(args)
        elif args.experiment_command == "prepare":
            _prepare(args)
        elif args.experiment_command == "lock":
            _lock(args)
        elif args.experiment_command == "derive":
            _derive(args)
        elif args.experiment_command == "bind-inputs":
            _bind_inputs(args)
        elif args.experiment_command == "export-config":
            _export_config(args)
        elif args.experiment_command == "bind":
            _bind(args)
        elif args.experiment_command == "run":
            _run(args)
        elif args.experiment_command == "collect":
            _collect(args)
        elif args.experiment_command == "explain":
            _explain(args)
        elif args.experiment_command == "reconstruct":
            _reconstruct(args)
        else:
            raise ValueError(
                f"unsupported experiment command {args.experiment_command}"
            )
    except (ValueError, OSError, TypeError, KeyError) as error:
        _emit(args, {"status": "error", "error": str(error)})
        raise SystemExit(2) from None


def add_commands(commands: argparse._SubParsersAction) -> None:
    from klab.experiments.recipe_cli import add_commands as add_recipe_commands

    add_recipe_commands(
        commands.add_parser("recipe", help="Inspect portable recipe metadata")
    )
    for name in (
        "validate",
        "inspect",
        "diff",
        "prepare",
        "lock",
        "derive",
        "bind",
        "bind-inputs",
        "export-config",
        "run",
        "collect",
        "explain",
        "reconstruct",
    ):
        command = commands.add_parser(
            name,
            **(
                {
                    "description": "Derive a new bound declaration with cold input and "
                    "scientific compatibility verification; does not lock, execute, "
                    "or verify runtime capability."
                }
                if name == "derive"
                else {}
            ),
        )
        if name == "bind-inputs":
            command.add_argument("run_config")
            command.add_argument("template")
        else:
            command.add_argument(
                "source"
                if name in {"validate", "inspect", "diff", "prepare", "lock", "derive"}
                else "lock"
            )
        command.add_argument("--json", action="store_true")
        if name in {
            "inspect",
            "diff",
            "lock",
            "bind",
            "bind-inputs",
            "export-config",
            "run",
            "collect",
            "explain",
            "reconstruct",
        }:
            command.add_argument("--cold-verify", action="store_true")
        if name in {"prepare", "run"}:
            command.add_argument("--allow-uncommitted-declaration", action="store_true")
        if name in {"validate", "inspect", "diff", "lock", "derive"}:
            command.add_argument("--max-runs", type=int, default=1000)
        if name == "prepare":
            command.add_argument("--resource-envelope", type=Path)
            from klab.cli.main import _tokenizer_batch_arguments

            _tokenizer_batch_arguments(command)
        if name in {"bind-inputs", "export-config"}:
            command.add_argument("--output", type=Path, required=True)
        if name == "bind-inputs":
            command.add_argument("--prepared-root", type=Path, required=True)
        if name == "derive":
            command.add_argument(
                "--set",
                dest="settings",
                action="append",
                required=True,
                metavar="FIELD=JSON",
                help="strict JSON assignment; replace complete sequences",
            )
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--prepared", type=Path)
        if name == "export-config":
            command.add_argument("--cell", required=True)
        if name in {"bind", "run"}:
            command.add_argument("--cell")
        if name == "bind":
            sources = command.add_mutually_exclusive_group(required=True)
            sources.add_argument("--runtime-profile", type=Path)
            sources.add_argument("--runtime", metavar="ID")
            sources.add_argument("--worker")
        if name == "run":
            command.add_argument("--phase")
            sources = command.add_mutually_exclusive_group()
            sources.add_argument("--runtime-profile", type=Path)
            sources.add_argument("--runtime", metavar="ID")
            sources.add_argument("--worker")
            sources.add_argument("--binding", type=Path)
        if name == "reconstruct":
            command.add_argument("--index")
        command.set_defaults(handler=_handle)
