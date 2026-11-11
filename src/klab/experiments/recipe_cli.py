"""Lightweight recipe routing with explicit inspection and publication operations."""

import argparse
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path


def add_commands(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="recipe_command", required=True)
    from klab.experiments.recipe_authoring import handle as create_handler
    from klab.experiments.recipe_export import handle as export_handler

    create = commands.add_parser(
        "create",
        help="Inventory explicit native declarations into a new recipe directory",
    )
    create.add_argument("root", type=Path)
    create.add_argument(
        "--profile", choices=("matrix", "preparation", "workflow"), required=True
    )
    create.add_argument("--id", required=True)
    create.add_argument("--entrypoint", required=True)
    create.add_argument(
        "--file",
        action="append",
        required=True,
        help="ROLE:RELATIVE_PATH; repeat for every metadata member",
    )
    create.add_argument(
        "--artifacts",
        type=Path,
        help="Existing profile artifact identity mapping; never future hashes",
    )
    create.add_argument(
        "--campaign",
        help="Workflow campaign path already included as a metadata member",
    )
    create.add_argument("--output", type=Path, required=True)
    create.set_defaults(handler=create_handler)
    export = commands.add_parser(
        "export",
        help="Copy an existing authenticated recipe inventory to a new directory",
    )
    export.add_argument("source", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(handler=export_handler)
    inspect = commands.add_parser(
        "inspect", help="Preview an inventoried native recipe; never execute"
    )
    inspect.add_argument("source", type=Path)
    inspect.add_argument("--max-runs", type=int, default=1000)
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(handler=handle)
    preparation = commands.add_parser(
        "preparation",
        help="Inspect or explicitly apply an inventoried local CPU preparation graph",
    )
    preparation_actions = preparation.add_subparsers(
        dest="preparation_command", required=True
    )
    preview = preparation_actions.add_parser("inspect")
    preview.add_argument("source", type=Path)
    preview.add_argument("--bindings", type=Path, required=True)
    preview.add_argument("--json", action="store_true")
    preview.set_defaults(handler=handle_preparation)
    for name in ("apply", "resume"):
        execution = preparation_actions.add_parser(name)
        execution.add_argument("source", type=Path)
        execution.add_argument("--bindings", type=Path, required=True)
        execution.add_argument("--admission", type=Path, required=True)
        execution.add_argument("--work-dir", type=Path, required=True)
        execution.add_argument("--json", action="store_true")
        execution.set_defaults(handler=handle_preparation_execution)
    workflow = commands.add_parser(
        "workflow",
        help="Inspect, publish or explicitly execute a prepared-input workflow",
    )
    actions = workflow.add_subparsers(dest="workflow_command", required=True)
    for name in ("inspect", "resolve", "publish"):
        action = actions.add_parser(name)
        action.add_argument("source", type=Path)
        action.add_argument("--bindings", type=Path, required=True)
        action.add_argument("--max-runs", type=int, default=1000)
        action.add_argument("--json", action="store_true")
        if name == "publish":
            action.add_argument("--workspace", type=Path, required=True)
        action.set_defaults(handler=handle_workflow)
    for name in ("apply", "resume", "approve"):
        action = actions.add_parser(name)
        action.add_argument("source", type=Path)
        action.add_argument("--bindings", type=Path, required=True)
        action.add_argument("--admission", type=Path, required=True)
        action.add_argument("--work-dir", type=Path, required=True)
        action.add_argument("--json", action="store_true")
        if name == "approve":
            action.add_argument("gate")
            action.add_argument(
                "--decision", choices=("approve", "reject"), default="approve"
            )
            action.add_argument("--note")
        else:
            action.add_argument("--execute-runs", action="store_true")
            action.add_argument("--max-wait-seconds", type=_wait_seconds, default=120)
        action.set_defaults(handler=handle_workflow_execution)
    report = actions.add_parser(
        "report", help="Verify and write a native publication report; never execute"
    )
    report.add_argument("lock", type=Path)
    report.add_argument("--workspace", type=Path)
    report.add_argument("--index", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--format", choices=("json", "markdown"), default="json")
    report.add_argument("--json", action="store_true")
    report.set_defaults(handler=handle_workflow_report)


def _wait_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError(
            "max wait seconds must be finite and nonnegative"
        )
    return seconds


def _workflow_result(command: str, result: dict) -> None:
    print(
        json.dumps(
            {"format": "klab-workflow-command-v1", "command": command, **result},
            sort_keys=True,
            allow_nan=False,
        )
    )


def handle_workflow_execution(args: argparse.Namespace) -> None:
    from klab.documents import read_document
    from klab.experiments.workflow_execution import (
        WorkflowAdmission,
        WorkflowExecution,
    )

    try:
        # Native adapters can emit progress; stdout remains one JSON document.
        with redirect_stdout(sys.stderr):
            context = WorkflowExecution(
                args.source,
                args.bindings,
                args.work_dir,
                WorkflowAdmission.model_validate(read_document(args.admission)),
            )
            engine = context.engine()
            if args.workflow_command == "approve":
                result = engine.approve(
                    args.gate, decision=args.decision, note=args.note
                )
            else:
                result = engine.apply(
                    resume=args.workflow_command == "resume",
                    execute_runs=args.execute_runs,
                    max_wait_seconds=args.max_wait_seconds,
                )
    except (OSError, TypeError, ValueError, KeyError) as error:
        _workflow_result(
            args.workflow_command,
            {
                "state": "FAILED",
                "error": {"type": type(error).__name__, "reason": str(error)},
            },
        )
        raise SystemExit(1) from None
    _workflow_result(args.workflow_command, result)
    if any(stage["state"] == "FAILED" for stage in result.get("stages", [])):
        raise SystemExit(1)


def handle_workflow_report(args: argparse.Namespace) -> None:
    from klab.experiments.publication_report import write_publication_report

    try:
        with redirect_stdout(sys.stderr):
            output = write_publication_report(
                args.lock,
                args.output,
                workspace=args.workspace,
                index_path=args.index,
                format=args.format,
            )
    except (OSError, TypeError, ValueError, KeyError) as error:
        _workflow_result(
            "report",
            {
                "state": "FAILED",
                "error": {"type": type(error).__name__, "reason": str(error)},
            },
        )
        raise SystemExit(1) from None
    _workflow_result("report", {"output": str(output), "executable": False})


def handle(args: argparse.Namespace) -> None:
    from klab.experiments.recipe import inspect_recipe

    try:
        result = inspect_recipe(args.source, max_runs=args.max_runs)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"klab: {error}") from None
    if args.json:
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    else:
        print(
            f"Recipe {result['id']}: {len(result['cells'])} native config cells; inspect-only, not executable"
        )
        print(f"Recipe SHA-256: {result['recipe_sha256']}")
        print(f"Installed lab SHA-256: {result['installed_lab']['sha256']}")
        for requirement in result["requirements"]:
            print(
                f"{requirement['kind']} {requirement.get('field', '')}: {requirement['status']}"
            )
        for limitation in result["limitations"]:
            print(limitation)


def handle_preparation(args: argparse.Namespace) -> None:
    from klab.experiments.preparation_recipe import inspect_preparation_recipe

    try:
        result = inspect_preparation_recipe(args.source, args.bindings)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"klab: {error}") from None
    print(json.dumps(result, sort_keys=True, allow_nan=False))


def handle_preparation_execution(args: argparse.Namespace) -> None:
    from klab.documents import read_document
    from klab.experiments.preparation_execution import (
        PreparationExecution,
        preparation_admission,
    )

    try:
        context = PreparationExecution(
            args.source,
            args.bindings,
            args.work_dir,
            preparation_admission(read_document(args.admission)),
        )
        result = context.engine().apply(resume=args.preparation_command == "resume")
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"klab: {error}") from None
    print(json.dumps(result, sort_keys=True, allow_nan=False))


def handle_workflow(args: argparse.Namespace) -> None:
    from klab.experiments.workflow_recipe import (
        inspect_workflow,
        publish_workflow,
        resolve_workflow,
    )

    try:
        if args.workflow_command == "inspect":
            result = inspect_workflow(
                args.source, args.bindings, max_runs=args.max_runs
            )
        elif args.workflow_command == "resolve":
            lock = resolve_workflow(args.source, args.bindings, max_runs=args.max_runs)
            result = {
                "format": "klab-workflow-resolution-v1",
                "scientific_sha256": lock.scientific_sha256,
                "plan_sha256": lock.plan_sha256,
                "published": False,
                "executable": False,
                "runtime_capability": "unknown",
                "cells": len(lock.cells),
            }
        else:
            path = publish_workflow(
                args.source, args.bindings, args.workspace, max_runs=args.max_runs
            )
            result = {
                "format": "klab-workflow-publication-v1",
                "lock": str(path),
                "executable": False,
                "runtime_capability": "unknown",
            }
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"klab: {error}") from None
    print(json.dumps(result, sort_keys=True, allow_nan=False))


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="klab experiment recipe", allow_abbrev=False)
    add_commands(parser)
    args = parser.parse_args(argv)
    args.handler(args)
