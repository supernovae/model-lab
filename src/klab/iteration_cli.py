"""CLI surface for the read-only native iteration inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from klab.iteration import check_iteration


def _check(args: argparse.Namespace) -> None:
    report = check_iteration(
        Path(args.declaration),
        Path(args.parent) if args.parent is not None else None,
        work_dir=args.work_dir,
        cold_verify=args.cold_verify,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True, allow_nan=False))
    else:
        for key, value in report.items():
            print(f"{key}: {json.dumps(value, sort_keys=True, allow_nan=False)}")
    if report["state"] == "BLOCKED":
        raise SystemExit(1)


def register_parser(commands: argparse._SubParsersAction) -> None:
    iteration = commands.add_parser(
        "iteration", help="Inspect a declared iteration without changing state"
    )
    actions = iteration.add_subparsers(dest="iteration_command", required=True)
    check = actions.add_parser(
        "check",
        help="Inspect a declaration or immutable lock with an optional exact parent",
    )
    check.add_argument("declaration", metavar="DECLARATION_OR_LOCK")
    check.add_argument("--parent", metavar="EXACT_GENERATION")
    check.add_argument("--json", action="store_true")
    check.add_argument("--cold-verify", action="store_true")
    check.set_defaults(handler=_check)
