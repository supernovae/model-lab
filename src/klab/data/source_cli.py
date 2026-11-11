"""Native declaration-driven dataset acquisition commands."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path


def _handle(args: argparse.Namespace) -> None:
    from klab.data.sources import lock_source, snapshot_source
    from klab.workdir import resolve_work_dir

    cache = args.cache_dir or resolve_work_dir(args.work_dir) / "downloads" / "datasets"
    try:
        with redirect_stdout(sys.stderr):
            result = (
                lock_source(args.source, args.output, cache)
                if args.data_command == "lock"
                else snapshot_source(args.lock, args.output, cache, resume=args.resume)
            )
    except (OSError, ValueError, TypeError, KeyError) as error:
        if args.json:
            print(json.dumps({"status": "blocked", "error": str(error)}))
        else:
            print(f"Data {args.data_command} blocked: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if args.json:
        print(json.dumps({"status": "complete", "path": str(result)}))
    else:
        print(result)


def register_parser(commands: argparse._SubParsersAction) -> None:
    lock = commands.add_parser(
        "lock", help="Pin a dataset declaration and source inventory"
    )
    lock.add_argument("source", type=Path)
    snapshot = commands.add_parser(
        "snapshot", help="Materialize a pinned generic text dataset"
    )
    snapshot.add_argument("lock", type=Path)
    snapshot.add_argument("--resume", action="store_true")
    for command in (lock, snapshot):
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--cache-dir", type=Path)
        command.add_argument("--json", action="store_true")
        command.set_defaults(handler=_handle)
