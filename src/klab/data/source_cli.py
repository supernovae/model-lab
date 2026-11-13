"""Native declaration-driven dataset acquisition commands."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path


def _run(args: argparse.Namespace) -> object:
    from klab.workdir import resolve_work_dir

    cache = args.cache_dir or resolve_work_dir(args.work_dir) / "downloads" / "datasets"
    if args.data_command == "lock":
        from klab.data.sources import lock_source

        return lock_source(args.source, args.output, cache)
    if args.data_command == "snapshot":
        from klab.data.sources import snapshot_source

        return snapshot_source(args.lock, args.output, cache, resume=args.resume)
    if args.data_command == "convert":
        from klab.data.conversion import convert_dataset

        return convert_dataset(args.declaration, args.output, cache)
    from klab.data.conversion import verify_conversion

    verify_conversion(args.directory, cache)
    return args.directory


def _handle(args: argparse.Namespace) -> None:
    try:
        with redirect_stdout(sys.stderr):
            result = _run(args)
    except (OSError, ValueError, TypeError, KeyError) as error:
        if args.json:
            print(json.dumps({"status": "blocked", "error": str(error)}))
        else:
            print(f"Data {args.data_command} blocked: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if args.json:
        status = "verified" if args.data_command == "verify-conversion" else "complete"
        print(json.dumps({"status": status, "path": str(result)}))
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
    convert = commands.add_parser(
        "convert",
        help="Turn pinned chat, instruction or multi-field rows into local JSONL",
    )
    convert.add_argument("declaration", type=Path)
    for command in (lock, snapshot, convert):
        command.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser(
        "verify-conversion",
        help="Re-read a conversion's pinned source and check its outputs",
    )
    verify.add_argument("directory", type=Path)
    for command in (lock, snapshot, convert, verify):
        command.add_argument("--cache-dir", type=Path)
        command.add_argument("--json", action="store_true")
        command.set_defaults(handler=_handle)
