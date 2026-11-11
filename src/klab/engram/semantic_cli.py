"""Command adapter for supplied-vector semantic probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from klab.engram.semantic_probe import run_semantic_probe


def _run(args: argparse.Namespace) -> None:
    try:
        report = run_semantic_probe(Path(args.declaration))
    except (OSError, OverflowError, RuntimeError, TypeError, ValueError) as error:
        if args.json:
            print(json.dumps({"status": "error", "error": str(error)}, sort_keys=True))
        else:
            print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    if args.json:
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return
    print("semantic probe")
    print(f"format: {report['format']}")
    print(f"role: {report['role']}")
    print("declaration:", json.dumps(report["declaration"], sort_keys=True))
    print("model:", json.dumps(report["model"], sort_keys=True))
    print("input_tokens:", json.dumps(report["input_tokens"], sort_keys=True))
    print(f"query_identity_sha256: {report['query_identity_sha256']}")
    for query_id, query in sorted(report["queries"].items()):
        print(f"query {query_id}:", json.dumps(query, sort_keys=True))
    print(f"logits_shape: {report['logits_shape']}")
    for attachment in report["attachments"]:
        print(
            f"attachment {attachment['name']}:",
            json.dumps(attachment, sort_keys=True),
        )


def register_parser(commands: argparse._SubParsersAction) -> None:
    """Register ``klab semantic probe DECLARATION [--json]``."""
    semantic = commands.add_parser(
        "semantic", help="Run supplied-vector semantic diagnostics"
    )
    subcommands = semantic.add_subparsers(dest="semantic_command", required=True)
    probe = subcommands.add_parser(
        "probe", help="Verify supplied semantic vectors and run one CPU forward"
    )
    probe.add_argument("declaration")
    probe.add_argument("--json", action="store_true", help="write one report object")
    probe.set_defaults(handler=_run)
