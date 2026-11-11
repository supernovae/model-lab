"""Command-line recovery inspection and explicit deterministic replay."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from klab.recovery.engine import (
    inspect_manifest,
    plan_manifest,
    reconstruct_manifest,
)
from klab.recovery.implementation_replay import ReplayFailure
from klab.workdir import resolve_work_dir


def _handle(args: argparse.Namespace) -> None:
    source = Path(args.manifest)
    root = resolve_work_dir(args.work_dir)
    try:
        if args.recovery_command == "inspect":
            result = inspect_manifest(source, root)
        elif args.recovery_command == "plan":
            result = plan_manifest(source, root)
        else:
            with redirect_stdout(sys.stderr):
                result = reconstruct_manifest(
                    source,
                    root,
                    allow_network=args.allow_network,
                    allow_uncommitted_declaration=args.allow_uncommitted_declaration,
                    evidence_output=args.evidence_output,
                    replay_pinned_implementation=args.replay_pinned_implementation,
                )
        failures = (
            [
                row["status"]
                for row in result.get("outcomes", [])
                if row["status"] not in {"PRESENT", "UNSEALED_RESULT"}
            ]
            if args.recovery_command == "reconstruct"
            else []
        )
        if args.recovery_command == "reconstruct" and any(
            row["status"] == "UNSEALED_RESULT" for row in result.get("outcomes", [])
        ):
            failures.append("UNSEALED_RESULT")
        reasons = set(failures)
        reasons.update(
            row["classification"]
            for row in result["steps"]
            if row["classification"]
            in {
                "MISSING_EXTERNAL",
                "MISSING_NONRECONSTRUCTABLE",
                "MISSING_RECONSTRUCTABLE",
                "PINNED_IMPLEMENTATION_REPLAY_REQUIRED",
                "MISSING_IMPLEMENTATION",
            }
        )
        result["reason_codes"] = sorted(reasons)
        result["status"] = (
            "BLOCKED"
            if reasons
            else "INSPECTED"
            if args.recovery_command != "reconstruct"
            else "RECONSTRUCTED"
        )
        print(
            json.dumps(
                result, sort_keys=True, default=str, indent=None if args.json else 2
            )
        )
        if failures:
            raise SystemExit(1)
    except (OSError, ValueError, TypeError) as error:
        reason = str(error).split(":", 1)[0]
        if not reason.isupper() or not reason.replace("_", "").isalnum():
            reason = "RECOVERY_BLOCKED"
        envelope = {
            "format": "klab-recovery-v1",
            "status": "BLOCKED",
            "reason_codes": [reason],
            "reason": str(error),
        }
        if isinstance(error, ReplayFailure):
            envelope["failed_replay"] = {
                "receipt_path": str(error.receipt_path),
                "record_sha256": error.record_sha256,
            }
        print(json.dumps(envelope, sort_keys=True, indent=None if args.json else 2))
        raise SystemExit(1) from error


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    recovery = subparsers.add_parser(
        "recovery", help="Inspect and reconstruct deterministic prerequisites"
    )
    commands = recovery.add_subparsers(dest="recovery_command", required=True)
    for name in ("inspect", "plan", "reconstruct"):
        command = commands.add_parser(name)
        command.add_argument("manifest", type=Path)
        command.add_argument("--json", action="store_true")
        if name == "reconstruct":
            command.add_argument("--allow-network", action="store_true")
            command.add_argument(
                "--replay-pinned-implementation",
                action="store_true",
                help="Explicitly execute the source commit's Corpus Forge in an isolated uv environment",
            )
            command.add_argument("--allow-uncommitted-declaration", action="store_true")
            command.add_argument("--evidence-output", type=Path)
        command.set_defaults(handler=_handle)
