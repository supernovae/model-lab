"""Campaign authoring, verified execution and explicit authorization commands."""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from klab.campaign.engine import CampaignEngine
from klab.campaign.plan import load_campaign
from klab.campaign.state import CampaignStore
from klab.workdir import resolve_work_dir

READ_ONLY_COMMANDS = frozenset({"validate", "plan", "status", "next", "explain"})


def _emit(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    envelope = {
        "format": "klab-campaign-command-v1",
        "command": args.campaign_command,
        **payload,
    }
    if args.json:
        print(json.dumps(envelope, sort_keys=True, allow_nan=False))
        return
    print(f"Campaign {payload.get('id', args.source)} ({args.campaign_command})")
    if "error" in payload:
        print(f"FAILED: {payload['error']['reason']}")
    for stage in payload.get("stages", []):
        print(
            f"{stage['id']} | {stage['kind']} | {stage['scope']} | {stage['state']} | {stage.get('outcome', '-')} | {stage.get('reason', '')}"
        )
        for field in ("deficits", "outputs", "blocked_by", "bindings"):
            if stage.get(field):
                print(f"  {field}: {json.dumps(stage[field], sort_keys=True)}")
    for row in payload.get("recoverability", []):
        print(
            f"recoverability {row['id']} | {row['kind']} | "
            f"{row['classification']} | {row.get('reason', '')}"
        )
    if "reconstruction" in payload:
        for row in payload["reconstruction"].get("steps", []):
            print(
                f"reconstructed {row['id']} | {row['kind']} | "
                f"{row['classification']} | {row.get('reason', '')}"
            )
    if "next_action" in payload:
        print(f"next_action: {json.dumps(payload['next_action'], sort_keys=True)}")


def _handle(args: argparse.Namespace) -> None:
    source = Path(args.source).absolute()
    try:
        if args.campaign_command == "validate":
            plan = load_campaign(source)
            store = CampaignStore(plan, resolve_work_dir(args.work_dir))
            payload = {
                "id": plan.id,
                "campaign_version": 1,
                "valid": True,
                "declaration_sha256": store.declaration_sha,
                "stages": [
                    {
                        "id": stage.id,
                        "kind": stage.kind,
                        "scope": stage.scope,
                        "requires": list(stage.requires),
                        "state": "NOT_STARTED",
                    }
                    for stage in plan.ordered_stages()
                ],
            }
        else:
            engine = CampaignEngine(
                source,
                resolve_work_dir(args.work_dir),
                runtime_profile=getattr(args, "runtime_profile_loaded", None),
                cold_verify=getattr(args, "cold_verify", False),
                observer=getattr(args, "observer", None),
            )
            # Lower-level adapters may print progress; stdout remains one JSON object.
            with redirect_stdout(sys.stderr):
                if args.campaign_command in READ_ONLY_COMMANDS:
                    payload = engine.inspect(args.campaign_command)
                elif args.campaign_command == "approve":
                    payload = engine.approve(
                        args.gate, decision=args.decision, note=args.note
                    )
                elif args.campaign_command == "reconstruct":
                    payload = engine.reconstruct(
                        allow_network=args.allow_network,
                        allow_uncommitted_declaration=args.allow_uncommitted_declaration,
                        evidence_output=args.evidence_output,
                    )
                else:
                    payload = engine.apply(
                        resume=args.campaign_command == "resume",
                        max_wait_seconds=args.max_wait_seconds,
                        execute_runs=args.execute_runs,
                        allow_uncommitted_declaration=args.allow_uncommitted_declaration,
                        only_stage=args.only_stage,
                    )
            if args.campaign_command == "explain":
                declarations = {
                    stage.id: stage.model_dump(mode="json")
                    for stage in engine.plan.stages
                }
                for stage in payload["stages"]:
                    stage["declaration"] = declarations[stage["id"]]
    except (OSError, ValueError, TypeError, KeyError) as error:
        _emit(
            args,
            {
                "state": "FAILED",
                "error": {"type": type(error).__name__, "reason": str(error)},
                "next_action": {
                    "action": "repair",
                    "reason": str(error),
                    "identities": {},
                },
            },
        )
        raise SystemExit(1) from error
    _emit(args, payload)
    if any(stage["state"] == "FAILED" for stage in payload.get("stages", [])):
        raise SystemExit(1)


def _wait_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError(
            "max wait seconds must be finite and nonnegative"
        )
    return seconds


def argument_error(command: str, reason: str) -> None:
    """Keep JSON command errors machine-readable before a workspace exists."""
    _emit(
        argparse.Namespace(campaign_command=command, json=True),
        {
            "state": "FAILED",
            "error": {"type": "ArgumentError", "reason": reason},
            "next_action": {"action": "repair", "reason": reason, "identities": {}},
        },
    )
    raise SystemExit(2)


class _CommandParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        if "--json" in sys.argv[1:]:
            argument_error(self.prog.rsplit(" ", 1)[-1], message)
        super().error(message)


def add_commands(commands: argparse._SubParsersAction) -> None:
    campaign = commands.add_parser(
        "campaign",
        help="Execute declared Campaign v1 DAGs with verified inputs and approvals",
    )
    verbs = campaign.add_subparsers(
        dest="campaign_command", required=True, parser_class=_CommandParser
    )
    for name in (
        "validate",
        "plan",
        "status",
        "next",
        "apply",
        "resume",
        "reconstruct",
        "explain",
        "approve",
    ):
        command = verbs.add_parser(name)
        command.add_argument("source", metavar="PATH", type=Path)
        command.add_argument("--json", action="store_true")
        if name in {"plan", "status", "next", "explain", "apply", "resume", "approve"}:
            command.add_argument("--cold-verify", action="store_true")
        if name in {"plan", "status", "next", "explain", "apply", "resume"}:
            command.add_argument("--observations-output", type=Path)
        if name in {"apply", "resume"}:
            command.add_argument("--max-wait-seconds", type=_wait_seconds, default=120)
            command.add_argument("--execute-runs", action="store_true")
            command.add_argument("--only-stage")
            runtime_source = command.add_mutually_exclusive_group()
            runtime_source.add_argument("--runtime-profile", type=Path)
            runtime_source.add_argument("--runtime", metavar="ID")
        if name in {"apply", "resume", "reconstruct"}:
            command.add_argument("--allow-uncommitted-declaration", action="store_true")
        if name == "reconstruct":
            command.add_argument("--allow-network", action="store_true")
            command.add_argument("--evidence-output", type=Path)
        if name == "approve":
            command.add_argument("gate", metavar="GATE")
            command.add_argument(
                "--decision", choices=("approve", "reject"), default="approve"
            )
            command.add_argument("--note")
        command.set_defaults(handler=_handle)
