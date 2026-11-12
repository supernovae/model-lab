"""CLI adapter for the independent operational monitor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from klab.operational_monitor import load_monitor_policy, monitor_command


def register_monitor_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `klab monitor` on the top-level CLI's subparsers."""
    from klab.lab_job import register_job_parser

    register_job_parser(subparsers)
    parser = subparsers.add_parser(
        "monitor", help="Guard one owned command and record operational evidence"
    )
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--runner-lock", type=Path)
    parser.add_argument("--runner-state", type=Path)
    parser.add_argument("--runner-blocked", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--reserve-bytes", type=int, default=0)
    parser.add_argument("--reserve-inodes", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "monitored_command", nargs=argparse.REMAINDER, help="-- COMMAND [ARG ...]"
    )
    parser.set_defaults(handler=_handle)
    baseline = subparsers.add_parser(
        "monitor-baseline", help="Capture one persistent workspace storage baseline"
    )
    baseline.add_argument("root", type=Path)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--seconds", type=float, default=4.0)
    baseline.add_argument("--json", action="store_true")
    baseline.set_defaults(handler=_capture_baseline)


def execute_monitor(args: argparse.Namespace) -> int:
    """Print final typed completion and propagate command/guard failure to CLI exit."""
    command = list(args.monitored_command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("monitor requires -- COMMAND [ARG ...]")
    runner_paths = [
        getattr(args, name, None)
        for name in ("runner_lock", "runner_state", "runner_blocked")
    ]
    runner_options = {}
    execute = monitor_command
    if any(path is not None for path in runner_paths):
        if any(path is None for path in runner_paths):
            raise ValueError("all three --runner-* paths are required")
        from klab.runner_control import run_controlled

        execute = run_controlled
        runner_options = {
            "lock_path": runner_paths[0],
            "state_dir": runner_paths[1],
            "blocked_path": runner_paths[2],
        }
    result = execute(
        command,
        load_monitor_policy(args.policy),
        workspace=args.workspace,
        log_dir=args.log_dir,
        policy_path=args.policy,
        cwd=args.cwd,
        reserved_bytes=args.reserve_bytes,
        reserved_inodes=args.reserve_inodes,
        baseline_path=args.baseline,
        **runner_options,
    )
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    return (
        result.returncode
        if result.status in {"COMPLETE", "FAILED"}
        and result.returncode is not None
        and result.returncode >= 0
        else 1
    )


def _handle(args: argparse.Namespace) -> None:
    raise SystemExit(execute_monitor(args))


def _capture_baseline(args: argparse.Namespace) -> None:
    from klab.operational_monitor import capture_workspace_baseline

    result = capture_workspace_baseline(args.root, args.output, seconds=args.seconds)
    print(
        json.dumps(
            result.model_dump(mode="json"),
            sort_keys=True,
            indent=None if args.json else 2,
        )
    )
