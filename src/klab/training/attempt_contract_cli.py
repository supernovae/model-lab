"""Public CLI adapter for the existing persistent attempt-contract ledger."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from klab.training.attempt_budget import AttemptBudget

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_RECEIPT_KINDS = (
    "train",
    "panel",
    "campaign_run",
    "campaign_panel",
    "campaign_evaluation",
    "none",
)


def _absolute(path: Path, name: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{name} must be absolute and not a symlink")
    return path


def _digest(value: str, name: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _id(value: str, name: str) -> str:
    if not _ID.fullmatch(value):
        raise ValueError(f"{name} must be a safe nonempty identifier")
    return value


def handle_attempt(args: argparse.Namespace) -> None:
    """Dispatch one explicit ledger action; all work remains in AttemptBudget."""
    if args.attempt_command == "run-phase":
        from klab.cli.main import build_parser
        from klab.training.preparation import (
            compile_phase,
            supervised_phase_command,
        )

        compiled = compile_phase(
            args.plan,
            _absolute(args.attempt_root, "attempt root"),
            _id(args.label, "phase label"),
            json.loads(args.bindings_json),
        )
        command = supervised_phase_command(
            compiled,
            work_root=args.workspace,
            ledger=args.ledger,
            label=args.label,
            content_identity=args.content_identity_sha256,
            whole_policy=args.policy,
            preparation_policy=args.preparation_policy,
            baseline=args.baseline,
        )
        # The same parser and handler enforce reservations, bindings and shutdown.
        parsed = build_parser().parse_args(command[1:])
        handle_attempt(parsed)
        return
    if args.attempt_command == "phase-command":
        from klab.training.preparation import compile_phase

        print(
            json.dumps(
                compile_phase(
                    args.plan,
                    _absolute(args.attempt_root, "attempt root"),
                    _id(args.label, "phase label"),
                    json.loads(args.bindings_json),
                ),
                sort_keys=True,
            )
        )
        return
    ledger = _absolute(args.ledger, "ledger")
    if args.attempt_command == "init":
        budget = AttemptBudget.create_contract(
            ledger,
            contract_path=_absolute(args.contract, "contract"),
            expected_sha256=_digest(args.contract_sha256, "contract SHA-256"),
            monitor_policy_path=(
                _absolute(args.policy, "monitor policy")
                if args.policy is not None
                else None
            ),
            workspace_baseline_path=(
                _absolute(args.baseline, "workspace baseline")
                if args.baseline is not None
                else None
            ),
            workspace_root=(
                _absolute(args.workspace, "workspace")
                if args.workspace is not None
                else None
            ),
        )
        print(json.dumps(budget.status(), sort_keys=True))
        return
    if args.attempt_command == "status":
        print(json.dumps(AttemptBudget(ledger).status(), sort_keys=True))
        return
    if args.attempt_command == "bind-artifact":
        result = AttemptBudget(ledger).bind_resolved_artifact(
            kind=args.kind,
            path=_absolute(args.path, "resolved artifact"),
            expected_sha256=_digest(args.sha256, "resolved artifact SHA-256"),
            content_identity_sha256=_digest(
                args.content_identity_sha256, "content identity"
            ),
            workspace_root=_absolute(args.workspace, "workspace"),
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.attempt_command != "run":
        raise ValueError("unknown attempt command")

    command = (
        args.command_args[1:] if args.command_args[:1] == ["--"] else args.command_args
    )
    if not command:
        raise ValueError("attempt run requires a command after --")
    label = _id(args.label, "phase label")
    content_identity = _digest(args.content_identity_sha256, "content identity")
    counters = (
        args.updates,
        args.target_positions,
        args.generation_calls,
        args.generated_tokens,
    )
    if any(type(counter) is not int or counter < 0 for counter in counters):
        raise ValueError("attempt reservations must be nonnegative integers")
    campaign_kind = args.receipt_kind in {
        "campaign_run",
        "campaign_panel",
        "campaign_evaluation",
    }
    if args.receipt_kind == "none":
        if args.receipt_path is not None or args.campaign_stage is not None:
            raise ValueError(
                "receipt-kind none cannot bind a receipt or Campaign stage"
            )
        if any(counters):
            raise ValueError("nonzero reservations require a native receipt binding")
        receipt = None
    else:
        if args.receipt_path is None:
            raise ValueError("native receipt path is required")
        receipt = _absolute(args.receipt_path, "native receipt")
    if campaign_kind:
        if args.campaign_stage is None:
            raise ValueError("Campaign receipt requires --campaign-stage")
        _id(args.campaign_stage, "Campaign stage")
        if args.campaign_work_dir is None:
            raise ValueError("Campaign receipt requires --campaign-work-dir")
        campaign_work_dir = _absolute(args.campaign_work_dir, "Campaign work directory")
    elif args.campaign_stage is not None:
        raise ValueError("Campaign stage is only valid for Campaign receipts")
    elif args.campaign_work_dir is not None:
        raise ValueError("Campaign work directory is only valid for Campaign receipts")
    else:
        campaign_work_dir = None

    parent_checkpoint = (
        _absolute(args.parent_checkpoint, "parent checkpoint")
        if args.parent_checkpoint is not None
        else None
    )
    budget = AttemptBudget(ledger)
    returncode = budget.run_contract(
        command,
        activity=args.activity,
        label=label,
        content_identity_sha256=content_identity,
        monitor_policy_path=_absolute(args.policy, "monitor policy"),
        workspace_baseline_path=_absolute(args.baseline, "workspace baseline"),
        workspace_root=_absolute(args.workspace, "workspace"),
        completion=_absolute(args.completion, "owned completion"),
        updates=args.updates,
        target_positions=args.target_positions,
        generation_calls=args.generation_calls,
        generated_tokens=args.generated_tokens,
        native_receipt_kind=args.receipt_kind,
        native_receipt_path=receipt,
        campaign_stage=args.campaign_stage,
        campaign_work_dir=campaign_work_dir,
        parent_checkpoint_path=parent_checkpoint,
    )
    if returncode != 0:
        raise SystemExit(returncode)


def register_attempt_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the canonical public attempt interface."""
    attempt = commands.add_parser(
        "attempt", help="Use a cumulative native attempt contract"
    )
    actions = attempt.add_subparsers(dest="attempt_command", required=True)
    phase_run = actions.add_parser(
        "run-phase", help="Dispatch one canonical zero-model preparation phase"
    )
    phase_run.add_argument("--plan", type=Path, required=True)
    phase_run.add_argument("--attempt-root", type=Path, required=True)
    phase_run.add_argument("--label", required=True)
    phase_run.add_argument("--bindings-json", default="{}")
    for name in ("ledger", "workspace", "policy", "preparation-policy", "baseline"):
        phase_run.add_argument(f"--{name}", type=Path, required=True)
    phase_run.add_argument("--content-identity-sha256", required=True)
    phase_run.set_defaults(handler=handle_attempt)
    compile_command = actions.add_parser(
        "phase-command", help="Expand canonical preparation arguments without execution"
    )
    compile_command.add_argument("--plan", type=Path, required=True)
    compile_command.add_argument("--attempt-root", type=Path, required=True)
    compile_command.add_argument("--label", required=True)
    compile_command.add_argument("--bindings-json", default="{}")
    compile_command.set_defaults(handler=handle_attempt)
    initialize = actions.add_parser("init")
    initialize.add_argument("--ledger", type=Path, required=True)
    initialize.add_argument("--contract", type=Path, required=True)
    initialize.add_argument("--contract-sha256", required=True)
    initialize.add_argument("--policy", type=Path)
    initialize.add_argument("--baseline", type=Path)
    initialize.add_argument("--workspace", type=Path)
    initialize.set_defaults(handler=handle_attempt)
    status = actions.add_parser("status")
    status.add_argument("--ledger", type=Path, required=True)
    status.set_defaults(handler=handle_attempt)
    bind = actions.add_parser("bind-artifact")
    bind.add_argument("--ledger", type=Path, required=True)
    bind.add_argument(
        "--kind",
        choices=(
            "train_config",
            "evaluation_baseline",
            "pre_freeze_project",
            "build_project",
            "release_acceptance",
        ),
        required=True,
    )
    bind.add_argument("--path", type=Path, required=True)
    bind.add_argument("--sha256", required=True)
    bind.add_argument("--content-identity-sha256", required=True)
    bind.add_argument("--workspace", type=Path, required=True)
    bind.set_defaults(handler=handle_attempt)
    run = actions.add_parser("run")
    run.add_argument("--ledger", type=Path, required=True)
    run.add_argument("--label", required=True)
    run.add_argument(
        "--activity",
        choices=("train", "warmup", "validate", "evaluate", "inspect"),
        required=True,
    )
    run.add_argument("--content-identity-sha256", required=True)
    for field in ("policy", "baseline", "workspace", "completion"):
        run.add_argument(f"--{field}", type=Path, required=True)
    for field in (
        "updates",
        "target-positions",
        "generation-calls",
        "generated-tokens",
    ):
        run.add_argument(f"--{field}", type=int, required=True)
    run.add_argument("--receipt-kind", choices=_RECEIPT_KINDS, required=True)
    run.add_argument("--receipt-path", type=Path)
    run.add_argument("--campaign-stage")
    run.add_argument("--campaign-work-dir", type=Path)
    run.add_argument("--parent-checkpoint", type=Path)
    run.add_argument("command_args", nargs=argparse.REMAINDER)
    run.set_defaults(handler=handle_attempt)
