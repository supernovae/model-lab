"""Parser integration for checkpoint-bound evaluation and model readiness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from klab.campaign.state import publish_immutable
from klab.evaluation.readiness import (
    assess_readiness,
    issue_review,
    verify_readiness_result,
)
from klab.evaluation.suite import run_suite, verify_evaluation_index


def _freeze_items(args: argparse.Namespace) -> None:
    from klab.evaluation.kml_card03_items import freeze_item_directory

    _print(
        freeze_item_directory(
            Path(args.release),
            Path(args.family_inventory),
            Path(args.draft),
            Path(args.output),
        ),
        args.json,
    )


def _fixed_slices(args: argparse.Namespace) -> None:
    from klab.config.loading import load_tokenizer_config
    from klab.evaluation.fixed_slices import (
        bind_fixed_slices,
        generate_fixed_continuations,
        score_fixed_slices,
    )
    from klab.evaluation.inference import load_run, write_inference_result

    fit = load_tokenizer_config(Path(args.tokenizer_config))
    bound = bind_fixed_slices(
        Path(args.profile),
        Path(args.release),
        fit.output_dir / "tokenizer.json",
        Path(args.tokenizer_config),
        Path(args.family_inventory),
        expected_profile_sha256=getattr(args, "expected_profile_sha256", None),
        expected_family_sha256=getattr(args, "expected_family_sha256", None),
    )
    if args.fixed_command == "verify":
        _print(
            {
                "profile_sha256": bound.profile_sha256,
                "slices": 24,
                "utility_pairs": 24,
                "continuations": 8,
            },
            args.json,
        )
        return
    from klab.training.attempt_budget import AttemptBudget

    if args.fixed_command == "score":
        required = 4608 if args.mode == "utility" else 3084
        if (
            any(key.startswith("KLAB_ATTEMPT_") for key in os.environ)
            and args.max_forward_positions != required
        ):
            raise ValueError("fixed score requires its exact declared forward cap")
        AttemptBudget.require_fixed_evaluation_allocation_from_environment(
            fixed_positions=required, total_positions=required, calls=0, tokens=0
        )
    else:
        AttemptBudget.require_fixed_evaluation_allocation_from_environment(
            fixed_positions=0, total_positions=8 * 6112, calls=8, tokens=512
        )
    loaded = load_run(
        args.run_id,
        Path(args.runs_dir),
        args.checkpoint,
        args.backend,
        authorization=getattr(args, "runtime_authorization", None),
    )
    if (
        loaded.engine is not None
        or loaded.config.model.memory != "none"
        or not isinstance(loaded.device, torch.device)
        or loaded.identity["tokenizer_sha256"] != bound.profile["tokenizer_sha256"]
    ):
        raise ValueError("fixed profile requires the declared dense PyTorch tokenizer")
    if args.fixed_command == "continuations":
        result = generate_fixed_continuations(
            bound, loaded.model, loaded.device, loaded.config.model.max_seq_len
        )
    else:
        result = score_fixed_slices(
            bound, loaded.model, loaded.device, args.mode, args.max_forward_positions
        )
    result["identity"] = loaded.identity
    path = write_inference_result(loaded.run, "fixed-slices", result)
    _print({"output": str(path), **result}, args.json)


def _fixed_selection(args: argparse.Namespace) -> None:
    from klab.evaluation.fixed_selection import (
        select_fixed_validation,
        verify_fixed_selection,
    )

    result = (
        select_fixed_validation(Path(args.declaration), Path(args.output))
        if args.fixed_command == "select"
        else verify_fixed_selection(Path(args.declaration), Path(args.receipt))
    )
    _print(result, args.json)


def _print(payload: dict, json_output: bool) -> None:
    print(
        json.dumps(payload, sort_keys=True)
        if json_output
        else json.dumps(payload, indent=2, sort_keys=True)
    )


def _suite_run(args: argparse.Namespace) -> None:
    if args.store is not None and args.worker is None:
        raise ValueError("--store requires --worker")
    output = run_suite(
        Path(args.suite),
        args.run_id,
        args.checkpoint,
        Path(args.runs_dir),
        args.backend,
        authorization=getattr(args, "runtime_authorization", None),
    )
    index = verify_evaluation_index(output)
    if args.evidence_output:
        publish_immutable(
            Path(args.evidence_output),
            {
                "format": "evaluation-index-reference-v1",
                "index_sha256": index["index_sha256"],
                "checkpoint_sha256": index["checkpoint_sha256"],
                "index": str(output.resolve()),
                "evaluations": [
                    {key: row[key] for key in ("id", "status", "sha256", "reason")}
                    for row in index["evaluations"]
                ],
            },
        )
    _print({"index": str(output), **index}, args.json)


def _model(args: argparse.Namespace) -> None:
    path = assess_readiness(
        Path(args.policy), Path(args.index), Path(args.review) if args.review else None
    )
    result = verify_readiness_result(path)
    if args.evidence_output:
        publish_immutable(
            Path(args.evidence_output),
            {
                "format": "model-readiness-reference-v1",
                "result_sha256": result["result_sha256"],
                "result": str(path.resolve()),
                "state": result["state"],
                "checkpoint_sha256": result["checkpoint_sha256"],
            },
        )
    _print({"output": str(path), **result}, args.json)


def _review(args: argparse.Namespace) -> None:
    result = issue_review(
        Path(args.index),
        args.reviewer,
        args.decision,
        args.note,
        Path(args.output),
        Path(args.surface_bundle) if args.surface_bundle else None,
    )
    _print({"output": args.output, **result}, args.json)


def register_evaluation_parser(
    subparsers: argparse._SubParsersAction,
    *,
    runs_dir_default: str | Path | None = None,
) -> None:
    """Add `evaluation suite run` beneath the existing top-level subparsers."""
    evaluation = subparsers.add_parser("evaluation")
    commands = evaluation.add_subparsers(dest="evaluation_command", required=True)
    freeze = commands.add_parser(
        "freeze-items", help="Freeze complete reviewed source-bound held-out items."
    )
    for name in ("release", "family-inventory", "draft", "output"):
        freeze.add_argument(f"--{name}", required=True)
    freeze.add_argument("--json", action="store_true")
    freeze.set_defaults(handler=_freeze_items)
    suite = commands.add_parser("suite")
    actions = suite.add_subparsers(dest="suite_command", required=True)
    run = actions.add_parser("run")
    run.add_argument("suite")
    run.add_argument("run_id")
    run.add_argument("--checkpoint", required=True)
    run.add_argument(
        "--runs-dir",
        default=str(runs_dir_default) if runs_dir_default else None,
        required=runs_dir_default is None,
    )
    run.add_argument("--backend")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--runtime-profile")
    source.add_argument("--runtime", metavar="ID")
    source.add_argument("--worker")
    run.add_argument("--store", help="controller store for the named local worker")
    run.add_argument("--json", action="store_true")
    run.add_argument("--evidence-output")
    run.set_defaults(handler=_suite_run)
    fixed = commands.add_parser("fixed-slices")
    fixed_actions = fixed.add_subparsers(dest="fixed_command", required=True)
    for action in ("verify", "score", "continuations"):
        command = fixed_actions.add_parser(action)
        command.add_argument("profile")
        command.add_argument("--release", required=True)
        command.add_argument("--tokenizer-config", required=True)
        command.add_argument("--family-inventory", required=True)
        command.add_argument("--expected-profile-sha256", required=action != "verify")
        command.add_argument("--expected-family-sha256", required=action != "verify")
        command.add_argument("--json", action="store_true")
        if action in {"score", "continuations"}:
            command.add_argument("run_id")
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--runs-dir", required=True)
            command.add_argument("--backend")
            runtime = command.add_mutually_exclusive_group()
            runtime.add_argument("--runtime-profile")
            runtime.add_argument("--runtime", metavar="ID")
            if action == "score":
                command.add_argument(
                    "--mode", choices=("validation", "test", "utility"), required=True
                )
                command.add_argument("--max-forward-positions", type=int, required=True)
        command.set_defaults(handler=_fixed_slices)
    select = fixed_actions.add_parser("select")
    select.add_argument("declaration")
    select.add_argument("--output", required=True)
    select.add_argument("--json", action="store_true")
    select.set_defaults(handler=_fixed_selection)
    verify_selection = fixed_actions.add_parser("verify-selection")
    verify_selection.add_argument("declaration")
    verify_selection.add_argument("receipt")
    verify_selection.add_argument("--json", action="store_true")
    verify_selection.set_defaults(handler=_fixed_selection)


def register_readiness_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add `readiness model|review` beneath existing readiness subparsers."""
    model = subparsers.add_parser("model")
    model.add_argument("policy")
    model.add_argument("index")
    model.add_argument("--review")
    model.add_argument("--json", action="store_true")
    model.add_argument("--evidence-output")
    model.set_defaults(handler=_model)
    review = subparsers.add_parser("review")
    review.add_argument("index")
    review.add_argument("--reviewer", required=True)
    review.add_argument("--decision", choices=("approve", "reject"), required=True)
    review.add_argument("--note", required=True)
    review.add_argument("--output", required=True)
    review.add_argument("--surface-bundle")
    review.add_argument("--json", action="store_true")
    review.set_defaults(handler=_review)
