"""Local Surface Review import and Streamlit launch commands."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from klab.evaluation.surface_triage_import import import_triage_reports


def _import(args: argparse.Namespace) -> None:
    if not args.triage_run:
        raise ValueError("triage requires repeated --triage-run RUN_ID RUNS_DIR")
    result = import_triage_reports(
        args.triage_run,
        args.output,
        args.sample,
        args.selection_seed,
        args.presentation_seed,
    )
    import json

    print(json.dumps(result, sort_keys=True))


def _launch(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve().parents[1] / "dashboard" / "surface_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--",
        "--mode",
        args.mode,
    ]
    if args.mode == "review":
        from klab.evaluation.surface_review import open_surface_bundle

        folder = Path(args.bundle).resolve(strict=True)
        open_surface_bundle(folder)
        cmd.extend(("--bundle", str(folder)))
    else:
        if not 2 <= len(args.cell) <= 4:
            raise ValueError(
                "exploratory chat requires 2–4 --cell ALIAS=GENERATION_PATH entries"
            )
        for item in args.cell:
            if "=" not in item or not item.split("=", 1)[0]:
                raise ValueError("--cell requires ALIAS=GENERATION_PATH")
            cmd.extend(("--cell", item))
        cmd.extend(("--backend", args.backend, "--seed", str(args.seed)))
        if args.work_dir:
            cmd.extend(("--work-dir", str(args.work_dir)))
    subprocess.run(cmd, check=True)


def add_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    surface = commands.add_parser("surface", help="Local self-blind checkpoint review")
    sub = surface.add_subparsers(dest="surface_command", required=True)
    importer = sub.add_parser("import", help="Seal existing verified generations")
    importer.add_argument("source", choices=("triage",))
    importer.add_argument("--output", required=True)
    importer.add_argument(
        "--sample", choices=("quick", "standard", "full"), default="quick"
    )
    importer.add_argument("--selection-seed", type=int, default=0)
    importer.add_argument("--presentation-seed", type=int, default=0)
    importer.add_argument(
        "--triage-run", nargs=2, action="append", metavar=("RUN_ID", "RUNS_DIR")
    )
    importer.set_defaults(handler=_import)
    review = sub.add_parser("review", help="Review a sealed bundle locally")
    review.add_argument("bundle")
    review.add_argument("--port", type=int, default=8502)
    review.set_defaults(handler=_launch, mode="review")
    chat = sub.add_parser("chat", help="Explore 2–4 verified checkpoints locally")
    chat.add_argument(
        "--cell", action="append", default=[], metavar="ALIAS=GENERATION_PATH"
    )
    chat.add_argument(
        "--backend",
        choices=("auto", "cpu", "rocm", "cuda", "xpu", "mps", "metal"),
        default="cpu",
    )
    chat.add_argument("--seed", type=int, default=11)
    chat.add_argument("--port", type=int, default=8502)
    chat.add_argument(
        "--work-dir",
        default=argparse.SUPPRESS,
        help="Work root for chat and parent scratch (overrides the global prefix)",
    )
    chat.set_defaults(handler=_launch, mode="chat")
