"""CLI for one verified, pinned local inference server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def serve(args: argparse.Namespace) -> None:
    from klab.evaluation.inference import load_run
    from klab.evaluation.serving import (
        LocalHTTPServer,
        LocalInference,
        ServerLimits,
    )

    if not 0 <= args.port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    limits = ServerLimits(
        args.max_new_tokens,
        args.max_request_bytes,
        args.request_timeout,
        args.max_clients,
    )
    loaded = load_run(
        args.run_id,
        Path(args.runs_dir),
        args.checkpoint,
        args.backend,
        authorization=args.runtime_authorization,
        quantized=args.quantized,
    )
    inference = LocalInference(loaded, args.model_id, limits)
    with LocalHTTPServer((args.host, args.port), inference) as server:
        print(
            json.dumps(
                {
                    "url": f"http://{args.host if ':' not in args.host else '[' + args.host + ']'}:{server.server_port}/v1",
                    "model": inference.model_id,
                    "identity": loaded.identity,
                    "streaming": False,
                }
            ),
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            pass
        finally:
            inference.closing.set()


def add_command(commands: argparse._SubParsersAction, runs_dir_default: str) -> None:
    parser = commands.add_parser(
        "serve", help="Serve one verified checkpoint through a local, non-streaming API"
    )
    parser.add_argument("run_id")
    parser.add_argument("--runs-dir", default=runs_dir_default)
    parser.add_argument(
        "--checkpoint",
        help="latest.json, best.json, or a generation path; resolved once at startup",
    )
    parser.add_argument(
        "--quantized",
        type=Path,
        metavar="DIR",
        help="Apply a `klab weights quantize` artifact made from this checkpoint",
    )
    parser.add_argument(
        "--backend", choices=("auto", "metal", "mps", "cuda", "rocm", "xpu", "cpu")
    )
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--runtime-profile", type=Path)
    runtime.add_argument("--runtime", metavar="ID")
    parser.add_argument("--host", choices=("127.0.0.1", "::1"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-request-bytes", type=int, default=65536)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--max-clients", type=int, default=8)
    parser.set_defaults(handler=serve)
