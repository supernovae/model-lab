"""Loopback-only CPU sampler. The dashboard owns UI; this process owns one model."""

import json
import os
import sqlite3
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# The deployed sampler uses a pinned inference runtime, with UI/catalog code beside
# this entry point. Do not replace that runtime or import dashboard app modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from snapshot_catalog import SnapshotOwner, available_catalog, resolve_candidate
from snapshot_controls import validate_controls


class Sampler:
    def __init__(self, runs):
        self.runs = Path(runs)
        self.owner = SnapshotOwner()

    def load_selected(self, selected):
        from klab.evaluation.inference import load_run

        location = resolve_candidate(self.runs, selected)
        return load_run(
            selected.run,
            self.runs,
            str(location),
            "cpu",
            expected_checkpoint_sha256=selected.digest,
        )

    def complete(self, request):
        if not isinstance(request, dict) or request.get("store") != str(self.runs):
            raise ValueError("This sampler only serves its configured run store.")
        prompt = request.get("prompt")
        count = request.get("max_new_tokens")
        temperature, top_k, seed = (
            request.get(k) for k in ("temperature", "top_k", "seed")
        )
        validate_controls(prompt, count, temperature, top_k, seed)
        selected = next(
            (
                item
                for item in available_catalog(self.runs)
                if item.key
                == (
                    request.get("run"),
                    request.get("checkpoint"),
                    request.get("checkpoint_sha256"),
                )
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                "Selected checkpoint is no longer available; no substitute loaded."
            )

        def action(loaded):
            from klab.evaluation.generation import generate

            if (
                len(loaded.tokenizer.encode(prompt, add_special_tokens=False).ids)
                + count
                > 1024
            ):
                raise ValueError(
                    "Prompt plus completion must fit the 1,024-token local preview limit."
                )
            result = generate(
                loaded.model,
                loaded.tokenizer,
                prompt,
                1024,
                count,
                loaded.device,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
                strict_context=True,
                engine=loaded.engine,
            )
            return {
                "prompt": prompt,
                "completion": result[len(prompt) :],
                "tokenizer_sha256": loaded.identity["tokenizer_sha256"],
                "settings": {
                    key: request[key]
                    for key in (
                        "store",
                        "run",
                        "checkpoint",
                        "checkpoint_sha256",
                        "temperature",
                        "top_k",
                        "seed",
                        "max_new_tokens",
                    )
                },
                "backend": "cpu",
            }

        return self.owner.execute(selected, self.load_selected, action)


def handler(sampler):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            # Prompts, completions and query strings never enter service logs.
            pass

        def reply(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError, ConnectionResetError:
                pass

        def allowed(self):
            # No browser-origin access or CORS: only the dashboard's server-side
            # JSON client. Host validation also prevents DNS rebinding.
            return self.headers.get(
                "Host"
            ) == f"127.0.0.1:{self.server.server_port}" and not self.headers.get(
                "Origin"
            )

        def do_GET(self):
            if not self.allowed():
                self.reply(403, {"error": "Only local dashboard requests are allowed."})
            elif self.path == "/health":
                state, identity = sampler.owner.status
                self.reply(
                    200, {"status": state, "identity": identity, "backend": "cpu"}
                )
            else:
                self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if not self.allowed():
                self.reply(403, {"error": "Only local dashboard requests are allowed."})
                return
            if self.path != "/complete":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if (
                    self.headers.get("Content-Type") != "application/json"
                    or not 0 < length <= 32768
                ):
                    raise ValueError("A bounded JSON completion request is required.")
                self.connection.settimeout(10)
                request = json.loads(self.rfile.read(length))
                self.reply(200, sampler.complete(request))
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
                self.reply(
                    409 if "busy" in str(error).lower() else 400, {"error": str(error)}
                )

    return Handler


def main():
    import torch

    config = json.loads(Path(os.environ["KLAB_SNAPSHOT_SERVICE_CONFIG"]).read_text())
    url = urlparse(config["url"])
    if url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port:
        raise ValueError("Sampler requires an explicit loopback HTTP port")
    torch.set_num_threads(2)
    sampler = Sampler(config["store"])
    ThreadingHTTPServer(("127.0.0.1", url.port), handler(sampler)).serve_forever()


if __name__ == "__main__":
    main()
