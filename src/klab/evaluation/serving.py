"""Bounded, non-streaming local OpenAI-compatible inference.

The HTTP adapter owns no artifact selection or runtime bypass: callers supply one
already verified InferenceRun, retained for the lifetime of this server.
"""

from __future__ import annotations

import ipaddress
import json
import math
import select
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
from typing import Any

from klab.evaluation.chat import ChatMessage, prepare_chat_prompt
from klab.evaluation.generation_request import GenerationCancelled, generate_result
from klab.evaluation.inference import InferenceRun


class APIError(ValueError):
    def __init__(self, message: str, status: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class ServerLimits:
    max_new_tokens: int = 256
    max_request_bytes: int = 65536
    request_timeout: float = 60.0
    max_clients: int = 8

    def __post_init__(self) -> None:
        for name in ("max_new_tokens", "max_request_bytes", "max_clients"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.request_timeout) or self.request_timeout <= 0:
            raise ValueError("request_timeout must be finite and positive")


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise APIError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if (
        type(value) not in (int, float)
        or not minimum <= value <= maximum
        or not math.isfinite(value)
    ):
        raise APIError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return float(value)


def _text(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise APIError("text must contain valid Unicode scalar values") from error
    return value


class LocalInference:
    """One pinned checkpoint with serialized generation and strict input validation."""

    def __init__(
        self,
        loaded: InferenceRun,
        model_id: str | None = None,
        limits: ServerLimits | None = None,
    ):
        self.loaded = loaded
        self.model_id = model_id or (
            f"{loaded.identity['run_id']}-{loaded.identity['checkpoint_sha256'][:12]}"
        )
        if (
            not self.model_id
            or len(self.model_id) > 200
            or any(ord(char) < 32 for char in self.model_id)
        ):
            raise ValueError("model ID must contain 1–200 printable characters")
        self.limits = limits or ServerLimits()
        self.lock = threading.Lock()
        self.closing = threading.Event()

    def models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                    "klab": {
                        "identity": self.loaded.identity,
                        "context_length": self.loaded.config.model.max_seq_len,
                        "max_new_tokens": self.limits.max_new_tokens,
                        "streaming": False,
                        "chat_format": "chat_transcript_v1",
                        "sampling": ["temperature", "top_k", "seed"],
                    },
                }
            ],
        }

    def complete(
        self,
        payload: Any,
        *,
        chat: bool,
        cancellation: threading.Event,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise APIError("request must be a JSON object")
        allowed = {
            "model",
            "messages" if chat else "prompt",
            "max_tokens",
            "temperature",
            "top_k",
            "seed",
            "stop",
            "stream",
            "n",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        }
        unknown = payload.keys() - allowed
        if unknown:
            raise APIError(f"unsupported options: {', '.join(sorted(unknown))}")
        if payload.get("model") != self.model_id:
            raise APIError(
                "model must match an ID returned by /v1/models", 404, "model_not_found"
            )
        if "stream" in payload and payload["stream"] is not False:
            raise APIError(
                "only stream=false is supported; token streaming is unavailable"
            )
        _integer(payload.get("n", 1), "n", 1, 1)
        for name, neutral in (
            ("top_p", 1),
            ("frequency_penalty", 0),
            ("presence_penalty", 0),
        ):
            _number(payload.get(name, neutral), name, neutral, neutral)
        max_tokens = _integer(
            payload.get("max_tokens", min(64, self.limits.max_new_tokens)),
            "max_tokens",
            1,
            self.limits.max_new_tokens,
        )
        temperature = _number(payload.get("temperature", 0.0), "temperature", 0, 2)
        top_k = _integer(
            payload.get("top_k", 0), "top_k", 0, self.loaded.tokenizer.get_vocab_size()
        )
        seed = _integer(payload.get("seed", 0), "seed", 0, 2**64 - 1)
        stops = payload.get("stop", [])
        if stops is None:
            stops = []
        if isinstance(stops, str):
            stops = [stops]
        if (
            not isinstance(stops, list)
            or len(stops) > 4
            or any(
                not isinstance(stop, str) or not stop or len(stop) > 256
                for stop in stops
            )
        ):
            raise APIError(
                "stop must be null, a nonempty string, or up to four nonempty strings (256 characters each)"
            )
        stops = [_text(stop) for stop in stops]
        if chat:
            prompt = self._chat_prompt(payload.get("messages"), max_tokens)
            stops = [*stops, "\nUser:", "\nSystem:", "\nAssistant:"]
        else:
            prompt = payload.get("prompt")
            if not isinstance(prompt, str):
                raise APIError(
                    "prompt must be one string; token arrays and batched prompts are unsupported"
                )
        _text(prompt)
        prompt_tokens = max(
            1, len(self.loaded.tokenizer.encode(prompt, add_special_tokens=False).ids)
        )
        if prompt_tokens + max_tokens > self.loaded.config.model.max_seq_len:
            raise APIError(
                "prompt and requested completion exceed native context; reduce input or max_tokens"
            )

        def cancelled() -> bool:
            return cancellation.is_set() or self.closing.is_set()

        while not self.lock.acquire(timeout=0.05):
            if cancelled():
                raise GenerationCancelled("request cancelled while waiting for model")
        try:
            if cancelled():
                raise GenerationCancelled("request cancelled before generation")
            result = generate_result(
                self.loaded.model,
                self.loaded.tokenizer,
                prompt,
                self.loaded.config.model.max_seq_len,
                max_tokens,
                self.loaded.device,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
                stop_sequences=stops,
                strict_context=True,
                engine=self.loaded.engine,
                cancellation=cancelled,
            )
        finally:
            self.lock.release()
        text = result.text[len(prompt) :]
        if chat:
            text = text.strip()
        choice = {"index": 0, "finish_reason": result.finish_reason, "logprobs": None}
        if chat:
            choice["message"] = {"role": "assistant", "content": text}
        else:
            choice["text"] = text
        return {
            "id": f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}",
            "object": "chat.completion" if chat else "text_completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [choice],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
            "klab": {
                "identity": self.loaded.identity,
                "prompt": prompt,
                "format": "chat_transcript_v1" if chat else "raw",
                "generation": {
                    "temperature": temperature,
                    "top_k": top_k,
                    "seed": seed,
                    "max_tokens": max_tokens,
                    "stop": stops,
                    "strict_context": True,
                    "use_cache": True,
                    "context_length": self.loaded.config.model.max_seq_len,
                },
            },
        }

    def _chat_prompt(self, messages: Any, max_tokens: int) -> str:
        if not isinstance(messages, list) or not 1 <= len(messages) <= 128:
            raise APIError("messages must contain 1–128 text messages")
        for item in messages:
            if (
                not isinstance(item, dict)
                or set(item) != {"role", "content"}
                or not isinstance(item["content"], str)
            ):
                raise APIError(
                    "each message must have only role and string content; multimodal and tools are unsupported"
                )
            _text(item["content"])
        messages = list(messages)
        system = None
        if messages[0]["role"] == "system":
            system = messages.pop(0)["content"]
        if not messages or len(messages) % 2 != 1:
            raise APIError(
                "chat requires complete user/assistant turns followed by a user message"
            )
        for index, item in enumerate(messages):
            if item["role"] != ("user" if index % 2 == 0 else "assistant"):
                raise APIError(
                    "roles must alternate user/assistant after an optional initial system message"
                )
        try:
            prompt, dropped = prepare_chat_prompt(
                [ChatMessage(**item) for item in messages[:-1]],
                messages[-1]["content"],
                self.loaded.tokenizer,
                self.loaded.config.model.max_seq_len,
                max_tokens,
                system=system,
            )
        except ValueError as error:
            raise APIError(str(error)) from error
        if dropped:
            raise APIError(
                "chat history and completion exceed native context; remove oldest complete turns explicitly"
            )
        return prompt


class LocalHTTPServer(ThreadingHTTPServer):
    """Bound HTTP workers, request bytes, socket waits, and generation lifetime."""

    daemon_threads = False
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], inference: LocalInference):
        if not ipaddress.ip_address(address[0]).is_loopback:
            raise ValueError(
                "serve only accepts a loopback IP address; no authentication is provided"
            )
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        self.inference = inference
        self.slots = threading.BoundedSemaphore(inference.limits.max_clients)
        super().__init__(address, LocalRequestHandler)
        bound_host, bound_port = self.server_address[:2]
        host = str(ipaddress.ip_address(bound_host))
        host = f"[{host}]" if ":" in host else host
        self.allowed_authorities = {
            f"{name}:{bound_port}": f"{name}:{bound_port}"
            for name in (host, "localhost")
        }
        if bound_port == 80:
            self.allowed_authorities.update(
                {name: f"{name}:80" for name in (host, "localhost")}
            )

    def server_bind(self) -> None:
        # HTTPServer resolves its display name with getfqdn(), which can wait on
        # reverse DNS. This API binds numeric loopback addresses and needs no DNS.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def get_request(self) -> tuple[socket.socket, Any]:
        request, address = super().get_request()
        request.settimeout(self.inference.limits.request_timeout)
        return request, address

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        if not self.slots.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(
        self, request: socket.socket, client_address: Any
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def server_close(self) -> None:
        self.inference.closing.set()
        super().server_close()


class LocalRequestHandler(BaseHTTPRequestHandler):
    server: LocalHTTPServer

    def setup(self) -> None:
        super().setup()
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.body_consumed = threading.Event()
        self.deadline = time.monotonic() + self.server.inference.limits.request_timeout
        self.watcher = threading.Thread(
            target=self._watch,
            args=(self.cancelled, self.finished, self.deadline),
            daemon=True,
        )
        self.watcher.start()

    def finish(self) -> None:
        try:
            super().finish()
        finally:
            self.finished.set()
            self.watcher.join()

    def log_message(self, format: str, *args: Any) -> None:
        # Prompts and request paths are deliberately not logged.
        pass

    def _reply(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body, ensure_ascii=True, allow_nan=False).encode("utf-8")
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, error: APIError) -> None:
        self._reply(
            error.status,
            {
                "error": {
                    "message": str(error),
                    "type": "invalid_request_error"
                    if error.status < 500
                    else "server_error",
                    "param": None,
                    "code": error.code,
                }
            },
        )

    def do_GET(self) -> None:
        try:
            self._validate_authority()
            if self.path != "/v1/models":
                raise APIError("unknown endpoint", 404, "not_found")
            self._reply(200, self.server.inference.models())
        except APIError as error:
            self._error(error)
        except OSError:
            pass

    def do_POST(self) -> None:
        cancelled = self.cancelled
        try:
            self._validate_authority()
            if self.path not in {"/v1/completions", "/v1/chat/completions"}:
                raise APIError("unknown endpoint", 404, "not_found")
            if self.headers.get("Transfer-Encoding") is not None:
                raise APIError(
                    "Transfer-Encoding is unsupported; send a bounded Content-Length"
                )
            lengths = self.headers.get_all("Content-Length", [])
            if (
                len(lengths) != 1
                or not lengths[0].isascii()
                or not lengths[0].isdigit()
            ):
                raise APIError("one nonnegative Content-Length is required")
            if len(lengths[0]) > 20:
                raise APIError(
                    "request body exceeds max_request_bytes", 413, "request_too_large"
                )
            size = int(lengths[0])
            if size > self.server.inference.limits.max_request_bytes:
                raise APIError(
                    "request body exceeds max_request_bytes", 413, "request_too_large"
                )
            if self.headers.get_content_type() != "application/json":
                raise APIError(
                    "Content-Type must be application/json",
                    415,
                    "unsupported_media_type",
                )
            raw = self.rfile.read(size)
            self.body_consumed.set()
            if len(raw) != size:
                raise APIError("incomplete request body")
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeError, RecursionError) as error:
                raise APIError("invalid JSON body") from error
            result = self.server.inference.complete(
                payload,
                chat=self.path == "/v1/chat/completions",
                cancellation=cancelled,
            )
            if cancelled.is_set():
                raise GenerationCancelled("request deadline or disconnect")
            self._reply(200, result)
        except APIError as error:
            try:
                self._error(error)
            except OSError:
                pass
        except GenerationCancelled:
            try:
                self._error(
                    APIError(
                        "request cancelled or deadline exceeded",
                        408,
                        "request_cancelled",
                    )
                )
            except OSError:
                pass
        except OSError, TimeoutError:
            cancelled.set()
        except RuntimeError, ValueError, TypeError, KeyError:
            # Keep internals/local paths out of unstructured error responses.
            try:
                self._error(APIError("local inference failed", 500, "inference_error"))
            except OSError:
                pass

    def _validate_authority(self) -> None:
        # Check before reading bodies or exposing model metadata. Loopback binding
        # alone does not prevent a browser from reaching us via DNS rebinding.
        hosts = self.headers.get_all("Host", [])
        authorities = self.server.allowed_authorities
        if len(hosts) != 1 or hosts[0].lower() not in authorities:
            raise APIError("unexpected Host authority", 403, "invalid_host")
        host = authorities[hosts[0].lower()]
        origins = self.headers.get_all("Origin", [])
        if not origins:
            return  # Native clients do not normally send Origin.
        origin = origins[0].lower()
        if (
            len(origins) != 1
            or not origin.startswith("http://")
            or authorities.get(origin[7:]) != host
        ):
            raise APIError("unexpected Origin", 403, "invalid_origin")

    def _watch(
        self, cancelled: threading.Event, finished: threading.Event, deadline: float
    ) -> None:
        while not finished.wait(0.05):
            if time.monotonic() >= deadline or self.server.inference.closing.is_set():
                cancelled.set()
                # Unblock header/body reads; decoding observes cancellation.
                try:
                    self.connection.shutdown(socket.SHUT_RD)
                except OSError:
                    pass
                return
            if not self.body_consumed.is_set():
                continue
            try:
                ready, _, _ = select.select([self.connection], [], [], 0)
                if (
                    ready
                    and self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                    == b""
                ):
                    cancelled.set()
                    return
            except BlockingIOError:
                continue
            except OSError:
                cancelled.set()
                return
