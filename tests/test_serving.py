from __future__ import annotations

import http.client
import json
import shutil
import socket
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
from test_inference import trained_run  # noqa: F401

from klab.cli.main import build_parser, main
from klab.evaluation import serving
from klab.evaluation.generation_request import (
    GenerationCancelled,
    GenerationResult,
)
from klab.evaluation.inference import load_run
from klab.evaluation.serving import (
    APIError,
    LocalHTTPServer,
    LocalInference,
    ServerLimits,
)


@pytest.fixture(scope="module")
def loaded(request):
    trained = request.getfixturevalue("trained_run")
    return load_run("original", trained.logging.root_dir)


def request(server, path, body=None, *, raw=None, headers=None):
    connection = http.client.HTTPConnection(*server.server_address[:2], timeout=5)
    try:
        data = (
            raw if raw is not None else json.dumps(body) if body is not None else None
        )
        connection.request(
            "GET" if data is None else "POST",
            path,
            body=data,
            headers=headers or {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        content = response.read()
        return response.status, json.loads(content) if content else None
    finally:
        connection.close()


@contextmanager
def running(loaded, **limits):
    app = LocalInference(loaded, "test-model", ServerLimits(**limits))
    with LocalHTTPServer(("127.0.0.1", 0), app) as server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        thread.start()
        try:
            yield server
        finally:
            app.closing.set()
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def payload(**changes):
    return {"model": "test-model", "prompt": "hello", "max_tokens": 2, **changes}


def test_models_and_real_raw_chat_roundtrip(loaded):
    with running(loaded) as server:
        status, models = request(server, "/v1/models")
        assert status == 200
        entry = models["data"][0]
        assert entry["id"] == "test-model"
        assert entry["klab"]["identity"] == loaded.identity
        assert entry["klab"]["streaming"] is False
        status, first = request(
            server, "/v1/completions", payload(temperature=0.8, seed=17, top_k=8)
        )
        assert status == 200
        _, second = request(
            server, "/v1/completions", payload(temperature=0.8, seed=17, top_k=8)
        )
        assert first["choices"] == second["choices"]
        assert first["klab"]["prompt"] == "hello"
        assert first["usage"]["total_tokens"] == sum(
            first["usage"][key] for key in ("prompt_tokens", "completion_tokens")
        )
        assert first["choices"][0]["finish_reason"] in {"stop", "length"}
        status, chat = request(
            server,
            "/v1/chat/completions",
            {
                "model": "test-model",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "n": 1,
                "top_p": 1.0,
                "frequency_penalty": 0,
            },
        )
        assert status == 200
        assert chat["object"] == "chat.completion"
        assert chat["choices"][0]["message"]["role"] == "assistant"
        assert chat["klab"]["prompt"] == "User: Hi\n\nAssistant:"


@pytest.mark.parametrize(
    "changes",
    [
        {"stream": True},
        {"top_p": 0.9},
        {"n": 2},
        {"tools": []},
        {"max_tokens": True},
        {"max_tokens": 257},
        {"temperature": float("nan")},
        {"temperature": True},
        {"seed": 2**64},
        {"top_k": 999999},
        {"stop": [""]},
        {"prompt": ["batch"]},
    ],
)
def test_unsupported_and_invalid_options_fail_before_generation(
    loaded, monkeypatch, changes
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid requests must not invoke model")

    monkeypatch.setattr(serving, "generate_result", forbidden)
    app = LocalInference(loaded, "test-model")
    with pytest.raises(APIError):
        app.complete(payload(**changes), chat=False, cancellation=threading.Event())


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        [{"role": "tool", "content": "Hi"}],
        [{"role": "user", "content": " "}],
        [{"role": "system", "content": "Hi"}],
        [{"role": "user", "content": "Hi"}, {"role": "user", "content": "Hi"}],
    ],
)
def test_chat_messages_rejected(loaded, messages):
    app = LocalInference(loaded, "test-model")
    with pytest.raises(APIError):
        app.complete(
            {"model": "test-model", "messages": messages, "max_tokens": 1},
            chat=True,
            cancellation=threading.Event(),
        )


def test_context_never_silently_truncated(loaded):
    app = LocalInference(loaded, "test-model")
    with pytest.raises(APIError, match="context"):
        app.complete(
            payload(prompt="hello " * 100), chat=False, cancellation=threading.Event()
        )
    with pytest.raises(APIError, match="history"):
        app.complete(
            {
                "model": "test-model",
                "max_tokens": 1,
                "messages": [
                    {"role": "user", "content": "hello " * 100},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "Hi"},
                ],
            },
            chat=True,
            cancellation=threading.Event(),
        )


def test_http_errors(loaded):
    with running(loaded, max_request_bytes=1024) as server:
        assert request(server, "/missing")[0] == 404
        assert request(server, "/missing", payload())[0] == 404
        assert request(server, "/v1/completions", payload(model="wrong"))[0] == 404
        assert request(server, "/v1/completions", raw="{")[0] == 400
        assert request(server, "/v1/completions", raw="[]")[0] == 400
        assert request(server, "/v1/completions", raw="x" * 1025)[0] == 413
        assert (
            request(
                server,
                "/v1/completions",
                raw="{}",
                headers={"Content-Type": "text/plain"},
            )[0]
            == 415
        )
        assert (
            request(
                server,
                "/v1/completions",
                raw="{}",
                headers={"Transfer-Encoding": "chunked"},
            )[0]
            == 400
        )
        assert request(server, "/v1/completions", payload(stream=True))[0] == 400


def test_empty_prompt_usage_accounts_for_bos(loaded):
    with running(loaded) as server:
        status, response = request(server, "/v1/completions", payload(prompt=""))
        assert status == 200
        assert response["usage"]["prompt_tokens"] == 1


def test_stop_metadata_and_actual_trajectory_count(loaded, monkeypatch):
    calls = []

    def fake(*args, **kwargs):
        calls.append(kwargs)
        return GenerationResult(args[2] + "ok", [1, 2, 3], 2, 4, "stop")

    monkeypatch.setattr(serving, "generate_result", fake)
    with running(loaded) as server:
        status, response = request(server, "/v1/completions", payload(stop="END"))
        assert status == 200
        assert response["choices"][0]["finish_reason"] == "stop"
        assert response["usage"]["completion_tokens"] == 4
        assert calls[0]["stop_sequences"] == ["END"]


def test_concurrent_requests_are_serialized_and_waiter_can_cancel(loaded, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()
    calls = []

    def fake(*args, **kwargs):
        calls.append(args[2])
        entered.set()
        assert release.wait(5)
        return GenerationResult(args[2], [], 1, 1, "stop")

    monkeypatch.setattr(serving, "generate_result", fake)
    app = LocalInference(loaded, "test-model")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            app.complete, payload(), chat=False, cancellation=threading.Event()
        )
        assert entered.wait(5)
        second = pool.submit(
            app.complete, payload(prompt="other"), chat=False, cancellation=cancelled
        )
        cancelled.set()
        try:
            with pytest.raises(GenerationCancelled):
                second.result(timeout=5)
            assert calls == ["hello"]
        finally:
            release.set()
        assert first.result(timeout=5)["choices"][0]["finish_reason"] == "stop"
    app.complete(payload(prompt="next"), chat=False, cancellation=threading.Event())
    assert calls == ["hello", "next"]


def test_disconnect_cancels_active_generation_and_server_recovers(loaded, monkeypatch):
    entered = threading.Event()
    stopped = threading.Event()

    def fake(*args, **kwargs):
        entered.set()
        while not kwargs["cancellation"]():
            stopped.wait(0.01)
        stopped.set()
        raise GenerationCancelled("disconnected")

    original = serving.generate_result
    monkeypatch.setattr(serving, "generate_result", fake)
    with running(loaded) as server:
        connection = socket.create_connection(server.server_address)
        body = json.dumps(payload()).encode()
        connection.sendall(
            f"POST /v1/completions HTTP/1.0\r\nHost: 127.0.0.1:{server.server_port}\r\nContent-Type: application/json\r\nContent-Length: ".encode()
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )
        assert entered.wait(5)
        connection.close()
        assert stopped.wait(5)
        monkeypatch.setattr(serving, "generate_result", original)
        assert request(server, "/v1/completions", payload())[0] == 200


def test_deadline_cancels_generation(loaded, monkeypatch):
    stopped = threading.Event()

    def fake(*args, **kwargs):
        while not kwargs["cancellation"]():
            stopped.wait(0.01)
        stopped.set()
        raise GenerationCancelled("deadline")

    monkeypatch.setattr(serving, "generate_result", fake)
    with running(loaded, request_timeout=0.2) as server:
        assert request(server, "/v1/completions", payload())[0] == 408
        assert stopped.is_set()


def test_connection_limit_rejects_excess_requests(loaded, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def fake(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return GenerationResult(args[2], [], 1, 1, "stop")

    monkeypatch.setattr(serving, "generate_result", fake)
    with (
        running(loaded, max_clients=1) as server,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        first = pool.submit(request, server, "/v1/completions", payload())
        assert entered.wait(5)
        try:
            assert request(server, "/v1/models")[0] == 503
        finally:
            release.set()
        assert first.result(timeout=5)[0] == 200


def test_server_shutdown_cancels_active_generation(loaded, monkeypatch):
    entered = threading.Event()
    stopped = threading.Event()

    def fake(*args, **kwargs):
        entered.set()
        while not kwargs["cancellation"]():
            stopped.wait(0.01)
        stopped.set()
        raise GenerationCancelled("shutdown")

    monkeypatch.setattr(serving, "generate_result", fake)
    with running(loaded) as server, ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(request, server, "/v1/completions", payload())
        assert entered.wait(5)
        server.inference.closing.set()
        assert first.result(timeout=5)[0] == 408
        assert stopped.is_set()


def test_loopback_and_limits(loaded):
    with pytest.raises(ValueError, match="loopback"):
        LocalHTTPServer(("0.0.0.0", 0), LocalInference(loaded))
    for options in (
        {"max_clients": 0},
        {"max_new_tokens": -1},
        {"request_timeout": float("nan")},
        {"max_request_bytes": True},
    ):
        with pytest.raises(ValueError):
            ServerLimits(**options)


def test_cli_wires_loader_runtime_and_pinned_identity(loaded, monkeypatch, capsys):
    from klab.cli.serve import serve

    calls = []

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        return loaded

    monkeypatch.setattr("klab.evaluation.inference.load_run", loader)
    monkeypatch.setattr(LocalHTTPServer, "serve_forever", lambda *args, **kwargs: None)
    args = build_parser().parse_args(
        [
            "serve",
            "original",
            "--checkpoint",
            "chosen",
            "--backend",
            "cpu",
            "--port",
            "0",
            "--model-id",
            "fixed",
        ]
    )
    args.runtime_authorization = object()
    serve(args)
    result = json.loads(capsys.readouterr().out)
    assert result["identity"] == loaded.identity
    assert result["model"] == "fixed"
    assert calls[0][0][2:] == ("chosen", "cpu")
    assert calls[0][1]["authorization"] is args.runtime_authorization


def test_serve_rejects_corrupt_checkpoint_before_binding(loaded, tmp_path, monkeypatch):
    shutil.copytree(loaded.run, tmp_path / "bad")
    (tmp_path / "bad" / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        ["klab", "serve", "bad", "--runs-dir", str(tmp_path), "--backend", "cpu"],
    )

    def forbidden(*args, **kwargs):
        pytest.fail("invalid artifacts must fail before binding a server")

    monkeypatch.setattr(LocalHTTPServer, "__init__", forbidden)
    with pytest.raises((ValueError, SystemExit)):
        main()


def test_serve_runtime_gate_precedes_loader(loaded, monkeypatch, tmp_path):
    from klab.cli.main import _prepare_runtime_command

    config = loaded.config.model_copy(
        update={"runtime": loaded.config.runtime.model_copy(update={"backend": "cuda"})}
    )
    run = tmp_path / "gated"
    run.mkdir()
    (run / "resolved_config.yaml").write_text(config.model_dump_json())
    args = build_parser().parse_args(["serve", "gated", "--runs-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="runtime"):
        _prepare_runtime_command(args)


@pytest.mark.parametrize("changes", [{"prompt": "\ud800"}, {"stop": "\ud800"}])
def test_invalid_unicode_is_client_error(loaded, changes):
    with running(loaded) as server:
        assert request(server, "/v1/completions", payload(**changes))[0] == 400


def test_header_read_has_deadline_and_releases_slot(loaded):
    with running(loaded, max_clients=1, request_timeout=0.2) as server:
        connection = socket.create_connection(server.server_address)
        connection.settimeout(5)
        try:
            connection.sendall(
                f"POST /v1/completions HTTP/1.0\r\nHost: 127.0.0.1:{server.server_port}\r\nX-Incomplete: ".encode()
            )
            # Header parsing must terminate on its absolute deadline, even when
            # no POST handler/body watcher has begun. No elapsed-time assertion.
            response = bytearray()
            while chunk := connection.recv(1024):
                response.extend(chunk)
            assert not response or b"400" in response or b"408" in response
        finally:
            connection.close()


def test_incomplete_body_is_released_on_deadline(loaded):
    with running(loaded, request_timeout=0.2) as server:
        connection = socket.create_connection(server.server_address)
        connection.settimeout(5)
        try:
            connection.sendall(
                f"POST /v1/completions HTTP/1.0\r\nHost: 127.0.0.1:{server.server_port}\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{{".encode()
            )
            response = bytearray()
            while chunk := connection.recv(1024):
                response.extend(chunk)
            assert not response or b"400" in response or b"408" in response
        finally:
            connection.close()


def test_model_failure_releases_lock_and_returns_structured_error(loaded, monkeypatch):
    original = serving.generate_result

    def broken(*args, **kwargs):
        raise RuntimeError("private/path must not be returned")

    monkeypatch.setattr(serving, "generate_result", broken)
    with running(loaded) as server:
        status, error = request(server, "/v1/completions", payload())
        assert status == 500
        assert error["error"]["code"] == "inference_error"
        assert "private/path" not in json.dumps(error)
        monkeypatch.setattr(serving, "generate_result", original)
        assert request(server, "/v1/completions", payload())[0] == 200


def test_no_file_routes_checkpoint_options_or_cors_allowance(loaded):
    with running(loaded) as server:
        assert request(server, "/../../etc/passwd")[0] == 404
        assert (
            request(server, "/v1/completions", payload(checkpoint="/etc/passwd"))[0]
            == 400
        )
        assert (
            request(
                server,
                "/v1/completions",
                payload(model="https://example.invalid/weights"),
            )[0]
            == 404
        )
        connection = http.client.HTTPConnection(*server.server_address[:2], timeout=5)
        try:
            connection.request(
                "GET", "/v1/models", headers={"Origin": "https://example.invalid"}
            )
            response = connection.getresponse()
            assert response.status == 403
            assert response.getheader("Access-Control-Allow-Origin") is None
            response.read()
        finally:
            connection.close()


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize(
    "hosts,origins,code",
    [
        ([], [], "invalid_host"),
        (["evil.invalid:{port}"], [], "invalid_host"),
        (["localhost:{port}"], ["null"], "invalid_origin"),
        (["localhost:{port}"], ["http://evil.invalid:{port}"], "invalid_origin"),
        (["localhost:{port}"], ["http://localhost:{port}"] * 2, "invalid_origin"),
    ],
)
def test_authority_rejected_before_metadata_or_inference(
    loaded, monkeypatch, method, hosts, origins, code
):
    def forbidden(*args, **kwargs):
        pytest.fail("rejected request reached model access")

    with running(loaded) as server:
        monkeypatch.setattr(server.inference, "models", forbidden)
        monkeypatch.setattr(server.inference, "complete", forbidden)
        connection = http.client.HTTPConnection(*server.server_address[:2], timeout=5)
        try:
            connection.putrequest(
                method,
                "/v1/models" if method == "GET" else "/v1/completions",
                skip_host=True,
            )
            for host in hosts:
                connection.putheader("Host", host.format(port=server.server_port))
            for origin in origins:
                connection.putheader("Origin", origin.format(port=server.server_port))
            # No body: invalid authority must fail before a body read.
            connection.putheader("Content-Length", "100")
            connection.endheaders()
            response = connection.getresponse()
            assert response.status == 403
            assert json.loads(response.read())["error"]["code"] == code
        finally:
            connection.close()


@pytest.mark.parametrize("origin", [False, True])
def test_native_and_same_origin_authorities(loaded, origin):
    with running(loaded) as server:
        authority = f"LOCALHOST:{server.server_port}"
        headers = {"Host": authority, "Content-Type": "application/json"}
        if origin:
            headers["Origin"] = f"http://{authority}"
        assert request(server, "/v1/models", headers=headers)[0] == 200
        assert request(server, "/v1/completions", payload(), headers=headers)[0] == 200


def test_default_http_port_authorities(loaded, monkeypatch):
    def bind_stub(server, address, handler):
        server.server_address = address

    monkeypatch.setattr(serving.ThreadingHTTPServer, "__init__", bind_stub)
    server = LocalHTTPServer(
        ("127.0.0.1", 80), LocalInference(loaded, "test-model", ServerLimits())
    )
    assert server.allowed_authorities == {
        "127.0.0.1": "127.0.0.1:80",
        "127.0.0.1:80": "127.0.0.1:80",
        "localhost": "localhost:80",
        "localhost:80": "localhost:80",
    }


def test_ipv6_authorities(loaded):
    app = LocalInference(loaded, "test-model", ServerLimits())
    try:
        server = LocalHTTPServer(("::1", 0), app)
    except OSError:
        pytest.skip("IPv6 loopback is unavailable")
    with server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            for host in ("[::1]", "localhost"):
                authority = f"{host}:{server.server_port}"
                assert (
                    request(
                        server,
                        "/v1/models",
                        headers={"Host": authority, "Origin": f"http://{authority}"},
                    )[0]
                    == 200
                )
            for host in ("::1", "127.0.0.1", "evil.invalid"):
                assert (
                    request(
                        server,
                        "/v1/models",
                        headers={"Host": f"{host}:{server.server_port}"},
                    )[0]
                    == 403
                )
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_numeric_loopback_binding_does_not_resolve_dns(loaded, monkeypatch):
    host = "127.0.0.1"

    def forbidden(*args, **kwargs):
        pytest.fail("numeric loopback binding must not call getfqdn")

    monkeypatch.setattr(socket, "getfqdn", forbidden)
    with LocalHTTPServer((host, 0), LocalInference(loaded)) as server:
        assert server.server_name == host
        assert server.server_port == server.socket.getsockname()[1]
        assert server.server_port > 0
        assert f"{host}:{server.server_port}" in server.allowed_authorities


def test_real_cli_process_serves_pinned_model_and_closes_on_sigint(loaded, tmp_path):
    import os
    import signal
    import subprocess
    import time
    from urllib.parse import urlsplit

    environment = {**os.environ, "KLAB_WORK_DIR": str(tmp_path / "work")}
    stdout_path = tmp_path / "serve.stdout.log"
    stderr_path = tmp_path / "serve.stderr.log"
    # Files avoid an unread stderr pipe blocking startup, and preserve diagnostics
    # for early exits, readiness deadlines, and failures after readiness alike.
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = None
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    # Execute the real module entry point, with child stacks
                    # retained if imports, validation, or socket startup stall.
                    (
                        "import faulthandler, runpy, sys; "
                        "print('CLI child entered', file=sys.stderr, flush=True); "
                        "faulthandler.dump_traceback_later(10, repeat=True); "
                        "runpy.run_module('klab', run_name='__main__', alter_sys=True)"
                    ),
                    "serve",
                    loaded.run.name,
                    "--runs-dir",
                    str(loaded.run.parent),
                    "--checkpoint",
                    loaded.identity["checkpoint_relative_path"],
                    "--backend",
                    "cpu",
                    "--model-id",
                    "cli-fixture",
                    "--port",
                    "0",
                ],
                stdout=stdout,
                stderr=stderr,
                env=environment,
            )
            deadline = time.monotonic() + 30
            while True:
                returncode = process.poll()
                assert returncode is None, (
                    f"server exited before readiness (returncode={returncode})"
                )
                output = stdout_path.read_bytes()
                if b"\n" in output:
                    announcement = json.loads(output.split(b"\n", 1)[0])
                    break
                remaining = deadline - time.monotonic()
                assert remaining > 0, (
                    "server still running; readiness deadline exceeded (30s)"
                )
                time.sleep(min(0.05, remaining))
            assert (
                announcement["identity"]["checkpoint_sha256"]
                == loaded.identity["checkpoint_sha256"]
            )
            url = urlsplit(announcement["url"])
            connection = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
            try:
                connection.request(
                    "POST",
                    "/v1/completions",
                    json.dumps(
                        {"model": "cli-fixture", "prompt": "hello", "max_tokens": 2}
                    ),
                    {"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                assert response.status == 200
                assert json.loads(response.read())["model"] == "cli-fixture"
            finally:
                connection.close()
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=15) == 0
            with pytest.raises(OSError):
                socket.create_connection((url.hostname, url.port), timeout=1)
        except BaseException as error:
            before_cleanup = process.poll() if process is not None else "not started"
            if process is not None and before_cleanup is None:
                try:
                    process.kill()
                    process.wait(timeout=15)
                except (OSError, subprocess.TimeoutExpired) as cleanup_error:
                    error.add_note(f"Server cleanup failed: {cleanup_error!r}")
            final_status = process.poll() if process is not None else "not started"
            error.add_note(
                f"Server status before cleanup: {before_cleanup}; "
                f"final returncode: {final_status}"
            )
            for path in (stdout_path, stderr_path):
                with path.open("rb") as log:
                    log.seek(max(0, path.stat().st_size - 16384))
                    tail = log.read().decode("utf-8", errors="replace")
                error.add_note(f"{path} (last 16 KiB):\n{tail}")
            raise
