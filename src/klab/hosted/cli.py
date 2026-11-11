"""Native commands for existing endpoints; never allocate provider resources."""

from __future__ import annotations

import argparse
import base64
import json
import math
import secrets
import shlex
import tempfile
import time
from pathlib import Path

from klab.hosted.bootstrap import (
    diagnostic_program,
    preflight_program,
    setup_program,
)
from klab.hosted.bootstrap_worker import strict_json
from klab.hosted.kernel import notebook_cell_program
from klab.hosted.models import HostedTarget, SetupRequest
from klab.hosted.transport import (
    ColabFileTransport,
    TransferManifest,
    assembly_python,
)
from klab.runtime_env_subprocess import run_bounded
from klab.training.manifest import sha256_file
from klab.workdir import ensure_scratch_dir


def add_parser(commands: argparse._SubParsersAction) -> argparse.ArgumentParser:
    hosted = commands.add_parser(
        "hosted", help="inspect or bootstrap an existing hosted endpoint"
    )
    operations = hosted.add_subparsers(dest="hosted_command", required=True)
    cell = operations.add_parser(
        "notebook-cell", help="write a fixed, user-run Colab enrollment cell"
    )
    cell.add_argument("--root", type=Path, required=True)
    cell.add_argument("--output", type=Path, required=True)
    cell.add_argument("--json", action="store_true")
    for name, default_timeout in (("inspect", 60), ("setup", 1800)):
        command = operations.add_parser(name)
        endpoint = command.add_mutually_exclusive_group(required=True)
        endpoint.add_argument("--colab-session")
        endpoint.add_argument("--ssh")
        command.add_argument("--root", type=Path, required=name == "setup")
        command.add_argument("--colab-config", type=Path)
        command.add_argument(
            "--colab-auth", choices=("oauth2", "adc"), default="oauth2"
        )
        command.add_argument("--timeout", type=float, default=default_timeout)
        command.add_argument("--json", action="store_true")
        if name == "setup":
            command.add_argument("--source-bundle", type=Path, required=True)
            command.add_argument("--source-commit", required=True)
            command.add_argument("--runtime-root", type=Path, required=True)
            command.add_argument(
                "--recipe", choices=("cuda-cu126-v1",), default="cuda-cu126-v1"
            )
    hosted.set_defaults(handler=_execute_command)
    return hosted


def _remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("hosted operation deadline exhausted")
    return value


def _colab_command(target: HostedTarget, script: Path, timeout: float) -> list[str]:
    command = ["colab"]
    if target.colab_config is not None:
        command.extend(["--config", str(target.colab_config)])
    command.extend(
        [
            "--auth",
            target.colab_auth,
            "exec",
            "-s",
            str(target.colab_session),
            "-f",
            str(script),
            "--timeout",
            str(timeout),
        ]
    )
    return command


def _ssh_command(target: HostedTarget, program: str) -> list[str]:
    encoded = base64.b64encode(program.encode()).decode()
    runner = f"import base64;exec(compile(base64.b64decode({encoded!r}),'<klab-hosted>','exec'))"
    return [
        "ssh",
        "-oBatchMode=yes",
        "-oStrictHostKeyChecking=yes",
        str(target.ssh),
        " ".join(shlex.quote(part) for part in ("python3", "-c", runner)),
    ]


def _idle(target: HostedTarget, scratch: Path, deadline: float) -> dict | None:
    if target.colab_session is None:
        return None
    from klab.workers.colab import verify_kernel_observer

    root = target.root or Path("/content/klab")
    transport = ColabFileTransport(
        target.colab_session,
        config_path=target.colab_config,
        auth=target.colab_auth,
        timeout=_remaining(deadline),
        deadline=deadline,
    )
    try:
        proof = verify_kernel_observer(
            transport, str(root / "work" / "worker"), scratch / "kernel-status.json"
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise ValueError(
            "Colab kernel occupancy is unknown; run the fixed hosted notebook-cell in the dedicated idle notebook first"
        ) from error
    if proof["state"] != "IDLE":
        raise ValueError("Colab kernel is busy; no execution was submitted")
    return proof


def _remote_json(
    target: HostedTarget, program: str, timeout: float, *, scratch: Path | None = None
) -> dict:
    if scratch is None:
        with tempfile.TemporaryDirectory(
            prefix="hosted-control-", dir=ensure_scratch_dir()
        ) as directory:
            return _remote_json(target, program, timeout, scratch=Path(directory))
    if target.colab_session is not None:
        script = scratch / f"exec-{secrets.token_hex(8)}.py"
        with script.open("x", encoding="utf-8") as output:
            output.write(program)
        script.chmod(0o600)
        try:
            result = run_bounded(
                _colab_command(target, script, timeout),
                timeout=timeout,
                output_limit=1024 * 1024,
            )
        finally:
            script.unlink(missing_ok=True)
    else:
        result = run_bounded(
            _ssh_command(target, program), timeout=timeout, output_limit=1024 * 1024
        )
    try:
        answer = strict_json(result.stdout)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(
            "hosted operation returned invalid JSON: "
            + result.stderr.decode("utf-8", "replace")[-2048:]
        ) from error
    if not isinstance(answer, dict):
        raise ValueError("hosted operation returned non-object JSON")  # noqa: TRY004 — invalid wire data
    if "hosted_error_version" in answer:
        if (
            set(answer) != {"hosted_error_version", "error_type", "message"}
            or type(answer["hosted_error_version"]) is not int
            or answer["hosted_error_version"] != 1
            or not isinstance(answer["error_type"], str)
            or not isinstance(answer["message"], str)
        ):
            raise ValueError("invalid hosted bootstrap error response")
        raise ValueError(
            f"hosted bootstrap failed: {answer['error_type']}: {answer['message']}"
        )
    if result.returncode:
        raise ValueError(
            result.stderr.decode("utf-8", "replace")[-2048:]
            or "hosted operation failed"
        )
    return answer


def inspect(target: HostedTarget, *, timeout: float = 60) -> dict:
    if not math.isfinite(timeout) or not 0 < timeout <= 7200:
        raise ValueError("timeout must be finite and in 0..7200 seconds")
    deadline = time.monotonic() + timeout
    with tempfile.TemporaryDirectory(
        prefix="hosted-inspect-", dir=ensure_scratch_dir()
    ) as directory:
        scratch = Path(directory)
        _idle(target, scratch, deadline)
        root = target.root or (Path("/content") if target.colab_session else Path("~"))
        answer = _remote_json(
            target,
            diagnostic_program(str(root), timeout=_remaining(deadline)),
            _remaining(deadline),
            scratch=scratch,
        )
    answer.update(
        hosted_inspect_version=1,
        requested="colab" if target.colab_session else "ssh",
        storage_scope="explicit" if target.root else "default",
    )
    if answer.get("tpu"):
        answer.update(
            status="TPU_UNSUPPORTED",
            reason="TPU observed; PyTorch/XLA training is not implemented",
        )
    elif not answer.get("nvidia_available"):
        answer.update(status="CUDA_UNAVAILABLE", reason="no NVIDIA GPU was observed")
    elif answer.get("cuda_available") is not True or answer.get("hip"):
        answer.update(
            status="FRAMEWORK_NOT_READY",
            reason="NVIDIA GPU observed; current framework does not verify CUDA",
        )
    else:
        answer["status"] = "READY"
    return answer


def setup(request: SetupRequest) -> dict:
    if not request.source_bundle.is_file() or request.source_bundle.is_symlink():
        raise ValueError("source_bundle must be an existing regular file")
    size = request.source_bundle.stat().st_size
    if not 0 < size <= 4 * 1024**3:
        raise ValueError("source bundle must contain 1 byte..4 GiB")
    deadline = time.monotonic() + request.timeout
    digest = sha256_file(request.source_bundle)
    with tempfile.TemporaryDirectory(
        prefix="hosted-setup-", dir=ensure_scratch_dir()
    ) as directory:
        scratch = Path(directory)
        verifier = scratch / "verify.git"
        for command in (
            ["git", "init", "--bare", str(verifier)],
            [
                "git",
                "-C",
                str(verifier),
                "bundle",
                "verify",
                str(request.source_bundle),
            ],
        ):
            checked = run_bounded(
                command, timeout=_remaining(deadline), output_limit=32768
            )
            if checked.returncode:
                raise ValueError("source_bundle is not a complete valid Git bundle")
        idle = _idle(request, scratch, deadline)
        observed = _remote_json(
            request,
            diagnostic_program(str(request.root), timeout=_remaining(deadline)),
            _remaining(deadline),
            scratch=scratch,
        )
        if observed.get("tpu"):
            raise ValueError("TPU is unsupported; PyTorch/XLA is required")
        if not observed.get("nvidia_available"):
            raise ValueError("CUDA setup requires an observed NVIDIA GPU")
        preflight = _remote_json(
            request,
            preflight_program(
                root=str(request.root),
                runtime_root=str(request.runtime_root),
                source_bytes=size,
                source_commit=request.source_commit,
                source_sha256=digest,
                recipe=request.recipe,
                instance_id=idle["instance_id"] if idle else None,
                timeout=_remaining(deadline),
            ),
            _remaining(deadline),
            scratch=scratch,
        )
        if preflight["reused"]:
            return preflight["result"]
        assert request.root is not None
        remote_bundle = str(
            request.root / "scratch" / f"source-{request.source_commit}.bundle"
        )
        nonce = idle["instance_id"] if idle else secrets.token_hex(32)
        if request.colab_session is not None:
            transport = ColabFileTransport(
                request.colab_session,
                config_path=request.colab_config,
                auth=request.colab_auth,
                timeout=_remaining(deadline),
                deadline=deadline,
            )
            manifest = transport.upload(request.source_bundle, remote_bundle)
        else:
            parts = (size + 16 * 1024**2 - 1) // (16 * 1024**2)
            with request.source_bundle.open("rb") as source:
                for index in range(parts):
                    chunk = scratch / f"bundle-{index:08d}"
                    chunk.write_bytes(source.read(16 * 1024**2))
                    chunk.chmod(0o600)
                    copied = run_bounded(
                        [
                            "scp",
                            "-oBatchMode=yes",
                            "-oStrictHostKeyChecking=yes",
                            str(chunk),
                            f"{request.ssh}:{remote_bundle}.part-{index:08d}",
                        ],
                        timeout=_remaining(deadline),
                        output_limit=32768,
                    )
                    chunk.unlink(missing_ok=True)
                    if copied.returncode:
                        raise ValueError(
                            copied.stderr.decode("utf-8", "replace")[:1024]
                            or "strict SSH source transfer failed"
                        )
            manifest = TransferManifest(remote_bundle, digest, size, parts)
        _idle(request, scratch, deadline)
        program = setup_program(
            root=str(request.root),
            runtime_root=str(request.runtime_root),
            source_bundle=remote_bundle,
            source_commit=request.source_commit,
            source_sha256=digest,
            recipe=request.recipe,
            instance_id=nonce,
            timeout=_remaining(deadline),
            assembly=assembly_python(manifest),
        )
        answer = _remote_json(request, program, _remaining(deadline), scratch=scratch)
    if answer.get("status") != "READY":
        raise ValueError("hosted setup did not return a verified READY receipt")
    return answer


def _execute_command(args: argparse.Namespace) -> None:
    print(json.dumps(execute(args), indent=2, sort_keys=True))


def execute(args: argparse.Namespace) -> dict:
    if args.hosted_command == "notebook-cell":
        if not args.output.is_absolute() or args.output.is_symlink():
            raise ValueError("notebook cell output must be an absolute fresh file")
        program = notebook_cell_program(args.root)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(program)
        args.output.chmod(0o600)
        return {
            "path": str(args.output),
            "action": "Run this fixed cell manually in the dedicated Colab notebook; this command submitted no execution.",
        }
    target = HostedTarget.model_validate(
        {
            "colab_session": args.colab_session,
            "ssh": args.ssh,
            "root": args.root,
            "colab_config": args.colab_config,
            "colab_auth": args.colab_auth,
        }
    )
    if args.hosted_command == "inspect":
        return inspect(target, timeout=args.timeout)
    return setup(
        SetupRequest.model_validate(
            {
                **target.model_dump(),
                "source_bundle": args.source_bundle,
                "source_commit": args.source_commit,
                "runtime_root": args.runtime_root,
                "recipe": args.recipe,
                "timeout": args.timeout,
            }
        )
    )
