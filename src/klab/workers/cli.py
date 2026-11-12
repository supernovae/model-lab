"""Command handlers for independent local and SSH workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import secrets
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from klab.config.loading import load_config
from klab.data.encoding import (
    TOKENIZER_BATCH_DOCUMENTS,
    TOKENIZER_BATCH_SOURCE_BYTES,
)
from klab.experiments.matrix import expand
from klab.training.manifest import config_sha256
from klab.workdir import ensure_scratch_dir

_BACKENDS = ("cpu", "mps", "cuda", "rocm", "xpu", "metal")
_ENGINES = ("pytorch", "mlx")
_TRANSFER_TIMEOUT = 1800
_TERMINAL_QUEUE_STATES = frozenset({"CANCELLED", "COMPLETE", "FAILED", "INTERRUPTED"})


def _json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, default=str))


def _controller(store: Path, *, transfer_timeout: float = _TRANSFER_TIMEOUT):
    from klab.workers.controller import Controller

    return Controller(store, transfer_timeout=transfer_timeout)


def _absolute(value: str, *, argument: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{argument} must be an absolute path")
    return path


def _relay_check(args: argparse.Namespace) -> None:
    from klab.training.manifest import sha256_file

    from .relay import RelayStore
    from .relay_models import RelayArtifactIdentity, load_relay_profile

    profile = load_relay_profile(args.profile)
    controller = _controller(Path(args.store))
    with tempfile.TemporaryDirectory(
        prefix=".relay-check-", dir=controller.root
    ) as directory:
        source = Path(directory) / "challenge.bin"
        source.write_bytes(secrets.token_bytes(32))
        identity = RelayArtifactIdentity(
            relative_path="verification/challenge.bin",
            sha256=sha256_file(source),
            size_bytes=32,
        )
        relay = RelayStore(profile, scratch_root=controller.root / ".relay" / "scratch")
        relay.put_verified(source, identity)
        relay.get_verified(identity, Path(directory) / "controller-readback.bin")
        if args.worker is not None:
            definitions = [
                controller._model("WorkerDefinition", row["definition"])
                for row in controller.store.worker_records()
                if row["definition"]["name"] == args.worker
            ]
            if len(definitions) != 1:
                raise ValueError("worker unavailable or ambiguous")
            result = controller._rpc_result(
                definitions[0],
                "configure_relay",
                {
                    "binding": profile.binding()
                    .worker_binding()
                    .model_dump(mode="json"),
                    "challenge": identity.model_dump(mode="json"),
                },
            )
        else:
            worker_store = RelayStore(
                profile,
                role="worker",
                scratch_root=controller.root / ".relay" / "scratch",
            )
            worker_store.get_verified(identity, Path(directory) / "worker-readback.bin")
            result = {"verified_challenge_sha256": identity.sha256}
        if result["verified_challenge_sha256"] != identity.sha256:
            raise ValueError("relay bindings do not address the same namespace")
        receipt = {
            "relay_check_version": 1,
            "namespace": profile.namespace,
            "sha256": identity.sha256,
            "worker": args.worker,
            "status": "VERIFIED",
            "remote_quota": None,
            "remote_quota_reason": "Not exposed by this binding.",
        }
        from .execution import _atomic_json

        _atomic_json(
            controller.root / ".relay" / "verification" / f"{identity.sha256}.json",
            receipt,
        )
        _json(receipt)


def _experiment_collect(args: argparse.Namespace) -> None:
    controller = _controller(Path(args.store))
    if args.relay_profile is not None:
        from .relay_collection import collect_relay
        from .relay_models import load_relay_profile

        result = collect_relay(
            controller,
            args.run_id,
            load_relay_profile(args.relay_profile),
            recover=args.recover,
            confirm_worker_lost=args.confirm_worker_lost,
        )
    else:
        if args.recover or args.confirm_worker_lost:
            raise ValueError("lost-worker recovery requires --relay-profile")
        attempt = controller.store.attempt_by_run(args.run_id)
        if attempt is None:
            raise KeyError(f"unknown run: {args.run_id}")
        worker = controller._worker_for_attempt(attempt)
        if worker is not None and worker.relay is not None:
            from .relay_collection import collect_relay

            result = collect_relay(controller, args.run_id, worker.relay)
        else:
            controller._poll_active(attempt)
            result = controller.store.attempt_by_run(args.run_id)
    _json(result)


def _collect_dispatch(args: argparse.Namespace) -> None:
    if Path(args.lock).is_file() and args.relay_profile is None and not args.recover:
        from klab.experiments.cli import _handle

        _handle(args)
        return
    args.run_id = args.lock
    _experiment_collect(args)


def _worker_definition(args: argparse.Namespace):
    from klab.workers.models import WorkerDefinition

    if args.backend != "cpu" and not args.python:
        raise ValueError(
            "accelerator worker registration requires an explicit absolute --python"
        )
    store = Path(args.store).resolve()
    ssh_host = args.ssh
    colab_session = args.colab_session
    relay = None
    if args.relay_profile is not None:
        from .relay_models import load_relay_profile

        relay = load_relay_profile(Path(args.relay_profile)).binding()
    colab = None
    if ssh_host is not None or colab_session is not None:
        python = _absolute(args.python, argument="--python") if args.python else None
        root = _absolute(args.root, argument="--root") if args.root else None
        if python is None or root is None:
            raise ValueError("remote worker registration requires --python and --root")
        transport = "colab" if colab_session is not None else "ssh"
        if colab_session is not None:
            from klab.hosted.transport import ColabFileTransport

            from .execution import _strict_json
            from .models import ColabEndpoint

            if relay is None:
                raise ValueError("Colab worker registration requires --relay-profile")
            with tempfile.TemporaryDirectory(
                prefix=".colab-register-", dir=ensure_scratch_dir()
            ) as directory:
                path = Path(directory) / "instance.json"
                ColabFileTransport(
                    colab_session,
                    config_path=args.colab_config,
                    auth=args.colab_auth or "oauth2",
                    timeout=60,
                ).download(str(root / "hosted-instance.json"), path)
                instance = _strict_json(path)
            if set(instance) != {"instance_id", "boot_id"}:
                raise ValueError("invalid hosted instance identity")
            colab = ColabEndpoint(
                session=colab_session,
                config_path=args.colab_config,
                auth=args.colab_auth,
                instance_id=instance["instance_id"],
                job_timeout_seconds=args.colab_job_timeout,
            )
    else:
        transport = "local"
        python = (
            _absolute(args.python, argument="--python")
            if args.python
            else Path(sys.executable).absolute()
        )
        root = (
            _absolute(args.root, argument="--root")
            if args.root
            else (store.parent / "workers" / args.name).resolve()
        )
    return WorkerDefinition(
        schema_version=1,
        worker_id=args.name,
        name=args.name,
        transport=transport,
        host=ssh_host,
        python=python,
        root=root,
        engine=args.engine,
        backend=args.backend,
        device_index=args.device_index,
        relay=relay,
        colab=colab,
    )


def _worker_register(args: argparse.Namespace) -> None:
    capability = _controller(Path(args.store)).register(_worker_definition(args))
    _json(capability.model_dump(mode="json"))


def _worker_status(args: argparse.Namespace) -> None:
    capabilities = _controller(Path(args.store)).workers(args.name, refresh=True)
    payload = [capability.model_dump(mode="json") for capability in capabilities]
    _json(payload if args.name is None else (payload[0] if payload else None))


def _serve_stdio(args: argparse.Namespace) -> None:
    """Run the finite stdio endpoint used by both local and SSH transport."""
    from klab.workers.agent import serve_stdio
    from klab.workers.models import WorkerDefinition

    definition = WorkerDefinition(
        schema_version=1,
        worker_id=args.worker_id,
        name=args.name,
        transport="local",
        host=None,
        python=Path(sys.executable).absolute(),
        root=_absolute(args.root, argument="--root"),
        engine=args.engine,
        backend=args.backend,
        device_index=args.device_index,
    )
    serve_stdio(definition)


def _execute(args: argparse.Namespace) -> None:
    from klab.workers.agent import execute_attempt
    from klab.workers.models import WorkerDefinition

    definition = WorkerDefinition(
        schema_version=1,
        worker_id=args.worker_id,
        name=args.name,
        transport="local",
        host=None,
        python=Path(sys.executable).absolute(),
        root=_absolute(args.root, argument="--root"),
        engine=args.engine,
        backend=args.backend,
        device_index=args.device_index,
    )
    execute_attempt(definition, args.attempt_id)


def _matrix_requests(args: argparse.Namespace) -> list[dict[str, object]]:
    expanded = expand(Path(args.matrix), max_runs=args.max_runs)
    if args.dry_run:
        _json(
            [
                {
                    "coordinate": item.coordinate,
                    "config_sha256": config_sha256(item.config.model_dump(mode="json")),
                    "matrix_sha256": item.matrix_sha256,
                    "preferred_worker": item.preferred_worker,
                    "requirements": item.requirements.model_dump(mode="json"),
                }
                for item in expanded
            ]
        )
        return []
    return [
        {
            "config": item.config,
            "worker": args.worker,
            "stage_bundle": Path(args.stage_bundle) if args.stage_bundle else None,
            "promote": Path(args.promote) if args.promote else None,
            "requirements": item.requirements,
            "preferred": item.preferred_worker,
            "matrix": {
                "matrix_sha256": item.matrix_sha256,
                "coordinate": item.coordinate,
            },
        }
        for item in expanded
    ]


def validate_submit_args(args: argparse.Namespace) -> None:
    """Reject incompatible submission shapes before any execution preflight."""
    if bool(args.config) == bool(args.matrix):
        raise ValueError("provide exactly one of CONFIG or --matrix")
    if args.matrix and (
        args.wait
        or args.runtime is not None
        or args.runtime_profile is not None
        or args.resource_envelope is not None
        or args.tokenizer_batch_documents is not None
        or args.tokenizer_batch_source_bytes is not None
        or args.transfer_timeout is not None
    ):
        raise ValueError(
            "--wait, runtime selection, resource envelope, tokenizer batching and "
            "transfer timeout require CONFIG, not --matrix"
        )
    if args.dry_run and not args.matrix:
        raise ValueError("--dry-run is only available with --matrix")


def _experiment_submit(args: argparse.Namespace) -> None:
    validate_submit_args(args)
    if args.matrix:
        requests = _matrix_requests(args)
        if args.dry_run:
            return
        submissions = _controller(Path(args.store)).submit_many(requests)
        _json([submission.model_dump(mode="json") for submission in submissions])
        return
    config = load_config(Path(args.config))
    controller, submission = _submit_config(args, config)
    if args.wait:
        _wait_for_submission(controller, submission)
    else:
        _json(submission.model_dump(mode="json"))


def _experiment_list(args: argparse.Namespace) -> None:
    _json(_controller(Path(args.store)).list_experiments())


def _experiment_cancel(args: argparse.Namespace) -> None:
    _json(_controller(Path(args.store)).cancel(args.run_id))


def _experiment_resume(args: argparse.Namespace) -> None:
    submission = _controller(Path(args.store)).resume(
        args.run_id,
        worker=args.worker,
        allow_runtime_drift=args.allow_runtime_drift,
    )
    _json(submission.model_dump(mode="json"))


def _controller_run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _controller(Path(args.store), transfer_timeout=args.transfer_timeout).run()


def _wait_for_run(controller: Any, run_id: str) -> dict[str, object]:
    """Observe an existing controller, or exclusively drive the submitted run."""
    previous: tuple[object, ...] | None = None
    with controller._process_lock(optional=True) as acquired:
        while True:
            if acquired:
                controller.tick()
            row = next(
                (
                    item
                    for item in controller.list_experiments()
                    if item.get("run_id") == run_id
                ),
                None,
            )
            if row is None:
                raise RuntimeError(
                    f"submitted run {run_id} disappeared from controller"
                )
            state = (
                row.get("status"),
                row.get("queued_reason"),
                row.get("ingestion_status"),
                row.get("ingestion_error"),
            )
            if state != previous:
                print(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "status": state[0],
                            "queued_reason": state[1],
                            "ingestion_status": state[2],
                            "ingestion_error": state[3],
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                previous = state
            if state[0] in _TERMINAL_QUEUE_STATES and (
                state[2] in {"COMPLETE", "NOT_REQUIRED"}
            ):
                return row
            worker = controller._worker_for_attempt(row)
            if (
                worker is not None
                and worker.transport == "colab"
                and state[0] == "UNKNOWN"
            ):
                print(
                    f"Colab observation UNKNOWN; do not resubmit. Use experiment collect {run_id} "
                    "--relay-profile PROFILE --store STORE; confirmed loss requires "
                    "--recover --confirm-worker-lost.",
                    file=sys.stderr,
                )
                return row
            time.sleep(controller.poll_seconds)


def _submit_config(args: argparse.Namespace, config: Any) -> tuple[Any, Any]:
    worker_name = args.worker
    local = worker_name is None and (args.wait or args.runtime or args.runtime_profile)
    if local:
        from klab.staging import inspect_runtime
        from klab.workers.models import WorkerDefinition

        runtime = inspect_runtime(config)
        if runtime.backend != "cpu" and args.runtime_authorization is None:
            raise ValueError(
                f"selected backend {runtime.backend} requires --runtime-profile "
                "or a registered --worker"
            )
    controller = _controller(
        Path(args.store),
        transfer_timeout=(
            args.transfer_timeout
            if args.transfer_timeout is not None
            else _TRANSFER_TIMEOUT
        ),
    )
    if local:
        root = controller.root.resolve()
        namespace = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
        worker_name = f"local-{runtime.engine}-{runtime.backend}-{config.runtime.device_index}-{namespace}"
        controller.register(
            WorkerDefinition(
                worker_id=worker_name,
                name=worker_name,
                transport="local",
                python=(
                    args.runtime_profile_python
                    if args.runtime_authorization is not None
                    else Path(sys.executable).resolve()
                ),
                root=root / ".workers" / worker_name,
                engine=runtime.engine,
                backend=runtime.backend,
                device_index=config.runtime.device_index,
            )
        )
    submission = controller.submit(
        config,
        worker=worker_name,
        stage_bundle=Path(args.stage_bundle) if args.stage_bundle else None,
        promote=Path(args.promote) if args.promote else None,
        resource_envelope=args.resource_envelope_value,
        tokenizer_batch_documents=(
            args.tokenizer_batch_documents
            if args.tokenizer_batch_documents is not None
            else TOKENIZER_BATCH_DOCUMENTS
        ),
        tokenizer_batch_source_bytes=(
            args.tokenizer_batch_source_bytes
            if args.tokenizer_batch_source_bytes is not None
            else TOKENIZER_BATCH_SOURCE_BYTES
        ),
    )
    return controller, submission


def _wait_for_submission(controller: Any, submission: Any) -> None:
    try:
        result = _wait_for_run(controller, submission.run_id)
    except KeyboardInterrupt:
        from shlex import quote

        run_id = quote(str(submission.run_id))
        store = quote(str(controller.root))
        print(
            "Controller disconnected; no cancellation requested. Worker execution may still be active.\n"
            f"Reconcile: klab experiment collect {run_id} --store {store}\n"
            f"Cancel explicitly: klab experiment cancel {run_id} --store {store}",
            file=sys.stderr,
            flush=True,
        )
        raise
    _json({"submission": submission.model_dump(mode="json"), "result": result})
    if result["status"] != "COMPLETE":
        raise SystemExit(1)


def _store_argument(parser: argparse.ArgumentParser, default_store: Path) -> None:
    parser.add_argument(
        "--store", default=str(default_store), help="Controller state root"
    )


def _endpoint_arguments(parser: argparse.ArgumentParser, *, attempt: bool) -> None:
    parser.add_argument("--root", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--engine", choices=_ENGINES, required=True)
    parser.add_argument("--backend", choices=_BACKENDS, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    if attempt:
        parser.add_argument("--attempt-id", required=True)


def add_commands(
    subparsers: argparse._SubParsersAction, *, default_store: Path
) -> None:
    """Add the worker/controller/experiment command families to the main parser."""
    worker = subparsers.add_parser(
        "worker", help="Register and inspect independent workers"
    )
    worker_commands = worker.add_subparsers(dest="worker_command", required=True)
    register = worker_commands.add_parser("register")
    register.add_argument("name")
    register.add_argument("--backend", choices=_BACKENDS, required=True)
    register.add_argument("--engine", choices=_ENGINES, default="pytorch")
    register.add_argument("--device-index", type=int, default=0)
    endpoint = register.add_mutually_exclusive_group()
    endpoint.add_argument("--ssh")
    endpoint.add_argument("--colab-session")
    register.add_argument("--colab-config", type=Path)
    register.add_argument("--colab-auth", choices=("oauth2", "adc"))
    register.add_argument("--colab-job-timeout", type=float, default=3600)
    register.add_argument("--relay-profile", type=Path)
    register.add_argument("--python")
    register.add_argument("--root")
    _store_argument(register, default_store)
    register.set_defaults(handler=_worker_register)
    status = worker_commands.add_parser("status")
    status.add_argument("name", nargs="?")
    status.add_argument("--json", action="store_true")
    _store_argument(status, default_store)
    status.set_defaults(handler=_worker_status)
    serve = worker_commands.add_parser("serve-stdio", help=argparse.SUPPRESS)
    _endpoint_arguments(serve, attempt=False)
    serve.set_defaults(handler=_serve_stdio)
    execute = worker_commands.add_parser("execute", help=argparse.SUPPRESS)
    _endpoint_arguments(execute, attempt=True)
    execute.set_defaults(handler=_execute)

    relay = worker_commands.add_parser("relay")
    relay_commands = relay.add_subparsers(dest="relay_command", required=True)
    relay_check = relay_commands.add_parser("check")
    relay_check.add_argument("profile", type=Path)
    relay_check.add_argument("--worker")
    relay_check.add_argument("--json", action="store_true", required=True)
    _store_argument(relay_check, default_store)
    relay_check.set_defaults(handler=_relay_check)
    experiment = subparsers.add_parser(
        "experiment", help="Submit and manage independent experiments"
    )
    experiment_commands = experiment.add_subparsers(
        dest="experiment_command", required=True
    )
    submit = experiment_commands.add_parser("submit")
    submit.add_argument("config", nargs="?")
    submit.add_argument("--matrix")
    runtime_source = submit.add_mutually_exclusive_group()
    runtime_source.add_argument("--worker")
    runtime_source.add_argument("--runtime-profile", type=Path)
    runtime_source.add_argument("--runtime", metavar="ID")
    submit.add_argument(
        "--wait",
        action="store_true",
        help="Drive or observe one CONFIG until terminal artifact ingestion; defaults to a local worker",
    )
    submit.add_argument("--resource-envelope", type=Path)
    submit.add_argument(
        "--transfer-timeout",
        type=float,
        default=None,
        help=f"Finite deadline in seconds for each bundle or artifact transfer RPC (default: {_TRANSFER_TIMEOUT})",
    )
    from klab.cli.main import _tokenizer_batch_arguments

    _tokenizer_batch_arguments(submit)
    submit.set_defaults(
        tokenizer_batch_documents=None, tokenizer_batch_source_bytes=None
    )
    submit.add_argument("--stage-bundle")
    submit.add_argument("--promote")
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--max-runs", type=int, default=1000)
    _store_argument(submit, default_store)
    submit.set_defaults(handler=_experiment_submit)
    listing = experiment_commands.add_parser("list")
    listing.add_argument("--json", action="store_true")
    _store_argument(listing, default_store)
    listing.set_defaults(handler=_experiment_list)
    cancel = experiment_commands.add_parser("cancel")
    cancel.add_argument("run_id")
    _store_argument(cancel, default_store)
    cancel.set_defaults(handler=_experiment_cancel)
    resume = experiment_commands.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("--worker")
    resume.add_argument("--allow-runtime-drift", action="store_true")
    _store_argument(resume, default_store)
    resume.set_defaults(handler=_experiment_resume)
    from klab.experiments.cli import add_commands as add_plan_commands

    add_plan_commands(experiment_commands)
    collect = experiment_commands.choices["collect"]
    collect.add_argument("--relay-profile", type=Path)
    collect.add_argument("--recover", action="store_true")
    collect.add_argument("--confirm-worker-lost", action="store_true")
    _store_argument(collect, default_store)
    collect.set_defaults(handler=_collect_dispatch)

    controller = subparsers.add_parser(
        "controller", help="Run the local experiment controller"
    )
    controller_commands = controller.add_subparsers(
        dest="controller_command", required=True
    )
    run_controller = controller_commands.add_parser("run")
    run_controller.add_argument(
        "--transfer-timeout",
        type=float,
        default=_TRANSFER_TIMEOUT,
        help="Finite deadline in seconds for each bundle or artifact transfer RPC",
    )
    _store_argument(run_controller, default_store)
    run_controller.set_defaults(handler=_controller_run)
