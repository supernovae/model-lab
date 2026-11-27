"""Opt-in Corpus command operators with pinned I/O and verified publication."""

from __future__ import annotations

import json
import os
import platform
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from klab.canonical import canonical_digest
from klab.config.models import StrictModel
from klab.corpus.build_stages import RECEIPT, StageStore, implementation_digests
from klab.corpus.command_contracts import (
    DOCUMENT_CONTRACTS,
    Kind,
    aligned_rows,
    scoped_rows,
    validated_rows,
)
from klab.corpus.relation_sources import Digest
from klab.corpus.relations import publish_relation_stage
from klab.corpus.release_members import shared_payload_lease
from klab.corpus.shards import ShardPlan, Single, declare_stage, run_sharded
from klab.corpus.stage_inspection import open_verified_stage
from klab.data.verification import HashingWriter
from klab.runtime_env_subprocess import run_bounded
from klab.safe_io import symlinked, write_canonical_json
from klab.training.manifest import sha256_file
from klab.workspace_preflight import check_storage, require_storage

_STAGE = "command-output"
_OPERATION = declare_stage(
    "authorized command operator",
    Single("one authorized process owns its staged output"),
)
_REQUEST_ENV = "KLAB_OPERATOR_REQUEST"
_ENV = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC", "PATH": "/usr/bin:/bin"}
_NAME = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]
_ENV_NAME = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")]
_BYTES = Annotated[int, Field(strict=True, ge=0, le=2**40)]
_LIMIT = Annotated[int, Field(strict=True, ge=1, le=2**40)]
_ROWS = Annotated[int, Field(strict=True, ge=1, le=2**40)]
_LINE_BYTES = Annotated[int, Field(strict=True, ge=1, le=64 * 1024**2)]
_ARG = Annotated[str, StringConstraints(max_length=8192, pattern=r"^[^\x00]*$")]
_LOG_LIMIT = 64 * 1024
_DECLARATION_LIMIT = 2 * 1024**2

# A fresh, isolated interpreter applies the per-file hard limit before exec.
# It has the caller's other authority. The command can still create many files.
_LAUNCH = """\
import os, resource, sys
limit = int(sys.argv[1])
_, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
if hard != resource.RLIM_INFINITY:
    limit = min(limit, hard)
resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
os.execv(sys.argv[2], sys.argv[2:])
"""


class FilePin(StrictModel):
    path: Path
    sha256: Digest
    size: _BYTES


class InputPin(FilePin):
    contract: Kind
    max_rows: _ROWS
    max_record_bytes: _LINE_BYTES = 1024**2


class Output(StrictModel):
    contract: Kind
    max_bytes: _LIMIT
    max_rows: _ROWS
    max_record_bytes: _LINE_BYTES = 1024**2
    aligned_to: _NAME | None = None
    scope: _NAME | None = None


class CommandSpec(StrictModel):
    operator: _NAME
    determinism: Literal["deterministic", "nondeterministic"]
    executable: FilePin
    arguments: Annotated[tuple[_ARG, ...], Field(max_length=128)] = ()
    environment: Annotated[dict[_ENV_NAME, _ARG], Field(max_length=64)] = Field(
        default_factory=dict
    )
    dependencies: Annotated[dict[_NAME, FilePin], Field(max_length=128)] = Field(
        default_factory=dict
    )
    inputs: Annotated[dict[_NAME, InputPin], Field(min_length=1, max_length=64)]
    outputs: Annotated[dict[_NAME, Output], Field(min_length=1, max_length=64)]
    timeout_seconds: Annotated[int, Field(strict=True, ge=1, le=86400)]
    max_validation_bytes: Annotated[int, Field(strict=True, ge=1024**2, le=2**40)] = (
        1024**2
    )

    @model_validator(mode="after")
    def contract_bindings(self):
        if _REQUEST_ENV in self.environment or "receipt.json" in self.outputs:
            raise ValueError("command declaration uses a reserved name")
        for name, output in self.outputs.items():
            if output.contract == "relation_scope":
                raise ValueError("relation_scope is an input-only command contract")
            if output.contract == "text_statistics" and output.aligned_to is None:
                raise ValueError(f"{name} requires a document-aligned input")
            if output.aligned_to is not None:
                parent = self.inputs.get(output.aligned_to)
                if parent is None or not (
                    (
                        parent.contract in DOCUMENT_CONTRACTS
                        and output.contract in DOCUMENT_CONTRACTS
                    )
                    or (parent.contract == output.contract == "local_text")
                ):
                    raise ValueError(
                        f"{name} alignment requires compatible declared contracts"
                    )
            if output.contract == "relation_records":
                scope = self.inputs.get(output.scope)
                if scope is None or scope.contract != "relation_scope":
                    raise ValueError(f"{name} requires a declared relation_scope input")
            elif output.scope is not None:
                raise ValueError(f"{name} scope applies only to relation_records")
        return self


@dataclass(frozen=True)
class CommandRequest:
    spec: CommandSpec
    locations: dict[str, Path]
    inputs: dict
    key: str
    authorization: str
    replay: str | None

    def pins(self):
        yield "executable", self.spec.executable
        for section in ("inputs", "dependencies"):
            for name, pin in sorted(getattr(self.spec, section).items()):
                yield f"{section}/{name}", pin


def _store(root):
    return StageStore(root / "cache" / "corpus" / "command-operators")


def _request(declaration, replay):
    if not declaration.is_file():
        raise ValueError("command declaration must be a local regular file")
    with declaration.open("rb") as stream:
        raw = stream.read(_DECLARATION_LIMIT + 1)
    if len(raw) > _DECLARATION_LIMIT:
        raise ValueError("command declaration exceeds the byte limit")
    spec = CommandSpec.model_validate_json(raw)
    base = declaration.resolve().parent
    locations = {"executable": (base / spec.executable.path).resolve()}
    options = spec.model_dump(
        mode="json", exclude={"executable", "inputs", "dependencies"}
    )
    options["executable"] = spec.executable.model_dump(mode="json", exclude={"path"})
    for section in ("inputs", "dependencies"):
        options[section] = {}
        for name, pin in sorted(getattr(spec, section).items()):
            locations[f"{section}/{name}"] = (base / pin.path).resolve()
            options[section][name] = pin.model_dump(mode="json", exclude={"path"})
    options["environment"] = {**_ENV, **spec.environment}
    inputs = {
        "implementation": implementation_digests(
            "command_operators.py",
            "command_contracts.py",
            "relation_sources.py",
            "mechanisms.py",
        ),
        "host": {
            "platform": sys.platform,
            "architecture": platform.machine(),
            "launcher_python": platform.python_version(),
        },
        "options": options,
    }
    key = StageStore.key(inputs)
    auth = canonical_digest(
        {
            "computation": key,
            "locations": {name: str(path) for name, path in locations.items()},
            "declaration": str(declaration.resolve()),
            "replay": replay,
        }
    )
    return CommandRequest(spec, locations, inputs, key, auth, replay)


def _entry(path, *, single_link=False):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or (single_link and info.st_nlink != 1):
        raise ValueError(f"command file is not a regular private file: {path.name}")
    return info


def _generation(info):
    return [
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    ]


def _copy(source, destination, *, size, expected=None, single_link=False):
    before = _entry(source, single_link=single_link)
    if before.st_size != size:
        raise ValueError(f"command file size drift: {source.name}")
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as incoming, destination.open("xb") as outgoing:
        if _generation(os.fstat(fd)) != _generation(before):
            raise ValueError(f"command file changed before staging: {source.name}")
        writer = HashingWriter(outgoing) if expected is not None else outgoing
        left = size
        while left:
            chunk = incoming.read(min(left, 1024**2))
            if not chunk:
                raise ValueError(f"command file shortened while staging: {source.name}")
            writer.write(chunk)
            left -= len(chunk)
        if incoming.read(1) or _generation(
            _entry(source, single_link=single_link)
        ) != _generation(before):
            raise ValueError(f"command file changed while staging: {source.name}")
        if expected is not None and writer.digest.hexdigest() != expected:
            raise ValueError(f"command file digest drift: {source.name}")
    return _generation(before)


def _rows(path, rule):
    return validated_rows(
        path,
        rule.contract,
        max_rows=rule.max_rows,
        max_record_bytes=rule.max_record_bytes,
    )


def _validate_output(path, output, request, work, scratch, input_counts):
    with closing(_rows(path, output)) as base:
        if output.scope is not None:
            scope = output.scope
            with (
                closing(
                    _rows(work / "inputs" / scope, request.spec.inputs[scope])
                ) as entities,
                closing(
                    scoped_rows(
                        base, entities, scratch, request.spec.max_validation_bytes
                    )
                ) as rows,
            ):
                return sum(1 for _ in rows)
        parent = output.aligned_to
        if parent is not None and output.contract in DOCUMENT_CONTRACTS:
            with closing(
                _rows(work / "inputs" / parent, request.spec.inputs[parent])
            ) as source:
                return sum(1 for _ in aligned_rows(base, source))
        count = sum(1 for _ in base)
        if parent is not None and count != input_counts[parent]:
            raise ValueError("command output does not align with its declared input")
        return count


def _limits(request):
    spec = request.spec
    output_limits = [output.max_bytes for output in spec.outputs.values()]
    return {
        "staging_bytes": sum(pin.size for _, pin in request.pins())
        + 2 * sum(output_limits)
        + spec.max_validation_bytes,
        "max_regular_file_bytes": max(output_limits),
        "diagnostic_stream_bytes": _LOG_LIMIT,
        "command_timeout_seconds": spec.timeout_seconds,
    }


def _run_command(staging, request):
    spec = request.spec
    limits = _limits(request)
    require_storage([check_storage(staging, projected_bytes=limits["staging_bytes"])])
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=".command-", dir=staging.parent
    ) as work_path:
        work = Path(work_path)
        for section in ("inputs", "outputs", "dependencies"):
            (work / section).mkdir(mode=0o700)
        originals, staged = {}, {}
        for name, pin in request.pins():
            path = work / name
            source = request.locations[name]
            originals[name] = _copy(source, path, size=pin.size, expected=pin.sha256)
            path.chmod(0o500 if name == "executable" else 0o400)
            staged[name] = _generation(_entry(path, single_link=True))
        counts = {
            name: sum(1 for _ in _rows(work / "inputs" / name, pin))
            for name, pin in spec.inputs.items()
        }
        request_path = work / "request.json"
        write_canonical_json(
            request_path,
            {
                "operator": spec.operator,
                "computation_key": request.key,
                "inputs": {name: f"inputs/{name}" for name in spec.inputs},
                "dependencies": {
                    name: f"dependencies/{name}" for name in spec.dependencies
                },
                "outputs": {name: f"outputs/{name}" for name in spec.outputs},
            },
        )
        argv = [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _LAUNCH,
            str(limits["max_regular_file_bytes"]),
            str(work / "executable"),
            *spec.arguments,
        ]
        try:
            process = run_bounded(
                argv,
                timeout=spec.timeout_seconds,
                output_limit=_LOG_LIMIT,
                env={
                    **request.inputs["options"]["environment"],
                    _REQUEST_ENV: str(request_path),
                },
                cwd=work,
            )
        except subprocess.TimeoutExpired:
            raise ValueError(
                "authorized command exceeded its wall-time limit"
            ) from None
        if process.returncode:
            raise ValueError(
                f"authorized command exited with status {process.returncode}"
            )
        for name, generation in staged.items():
            if _generation(_entry(work / name, single_link=True)) != generation:
                raise ValueError(
                    f"command changed its staged input or dependency: {name}"
                )
        out_dir = work / "outputs"
        if (
            symlinked(out_dir)
            or not out_dir.is_dir()
            or {p.name for p in out_dir.iterdir()} != set(spec.outputs)
        ):
            raise ValueError("command must write exactly its declared output files")
        results = {}
        for name, output in sorted(spec.outputs.items()):
            member = out_dir / name
            size = _entry(member, single_link=True).st_size
            if size > output.max_bytes:
                raise ValueError(
                    f"command output violates its file size contract: {name}"
                )
            _copy(member, staging / name, size=size, single_link=True)
            with tempfile.TemporaryDirectory(
                prefix=".validate-", dir=staging
            ) as scratch:
                count = _validate_output(
                    staging / name,
                    output,
                    request,
                    work,
                    Path(scratch) / "scope.sqlite",
                    counts,
                )
            results[name] = {
                "contract": output.contract,
                "records": count,
                "size": size,
            }
        # A command has the caller's authority; catch accidental changes before publishing.
        for name, generation in staged.items():
            if _generation(_entry(work / name, single_link=True)) != generation:
                raise ValueError(
                    f"command staged input changed during validation: {name}"
                )
        for name, generation in originals.items():
            if _generation(_entry(request.locations[name])) != generation:
                raise ValueError(f"command source changed during execution: {name}")
    if {member.name for member in staging.iterdir()} != set(spec.outputs):
        raise ValueError("command stage contains files outside its validated outputs")
    return {
        "computation_key": request.key,
        "input_records": counts,
        "limits": limits,
        "outputs": results,
        "locations": {name: str(path) for name, path in request.locations.items()},
        "generations": originals,
        "wall_seconds": time.monotonic() - started,
        "diagnostic_bytes": {
            "stdout": len(process.stdout),
            "stderr": len(process.stderr),
        },
        "replay_of": request.replay,
    }


def _produce(staging, request):
    plan = ShardPlan(
        _OPERATION, [(staging, request)], sum(pin.size for _, pin in request.pins())
    )
    with closing(run_sharded(plan, _run_command)) as run:
        return next(run)


def _verified_pins(request, previous):
    for name, pin in request.pins():
        path = request.locations[name]
        before = _entry(path)
        generation = _generation(before)
        if (
            generation == previous["generations"][name]
            and str(path) == previous["locations"][name]
        ):
            continue
        if (
            before.st_size != pin.size
            or sha256_file(path) != pin.sha256
            or _generation(_entry(path)) != generation
        ):
            raise ValueError(f"command file digest or size drift: {path.name}")


def _view(path, receipt, *, reused):
    return {
        "key": receipt["key"],
        "path": str(path),
        "inputs": receipt["inputs"],
        "files": receipt["files"],
        "reused": reused,
        **receipt["result"],
    }


def inspect_command_operator(root, key):
    with open_verified_stage(_store(root), _STAGE, key) as (path, receipt, _):
        return _view(path, receipt, reused=True)


def plan_command_operator(root, declaration, *, replay=None):
    request = _request(declaration, replay)
    if (
        replay is not None
        and inspect_command_operator(root, replay)["computation_key"] != request.key
    ):
        raise ValueError("command replay does not match the published computation")
    return {
        "computation_key": request.key,
        "authorization": request.authorization,
        "inputs": request.inputs,
        "locations": {name: str(path) for name, path in request.locations.items()},
        "replay_of": replay,
        "limits": _limits(request),
        "authority": "caller_permissions",
        "candidate": request.spec.determinism == "deterministic"
        and _store(root).directory(_STAGE, request.key).is_dir(),
    }


def run_command_operator(root, declaration, *, authorization, replay=None):
    request = _request(declaration, replay)
    if authorization != request.authorization:
        raise ValueError(
            "command execution requires --allow-command-operators with the exact plan authorization"
        )
    store = _store(root)
    if (
        replay is not None
        and inspect_command_operator(root, replay)["computation_key"] != request.key
    ):
        raise ValueError("command replay does not match the published computation")
    if request.spec.determinism == "nondeterministic":
        inputs = {**request.inputs, "attempt": secrets.token_hex(16)}
        return publish_relation_stage(
            store,
            _STAGE,
            inputs,
            lambda staged, result: result.update(_produce(staged, request)),
        )

    destination = store.directory(_STAGE, request.key)
    with shared_payload_lease(destination):
        if destination.is_dir():
            with open_verified_stage(store, _STAGE, request.key) as (path, receipt, _):
                if replay is None:
                    _verified_pins(request, receipt["result"])
                    return _view(path, receipt, reused=True)
                with tempfile.TemporaryDirectory(
                    prefix=".replay-", dir=destination.parent
                ) as temporary:
                    temporary = Path(temporary)
                    replay_result = _produce(temporary, request)
                    observed = {
                        p.name: {"size": p.stat().st_size, "sha256": sha256_file(p)}
                        for p in temporary.iterdir()
                    }
                    if observed != receipt["files"]:
                        raise ValueError(
                            "deterministic command replay changed published output bytes"
                        )
                return {**_view(path, receipt, reused=True), "replay": replay_result}
        if replay is not None:
            raise ValueError("command replay requires its retained original artifact")
        with store.build(_STAGE, request.inputs) as (staging, result):
            result.update(_produce(staging, request))
        published = json.loads((destination / RECEIPT).read_bytes())
        return _view(destination, published, reused=False)
