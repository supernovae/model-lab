"""Immutable native lab handoffs over the host runner; no scheduler or retry."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from klab import runner_control as runner
from klab.campaign.state import digest, publish_immutable, read_canonical
from klab.canonical import canonical_json
from klab.lab_job_workload import (
    TryWorkload,
    declarations,
    runtime_identity,
    snapshot_inputs,
    verify_native_result,
)
from klab.operational_monitor import MonitorCompletion, MonitorPolicy
from klab.training.manifest import sha256_file
from klab.workdir import ensure_scratch_dir
from klab.workers.transport import _strict_json


class Host(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    repository: Path
    lock: Path
    state: Path
    blocked: Path
    evidence: Path
    work: Path
    python: Path

    @field_validator("*")
    @classmethod
    def absolute(cls, value: Path) -> Path:
        if not value.is_absolute() or ".." in value.parts:
            raise ValueError("host paths must be explicit absolute paths")
        return value

    @model_validator(mode="after")
    def separate(self) -> Host:
        roots = [self.state, self.evidence, self.work, self.repository]
        for index, left in enumerate(roots):
            for right in roots[index + 1 :]:
                if left.is_relative_to(right) or right.is_relative_to(left):
                    raise ValueError(
                        "source, evidence, state and work must be separate"
                    )
        for marker in (self.lock, self.blocked):
            if any(marker.is_relative_to(root) for root in roots):
                raise ValueError("lock and blocked marker must be outside roots")
        return self


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    format: Literal["klab-lab-job-v1"]
    job_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    attempt_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    validation_reference: str = Field(min_length=1)
    backend: Literal["cpu"]
    host: Host
    python_version: str
    torch_version: str
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    workload: TryWorkload
    policy: MonitorPolicy
    max_evidence_bytes: int = Field(strict=True, gt=0)

    @model_validator(mode="after")
    def bounded(self) -> Job:
        if (
            self.policy.max_wall_seconds is None
            or self.policy.max_tree_rss_bytes is None
        ):
            raise ValueError("bounded job requires explicit wall and tree RSS limits")
        if (
            self.policy.max_added_workspace_bytes is not None
            or self.policy.max_added_workspace_inodes is not None
        ):
            raise ValueError(
                "use native disk floors; workspace baseline mode is not supported"
            )
        if self.policy.expected_device_uuid is not None:
            raise ValueError("GPU admission is outside this CPU adapter")
        return self

    def payload(self) -> dict:
        return self.model_dump(mode="json")

    def identity(self) -> str:
        return digest("klab-lab-job-v1", self.payload())


def load_job(path: Path) -> Job:
    # Reuse native wire parsing: allow formatted input, reject duplicate/nonfinite keys.
    return Job.model_validate(_strict_json(path.read_bytes()))


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _source(job: Job, root: Path, *, detached: bool) -> None:
    if _git(root, "rev-parse", "HEAD") != job.source_sha:
        raise ValueError("stale source SHA")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source workdir is not clean")
    if detached and _git(root, "rev-parse", "--abbrev-ref", "HEAD") != "HEAD":
        raise ValueError("source workdir must be detached")


def _paths(job: Job) -> tuple[Path, Path, Path]:
    evidence = job.host.evidence / job.job_id
    scratch = job.host.work / "scratch" / job.attempt_id
    return evidence, scratch, scratch / "source"


def _binding(job: Job, native_id: str) -> dict[str, str]:
    return {
        "job_id": job.job_id,
        "attempt_id": job.attempt_id,
        "manifest_sha256": job.identity(),
        "runner_run_id": native_id,
    }


def _record(job: Job) -> dict:
    evidence, _, _ = _paths(job)
    runner._safe(evidence, directory=True)
    saved = read_canonical(evidence / "manifest.json")
    if saved != job.payload():
        raise ValueError("conflicting job handoff")
    prepared = read_canonical(evidence / "prepared.json")
    if not re.fullmatch(r"[0-9a-f]{32}", prepared.get("runner_run_id", "")):
        raise ValueError("invalid native runner identity")
    if prepared != _binding(job, prepared["runner_run_id"]):
        raise ValueError("conflicting job binding")
    if runner._read(job.host.evidence / f"attempt-{job.attempt_id}.json") != prepared:
        raise ValueError("conflicting authorized attempt")
    return prepared


def _refusal(job: Job, prepared: dict) -> dict | None:
    evidence, _, _ = _paths(job)
    path = evidence / "refusal.json"
    if not os.path.lexists(path):
        return None
    runner._safe(path)
    sealed = read_canonical(path)
    payload = sealed.get("payload")
    if (
        set(sealed) != {"payload", "sha256"}
        or not isinstance(payload, dict)
        or set(payload)
        != set(prepared)
        | {"status", "boundary", "reason", "workload_started", "coordinator_acceptance"}
        or any(payload.get(key) != value for key, value in prepared.items())
        or payload.get("status") != "REFUSED_BEFORE_LAUNCH"
        or payload.get("boundary") != "before-attempt-directory"
        or payload.get("workload_started") is not False
        or payload.get("coordinator_acceptance") != "PENDING"
        or not isinstance(payload.get("reason"), dict)
        or set(payload["reason"]) != {"type", "message"}
        or not all(isinstance(v, str) for v in payload["reason"].values())
        or sealed["sha256"] != digest("klab-lab-job-refusal-v1", payload)
        or os.path.lexists(evidence / "summary.json")
    ):
        raise ValueError("invalid sealed prelaunch refusal")
    return payload


def status(job: Job) -> dict:
    prepared = _record(job)
    evidence, _, _ = _paths(job)
    refusal = _refusal(job, prepared)
    if refusal is not None:
        return refusal
    summary = evidence / "summary.json"
    if summary.exists():
        return collect(job)
    complete = job.host.state / prepared["runner_run_id"] / "completion.json"
    return {
        **prepared,
        "status": "COLLECTABLE" if complete.exists() else "UNKNOWN_OR_RUNNING",
        "coordinator_acceptance": "PENDING",
    }


def admit(job: Job) -> dict:
    evidence, scratch, source = _paths(job)
    if os.path.lexists(evidence):
        return status(job)  # No launch branch on redelivery, including PREPARED.
    for root in (job.host.repository, job.host.state, job.host.evidence, job.host.work):
        runner._safe(root, directory=True)
        if root.resolve() != root:
            raise ValueError("host roots must be canonical, without symlink aliases")
    _source(job, job.host.repository, detached=False)
    if Path(__file__).resolve() != job.host.repository / "src/klab/lab_job.py":
        raise ValueError("adapter import does not match selected source")
    # No fetch and no moving target: the full selected SHA must already be merged.
    subprocess.run(
        [
            "git",
            "-C",
            str(job.host.repository),
            "merge-base",
            "--is-ancestor",
            job.source_sha,
            "refs/remotes/origin/main",
        ],
        check=True,
    )
    runner._unblocked(job.host.blocked)
    if job.host.work.stat().st_dev != job.host.evidence.stat().st_dev:
        raise ValueError(
            "native evidence and scratch must share the monitored filesystem"
        )
    declarations(job.workload, job.workload.root)
    native_id = uuid.uuid4().hex
    prepared = _binding(job, native_id)
    try:
        evidence.mkdir(mode=0o700)
    except FileExistsError:
        return status(job)  # A concurrent partial publisher refuses in _record.
    runner._sync_dir(evidence.parent)
    publish_immutable(evidence / "manifest.json", job.payload())
    # Exclusive authorized-attempt index across job IDs. Never replace or reclaim.
    runner._write(job.host.evidence / f"attempt-{job.attempt_id}.json", prepared)
    publish_immutable(evidence / "prepared.json", prepared)
    snapshot_inputs(job.workload, evidence / "inputs")
    ensure_scratch_dir(job.host.work)
    scratch.mkdir(mode=0o700)
    runner._sync_dir(scratch.parent)
    publish_immutable(evidence / "scratch.json", {"path": str(scratch), **prepared})
    subprocess.run(
        [
            "git",
            "-C",
            str(job.host.repository),
            "-c",
            "core.hooksPath=/dev/null",
            "worktree",
            "add",
            "--detach",
            "--quiet",
            str(source),
            job.source_sha,
        ],
        check=True,
    )
    policy_path = evidence / "policy.json"
    publish_immutable(policy_path, job.policy.model_dump(mode="json"))
    command = [
        str(job.host.python),
        "-I",
        "-B",
        "-c",
        (
            f"import sys; sys.path.insert(0, {str(source / 'src')!r}); "
            "from klab.lab_job import execute; "
            f"raise SystemExit(execute({str(evidence / 'manifest.json')!r}))"
        ),
    ]
    try:
        runner.run_controlled(
            command,
            job.policy,
            lock_path=job.host.lock,
            state_dir=job.host.state,
            blocked_path=job.host.blocked,
            run_id=native_id,
            job_binding=prepared,
            workspace=job.host.work,
            log_dir=evidence / "monitor",
            policy_path=policy_path,
            cwd=source,
        )
    except runner.RunnerPreflightRefused as error:
        refusal = {
            **prepared,
            "status": "REFUSED_BEFORE_LAUNCH",
            "boundary": "before-attempt-directory",
            "reason": {"type": error.error_type, "message": str(error)},
            "workload_started": False,
            "coordinator_acceptance": "PENDING",
        }
        publish_immutable(
            evidence / "refusal.json",
            {
                "payload": refusal,
                "sha256": digest("klab-lab-job-refusal-v1", refusal),
            },
        )
        return status(job)
    return collect(job)


def _runtime_probe(job: Job, source: Path) -> list[str]:
    code = (
        "import sys,platform,json; "
        f"sys.path.insert(0,{str(source / 'src')!r}); "
        "import klab,torch; from pathlib import Path; "
        "from klab.lab_job_workload import runtime_identity; "
        "actual=[platform.python_version(),str(torch.__version__)]; "
        "identity=runtime_identity(); print(json.dumps({'versions':actual,'runtime':identity,'import':klab.__file__})); "
        f"assert actual == {[job.python_version, job.torch_version]!r}, 'stale runtime'; "
        f"assert identity['runtime_sha256'] == {job.runtime_sha256!r}, 'stale runtime distribution identity'; "
        f"assert Path(klab.__file__).resolve() == Path({str(source / 'src/klab/__init__.py')!r}), 'stale import'"
    )
    return [str(job.host.python), "-I", "-B", "-c", code]


def _gate_names(job: Job) -> tuple[str, ...]:
    return ("source", "runtime", "inputs", "tokenizer", "try", "report")


def _command(
    job: Job, source: Path, evidence: Path, scratch: Path, gate: str
) -> list[str]:
    if gate == "runtime":
        return _runtime_probe(job, source)
    prefix = [str(job.host.python), "-I", "-B", "-c"]
    bootstrap = f"import sys; sys.path.insert(0,{str(source / 'src')!r}); "
    if gate == "try":
        return [
            *prefix,
            bootstrap + "from klab.lab_job_workload import run_native_try; "
            f"raise SystemExit(run_native_try({str(evidence / 'manifest.json')!r}))",
        ]
    cli = [*prefix, bootstrap + "from klab.cli.entry import main; main()"]
    if gate == "tokenizer":
        return [
            *cli,
            "--work-dir",
            str(scratch / "work"),
            "tokenizer",
            "train",
            str(evidence / "inputs" / job.workload.tokenizer),
        ]
    reference = read_canonical(evidence / "native-result.json")
    return [
        *cli,
        "--work-dir",
        str(evidence / "native"),
        "report",
        reference["record"],
        "--json",
    ]


def execute(manifest: str) -> int:
    """Fixed gate payload, invoked only as the native monitor's owned child."""
    os.umask(0o077)  # Private evidence from this owned child.
    job = load_job(Path(manifest))
    prepared = _record(job)
    evidence, scratch, source = _paths(job)
    env = dict(os.environ)
    env.update(
        dict.fromkeys(
            (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "HF_HUB_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "PYTHONDONTWRITEBYTECODE",
            ),
            "1",
        )
    )
    env.update(
        dict.fromkeys(
            ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"), ""
        )
    )
    env["KLAB_WORK_DIR"] = str(scratch / "work")
    temporary = ensure_scratch_dir(scratch / "work")
    env.update(dict.fromkeys(("TMPDIR", "TEMP", "TMP"), str(temporary)))
    failed = False
    gates = []
    for name in _gate_names(job):
        code = None
        detail = None
        if not failed:
            if name in {"source", "inputs"}:
                try:
                    if name == "source":
                        _source(job, source, detached=True)
                    else:
                        declarations(job.workload, evidence / "inputs")
                    code = 0
                except (ValueError, OSError, subprocess.CalledProcessError) as error:
                    detail, code = str(error), 1
            else:
                command = _command(job, source, evidence, scratch, name)
                with (
                    (evidence / f"{name}.stdout").open("xb") as out,
                    (evidence / f"{name}.stderr").open("xb") as err,
                ):
                    code = subprocess.run(
                        command,
                        cwd=source,
                        env=env,
                        stdout=out,
                        stderr=err,
                        check=False,
                    ).returncode
                    os.fsync(out.fileno())
                    os.fsync(err.fileno())
            failed = code != 0
        gate = {
            "name": name,
            "executed": code is not None,
            "returncode": code,
            "detail": detail,
        }
        publish_immutable(evidence / f"gate-{name}.json", {**prepared, **gate})
        gates.append(gate)
    publish_immutable(evidence / "gates.json", {**prepared, "gates": gates})
    return int(failed)


def collect(job: Job) -> dict:
    """Verify and publish evidence only. Publication retry cannot reach admission."""
    prepared = _record(job)
    evidence, scratch, source = _paths(job)
    refusal = _refusal(job, prepared)
    if refusal is not None:
        return refusal
    contract = runner._read(job.host.state / "contract.json")
    identity = runner._safe(job.host.lock)
    if (identity.st_dev, identity.st_ino) != (
        contract["lock_device"],
        contract["lock_inode"],
    ):
        raise ValueError("runner lock identity changed")
    runner._prior_runs(job.host.state, contract)
    pending = runner._read(job.host.state / prepared["runner_run_id"] / "pending.json")
    if contract["lock"] != str(job.host.lock) or contract["blocked"] != str(
        job.host.blocked
    ):
        raise ValueError("host contract changed")
    if pending.get("job_binding") != prepared or pending["log_dir"] != str(
        evidence / "monitor"
    ):
        raise ValueError("native completion binding mismatch")
    completion = MonitorCompletion.model_validate_json(
        (evidence / "monitor/completion.json").read_bytes()
    )
    gates = read_canonical(evidence / "gates.json")
    if {k: gates[k] for k in prepared} != prepared:
        raise ValueError("gate binding mismatch")
    for name, gate in zip(_gate_names(job), gates["gates"], strict=True):
        if gate["name"] != name or read_canonical(evidence / f"gate-{name}.json") != {
            **prepared,
            **gate,
        }:
            raise ValueError("missing or conflicting gate evidence")
    for gate in gates["gates"]:
        if gate["executed"] and gate["name"] not in {"source", "inputs"}:
            for suffix in ("stdout", "stderr"):
                runner._safe(evidence / f"{gate['name']}.{suffix}")
    for name in ("command.stdout.log", "command.stderr.log", "events.jsonl"):
        runner._safe(evidence / "monitor" / name)
    passed = completion.status == "COMPLETE" and all(
        g["executed"] and g["returncode"] == 0 for g in gates["gates"]
    )
    native = verify_native_result(job, evidence, prepared) if passed else None
    files = {}
    for path in sorted(evidence.rglob("*")):
        if path == evidence / "summary.json":
            continue
        if path.is_dir():
            runner._safe(path, directory=True)
            continue
        runner._safe(path)
        files[str(path.relative_to(evidence))] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    size = sum(item["bytes"] for item in files.values())
    try:
        _source(job, source, detached=True)
        source_unchanged = True
    except ValueError, OSError, subprocess.CalledProcessError:
        source_unchanged = False
    checks = {
        "source_unchanged": source_unchanged,
        "manifest": read_canonical(evidence / "manifest.json") == job.payload(),
        "size": size <= job.max_evidence_bytes,
        "time": completion.elapsed_seconds <= job.policy.max_wall_seconds,
    }
    summary = {
        **prepared,
        "controller_completion": "PASS" if passed and all(checks.values()) else "FAIL",
        "coordinator_acceptance": "PENDING",
        "cleanup_verified": True,
        "native_status": completion.status,
        "native_returncode": completion.returncode,
        "gates": gates["gates"],
        "native_workflow": native,
        "elapsed_seconds": completion.elapsed_seconds,
        "peak_tree_rss_bytes": completion.peak_tree_rss_bytes,
        "peak_tree_swap_bytes": completion.peak_tree_swap_bytes,
        "evidence_bytes": size,
        "checks": checks,
        "files": files,
        "scratch": str(scratch),
    }
    # Include the summary itself in the final byte check. Converge its byte-count
    # field and PASS/FAIL spelling before immutable publication.
    while True:
        total = size + len(canonical_json(summary)) + 1
        within_cap = total <= job.max_evidence_bytes
        completion_status = (
            "PASS"
            if passed and all({**checks, "size": within_cap}.values())
            else "FAIL"
        )
        if (
            summary["evidence_bytes"] == total
            and checks["size"] == within_cap
            and summary["controller_completion"] == completion_status
        ):
            break
        summary["evidence_bytes"] = total
        checks["size"] = within_cap
        summary["controller_completion"] = completion_status
    publish_immutable(evidence / "summary.json", summary)
    return {**prepared, "status": "FINALIZED", "summary": summary}


def cleanup_preview(job: Job, candidate: Path) -> dict:
    """Return a plan only. No deletion, unlinking, worktree removal or pruning."""
    prepared = _record(job)
    evidence, scratch, _ = _paths(job)
    if candidate != scratch or candidate.resolve() != scratch:
        raise ValueError("unsafe cleanup plan: only exact owned scratch is eligible")
    if read_canonical(evidence / "scratch.json") != {"path": str(scratch), **prepared}:
        raise ValueError("unsafe cleanup plan: scratch ownership mismatch")
    result = collect(job)
    for pending in job.host.state.glob("*/pending.json"):
        if Path(runner._read(pending)["log_dir"]).is_relative_to(scratch):
            raise ValueError("unsafe cleanup plan: receipt-referenced logs")
    runner._safe(scratch, directory=True)
    for path in scratch.rglob("*"):
        if path.is_symlink() or (path.is_file() and path.stat().st_nlink != 1):
            raise ValueError("unsafe cleanup plan: linked scratch content")
    return {
        "preview_only": True,
        "candidate": str(scratch),
        "delete_implemented": False,
        "retain": [
            str(job.host.evidence),
            str(job.host.state),
            str(job.host.lock),
            str(job.host.blocked),
        ],
        "job_status": result["status"],
    }


def register_job_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "lab-job", help="Admit or collect an immutable native lab handoff"
    )
    _arguments(parser)
    parser.set_defaults(handler=_handle)


def _arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "action",
        choices=(
            "admit",
            "status",
            "collect",
            "cleanup-preview",
            "runtime-identity",
            "seal",
        ),
    )
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path)


def _handle(args: argparse.Namespace) -> None:
    if args.action == "runtime-identity":
        if (
            args.manifest is not None
            or args.candidate is not None
            or args.output is not None
        ):
            raise ValueError("runtime-identity takes no manifest or candidate")
        print(json.dumps(runtime_identity(), sort_keys=True))
        return
    if args.manifest is None:
        raise ValueError("a job manifest is required")
    if args.action == "seal":
        if args.output is None or args.candidate is not None:
            raise ValueError("seal requires an unused --output and no --candidate")
        raw = _strict_json(args.manifest.read_bytes())
        if Path(raw["host"]["python"]) != Path(sys.executable):
            raise ValueError("seal must run in the exact selected interpreter")
        import torch

        identity = runtime_identity()
        raw.update(
            runtime_sha256=identity["runtime_sha256"],
            python_version=identity["python_version"],
            torch_version=str(torch.__version__),
        )
        workload = raw["workload"]
        files = workload["files"]
        if (
            not isinstance(files, list)
            or any(
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                for name in files
            )
            or len(files) != len(set(files))
        ):
            raise ValueError("seal requires a list of unique flat input filenames")
        workload["files"] = {
            name: sha256_file(Path(workload["root"]) / name) for name in files
        }
        job = Job.model_validate(raw)
        _source(job, job.host.repository, detached=False)
        declarations(job.workload, job.workload.root)
        if os.path.lexists(args.output):
            raise FileExistsError(args.output)
        publish_immutable(args.output, job.payload())
        print(
            json.dumps(
                {
                    "job_id": job.job_id,
                    "attempt_id": job.attempt_id,
                    "manifest_sha256": job.identity(),
                    "manifest": str(args.output),
                    "status": "SEALED_NOT_ADMITTED",
                },
                sort_keys=True,
            )
        )
        return
    if args.output is not None:
        raise ValueError("--output is only valid for seal")
    job = load_job(args.manifest)
    if args.action == "cleanup-preview":
        if args.candidate is None:
            raise ValueError("cleanup-preview requires --candidate")
        result = cleanup_preview(job, args.candidate)
    else:
        if args.candidate is not None:
            raise ValueError("--candidate is only valid for cleanup-preview")
        result = {"admit": admit, "status": status, "collect": collect}[args.action](
            job
        )
    print(json.dumps(result, sort_keys=True))
    if result.get("summary", {}).get("controller_completion") == "FAIL":
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _arguments(parser)
    _handle(parser.parse_args(argv))


if __name__ == "__main__":
    main()
