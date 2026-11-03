"""Typed native references and restricted profiles for shared-host lab admission.

Sequencing inside a try belongs to lab_mode, not this adapter. No generic DAG,
command string, dispatch retry or alternate trainer lives here.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sparselab.campaign.state import digest, publish_immutable, read_canonical
from sparselab.training.manifest import sha256_file


class TryWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["lab_try"]
    profile: Literal["tiny-cpu-lab"]
    root: Path
    tokenizer: str
    baseline: str
    candidate: str
    declaration: str
    files: dict[str, str] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def references(self) -> TryWorkload:
        if not self.root.is_absolute() or self.root.resolve() != self.root:
            raise ValueError("native input root must be canonical and absolute")
        for name, sha in self.files.items():
            if not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*", name
            ) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise ValueError(
                    "profile requires flat named inputs with SHA-256 identities"
                )
        if (
            not {self.tokenizer, self.baseline, self.candidate, self.declaration}
            <= self.files.keys()
        ):
            raise ValueError("every native declaration must be digest-bound")
        return self


def runtime_identity() -> dict:
    """Read-only installed-distribution/interpreter identity; never probes a device."""
    packages = sorted(
        (item.metadata["Name"], item.version)
        for item in importlib.metadata.distributions()
    )
    body = {
        "python_version": platform.python_version(),
        "python_sha256": sha256_file(Path(sys.executable)),
        "packages": packages,
    }
    return {**body, "runtime_sha256": digest("sparselab-lab-runtime-v1", body)}


def verify_inputs(workload: TryWorkload, root: Path) -> None:
    from sparselab.runner_control import _safe

    total = 0
    for name, sha in workload.files.items():
        path = root / name
        total += _safe(path).st_size
        if total > 1024 * 1024:
            raise ValueError("tiny-cpu-lab input closure exceeds 1 MiB")
        if sha256_file(path) != sha:
            raise ValueError(f"native input identity changed: {name}")


def snapshot_inputs(workload: TryWorkload, destination: Path) -> None:
    """Copy the exact declared closure once; no config rewriting or acquisition."""
    import os

    from sparselab.runner_control import _sync_dir

    verify_inputs(workload, workload.root)
    destination.mkdir(mode=0o700)
    for name in workload.files:
        fd = os.open(
            destination / name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "wb") as output:
            output.write((workload.root / name).read_bytes())
            output.flush()
            os.fsync(output.fileno())
    _sync_dir(destination)
    _sync_dir(destination.parent)
    verify_inputs(workload, destination)


def declarations(workload: TryWorkload, root: Path) -> tuple:
    """Validate this small profile with native loaders before any model work."""
    from sparselab.config.loading import load_config, load_tokenizer_config
    from sparselab.derivation import _config
    from sparselab.experiments.compiler import apply_patch
    from sparselab.lab_mode import read_candidate
    from sparselab.resource_envelope import load_resource_envelope
    from sparselab.try_config import load_try_config

    verify_inputs(workload, root)
    base = load_config(root / workload.baseline)
    token = load_tokenizer_config(root / workload.tokenizer)
    delta = read_candidate(root / workload.candidate)
    if delta.kind != "delta" or not set(delta.settings) <= {
        "model.ffn_dim",
        "optimizer.peak",
        "seed",
    }:
        raise ValueError(
            "tiny-cpu-lab permits an FFN-width, learning-rate or seed delta"
        )
    candidate = _config(apply_patch(base, delta.settings), root)
    policy = load_try_config(str(root / workload.declaration))
    outputs = root.parent / "outputs"
    if (
        token.output_dir != outputs / "tokenizer"
        or base.tokenizer.path != token.output_dir / "tokenizer.json"
    ):
        raise ValueError(
            "tokenizer producer and baseline consumer paths must bind in ../outputs/tokenizer"
        )
    if (
        policy.lab_dir != root.parent / "native/lab"
        or policy.runtime_profile is not None
        or not policy.fresh_baseline
    ):
        raise ValueError(
            "tiny-cpu-lab requires fresh_baseline, ../native/lab and the admitted interpreter"
        )
    if policy.probe_tier not in {"none", "fast"}:
        raise ValueError("tiny-cpu-lab supports only none/fast probe tiers")
    if (
        policy.resource_envelope is None
        or policy.resource_envelope.parent != root
        or policy.resource_envelope.name not in workload.files
    ):
        raise ValueError("native resource envelope must be in the bound input closure")
    envelope = load_resource_envelope(policy.resource_envelope)
    if (
        envelope.lab_try is None
        or envelope.lab_try.max_wall_seconds_per_arm is None
        or envelope.lab_try.max_wall_seconds_per_arm > 120
    ):
        raise ValueError(
            "tiny-cpu-lab requires a native per-arm wall bound <=120 seconds"
        )
    for data in (base.dataset, candidate.dataset, token.dataset):
        if data.source != "local_text" or data.cache_dir != outputs / "data":
            raise ValueError("tiny-cpu-lab requires local_text and ../outputs/data")
        for path in (data.train_path, data.validation_path):
            if path is None or path.parent != root or path.name not in workload.files:
                raise ValueError(
                    "all local data must belong to the immutable input closure"
                )
        if data.train_path == data.validation_path or sha256_file(
            data.train_path
        ) == sha256_file(data.validation_path):
            raise ValueError("training and held-out inputs must be distinct")
        if any(
            getattr(data, key) is not None
            for key in (
                "allocation_manifest_path",
                "source_manifest_path",
                "corpus_release_path",
                "corpus_export_path",
                "mixture_declaration_path",
                "mixture_output_path",
            )
        ):
            raise ValueError("external data artifacts are outside tiny-cpu-lab")
        if (
            data.train_max_tokens > 32768
            or data.validation_max_tokens > 8192
            or data.train_max_documents > 2000
            or data.validation_max_documents > 100
        ):
            raise ValueError("data exceeds the existing tiny smoke profile")
    if (
        token.dataset != base.dataset
        or token.max_documents > 2000
        or token.vocab_size > 512
    ):
        raise ValueError("tokenizer must consume the same bounded input declaration")
    for config in (base, candidate):
        if (
            config.model.ffn != "dense"
            or config.model.memory != "none"
            or config.model.vocab_size != token.vocab_size
        ):
            raise ValueError(
                "tiny-cpu-lab requires dense FFN, no external memory and matching tokenizer vocabulary"
            )
        targets = (
            config.training.max_steps
            * config.training.seq_len
            * config.training.micro_batch_size
        )
        if config.training.max_tokens < targets:
            raise ValueError(
                "tiny-cpu-lab token budget must cover its declared updates"
            )
        if (
            config.runtime.engine != "pytorch"
            or config.runtime.backend != "cpu"
            or config.runtime.precision != "fp32"
        ):
            raise ValueError("tiny-cpu-lab requires PyTorch CPU fp32")
        if (
            config.model.hidden_dim > 32
            or config.model.num_layers > 2
            or config.model.num_heads > 2
            or config.model.ffn_dim > 128
            or config.model.vocab_size > 512
            or config.model.max_seq_len > 64
        ):
            raise ValueError("model exceeds the existing tiny smoke profile")
        if (
            config.training.max_steps > 12
            or config.training.max_tokens > 1536
            or config.training.seq_len > 32
            or config.training.micro_batch_size > 4
            or config.training.gradient_accumulation != 1
            or config.evaluation.batch_limit > 4
        ):
            raise ValueError(
                "training/evaluation exceeds the existing tiny smoke profile"
            )
        if not config.checkpoint.keep_periodic:
            raise ValueError("tiny-cpu-lab retains all checkpoint generations")

        # Typed paths catch future optional external packages too. Only the
        # declared tokenizer/data and task-owned output locations are allowed.
        def paths(value):
            if isinstance(value, Path):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from paths(item)
            elif isinstance(value, list):
                for item in value:
                    yield from paths(item)

        allowed = {
            outputs / "data",
            outputs / "tokenizer/tokenizer.json",
            outputs / "unused-runs",
            base.dataset.train_path,
            base.dataset.validation_path,
        }
        if any(path not in allowed for path in paths(config.model_dump())):
            raise ValueError("unbound external path in native run config")
    return base, candidate, token, policy, envelope


def run_native_try(manifest: str) -> int:
    """Delegate preparation/train/score/probe lifecycle and IDs to native try."""
    from sparselab.lab_job import _paths, _record, load_job
    from sparselab.lab_mode import run_try

    job = load_job(Path(manifest))
    binding = _record(job)
    evidence, scratch, _ = _paths(job)
    workload = job.workload
    root = evidence / "inputs"
    _, _, _, policy, envelope = declarations(workload, root)
    record, path = run_try(
        root / workload.candidate,
        root / workload.baseline,
        work_dir=scratch / "work",
        lab_dir=policy.lab_dir,
        seed=policy.seed,
        backend="cpu",
        resource_envelope=envelope,
        fresh_baseline=policy.fresh_baseline,
        probe_tier=None if policy.probe_tier == "none" else policy.probe_tier,
        request={
            **policy.model_dump(mode="json"),
            "baseline": str(root / workload.baseline),
            "job_binding": binding,
        },
    )
    # Native try wrote a sealed started record containing this binding before
    # preparation/training; its terminal record supplies the report dependency.
    publish_immutable(
        evidence / "native-result.json",
        {
            **binding,
            "try_id": record["try_id"],
            "record": str(path),
            "record_sha256": record["record_sha256"],
        },
    )
    return record["exit_status"]


def verify_native_result(job, evidence: Path, binding: dict) -> dict:
    """Read native sealed evidence and verify checkpoints, never evaluate/train."""
    from sparselab.lab_mode import current_inputs, read_record
    from sparselab.training.checkpoints import CheckpointManager

    reference = read_canonical(evidence / "native-result.json")
    if {key: reference[key] for key in binding} != binding:
        raise ValueError("native result binding mismatch")
    path = evidence / "native/lab/tries" / reference["try_id"] / "try.json"
    if str(path) != reference["record"]:
        raise ValueError("native record outside admitted lab root")
    record = read_record(path)
    if (
        record["record_sha256"] != reference["record_sha256"]
        or record["request"]["job_binding"] != binding
    ):
        raise ValueError("native record identity mismatch")
    if (
        record["status"] != "completed"
        or record["exit_status"] != 0
        or not record["comparison"]["comparable"]
    ):
        raise ValueError("native try did not complete a comparable result")
    base, candidate, _, _, _ = declarations(job.workload, evidence / "inputs")
    if (
        record["inputs"]["baseline"]["sha256"]
        != job.workload.files[job.workload.baseline]
        or record["inputs"]["candidate"]["sha256"]
        != job.workload.files[job.workload.candidate]
    ):
        raise ValueError("native declaration identity mismatch")
    arms = {}
    for name, config in (("baseline", base), ("candidate", candidate)):
        arm = record["arms"][name]
        if record[f"{name}_inputs"] != current_inputs(config):
            raise ValueError("native data/tokenizer input binding changed")
        if arm["data"]["train_sha256"] == arm["data"]["validation_sha256"]:
            raise ValueError("held-out identity is not distinct")
        run = evidence / "native/lab/runs" / arm["run_id"]
        if str(run) != arm["run_dir"] or arm["reused"] or arm["status"] != "completed":
            raise ValueError("native arm did not execute in the admitted root")
        expected_targets = min(
            config.training.max_tokens,
            config.training.max_steps
            * config.training.seq_len
            * config.training.micro_batch_size,
        )
        if (
            arm["step"] != config.training.max_steps
            or arm["tokens_seen"] != expected_targets
        ):
            raise ValueError("native optimizer did not reach declared updates/tokens")
        heldout = arm["heldout"]
        if not math.isfinite(heldout["loss"]) or heldout["valid_targets"] <= 0:
            raise ValueError("missing finite held-out model output")
        checkpoint = run / arm["checkpoint"]
        verified = CheckpointManager(run).verify(checkpoint)
        if not verified.valid:
            raise ValueError("native checkpoint failed verification")
        checkpoint_manifest = json.loads((checkpoint / "manifest.json").read_bytes())
        if (
            checkpoint_manifest["sha256"] != arm["checkpoint_sha256"]
            or checkpoint_manifest["step"] != arm["step"]
            or checkpoint_manifest["tokens_seen"] != arm["tokens_seen"]
        ):
            raise ValueError("native checkpoint counters/digest mismatch")
        observation = Path(heldout["observation"])
        if not observation.is_relative_to(run):
            raise ValueError("evaluation observation outside native run")
        measured = json.loads(observation.read_bytes())
        if measured["identity"]["checkpoint_sha256"] != arm["checkpoint_sha256"]:
            raise ValueError("evaluation was not bound to the trained checkpoint")
        arms[name] = {
            key: arm[key]
            for key in (
                "run_id",
                "step",
                "tokens_seen",
                "checkpoint_sha256",
                "heldout",
                "data",
            )
        }
    reported = json.loads((evidence / "report.stdout").read_bytes())
    if reported != record:
        raise ValueError("report did not consume the exact native result")
    return {
        "try_id": record["try_id"],
        "record_sha256": record["record_sha256"],
        "arms": arms,
        "comparison": record["comparison"],
    }
