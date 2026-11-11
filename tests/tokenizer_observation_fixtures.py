"""Tiny closed-job metadata for observation checks; no tokenizer fit is run."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import yaml

from klab.canonical import canonical_digest, canonical_json
from klab.config.loading import load_tokenizer_config
from klab.corpus.export import _export_binding, _export_request
from klab.training.manifest import sha256_file


def observed_tokenizer(root):
    release_id = "d" * 64
    files = {}

    def write(name, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            value if isinstance(value, bytes) else canonical_json(value) + b"\n"
        )
        files[name] = {"path": str(path), "sha256": sha256_file(path)}
        return path

    release = root / "corpora/tiny/builds/build"
    manifest = {
        "release_id": release_id,
        "corpus_id": "tiny",
        "build_id": "build",
        "build_identity": {
            "completed_preparation": {"policy": "completed_local_preparation"}
        },
        "files": {
            name: {"sha256": "a" * 64}
            for name in (
                "report.json",
                "license-report.json",
                "lm/train.jsonl",
                "lm/validation.jsonl",
            )
        },
    }
    write("release", release / "manifest.json", manifest)
    request = _export_request(release_id, "lm", "b" * 64, 512, 1)
    export_dir = (
        root / "corpora/tiny/exports" / release_id / "lm" / canonical_digest(request)
    )
    dataset = {
        "license": "CC0-1.0",
        "source": "local_text",
        "revision": release_id,
        "cache_dir": str(root / "cache"),
        "corpus_release_path": str(release),
        "corpus_export_path": str(export_dir),
        "train_path": str(release / "lm/train.jsonl"),
        "validation_path": str(release / "lm/validation.jsonl"),
        "train_max_documents": 1,
        "validation_max_documents": 1,
        "train_max_tokens": 10,
        "validation_max_tokens": 10,
    }
    sampling = {
        "source_strata": {"tiny": "text"},
        "byte_weights": {"text": 1},
        "max_fit_bytes": 8,
    }
    tokenizer_config = {
        "schema_version": 1,
        "vocab_size": 512,
        "min_frequency": 1,
        "max_documents": 1,
        "output_dir": str(root / "fit"),
        "dataset": dataset,
        "sampling": sampling,
    }
    config_path = write(
        "tokenizer_config",
        export_dir / "tokenizer.yaml",
        yaml.safe_dump(tokenizer_config).encode(),
    )
    config = load_tokenizer_config(config_path)
    run = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/smoke_cpu.yaml").read_text()
    )
    run["dataset"] = dataset
    write("run", export_dir / "run.yaml", yaml.safe_dump(run).encode())
    export = {
        **request,
        "schema_version": 1,
        "run_config_sha256": files["run"]["sha256"],
        "tokenizer_config_sha256": files["tokenizer_config"]["sha256"],
        "release_manifest_sha256": files["release"]["sha256"],
        "report_sha256": "a" * 64,
        "license_report_sha256": "a" * 64,
        "splits": {
            split: {
                "path": f"lm/{split}.jsonl",
                "sha256": "a" * 64,
                "records": 1,
                "rendered_bytes": 8,
            }
            for split in ("train", "validation")
        },
    }
    write("export", export_dir / "export.json", export)
    token = write("tokenizer", root / "fit/tokenizer.json", b"observed token bytes")
    contract = {
        "corpus_export": _export_binding(export),
        "vocab_size": 512,
        "min_frequency": 1,
        "max_documents": 1,
        "source": "local_text",
        "revision": release_id,
        "sampling": config.sampling.model_dump(mode="json"),
    }
    token_manifest = {
        "corpus_export": _export_binding(export),
        "training_contract": contract,
        "sha256": files["tokenizer"]["sha256"],
        "revision": release_id,
        "source": "local_text",
        "vocab_size": 512,
        "sampling_audit": {"complete": True, "disjoint_from_fit": True},
    }
    write(
        "tokenizer_manifest", token.with_name("tokenizer_manifest.json"), token_manifest
    )
    receipts = root / "receipts"
    session_root = receipts / "session"
    job_root = session_root / "slot"
    argv = ["klab", "tokenizer", "train", str(config_path)]
    source = {"path": str(root / "source"), "commit": "c" * 40}
    job_spec = {
        "argv": argv,
        "output_root": str(job_root),
        "job_id": "job",
        "source": source,
        "required_files": {str(config_path): files["tokenizer_config"]["sha256"]},
    }
    write("job_spec", root / "job.json", job_spec)
    session_spec = {
        "session_id": "session",
        "receipt_root": str(receipts),
        "source_paths": [source["path"]],
        "slots": [{"id": "slot", "root": "slot", "argv": argv}],
    }
    write("session_spec", root / "session.json", session_spec)
    write("stdout", job_root / "job/stdout.log", str(token).encode())
    write("stderr", job_root / "job/stderr.log", b"")
    job = {
        "job_id": "job",
        "source": source,
        "spec_sha256": files["job_spec"]["sha256"],
        "returncode": 0,
        "reason": "exit",
        "launched": True,
        "process_success": True,
        "cleanup_verified": True,
        "evidence_complete": True,
        "logs_complete": True,
        "errors": [],
        "artifacts": {"files": []},
    }
    write("job_report", job_root / "job/report.json", job)
    progress = {
        "job_id": "job",
        "slot": "slot",
        "root": "slot",
        "launched": True,
        "returncode": 0,
        "spec_sha256": files["job_spec"]["sha256"],
        "report_sha256": files["job_report"]["sha256"],
        "stdout_sha256": files["stdout"]["sha256"],
        "stderr_sha256": files["stderr"]["sha256"],
    }
    session = {
        "session_id": "session",
        "reason": "complete",
        "slots_completed": 1,
        "slots_total": 1,
        "pending_failure": None,
        "incomplete_jobs": [],
        "inventory_error": None,
        "cleanup_verified": True,
        "evidence_complete": True,
        "progress": {"jobs": [progress]},
    }
    write("session_report", session_root / "shared/report.json", session)
    outer = {
        "session_spec_sha256": files["session_spec"]["sha256"],
        "reason": "exit",
        "engine_exit": 0,
        "container_exit": 0,
        "oom_killed": False,
        "cleanup_verified": True,
        "logs_complete": True,
        "errors": [],
        "ended_at_unix": 1,
    }
    write("container_report", session_root / "shared/container-report.json", outer)
    # The envelope schema is not under test; real files/config loaders and all
    # observation receipt/provenance/split checks execute without mocks.
    files.pop("release")
    files.pop("run")
    observed_at = datetime(2026, 1, 1, tzinfo=UTC)
    observation = SimpleNamespace(
        observed_at_utc=observed_at,
        model_dump=lambda **kwargs: {
            "files": files,
            "observed_at_utc": observed_at.isoformat(),
        },
    )
    return (
        observation,
        config_path,
        config,
        release,
        manifest,
        export,
        token_manifest,
        write,
    )
