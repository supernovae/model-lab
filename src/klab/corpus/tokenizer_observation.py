"""Explicitly adopt a post-run tokenizer observation without replaying its fit.

This binds current artifact bytes to a completed job's receipts and records the
limit of that evidence: the original job did not attest the output digests.
It is opt-in and does not mint a cold-verification proof.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from klab.canonical import canonical_digest, canonical_json
from klab.config.loading import load_config, load_tokenizer_config
from klab.corpus.export import _export_binding, _export_request


def _read(path: Path, expected: str, *, limit: int = 4 * 1024 * 1024) -> bytes:
    if (
        not path.is_absolute()
        or any(part.is_symlink() for part in (path, *path.parents))
        or not path.is_file()
        or path.stat().st_size > limit
    ):
        raise ValueError("unsafe or oversized tokenizer observation input")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("tokenizer observation input changed")
    return raw


def verify_observed_tokenizer(
    observation, config_path, config, release, release_manifest
):
    """Validate bounded metadata and current bytes, not a historical digest proof."""
    values = observation.model_dump(mode="json")
    files = values["files"]

    def read(name):
        entry = files[name]
        return _read(Path(entry["path"]), entry["sha256"])

    job_spec = json.loads(read("job_spec"))
    session_spec = json.loads(read("session_spec"))
    job = json.loads(read("job_report"))
    session = json.loads(read("session_report"))
    outer = json.loads(read("container_report"))
    stdout = read("stdout").decode("utf-8").strip()
    read("stderr")
    config_raw = read("tokenizer_config")
    token_raw = read("tokenizer")
    manifest_raw = read("tokenizer_manifest")
    token_manifest = json.loads(manifest_raw)

    token_path = config.output_dir / "tokenizer.json"
    names = {name: Path(entry["path"]).resolve() for name, entry in files.items()}
    if (
        names["tokenizer_config"] != Path(config_path).resolve()
        or names["tokenizer"] != token_path.resolve()
        or names["tokenizer_manifest"]
        != token_path.with_name("tokenizer_manifest.json").resolve()
        or stdout != str(token_path)
    ):
        raise ValueError("tokenizer observation paths do not match the completed job")
    slots = session_spec.get("slots", [])
    progress = session.get("progress", {}).get("jobs", [])
    if len(slots) != 1 or len(progress) != 1:
        raise ValueError("tokenizer observation requires one completed job")
    entry = progress[0]
    argv = job_spec.get("argv", [])
    job_root = Path(job_spec.get("output_root", ""))
    receipt_root = job_root.parent
    expected_job = job_root / str(job_spec.get("job_id", ""))
    if (
        len(argv) < 3
        or argv[-3:]
        != [
            "tokenizer",
            "train",
            str(config_path),
        ]
        or slots[0].get("argv") != argv
        or job_spec.get("required_files", {}).get(str(config_path))
        != hashlib.sha256(config_raw).hexdigest()
        or job_spec.get("job_id") != job.get("job_id")
        or job_spec.get("source", {}).get("path")
        not in session_spec.get("source_paths", [])
        or not job_root.is_absolute()
        or job_root.resolve()
        != (
            Path(session_spec.get("receipt_root", ""))
            / str(session_spec.get("session_id", ""))
            / str(slots[0].get("root", ""))
        ).resolve()
        or names["job_report"] != (expected_job / "report.json").resolve()
        or names["stdout"] != (expected_job / "stdout.log").resolve()
        or names["stderr"] != (expected_job / "stderr.log").resolve()
        or names["session_report"] != (receipt_root / "shared/report.json").resolve()
        or names["container_report"]
        != (receipt_root / "shared/container-report.json").resolve()
        or receipt_root.name != session_spec.get("session_id")
        or job.get("source") != job_spec.get("source")
        or job.get("spec_sha256") != files["job_spec"]["sha256"]
        or entry.get("job_id") != job.get("job_id")
        or entry.get("slot") != slots[0].get("id")
        or entry.get("root") != slots[0].get("root")
        or entry.get("launched") is not True
        or entry.get("spec_sha256") != files["job_spec"]["sha256"]
        or entry.get("report_sha256") != files["job_report"]["sha256"]
        or entry.get("stdout_sha256") != files["stdout"]["sha256"]
        or entry.get("stderr_sha256") != files["stderr"]["sha256"]
        or entry.get("returncode") != 0
        or session.get("session_id") != session_spec.get("session_id")
        or session.get("reason") != "complete"
        or session.get("slots_completed") != 1
        or session.get("slots_total") != 1
        or session.get("pending_failure") is not None
        or session.get("incomplete_jobs") != []
        or session.get("inventory_error") is not None
        or not session.get("cleanup_verified")
        or not session.get("evidence_complete")
        or outer.get("session_spec_sha256") != files["session_spec"]["sha256"]
        or outer.get("reason") != "exit"
        or outer.get("engine_exit") != 0
        or outer.get("container_exit") != 0
        or outer.get("oom_killed") is not False
        or not outer.get("cleanup_verified")
        or not outer.get("logs_complete")
        or outer.get("errors") != []
        or not isinstance(outer.get("ended_at_unix"), (int, float))
        or observation.observed_at_utc.timestamp() < outer["ended_at_unix"]
        or job.get("returncode") != 0
        or job.get("reason") != "exit"
        or job.get("launched") is not True
        or not all(
            job.get(key)
            for key in (
                "process_success",
                "cleanup_verified",
                "evidence_complete",
                "logs_complete",
            )
        )
        or job.get("errors") != []
        or job.get("artifacts", {}).get("files") != []
    ):
        raise ValueError("tokenizer observation lacks successful closed execution")

    export_path = config.dataset.corpus_export_path
    from klab.corpus.completed_release import in_place

    if not in_place(release, release_manifest):
        raise ValueError("tokenizer observation requires a retained native release")
    if export_path is None or config.dataset.corpus_release_path.resolve() != release:
        raise ValueError("tokenizer observation release differs")
    export_raw = read("export")
    if names["export"] != (export_path / "export.json").resolve():
        raise ValueError("tokenizer observation export path differs")
    export = json.loads(export_raw)
    if export_raw != canonical_json(export) + b"\n":
        raise ValueError("tokenizer observation export is not canonical")
    # Validate untrusted split fields before the binding projection indexes them.
    # Keep exception translation local to metadata access/arithmetic, not loaders
    # or provenance checks whose internal failures must remain visible.
    try:
        split_values = {
            split: (
                export["splits"][split]["path"],
                export["splits"][split]["sha256"],
                export["splits"][split]["records"],
                export["splits"][split]["rendered_bytes"]
                + export["splits"][split]["records"]
                + 1,
            )
            for split in ("train", "validation")
        }
    except (KeyError, TypeError) as error:
        raise ValueError("tokenizer observation split differs") from error
    for split in ("train", "validation"):
        view = f"lm/{split}.jsonl"
        dataset_path = getattr(config.dataset, f"{split}_path")
        if (
            dataset_path is None
            or dataset_path.resolve() != (release / view).resolve()
            or split_values[split][0] != view
            or split_values[split][1] != release_manifest["files"][view]["sha256"]
            or split_values[split][2]
            != getattr(config.dataset, f"{split}_max_documents")
            or split_values[split][3] != getattr(config.dataset, f"{split}_max_tokens")
        ):
            raise ValueError("tokenizer observation split differs")
    request = _export_request(
        release_manifest["release_id"],
        "lm",
        export.get("base_config_sha256"),
        config.vocab_size,
        export.get("min_frequency", 1),
    )
    if not isinstance(export.get("base_config_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", export["base_config_sha256"]
    ):
        raise ValueError("tokenizer observation export request is invalid")
    _read(export_path / "run.yaml", export.get("run_config_sha256"))
    _read(export_path / "tokenizer.yaml", export.get("tokenizer_config_sha256"))
    exported_tokenizer = load_tokenizer_config(export_path / "tokenizer.yaml")
    if (
        export.get("release_id") != release_manifest["release_id"]
        or export.get("schema_version") != 1
        or export.get("view") != "lm"
        or export.get("vocab_size") != config.vocab_size
        or export_path.name != canonical_digest(request)
        or export_path.parts[-6:-1]
        != (
            "corpora",
            release_manifest["corpus_id"],
            "exports",
            release_manifest["release_id"],
            "lm",
        )
        or export.get("release_manifest_sha256")
        != hashlib.sha256((release / "manifest.json").read_bytes()).hexdigest()
        or export.get("report_sha256")
        != release_manifest["files"]["report.json"]["sha256"]
        or export.get("license_report_sha256")
        != release_manifest["files"]["license-report.json"]["sha256"]
        or load_config(export_path / "run.yaml").dataset.model_dump(mode="json")
        != config.dataset.model_dump(mode="json")
        or exported_tokenizer.dataset.model_dump(mode="json")
        != config.dataset.model_dump(mode="json")
        or exported_tokenizer.vocab_size != config.vocab_size
        or exported_tokenizer.min_frequency != request.get("min_frequency", 1)
        or token_manifest.get("corpus_export") != _export_binding(export)
        or token_manifest.get("training_contract", {}).get("corpus_export")
        != _export_binding(export)
        or token_manifest.get("sha256") != hashlib.sha256(token_raw).hexdigest()
        or token_manifest.get("revision") != release_manifest["release_id"]
        or token_manifest.get("source") != "local_text"
        or token_manifest.get("vocab_size") != config.vocab_size
        or config.sampling is None
        or token_manifest.get("training_contract", {}).get("vocab_size")
        != config.vocab_size
        or token_manifest.get("training_contract", {}).get("min_frequency")
        != config.min_frequency
        or token_manifest.get("training_contract", {}).get("max_documents")
        != config.max_documents
        or token_manifest.get("training_contract", {}).get("source")
        != config.dataset.source
        or token_manifest.get("training_contract", {}).get("revision")
        != config.dataset.revision
        or token_manifest.get("training_contract", {}).get("sampling")
        != config.sampling.model_dump(mode="json")
        or token_manifest.get("sampling_audit", {}).get("complete") is not True
        or token_manifest.get("sampling_audit", {}).get("disjoint_from_fit") is not True
    ):
        raise ValueError("tokenizer observation provenance differs")
    result = {
        "method": "post_run_observation",
        "observed_at_utc": values["observed_at_utc"],
        "session_id": session["session_id"],
        "source_commit": job_spec["source"]["commit"],
        "job_spec_sha256": files["job_spec"]["sha256"],
        "session_spec_sha256": files["session_spec"]["sha256"],
        "tokenizer_config_sha256": files["tokenizer_config"]["sha256"],
        "export_sha256": files["export"]["sha256"],
        "stdout_sha256": files["stdout"]["sha256"],
        "stderr_sha256": files["stderr"]["sha256"],
        "tokenizer_sha256": files["tokenizer"]["sha256"],
        "tokenizer_manifest_sha256": files["tokenizer_manifest"]["sha256"],
        "job_report_sha256": files["job_report"]["sha256"],
        "session_report_sha256": files["session_report"]["sha256"],
        "container_report_sha256": files["container_report"]["sha256"],
        "limitation": "output digests were observed after completion; the job receipt did not attest them",
        "deferred": "export split and license recomputation, tokenizer sample selection and disjoint audit were not replayed",
    }
    return result
