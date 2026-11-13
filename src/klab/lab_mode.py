"""Lab mode: one local command from a YAML delta to a scored comparison.

``klab try`` is the fast path for normal, authorized local experiments. It
derives the candidate through native ``config derive``, trains both arms through
the native trainer, scores both on the held-out validation split through native
checkpoint-bound evaluation and writes one compact, self-hashed run record.

It keeps:

* resource limits: storage preflight, optional ``ResourceEnvelope`` and an
  optional, caller-chosen wall limit (no implicit short timeout);
* data identity: tokenizer, train and validation digests for each arm, and a
  cached baseline is reused only when the inputs hashed *now* still match;
* safe cancellation: SIGINT/SIGTERM or a cancel sentinel stop the try at a safe
  point (checkpointed step, or before/after scoring) and an interrupted record
  is still written atomically;
* held-out checks: both arms are scored under one shared evaluation protocol
  (same tokenized validation bytes, context windows and batch boundaries) with
  the same tokenizer, or the comparison is refused.

A lab record is evidence of one local comparison, not a release, promotion or
scientific conclusion.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import resource
import secrets
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from klab.lab_context import (
    LabCancelled,
    LabContext,
    LabSignal,
)
from klab.lab_records import (
    TRY_FORMAT as RECORD_FORMAT,
)
from klab.lab_records import (
    canonical as _canonical,
)
from klab.lab_records import (
    eval_group,
    read_lab_record,
    write_sealed,
)

DELTA_VERSION = 1
_DELTA_KEYS = {"lab_delta_version", "question", "set"}
EXIT_NOT_COMPARABLE = 3
EXIT_INTERRUPTED = 130
_LIMIT_SEED = "single local comparison with one seed per arm; not a significance test"
_LIMIT_SCOPE = (
    "lab records are not release, promotion or scientific-conclusion evidence"
)


@dataclass(frozen=True)
class Candidate:
    """A candidate arm: either a delta over the baseline or a full config."""

    kind: str  # "delta" or "config"
    source: Path
    settings: dict[str, Any]
    question: str | None


def lab_root(work_dir: Path, override: Path | None = None) -> Path:
    """Lab-mode state lives under the persistent work root, outside Git."""
    return (override if override is not None else work_dir / "lab").absolute()


def read_candidate(path: Path) -> Candidate:
    """Classify CANDIDATE as a lab delta document or a full RunConfig."""
    from klab.derivation import _settings
    from klab.documents import read_document

    path = Path(path).absolute()
    document = read_document(path)
    if "schema_version" in document:
        return Candidate("config", path, {}, None)
    unknown = set(document) - _DELTA_KEYS
    if unknown:
        raise ValueError(
            "lab delta accepts only lab_delta_version, question and set; "
            f"unexpected: {', '.join(sorted(unknown))}"
        )
    version = document.get("lab_delta_version", DELTA_VERSION)
    if version != DELTA_VERSION or type(version) is not int:
        raise ValueError(f"unsupported lab_delta_version: {version!r}")
    settings = document.get("set")
    if not isinstance(settings, dict) or not settings:
        raise ValueError("lab delta requires a nonempty 'set' mapping of dotted fields")
    _settings(settings)
    question = document.get("question")
    if question is not None and not isinstance(question, str):
        raise ValueError("lab delta question must be a string")
    return Candidate("delta", path, dict(settings), question)


def _sha256_file(path: Path) -> str:
    from klab.training.manifest import sha256_file

    return sha256_file(path)


def _with_runs_dir(config: Any, runs: Path) -> Any:
    """Operational relocation of the run store; scientific settings unchanged."""
    return config.model_copy(
        update={"logging": config.logging.model_copy(update={"root_dir": runs})}
    )


def _with_backend(config: Any, backend: str | None) -> Any:
    if backend is None:
        return config
    return config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"backend": backend})}
    )


def _config_key(config: Any) -> str:
    from klab.training.manifest import config_sha256, source_identity

    return hashlib.sha256(
        _canonical(
            {
                "config_sha256": config_sha256(config.model_dump(mode="json")),
                "source_identity_sha256": source_identity()["sha256"],
            }
        )
    ).hexdigest()


# Dataset path fields that are inputs to preparation. ``cache_dir`` holds derived
# products and ``mixture_output_path`` is written by preparation, so neither is
# part of the current input identity.
_INPUT_PATH_FIELDS = (
    "train_path",
    "validation_path",
    "source_manifest_path",
    "corpus_release_path",
    "corpus_export_path",
    "mixture_declaration_path",
)


def _path_digest(path: Path) -> str | dict[str, str]:
    """Content digest of an input file, or of every file under an input folder."""
    if path.is_symlink():
        return {"symlink": os.readlink(path)}
    if path.is_file():
        return _sha256_file(path)
    if path.is_dir():
        tree = {
            str(child.relative_to(path)): _sha256_file(child)
            for child in sorted(path.rglob("*"))
            if child.is_file() and not child.is_symlink()
        }
        return {"tree_sha256": hashlib.sha256(_canonical(tree)).hexdigest()}
    return "missing"


# External memory artifacts a run loads by path: the portable Engram package
# (``model.memory_package_path``, a file or folder). Its *contents* are part of
# what a run trained with.
MEMORY_PACKAGE_FIELDS = (("model", "memory_package_path"),)


def memory_packages(config: Any) -> dict[str, Any]:
    """Content digests, computed now, of every memory package CONFIG references."""
    packages: dict[str, Any] = {}
    for section, field in MEMORY_PACKAGE_FIELDS:
        value = getattr(getattr(config, section), field, None)
        if value is not None:
            packages[f"{section}.{field}"] = _path_digest(Path(value))
    return packages


def frozen_checkpoints(config: Any) -> dict[str, str]:
    """Digest the parent and teacher checkpoints a config starts or learns from."""
    from klab.training.manifest import checkpoint_selection

    digests = {}
    for role, path in (
        ("parent", config.training.parent_checkpoint),
        ("teacher", config.training.objective.teacher_checkpoint),
    ):
        if path is None:
            continue
        selected = checkpoint_selection(Path(path))
        identity = selected if selected.is_file() else selected / "manifest.json"
        digests[role] = _sha256_file(identity) if identity.is_file() else "missing"
    return digests


def current_inputs(config: Any) -> dict[str, Any]:
    """Hash tokenizer, dataset and memory-package inputs as they are on disk now."""
    from klab.data.packing import canonical_tokenizer_sha256
    from klab.data.tokenizer import load_tokenizer

    tokenizer_path = Path(config.tokenizer.path)
    try:
        # Same digest prepared data records: the complete serialized tokenizer.
        tokenizer_sha256 = canonical_tokenizer_sha256(load_tokenizer(tokenizer_path))
    except Exception:  # noqa: BLE001 - an unreadable tokenizer is never reusable
        tokenizer_sha256 = "unreadable"
    inputs: dict[str, Any] = {
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_file_sha256": _sha256_file(tokenizer_path)
        if tokenizer_path.is_file()
        else "missing",
        "dataset_source": config.dataset.source,
        "dataset_paths": {},
    }
    for field in _INPUT_PATH_FIELDS:
        value = getattr(config.dataset, field, None)
        if value is not None:
            inputs["dataset_paths"][field] = _path_digest(Path(value))
    inputs["memory_packages"] = memory_packages(config)
    checkpoints = frozen_checkpoints(config)
    if checkpoints:
        inputs["checkpoints"] = checkpoints
    inputs["sha256"] = hashlib.sha256(_canonical(inputs)).hexdigest()
    return inputs


def reuse_key(config: Any, inputs: Mapping[str, Any]) -> str:
    """Config + code + current tokenizer/data contents; any change retrains."""
    return hashlib.sha256(
        _canonical(
            {"config_key": _config_key(config), "inputs_sha256": inputs["sha256"]}
        )
    ).hexdigest()


def _progress(run: Path) -> dict[str, Any]:
    path = run / "progress.json"
    if path.is_symlink() or not path.is_file():
        return {"status": "missing"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return {"status": "unreadable"}
    return value if isinstance(value, dict) else {"status": "unreadable"}


def _stop_reason(progress: Mapping[str, Any]) -> str | None:
    stages = progress.get("stages")
    if isinstance(stages, list) and stages and isinstance(stages[-1], dict):
        reason = stages[-1].get("reason")
        if isinstance(reason, str):
            return reason
    return None


def _reusable_baseline(
    root: Path, runs: Path, key: str, inputs: Mapping[str, Any]
) -> str | None:
    """Return a baseline run proven current by a completed, sealed try record.

    A baseline is reused only when an earlier try *completed* with the same
    reuse key (config, code and the tokenizer/data contents hashed now), its run
    still reports completion, and its prepared tokenizer digest equals the
    tokenizer on disk now. Interrupted or failed tries are never a source.
    """
    candidates: list[tuple[str, str]] = []
    for path in (root / "tries").glob("*/try.json"):
        try:
            record = read_record(path)
        except OSError, ValueError, TypeError:
            continue
        base = (record.get("arms") or {}).get("baseline") or {}
        if (
            record.get("status") == "completed"
            and record.get("continuation") is None
            and base.get("reuse_key") == key
            and isinstance(base.get("run_id"), str)
        ):
            candidates.append((str(record.get("created_at")), base["run_id"]))
    for _, run_id in sorted(candidates, reverse=True):
        run = runs / run_id
        if run.is_symlink() or _progress(run).get("status") != "completed":
            continue
        try:
            data = _data_identity(run)
        except OSError, ValueError, TypeError:
            continue
        if data["tokenizer_sha256"] == inputs["tokenizer_sha256"]:
            return run_id
    return None


def _data_identity(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / "data" / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise TypeError("prepared-data manifest must be an object")
    return {
        "source": manifest.get("source"),
        "prepared_manifest_sha256": manifest.get("manifest_sha256"),
        "source_identity_sha256": manifest.get("source_identity_sha256"),
        "tokenizer_sha256": manifest.get("tokenizer_sha256"),
        "train_sha256": manifest.get("train", {}).get("sha256"),
        "train_tokens": manifest.get("train", {}).get("tokens"),
        "validation_sha256": manifest.get("validation", {}).get("sha256"),
        "validation_tokens": manifest.get("validation", {}).get("tokens"),
    }


def _run_identity(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text())
    return {
        "effective_config_sha256": manifest.get("effective_config_sha256"),
        "git_commit": manifest.get("git_commit"),
        "git_dirty": manifest.get("git_dirty"),
        "package_version": manifest.get("package_version"),
        "source_identity_sha256": (manifest.get("source_identity") or {}).get("sha256")
        if isinstance(manifest.get("source_identity"), dict)
        else manifest.get("source_identity"),
    }


def code_revision() -> dict[str, Any]:
    """Best-effort Git revision of the running checkout (read-only Git calls)."""
    import subprocess

    def git(*arguments: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).parent), *arguments],
            check=True,
            capture_output=True,
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        dirty = bool(git("status", "--porcelain=v1", "--untracked-files=no"))
    except OSError, subprocess.CalledProcessError, UnicodeError:
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def _peak_rss_bytes() -> int | None:
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except OSError, ValueError:
        return None
    # Linux reports KiB; macOS reports bytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def eval_protocol(config: Any) -> dict[str, int]:
    """The evaluation windowing and batching both arms are scored with."""
    return {
        "seq_len": config.training.seq_len,
        "batch_size": config.evaluation.batch_size or config.training.micro_batch_size,
        "max_batches": config.evaluation.max_batches,
    }


def _with_eval_protocol(loaded: Any, protocol: Mapping[str, int]) -> Any:
    """Score a loaded run under PROTOCOL without touching its training config."""
    config = loaded.config
    return dataclasses.replace(
        loaded,
        config=config.model_copy(
            update={
                "training": config.training.model_copy(
                    update={
                        "seq_len": protocol["seq_len"],
                        "micro_batch_size": protocol["batch_size"],
                    }
                ),
                "evaluation": config.evaluation.model_copy(
                    update={
                        "max_batches": protocol["max_batches"],
                        "batch_size": protocol["batch_size"],
                    }
                ),
            }
        ),
    )


def _train_and_score(
    *,
    arm: str,
    config: Any,
    run_id: str,
    reuse: bool,
    key: str | None,
    protocol: Mapping[str, int],
    context: LabContext,
    max_wall_seconds: float | None,
    authorization: Any,
    tokenizer_batch_documents: int,
    tokenizer_batch_source_bytes: int,
    train_fn: Callable[..., str],
    measurements: dict[str, Any] | None = None,
    continuation: Any = None,
    continuation_identity: Mapping[str, Any] | None = None,
    text_documents: int | None = None,
    gradient_bottleneck: bool = False,
) -> dict[str, Any]:
    """Train (or reuse) and score one arm under CONTEXT.

    With MEASUREMENTS, the scoring pass also records the probe battery's
    validation statistics for this arm (one forward pass serves both). With
    TEXT_DOCUMENTS (the baseline's ``evaluation.text_documents``), it also
    scores that many held-out documents for bits per byte; with
    GRADIENT_BOTTLENECK, it measures the output head's gradient bottleneck on
    the same held-out targets.
    """
    from klab.evaluation.inference import load_run, write_inference_result

    runs = config.logging.root_dir
    context.row = None
    context.enter(arm, "training")
    context.checkpoint()
    # The memory this arm trains with, hashed now (a reused baseline's key
    # already bound the same digests moments ago).
    packages = memory_packages(config)
    context.checkpoint()
    if continuation_identity is not None and (
        _sha256_file(continuation.checkpoint / "manifest.json")
        != continuation_identity["checkpoint_manifest_sha256"]
    ):
        raise ValueError("parent checkpoint changed between try arms")
    started = time.monotonic()
    if not reuse:
        try:
            returned = train_fn(
                config,
                **(
                    {
                        "extend_budget": continuation.checkpoint,
                        "append_data": continuation.append_data,
                        "append_from_checkpoint": continuation.append_from_checkpoint,
                        "allow_runtime_optimization": continuation.allow_runtime_optimization,
                        "allow_runtime_drift": continuation.allow_runtime_drift,
                    }
                    if continuation is not None
                    else {}
                ),
                run_id=run_id,
                cancel_path=context.cancel_path,
                max_wall_seconds=max_wall_seconds,
                authorization=authorization,
                resource_envelope=context.resource_envelope,
                tokenizer_batch_documents=tokenizer_batch_documents,
                tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
            )
        except InterruptedError:
            # Before run creation the trainer has no progress/checkpoint to read.
            # Only the actual cancel marker authorizes cancellation classification.
            context.check_cancel()
            raise
        if returned != run_id:
            raise ValueError(f"trainer returned unexpected run id {returned!r}")
    train_seconds = time.monotonic() - started
    run = runs / run_id
    progress = _progress(run)
    row: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run),
        "reused": reuse,
        "reuse_key": key,
        "status": progress.get("status"),
        "stop_reason": _stop_reason(progress),
        "train_seconds": None if reuse else round(train_seconds, 3),
        "seed": config.seed,
        "memory_packages": packages,
    }
    context.row = row
    if progress.get("status") != "completed":
        row["status"] = progress.get("status") or "interrupted"
        raise LabCancelled(
            arm, str(row["stop_reason"] or row["status"]), row, "training"
        )
    context.checkpoint()
    context.enter(arm, "scoring")
    started = time.monotonic()
    loaded = load_run(run_id, runs, None, None, authorization=authorization)
    data = _data_identity(run)
    shareable = (
        protocol["seq_len"] <= loaded.config.model.max_seq_len and loaded.engine is None
    )
    # Score under the shared protocol when the model can; otherwise score
    # natively and let the differing protocol identity refuse the comparison.
    scored = _with_eval_protocol(
        loaded, protocol if shareable else eval_protocol(loaded.config)
    )
    used = eval_protocol(scored.config)
    blocks = len(scored.validation_dataset())
    protocol_identity = {
        "split": "validation",
        "validation_sha256": data["validation_sha256"],
        "tokenizer_sha256": data["tokenizer_sha256"],
        "packing_version": _packing_version(run),
        # Which positions are scored: identical tokens and target counts can
        # still differ in position under assistant-only objectives.
        **_supervision_identity(run),
        **used,
        "scored_blocks": min(blocks, used["batch_size"] * used["max_batches"]),
        "evaluator": "pytorch" if loaded.engine is None else "mlx",
    }
    stats = None
    if loaded.engine is None:
        # The probe battery's own observer: per-window sums make this arm
        # pairable with any probe or try result in the same eval group.
        from klab.probes.scoring import ValidationStats

        stats = ValidationStats()
    bottleneck = None
    if gradient_bottleneck and loaded.engine is None:
        from klab.evaluation.gradient_bottleneck import GradientBottleneck

        bottleneck = GradientBottleneck(loaded.model)
    from klab.evaluation.gradient_bottleneck import observers

    result = scored.evaluate(observer=observers(stats, bottleneck))
    if bottleneck is not None:
        result["gradient_bottleneck"] = bottleneck.result()
    text = loaded.text_bits(text_documents) if text_documents else None
    if text is not None:
        result["text"] = text
    validation = stats.result(result) if stats is not None else None
    if validation is not None and measurements is not None and shareable:
        measurements[arm] = {
            "checkpoint_sha256": loaded.identity.get("checkpoint_sha256"),
            "validation": {**validation, "protocol": dict(protocol)},
        }
    result.update(
        {
            "source": "lab_try_eval",
            "identity": loaded.identity,
            "eval_protocol": protocol_identity,
        }
    )
    observation = write_inference_result(loaded.run, "eval", result)
    identity = loaded.identity
    pareto: dict[str, Any] = {"eval_group": None}
    if validation is not None:
        from klab.probes.runner import _validation_identity, footprint

        pareto = {
            # Same digest the probe battery records for this checkpoint.
            "eval_group": eval_group(_validation_identity(loaded), used),
            **footprint(loaded.model, identity.get("parameter_inventory")),
        }
    row.update(
        {
            "eval_seconds": round(time.monotonic() - started, 3),
            "checkpoint": identity.get("checkpoint_relative_path"),
            "checkpoint_sha256": identity.get("checkpoint_sha256"),
            "step": identity.get("step"),
            "tokens_seen": identity.get("tokens_seen"),
            "training_targets_applied": identity.get("tokens_seen")
            - (
                continuation_identity["tokens_seen"]
                if continuation_identity is not None
                else 0
            ),
            "parameters": (identity.get("parameter_inventory") or {}).get("total"),
            **pareto,
            "eval_protocol": protocol_identity,
            "eval_protocol_sha256": hashlib.sha256(
                _canonical(protocol_identity)
            ).hexdigest(),
            "heldout": {
                "split": "validation",
                "loss": result["loss"],
                "perplexity": result["perplexity"],
                "valid_targets": result["valid_targets"],
                "batches": result["batches"],
                **({"text": text} if text is not None else {}),
                **(
                    {"gradient_bottleneck": result["gradient_bottleneck"]}
                    if bottleneck is not None
                    else {}
                ),
                "observation": str(observation),
                **(
                    {
                        "ms_per_token": validation["ms_per_token"],
                        "window_sums": [float(v) for v in validation["window_sums"]],
                        "window_counts": [int(v) for v in validation["window_counts"]],
                    }
                    if validation is not None
                    else {}
                ),
            },
            "data": {
                **data,
                "eval_data_sha256": identity.get("data_sha256"),
                "eval_tokenizer_sha256": identity.get("tokenizer_sha256"),
            },
            "code": _run_identity(run),
        }
    )
    context.checkpoint()
    return row


def _supervision_identity(run: Path) -> dict[str, Any]:
    """Objective mode and a digest of the validation loss mask actually scored."""
    manifest = json.loads((run / "data" / "manifest.json").read_text())
    supervision = manifest.get("supervision") if isinstance(manifest, dict) else None
    kind = supervision.get("kind") if isinstance(supervision, dict) else None
    mask = run / "data" / "validation_supervision.npy"
    return {
        "objective": kind or "all_tokens",
        "supervision_mask_sha256": _sha256_file(mask) if mask.is_file() else None,
    }


def _packing_version(run: Path) -> Any:
    manifest = json.loads((run / "data" / "manifest.json").read_text())
    return manifest.get("packing_version") if isinstance(manifest, dict) else None


def heldout_checks(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    changed_variables: Iterable[str] = (),
) -> dict:
    """Refuse comparisons that are not scored on identical held-out inputs."""
    base_data = baseline["data"]
    cand_data = candidate["data"]
    changed = list(changed_variables)
    data_declared = any(name.startswith(("dataset.", "tokenizer.")) for name in changed)
    memory_declared = any(
        f"{section}.{field}" in changed for section, field in MEMORY_PACKAGE_FIELDS
    )
    checks = {
        "validation_distinct_from_train": all(
            data["validation_sha256"] is not None
            and data["validation_sha256"] != data["train_sha256"]
            for data in (base_data, cand_data)
        ),
        "same_validation_data": base_data["validation_sha256"]
        == cand_data["validation_sha256"]
        and base_data["validation_sha256"] is not None,
        "same_tokenizer": base_data["eval_tokenizer_sha256"]
        == cand_data["eval_tokenizer_sha256"],
        # Unless the delta declares a data change, both arms train on the same
        # current prepared bytes (a cached baseline cannot predate an edit).
        "same_training_data": data_declared
        or (
            base_data["train_sha256"] == cand_data["train_sha256"]
            and base_data["train_sha256"] is not None
        ),
        # Both arms trained with the same memory package contents, unless the
        # delta explicitly changes the package.
        "same_memory_packages": memory_declared
        or baseline.get("memory_packages") == candidate.get("memory_packages"),
        "same_eval_protocol": baseline.get("eval_protocol_sha256") is not None
        and baseline.get("eval_protocol_sha256")
        == candidate.get("eval_protocol_sha256"),
        "same_scored_targets": baseline["heldout"]["valid_targets"]
        == candidate["heldout"]["valid_targets"]
        and baseline["heldout"]["valid_targets"] > 0,
        "finite_losses": all(
            math.isfinite(row["heldout"]["loss"]) for row in (baseline, candidate)
        ),
    }
    return {
        "comparable": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
    }


# Token-level checks a tokenizer change fails by construction; bits per byte on
# the same held-out text stands in for them.
_TOKEN_LEVEL_CHECKS = frozenset(
    {
        "same_tokenizer",
        "same_validation_data",
        "same_eval_protocol",
        "same_scored_targets",
    }
)


def text_checks(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict:
    """Both arms scored the same held-out text for bits per byte."""
    texts = [row["heldout"].get("text") for row in (baseline, candidate)]
    present = all(isinstance(text, Mapping) for text in texts)
    base, cand = texts if present else ({}, {})
    return {
        "text_scored": present,
        "same_text": present
        and base.get("protocol") == cand.get("protocol")
        and base.get("text_sha256") == cand.get("text_sha256")
        and base.get("bytes") == cand.get("bytes")
        and bool(base.get("bytes")),
        "finite_bits_per_byte": present
        and all(
            isinstance(text.get("bits_per_byte"), (int, float))
            and math.isfinite(text["bits_per_byte"])
            for text in (base, cand)
        ),
    }


def compare(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    changed_variables: Iterable[str] = (),
) -> dict:
    """Candidate against baseline on held-out loss, or on bits per byte.

    Arms with the same tokenizer compare held-out loss per token (``metric:
    loss``). Arms whose tokenizers differ cannot; they compare bits per byte on
    the same held-out documents (``metric: bits_per_byte``) when the baseline's
    ``evaluation.text_documents`` scored them, and are refused otherwise.
    """
    checks = heldout_checks(baseline, candidate, changed_variables)
    texts = text_checks(baseline, candidate)
    metric = "loss"
    if not checks["checks"]["same_tokenizer"]:
        metric = "bits_per_byte"
        kept = {
            name: passed
            for name, passed in checks["checks"].items()
            if name not in _TOKEN_LEVEL_CHECKS
        }
        merged = {**kept, **texts}
        checks = {
            "comparable": all(merged.values()),
            "checks": merged,
            "failed": sorted(name for name, passed in merged.items() if not passed),
        }
    if not checks["comparable"]:
        return {"verdict": "NOT_COMPARABLE", "metric": metric, **checks}
    if metric == "loss":
        base_value = float(baseline["heldout"]["loss"])
        cand_value = float(candidate["heldout"]["loss"])
    else:
        base_value = float(baseline["heldout"]["text"]["bits_per_byte"])
        cand_value = float(candidate["heldout"]["text"]["bits_per_byte"])
    delta = cand_value - base_value
    verdict = (
        "CANDIDATE_LOWER_LOSS"
        if delta < 0
        else "CANDIDATE_HIGHER_LOSS"
        if delta > 0
        else "TIE"
    )
    result = {
        "verdict": verdict,
        "metric": metric,
        "heldout_delta": delta,
        "heldout_relative": delta / base_value if base_value else None,
        **checks,
    }
    if metric == "loss" and texts["same_text"] and texts["finite_bits_per_byte"]:
        result["bits_per_byte_delta"] = float(
            candidate["heldout"]["text"]["bits_per_byte"]
        ) - float(baseline["heldout"]["text"]["bits_per_byte"])
    return result


def write_record(
    path: Path, record: dict[str, Any], *, replace_started: bool = False
) -> dict[str, Any]:
    """Seal and publish RECORD atomically.

    An existing record is never replaced, except that the final record of a try
    may replace that same try's own sealed ``started`` record.
    """

    def own_started(existing: Path) -> bool:
        current = read_record(existing)
        return current.get("status") == "started" and current.get(
            "try_id"
        ) == record.get("try_id")

    return write_sealed(
        path, record, may_replace=own_started if replace_started else None
    )


def read_record(path: Path) -> dict[str, Any]:
    """Read a lab try record and reject edited content (shared reader)."""
    return read_lab_record(path, "try")[1]


def run_try(
    candidate_path: Path,
    baseline_path: Path,
    *,
    work_dir: Path,
    lab_dir: Path | None = None,
    seed: int | None = None,
    backend: str | None = None,
    max_wall_seconds: float | None = None,
    authorization: Any = None,
    resource_envelope: Any = None,
    tokenizer_batch_documents: int | None = None,
    tokenizer_batch_source_bytes: int | None = None,
    fresh_baseline: bool = False,
    repeat_baseline: bool = False,
    continuation: Any = None,
    train_fn: Callable[..., str] | None = None,
    now: Callable[[], datetime] | None = None,
    probe_tier: str | None = "fast",
    request: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Train/score baseline and candidate locally and write one lab record."""
    import yaml
    from pydantic import TypeAdapter

    from klab.config.loading import load_config
    from klab.config.models import RunSeed
    from klab.derivation import _config, _delta, derive_config
    from klab.documents import read_document
    from klab.experiments.compiler import patchable_config
    from klab.resource_envelope import (
        check_envelope,
        current_process_rss_bytes,
        resolve_lab_try_limits,
    )
    from klab.training.manifest import source_identity
    from klab.try_config import TryConfig

    # Only seed errors precede attempt creation. Other parsing/setup failures
    # still produce the established failed/interrupted record below.
    policy = TryConfig(
        seed=seed, repeat_baseline=repeat_baseline, continuation=continuation
    )
    continuation = policy.continuation
    seed_adapter = TypeAdapter(RunSeed)
    for path, is_candidate in ((baseline_path, False), (candidate_path, True)):
        try:
            # Match each consumer: baseline load_config uses SafeLoader,
            # while read_candidate uses the strict experiment document reader.
            document = (
                read_document(Path(path))
                if is_candidate
                else yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            )
        except OSError, ValueError, TypeError, yaml.YAMLError:
            continue  # Let normal setup retain its failure evidence.
        if not isinstance(document, dict):
            continue
        try:
            if "seed" in document and (
                not is_candidate or "schema_version" in document
            ):
                seed_adapter.validate_python(document["seed"])
            if is_candidate and "schema_version" not in document and seed is None:
                settings = document.get("set")
                if isinstance(settings, dict) and "seed" in settings:
                    seed_adapter.validate_python(settings["seed"])
        except ValueError as error:
            raise ValueError(f"invalid seed in {path}: {error}") from error

    if train_fn is None:
        from klab.training.trainer import train as train_fn
    declared_envelope = resource_envelope
    max_wall_seconds, tokenizer_batch_documents, tokenizer_batch_source_bytes = (
        resolve_lab_try_limits(
            resource_envelope,
            max_wall_seconds=max_wall_seconds,
            tokenizer_batch_documents=tokenizer_batch_documents,
            tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
        )
    )
    # Scoped limits are now explicit native trainer arguments. Delegated
    # consumers receive only common bounds and cannot silently ignore lab_try.
    if resource_envelope is not None and resource_envelope.lab_try is not None:
        resource_envelope = resource_envelope.model_copy(update={"lab_try": None})
    started_at = (now or (lambda: datetime.now(UTC)))()
    started = time.monotonic()
    root = lab_root(Path(work_dir), lab_dir)
    runs = root / "runs"
    from klab.lab_results import baseline_provenance

    request = dict(request) if request is not None else {"baseline": str(baseline_path)}
    provenance = baseline_provenance(request.get("baseline", str(baseline_path)))
    baseline_path = Path(baseline_path).absolute()
    provenance.update(
        status="resolved", resolved={"kind": "config", "path": str(baseline_path)}
    )
    candidate_path = Path(candidate_path).absolute()
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    try_id = f"try-{stamp}-{secrets.token_hex(4)}"
    try_dir = root / "tries" / try_id
    try_dir.mkdir(parents=True, exist_ok=False)
    record_path = try_dir / "try.json"
    cancel_path = try_dir / "CANCEL"
    # After seed preflight, cancellation and a sealed started record precede
    # other parsing, hashing and setup, retaining interrupted execution evidence.
    context = LabContext(
        cancel_path, resource_envelope=resource_envelope, workspace=root
    )
    context.enter(None, "setup")
    record: dict[str, Any] = {
        "format": RECORD_FORMAT,
        "mode": "lab",
        "try_id": try_id,
        "created_at": started_at.isoformat(),
        "inputs": {
            "baseline": {"path": str(baseline_path)},
            "candidate": {"path": str(candidate_path)},
            "seed_override": seed,
            "backend_override": backend,
        },
        "resources": {"cancel_path": str(cancel_path)},
        "arms": {},
        "status": "started",
        "baseline_provenance": provenance,
    }
    if request is not None:
        record["request"] = request
    exit_status = 0
    setup_error: BaseException | None = None
    context.install()
    try:
        try:
            write_record(record_path, record)
            print(
                f"lab try {try_id}: touch {cancel_path} (or Ctrl-C) to stop at a "
                "safe step",
                file=sys.stderr,
            )
            context.check_cancel()
            candidate = read_candidate(candidate_path)
            baseline_file = baseline_path
            if seed is not None:
                baseline_file = try_dir / "baseline.yaml"
                derive_config(baseline_path, baseline_file, {"seed": seed})
            declared: dict[str, Any] | None = None
            if candidate.kind == "delta":
                settings = dict(candidate.settings)
                if seed is not None:
                    # --seed is authoritative for both arms, over any delta seed.
                    settings["seed"] = seed
                candidate_file = try_dir / "candidate.yaml"
                declared = derive_config(baseline_path, candidate_file, settings)[
                    "declared_delta"
                ]
            else:
                candidate_file = candidate.source
                if seed is not None:
                    candidate_file = try_dir / "candidate.yaml"
                    derive_config(candidate.source, candidate_file, {"seed": seed})
            base_config = _with_backend(
                _with_runs_dir(load_config(baseline_file), runs), backend
            )
            cand_config = _with_backend(
                _with_runs_dir(load_config(candidate_file), runs), backend
            )
            delta = _delta(
                _config(
                    patchable_config(load_config(baseline_file)), baseline_file.parent
                ),
                _config(
                    patchable_config(load_config(candidate_file)),
                    candidate_file.parent,
                ),
            )
            context.check_cancel()
            # Bind baseline reuse to tokenizer, data and memory-package contents
            # as they are now; the candidate's identity is hashed the same way.
            inputs = current_inputs(base_config)
            context.check_cancel()
            cand_inputs = current_inputs(cand_config)
            context.check_cancel()
            base_key = reuse_key(base_config, inputs)
            if not delta or base_key == reuse_key(cand_config, cand_inputs):
                raise ValueError(
                    "candidate does not change any setting from the baseline"
                )
            # Resource limits are checked before any training starts.
            envelope_reading = (
                check_envelope(
                    resource_envelope,
                    workspace=root,
                    rss_bytes=current_process_rss_bytes(),
                )
                if resource_envelope is not None
                else None
            )
            baseline_sha256 = _sha256_file(baseline_path)
            candidate_sha256 = _sha256_file(candidate.source)
            context.check_cancel()
        except LabCancelled:
            raise
        except Exception as error:
            # Keep the failed attempt visible, then surface the setup error.
            record["status"] = "failed"
            record["failure"] = f"{type(error).__name__}: {error}"
            exit_status = 2
            setup_error = error
            raise
        runs.mkdir(parents=True, exist_ok=True)
        reuse_id = (
            None
            if fresh_baseline or repeat_baseline or continuation is not None
            else _reusable_baseline(root, runs, base_key, inputs)
        )
        new_id = f"lab-base-{base_key[:16]}-{try_id}"
        changed_variables = sorted(delta)
        memory_changed = any(
            f"{section}.{field}" in delta for section, field in MEMORY_PACKAGE_FIELDS
        )
        record.update(
            {
                "question": candidate.question,
                "inputs": {
                    "baseline": {"path": str(baseline_path), "sha256": baseline_sha256},
                    "candidate": {
                        "kind": candidate.kind,
                        "path": str(candidate.source),
                        "sha256": candidate_sha256,
                        "derived_config": str(candidate_file),
                    },
                    "seed_override": seed,
                    "backend_override": backend,
                },
                "declared_delta": declared,
                "delta": delta,
                "changed_variables": changed_variables,
                "seed": {
                    "override": seed,
                    "baseline": base_config.seed,
                    "candidate": cand_config.seed,
                    "changed_by_delta": "seed" in delta,
                },
                "memory_packages": {
                    "baseline": inputs["memory_packages"],
                    "candidate": cand_inputs["memory_packages"],
                    "changed_by_delta": memory_changed,
                },
                "baseline_inputs": inputs,
                "candidate_inputs": cand_inputs,
                "eval_protocol": eval_protocol(base_config),
                "code": {
                    "source_identity_sha256": source_identity()["sha256"],
                    **code_revision(),
                },
                "limitations": [
                    _LIMIT_SEED,
                    _LIMIT_SCOPE,
                ],
                "status": "running",
            }
        )
        if continuation is not None:
            record["continuation"] = {
                **continuation.model_dump(mode="json"),
                "checkpoint_manifest_sha256": _sha256_file(
                    continuation.checkpoint / "manifest.json"
                ),
                "mode": "native_terminal_budget_extension",
            }
            parent_metadata = json.loads(
                (continuation.checkpoint / "manifest.json").read_text()
            )
            record["continuation"].update(
                {key: parent_metadata[key] for key in ("step", "tokens_seen", "sha256")}
            )
        record["execution_order"] = ["baseline", "candidate"] + (
            ["baseline_repeat"] if repeat_baseline else []
        )
        record["resources"].update(
            {
                "resource_envelope": None
                if declared_envelope is None
                else declared_envelope.model_dump(mode="json"),
                "envelope_reading": envelope_reading,
                "max_wall_seconds_per_arm": max_wall_seconds,
                "tokenizer_batch_documents": tokenizer_batch_documents,
                "tokenizer_batch_source_bytes": tokenizer_batch_source_bytes,
            }
        )
        measurements: dict[str, Any] | None = {} if probe_tier is not None else None
        common = {
            "max_wall_seconds": max_wall_seconds,
            "authorization": authorization,
            "measurements": measurements,
            "tokenizer_batch_documents": tokenizer_batch_documents,
            "tokenizer_batch_source_bytes": tokenizer_batch_source_bytes,
            "train_fn": train_fn,
            "context": context,
            "continuation": continuation,
            "continuation_identity": record.get("continuation"),
            "text_documents": base_config.evaluation.text_documents,
            "gradient_bottleneck": base_config.evaluation.gradient_bottleneck,
        }
        protocol = eval_protocol(base_config)
        record["arms"]["baseline"] = _train_and_score(
            arm="baseline",
            config=base_config,
            run_id=reuse_id or new_id,
            reuse=reuse_id is not None,
            key=base_key if continuation is None else None,
            protocol=protocol,
            **common,
        )
        record["eval_protocol"] = record["arms"]["baseline"]["eval_protocol"]
        record["arms"]["candidate"] = _train_and_score(
            arm="candidate",
            config=cand_config,
            run_id=f"lab-{try_id}",
            reuse=False,
            key=None,
            protocol=protocol,
            **common,
        )
        if repeat_baseline:
            record["arms"]["baseline_repeat"] = _train_and_score(
                arm="baseline_repeat",
                config=base_config,
                run_id=f"lab-base-repeat-{try_id}",
                reuse=False,
                key=base_key if continuation is None else None,
                protocol=protocol,
                **common,
            )
            record["baseline_repeat_comparison"] = compare(
                record["arms"]["baseline"],
                record["arms"]["baseline_repeat"],
                [],
            )
        context.enter("candidate", "comparison")
        record["comparison"] = compare(
            record["arms"]["baseline"],
            record["arms"]["candidate"],
            changed_variables,
        )
        context.checkpoint()
        record["leakage"] = leakage_screen(record["arms"], runs, root)
        record["status"] = "completed"
        if record["comparison"]["verdict"] == "NOT_COMPARABLE":
            exit_status = EXIT_NOT_COMPARABLE
        if probe_tier is not None:
            context.enter("candidate", "probing")
            record["probe"] = _probe_arms(
                record,
                runs,
                protocol,
                probe_tier,
                try_dir,
                authorization,
                context,
                measurements or {},
            )
            # A signal stops the battery at a safe point; the partial result is
            # already in the record. Honor it (and a late cancel) now.
            if context.signals:
                raise LabSignal(context.signals[0])
            if (record["probe"].get("stop") or {}).get("kind") == "interrupted":
                raise KeyboardInterrupt
            context.check_cancel()
    except LabCancelled as error:
        if error.phase == "probing":
            # Both arms are scored and compared; only the probe battery stopped.
            record.setdefault("probe", {"status": "interrupted"})
        elif error.arm is not None and error.row is not None:
            record["arms"][error.arm] = error.row
        record["status"] = "interrupted"
        record["interruption"] = {
            "arm": error.arm,
            "phase": error.phase,
            "reason": error.reason,
        }
        exit_status = EXIT_INTERRUPTED
    except (LabSignal, KeyboardInterrupt) as error:
        if context.phase == "probing":
            # Both arms are scored; only the optional probe battery stopped.
            # Keep the finalized partial battery when there is one.
            record.setdefault("probe", {"status": "interrupted"})
        else:
            if context.arm is not None and context.row is not None:
                record["arms"][context.arm] = context.row
            record.pop("comparison", None)
        record["status"] = "interrupted"
        record["interruption"] = {
            "arm": context.arm,
            "phase": context.phase,
            "reason": error.name
            if isinstance(error, LabSignal)
            else "keyboard_interrupt",
        }
        exit_status = EXIT_INTERRUPTED
    except Exception as error:  # noqa: BLE001 - every failure is retained in the record
        if setup_error is None:
            record["status"] = "failed"
            record["failure"] = f"{type(error).__name__}: {error}"
            exit_status = 1
    finally:
        # From here on signals are noted, never raised, so the record is written.
        context.finalizing = True
        if context.signals:
            record["signals_received"] = list(context.signals)
        record["resources"]["wall_seconds"] = round(time.monotonic() - started, 3)
        record["resources"]["peak_rss_bytes"] = _peak_rss_bytes()
        record["exit_status"] = exit_status
        try:
            sealed = write_record(record_path, record, replace_started=True)
        finally:
            context.restore()
    if setup_error is not None:
        raise setup_error
    return sealed, record_path


def leakage_screen(
    arms: Mapping[str, Mapping[str, Any]], runs: Path, root: Path
) -> dict[str, Any]:
    """Screen each arm's prepared training data for held-out probe items.

    Arms that trained on the same prepared data with the same tokenizer share
    one screen. A screen that cannot run is recorded as ``ERROR`` (missing
    evidence), never as clean.
    """
    from klab.probes.contamination import (
        CLEAN,
        CONTAMINATED,
        ERROR,
        WINDOW,
        check_prepared,
        leakage_cache_dir,
    )

    started = time.monotonic()
    screened: dict[tuple[Any, Any], dict[str, Any]] = {}
    rows: dict[str, Any] = {}
    for name, arm in arms.items():
        data = arm.get("data") or {}
        key = (data.get("prepared_manifest_sha256"), data.get("tokenizer_sha256"))
        if key not in screened:
            run = runs / str(arm["run_id"])
            try:
                screened[key] = check_prepared(
                    run / "data",
                    run / "tokenizer.json",
                    cache_dir=leakage_cache_dir(root),
                )
            except Exception as error:  # noqa: BLE001 - recorded as missing evidence
                screened[key] = {
                    "status": ERROR,
                    "error": f"{type(error).__name__}: {error}",
                }
        rows[name] = screened[key]
    statuses = {row["status"] for row in rows.values()}
    return {
        "status": CONTAMINATED
        if CONTAMINATED in statuses
        else ERROR
        if ERROR in statuses
        else CLEAN,
        "n": WINDOW,
        "arms": rows,
        "seconds": round(time.monotonic() - started, 3),
    }


def _probe_arms(
    record: Mapping[str, Any],
    runs: Path,
    protocol: Mapping[str, int],
    tier: str,
    try_dir: Path,
    authorization: Any,
    context: LabContext,
    measurements: Mapping[str, Any],
) -> dict[str, Any]:
    """Probe the candidate vs the baseline under the try's own context.

    Arms load one at a time; each reuses the validation statistics recorded by
    its scoring pass (same checkpoint and protocol), so validation is never
    scored twice. Probe problems never fail a try: cancellation, resources and
    OOM stop the battery (``incomplete``) and the comparison stands.
    """
    from klab.evaluation.inference import load_run
    from klab.probes.runner import Arm, progress_writer, run_battery

    def arm(name: str) -> Arm:
        row = record["arms"][name]
        cache: dict[str, Any] = {}
        scored = measurements.get(name) or {}
        if scored.get("checkpoint_sha256") == row.get("checkpoint_sha256"):
            cache["validation"] = scored.get("validation")
        return Arm(
            load=lambda: load_run(
                row["run_id"], runs, None, None, authorization=authorization
            ),
            cache=cache,
        )

    try:
        return run_battery(
            arm("candidate"),
            arm("baseline"),
            tier=tier,
            protocol=protocol,
            progress=progress_writer(try_dir / "probe-progress.json"),
            context=context,
            comparison=record.get("comparison"),
            leakage=record.get("leakage"),
        )
    except Exception as error:  # noqa: BLE001 - probes inform, never fail a try
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return "n/a" if value is None else str(value)


def summarize(
    record: Mapping[str, Any], path: Path | None = None, *, color: bool = False
) -> str:
    """Human-readable comparison; ``--json`` carries the same record."""
    lines = [f"LAB TRY {record['try_id']}  status: {record['status']}  (lab mode)"]
    from klab.lab_results import baseline_text

    lines.append(baseline_text(record))
    if record.get("question"):
        lines.append(f"  question: {record['question']}")
    for field, change in (record.get("delta") or {}).items():
        lines.append(f"  delta {field}: {change['base']!r} -> {change['variant']!r}")
    for arm in ("baseline", "candidate"):
        row = record.get("arms", {}).get(arm)
        if row is None:
            lines.append(f"  {arm:<9} not scored")
            continue
        heldout = row.get("heldout", {})
        reused = " (reused)" if row.get("reused") else ""
        text = heldout.get("text")
        bits = f"  bits/byte {_fmt(text['bits_per_byte'])}" if text else ""
        lines.append(
            f"  {arm:<9} {row['run_id']}{reused}  held-out loss "
            f"{_fmt(heldout.get('loss'))}  ppl {_fmt(heldout.get('perplexity'), 2)}"
            f"  targets {_fmt(heldout.get('valid_targets'))}{bits}"
        )
    comparison = record.get("comparison")
    if comparison:
        if comparison["verdict"] == "NOT_COMPARABLE":
            lines.append(
                "  verdict: NOT_COMPARABLE (failed held-out checks: "
                + ", ".join(comparison["failed"])
                + ")"
            )
        else:
            measure = (
                "held-out loss"
                if comparison.get("metric") == "loss"
                else "bits per byte"
            )
            lines.append(
                f"  verdict: {comparison['verdict']}  Δ {measure} "
                f"{comparison['heldout_delta']:+.4f}"
            )
    leakage = record.get("leakage")
    if leakage:
        lines.append(f"  leakage: {leakage['status']}  (n={leakage.get('n')})")
        for arm, row in (leakage.get("arms") or {}).items():
            if row.get("error"):
                lines.append(f"    {arm:<9} {row['error']}")
            elif row.get("checked"):
                counts = ", ".join(
                    f"{split} {row['contaminated'][split]}/{checked}"
                    for split, checked in row["checked"].items()
                )
                items = ", ".join(row.get("items") or [])
                lines.append(
                    f"    {arm:<9} contaminated {counts}"
                    + (f"  held-out items: {items}" if items else "")
                )
    if record.get("interruption"):
        stop = record["interruption"]
        lines.append(
            f"  stopped safely: arm={stop['arm']} phase={stop.get('phase')} "
            f"reason={stop['reason']}"
        )
    if record.get("failure"):
        lines.append(f"  failure: {record['failure']}")
    resources = record.get("resources", {})
    lines.append(
        f"  wall {_fmt(resources.get('wall_seconds'), 1)}s  "
        f"peak rss {_fmt(resources.get('peak_rss_bytes'))} bytes"
    )
    probe = record.get("probe")
    if isinstance(probe, Mapping) and probe.get("format"):
        from klab.probes.render import render
        from klab.probes.verdict import respect_try_comparison

        shown = {
            **probe,
            "verdict": respect_try_comparison(probe.get("verdict"), comparison),
        }
        lines.append("")
        lines.append(render(shown, color=color))
    elif isinstance(probe, Mapping):
        lines.append(
            f"  probe battery: {probe.get('status')} {probe.get('error') or ''}".rstrip()
        )
    if path is not None:
        lines.append(f"  record: {path}")
    return "\n".join(lines)
