"""Controlled offline lm-eval protocol; task implementations remain upstream."""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from importlib.metadata import version
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from klab.config.models import StrictModel

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Count = Annotated[int, Field(strict=True, ge=0)]


class TaskPin(StrictModel):
    config_sha256: Digest
    dataset_fingerprints: dict[str, str]
    evaluation_split: str
    primary_metric: str
    context: Literal["standard", "support_passage"] = "standard"

    @model_validator(mode="after")
    def complete(self) -> TaskPin:
        if not self.dataset_fingerprints or any(
            not k or not v for k, v in self.dataset_fingerprints.items()
        ):
            raise ValueError("pin every cached dataset split fingerprint")
        if (
            self.evaluation_split not in self.dataset_fingerprints
            or not self.primary_metric
        ):
            raise ValueError("evaluation split and primary metric required")
        return self


class BenchmarkProtocol(StrictModel):
    harness_version: Literal["0.4.13"]
    dependency_versions: dict[str, str]
    tasks: dict[
        Literal["lambada_openai", "piqa", "arc_easy", "hellaswag", "sciq"], TaskPin
    ]
    limit: Annotated[int, Field(strict=True, ge=1)]
    selection: Literal["first_n"] = "first_n"
    num_fewshot: Count
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
    fewshot_seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
    bootstrap_iters: Count
    cache_policy: Literal["offline"] = "offline"

    @model_validator(mode="after")
    def complete(self) -> BenchmarkProtocol:
        if not self.tasks:
            raise ValueError("at least one pinned task required")
        if (
            not {"datasets", "tokenizers", "torch", "numpy"}
            <= self.dependency_versions.keys()
        ):
            raise ValueError("pin datasets, tokenizers, torch and numpy versions")
        if "lm-eval" in self.dependency_versions:
            raise ValueError("harness version is pinned separately")
        if any(not v for v in self.dependency_versions.values()):
            raise ValueError("nonempty dependency versions required")
        if "sciq" in self.tasks and self.tasks["sciq"].context != "support_passage":
            raise ValueError("SciQ is support-passage conditioned, not closed-book")
        return self


def digest(value: Any) -> str:
    def scalar(obj: Any) -> Any:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"non-JSON benchmark evidence: {type(obj).__name__}")

    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=scalar
        ).encode()
    ).hexdigest()


def task_config_digest(config: dict[str, Any]) -> str:
    """Bind effective task content, excluding only its machine-local YAML path."""
    # lm-eval v0.4.13 tasks/_factory.py: TaskFactory._load_full_config
    # writes the YAML location under metadata, beside semantic fields like version.
    canonical = dict(config)
    metadata = canonical.get("metadata")
    if isinstance(metadata, dict) and "config_source" in metadata:
        canonical["metadata"] = {
            key: value for key, value in metadata.items() if key != "config_source"
        }
    return digest(canonical)


def finite_evidence(value: Any) -> tuple[Any, bool]:
    """Keep nonfinite observations JSON-safe, with an explicit completeness flag."""
    import numpy as np

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return (
            "NaN" if math.isnan(value) else "+Infinity" if value > 0 else "-Infinity"
        ), False
    if isinstance(value, (dict, list, tuple)):
        entries = value.items() if isinstance(value, dict) else enumerate(value)
        converted = {k: finite_evidence(v) for k, v in entries}
        clean = {k: v for k, (v, _) in converted.items()}
        return (clean if isinstance(value, dict) else list(clean.values())), all(
            ok for _, ok in converted.values()
        )
    return value, True


@contextmanager
def offline():
    # Importing these configuration modules does not load tasks or datasets.
    import datasets.config
    import huggingface_hub.constants

    # The probe worker is single-threaded. Restore flags even on a cache miss.
    settings = {
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    previous = {key: os.environ.get(key) for key in settings}
    flags = []
    try:
        os.environ.update(settings)
        for module, key in (
            (datasets.config, "HF_DATASETS_OFFLINE"),
            (datasets.config, "HF_HUB_OFFLINE"),
            (huggingface_hub.constants, "HF_HUB_OFFLINE"),
        ):
            flags.append((module, key, getattr(module, key)))
            setattr(module, key, True)
        yield
    finally:
        for module, key, value in flags:
            setattr(module, key, value)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def pinned_tasks(protocol: BenchmarkProtocol) -> list[Any]:
    """Load cached standard tasks and verify identities before any scoring."""
    from lm_eval.tasks import TaskManager

    for package, expected in {
        "lm-eval": protocol.harness_version,
        **protocol.dependency_versions,
    }.items():
        if version(package) != expected:
            raise ValueError(f"benchmark dependency version mismatch: {package}")
    tasks = TaskManager().load(list(protocol.tasks))["tasks"]
    if set(tasks) != set(protocol.tasks):
        raise ValueError("benchmark task expansion differs from pinned task names")
    for name, task in tasks.items():
        pin = protocol.tasks[name]
        config = task.dump_config()
        if task_config_digest(config) != pin.config_sha256:
            raise ValueError(f"benchmark task config mismatch: {name}")
        observed = {split: data._fingerprint for split, data in task.dataset.items()}
        if observed != pin.dataset_fingerprints:
            raise ValueError(f"benchmark dataset fingerprint mismatch: {name}")
        split = config.get("test_split") or config.get("validation_split")
        if split != pin.evaluation_split:
            raise ValueError(f"benchmark evaluation split mismatch: {name}")
        if config.get("num_fewshot") == 0 and protocol.num_fewshot != 0:
            raise ValueError(f"task fixes zero-shot; cannot override: {name}")
        if protocol.num_fewshot:
            # Mirror ConfigurableTask.fewshot_docs in the pinned harness: an
            # explicit split wins over samples, then train/validation/test.
            fewshot = config.get("fewshot_config") or {}
            source = fewshot.get("split") or config.get("fewshot_split")
            if source is None and fewshot.get("samples") is None:
                source = (
                    config.get("training_split")
                    or config.get("validation_split")
                    or config.get("test_split")
                )
            if source == split:
                raise ValueError(f"few-shot source overlaps evaluation split: {name}")
        if getattr(task, "UNSAFE_CODE", False):
            raise ValueError("unsafe task code unsupported")
    return list(tasks.values())


def controlled_results(
    results: dict[str, Any], protocol: BenchmarkProtocol, loaded: Any, elapsed: float
) -> dict[str, Any]:
    """Keep metrics and actual evidence; incompleteness never becomes a score."""
    rows = {}
    gaps = []
    for name, pin in protocol.tasks.items():
        metrics = (results.get("results") or {}).get(name, {})
        samples = (results.get("samples") or {}).get(name, [])
        ids = [s.get("doc_id") for s in samples]
        counts = (results.get("n-samples") or {}).get(name, {})
        primary = metrics.get(pin.primary_metric)
        metrics, metrics_finite = finite_evidence(metrics)
        samples, samples_finite = finite_evidence(samples)
        complete = (
            metrics_finite
            and samples_finite
            and len(ids) == protocol.limit
            and all(type(i) is int for i in ids)
            and set(ids) == set(range(protocol.limit))
            and all(
                s.get("doc_hash") is not None
                and s.get("arguments") is not None
                and "target" in s
                for s in samples
            )
            and counts.get("effective") == protocol.limit
            and type(primary) in (int, float)
            and (results.get("n-shot") or {}).get(name) == protocol.num_fewshot
        )
        if not complete:
            gaps.append(name)
        rows[name] = {
            "metrics": metrics,
            "counts": counts,
            "logged_items": len(samples),
            "primary_metric": pin.primary_metric,
            "complete": complete,
            "nonfinite_evidence": not (metrics_finite and samples_finite),
            "context": pin.context,
            "items": [
                {"doc_id": s.get("doc_id"), "sha256": digest(s)} for s in samples
            ],
            "uncertainty": {
                "source": "lm-evaluation-harness",
                "bootstrap_iters": protocol.bootstrap_iters,
                "reported": {k: v for k, v in metrics.items() if "stderr" in k},
            },
        }
    identity = getattr(loaded, "identity", {})
    subject = {
        k: identity.get(k)
        for k in (
            "run_id",
            "checkpoint_relative_path",
            "checkpoint_sha256",
            "tokenizer_sha256",
            "step",
            "tokens_seen",
        )
    }
    return {
        "protocol": protocol.model_dump(mode="json"),
        "protocol_sha256": digest(protocol.model_dump(mode="json")),
        "tasks": rows,
        "missing_tasks": gaps,
        "evidence_complete": not gaps,
        "scientific_result": "not_assessed",
        "subject": subject,
        "evaluation_wall_seconds": elapsed,
        "training_time_seconds": None,
        "training_compute": None,
        "cost": None,
        "target_count_basis": "checkpoint tokens_seen; not independently audited exposure",
        "device": str(getattr(loaded, "device", "unknown")),
    }
