"""Read-only post-training evidence, bounded physical probes, and immutable advice."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import sqlite3
import uuid
from collections import Counter
from collections.abc import Collection
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from klab.canonical import canonical_digest, canonical_json
from klab.config.models import RunConfig
from klab.data.packing import TokenBlockDataset, supervision_requires_mask
from klab.evaluation.evidence import validate_held_out_report
from klab.evaluation.generation import generate_with_token_ids
from klab.evaluation.inference import load_run
from klab.evaluation.reference_exercise import PROMPTS
from klab.model.inspection import parameter_inventory
from klab.runtime_profile import RuntimeAuthorization
from klab.training.checkpoints import CheckpointManager
from klab.training.manifest import IDENTITY_VERSION, read_manifest, sha256_file

FORMAT = "klab_post_train_triage_v1"
MAX_BYTES = 256 * 1024
_TRIGGERS = (
    "LOSS_BEHAVIOR_DIVERGENCE",
    "DECODER_SENSITIVITY_DETECTED",
    "ENDPOINT_STILL_LEARNING",
    "POSSIBLE_PLATEAU",
    "SCALE_WITHOUT_BEHAVIOR_GAIN",
    "SEED_DISAGREEMENT",
    "MECHANISM_PRESENT_BUT_USAGE_UNKNOWN",
)
_COST = {"negligible": 0, "low": 1, "medium": 2, "high": 3}
_GENERATION = re.compile(r"step_\d+_gen_\d+\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _run_path(run_id: str, runs_dir: Path) -> Path:
    if (
        not run_id
        or run_id in {".", ".."}
        or Path(run_id).name != run_id
        or "\\" in run_id
    ):
        raise ValueError("unsafe run ID")
    root = runs_dir.resolve(strict=True)
    run = root / run_id
    if run.is_symlink() or not run.is_dir() or run.resolve() != run:
        raise ValueError("unsafe run directory")
    return run


def _owned(run: Path, relative: str, *, directory: bool = False) -> Path:
    part = Path(relative)
    if (
        part.is_absolute()
        or not part.parts
        or any(p in {"..", "."} for p in part.parts)
    ):
        raise ValueError(f"unsafe run path: {relative}")
    path = run
    for name in part.parts:
        path = path / name
        if path.is_symlink():
            raise ValueError(f"symlink in run path: {relative}")
    if not path.resolve().is_relative_to(run):
        raise ValueError(f"path escapes run: {relative}")
    if directory and path.exists() and not path.is_dir():
        raise ValueError(f"not a directory: {relative}")
    return path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")  # noqa: TRY004 - serialized artifact
    return value


def _finite(value: object) -> bool:
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _metrics(
    runs_dir: Path, run_id: str
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object], str | None]:
    database = runs_dir / "experiments.sqlite3"
    if database.is_symlink() or not database.is_file():
        return [], {}, {}, "read-only metrics database unavailable"
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT step,tokens_seen,wall_time,name,value FROM metrics WHERE run_id=? ORDER BY step,name",
            (run_id,),
        ).fetchall()
        final = con.execute(
            "SELECT payload_json FROM events WHERE run_id=? AND kind='runtime_final_observation' ORDER BY id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        recorded = con.execute(
            "SELECT metadata_json FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
    return (
        [
            {"step": s, "targets": t, "wall_seconds": w, "name": n, "value": v}
            for s, t, w, n, v in rows
        ],
        _json_value(final[0]) if final else {},
        _json_value(recorded[0]) if recorded else {},
        None,
    )


def _json_value(raw: str) -> dict[str, object]:
    value = json.loads(raw)
    return value if isinstance(value, dict) else {}


def loss_trajectory(points: list[dict[str, object]]) -> dict[str, object]:
    """Compute transparent target-exposure-based rates, without a convergence claim."""
    series = sorted(
        (
            p
            for p in points
            if _finite(p.get("loss")) and isinstance(p.get("targets"), int)
        ),
        key=lambda p: (int(p["targets"]), int(p["step"])),
    )
    intervals = []
    for left, right in itertools.pairwise(series):
        added = int(right["targets"]) - int(left["targets"])
        intervals.append(
            {
                "from_step": left["step"],
                "to_step": right["step"],
                "from_targets": left["targets"],
                "to_targets": right["targets"],
                "added_target_exposures": added,
                "loss_decrease": float(left["loss"]) - float(right["loss"]),
                "nats_per_target_per_million_additional_target_exposures": (
                    float(left["loss"]) - float(right["loss"])
                )
                * 1e6
                / added
                if added > 0
                else None,
            }
        )
    result: dict[str, object] = {
        "points": series,
        "initial_loss": series[0]["loss"] if series else None,
        "terminal_loss": series[-1]["loss"] if series else None,
        "delta_loss": float(series[-1]["loss"]) - float(series[0]["loss"])
        if len(series) > 1
        else None,
        "intervals": intervals,
        "trend": "insufficient_history",
        "trend_intervals": None,
        "thresholds": {
            "terminal_regression_nats": 0.01,
            "recent_decrease_nats": 0.01,
            "plateau_recent_to_early_rate": 0.25,
        },
    }
    if len(series) < 4 or any(
        int(b["targets"]) <= int(a["targets"]) for a, b in itertools.pairwise(series)
    ):
        return result
    middle = (int(series[0]["targets"]) + int(series[-1]["targets"])) / 2
    pivot = min(
        range(1, len(series) - 1),
        key=lambda i: (abs(int(series[i]["targets"]) - middle), i),
    )
    early = float(series[0]["loss"]) - float(series[pivot]["loss"])
    recent = float(series[pivot]["loss"]) - float(series[-1]["loss"])
    early_rate = (
        early * 1e6 / (int(series[pivot]["targets"]) - int(series[0]["targets"]))
    )
    recent_rate = (
        recent * 1e6 / (int(series[-1]["targets"]) - int(series[pivot]["targets"]))
    )
    result["trend_intervals"] = {
        "early": {
            "from": series[0],
            "to": series[pivot],
            "loss_decrease": early,
            "rate": early_rate,
        },
        "recent": {
            "from": series[pivot],
            "to": series[-1],
            "loss_decrease": recent,
            "rate": recent_rate,
        },
        "penultimate_to_terminal_loss_increase": float(series[-1]["loss"])
        - float(series[-2]["loss"]),
    }
    regression = float(series[-1]["loss"]) - float(series[-2]["loss"])
    if regression >= 0.01 or math.isclose(regression, 0.01, abs_tol=1e-12):
        result["trend"] = "regressing"
    elif recent >= 0.01 or math.isclose(recent, 0.01, abs_tol=1e-12):
        result["trend"] = "still_improving"
    elif early > 0 and 0 <= recent < 0.01 and recent_rate < early_rate / 4:
        result["trend"] = "plateau_possible"
    else:
        result["trend"] = "unclear"
    return result


def _validation_mask_required(run: Path, artifacts: dict[str, str]) -> bool:
    manifest_path = _owned(run, "data/manifest.json")
    if artifacts.get("data/manifest.json") != sha256_file(manifest_path):
        raise ValueError("prepared-data manifest identity mismatch")
    required = supervision_requires_mask(_json(manifest_path))
    mask_path = _owned(run, "data/validation_supervision.npy")
    if (mask_path.is_file(), "data/validation_supervision.npy" in artifacts) != (
        required,
        required,
    ):
        raise ValueError("validation supervision inventory does not match descriptor")
    return required


def _expected_validation(
    run: Path,
    config: dict[str, Any],
    mask_required: bool,
) -> tuple[int, int]:
    if config["evaluation"].get("panels"):
        from klab.evaluation.panel_data import panel_coverage

        return panel_coverage(config["evaluation"])
    seq = config["training"]["seq_len"]
    batch = (
        config["evaluation"].get("batch_size") or config["training"]["micro_batch_size"]
    )
    supervision = "data/validation_supervision.npy"
    data = TokenBlockDataset(
        np.load(_owned(run, "data/validation.npy"), mmap_mode="r", allow_pickle=False),
        seq,
        supervision=np.load(_owned(run, supervision), mmap_mode="r", allow_pickle=False)
        if mask_required
        else None,
    )
    blocks = min(len(data), batch * config["evaluation"]["max_batches"])
    targets = blocks * seq
    if data.supervision is not None:
        targets = sum(
            int(
                np.count_nonzero(
                    data.supervision[int(block) * seq + 1 : int(block) * seq + 1 + seq]
                )
            )
            for block in data.block_indices[:blocks]
        )
    return targets, math.ceil(blocks / batch)


def _checkpoint_metadata(
    run: Path, name: str, run_digest: str
) -> dict[str, object] | None:
    if not _GENERATION.fullmatch(name):
        return None
    path = _owned(run, f"checkpoints/{name}/manifest.json")
    if not path.is_file():
        return None
    raw = _json(path)
    digest = raw.pop("sha256", None)
    if (
        not isinstance(digest, str)
        or canonical_digest(raw) != digest
        or raw.get("manifest_sha256") != run_digest
    ):
        return None
    return {
        "digest": digest,
        "step": raw.get("step"),
        "tokens_seen": raw.get("tokens_seen"),
        "validation_loss": raw.get("validation_loss"),
    }


def _evidence(
    run: Path,
    manifest: dict[str, Any],
    progress: dict[str, Any],
    digest: str,
    metrics: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    recorded = [
        {
            "step": row["step"],
            "targets": row["targets"],
            "loss": row["value"],
            "evidence": "recorded_not_checkpoint_verified",
        }
        for row in metrics
        if row["name"] == "validation/loss" and _finite(row["value"])
    ]
    unverified = {"points": recorded, "reports": [], "rejected_reports": []}
    endpoint = progress.get("latest")
    outcome: dict[str, object] = {
        "status": "UNKNOWN",
        "reason": "missing final immutable generation",
    }
    if not isinstance(endpoint, dict) or not isinstance(
        endpoint.get("relative_path"), str
    ):
        return outcome, unverified
    name = endpoint["relative_path"]
    if not _GENERATION.fullmatch(name):
        return {"status": "FAIL", "reason": "invalid final generation path"}, unverified
    try:
        generation = _owned(run, f"checkpoints/{name}", directory=True)
        manager = CheckpointManager(run, manifest_sha256=digest)
        verified = manager.verify(
            generation, expected_manifest=digest, require_training_state=True
        )
        metadata = _checkpoint_metadata(run, name, digest)
        if (
            not verified.valid
            or metadata is None
            or metadata["digest"] != endpoint.get("manifest_sha256")
            or metadata["step"] != progress.get("step")
            or metadata["tokens_seen"] != progress.get("tokens_seen")
        ):
            return {
                "status": "FAIL",
                "reason": "final checkpoint integrity/counters mismatch",
                "errors": list(verified.errors),
            }, unverified
        outcome = {
            "status": "PASS",
            "generation": name,
            "generation_manifest_sha256": metadata["digest"],
            "step": metadata["step"],
            "targets": metadata["tokens_seen"],
            "verification_scope": verified.resume_level,
        }
        artifacts = {
            item["relative_path"]: item["sha256"] for item in manifest["artifacts"]
        }
        try:
            mask_required = _validation_mask_required(run, artifacts)
        except (OSError, ValueError, TypeError, KeyError) as error:
            return outcome, {
                **unverified,
                "rejected_reports": [
                    {"path": "data/manifest.json", "reason": str(error)}
                ],
            }
        for asset in (
            "data/validation.npy",
            "tokenizer.json",
            *(("data/validation_supervision.npy",) if mask_required else ()),
            "data/validation_byte_addresses.npy",
        ):
            if (
                asset in artifacts
                and sha256_file(_owned(run, asset)) != artifacts[asset]
            ):
                return outcome, {
                    **unverified,
                    "rejected_reports": [
                        {
                            "path": asset,
                            "reason": "run-owned held-out artifact hash mismatch",
                        }
                    ],
                }
        try:
            expected, batches = _expected_validation(
                run, manifest["effective_config"], mask_required
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            return outcome, {
                **unverified,
                "rejected_reports": [
                    {
                        "path": "data/validation.npy",
                        "reason": f"held-out data unavailable: {error}",
                    }
                ],
            }
        lookup = {name: metadata}
        points: list[dict[str, object]] = list(recorded)
        reports: list[dict[str, object]] = []
        rejected: list[dict[str, str]] = []
        try:
            folder = _owned(run, "evaluations", directory=True)
        except ValueError as error:
            return outcome, {
                **unverified,
                "rejected_reports": [{"path": "evaluations", "reason": str(error)}],
            }
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if not re.fullmatch(r"validation_step_\d+_gen_\d+\.json", path.name):
                    continue
                try:
                    _owned(run, f"evaluations/{path.name}")
                    report_raw = _json(path)
                    checkpoint = report_raw.get("checkpoint")
                    if isinstance(checkpoint, str) and checkpoint not in lookup:
                        old = _checkpoint_metadata(run, checkpoint, digest)
                        if old is not None:
                            lookup[checkpoint] = old
                    valid, reason = validate_held_out_report(
                        path, lookup, artifacts, manifest, expected, batches
                    )
                    if valid is None:
                        rejected.append(
                            {"path": path.name, "reason": reason or "invalid report"}
                        )
                        continue
                    scope = (
                        "checkpoint_weights_verified"
                        if checkpoint == name
                        else "checkpoint_manifest_only_weights_not_verified"
                    )
                    reports.append(
                        {
                            "path": path.name,
                            "sha256": sha256_file(path),
                            "checkpoint": checkpoint,
                            "step": valid["step"],
                            "targets": valid["tokens_seen"],
                            "loss": valid["loss"],
                            "valid_targets": valid["valid_targets"],
                            "protocol": valid["identities"]["protocol"],
                            "verification_scope": scope,
                        }
                    )
                    points.append(
                        {
                            "step": valid["step"],
                            "targets": valid["tokens_seen"],
                            "loss": valid["loss"],
                            "evidence": scope,
                            "report": path.name,
                        }
                    )
                except (ValueError, TypeError, KeyError, OSError) as error:
                    rejected.append({"path": path.name, "reason": str(error)})
        dedup = {(p["step"], p["targets"]): p for p in points}
        return outcome, {
            "points": list(dedup.values()),
            "reports": reports,
            "rejected_reports": rejected,
            "expected_validation_targets": expected,
            "expected_validation_batches": batches,
        }
    except (ValueError, TypeError, KeyError, OSError) as error:
        return {"status": "FAIL", "reason": str(error)}, unverified


def _mechanics(ids: list[int], text: str, special: set[int]) -> dict[str, object]:
    def excess(sequence: list[object]) -> int:
        return sum(n - 1 for n in Counter(sequence).values() if n > 1)

    return {
        "emitted_length": len(ids),
        "repeated_token_excess": excess(ids),
        "repeated_bigram_excess": excess(list(itertools.pairwise(ids))),
        "repeated_trigram_excess": excess(list(zip(ids, ids[1:], ids[2:]))),
        "empty": not text.strip() or not ids,
        "special_token_ids": [i for i in ids if i in special],
    }


def _probe(
    run_id: str,
    runs_dir: Path,
    endpoint: dict[str, object],
    config: RunConfig,
    remaining_seconds: float | None,
    authorization: RuntimeAuthorization | None = None,
) -> dict[str, object]:
    inventory = parameter_inventory(config)
    backend = config.runtime.backend
    result: dict[str, object] = {
        "status": "AVAILABLE",
        "reason": None,
        "parameter_inventory": asdict(inventory),
        "greedy": [],
        "sampled": [],
        "settings": {
            "greedy": {
                "temperature": 0,
                "top_k": 0,
                "seed": 42042,
                "max_new_tokens": 32,
            },
            "sampled": {
                "temperature": 0.8,
                "top_k": [0, 40],
                "seeds": [11, 29],
                "max_new_tokens": 24,
            },
        },
    }
    if inventory.total > 100_000_000 or (
        backend == "cpu" and inventory.total > 10_000_000
    ):
        result["reason"] = "cost_ceiling"
        return result
    if remaining_seconds is not None and remaining_seconds < 120:
        result["reason"] = "deadline_less_than_120_seconds"
        return result
    try:
        loaded = load_run(
            run_id,
            runs_dir,
            str(
                _owned(
                    _run_path(run_id, runs_dir), f"checkpoints/{endpoint['generation']}"
                )
            ),
            authorization=authorization,
        )
    except Exception as error:  # noqa: BLE001
        result.update(
            status="UNKNOWN",
            reason=f"runtime/backend unavailable: {type(error).__name__}: {error}",
        )
        return result
    result["status"] = "OBSERVED"
    special = {
        v
        for token in ("<pad>", "<bos>", "<eos>", "<unk>")
        if (v := loaded.tokenizer.token_to_id(token)) is not None
    }
    greedy: dict[str, dict[str, object]] = {}
    for prompt_id, prompt in PROMPTS:
        prompt_ids = loaded.tokenizer.encode(prompt, add_special_tokens=False).ids
        if len(prompt_ids) + 32 > config.model.max_seq_len:
            case = {
                "id": prompt_id,
                "status": "UNKNOWN",
                "reason": "prompt and requested completion exceed context",
                "prompt_tokens": len(prompt_ids),
                "requested_new_tokens": 32,
            }
        else:
            try:
                text, ids = generate_with_token_ids(
                    loaded.model,
                    loaded.tokenizer,
                    prompt,
                    config.model.max_seq_len,
                    32,
                    loaded.device,
                    temperature=0,
                    top_k=0,
                    seed=42042,
                    strict_context=True,
                    engine=loaded.engine,
                )
                case = {
                    "id": prompt_id,
                    "status": "OBSERVED",
                    "prompt": prompt,
                    "text": text[:4096],
                    "token_ids": ids,
                    "mechanics": _mechanics(ids, text[len(prompt) :], special),
                    "settings": result["settings"]["greedy"],
                }
                greedy[prompt_id] = case
            except Exception as error:  # noqa: BLE001
                case = {
                    "id": prompt_id,
                    "status": "UNKNOWN",
                    "error": f"{type(error).__name__}: {error}"[:512],
                }
        result["greedy"].append(case)
    for prompt_id, prompt in PROMPTS[:2]:
        baseline = greedy.get(prompt_id)
        if baseline is None:
            continue
        for top_k in (0, 40):
            for seed in (11, 29):
                try:
                    text, ids = generate_with_token_ids(
                        loaded.model,
                        loaded.tokenizer,
                        prompt,
                        config.model.max_seq_len,
                        24,
                        loaded.device,
                        temperature=0.8,
                        top_k=top_k,
                        seed=seed,
                        strict_context=True,
                        engine=loaded.engine,
                    )
                    mechanics = _mechanics(ids, text[len(prompt) :], special)
                    matched = _mechanics(
                        list(baseline["token_ids"])[:24],
                        loaded.tokenizer.decode(
                            list(baseline["token_ids"])[:24], skip_special_tokens=True
                        ),
                        special,
                    )
                    case = {
                        "id": prompt_id,
                        "status": "OBSERVED",
                        "text": text[:4096],
                        "token_ids": ids,
                        "mechanics": mechanics,
                        "greedy_first_24_mechanics": matched,
                        "trigram_excess_difference": mechanics[
                            "repeated_trigram_excess"
                        ]
                        - matched["repeated_trigram_excess"],
                        "settings": {
                            "temperature": 0.8,
                            "top_k": top_k,
                            "seed": seed,
                            "max_new_tokens": 24,
                        },
                    }
                except Exception as error:  # noqa: BLE001
                    case = {
                        "id": prompt_id,
                        "status": "UNKNOWN",
                        "settings": {"top_k": top_k, "seed": seed},
                        "error": f"{type(error).__name__}: {error}"[:512],
                    }
                result["sampled"].append(case)
    if not greedy:
        result["status"] = "UNKNOWN"
        result["reason"] = (
            "no frozen prompt fits the full requested completion in context"
        )
    result["substantive_behavior_applicability"] = (
        "UNKNOWN: frozen TinyStories prompts can be out of domain"
    )
    if config.attention.kind == "mla":
        cache: dict[str, object] = {
            "status": "UNKNOWN",
            "reason": "cached inference unsupported or prompt does not fit",
        }
        if isinstance(loaded.device, torch.device) and getattr(
            loaded.model.incremental_cache_capability, "supported", False
        ):
            ids = loaded.tokenizer.encode(PROMPTS[0][1], add_special_tokens=False).ids
            if ids and len(ids) <= config.model.max_seq_len:
                try:
                    loaded.model.eval()
                    with torch.inference_mode():
                        _, state = loaded.model.forward_cached(
                            torch.tensor([ids], device=loaded.device),
                            cache_capacity=len(ids),
                        )
                    cache = {
                        "status": "OBSERVED",
                        "allocated_bytes": state.allocated_bytes,
                        "mode": state.mode,
                        "capacity": state.capacity,
                        "storage": "expanded projected K/V; not compressed-cache savings",
                    }
                except Exception as error:  # noqa: BLE001
                    cache["reason"] = f"{type(error).__name__}: {error}"[:512]
        result["mla_cache"] = cache
    return result


def _mechanisms(
    config: dict[str, Any], metrics: list[dict[str, object]]
) -> dict[str, object]:
    model = config["model"]
    kinds = {
        "engram": model.get("memory") != "none",
        "moe": model.get("ffn") == "moe",
        "sparse_attention": config["attention"].get("kind") == "block_sparse",
        "mla": config["attention"].get("kind") == "mla",
    }
    names = {
        "engram": (
            "engram/lookup_count",
            "engram/gate_mean",
            "engram/bucket_reuse_rate",
        ),
        "moe": ("router_entropy", "maximum_expert_fraction", "mean_topk_probability"),
        "sparse_attention": ("available_tokens", "selected_tokens", "selection_ratio"),
    }
    result: dict[str, object] = {}
    for kind, configured in kinds.items():
        if not configured:
            result[kind] = {"status": "NOT_APPLICABLE", "measurements": []}
            continue
        matched = [
            row
            for row in metrics
            if kind in names
            and (
                row["name"] in names[kind]
                if kind == "engram"
                else row["name"].startswith("moe/layer_")
                and row["name"].rsplit("/", 1)[-1] in names[kind]
                if kind == "moe"
                else row["name"].startswith("attention/layer_")
                and row["name"].rsplit("/", 1)[-1] in names[kind]
            )
        ]
        present = {
            row["name"] if kind == "engram" else row["name"].rsplit("/", 1)[-1]
            for row in matched
        }
        missing = sorted(set(names.get(kind, ())) - present)
        result[kind] = {
            "status": "UNKNOWN" if missing or kind == "mla" else "OBSERVED",
            "measurements": matched[-48:],
            "missing_measures": missing,
            "summary": {
                name: next((r for r in reversed(matched) if r["name"] == name), None)
                for name in dict.fromkeys(r["name"] for r in matched)
            },
            "interpretation": "usage telemetry, not causal dependence",
        }
    return result


def _parent(
    run: Path, manifest: dict[str, Any], current: dict[str, Any], runs_dir: Path
) -> dict[str, object]:
    parent_id = manifest.get("parent_run_id")
    if not isinstance(parent_id, str):
        return {"status": "UNKNOWN", "reason": "no declared parent_run_id"}
    try:
        parent_run = _run_path(parent_id, runs_dir)
        parent_manifest = read_manifest(_owned(parent_run, "manifest.json"))
        parent_progress = _json(_owned(parent_run, "progress.json"))
        parent_digest = canonical_digest(parent_manifest)
        rows, final, _, _ = _metrics(runs_dir, parent_id)
        integrity, evidence = _evidence(
            parent_run, parent_manifest, parent_progress, parent_digest, rows
        )
        mismatches: list[str] = []
        if (
            manifest["source_identity"]["sha256"]
            != parent_manifest["source_identity"]["sha256"]
        ):
            mismatches.append("source implementation identity mismatch")

        def artifact(m: dict[str, Any], key: str) -> str | None:
            return next(
                (a["sha256"] for a in m["artifacts"] if a["relative_path"] == key),
                None,
            )

        for key in (
            "data/validation.npy",
            "tokenizer.json",
            "data/validation_supervision.npy",
        ):
            if artifact(manifest, key) != artifact(parent_manifest, key):
                mismatches.append(f"{key} identity mismatch")
        for key in ("micro_batch_size", "seq_len"):
            if (
                manifest["effective_config"]["training"][key]
                != parent_manifest["effective_config"]["training"][key]
            ):
                mismatches.append(f"validation {key} mismatch")
        if (
            manifest["effective_config"]["evaluation"]["max_batches"]
            != parent_manifest["effective_config"]["evaluation"]["max_batches"]
        ):
            mismatches.append("validation max_batches mismatch")
        if current.get("expected_validation_targets") != evidence.get(
            "expected_validation_targets"
        ):
            mismatches.append("targets per evaluation mismatch")
        if integrity["status"] != "PASS":
            mismatches.append(f"parent checkpoint integrity {integrity['status']}")
        if manifest.get("checkpoint_sha256") != integrity.get(
            "generation_manifest_sha256"
        ):
            mismatches.append(
                "declared parent checkpoint differs from verified parent endpoint"
            )
        parent_report = next(
            (
                r
                for r in evidence["reports"]
                if r["checkpoint"] == integrity.get("generation")
                and r["verification_scope"] == "checkpoint_weights_verified"
            ),
            None,
        )
        own_report = next(
            (
                r
                for r in current["reports"]
                if r["verification_scope"] == "checkpoint_weights_verified"
            ),
            None,
        )
        if parent_report is None or own_report is None:
            mismatches.append("verified endpoint held-out report missing")
        if mismatches:
            return {
                "status": "UNKNOWN",
                "parent_run_id": parent_id,
                "reason": "; ".join(mismatches),
            }
        parent_triage = read_triage(parent_id, runs_dir)
        parent_panel = (
            parent_triage.get("tier1", {}).get("greedy", []) if parent_triage else []
        )
        return {
            "status": "COMPATIBLE",
            "parent_run_id": parent_id,
            "parent_loss": parent_report["loss"],
            "current_loss": own_report["loss"],
            "loss_decrease": parent_report["loss"] - own_report["loss"],
            "parent_generation": integrity["generation"],
            "parent_panel": parent_panel,
            "parent_optimizer_seconds": final.get("phases", {})
            .get("optimizer_update", {})
            .get("seconds"),
            "parent_parameters": parameter_inventory(
                RunConfig.model_validate(parent_manifest["effective_config"])
            ).total,
            "parent_memory": {
                r["name"]: r["value"] for r in rows if r["name"].startswith("memory/")
            },
        }
    except (ValueError, OSError, KeyError, TypeError, sqlite3.Error) as error:
        return {
            "status": "UNKNOWN",
            "parent_run_id": parent_id,
            "reason": str(error)[:512],
        }


def _panel(parent: dict[str, object], tier1: dict[str, object]) -> dict[str, object]:
    old = {
        case["id"]: case
        for case in parent.get("parent_panel", [])
        if isinstance(case, dict) and case.get("status") == "OBSERVED"
    }
    new = {
        case["id"]: case
        for case in tier1.get("greedy", [])
        if isinstance(case, dict) and case.get("status") == "OBSERVED"
    }
    if parent.get("status") != "COMPATIBLE" or any(
        key not in old or key not in new for key, _ in PROMPTS
    ):
        return {
            "status": "UNKNOWN",
            "reason": "verified compatible matched 32-token greedy cases unavailable",
        }
    cases = {}
    for key, _ in PROMPTS:
        a, b = old[key]["mechanics"], new[key]["mechanics"]
        before = bool(a["empty"] or a["special_token_ids"])
        after = bool(b["empty"] or b["special_token_ids"])
        delta = b["repeated_trigram_excess"] - a["repeated_trigram_excess"]
        label = (
            "improved"
            if before and not after
            else "worse"
            if after and not before
            else "worse"
            if delta >= 2
            else "improved"
            if delta <= -2 and not after
            else "unchanged"
        )
        cases[key] = {
            "label": label,
            "trigram_excess_delta": delta,
            "parent_invalid": before,
            "current_invalid": after,
        }
    labels = {c["label"] for c in cases.values()}
    status = (
        "MIXED"
        if {"improved", "worse"} <= labels
        else "REGRESSED"
        if "worse" in labels
        else "IMPROVED"
        if "improved" in labels
        else "FLAT"
    )
    return {
        "status": status,
        "cases": cases,
        "interpretation": "mechanical changes, not subjective text quality",
    }


def _matrix(manifest: dict[str, Any]) -> dict[str, Any] | None:
    for decision in manifest.get("resource_decisions", []):
        if isinstance(decision, dict) and decision.get("kind") == "worker_dispatch":
            matrix = decision.get("matrix")
            if isinstance(matrix, dict):
                return matrix
    return None


def _seed_comparison(
    run: Path,
    manifest: dict[str, Any],
    core: dict[str, Any],
    tier1: dict[str, Any],
    runs_dir: Path,
) -> dict[str, object]:
    matrix = _matrix(manifest)
    coordinate = matrix.get("coordinate") if matrix else None
    if not isinstance(coordinate, dict) or "seed" not in coordinate:
        return {
            "fired": None,
            "comparisons": [],
            "reason": "no declared sealed matrix seed coordinate",
        }
    entries = list(runs_dir.iterdir())
    if len(entries) > 256:
        return {
            "fired": None,
            "comparisons": [],
            "reason": "matrix sibling inventory exceeds bounded 256-run scan",
        }
    comparisons: list[dict[str, object]] = []
    current_loss = next(
        (
            r["loss"]
            for r in core["evidence"]["reports"]
            if r["verification_scope"] == "checkpoint_weights_verified"
        ),
        None,
    )
    if current_loss is None:
        return {
            "fired": None,
            "comparisons": [],
            "reason": "verified endpoint loss missing",
        }
    current_panel = {
        r["id"]: r for r in tier1.get("greedy", []) if r.get("status") == "OBSERVED"
    }
    for sibling in entries:
        if sibling.name == run.name or sibling.is_symlink() or not sibling.is_dir():
            continue
        try:
            sibling_run = _run_path(sibling.name, runs_dir)
            sibling_manifest = read_manifest(_owned(sibling_run, "manifest.json"))
            other = _matrix(sibling_manifest)
            if (
                not other
                or other.get("matrix_sha256") != matrix["matrix_sha256"]
                or not isinstance(other.get("coordinate"), dict)
            ):
                continue
            other_coordinate = other["coordinate"]
            if other_coordinate.get("seed") == coordinate["seed"] or {
                k: v for k, v in other_coordinate.items() if k != "seed"
            } != {k: v for k, v in coordinate.items() if k != "seed"}:
                continue

            def same_except_seed(config: dict[str, Any]) -> dict[str, Any]:
                normalized = json.loads(json.dumps(config))
                normalized.pop("seed", None)
                normalized.pop("logging", None)
                normalized.get("dataset", {}).pop("cache_dir", None)
                normalized.get("tokenizer", {}).pop("path", None)
                return normalized

            if same_except_seed(manifest["effective_config"]) != same_except_seed(
                sibling_manifest["effective_config"]
            ):
                continue
            if any(
                next(
                    (
                        a["sha256"]
                        for a in manifest["artifacts"]
                        if a["relative_path"] == item
                    ),
                    None,
                )
                != next(
                    (
                        a["sha256"]
                        for a in sibling_manifest["artifacts"]
                        if a["relative_path"] == item
                    ),
                    None,
                )
                for item in (
                    "data/validation.npy",
                    "tokenizer.json",
                    "data/validation_supervision.npy",
                )
            ):
                continue
            report = read_triage(sibling.name, runs_dir)
            if report is None or report["core"]["integrity"]["status"] != "PASS":
                continue
            sibling_evidence = report["core"]["evidence"]
            other_loss = next(
                (
                    r["loss"]
                    for r in sibling_evidence["reports"]
                    if r["verification_scope"] == "checkpoint_weights_verified"
                ),
                None,
            )
            if other_loss is None or core["evidence"][
                "expected_validation_targets"
            ] != sibling_evidence.get("expected_validation_targets"):
                continue
            sibling_panel = {
                r["id"]: r
                for r in report["tier1"].get("greedy", [])
                if r.get("status") == "OBSERVED"
            }
            differences = [
                key
                for key in current_panel.keys() & sibling_panel.keys()
                if abs(
                    current_panel[key]["mechanics"]["repeated_trigram_excess"]
                    - sibling_panel[key]["mechanics"]["repeated_trigram_excess"]
                )
                >= 3
            ]
            comparisons.append(
                {
                    "run_id": sibling.name,
                    "seed_coordinate": other_coordinate["seed"],
                    "loss_difference": abs(current_loss - other_loss),
                    "mechanical_cases_difference_at_least_3": sorted(differences),
                    "fired": abs(current_loss - other_loss) >= 0.05
                    or len(differences) >= 2,
                }
            )
        except OSError, ValueError, TypeError, KeyError, IndexError:
            continue
    return {
        "fired": any(r["fired"] for r in comparisons) if comparisons else None,
        "comparisons": comparisons,
        "reason": None
        if comparisons
        else "no completed sealed same-settings sibling with verified comparison protocol",
    }


def _advice(
    core: dict[str, Any],
    tier1: dict[str, Any],
    mechanisms: dict[str, Any],
    parent: dict[str, Any],
    panel: dict[str, Any],
    seed: dict[str, object],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, object]]]:
    triggers: dict[str, Any] = {}
    specs = {
        "LOSS_BEHAVIOR_DIVERGENCE": ("independent behavior/decoding suite", "medium"),
        "DECODER_SENSITIVITY_DETECTED": (
            "new preregistered independent decoder/behavior study",
            "medium",
        ),
        "ENDPOINT_STILL_LEARNING": ("bounded independent behavior check", "low"),
        "POSSIBLE_PLATEAU": ("capacity/data/task hypothesis review", "negligible"),
        "SCALE_WITHOUT_BEHAVIOR_GAIN": ("bounded behavior/decoding probe", "low"),
        "SEED_DISAGREEMENT": ("inspect retained sibling variance", "negligible"),
        "MECHANISM_PRESENT_BUT_USAGE_UNKNOWN": (
            "smallest mechanism-specific telemetry probe",
            "low",
        ),
    }

    def put(
        code: str,
        fired: bool | None,
        operands: dict[str, object],
        threshold: object,
        missing: str | None = None,
    ) -> None:
        triggers[code] = {
            "fired": fired,
            "operands": operands,
            "threshold": threshold,
            "missing_data_reason": missing,
        }

    if core["integrity"]["status"] != "PASS":
        reason = "verified final checkpoint unavailable; post-hoc advice suppressed"
        for code in _TRIGGERS:
            put(code, None, {}, None, reason)
        return (
            triggers,
            {
                code: {
                    "status": "UNKNOWN",
                    "trigger": code,
                    "reason": reason,
                    "estimated_cost_class": specs[code][1],
                    "requires_training": False,
                    "evidence_references": ["core.integrity"],
                }
                for code in _TRIGGERS
            },
            [],
        )
    valid = parent.get("status") == "COMPATIBLE" and panel.get("status") != "UNKNOWN"
    decrease = parent.get("loss_decrease")
    put(
        "LOSS_BEHAVIOR_DIVERGENCE",
        decrease >= 0.05 and panel["status"] in {"MIXED", "REGRESSED", "FLAT"}
        if valid
        else None,
        {"verified_loss_decrease": decrease, "mechanical_panel": panel["status"]},
        0.05,
        None
        if valid
        else "compatible verified loss and matched behavior panel required",
    )
    sampled = [
        row for row in tier1.get("sampled", []) if row.get("status") == "OBSERVED"
    ]
    sensitive = [row for row in sampled if abs(row["trigram_excess_difference"]) >= 3]
    complete = len(sampled) == 8
    put(
        "DECODER_SENSITIVITY_DETECTED",
        bool(sensitive) if complete else None,
        {
            "sampled_cases": len(sampled),
            "matching_cases": [
                {
                    "id": r["id"],
                    "settings": r["settings"],
                    "difference": r["trigram_excess_difference"],
                }
                for r in sensitive
            ],
        },
        {"minimum_repeated_trigram_excess_difference": 3},
        None if complete else "matched sampled/greedy 24-token cases unavailable",
    )
    trend = core["loss"]["trend"]
    verified_points = core["loss"]["points"]
    verified_history = (
        core["integrity"]["status"] == "PASS"
        and len(verified_points) >= 4
        and all(p["evidence"] == "checkpoint_weights_verified" for p in verified_points)
    )
    put(
        "ENDPOINT_STILL_LEARNING",
        trend == "still_improving" if verified_history else None,
        {
            "trend": trend,
            "verified_comparable_points": sum(
                p["evidence"] == "checkpoint_weights_verified" for p in verified_points
            ),
        },
        "still_improving",
        None
        if verified_history
        else "four comparable weight-verified held-out points required; recorded-only trajectory is diagnostic",
    )
    put(
        "POSSIBLE_PLATEAU",
        trend == "plateau_possible" if verified_history else None,
        {"trend": trend},
        "plateau_possible",
        None
        if verified_history
        else "four comparable weight-verified held-out points required; recorded-only trajectory is diagnostic",
    )
    params = tier1["parameter_inventory"]["total"]
    current_update = (
        core["runtime"].get("phases", {}).get("optimizer_update", {}).get("seconds")
    )
    parent_update = parent.get("parent_optimizer_seconds")
    scale_ready = (
        valid
        and _finite(current_update)
        and _finite(parent_update)
        and parent_update > 0
    )
    put(
        "SCALE_WITHOUT_BEHAVIOR_GAIN",
        bool(
            params > parent["parent_parameters"]
            and current_update >= parent_update * 1.2
            and decrease >= 0.05
            and panel["status"] in {"MIXED", "FLAT"}
        )
        if scale_ready
        else None,
        {
            "parameters": params,
            "parent_parameters": parent.get("parent_parameters"),
            "optimizer_seconds": current_update,
            "parent_optimizer_seconds": parent_update,
            "verified_loss_decrease": decrease,
            "mechanical_panel": panel["status"],
        },
        {"optimizer_increase_fraction": 0.2, "loss_decrease": 0.05},
        None
        if scale_ready
        else "compatible parent and measured optimizer time and matched panel required",
    )
    put(
        "SEED_DISAGREEMENT",
        seed["fired"],
        {"sibling_seed_coordinates": seed["comparisons"]},
        {"loss_difference": 0.05, "trigram_difference": 3, "minimum_cases": 2},
        seed["reason"],
    )
    missing = sorted(
        name for name, value in mechanisms.items() if value["status"] == "UNKNOWN"
    )
    put(
        "MECHANISM_PRESENT_BUT_USAGE_UNKNOWN",
        bool(missing),
        {"mechanisms": missing},
        "configured mechanism missing applicable usage telemetry",
    )
    diagnostics: dict[str, Any] = {}
    recommendations: list[dict[str, object]] = []
    for code in _TRIGGERS:
        fired = triggers[code]["fired"]
        action, cost = specs[code]
        if code == "ENDPOINT_STILL_LEARNING" and fired:
            action = "bounded independent behavior check; new tokens NOT_NEEDED for unanswered behavior question"
        diagnostic = {
            "status": "RECOMMENDED"
            if fired
            else "UNKNOWN"
            if fired is None
            else "NOT_NEEDED",
            "trigger": code,
            "reason": triggers[code]["missing_data_reason"]
            if fired is None
            else action
            if fired
            else "threshold not met",
            "estimated_cost_class": cost,
            "requires_training": False,
            "evidence_references": ["core", "tier1", "triggers." + code],
        }
        diagnostics[code] = diagnostic
        if fired:
            recommendations.append(
                {
                    "trigger": code,
                    "action": action,
                    "estimated_cost_class": cost,
                    "requires_training": False,
                    "evidence_references": diagnostic["evidence_references"],
                }
            )
    recommendations.sort(
        key=lambda item: (_COST[item["estimated_cost_class"]], item["trigger"])
    )
    if any(
        triggers[c]["fired"]
        for c in (
            "LOSS_BEHAVIOR_DIVERGENCE",
            "DECODER_SENSITIVITY_DETECTED",
            "ENDPOINT_STILL_LEARNING",
            "SCALE_WITHOUT_BEHAVIOR_GAIN",
        )
    ):
        recommendations.append(
            {
                "trigger": "TRAINING_POLICY",
                "action": "NOT RECOMMENDED YET: resolve cheaper independent behavior question before new training",
                "estimated_cost_class": "high",
                "requires_training": True,
                "evidence_references": ["triggers"],
            }
        )
    return triggers, diagnostics, recommendations


def _surfaces(
    core: dict[str, Any], tier1: dict[str, Any], mechanisms: dict[str, Any]
) -> dict[str, Any]:
    verified_loss = any(
        r["verification_scope"] == "checkpoint_weights_verified"
        for r in core["evidence"]["reports"]
    )
    greedy = tier1.get("greedy", [])
    return {
        "held_out_objective": {
            "applicability": "APPLICABLE",
            "observation": "OBSERVED"
            if verified_loss
            else "INVALID"
            if core["evidence"]["rejected_reports"]
            else "NOT_TESTED",
        },
        "frozen_mechanical_panel": {
            "applicability": "APPLICABLE",
            "observation": "OBSERVED"
            if len(greedy) == 6 and all(r["status"] == "OBSERVED" for r in greedy)
            else "NOT_TESTED",
        },
        "declared_task_capability_cards": {
            "applicability": "UNKNOWN",
            "observation": "NOT_TESTED",
        },
        "independent_behavior_text": {
            "applicability": "UNKNOWN",
            "observation": "NOT_TESTED",
        },
        "mechanism_usage": {
            "applicability": "APPLICABLE"
            if any(v["status"] != "NOT_APPLICABLE" for v in mechanisms.values())
            else "NOT_APPLICABLE",
            "observation": "OBSERVED"
            if any(v["status"] == "OBSERVED" for v in mechanisms.values())
            else "NOT_TESTED",
        },
        "mechanism_causal_dependence": {
            "applicability": "UNKNOWN",
            "observation": "NOT_TESTED",
        },
        "human_review": {"applicability": "UNKNOWN", "observation": "NOT_TESTED"},
    }


def _card_surface(
    run: Path, core: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, object]:
    try:
        folder = _owned(run, "evaluations", directory=True)
    except ValueError as error:
        return {
            "applicability": "UNKNOWN",
            "observation": "INVALID",
            "reason": str(error),
        }
    if not folder.exists():
        return {"applicability": "UNKNOWN", "observation": "NOT_TESTED"}
    cards = sorted(
        p
        for p in folder.iterdir()
        if p.name.startswith("capability-") and p.suffix == ".json"
    )
    if not cards:
        return {"applicability": "UNKNOWN", "observation": "NOT_TESTED"}
    if len(cards) > 32:
        return {
            "applicability": "UNKNOWN",
            "observation": "INVALID",
            "reason": "too many retained cards to inspect",
        }
    endpoint = core["integrity"]
    artifacts = {
        item["relative_path"]: item["sha256"] for item in manifest["artifacts"]
    }
    observed: list[str] = []
    rejected: list[str] = []
    for path in cards:
        try:
            _owned(run, f"evaluations/{path.name}")
            payload = _json(path)
            digest = payload.pop("result_digest", None)
            identity = payload.get("identity")
            if (
                not isinstance(digest, str)
                or canonical_digest(payload) != digest
                or not isinstance(identity, dict)
                or identity.get("run_id") != manifest["run_id"]
                or not isinstance(identity.get("checkpoint_sha256"), str)
                or identity.get("source_identity_sha256")
                != manifest["source_identity"]["sha256"]
                or identity.get("tokenizer_sha256") != artifacts.get("tokenizer.json")
                or not isinstance(identity.get("data_sha256"), dict)
                or identity["data_sha256"].get("validation")
                != artifacts.get("data/validation.npy")
                or payload.get("valid") is not True
            ):
                rejected.append(path.name)
            elif endpoint.get("status") == "PASS" and identity[
                "checkpoint_sha256"
            ] == endpoint.get("generation_manifest_sha256"):
                observed.append(path.name)
        except ValueError, OSError, TypeError, KeyError:
            rejected.append(path.name)
    return {
        "applicability": "APPLICABLE" if observed else "UNKNOWN",
        "observation": "INVALID"
        if rejected
        else "OBSERVED"
        if observed
        else "NOT_TESTED",
        "verified": observed,
        "rejected": rejected,
    }


def _triage_choice(value: object, choices: Collection[str]) -> bool:
    """Reject malformed JSON scalars before membership can raise an incidental TypeError."""
    return isinstance(value, str) and value in choices


def _triage_manifest(path: Path) -> dict[str, Any]:
    """Reject serialized shape errors locally, then retain native identity verification.

    Other manifest callers keep their TypeError contract. No reader exceptions
    are caught here: an unexpected TypeError/KeyError in native verification
    still propagates after these explicit artifact shape checks.
    """
    data = _json(path)
    if data.get("identity_version") == IDENTITY_VERSION:
        requested, effective = (
            data.get("requested_config"),
            data.get("effective_config"),
        )
        if not isinstance(requested, dict) or not isinstance(effective, dict):
            raise ValueError("post-train triage manifest requires config mappings")
        model = effective.get("model", effective)
        if not isinstance(model, dict) or not isinstance(
            effective.get("attention", {}), dict
        ):
            raise ValueError(
                "post-train triage manifest requires model/attention mappings"
            )
        if not isinstance(model.get("memory", "none"), str):
            raise ValueError("post-train triage manifest requires a memory kind string")
        if not isinstance(data.get("artifacts"), list):
            raise ValueError("post-train triage manifest artifacts must be an array")
    return read_manifest(path)


def read_triage(run_id: str, runs_dir: Path) -> dict[str, object] | None:
    """Read one verified run-local immutable report; never choose between conflicts."""
    run = _run_path(run_id, runs_dir)
    folder = _owned(run, "post-train-triage", directory=True)
    if not folder.exists():
        return None
    paths = list(folder.iterdir())
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError("ambiguous post-train triage artifacts")
    path = _owned(run, f"post-train-triage/{paths[0].name}")
    if (
        not _DIGEST.fullmatch(path.stem)
        or path.suffix != ".json"
        or not path.is_file()
        or path.stat().st_size > MAX_BYTES
    ):
        raise ValueError("invalid post-train triage artifact")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != path.stem:
        raise ValueError("post-train triage filename hash mismatch")
    payload = _json(path)
    identity = payload.pop("identity", None)
    if not isinstance(identity, dict) or identity.get("sha256") != canonical_digest(
        payload
    ):
        raise ValueError("post-train triage embedded identity mismatch")
    payload["identity"] = identity
    if payload.get("format") != FORMAT or identity.get("run_id") != run_id:
        raise ValueError("post-train triage schema/run mismatch")
    for key in ("inputs", "core", "tier1", "surfaces", "triggers", "diagnostics"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"post-train triage missing {key}")  # noqa: TRY004 - serialized artifact
    if not isinstance(payload.get("recommendations"), list) or not isinstance(
        payload.get("known_unknowns"), list
    ):
        raise ValueError("post-train triage malformed recommendations/unknowns")  # noqa: TRY004 - serialized artifact
    core = payload["core"]
    endpoint = core.get("integrity")
    if not isinstance(endpoint, dict) or not _triage_choice(
        endpoint.get("status"),
        {
            "PASS",
            "FAIL",
            "UNKNOWN",
        },
    ):
        raise ValueError("post-train triage malformed integrity status")
    loss = core.get("loss")
    if (
        not isinstance(loss, dict)
        or not _triage_choice(
            loss.get("trend"),
            {
                "insufficient_history",
                "regressing",
                "still_improving",
                "plateau_possible",
                "unclear",
            },
        )
        or not isinstance(loss.get("points"), list)
    ):
        raise ValueError("post-train triage malformed loss trajectory")
    tier1 = payload["tier1"]
    if (
        not _triage_choice(tier1.get("status"), {"OBSERVED", "AVAILABLE", "UNKNOWN"})
        or not isinstance(tier1.get("greedy"), list)
        or not isinstance(tier1.get("sampled"), list)
    ):
        raise ValueError("post-train triage malformed Tier 1")
    if set(payload["triggers"]) != set(_TRIGGERS):
        raise ValueError("post-train triage missing trigger")
    for code in _TRIGGERS:
        trigger = payload["triggers"][code]
        if (
            not isinstance(trigger, dict)
            or type(trigger.get("fired")) not in {bool, type(None)}
            or not isinstance(trigger.get("operands"), dict)
            or "threshold" not in trigger
            or "missing_data_reason" not in trigger
        ):
            raise ValueError(f"post-train triage malformed trigger: {code}")
        diagnostic = payload["diagnostics"].get(code)
        if (
            not isinstance(diagnostic, dict)
            or not _triage_choice(
                diagnostic.get("status"),
                {
                    "NOT_APPLICABLE",
                    "NOT_NEEDED",
                    "AVAILABLE",
                    "RECOMMENDED",
                    "OBSERVED",
                    "UNKNOWN",
                },
            )
            or not _triage_choice(diagnostic.get("estimated_cost_class"), _COST)
            or type(diagnostic.get("requires_training")) is not bool
        ):
            raise ValueError(f"post-train triage malformed diagnostic: {code}")
    for name, surface in payload["surfaces"].items():
        if (
            not isinstance(surface, dict)
            or not _triage_choice(
                surface.get("applicability"),
                {"APPLICABLE", "NOT_APPLICABLE", "UNKNOWN"},
            )
            or not _triage_choice(
                surface.get("observation"), {"OBSERVED", "NOT_TESTED", "INVALID"}
            )
        ):
            raise ValueError(f"post-train triage malformed surface: {name}")
    for item in payload["recommendations"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("action"), str)
            or not isinstance(item.get("trigger"), str)
            or not _triage_choice(item.get("estimated_cost_class"), _COST)
            or type(item.get("requires_training")) is not bool
        ):
            raise ValueError("post-train triage malformed recommendation")
    manifest = _owned(run, "manifest.json")
    if identity.get("manifest_sha256") != canonical_digest(_triage_manifest(manifest)):
        raise ValueError("post-train triage manifest binding mismatch")
    progress = _json(_owned(run, "progress.json"))
    if (
        progress.get("status") != "completed"
        or progress.get("manifest_sha256") != identity["manifest_sha256"]
    ):
        raise ValueError("post-train triage no longer binds completed run progress")
    endpoint = core["integrity"]
    if endpoint.get("status") == "PASS":
        name = endpoint.get("generation")
        if not isinstance(name, str) or not _GENERATION.fullmatch(name):
            raise ValueError("invalid triage generation binding")
        checkpoint = _checkpoint_metadata(run, name, identity["manifest_sha256"])
        latest = progress.get("latest")
        if (
            checkpoint is None
            or type(checkpoint["step"]) is not int
            or checkpoint["step"] < 0
            or type(checkpoint["tokens_seen"]) is not int
            or checkpoint["tokens_seen"] < 0
            or type(progress.get("step")) is not int
            or progress["step"] < 0
            or type(progress.get("tokens_seen")) is not int
            or progress["tokens_seen"] < 0
            or checkpoint["digest"] != endpoint.get("generation_manifest_sha256")
            or endpoint.get("generation_manifest_sha256")
            != identity.get("generation_manifest_sha256")
            or not isinstance(latest, dict)
            or latest.get("relative_path") != name
            or latest.get("manifest_sha256") != checkpoint["digest"]
            or (checkpoint["step"], checkpoint["tokens_seen"])
            != (progress.get("step"), progress.get("tokens_seen"))
        ):
            raise ValueError("post-train triage checkpoint/progress binding mismatch")
    return payload


def triage_completed_run(
    run_id: str,
    runs_dir: Path,
    *,
    remaining_seconds: float | None = None,
    authorization: RuntimeAuthorization | None = None,
) -> Path:
    """Interpret only completed run evidence and persist one bounded immutable report."""
    run = _run_path(run_id, runs_dir)
    previous = read_triage(run_id, runs_dir)
    if previous is not None:
        return next(_owned(run, "post-train-triage", directory=True).iterdir())
    progress = _json(_owned(run, "progress.json"))
    if progress.get("status") != "completed":
        raise ValueError("post-train triage requires completed progress status")
    manifest_path = _owned(run, "manifest.json")
    manifest = read_manifest(manifest_path)
    digest = canonical_digest(manifest)
    if manifest.get("run_id") != run_id or progress.get("manifest_sha256") != digest:
        raise ValueError("run/manifest identity mismatch")
    config = RunConfig.model_validate(manifest["effective_config"])
    rows, final, recorded_metadata, metrics_error = _metrics(runs_dir, run_id)
    integrity, evidence = _evidence(run, manifest, progress, digest, rows)
    loss = loss_trajectory(evidence["points"])
    checkpoint_losses = [
        {
            "step": r["step"],
            "targets": r["targets"],
            "loss": r["loss"],
            "delta_from_initial": r["loss"] - loss["initial_loss"]
            if loss["initial_loss"] is not None
            else None,
            "verification_scope": r["verification_scope"],
        }
        for r in evidence["reports"]
    ]
    memory: dict[str, dict[str, object]] = {}
    for row in rows:
        if row["name"].startswith("memory/"):
            name = str(row["name"])
            if (
                "peak" not in name
                or name not in memory
                or row["value"] > memory[name]["value"]
            ):
                memory[name] = {
                    "value": row["value"],
                    "step": row["step"],
                    "targets": row["targets"],
                }
    mechanisms = _mechanisms(manifest["effective_config"], rows)
    core = {
        "integrity": integrity,
        "loss": loss,
        "evidence": evidence,
        "milestone_losses": checkpoint_losses,
        "runtime": final,
        "parameter_inventory": {
            "recorded": recorded_metadata.get("inspection"),
            "configured": asdict(parameter_inventory(config)),
        },
        "metric_units": {
            "held_out_loss": "nats_per_target",
            "targets": "target_exposures",
            "optimizer_update": "seconds",
        },
        "memory": {
            "observations": memory,
            "method": "native"
            if "memory/device_peak_allocated_bytes" in memory
            else "sampled_lower_bound"
            if memory
            else "unavailable",
            "unavailable_fields": [
                key
                for key in (
                    "memory/device_peak_allocated_bytes",
                    "memory/device_peak_reserved_bytes",
                )
                if key not in memory
            ],
        },
        "configured_targets": config.training.max_tokens,
        "actual_targets": progress["tokens_seen"],
        "actual_steps": progress["step"],
        "configured_steps": config.training.max_steps,
        "cumulative_optimizer_seconds": progress.get("cumulative_update_seconds"),
        "total_wall_seconds": progress.get("cumulative_wall_seconds"),
        "mechanisms": mechanisms,
    }
    tier1 = (
        _probe(run_id, runs_dir, integrity, config, remaining_seconds, authorization)
        if integrity["status"] == "PASS"
        else {
            "status": "UNKNOWN",
            "reason": "invalid or unavailable endpoint checkpoint",
            "greedy": [],
            "sampled": [],
            "parameter_inventory": asdict(parameter_inventory(config)),
        }
    )
    if tier1.get("mla_cache", {}).get("status") == "OBSERVED":
        mechanisms["mla"] = {
            "status": "OBSERVED",
            "measurements": [tier1["mla_cache"]],
            "interpretation": "actual expanded projected inference K/V allocation, not savings",
        }
    parent = (
        _parent(run, manifest, evidence, runs_dir)
        if integrity["status"] == "PASS"
        else {"status": "UNKNOWN", "reason": "invalid endpoint"}
    )
    core["parent_comparison"] = {
        k: v for k, v in parent.items() if k not in {"parent_panel", "parent_memory"}
    }
    if parent.get("status") == "COMPATIBLE":
        core["parent_comparison"]["optimizer_seconds_difference"] = (
            final.get("phases", {}).get("optimizer_update", {}).get("seconds")
            - parent["parent_optimizer_seconds"]
            if _finite(
                final.get("phases", {}).get("optimizer_update", {}).get("seconds")
            )
            and _finite(parent["parent_optimizer_seconds"])
            else None
        )
        core["parent_comparison"]["memory_comparison"] = {
            name: {"current": row["value"], "parent": parent["parent_memory"][name]}
            for name, row in memory.items()
            if name in parent["parent_memory"]
            and name
            in {
                "memory/device_peak_allocated_bytes",
                "memory/device_peak_reserved_bytes",
            }
        }
    panel = _panel(parent, tier1)
    core["parent_comparison"]["mechanical_panel"] = panel
    seed = (
        _seed_comparison(run, manifest, core, tier1, runs_dir)
        if integrity["status"] == "PASS"
        else {"fired": None, "comparisons": [], "reason": "invalid endpoint"}
    )
    triggers, diagnostics, recommendations = _advice(
        core, tier1, mechanisms, parent, panel, seed
    )
    diagnostics["INFERENCE_PANEL"] = {
        "status": tier1["status"],
        "trigger": None,
        "reason": tier1.get("reason") or "frozen bounded mechanical prompts",
        "estimated_cost_class": "high"
        if tier1.get("reason") == "cost_ceiling"
        else "low",
        "requires_training": False,
        "evidence_references": ["core.integrity", "tier1.greedy", "tier1.sampled"],
    }
    diagnostics["MLA_CACHE"] = {
        "status": "NOT_APPLICABLE"
        if config.attention.kind != "mla"
        else "OBSERVED"
        if tier1.get("mla_cache", {}).get("status") == "OBSERVED"
        else "UNKNOWN"
        if tier1["status"] == "UNKNOWN"
        else "AVAILABLE",
        "trigger": "MECHANISM_PRESENT_BUT_USAGE_UNKNOWN",
        "reason": "actual projected K/V storage only; never compressed-cache savings",
        "estimated_cost_class": "low",
        "requires_training": False,
        "evidence_references": ["tier1.mla_cache", "core.mechanisms.mla"],
    }
    artifacts = {
        item["relative_path"]: item["sha256"] for item in manifest["artifacts"]
    }
    inputs = {
        "manifest_sha256": digest,
        "generation_manifest_sha256": integrity.get("generation_manifest_sha256"),
        "source_identity_sha256": manifest["source_identity"]["sha256"],
        "effective_config_sha256": manifest["effective_config_sha256"],
        "validation_data_sha256": artifacts.get("data/validation.npy"),
        "tokenizer_sha256": artifacts.get("tokenizer.json"),
        "report_sha256": {r["path"]: r["sha256"] for r in evidence["reports"]},
        "telemetry_sha256": canonical_digest({"metrics": rows, "runtime": final}),
        "policy": {
            "tier1_parameter_ceiling": 100_000_000,
            "cpu_parameter_ceiling": 10_000_000,
            "minimum_remaining_seconds": 120,
            "remaining_seconds": remaining_seconds,
        },
    }
    surfaces = _surfaces(core, tier1, mechanisms)
    surfaces["declared_task_capability_cards"] = _card_surface(run, core, manifest)
    payload: dict[str, Any] = {
        "format": FORMAT,
        "inputs": inputs,
        "core": core,
        "tier1": tier1,
        "surfaces": surfaces,
        "triggers": triggers,
        "diagnostics": diagnostics,
        "recommendations": recommendations,
        "known_unknowns": [
            s
            for s in (
                metrics_error,
                "historical checkpoint weight files not verified",
                "mechanical prompts do not establish subjective text quality",
                "no Tier-2 or Tier-3 experiments automatically dispatched",
            )
            if s
        ],
    }
    payload["identity"] = {
        "sha256": canonical_digest(payload),
        "run_id": run_id,
        "manifest_sha256": digest,
        "generation_manifest_sha256": integrity.get("generation_manifest_sha256"),
    }
    raw = canonical_json(payload) + b"\n"
    if len(raw) > MAX_BYTES:
        raise ValueError("post-train triage exceeds 256 KiB bound")
    folder = _owned(run, "post-train-triage", directory=True)
    folder.mkdir(exist_ok=True)
    path = folder / (hashlib.sha256(raw).hexdigest() + ".json")
    if any(folder.iterdir()):
        raise ValueError("post-train triage artifact appeared concurrently")
    temporary = folder / ("." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as out:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
        if any(p != temporary for p in folder.iterdir()):
            raise ValueError("post-train triage artifact appeared concurrently")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def triage_summary(report: dict[str, object]) -> str:
    """A concise advisory for read-only CLI and worker logs."""
    core = report["core"]
    advice = report["recommendations"]
    choice = next((r for r in advice if r["trigger"] != "TRAINING_POLICY"), None)
    return (
        f"integrity={core['integrity']['status']}; trend={core['loss']['trend']}; "
        f"Tier 1={report['tier1']['status']} ({report['tier1'].get('reason') or 'bounded probe'}); "
        f"next={choice['action'] if choice else 'none'} "
        f"({choice['estimated_cost_class'] if choice else 'none'})"
    )
