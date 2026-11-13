"""Run the probe battery, cheapest probes first, one loaded arm at a time.

Each arm (baseline, candidate) is loaded, measured for the current tier and
released before the other is loaded, so a battery never holds two models in
memory. Measurements are cached per arm (validation is scored once and shared
by held-out loss and calibration; inside ``klab try`` it is the try's own
scoring pass). A :class:`~klab.lab_context.LabContext` is checked before
every probe: a cancel sentinel or a violated resource envelope stops the
battery at that safe point, and an out-of-memory error stops further probe work
instead of becoming an ordinary error row. A stopped battery keeps everything
measured so far and is ``incomplete``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from klab.lab_context import (
    LabCancelled,
    LabContext,
    LabSignal,
    is_out_of_memory,
    release_memory,
)
from klab.lab_records import (
    PROBE_FORMAT,
    comparison_group,
    eval_group,
    write_json_atomic,
)
from klab.probes import metrics, scoring
from klab.probes.suite import (
    BY_ID,
    ProbeSpec,
    fact_items,
    final_verdict_access,
    needle_items,
    ordered,
    parametric_items,
    parametric_manifest_sha256,
    prompts,
    suite_identity,
    tiers_through,
    verdict_split,
)
from klab.probes.verdict import (
    INCOMPLETE_STOPS,
    MISSING_STATUSES,
    decide,
    final_verdict,
    judge,
    numerical_failure,
    overfit_guard,
)

RESULT_FORMAT = PROBE_FORMAT
NEAR_IDENTICAL_JS = 0.01
Progress = Callable[[dict[str, Any]], None]


class ProbeUnsupported(ValueError):
    """The checkpoint cannot run this probe (e.g. a non-PyTorch engine)."""


class _NotComparable(Exception):
    pass


class ProbeInapplicable(Exception):
    """The probe does not apply to this checkpoint (recorded as skipped)."""


@dataclass
class Arm:
    """One side of a comparison: how to load it and what is already measured."""

    load: Callable[[], Any]
    cache: dict[str, Any] = field(default_factory=dict)
    # A pinned public checkpoint (``ref:NAME``): only the probes in
    # REFERENCE_PROBES apply, since its tokenizer and data are not ours.
    reference: bool = False


# Probes that mean the same thing for a public reference model: lm-eval tasks
# are defined by the harness, and the text-level ranking probes (needle
# included: its lengths are in characters) ask the same strings of any
# tokenizer (their ``item_group`` proves it).
REFERENCE_PROBES = frozenset({"lm_eval", "fact_recall", "parametric_recall", "needle"})
TEXT_PROBES = ("fact_recall", "parametric_recall", "needle")


def as_arm(value: Any) -> Arm | None:
    if value is None or isinstance(value, Arm):
        return value
    return Arm(load=lambda: value)


def _require_torch_engine(loaded: Any) -> None:
    if loaded.engine is not None:
        raise ProbeUnsupported("probes need the PyTorch engine (MLX not supported)")
    if getattr(loaded.model, "semantic_memories", None):
        raise ProbeUnsupported("probes do not support attached semantic packs")


def footprint(model: Any, inventory: Mapping[str, Any] | None) -> dict[str, int]:
    """Resident vs. active parameters (and bytes) of a loaded model.

    Resident counts every weight held in memory; active counts what one token
    touches (one embedding row, the selected experts/memory rows) per the
    model's parameter inventory. Shared by probe records and lab try arms.
    """
    parameters = list(model.parameters())
    total = sum(p.numel() for p in parameters)
    size = sum(p.numel() * p.element_size() for p in parameters)
    active = int((inventory or {}).get("active_per_token") or total)
    active = min(active, total)
    return {
        "parameters": total,
        "parameter_bytes": size,
        "active_parameters": active,
        "active_parameter_bytes": round(size * active / total) if total else 0,
    }


def arm_identity(loaded: Any) -> dict[str, Any]:
    """Checkpoint identity and footprint of one loaded arm (probe and bench records)."""
    identity = loaded.identity
    return {
        "run_id": identity.get("run_id"),
        "run_dir": str(loaded.run),
        "checkpoint": identity.get("checkpoint_relative_path"),
        "checkpoint_sha256": identity.get("checkpoint_sha256"),
        "step": identity.get("step"),
        "tokens_seen": identity.get("tokens_seen"),
        **footprint(loaded.model, identity.get("parameter_inventory")),
        "tokenizer_sha256": identity.get("tokenizer_sha256"),
        "validation_sha256": (identity.get("data_sha256") or {}).get("validation"),
        "max_seq_len": loaded.config.model.max_seq_len,
        **(
            {"reference": loaded.describe()}
            if getattr(loaded, "reference", None) is not None
            else {}
        ),
    }


def _validation_identity(loaded: Any) -> dict[str, Any]:
    from klab.lab_mode import _data_identity, _supervision_identity

    if getattr(loaded, "reference", None) is not None:
        # Never equal to a lab run's: held-out loss is not comparable.
        return {
            "reference": loaded.identity["checkpoint_relative_path"],
            "tokenizer_sha256": loaded.identity["tokenizer_sha256"],
        }

    data = _data_identity(loaded.run)
    return {
        "validation_sha256": data["validation_sha256"],
        "tokenizer_sha256": data["tokenizer_sha256"],
        **_supervision_identity(loaded.run),
    }


def _tokenizer_digest(loaded: Any) -> str:
    return hashlib.sha256(loaded.tokenizer.to_str().encode()).hexdigest()


def _describe(loaded: Any, arm: Arm) -> None:
    """Identity facts recorded the first time an arm is opened."""
    if "identity" not in arm.cache:
        arm.cache["identity"] = arm_identity(loaded)
        arm.cache["validation_identity"] = _validation_identity(loaded)
        arm.cache["tokenizer_digest"] = _tokenizer_digest(loaded)
        arm.cache["config"] = loaded.config


# --- Per-arm measurements ----------------------------------------------------


def validation(loaded: Any, arm: Arm, protocol: Mapping[str, int]) -> dict[str, Any]:
    """Native held-out scoring with the probe observer, computed once per arm."""
    cached = arm.cache.get("validation")
    if cached is not None and cached.get("protocol") == dict(protocol):
        return cached
    if protocol["seq_len"] > loaded.config.model.max_seq_len:
        raise _NotComparable("baseline eval window exceeds this model's max_seq_len")
    from klab.lab_mode import _with_eval_protocol

    stats = scoring.ValidationStats()
    try:
        native = _with_eval_protocol(loaded, protocol).evaluate(observer=stats)
    except FloatingPointError as error:
        # NaN/inf loss from the native evaluator: a numerical failure, kept
        # as an outcome (not a probe error) so the battery can hard-fail on it.
        arm.cache["validation"] = {
            "numerical_failure": str(error),
            "protocol": dict(protocol),
        }
        return arm.cache["validation"]
    arm.cache["validation"] = {**stats.result(native), "protocol": dict(protocol)}
    return arm.cache["validation"]


def generations(loaded: Any, max_new_tokens: int) -> list[dict[str, Any]]:
    from klab.evaluation.generation import generate_with_token_ids

    budget = max(1, min(max_new_tokens, loaded.config.model.max_seq_len // 2))
    rows = []
    for prompt in prompts(verdict_split()):
        text, ids = generate_with_token_ids(
            loaded.model,
            loaded.tokenizer,
            prompt,
            loaded.config.model.max_seq_len,
            budget,
            loaded.device,
            temperature=0.0,
        )
        rows.append(
            {"prompt": prompt, "continuation": text.removeprefix(prompt), "ids": ids}
        )
    return rows


def recall_items(split: str) -> list[dict[str, Any]]:
    return [
        {
            "prefix": f"{item['context']} {item['question']}",
            "answer": item["answer"],
            "candidates": item["candidates"],
        }
        for item in fact_items(split)
    ]


def parametric_recall_items(
    split: str, *, seed: int, control: bool = False
) -> list[dict]:
    return [
        {
            "prefix": item["question"],
            "answer": item["answer"],
            "candidates": item["candidates"],
        }
        for item in parametric_items(split, seed=seed, control=control)
    ]


def parametric_binding(loaded: Any, *, reference: bool = False) -> dict[str, Any]:
    """Bind closed-book recall to the facts this checkpoint actually trained on.

    The ``withheld_facts`` source trains on ``split_facts(synthetic_seed)``;
    the probe uses the same seed's manifest, read from the run's resolved
    config and cross-checked against its prepared-data provenance
    (``data/manifest.json``). A run (or reference model) not trained on
    ``withheld_facts`` raises :class:`ProbeInapplicable`; a withheld-facts run
    whose provenance is missing or disagrees raises :class:`ProbeUnsupported`
    (missing evidence), so trained facts are never scored as never-trained.
    """
    if reference:
        raise ProbeInapplicable(
            "inapplicable: a pinned public model was not trained on a "
            "withheld-facts manifest"
        )
    dataset = getattr(getattr(loaded, "config", None), "dataset", None)
    source = getattr(dataset, "source", None)
    task = getattr(dataset, "synthetic_task", None)
    if source is None:
        raise ProbeUnsupported(
            "no dataset provenance in the run config: cannot tell which facts "
            "were trained"
        )
    if (source, task) != ("synthetic", "withheld_facts"):
        raise ProbeInapplicable(
            f"inapplicable: the run trained on {source}, not withheld_facts "
            "(no facts manifest to bind)"
        )
    seed = getattr(dataset, "synthetic_seed", None)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ProbeUnsupported("withheld_facts run records no synthetic_seed")
    run = getattr(loaded, "run", None)
    if isinstance(run, Path):
        path = run / "data" / "manifest.json"
        try:
            recorded = json.loads(path.read_text(encoding="utf-8"))
            prepared = recorded["cache_identity"]["dataset"]
        except OSError, ValueError, KeyError, TypeError:
            raise ProbeUnsupported(
                "the run's prepared-data provenance (data/manifest.json) is "
                "missing or unreadable: cannot bind the facts manifest"
            ) from None
        if (
            prepared.get("source"),
            prepared.get("synthetic_task"),
            prepared.get("synthetic_seed"),
        ) != (source, task, seed):
            raise ProbeUnsupported(
                "the run's prepared data does not match its config "
                f"(prepared {prepared.get('source')} seed "
                f"{prepared.get('synthetic_seed')}, config {source} seed {seed})"
            )
    return {"seed": seed, "manifest_sha256": parametric_manifest_sha256(seed)}


def item_group(probe: str, items: list[dict[str, Any]]) -> str:
    """Two ranking results compare only on the same rendered items and scoring."""
    return comparison_group(
        {
            "probe": probe,
            "scoring": scoring.RANKING_PROTOCOL,
            "items": [
                {k: item[k] for k in ("prefix", "answer", "candidates")}
                for item in items
            ],
        }
    )


def _ranked(loaded: Any, items: list[dict[str, Any]]) -> dict[str, Any]:
    scores = scoring.ranking_scores(loaded, items)
    return {
        "credits": [
            scoring.tie_credit(s, item["answer"])
            for s, item in zip(scores, items, strict=True)
        ],
        "picked": [scoring.top_candidate(s) for s in scores],
    }


def _measure(
    spec: ProbeSpec, loaded: Any, arm: Arm, protocol: Mapping[str, int]
) -> dict[str, Any]:
    """Everything one arm contributes to SPEC; judged later against the other."""
    if spec.id in {"heldout_loss", "calibration"}:
        val = validation(loaded, arm, protocol)
        if "numerical_failure" in val:
            if spec.id == "heldout_loss":
                return {
                    "value": math.nan,
                    "numerical_failure": val["numerical_failure"],
                }
            return {
                "status": "error",
                "note": f"numerical failure: {val['numerical_failure']}",
            }
        if spec.id == "heldout_loss":
            return {
                "value": val["loss"],
                "window_sums": val["window_sums"],
                "window_counts": val["window_counts"],
                "valid_targets": val["valid_targets"],
                "windows": val["windows"],
                "top1_accuracy": val["accuracy"],
                "ms_per_token": val["ms_per_token"],
            }
        bins = int(spec.params.get("bins", 15))
        return {
            "value": metrics.expected_calibration_error(
                val["confidences"], val["correct"], bins
            ),
            "reliability": metrics.reliability_bins(
                val["confidences"], val["correct"], bins
            ),
            "positions": int(val["correct"].size),
            "bins": bins,
        }
    if spec.id == "token_agreement":
        rows = []
        for prompt in prompts(verdict_split()):
            ids = scoring.encode(loaded, prompt)[-loaded.config.model.max_seq_len :]
            rows.append(scoring.log_probs(loaded, ids).astype(np.float32))
        return {"log_probs": rows}
    if spec.id == "repetition":
        rows = generations(loaded, int(spec.params.get("max_new_tokens", 32)))
        ids = [r["ids"] for r in rows]
        return {
            "value": metrics.mean_seq_rep_n(ids, 4),
            "seq_rep_4": metrics.mean_seq_rep_n(ids, 4),
            "distinct_1": metrics.distinct_n(ids, 1),
            "distinct_2": metrics.distinct_n(ids, 2),
            "samples": [
                {"prompt": r["prompt"], "continuation": r["continuation"]}
                for r in rows[:2]
            ],
        }
    if spec.id in TEXT_PROBES:

        def items(split: str) -> list[dict[str, Any]]:
            if spec.id == "fact_recall":
                return recall_items(split)
            if spec.id == "parametric_recall":
                return parametric_recall_items(split, seed=binding["seed"])
            return [
                # Token count is this model's view of the item: informational,
                # never part of the item identity.
                {**item, "tokens": len(scoring.encode(loaded, item["prefix"]))}
                for item in needle_items(split, spec.params["char_lengths"])
            ]

        binding = (
            parametric_binding(loaded, reference=arm.reference)
            if spec.id == "parametric_recall"
            else {}
        )
        held = items(verdict_split())
        ranked = _ranked(loaded, held)
        credits = ranked["credits"]
        out = {
            "value": float(np.mean(credits)) if credits else None,
            "credits": credits,
            "picked": ranked["picked"],
            "constant_pick": scoring.constant_pick(held, ranked["picked"]),
            "dev_credits": scoring.ranking_credit(loaded, items("dev")),
            "chars": [i.get("chars") for i in held],
            "tokens": [i.get("tokens") for i in held],
            "chance": 1 / len(held[0]["candidates"]) if held else None,
            "item_group": item_group(spec.id, held),
            "items": [
                {
                    "prompt": i["prefix"] if spec.id != "needle" else None,
                    "answer": i["answer"],
                    "chars": i.get("chars"),
                    "tokens": i.get("tokens"),
                }
                for i in held
            ],
        }
        if spec.id == "parametric_recall":
            control = _ranked(
                loaded,
                parametric_recall_items(
                    verdict_split(), seed=binding["seed"], control=True
                ),
            )
            out["control_accuracy"] = float(np.mean(control["credits"]))
            out["manifest_sha256"] = binding["manifest_sha256"]
            out["manifest_seed"] = binding["seed"]
        return out
    if spec.id == "lm_eval":
        from klab.probes.lm_eval_adapter import LmEvalUnavailable, run_lm_eval

        try:
            out = run_lm_eval(
                loaded,
                spec.params["tasks"],
                spec.params["limit"],
                **(
                    {"protocol": spec.params["protocol"]}
                    if "protocol" in spec.params
                    else {}
                ),
            )
        except LmEvalUnavailable as error:
            raise ProbeUnsupported(str(error)) from None
        return {"value": out["mean_accuracy"], **out}
    raise ValueError(f"unknown probe {spec.id}")


def probe_cache_key(probe_id: str) -> str:
    """Per-arm cache key of a probe measurement, scoped to the verdict split.

    Item-based probes read ``heldout`` or (inside a final verdict) ``final``
    items, so a measurement made under one split is never reused under the
    other. Validation statistics (held-out loss, calibration) are cached
    separately under ``validation``; they do not depend on the item split.
    """
    return f"probe:{verdict_split()}:{probe_id}"


def _measure_safely(
    spec: ProbeSpec, loaded: Any, arm: Arm, protocol: Mapping[str, int]
) -> dict[str, Any]:
    """Probe failures become evidence gaps; OOM and stops propagate."""
    key = probe_cache_key(spec.id)
    if key in arm.cache:
        return arm.cache[key]
    try:
        measured = _measure(spec, loaded, arm, protocol)
    except _NotComparable as error:
        measured = {"status": "not_comparable", "note": str(error)}
    except ProbeUnsupported as error:
        measured = {"status": "unavailable", "note": str(error)}
    except ProbeInapplicable as error:
        measured = {"status": "skipped", "note": str(error)}
    except Exception as error:
        if is_out_of_memory(error):
            raise
        measured = {"status": "error", "note": f"{type(error).__name__}: {error}"}
    arm.cache[key] = measured
    return measured


# --- Judging ------------------------------------------------------------------


def _row(spec: ProbeSpec, **extra: Any) -> dict[str, Any]:
    return {
        "id": spec.id,
        "title": spec.title,
        "tier": spec.tier,
        "cost": spec.cost,
        "metric": spec.metric,
        "higher_is_better": spec.higher_is_better,
        "hard": spec.hard,
        "thresholds": {"mode": spec.mode, "warn": spec.warn, "fail": spec.fail},
        "uncertainty": spec.uncertainty,
        "suggests": spec.suggests,
        "explains": spec.explains,
        "reference": spec.reference,
        "status": "skipped",
        "value": None,
        "baseline_value": None,
        "delta": None,
        "regression": None,
        "delta_se": None,
        "within_noise": None,
        "improved": False,
        "note": None,
        "details": {},
        "seconds": None,
        **extra,
    }


def _paired(t: list[float], b: list[float]) -> float | None:
    if len(t) != len(b) or len(t) < 2:
        return None
    return metrics.paired_mean_and_se(np.asarray(t) - np.asarray(b))[1]


def lm_eval_paired(
    t: Mapping[str, Any], b: Mapping[str, Any]
) -> tuple[float, float | None] | None:
    """Paired task-mean accuracy difference of two lm-eval summaries."""

    def items(side: Mapping[str, Any]) -> dict[str, list[float]]:
        return {task: row.get("items") or [] for task, row in side["tasks"].items()}

    return metrics.task_mean_difference(items(t), items(b))


def _judge(
    spec: ProbeSpec,
    t: Mapping[str, Any],
    b: Mapping[str, Any] | None,
    comparable: Mapping[str, Any],
    dev: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row = _row(spec)
    for name, side in (("candidate", t), ("baseline", b)):
        if side is not None and "status" in side:
            note = f"{name}: {side['note']}"
            return {**row, "status": side["status"], "note": note}
    if (
        b is not None
        and spec.id in {"heldout_loss", "calibration"}
        and not comparable["validation"]
    ):
        return {
            **row,
            "status": "not_comparable",
            "note": comparable["validation_reason"],
        }
    if b is not None and spec.id == "token_agreement" and not comparable["tokenizer"]:
        return {**row, "status": "not_comparable", "note": "tokenizers differ"}
    if spec.id == "heldout_loss" and (b or {}).get("numerical_failure"):
        # The reference is broken: nothing can be judged against it.
        note = f"baseline: numerical failure: {b['numerical_failure']}"
        return {**row, "status": "error", "note": note}
    if spec.id == "heldout_loss" and t.get("numerical_failure"):
        row.update(judge(spec, math.nan, b["value"] if b else None))
        row.update(
            status="fail",
            note=f"numerical failure: {t['numerical_failure']}",
            details={"numerical_failure": t["numerical_failure"]},
        )
        return row
    if spec.id == "heldout_loss":
        se = (
            metrics.ratio_difference_se(
                t["window_sums"],
                t["window_counts"],
                b["window_sums"],
                b["window_counts"],
            )
            if b is not None
            else None
        )
        row.update(judge(spec, t["value"], b["value"] if b else None, se=se))
        row["details"] = {
            "perplexity": math.exp(t["value"]) if t["value"] < 700 else None,
            "baseline_perplexity": math.exp(b["value"])
            if b and b["value"] < 700
            else None,
            "valid_targets": t["valid_targets"],
            "windows": t["windows"],
            "top1_accuracy": t["top1_accuracy"],
            "baseline_top1_accuracy": b["top1_accuracy"] if b else None,
            "ms_per_token": t["ms_per_token"],
            # Per-window sums let later comparisons (klab compare) pair
            # this result with another one in the same eval group.
            "window_sums": [float(v) for v in t["window_sums"]],
            "window_counts": [int(v) for v in t["window_counts"]],
        }
        return row
    if spec.id == "calibration":
        row.update(judge(spec, t["value"], b["value"] if b else None))
        row["details"] = {
            "bins": t["bins"],
            "positions": t["positions"],
            "reliability": t["reliability"],
            "baseline_reliability": b["reliability"] if b else None,
        }
        return row
    if spec.id == "token_agreement":
        if b is None:
            return {**row, "status": "info", "note": "needs a baseline"}
        log_p = np.concatenate(b["log_probs"]).astype(np.float64)
        log_q = np.concatenate(t["log_probs"]).astype(np.float64)
        stats = {
            "top1_agreement": metrics.top1_agreement(log_p, log_q),
            "kl_base_to_candidate": float(metrics.kl_divergence(log_p, log_q).mean()),
            "js": float(metrics.js_divergence(log_p, log_q).mean()),
            "positions": int(log_p.shape[0]),
        }
        row.update(judge(spec, stats["top1_agreement"], None))
        if stats["js"] < NEAR_IDENTICAL_JS and row["status"] != "pass":
            # Argmax ties between near-identical (often near-uniform) predictions
            # are arbitrary; low agreement then says nothing.
            row.update(
                status="pass",
                note=f"near-identical distributions (JS < {NEAR_IDENTICAL_JS}); "
                "argmax agreement is not informative",
            )
        row["details"] = {**stats, "informative": stats["js"] >= NEAR_IDENTICAL_JS}
        return row
    if spec.id == "repetition":
        row.update(judge(spec, t["value"], b["value"] if b else None))
        keep = ("seq_rep_4", "distinct_1", "distinct_2")
        row["details"] = {
            **{k: t[k] for k in keep},
            "baseline": {k: b[k] for k in keep} if b else None,
            "samples": t["samples"],
            "baseline_samples": b["samples"] if b else [],
        }
        return row
    if spec.id in TEXT_PROBES:
        if b is not None and t["item_group"] != b["item_group"]:
            return {
                **row,
                "status": "not_comparable",
                "note": "different items (prompts, answers or candidates differ)",
            }
        se = _paired(t["credits"], b["credits"]) if b else None
        row.update(judge(spec, t["value"], b["value"] if b else None, se=se))
        details: dict[str, Any] = {
            "n": len(t["credits"]),
            "chance": t["chance"],
            "credit": "fractional_ties",
            "dev_accuracy": float(np.mean(t["dev_credits"]))
            if t["dev_credits"]
            else None,
            "baseline_dev_accuracy": float(np.mean(b["dev_credits"]))
            if b and b["dev_credits"]
            else None,
            "item_group": t["item_group"],
            "scoring": scoring.RANKING_PROTOCOL,
            "constant_pick": t.get("constant_pick"),
            "baseline_constant_pick": b.get("constant_pick") if b else None,
            # Per held-out item: what was asked, the answer, each arm's pick and
            # credit (the Behaviors page shows the misses).
            "items": [
                {
                    **{k: v for k, v in item.items() if v is not None},
                    "picked": t["picked"][i],
                    "credit": t["credits"][i],
                    **(
                        {
                            "baseline_picked": b["picked"][i],
                            "baseline_credit": b["credits"][i],
                        }
                        if b
                        else {}
                    ),
                }
                for i, item in enumerate(t["items"])
            ],
        }
        if spec.id == "parametric_recall":
            details["control_accuracy"] = t["control_accuracy"]
            details["baseline_control_accuracy"] = b["control_accuracy"] if b else None
            details["manifest_sha256"] = t["manifest_sha256"]
            details["manifest_seed"] = t["manifest_seed"]
        if spec.id == "needle":
            by_length: dict[str, dict[str, Any]] = {}
            for length in spec.params["char_lengths"]:
                picks = [i for i, c in enumerate(t["chars"]) if c == length]
                by_length[str(length)] = {
                    "chars": length,
                    "tokens": max((t["tokens"][i] for i in picks), default=0),
                    "accuracy": float(np.mean([t["credits"][i] for i in picks]))
                    if picks
                    else None,
                    "baseline_accuracy": float(
                        np.mean([b["credits"][i] for i in picks])
                    )
                    if picks and b
                    else None,
                }
            details["by_length"] = by_length
        row["details"] = details
        if b is not None:
            dev[spec.id] = {
                "value": details["dev_accuracy"],
                "baseline_value": details["baseline_dev_accuracy"],
                "se": _paired(t["dev_credits"], b["dev_credits"]),
            }
        return row
    if spec.id == "lm_eval":
        if "controlled" in t:
            complete = t["controlled"]["evidence_complete"] and (
                b is None
                or "controlled" not in b
                or b["controlled"].get("evidence_complete")
            )
            compatible = b is None or (
                t.get("benchmark_group") is not None
                and t.get("benchmark_group") == b.get("benchmark_group")
            )
            return {
                **row,
                "status": "error"
                if not complete
                else "not_comparable"
                if not compatible
                else "info",
                "note": "controlled task metrics; scientific interpretation not assessed",
                "details": {
                    "tasks": t["tasks"],
                    "benchmark_group": t.get("benchmark_group"),
                    "controlled": t["controlled"],
                    "baseline_controlled": b.get("controlled") if b else None,
                },
            }
        if b is not None and t.get("benchmark_group") != b.get("benchmark_group"):
            return {
                **row,
                "status": "not_comparable",
                "note": "different benchmark group "
                "(tasks, limit, few-shot or task versions differ)",
            }
        paired = lm_eval_paired(t, b) if b is not None else None
        row.update(
            judge(spec, t["value"], b["value"] if b else None, se=paired and paired[1])
        )
        row["details"] = {
            "tasks": t["tasks"],
            "baseline_tasks": b["tasks"] if b else None,
            "benchmark": t.get("benchmark"),
            "benchmark_group": t.get("benchmark_group"),
            "limit": spec.params["limit"],
            "chance": spec.params["chance"],
            "lm_eval_version": t["version"],
            "caveat": "tiny models usually score near chance on these tasks; "
            "small differences are noise at this limit",
        }
        return row
    raise ValueError(f"unknown probe {spec.id}")


# --- Battery ------------------------------------------------------------------


def lm_eval_spec(
    tasks: Sequence[str] | None = None, limit: int | None = None, protocol: Any = None
) -> ProbeSpec:
    """The lm-eval probe, optionally with other tasks or a larger ``limit``.

    Overrides change the benchmark, never the suite: they are recorded in the
    row's ``benchmark_group``, so results compare only with the same tasks and
    limit. Chance is only known for the default tasks (None otherwise).
    """
    spec = BY_ID["lm_eval"]
    if protocol is not None:
        from klab.probes.benchmark_protocol import BenchmarkProtocol

        protocol = BenchmarkProtocol.model_validate(protocol)
        if tasks is not None or limit is not None:
            raise ValueError("controlled protocol conflicts with task/limit overrides")
        return dataclasses.replace(
            spec,
            params={
                **spec.params,
                "tasks": list(protocol.tasks),
                "limit": protocol.limit,
                "protocol": protocol.model_dump(mode="json"),
                "chance": {t: None for t in protocol.tasks},
            },
        )
    if tasks is None and limit is None:
        return spec
    chosen = list(tasks) if tasks is not None else list(spec.params["tasks"])
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("lm_eval_tasks needs distinct task names")
    if limit is not None and limit < 1:
        raise ValueError("lm_eval_limit must be at least 1")
    known = spec.params["chance"]
    return dataclasses.replace(
        spec,
        params={
            **spec.params,
            "tasks": chosen,
            "limit": limit if limit is not None else spec.params["limit"],
            "chance": {t: known.get(t) for t in chosen},
        },
    )


def _require_complete_benchmark(results: list[dict[str, Any]], spec: ProbeSpec) -> None:
    """A reference's lm-eval row counts only with every task and item scored.

    An errored, skipped (stopped, OOM) or partial row names its unscored
    tasks in ``details.missing_tasks`` and is an error, i.e. missing evidence.
    """
    from klab.probes.lm_eval_adapter import missing_tasks

    for row in results:
        if row["id"] != "lm_eval":
            continue
        details = row.get("details") or {}
        gaps = (
            missing_tasks(details, spec.params["tasks"], spec.params["limit"])
            if row["status"] == "info"
            else list(spec.params["tasks"])
        )
        if gaps:
            row["details"] = {**details, "missing_tasks": gaps}
        if row["status"] == "info" and gaps:
            row.update(
                status="error",
                note="incomplete benchmark: unscored items in " + ", ".join(gaps),
            )
        elif row["status"] == "info":
            row["note"] = (
                "reference point: place results on it with "
                "`klab compare RESULT --references`"
            )


def run_battery(
    target: Any, baseline: Any | None = None, *, final: bool = False, **options: Any
) -> dict[str, Any]:
    """Run the battery (see :func:`_run_battery`); FINAL scores the held-back split.

    ``final=True`` is the final verdict on an already chosen candidate: text
    probes, agreement and generations read the ``final`` split instead of
    ``heldout`` (held-out loss stays on the run's validation split). Only
    ``klab probe --config final`` passes it; ``try`` and ordinary probes never
    do, so no selection loop reads the held-back items.
    """
    if not final:
        return _run_battery(target, baseline, **options)
    if options.get("comparison") is not None or options.get("leakage") is not None:
        raise ValueError("a final verdict is a standalone probe, not part of a try")
    with final_verdict_access():
        result = _run_battery(target, baseline, **options)
    return {**result, "verdict": final_verdict(result["verdict"]), "final": True}


def _run_battery(
    target: Any,
    baseline: Any | None = None,
    *,
    tier: str = "fast",
    fast_fail: bool = True,
    protocol: Mapping[str, int] | None = None,
    progress: Progress | None = None,
    context: LabContext | None = None,
    now: Callable[[], datetime] | None = None,
    lm_eval: ProbeSpec | None = None,
    comparison: Mapping[str, Any] | None = None,
    leakage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the battery through TIER and return one result (never raises on probes).

    COMPARISON is the enclosing try's held-out comparison, when there is one;
    a ``NOT_COMPARABLE`` try gets a non-escalating verdict whatever the probes
    say. LEAKAGE is the try's held-out leakage screen: contaminated training
    data makes the verdict ``rerun``.

    TARGET/BASELINE are :class:`Arm` loaders or already loaded runs. Arms are
    opened one at a time per tier: the baseline first, then the candidate,
    which is judged probe by probe so a hard failure stops it immediately.
    """
    from klab.lab_mode import eval_protocol

    started_at = (now or (lambda: datetime.now(UTC)))()
    started = time.monotonic()
    tiers_through(tier)
    suite = suite_identity()
    t_arm = as_arm(target)
    b_arm = as_arm(baseline)
    assert t_arm is not None
    if b_arm is not None and b_arm.reference:
        raise ValueError(
            "a reference model is not a probe baseline; "
            "use `klab compare RESULT --references`"
        )
    lm_spec = lm_eval or BY_ID["lm_eval"]
    specs = [
        lm_spec if s.id == "lm_eval" else s
        for s in ordered(tier)
        if not t_arm.reference or s.id in REFERENCE_PROBES
    ]
    results: list[dict[str, Any]] = []
    dev: dict[str, dict[str, Any]] = {}
    tiers_run: list[str] = []
    stop: dict[str, Any] = {"stopped": False, "kind": None, "at": None, "reason": None}
    state: dict[str, Any] = {"protocol": dict(protocol) if protocol else None}

    def report(phase: str, arm: str | None, current: str | None) -> None:
        state["current"] = current
        if progress is None:
            return
        progress(
            {
                "state": phase,
                "suite": suite["sha256"],
                "tier": tier,
                "arm": arm,
                "current": current,
                "done": [r["id"] for r in results],
                "statuses": {r["id"]: r["status"] for r in results},
                "total": len(specs),
                "target": (t_arm.cache.get("identity") or {}).get("run_id"),
                "started_at": started_at.isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )

    def checkpoint() -> None:
        if context is not None:
            context.checkpoint()

    def open_arm(
        arm: Arm,
        name: str,
        work: Callable[[Any, list[ProbeSpec]], None],
        tier_specs: list[ProbeSpec],
    ) -> None:
        if context is not None:
            context.enter(name, "probing")
        checkpoint()
        loaded = arm.load()
        try:
            _require_torch_engine(loaded)
            _describe(loaded, arm)
            if state["protocol"] is None and not arm.reference:
                state["protocol"] = eval_protocol(loaded.config)
            was_training = loaded.model.training
            loaded.model.eval()
            try:
                work(loaded, tier_specs)
            finally:
                loaded.model.train(was_training)
        finally:
            del loaded
            release_memory()

    def comparable() -> dict[str, Any]:
        out = {"validation": True, "validation_reason": None, "tokenizer": True}
        if b_arm is None or "identity" not in b_arm.cache:
            return out
        t_id = t_arm.cache["validation_identity"]
        b_id = b_arm.cache["validation_identity"]
        if t_id != b_id:
            out["validation"] = False
            out["validation_reason"] = "validation identity differs: " + ", ".join(
                sorted(k for k in t_id if t_id.get(k) != b_id.get(k))
            )
        out["tokenizer"] = (
            t_arm.cache["tokenizer_digest"] == b_arm.cache["tokenizer_digest"]
        )
        return out

    def stop_battery(kind: str, at: str | None, reason: str) -> None:
        stop.update(stopped=True, kind=kind, at=at, reason=reason)

    def measure_baseline(loaded: Any, tier_specs: list[ProbeSpec]) -> None:
        assert b_arm is not None
        for spec in tier_specs:
            checkpoint()
            report("running", "baseline", spec.id)
            _measure_safely(spec, loaded, b_arm, state["protocol"])

    def measure_candidate(loaded: Any, tier_specs: list[ProbeSpec]) -> None:
        for spec in tier_specs:
            checkpoint()
            report("running", "candidate", spec.id)
            tick = time.monotonic()
            measured = _measure_safely(spec, loaded, t_arm, state["protocol"])
            base = (
                b_arm.cache.get(probe_cache_key(spec.id)) if b_arm is not None else None
            )
            row = _judge(spec, measured, base, comparable(), dev)
            row["seconds"] = round(time.monotonic() - tick, 3)
            results.append(row)
            if fast_fail and row["status"] == "fail" and spec.hard:
                what = "numerical failure" if numerical_failure(row) else "hard failure"
                stop_battery("fast_fail", spec.id, f"fast-fail: {what} in {spec.id}")
                return

    try:
        for tier_name in tiers_through(tier):
            tier_specs = [s for s in specs if s.tier == tier_name]
            if stop["stopped"]:
                break
            if not tier_specs:
                continue  # e.g. a reference model has nothing in this tier
            if (
                tiers_run
                and fast_fail
                and b_arm is not None
                and any(r["status"] == "fail" for r in results)
            ):
                stop_battery(
                    "not_promising",
                    tier_specs[0].id,
                    f"not promising after the {tiers_run[-1]} tier; not escalating",
                )
                break
            if (
                tiers_run
                and fast_fail
                and any(r["status"] in MISSING_STATUSES for r in results)
            ):
                stop_battery(
                    "missing_evidence",
                    tier_specs[0].id,
                    f"missing evidence in the {tiers_run[-1]} tier; not escalating",
                )
                break
            tiers_run.append(tier_name)
            if b_arm is not None:
                open_arm(b_arm, "baseline", measure_baseline, tier_specs)
            open_arm(t_arm, "candidate", measure_candidate, tier_specs)
    except LabCancelled as error:
        stop_battery(error.kind, state.get("current"), error.reason)
    except (LabSignal, KeyboardInterrupt) as error:
        # Ctrl-C/SIGTERM: keep everything measured so far and finalize the
        # result; the caller re-raises after publishing it.
        release_memory()
        name = error.name if isinstance(error, LabSignal) else "keyboard_interrupt"
        stop_battery("interrupted", state.get("current"), f"interrupted: {name}")
    except Exception as error:
        if not is_out_of_memory(error):
            raise
        release_memory()
        stop_battery(
            "oom",
            state.get("current"),
            f"out of memory: {type(error).__name__}: {error}",
        )
    if context is not None:
        # From here on signals are noted, never raised, so the partial result
        # and its progress state are finalized; callers check context.signals.
        context.finalizing = True
    done = {r["id"] for r in results}
    for spec in specs:
        if spec.id not in done:
            note = f"skipped: {stop['reason']}" if stop["stopped"] else "skipped"
            results.append(_row(spec, note=note))
    order = {spec.id: i for i, spec in enumerate(specs)}
    results.sort(key=lambda r: order[r["id"]])
    guard = overfit_guard(
        dev,
        {
            r["id"]: {"value": r["value"], "baseline_value": r["baseline_value"]}
            for r in results
            if r["id"] in dev
        },
    )
    if t_arm.reference:
        _require_complete_benchmark(results, lm_spec)
    verdict = decide(
        results,
        has_baseline=b_arm is not None,
        tiers_run=tiers_run,
        requested_tier=tier,
        guard=guard,
        specs={**BY_ID, "lm_eval": lm_spec},
        stop=stop,
        reference=t_arm.reference,
        comparison=comparison,
        leakage=leakage,
    )
    protocol_used = state["protocol"] or {}

    def identity(arm: Arm | None) -> dict[str, Any] | None:
        if arm is None or "identity" not in arm.cache:
            return None
        return {
            **arm.cache["identity"],
            "eval_group": eval_group(arm.cache["validation_identity"], protocol_used),
        }

    result = {
        "format": RESULT_FORMAT,
        "created_at": started_at.isoformat(),
        "suite": suite,
        "tier": tier,
        "tiers_run": tiers_run,
        "stop": stop,
        "target": identity(t_arm),
        "baseline": identity(b_arm),
        "comparable": comparable() if "identity" in t_arm.cache else None,
        "protocol": dict(protocol_used),
        "probes": results,
        "guard": guard,
        "verdict": verdict,
        "seconds": round(time.monotonic() - started, 3),
    }
    report(
        "stopped" if stop["kind"] in INCOMPLETE_STOPS else "done",
        None,
        None,
    )
    return result


def progress_writer(path: Path) -> Progress:
    """Live progress for the dashboard: an atomically replaced JSON file."""

    def write(state: dict[str, Any]) -> None:
        try:
            write_json_atomic(path, state)
        except OSError:
            pass

    return write
