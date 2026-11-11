"""Judge probe values against a baseline and turn them into a next action.

Pure functions: no model code, so the policy is easy to test and to read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from klab.probes.suite import TIERS, ProbeSpec, verdict_split

STATUSES = (
    "pass",
    "warn",
    "fail",
    "info",
    "skipped",
    "unavailable",
    "not_comparable",
    "error",
)
# A row in one of these states is a check that should have run and did not.
MISSING_STATUSES = frozenset({"unavailable", "not_comparable", "error"})
# Battery stops that leave requested checks unrun (fast-fail stops do not).
INCOMPLETE_STOPS = frozenset({"cancelled", "resources", "oom", "interrupted"})
ACTIONS = (
    "abandon",
    "tweak",
    "rerun",
    "escalate",
    "longer_run",
    "compare",
    "report",
)
NOISE_SIGMAS = 2.0
OVERFIT_GAP = 0.15
# The held-out loss SE is window-clustered *within one run*: it measures how
# noisy the evaluation is, not how much another seed would move the loss. At
# tiny budgets seed-to-seed spread is about 5-10x larger, so a single-seed
# "beyond noise" gain is a screening signal, not a win.
EVAL_NOISE_LABEL = "within-run eval noise"
SEED_CAVEAT = (
    " The ± on held-out loss is within-run eval noise from one seed per arm; "
    "seed-to-seed spread is typically 5-10x larger at small budgets. Before "
    "claiming a win, re-run both arms with the same 2-3 extra seeds (paired "
    "seeds) and check the gain holds on every pair."
)
NEXT_TIER_CHECKS = {
    "standard": "reworded fact recall and needle retrieval",
    "full": "standard lm-eval tasks; needs `uv sync --extra lmeval`",
}


def judge(
    spec: ProbeSpec,
    value: float | None,
    baseline: float | None,
    *,
    se: float | None = None,
) -> dict[str, Any]:
    """Classify one probe value; positive ``regression`` means worse."""
    out: dict[str, Any] = {
        "value": value,
        "baseline_value": baseline,
        "delta": None,
        "regression": None,
        "delta_se": se,
        "within_noise": None,
        "improved": False,
    }
    if value is None or not math.isfinite(value):
        hard_bad = spec.id == "heldout_loss" and value is not None
        out["status"] = "fail" if hard_bad else "error"
        out["note"] = "non-finite value" if value is not None else "no value"
        return out
    if spec.mode == "value_min":
        if spec.fail is not None and value < spec.fail:
            out["status"] = "fail"
        elif value < spec.warn:
            out["status"] = "warn"
        else:
            out["status"] = "pass"
        return out
    if baseline is None or not math.isfinite(baseline):
        out["status"] = "info"
        out["note"] = "absolute value; pass --vs BASELINE for a verdict"
        if spec.id == "repetition" and value >= 0.95:
            out["status"] = "warn"
            out["note"] = "greedy generations are stuck in loops"
        return out
    delta = value - baseline
    regression = -delta if spec.higher_is_better else delta
    if spec.mode == "delta_rel":
        regression = regression / abs(baseline) if baseline else regression
    noise = se is not None and abs(delta) <= NOISE_SIGMAS * se
    out.update(
        {
            "delta": delta,
            "regression": regression,
            "within_noise": noise if se is not None else None,
            "improved": regression < 0 and not noise,
        }
    )
    if noise:
        out["status"] = "pass"
        out["note"] = f"within noise (|delta| <= {NOISE_SIGMAS:g} SE)"
    elif spec.fail is not None and regression > spec.fail:
        out["status"] = "fail"
    elif regression > spec.warn:
        out["status"] = "warn"
    else:
        out["status"] = "pass"
    return out


def overfit_guard(
    dev: Mapping[str, Mapping[str, float | None]],
    heldout: Mapping[str, Mapping[str, float | None]],
) -> dict[str, Any]:
    """Flag a dev-split gain the held-out split does not share.

    ``dev``/``heldout`` map probe id -> {"value", "baseline_value"} (accuracy);
    a dev row may carry ``se`` (paired SE of the dev difference) so small dev
    splits do not cry wolf.
    """
    rows: dict[str, Any] = {}
    suspected = False
    for probe, dev_row in dev.items():
        held = heldout.get(probe) or {}
        values = (
            dev_row.get("value"),
            dev_row.get("baseline_value"),
            held.get("value"),
            held.get("baseline_value"),
        )
        if any(v is None for v in values):
            continue
        dev_gain = float(values[0]) - float(values[1])
        held_gain = float(values[2]) - float(values[3])
        noise = NOISE_SIGMAS * float(dev_row.get("se") or 0.0)
        flag = dev_gain - held_gain > max(OVERFIT_GAP, noise) and dev_gain > noise
        suspected = suspected or flag
        rows[probe] = {
            "dev_gain": dev_gain,
            "heldout_gain": held_gain,
            "dev_noise": noise,
            "flag": flag,
        }
    return {
        "verdict_split": verdict_split(),
        "overfit_suspected": suspected,
        "probes": rows,
        "rule": f"flag when the dev gain is beyond noise and exceeds the held-out "
        f"gain by more than max({OVERFIT_GAP}, {NOISE_SIGMAS:g} SE)",
    }


def missing_evidence(
    results: Sequence[Mapping[str, Any]], stop: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Checks that should have produced evidence and did not, with why."""
    missing = [
        {"id": r["id"], "status": r["status"], "note": r.get("note")}
        for r in results
        if r["status"] in MISSING_STATUSES
    ]
    if stop and stop.get("kind") in INCOMPLETE_STOPS:
        unrun = [r["id"] for r in results if r["status"] == "skipped"]
        if unrun:
            missing.append(
                {
                    "id": ",".join(unrun),
                    "status": f"stopped:{stop['kind']}",
                    "note": stop.get("reason"),
                }
            )
    return missing


REFERENCE_SUGGESTION = (
    "Reference point (pinned public checkpoint): place a result on this "
    "curve with `klab compare RESULT --references`."
)
REFERENCE_RERUN_SUGGESTION = (
    "Reference evaluation is incomplete: fix the cause (missing extras, memory) "
    "and re-run `klab probe ref:NAME --config full` before placing results "
    "on this reference."
)

NUMERICAL_SUGGESTION = (
    "Numerical failure: the candidate's loss is NaN/inf. Lower the learning "
    "rate or lengthen warmup, check precision and initialization, and look for "
    "a division or log of zero before trying again."
)


def numerical_failure(row: Mapping[str, Any]) -> str | None:
    """The numerical-failure message of a probe row, if it has one."""
    return (row.get("details") or {}).get("numerical_failure")


def _verdict(
    status: str,
    action: str,
    reasons: list[str],
    suggestion: str,
    missing: list[dict[str, Any]],
    next_tier: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "action": action,
        "next_tier": next_tier,
        "reasons": reasons,
        "missing": missing,
        "suggestion": suggestion,
    }


def _incomplete_verdict(missing: list[dict[str, Any]]) -> dict[str, Any]:
    named = "; ".join(f"{m['id']} ({m['status']}: {m['note']})" for m in missing)
    return _verdict(
        "incomplete",
        "rerun",
        [f"missing evidence: {m['id']} ({m['status']})" for m in missing],
        f"Not enough evidence for a verdict. Fix and re-run: {named}.",
        missing,
    )


NOT_COMPARABLE_SUGGESTION = (
    "The try's arms are not comparable (failed held-out checks: {failed}), so "
    "no probe result can promote it. Fix the cause (same data, tokenizer, "
    "memory packages and eval protocol for both arms) and re-run the try."
)


FINAL_REPORT_SUGGESTION = (
    "Final verdict on the held-back split ({status}): report this result as it "
    "is. It is terminal: do not change the idea, re-tune or re-run the final declaration. A "
    "new idea starts again with `klab try` on the held-out split."
)
FINAL_RERUN_SUGGESTION = (
    "Final verdict incomplete: fix the missing evidence ({missing}) and re-run "
    "the same final declaration unchanged (same candidate, baseline and tier). Do not "
    "read the partial final scores."
)


def final_verdict(verdict: Mapping[str, Any]) -> dict[str, Any]:
    """Terminal form of a final verdict, or an unchanged re-run.

    The choice follows the evidence, not the status: any requested check that
    produced no evidence (``missing``: unavailable, error, not comparable, a
    cancel/resource/OOM stop) makes the final battery ``incomplete`` with
    action ``rerun`` (the same final declaration, unchanged), whatever status
    ``decide`` gave it (e.g. a soft ``fail``).
    Only a decisive failure (``abandon``: a NaN/inf loss or a hard probe
    failure) is terminal despite gaps. A complete battery keeps its status and
    reasons and becomes ``report`` with no next tier and a suggestion that
    never invites selection (escalate, tweak or a longer run).
    """
    reasons = [*verdict.get("reasons", []), "final verdict: held-back final split"]
    missing = list(verdict.get("missing") or [])
    decisive = verdict.get("action") == "abandon"
    if verdict.get("status") == "incomplete" or (missing and not decisive):
        named = ", ".join(str(m.get("id")) for m in missing) or "?"
        gaps = [
            f"missing evidence: {m.get('id')} ({m.get('status')})"
            for m in missing
            if f"missing evidence: {m.get('id')} ({m.get('status')})" not in reasons
        ]
        return {
            **verdict,
            "status": "incomplete",
            "action": "rerun",
            "next_tier": None,
            "reasons": [*gaps, *reasons],
            "suggestion": FINAL_RERUN_SUGGESTION.format(missing=named),
        }
    return {
        **verdict,
        "action": "report",
        "next_tier": None,
        "reasons": reasons,
        "suggestion": FINAL_REPORT_SUGGESTION.format(status=verdict.get("status")),
    }


def try_not_comparable(comparison: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The verdict for a try whose comparison is ``NOT_COMPARABLE``, else None.

    A non-comparable try never escalates: this takes precedence over every
    probe-based outcome except a numerical failure (itself decisive).
    """
    if not comparison or comparison.get("verdict") != "NOT_COMPARABLE":
        return None
    failed = ", ".join(comparison.get("failed") or []) or "unknown"
    return _verdict(
        "incomplete",
        "rerun",
        [f"try comparison NOT_COMPARABLE (failed held-out checks: {failed})"],
        NOT_COMPARABLE_SUGGESTION.format(failed=failed),
        [{"id": "comparison", "status": "not_comparable", "note": failed}],
    )


def respect_try_comparison(
    verdict: Mapping[str, Any] | None, comparison: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    """A recorded probe verdict as shown for a try (older records included)."""
    override = try_not_comparable(comparison)
    if override is None or (verdict or {}).get("action") == "abandon":
        return verdict
    return override


def decide(
    results: Sequence[Mapping[str, Any]],
    *,
    has_baseline: bool,
    tiers_run: Sequence[str],
    requested_tier: str,
    guard: Mapping[str, Any] | None,
    specs: Mapping[str, ProbeSpec],
    stop: Mapping[str, Any] | None = None,
    reference: bool = False,
    comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Overall verdict and a recommended next action for humans and agents.

    Missing evidence never reads as success: an unavailable optional tier, a
    probe error, a non-comparable probe or a battery stopped by cancellation,
    resources or OOM makes the battery ``incomplete`` (action ``rerun``) unless
    a failure already decides it. ``comparison`` is the enclosing try's
    held-out comparison: ``NOT_COMPARABLE`` takes precedence over every
    probe outcome except a numerical failure.
    """
    missing = missing_evidence(results, stop)
    by_id = {row["id"]: row for row in results}
    numerical = [r for r in results if numerical_failure(r)]
    if numerical:
        # A NaN/inf loss is a decisive outcome with or without a baseline.
        return _verdict(
            "fail",
            "abandon",
            [
                f"hard fail: {r['id']} (numerical failure: {numerical_failure(r)})"
                for r in numerical
            ],
            NUMERICAL_SUGGESTION,
            missing,
        )
    not_comparable = try_not_comparable(comparison)
    if not_comparable is not None:
        return not_comparable
    last = tiers_run[-1] if tiers_run else requested_tier
    next_tier = (
        TIERS[TIERS.index(last) + 1] if last in TIERS and last != TIERS[-1] else None
    )
    if reference:
        # A reference point is only usable with complete benchmark evidence.
        if missing:
            reasons = [
                f"missing: {m['id']} ({m['status']}): {m.get('note') or ''}".rstrip(
                    ": "
                )
                for m in missing
            ]
            unscored = sorted(
                {
                    t
                    for r in results
                    for t in (r.get("details") or {}).get("missing_tasks") or []
                }
            )
            if unscored:
                reasons.append("unscored tasks: " + ", ".join(unscored))
            return _verdict(
                "incomplete", "rerun", reasons, REFERENCE_RERUN_SUGGESTION, missing
            )
        return _verdict(
            "info",
            "compare",
            ["reference point: every benchmark task and item scored"],
            REFERENCE_SUGGESTION,
            missing,
        )
    if not has_baseline:
        if missing:
            verdict = _incomplete_verdict(missing)
            verdict["suggestion"] += " Add --vs BASELINE for pass/warn/fail."
            return verdict
        return _verdict(
            "info",
            "compare",
            ["no baseline: values are absolute"],
            "Re-run with --vs BASELINE to get pass/warn/fail and a recommended action.",
            missing,
        )
    hard = [r for r in results if r["status"] == "fail" and specs[r["id"]].hard]
    fails = [r for r in results if r["status"] == "fail"]
    warns = [r for r in results if r["status"] == "warn"]
    if hard:
        return _verdict(
            "fail",
            "abandon",
            [f"hard fail: {r['id']}" for r in hard],
            specs[hard[0]["id"]].suggests,
            missing,
        )
    if fails:
        return _verdict(
            "fail",
            "tweak",
            [f"fail: {r['id']}" for r in fails],
            specs[fails[0]["id"]].suggests,
            missing,
        )
    if missing:
        return _incomplete_verdict(missing)
    if guard and guard.get("overfit_suspected"):
        return _verdict(
            "warn",
            "tweak",
            ["dev split improved but held-out did not"],
            "You may be fitting the probe items: change the idea, not the "
            "prompts. Verdicts only use the held-out split.",
            missing,
        )
    status = "warn" if warns else "pass"
    reasons = [f"warn: {r['id']}" for r in warns]
    loss = by_id.get("heldout_loss")
    if loss is not None and loss.get("improved"):
        reasons.append(
            f"held-out loss improved beyond {EVAL_NOISE_LABEL} (one seed per arm)"
        )
        if next_tier is not None:
            return _verdict(
                status,
                "escalate",
                reasons,
                f"Promising: run `--config {next_tier}` ("
                + NEXT_TIER_CHECKS.get(next_tier, "the next tier")
                + ") before a longer run."
                + SEED_CAVEAT,
                missing,
                next_tier,
            )
        return _verdict(
            status,
            "longer_run",
            reasons,
            "Every tier holds up: schedule a longer run (more tokens or seeds) "
            "to confirm the gain. Probes screen; they do not prove usefulness."
            + SEED_CAVEAT,
            missing,
        )
    reasons.append("no held-out loss gain beyond noise")
    return _verdict(
        status,
        "tweak",
        reasons,
        "No measurable gain at this budget: try a bolder change, or a longer "
        "run only if the idea needs more tokens to show.",
        missing,
    )
