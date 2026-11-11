"""Disposable, measured proposals for divisible batch configurations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median

import yaml

from klab.config.models import RunConfig
from klab.staging import stage
from klab.training.manifest import config_sha256
from klab.workspace_preflight import require_storage, training_storage_checks


def batch_candidates(config: RunConfig, max_candidates: int = 8) -> tuple[int, ...]:
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    batch = config.training.micro_batch_size * config.training.gradient_accumulation
    original = config.training.micro_batch_size
    divisors = [
        value for value in range(1, math.isqrt(batch) + 1) if batch % value == 0
    ]
    values = sorted(set(divisors + [batch // value for value in divisors]))
    ranked = sorted(values, key=lambda value: (abs(math.log2(value / original)), value))
    selected = set(ranked[:max_candidates])
    selected.add(original)
    return tuple(sorted(selected))


def _summarize_pilot(
    pilot: dict[str, object], ceiling: int | None
) -> dict[str, object]:
    updates = pilot.get("update_observations")
    if not isinstance(updates, list) or len(updates) < 6:
        return {
            "status": "rejected",
            "reason": "insufficient post-initialization updates",
        }
    steady = updates[2:]
    timings = [float(item["update_seconds"]) for item in steady]
    targets = [int(item["targets"]) for item in steady]
    if any(not math.isfinite(value) or value <= 0 for value in timings) or any(
        value <= 0 for value in targets
    ):
        return {"status": "rejected", "reason": "invalid update observations"}
    step_median = median(timings)
    relative_mad = median(abs(value - step_median) for value in timings) / step_median
    throughput = sum(targets) / sum(timings)
    peak = pilot.get("observed_peak_bytes")
    peak_bytes = int(peak) if isinstance(peak, (int, float)) else None
    headroom = (
        ceiling - peak_bytes if ceiling is not None and peak_bytes is not None else None
    )
    result: dict[str, object] = {
        "step_seconds_median": step_median,
        "step_seconds_min": min(timings),
        "step_seconds_max": max(timings),
        "relative_median_absolute_deviation": relative_mad,
        "targets_per_second": throughput,
        "observed_peak_bytes": peak_bytes,
        "capacity_ceiling_bytes": ceiling,
        "headroom_bytes": headroom,
        "backend": pilot.get("runtime", {}).get("backend")
        if isinstance(pilot.get("runtime"), dict)
        else None,
        "peak_method": pilot.get("peak_method"),
    }
    if relative_mad > 0.25:
        result.update(status="rejected", reason="unstable update timing")
    elif headroom is None:
        result.update(status="rejected", reason="device headroom unavailable")
    elif headroom < ceiling * 0.1:
        result.update(status="rejected", reason="insufficient measured headroom")
    else:
        result.update(status="eligible", reason=None)
    return result


def calibrate_batch(
    config: RunConfig, output: Path, *, max_candidates: int = 8
) -> Path:
    """Run bounded warmups and write a separate proposal; leave source untouched."""
    candidates = batch_candidates(config, max_candidates)
    output = output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(output)
    require_storage(training_storage_checks(config))
    output.mkdir(parents=True)
    effective_batch = (
        config.training.micro_batch_size * config.training.gradient_accumulation
    )
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "schema_version": 1,
        "source_config_sha256": config_sha256(config.model_dump(mode="json")),
        "effective_batch": effective_batch,
        "warmup_steps": max(config.staging.warmup_steps, 8),
        "candidates": rows,
        "selected_micro_batch_size": None,
        "proposal": None,
    }
    for micro in candidates:
        candidate = config.model_copy(
            update={
                "training": config.training.model_copy(
                    update={
                        "micro_batch_size": micro,
                        "gradient_accumulation": effective_batch // micro,
                    }
                ),
                "staging": config.staging.model_copy(
                    update={"warmup_steps": max(config.staging.warmup_steps, 8)}
                ),
            }
        )
        row: dict[str, object] = {
            "micro_batch_size": micro,
            "gradient_accumulation": effective_batch // micro,
            "config_sha256": config_sha256(candidate.model_dump(mode="json")),
            "stage_path": f"candidate-{micro}",
        }
        try:
            stage(candidate, output / f"candidate-{micro}", "warmup")
            report = json.loads(
                (output / f"candidate-{micro}" / "stage.json").read_text()
            )
            pilots = report["pilot_reports"]
            warmup = next(
                item["report"] for item in pilots if item["purpose"] == "warmup"
            )
            ceiling = report["estimate"].get("capacity_ceiling_bytes")
            row.update(_summarize_pilot(warmup, ceiling))
            row["pilot_report_sha256"] = warmup["sha256"]
            row["runtime"] = report["runtime"]
        except Exception as error:  # noqa: BLE001 - retain every failed candidate
            row.update(status="rejected", reason=f"{type(error).__name__}: {error}")
        rows.append(row)
        (output / "calibration.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    eligible = [row for row in rows if row["status"] == "eligible"]
    if eligible:
        selected = max(eligible, key=lambda row: float(row["targets_per_second"]))
        micro = int(selected["micro_batch_size"])
        payload = config.model_dump(mode="json")
        payload["training"]["micro_batch_size"] = micro
        payload["training"]["gradient_accumulation"] = effective_batch // micro
        proposal = output / "proposal.yaml"
        proposal.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        summary["selected_micro_batch_size"] = micro
        summary["proposal"] = str(proposal)
    (output / "calibration.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output / "calibration.json"
