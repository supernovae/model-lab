"""Compare completed warmed arms; never automatically certify or promote one."""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("results", nargs="+", type=Path)
a = p.parse_args()
results = [json.loads(path.read_text()) for path in a.results]
base = next(r for r in results if r["arm"] == "baseline")
if any(
    r["status"] != "completed" or r["plan_sha256"] != base["plan_sha256"]
    for r in results
):
    raise ValueError("All results must complete the same plan")
for r in results:
    if any(
        r.get(key) != base.get(key)
        for key in ("runtime_profile_sha256", "envelope_sha256", "loaded_gpu_libraries")
    ):
        raise ValueError("Runtime/library/envelope scopes differ")
    if [p["phase"] for p in r["passes"]] != ["cold", "warm"]:
        raise ValueError("Both cold and warm passes required")
for result in results:
    for phase, reference in zip(result["passes"], base["passes"], strict=True):
        if [
            (r["group"], r["step"], r["targets"], r["learning_rate"])
            for r in phase["updates"]
        ] != [
            (r["group"], r["step"], r["targets"], r["learning_rate"])
            for r in reference["updates"]
        ]:
            raise ValueError("Cold/warm update or LR mismatch")
    for key in ("runtime", "attention", "optimizer"):
        if result["config"][key] != base["config"][key]:
            raise ValueError(
                "Compare identical runtime/attention/optimizer settings first"
            )
b = base["passes"][-1]
for result in results:
    warm = result["passes"][-1]
    if [
        (r["group"], r["step"], r["targets"], r["learning_rate"])
        for r in warm["updates"]
    ] != [
        (r["group"], r["step"], r["targets"], r["learning_rate"]) for r in b["updates"]
    ]:
        raise ValueError("Update membership or LR mismatch")
    names = b["parameter_samples"].keys()
    if names != warm["parameter_samples"].keys():
        raise ValueError("Parameter set differs")
    deltas = [
        abs(x - y)
        for n in names
        for x, y in zip(
            b["parameter_samples"][n], warm["parameter_samples"][n], strict=True
        )
    ]
    print(
        json.dumps(
            {
                "arm": result["arm"],
                "warm_targets_per_second": warm["targets_per_update_second"],
                "relative_speed": warm["targets_per_update_second"]
                / b["targets_per_update_second"],
                "cold_update_seconds": result["passes"][0]["update_seconds"],
                "warm_wall_targets_per_second": warm["targets_per_wall_second"],
                "max_abs_loss_delta": max(
                    abs(x["loss"] - y["loss"])
                    for x, y in zip(b["updates"], warm["updates"], strict=True)
                ),
                "max_abs_sampled_parameter_delta": max(deltas),
                "compute_slots": warm["compute_slots"],
                "attention_pairs": warm["attention_pairs"],
                "reserved_headroom_bytes": warm["reserved_headroom_bytes"],
                "peak_allocated_bytes": warm["peak_allocated_bytes"],
                "peak_reserved_bytes": warm["peak_reserved_bytes"],
                "qualified": False,
                "missing_qualification_evidence": [
                    "full_tensor_model_numerics",
                    "reviewed_memory_headroom",
                    "matched_repeated_trials",
                ],
                "qualification": "Throughput evidence only; this tool never certifies a winner",
            }
        )
    )
