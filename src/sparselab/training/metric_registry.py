"""Canonical metric metadata shared by training and dashboard views."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    summary: str
    help_slug: str
    producer: str


METRICS = {
    spec.name: spec
    for spec in (
        MetricSpec(
            "train/loss",
            "nats/target",
            "Mean next-token negative log-probability.",
            "loss",
            "trainer",
        ),
        MetricSpec(
            "validation/loss",
            "nats/target",
            "Held-out next-token negative log-probability.",
            "loss",
            "evaluation",
        ),
        MetricSpec(
            "validation/perplexity",
            "ratio",
            "exp(validation/loss).",
            "perplexity",
            "evaluation",
        ),
        MetricSpec(
            "optimizer/learning_rate",
            "ratio",
            "AdamW update learning rate, or Adafactor relative learning-rate cap.",
            "learning-rate",
            "optimizer",
        ),
        MetricSpec(
            "optimizer/grad_norm",
            "L2",
            "Global norm before clipping.",
            "gradient-norm",
            "trainer",
        ),
        MetricSpec(
            "performance/tokens_per_second",
            "targets/s",
            "Valid targets processed per elapsed second.",
            "throughput",
            "trainer",
        ),
        MetricSpec(
            "moe/router_auxiliary_loss",
            "loss",
            "Auxiliary router load-balance term.",
            "moe",
            "trainer",
        ),
        MetricSpec(
            "memory/device_allocated_bytes",
            "bytes",
            "Native device allocation at an update boundary; MLX records Metal active memory.",
            "memory",
            "memory-monitor",
        ),
        MetricSpec(
            "memory/process_rss_bytes",
            "bytes",
            "Resident host process memory at an update boundary.",
            "memory",
            "memory-monitor",
        ),
        MetricSpec(
            "batch/effective_tokens_per_update",
            "targets",
            "Committed valid prediction targets in one optimizer update.",
            "accumulation",
            "trainer",
        ),
        MetricSpec(
            "offload/bytes_to_cpu",
            "bytes",
            "Saved activation bytes copied from device to host.",
            "offload",
            "offload",
        ),
        MetricSpec(
            "offload/bytes_to_device",
            "bytes",
            "Saved activation bytes restored for backward.",
            "offload",
            "offload",
        ),
        MetricSpec(
            "offload/peak_host_bytes",
            "bytes",
            "Peak live host activation-offload storage.",
            "offload",
            "offload",
        ),
        MetricSpec(
            "performance/step_seconds",
            "seconds",
            "Synchronized optimizer-update duration, excluding evaluation and checkpointing.",
            "throughput",
            "trainer",
        ),
        MetricSpec(
            "memory/device_reserved_bytes",
            "bytes",
            "Native allocator-reserved device memory where the runtime exposes it.",
            "memory",
            "memory-monitor",
        ),
        MetricSpec(
            "memory/device_peak_allocated_bytes",
            "bytes",
            "Native per-update peak allocated device memory.",
            "memory",
            "memory-monitor",
        ),
        MetricSpec(
            "memory/device_peak_reserved_bytes",
            "bytes",
            "Native per-update peak reserved device memory.",
            "memory",
            "memory-monitor",
        ),
        MetricSpec(
            "memory/device_sampled_peak_bytes",
            "bytes",
            "Sampled device-memory peak; on MPS this is a lower bound, not allocator high water.",
            "memory",
            "memory-monitor",
        ),
        MetricSpec(
            "memory/driver_allocated_bytes",
            "bytes",
            "Driver-reported allocation where available.",
            "memory",
            "memory-monitor",
        ),
        MetricSpec(
            "memory/device_cache_bytes",
            "bytes",
            "MLX Metal framework cache memory; not allocator-reserved or driver allocation.",
            "memory",
            "memory-monitor",
        ),
        MetricSpec(
            "memory/process_peak_rss_bytes",
            "bytes",
            "Peak resident host process memory sampled during an update.",
            "memory",
            "memory-monitor",
        ),
        MetricSpec(
            "memory/system_available_bytes",
            "bytes",
            "Host RAM available at an update boundary.",
            "memory",
            "memory-monitor",
        ),
        MetricSpec(
            "memory/sampling_seconds",
            "seconds",
            "Wall time spent collecting update memory observations, including sampling contention.",
            "memory",
            "memory-monitor",
        ),
        MetricSpec(
            "memory/observer_active_seconds",
            "seconds",
            "Time the optional background sampler was active; not its CPU overhead.",
            "memory",
            "memory-monitor",
        ),
        MetricSpec(
            "batch/micro_batch_size",
            "examples",
            "Configured examples in each microbatch.",
            "accumulation",
            "trainer",
        ),
        MetricSpec(
            "batch/accumulation_steps",
            "microbatches",
            "Microbatches accumulated into the committed update.",
            "accumulation",
            "trainer",
        ),
        MetricSpec(
            "batch/effective_batch_size",
            "examples",
            "Examples consumed by the committed update window.",
            "accumulation",
            "trainer",
        ),
        MetricSpec(
            "batch/tokens_per_micro_batch",
            "targets",
            "Valid prediction targets in a microbatch.",
            "accumulation",
            "trainer",
        ),
        MetricSpec(
            "recompute/block_call_ratio",
            "ratio",
            "Recomputed block calls divided by original block calls; one means each block was recomputed.",
            "recomputation",
            "trainer",
        ),
        MetricSpec(
            "offload/transfer_seconds",
            "seconds",
            "Synchronized saved-tensor transfer time included in total update time.",
            "offload",
            "offload",
        ),
        MetricSpec(
            "offload/transfer_fraction",
            "ratio",
            "Fraction of total update time spent transferring saved tensors.",
            "offload",
            "offload",
        ),
        MetricSpec(
            "engram/injection/final",
            "indicator",
            "Memory was injected after the final RMSNorm and before the LM head.",
            "memory",
            "model",
        ),
        MetricSpec(
            "engram/injection/after_block",
            "indicator",
            "Memory was injected after configured decoder blocks.",
            "engram",
            "model",
        ),
        MetricSpec(
            "engram/injection/embedding",
            "indicator",
            "Memory was injected after token embedding and before decoder block 0.",
            "memory",
            "model",
        ),
    )
}


# These are per-forward diagnostics, target-weighted over a committed update.
_ENGRAM = {
    "lookup_count": (
        "lookups",
        "Address lookups per forward; not a cumulative training count.",
    ),
    "unique_addresses": (
        "addresses",
        "Distinct addresses observed per forward across tables.",
    ),
    "collision_count": (
        "reused addresses",
        "Lookups minus distinct addresses, including repeated input; not proven hash collisions.",
    ),
    "bucket_reuse_rate": (
        "fraction",
        "Fraction of lookups that reuse an observed address; not retrieval accuracy.",
    ),
    "table_utilization": (
        "fraction",
        "Observed distinct addresses divided by table capacity; not lifetime occupancy.",
    ),
    "maximum_address_fraction": (
        "fraction",
        "Largest address count divided by lookup count.",
    ),
    "gate_mean": (
        "gate",
        "Mean learned gate value; not memory quality or a gradient measurement.",
    ),
    "value_norm": ("L2", "Mean memory value-vector norm."),
    "hidden_norm": ("L2", "Mean hidden-vector norm at the memory adapter."),
}
METRICS.update(
    {
        f"engram/{name}": MetricSpec(f"engram/{name}", unit, summary, "engram", "model")
        for name, (unit, summary) in _ENGRAM.items()
    }
)


def metric_spec(name: str) -> MetricSpec | None:
    """Return canonical metadata for an exact persisted metric name."""
    spec = METRICS.get(name)
    if spec is not None:
        return spec
    parts = name.split("/")
    if len(parts) == 4 and parts[:2] == ["engram", "port"] and parts[2].isdigit():
        base = METRICS.get(f"engram/{parts[3]}")
        if base is not None:
            return replace(base, name=name)
    return None
