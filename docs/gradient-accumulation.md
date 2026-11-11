# Gradient accumulation

Gradient accumulation lets a run use several small microbatches for one optimizer update. It is a memory-management and optimization-boundary choice, not a disk checkpoint, a distributed collective, or a way to make several hosts share gradients.

## Configuration and meaning

```yaml
training:
  micro_batch_size: 1
  gradient_accumulation: 4
```

The nominal effective example batch is `micro_batch_size × gradient_accumulation`. Model Lab builds an update window from deterministic epoch order, up to that many examples, stopping at an epoch boundary or token-budget boundary. A short final window is real: its actual valid next-token targets, not the nominal product, determine its loss denominator and token counters.

For each microbatch, the trainer computes summed next-token cross-entropy, ignores labels marked `-100`, and weights the auxiliary routing loss by that microbatch's share of valid targets. It divides the accumulated loss by the update window's total valid targets. This makes ordinary dense no-dropout accumulation comparable to a corresponding large batch while preserving correct masking at a partial final block.

## One successful update

For each window, Model Lab:

1. clears gradients once;
2. runs backward for every nonempty microbatch;
3. unscales once for FP16, clips global gradients once, sets the scheduled learning rate once, and steps the optimizer once;
4. advances cursor, update counter, and target counter only after a successful step.

FP16 overflow restores the window RNG/cursor state, reduces the scaler, and retries identical data. Three consecutive skipped attempts fail while retaining the last good checkpoint. Checkpoint cadence and interruption handling act only after a completed update window.

The metrics record `batch/micro_batch_size`, `batch/accumulation_steps`, actual `batch/effective_batch_size`, and `batch/effective_tokens_per_update`, alongside the successful update's loss, learning rate, and throughput. Read actual counts when comparing a short final window.

## Several devices on one host

`runtime.secondary_device_indices` lists additional devices of the same backend
(`cuda`, `rocm`, `xpu`, or `cpu` for wiring checks). Each holds an in-process
replica of the model. The microbatches of one update are dealt round-robin across
the primary and the replicas, their gradients are reduced onto the primary, and
the primary alone clips, steps the optimizer and saves state. The update, its
effective batch and its checkpoint format are the same as on one device, so a run
can resume with or without secondaries.

```yaml
runtime:
  backend: xpu
  device_index: 0
  secondary_device_indices: [1]
training:
  gradient_accumulation: 16   # at least two microbatches per update are needed
```

Every device runs its share of the update on its own host thread, with compiled
or eager kernels alike. A replica's thread copies the updated weights before its
share and moves its gradients to the primary after it, so that exchange overlaps
with the primary's compute; gradients are added in replica order, so the sum is
the same on every run.

Each update that used replicas reports where its time went:

| Metric | Meaning |
|---|---|
| `performance/primary_compute_seconds` | The primary's forward and backward time for its share |
| `performance/replica_compute_seconds` | The slowest replica's time, including its weight copy |
| `performance/device_wait_seconds` | Time finished devices waited for the slowest, summed over devices |
| `performance/gradient_merge_seconds` | Adding the collected gradients on the primary |

A persistent `device_wait_seconds` means the devices are not equally fast or the
microbatches are not equally costly. Clipping, the optimizer step and reporting
are fixed costs per update, so give each device a share that is long compared
with them: with more devices, raise `gradient_accumulation` rather than letting
updates shrink. See the [two-B70 measurements](../benchmarks/multi_device_b70.md)
and use `benchmarks/device_busy.py` to see device idle time directly.

An update with a single microbatch (for example `gradient_accumulation: 1` or a
short final window) runs on the primary alone. `batch/devices_used` records how
many devices executed each committed update, so idle secondaries are visible in
the run's metrics. FP16, activation offload, semantic memory, allocation sidecars
and portability runs are refused with secondaries. Validation runs on the primary.
Throughput scaling depends on the backend and model and must be measured per run.

## What it does not mean

Accumulation does not reduce resident parameters, gradients, or optimizer state. It usually reduces peak activation size because each forward/backward sees a smaller microbatch, but it can increase update wall time because more microbatches are executed serially on one device. It does not provide multi-host data parallelism, expert exchange, model sharding, remote workers, or a KV decode cache.

MoE routing balance remains microbatch-local. The implementation masks invalid targets before routing summaries/auxiliary reductions, but it does not claim that accumulation recreates a whole-window global routing objective.

For the other resource mechanisms, see [memory accounting](memory.md), [activation recomputation](activation-checkpointing.md), and [disk checkpoints](checkpointing.md).
