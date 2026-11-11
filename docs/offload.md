# Activation offload

Activation offload moves eligible autograd-saved tensors to host storage after forward and restores them for backward. It is distinct from activation recomputation, which reruns forward computation, and from disk checkpoints, which preserve durable restart state.

## Current implementation

Request it explicitly:

```yaml
runtime:
  memory:
    activation_offload:
      enabled: true
```

For a selected non-CPU PyTorch accelerator, the trainer creates
`ActivationOffload(device, model)` and installs
`torch.autograd.graph.saved_tensors_hooks` around each microbatch forward.
Construction excludes all parameter and buffer storage identities, including
non-leaf aliases and views; ordinary non-parameter leaf activations are still
eligible. Eligible aliases share one pageable CPU backing copy per storage
version. In-place updates therefore retain distinct saved values. A weak storage
identity prevents allocator-address reuse without pinning the original device
allocation. Each packed view retains its shape, stride, storage offset, and dtype;
host storage remains live until all Autograd packed handles release it.

The reference path is synchronous and pageable. Transfer timing starts after
the pre-copy synchronization and includes the copy and completion synchronization;
queued forward computation is not counted as transfer time. Per-update metrics are
`offload/bytes_to_cpu`, `offload/bytes_to_device`, `offload/peak_host_bytes`,
`offload/transfer_seconds`, and (when the update duration is available)
`offload/transfer_fraction`. Transfer time is included in normal update time;
it is never subtracted to improve reported throughput. `reset()` clears those
per-update counters only after no packed state remains, returning `False` if a
caller retained a graph.

`probe_activation_offload(device)` performs a disposable real saved-tensor
roundtrip/backward and returns structured support, reason, unified-memory, and
measurement data. CPU requests are rejected as meaningless. A passing probe
only proves this hook path on the selected runtime; it does not prove capacity
relief. MPS is unified memory, so its reported estimated device-capacity saving
is always zero. Discrete CUDA, ROCm (via CUDA APIs), and XPU still require a
paired real-hardware memory observation before making a capacity claim.

## Preflight and estimates

Before model initialization or run creation, the trainer checks live available
host RAM and process RSS. The incremental host requirement includes predicted
saved activations, conservative checkpoint staging (twice the resident weights,
buffers, and optimizer storage), and a 15% uncertainty margin. Available RAM is
not increased by the process's existing RSS.

On unified memory, the model estimate and additional host staging share one
budget; an explicit artificial device ceiling also constrains that combined
requirement. Insufficient headroom raises an error before creating a run. The
accepted calculation is recorded in the immutable resource decisions.

Device estimates may subtract eligible retained activations only for a
supported discrete accelerator; weights, gradients, optimizer state, and active
backward workspace remain resident. CPU and unified-memory backends report zero
physical-capacity gain. A one-run report has no invented baseline.

## Actual MPS comparison

The acceptance study ran three matched FP32 CLI jobs on Apple Silicon with PyTorch 2.14.0:
six updates and 192 committed targets each, discarding the first two updates
for median timing. These tiny-run measurements are diagnostics, not a general
performance benchmark.

| Policy | Median update | Median transfer | Peak live offload host bytes | Sampled device peak |
|---|---:|---:|---:|---:|
| Baseline | 20.77 ms | 0 | 0 | 236,032 B |
| Offload | 57.47 ms | 20.56 ms | 24,384 B | 217,856 B |
| Offload + recomputation | 30.38 ms | 5.20 ms | 4,288 B | 217,856 B |

Offload transferred 48,768 bytes in each direction per update; recomputation
reduced that to 8,576 bytes. Final parameters matched the baseline within
`rtol=3e-4, atol=1e-5`. Device peaks are sampled lower bounds, not native allocator
high-water marks. MPS remains unified memory: **estimated physical-capacity
savings are zero**, despite the smaller sampled device allocation.

An artificial budget that fit the model but not additional host staging was
rejected before run creation. Saved-version, parameter/buffer-alias, and device
allocation-lifetime regressions also passed. [Recorded configurations, runtime
identities, headroom decisions, and measurements](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/acceptance/offload_2026_09_22.json)
preserve the acceptance evidence. Discrete-device capacity claims remain gated
on actual CUDA/ROCm/XPU hardware.

## What it is not

Offload does not move model parameters or optimizer state, does not make an
inactive MoE expert disappear, and does not create disk-resumable state.
`optimizer.state_offload: true` remains deferred. It does not create a KV decode
cache, remote worker, or distributed backward path.

Use [memory accounting](memory.md) for the estimate/measurement boundary,
[activation recomputation](activation-checkpointing.md) for the compute-for-memory
alternative, and [checkpointing](checkpointing.md) for durable restart state.
