# Training memory accounting

Training memory is not one number. Model Lab separates a conservative **estimate** from per-update **observations**, and keeps categories disjoint so a total is not inflated by counting the same tensor twice.

## What occupies memory

The `inspect` estimator currently reports these byte categories:

- **Resident weights** — all model parameters, including stored but inactive MoE experts and frozen portable-memory tables. Tied embeddings/output weights are counted once.
- **Runtime buffers** — RoPE caches and token-memory maps. Attention masks are built per call and are not registered. They are not parameter bytes merely because they are persistent runtime state.
- **Gradients** — FP32 gradients for trainable parameters.
- **Optimizer state** — AdamW moment state, or the modeled Adafactor factors. Optimizer state is separate from weights and gradients.
- **Retained activations** — tensors held for backward, including logits/loss allowance and memory-adapter values.
- **Attention working tensors** — projection and score/probability allowances, sized by how many keys each query scores: two windows for `sliding_window`; the local window plus selected blocks (and one gathered copy of those keys and values, plus one summary per block with the learned selector) for `local_selected`; none for `sdpa` dense or MLA attention on CUDA/XPU, which is an assumption to confirm with a measured warmup; and a full $T^2$ allowance otherwise (reference dense, MLA and block-sparse, and `sdpa` on CPU, MPS and ROCm). The estimate version is `reference-v2`.
- **Temporary workspace** — a conservative allocator/kernel/logit allowance.
- **Headroom** — 15% of the preceding subtotal, reserved for uncertainty rather than an allocation category available to the model.

The estimator uses FP32 master-parameter and gradient accounting. Autocast does not halve resident weights or optimizer state. Its `reference-v1` activation assumptions are deliberately conservative. A calibration multiplier may only raise an estimate when a matching native peak is observed; sampled MPS observations cannot lower an estimate or certify a bound.

A proposal retains any positive baseline calibration correction as an additive
uncertainty margin. That is not calibration of the changed configuration: its
matching-observation count is zero and it requires its own warmup. Reducing a
microbatch must not manufacture a fit by silently dropping an observed underestimate.

Engram and MoE are capacity subsets, not extra totals. A displayed expert or memory-table subtotal is already part of resident weights/gradients/optimizer state. “Active per token” is a static direct-use convention, not a claim that inactive expert optimizer state disappears.

## Capacity is one physical budget

`runtime.memory.max_device_memory_fraction` limits a single ceiling. CPU uses the lesser of the configured fraction of total RAM and currently available RAM. Discrete CUDA/ROCm/XPU uses the lesser of the fraction of device total and free memory. MPS/Metal uses the runtime's recommended working-set bound, available current-allocation readings, and available system RAM; absent counters are not an invented second pool.

Unified-memory MPS/Metal memory is shared with the host; do not add host RAM and device capacity. If a required reading is unavailable the fit result is `UNKNOWN`. `budget_bytes` is an additional artificial cap, useful to disprove a fit in planning or tests, but cannot prove that real hardware will fit.

## Choose estimate admission per run

`runtime.memory.estimate_policy` defaults to `enforce`, preserving the existing
refusal when estimated training memory exceeds its planning ceiling. For an
explicit observational run, select:

```yaml
runtime:
  memory:
    estimate_policy: observe
```

Both `train` and staging retain the estimate and its assumptions, warn about an
exceedance, and continue with the same requested configuration. Staging also
records a pilot's measured peak above the planning ceiling without treating that
comparison as an allocator failure. Resolved configurations record observation mode explicitly. No proposal is applied automatically, and model, batch, optimizer,
precision and token budgets are unchanged.

This also applies when `budget_bytes` lowers the planning ceiling: with
`estimate_policy: observe`, an estimate above that ceiling produces a warning rather
than a refusal. `budget_bytes` is not an enforced allocation cap in observation
mode.

Actual allocation failures, failed pilots, cancellation, input verification and
explicit resource envelopes still apply. Observation mode does not certify a
physical fit or implement a hard memory cap. This option addresses a potential
policy mismatch; an estimate alone is not evidence that a particular model fails
on a particular device.

Full-state continuation can explicitly change `estimate_policy` from `enforce`
to `observe` with `--allow-runtime-optimization`, including an appended-data
budget extension. An omitted historical value means `enforce`; the child records
the policy transition. This does not admit changes to memory budgets or fractions,
model/data settings, optimizer or effective batch. It does not reduce the retained
estimate, claim physical fit, or provide a hard GPU-memory cap.

## Observe an update

During PyTorch training, memory monitoring samples at begin, after forward, after backward, after optimizer work, and end of each update. Relevant metric families include:

- `memory/process_rss_bytes` and `memory/process_peak_rss_bytes` for process memory;
- `memory/device_allocated_bytes` and `memory/device_reserved_bytes` when the backend exposes them;
- `memory/device_peak_allocated_bytes` and `memory/device_peak_reserved_bytes` only when the native counter was successfully reset/read for that update;
- `memory/driver_allocated_bytes` for available MPS driver readings; and
- `memory/device_sampled_peak_bytes` for MPS's observed sample maximum, explicitly a lower bound rather than a native peak.

Unavailable metrics are omitted, not written as `0`. Optional background sampling is supported by `MemoryMonitor`, but normal trainer construction does not enable it. Sampling time is recorded so observation overhead is visible.

MLX/Metal uses the same process-RSS monitor plus native active/peak/cache counters.
Its forward/backward transform is sampled as one combined phase. Native peak
values are emitted only after a successful per-update reset; cache bytes are
reported separately, not mislabeled as driver or reserved memory. Update timing
includes synchronization, finite checks, and monitoring.

## Adafactor state accounting

PyTorch AdamW retains two full-size moments plus step scalars. Adafactor retains
one unfactored second moment for scalar/vector parameters. For a tensor shaped
`(..., rows, columns)`, it retains row and column factors with shapes
`(..., rows, 1)` and `(..., 1, columns)`, plus a step scalar. Tied parameters
are counted once. Neither optimizer removes parameter or gradient storage.

The estimator uses these actual factor shapes; it does not assume a fixed
percentage saving. Adafactor's scheduled learning rate is a relative step-size
cap with parameter-RMS scaling, not AdamW-equivalent update magnitude. Changing
optimizer is an explicit new experiment or promotion.

The acceptance record reports 86,444 optimizer-tensor bytes for AdamW and 3,436 for Adafactor
on the same 10,800-parameter tied tiny decoder, including 11 step scalars.
Both exactly match the estimator. This is persistent-state accounting, not a
total-process memory reduction or quality comparison.
[The record](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/acceptance/memory_optimizer_2026_09_22.json) includes
factor shapes, aliases, CPU RSS, native MLX fields, and calibration limits.

## Reduce memory without changing the science silently

A smaller `training.micro_batch_size` reduces activation dimensions. Increasing `training.gradient_accumulation` can preserve the requested effective example batch, at the cost of more forward/backward passes per update. Transformer-block activation recomputation trades retained activation memory for repeated backward-time computation. Both are explicit config changes.

`inspect` can write a separate proposal rather than editing its input:

```sh
uv run --locked --extra cpu klab inspect configs/runtime_smoke_cpu.yaml \
  --write-proposal klab-work/runtime-proposal.yaml
```

The adjacent decision report describes each ordered candidate and before/after estimated peak. A proposed config is only effective if a user deliberately supplies it to a later command. The planner does not claim savings from unsupported activation offload, optimizer-state offload, selective checkpointing, or unvalidated precision.

Proposal publication never overwrites either destination. Both files are
written and synced privately; the decision report is published durably before
the YAML becomes visible. Report format 1 binds the exact YAML bytes with
`yaml_sha256` and the loadable scientific configuration with `config_sha256`.
An interrupted process can leave an uncommitted report without a YAML; it is
not a usable proposal. Report or config conflicts fail rather than replacing
user-owned artifacts. Storage failures during publication roll back files
created by that invocation.

The [CLI acceptance record](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/acceptance/resource_policy_2026_09_22.json)
covers low-memory and capacity-constrained balanced proposals, unchanged source
files, preserved effective batch, loadable hash-bound YAML, and overwrite refusal.
These are planner/publication checks, not measured training-capacity claims.

## Do not conflate these mechanisms

- **Disk checkpoints** preserve durable model/training state for restart; see [checkpointing](checkpointing.md).
- **Activation recomputation** drops selected forward intermediates and reruns blocks in backward; see [activation checkpointing](activation-checkpointing.md).
- **Gradient accumulation** changes when a successful optimizer update is committed; see [gradient accumulation](gradient-accumulation.md).
- **Activation offload** copies saved tensors to host storage. Discrete-device estimates are conditional; measured capacity claims need paired hardware evidence. Unified-memory savings remain zero; see [offload](offload.md).
- **KV caches** are request-local inference/decode structures. Training does not reuse them and receives no cache-related memory discount.

A lower estimate or an architecture-specific diagnostic is not a throughput, quality, or hardware-capacity result. Compare matched observed runs before making any of those claims.
