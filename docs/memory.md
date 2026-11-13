# Memory

`klab inspect CONFIG` estimates training memory before anything runs, and
training measures it every update. This page covers both and the settings that
trade memory for time.

## The estimate

`inspect` reports disjoint categories, so nothing is counted twice:

- **Resident weights:** every parameter, including inactive experts and frozen
  memory tables. Tied embeddings count once.
- **Runtime buffers:** RoPE caches and token-memory maps.
- **Gradients:** FP32, for trainable parameters.
- **Optimizer state:** AdamW's two moments, or Adafactor's factors (below).
- **Retained activations:** tensors kept for backward, including logits and
  memory values.
- **Attention working tensors**, sized by how many keys each query scores: two
  windows for `sliding_window`; the local window plus selected blocks for
  `local_selected`; nothing for `sdpa` dense or latent attention on CUDA and XPU
  (confirm with a warmup); a full $T^2$ allowance otherwise.
- **Temporary workspace:** an allowance for allocator, kernels and logits.
- **Headroom:** 15% of the subtotal, for uncertainty.

Weights and gradients are counted in FP32 even with BF16 autocast. Activation
assumptions are conservative. When a matching run has measured a higher peak,
calibration raises the estimate; it never lowers it.

## The capacity it is compared with

`runtime.memory.max_device_memory_fraction` sets one ceiling. CPU uses the
lesser of that fraction of total RAM and the RAM available now. CUDA, ROCm and
XPU use the lesser of the fraction of device memory and the free amount. MPS and
Metal share memory with the host, so their ceiling comes from the recommended
working set and available system RAM; host and device memory are never added
together. If a reading is unavailable the fit is `UNKNOWN`.

`runtime.memory.budget_bytes` sets an artificial lower ceiling, useful for
planning.

`runtime.memory.estimate_policy` is `enforce` by default: a config whose
estimate exceeds the ceiling is refused. With `observe`, `train` and `stage`
warn and continue unchanged, including above `budget_bytes`; real allocation
failures still stop the run.

## Make it fit

`inspect` can propose a config that fits, written beside the original without
changing it:

```sh
klab inspect configs/runtime_smoke_cpu.yaml --write-proposal "$WORK/runtime-proposal.yaml"
```

The decision report lists each candidate change and its estimated peak. Use the
proposal only by passing it to a later command. Every option below is an
explicit config change.

### Gradient accumulation

```yaml
training:
  micro_batch_size: 1
  gradient_accumulation: 4
```

An update runs `gradient_accumulation` microbatches, summing their losses and
dividing by the update's total valid targets, then clips, sets the learning rate
and steps once. The effective batch is `micro_batch_size × gradient_accumulation`
examples; a final window cut short by the budget uses its real target count.
Smaller microbatches reduce activation memory at the cost of more passes per
update. Accumulation does not reduce weights, gradients or optimizer state.

With FP16, an overflow restores the window's state, lowers the loss scale and
retries the same data; three overflows in a row fail the run.

Mixture-of-experts load balancing is computed per microbatch, not over the whole
update.

### Chunked output head

```yaml
runtime:
  output_head_chunk_tokens: 2048
```

The output head turns every position's hidden state into one logit per
vocabulary entry, so a microbatch of `T` tokens holds `T × vocab_size` logits,
their FP32 copy and their gradient: several GiB once `T` reaches tens of
thousands. With this setting the head projects and scores that many positions
at a time, drops each chunk's logits after its loss and rebuilds them during
backward. The loss and gradients are the same up to summation order; the head
is computed twice. Validation uses the same chunks. It applies to the default
next-token objective on the PyTorch engine, on one device or several, and
cannot be combined with `compile_output_head_loss`.

On one Arc Pro B70 (bf16, activation recomputation on), a 730M-parameter model
with a 32,768-entry vocabulary and one 32,768-token row ran out of device memory
while building its logits; with 2,048-token chunks the first update completed at
a 26.9 GiB peak, too close to the card to train on. Adding
`strategy: transformer_block_and_attention` and `optimizer.state_offload` brought
the same row to an 18.7 GiB reserved peak over 20 steady updates (see below).

### Activation recomputation

```yaml
runtime:
  memory:
    activation_checkpointing:
      enabled: true
      strategy: transformer_block
```

Each decoder block keeps only its input during forward and runs again during
backward. This cuts retained activations for extra compute; it does not reduce
weights, gradients or optimizer state. `recompute/block_call_ratio` counts the
extra block calls. MLX supports it too.

`strategy: transformer_block_and_attention` also recomputes `local_selected`
attention inside each block one chunk of query blocks at a time. Without it a
block's recomputation holds every chunk's gathered keys, mask and scores at once
for its backward: about 13 GiB for one layer of a 32,768-token row at width
1,536 with 24 heads, against 2.7 GiB with it. The result is the same; attention
runs forward once more, about 30% more attention time. Other attention kinds
recompute per block only. The estimate charges such a layer one chunk of scores
and gathered keys.

### Activation offload

```yaml
runtime:
  memory:
    activation_offload:
      enabled: true
```

On a PyTorch accelerator, tensors saved for backward are copied to host memory
after forward and back before backward. Parameters and buffers stay on the
device. Copies are synchronous; their time is part of the update time and is
reported as `offload/transfer_seconds` and `offload/transfer_fraction`, with
`offload/bytes_to_cpu`, `bytes_to_device` and `peak_host_bytes`.

Before training, Lab checks that available host RAM covers the offloaded
activations plus checkpoint staging with a 15% margin. On MPS, host and device
share memory, so offload saves no capacity. Activation offload does not move
parameters or optimizer state; optimizer state has its own setting, below.

### Optimizer state offload

```yaml
optimizer:
  state_offload: true
```

On the PyTorch engine, AdamW's two moments (or Adafactor's factors) wait in
host memory between updates, pinned on CUDA and XPU. The optimizer step streams
them to the device one bucket of parameters at a time (128 MiB of parameters, or
one larger tensor alone), runs PyTorch's own single-tensor update on that bucket
and copies the state back. The update is the same arithmetic, so a run with
offload matches one without, and checkpoints save and resume the state the same
way. It costs one round trip of the state per update over the host link. MLX
refuses it. On CPU it changes nothing.

The estimate charges the device one bucket of state instead of all of it and
reports the full state as `host_optimizer_bytes`. Before training, Lab checks
that available host RAM covers that state plus checkpoint staging with a 15%
margin, as for activation offload. On an accelerator, `runtime.memory.policy:
max_fit` proposes it before any change of optimizer.

### Adafactor

`optimizer.name: adafactor` keeps one second-moment factor per row and per
column of each matrix (plus a full second moment for vectors) instead of AdamW's
two full moments. The estimate uses the actual factor shapes. Its learning rate
is a relative step-size cap, not an AdamW learning rate, so switching optimizer
is a new experiment.

## Measured memory

Each update samples memory at its start, after forward, after backward, after
the optimizer step and at its end:

- `memory/process_rss_bytes`, `memory/process_peak_rss_bytes`
- `memory/device_allocated_bytes`, `memory/device_reserved_bytes`
- `memory/device_peak_allocated_bytes`, `memory/device_peak_reserved_bytes`
  (where the backend's peak counter can be reset each update)
- `memory/driver_allocated_bytes` and `memory/device_sampled_peak_bytes` on MPS;
  the sampled peak is a lower bound

MLX reports active, peak and cache memory. Unavailable readings are omitted,
never written as zero.

## Spill guard

```yaml
runtime:
  memory:
    spill_guard:
      enabled: true
      host_drop_fraction: 0.5
```

A run whose memory outgrows the device may not fail; on some drivers it slows to
a stop. The spill guard stops it instead. Before every decoder block (including
its recomputation during backward) and at each sample above, it reads the same
counters and fails the update when:

- device memory in use (allocated, reserved or their peaks; the driver
  allocation on MPS) exceeds `max_device_memory_fraction` of the device total
  (of the recommended working set on MPS and Metal), or
- host available memory falls by more than `host_drop_fraction` of what was
  available after the first update.

On CUDA, ROCm and XPU the PyTorch caching allocator is also capped at
`max_device_memory_fraction` of the device, so an allocation past it fails at
once instead of waiting on the driver; that failure carries the same reason.
Near a full card the Intel driver can stall inside one kernel for minutes before
the next check runs, which is why the cap matters there.

The run then finishes as `failed` with a reason beginning `device memory spill:`
that names the reading and the setting. No checkpoint is written for the
interrupted update; earlier checkpoints stay resumable. The guard reads counters
only and never synchronizes the device. It watches the primary device only.

On a host with two Intel Arc cards, every device allocation on one card is also
resident in host memory for the other (`drm-resident-gtt` in the process's DRM
`fdinfo`), so host available memory falls by about as much as the device
allocates. That is normal and is why the host reading is compared with its value
after the first update, not before training.

A generation key/value cache belongs to inference only; it has no part in
training memory.
