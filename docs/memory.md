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
a 26.9 GiB peak. That is one update of one run and still too close to the card's
32 GiB to train on: the run then slowed as allocations spilled into host memory.

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
share memory, so offload saves no capacity. Offload does not move parameters or
optimizer state.

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

A generation key/value cache belongs to inference only; it has no part in
training memory.
