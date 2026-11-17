# Design: sharded training across the devices of one host

Status: proposal for owner review; the implementation follows in the next two
pull requests. Backlog: [#497](https://github.com/supernovae/model-lab/issues/497).

Marking: **Verified** was run and the result is stated. **Gloo only** was run on
CPU with `gloo` collectives and not on a card. **Not verified** is a design
claim that needs the hardware. Section 6 lists what the owner decides.

## 0. Summary

1. Today `runtime.secondary_device_indices` trains data-parallel *replicas*:
   every device holds the whole model and its gradients, the primary alone holds
   the optimizer, and weights and gradients cross between processes through
   shared host tensors. Nothing is sharded and the interconnect is host memory.
2. A 1.05B-parameter model (845M active) with FP32 master weights, gradients
   and AdamW state needs 3.9 + 3.9 + 7.8 GiB before activations; `klab inspect`
   estimates 35.1 GiB against a 31.9 GiB card.
3. The design adds one recorded option, `runtime.parallel.mode`, with
   `replicated` (today, the A/B baseline), `zero1`, `zero2` and `fsdp`. The
   sharded modes run real `torch.distributed` collectives (`xccl` on XPU,
   `nccl` on CUDA and ROCm, `gloo` on CPU) over the same device list and the same
   one-process-per-device topology, and are all settings of PyTorch's FSDP2
   (`fully_shard`).
4. Checkpoints stay in the one-device layout (whole tensors, named), so a run
   resumes in any mode and on any device count. Sharded checkpoints are a
   later option, not a second format (section 5).
5. The estimator reports each mode's per-device memory, and `--write-proposal`
   offers sharding before it offers a smaller batch or recomputation.

## 1. Feasibility on this host

The host has four Intel Arc Pro B70 cards (32 GB) and PyTorch 2.15.0+xpu.

| Question | Result | Where |
| --- | --- | --- |
| Does the build ship `xccl`, `gloo` and FSDP2? | `is_xccl_available()`, `is_gloo_available()` are true; `torch.distributed.fsdp.fully_shard` imports | **Verified** |
| `all_reduce`, `reduce_scatter_tensor`, `all_gather_into_tensor` on a card | correct results, world size 1, card 3 | **Verified** |
| FSDP2 `fully_shard` with forward, backward and gradients on a card | parameters are `DTensor`, gradients arrive, world size 1, card 3 | **Verified** |
| Two ranks sharing card 3 | the same three collectives and FSDP2 gave correct results on rank 0; oneCCL printed an error stack from the other rank or from teardown, not investigated | **Partly verified** |
| Collectives between *different* cards, each process confined to one card by `ZE_AFFINITY_MASK` as Lab does | not run: needs a window with all four cards | **Not verified** |
| Bandwidth over PCIe, which collectives are fast | not measured, same reason | **Not verified** |
| FSDP2 with this model (blocks, tied embedding, Engram tables, recomputation, chunked head) | equal to one device after two updates in every mode, 2 ranks | **Gloo only** |
| Compiled blocks and the compiled output head under FSDP2 | compiled blocks agree with one device in `zero1`; the compiled output head is XPU-only | **Gloo only** (blocks), **Not verified** (head) |
| Resume across modes and device counts | sharded to one device and back, then the next update agrees | **Gloo only** |

The pull request that adds the option carries an `xpu`-marked test that repeats
the equality check with one process per card over `xccl`. It runs when the
owner grants a window; until then every claim in the last three rows rests on
`gloo`.

The main risk is the cross-card row. Lab confines each process to one card so a
process does not mirror its device memory into host RAM. If `xccl` cannot reach
a peer card from a confined process, the fallback is a rank that sees every card
and pays that host-RAM cost, which `klab` already warns about.

## 2. What is sharded

Per device, with $N$ devices, $W$ the FP32 weights, $G$ the gradients and $O$
the optimizer state:

| Mode | Weights | Gradients | Optimizer | Communication |
| --- | --- | --- | --- | --- |
| `replicated` | $W$ | $G$ | $O$ on the primary only | host-staged copies |
| `zero1` | $W + W/N$ | $G + G/N$ | $O/N$ | reduce-scatter once, all-gather once |
| `zero2` | $W + W/N$ | $G/N$ plus the groups in flight | $O/N$ | reduce-scatter per microbatch, all-gather once |
| `fsdp` | $W/N$ plus the gather transient | $G/N$ plus the groups in flight | $O/N$ | all-gather per group, twice, reduce-scatter per microbatch |

The sharded modes keep a sharded FP32 master beside the whole parameters
(`zero1`, `zero2`), which costs $W/N$ over a textbook ZeRO that aliases them.
`fsdp` gathers one group at a time, where a group is a transformer block, a token
memory (Engram table set), or the rest (embedding, norm, output head). The rest
is gathered once and held through backward, because the loss reads the output
weight after the forward.

Why FSDP2 and not a hand-written ZeRO: it already implements all three layouts,
bucketing, prefetch and `DTensor` optimizer state, and exposes the switches the
modes need (`reshard_after_forward`, `reshard_after_backward`,
`set_requires_gradient_sync`). A hand-written ZeRO-1/2 would be about as long as
the glue this design needs and would not give `fsdp`. Lab's own code is the glue.

## 3. The mechanism

- **Topology.** Rank 0 is the trainer's process on `device_index`; ranks 1..N-1
  are the processes `secondary_device_indices` already starts, one per card, each
  confined to its card. They form one process group on `127.0.0.1`.
- **Commands.** Every collective must run on every rank in the same order, so the
  primary sends each worker a command (update, evaluate, gather weights, gather
  optimizer state, restore) and runs the same operation itself.
- **Microbatches.** Dealt round-robin as today. Each forward and backward is a
  collective, so every rank runs the same number; short ranks run a masked
  microbatch that carries no loss or diagnostic.
- **Equality.** Gradients are summed (divide factor 1, plain `SUM`); the loss is
  already divided by the update's target count. Clipping and the finite checks
  see the whole gradient. AdamW is elementwise, so sharding its step is exact.
  Adafactor is not elementwise and is refused.
- **Forward routing.** FSDP gathers in hooks on `forward`; Lab's model runs
  `forward_with_aux`. In a sharded run the module's `forward` *is*
  `forward_with_aux`.
- **Checkpoints.** Weights and optimizer state are gathered whole to the
  primary's host and written in the one-device layout. Restore scatters the
  optimizer state with `distribute_tensor`. A run moves between modes and device
  counts, which is the resume rule of
  [#485](https://github.com/supernovae/model-lab/issues/485).
- **Options, not versions.** `runtime.parallel` is omitted from the recorded
  config while `replicated`, so existing records and digests do not change.

## 4. Memory estimate

`klab inspect` prints each mode's per-device figures from the config's device
list. For a 12B-token pretraining config (28 layers, hidden 1536, 32,768
vocabulary, Engram tables, bf16, sequence 4096, microbatch 1, no recomputation,
four devices), in GiB per device:

| Mode | Weights | Gradients | Optimizer | Gather | Activations | Peak (with workspace and headroom) |
| --- | --- | --- | --- | --- | --- | --- |
| `replicated` | 3.90 | 3.90 | 7.79 | 0.00 | 10.38 | 35.10 |
| `zero1` | 4.87 | 4.87 | 1.95 | 0.00 | 10.38 | 30.17 |
| `zero2` | 4.87 | 1.92 | 1.95 | 0.00 | 10.38 | 26.43 |
| `fsdp` | 0.97 | 1.92 | 1.95 | 0.94 | 10.38 | 22.59 |

These are analytic, not measured. Activations are not sharded and are the next
largest term in every mode: recomputation and the chunked output head apply on
top. A measured comparison is section 7.

## 5. Not in this design

- **Sharded checkpoints** (per-rank files). Gathering whole tensors needs the
  primary's host RAM to hold weights and optimizer state at save time (about
  12 GiB here) and cannot scale to a model that does not fit one host. A sharded
  layout must still resume on another device count; that is a resharding reader,
  and per the rule against compatibility readers it should arrive as a named
  option with its own tests.
- **Several hosts.** Same-host only (`127.0.0.1`).
- **BF16 gathers.** Gathering parameters in BF16 halves the transient and the
  traffic but changes numerics for the non-matmul ops, so it would be a named
  option after a measurement.
- **Sharded validation.** Every rank runs the same validation batches.

## 6. Owner decisions and concerns

1. **Adopt FSDP2 as the mechanism** for all three modes (recommended), or
   write ZeRO-1/2 by hand.
2. **Gather dtype.** Keep FP32 gathers so numerics match replicated
   (recommended now), then measure a BF16 gather option.
3. **A 4-GPU window** for the unverified rows in section 1 and the measured
   comparison in section 7. Concern, unvalidated: `xccl` from confined processes.
4. **A dead rank hangs the others** until the 20-minute collective timeout. A
   watchdog that aborts the group when a worker exits is recommended before
   unattended multi-day runs. Unvalidated on hardware.
5. **Validation repeats on every rank**, which costs $N$ times the compute and
   no wall time; split the batches if validation becomes a large share.
6. **Sharded checkpoints** (section 5) before any run whose state does not fit
   the primary's RAM.
7. **Comparability.** `runtime.parallel` changes the config digest, so a sharded
   try is a different experiment from its replicated baseline in lab mode even
   though the update is the same up to summation order. Decide whether `try`
   should treat the mode as non-semantic.
8. **`gradient_accumulation` should be a multiple of the device count**; a
   remainder runs masked microbatches, which waste compute.

## 7. Measured comparison

Peak memory and targets per second for each mode, with and without
recomputation, on the same-shaped model, need all four cards. Not run yet;
the per-mode figures above are the projection.
