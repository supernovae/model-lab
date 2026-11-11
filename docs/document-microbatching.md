# Independent document microbatches

Whole-document matched groups can opt into bounded padded-row execution:

```yaml
training:
  micro_batch_size: 4
  gradient_accumulation: 1
  matched_stories:
    layout: story
    targets_per_update: 4096
    unpadded_documents: true
    document_batching: padded
    padding_multiple: 128
evaluation:
  batch_size: 1
```

`unpadded_documents` continues to mean the dataset emits entire, untruncated
individual documents. `document_batching: padded` changes only their execution:
stable sort by length inside each original optimizer-update group, then right-pad
independent rows. No document crosses an update boundary. The original group
cursor, virtual preceding EOS, labels, row-local positional indices and
ngram/convolution resets are preserved. Causal attention never reads padding to
the right. Input positions are distinct from supervised target positions; masked
prompt tokens remain context. Loss is summed across all valid targets and divided
by the original update's total once. The maximum row count does not multiply the
optimizer batch or its scheduler steps. This path requires dense attention and
FFN, PyTorch, none/ngram memory, and no allocation/portability sidecars. Evaluation
remains separate-document batch one.

Sequence dimensions round up to `padding_multiple`, capped at `training.seq_len`.
There are no dummy rows: final microbatches may be partial. This bounds shapes but
does not guarantee compiler cache behavior or memory fit. Use native batch
telemetry to measure the chosen workload and runtime. Default `document_batching: separate` retains
legacy mb1 execution and diagnostic semantics; omitted new defaults preserve
historical config identities.

Opt-in telemetry records `batch/document_diagnostics_version = 2`. Engram lookup,
unique-address and collision counts are sums over documents (an address shared by
two documents counts once in each). Gate and norm statistics weight real input
positions, including masked prompt positions and excluding padding. Occupancy and
maximum-address fraction are real-position-weighted document statistics, averaged
across streams/ports as documented by their aggregate definitions; they are not
whole-update distinct-address counts or whole-update maxima. This definition is
invariant to partitioning documents into microbatches. Empty-supervision documents
still contribute real-position diagnostics. `batch/compute_slots` counts all padded
input slots and `batch/real_positions` counts true document positions. Legacy v1
metrics must not be compared as if they used the v2 definition.

A full-state change requires native `--allow-runtime-optimization`, which records
the parent checkpoint and explicit execution deltas with `best_effort` numerical
continuity. For these variable groups, effective batch is the original immutable
group definition, not microbatch × accumulation. Data, masks, group boundaries,
budgets, optimizer, LR and evaluation settings remain checked; changing any of
those is not admitted as a batching optimization. New implementation source also
requires the existing explicit `--allow-runtime-drift` path. Budget extension and
runtime optimization remain separate supported transitions; their mutual-exclusion
rule is not bypassed. Defaults, historical identities and immutable parent files
are not rewritten.

CPU tests cover Engram off/on with two convolved mapped ngram ports, reference/SDPA
attention, mb1/4/8, ragged and masked documents, all-masked documents, partial groups,
4096-token context, causality/isolation, all parameter gradients, AdamW steps,
diagnostics, native train/save/resume, group cursors and explicit transitions.
FP32 logits use rtol 3e-5/atol 3e-6; gradients 2e-4/3e-6; direct AdamW parameters
3e-5/2e-6. The native engine allows atol 5e-6 (observed 3.59e-6 near a zero gradient
through Adam's epsilon division). Summed CE uses rtol 2e-6. Same-path checkpoint
resume remains bitwise tested; different batching is numerically, not bitwise,
equivalent. CPU success does not certify accelerator BF16 numerics or performance.


## Author a fresh schedule from prepared groups

Do not estimate whole-document updates by dividing targets by the nominal group
size. Documents can overshoot a group target. After publishing and cold-verifying
prepared inputs for a valid packing config, use:

```sh
sparselab config schedule packing.yaml --prepared-inputs prepared-packing --output final.yaml
sparselab data prepared-inputs publish final.yaml --output prepared-final
sparselab data prepared-inputs verify final.yaml prepared-final
```

This command cold-verifies the original bundle and derives `training.max_steps`
and AdamW `optimizer.decay_steps` from native `MatchedStoryDataset.horizon` over
the actual token array. It preserves the declared target budget, warmup, source
identity and packing settings. If retention is configured, it updates terminal
boundaries and each existing milestone's target count from the actual groups;
milestone steps beyond the new horizon are refused. Whole-document target budgets
must end on a document boundary. A horizon incompatible with warmup or stable
steps is rejected rather than silently changing those settings.

The output is a new config with a native derivation receipt. The JSON result
includes the original prepared-input hash, group count and exact horizon. The
original bundle remains immutable and bound to its original config: publish a
new final-config bundle (reusing the native prepared cache) before execution.
The command reports committed targets and the final group's complete, consumed
and clipped target counts. A target budget larger than the prepared train supply
is rejected: fresh scheduling never implicitly repeats data. Checkpoint-bound
`target_group_transition` is rejected; use native continuation for that workflow.
Explicit `frozen_update_ends` are retained as part of the declared fresh packing
layout, rather than inventing or erasing document-group boundaries.
This authors a fresh schedule, not a checkpoint continuation.

Fixed validation accepts any number of uniquely named panels, including a
five-pillar evaluation. Each panel retains its own immutable selection and
forward/target budget; duplicate panel names are rejected.
