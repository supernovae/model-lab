# Explicit larger-target continuation

This is staged implementation for the Devmind run based on `f515057`. No live
training source, config, process, GPU allocation or checkpoint tensor was changed
or loaded during implementation. Storage migration is separate.

The recommended eventual target is **131072 supervised targets per optimizer
update**, restoring the intended effective-group scale. Keep 4096 context, up to
four independent document rows, dynamic transformer and output-head/loss compilation,
activation checkpointing off, private Intel 26.40 and optimizer/clipping
`foreach=False`. The new collator also caps `B*L <= 8192` and `B*L*L <= 16777216`.
These caps retain four rows for shorter documents and reduce rows for longer ones.
They bound activation-related shapes; they do not prove allocator or compiler-cache
memory residency will fit. GPU execution remains untested.

## Migration contract

The opt-in `training.matched_stories.target_group_transition` records the parent
checkpoint SHA256, its committed token boundary and the previous target budget.
The data plan reconstructs the historical groups with the previous budget up to
that boundary; only the remaining suffix uses the new budget. Existing frozen
curriculum boundaries remain group ends. The unchanged `(epoch, next_group)` cursor
is checked against both old and new plans and the checkpoint's committed targets.
No group-index reinterpretation, document repetition, skipping, data replacement,
or implicit epoch reset occurs. This initial implementation supports one transition
within the first source-order pass, for whole unpadded all-token documents.

The native continuation loader restores model, Adam moments and per-parameter step
counts, RNG, scaler, cumulative counters and cadence watermarks. It records an
explicit scientific transition. The global update count continues from the parent;
`max_steps` becomes old completed updates plus the exact new suffix horizon. Token
budget and data/curriculum stay fixed. Sorting still happens only inside each update;
this was already present before the change. All microbatch losses divide by the
whole new update's valid-target total, followed by one clipping/AdamW step.

This is **not** equivalent to doing the old small updates. Adam beta decay and
bias correction advance once per larger group; weight decay and gradient clipping
also happen less often per token. The RNG starts from the saved state, but shape
and reduction changes can change subsequent consumption and numerical trajectory.
No beta, LR or weight-decay rescaling is implied.

The supported migration preserves an already reached, explicit decay floor. For
this source, warmup=10 and decay_steps=408 remain unchanged, with backbone LR=1e-5
and table LR=5e-5. There is no warmup restart or peak-LR restoration. Choosing a
new learning schedule is a separate scientific decision, not a performance fix.
The config comparison prevents silently changing runtime, optimizer, evaluation,
data or curriculum during regrouping. Old configurations keep old behavior.

Periodic checkpointing in the planner changes from every 1000 updates to every
5,000,000 targets, approximately preserving the former recovery distance. Existing
cadence watermarks are carried forward; the next update can therefore be due
immediately. Best, terminal, interruption, and recovery saves remain unchanged.
Evaluation remains every 100 absolute updates, so its token spacing grows to about
13M targets for 128K groups (previously about 0.49M). This is a deliberate documented
consequence; choose a separate evaluation-cadence policy before adoption if that
spacing is unwanted. The implementation does not silently alter validation policy.

## Bounded sample evidence

`evidence/target-group-sample.json` and `evidence/simulate_target_groups.py` record
an at-most-8-MiB read of prepared tokens starting at offset 61,981,272, the next
unconsumed token in completed checkpoint `step_00012500_gen_000005`. It contains
2,096,919 targets after trimming the final incomplete document. This is a local
suffix sample, not the full corpus and not a GPU speed measurement. Current-group
baseline uses the existing 128 alignment and width four. Larger candidates use
that same alignment/width plus the explicit shape caps.

| Goal | Updates | Mean targets/update | Valid slots | Padded slots | Padding | Update reduction |
|---|---:|---:|---:|---:|---:|---:|
| 4096 | 435 | 4,820.5 | 2,096,919 | 1,457,257 | 41.00% | 1x |
| 32768 | 63 | 33,284.4 | 2,096,919 | 330,985 | 13.63% | 6.90x |
| 65536 | 32 | 65,528.7 | 2,096,919 | 251,881 | 10.72% | 13.59x |
| 131072 | 16 | 131,057.4 | 2,096,919 | 203,625 | 8.85% | 27.19x |

All sample cases reached B=4 and L=4096 separately. The baseline peaks were
B*L=16384 and B*L²=67108864; every larger candidate stayed at 8192 and 16777216.
Peak microbatches/update were 3, 14, 26 and 47 respectively. The final sample group
is partial, hence means need not exceed goals. JSON retains min/median/max and shape
counts. 128K best matches the original intended scale and has the lowest sampled
padding and optimizer frequency; 32K is an available conservative trajectory change,
not an established GPU optimum. No targets/s or peak-FLOPs speed claim follows.

## Preparing and adopting later

Do not run the training command concurrently with the active run. First coordinate
a completed-checkpoint handoff through the owner. Choose the final checkpoint at
that handoff, not the older sample checkpoint (which would repeat subsequent work).
The planner reads the full prepared token stream to compute the exact horizon;
one such scan was subsequently performed against the existing NVMe prepared copy
for the staged step-13500 configuration. Refresh it for the actual handoff.
It opens no model/optimizer tensors and never initializes XPU. It was tested against
a tiny native completed-checkpoint fixture.

From this worktree, using the provisioned Python environment and `PYTHONPATH=src`:

```sh
python -B tools/plan_target_groups.py \
  --config /home/byron/Documents/Codex/2026-11-04/task-9/compiled-launch/continuation.yaml \
  --checkpoint "$COMPLETED_GENERATION" \
  --targets 131072 --output "$FUTURE_CONFIG"
```

The generated complete YAML preserves the current runtime choices and fills the
exact prefix provenance, target budget, shape caps and new max_steps. It does not
change the live YAML or choose storage destinations. The staged complete YAML is bound to step 13500 and must be regenerated
if the handoff selects a later checkpoint. The relevant new declaration is:

```yaml
training:
  matched_stories:
    targets_per_update: 131072
    max_padded_slots: 8192
    max_attention_cells: 16777216
    target_group_transition:
      after_tokens: <completed checkpoint tokens_seen>
      previous_targets_per_update: 4096
      parent_checkpoint_sha256: <completed generation digest>
checkpoint:
  every_steps: null
  every_tokens: 5000000
```

Use the existing owned launcher with this worktree, its private Intel26.40 runtime,
new cache scope, device lease, physical-memory budget and new run identity. Its
native training call becomes:

```python
train(config, resume=completed_generation, run_id=new_run_id,
      stop_after_step=parent_step + 3, allow_runtime_drift=True,
      worker_id=owned_worker, authorization=authorization,
      resource_envelope=envelope, checkpoint_committed=committed)
```

This is the smallest suggested real adoption: three actual updates, committed
checkpoint, then full-state reload for continuation. Inspect observed physical
memory, compiler specialization/cache growth, loss and update timing at this new
grouping. Do not reset or replay the three updates. The corresponding native CLI is
`sparselab train "$FUTURE_CONFIG" --resume "$COMPLETED_GENERATION"
--run-id NEW_ID --allow-runtime-drift --stop-after-step PARENT_PLUS_3`, with the same
runtime profile, resource envelope and owned launch environment. The code revision
change requires existing explicit runtime-drift acknowledgment; regrouping itself
is declared by the config policy, not disguised as runtime optimization.

## Reporting inspection and validation

PyTorch accumulates CE/auxiliary/diagnostic tensors across microbatches and reads
scalars at the update boundary. Its finite-loss/gradient synchronization protects
optimizer mutation. Trainer `store.log_metrics` and `progress()` run every update;
`logging.every_steps=10` does not throttle this path. Larger groups reduce how often
these operations happen per target. Architecture diagnostics still run per forward,
and more microbatches can still incur dispatch/progress overhead. No telemetry,
fsync, finite-check or checkpoint barrier was removed; attribution of GPU dips
requires a later actual trace.

Focused CPU tests cover source-order suffix coverage, masks, curriculum boundaries,
shape caps, whole-update normalization/gradients including Engram, unchanged old
batching, native Adam/RNG/scheduler/cursor save-reload equality, source-setting
refusals, planner output and existing document-budget extensions. The broad suite,
GPU memory validation and real-training comparison were not run. The later NVMe
follow-up did run the full token-only planner scan, without checkpoint tensor loads. See the handoff for exact test results.

## NVMe preparation follow-up

The subsequent explicit NVMe request produced a complete staged YAML and launcher
at `/home/byron/Documents/Codex/2026-11-04/task-12/nvme-handoff/`. A single sequential
scan of the existing NVMe prepared tokens computed 587 remaining 131072-target
updates from step 13500 (66,864,692 tokens), with terminal step 14087. The source
run remains active; refresh the selected generation/config at the actual handoff.

The planner now accepts `--runs-dir` and `--train-tokens`. The latter selects an
existing prepared copy for CPU planning only; native continuation still verifies
its parent-bound data. Native resume maps the parent run's data, so the immutable
selected parent and its bound artifacts must be copied to NVMe before launching
if every resumed update must read from NVMe. No copy or launch was performed here.

The staged config preserves all periodic checkpoints. Native local retention does
not archive before deleting, and the separate async archive prototype remains WIP.
Observed ~889 GiB free NVMe covers a conservative 160 GiB finite-run working budget.
Automatic verified archival/eviction remains separate pending work; no checkpoint
was deleted or claimed archived. See `nvme-handoff/HANDOFF.md` for the exact paths,
capacity assumptions and launcher command.
