# Checkpoint exposure

New training checkpoints seal version 1 `exposure` metadata in their existing
manifest. Read it through the normal, read-only public report:

```sh
klab report RUN_ID --kind run --json
```

The JSON field `evidence.checkpoints[].exposure` contains the prepared-data supply,
committed supervised-target count, effective passes, loader cursor, identities and
timing. Historical checkpoints without this metadata report `null`; reporting
never reconstructs missing history, trains a model or rewrites a checkpoint.
No additional forward pass or evaluation is introduced.

`usable_supervised_targets_per_pass` counts actual next-token labels in full
`TokenBlockDataset` blocks, after the supervision mask. Position zero is context,
not a target. Trailing positions outside full blocks are excluded; blocks with no
supervised targets are skipped. Raw packed positions, full/eligible/skipped blocks
and trailing target positions remain separate counts. `trailing_target_positions`
counts positions regardless of the supervision mask, not supervised targets.
Prepared manifest, packed training tokens, tokenizer and optional supervision-mask
digests come from the
existing owned artifact inventory. The exposure protocol also binds sequence
length, shuffle seed and target-count semantics.

`committed_supervised_targets` reuses the checkpoint's `tokens_seen` counter.
Only applied optimizer updates advance it; overflow retries never double count.
`effective_passes` is that count divided by usable supervised targets per pass.
This measures exposure to prepared data, not deduplicated content or overfitting.

Both raw cursor fields and normalized completed loader passes/next eligible block
are retained. A budget may mask some labels in the last selected block while the
cursor advances beyond that block. Completed loader passes therefore do not prove
that every label was consumed. `finer_consumed_positions` stays unknown (`null`).
Full-state resume keeps cumulative counters and the known origin run; a promoted
warm start starts a new exposure scope. Missing historical origin metadata stays
unknown even when the existing committed counter is available.

Timing fields have distinct scopes:

- `optimizer_update_seconds`: existing cumulative engine-measured applied-update
  time, including overflow attempts that eventually applied. It excludes wholly
  failed updates, validation, preparation and checkpoint writing.
- `training_loop_elapsed_seconds`: the existing cumulative loop wall counter,
  including inline validation/reporting and earlier checkpoint writes. It excludes
  preparation and the checkpoint currently being written.
- `invocation_elapsed_seconds_at_checkpoint`: measured elapsed time in this
  invocation through the start of this checkpoint, including preparation; it is
  not a cumulative total across resumed invocations.
- `other_training_overhead_seconds`: unknown. No residual is presented as a
  measured overhead. Existing final runtime phase reports remain separate.

On resume, cumulative timing is available only when its prior metadata and native
counter are available. Otherwise it remains null through subsequent resumes; a
missing historical time is never silently treated as zero in this metadata.

Stage A does not change repetition caps or sampling, add train-reference scoring,
consume a final holdout, infer an overfitting threshold, or run an overlap audit.
Those require their own protocols and execution approval.
