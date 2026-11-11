# Explicit target-group continuation

Whole-document training can explicitly change the target count per optimizer
update at an immutable checkpoint boundary. This is a scientific transition,
not a runtime batching optimization and not a fresh schedule. Fresh runs use
[`sparselab config schedule`](document-microbatching.md#author-a-fresh-schedule-from-prepared-groups).

## Checkpoint-bound contract

`training.matched_stories.target_group_transition` declares the parent checkpoint
digest, its committed token boundary and the previous targets-per-update setting:

```yaml
training:
  matched_stories:
    layout: story
    unpadded_documents: true
    targets_per_update: 131072
    document_batching: padded
    max_padded_slots: 8192
    max_attention_cells: 16777216
    target_group_transition:
      after_tokens: <checkpoint tokens_seen>
      previous_targets_per_update: <parent targets_per_update>
      parent_checkpoint_sha256: <immutable checkpoint digest>
```

The numbers above are illustrative workload settings, not a recommended model or
hardware budget. Choose and declare them for the intended experiment.

The native `MatchedStoryDataset` reconstructs the consumed groups using the old
setting and groups only the remaining suffix with the new setting. Existing
`frozen_update_ends` remain boundaries. Its `horizon(max_tokens)` gives the exact
new total update count; do not divide tokens by the nominal target-group size.
The saved `(epoch, next_group)` cursor and committed target count must agree with
both plans. The supported transition is within the first source-order pass of
whole, unpadded, all-token documents. Data replacement, repetition, skipped
examples, implicit epoch reset and cursor reinterpretation are refused.

The continuation validator checks the checkpoint binding, consumed prefix and
scientific settings before adoption. Model weights, Adam moments and parameter
steps, RNG, scaler, cumulative counters and checkpoint cadence watermarks are
restored. The new `training.max_steps` is the exact horizon of the declared plan;
`training.max_tokens` and data remain unchanged.

Regrouping preserves an already reached explicit scheduler decay floor. It does
not restart warmup, restore peak learning rate, rescale betas or change weight
decay. Larger groups mean fewer Adam and clipping operations per target and can
change the numerical trajectory. Keep optimizer, runtime and validation settings
unchanged unless their separate native transition contract explicitly allows the
change. Changes in validation or checkpoint cadence are not inferred by a planner.

## Use the native continuation path

Prepare a reviewed target configuration from the current native schema and exact
`MatchedStoryDataset` group boundaries. Use independently verified immutable
checkpoint/data identities, not a mutable `latest` selection made earlier. This
operation must not race an active owner of the same run or devices.

```sh
sparselab train target.yaml \
  --resume "$COMPLETED_GENERATION" \
  --run-id "$NEW_RUN_ID"
```

Use the normal runtime-profile and resource-envelope selectors for the selected
machine. If the code/runtime identity changed, the existing explicit
`--allow-runtime-drift` acknowledgment is also required; it is not evidence that
the changed execution is numerically equivalent or faster. Use native checkpoint
and evidence readers to verify the new run and subsequent resume. There is no
campaign-specific launcher or host-specific environment wrapper in this workflow.

`max_padded_slots` bounds B×L and `max_attention_cells` bounds B×L² in the native
collator. These limits preserve independent document rows and whole-update loss
normalization. They bound shapes, not physical accelerator residency or compiler
cache growth. Measure the chosen configuration on the intended runtime separately.

## Coverage and limits

The retained CPU tests cover source-order suffix coverage, frozen boundaries,
shape caps, saved cursor validation, refusal of changed scientific settings,
Adam/RNG/scheduler restoration and exact interrupted save/reload equivalence.
The document-batching tests separately cover logits, gradients, diagnostics and
normalization across padded microbatches. See `tests/test_target_group_transition.py`
and `tests/test_document_batching.py`.

Historical campaign sample counts, handoff paths, hardware choices and one-off
planning scripts are not generic engine evidence and are not shipped here. CPU
fixtures do not qualify real-corpus quality, accelerator performance, archival
safety or permission to start a run.
