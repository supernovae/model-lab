# Whole-document batching

By default Lab cuts the packed token stream into fixed blocks of
`training.seq_len` and shuffles them, so a block can start mid-document and span
several documents. `training.matched_stories` instead groups whole documents
into updates. Use it when document boundaries matter, or when two arms must see
exactly the same documents in exactly the same updates.

It needs plain all-token data delimited by EOS (EOS must not appear inside the
source text). Models with n-gram memory need `unpadded_documents: true`; other
memory kinds are not supported.

## Groups

Documents are read in source order, repeating that order each pass. An update's
group closes at the first document end that reaches `targets_per_update` (4096
by default), so a long document makes a larger group: the setting is a target,
not an exact batch size. `micro_batch_size` splits a group's rows into
microbatches; `gradient_accumulation` does not size groups. Both arms of a
comparison get the same group at every update.

```yaml
training:
  seq_len: 512
  micro_batch_size: 8
  gradient_accumulation: 1
  max_tokens: 100003840
  max_steps: HORIZON        # from klab config schedule, below
  matched_stories:
    layout: story
    targets_per_update: 4096
```

`layout` decides how a group becomes rows:

- `story`: each document starts its own right-padded row; a long document
  continues in chunks of at most `seq_len` targets, each with the preceding
  token as context.
- `contiguous`: rows are filled across document boundaries within the group.

Every stored token is a target, including the first token (which sees a virtual
preceding EOS) and every EOS. Padding has label `-100` and does not count toward
the budget. The loss is the sum over all targets in the update divided by their
count.

## Unpadded documents and padded execution

`unpadded_documents: true` emits each document whole and untruncated, with
evaluation at batch size one. Add `document_batching: padded` to run them in
batches:

```yaml
training:
  micro_batch_size: 4
  matched_stories:
    layout: story
    targets_per_update: 4096
    unpadded_documents: true
    document_batching: padded
    padding_multiple: 128
    max_padded_slots: 8192
    max_attention_cells: 16777216
evaluation:
  batch_size: 1
```

Within each update, documents are sorted by length and right-padded into
independent rows; no document crosses an update. Row positions, n-gram and
convolution state restart per document, and causal attention never reads the
padding to its right. Sequence length rounds up to `padding_multiple`, capped at
`seq_len`. `max_padded_slots` caps batch × length and `max_attention_cells` caps
batch × length². These bound shapes, not device memory; measure the
configuration on your hardware.

Padded batching needs dense attention and feed-forward, the PyTorch engine, and
`gradient_accumulation: 1`. The default, `document_batching: separate`, runs one
document per microbatch.

With padded batching, `batch/document_diagnostics_version` is 2: Engram lookup
counts are summed per document, gate and norm statistics are weighted by real
positions (masked prompt tokens included, padding excluded), and
`batch/compute_slots` and `batch/real_positions` count padded and real positions.

## Schedule from the prepared groups

Because groups vary in size, do not compute the number of updates by dividing
tokens by `targets_per_update`. Derive it from the prepared data:

```sh
klab data prepared-inputs publish packing.yaml --output prepared-packing
klab config schedule packing.yaml --prepared-inputs prepared-packing --output final.yaml
klab data prepared-inputs publish final.yaml --output prepared-final
klab data prepared-inputs verify final.yaml prepared-final
```

`config schedule` verifies the prepared inputs and writes a new config whose
`training.max_steps` and AdamW `optimizer.decay_steps` match the exact number of
groups in the token budget. It keeps the budget, warmup and packing settings,
updates checkpoint retention boundaries if configured, and refuses a budget
larger than the prepared data, a budget that ends mid-document, or a horizon that
conflicts with warmup. Publish the inputs again for the new config, since they
are bound to it.

`frozen_update_ends` fixes group boundaries at given token counts; schedules and
continuations keep them.

## Change the group size at a checkpoint

`target_group_transition` changes `targets_per_update` partway through a run:

```yaml
training:
  matched_stories:
    layout: story
    unpadded_documents: true
    targets_per_update: 131072
    document_batching: padded
    target_group_transition:
      after_tokens: <the checkpoint's tokens_seen>
      previous_targets_per_update: <the parent's targets_per_update>
      parent_checkpoint_sha256: <the checkpoint's digest>
```

```sh
klab train target.yaml --resume "$CHECKPOINT_GENERATION" --run-id "$NEW_RUN_ID"
```

Lab rebuilds the consumed groups with the old size, groups the rest with the new
one and checks that the checkpoint's cursor and token count agree with both. It
restores weights, optimizer state, RNG and counters, and sets `max_steps` to the
new exact horizon. The learning rate schedule continues where it was; warmup does
not restart. The transition must happen within the first pass of the data.
Larger groups mean fewer optimizer steps per token, which can change the
trajectory.

Changing only how documents are batched (for example `padding_multiple`) on
resume is a runtime change: pass `--allow-runtime-optimization`, which checks
that data, masks, groups, budgets, optimizer and evaluation are unchanged and
records the change. See [checkpointing](checkpointing.md).

## Append data to a run

`train TARGET --extend-budget GENERATION --append-data` continues an unpadded,
single-pass parent on a larger, already prepared `local_text` stream. The new
stream must keep every trained token in order, every earlier update boundary and
the same validation tokens. Set `training.matched_stories.frozen_update_ends` to
the old stream length (plus any earlier frozen ends) so the parent's last update
is never merged with new records. The budget includes the consumed prefix;
training starts after it. Derive caps with `MatchedStoryDataset.horizon` or
`targets_for_updates`, keep the parent's `optimizer.decay_steps`, and the
schedule continues at its floor without a new warmup.

This is a full-state continuation: optimizer moments, random state, cursor and
counters carry over. After its first checkpoint, continue with ordinary
`--resume`; do not repeat `--append-data` without another extension.

To start the new data before the parent's planned budget finishes, add
`--append-from-checkpoint`. It accepts a verified checkpoint at a complete
update, and the new caps must lie beyond it. The decision records the consumed
counters and the source tokens left unconsumed.

`--allow-runtime-optimization` can be combined with either form; both
validators run, and the child records both decisions. See
[resume](checkpointing.md#resume).
