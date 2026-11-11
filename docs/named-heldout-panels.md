# Named development heldout panels

Use one native evaluation invocation per fixed panel against the same immutable
checkpoint. Workshop owns panel definitions, cadence, outer wall/memory limits and
orchestration; Lab verifies inputs and computes the loss. No training configuration,
data cursor, checkpoint or source dataset is changed.

```sh
klab eval RUN_ID --runs-dir /work/runs --checkpoint step_00013042_gen_000001 \
  --panel /work/recipe/english.json --output /work/results/english-step13042.json
klab eval RUN_ID --runs-dir /work/runs --checkpoint step_00013042_gen_000001 \
  --panel /work/recipe/technical.json --output /work/results/technical-step13042.json
```

Each declaration pins the same checkpoint manifest SHA256. Checkpoint pointers
are accepted only if their resolved digest matches that pin; prefer an immutable
generation name. The existing `--backend`, `--runtime` and `--runtime-profile`
selectors still apply. Inference uses its existing FP32 evaluation behavior; this
is distinct from training BF16 autocast. This slice supports dense PyTorch models
without memory/allocation sidecars. Unsupported engines/models fail explicitly.

A declaration is strict JSON with these fields (values below are placeholders,
not an executable experiment or approval):

```json
{
  "schema_version": 1,
  "name": "english",
  "purpose": "development",
  "prepared_path": "../prepared/english",
  "prepared_manifest_sha256": "REPLACE_WITH_MANIFEST_IDENTITY",
  "tokenizer_file_sha256": "REPLACE_WITH_RUN_TOKENIZER_FILE_SHA256",
  "tokenizer_canonical_sha256": "REPLACE_WITH_PREPARED_TOKENIZER_IDENTITY",
  "checkpoint_sha256": "REPLACE_WITH_CHECKPOINT_MANIFEST_IDENTITY",
  "split": "validation",
  "sampling": "first_eligible_blocks",
  "sequence_length": 2048,
  "blocks": 128,
  "supervised_targets": 262144,
  "batch_size": 2,
  "max_forward_positions": 262144
}
```

`prepared_path` is relative to the declaration, or absolute. It must point to an
already prepared native directory: Lab never acquires or prepares data here.
Native deep verification authenticates the complete prepared inventory, including
its train member; account for that I/O separately from evaluation. The prepared
manifest identity is its `manifest_sha256` field, not the manifest file hash.
The tokenizer file hash and canonical tokenizer identity are distinct pins.

Only the prepared `validation` member is scored. `purpose` must be `development`;
this command does not open the lab's protected final holdout. Workshop must ensure
that input documents belong to the approved development split; labeling arbitrary
bytes as validation does not establish their scientific eligibility.

Selection is the first declared number of eligible full blocks in stored order,
without shuffling or repeats. Zero-target blocks are skipped using native mask
semantics. A trailing partial block is excluded. The exact supervised-target
count must match the declaration before any forward pass. A panel smaller than
requested, empty panel, invalid identity or missing file is incomplete, never a
successful short sample. With masks, choose the block count and measured target
count together; this command does not silently truncate or fill to a target cap.

Every forward's input positions are charged before the model call. The maximum
is `blocks * sequence_length`, which must fit `max_forward_positions`; padding
is not added to the last smaller batch. An existing owned attempt contract must
also permit this fixed-profile evaluation and each batch is reserved through the
native attempt ledger. These position limits do not replace external wall-time,
host-memory, GPU-memory or process-cleanup controls.

The output parent directory must already exist. The output file must be new and
outside both input roots; an existing result is never overwritten, even on replay.
Each result contains panel/protocol identity, authenticated validation/mask/packing
identities, native checkpoint/tokenizer/runtime identity and committed training
`tokens_seen`, plus NLL sum, actual supervised targets, loss, perplexity, completed
batches, attempted forward positions and evaluation/total elapsed seconds.

Exit status is zero only for complete coverage. A caught failure/interruption
retains completed-batch NLL/counts, includes the failed forward in attempted
positions, leaves `loss` null and exits nonzero. A reservation failure before a
forward is not counted as an attempted forward. Pre-invocation schema/CLI errors
refuse before creating a result. A durable initial `incomplete` record has
`accounting_complete=false`; after an uncatchable termination, its zero counters
are not evidence that no work occurred. Preserve the external supervisor's exit
and logs. Never treat an absent/skipped result as zero loss or complete coverage.

A panel is comparable only when both results are complete and their
`comparison_group` values match. That group binds the panel protocol and actual
runtime identity; checkpoint identity stays separate so a trajectory can compare
checkpoints. Locations do not affect the protocol. A changed mask, dataset,
tokenizer, selection, batch or budget changes it. Timings are observations and
need not reproduce byte-for-byte. No cross-panel aggregate, automatic comparison,
bootstrap, scientific stopping rule or scheduler is introduced.

Within an attempt ledger, replaying the same panel and checkpoint needs a new
owned evaluation phase: reservation labels are unique within a phase, so repeating
the same invocation there refuses at the first duplicate forward reservation.
