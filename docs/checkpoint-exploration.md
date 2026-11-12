# Explore retained checkpoints

Commands below use the [activated, provisioned environment](../README.md#first-run).

Use the native lab interfaces to inspect and compare your own retained models.
The [model exploration guide](tinytext-model-guide.md) covers the complete
verify → raw completion → chat workflow and debugging controls. For client
applications use the [local API guide](local-api.md); for the retained DevMind
base checkpoint see [MODEL-0](history/model-0.md).
For a new training run, start with the [TinyStories microlab](tinystories-microlab.md).
Existing weights, run-owned assets and a compatible runtime must be available;
commands cannot reconstruct a missing checkpoint from its metadata.

## Verify and explore

Set `GENERATION_A` and `GENERATION_B` to two immutable `step_*_gen_*`
checkpoint directories. Use fresh output under your external work root.
The example selects CPU; accelerator execution needs its provisioned environment
and the supported [runtime setup](runtime.md).

```sh
klab checkpoint verify "$GENERATION_A" --json
klab checkpoint verify "$GENERATION_B" --json
klab surface chat \
  --cell "first=$GENERATION_A" --cell "second=$GENERATION_B" \
  --backend cpu --seed 11 \
  --work-dir "$KLAB_WORK_DIR/experiments/checkpoint-exploration"
```

Surface chat uses each model's native tokenizer and shows anonymous response
cards. Saved exploratory replies and votes are not sealed evaluation evidence.
Different data, tokenizer, size, context or precision prevent attributing a
response difference to a single architectural feature.

## Review existing outputs

For a sealed comparison, use `klab surface import triage` on compatible
retained triage outputs, then `surface review` on the new bundle. The
[Surface Review guide](research/surface-review-v1.md) owns the complete commands,
required inputs, study-specific importers, privacy limits and verified dashboard
overlay. Imports verify existing output bytes and do not regenerate missing text.
One person's review does not establish population preference or promote a model.

For historical findings and checkpoint-bound records, consult the
[experiment ledger](research/experiment-ledger.md). Historical comparator scripts
remain with their original research packets; they are not the general lab workflow.
