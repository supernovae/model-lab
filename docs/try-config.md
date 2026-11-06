# Try declarations

The default loop is unchanged: `sparselab try DELTA.yaml --vs BASELINE.yaml`.
Five executable options remain: `--vs`, `--config`, `--backend`, `--runtime`,
`--json`. `--config` selects `fast` (default), `standard`, `full`, `no-probes`,
or a YAML path. Presets use the same validated schema as files. Preset names
win over same-named files; use `./standard` to select such a file explicitly.
Empty, comment-only, null and `{}` files select default policy. Unknown fields,
wrong versions, coercible seeds/booleans and unsupported probe tiers fail before
runtime lookup or output creation.

```yaml
try_config_version: 1
seed: 7
probe_tier: standard
fresh_baseline: false
repeat_baseline: false
lab_dir: ./lab
runtime_profile: ./cpu-profile.yaml
resource_envelope: ./bounds.yaml
```

All fields are optional. `seed` is an authoritative paired override for both
arms. Run configs and try overrides accept seeds from `0` through `4294967295`
(`2**32 - 1`), inclusive, on both PyTorch and MLX. Values outside this range
are rejected, never wrapped or normalized. Try checks declared seeds before
creating its attempt directory or preparing either arm. Without an override,
native candidate derivation retains a delta's explicit seed change.
`fresh_baseline` bypasses reuse; ordinary reuse still requires a completed try
with matching config, code and current tokenizer/data/memory identities.
`probe_tier` selects held-out screening, never the protected final split.

Paths expand `~` and resolve relative to the declaration file. Select either
`runtime_profile` or `--runtime ID`; using both fails. Native profile loading,
interpreter checks, explicit device selection and authorization remain intact.
`--backend` retains the native explicit override behavior.

Runtime bounds belong in the existing resource envelope, separate from science:

```yaml
resource_envelope_version: 1
max_rss_bytes: 2147483648
lab_try:
  max_wall_seconds_per_arm: 600
  tokenizer_batch_documents: 32
  tokenizer_batch_source_bytes: 1048576
```

`lab_try` is optional and applies to try arms. Wall limits must be positive and
finite; there is no implicit timeout. Batching uses strict integers: documents
1–256 and source bytes 1–8388608. Omitted batching uses native defaults (256
and 1048576); smaller batches bound transient preparation memory. Direct API
limits remain supported and the tighter of explicit and declared limits wins.
Native trainer preparation, checkpointed wall stops, resource preflight,
cancellation and accounting retain their current behavior. Other envelope consumers reject a non-null `lab_try` section before execution,
including an empty section. Remove it when using the envelope outside try. Try
resolves these limits into native trainer arguments and delegates only the
remaining common host/storage bounds; sealed records retain the full original
envelope. Absent `lab_try` preserves legacy serialization.

The removed `--seed`, `--probe-tier`, `--fresh-baseline`, `--lab-dir`,
`--runtime-profile`, `--resource-envelope`, `--max-wall-seconds` and
`--tokenizer-batch-*` flags are rejected. Replace them with declaration fields
and envelope limits above. `--work-dir` remains the global state selector.

Try seals the validated effective request, loaded profile/envelope values and
source paths before setup. Files can change later without changing that record.
Resolved wall/batching limits are also recorded under `resources` when setup
reaches execution. `report --json` returns the entire sealed record, including
failed/interrupted results and missing observations; historical records remain
readable without rewriting them. Native checkpoint/digest binding and held-out
scoring are unchanged. One paired seed is still insufficient for a scientific
claim; this migration does not add across-seed aggregation or dashboard replay
of custom settings.

## Baseline/candidate/baseline bracket

Set `repeat_baseline: true` to execute three independent native training/scoring
arms in that order. Both baselines are fresh even when a reusable baseline exists.
The final baseline uses the identical config, seed, prepared inputs and evaluation
protocol as the first; it does not continue from the candidate. The sealed record
retains all three arms and a separate `baseline_repeat_comparison`. Per-arm limits
and safe cancellation apply to the third arm too. A failure or cancellation keeps
completed arms and withholds an overall completed result.

`train_seconds` is end-to-end native training wall time, including preparation,
compilation, evaluation during training and checkpoint writes, and excluding the
subsequent scoring pass. Do not call it steady-state device throughput or compare
it with profiler timing. Arms share one Python process and its compiler/cache
scope; A2 may be warm. There is no automatic cache reset or cold-start equivalence
claim. Record runtime/cache policy separately for a performance experiment.
This option adds drift/repeatability evidence; it does
not infer a speed winner, choose a tolerance or replace separate trace runs.

Without `continuation`, each arm starts from the declared initial model. For a
same-checkpoint full-state bracket, use the native terminal budget extension:

```yaml
repeat_baseline: true
probe_tier: none
continuation:
  checkpoint: /absolute/parent/checkpoints/step_00023691_gen_000001
  append_data: true
  allow_runtime_optimization: true
  allow_runtime_drift: false
```

Replace the example generation with a verified actual terminal generation. The
baseline config declares strictly larger whole-update caps and the native-verified
append-only dataset. A 24-update test sets `max_steps` to parent step + 24 and
`max_tokens` to the corresponding native group offset. The candidate delta changes
only a native-allowed runtime setting, such as `training.matched_stories.padding_multiple`.
Every arm invokes native `extend_budget` from that same parent, preserving its
model, Adam, RNG, cursor and schedule. All baselines are fresh. The sealed record
retains the declared continuation and parent manifest hash, and native run lineage
records retain the verified parent checkpoint identity. Native parent-terminal,
append-prefix, scientific-setting and full-state checks remain authoritative.

`allow_runtime_drift` defaults false; if an experiment uses a changed implementation,
explicitly record and authorize that change before setting it true. This is not a
model-weights-only fork. Use the existing resource envelope for per-arm wall bounds;
a checkpointed wall stop remains interrupted, not a successful throughput result.
The caller must arrange exclusive device ownership and production fallback before
invoking this workflow; try does not interrupt or resume another trainer.

For an explicitly selected nonterminal checkpoint, `append_from_checkpoint: true`
requires `append_data: true` and retains native consumed-prefix/cursor verification.
The bounded target dataset must cover exactly its declared new single-pass budget;
limiting training caps alone while retaining a larger prepared stream is refused.
Prepare that bounded dataset once with native `prepare_data`, then reuse its cache
for all arms. This feature does not supply a zero-tokenization cache-slicing API.
