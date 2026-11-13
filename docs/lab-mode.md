# Lab mode

Lab mode answers one question at a time: does this change help? One command
trains a baseline and a candidate, scores both on the same held-out split, runs
behavioral probes and writes one sealed record.

```sh
cat > wider-ffn.yaml <<'YAML'
question: Does a wider feed-forward lower held-out loss at a fixed token budget?
set:
  model.ffn_dim: 128
YAML
klab try wider-ffn.yaml --vs BASELINE.yaml
klab report TRY_ID
```

`BASELINE.yaml` is any run config whose tokenizer exists. The candidate is a
**lab delta** (`set:` with dotted fields and an optional `question:`) or a whole
run config. The output looks like:

```text
LAB TRY try-20261010T052417Z-b9361b7d  status: completed  (lab mode)
  question: Does a wider feed-forward lower held-out loss at a fixed token budget?
  delta model.ffn_dim: 96 -> 128
  baseline  lab-base-759596ad32b5bf50  held-out loss 3.8764  ppl 48.25  targets 512
  candidate lab-try-20261010T052417Z-b9361b7d  held-out loss 3.8451  ppl 46.76  targets 512
  verdict: CANDIDATE_LOWER_LOSS  Δ held-out loss -0.0313
```

`report TRY_ID` prints the record again without computing anything; `--json`
gives all of it.

## Probes

After scoring, `try` runs the fast [probe](probes.md) tier on both arms: held-out
loss with a standard error, calibration, top-1 agreement and degeneration. It
prints a verdict and a next action. `--config standard` or `--config full` runs
more probes and `--config no-probes` skips them.

A probe that cannot run (an optional dependency missing, an error, a resource
stop) makes the battery `incomplete`, with action `rerun` and the gaps listed in
`probe.verdict.missing`. It never fails the try and never counts as a pass.

Before probing, `try` screens each arm's training data for the held-out and
final probe items ([leakage](probes.md#held-out-leakage)) and records the
result under `leakage`. Contaminated data also makes the verdict `rerun`.

## What is kept the same

Both arms are scored under one evaluation protocol: the baseline's context
length, batch size and batch count over the same tokenized validation data. Each
arm records its protocol: split, validation and tokenizer digests, packing
version, loss mode and loss-mask digest. The comparison also requires the same
training data unless the delta changes `dataset` or `tokenizer`, and a
validation split distinct from training. Anything else makes the verdict
`NOT_COMPARABLE` (exit 3) with no delta. So does a candidate that cannot read the
baseline's evaluation context.

A delta that changes the tokenizer cannot keep the tokenized validation data,
so it is compared on [bits per byte](evaluation.md#bits-per-byte) instead
(`comparison.metric: bits_per_byte`). Set `evaluation.text_documents` in the
baseline; both arms then score that many held-out documents, and the comparison
requires the same text (`text_sha256` and byte count) in place of the same
tokens. Without it, a tokenizer change is `NOT_COMPARABLE`. Arms with the same
tokenizer still compare held-out loss (`metric: loss`) and, when text was
scored, also report `bits_per_byte_delta`. `comparison.heldout_delta` is
candidate minus baseline in the comparison's metric.

## Reusing a baseline

A baseline is reused only from an earlier completed try with the same effective
config, the same Lab source, and the same tokenizer, data files, memory packages
and parent or teacher checkpoints, all hashed now. Editing any of them retrains
it; interrupted and failed tries are never reused. `fresh_baseline: true` always
retrains.

## Declarations

`--config` takes a preset (`fast`, `standard`, `full`, `no-probes`) or a YAML
file; `./standard` selects a file named like a preset. Every field is optional:

```yaml
try_config_version: 1
seed: 7
probe_tier: standard          # none, fast, standard or full
fresh_baseline: false
repeat_baseline: false
lab_dir: ./lab
runtime_profile: ./cpu-profile.yaml
resource_envelope: ./bounds.yaml
```

- **`seed`** applies to both arms, overriding any seed in the baseline or the
  delta (0 to 2³²−1). Without it, a delta that sets `seed` makes the seed a
  changed variable. The record shows which.
- **Paths** resolve relative to the declaration.
- **Runtime.** Use `runtime_profile` or `--runtime ID`, not both. `--backend`
  overrides the backend.
- **Resource limits** go in the envelope:

  ```yaml
  resource_envelope_version: 1
  max_rss_bytes: 2147483648
  lab_try:
    max_wall_seconds_per_arm: 600
    tokenizer_batch_documents: 32
    tokenizer_batch_source_bytes: 1048576
  ```

  There is no default wall-time limit. Smaller tokenizer batches lower peak
  preparation memory.

### Repeat the baseline

`repeat_baseline: true` trains three fresh arms (baseline, candidate, baseline
again) with identical inputs for the two baselines, and records how far the two
baselines differ. Use it to see run-to-run drift next to the candidate's effect.
`train_seconds` covers each arm's whole training, including preparation,
compilation, validation and checkpoints, and the arms share one process, so it
is not a throughput benchmark.

### Start every arm from one checkpoint

```yaml
repeat_baseline: true
probe_tier: none
continuation:
  checkpoint: /data/runs/parent/checkpoints/step_00023691_gen_000001
  append_data: true
  allow_runtime_optimization: true
  allow_runtime_drift: false
```

Every arm extends the same finished checkpoint with
[`--extend-budget`](checkpointing.md#extend-a-finished-run), keeping its weights,
optimizer, random state, cursor and schedule. The baseline config sets larger
limits and an appended dataset; the delta changes one runtime setting (for
example `training.matched_stories.padding_multiple`). `append_from_checkpoint:
true` continues from a checkpoint that is not the final one.

## Records

Records live in `WORK_DIR/lab/tries/<try_id>/try.json` and runs in
`WORK_DIR/lab/runs`. A record carries its own digest, and `report` refuses an
edited record. Each arm also records what `klab compare` and the dashboard's
Pareto view need: its evaluation group, parameter counts and bytes, time per
token and per-window losses for paired standard errors.

The same record ID works with `report`, `compare`, `probe` and `explore`, which
load the record's own candidate checkpoint, never a newer one. See
[results](results.md).

Exit status: 0 for a completed comparison, 3 for `NOT_COMPARABLE`, 130 for an
interrupted try, 1 for a failure. The record keeps the error.

## Stopping

Ctrl-C, SIGTERM or `touch <try>/CANCEL` stops at a checkpointed step during
training and between phases otherwise. The record is sealed as `interrupted`
with the arm, the phase (`setup`, `training`, `scoring` or `probing`) and the
reason, and keeps everything already scored. Interrupted tries are kept and
never reused.

## Sweeps

To see how a model responds across several values of a setting, run a sweep:
every value is a try against the same baseline, at every seed.

```yaml
# sweep.yaml
baseline: baseline.yaml
question: Does a longer context help?
vary:
  training.seq_len: [128, 512, 1024, 4096]
seeds: [1, 2, 3]
config: standard        # try policy: fast, standard, full, no-probes or a YAML path
```

```sh
klab sweep sweep.yaml
klab report SWEEP_ID
```

- **Grid.** Several fields under `vary` run every combination, in the order
  written. A sweep runs at most 256 tries.
- **Points.** To change several fields together, list the points instead of
  `vary` (a sweep has one or the other). For example, a tokenizer and its
  vocabulary size, compared on bits per byte:

  ```yaml
  points:
    - {tokenizer.path: tok-8k/tokenizer.json, model.vocab_size: 8192}
    - {tokenizer.path: tok-32k/tokenizer.json, model.vocab_size: 32768}
  ```
- **Baseline once per seed.** The first try at each seed trains the baseline;
  the others reuse it. A value equal to the baseline's own is reported as the
  baseline point, not trained again.
- **Seeds.** Without `seeds`, each try uses the baseline's seed. With seeds,
  each point is compared at every seed against the baseline at the same seed.
- **Paths.** `baseline` and a `config` file resolve from the sweep file.

`report` shows one row per point: the mean change in held-out loss across seeds
(in bits per byte for a point that changes the tokenizer) with its standard
error, a verdict, parameters and wall time, then every try ID.
A verdict needs two or more seeds: `LOWER_LOSS` or `HIGHER_LOSS` when the mean is
more than two standard errors from zero, otherwise `WITHIN_SEED_NOISE`. With one
seed the verdict says `SINGLE_SEED_*`, a direction to follow up rather than a
result. The sealed record lives in `LAB/sweeps/SWEEP_ID/sweep.json` beside a
`series.csv` for plotting.

An interrupted try stops the sweep and records how far it got. A failed or not
comparable try is recorded and the sweep continues. Each try is an ordinary
record, so `report`, `probe`, `explore` and `compare` work on any of them.

## Limits

One seed per arm and one held-out loss make a lab verdict a direction to
follow up, not a significance test. Confirm a gain with a sweep over paired
seeds before claiming it. Each try runs in one process on one device; spread
larger independent runs over [workers](workers.md).
