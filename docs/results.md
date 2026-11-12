# Results

`try` and `probe` print a record ID. Keep it: the everyday commands all accept
it.

```sh
klab report TRY_ID
klab compare TRY_ID --references
klab probe TRY_ID --vs BASELINE_RUN --backend cpu --config standard
klab explore TRY_ID
```

`report` and `compare` read records and never load a model. `probe` scores and
`explore` inspects; both load the checkpoint the record names: a try's
candidate, or a probe's target. `--vs OTHER_RECORD` means the other record's
candidate. A record never implies its baseline, tier, final split, backend or
resource limits.

`--vs BASELINE_RUN` loads that run's current latest checkpoint, which may have
moved on since the try. To probe against exactly the baseline the try scored,
take `arms.baseline.checkpoint` from `klab report TRY_ID --json`:

```sh
klab probe TRY_ID --vs /data/lab/runs/BASELINE_RUN/checkpoints/GENERATION --backend cpu --config standard
```

## Reports

```sh
klab report TRY_OR_PROBE_ID [--json]
klab report RUN_ID --kind run [--runs-dir RUNS] [--json]
```

A record report shows the whole sealed record, including failed and interrupted
ones; `--lab-dir` selects another lab. Lab verifies a record's digest whenever
it is read, so an edited record is refused everywhere.

A run report shows two things side by side under separate JSON keys:

- **`evidence`:** the verified checkpoints and their held-out reports, including
  rejected or missing reports, and each checkpoint's
  [exposure](checkpointing.md#what-a-checkpoint-records).
- **`triage`:** the advisory post-training diagnosis saved by `train`. Missing
  or invalid triage is shown as `UNKNOWN` with the reason.

If the run's evidence cannot be read, the report says `missing_evidence` or
`invalid_evidence` and exits 1. Reports never train, load a model or create a
run store.

## How lookup works

- **Record IDs** are found under `LAB/tries` and `LAB/probes` (`LAB` is
  `WORK_DIR/lab`). A bare ID that also matches something in the current
  directory is reported as `ambiguous_result`; pass a path to choose.
- **Record files** can be passed directly, including exported copies under
  another name. Their digest is checked.
- **Runs and checkpoints** work with `probe` and `explore`: a run ID under
  `LAB/runs` or a declaration's `runs_dir`, a run directory (its latest
  checkpoint) or a checkpoint path (verified first).
- **A record pins its exact checkpoint.** If that checkpoint has moved or been
  deleted, restore it or pass its new path; Lab never falls back to another
  checkpoint.
- **`report` and `compare` need a record**, not a run: one run can have many
  records, and Lab does not guess which you meant.

A lookup failure exits 1; with `--json` it returns `status`
(`missing_evidence`, `ambiguous_result` or `invalid_evidence`), `result`,
`reason` and `hint`.

## Devices

A standalone `probe` uses each run's saved backend unless `--backend` is given.
Installing the CPU extra does not turn a CUDA or ROCm run into a CPU one: a saved
accelerator without an authorized runtime fails with `runtime_required` and
asks for `--runtime ID` or `--backend cpu`. `explore` runs on CPU unless you
choose a runtime. An MLX checkpoint cannot be probed as PyTorch.

## Baseline provenance

`try` and `probe` records keep the baseline exactly as you named it (path or
ID), the working directory, whether it resolved, and what it resolved to. The
scored baseline row records the checkpoint and digest actually used.
`report` shows the original selector and its resolution status.

## Comparing results

`klab compare RESULT [RESULT...] [--references] [--final] [--json]` places
results side by side using only recorded measurements, and compares a metric
only within one comparison group. See
[reference models and `klab compare`](probes.md#reference-models-and-klab-compare).
