# Experiments beyond one comparison

Most work is one change against one baseline: use [lab mode](lab-mode.md).
This page covers what comes after that: queues of independent runs, seed and
architecture matrices, authored plans with locked inputs, and the checks to run
between iterations of a long-running model.

## Choose a path

| Need | Command | Guide |
| --- | --- | --- |
| Compare one local change | `klab try` | [lab mode](lab-mode.md) |
| Train, resume, extend or promote a checkpoint directly | `klab train` | [checkpointing](checkpointing.md) |
| A reusable readiness bundle or a disposable pilot | `klab stage` | [runtime](runtime.md) |
| Queue a config or an independent matrix | `klab experiment submit` | [workers](workers.md) |
| Run selected cells and phases of a locked plan | `klab experiment run` | [training programs](experiment-programs.md) |
| Supervise a reserved native phase with an attempt budget | `klab attempt run` | [corpus preparation](corpus-preparation.md) |
| Orchestrate releases, approvals and dependent stages | `klab campaign` | [campaigns](campaigns.md) |

Inspect every config before you run it. Do not infer a parameter count or a
scientific equivalence from a filename:

```sh
klab inspect configs/dense_10m.yaml --json
```

## Matrices

A matrix expands one base config over labeled axes. Each coordinate is a normal
config with its own experiment, attempt and run IDs. This one, checked in as
`tests/fixtures/runtime-matrix.yaml`, runs the CPU smoke config over three seeds:

```yaml
matrix_version: 1
base_config: ../../configs/runtime_smoke_cpu.yaml
scheduling:
  requirements:
    backend: [cpu]
axes:
  seed:
    - label: seed-7
      set:
        seed: 7
    - label: seed-17
      set:
        seed: 17
    - label: seed-41
      set:
        seed: 41
```

Add an `architecture` axis (for example `baseline` with `set: {}` and
`wider-ffn` with `model.ffn_dim: 48`) and every architecture runs at every seed.

```sh
WORK="$KLAB_WORK_DIR/experiments/runtime-matrix"
klab experiment submit --matrix tests/fixtures/runtime-matrix.yaml \
  --dry-run --store "$WORK/runs"
```

`--dry-run` prints every resolved coordinate and hash without preparing data,
creating the store or queueing anything. To execute, prepare the tokenizer,
[register a worker](workers.md), drop `--dry-run` and run
`klab controller run --store "$WORK/runs"` in another terminal. Preparation must
succeed for every coordinate before any job is queued.

Axes apply in order. Duplicate labels, conflicting patches, invalid configs and
oversized expansions are refused. Labels such as `50M` or `MoE` never select a
model: only the `set` patches change the config. Workers train independent
optimizers, not distributed gradients; a coordinate whose requirements no
worker meets stays queued with the reason. See
[workers](workers.md#explicit-matrices) for the full format.

Keep data, tokenizer, code revision, runtime and token budget matched across an
architecture comparison, and pair seeds across arms. Parameter count is not a
result: report validation loss, throughput, device and budget for every
completed run.

## Plans and locks

An experiment plan (`plan_version: 1`, [JSON Schema](../schemas/experiment-plan-v1.schema.json))
declares a base config, labeled axes, typed source and artifact inputs, optional
Corpus Forge release recipes, comparison contracts and ordered checkpoint
phases. A plan is not executable until it is locked:

```sh
klab experiment validate PLAN.yaml --json
klab experiment prepare PLAN.yaml --json
klab experiment diff PLAN.yaml --json
klab experiment lock PLAN.yaml --json
```

- **`validate`** expands the configuration without checking that files,
  runtimes or checkpoints exist.
- **`prepare`** builds what `corpus_variants` declare: it acquires pinned
  local sources, freezes releases, exports data, trains tokenizers and verifies
  prepared arrays. External sources must already be downloaded. A plain
  `base_run` gets no preparation here; prepare its data yourself.
- **`diff`** checks each comparison and refuses any resolved field that differs
  without being declared.
- **`lock`** verifies artifact digests and runtime requirements, then writes
  an immutable, content-addressed plan. A sidecar records where the inputs are
  on this host.

Comparisons name concrete leaf fields in `interventions` and `invariants`, such
as `model.ffn_dim`, `training.max_tokens` or `artifacts.tokenizer.sha256`; a
section name like `training` is not a field. `phases: [pretrain]` limits a
comparison to those phases.

A corpus variant picks one tokenizer strategy: `vocab_size: 300` trains one from
that release, or `tokenizer_artifact: shared_tokenizer` reuses a named external
artifact (`kind: tokenizer`, version, producer, identifier, path to
`tokenizer.json` and SHA-256). Reuse verifies the tokenizer's manifest and
vocabulary before building anything. Comparisons that share a tokenizer keep
`artifacts.tokenizer.sha256` invariant. Fractional releases need a pinned
selector file and its SHA.

### Run a lock

```sh
klab experiment bind LOCK --runtime-profile PROFILE
klab experiment run LOCK --cell CELL --json
klab controller run --store "$KLAB_WORK_DIR/experiments/PLAN_ID/controller"
```

Accelerator cells need a runtime binding from `experiment bind` (with
`--runtime-profile`, `--runtime` or `--worker`); `experiment run` accepts those
directly or a saved `--binding RECEIPT`, and revalidates every time. Only
PyTorch CPU runs without one. Binding a runtime does not change the lock digest.

A dependent phase is submitted on its own with `--phase` once its parent has a
verified, ingested generation. The parent generation is pinned before dispatch;
`latest` and `best` pointers are never identities.

### Collect results

```sh
klab experiment collect LOCK --json
klab experiment reconstruct LOCK --index INDEX --json
```

`collect` builds a content-addressed index of every cell's specs, receipts and
manifests. Missing, failed, interrupted and invalid cells stay in it.
`reconstruct` verifies that index and returns the plan's identity, contrasts,
phase lineage and coverage; without `--index` it shows the planned view only.
Neither draws a scientific conclusion or promotes a baseline.

## Between iterations

When a model trains in stages, check the next step before spending compute:

```sh
klab checkpoint verify "$GENERATION" --json
klab report "$RUN_ID" --kind run --runs-dir "$RUNS" --json
klab iteration check PLAN_OR_LOCK --parent "$GENERATION" --json
```

Pin an exact `step_*_gen_*` directory as the parent, never `latest.json` or
`best.json`. `iteration check` reads without changing anything: it verifies the
parent, compares its saved config with the next cell, and reports the changed
fields, artifacts, storage, runtime evidence and the next command. Add
`--cold-verify` at a new host or trust boundary. For a fresh phase, omit
`--parent`.

Its `state` is one of:

| State | Meaning |
| --- | --- |
| `SAFE_TO_PREPARE` | The declaration needs a lock. |
| `NEEDS_PILOT` | The runtime or full-shape pilot is unproven. |
| `NEEDS_APPROVAL` | A campaign gate is waiting. |
| `READY_TO_DISPATCH` | Every inspected gate passed. |
| `RUNNING` | Campaign work is in flight. |
| `BLOCKED` | A prerequisite failed, is missing or cannot be verified. |

The exit status is 1 for `BLOCKED` and 0 otherwise. A ready state is not
permission to run: the next command validates again when it executes.

What to repeat depends on what changed:

| Change | Repeat |
| --- | --- |
| A lab-mode delta | Nothing; `klab try` handles it and reuses a verified identical baseline. |
| Budget or another config field | Inspect, lock a new plan, check the parent, storage and pilot. |
| Corpus, tokenizer or prepared data | Verify the changed artifact and what depends on it, then prepare again. |
| Lab code | The nearest tests and the affected [readiness](runtime.md#readiness-smoke) families. |
| Backend, device or worker | A fresh runtime acceptance and full-shape pilot on that runtime. |
| New host or store | Cold verification at the boundary. |

Verified reuse skips rehashing unchanged artifacts in an owner-only registered
store, using signed host-local proofs; anything changed or unproven falls back
to cold verification. `--cold-verify` turns reuse off. See
[workspaces](workspaces.md#verified-reuse).
