# Independent experiments and controlled comparisons

Commands below use the [activated, provisioned environment](../README.md#first-run).

This guide explains the execution machinery. Neutral teaching walkthroughs are
indexed under [`experiments/samples/`](../experiments/samples/). Keep your current
research declarations in a user-owned recipe directory; the local
[`experiments/research/`](../experiments/research/) tree retains historical references. Keep downloads, prepared
data, run stores, checkpoints, logs, and generated reports under the external
persistent root at
`$KLAB_WORK_DIR/experiments/<campaign>/`.
Report code gaps through the
[contribution workflow](../CONTRIBUTING.md#planning-and-contributions); scientific
next steps remain in the research roadmap/lifecycle.

For one local comparison, start with [the TinyStories lab loop](tinystories-microlab.md)
or [lab mode](lab-mode.md). This advanced guide covers independent queues and
matrices; [training programs](experiment-programs.md) adds authored plans and
checkpoint phases.

Run each concrete configuration with its declared tokenizer, data source, engine/backend, precision, optimizer, sequence length, effective batch, token budget, and seed. Inspect first; do not infer parameter counts or scientific equivalence from a filename.

```sh
klab inspect configs/micro_dense.yaml --json
klab inspect configs/dense_7m.yaml --json
klab inspect configs/dense_10m.yaml --json
klab inspect configs/dense_25m.yaml --json
klab inspect configs/dense_50m.yaml --json
klab inspect configs/smoke_sliding_cpu.yaml --json
klab inspect configs/smoke_mla_cpu.yaml --json
klab inspect configs/smoke_combined_cpu.yaml --json
```


The retained dense scale presets inspect at 3,344,064, 6,917,376, 10,244,160,
29,893,120, and 50,274,752 parameters. They stream TinyStories as a generic
[`huggingface` source](datasets.md#source-kinds). Use the microlab's
snapshot-based matrix for a runnable width comparison on whole documents. Parameter count alone is
not a comparison result: report observed validation loss, perplexity, throughput,
device, and metric coordinates for each completed run.

## Authored experiment plans and resolved locks

[Campaigns](campaigns.md) orchestrates direct snapshots, Corpus Forge releases and these plans
with typed dependencies, declared readiness policies, input-bound authorization
and recoverable runtime-bound execution. It does not replace ExperimentPlan's
scientific configuration or infer data/architecture/quality thresholds.

`experiment plan_version: 1` declarations describe a concrete base `RunConfig`,
bounded labeled axes, typed source/artifact inputs, optional Corpus Forge release
recipes, exact-field comparison contracts and ordered checkpoint phases. The
versioned [JSON Schema](../schemas/experiment-plan-v1.schema.json) describes the
authoring format. A declaration is not executable: `validate`/`inspect` expand
configuration values without asserting that referenced files, worker runtime or
future checkpoints exist. For declared `corpus_variants`, `prepare` explicitly
acquires pinned local/offline sources, freezes releases, exports data, trains
tokenizers and verifies prepared arrays without starting training. External
sources must already be acquired. A plain `base_run` does not get tokenizer or
data preparation from this command; prepare those assets explicitly.
`diff` requires those inputs when a comparison
selects corpus variants, and rejects every undeclared resolved-field difference.

Corpus variants choose exactly one tokenizer strategy: `vocab_size: 300` trains
from that release, or `tokenizer_artifact: shared_tokenizer` references a named
external `artifacts.shared_tokenizer` with `kind: tokenizer`, version, producer,
identifier, path to `tokenizer.json`, and SHA-256. Reuse verifies the
tokenizer's provenance manifest and actual vocabulary before acquiring or
building anything. Corpus Forge's immutable export `run.yaml` remains a generic
tokenizer-training template; the prepared and locked cell instead binds the
verified external tokenizer path.
Comparisons using one tokenizer declare the resulting release, export, prepared
data and dataset-revision identity differences explicitly and keep
`artifacts.tokenizer.sha256` invariant; a changed tokenizer SHA is independent
tokenizer drift, not an incidental corpus difference. Fractional releases still
require a project-relative pinned selector file and SHA. In reuse mode its SHA
must match the reused artifact unless `fraction_tokenizer` explicitly names a
second verified external tokenizer artifact with that selector's SHA.
`release_set.accepted_generation_statuses` accepts Corpus Forge's verification
labels as unordered filter classes, not a quality ranking or score.

The following retained neutral fixture teaches corpus-shape and exact-fraction
semantics. It is not a model-quality experiment. For another plan, replace the
sample path with your explicit current declaration and its bound inputs.
Preparation can train tokenizers and materialize data; use separately authorized
resources. The [disposition map](history/sample-disposition.json) identifies the fixture
and its unchanged historical preservation.

```sh
klab experiment validate experiments/samples/corpus-shape-fraction.yaml --json
klab experiment prepare experiments/samples/corpus-shape-fraction.yaml --json
klab experiment diff experiments/samples/corpus-shape-fraction.yaml --json
klab experiment lock experiments/samples/corpus-shape-fraction.yaml --json
```

`lock` re-verifies artifact digests and hardware-free runtime requirements,
then publishes an immutable content-addressed scientific plan. Its separate
availability sidecar records local input paths, not executable capability.
Accelerator runs require per-cell operational receipts from `experiment bind
LOCK --runtime-profile PROFILE` or `--worker NAME`; `experiment run` accepts
those sources directly or a previously published `--binding RECEIPT`, always
with fresh revalidation. Only explicit PyTorch CPU supports ambient local
registration. Runtime binding does not change either lock digest.
`experiment run LOCK --cell CELL --json` submits the selected cell;
`klab controller run --store "$KLAB_WORK_DIR/experiments/<plan-id>/controller"`
executes and ingests it. A dependent phase must be submitted separately with
`--phase` after the parent has an ingested, verified generation. Terminal and
best-validation selectors bind the exact parent generation before dispatch.
The parent `execution_binding_sha256` remains distinct from the operational
`runtime_binding_sha256`; mutable aliases are never checkpoint identities.
Resume, guarded budget extension, and weight promotion preserve their distinct
trainer semantics.

Comparison `interventions` and `invariants` name concrete leaf fields, such as
`model.ffn_dim`, `training.max_tokens` or `artifacts.tokenizer.sha256`; section
names such as `training` are not field selectors. An optional `phases: [pretrain]`
limits a comparison to those declared phases. Omitting it retains comparison in
every phase. This lets fresh architecture contrasts coexist with continuations
whose distinct parent checkpoint identities require separate interpretation.

`experiment collect LOCK --json` creates a hash-addressed evidence index from
matching worker specs/receipts and ingested manifests. Missing, failed,
interrupted, and invalid cells remain visible. `experiment reconstruct LOCK
--index INDEX --json` re-verifies it and returns identity/provenance, contrasts,
phase lineage, evidence coverage, and readiness/decision views. Planned-only
views omit `--index`. These are software and provenance reports, not automatic
scientific findings or baseline promotion; preregistered held-out evaluations
and human review remain separate. See the
[four-arm CPU example](../experiments/samples/README.md#corpus-shape-and-exact-generated-fraction)
and [retrospective limits](research/experiment-dsl-retrospectives.md).

Existing `matrix_version: 1` and `study_version: 1` callers remain usable:
their axis expansion and exact-difference checks now share the plan compiler,
but their receipts are not relabeled as new locks. To migrate, copy the base
config and axis choices into a new `plan_version: 1` document, declare artifact
identities/release recipes and exact interventions, run `prepare` when needed,
and create a new lock before dispatch. Preserve old receipts and research
reports with their original protocol identities. A lock cannot be retroactively
assigned to a prior result solely because its config fields happen to match.

## Explicit matrices and worker execution

The checked-in matrix expands the CPU runtime smoke configuration over seeds 7, 17, and 41:

```sh
WORK=klab-work/experiments/runtime-matrix
export KLAB_WORK_DIR="$WORK"
klab experiment submit --matrix tests/fixtures/runtime-matrix.yaml \
  --dry-run --store "$WORK/runs"
```

Dry-run prints all resolved coordinates and hashes without downloading/preparing data, creating the store, or enqueueing. For execution, prepare the tokenizer, register an eligible worker, remove `--dry-run`, and run `klab controller run --store "$WORK/runs"`. Every coordinate gets distinct experiment/attempt/run IDs. Preparation must succeed for all coordinates before the enqueue transaction.

Matrix v1 uses an explicit base config and insertion-ordered axes with labeled dotted-path patches. Duplicate labels, conflicting patches, invalid configs, and excessive expansion are rejected. Labels such as “50M” or “MoE” never infer a model. Workers execute independent optimizers, not distributed gradients; unsupported requirements remain queued with a reason. See [worker contracts and matrix format](workers.md#explicit-matrices).

## Architecture study campaigns

An architecture study is an optional campaign wrapper over the existing explicit matrix format. It does not add a closed architecture registry or change `RunConfig`: every matrix coordinate still resolves to a normal concrete config, with the same dotted-path validation as `experiment submit --matrix`. Use [lab mode](lab-mode.md) for a single delta; direct training remains available for explicit run control:

```sh
klab train --runs-dir klab-work/runs configs/my_architecture.yaml
klab inspect configs/my_architecture.yaml --json
```

If an architecture needs model-code or config-schema changes, make those normally and continue to train its concrete config directly. The campaign layer does not make arbitrary Python model code auto-configurable; it does not restrict existing direct builds to named presets.

For a controlled FFN-width experiment, save these two files as `experiments/ffn-matrix.yaml` and `experiments/ffn-study.yaml`:

```yaml
# experiments/ffn-matrix.yaml
matrix_version: 1
base_config: ../configs/runtime_smoke_cpu.yaml
axes:
  architecture:
    - label: baseline
      set: {}
    - label: wider-ffn
      set:
        model.ffn_dim: 48
```

```yaml
# experiments/ffn-study.yaml
study_version: 1
name: ffn-width-smoke
matrix: ffn-matrix.yaml
cards:
  - engram-recall-v1
comparisons:
  - id: wider-ffn
    vary: custom
    vary_fields:
      - model.ffn_dim
    baseline:
      architecture: baseline
    variant:
      architecture: wider-ffn
```

The smoke config demonstrates wiring, not model quality; use a task-appropriate trained config and capability card for useful evidence. Built-in cards are named in [capabilities](capabilities.md); custom cards are referenced by JSON path. A card is a fixed post-training probe, not a training reward.

Plan before allocating workers:

```sh
klab study plan experiments/ffn-study.yaml
```

The plan expands every config, hashes the inputs, reports parameter inventory and estimated weight/optimizer/checkpoint bytes, and requires each declared comparison to match one-to-one over all unselected axes. It rejects undeclared config changes before submission. For multi-seed ablations, add a `seed` matrix axis and leave it out of both selectors; the same seed then pairs baseline and variant. Seeds cannot be the varied field. `vary_fields` must list exactly the changed dotted config fields. The built-in `memory`, `attention`, `ffn`, and `scale` modes enforce their corresponding model field families.

Submit the immutable run inventory to the existing independent-worker controller, then run that controller in a separate terminal:

```sh
WORK=klab-work/experiments/ffn-width-smoke
export KLAB_WORK_DIR="$WORK"
mkdir -p "$WORK"
klab worker register cpu-one --backend cpu --store "$WORK/runs"
klab study submit experiments/ffn-study.yaml \
  --worker cpu-one --store "$WORK/runs" \
  --receipt "$WORK/receipt.json"
klab controller run --store "$WORK/runs"
```

After all coordinates finish, collect held-out validation loss/perplexity and each card against the run checkpoint:

```sh
klab study collect experiments/ffn-study.yaml \
  "$WORK/receipt.json" --runs-dir "$WORK/runs"
```

Collection verifies the receipt against the current study and each run's resolved config, records checkpoint identities, and emits a content-addressed report beside the receipt. Missing runs or invalid evaluations stay explicitly inconclusive; they are not dropped from the denominator silently. `--checkpoint NAME` selects the same named checkpoint for every run; the default is each run's latest checkpoint. Paired deltas are descriptive, not statistical significance or a universal architecture ranking.

A study submitted without path overrides uses `<work-dir>/experiments/<study-name>/runs` and one sibling `receipt.json`. Collection defaults to the receipt’s sibling `runs/`. Pass the same explicit store as `--runs-dir` for custom layouts. Seeds, architecture cells, budget coordinates, and resumed children share this store; they do not create peer workspaces. See [workspace policy](workspaces.md) for external-disk overrides and evidence retention.

Submission reserves `receipt.json.intent.json` exclusively and durably before
preparing or enqueueing jobs. This immutable intent binds a unique submission,
study, matrix, cards, coordinates, configs and controller store. A failed or
interrupted attempt retains it: do not delete the intent to retry submission.
Existing complete or partial receipts and intents prevent another enqueue.
Receipt publication uses complete staged bytes and an exclusive atomic link.

An ordinary preparation failure (for example, a missing tokenizer) can leave
an intent with zero queued jobs. The original receipt path remains reserved;
recovery distinguishes no bound submissions from partial evidence and from
legacy/unbound records. These are read-only observations, not proof that another
submitter has stopped. Never delete, rename or automatically abandon an intent
because a query currently finds no rows.

After confirming the original submission has finished and failed **before
enqueue**, fix its preparation error and choose a fresh receipt filename in the
**same directory**, keeping the same explicit store:

```sh
klab study submit experiments/ffn-study.yaml \
  --receipt /path/to/receipt-retry-1.json --store /path/to/runs
```

With no `--store` override, a new `--receipt` in the original receipt directory
also keeps its sibling `runs/` store. The retry gets a new submission identity
and its own intent. Preserve the old intent and any receipt bytes. A fresh
filename is not deduplication: if enqueue happened, or the original submitter
may still be active, recover or wait instead of submitting another copy.

If publication failed after enqueue, use the original study and explicit native
store to recover into a **distinct unused** output:

```sh
klab study recover experiments/ffn-study.yaml \
  --intent /path/to/receipt.json.intent.json \
  --store /path/to/runs --output /path/to/recovered-receipt.json
```

Recovery only reads the intent and controller database; it does not submit,
resume, tick, contact workers, or train. Every planned coordinate must have one
unambiguous persisted submission with the matching intent and full identity.
Missing, duplicate, altered or legacy unbound evidence is unrecoverable. An
intent alone does not prove enqueue occurred. Preserve original partial or
complete receipt bytes; the recovered receipt includes their observed digest,
intent provenance and store location. `recovery.intent_sha256` is the canonical
JSON digest used by the native binding; `recovery.intent_file_sha256` hashes the
observed file bytes, including the trailing newline. Collect it through the ordinary
`study collect` command, specifying `--runs-dir` when the output is elsewhere.
Recovery proves submission identity only, not job completion or scientific
results. No historical queue or receipt is upgraded or rewritten.

A write/flush/file-fsync failure publishes no destination. A directory-fsync
failure after publication leaves any complete destination intact and reports an
error: its crash durability is uncertain. Preserve surviving files and inspect
the evidence rather than interpreting an error as proof that nothing happened.
A surviving intent whose durability failed blocks another submit, and no jobs
are enqueued by that invocation. Recovery also refuses any preexisting output.
An abrupt process exit can leave hidden staging files such as
`.receipt.json.<uuid>.tmp` or `.receipt.json.intent.json.<uuid>.tmp`. They are
harmless to ignore: they neither reserve the destination nor prove enqueue or
publication. This recovery workflow does not authorize deleting them.

The study layer compares only configurations and evidence supported by the current training/evaluation stack. It does not add RL reward training or wire record-based Engram packs into model execution; those require separate model/trainer work. Keep data, tokenizer, source revision, runtime, and actual step/token budgets matched within each architecture comparison.

## Learn from retained comparisons

The [experiment ledger](research/experiment-ledger.md) links completed studies to
their bounded findings and limitations. Those records illustrate how to retain
comparisons; their datasets, run IDs and budgets are not required by this DSL.

## Evidence before comparison

Every serious local run should have verified checkpoint/held-out evidence before it enters a comparison:

```sh
klab checkpoint verify klab-work/runs/RUN_ID/checkpoints/latest.json --json
klab report --kind run RUN_ID --json
```

Training records held-out validation at initial, configured periodic, and terminal boundaries. A validation metric is not by itself a saved model: checkpoint cadence, a new best loss, or termination triggers a verified generation. Use checkpoint-bound evaluation reports when claiming results for exact weights. See [experiment evidence](evidence.md) for evidence levels, controlled-comparison requirements, and the separately unimplemented hardware/reference-harness protocol.

For a named architectural hypothesis rather than aggregate loss alone, use a [capability card](capabilities.md). Cards preserve their own prompt set, scorer, baseline controls, and scope-limited conclusion.
