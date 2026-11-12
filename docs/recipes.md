# Portable recipes

A recipe is a directory of declarations that lives outside Lab, in
[Kybern Workshop](https://github.com/supernovae/kybern-workshop) or your own
repository. Lab validates, composes, plans and runs those declarations with its
own commands, so a recipe repository needs no scripts and no copy of Lab.

Most recipe repositories need only [config validation and composition](#validate-and-compose-configs).
A descriptor that inventories every file by digest adds one of three profiles:

| Profile | Descriptor | What it does |
| --- | --- | --- |
| [Matrix](#matrix-recipes) | `portable_recipe_version: 1` | Previews an experiment plan's configurations |
| [Preparation](#preparation-recipes) | `portable_preparation_version: 1` | Builds tokenizers and prepared data through a campaign |
| [Workflow](#workflow-recipes) | `portable_workflow_version: 1` | Binds prepared inputs, publishes a lock and runs it |

## Validate and compose configs

Validate one section, or every declaration under a directory, without loading
a model, data or runtime:

```sh
klab config validate model.json --kind model --json
klab config validate optimizer.yaml --kind optimizer --json
klab config validate run.yaml --kind run --json
klab config validate tokenizer.yaml --kind tokenizer --json
klab config validate path/to/project
```

Unknown fields, malformed YAML or JSON and duplicate keys are refused. For a
directory, every complete run, tokenizer, model and optimizer declaration is
checked, other documents are counted as skipped, and the exit status is nonzero
if any fails. This is the check a recipe repository runs by hand or in CI. A
valid section says nothing about artifacts, hardware or whether sections fit
together; `klab inspect CONFIG` checks a full run config.

Assemble a run config from parts:

```sh
klab config compose run.yaml model=model.json optimizer=optimizer.json \
  local-paths.yaml --output "$WORK/run.yaml"
```

Parts merge in order, later parts winning, and `FIELD=PATH` places a file at a
dotted field. Keep a model section, an optimizer section and a file of local
paths (tokenizer, cache, run store) outside Git and compose them. Relative paths
resolve from the part that declares them. The result must be a complete run
config and is written beside a `.derivation.json` receipt with each part's
digest.

Change single fields of a complete config:

```sh
klab config derive configs/smoke_cpu.yaml \
  --set optimizer.peak=0.001 --set training.micro_batch_size=2 \
  --output "$WORK/smoke-peak.yaml" --json
```

Each `--set` is strict JSON, so strings are quoted. Paths in assignments resolve
from the source config's directory. The output and its `.derivation.json` (the
requested assignments and the observed change) must not exist. Deriving never
adjusts other settings: changing a batch size does not change the token budget.

## Recipe descriptors

All three profiles share these rules:

- **Root.** The descriptor's directory is the recipe root, whatever the working
  directory or enclosing Git repository. Every declaration reference resolves
  inside it and must be inventoried.
- **Inventory.** `files` lists each member's canonical relative path, role and
  SHA-256. Lab reads at most 64 members, 1 MiB each and 8 MiB in total, and
  snapshots their bytes before validating them. Absolute paths, `..`, `./`,
  Windows paths and symlinks are refused.
- **Identity.** The recipe fingerprint is a SHA-256 over a profile-specific
  prefix and the canonical descriptor with files sorted by path, so formatting
  does not change it; `descriptor_sha256` covers the exact bytes. Moving the
  directory changes only the displayed location. Local bindings have their own
  digest.
- **Local bindings.** Where outputs and external inputs live on this host is
  kept in a separate bindings document, outside the recipe. Relative locations
  resolve from that document.

### Create, export and move a recipe

```sh
klab experiment recipe create /my/authored-inputs --profile matrix \
  --id new-study --entrypoint plans/plan.yaml \
  --file experiment_plan:plans/plan.yaml --file run_config:configs/run.json \
  --file metadata:notes.json --artifacts /my/expected-input-identities.json \
  --output /my/new-recipe
klab experiment recipe export /my/new-recipe/recipe.json --output /my/copied-recipe
```

`create` hashes exactly the files you name with `--file ROLE:PATH` and writes
`recipe.json`; it never scans a directory, so it cannot pick up a secret by
accident. `--artifacts` maps native input fields to digests you already know.
`export` copies an existing descriptor and its members. Both refuse an existing
destination, write the descriptor last, and leave a partial directory in place
after an I/O error rather than overwriting it.

| Profile | Entrypoint role | Other roles | Next command |
| --- | --- | --- | --- |
| matrix | `experiment_plan` | `run_config`, `metadata` | `recipe inspect` |
| preparation | `campaign` | `run_config`, `tokenizer_config`, `metadata`, `source_lock` | `recipe preparation inspect` |
| workflow | `experiment_plan` | `run_config`, `metadata`; optional `--campaign` | `recipe workflow inspect` |

Commit the directory to your own repository with Git.

### Admit execution

Preparation and workflow execution need an admission document naming exactly
one source of trust:

- `reviewed_commit`: the recipe repository's current HEAD, with every member
  clean and committed. Any later commit invalidates it.
- `reviewed_descriptor_sha256`: the SHA-256 of the exact descriptor bytes, for a
  recipe in a plain directory (`sha256sum recipe.json`).

Either way the admission also carries the `recipe_sha256` and `binding_sha256`
from inspection and `cpu_python`, the exact interpreter running the command
(`sys.executable`). Admission is your assertion that you reviewed this content;
Git proves committed bytes, not authorship. Changing the recipe, the bindings or
the admission invalidates a resume. Outputs must go under an absolute work
directory outside Git. Lab never fills in or grants an admission.

## Matrix recipes

`klab experiment recipe inspect RECIPE.yaml --json` previews an experiment
plan's cells with the native compiler. It never runs anything, reports
`status: incomplete` and `executable: false`, and calculates no payload hashes.

```yaml
portable_recipe_version: 1
id: width-study
entrypoint: plans/plan.yaml
contracts:
  experiment_plan: 1
  run_config: 2
files:
  - path: plans/plan.yaml
    role: experiment_plan
    sha256: <sha256 of the plan bytes>
  - path: configs/run.json
    role: run_config
    sha256: <sha256 of the config bytes>
artifacts:
  tokenizer.path: <expected sha256 of the tokenizer>
```

The plan's `base_run: configs/run.json` is relative to the recipe root. Typed
paths inside a config stay relative to that config's directory:
`tokenizer.path: artifacts/tokenizer.json` in `configs/run.json` means
`configs/artifacts/tokenizer.json`.

- **Inputs** (`tokenizer.path`, `model.memory_package_path`, the `dataset`
  source and corpus paths, `training.portability_manifest_path`) need not exist;
  list their expected digests under `artifacts`. Missing inputs report
  `unavailable`, present ones `unverified`, and runtime capability `unknown`.
- **Outputs** (`dataset.cache_dir`, `logging.root_dir`) are bound but never
  created.
- **Supported**: axes, derived fields, comparisons with one cell per arm, and up
  to `--max-runs` cells (default 1000).
- **Refused**: inline base configs, mixture paths, path-changing axes,
  non-checkpoint artifacts, corpus variants, evaluation suites and attempt
  contracts. The preview never silently drops work it cannot compile.

A phase may continue from an external checkpoint artifact (identity, SHA-256,
`full` or `weights` state and a relative path) with `resume`, `extend_budget` or
`promote`, or from an earlier phase through `parent` and a selector. Inspection
checks the phase graph and each phase's patch, and reports
`supplied_checkpoint`, `produced_checkpoint` or `fresh_training` per phase. It
does not open checkpoints or check that parent and child are compatible; the
real lock does.

## Preparation recipes

A preparation recipe builds tokenizers and prepared data through a campaign
restricted to `tokenizer_train`, `data_prepare`, `dataset_snapshot` and external
`artifact_reference` stages. Its contracts are
`{campaign: 1, run_config: 2, tokenizer_config: 1}`, and the entrypoint is its
only campaign member. Two graph shapes are supported:

- `tokenizer_train → data_prepare`: train a tokenizer, then prepare with it.
- `artifact_reference(tokenizer) → data_prepare`: prepare with an existing
  tokenizer.

The bindings name every config reference and each of its typed path fields:

```json
{
  "artifacts": {},
  "config_paths": {
    "tokenizer.json": {"output_dir": "payload/tok", "dataset.cache_dir": "payload/cache"},
    "run.json": {"tokenizer.path": "payload/tok/tokenizer.json", "dataset.cache_dir": "payload/cache", "logging.root_dir": "payload/runs"}
  }
}
```

`artifacts` maps each external tokenizer stage to its location. A consumer's
tokenizer path must match its producer's output.

```sh
klab experiment recipe preparation inspect recipe.json --bindings local.json --json
klab experiment recipe preparation apply recipe.json --bindings local.json \
  --admission admission.json --work-dir /absolute/work --json
klab experiment recipe preparation resume recipe.json --bindings local.json \
  --admission admission.json --work-dir /absolute/work --json
```

`inspect` reports the operations, edges, config digests and local paths without
opening payloads. `apply` and `resume` run the campaign under an admission with
explicit ceilings:

```json
{
  "reviewed_commit": "<full recipe commit SHA>",
  "recipe_sha256": "<recipe_sha256 from inspect>",
  "binding_sha256": "<binding_sha256 from inspect>",
  "cpu_python": "/absolute/environment/bin/python",
  "max_documents": 32,
  "max_tokens": 1024,
  "max_vocab_size": 512,
  "min_free_bytes": 1073741824,
  "min_free_inodes": 128,
  "max_snapshot_bytes": 1048576,
  "max_snapshot_records": 40
}
```

All limits are strict positive integers (vocabulary at least 260). Each config's
total documents and tokens, and the tokenizer's documents and vocabulary, must
fit; the ceilings never rewrite a config. Free space is checked again before
dispatch. The snapshot limits are required only when the graph reads or
acquires a snapshot: `max_snapshot_bytes` counts every file in each snapshot
directory and `max_snapshot_records` the sum of declared per-split
`max_source_records`, across all references.

Outputs must lie under the work directory, must not overlap each other or the
operational directories (`campaigns`, `scratch`, `execution`, `ownership`), and
must not overlap snapshot inputs. A completed stage replays from its receipt
without training or packing again, after its outputs are verified. An
interrupted stage keeps its attempt and is never rerun or overwritten
automatically. An existing store refuses a changed recipe, bindings or
admission; use a new work directory for a new context.

### Consume a dataset snapshot

An already materialized snapshot enters as an `artifact_reference` of kind
`dataset_snapshot`:

```yaml
- id: input
  kind: artifact_reference
  scope: corpus
  artifact:
    kind: dataset_snapshot
    version: 1
    producer: native dataset snapshot
    identifier: <source lock_sha256>
    sha256: <snapshot manifest_sha256>
    path: snapshot/manifest.json
- id: tokenizer
  kind: tokenizer_train
  scope: tokenizer
  requires: [input]
  snapshot: input
  config: tokenizer.json
- id: packed
  kind: data_prepare
  scope: model
  requires: [input, tokenizer]
  snapshot: input
  tokenizer: tokenizer
  config: run.json
```

Bind `artifacts.input` to the local `manifest.json`. Each consuming config
declares `dataset.source: snapshot` with `source_manifest_path`, `train_path` and
`validation_path`, bound in its `config_paths` to that manifest and its sibling
`train.jsonl` and `validation.jsonl`. The snapshot must be
`klab-dataset-snapshot-v1`. Verification covers the lock and manifest
identities, split contents, exclusions, event history, revision, license and
locations; it does not grant redistribution rights. A missing or changed
payload is refused, with no network fallback.

### Acquire from the Hub

A `source_lock` member can feed a `dataset_snapshot` stage (`id: input`,
`scope: corpus`, `lock: source.lock.json`, `output: snapshot`,
`cache_dir: acquisition-cache`). The bindings place the output and cache:

```yaml
artifacts: {}
acquisitions:
  input:
    output: /absolute/work/snapshot
    cache_dir: /absolute/work/acquisition-cache
config_paths: {}
```

The admission then needs an `acquisitions` grant per acquiring stage:

| Field | Meaning |
| --- | --- |
| `acquisition_version`, `provider`, `decoder_profile` | `1`, `huggingface`, `scalar-parquet-v1` |
| `repo_id`, `revision`, `lock_sha256` | The exact source lock: 40-hex revision, 64-hex digest |
| `allow_network` | Must be `true` |
| `max_source_body_bytes`, `max_transfers`, `max_retries_per_shard` | Each file's full size is reserved before every request, retries included |
| `max_metadata_body_bytes`, `max_requests` | Bodies of redirects and errors, and the request count |
| `max_wall_seconds` | A deadline from the first start, not reset on resume |
| `max_cache_bytes`, `max_cache_files` | Cache reservations, including partial files |
| `max_scanned_rows` | Rows reserved before decoding, replays included |
| `max_verification_bytes` | Logical bytes reserved for hashing and verification |
| `max_observed_owned_bytes`, `max_observed_owned_inodes` | Observed work-directory growth; a stop rule, not a filesystem quota |

The source lock must already list its files in order. Only whole pinned Parquet
files with scalar or text columns are accepted; projections, filters, casts and
media features are refused. Downloads use HTTPS with credentials stripped on
cross-origin redirects, and Git and LFS hashes are checked. The ledger lives at
`WORK/execution/acquisition/<binding>/transport.sqlite`. Reservations are never
refunded, so exhausting an allowance or the deadline can refuse a replay.
Creating `CANCEL` there stops acquisition and keeps both journals; remove it to
resume. `stage.resume` stays false: continue with `recipe preparation resume`.

## Workflow recipes

A workflow recipe starts after preparation has produced a tokenizer and prepared
data. It binds them to an experiment plan, publishes the lock and, with a
campaign, runs it.

```yaml
portable_workflow_version: 1
id: prepared-width-study
entrypoint: plans/plan.yaml
contracts: {experiment_plan: 1, run_config: 2}
files:
  - {path: plans/plan.yaml, role: experiment_plan, sha256: <file SHA-256>}
  - {path: configs/run.json, role: run_config, sha256: <file SHA-256>}
artifacts:
  token:
    kind: tokenizer
    version: 1
    producer: klab tokenizer train
    identifier: tokenizer
    sha256: <tokenizer SHA-256>
  packed:
    kind: prepared_data
    version: 1
    producer: klab data prepare
    identifier: <prepared settings SHA-256>
    sha256: <prepared manifest SHA-256>
```

The plan declares `base_run: configs/run.json` and
`inputs: {tokenizer: token, prepared_data: packed}`. Take each identity from the
completed preparation stage's `outputs` (`kind`, `identifier`, `sha256`); never
write one for a missing or failed stage. The bindings place the inputs and the
run store:

```yaml
workflow_bindings_version: 1
artifacts:
  token: /data/tokenizer/tokenizer.json
  packed: /data/cache/PREPARED_ID
logging_root: /work/runs
config_paths: {}
```

`config_paths` supplies any other typed input path, such as snapshot locations.
The profile takes exactly one tokenizer and one prepared-data artifact; corpus
variants, authored phases, inline evaluations, attempt contracts, path-changing
axes and mixture paths are refused.

```sh
klab experiment recipe workflow inspect recipe/workflow.yaml --bindings local.yaml --json
klab experiment recipe workflow resolve recipe/workflow.yaml --bindings local.yaml --json
klab experiment recipe workflow publish recipe/workflow.yaml --bindings local.yaml \
  --workspace /absolute/work/experiment --json
```

`inspect` reads metadata only. `resolve` verifies the artifacts and resolves the
plan without publishing. `publish` writes the lock and its availability sidecar,
which records the recipe and binding identity. A published workflow lock is not
permission to run: `experiment run` refuses it with
`WORKFLOW_EXECUTION_UNAVAILABLE`.

### Run a workflow

Add `campaign: campaign.json` to the descriptor and inventory the campaign and
its evaluation suite as `metadata`. The campaign connects two artifact
references (with the descriptor's identity fields and no path), one plan lock,
CPU runtime acceptance, an approval, the run, checkpoint collection, a
`heldout_lm` evaluation and a final `experiment_collect` stage covering every
evaluation. Stage IDs are yours to choose.

```sh
klab experiment recipe workflow apply recipe/workflow.yaml --bindings local.yaml \
  --admission admission.json --work-dir /absolute/work --json
klab experiment recipe workflow approve recipe/workflow.yaml APPROVAL_STAGE --bindings local.yaml \
  --admission admission.json --work-dir /absolute/work --note "approved bounded CPU run" --json
klab experiment recipe workflow apply recipe/workflow.yaml --bindings local.yaml \
  --admission admission.json --work-dir /absolute/work --execute-runs --max-wait-seconds 540 --json
klab experiment recipe workflow resume recipe/workflow.yaml --bindings local.yaml \
  --admission admission.json --work-dir /absolute/work --json
```

```json
{
  "reviewed_commit": "<recipe commit SHA>",
  "recipe_sha256": "<recipe_sha256 from inspect>",
  "binding_sha256": "<binding_sha256 from inspect>",
  "cpu_python": "/absolute/environment/bin/python",
  "limits": {
    "max_training_updates": 54,
    "max_target_positions": 1728,
    "max_eval_batches": 4
  }
}
```

`limits` refuses a recipe that declares more work than you approved; it is not
a compute allocation. Training updates sum `max_steps`, `smoke_steps` and
`warmup_steps` across cells; target positions multiply each cell's updates by
sequence length, microbatch and accumulation; evaluation batches bound each
scoring call. Budget wall time, memory and storage separately with
[`klab monitor`](workspaces.md#guard-a-command) around the whole command.

`apply` without `--execute-runs` still runs runtime acceptance probes; it only
withholds the training run. Nothing is approved implicitly (`approve --decision
reject` records a rejection). The default controller wait is 120 seconds; zero
queues without ticking a worker. Each command prints one
`klab-workflow-command-v1` JSON object and exits 1 on errors or a failed stage.
Exit 0 can still mean waiting or blocked, so read the stage states.

The context is checked again at admission, approval, every dispatch and every
receipt, so an old approval never admits a changed recipe or bindings. The
supported workload is one synthetic CPU PyTorch cell, or the pair below.

### A controlled pair

A workflow can compare two cells on one snapshot dataset: one axis with two
named choices and one `controlled` comparison of them, with exactly one
intervention under `model`, `attention` or `optimizer`. Data, tokenizer, seed,
runtime, exposure and evaluation cannot be the intervention. The campaign has two
run, collect and evaluate branches: one run requires the other (so they run in
sequence), each collection requires both runs, and both evaluations use the
same suite. A comparison is available only when both cells have complete valid
evidence.

### Report

```sh
klab experiment recipe workflow report /work/plan.lock.json --workspace /work/experiment \
  --index /work/evidence/index.json --output /work/report.json --json
klab experiment recipe workflow report /work/plan.lock.json --workspace /work/experiment \
  --index /work/evidence/index.json --output /work/report.md --format markdown
```

Use the plan stage's lock and workspace and the final collect stage's evidence
index (not an evaluation index). Give both `--workspace` and `--index`, or
neither for a planned-only report. The report (`klab-publication-report`)
verifies the evidence, keeps missing and incomplete items, writes a new file
(an identical rerun is allowed) and never collects, trains or scores. The
evidence index (`klab-experiment-evidence-index`) records `diagnostic_status`
and one diagnostic per completed cell.
