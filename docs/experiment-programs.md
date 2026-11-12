# Training programs

A training program is an experiment plan with ordered checkpoint phases: train,
then continue, branch or promote from a chosen parent generation. Lab locks the
plan, runs each phase through a worker, and records which parent each child
came from. For one comparison, use [lab mode](lab-mode.md) instead.

This guide runs [`examples/training-program`](../examples/training-program/):
20 steps of fresh training on synthetic data, then a full-state extension to 40
steps. It needs no download and runs on a laptop CPU.

## Read the program

```yaml
phases:
  - id: pretrain
    transition: fresh
  - id: continue
    transition: extend_budget
    parent: pretrain
    selector: terminal
    at_step: 20
    set:
      training.max_steps: 40
      training.max_tokens: 1280
      optimizer.decay_steps: 20
```

The base config has sequence length 16, microbatch 1 and accumulation 2: 32
targets per update. The parent ends at 20 updates and 640 targets; the child
continues the parent's full training state to 40 updates and 1,280 targets on
the original decay horizon.

| Transition | What the child carries forward |
| --- | --- |
| `fresh` | Nothing: new weights and training state. |
| `resume` | The full training state, with unchanged settings and budget. |
| `extend_budget` | The full final AdamW state, with a larger budget and the original decay horizon. PyTorch only. |
| `promote` | Weights and tokenizer, with a fresh optimizer, schedule, random state and cursor. |

A phase names an earlier parent with the same axis coordinate. A `terminal`
selector names the parent's final update with `at_step`; `best_validation`
picks the verified generation with the lowest validation loss. The chosen
checkpoint's digest is recorded in the child's binding.

## Prepare and bind the inputs

Copy the example into a workspace outside the checkout; its configs place the
tokenizer, data and runs beside it.

```sh
WORK="$KLAB_WORK_DIR/experiments/offline-training-chain-v1"
mkdir -p "$WORK"
cp -r examples/training-program "$WORK/inputs"
cd "$WORK/inputs"
klab tokenizer train tokenizer.yaml
PREPARED=$(klab data prepare run.yaml)
klab experiment bind-inputs run.yaml offline-chain.yaml \
  --prepared-root "$PREPARED" --output plan.yaml --json
klab experiment validate plan.yaml --json
klab experiment inspect plan.yaml --json
LOCK=$(klab experiment lock plan.yaml --json | jq -r '.lock')
klab experiment explain "$LOCK" --json
```

`PREPARED` is the directory `data prepare` prints: the one holding
`manifest.json`. `bind-inputs` checks the real tokenizer and prepared arrays
against the run config and writes a complete plan; it never trains, downloads
or prepares anything. The names `tokenizer`, `packed`, `release` and `export`
are reserved for inputs it binds.

Locking verifies every input once, then publishes the content-addressed plan.
Every later reopen (`explain`, `run`, a campaign) verifies the inputs again from
disk; `--cold-verify` also turns off [verified reuse](workspaces.md#verified-reuse).
Keep the code revision, plan and inputs fixed while it runs. To change it, copy
the plan under a new ID; never rewrite a running plan.

Plans are normally committed to Git before they run. This example's plan lives
outside the repository, so `run` passes `--allow-uncommitted-declaration`. The
lock still pins the plan's bytes, and the flag does not skip input or runtime
checks.

### Snapshot datasets

A run on a `dataset.source: snapshot` dataset binds the same two inputs,
`tokenizer` and `prepared_data`. Lock and snapshot the source with the
[dataset interface](datasets.md), prepare it, and pass the prepared root to
`bind-inputs`. Locking checks the tokenizer's training provenance, the snapshot
identity and how the prepared data was built from both. Workers receive the
verified arrays and tokenizer, so they never fetch the source.

### Corpus Forge inputs

For a Corpus Forge release, preparation seals portable provenance with the
inputs: tokenizer and manifest, packed arrays, release manifest, corpus,
license and audit reports, export metadata and `corpus/binding.json`. Pilots and
workers verify that evidence without the original export, a run copies it into
its artifact inventory, and a resumed child carries its parent's. Missing or
changed evidence is refused.

### Attempt contracts

`execution.attempt_contract: {path, sha256}` pins an operational allocation
file to the plan. Locking checks its bytes, and the digest becomes part of the
plan's operational identity, which campaign approval binds. The contract alone
grants nothing.

### Operational source compatibility

A fix that touches only operational code still changes the package digest, so
an existing lock and its prepared data no longer match. To keep using them,
publish a reviewed `OperationalSourceCompatibility` record that binds the
baseline and execution commits, their package inventories and the exact changed
files, then pin it with `KLAB_SOURCE_COMPATIBILITY` (absolute path),
`KLAB_SOURCE_COMPATIBILITY_SHA256` and `KLAB_SOURCE_COMPATIBILITY_COMMIT`. Only
the listed operational paths are eligible. New data is never prepared under the
old digest, and runs record the real execution source.

## Derive a new plan

`experiment derive` writes a new plan for changes a phase's `set` cannot make:
retention, comparisons, axes or a base-run field.

```sh
klab experiment derive plan.yaml \
  --set 'id="offline-training-chain-v2"' \
  --set retention.keep_periodic=false \
  --output plan-v2.yaml --json
LOCK=$(klab experiment lock plan-v2.yaml --json | jq -r '.lock')
```

Values are strict JSON, one `--set` per field, and lists are replaced whole.
Neither output may already exist. Beside `plan-v2.yaml` it writes
`plan-v2.yaml.derivation.json`, binding the source and output bytes and the
observed change. Derivation resolves the candidate as completely as `lock`
would, and refuses a move that would change what a reference points at. Pass
`--prepared PREPARATION.json` when the new plan needs an existing prepared
corpus variant; its plan ID must match.

## Pilot the resolved configs

```sh
klab experiment export-config "$LOCK" --cell pretrain:single \
  --output "$WORK/pretrain-effective.yaml" --json
klab inspect "$WORK/pretrain-effective.yaml" --json
klab stage "$WORK/pretrain-effective.yaml" --through warmup --output "$WORK/stages/pretrain"
```

Each cell's exported config is what will run, after input binding and phase
changes. A pilot shows it fits and runs; it does not run the transition.

## Run the parent, then the child

```sh
klab experiment run "$LOCK" --phase pretrain --allow-uncommitted-declaration --json
klab controller run --store "$WORK/controller"
```

`run` queues work (registering a local worker if none is named) and returns.
The controller stays in the foreground. From another terminal, watch for the
parent's `status: COMPLETE` and `ingestion_status: COMPLETE`, then queue the
child:

```sh
klab experiment list --store "$WORK/controller"
klab experiment run "$LOCK" --phase continue --allow-uncommitted-declaration --json
```

Without `--phase`, `run` selects root phases. A dependent phase needs exactly one
completed, ingested parent; failed, interrupted or pending parents do not count.
`--cell pretrain:single` selects a single cell; matrix cells include their axis
labels.

## Collect the results

```sh
klab experiment collect "$LOCK" --json > "$WORK/collection.json"
INDEX=$(jq -r '.index_path' "$WORK/collection.json")
klab experiment reconstruct "$LOCK" --index "$INDEX" --json
klab dashboard --runs-dir "$WORK/controller"
```

The index keeps pending, failed, interrupted and invalid cells. `reconstruct`
reports input identity, contrasts, phase lineage, coverage and readiness. Use
the returned run IDs with the ordinary commands:

```sh
klab eval RUN_ID --runs-dir "$WORK/controller"
klab report RUN_ID --kind run --runs-dir "$WORK/controller" --json
klab generate RUN_ID --runs-dir "$WORK/controller" --prompt "A path" --max-new-tokens 8
```
