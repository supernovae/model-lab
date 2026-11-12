# Using Kybern Model Lab

Commands below use the [activated, provisioned environment](../README.md#first-run).

Start with [TinyStories](tinystories-microlab.md) for prepared inputs and a first
comparison, or [lab mode](lab-mode.md) if you already have a RunConfig and
tokenizer. The normal local path is `try` → `report`, followed by the next check
suggested by the evidence. A plan, worker queue or Campaign is not a prerequisite.

## Everyday commands

Run from the checkout in the [locked environment](../README.md#first-run).
`DELTA.yaml` and `BASELINE.yaml` are your files; replace the ID placeholders with
values printed by `try` or `report`.

```sh
klab try DELTA.yaml --vs BASELINE.yaml
klab report TRY_ID
klab probe CANDIDATE_RUN --vs BASELINE_RUN --backend cpu --config standard
klab compare TRY_ID --references
klab explore CANDIDATE_RUN
klab dashboard
```

`try` trains; `probe` scores; `explore` computes checkpoint diagnostics. `report`
and `compare` read sealed records only. The dashboard reads saved lab results
and exposes on-demand explorer computation. A **run ID** identifies weights;
a **try/probe ID** identifies a result record. `probe` and `explore` search
`WORK_DIR/lab/runs`. For other saved runs, both commands accept an explicit run path or the
`runs_dir` field in their declarations ([probe](probe-config.md),
[explore](explore-config.md)).
`report` and `compare` accept sealed result paths as well as record IDs.

`probe` and `explore` also accept try/probe record IDs or paths, selecting the
record's exact candidate/target checkpoint rather than the run's current
`latest`; see [result lookup](result-lookup.md) for pinning, paths and missing evidence.

Use `klab COMMAND --help` for exact arguments. Current help groups commands
but does not remove them; the [feature roadmap](../ROADMAP.md#current-direction)
summarizes the direction for a smaller CLI. There is no `advanced` or `release` command alias.

## Find the next guide

| Need | Guide |
| --- | --- |
| Build a model from your own text | [Build guide](build.md) |
| Train a larger first model with repeated seeds | [17.3M FineWeb-Edu walkthrough](first-model.md) |
| Understand verdicts, tiers and references | [Probe battery](probe-battery.md) |
| Browse experiments, model comparisons and internals | [Dashboard](dashboard.md) |
| Select CPU, ROCm, CUDA, XPU, MPS or MLX | [Runtime](runtime.md), [hosted environments](hosted-environments.md) |
| Prepare a new dataset or tokenizer | [Datasets](datasets.md), [Hub access](huggingface-access.md) |
| Train directly or continue a checkpoint | [Training](training.md), [checkpointing](checkpointing.md) |
| Complete text, chat or serve saved weights | [Model guide](tinytext-model-guide.md), [local API](local-api.md) |
| Change architecture or instruction objectives | [Architecture](architecture.md), [instruction training](instruction-training.md) |
| Schedule independent runs or matrices | [Workers](workers.md), [experiments](experiments.md) |
| Declare release lineage, approvals and recovery | [Training programs](experiment-programs.md), [Campaigns](campaigns.md), [lifecycle](research/lifecycle-recovery.md) |
| Run focused code checks | [Test feedback](test-speed.md) |

Advanced interfaces below serve specific authoring and diagnostic needs. They
are not extra steps to complete after every lab try.

## Derive a validated config variant

Use `config derive` to author a new standalone v2 run configuration without
rewriting YAML by hand. Each `--set` value is strict JSON, so strings must be
quoted. The command publishes a new YAML file and adjacent operational
`.derivation.json` receipt; neither an existing payload nor receipt is
replaced. The receipt is not an artifact identity or trusted experiment lock.

```sh
mkdir -p "$KLAB_WORK_DIR/handoffs"
klab config derive configs/smoke_cpu.yaml \
  --set optimizer.peak=0.001 --set training.micro_batch_size=2 \
  --output "$KLAB_WORK_DIR/handoffs/smoke-peak.yaml" --json
```

The derived configuration preserves path targets relative to the source
configuration, validates the complete typed configuration, and records the
requested assignments plus observed field delta. It does not verify prepared
data, tokenizer bytes, checkpoints, continuation compatibility, or runtime
execution; use the normal preparation, experiment, and checkpoint commands for
those boundaries.

Paths supplied by an assignment, including a whole replacement object, are
anchored to the source YAML directory, never the output directory or cwd.
Optional default fields such as `optimizer.decay_steps` remain patchable.
Equal assignments are allowed; only actual normalized changes enter the
observed delta. The output parent must exist and have no symlinked components.
Changing a token budget or batch setting never adjusts other scientific
settings implicitly or turns a config into an approved continuation.

## Scratch and artifact locations

The work root's `scratch/` child must be a real directory contained in the resolved
root, never a symlink. If scratch is symlinked, choose a different `--work-dir` or
`KLAB_WORK_DIR`. Symlink aliases for the work root itself remain supported.

Implicit persistent state defaults to `${XDG_DATA_HOME}/klab` when `XDG_DATA_HOME` is absolute and nonempty, otherwise `~/.local/share/klab`. For substantial campaigns, set `KLAB_WORK_DIR=/data/klab` on a sufficiently large filesystem. Global `--work-dir PATH` (before the subcommand) overrides that environment variable; an explicit relative path stays relative to the current directory. Durable `experiments/`, `runs/` and artifact/receipt stores share that root; temporary files go to its disposable `scratch/`, and optional `cache/` is reconstructable, not evidence. Read-only inspection does not create the root. Long-lived payloads inside **any** Git checkout trigger a containment warning but are not redirected; check explicit destinations too.

Explicit `--runs-dir`, `--store`, `--output`, `dataset.cache_dir` and `logging.root_dir` retain their literal paths, including historical receipts. Selecting an external root does **not** move an existing `klab-work/` directory or reinterpret prior data; intentional relocation requires a new location binding/manifest and re-verification of referenced bytes. Study submission and research scaffolding derive implicit workspaces from `<work-dir>/experiments/<study-name>`; SSH hosts need their own adequately sized persistent root.


For a replicated experiment, use one [experiment workspace](workspaces.md) and pass its persistent `runs/` directory to every producer and consumer. `train --runs-dir PATH` overrides an execution destination without editing the YAML; historical config paths remain unchanged.

## Native diagnostic interfaces

These commands expose operational or descriptive observations, not model-quality
gates or authorization to train:

```sh
klab semantic probe probe.yaml --json
klab memorization analyze overlap.yaml --json
klab data benchmark-preparation benchmark.yaml --json
```

`semantic probe` authenticates packs and runs one bounded CPU/PyTorch forward with
explicitly supplied vectors, encoder identities, masks and times. Initialized
backbones/adapters are labeled untrained; checkpoint backbones retain their native
generation identity, including declared restored allocation adapters. The
[semantic lesson](research/semantic-memory.md) scaffolds a ready `probe.yaml`.
No implicit text encoder, training callback or run store is created.

`memorization analyze` binds an exact UTF-8 file digest or an explicit completed
row from a natively authenticated generation panel. It compares only supplied
source passages and preserves raw hashes, normalized overlap and unavailable
bounded-edit values. It emits no eligibility threshold or memorization verdict;
see [the declaration and provenance contract](memorization.md).

`data benchmark-preparation` exclusively claims a declared absent workspace,
generates bounded offline data, trains a tiny tokenizer and prepares each case
in a fresh thread-bounded worker. It retains failures, memory-coverage labels and
exact per-size prepared-identity comparisons, without model training or automatic
batch recommendations. See [runtime observations](runtime.md).

`data prepare`, `stage` and Campaign `plan|status|next|explain|apply|resume` accept
`--observations-output PATH.json`. The parent must already exist; the destination
must be absent and outside scientific inventories. The optional envelope records
the complete handler separately from nested native phases, actual counter coverage,
null/unavailable measurements and operation-scoped snapshot verifier statistics.
Diagnostic publication failures warn without changing the workflow result.
See [phase-observation boundaries](capacity-aware-execution.md).

Generic snapshots reuse full authentication only within one preparation, staging
or Campaign operation; every invocation rechecks config bindings and inventory.
`--cold-verify` disables reuse on preparation, staging and Campaign
`plan|status|next|explain|apply|resume|approve`. Imports and publication remain cold;
see [snapshot verification lifetime](datasets.md#campaign-and-existing-input-reuse).
