# Documentation index

Every page under `docs/` is listed here once. Pages in the sections below
describe how Lab behaves now; a page is the one place its subject is
explained, and other pages link to it. Dated records live under
[`history/`](history/README.md) and are evidence, not instructions.

New to Lab? Start with the [README](../README.md), then the first section
below.

## Start here

Navigation and the shortest paths.

| Page | The place for |
| --- | --- |
| [Using Kybern Model Lab](using-klab.md) | Everyday commands and where to go next |
| [Build a model from your own text](build.md) | Corpus to tokenizer to model with five commands; keeping a model in its own repository |
| [Lab mode: from a YAML delta to a scored comparison](lab-mode.md) | `try` and `report`: one change against a baseline |
| [First real model: a 17M dense decoder on FineWeb-Edu](first-model.md) | A 17M decoder on a real corpus with repeated seeds |
| [From flashcards to a useful local assistant](from-toy-to-useful.md) | From a toy exercise to a narrow useful assistant |
| [TinyStories recipe location](tinystories-microlab.md) | Pointer to the TinyStories starter recipe in Workshop |

## Model architecture

What a decoder is built from and how to change it.

| Page | The place for |
| --- | --- |
| [Decoder architecture](architecture.md) | The decoder and each shipped mechanism |
| [Model components, per-layer schedules and architecture presets](components.md) | The component registry, per-layer schedules, presets and kinds from other packages |
| [Sparse attention](sparse-attention.md) | Block-sparse and local-plus-selected attention, with measurements |
| [Latent Attention](mla.md) | Latent attention |
| [Mixture of Experts](moe.md) | Mixture-of-experts feed-forward |
| [Engram Memory](engram.md) | Engram memory overview |
| [Context-gated token memory and dense-checkpoint adaptation](ngram-memory.md) | Token n-gram memory and adapting a dense checkpoint to it |
| [Raw UTF-8 byte addressing](byte-addressing.md) | Byte-addressed memory and its preprocessing contract |
| [Portable Engram](portable-engram.md) | Portable Engram packages |
| [Model scaling and accounting](model-scaling.md) | Parameter accounting conventions |
| [References](references.md) | Papers the mechanisms follow |

## Data and tokenizers

Getting text into prepared arrays.

| Page | The place for |
| --- | --- |
| [Declare, pin and reuse datasets](datasets.md) | Source kinds, locks, snapshots, coverage and budgets |
| [Hugging Face access](huggingface-access.md) | Hub credentials |
| [Reusable token store](token-store.md) | The reusable binary token store and mixtures |
| [Derive a bounded window from an existing encoded cache](prepared-prefix.md) | A bounded window from an existing encoded cache |
| [Reuse a verified parent when preparing a fresh append](prepared-append.md) | Encoding only newly appended documents |
| [Independent document microbatches](document-microbatching.md) | Whole-document microbatches |
| [Matched story packing](matched-story-packing.md) | Matched story packing |
| [Explicit target-group continuation](target-group-continuation.md) | Changing targets per update at a checkpoint boundary |
| [Corpus-to-prepared-bundle public surface](corpus-preparation.md) | The release corpus-to-bundle interface and its phase map |
| [Corpus Forge provenance and shape experiments](corpus-provenance.md) | Origin, verification and lineage classifications |
| [Corpus Forge rights policy (prospective v2)](rights-policy.md) | Training, redistribution and weight-licensing decisions |

## Training

Running, continuing and sizing a run.

| Page | The place for |
| --- | --- |
| [Training, checkpoints, and local continuation](training.md) | The trainer, smoke runs, forecasts and keeping accelerators busy |
| [Disk checkpoints, recovery, and promotion](checkpointing.md) | Disk checkpoints, resume, recovery and promotion |
| [Scientific checkpoint retention](checkpoint-milestones.md) | Checkpoint retention policy |
| [Checkpoint exposure](checkpoint-exposure.md) | Exposure accounting per checkpoint |
| [Training memory accounting](memory.md) | The memory estimate and what it assumes |
| [Gradient accumulation](gradient-accumulation.md) | Micro-batches and accumulation |
| [Activation recomputation](activation-checkpointing.md) | Activation recomputation |
| [Activation offload](offload.md) | Activation offload |
| [Chat-oriented training](instruction-training.md) | Chat records and loss modes |
| [Post-training: instruction tuning, distillation, preference and export](post-training.md) | Parent checkpoints, objectives, chat templates, preference pairs and export |
| [Reproducibility and artifact identity](reproducibility.md) | What a run records and which continuation to choose |
| [Metrics](metrics.md) | Logged metric names and meanings |

## Evaluating and inspecting

Reading what a model learned.

| Page | The place for |
| --- | --- |
| [Probe battery: fast-fail checks for native checkpoints](probe-battery.md) | Probe tiers, verdicts and final evaluation |
| [Probe declarations](probe-config.md) | Probe declarations |
| [Try declarations](try-config.md) | Try declarations |
| [Explorer declarations](explore-config.md) | Explorer declarations |
| [Finding a lab result](result-lookup.md) | Finding a result by ID or path |
| [Read a result](result-reports.md) | Reading run, try and research reports |
| [Experiment evidence](evidence.md) | Evidence levels and held-out evaluation |
| [Capability experiments](capabilities.md) | Capability cards |
| [Fixed panels during training](active-validation-panels.md) | Fixed panels evaluated during training |
| [Named development heldout panels](named-heldout-panels.md) | Named development panels |
| [Controlled benchmarks](controlled-benchmarks.md) | Public benchmark tasks through the probe entry point |
| [Continuation/source overlap diagnostic](memorization.md) | Continuation and source overlap diagnostic |
| [Withheld-fact diagnostic](withheld-facts.md) | Withheld-fact split manifests |
| [Explore retained checkpoints](checkpoint-exploration.md) | Exploring retained checkpoints |
| [TinyText model guide: find, inspect and try a checkpoint](tinytext-model-guide.md) | Verify, complete and chat with a checkpoint |
| [A local API for verified checkpoints](local-api.md) | The local HTTP API |
| [Speculative decoding with n-gram drafts](speculative-decoding.md) | Prompt-lookup speculative decoding |
| [Dashboard tour](dashboard.md) | The dashboard |

## Runtime and hardware

Where and on what a run executes.

| Page | The place for |
| --- | --- |
| [Runtime policy](runtime.md) | Backends, precision, runtime environments and policy |
| [Location and runtime selection](location-runtime-controls.md) | Work directories and runtime selection |
| [Intel Arc acceleration](intel-xpu.md) | Intel Arc |
| [Native MLX on Apple Silicon](macos-mlx.md) | Apple Silicon with MLX |
| [Run Model Lab in a CPU container](containers.md) | The CPU container |
| [CUDA container](containers-cuda.md) | The CUDA container |
| [ROCm image for Radeon RX 7900 XTX](containers-rocm.md) | The ROCm container |
| [Intel XPU container](containers-xpu.md) | The Intel XPU container |
| [Accelerator image builds](container-builds.md) | How accelerator images are built |
| [Optional hosted GPU environments](hosted-environments.md) | Hosted GPU environments |
| [Capacity-aware execution and verified reuse](capacity-aware-execution.md) | Reuse and host-bounded preparation |
| [Experiment workspaces and retained evidence](workspaces.md) | Experiment workspaces and retained evidence |
| [Source, definitions and runtime data](repository-storage.md) | What lives in the repository and what does not |

## Scheduling and release workflows

Beyond one local run. Lab mode needs none of these.

| Page | The place for |
| --- | --- |
| [Independent experiments and controlled comparisons](experiments.md) | Independent experiments and matrices |
| [Independent workers](workers.md) | Independent workers |
| [Bounded shared-host lab jobs](lab-jobs.md) | Bounded shared-host jobs |
| [Launch contracts](launch-contracts.md) | Which launch route to use |
| [Chain a training program](experiment-programs.md) | Multi-phase training programs |
| [Campaigns: declare and execute an experimental workflow](campaigns.md) | Campaigns with approvals |
| [Validate an iteration with the lab](iteration.md) | Validating an iteration |
| [Portable recipe inspection](portable-recipes.md) | Recipe inventories, `config compose` and directory validation |
| [Portable experiment ownership](portable-experiment-migration.md) | What a recipe owns and what Lab owns |
| [Portable preparation recipes](preparation-recipes.md) | Portable preparation recipes |
| [Compiled native preparation bindings](preparation-composition.md) | Compiled preparation bindings |
| [External prepared-input workflow locks](workflow-recipes.md) | Prepared-input workflow locks |
| [Document loading and portable experiment boundaries](document-contracts.md) | The strict document loader and recipe context |
| [External research authoring resources](external-research-resources.md) | Authoring studies from an external resource directory |
| [Offline artifact adapters](verification-adapters.md) | Offline artifact verification commands |
| [Evidence-only publication drafts](publication-reports.md) | Evidence-only report drafts |

## Contributing to Lab

Working on the code.

| Page | The place for |
| --- | --- |
| [Fast test feedback](test-speed.md) | Fast and slow suites and CI jobs |
| [Model construction readiness](lab-readiness.md) | The mechanism readiness check |

## Other collections

| Collection | What it holds |
| --- | --- |
| [Research workbench](research/README.md) | Study authoring, lessons, the experiment ledger and the scientific roadmap |
| [Architecture decision records](decisions/README.md) | Why earlier design choices were made; not a description of current behavior |
| [History](history/README.md) | Retirement maps, acceptance records, audits and review records, each dated |

## Keeping this current

- A change to a command, a config field or a default updates its page in the
  same pull request.
- A new page is added to this index, or to `history/` if it is a dated record.
- A page that only records what happened on a date moves to `history/`.
- `tests/test_docs.py` checks that every page is indexed, that links and
  anchors resolve, and that every `klab` command shown in a current page
  exists with the options shown.
