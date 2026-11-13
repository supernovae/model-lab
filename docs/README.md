# Documentation

Each subject has one page, and other pages link to it. New to Lab? Start with
the [README](../README.md), then the first two pages under Start.

## Everyday commands

```sh
klab try DELTA.yaml --vs BASELINE.yaml     # train and score one change
klab sweep sweep.yaml                      # several values or seeds of a change
klab report TRY_ID                         # read the sealed result
klab probe TRY_ID --vs BASELINE_RUN --backend cpu --config standard
klab compare TRY_ID --references           # place it next to public models
klab explore TRY_ID                        # look inside the checkpoint
klab dashboard                             # browse everything
```

`try` trains, `probe` scores and `explore` inspects; `report` and `compare`
only read records. A run ID names weights; a try or probe ID names a result. Use
`klab COMMAND --help` for every option.

## Start

| Page | Covers |
| --- | --- |
| [Build a model from your own text](build.md) | Corpus to tokenizer to model in five commands; keeping a model in its own repository |
| [Lab mode](lab-mode.md) | `try`, `sweep` and `report`: changes against a baseline |
| [Configs](configs.md) | Validating, composing and deriving configs |
| [First real model](first-model.md) | A 17M decoder on FineWeb-Edu with paired seeds |
| [From a toy model to a useful one](from-toy-to-useful.md) | From an instruction-learning toy to a narrow useful model |

## Models

| Page | Covers |
| --- | --- |
| [Architecture](architecture.md) | The decoder and each shipped mechanism |
| [Components](components.md) | The component registry, per-layer schedules, presets and kinds from other packages |
| [Sparse attention](sparse-attention.md) | Block-sparse and local-plus-selected attention |
| [Engram memory](engram.md) | Hashed n-gram memory, byte addressing and portable packs |

## Data

| Page | Covers |
| --- | --- |
| [Datasets](datasets.md) | Source kinds, pinning, preparation, shared tokenizers and Hub credentials |
| [Whole-document batching](document-batching.md) | Document groups, unpadded rows and appending data to a run |
| [Token store](token-store.md) | The reusable binary token store and mixtures |
| [Corpus preparation](corpus-preparation.md) | Corpus Forge releases through to a prepared bundle |

## Training

| Page | Covers |
| --- | --- |
| [Training](training.md) | The trainer, smoke runs, multi-GPU replicas and keeping accelerators busy |
| [Checkpoints](checkpointing.md) | Resume, recover, extend, promote, import weights and retention |
| [Memory](memory.md) | The memory estimate, accumulation, recomputation and offload |
| [Post-training](post-training.md) | Instruction tuning, distillation, preference optimization, chat templates and export |
| [Metrics](metrics.md) | Logged metric names, forecasts and progress |

## Evaluate and inspect

| Page | Covers |
| --- | --- |
| [Results](results.md) | Record IDs, reports and how lookup works |
| [Probes](probes.md) | Probe tiers, verdicts, the final split, reference models and execution-scored code benchmarks |
| [Evaluation](evaluation.md) | Held-out loss and fixed evaluation panels |
| [Inspecting models](models.md) | Generate, chat and explore a checkpoint |
| [Local API](local-api.md) | Serving a checkpoint over HTTP |
| [Speculative decoding](speculative-decoding.md) | Prompt-lookup and draft-model speculative decoding |
| [Dashboard](dashboard.md) | The dashboard pages |

## Runtime and hardware

| Page | Covers |
| --- | --- |
| [Runtime](runtime.md) | Backends, precision, interpreters, profiles, staging, readiness and resource envelopes |
| [Workspaces and storage](workspaces.md) | The work directory, which option selects what, verified reuse, monitoring and cleanup |
| [Intel Arc](intel-xpu.md) | Intel GPUs |
| [MLX on Apple Silicon](macos-mlx.md) | Apple Silicon with MLX |
| [Containers](containers.md) | CPU, CUDA, ROCm and XPU images |
| [Hosted GPU environments](hosted-environments.md) | Colab and full-SSH GPU hosts |

## Beyond one comparison

Lab mode needs none of these.

| Page | Covers |
| --- | --- |
| [Experiments](experiments.md) | Choosing a launch path, queues and matrices |
| [Workers](workers.md) | Independent local, SSH and Colab workers |
| [Lab jobs](lab-jobs.md) | Bounded CPU handoffs from an external scheduler |

## Contributing

| Page | Covers |
| --- | --- |
| [Test feedback](test-speed.md) | Fast and slow suites and the CI jobs |

`tests/test_docs.py` checks that every page is listed here once, that links and
anchors resolve, and that every `klab` command a page shows exists with the
options shown. A change to a command, config field or default updates its page
in the same pull request.
