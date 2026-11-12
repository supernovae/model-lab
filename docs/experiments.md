# Experiments beyond one comparison

Most work is one change against one baseline: use [lab mode](lab-mode.md).
This page covers running many independent runs: queues, seed and architecture
matrices, and sending work to other machines.

## Choose a path

| Need | Command | Guide |
| --- | --- | --- |
| Compare one local change | `klab try` | [lab mode](lab-mode.md) |
| Train, resume, extend or promote a checkpoint directly | `klab train` | [checkpointing](checkpointing.md) |
| A reusable readiness bundle or a disposable pilot | `klab stage` | [runtime](runtime.md) |
| Queue a config or an independent matrix, here or on another machine | `klab experiment submit` | [workers](workers.md) |
| Supervise a reserved native phase with an attempt budget | `klab attempt run` | [corpus preparation](corpus-preparation.md) |

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
