# Training

`klab train CONFIG --run-id NAME` trains one config on one host. It prepares
data if needed, validates at step zero, at the configured cadence and at the
end, writes verified checkpoints and records metrics and events in the run
store: the config's `logging.root_dir`, or `--runs-dir` when given. Commands that
read runs (`eval`, `chat`, `generate`, `report --kind run`, `dashboard`) default
to `$KLAB_WORK_DIR/runs`.

Data is packed as token IDs with one EOS after each document, cut into blocks of
`training.seq_len` and shuffled. A block can span documents; EOS separates them
but does not mask attention. For whole-document training see
[document batching](document-batching.md). Training budgets count supervised
next-token targets.

## Where it runs

`runtime.backend` is `cpu`, `mps`, `cuda`, `rocm`, `xpu` or `auto` with the
PyTorch engine (`auto` picks an available device and never falls back silently),
or `runtime.engine: mlx` with `backend: metal`. Accelerator commands select a
registered interpreter with `--runtime ID`; see [runtime](runtime.md).

For bitwise repeatable CPU runs, keep `training.deterministic: true` (the
default). Accelerators run the same lifecycle but are not bitwise identical to
CPU.

## Prepare before you hold an accelerator

`klab train` tokenizes a missing data cache itself, which for a large corpus can
take a long time while a GPU waits. Prepare first on CPU, then make training
refuse to prepare:

```sh
klab data prepare RUN.yaml
klab train RUN.yaml --run-id RUN --require-prepared
```

With `--require-prepared`, a missing cache fails immediately and names the
`data prepare` command to run.

## Smoke runs

```sh
klab train configs/smoke_cpu.yaml --run-id dense-smoke
klab train configs/smoke_moe_cpu.yaml --run-id moe-smoke
klab train configs/smoke_sparse_cpu.yaml --run-id sparse-smoke
klab train configs/smoke_combined_cpu.yaml --run-id combined-smoke
```

A smoke run checks the path through training, checkpointing, evaluation and
metrics. It says nothing about quality.

## Keep accelerators busy

An accelerator idles whenever the trainer waits on something other than an
update. [`configs/accelerator_scale.yaml`](../configs/accelerator_scale.yaml)
sets all of the following and is the starting point for a long run.

| Setting | Effect |
| --- | --- |
| `checkpoint.background` (default `true`) | Checkpoints are copied to host memory, then written, hashed and verified off the training thread. Training waits only for the copy. |
| `checkpoint.every_tokens`, `every_minutes` | Checkpoint by trained tokens or wall time. Set `every_steps: null` to use only these. |
| `checkpoint.save_on_new_best: false` | A better validation loss no longer writes a checkpoint by itself. |
| `evaluation.every_tokens` | Validate by trained tokens. Set `every_steps: null` to use only this. |
| `logging.every_steps` | Record update metrics every N updates (first and last always kept). |
| `training.deterministic: false` | Lets the backend pick its fastest kernels and use all host threads. |
| `runtime.precision: bf16`, `attention.implementation: sdpa`, `runtime.compile_transformer_blocks: true` | Faster updates where the backend supports them. |
| `runtime.compile_output_head_loss: true` | Compiles the output projection and FP32 cross-entropy together (PyTorch XPU, BF16). |

Background checkpoints need host memory for one copy of the weights and
optimizer state; with less than 1.5 times that available, a checkpoint is
written inline. The final checkpoint, retention-policy runs and MLX runs always
write inline. One checkpoint is written at a time, and a failed background write
fails the run at the next update. Metrics and `progress.json` are written at most
once a second, so the dashboard can trail training slightly.

`runtime_final_observation` records where the run's wall time went: `checkpoint`
(training waited), `checkpoint_background` (overlapped), `validation`, and the
other phases.

Measured on one Arc Pro B70 with a 155M-parameter dense model (bf16, 200 updates
of 4,096 tokens, about 1.9 GB per checkpoint), one run per row:

| Configuration | End to end | Waiting on checkpoints | Validation | Checkpoints |
| --- | --- | --- | --- | --- |
| Default cadences | 68.8 s | 10.6 s | 9.4 s | 6 |
| Token cadences, no best saves | 60.2 s | 10.5 s | 5.3 s | 3 |
| Plus SDPA, compiled blocks, relaxed determinism | 51.9 s | 11.9 s | 10.6 s | 3 |

About 8 s of each wait is the final checkpoint, written inline. The last row's
validation includes compiling the blocks.

## Several GPUs on one host

`runtime.secondary_device_indices` lists more devices of the same backend. Each
holds a replica in its own process; one update's microbatches are dealt across
the primary and the replicas, their gradients are summed on the primary in a
fixed order, and the primary alone clips, steps and saves. The update and its
checkpoint are the same as on one device, so a run can resume with or without
secondaries.

```yaml
runtime:
  backend: xpu
  device_index: 0
  secondary_device_indices: [1]
training:
  gradient_accumulation: 16   # every device needs a share of each update
```

Before each slice the primary copies its updated weights into shared host
memory, which every replica reads, and each replica writes its gradients into
its own shared buffer, which the primary adds to its own in replica order. That
takes host RAM of about four bytes per trainable parameter for the weights plus
as much again per replica: 11.7 GB for a 730M model on four devices. Updates
that used replicas report `performance/weight_publish_seconds`,
`primary_compute_seconds`, `replica_compute_seconds`, `device_wait_seconds` and
`gradient_merge_seconds`, and `batch/devices_used` shows how many devices took
part. A persistent device wait means the devices or
their shares are uneven. Clipping, the optimizer step and reporting are fixed
costs per update, so with more devices raise `gradient_accumulation` rather than
shrinking updates. `benchmarks/device_busy.py` shows device idle time directly.

An update with a single microbatch runs on the primary alone. FP16, activation
offload and memory-allocation runs are refused with secondaries. Validation runs
on the primary.

On Intel XPU every training process sees only its own card. A process that can
see several Arc cards also holds its device memory in host RAM, so Lab sets
`ZE_AFFINITY_MASK` to the configured device before the process loads Torch,
and starts each replica the same way; `device_index` keeps naming the physical
card. A `ZE_AFFINITY_MASK` you set yourself is left alone, and indices then
count within the cards it shows.

Measured on two Arc Pro B70s with a 730M dense model, 4,096-token rows and
131,072 targets per update, compiled, on a desktop whose display ran on the
first card:

| Devices | Targets/s | Fall in host available memory |
| --- | --- | --- |
| first card | 6,773 | 12.1 GiB |
| second card | 8,124 | 8.3 GiB |
| both, primary on the first | 12,682 (1.87× the first card) | 20.9 GiB |

Each update waits for the slower device: here the first card, which shared its
time with the display. Publishing weights and merging gradients took about
0.45 s of each 10.3 s update. Host readings include every other program on
the machine. Before Lab confined each process, one card in a process that could
see both took about 37 GiB of host RAM.

## Forecasts and progress

```sh
klab inspect CONFIG --estimate-runtime --json
klab stage CONFIG --through warmup --output "$WORK/stages/warmup"
klab train CONFIG --stage-bundle "$WORK/stages/warmup"
klab runtime status RUN_ID --runs-dir "$RUNS" --json
```

`inspect --estimate-runtime` forecasts update time from earlier runs on the
same runtime and device; without one it reports the forecast as unavailable.
`stage --through warmup` runs a disposable pilot that measures real updates
without advancing training. While training, progress counts completed
supervised targets, and `optimizer_only_eta` estimates the remaining optimizer
work, not the whole job. Progress is emitted as JSON lines on stderr;
`runtime status` and the dashboard read the latest snapshot. See
[metrics](metrics.md#forecasts-and-progress).

## Stopping

Ctrl-C, SIGTERM or creating the printed `CANCEL` file stops at the next update
boundary with a validated checkpoint. A cancellation before the run exists stops
without creating it. Never kill a process while it writes a checkpoint.

## Evaluation during training

Validation runs at step zero, at `evaluation.every_steps` or `every_tokens`, and
at the end or interruption, scoring `evaluation.max_batches` batches of the
held-out split. Every report is bound to its checkpoint, with the exact number of
targets scored. `evaluation.panels` replaces the ordinary sample with named fixed
panels; see [evaluation](evaluation.md#during-training).

After a run completes, `train` writes a bounded post-training triage (loss
trend, a short generation probe and integrity checks); `klab report RUN_ID --kind
run` shows it beside the run's evidence.

## Context length

Pretrain on short rows and lengthen the context afterwards. A model learns
most of what it knows from 2,048 to 4,096-token rows, and longer rows cost
much more per token. Measured on one Arc Pro B70 with a 730M dense model:

| Rows | Settings | Targets/s | Device peak |
| --- | --- | --- | --- |
| 4,096 | compiled blocks and loss | 8,076 | 23.1 GiB |
| 8,192 | compiled blocks and loss | 5,493 | 28.6 GiB |
| 16,384 | activation recomputation | 2,348 | 21.2 GiB |
| 32,768 | recomputation and optimizer state offload | 1,285 | 22.1 GiB |

To extend a pretrained model, train a short child stage on long documents
(`document_start` packing; see [datasets](datasets.md)), a few percent of the
pretraining tokens, with a longer `training.seq_len`. Name the pretrained run
as `training.parent_checkpoint`, raise `model.max_seq_len`, and change RoPE
for the longer context (see [rotary positions](architecture.md#rotary-positions)):

```yaml
model:
  max_seq_len: 16384
attention:
  rope_base: 500000.0           # ABF, or:
  rope_scaling:
    type: yarn
    original_max_seq_len: 4096
training:
  seq_len: 16384
  parent_checkpoint: /runs/pretrain
```

A promoted child may change only `attention.rope_base`,
`attention.rope_scaling` and `model.max_seq_len`, which may not shrink, from
its parent's architecture; RoPE has no weights, so every tensor loads
unchanged. The run's manifest records the change as a `rope_extension`
decision under `resource_decisions`. A full-state resume still requires the
same architecture.

Which schedule works best for a model is an experiment. Run each as a `klab
try` arm from the same parent, on the same long held-out data, and compare
them on held-out loss at long positions and on retrieval from far back in the
context (see [evaluation](evaluation.md) and [probes](probes.md)). Check the
short-context probes too: extension usually costs a little there.

## Continue, branch and recover

Checkpoints, resume, promotion, budget extension and recovery are described in
[checkpointing](checkpointing.md). Memory planning, gradient accumulation,
recomputation and offload are in [memory](memory.md).
