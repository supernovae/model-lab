# Runtime

A run executes in one process on one host. Its manifest records the engine,
backend, precision, device, memory readings and framework versions it actually
used. That record describes the machine that ran it, not what another machine
can do.

- An **engine** is the framework running the model: PyTorch or MLX.
- A **backend** is its execution route: PyTorch CPU, NVIDIA CUDA, AMD ROCm,
  Intel XPU or Apple MPS; MLX uses Metal on Apple Silicon.
- A **device** is the selected processor.

## Choose where a run executes

```yaml
runtime:
  engine: pytorch       # pytorch | mlx
  backend: auto         # auto | cpu | mps | cuda | rocm | xpu | metal
  device_index: 0
  precision: fp32       # auto | fp32 | bf16 | fp16
```

With PyTorch, `auto` picks an available MPS device, then CUDA or ROCm, then XPU,
then CPU. An explicit backend that is unavailable is an error; Lab never retries
on CPU. ROCm uses PyTorch's `cuda` device API but is recorded as `rocm`, and a
CUDA request never selects a HIP build. CPU and MPS have only index 0.

`precision: auto` means FP32. BF16 and FP16 are checked by a disposable
subprocess that runs a tiny forward, backward and optimizer update at that
precision. CPU and MPS reject FP16; BF16 runs through autocast. Parameters,
gradients and the memory estimate stay FP32.

MLX is a separate engine: `engine: mlx` with `backend: metal`. It supports FP32
dense and native block-sparse attention, recomputation, resume, promotion,
evaluation, generation and chat, and decodes without a KV cache. MoE, Engram
memory, MLA, sliding-window attention, mixed precision, offload and Adafactor
are refused. See [MLX on Apple Silicon](macos-mlx.md).

### Host and backend are separate

The host OS and the compute backend are independent. WSL2 is a Linux
environment; it does not select AMD, NVIDIA, Intel or CPU. macOS does not imply
the GPU. Discovery records `host_os`, `host_environment` and `host_architecture`
beside the backend, and never initializes a GPU to do so.

Under WSL2, keep training data, caches and checkpoints in the Linux filesystem;
crossing into Windows-mounted storage is slow (see Microsoft's
[filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems)).
Memory readings describe the Linux environment, not the Windows host. Install
drivers per vendor: NVIDIA's [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/)
uses the Windows driver, PyTorch's [Intel GPU guide](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
lists supported XPU setups, and AMD's
[PyTorch install guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
covers ROCm. Lab never installs drivers.

### Framework versions

The base package has no PyTorch dependency. `klab runtime env` works without
it; the rest of the CLI needs a provisioned framework. For CPU development,
`uv sync --locked --extra cpu --dev`. Never sync the CPU extra into an
accelerator environment: it replaces the vendor PyTorch.

| Runtime | Package | Notes |
| --- | --- | --- |
| Linux CPU | `torch 2.15.0+cpu` (`uv.lock`) | CPU tests |
| Mac CPU / MPS | `torch 2.15.0` (`uv.lock`) | Validate on the Mac |
| Intel XPU | `torch 2.15.0+xpu` | [Intel Arc](intel-xpu.md) |
| AMD gfx1100 | `torch 2.15.0+rocm10.0`, ROCm SDK `10.0.0` | [ROCm image](containers.md#rocm) |
| NVIDIA container | `torch 2.14.1+cu132` | [CUDA image](containers.md#cuda) |
| Mac MLX | `mlx==0.32.4`, `mlx-metal==0.32.4` | Separate engine |

Matching versions do not mean identical kernels, numerics or speed.

| Feature | PyTorch CPU | PyTorch MPS | PyTorch XPU / ROCm / CUDA | MLX Metal |
| --- | --- | --- | --- | --- |
| Dense training, evaluation, generation | Yes | Yes | Yes | Yes |
| BF16 | Probe | Probe | Probe | No |
| FP16 training | No | No | Probe with GradScaler | No |
| Block-sparse attention | Reference | Reference | Reference kernels | Native |
| MoE, Engram, MLA, sliding window | Yes | Yes | Yes | No |
| Compiled transformer blocks | Opt-in | Not validated | Opt-in | No |
| Probes and explorer | Yes | Yes | Yes | No |

Discovery reports `tested_precisions` and `tested_features` only for what it
exercised. Lab never switches device, precision, engine, tokenizer or model to
make a run work.

## Register an interpreter

`klab runtime env` manages interpreters on this host. It keeps four things
apart: hardware it observed, Python environments installed, registered
profiles, and authorization to run. A GPU seen by `rocminfo` does not make a
CPU PyTorch interpreter ROCm-ready.

```sh
klab runtime env discover --json
klab runtime env register cpu-py314 \
  --python "$PWD/.venv/bin/python" --backend cpu --json
klab runtime env list --json
klab runtime env show cpu-py314 --json
klab runtime env doctor cpu-py314 --json
klab runtime env profile cpu-py314
klab stage configs/runtime_smoke_cpu.yaml \
  --through inspect --runtime cpu-py314 --output /absolute/disposable/stage
```

Accelerator interpreters register the same way:

```sh
klab runtime env register cuda-local --python /absolute/cuda/bin/python --backend cuda --json
klab runtime env register xpu-local --python /absolute/xpu/bin/python --backend xpu --json
klab runtime env register mps-local --python /absolute/mps/bin/python --backend mps --json
klab runtime env register mlx-local --python /absolute/mlx/bin/python --backend metal --json
```

`register` checks the installed Lab source, backend, device and requirements,
then runs the same optimizer check as `doctor` before saving anything. It does
not touch data, train or create checkpoints. If the source does not match this
checkout it prints the fix:
`uv pip install --python <registered-python> --no-deps --editable <checkout>`,
which keeps the vendor PyTorch. `unregister ID` removes only the registration.

- **Registry:** `$XDG_CONFIG_HOME/klab/runtimes.yaml`, else
  `~/.config/klab/runtimes.yaml`. Unknown fields, duplicates and relative
  paths are refused. Never copy it to another host; register there.
- **Runtime root** (where `provision` installs): `KLAB_RUNTIME_DIR`, else
  `$XDG_DATA_HOME/klab/runtimes`, else `~/.local/share/klab/runtimes`. It must
  be absolute, outside Git and not equal to the work directory. Registered
  interpreters can live anywhere.
- **Discovery** checks the active Python, the project `.venv`, and up to 32
  runtime-root children and 32 registered interpreters, each with a time and
  output bound. Each reports `READY`, `UNAVAILABLE`, `SOURCE_MISMATCH`,
  `ERROR`, `NOT_PROVISIONED` or `UNOBSERVED`. `READY` is not authorization to
  run.

### Provision a vendor environment

`provision` is the only command that installs packages. It builds a new
environment under the runtime root from a pinned recipe, installs Lab with
`--no-deps`, and registers it only after a real device check passes.

| Recipe | Pins | Check |
| --- | --- | --- |
| `rocm-gfx1100-v1` | [`requirements/rocm-gfx1100.txt`](../requirements/rocm-gfx1100.txt): PyTorch 2.15.0, ROCm 10.0.0 | BF16 optimizer update |
| `cuda-cu126-v1` | [`requirements/cuda-cu126.txt`](../requirements/cuda-cu126.txt): CUDA 12.6 PyTorch | FP16 update with GradScaler |
| `xpu-arc-v1` | Upstream PyTorch XPU | FP32 optimizer update; see [Intel Arc](intel-xpu.md) |

```sh
# On a host whose driver stack already supports the device:
export KLAB_RUNTIME_DIR="$HOME/.local/share/klab/runtimes"
klab runtime env provision rocm-7900xtx --recipe rocm-gfx1100-v1 --json
klab runtime env doctor rocm-7900xtx --json
klab stage /absolute/rocm-run.yaml \
  --through warmup --runtime rocm-7900xtx --output /absolute/disposable/rocm-stage
```

Check the driver and GPU against the vendor's support matrix first (for AMD,
the [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)).
Provisioning needs Linux x86_64, an existing Python 3.14 (`--python` to choose
it), a new ID, local writable storage with 20 GiB and 100,000 inodes free, and
never downloads an interpreter. Network, FUSE and Windows-share filesystems are
refused; overlay mounts are accepted when writable with enough headroom.

The target keeps three evidence files: `provision-registration.json` (the
intent, written before registering), `provision-ready-staged.json` and
`provision-receipt.json` (the READY receipt, two links to one file). A failure
writes `provision-failure.json` with `registration_state`:

| State | Meaning | What to do |
| --- | --- | --- |
| no intent | Failed before registering | Keep the logs; do not reuse the target |
| `refused` | Registration rejected; registry unchanged | Keep any conflicting entry |
| `unknown` | Registry write may have happened | Inspect the registry path in the intent |
| `registered` | Registered; receipt not published | Resolve the storage error before use |

Never rerun provisioning into the same target or fabricate a receipt; choose a
new ID. Lab never repairs the registry for you.

## Runtime profiles

Accelerator execution needs a registered runtime ID, a runtime profile file or
a registered worker. CPU runs without one, and `auto` resolving to a GPU does not
authorize it.

```yaml
runtime_profile_version: 1
id: vendor-runtime
python: /absolute/vendor-environment/bin/python
engine: pytorch
backend: rocm
device_index: 0
requirements:
  torch_hip: true
  bf16: true
  device_name_regex: "Radeon"
```

Profiles refuse unknown fields and need an absolute, executable interpreter.
`torch_hip` is valid only for ROCm; `bf16` needs a disposable optimizer update;
`device_name_regex` searches the device name. `klab runtime probe PROFILE --json`
reports the interpreter, framework, device and source identity it finds.

`train`, `stage`, `eval`, `generate`, `chat`, `serve` and `explore` take
`--runtime ID` or `--runtime-profile FILE`; `try` and `probe` take `--runtime ID`
or `runtime_profile` in their declaration. Single-config `experiment submit`
accepts them too.
A different interpreter replaces the CLI process once, with the same arguments
and environment, after authorization and before any workspace is created.
Profiles and probe results are operational evidence; they never enter a config
or cache identity.

## Inspect and stage before a run

```sh
klab inspect configs/runtime_smoke_cpu.yaml --json
klab stage configs/runtime_smoke_cpu.yaml --through smoke --output "$WORK/stages/smoke"
klab stage configs/runtime_smoke_cpu.yaml --through warmup --output "$WORK/stages/warmup"
```

`inspect` builds a shape-only parameter inventory and memory estimate without
loading the model, tokenizer or data; see [memory](memory.md). A stage bundle is
a verified copy of the effective inputs: `inspect` and `validate` check them,
`smoke` and `warmup` run short disposable pilots that never advance the real
run. A stage output must be new, or an identical complete bundle is verified
and reused. If a pilot exceeds its ceiling, staging writes a proposal and stops
without editing the config. `train --stage-bundle DIR` links the pilot to the
run; it never initializes from pilot weights.

`stage --pilot-deadline-policy POLICY.yaml` bounds pilots. The defaults:

```yaml
pilot_deadline_version: 1
initialization_timeout_seconds: 1800
no_progress_timeout_seconds: 1200
absolute_timeout_seconds: 7200
termination_grace_seconds: 5
smoke:
  initialization_timeout_seconds: 3600
warmup:
  absolute_timeout_seconds: 14400
```

Initialization covers input verification and model, data and runtime setup.
No-progress resets only when real counters advance. The absolute cap applies
even while progress continues. `smoke` and `warmup` override per pilot. A
timeout stops an owned pilot and keeps its bundle; it says the pilot did not
finish in time, not that the model does not fit.

### Readiness smoke

`klab readiness smoke` runs every shipped mechanism through the public CLI on
tiny synthetic data: inspect, preflight, prepare, stage, train, verify,
evaluate, generate and resume a child.

```sh
klab readiness smoke --output "$KLAB_WORK_DIR/experiments/readiness-1"
klab readiness smoke --family dense --output "$KLAB_WORK_DIR/experiments/readiness-2"
```

| Family | Config |
| --- | --- |
| Dense | `smoke_cpu.yaml` |
| MoE | `smoke_moe_cpu.yaml` |
| Block sparse | `smoke_sparse_cpu.yaml` |
| Local + selected blocks | `smoke_local_selected_cpu.yaml` |
| Sliding window | `smoke_sliding_cpu.yaml` |
| MLA | `smoke_mla_cpu.yaml` |
| Token Engram | `smoke_memory_cpu.yaml` |
| Byte Engram | `smoke_byte_memory_cpu.yaml` |
| Combined | `smoke_combined_cpu.yaml` |

All families run on CPU in FP32. The output must be a new directory; every
command, outcome and identity goes into `readiness.json`, and a failure leaves
its runs in place. Passing shows the wiring works, not that a model learns or
fits a GPU. New architecture code needs a focused test and a path here first.

### Scale up on a GPU

`configs/scale_dense_160m_cuda.yaml` (162,554,880 parameters) is the first CUDA
capacity pilot, and `configs/scale_dense_1b_cuda.yaml` (950,621,952) a larger
skeleton. Run these from the provisioned CUDA environment (`uv run --locked
--no-sync`, or the interpreter directly) so the CPU extra never replaces its
PyTorch. Both read FineWeb-Edu sample-10BT with
`configs/tokenizer_fineweb_scale_8k.yaml`. They are capacity checks, not
recipes for a useful model.

```sh
WORK="$KLAB_WORK_DIR/experiments/dense-scale"
klab inspect configs/scale_dense_160m_cuda.yaml --json
klab workspace preflight configs/scale_dense_160m_cuda.yaml
klab tokenizer train configs/tokenizer_fineweb_scale_8k.yaml
klab data prepare configs/scale_dense_160m_cuda.yaml
klab stage configs/scale_dense_160m_cuda.yaml --runtime cuda-local \
  --through warmup --output "$WORK/staging/160m"
klab batch calibrate configs/scale_dense_160m_cuda.yaml \
  --output "$WORK/calibration/160m"
```

`workspace preflight` budgets three checkpoint generations with 25% headroom
plus prepared data, against the real free bytes and inodes of each destination.
`batch calibrate` tries microbatch and accumulation pairs with the same
effective batch, ranks them by measured throughput after two warmup updates,
and writes `proposal.yaml` only for a stable candidate with headroom; the
original config is unchanged.

Before a long allocation, train a short parent and resume a child from it:

```sh
klab train configs/scale_dense_160m_cuda.yaml --runtime cuda-local \
  --run-id dense-160m-parent --runs-dir "$WORK/runs" --stop-after-step 10
klab train configs/scale_dense_160m_cuda.yaml --runtime cuda-local \
  --run-id dense-160m-child --runs-dir "$WORK/runs" \
  --resume "$WORK/runs/dense-160m-parent/checkpoints/latest.json" --stop-after-step 20
klab checkpoint verify "$WORK/runs/dense-160m-child/checkpoints/latest.json" --json
```

Run long jobs from a fixed, committed checkout, and develop in a separate
worktree while they train.

### Accept a provisioned host

To check that Lab trains, checkpoints and evaluates on a newly provisioned
host, run the 20-step, 640-target runtime smoke with its registered runtime:

```sh
WORK="$KLAB_WORK_DIR/experiments/runtime-acceptance"
CONFIG=configs/runtime_smoke_cpu.yaml     # or the ROCm or XPU smoke config
RUNTIME=cpu-py314
RUN_ID="runtime-acceptance-$(date -u +%Y%m%dT%H%M%SZ)"
klab tokenizer train configs/tokenizer_smoke.yaml
klab inspect "$CONFIG" --json
klab stage "$CONFIG" --runtime "$RUNTIME" --through warmup --output "$WORK/staging/$RUN_ID"
klab train "$CONFIG" --runs-dir "$WORK/runs" --runtime "$RUNTIME" \
  --stage-bundle "$WORK/staging/$RUN_ID" --run-id "$RUN_ID"
klab eval "$RUN_ID" --runs-dir "$WORK/runs" --runtime "$RUNTIME"
```

The stage, manifest and evaluation must record the requested backend, training
must commit 20 steps and 640 targets, and evaluation must use that checkpoint.
For CUDA, copy the config and change `runtime.backend`.

## Forecasts and progress

```sh
klab inspect CONFIG --estimate-runtime --json
klab stage CONFIG --through warmup --output "$WORK/stages/forecast"
klab train CONFIG --runs-dir "$WORK/runs" --stage-bundle "$WORK/stages/forecast"
klab runtime status RUN_ID --runs-dir "$WORK/runs" --json
```

Lab keeps four records apart:

- **Planning estimate** (`inspect --estimate-runtime`): from up to 25 earlier
  runs whose runtime signature matches exactly (device, framework and driver
  versions, precision, architecture, optimizer, sequence length, batch shape,
  recomputation and offload). Without one it is unavailable.
- **Warmup estimate**: from the measured updates of a matching warmup pilot,
  excluding the first.
- **Live progress**: completed supervised targets against the budget, with
  `optimizer_only_eta`. When targets stop advancing the state becomes
  `NO_PROGRESS` and the ETA is withdrawn; the run keeps going.
- **Final observation** (`runtime_final_observation`): preparation, updates,
  validation, checkpointing and wall time, kept separate.

Progress is JSON lines on stderr, so stdout results are unchanged. Missing
measurements are `null`, never zero. A compatible history is a guide, not a
promise: contention and thermals change throughput without changing the
signature.

`performance/step_seconds` and `tokens_per_second` cover update work only.
Memory sampling reads process RSS and allocator counters where the backend has
them. MPS and MLX share unified memory with the host, so the two are never
added together.

## Resource envelopes

`data prepare`, `stage`, `train`, `explore`, single-config `experiment submit`
and `experiment prepare` accept `--resource-envelope FILE`; `try` and `probe`
take `resource_envelope` in their declaration.

```yaml
resource_envelope_version: 1
max_rss_bytes: 2147483648
max_host_memory_fraction: 0.5
min_available_ram_bytes: 134217728
min_swap_bytes: 0
min_disk_bytes: 268435456
min_inodes: 128
max_workers: 4
max_queue_depth: 1
spill_to_disk: true
```

Every limit is optional. Unknown fields fail. RAM and swap come from psutil and
storage from the destination's filesystem; a missing measurement fails only if
a limit needs it. `max_host_memory_fraction` is process RSS over host RAM, not
accelerator memory. Limits are checked before writing, at streaming boundaries
and before publishing. An envelope never changes batch size, sequence length,
precision or data.

A try envelope may add a section that applies only to its arms, and is refused
anywhere else:

```yaml
resource_envelope_version: 1
lab_try:
  max_wall_seconds_per_arm: 600
  tokenizer_batch_documents: 32
  tokenizer_batch_source_bytes: 1048576
```

## Prepared data

`data prepare` writes a progress stream and a sibling `<cache>.preparation.json`
with records, bytes, tokens, throughput, sampled memory, free disk and the time
spent iterating the source, encoding, writing, syncing and hashing. These are
operational measurements and never enter the cache identity.

- **Batches.** For `local_text`, `--tokenizer-batch-documents` (1 to 256,
  default 256) and `--tokenizer-batch-source-bytes` (up to 8 MiB, default 1 MiB)
  bound each encode call. An oversized document fails rather than being cut.
  Encoding uses the shared [encoder pool](token-store.md#one-encoder).
- **Supervision.** `contiguous-eos-v6` caches state their supervision:
  `{kind: all_tokens}` stores no mask, `{kind: token-loss-mask-v1}` stores a
  boolean mask for both splits.
- **Hashing.** Arrays are hashed while they are written, and later checks in
  the same process reuse that digest only while the file's identity is
  unchanged. A new process always hashes from disk.

### Interrupted preparation

Streaming preparation writes sealed chunks of at most 4,096 records and 16 MiB,
ending on whole documents. A restart verifies the source, tokenizer and every
sealed chunk, discards only the unfinished one, and continues from the exact
record where it stopped without re-encoding earlier ones. A finished train split
survives an interrupted validation split. Preparation holds a lock, so a live
producer is never mistaken for a crashed one.

### Parallel preparation

Streaming `local_text` and `snapshot` preparation can encode chunks in parallel
within each split:

```yaml
resource_envelope_version: 1
max_workers: 4
preparation_shard_workers: 4
max_queue_depth: 4
spill_to_disk: true
```

`max_workers` is the total thread budget and `preparation_shard_workers` (1 to
32) the number of chunk workers, both bounded by measured CPU and memory.
Without an envelope, preparation sizes itself: one four-thread worker per 32M
declared tokens, up to capacity. Worker counts never change the prepared arrays:
one coordinator selects records, enforces quotas and commits chunks in source
order. Parallel encoding needs a deterministic ByteLevel BPE tokenizer; others
run serially and the receipt says why. The receipt's `shards` and `worker_pool`
sections report per-split and per-worker measurements.

To measure preparation on a host:

```sh
klab data benchmark-preparation declaration.yaml --json
```

The declaration (`klab-preparation-benchmark-v1`) names a new workspace and
bounds the generated sizes (32 MiB each, 64 MiB total), coordinates, batches,
threads and timeout. Each coordinate runs in a fresh offline CPU process and
reports its arrays and cache identity; a mismatch fails the report. It measures,
it does not choose settings.

## Storage and checkpoint headroom

`experiment inspect` and `workspace preflight` report checkpoint write volume
apart from peak disk growth. Training writes a generation at the start and the
end, plus at each step, token or minute cadence and each new best; coincident
triggers write once. A 100-update run with checkpoint and validation every 40
updates writes four (initial, 40, 80, final).

`checkpoint.keep_periodic: false` keeps latest, previous and best: up to three
generations, plus room for a fourth in flight. `checkpoint.retention` keeps a
declared set within a byte budget; see
[checkpointing](checkpointing.md#keep-milestones). Preflight checks future
checkpoints, the run's copy of prepared arrays and any unverified cache growth
against each destination's free bytes and inodes. A worker's copy needs its own
headroom.

## Memory proposals and optimizers

The memory policy can propose a complete alternative config; it never changes
the one you train. `fast`, `balanced`, `low_memory` and `max_fit` order the
candidates differently. Changing microbatch and accumulation keeps the effective
batch; changing precision or sequence length is a scientific change and is
shown as one. Recomputation and offload are covered in [memory](memory.md).

AdamW is the default. PyTorch Adafactor factors its second moment, and the
learning-rate schedule sets its relative step-size cap rather than an absolute
rate. Changing the optimizer needs a fresh or promoted run, never a resume.
