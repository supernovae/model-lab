# Model construction readiness

Model Lab has a reusable local check for its current PyTorch model mechanisms.
It exercises the public CLI with isolated, tiny synthetic data: inspect,
filesystem preflight, preparation, disposable stage smoke, training, checkpoint
verification, evaluation, generation, and full-state child resume. The check
records commands, outcomes, config identities, and checkpoint paths in a named
ignored workspace. It establishes executable wiring and recovery, not learning,
speed, device fit, or model quality.

## Smallest capability gate

Run from the repository root with the locked CPU-development environment. The output must be a
new task-owned directory; the command will not reuse or overwrite one.

```sh
uv run --locked --extra cpu klab readiness smoke \
  --output klab-work/experiments/lab-readiness-YYYYMMDD
```

Use repeated `--family NAME` to limit a runner session to an affected path.
With no filter, the versioned matrix is:

| Family | Source config | Local gate |
|---|---|---|
| Dense | `smoke_cpu.yaml` | CPU/PyTorch FP32 |
| MoE | `smoke_moe_cpu.yaml` | CPU/PyTorch FP32 |
| Block sparse | `smoke_sparse_cpu.yaml` | CPU/PyTorch FP32 reference |
| Sliding window | `smoke_sliding_cpu.yaml` | CPU/PyTorch FP32 |
| MLA | `smoke_mla_cpu.yaml` | CPU/PyTorch FP32 reference |
| Token Engram | `smoke_memory_cpu.yaml` | CPU/PyTorch FP32 |
| Byte Engram | `smoke_byte_memory_cpu.yaml` | CPU/PyTorch FP32 |
| Combined | `smoke_combined_cpu.yaml` | CPU/PyTorch FP32 compatibility |

The command derives runnable configs that redirect tokenizer, data, run store,
staging, and temporary files into its workspace; it does not edit the source
configs. Every family receives its own run IDs and a resumed child. A failed
command leaves its evidence and partially completed runs in place for diagnosis.
The source package digest and each resolved config digest are retained in
`readiness.json`.

New architecture code must first gain a focused contract test and a tiny
end-to-end path here, then a device-specific warmup if its intended backend is
available. Register a model-quality question separately in the research
lifecycle. A small pilot cannot establish architecture advantage. MLX supports
only its documented dense and native block-sparse paths; these readiness runs
execute on CPU. To check workload compatibility with a GPU
environment, run the appropriate pilot there. CUDA block-sparse selection still
uses the reference path until the native CUDA direction in [ROADMAP.md](../ROADMAP.md#longer-term-directions) is implemented
and measured on CUDA hardware.

## Before scaling on a GPU

The first CUDA capacity pilot is `configs/scale_dense_160m_cuda.yaml` (162,554,880
parameters). `configs/scale_dense_1b_cuda.yaml` is a 950,621,952-parameter
construction skeleton. Both use the pinned FineWeb-Edu sample-10BT source and a
separate 8,192-entry tokenizer definition in
`configs/tokenizer_fineweb_scale_8k.yaml`. Their dataset caps and training
ceilings are development settings, not a preregistered scientific comparison or
a recipe for a useful language model. Copy a config and freeze a research
protocol before a long training campaign.

Inspect is shape-only and works without a GPU, corpus, or tokenizer. On this
machine it estimated 3.68 GB training peak and 1.95 GB checkpoint tensors for
the 160M pilot, and 20.02 GB training peak and 11.41 GB checkpoint tensors for
the 950M skeleton. Both physical-fit results were `UNKNOWN` because CUDA was
unavailable. Real checkpoint generations include metadata and temporary writes;
`workspace preflight` budgets three generations with a 25% allowance, plus
prepared-array growth and reserves. It checks free bytes and inodes on the
actual destination filesystems before training publishes a run. Remote dataset
download cache growth is source-dependent and is not included in the local
array estimate; keep additional room for the provisioned source cache.

On a provisioned CUDA host, use its vendor environment as documented in
`AGENTS.md` and `docs/workers.md`. Keep a single named workspace, or copy the
configs to change every explicit tokenizer, data, and run path to another disk.
For an already provisioned environment, the command prefix below is
`uv run --locked --no-sync`; do not replace its CUDA framework with the default
Linux dependency set.

```sh
WORK=klab-work/experiments/dense-scale
mkdir -p "$WORK"
export KLAB_WORK_DIR="$WORK"
uv run --locked --no-sync klab inspect configs/scale_dense_160m_cuda.yaml --json
uv run --locked --no-sync klab workspace preflight configs/scale_dense_160m_cuda.yaml
uv run --locked --no-sync klab tokenizer train configs/tokenizer_fineweb_scale_8k.yaml
uv run --locked --no-sync klab data prepare configs/scale_dense_160m_cuda.yaml
uv run --locked --no-sync klab stage configs/scale_dense_160m_cuda.yaml \
  --through warmup --output "$WORK/staging/160m"
uv run --locked --no-sync klab batch calibrate configs/scale_dense_160m_cuda.yaml \
  --output "$WORK/calibration/160m"
```

The calibration command tries a bounded set of divisible microbatch and
accumulation pairs with the same effective batch. Each disposable candidate
has at least eight updates; the first two are excluded from throughput ranking.
It records measured target throughput, timing spread, peak memory, device,
headroom, and every rejection or failure. Only a stable candidate with measured
headroom gets a separate `proposal.yaml`. The original config is unchanged;
choosing the proposal for a later run is explicit.

Before spending a long GPU allocation, execute a short parent/child training
pilot with the same selected config in both commands:

```sh
uv run --locked --no-sync klab train configs/scale_dense_160m_cuda.yaml \
  --run-id dense-160m-pilot-parent --runs-dir "$WORK/runs" --stop-after-step 10
uv run --locked --no-sync klab checkpoint verify \
  "$WORK/runs/dense-160m-pilot-parent/checkpoints/latest.json" --json
uv run --locked --no-sync klab train configs/scale_dense_160m_cuda.yaml \
  --run-id dense-160m-pilot-child --runs-dir "$WORK/runs" \
  --resume "$WORK/runs/dense-160m-pilot-parent/checkpoints/latest.json" \
  --stop-after-step 20
uv run --locked --no-sync klab checkpoint verify \
  "$WORK/runs/dense-160m-pilot-child/checkpoints/latest.json" --json
```

Record the actual backend, peak, update times, checkpoint write time, and
dataset identity. A near-1B run requires its
own GPU warmup and capacity gate; inspection cannot certify it. A GPU sparse
campaign also requires native CUDA sparse correctness and component benchmarks
before treating that path as ready for scaled execution. Single-host
[multi-device replicas](gradient-accumulation.md#several-devices-on-one-host)
need their own measured throughput on the target devices; multi-host training
and spot-specific cadence remain separate future work; see the
[feature roadmap](../ROADMAP.md#longer-term-directions) for direction and tracking ownership.

Keep implementation and experiment work separate. Commit a tested lab change
and, when authorized, push the branch before launching hours-long remote work;
record the commit, dirty-tree state, package identity, and effective config.
Run from the fixed checkout, leaving active inputs and checkpoints untouched
while other lab development proceeds in a different checkout. Runner sessions
on fixed code use focused readiness checks and staging; code changes receive
their relevant focused tests and broader suite before another long launch.
