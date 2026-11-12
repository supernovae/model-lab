# First real model: a 17M dense decoder on FineWeb-Edu

Commands below use the [activated, provisioned environment](../README.md#install).

This is the shortest path from the lab-mode tools to one small real model,
using only existing configs, loaders and commands:

- [`configs/first_model_20m_cuda.yaml`](../configs/first_model_20m_cuda.yaml):
  one consumer GPU, bf16, about 49M training tokens. It runs on CUDA as
  written, or on ROCm with `--backend rocm` (see [setup](#0-setup)).
- [`configs/first_model_20m_cpu.yaml`](../configs/first_model_20m_cpu.yaml):
  the same model, data and eval protocol, with a 4.9M-token budget sized for
  a CPU.

| | |
|---|---|
| Model | dense decoder, 8 layers, width 384, 6 heads, FFN 1024, context 512, tied embeddings |
| Parameters | **17,308,032** (embedding 3.1M, attention 4.7M, FFN 9.4M), from `klab inspect` |
| Tokenizer | 8k BPE from [`configs/tokenizer_fineweb_scale_8k.yaml`](../configs/tokenizer_fineweb_scale_8k.yaml) |
| Data | FineWeb-Edu `sample-10BT` at a pinned revision, streamed and capped at 60M train tokens; the validation documents come after the training documents |
| Paths | `klab-work/experiments/dense-scale/` (tokenizer, prepared data) and `.../first-model/runs` (gitignored) |

49M tokens is about 3 tokens per parameter, far below compute-optimal
(about 20). The aim is a real model that exercises the whole workflow,
not a converged one.

## Runtime estimates

| Step | Estimate | Basis |
|---|---|---|
| Tokenizer (20k documents) | about 1 min | 3k documents took 9 s here |
| Data preparation (60M tokens) | 5–15 min, mostly download | 2M tokens prepared in 8 s here |
| Training, GPU config (49M tokens) | **about 15–45 min** per arm, depending on the GPU | about 5 PFLOP (6·N·D); bf16 eager PyTorch on a current consumer card (CUDA or ROCm). Not measured yet: the first run records `targets_per_second` in `progress.json` |
| Training, CPU config (4.9M tokens) | about 2 h per arm on an 8-vCPU box; less on a 12–16-core desktop | **measured** 716 tokens/s on an 8-vCPU box (fp32, deterministic, micro-batch 8). Estimated peak RAM about 5.4 GB (`klab inspect`) |
| Probe, standard tier (`--backend cpu`) | under 1 min | 15–29 s on the smoke run |
| Probe, full tier (`--backend cpu`) | 1–5 min | 61 s on the smoke run with lm-eval datasets cached; lm-eval 4 tasks at `limit=50` |
| Explore | seconds | 4 s on the smoke run |

A `try` trains both arms: a baseline and a candidate. The baseline is reused
by later tries with the same seed, code and data, so the first try costs two
trainings and later ones cost one.

## 0. Setup

```sh
export KLAB_WORK_DIR="$PWD/klab-work/experiments/first-model"
mkdir -p "$KLAB_WORK_DIR"
uv sync --locked --extra cpu --extra lmeval      # CPU env: probes, compare, dashboard
klab tokenizer train configs/tokenizer_fineweb_scale_8k.yaml
klab data prepare configs/first_model_20m_cuda.yaml
```

Both first-model configs share one tokenizer, one data cache and one lab eval
protocol: the same `seq_len` (512), `micro_batch_size` (16) and
`evaluation.max_batches` (32), so both score the same held-out windows. Their
held-out losses are therefore comparable. Accelerator runs train in bf16, but
scoring and probes run in fp32.

**GPU.** Accelerators need a registered runtime
([runtime](runtime.md#register-an-interpreter)); a `--backend`
flag alone does not authorize one. Provision one once, then pass the same
`GPU` flags to every training command below. Choose exactly one device block.

```sh
export KLAB_RUNTIME_DIR="$HOME/.local/share/klab/runtimes"
```

**NVIDIA CUDA** (Linux x86_64, existing Python 3.14):

```sh
klab runtime env provision cuda-gpu \
  --recipe cuda-cu126-v1 --python "$(command -v python3.14)" --json
klab runtime env doctor cuda-gpu --precision bf16 --json
GPU=(--runtime cuda-gpu)
```

**AMD ROCm** (RX 7900 XTX / gfx1100, Linux or WSL2 with AMD's driver stack):

```sh
klab runtime env provision rocm-7900xtx \
  --recipe rocm-gfx1100-v1 --json
klab runtime env doctor rocm-7900xtx --precision bf16 --json
GPU=(--backend rocm --runtime rocm-7900xtx)
```

The built-in `rocm-gfx1100-v1` recipe is pinned for the RX 7900 XTX. The config
says `backend: cuda`, and a CUDA request never silently selects a
HIP build, so ROCm needs the explicit `--backend rocm` on both arms. Model Lab
does not install GPU drivers: check AMD's WSL/ROCm prerequisites first.
On a GPU without bf16, set `runtime.precision: fp16`.

On a CPU, use the `_cpu` config, set `GPU=()`, and replace `_cuda` with `_cpu`
in the commands below.

## 1. Try a change against the baseline

```sh
cat > wider-ffn.yaml <<'YAML'
question: Does a wider FFN lower held-out loss at 49M tokens?
set:
  model.ffn_dim: 1536
YAML
klab try wider-ffn.yaml \
  --vs configs/first_model_20m_cuda.yaml "${GPU[@]}" --config standard
klab report TRY_ID
```

The first try trains the baseline (the config as-is) and the candidate, then
scores both on the same held-out windows. Any single-variable `set:` works.
The try's own probes (declaration `probe_tier`) run under the try's runtime
authorization; standalone probes in step 3 need `--backend cpu`.

## 2. Seed-noise check: 3–5 paired seeds before believing a verdict

The `±SE` printed next to Δ held-out loss is a window-clustered error over one
pair of runs. It measures evaluation noise, not training noise. At tiny
budgets the seed-to-seed spread was **5–10× larger** than that SE: roughly
0.03–0.2 nats across seeds, against a per-try SE near 0.006. A single-seed "CANDIDATE_LOWER_LOSS" can be pure seed luck
(see the [roadmap](../ROADMAP.md#now)).

Run the same delta on paired seeds (both arms share each seed). A lab delta
changes only the candidate, so first derive a short-budget **baseline** to make
this fit tonight:

```sh
klab config derive configs/first_model_20m_cuda.yaml \
  --set training.max_steps=1000 --set training.max_tokens=8192000 \
  --output "$KLAB_WORK_DIR/first_model_short.yaml"
for seed in 1 2 3; do
  printf 'seed: %s\nprobe_tier: none\n' "$seed" > "$KLAB_WORK_DIR/paired-try.yaml"
  klab try wider-ffn.yaml \
    --vs "$KLAB_WORK_DIR/first_model_short.yaml" "${GPU[@]}" \
    --config "$KLAB_WORK_DIR/paired-try.yaml"
done
```

On a GPU, each short arm is about 1/6 of a full one (roughly 3–8 min). On
CPU, run the same loop against the `_cpu` config, which is already short.
Add seeds 4–5 when the first three disagree in sign.

Read it like this: if every paired Δ has the same sign and the mean is more
than about 2× their spread, treat it as real. Otherwise call it noise,
whatever each try's verdict says. A seed-only A/A try (`set: {seed: 7}`, no
other change) measures the noise floor directly.

## 3. Probe at standard and full tiers

```sh
klab probe CANDIDATE_RUN --vs BASELINE_RUN --backend cpu --config standard
klab probe CANDIDATE_RUN --vs BASELINE_RUN --backend cpu --config full   # needs --extra lmeval
```

`klab report TRY_ID` prints both arms' run ids. A run trained with plain
`klab train` can be probed by its explicit run directory, for example
`klab probe klab-work/experiments/first-model/runs/RUN --backend cpu`.

Keep `--backend cpu`. `probe` has no default backend: without the flag it
reuses the run's saved training backend (`cuda` or `rocm` for a GPU try), and
that needs a runtime authorization, so it fails with "cuda requires
--runtime-profile or a registered compatible worker". With `--backend cpu`
it scores in fp32 on CPU, which needs no authorization and is fine at 17M
parameters. The standard
tier adds in-context fact recall and needle retrieval. Closed-book recall is
skipped as inapplicable, because this model is not trained on
`withheld_facts`. The full tier adds lambada/hellaswag/arc_easy/piqa at
`limit=50`; expect near-chance accuracy at this size, where a few points is noise.

## 4. Place it on the reference curve

```sh
klab compare TRY_ID --references
```

Run the full-tier probe first: compare reads records only and, where lm-eval
evidence is missing, prints the probe command that produces it. This compares
lm-eval accuracy with pinned Pythia-70M/160M and
SmolLM2-135M/360M, using paired SEs, within one benchmark group only. Held-out loss is
**not comparable** with references (different tokenizer), and the output says
so. Nothing is downloaded or re-scored, and no model is loaded, so `compare`
(like `report` and `dashboard`) takes no backend flag.

## 5. Look inside

For a custom sentence, save `explore.yaml` (see [looking inside](models.md#look-inside)):

```yaml
explore_config_version: 1
text: The water cycle describes how water moves
```

```sh
klab explore CANDIDATE_RUN --config explore.yaml
```

`explore` already defaults to CPU, so it needs no flag after a GPU
try. 17.3M parameters is under the explorer's 60M cap, so you get per-token loss,
top-k guesses, attention maps for all 8 layers × 6 heads, and weight stats.
A run trained with `klab train` (not `try`) also needs
a `runs_dir: klab-work/experiments/first-model/runs` field in the
explorer declaration (relative to that file), or pass the run directory directly.

## 6. Read the dashboard

```sh
klab dashboard
```

- **Home**: the newest verdict and next-step commands.
- **Experiments**: the try's banner, probe table, and per-probe trends across
  the seed tries. Use **Compare** for paired deltas; cells read better, worse
  or within noise.
- **Models**: the catalog row (17.3M resident and active) and the Pareto
  frontier against the reference diamonds on lm-eval accuracy.
- **Behaviors**: greedy generations side by side (look for loops), recall hits
  and misses, needle by length, and calibration.
- **Explorer**: the exploration from step 5.

See the [dashboard tour](dashboard.md).
