# Training, checkpoints, and local continuation

Commands below use the [activated, provisioned environment](../README.md#first-run).

For an ordinary baseline/candidate comparison use [lab mode](lab-mode.md), or
start with [TinyStories](tinystories-microlab.md). This reference explains direct
training, runtime behavior and explicit checkpoint transitions.

Model Lab packs plain token IDs with one EOS per document. A sequence may cross an EOS boundary: EOS separates records but does not create an attention mask. Source-acquisition budgets include EOS; training budgets count valid next-token targets.

## Local runtime contract

A run uses one local process on one host and one explicitly configured or automatically detected device. Use `runtime.backend: cpu|mps|cuda|rocm|xpu|auto` with the PyTorch engine. `auto` selects only an available local backend; an explicitly unavailable backend fails rather than falling back to CPU. The optional MLX engine requires `runtime.engine: mlx` and `runtime.backend: metal`.

CPU reproducibility uses `runtime.backend: cpu` and `training.deterministic: true`. MPS and other accelerators can exercise the same local lifecycle, but are not claimed to be bitwise-identical to CPU.

## Prepare on CPU before holding an accelerator

`klab train` tokenizes and packs a missing data cache itself, before the model is
built. For a large corpus that can take a long time on CPU while a reserved or
rented accelerator waits. Prepare first, then make the training command refuse
to do that work:

```sh
klab data prepare RUN.yaml                 # CPU only; uses host cores, resumable
klab train RUN.yaml --run-id RUN --require-prepared
```

For `local_text` sources, tokens are kept in a [reusable token store](token-store.md),
so preparing the same data for another model or a larger budget copies them
instead of tokenizing again.

With `--require-prepared`, a published cache for this config is verified and
used; a missing cache fails immediately with the `data prepare` command to run,
and nothing is tokenized. Use it in any launch script that starts on a GPU host.

## Smoke runs

```sh
klab train --runs-dir klab-work/runs configs/smoke_cpu.yaml --run-id dense-smoke
klab train --runs-dir klab-work/runs configs/smoke_moe_cpu.yaml --run-id moe-smoke
klab train --runs-dir klab-work/runs configs/smoke_sparse_cpu.yaml --run-id sparse-smoke
klab train --runs-dir klab-work/runs configs/smoke_combined_cpu.yaml --run-id combined-smoke
```

A smoke run validates a path through training, checkpointing, evaluation, and metrics. It is not a quality benchmark. Compare runs only when data, tokenizer, device, sequence length, token budget, optimizer, and seed are recorded and intentionally matched.

## Compiled output projection and loss

For PyTorch XPU BF16 runs, `runtime.compile_output_head_loss: true` optionally
compiles the output projection and summed FP32 cross-entropy together. It keeps
the same token IDs, tied weights, masking and effective batch; measured speed
depends on the device and workload. The option is currently unavailable with
activation offload, allocation ownership or portability runs. The default is
off. Enabling it on a full-state continuation is an explicit runtime change:
use `--allow-runtime-optimization` and retain its recorded transition.

## Runtime forecast and target progress

Planning is available only from compatible historical optimizer timings; without
matching runtime/device identity, the estimate is unavailable rather than
inferred from model size. An optional disposable warmup adds a separate
measurement without advancing training state:

```sh
klab inspect CONFIG --estimate-runtime --json
klab stage CONFIG --through warmup --output klab-work/stages/warmup
klab train --runs-dir klab-work/runs CONFIG --stage-bundle klab-work/stages/warmup
klab runtime status RUN_ID --runs-dir klab-work/runs --json
```

Live progress uses completed supervised targets as its primary denominator.
`optimizer_only_eta` forecasts remaining optimizer work; it is not a whole-job
ETA. Preparation, validation, checkpoint, reporting, and end-to-end costs stay
separate in the final observation. Evaluation and generation are unavailable in
a training record unless observed as separate operations. Stalled target
progress suspends ETA but does not terminate the run.

Progress JSON Lines are emitted on stderr. `runtime status` and the dashboard
read the latest bounded snapshot plus the final timing record without changing
run state. The dashboard's Training and Runtime views are operational telemetry,
not a scheduler or automatic tuning interface. See [runtime policy](runtime.md#runtime-forecasting-and-progress)
for exact-history matching and interpretation limits.

## Keep accelerators busy

An accelerator is idle whenever the trainer waits on something other than an
optimizer update. These settings bound that time; the
[`accelerator_scale.yaml`](../configs/accelerator_scale.yaml) preset sets all of
them and is the starting point for a long run.

| Setting | Effect |
| --- | --- |
| `checkpoint.background` (default `true`) | A PyTorch checkpoint is copied to host memory, then written, hashed and verified off the training thread. Training waits only for the copy. |
| `checkpoint.every_tokens`, `every_minutes` | Checkpoint by trained tokens or wall time instead of a step count. Set `every_steps: null` to use only these. |
| `checkpoint.save_on_new_best: false` | An improving validation loss no longer writes a checkpoint by itself; `best` is then the best of the cadence checkpoints. |
| `evaluation.every_tokens` | Validate by trained tokens. Set `every_steps: null` to use only this. |
| `logging.every_steps` | Record update metrics every N updates (the first and last are always kept). |
| `training.deterministic: false` | Lets the backend pick its fastest kernels and use all host threads. Runs are then not bitwise repeatable. |
| `runtime.precision: bf16`, `attention.implementation: sdpa`, `runtime.compile_transformer_blocks: true` | Faster updates where the backend supports them. Test the workload first. |

Background checkpoints need host memory for one copy of the weights and
optimizer state. If less than 1.5 times that is available, that checkpoint is
written inline instead. The final checkpoint of a run, milestone-retention
runs and MLX runs are always written inline. One checkpoint is written at a
time: if the next one is due before the previous has finished, training waits
for it. A failed background write fails the run at the next update.

Update metrics and `progress.json` are written at most once a second and are
held while a checkpoint is being written, because small synced writes stall
behind its disk writeback. The dashboard can therefore trail training by that
long.

Every run records where its wall time went in the `runtime_final_observation`
event: `checkpoint` is time training waited, `checkpoint_background` is write
time that overlapped training, and `validation` is inline validation.

Measured on one Arc Pro B70 with a 155M-parameter dense model (bf16, 200
updates of 4,096 tokens, about 1.9 GB per checkpoint):

| Run | End to end | Waiting on checkpoints | Validation | Checkpoints |
| --- | --- | --- | --- | --- |
| Before, default cadences | 105.6 s | 46.2 s (43.8%) | 9.4 s | 6 |
| After, same config | 68.8 s | 10.6 s (15.5%) | 9.4 s | 6 |
| After, token cadences and no best saves | 60.2 s | 10.5 s (17.4%) | 5.3 s | 3 |
| After, plus SDPA, compiled blocks, relaxed determinism | 51.9 s | 11.9 s (22.8%) | 10.6 s | 3 |

About 8 s of the remaining wait in each row is the final checkpoint, which is
written inline; on a long run that is a fixed cost. The last row's validation
time includes compiling the blocks. This is one short run per row on one
machine, not a benchmark of long runs.

## Checkpoint and resume workflow

PyTorch checkpoints are immutable local generations under `klab-work/runs/<run-id>/checkpoints/`; `latest.json` points to the newest validated generation. Verify a checkpoint before continuing it:

```sh
klab train --runs-dir klab-work/runs configs/smoke_moe_cpu.yaml --run-id moe-part --stop-after-step 20
klab checkpoint inspect klab-work/runs/moe-part/checkpoints/latest.json --json
klab checkpoint verify klab-work/runs/moe-part/checkpoints/latest.json --json
klab train --runs-dir klab-work/runs configs/smoke_moe_cpu.yaml --run-id moe-resumed \
  --resume klab-work/runs/moe-part/checkpoints/latest.json
```

Resume creates a child run and retains parent telemetry. It rejects changed model, optimizer, data, tokenizer, budget, engine, or backend configuration. Source or runtime identity drift requires `--allow-runtime-drift` and is recorded as `best_effort`; the flag never bypasses scientific compatibility. Recovery applies the same checks. Completed budgets cannot be resumed. Run-owned data, tokenizer, and memory-package artifacts—including their manifests—are verified before continuation.

For supported dense PyTorch resumes and recovery, `--allow-runtime-optimization` permits a narrow set of execution changes while preserving scientific settings, effective training batch and validation coverage. The child records a best-effort transition; see [explicit runtime optimization on resume](checkpointing.md#explicit-runtime-optimization-on-resume).

An explicit PyTorch/AdamW `--extend-budget CHECKPOINT` is different from ordinary
resume: only a verified terminal full-state generation may grow both whole-update
caps, with `optimizer.decay_steps` fixed to the parent's original step limit.
It retains optimizer moments, cursor, RNG, and the old cosine schedule's floor;
it does not restart a new longer cosine. An interrupted extended child then
uses ordinary `--resume` with its unchanged extended config. The
[checkpointing contract](checkpointing.md#explicit-terminal-budget-extension)
specifies the compatibility checks; the [dense-LM budget study](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/dense-lm-token-budget-v1/results.md)
records one executed ROCm continuation rather than a generic performance guarantee.

Current run manifests retain `manifest_version: 1` with `identity_version: run-identity-v2`. Requested/effective configuration digests exclude machine-local paths; architecture identity covers model and attention semantics separately. Historical identity-less manifests keep their original digest interpretation. A child's `parent_run_id` and `checkpoint_sha256` bind the exact selected parent generation, not a moving pointer.

Full PyTorch continuation uses checkpoint directory format 2 with native state codec version 2. Verification checks the resolved RunConfig and its digests, canonical tensors/aliases/trainability, named optimizer coverage and moment shapes, completed-update schedule and learning rate, counters/cursor, and safe host/selected-device RNG envelopes. PyTorch saves copy the state to host memory and write it off the training thread; see [keep accelerators busy](#keep-accelerators-busy). Older format-2 native envelopes remain weights-only: use `checkpoint verify --weights-only` for inference/promotion eligibility, not as proof that full optimizer continuation is supported.

A POSIX advisory writer lease covers either engine's training session; another writer or recovery operation fails rather than modifying the same run. Explicit `--recover klab-work/runs/<parent-id>` reconciles finalized verified generations, repairs `latest.json`/`best.json`, and starts a child from the selected generation. Temporary or corrupt generations are not adopted. Equal finite validation losses retain the earlier generation as best. `checkpoint.keep_periodic: false` preserves latest, best, and the immediate verified predecessor; it does not preserve a fully verifiable historical learning curve.

Promotion reuses compatible weights with a fresh optimizer/cursor, using the **destination** dataset and budgets. Source architecture semantics and tokenizer contents must match. It is useful for continuing a model on a new domain/chat corpus, not for automatically resizing a backbone or converting dense weights into MoE.

MLX uses the same immutable generation, artifact, provenance, promotion, recovery, and evidence contract. Its native codec stores validated JSON and optimizer safetensors. Same-engine continuation restores native state; cross-engine promotion copies canonical weights and starts fresh training state. Native files can be verified without the MLX SDK; execution still requires it. See [checkpointing](checkpointing.md#native-mlx-generations).

A cancellation already present during startup stops before the next startup phase, without creating a run or inventing a checkpoint. The trainer raises `InterruptedError`; native workers acknowledge a retained regular, non-symlink cancel marker and lab tries seal an interruption without a completed arm. A worker receiving `InterruptedError` without that marker records `FAILED` and re-raises the original error. The trainer's broader existence check also stops on a directory or a symlink to an existing target, but those are not worker cancellation acknowledgment authority; they remain retained failure evidence. If cancellation or a handled signal arrives after run creation but before initial validation, the run records `interrupted` with no new checkpoint. Existing parent checkpoints and cancel markers remain untouched. A stop after committed updates retains the normal safe checkpoint boundary behavior.

Both engines validate at step zero (unless startup was cancelled), each configured cadence, and the terminal update. Exact supervised-target counts, finite loss, protocol settings, and artifact identities accompany each checkpoint-bound report. Non-multiple budgets mask only the remaining valid targets; wholly masked chunks do not update state. Validation selects nonempty blocks before applying its batch limit and binds supervision/byte-address files when present. Training and validation modes are restored; diagnostics are captured before validation can replace them.

## Mechanism-specific diagnostics

The MoE smoke configuration has local Top-K experts with configurable selection. It has no capacity clipping or distributed expert dispatch. Routing metrics include the auxiliary balance loss, entropy, maximum expert fraction, and selected-probability mass; none demonstrates expert specialization by itself.

Block-sparse attention reports selected and available causal key positions, selection ratio, and a relative selected-key score-product estimate. These are resource diagnostics, not a quality conclusion or a custom-kernel speed claim. Use a matched dense run for a controlled ablation.

See [runtime policy](runtime.md), [metrics](metrics.md), and [architecture](architecture.md) for the terms used by run reports.

Opt-in [`checkpoint.retention`](checkpoint-milestones.md) additionally preserves
scientific milestones and reserves its declared peak bytes plus free-space
headroom. Its keep-set replaces the default latest/previous/best-only bound.
This RunConfig policy differs from experiment-plan `retention.keep_periodic`,
which describes plan storage expectations and does not configure the trainer.
