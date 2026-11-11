# Disk checkpoints, recovery, and promotion

Commands below use the [activated, provisioned environment](../README.md#first-run).

A checkpoint generation is a durable, immutable local snapshot shared by the PyTorch and MLX execution paths. It is not activation recomputation, gradient accumulation, or an inference KV cache. Checkpoints are written at successful update boundaries only; partial accumulation gradients are not a resumable state.

## Verify before reuse

```sh
sparselab train --runs-dir sparselab-work/runs configs/runtime_smoke_cpu.yaml --run-id runtime-part --stop-after-step 10
sparselab checkpoint inspect sparselab-work/runs/runtime-part/checkpoints/latest.json --json
sparselab checkpoint verify sparselab-work/runs/runtime-part/checkpoints/latest.json --json
sparselab train --runs-dir sparselab-work/runs configs/runtime_smoke_cpu.yaml --run-id runtime-resumed \
  --resume sparselab-work/runs/runtime-part/checkpoints/latest.json
```

Both engines store generations under `sparselab-work/runs/<run-id>/checkpoints/step_<step>_gen_<generation>/`. Each finalized generation has a manifest and canonical safetensor weight shards/index; its engine selects the native training-state codec. `latest.json` and `best.json` are atomic lookup indexes; verified generation manifests and file digests are the durable truth. `checkpoint inspect` reports inventory and lineage, not a standalone integrity claim. `checkpoint verify` validates the selected path and returns structured failures with a nonzero exit when invalid.

The checkpoint manager writes a temporary sibling, verifies the completed generation, atomically publishes it, then updates lookup pointers. A writer lease prevents concurrent writers/recovery in the same run. Recovery scans finalized generations and can repair stale projections/pointers, but an explicit corrupt `--resume` selection fails rather than silently falling back.

## Full resume

`--resume` creates a new child run. It restores canonical model weights, named optimizer state, warmup/cosine schedule, optional scaler, committed counters, data cursor, cadence watermarks, cumulative timing, and the engine's safe RNG envelope. The child records `parent_run_id` and the exact parent `checkpoint_sha256`; it does not bind itself to a moving `latest` pointer.

By default full resume requires the same engine/backend, architecture semantics, tokenizer/data/package identities, optimizer settings, schedule horizon, training budget, micro-batch/accumulation, sequence length, deterministic setting, and verified run-owned artifacts. The explicit runtime-optimization transition described below permits a narrow, recorded exception for supported execution settings. Source/runtime or device-identity drift requires `--allow-runtime-drift`; that produces an explicit best-effort record and never bypasses scientific compatibility. A completed budget cannot be resumed.

Use `--recover RUN_DIR` only to seek the latest valid finalized generation after interruption/corruption. It creates a child after checking compatible state; it is not permission to overwrite the existing run. Old or incomplete historical artifacts that lack required continuation state are not full-resumable.

## Explicit terminal budget extension

`train --extend-budget CHECKPOINT` takes an immutable full-state generation
directory, not `latest.json` or a weights-only promotion. It is mutually
exclusive with `--resume`, `--recover`, and `--promote`. The parent must be
terminal at both its configured update and target caps; both old and new target
caps must equal `max_steps × seq_len × micro_batch_size × gradient_accumulation`.
The new caps must both grow, and the new AdamW `optimizer.decay_steps` must equal
the parent's `training.max_steps`. The parent may have an implicit horizon or
an explicit horizon equal to that same original update cap. Only name, logging,
checkpoint paths, the two caps, and explicit representation of that original
horizon may differ. A relocated tokenizer pathname is accepted only after its
bytes match the authenticated run-owned parent tokenizer. Engine/backend and
artifact identity remain verified; source/runtime drift still needs explicit
`--allow-runtime-drift` and appears in the child's `budget_extension` decision.
ExperimentPlan budget extensions do not authorize that drift.

The child restores weights, AdamW moments and learning rate, scaler, cursor,
RNG, counters and lineage; it continues at the original cosine floor rather
than restarting a longer cosine. A subsequent interrupted child uses ordinary
strict `--resume` with its **unchanged** extended config. The source checkpoint
and run store remain immutable. This mode is PyTorch-only; explicit AdamW decay
horizons are not supported by MLX.

## Promotion is weights-only continuation

`--promote CHECKPOINT` creates a child with compatible canonical model weights but **fresh** optimizer, schedule, scaler, RNG, cursor, and zero committed counters. Its destination config supplies the new data/budget/runtime choices. Promotion still requires the same architecture semantics, tokenizer contents, and portable package identity when present; equal tensor shapes alone are insufficient. It does not resize a backbone or turn dense weights into MoE.

Use full resume for the same experiment and promotion for a compatible
weights-only starting point. The public path is local; remote scheduling and
controller-side artifact ingestion are separate contracts.

## Import compatible external weights

`weights import` accepts closed, validated source formats:
`sparselab_legacy_v1` and `hf_llama_safetensors`. The latter is a narrow
bias-free dense Llama mapping with compatible attention/RoPE/tokenizer semantics,
including exact dense or grouped-query layouts when the destination's
`num_kv_heads` exactly matches the source `num_key_value_heads`. It is not a
general Hugging Face, quantization, or chat-template importer.

```sh
sparselab weights import SOURCE OUTPUT_RUN \
  --config COMPATIBLE_V2.yaml --format sparselab_legacy_v1 \
  --source-tokenizer SOURCE_TOKENIZER.json --provenance PROVENANCE.json
```

The destination config must preserve architecture semantics and exact compatible
tokenizer contents. Provenance format 1 requires explicit source and license
statements; importing does not grant redistribution rights. The output is a
verified, promotion-only generation with null native-state codec fields, not
invented optimizer or RNG state. Use its checkpoint with `train --promote`.

Historical `.pt` files remain inspectable/verifiable through the same checkpoint
CLI. Their hash, migrated configuration, tensor shapes/values, and tied aliases
are checked. Their reported scope is always `weights_only`; they cannot become
full-resumable by supplying `--config`.

## Training-state comparison API

For offline diagnostics, `sparselab.training.state_digest` provides
`canonical_training_state(snapshot, observation=observed_update)` and
`compare_training_states(left, right)`. Load independently verified native
`TrainingSnapshot` objects at matching committed update boundaries and supply
the actual step, tokens seen, learning rate, loss, gradient norm, overflow flag,
and retry state. The comparison reports differing components or
`BITWISE_EQUIVALENT`; it uses exact tensor bytes and typed state, without a
numerical tolerance. Missing loss or gradient observations may be `None`, so
equal unavailable observations establish no equality of those measurements.

The versioned fingerprint covers weights and trainability, optimizer, schedule,
RNG, scaler, cursor, counters, scientific configuration, validation/lineage
values and checkpoint cadence. It excludes run/file/source identities,
operational config paths and cumulative timings. A disabled time-based cadence's
minutes watermark is excluded; enabled time-based cadence and unknown snapshot
fields fail closed. Supported tensors are dense PyTorch tensors and compatible
native-endian NumPy arrays; unsupported state types fail rather than being
silently omitted.

This API does not authenticate checkpoints or the contents of excluded paths,
validate optimizer mathematics, grant source compatibility or authorize a run.
Equal fingerprints describe only the supplied states and observations; they do
not prove successful resume, future trajectory equivalence, or model quality.

## Evaluation, best, and lineage

Child lineage discovery ranks generation manifests before reading tensor payloads.
It verifies the explicitly selected parent and any candidate that can contribute
the best local or inherited loss, falling back when a candidate is corrupt.
Unrelated retained generations do not require a payload scan during an ordinary
continuation. Explicit recovery still verifies the full retained store; full-state
resume verification and older-parent boundaries are unchanged.

Validation runs at step 0, configured evaluation intervals, and final/interruption boundaries. A new best validation loss causes a checkpoint; a cadence-only checkpoint has `validation_loss: null` rather than inheriting stale attribution. `best.json` points only to a verified evaluated generation physically present in that run. A resumed child may carry an ancestor `lineage_best` record whose checkpoint is not local; it is lineage metadata, not a fabricated local best pointer.

When selecting an older parent checkpoint, inherited best-state discovery stops
at that selected generation and step. Later same-run evaluations cannot leak
backward into the child's lineage. An ancestor best that is not copied locally
remains explicitly unavailable as a local checkpoint.

`checkpoint.keep_periodic: false` prunes only after successful publication and protects latest, best, and the immediately previous verified generation. Keeping fewer generations reduces recovery/history options; it never makes old state safer.

In [actual signal/recovery acceptance](https://github.com/supernovae/tiny-sparse-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/acceptance/signal_recovery_2026_09_22.json),
SIGTERM ended cleanly at a committed update; deliberate latest-generation
corruption made explicit resume fail before child creation. Explicit recovery
selected the previous verified generation and completed a new child update,
leaving that selected parent's files unchanged.

## Native MLX generations

MLX uses the common format-2 generation, pointer, artifact, lineage, recovery,
and evidence contract. Its native state is `training_state.json` plus
`optimizer.safetensors`; both are hashed and fsynced before generation
publication. The reader checks the exact optimizer array inventory, named
groups, counters, schedule, primitive metadata, and Python/NumPy/MLX RNG.
Malformed containers reject the generation rather than escaping recovery.

Offline inspection and verification do not require MLX. Execution does require
the pinned optional runtime. Same-engine continuation restores native state;
cross-engine continuation is canonical-weight promotion with fresh training
state, never an optimizer conversion.

## Optional spot-safety policy

The canonical `sparselab train CONFIG --spot-policy POLICY.json` accepts a
separate operational policy; Python callers pass `spot_policy=SpotSafetyPolicy(...)`
to `train`. Read-only `sparselab checkpoint plan-spot POLICY.json --workspace RUNS_DIR`
prints the decision and capacity without creating a run. Queued worker declarations
do not implicitly inherit this host policy; this interface is explicit direct training.
It adds checkpoint boundaries without editing RunConfig, evaluation
intervals, scheduler, optimizer, data, precision or target budget. Each fresh or
resumed child records its chosen policy, cadence and initial capacity readings
in `spot_safety.json`; a child must explicitly supply its own current policy.
Operational saves retain scientific cadence watermarks; separate child-local
operational time starts at child initialization and resets on every save.
Historical generations and their identities remain unchanged.

The strict JSON fields are `format: "sparselab-spot-safety-v1"`,
`observation_reference` (retained measurement reference),
`checkpoint_write_seconds`, `boundary_seconds`, `restart_seconds`,
`interruption_notice_seconds`, `max_recovery_seconds`, `checkpoint_bytes`,
`checkpoint_inodes`, `reserve_bytes` and `reserve_inodes`. Times must be finite;
write, boundary and restart observations must be positive. Missing observations
fail closed. Boundary latency must include the longest observed update plus
any intervening evaluation before reaching a checkpoint boundary. Write time
must include durable checkpoint commit, and restart cost must include restore
and preparation needed to resume updates on the selected runtime.

Cadence in seconds is `max_recovery_seconds - restart_seconds -
checkpoint_write_seconds - boundary_seconds`; a nonpositive result is rejected.
Interruption notice must cover the observed boundary plus write time. This
bounds the planned replay/restart envelope using supplied observations; it is
not a provider SLA or a guarantee against longer future operations. Scientific
checkpoint triggers remain enabled, and time cadence is checked at native
completed-update boundaries, measured from the last checkpoint start watermark
(conservative with respect to time spent writing). Native signal handling still owns shutdown; this
policy does not add a provider polling service or authorize allocation.

Native storage checks run before engine creation and again before each save.
Required free bytes/inodes are twice the measured generation footprint plus
explicit reserves (staging and committed-generation headroom); existing retained
generations already consume free space. Missing inode observations, unavailable
storage or insufficient capacity reject the operation. Checkpoint growth and
competing writers still require conservative observed bounds and reserves.

Offline tests establish strict loading, cadence, capacity failures and canonical
CLI wiring with training mocked. Live qualification remains separately allocated:
measure full-state durable write size/time, longest boundary latency, and cold
child restart on the actual model/runtime/filesystem; verify notice delivery and
signal-to-durable-commit under bounded interruption, then verify the exact
committed generation and child resume. No fixture establishes these measurements
or provider interruption safety.

Opt-in [`checkpoint.retention`](checkpoint-milestones.md) additionally preserves
scientific milestones and reserves its declared peak bytes plus free-space
headroom. Its keep-set replaces the default latest/previous/best-only bound.
This RunConfig policy differs from experiment-plan `retention.keep_periodic`,
which describes plan storage expectations and does not configure the trainer.

## Explicit runtime optimization on resume

A full-state `train --resume` or `train --recover` may opt in to
`--allow-runtime-optimization` to change dense attention implementation,
activation checkpointing, block compilation, compiled output projection and loss,
determinism, or microbatch and
accumulation while preserving their product. Fixed validation panels retain their
own batch sizes. Without fixed panels, set `evaluation.batch_size` to the prior
microbatch size when changing microbatch so the validation sample is unchanged.
All other training, data, model, optimizer, schedule and budget settings remain bound. The child manifest records
the old and new values and parent checkpoint digest as a best-effort transition;
results are not bitwise comparable when these settings change. Original
checkpoints are not modified. Source/runtime identity changes separately require
`--allow-runtime-drift`.

Set `runtime.compile_transformer_blocks: true` to compile PyTorch transformer
blocks. This currently requires activation checkpointing and offload to be
disabled; startup compilation cost and device compatibility depend on the
installed PyTorch backend. Test the target workload before using it for a long
run.

Set `runtime.compile_output_head_loss: true` to compile the output projection
and summed FP32 cross-entropy together on PyTorch XPU BF16. It defaults to off;
changing it on full-state resume or recovery requires
`--allow-runtime-optimization`. See [compiled output projection and loss](training.md#compiled-output-projection-and-loss)
for its supported scope and exclusions.
