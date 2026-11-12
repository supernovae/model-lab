# Checkpoints

A checkpoint is an immutable, verified snapshot written at an update boundary.
It holds the model weights and, for a full-state checkpoint, everything needed to
continue exactly: optimizer state, learning rate schedule, data cursor, counters
and random state.

## Layout and verification

Checkpoints live under `RUNS/<run-id>/checkpoints/step_<step>_gen_<generation>/`.
Each holds a manifest, safetensors weight shards and the engine's training state.
`latest.json` and `best.json` point at generations; the manifests and file
digests are what count.

```sh
klab checkpoint inspect "$RUNS/run/checkpoints/latest.json" --json
klab checkpoint verify "$RUNS/run/checkpoints/latest.json" --json
klab checkpoint verify "$RUNS/run/checkpoints/latest.json" --weights-only --json
```

`inspect` lists contents and lineage. `verify` checks every file and digest, the
config and the training state, and exits non-zero with structured errors when
something is wrong. `--weights-only` checks only what generation or promotion
needs.

A checkpoint is written to a temporary sibling, verified, then published
atomically before the pointers move. A per-run lock keeps two writers out of the
same run.

## Continue a run

All continuations create a new child run that records its parent run and the
exact parent checkpoint digest. The parent is never modified.

| Flag | Starts from | Keeps | Use for |
| --- | --- | --- | --- |
| `--resume CHECKPOINT` | a full-state checkpoint | everything | continuing the same run |
| `--recover RUN_DIR` | the newest valid checkpoint in a run | everything | continuing after a crash or corruption |
| `--extend-budget GENERATION` | a finished run's final checkpoint | everything, with larger limits | training longer |
| `--promote CHECKPOINT` | any checkpoint's weights | weights only | a new stage on new data or settings |

### Resume

`--resume` restores weights, optimizer state, schedule, scaler, counters, data
cursor, checkpoint cadence, timing and random state. It refuses any change to
architecture, tokenizer, data, optimizer, schedule, budget, batch shape,
sequence length, determinism, engine or backend. A finished run cannot be
resumed.

- **`--allow-runtime-drift`** accepts a different Lab version or runtime on
  the same backend, recording the continuation as best effort.
- **`--allow-runtime-optimization`** accepts changes to how the same training
  runs: attention implementation, activation recomputation, block compilation,
  the compiled output head, determinism, `runtime.memory.estimate_policy`, and
  microbatch size and accumulation while keeping their product. When changing
  the microbatch size without fixed validation panels, set
  `evaluation.batch_size` to the old microbatch size so validation scores the
  same sample. The child records old and new values; results are no longer
  bitwise comparable.

### Recover

`--recover RUN_DIR` scans the run's finished generations, skips temporary or
corrupt ones, repairs `latest.json` and `best.json`, and continues from the
newest valid generation. An explicitly named `--resume` checkpoint that turns out
corrupt fails rather than falling back.

### Extend a finished run

`--extend-budget` takes the final full-state generation directory of a run that
reached both its step and token limits. Both limits must grow, and with AdamW
`optimizer.decay_steps` must equal the parent's original `training.max_steps`,
so the schedule continues at its floor rather than starting a new cosine. Only
the name, logging paths, the two limits and that decay horizon may differ. An
interrupted extension continues with ordinary `--resume` and the same config.
PyTorch only.

With `append_data`, the extension may also train on newly appended data; see
[datasets](datasets.md#bounded-windows-and-appended-data).

### Promote

`--promote` copies weights into a fresh run: new optimizer, schedule, random
state, cursor and counters, with the target config's data and budget. The
architecture and tokenizer must match. `training.parent_checkpoint` in a config
does the same thing declaratively, so [post-training](post-training.md) stages
chain and both arms of a `klab try` start from one parent.

To add n-gram memory to a dense model while promoting, see
[Engram memory](engram.md#add-memory-to-a-trained-dense-model).

## Import weights

`klab weights import` brings a local Hugging Face Llama safetensors directory
into a weights-only run you can promote from:

```sh
klab weights import SOURCE_DIR OUTPUT_RUN --config target.yaml \
  --format hf_llama_safetensors --source-tokenizer SOURCE_DIR/tokenizer.json \
  --provenance provenance.json
```

The target config must have the same dense or grouped-query layout and the same
tokenizer. `provenance.json` (`format_version: 1`) states the source and license.
`klab weights export` writes the reverse; see
[post-training](post-training.md#export-for-hugging-face).

## Best checkpoint and pruning

Validation at a new best loss writes a checkpoint (unless
`checkpoint.save_on_new_best: false`); `best.json` points at the best evaluated
generation in the run. A child keeps its ancestors' best as lineage metadata.

`checkpoint.keep_periodic: false` deletes older generations after each
successful save, keeping latest, best and the previous one.

### Keep milestones

`checkpoint.retention` keeps a declared set of checkpoints within a storage
budget:

```yaml
checkpoint:
  every_steps: 500
  keep_periodic: false
  retention:
    milestones:
      - {name: early, step: 2000, committed_targets: 262144000}
      - {name: final, step: 10000, committed_targets: 1310720000}
    terminal_step: 10000
    terminal_targets: 1310720000
    max_generations: 7
    max_generation_bytes: 5368709120
    max_retained_bytes: 37580963840
    max_peak_bytes: 42949672960
    reserve_free_bytes: 107374182400
```

A milestone is the first checkpoint whose step and committed targets both reach
its values. The kept set is the milestones plus latest, previous, best and
terminal. Before each save Lab checks free space against the remaining peak
budget and reserve; after it, it checks the actual size before publishing.
`checkpoints/retention.json` records what was kept and why, before anything is
deleted. Set `max_generation_bytes` from a real full-state checkpoint's size.
The policy is fixed when a run starts; a resumed child keeps its parent's
milestones and can add later ones.

## Checkpoints on interruptible machines

`klab train CONFIG --spot-policy POLICY.json` adds time-based checkpoints sized
so that a preemption loses at most a declared amount of work. The policy
(`format: klab-spot-safety-v1`) gives measured `checkpoint_write_seconds`,
`boundary_seconds`, `restart_seconds`, `interruption_notice_seconds`,
`max_recovery_seconds`, `checkpoint_bytes`, `checkpoint_inodes`, `reserve_bytes`
and `reserve_inodes`. The cadence is `max_recovery_seconds − restart − write −
boundary`; a non-positive result is refused. Free space for two checkpoints plus
the reserves is checked before training and before every save.
`klab checkpoint plan-spot POLICY.json --workspace RUNS` prints the decision
without starting a run.

## What a checkpoint records

- **Identity.** A run manifest records requested and effective config digests
  (machine-local paths excluded), an architecture digest, the Lab source
  identity and the runtime. Each checkpoint records its config digest and parent.
- **Exposure.** `klab report RUN_ID --kind run --json` lists, per checkpoint:
  usable supervised targets per pass over the prepared data, committed targets,
  effective passes, the loader cursor, and three timings (cumulative update time,
  training-loop time, and time in this invocation). Promotion starts a new
  exposure count; resume continues it.
- **Run-owned inputs.** A run copies its tokenizer, prepared data and any memory
  package into the run directory, so continuation does not depend on the
  original paths. Keep the whole run directory, not only the weights, if you may
  resume it.

## MLX checkpoints

MLX runs use the same generations, pointers and verification. Their training
state is `training_state.json` plus `optimizer.safetensors`. Checkpoints can be
inspected and verified without MLX installed. Continuing on the other engine is
a promotion, with fresh training state.
