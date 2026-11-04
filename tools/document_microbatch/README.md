# Matched document microbatch replay (GPU execution pending)

This disposable harness extends the previously executed step-420 replay, using
native collation, training engine, checkpoint verification/restoration and device
lease. It never saves a training checkpoint or restarts continuation. Preparation
is CPU-only. Execution needs an exclusive operator-scheduled B70 window and fresh
host compute-idle evidence; no GPU run has been made for these changes.

Primary arms all use checkpointing OFF, block compilation OFF and head compilation
OFF. `baseline` preserves legacy separate mb1; `padded_mb1` isolates new padding and
diagnostics overhead; `mb4` and optional `mb8` use independent padded rows. All
retain the identical 100 groups, 507,591 targets, step-420 checkpoint, optimizer
state, RNG, scheduler and LR sequence. Each cold and warm pass restores that exact
state. Bucket lengths round up to 128, capped at 4096; a partial final microbatch
has fewer rows. Padding costs are explicit input slots and dense attention matrix
slots (B*T*T), **not measured FLOPs**. Useful target throughput is the primary rate.

After committing the reviewed implementation, from repository root:

```bash
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/byron/src/tiny-sparse-lab/.venv/bin/python tools/document_microbatch/prepare.py \
  --parent-plan /home/byron/Documents/Codex/2026-11-04/task-6/performance-prep/plan.json \
  --runtime-profile /mnt/sparselab/operations/lab86-torch215-compare-20261103-r1/runtime-profile.json \
  --output /home/byron/Documents/Codex/2026-11-04/task-9/benchmark-prepared
```

This writes a new source-pinned plan and explicit unqualified admission receipts;
it does not edit the original plan or its evidence. It verifies the parent config
against the checkpoint, records source drift, and checks each delta through native
`runtime_optimization_transition`. Full cold checkpoint verification occurs again
at execution. A new tested source revision requires fresh preparation.

Only when the parent has reserved the GPU, with NEW output directories and fresh
idle evidence (same schema as the original replay, including observed UTC time,
physical device ID and zero reported CCS), execute ONE arm at a time:

```bash
PREP=/home/byron/Documents/Codex/2026-11-04/task-9/benchmark-prepared
IDLE=/absolute/path/to/fresh-host-idle-evidence.json
OUT=/mnt/sparselab/projects/devmind/experiments/document-microbatch-v1
bash tools/document_microbatch/execute-arm.sh baseline "$PREP" "$IDLE" "$OUT/baseline-r1" > baseline-r1.stdout.log 2> baseline-r1.stderr.log
# Review baseline completion before this command:
bash tools/document_microbatch/execute-arm.sh mb4 "$PREP" "$IDLE" "$OUT/mb4-r1" > mb4-r1.stdout.log 2> mb4-r1.stderr.log
python tools/document_microbatch/compare.py "$OUT/baseline-r1/result.json" "$OUT/mb4-r1/result.json"
# Optional only after measured headroom review; no automatic matrix launch:
bash tools/document_microbatch/execute-arm.sh padded_mb1 "$PREP" "$IDLE" "$OUT/padded-mb1-r1"
bash tools/document_microbatch/execute-arm.sh mb8 "$PREP" "$IDLE" "$OUT/mb8-r1"
```

Refresh idle evidence if older than 30 minutes. The native physical-device lease
and harness flock are held through the run, with lease heartbeats every update.
Keep the prepared private Intel loader/python environment in `execute-arm.sh`.
The harness checks loaded driver/compiler/loader paths, host/storage envelope,
nonfinite outcomes, and peak reserved memory against the configured device ceiling
(90% for this plan). OOM, ceiling breach or compilation failure terminates that arm;
there is no retry with changed settings. An OOM can occur before the post-update
ceiling check; it is retained as failed evidence, never inferred to fit from CPU
shape counts. The preflight receipt is not a capacity guarantee.

Report both cold and warm update and replay-wall throughput, shapes/padded slots,
peak allocated/reserved bytes and headroom, compiler counters and stderr. The
comparison refuses mixed runtime/attention/optimizer settings. GPU numerical
screening reports per-update loss/gradient norm/LR and parameter samples; full
GPU tensor equivalence and repeated/reversed-order runs remain required before
qualification. No throughput winner or production profile is certified here.
