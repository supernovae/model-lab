# Implementation brief: runtime forecasting and throughput calibration

Use this prompt for the next bounded implementation campaign. It supersedes the
earlier broad runtime-estimation prompt where the repository has already moved
forward.

## Current baseline to preserve

Treat this as follow-up work to the existing preprocessing progress, workdir,
staging, memory-planning, and research-lifecycle implementation.

- `src/klab/data/progress.py` already emits flushed JSON-Line
  start/heartbeat/finish records around preprocessing phases. Extend or replace
  that contract; do not build an unrelated second progress system.
- Tokenizer and packing paths already identify several coarse preprocessing
  phases. They do not yet report meaningful counters, rates, or ETA.
- `klab inspect` already reports architecture inventory, runtime identity,
  a calibrated memory estimate, and a separate resource proposal.
- `klab stage --through smoke|warmup` already runs disposable pilots without
  advancing the eventual run's optimizer, cursor, counters, schedule, or RNG.
- The `fast` memory policy can propose a larger divisible microbatch while
  preserving effective batch. It is a capacity-based proposal, not a measured
  throughput search.
- `--work-dir` / `KLAB_WORK_DIR` already redirects Model Lab-managed
  temporary work into the ignored project-local `klab-work/` by default.
- The research lifecycle already separates findings, next tests, evidence
  availability, and reviewed baseline decisions.

Do not interrupt, mutate, or reuse the directory of an active experiment. Use a
new named directory beneath `klab-work/` and confirm its free bytes and
inodes before launching anything expensive.

## Goal

Make long Model Lab operations predictable enough that a user can answer:

- Is this seconds, minutes, hours, or days before launch?
- What did a bounded warmup change about that estimate?
- Is the current phase advancing normally?
- What is the recent and long-window throughput?
- When might this phase and the whole job finish?
- Is the machine underfilled because of batch configuration, input work,
  synchronization, memory pressure, or something still unknown?
- Which microbatch/accumulation choice is the best measured proposal for this
  config and device without silently changing the experiment?

Runtime forecasts are planning aids, not scientific results or guarantees. Never
convert timing evidence into a model-quality claim.

## Required conceptual separation

Preserve these as distinct records:

1. **Planning estimate**: computed before execution from the requested config,
   architecture/resource accounting, cache state, runtime/device identity, and
   only compatible historical observations.
2. **Warmup-calibrated estimate**: computed from a bounded disposable pilot. It
   may supersede the planning estimate for display, but never overwrites it.
3. **Live ETA**: derived from actual progress using robust recent and longer
   windows.
4. **Final observed cost**: actual phase and end-to-end durations, device/update
   time, preparation, evaluation, checkpoint, and reporting cost.

Persist predictions and observations separately so forecast error can be
measured honestly after completion.

## First implementation slice

Implement a coherent vertical slice rather than every predictor at once.

### 1. Common progress record

Define one versioned machine-readable schema usable by preprocessing, training,
evaluation, checkpointing, and reporting. It should include, where known:

- operation/run identity and phase;
- event kind and monotonic observation timestamp;
- elapsed seconds;
- completed and total work with named units;
- completed optimizer targets and steps;
- recent-window and cumulative rate;
- last meaningful-progress time;
- state such as `RUNNING`, `RUNNING_SLOW`, `NO_PROGRESS`,
  `WAITING_EXTERNAL`, or `COMPLETING`;
- ETA low/high seconds, basis, and status;
- unavailable reason rather than fabricated zeroes.

Keep raw numeric values in JSON. Human formatting and completion timestamps are
derived views. Preserve current stdout payloads and emit progress JSON Lines on
stderr unless an existing durable event path is the better canonical sink.

### 2. Phase-aware preprocessing progress

Instrument meaningful counters for source acquisition/iteration, tokenizer
corpus selection, tokenizer training where the backend exposes progress, data
encoding/packing, serialization, verification, and hashing. Useful rates include
documents/sec, source bytes/sec, encoded tokens/sec, and verified bytes/sec.

Show valid cache reuse explicitly. A verified cached phase should be marked
reused/verification-only, not estimated as if it will rebuild. Do not infer that
an external download is local unless the code can verify the relevant cache.

A heartbeat without counter advancement must eventually suspend ETA and report
`NO_PROGRESS`; it must not automatically kill the process.

### 3. Training progress and live ETA

Use completed supervised targets as the primary training progress denominator,
with steps retained separately. Maintain robust recent and longer windows,
exclude initialization/transient steps from calibrated throughput, smooth the
display, and report instability when windows materially disagree.

An optimizer-only ETA must be named `optimizer_only_eta`. Estimate validation,
checkpoint, evaluation, generation, and reporting overhead separately only when
evidence exists. Unknown future phase cost makes overall ETA partial or
unavailable; it does not equal zero.

### 4. Bounded throughput calibration and batch recommendation

Add an explicit disposable calibration mode to staging; do not make training
silently tune itself.

Treat effective batch
`micro_batch_size * gradient_accumulation` as fixed unless the user explicitly
authorizes a scientific config change. Generate only divisible candidate pairs
that preserve that effective batch. Start from safe memory-fit information, then
measure a small bounded set that can reveal whether fewer, larger microbatches
improve target throughput.

For each candidate retain:

- complete effective config identity;
- backend/device/precision and runtime versions;
- initialization steps excluded and successful measured updates included;
- median and robust spread of update seconds and target tokens/sec;
- observed peak memory and available headroom where trustworthy;
- OOM, overflow, unsupported, timeout, or unstable outcome;
- host/input/device observations when available;
- reason selected or rejected.

Rank by measured stable target throughput subject to an explicit memory-headroom
rule. Do not reward one fast outlier. Write a separate proposal/config and an
evidence record; never mutate the requested source config. A changed proposal
requires its own warmup and remains distinct from the original experiment.

Do not use CPU or accelerator utilization as the optimization objective. Less
than 100% can be normal; classify bottlenecks conservatively as
accelerator-bound, host/input-bound, memory-pressure, synchronization/overhead,
or unknown only when telemetry supports the label.

### 5. Planning estimate

Extend existing inspection/resource accounting rather than duplicating it. Add a
read-only explicit command or option consistent with the CLI, such as
`klab inspect CONFIG --estimate-runtime`.

Use information genuinely available before execution: active/trainable/total
parameters, layer and attention shapes, sparse/MoE/MLA/Engram active work,
sequence length, microbatch and accumulation, target budget, optimizer,
precision, recomputation/offload, engine/backend, memory pressure, device
identity, cache state, and compatible historical measurements.

If an architectural work approximation is used, label it as estimated work and
document exclusions. Do not present theoretical FLOPs or `6 * params * tokens`
as a precise wall-time model. Mechanisms without a trustworthy model must mark
the estimate incomplete.

### 6. Historical compatibility

Initially use a deterministic, explainable matcher. Reject observations with
incompatible device/backend/engine, precision, architecture family, attention
family, optimizer, recomputation/offload, or materially different sequence/work
shape. Identify every contributing run and matching dimension. Do not average
unrelated runs to avoid saying that evidence is unavailable.

### 7. Status, persistence, and UI

Persist the initial forecast, calibration result, bounded live snapshots, phase
observations, and final cost. Extend an existing status/controller path where
possible. Add read-only dashboard presentation only after the JSON/data contract
works; the dashboard must not schedule, tune, or mutate runs.

Recommended human view:

```text
dense-lm-v1  TRAINING
7,864,320 / 20,000,000 targets (39.3%)
recent 68,400 targets/s; long 66,900 targets/s
elapsed 1h57m; ETA 3h00m–3h14m; stable
last meaningful progress 1.1s ago
```

All formatted values must have raw numeric counterparts in JSON.

## Workspace reliability in this slice

Before long calibration or execution, report the actual named workspace,
filesystem free bytes, free inodes where supported, and projected known growth
from checkpoints/prepared data. Do not rely on `/tmp` capacity. Fail before
partial publication if required output clearly cannot fit. If a projection is
unknown, say so; do not invent a safe value.

## Tests

Add deterministic focused tests for at least:

- zero observations produce preliminary or unavailable ETA;
- exact progress math and completed-run zero ETA;
- ETA never becomes negative;
- cached phases are marked reused and excluded from rebuild work;
- counter-stalled jobs suspend ETA while heartbeat-only jobs remain alive;
- recent/long windows and smoothing use controlled timestamps;
- initialization steps are excluded from calibration;
- candidate generation preserves effective batch;
- candidate ranking rejects OOM/unstable results and does not mutate input;
- validation/checkpoint overhead remains distinct;
- incompatible historical observations are refused with reasons;
- raw JSON uses null/unavailable rather than missing telemetry as zero;
- estimation and dashboard reads cannot mutate run state;
- prediction records cannot become scientific evidence automatically;
- workspace selection and capacity checks honor `KLAB_WORK_DIR`.

Avoid wall-clock-sensitive CI assertions and live hardware requirements in the
portable suite. Add separately marked backend tests for real devices.

## Documentation and success criterion

Document the progression:

`architecture planning → disposable warmup → live ETA → final observation → compatible historical calibration`

Include one honest example where the planning estimate is wrong and explain why.
Document batch calibration as a measured proposal that preserves the declared
effective batch, not an automatic hyperparameter search.

The slice succeeds when a long preparation or training job can say either:

```text
Pre-run: 3–5 hours, preliminary.
Warmup: 3h42m–4h05m from measured device throughput.
At 41%: recent ETA 2h11m; long-window ETA 2h18m; stable.
```

or, without pretending progress exists:

```text
NO_PROGRESS for 8m13s. ETA suspended.
Last advancement: 4,194,304 encoded tokens.
Process heartbeat continues; limiting cause unknown.
```

Do not use the current dense reference as calibration until its lifecycle state
and owner make that scientifically and operationally safe. A disposable copy in
a separate named workspace is preferred.
