# Offline 100M-target readiness assessment

**2026-10-10 host update:** The earlier assessment below describes a different
workspace that lacked `/srv/klab`. The present host has retained artifacts;
their fresh, bounded cold-verification results and proposed retained-only
admission/yield allocation are recorded in [the updated proposal](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/kernel-memory-lab/CARD05_BASE_100M_READINESS_PROPOSAL.md).
That verification does not admit or measure a new v3 release. The older text is
kept as a historical forecast, not a current host inventory.

Assessment date: 2026-10-10. Code baseline: merged PR #56,
`1c1fbc5` (reviewed parent `2b981988dc454820d296bf6c950683be3123c975`).
The owner selected **100,000,000 training targets** as the next research target.
This assessment accompanies the six conditional code changes; it is not a
production preparation packet, corpus admission, runtime allocation or training
result. Historical declarations and spent ledgers remain unchanged.

**Not ready to launch.** The accepted historical supply is insufficient in every
stratum at the previous mix/exposure limit. No admitted replacement release or
100M prepared bundle is recorded. This workspace lacks `/srv/klab`, including
the cited B7 preparation and B8 training roots (read-only path inspection).
Consequently, the numbers below are checked-in historical measurements, **not
fresh authentication of retained corpus, checkpoints, receipts or hardware**.
[Current KML status](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/kernel-memory-lab/STATUS.md)
records B8 as the latest completed scientific run, B16 as the latest initialized
preparation attempt and B17 as a pre-ledger stop. The older foundation audit's
all-not-started language is not current execution status.

## Supply and scientific assumptions

The [current mixture template](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/kernel-memory-lab/card05-base-50m/current/mixture.template.yaml)
still declares **50M** targets, 81/18/1 general/explanatory/incident quotas and
at most two exposures per eligible position. The
[prospective plan](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/kernel-memory-lab/CARD05_BASE_50M_PROPOSAL.md)
retains the 341,885,952-parameter dense model, unchanged 32,768-entry tokenizer,
context 1,024 and fresh seed 17. These are the basis of this conditional 100M
calculation, not a newly approved fixed mix or permission to change templates.

| Stratum | B7 measured unique train positions | Required 100M targets | Unique floor at two exposures | Unique shortfall against B7 |
| --- | ---: | ---: | ---: | ---: |
| General | 20,802,818 | 81,000,000 | 40,500,000 | 19,697,182 |
| Explanatory | 3,155,976 | 18,000,000 | 9,000,000 | 5,844,024 |
| Incident | 343,703 | 1,000,000 | 500,000 | 156,297 |
| Total | 24,302,497 | 100,000,000 | 50,000,000 | 25,697,503 |

Source: [B7 preparation report](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/kernel-memory-lab/results/2026-10-08-card05-base-preparation-v4-readiness.md).
These are distinct retained train content positions, excluding EOS, measured
with tokenizer SHA-256
`308b33a6edbed613f3caf5232b124a7c6105a7a3dc273c999725be51c3ffeaa9`.
At two exposures B7 provides **48,604,994 content-position exposures excluding
EOS**, even with a changed mix; this is not an exact supervised target capacity.
At 81/18/1, the explanatory stratum limits content-position exposure capacity
to about **35.07M**. Native mixture accounting establishes exact scheduled targets.
Extra repetitions or relabeling domains would change the scientific protocol.

The old 50M proposal's one extra Wikimedia bucket forecast (~1.55M unique
positions) cannot establish 100M supply, and adds no general or incident supply.
[B16](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/kernel-memory-lab/results/2026-10-10-card05-50m-preparation-attempt1-receipt-collision.md)
reports authenticating four retained snapshots, but never reached admission or
measured replacement-release supply. Do not infer that those retained snapshots
have B7's counts or that they need reacquisition. First inventory and authenticate
what the owner's host actually retains; then measure the exact admitted,
post-cleaning/deduplicated train release with protected family assignments.
Any missing supply needs a reviewed source-selection or protocol decision, with
separate preparation/acquisition authority if necessary.

## Updates, scheduler and evaluation

At microbatch 1, gradient accumulation 1 and context 1,024, exactly 100M targets
require **97,656 full updates plus one 256-target update: 97,657 updates**, with
768 final padding positions masked. The
[current run template](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/kernel-memory-lab/card05-base-50m/current/prepared-run.template.yaml)
has `max_steps: 48829`, `max_tokens: 50000000` and `decay_steps: 48829`.
Increasing only the token cap is incorrect: the explicit decay horizon in the
[native scheduler](../../src/klab/training/optimizer.py) overrides max steps
and clamps at its floor. A future reviewed fresh run should bind the new target,
update and decay horizon together. Keeping 500-step warmup and peak/floor
`3e-4`/`3e-5` is a hypothesis to review; do not scale warmup automatically or
resume B8's completed 25M scheduler under a new identity.

The unchanged base evaluation profile remains compatible only while its original
C03-C1 held-out release, document windows, tokenizer and protected families stay
bound. Its SHA-256 is
`bb4c1b719b29fa96c4429516022a5038f0e3595b9d8ef63394bf0a17a5c6851d`.
Keep earliest-minimum finite fixed-validation checkpoint selection, the 12 fixed
test windows, 24 true/decoy pairs and eight 64-token greedy prose prompts per
selected model. Preserve the historical v2 comparison and read-only B8 outcomes;
a new training release must not replace the profile's origin release.

Checkpoint cadence needs an explicit choice. Keeping every 5,000 updates yields
**21 checkpoints** (initial, 19 periodic, terminal), rather than 11. That adds
30,720 fixed-validation and 10,240 operational-validation input positions to
B8's accounting: approximately **83,112 fixed-profile** and **202,408 aggregate
charged nontraining inputs**, if every other comparison and reservation remains
unchanged. The old 55,000/170,000 caps would fail. Every 10,000 updates gives 11
checkpoints and can preserve that evaluation count, but changes selection
resolution and recovery exposure; it is a review option, not an enacted cadence.
Any spot-safety cadence must also be reconciled with explicitly selected
scientific evaluation checkpoints and measured retention capacity.

The original base-learning thresholds remain unchanged (at least 10% lower
fixed test loss than v2, at least 18/24 utility preferences and four more than
v2, at least 6/8 sustained prose continuations). B8 achieved 15/24 and 0/8;
more targets are not evidence those gates will pass. The 200-item language and
400-item evidence suites are separate work; C05-N1 remains 0/200 and Card 06
remains blocked.

## Resource forecast and missing allocation

[B8's measured report](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/kernel-memory-lab/results/2026-10-08-card05-base-training-v4.md)
records 25M targets in a 6,348-second training monitor (~3,938 targets/second
including its operational overhead), about 7,960 seconds end to end, 11 retained
checkpoints, 10,953,015,296 bytes sampled device VRAM and 4,019,204,096 bytes
sampled process-tree RSS. Assuming the same hardware, stack, architecture,
batch/context and similar throughput, `4 × 6348 + (7960 - 6348)` gives
**27,004 seconds (~7.50 hours)** for a comparable 11-checkpoint 100M runtime
path, before newly required corpus preparation. It scales aggregate monitor
overhead too, so it is a rough extrapolation, not a pure update benchmark or a
confidence bound. Additional checkpoints, I/O stalls and changed input throughput
need separate allowance. B7 preparation historically took about 2,676 seconds;
its smaller input cannot bound new 100M preparation.

B8's cumulative added apparent storage was **46,863,235,311 bytes (~43.65 GiB)**,
including retained preparation. Runtime state does not grow proportionally to
target count at fixed shape, but mixtures, logs, preparation and checkpoints do.
Keeping 21 checkpoints could exceed the old 72 GiB limit: its rough B8
post-preparation growth per checkpoint is ~3.89 GB, implying another ~38.9 GB
for ten extra checkpoints before increased corpus storage. This is an average
inference, not a measured checkpoint size. Recompute from authenticated actual
files, retention requirements, temporary staging and a fresh common-root
byte/inode/free-space baseline. Do not double memory, disk or evaluation caps.

Before any run, review one exact new allocation tied to the final code revision,
new immutable attempt identity, admitted release/family/mixture/bundle hashes,
tokenizer origin and bytes, config, scheduler, evaluation profile and checkpoint
cadence. It must enumerate preparation and runtime wall deadlines, **100M targets
and at most 97,657 updates**, all nontraining inputs/generation/reviewer limits,
VRAM/RSS, total and staging bytes/inodes, free-space margin, shutdown monitoring,
restart/retry policy and zero-cloud or separately approved spending. Verify the
owner's retained corpus and actual hardware/stack first; unavailable sensors or
identity/cap failures must stop. Earlier 50M authority does not cover this target.
This code task performs no acquisition, fitting, production preparation, model
initialization, forward, update, generation or device job.
