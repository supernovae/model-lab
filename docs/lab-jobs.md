# Bounded shared-host lab jobs

`lab-job` binds an immutable external handoff to the existing Linux
`runner_control` and owned operational monitor. The useful native path is
**tokenizer train → try → report**: `try` owns data preparation, candidate
derivation, initialized baseline/candidate training, held-out evaluation, probes,
sealed records and cancellation. The adapter verifies their dependent outputs;
it does not implement another trainer, DAG, queue or scheduler. Ordinary lab work
needs no Campaign or locked ExperimentPlan. Those remain available for release
work and larger queued workflows. There is no shell-command field, dependency
sync, automatic execution retry or deletion operation. GPU admission is outside
this slice and remains parent-gated after CPU verification.

## Select source, host and native declarations

The coordinator selects a **merged, validated full SHA** containing this adapter
before admission. Use a separate clean detached controller worktree at that SHA;
do not switch the owner's checkout. If necessary, the parent/operator can create
it from locally available Git objects:

```sh
git -C /owner/checkout -c core.hooksPath=/dev/null worktree add --detach /absolute/selected/controller FULL_VALIDATED_SHA
```

`validation_reference` records review/CI evidence. Admission verifies local source
and ancestry in `refs/remotes/origin/main`; it does not fetch or independently
attest CI. Source is never refreshed during a job. Each admitted job gets a fresh
owned detached workload worktree. No checkout hooks or dependency sync run.

All host roots must already exist, be canonical absolute paths, owned by the
operator and not group/other writable. Source, state, evidence and disposable work
roots must be disjoint. Evidence and work must share a filesystem for the native
disk monitor. Use the actual existing CPU **state-v2** directory,
`.lab-cpu.blocked.json` and agreed canonical host lock. ROCm's existing marker is
`.lab-rocm.blocked.json`; no state/marker is migrated or renamed. Do not wrap this
command in another `flock` or launcher taking the same lock.

The operational monitor creates new output directories, including missing ancestors,
with private mode `0700`, and output files with private mode `0600`. Newly created
directories therefore do not allow group traversal, even under a permissive caller
umask. Existing directories and evidence are not permission-repaired; unsafe
existing evidence remains subject to the collector's refusal.

The provisioned interpreter must already contain the locked dependencies.
`seal` records exact Python/Torch versions and a fingerprint of interpreter bytes
plus installed distribution names/versions. The monitored runtime gate rechecks
that fingerprint and imports from the selected workload source. This is not a
hash of every installed package file. Accelerator visibility is disabled, thread
counts are one and Hugging Face/Transformers offline flags are set. These are
execution controls, not an OS network sandbox or protection against hostile code
under the same UID. No installation or persistent environment mutation occurs.

[The checked-in tiny example](../examples/lab-job-try/README.md) uses only generated
local JSONL: a 260-token tokenizer, one-layer 16-wide CPU fp32 models, an explicit
FFN width delta 32 → 48, paired seed 7, **3 updates / 96 targets per arm** and
64 held-out targets per arm. A higher, lower or tied candidate loss is a valid
result; this wiring fixture makes no quality claim. Distinct file hashes do not
prove document-level split disjointness; inspect selected data before using other
inputs. This example uses distinct generated training and validation sentences. The manifest names native
YAML declarations and hashes their complete input closure. Relative producer and
consumer paths resolve inside retained evidence after the exact bytes are copied.

The `tiny-cpu-lab` profile bounds dense models to width 32, two layers, FFN 128,
vocabulary 512; training to 12 updates / 1,536 targets per arm; evaluation to four
batches per pass (the sum of both fixed panels' rounded-up batch counts, when
declared); local input closure
to 1 MiB. It requires fresh arms, retained checkpoints, CPU fp32 and explicit
native per-arm wall bounds (at most 120 seconds). The example selects 30 seconds
per arm and 90 seconds for the entire monitored job. The lab-loop CI's separate
900-second ceiling is **not** a host allocation. Every host pilot still needs the
owner-approved outer RSS/time/disk envelope in its manifest.

## Seal one immutable handoff

Write this draft JSON outside the checkout. Replace every placeholder. Values
below are a bounded recipe for parent approval, not proof of host capacity. The
64 MiB evidence cap allows native tokenizer, prepared data, checkpoints and
observations; it replaces the earlier test-only 16 MiB example.

```json
{
  "format": "sparselab-lab-job-v1",
  "job_id": "cpu-native-001",
  "attempt_id": "cpu-native-authorized-001",
  "source_sha": "<exact merged validated 40-character SHA>",
  "validation_reference": "<review and exact-head CI URLs>",
  "backend": "cpu",
  "host": {
    "repository": "<absolute selected controller>",
    "lock": "<absolute existing canonical CPU lock>",
    "state": "<absolute existing CPU state-v2 directory>",
    "blocked": "<absolute parent>/.lab-cpu.blocked.json",
    "evidence": "<absolute precreated durable evidence root>",
    "work": "<absolute precreated disposable work root>",
    "python": "<absolute provisioned interpreter>"
  },
  "workload": {
    "kind": "lab_try",
    "profile": "tiny-cpu-lab",
    "root": "<absolute selected controller>/examples/lab-job-try",
    "tokenizer": "tokenizer.yaml",
    "baseline": "baseline.yaml",
    "candidate": "delta.yaml",
    "declaration": "try.yaml",
    "files": ["tokenizer.yaml", "baseline.yaml", "delta.yaml", "try.yaml", "bounds.yaml", "train.jsonl", "validation.jsonl"]
  },
  "policy": {
    "monitor_policy_version": 1,
    "max_wall_seconds": 90,
    "max_tree_rss_bytes": 2147483648,
    "termination_grace_seconds": 5,
    "min_disk_free_bytes": 1073741824,
    "min_disk_free_inodes": 1000
  },
  "max_evidence_bytes": 67108864
}
```

Bootstrap from the selected controller using command-scoped `PYTHONPATH` even if
an editable install points elsewhere. `-P` excludes implicit current-directory
imports; `-B` suppresses bytecode writes. Do not add `-I` here (it ignores
`PYTHONPATH`); internal children use isolated mode with explicit source selection.

```sh
env PYTHONPATH=/absolute/selected/controller/src \
  /absolute/provisioned/bin/python -P -B -m sparselab lab-job seal /absolute/draft.json --output /absolute/job.json
```

Seal validates native declarations, fingerprints the current selected interpreter,
replaces `files` with filename-to-SHA-256 identities, and exclusively publishes
`job.json`. It prints job ID, attempt ID, manifest hash and `SEALED_NOT_ADMITTED`;
it runs no model and reserves no attempt. Review this sealed artifact before
admitting it. `runtime-identity` also prints the read-only runtime fingerprint.
The external identifiers are owner-authorized labels, not credentials. Keep the
same evidence root as the uniqueness namespace. Changed content or a reattempt
needs a newly authorized identity. Duplicate keys, unknown fields and nonfinite
values refuse; manifest identity ignores JSON whitespace.

## Admit, inspect and collect

```sh
env PYTHONPATH=/absolute/selected/controller/src \
  /absolute/provisioned/bin/python -P -B -m sparselab lab-job admit /absolute/job.json
env PYTHONPATH=/absolute/selected/controller/src \
  /absolute/provisioned/bin/python -P -B -m sparselab lab-job status /absolute/job.json
env PYTHONPATH=/absolute/selected/controller/src \
  /absolute/provisioned/bin/python -P -B -m sparselab lab-job collect /absolute/job.json
```

Admission exclusively reserves the job and unique attempt index, seals the
manifest and `prepared.json` binding external IDs/digest to a native runner UUID,
and retains the exact input closure **before launch**. The same binding enters
native `pending.json` before monitor spawn. Native `try` then writes its own
sealed started record with `request.job_binding` before optimizer execution; its
actual try/run IDs, checkpoints and observation identities are retained. Queue
workers keep their native PREPARED/claim/replay protocol unchanged.

Duplicate delivery never reaches launch. It returns `UNKNOWN_OR_RUNNING`,
`COLLECTABLE`, a sealed `REFUSED_BEFORE_LAUNCH`, or a reverified `FINALIZED`
result. Partial/conflicting handoffs refuse. Even a crash before native pending
exists does not authorize execution retry. Uncertain launches stay blocked for inspection; PID absence proves
nothing. The existing native history, lock inode, state, blocked markers and
receipt-referenced logs remain authoritative. Collection currently checks all
native runner history, so unrelated incomplete history can block it too.

`REFUSED_BEFORE_LAUNCH` requires a typed native runner preflight refusal and an
immutable `refusal.json` bound to the prepared job. It proves that attempt-directory
creation was not entered and the monitor/workload was not launched; it does not
clear locks, repair history, approve a retry or release the consumed job/attempt
IDs. Status and collect verify and return that receipt. A crash before receipt
publication, a signal, or any error once attempt-directory creation is entered
stays uncertain, even if `pending.json` is absent. A new attempt needs new explicit
authorization. Initial missing state still refuses before admission publication.
For `admit`, `status` and `collect`, a successfully returned refusal JSON currently
has exit code zero. Callers must inspect the JSON `status`; exit zero alone does
not establish that a workload launched or completed.

Fixed source/runtime/inputs/tokenizer/try/report gates record actual exits and
executed/skipped status. The final verifier checks native completion and owned
cleanup, manifest/source/size/time, sealed try/report equality, declaration and
data/tokenizer bindings, executed non-reused arms, checkpoint hashes/counters,
finite held-out output and its checkpoint identity. It does no training or
scoring. Missing or semantically inconsistent evidence refuses finalization and
remains available as `COLLECTABLE` for inspection; no replacement failure summary
or launch is invented. Failed/skipped gates cannot pass.
All native artifacts live in durable evidence, separately from disposable scratch.
`summary.json` indexes retained bytes/hashes and includes native IDs, comparison,
measured RSS/time and checks. `controller_completion` remains distinct from
`coordinator_acceptance: PENDING`, even on operational `PASS`.

On final-publication failure retain everything and retry **collect**. Collection
has no launch path. For a running try, its native printed `CANCEL` file is the
normal cancellation mechanism; the watchdog remains the outer limit. Interrupted
records and failed gate evidence remain, with no automatic retry.

## CPU acceptance and retained cleanup preview

After external review, merge and applicable CI for the selected merged source:

1. Parent verifies actual lock/state-v2/marker paths, retained prior receipts,
   filesystem/free bytes/inodes and provisioned runtime. Seal and approve the
   explicit envelope and generated local input hashes. No host qualification was
   performed by this authoring PR.
2. Admit once. Inspect all six gates, native `COMPLETE`, cleanup verification,
   summary checks and retained logs. Confirm two fresh trained runs, declared
   update/token counters, finite held-out losses, native report and comparison.
   Accept lower/higher/tied loss; acceptance concerns correct execution.
3. Redeliver and collect; confirm identical manifest hash, runner/try/run IDs,
   unchanged summary and one native pending/completion pair. Coordinator records
   acceptance separately against job ID/hash. Failed/uncertain attempts require
   inspection and new authorization, never a replay launch.
4. GPU rollout requires separate parent approval after CPU verification. This
   profile neither executes nor qualifies ROCm or other accelerators.

The public workload is the typed native `lab_try` profile; pytest and arbitrary
command payloads are rejected. The original pytest pilot surface was removed
because it had no product use beyond the pilot scaffold. Global work-directory
and token-file options are unsupported: this path is offline and manifest-bound.

Product regressions use temporary repositories and internally substituted native
operations to test admission, replay, actual gate exits/skips, evidence retention
and cleanup refusal without model work. They do not qualify a real CPU host.
These subprocess cases remain explicit in serial safe CI; measured slow cases
are excluded from parallel fast CI. Their 90-second synthetic fixture allowance
accounts for cold imports on CI; it does not set host job budgets. Separate
lab-loop integration uses the real
native tokenizer/model/report path and checked-in inputs, including cancellation
and verifier refusal mutations against one completed result.

Cleanup is **preview only**:

```sh
env PYTHONPATH=/absolute/selected/controller/src \
  /absolute/provisioned/bin/python -P -B -m sparselab lab-job cleanup-preview /absolute/job.json \
  --candidate /absolute/disposable-work/scratch/cpu-native-authorized-001
```

Only exact owned scratch with verified completion or a verified prelaunch refusal
is eligible for this preview. For a refused job, `job_status` is
`REFUSED_BEFORE_LAUNCH`; its detached source worktree is still retained.
Linked/foreign content, ancestors and evidence/state/lock/marker targets refuse. Output says
`delete_implemented: false`. Nothing is removed, including Git worktrees. Retain
all history, state, locks, markers and referenced logs: future admission reads
them. Scratch/source retention is also required for final source rechecking.
