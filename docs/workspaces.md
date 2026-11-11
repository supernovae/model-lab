# Experiment workspaces and retained evidence

Commands below use the [activated, provisioned environment](../README.md#first-run).

For command-by-command storage and interpreter choices, see
[location and runtime selection](location-runtime-controls.md), including why
historical lab/run readers and controller stores retain separate selectors.

One experiment has one local workspace. Seeds, architecture cells, budget
coordinates, and resumed children are scientific coordinates identified by run
IDs and immutable manifests; they are not peer repository-root directories.

Name every workspace for its task or experiment, independently of its execution
backend. CPU, ROCm, CUDA, and other backends are runtime parameters recorded in
configs and manifests. For example, runtime acceptance across devices shares
`$KLAB_WORK_DIR/experiments/runtime-acceptance/runs/`; runtime smoke configs
share `$KLAB_WORK_DIR/experiments/runtime-smoke/runs/`. Use distinct run IDs
for each execution, including backend labels when useful, rather than separate
`runs-rocm` or `runs-cpu` stores. Worker environments remain backend-specific.

The implicit persistent root is `${XDG_DATA_HOME}/klab` when
`XDG_DATA_HOME` is nonempty and absolute; otherwise it is
`~/.local/share/klab`. For large campaigns, explicitly choose an adequately
sized filesystem, for example `export KLAB_WORK_DIR=/data/klab`.
The global `--work-dir PATH` overrides that environment variable, which overrides
the implicit root. An explicit relative path remains relative to the current
directory, including legacy `KLAB_WORK_DIR=klab-work`. The root is not
derived from the repository, branch, or worktree. Merely resolving it does not
create it; read-only inspection does not need to initialize it.

```text
<persistent-root>/
  experiments/dense-lm-v1/
    receipt.json
    runs/              # shared controller/store, run IDs, parent and child runs
    staging/
    exercises/
    captures/
    local-reports/
  runs/                # implicit generic controller/store
  scratch/             # disposable temporary files; real directory, not a symlink
  cache/               # optional reconstructable caches, never scientific evidence
```

The `scratch/` child must be a real directory contained in the resolved work
root; scratch symlinks, including links within that root, are refused. Choose
a different `--work-dir` or `KLAB_WORK_DIR` if the selected root has a
symlinked scratch child. Symlink aliases for the work root itself remain supported.

The source tree holds source, tests, configs, documentation, and declarative
research/lifecycle records. The persistent root holds downloads, prepared arrays,
checkpoints, immutable execution evidence, receipts, and logs. Its `scratch/`
directory contains disposable temporary files; `cache/` is reserved for optional
reconstructable caches. The old ignored in-checkout `klab-work/` remains
selectable explicitly, but is not moved or pruned automatically. A selected
persistent root or explicit long-lived output/cache destination inside any Git
checkout receives a `STORAGE_INSIDE_GIT_CHECKOUT` warning: execution is permitted,
but losing the checkout may lose the bytes. Published compact evidence may be
versioned separately; a digest or commit identifies content, not its root path.
Promoting a model does not promote its full mutable workspace into version
control. Reusable tokenizer and prepared-data caches keep their configured
identities and locations; they need not be duplicated into each experiment.

## Guarding one operational command

`klab monitor` launches one command once, records versioned launch,
per-sample and completion JSON under a new exclusive log directory, and returns
a nonzero exit code on any guard violation, monitor error or command failure.
For example:

```sh
klab monitor --policy monitor.yaml --workspace /data/klab \
  --log-dir /data/klab/logs/unique-launch \
  --reserve-bytes 1073741824 --reserve-inodes 100 --json -- \
  klab corpus measure-tokens ...
```

For a 12 GiB owned-tree RSS cap, 8 GiB host available-RAM floor, 1 GiB
owned-tree swap cap and independently selected free-swap floor:

```yaml
monitor_policy_version: 1
max_tree_rss_bytes: 12884901888
min_host_available_ram_bytes: 8589934592
max_tree_swap_bytes: 1073741824
min_host_free_swap_bytes: 1073741824
min_projected_disk_free_bytes: 10000000000
min_projected_disk_free_inodes: 10000
```

The strict YAML policy starts with `monitor_policy_version: 1`; optional guards
are `max_tree_rss_bytes`, `max_tree_swap_bytes`,
`min_host_available_ram_bytes`, `min_host_free_swap_bytes`,
`min_disk_free_bytes`, `min_disk_free_inodes`,
`min_projected_disk_free_bytes` and `min_projected_disk_free_inodes`.
`interval_seconds` defaults to 1 (maximum 2) and
`termination_grace_seconds` to 5. Reservations subtract from the current free
space/inodes on the workspace filesystem before projected checks. The policy is
operational: it is **not** a scientific config, and unlike
`ResourceEnvelope.min_swap_bytes` the tree swap guard applies only to the
command's PID/creation-time identified descendants. Host swap *used* is logged
but never a tree-swap threshold; free host swap has its own minimum.

Python callers can load `MonitorPolicy` with `load_monitor_policy(path)` and call
`monitor_command(command, policy, workspace=..., log_dir=...,
policy_path=..., reserved_bytes=..., reserved_inodes=..., cwd=...)`; the result
is a typed `MonitorCompletion`. The command working directory defaults to the
caller's directory and is recorded separately from the monitored storage
workspace; `--cwd` selects it explicitly. The event stream (`events.jsonl`)
includes the command, root PID and creation time, source/policy paths and SHA-256,
and log paths. Child stdout/stderr are retained separately as
`command.stdout.log` and `command.stderr.log`; the CLI emits the final completion
JSON. `completion.json` repeats that terminal record. `FAILED` preserves a nonzero
child return code and is not an accepted successful operation. Peaks are sampled
observations; unavailable readings remain null, not invented zeroes.
Monitored descendants retain their identity after reparenting if observed before
orphaning. Signals
use per-PID Linux pidfds after identity checks, never process-group signals.
Sampling cannot discover a process that was born and orphaned entirely between
samples; plan critical workloads accordingly. Missing requested metrics fail
closed; monitoring does not resume, retry or authenticate scientific inputs.

For immutable bounded native CPU lab handoffs with job IDs, replay-safe admission and
retained finalization, use [lab jobs](lab-jobs.md). Its host configuration is
explicit; existing CPU state-v2 and `.lab-cpu.blocked.json`, and ROCm
`.lab-rocm.blocked.json`, must not be renamed to match the illustrative paths below.

## Shared Linux runner ownership

Opt into the same native monitor with all three paths below. Every contender
on a host must use the same paths on a local filesystem with reliable `flock`
and `fsync`. The actual CPU host/SSH alias is `cpu`; it runs CPU workloads.
The agreed lock paths and suggested local state locations are:

| Host | Canonical lock | Blocked marker | Receipt directory |
| --- | --- | --- | --- |
| `cpu` | `~/src/.lab-cpu-runner.lock` | `~/src/.lab-cpu-runner.blocked` | `~/src/.lab-cpu-runner-state` |
| `rocm` | `~/src/.lab-rocm.lock` | `~/src/.lab-rocm.blocked` | `~/src/.lab-rocm-state` |

Marker and receipt locations are configurable so an external runner can keep
its existing marker convention; agree on one triplet before adopting this tool.
Parent directories and the receipt directory must already exist, belong to the
caller, and exclude group and other write access. After establishing that the
host has no outstanding work, provision an empty receipt directory once (for
example, `mkdir -m 700 "$HOME/src/.lab-cpu-runner-state"`). A missing directory
may mean lost evidence and is refused, never automatically recreated. The tool
creates a missing lock; existing regular files must have one link, belong to the
caller, and exclude group/other writes. Symlinked lock/state entries are refused.
These are trusted local owner files, not a boundary against malicious workloads
under that UID. Never unlink, replace or truncate the lock: its inode is the
stable lock identity.

Inspect the **existing agreed triplet** without launching or provisioning anything:

```sh
klab runner-status --json \
  --runner-lock "$RUNNER_LOCK" --runner-state "$RUNNER_STATE" \
  --runner-blocked "$RUNNER_BLOCKED"
```

Here the variables denote the paths already configured for the job, not new
locations. This Linux-only read opens the existing lock read-only and uses a
nonblocking shared probe while verifying history. It never initializes a missing
lock/state/contract, truncates a lock, removes a marker or repairs receipts.
An initially present marker yields `BLOCKED`, including malformed or dangling
entries; a marker arriving during verification yields `UNRESOLVED` and remains
untouched. `BUSY`
means the lock is contended, with no claim about who owns it. Empty existing state
with an existing safe lock is `UNINITIALIZED`; a missing lock is `UNRESOLVED` and
is never created by inspection. `UNRESOLVED` reports the failing evidence path and exception;
missing or invalid history never becomes proof of a dead owner. `VERIFIED_HISTORY`
means existing native receipts checked at this observation only; it cannot
resolve a lab job with missing launch evidence. Every result has
`launch_authorized: false`; availability can change immediately afterward.
Inspection returns JSON with exit zero, so callers must inspect its `status`.
Ordinary exclusive launch admission remains mandatory. PID/heartbeat metadata
never permits reclaiming a lock. There is no repair, reconcile or unblock action.

The shared probe can make a concurrent exclusive launch admission refuse.
For `lab-job`, that refusal is sealed as `REFUSED_BEFORE_LAUNCH` and consumes the
job/attempt IDs even though no workload starts; redelivery returns the refusal.
Do not poll status while launches may begin. Coordinate inspection before
scheduling admission. A new attempt needs explicit new authorization; inspection
does not grant an automatic retry.

For a CPU command already authorized by the owner (use a fresh log directory):

```sh
klab monitor --policy /data/klab/monitor.yaml \
  --workspace /data/klab --log-dir /data/klab/logs/unique-launch \
  --runner-lock "$HOME/src/.lab-cpu-runner.lock" \
  --runner-state "$HOME/src/.lab-cpu-runner-state" \
  --runner-blocked "$HOME/src/.lab-cpu-runner.blocked" -- \
  python -c 'print("tiny CPU command")'
```

Invoke the workload directly. Do not wrap a legacy launcher that acquires the
same lock again. Contention refuses immediately; there is no queue, retry,
takeover, unblock command or lock-keeper process. Omitting all three flags keeps
the existing monitor CLI behavior. Supplying a subset is an error.

The minimal local contract is `contract.json` (format `klab-runner-v1`,
canonical lock path, device/inode and blocked path) plus one UUID directory per
attempt. Its exclusive, fsynced `pending.json` records the command, policy,
owner PID and canonical log directory **before** the native monitor spawns. Its
exclusive `completion.json` binds that pending record's SHA-256 and run ID to
the monitor and owned-completion receipt hashes and supervisor PID/creation
time. Successful, nonzero-exit and guard-stopped commands can all certify clean
shutdown; only success certifies workload success. Prior receipts and logs must
remain available unchanged. Each launch re-verifies retained history, so its
preflight cost grows with the number of attempts. Log-path parents must already
exist and are canonicalized before recording pending state; the same path goes
to the monitor. The final log directory is created exclusively, and final
symlinks are refused. Missing, partial, foreign or changed records refuse launch,
even if the recorded owner PID no longer exists.

The wrapper checks the blocked marker before and after acquiring its single
non-inherited flock. Any marker, including a dangling symlink, is authoritative.
It always uses the existing Linux subreaper, tree RSS sampling and storage
guards. Detached descendants retain subreaper attribution. Shutdown sends TERM,
then KILL after the policy grace, with at most two additional seconds of helper
cleanup. The monitor also bounds its cleanup wait and refuses unverifiable
owned receipts. These are userspace deadlines; kernel/filesystem stalls cannot
be given a hard realtime guarantee. Workload time itself remains governed by
optional `max_wall_seconds`; monitoring introduces no new workload budget.

Preflight refusals before an attempt directory exists leave no new host marker.
Once an attempt exists on disk, interruption, uncertain cleanup or completion
write failure leaves a durable blocked marker before lock release. A storage
failure can prevent even that write: the retained incomplete attempt still
forbids a later launch. SIGKILL or
power loss cannot write a new marker; pending state remains authoritative and
is never automatically recovered. A human must establish that owned work is
gone, preserve the failed evidence and reconcile the host before provisioning a
new state contract. Removing the marker alone does not repair an incomplete
attempt. The tool never deletes state, trusts a stale PID as clearance, acquires
credentials or contacts Notion. Updating a remote task/blocker is best effort
work for the external runner after enforcing local launch safety.

## Explicit campaign cleanup

Choose the campaign workspace under the selected persistent root when preparing
campaign-local data caches. New prepared-data caches inside that workspace receive
an ownership marker; existing caches without a marker and caches outside it are
never cleanup candidates. Shared data and worker dispatch caches are outside this
workflow. Keep `KLAB_WORK_DIR` pointing at the persistent root, not a nested
experiment: explicit cache locations remain separate.

```sh
WORK="${KLAB_WORK_DIR:-$HOME/.local/share/klab}/experiments/dense-lm-v1"
klab workspace cleanup plan "$WORK" \
  --output "$WORK/local-reports/cleanup-plan.json"
# Review the JSON paths, identities, and reclaimable bytes before applying.
klab workspace cleanup apply \
  "$WORK/local-reports/cleanup-plan.json"
```

The plan is read-only and defaults to retaining two additional eligible periodic
checkpoint generations per run and two campaign-owned prepared-data cache entries.
Change those limits with `--max-extra-periodic` and `--max-cache-entries` when
creating the plan. Latest, best, and the previous verified generation are always
protected. Every generation registered in the run database or bound by a child
run manifest is also protected,
even when this exceeds the count limit; most ordinary training checkpoints are
registered, so checkpoint proposals can be empty. Only verified, unregistered
generations in terminal runs can be proposed. Cache candidates require the
workspace ownership marker, a matching prepared-data manifest, and no active
campaign run or worker attempt. Applying a plan rechecks the database, candidate
inventory, and writer leases; changed plans must be created again. The command
never scans arbitrary sibling workspaces or deletes unmarked directories.

Use one variable for a study, including worker registration, submission,
controller operation, collection, and resume:

```sh
WORK="${KLAB_WORK_DIR:-$HOME/.local/share/klab}/experiments/dense-lm-v1"
mkdir -p "$WORK"
klab study submit configs/references/dense-lm-v1/study.yaml \
  --store "$WORK/runs" --receipt "$WORK/receipt.json"
klab controller run --store "$WORK/runs"
# After completion and verified ingestion:
klab study collect configs/references/dense-lm-v1/study.yaml \
  "$WORK/receipt.json" --runs-dir "$WORK/runs"
```

Register eligible workers in that same store before execution. One receipt
contains all submitted seed coordinates. Device leases serialize work where
required; separate seed stores are not required. Resume a controller run using
`experiment resume RUN_ID --store "$WORK/runs"`. The parent and child remain in
that store, linked by IDs and checkpoint identities. Keep the original submission
receipt as history; do not replace its submitted run IDs with resumed child IDs.

Choose `export KLAB_WORK_DIR=/data/klab` on a disk sized for the
campaign, then set `WORK="$KLAB_WORK_DIR/experiments/dense-lm-v1"`.
Explicit `--store`, `--runs-dir`, `--output`, transcript destinations, old
`logging.root_dir`, and `dataset.cache_dir` retain their supplied meaning.
The global `--work-dir /data/klab` (before the subcommand) overrides the
environment for implicit stores and scratch but never relocates explicit paths.
Direct training should pass `--runs-dir "$WORK/runs"` on every parent and child
command when using a task-specific store.

A study without store or receipt overrides chooses
`<work-dir>/experiments/<study-name>/runs` and a sibling `receipt.json`. The generic
non-study CLI store/reader default is `<work-dir>/runs/`. Existing root-level
runs remain readable with explicit `--runs-dir runs`; defaults do not search
historical locations or silently reinterpret configured paths.

Future experiments follow the same pattern:

```text
<persistent-root>/experiments/
  new-study/
  independent-replication/
```

## Historical paths and relocation

The original dense-lm-v1 seed YAMLs explicitly selected three root-level
`runs-dense-lm-v1-seed*` directories. The reference matrix repeated those paths,
and the manual instructions invoked direct training without overriding them.
This was a workflow convention, not controller isolation. New instructions use
one explicit shared store; frozen source configs and published evidence retain
their original values.

Intentional relocation to a genuinely different persistent root requires a new
location binding or manifest and re-verification of the referenced bytes. Do not
rewrite past receipts to imply that historical absolute paths moved.

Before moving retained bytes, distinguish scientific identity (hashes, run IDs,
checkpoint and input identities), historical location (where execution or capture
occurred), and current local location (where readers should find retained bytes).
Preserve the first two. Update only supported current-location indexes or explicit
reader paths, and verify checkpoint integrity, lineage, evidence, observations,
static reports, and baseline descriptions afterward. Never edit old observations
or acceptance records to claim they were originally captured at a new path.
Do not move active runs or blindly merge controller databases. Prefer same-device
atomic renames when relocation is supported. If the evidence cannot be loaded
without rewriting scientific artifacts, retain the original locations and record
the limitation.

Some older instructional documents are themselves registered, byte-hashed
research evidence: [domain corpus study](path-domain-corpus.md) and
[portable adapter notes](portable-engram.md). Their original commands remain
historical records. Apply the workspace convention above to new execution rather
than copying their old root-level output paths. Updating those bytes would
invalidate the evidence; no registry hash is changed by this hygiene work.

The original context-study guide bytes are preserved in
[Workshop at `b74dff53259343aafd336cd00e3480246d70133b`](https://github.com/supernovae/kybern-workshop/blob/b74dff53259343aafd336cd00e3480246d70133b/provenance/lab/docs/context-engram-study.md).
The [Lab context-study page](context-engram-study.md) is navigation only; its
original byte-hashed guide is mapped by [the retirement record](context-retirement.json).

The [dense-lm-v1 migration audit](dense-lm-v1-workspace-audit.md) records the actual
consolidation, preserved history, verification results, and remaining unrelated
lifecycle availability limits.

On 2026-09-27, the remaining local `runs-rocm/` store was relocated to
`klab-work/experiments/runtime-acceptance/runs/`. Its sole completed run,
`rocm-wsl2-acceptance-20260926T035145Z`, retains its original ID, backend metadata,
and historical paths. All 34 run files retained their hashes, inodes, sizes, and
modification times; all five checkpoint generations verified before and after
the move, and the integrity-bound evaluation evidence was identical. The current
checkpoint locator was rebased using the same standalone-store helper. The
original database and sidecars, inventory, relocation script, and verification
record are retained under the workspace's `local-reports/relocation/`. No new
training or hardware acceptance was performed. Earlier accounts of leaving
`runs-rocm/` untouched describe those earlier operations and remain unchanged.
