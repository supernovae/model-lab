# Workspaces and storage

The repository holds source, tests, configs and documentation. Model
definitions, named datasets and studies live in
[Kybern Workshop](https://github.com/supernovae/kybern-workshop) or your own
repository. Everything a run produces (downloads, prepared data, checkpoints,
logs, receipts) lives in a work directory outside Git.

## Choose the work directory

The work directory is the first of:

1. the global `--work-dir PATH`, given before the command;
2. `KLAB_WORK_DIR`;
3. `$XDG_DATA_HOME/klab`, when `XDG_DATA_HOME` is absolute and nonempty;
4. `~/.local/share/klab`.

For large work, put it on a filesystem with room to spare:
`export KLAB_WORK_DIR=/data/klab`. A relative path stays relative to the
current directory. The work directory never depends on the checkout, branch or
worktree, and resolving it does not create it. A work directory or long-lived
output inside a Git checkout works but warns `STORAGE_INSIDE_GIT_CHECKOUT`.

```text
WORK_DIR/
  lab/                  # lab-mode tries, probes, explorer cache and their runs
  runs/                 # default run store and controller store
  experiments/NAME/     # one workspace per experiment
    runs/               # its controller store: seeds, cells, parents and children
    staging/
    local-reports/
  scratch/              # disposable temporary files; must be a real directory
  cache/                # optional caches that can be rebuilt
```

`scratch/` must be a real directory inside the work directory, not a symlink.
The work directory itself may be reached through a symlink.

Name a workspace for its task, not its backend. Backends are runtime settings
recorded in configs and manifests: a runtime acceptance across CPU, ROCm and
CUDA shares `experiments/runtime-acceptance/runs/` with distinct run IDs, not
separate `runs-rocm` and `runs-cpu` stores. Seeds, architecture cells and
resumed children are run IDs within one store.

## Which option selects what

| Command | Storage | Runtime |
| --- | --- | --- |
| `try` | Work directory, or `lab_dir` in the declaration | `--runtime ID` or `runtime_profile`; `--backend` overrides the workload |
| `report`, `compare` | Work directory, a record path or `--lab-dir`; `report --kind run` uses `--runs-dir` | None: read only |
| `probe` | Work directory, or `lab_dir` and `runs_dir` in the declaration | `--runtime ID` or `runtime_profile`; the saved backend unless `--backend` |
| `explore` | Work directory, or `lab_dir` and `runs_dir` in the declaration | `--runtime ID` or `--runtime-profile FILE`; CPU by default |
| `eval`, `generate`, `chat`, `serve` | `WORK_DIR/runs`, or `--runs-dir` | `--runtime ID` or `--runtime-profile FILE` |
| `train`, `stage` | The config's `logging.root_dir`, or `train --runs-dir`; `stage --output` | `--runtime ID` or `--runtime-profile FILE` |
| `experiment`, `worker`, `controller` | `--store`: the same store for registration, submission and control | A named worker, or a local runtime for single-config submissions |
| `runtime env` | Host-local registry and runtime root, independent of the work directory | Manages interpreters; never launches an experiment |

- **`--lab-dir`** selects lab records; **`--runs-dir`** selects runs and
  checkpoints; **`--store`** selects a controller database with its workers,
  attempts and receipts. They are not interchangeable.
- **`--runtime ID`** names a registered interpreter on this host;
  **`--runtime-profile FILE`** gives the same description as a file, without
  registering it. See [runtime](runtime.md).
- **`--backend`** requests where the workload runs. It never selects an
  interpreter or installs a framework.

Paths inside try, probe and explore declarations resolve relative to the
declaration file, with `~` expanded. Explicit paths are always used exactly as
given; choosing a work directory never moves them.

## Verified reuse

Iterations often share a corpus, tokenizer and prepared arrays. Rehashing all of
them before every phase can cost more than the phase itself, so Lab keeps signed
proofs of what it has verified:

| Situation | Verification |
| --- | --- |
| A new or changed input | Full byte and content verification. |
| An unchanged input in an owner-only registered store | Signed host-local proofs are reused. |
| Transfer to another worker or store | Bytes and contents verified before publishing. |

Changed fingerprints, dependencies, code identity, unsafe paths, or missing or
foreign proofs fall back to full verification. Another process running as the
same user is outside what local proofs protect against. `stage --cold-verify`
turns reuse off; use it at a new host or trust boundary.

Preparation work is bounded by the [resource envelope](runtime.md#resource-envelopes):
worker counts follow available CPU, memory and explicit limits, and results are
reduced in input order so concurrency never changes the prepared data.

### Phase observations

`data prepare` and `stage` accept `--observations-output PATH.json`, a new
file under an existing directory outside the prepared cache and stage output. It records each phase's timings, host coverage and verification
counters. Observations are operational evidence only and never enter a model,
data or run identity. Unavailable counters are `null` with a reason, and a
failure to write observations only warns.

## Guard a command

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

For bounded CPU handoffs from an external scheduler, use [lab jobs](lab-jobs.md).

### Share a Linux host between runners

Opt into the same native monitor with all three paths below. Every contender
on a host must use the same paths on a local filesystem with reliable `flock`
and `fsync`. For example, with hosts named
`cpu` and `rocm`:

| Host | Canonical lock | Blocked marker | Receipt directory |
| --- | --- | --- | --- |
| `cpu` | `~/src/.lab-cpu-runner.lock` | `~/src/.lab-cpu-runner.blocked` | `~/src/.lab-cpu-runner-state` |
| `rocm` | `~/src/.lab-rocm.lock` | `~/src/.lab-rocm.blocked` | `~/src/.lab-rocm-state` |

All three paths are configurable; every job on a host must agree on one set.
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

For a CPU command already authorized by the owner (use a fresh log directory):

```sh
klab monitor --policy /data/klab/monitor.yaml \
  --workspace /data/klab --log-dir /data/klab/logs/unique-launch \
  --runner-lock "$HOME/src/.lab-cpu-runner.lock" \
  --runner-state "$HOME/src/.lab-cpu-runner-state" \
  --runner-blocked "$HOME/src/.lab-cpu-runner.blocked" -- \
  python -c 'print("tiny CPU command")'
```

Invoke the workload directly, not through another launcher that takes the
same lock. Contention refuses immediately; there is no queue, retry,
takeover, unblock command or lock-keeper process. Give all three flags or none.

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
attempt. The tool never deletes state or trusts a stale PID as clearance.

## Clean up

Prepared-data caches created inside an experiment workspace get an ownership
marker; caches without one, or outside the workspace, are never cleanup
candidates. Shared data and worker dispatch caches are outside this
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
checkpoint generations per run and two workspace-owned prepared-data cache entries.
Change those limits with `--max-extra-periodic` and `--max-cache-entries` when
creating the plan. Latest, best, and the previous verified generation are always
protected. Every generation registered in the run database or bound by a child
run manifest is also protected,
even when this exceeds the count limit; most ordinary training checkpoints are
registered, so checkpoint proposals can be empty. Only verified, unregistered
generations in terminal runs can be proposed. Cache candidates require the
workspace ownership marker, a matching prepared-data manifest, and no active
run or worker attempt. Applying a plan rechecks the database, candidate
inventory, and writer leases; changed plans must be created again. The command
never scans arbitrary sibling workspaces or deletes unmarked directories.

## Move retained data

Separate three things before moving anything: identity (hashes, run IDs,
checkpoint and input digests), where the work originally ran, and where readers
should find the bytes now. Keep the first two; update only the current
location. Never move an active run or merge controller databases by hand.
Prefer same-device renames, then verify checkpoints, lineage and evidence
afterward.
