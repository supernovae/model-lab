# Independent workers

A worker runs a whole experiment with its own optimizer, random state, local
database, checkpoints and immutable inputs. The controller owns a separate local
queue and a read-only view of results. Local and SSH workers speak the same
versioned stdio protocol; neither is a persistent GPU daemon. This is
independent experiment scheduling, not distributed training.

Controller commands default to `<work-dir>/runs` (see
[workspaces](workspaces.md) for how the work directory is chosen). Local worker
processes put temporary files under `<work-dir>/scratch`. SSH workers resolve
their own work directory on the remote host, so configure that environment
separately. Explicit `--store`, `--runs-dir`, `--output` and cache paths are
used exactly as given.

For bounded CPU work handed in by an external scheduler, see
[lab jobs](lab-jobs.md); it runs the ordinary tokenizer, `try` and `report`
lifecycle and adds no second scheduler.

## Local operation

Prepare the tokenizer and use a concrete configuration whose backend and precision the worker supports:

```sh
export KLAB_WORK_DIR=/data/klab
WORK="$KLAB_WORK_DIR/experiments/runtime-smoke"
mkdir -p "$WORK"
klab worker register cpu-one --backend cpu --store "$WORK/runs"
klab worker status cpu-one --json --store "$WORK/runs"
klab experiment submit configs/runtime_smoke_cpu.yaml --worker cpu-one --store "$WORK/runs"
klab controller run --store "$WORK/runs"
```

Run the controller in its own terminal. From another terminal, use the same `--store`:

```sh
klab experiment list --json --store "$WORK/runs"
klab experiment cancel RUN_ID --store "$WORK/runs"
klab experiment resume RUN_ID --worker cpu-one --store "$WORK/runs"
```

`--worker NAME` is a hard binding. Without one, queued submissions use eligible idle registered workers; a preferred worker is only a tie-breaker. Eligibility checks engine, backend, features, tested precision, required schema/codecs, source identity, and measured memory requirements. Unknown capacity never satisfies a numeric minimum. No eligible worker means `QUEUED` with a reason, not a silently changed configuration.

Use `try DELTA.yaml --vs BASELINE.yaml` for a local comparison. For advanced
queue dispatch, `experiment submit CONFIG --store ROOT` queues work for a
registered worker; add `--worker NAME` to select one explicitly. Use
`experiment submit CONFIG --wait --store ROOT` to drive or observe the queue
through terminal artifact ingestion. With `--wait` and no worker selection it
registers a controller-scoped local endpoint; `--runtime ID` or
`--runtime-profile PROFILE` selects an authorized local interpreter. Runtime
selection also supports submission without waiting: the submission stays queued
until a controller processes that store. If none is running, execute
`klab controller run --store ROOT` with the same store to process the
existing queue entry. Submitting again creates another experiment; it does not
resume processing the queued entry. If another controller owns the store lock,
waiting observes it rather than competing for assignments.
Single-config submissions accept resource envelopes, tokenizer batch limits
and transfer deadlines. Matrix submissions retain their declared worker
requirements and atomic submission; waiting and these single-config options
are not supported for matrices.

Ctrl-C while waiting disconnects the observer without cancelling the worker.
Use the printed `experiment collect` or `experiment cancel` command to reconcile
or explicitly cancel it. Completion waits for artifact ingestion; a failed,
cancelled, interrupted or unresolved Colab outcome exits nonzero. For direct
checkpoint transitions use `train`; see [choosing a path](experiments.md#choose-a-path).

`controller run` and single-config `experiment submit` accept `--transfer-timeout SECONDS` (default 1800). It is a finite per-RPC deadline for bundle and artifact transfers, including hashing and transmission; increase it explicitly for a slow link or large input. Metadata requests have shorter bounded deadlines. Killing the controller never authorizes re-execution of an assigned optimizer attempt.

## Machine-local runtime selection

`klab runtime env` manages a host-local interpreter registry, not worker
registrations. Inspect hardware and installed Python environments
separately; inventory alone does not authorize a backend. A CPU checkout example:

```sh
klab runtime env discover --json
klab runtime env register cpu-py314 --python "$PWD/.venv/bin/python" --backend cpu --json
klab runtime env list --json
klab runtime env doctor cpu-py314 --json
klab stage configs/runtime_smoke_cpu.yaml --through inspect \
  --runtime cpu-py314 --output /absolute/disposable/stage
```

The provision-created runtime root is `KLAB_RUNTIME_DIR` if explicitly
absolute, else `$XDG_DATA_HOME/klab/runtimes` if XDG is absolute/nonblank,
else `~/.local/share/klab/runtimes`. The registry uses the corresponding
`$XDG_CONFIG_HOME/klab/runtimes.yaml` or
`~/.config/klab/runtimes.yaml`. Both are independent of
`KLAB_WORK_DIR`. Existing interpreters can be
registered outside the runtime root. Never copy registry entries to another
host: register and doctor the actual interpreter there; remote worker capabilities
remain authoritative on that host. `--runtime ID` resolves a local profile,
while `--worker NAME` selects a worker endpoint; they are not interchangeable.

The following **user-provisioned** registrations select the framework/device
used to execute a workload. They are setup examples, not experiment definitions.
Each interpreter
must already contain a compatible Model Lab source and actual working device;
registration runs a real optimizer doctor and refuses unavailable backends.
MLX uses `--backend metal` (its profile engine is MLX).

```sh
klab runtime env register cuda-local --python /absolute/cuda/bin/python --backend cuda --json
klab runtime env register xpu-local --python /absolute/xpu/bin/python --backend xpu --json
klab runtime env register mps-local --python /absolute/mps/bin/python --backend mps --json
klab runtime env register mlx-local --python /absolute/mlx/bin/python --backend metal --json
klab runtime env doctor cuda-local --json
```

The sole automatic vendor recipe is the **candidate** `rocm-gfx1100-v1`:
`klab runtime env provision rocm-7900xtx --recipe rocm-gfx1100-v1 --json`.
It requires verified host/driver support, Linux x86_64, an existing Python 3.14
executable, writable local runtime storage with at least 20 GiB free and 100,000
free inodes. This command installs into a new runtime-root path, not the checkout
`.venv`; failure retains the partial environment unregistered. There is no automatic
garbage collection. Do not sync a CPU extra into a vendor environment or treat
installation success as confirmation that the selected workload will run. Details:
[machine-local runtime environments](runtime.md#register-an-interpreter).

## User-provisioned SSH workers

The following is an example for an already provisioned host, not evidence that such a host is available:

```sh
klab worker register amd --backend rocm --ssh amd-host \
  --python /opt/klab/bin/python --root /srv/klab/amd \
  --store "$WORK/runs"
```

The SSH alias must already be configured in the user's SSH environment. Transport uses strict host-key checking and `BatchMode=yes`; it does not store passwords, accept unknown host keys, install software remotely, or execute commands from experiment configurations. Interpreter and worker-root paths are absolute. The endpoint runs a fixed, shell-quoted agent command. Worker roots are private and bound to an immutable registration.

Provision each target with its vendor-compatible PyTorch/runtime and the
project wheel. Base Model Lab is Torch-free; `--extra cpu` explicitly opts into
CPU PyTorch for development and CI. The lightweight `klab runtime env`
entry works without Torch, but the full CLI still imports a backend framework.
Do not apply the CPU extra to accelerator workers. **Select the provisioned
interpreter directly, or set `UV_PROJECT_ENVIRONMENT` to its environment prefix
and use `uv run --locked --no-sync` for every command in a provisioned accelerator
checkout.** A syncing `uv run --locked --extra cpu` can replace a vendor wheel
with CPU PyTorch. Check the installed framework API and actual device before
staging (for ROCm: `torch.version.hip`, `torch.cuda.is_available()`, and
`torch.cuda.get_device_name(0)`), and require explicit backend validation at
warmup. An unknown device or missing vendor runtime is a provisioning decision
to escalate, not a signal to select another backend from the host OS or an old
run manifest. Record actual driver/framework/runtime versions and perform the
worker's concrete validation. PyTorch MPS and optional MLX Metal are different
engines sharing the same physical Apple GPU lease.

Choose the host environment and backend independently: a WSL2 worker is a Linux
worker that can request CPU, CUDA, ROCm, or XPU when its installed stack and
hardware support that backend. Registration does not infer a GPU vendor from
WSL2 or from the worker's name. Apply the same protocol, staging, device leases,
and task-owned store conventions on native Linux and WSL2. See
[host environment and compute backend](runtime.md#host-and-backend-are-separate)
for filesystem, driver, telemetry, and acceptance boundaries.

The optional [gfx1100 provisioning requirements](../requirements/rocm-gfx1100.txt)
retain the existing AMD worker's Python 3.14/ROCm package pins. They are an
explicit device-specific provisioning input, not a Linux or WSL2 default and
not a prescription for every AMD GPU. Other AMD devices, NVIDIA CUDA, and Intel
XPU workers use their own compatible vendor stack. The base project wheel has
no PyTorch dependency; provision the vendor framework separately and install
Model Lab without resolving or replacing that framework (for example, with
`uv pip install --python /absolute/vendor/bin/python --no-deps --editable .`
from the checkout). Keep the vendor package constraints when resolving other
dependencies. On Apple arm64, MLX remains opt-in; the full CLI currently needs
both `--extra mlx` and `--extra cpu` in a managed project environment.

## Cost and spend caps

Give a rented worker its hourly price when you register it, and optionally a
cap on what it may spend in this store:

```sh
klab worker register vast-4090 --backend cuda --ssh vast-4090 \
  --python /opt/klab/bin/python --root /srv/klab/worker \
  --hourly-cost 0.42 --currency USD --max-spend 25 --store "$WORK/runs"
```

Every attempt on a priced worker then carries a `cost` in
`klab experiment list`: the attempt's wall time from its receipt, the hourly
rate and the amount. `klab worker status NAME` shows the worker's attributed
spend beside its cap. Once the attributed spend reaches `--max-spend`, the
controller stops assigning that worker new work and says why in the queued
reason; work already running finishes. Register the worker again with a higher
cap to continue.

This is attributed cost: run time at the hourly rate. A rented machine bills
for every hour it exists, including setup, idle time and storage, so the
provider's bill is at least this much. Lab never starts, stops or pays for
an instance; turn machines off yourself when the queue is empty.

## Supplied Colab workers and relay-backed recovery

Colab is a user-supplied, single foreground notebook kernel, not an SSH host,
daemon, or scheduler. Before any automatic kernel execution, generate the
fixed enrollment cell with `klab hosted notebook-cell --root
/content/klab --output ABSOLUTE_PATH --json`, then run that cell manually
in a dedicated notebook. The observer records boot, PID, and process-start
identity through the Contents API. Unknown or `BUSY` occupancy refuses work; it
is never queued behind another cell.

Register only the setup-returned absolute interpreter and worker root with
`worker register NAME --colab-session SESSION --backend cuda --python PYTHON
--root ROOT --relay-profile PROFILE --store STORE`. Colab requires the relay.
The local controller starts one official foreground execution only after a
native prepared receipt; while it is occupied, passive status and relay
collection make no kernel `exec` call. See the
[hosted guide](hosted-environments.md) for enrollment, setup, and exact
provider boundaries.

Relay checkpoint publication is synchronous: the optimizer cannot advance past
a finalized checkpoint boundary until its immutable, authenticated recovery
closure has been uploaded and read back. Transfer time is recorded separately.
Each attempt has a private 32-byte HMAC key outside the relay and scientific
artifacts; credentials and keys never enter a profile, declaration, manifest,
or log. Relay is a verified object copy, never a live SQLite/WAL or training
filesystem.


## Preparation and durable identity

Submission verifies and seals an engine-neutral dispatch bundle before atomically reserving experiment, attempt, and run IDs. It contains exact tokenizer/data/package bytes, configuration and source identities, supported version requirements, and selected continuation artifacts. Original input paths are not needed after successful preparation. Promotion must match the requested tokenizer/package identity before any path is rebased; full resume can prepare from verified parent-owned inputs after the originals are gone.

The worker checks its content-addressed cache, installs only missing verified assets, revalidates its actual runtime, and runs disposable smoke/warmup pilots. Pilots are separate runs and never advance the experiment's optimizer, schedule, cursor, RNG, or target budget. Memory failure produces a proposal rather than silently reducing scientific settings.

A durable `PREPARED` receipt precedes spawning. A nonblocking attempt claim prevents duplicate wrappers from training. A claimed execution records PID, process-start identity, boot identity, start token, and phase before training. Replaying the same launch returns the same receipt; conflicting identities fail. `RUNNING` and terminal attempts are never restarted under their old IDs.

Each configured worker has one logical slot. Host-wide OS leases additionally serialize aliases of the same accelerator across scratch roots; an unknown accelerator identity conservatively excludes other accelerator work. Isolated pilots inherit the lease, so killing their wrapper cannot release hardware while a child still owns it. This namespace assumes one service account per host, not coordination across unrelated user accounts.

## Cancellation, disconnects, and explicit recovery

- Cancelling a queued experiment creates no executor. An assigned cancellation remains durable through a lost RPC and is redelivered until evidence arrives.
- A marker arriving before preparation is retained without inventing a receipt or acknowledgement. Once preparation is known and unclaimed, cancellation can acknowledge without initializing a model.
- Staging checks cancellation at pilot boundaries; an executing pilot and the trainer stop at committed update boundaries. After model initialization, even a step-zero interruption has a real full checkpoint. There is no fabricated pre-initialization checkpoint.
- `CANCEL_REQUESTED` is intent. Acknowledged interruption becomes `CANCELLED`; natural completion already committed at the budget wins a late request.
- A controller disconnect does not stop worker updates. Reconnection resumes record/artifact ingestion, not optimizer execution.
- A lost executor is finalized only after its attempt claim and inherited resource holders are gone. It remains `UNKNOWN`; verified finalized generations are available for an explicit child resume. Missing crash-window manifest/checkpoint projections can be reconstructed from those verified files without creating an echo outbox. Conflicting metadata is rejected, never overwritten.
- Resume creates a new visible child with the selected full-state digest. It requires compatible same-engine/backend training state and a remaining scientific budget. Complete budgets cannot resume. Source/runtime drift requires explicit `--allow-runtime-drift` and remains best-effort, not an architecture/data compatibility bypass.
- Supported dense PyTorch full-state resumes and recovery may separately opt into `--allow-runtime-optimization` for the narrow execution changes described in [resume](checkpointing.md#resume). Scientific settings, effective training batch and validation coverage remain bound; the child records a best-effort transition.

Execution state and `ingestion_status` are separate durable facts. A remotely complete receipt does not mean its checkpoints are already locally available. `experiment submit CONFIG --wait` waits for `COMPLETE` or `NOT_REQUIRED` ingestion for every terminal outcome, including cancellation and failure. Queued cancellation and failures with no run artifacts require no transfer; they do not fabricate a checkpoint. Full resume is admitted only after verified local ingestion. Artifact-transfer errors remain visible and can be retried without rerunning the optimizer.

## Local aggregation and integrity

Every worker's database/WAL stays local. The controller imports contiguous `(origin_id, sequence)` envelopes, rejects conflicting duplicates or gaps, and never re-emits imported records into its own outbox. Live runs therefore have real local metric series before completion.

Terminal artifacts are downloaded into a private temporary directory, checked against the receipt and sealed dispatch, then atomically published. Checks bind attempt/run/experiment/worker IDs, requested and effective configurations, source policy, continuation lineage, matrix coordinate, exact input identities, full-native checkpoint state, terminal counters, and replicated manifest/checkpoint metadata. Checkpoint generations use their embedded canonical digest; a file's transport hash is not that identity. Logs are sealed snapshots rather than mutable live files. Failed pre-training attempts with only diagnostic logs do not claim a resumable run.

Protocol v1 uses a bounded canonical JSON header and length/hash-framed attachments. Unknown versions/operations, duplicate JSON keys, nonfinite/coerced scalar values, unsafe names, symlink destinations, excess sizes, truncated bodies, and conflicting hashes fail closed. Error responses cannot carry attachments. A `cancel` response may contain no receipt only while its marker precedes preparation; acknowledgement is then false.

Native state codecs are engine-specific: PyTorch uses `pytorch_native/v2`; MLX uses `mlx_native/v1`. Capability advertisement and dispatch requirements derive these versions from their implemented codecs rather than treating a generic checkpoint schema version as every engine's native-state version.

## Explicit matrices

A version-1 matrix names a base configuration and insertion-ordered axes. Each option has a label and explicit dotted-path `set` patch. Expansion rejects duplicate labels, overlapping paths, unknown keys, invalid resulting configurations, and more than 1000 coordinates unless `--max-runs` is explicitly increased.

```sh
klab experiment submit --matrix MATRIX.yaml --dry-run --store "$WORK/runs"
klab experiment submit --matrix MATRIX.yaml --store "$WORK/runs"
```

Dry-run resolves the complete coordinates, configuration hashes, and scheduling constraints without creating the controller store, preparing data, probing devices, or enqueueing. Non-dry-run preparation must succeed for every coordinate before the queue transaction. Each coordinate receives independent IDs and retains the matrix digest and labels in its spec and resulting manifest.

## Comparison and hardware limits

The read-only dashboard can use the controller root as `--runs-dir`. Compare actual observed losses/counters and keep differences in optimizer, precision, architecture, data/tokenizer, effective batch, backend, and budget visible. Equal worker names or similar parameter counts do not make a controlled scientific comparison.

Worker names are labels; their reported backend describes the actual execution
device. A worker running on CPU reports CPU regardless of its name. Checking
GPU dispatch or simultaneous work across hosts requires those provisioned
environments. SSH framing and quoting tests check transport behavior; they do
not measure remote driver or kernel performance.

No worker shares optimizer state or exchanges gradients/expert tokens with another. There is no process group, all-reduce, expert all-to-all, tensor/model sharding, or heterogeneous distributed backward.

## Manifest replication

Worker metrics envelopes remain bounded to 64 KiB. A run manifest is retained in
full in the worker store and native `manifest.json`; its outbox record carries a
`manifest_reference` with the canonical payload SHA-256 and byte length. The
manifest file's enclosing hash and newline are not part of that payload length.

Controllers retain the reference until native artifact collection verifies the
complete manifest, dispatch identity, input inventory and checkpoints. Successful
publication resolves the reference to the full manifest without emitting another
outbox record. Relay collection also checks that each reference has its matching
immutable manifest artifact. Standalone store consolidation verifies the source
manifest against the reference before copying its projection. Cleanup refuses an
unresolved reference so continuation ancestry cannot silently disappear.

While a manifest reference awaits artifact collection, the controller dashboard
labels unresolved worker/continuation/checkpoint fields as **pending artifact
collection**. It retains the parent run and worker assignment from the controller
store when available. Collection replaces this presentation with the verified full
manifest; an unresolved reference never establishes that a run has no parent.
