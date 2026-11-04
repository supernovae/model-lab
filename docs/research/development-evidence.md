# Retained development evidence

These notes preserve measurements and campaign-specific procedures used while
building the lab. They are historical context, not prerequisites or default
workflows for a new experiment. Use [Campaigns](../campaigns.md),
[iteration](../iteration.md), [capacity and reuse](../capacity-aware-execution.md)
and [lifecycle recovery](lifecycle-recovery.md) for general operation.
Original protocols, artifacts, failed observations and identities remain authoritative.

## Capacity execution measurements and implementation notes

### Historical baseline

[Compact baseline](https://github.com/supernovae/tiny-sparse-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/benchmarks/capacity-aware-execution-v1-baseline.json)
binds exact JSON pointers into versioned
[execution records](https://github.com/supernovae/tiny-sparse-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/devmind-pretrain-v5/model0-execution.md),
[preparation](https://github.com/supernovae/tiny-sparse-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/devmind-pretrain-v5/model0-preparation-verification.json),
[lock acceptance](https://github.com/supernovae/tiny-sparse-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/devmind-pretrain-v5/model0-plan-lock-acceptance.json),
[current-source readiness](https://github.com/supernovae/tiny-sparse-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/devmind-pretrain-v5/model0-current-source-full-shape-readiness.json)
and [completed result](https://github.com/supernovae/tiny-sparse-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/devmind-pretrain-v5/model0-result.json).
These are retrospective operational evidence, not new scientific results.

| Boundary | Observed wall seconds | Interpretation |
| --- | ---: | --- |
| Source denominator | 29.325819 | Native encoding/scan; 1,469,599,744 B peak tree RSS |
| Preparation | 2,274.680755 | Inclusive; 8,834,408,448 B peak tree RSS |
| Lock / independent cold explain | 1,604.321869 / 1,538.799164 | Repeated full input closure authentication |
| Pilot input validation | 621.631 / 628.283 | Smoke/warmup inclusive authentication |
| Pilot materialization | 638.220 / 609.643 | Smoke/warmup inclusive materialization |
| Optimizer updates | 1,979.606588 | 5,525 updates; 22,863.533 targets/s |
| Training operation | 2,459.213342 | Excludes worker staging and Campaign reconciliation |
| Native heldout | 7.823494 | Not a demonstrated host bottleneck |
| Four training checkpoints | 13.698850 | Nested in training operation |
| Reconciliation censored windows | 25,550.210 / 21,601.291 | Separate interrupted windows, not successful elapsed |
| Final closure | 8,309.562 | Successful same-attempt closure |

Preparation records 4,081,965,445 logical input B, 4,031,007,060 logical
output B and 812,974 records. Native tokenizer encoding is 598.125071 s;
final fsync is 17.091629 s. These are nested counters, not disjoint phases.
The retained prepared root has four files (two arrays, manifest and ownership
marker), totaling 4,031,011,511 B. Per-file sizes and frozen array SHA-256
are in the baseline. CPU and physical I/O counters absent from the records
are null. Disk free delta is not physical write traffic. Historical preparation
swap is host-wide used swap; later monitored swap is owned process-tree swap.

### Call graph and data-flow boundaries

| Entry | Repeated downstream checks/copies | Identity authority | Proposed reuse boundary |
| --- | --- | --- | --- |
| `resolve_plan` / `open_lock` | `verify_artifact` for each input and corpus variant | Typed kind/version/identifier/domain SHA plus package source | Explicit trusted store; direct calls remain cold |
| Tokenizer domain verifier | Export, release, build, snapshots via provenance | Tokenizer bytes and exact export/release closure | Unchanged typed ancestor closure |
| Prepared domain verifier | `load_prepared_data`, array SHA and header validation | Manifest/settings/tokenizer/source SHA | Host-local signed proof; new process-local seal |
| `stage` / pilot `input_validation` | Tokenizer closure, stage inventory, prepared input checks | Stage inventories and prepared seals | Explicit mode passed into pilot child |
| Pilot `input_materialization` | Source arrays into private prepared and assets trees | Canonical SHA, source fingerprint, destination verification | Private copies/clones; private stage-only links |
| Dispatch install/materialize | Asset SHA cache install, cache rescan, private copy | Existing dispatch asset/bundle CAS SHA | TRANSFER cold; same-store signed reuse |
| Campaign / Controller reconciliation | Artifact checks, RPC status, receipt ingestion and collection | Durable attempt state and typed output identities | Unchanged local evidence only; terminal ingestion gating retained |
| Archive / recovery / family | Independent byte and semantic verification | Final durable evidence authority | Always COLD |

### Ranked measured opportunity, before tuning

1. Repeated typed closure authentication: 1,539–1,604 s lock boundaries and
   622–628 s pilot gate. Highest eligible operational reuse opportunity; byte
   hashing versus semantic parsing share is not yet isolated.
2. Private materialization: 610–638 s inclusive pilot phase, roughly 4 GB
   canonical arrays. Clone/copy mechanisms require measured destination safety.
3. Preparation native encoding: 598 s nested component. Already Rayon-backed;
   no Python parallel producer or larger thread default is justified yet.
4. Reconciliation: largest observed windows but censored and unattributed.
   Profile before claiming improvement; 0.1 s sleep does not explain hours.
5. Independent multi-file cold SHA: real prepared deep check 15.172 s; eligible
   only when comparable medium timings demonstrate benefit. Single-file SHA
   remains native sequential SHA-256.

Heldout (7.823 s), checkpoint writes (13.699 s), ISA-specific hashing, GPU
parallelism, source parsing parallelism and new package dependencies are
unmeasured/ineligible optimizations, not claimed gains.

### Trust modes

- **COLD**: independent full byte and domain-semantic verification. Default for
  direct Python APIs, external paths and final archival.
- **VERIFIED_REUSE**: unchanged bytes and typed upstream closure in the same
  authenticated, owner-only registered persistent store, using signed local
  receipts. Not protection against a compromised same-UID owner.
- **STRUCTURAL**: same-operation, process-owned sealed `VerifiedFile` or
  `_VerifiedArtifact`; fingerprints, inventory and headers still checked.
- **TRANSFER**: full SHA and semantic authentication once upon entry to a new
  worker/cache trust domain, before publication.

Operational hardware, selected workers, proof/key paths and trust modes never
enter scientific, prepared, artifact or plan digests. Relocation, missing or
foreign receipts, changed fingerprints/closure or implementation, unsafe paths
and signature failure require COLD fallback; failed cold verification is fatal.

### Bounded host planner

`host_capacity.plan_host_workers` bounds native independent tasks by affinity,
physical CPUs (logical fallback), measured available RAM after explicit reserve,
per-worker memory and operator cap. Unknown RAM permits only one <=8 MiB SHA
worker. Callers reserve max(1 GiB, total RAM/10); no memory-heavy unknown-RAM work.
The SHA bound includes the 1 MiB hasher buffer and thread overhead; observed
medium process RSS differs by only a few MiB across 1/2/4 workers.
`run_ordered` keeps at most the selected worker count outstanding, reduces in
input ordinal order and cancels/joins on failure or iterator close.

[Before measurements](https://github.com/supernovae/tiny-sparse-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/benchmarks/capacity-aware-execution-v1-before.json)
use independent processes and uncontrolled OS cache, not evicted-cache claims.
Their elapsed times include package imports. Threaded SHA did not demonstrate a
material end-to-end benefit; production cold SHA defaults to one worker.
The bounded host probe reports x86 SSE4.1/4.2, AVX/AVX2, SHA-NI and AES here;
AVX512F is absent. Native SHA is `_hashlib`, linked OpenSSL 3.5.5. Capability
presence is not an acceleration result; no custom hashing or dependency added.

### Host-local proof protocol

`verification_proofs.ProofStore(work_root)` stores HMAC-SHA256 receipts under
`cache/verification-v1/`. The separate 32-byte secret is
`$XDG_CONFIG_HOME/sparselab/verification-key-v1` (absolute XDG only), otherwise
`~/.config/sparselab/verification-key-v1`. Receipt/key files are 0600 and their
private directories 0700. No proof is written inside an inventoried artifact.

Bindings include typed kind/version/identifier/SHA, canonical absolute path,
manifest/inventory digest, exact upstream identities, full member/dependency
device/inode/mode/size/mtime-ns/ctime-ns fingerprints and verification success.
Receipt schema 2 binds a *typed verifier authority*: SHA-256 of actual source
bytes for domain verifier entry points, recursively discovered local static and
lazy imports and package initializers, except audited operational import edges
whose existing uses are AST-pinned. Typed artifact authorities include their
own domain branch in `experiments.artifacts`, not unrelated stage, checkpoint
or evaluator branches. Prepared array and directory receipts exclude producer
and execution-only imports, including training. Changes to an excluded use
disable reuse until that edge is re-audited; newly imported local helpers enter
automatically even outside the original domain package. Missing mandatory
local import modules or unexported package helpers disable receipts rather than
silently shortening the authority closure. Nonliteral dynamic imports cannot
authorize reuse and remain cold; constant local dynamic imports, including
direct imported aliases, join the closure. Escaped import loaders and dynamic
`globals`/`locals` namespace access disable receipts rather than concealing an
indirect verifier dependency. File kinds include prepared arrays, stage inventories, checkpoint
members, dispatch-cache assets and run evidence.
Unknown kinds never accept or publish receipts; cold authentication still
succeeds. Unlike the former package-wide source digest, editing unrelated CLI,
UI, training code or documentation does not invalidate an unchanged
prepared-input receipt. Edits to relevant verifier code do; an old
schema-1 receipt is cold-verified and sealed anew only after successful cold
authentication, never accepted or re-signed from unsigned metadata. This is an
operational trust binding, not a change to scientific or artifact identity.
Only currently minted, process-owned cold verifier seals can publish receipts.
Warm validation checks the complete signature and binding, trusted ownership
and fingerprints before and after lookup, then mints a new process-local seal.
Prepared inventory, manifest digest, dtype and shape checks still run.

Trust requires the selected root and paths below it to be service-UID-owned and
not group/other writable. Symlinks, multiply linked regular files, unsafe stores,
foreign/missing keys, unsigned/stale receipts and changed bindings miss to cold.
Cold corruption fails; there is no mtime/size-only authority or same-UID attacker
protection. Direct Python verification remains cold unless explicitly opted in.

`ProofStore.diagnostics()` returns cumulative hits, misses, recorded counts,
reason counts and per-lookup events with artifact path/kind, typed authority,
reason, and measured file-verifier bytes hashed/avoided. Reasons include
`no_receipt`, `untrusted_store`, `changed_fingerprint`,
`changed_dependency`, `changed_manifest_binding`,
`verifier_authority_changed`, `corrupt_or_invalid_receipt`, `explicit_cold`,
and `unknown`. An unavailable authority includes `authority_error` on the event,
misses with `unknown`, and cannot publish a receipt even after successful cold
authentication. The receipt locator uses stable typed artifact fields, excluding
the manifest digest; the complete manifest/upstream binding remains HMAC-signed,
so a changed manifest reports `changed_manifest_binding` directly without
scanning unrelated receipts. Unknown artifact-verifier byte counts are null.
`read_only=True` permits lookup and diagnostics without publishing receipts or
creating keys.

Trust walks compare every signed member/dependency fingerprint with fresh
`DirEntry.stat(follow_symlinks=False)` metadata. Ancestor checks are deduplicated
only within that invocation; no trust decision survives a lookup/publication.
Tokenizer selection passes the explicit verification context to its nested
release verifier. Recovery/archive callers retain their cold defaults.

The [pre-MODEL-1 hash comparison](https://github.com/supernovae/tiny-sparse-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/benchmarks/pre-model1-hash.json)
uses fresh processes and equally pre-read inputs; it is not a disk-cold result.
On the 3,937,267,228-byte retained training array, median SHA-256/1 MiB was
1.921 seconds, BLAKE3/single-thread 1.109 seconds, bounded four-thread BLAKE3
0.540 seconds, and non-authoritative XXH3-128 0.414 seconds. SHA-256/4–16 MiB
and `file_digest` did not materially improve the existing SHA path. Keep
authoritative SHA-256 and serial 1 MiB reads; no dependency or blanket threading
change follows from these component-only measurements.

Experiment `inspect`, `diff`, `lock`, `bind`, `run`, `collect`, `explain` and
`reconstruct`, and `stage`, accept `--cold-verify`. Normal commands select the
registered persistent root; only inputs beneath its trusted paths can reuse.
Cold mode constructs no proof store and reads/writes no proof. Operational
CLI proof counters are outside plan/scientific digests. Stage/pilot/worker
parameters forward mode explicitly; portable corpus attachments retain their
independent binding verification. Archive, recovery and family remain cold.

### Prepared array verification

Cold independent array hashes use `run_ordered` with canonical filenames and
native full-file SHA-256. The measured default remains one worker, including for
the single large `train.npy`; no tree/chunk digest substitution. All 1/2/N choices
preserve exact array SHA and manifest bytes in regression checks.

Per-array signed receipts bind SHA, full fingerprint, shape/dtype metadata and
cache identity. `_receipt_from_proofs` independently rechecks the current
manifest digest, exact inventory, every new process-owned file seal and each
NPY header. A changed array node is cold-checked; unchanged array-node proofs
remain eligible while the prepared/plan closure is invalidated.
Same-size mutation with restored mtime, inode replacement, truncation, altered
manifest/dependency, relocation and forged/stale/missing receipts are covered.
Independent cold loads remain reliable; unsigned metadata cannot mint seals.

### Owned stage and worker materialization

`owned_copy` authenticates the source seal/fingerprint and destination SHA before
exclusive publication and fsync. Linux FICLONE and macOS clone are capability
attempts, not assumed available. Partial native copies reset before bounded
buffered fallback. Published destinations are private inodes (0600); source
permissions and bytes are unchanged. Directory layout and scientific inventories
still contain the same names and exact bytes.

Only a freshly materialized stage-private `prepared/assets` tree may link to the
same stage's `assets`. Both historical paths remain in the inventory. Link
creation changes ctime, so that intentional transition receives an independent
SHA check before refreshing process-owned seals. Canonical inputs, supplied
external stage/prepared roots and worker CAS never link to mutable worker paths.
Persistent receipt trust rejects multiply-linked files; stage-private links use
operation-local seals and cold verification, not a relaxed persistent policy.

Existing worker `.dispatch-cache` is unchanged: TRANSFER authenticates missing
content and a semantic/portable/continuation closure before signing cache
receipts. Same trusted closure hits avoid CAS SHA rescans; a changed/corrupt
existing CAS asset is fatal, never silently repaired. Destination copies remain
independently authenticated. Public `verify_dispatch_bundle` remains cold.
Copy/cache observations are optional return-side collections, never inventory or
scientific digest members; unavailable physical I/O remains null.

Three alternating fresh-process copies of the same prehashed 67,108,992-byte
medium array on this host used `copy_file_range` (FICLONE unavailable):
0.123528 s median, 0.122716–0.123896 s range, versus bounded buffered copies
0.130750 s median, 0.129062–0.138465 s range. Exact SHA matched all six outputs.
CPU medians were 0.076741/0.075849 s and process high-water RSS maxima
377,782,272/377,884,672 bytes respectively. OS cache was uncontrolled, source
prehash preceded each timed copy; this is a copy-plus-destination-SHA/fsync
comparison, not end-to-end preparation or a physical-I/O claim.

Materialization gate: 74 focused copy/proof/stage/worker tests passed, including
late competing-directory no-replace publication. Actual disposable CPU runtime
exercised stage validate → dispatch prepare → TRANSFER → warm install/private
materialize: unchanged worker manifest digest, zero warm CAS SHA reads, isolated
inodes, private corruption rejected and canonical array SHA unchanged. Config
key creation preflights symlink ancestry before creating directories.

### Corpus ancestor reuse and bounded parsing

Mutation commands forward the same host-local proof mode through snapshot,
build, release, export and tokenizer verification. Defaults on direct APIs and
recovery remain cold. Mutable local acquisition inputs are independently read
and hashed: a published snapshot receipt never blesses its original raw pathname.
Full inventory/dependency fingerprints, not a manifest-only memo, bind release
reuse. Rendering, split statistics and tokenizer document iteration still read
the bytes their semantics require. Config/evaluation changes reuse unchanged
ancestors; changed tokenizer/export nodes receive their own canonical bindings.

The measured parser bottleneck included research-exclusion normalization
(3.082 s of 4.519 s profiled record parsing), not just JSON decoding. Independent
HF/Wiki JSONL shards at least 32 MiB, with rows bounded to 1 MiB, can use two
processes. Selection uses the shared CPU/affinity/RAM planner, a 512 MiB bound per
worker and its reserve; unknown/inadequate capacity, long rows, nested metadata,
other formats and already prepared shards retain serial processing. Submission
is bounded and results/counters reduce in canonical source order. SQLite
deduplication, release identity, provenance, emitted bytes and error ordering
are unchanged. No arbitrary regex/ISA rewrite or blanket source parallelism.

Three alternating fresh-process observations of two 32 MiB shards include row
preflight, process startup and shard receipt/output hashing. Serial median:
5.076521 s (5.059783–5.076647); two-process median: 3.415149 s
(3.383993–3.447172). Exact output SHA matched all six observations. Process-tree
RSS maxima were 270,241,792/908,754,944 bytes, observed swap zero; sampled CPU
lower-bound medians 5.23/7.98 s, physical I/O unavailable. This trades CPU/RAM for
shard preparation wall time, not a measured end-to-end corpus speedup.
Separate warmed multi-file SHA observations were not comparable end-to-end
release measurements; release hashing remains serial.

Raw measurement: task-owned `corpus-process-final.json`,
SHA-256 `7222e089bf001452b96f0a38aaac82fae39456601c07ade4547627726b775ce9`;
package implementation `3d55097f77cee97595f50030ca57e9669c2077943187bbfa5d56a3e336fc7557`.
The 138-test focused gate covers snapshot/ancestor reuse, same-size restored-mtime
mutation, canonical tokenizer/export binding, config/evaluation-only changes,
and serial/process full-build bytes plus failure counters. Actual CLI acquire,
repeated offline build and freeze completed in the disposable sample workspace.

### Native encoding and evaluation capacity

Medium profiling retained the existing 256-document/1 MiB tokenizer batches and
fresh native Rayon child. Three alternating fresh cases per 1/2/4 choice encoded
66,272,100 source UTF-8 bytes into 19,011,624 tokens. Before planner integration,
wall medians were 24.871841/15.423598/10.363416 s; native encoding medians
20.999457/11.467340/6.838632 s; process-tree RSS maxima
477,204,480/479,256,576/481,943,552 bytes. Full arrays, manifest and cache identity
matched across worker choices within that revision.

`PreparationEncoder` now selects CPU/affinity/RAM-bounded native threads with the
shared planner before creating its child. Operator `max_workers` is an upper
bound, not permission to exceed measured four-thread capacity. The conservative
per-thread bound is 512 MiB, reserve max(1 GiB, total RAM/10); unknown/inadequate
RAM rejects memory-heavy work. The plan is operational receipt telemetry only.
Same-input post-change medians were 24.585059/15.442836/10.392618 s, tree RSS maxima
476,958,720/479,350,784/481,722,368 bytes. Four-thread before/after ranges
10.312176–10.419641 / 10.375050–10.414132 s overlap; the 0.29% median difference
does not demonstrate a throughput gain. Retained the existing four-thread native
choice with capacity guards, not extra producer processes or larger batches.
Arrays match across the cutover; scientific identities match among worker
choices within each revision, not across changed package identities.

Three warm-process, line-traced CPU-only evaluations of a two-update synthetic
checkpoint preserve exact metrics and canonical index bytes on replay. Median
host preprocessing/reporting were about 1.060 ms/0.012 ms versus native model
7.422 ms, authenticated model load 10.501 ms and suite wall 28.523 ms. CPU
pre/postprocessing did not dominate: no host worker pool added, GPU execution
unchanged and unmeasured. Process-lifetime RSS (497,868,800 bytes) includes
fixture creation and is not an evaluation-specific peak. Physical I/O and
accelerator utilization were unavailable, not inferred from wall time.

Raw task-owned evidence SHA-256:
- `encoding-profile.json`: `418cea59da8d1211697378066b8d1780f6a475be6e2eb5ce7ecbe4510c394c24`
- `encoding-profile-after.json`: `d6d58da9c27d4b6631721a3d6876caabaa9b864a71a78c1c9f4e99990e96c690`
- `evaluation-profile.json`: `324b9c074c90656f809d90e83573a9f6b30a2f6b97d1a9ee910d7c9aea980380`

The 92-test gate covers host bounds, fresh Rayon exact-output parity, streaming
packing, resource envelope enforcement and checkpoint-bound evaluation suites.

### Durable reconciliation and measured observations

Campaign run reconciliation re-queries durable attempt state immediately after
each controller tick. COMPLETE requires ingestion COMPLETE; a changed durable
row is re-evaluated without sleeping. Only an unchanged row before the shared
deadline consumes bounded idle sleep. Replay IDs, terminal/ingestion distinction,
pending-on-deadline and crash recovery remain authoritative.

Run inventories and checkpoint members use the existing signed file proof index
with authoritative run inventory/checkpoint/manifest bindings; semantic checks,
prepared headers and evaluation/checkpoint correspondence still execute.
`experiment_evidence`, collection, retrospective reads and Campaign evaluation
consumers forward explicit proof mode. Native inference still reads model weights
for loading and evaluation inputs for computation; avoiding their redundant SHA
scan is not avoiding the required computation/input read. Independent direct
APIs, recovery and archival remain cold.

CLI `experiment run --cold-verify` now reaches controller preparation, worker
TRANSFER/cache installation, private materialization and pilot/training gates.
RPC `cold_verify` is operational, outside ExperimentSpec and bundle digests.
Missing direct/external launch policy defaults cold; cold requests are sticky
across PREPARED same-ID replay. Running/terminal replay is not mutated or retrained.
The worker's private `verification.json` is outside scientific inventories.

`BottleneckObserver` is opt-in on preparation, staging, Controller and Campaign.
Phase records measure matched surviving-process user/system CPU and available
process I/O, endpoint-sampled tree RSS/swap and an optional real accelerator
utilization probe. Missing counters/probes are null. CPU/I/O exclude children not
alive at both endpoints; memory maxima are sampled endpoints, not lifetime peaks.
Records never enter prepared/stage inventories, scientific/plan hashes or metric
schemas. Probe failures report unavailable observations without blocking work.
Labels are diagnostic heuristics, not scientific findings or acceptance gates:
accelerator, input_host, memory_pressure, unknown; host cpu_bound, io_bound,
serialization_bound, verification_bound, copy_bound, cache_hit, cache_miss,
unknown. Phase and cache labels require measured activity/counters plus caller
provenance; low CPU/device utilization alone is not a failure.

Before this change, bounded read-only fixtures observed expired controller tick
0.000378 s and cold evidence read of an existing two-update synthetic run
0.010273 s. They exercised no live RPC, transfer, ingestion, collection,
evaluation or idle wait; those fields remain null, not a control-plane speedup.
Actual normal CPU integration completed one two-update worker run and ingestion,
preserving canonical input arrays and cold/warm evidence. Its 39 records observed
cache_hit/cache_miss/copy_bound/unknown, endpoint tree RSS at most 400,863,232
bytes and observed swap zero; accelerator and physical I/O unavailable.
Inclusive totals included discovery RPC 2.588331 s, bundle transfer 4.267456 s,
status RPC 13.692612 s and active polling 56.961635 s. They overlap and must not
be summed. A final 31.992090 s tick included ingestion: the 0.1 s idle sleep does
not explain long reconciliation windows.

The independent cold-worker runtime also reached COMPLETE + ingestion COMPLETE
with `cold_verify=true` persisted outside the spec, unchanged canonical arrays
and equivalent cold/warm evidence. Its 51 records separately observed record
ingestion 13.578576 s, artifact transfer/verification 28.839650 s and terminal
receipt ingestion 30.144051 s (inclusive/nested); endpoint tree RSS at most
400,814,080 bytes. No physical-I/O or accelerator counters were manufactured.
Source package implementation:
`18d5ac531bf89cf60333f63d2d57be015e17edcb44bfe1d0b6b9082a0467356d`.

The broad 231-case gate initially passed 229 cases; two new counters incorrectly
counted package-source hashing or verifier entry calls as payload SHA. Corrected
their measurement scope, then all seven affected payload/observation cases
passed. Twelve follow-on controller deadline, durable replay and immediate
Campaign progress cases also passed. Different-process run evidence skips
unchanged payload SHA; restored-mtime corruption still rejects; suite replay
preserves canonical index bytes; observation failure preserves real preparation
identity. The optional-observation TODO is closed; worker-provenance TODO stays
open.

Raw task-owned evidence SHA-256:
- `control-evidence-before.json`: `b7e8ccaf32add93abf334f02a90960b7a91f8034d3693eff19f599960a5b50b0`
- `step9-smoke.json`: `ae1c5a16e9ba8ee0fdd6177b7defffb2f2f2614bebad8cc629e83cbc46a7e6e7`
- `step9-cold-smoke.json`: `2feb7c21fd1ed2be98eb99e71ece06541709ca7e6ebfe2ab8111eb9aabc35e67`

### Final comparable harness

The original cold benchmark operations and their import-inclusive timing
boundaries remain intact. Additional cases prime signed proofs before fresh
processes measure prepared/plan/lock warm reads, worker TRANSFER/cache reuse and
owned materialization. Planned cold arrays alternate requested 1/2/4 workers.
Logical payload-file SHA and private-copy authentication SHA bytes are separate;
physical I/O remains unavailable. Independent output-array SHA validation is
reported outside the additional operation timer, not silently excluded from the
original baseline materialization operation.

Copy-method equivalence compares the **same frozen stage closure** using native
and forced buffered fallback, and dispatch manifests created from the same
frozen stage. Independent `stage(..., through="validate")` invocations legitimately
have different runtime timestamps/RAM observations and therefore different stage
receipt digests; those receipts are never stripped or re-pinned for equality.
The frozen copy smoke observed `copy_file_range` and buffered transfer with
exactly equal stage digest, array bytes and SHA. Reflink is unavailable on this
filesystem; simulated clone/reflink and cross-device failures remain safety
regressions, not measured native acceleration.

An authentic HMAC receipt from an older implementation now has a dedicated
consumer regression: a current verifier falls back to full array SHA rather
than accepting stale signed authority. Config/evaluation changes resolve a new
protocol identity while retaining eligible unchanged ancestors; this does not
authorize in-place rewriting of an accepted Campaign declaration or migration
of its immutable stage receipts.

### Final observations and remaining ROI

Measured clean revision `63cc36ae73bc202603e399261b0c43d829842677`,
package implementation
`18d5ac531bf89cf60333f63d2d57be015e17edcb44bfe1d0b6b9082a0467356d`.
The final harness passed 28 small and 84 medium cases; every medium case has
three repetitions. Full median/range, user/system CPU, sampled peak tree RSS,
logical SHA/copy authentication bytes and identity assertions are in
`artifacts/benchmarks/capacity-aware-execution-v1-results.json`.

| Workload | Before wall s | After wall s | Avoided SHA bytes | Workers | Peak tree RSS B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Medium prepared cold | 1.403549 | 1.446355 | 0 | 1 | 384479232 |
| Medium resolve cold | 1.486785 | 1.501696 | 0 | 1 | 387932160 |
| Medium open cold | 1.455063 | 1.469207 | 0 | 1 | 387940352 |
| Medium materialize cold | 1.626155 | 1.680747 | 0 | 1 | 389292032 |
| Medium prepared warm | not measured | 1.437198 | 68157696 | 1 | 383152128 |
| Medium worker install warm | not measured | 1.464224 | 68157696 | 1 | 384413696 |
| Original MODEL-0 cold, single/cache uncontrolled | 3.363676 | 8.793402 | 0 | 1 | 383934464 |
| Task-owned MODEL-0 warm, three repetitions | not measured | 1.414212 | 4031007060 | 1 | 382341120 |
| Independent two 32 MiB JSONL shard preparation | 5.076521 | 3.415149 | 0 | 2 | 908754944 |
| Isolated 64 MiB copy, buffered → native | 0.130750 | 0.123528 | 0 | 1 | 377782272 |

Warm real verification range: 1.405802–1.416089 s, zero payload SHA calls in
each fresh process. Frozen original train/validation SHA and manifest identity
remain exact. The task-owned real copy exercised actual cross-filesystem
buffered fallback (`st_dev` 2096 → 2112), independent destination authentication
and cold semantic/index priming: 13.440605 s, 385253376 B peak tree RSS.
This transfer/prime boundary includes an extra destination cold scan to prime
the standalone harness; it is not the sealed structural materialization path.

No cold end-to-end speedup is established. Medium import-inclusive medians
overlap or modestly regress; original real cold observations have uncontrolled
OS-cache/storage conditions and must not be treated as a code comparison.
Physical read/write counters remain null/unavailable. The native isolated copy
gain does not imply end-to-end gain: frozen-stage native/buffered medians
1.970161/1.961420 s have overlapping ranges; dispatch-create native/buffered
1.992360/1.989157 s also overlap. Keep serial cold SHA and serial evaluation;
retain measured bounded two-process shard preparation and the isolated native
copy mechanism with safe buffered fallback. Tokenizer throughput remains
statistically overlapping, not a claimed acceleration. ISA capability presence
and OpenSSL `_hashlib` use are operational observations, not benchmarked SIMD.

Remaining measured ranking: native CLI/worker import and serialized RPC startup;
artifact transfer plus receipt/checkpoint ingestion; required destination SHA,
semantic/tensor validation and mandatory private-stage hardlink cold checks;
then native tokenizer work. Durable state idle sleep is not the demonstrated
reconciliation bottleneck. No new DAG, CAS, scientific defaults, metrics,
acceptance criteria or worker provenance authority was introduced.

### Full-gate recovery correction

The first full suite completed 1602 passes, 12 skips and eight failures; the
300-case focused gate and independent lock/explain CLI proof passed first.
Published-preparation recovery still compared its historical four-field raw
chunk fingerprint to the new six-field `VerifiedFile` seal. Recovery now uses
the verifier fingerprint (including mode and ctime) for published arrays; raw
chunk receipt schema and length indexing remain unchanged. Existing publication
and cleanup crash-window regressions pass without re-encoding.

A real external-root smoke injected a crash immediately after atomic
publication, then another interpreter cold-verified all four ID/byte-address
arrays exactly once, completed cleanup and recovered without encoding.
Implementation identity:
`466b154e5c7e0b2d7d8250929cdf3a515723b4169f0896a6e092f0232bde5f7b`.
The corpus corruption fixture now owns a copied snapshot tree rather than
passing an unsafe symlink. Runtime registry-relocation coverage uses a real
resolved/published tokenizer+prepared lock, not a mocked `open_lock` signature
or incomplete fabricated availability. Obsolete exception-wording/private
inventory tests and the unsafe persistent-hardlink warm-skip expectation were
removed rather than re-pinned. Consumer-level mutation, cold verification,
zero-hash trusted reuse and copy isolation coverage remain.

The affected 70 cases passed across the 69-pass broad subset and the corrected
runtime-relocation case. Corrected-source small/medium/real benchmark and CLI
gates are required before final publication. The earlier measurement table is
revision-labelled evidence, not silently relabelled.

### Corrected-source final gate

Clean benchmark revision `1b644745999b0f27e462ddfc850b33d29eaab13c`
uses the corrected implementation `466b154e…de5f7b`. Its 28 small and 84
medium cases passed all array, prepared/plan/scientific, frozen-stage and
dispatch identity comparisons. Each medium row again has three repetitions.
The full locked CPU suite passed **1606 tests, 12 skipped in 1225.73 s**.
Ruff check and format check passed (536 files). No accelerator or real
MODEL-0/MODEL-1 training was launched.

| Workload | Before wall s | After wall s | Avoided SHA bytes | Workers | Peak tree RSS B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Medium prepared cold | 1.403549 | 1.432169 | 0 | 1 | 384638976 |
| Medium resolve cold | 1.486785 | 1.509235 | 0 | 1 | 388173824 |
| Medium open cold | 1.455063 | 1.456826 | 0 | 1 | 385310720 |
| Medium materialize cold | 1.626155 | 1.724634 | 0 | 1 | 389394432 |
| Medium prepared warm | not measured | 1.451629 | 68157696 | 1 | 382468096 |
| Medium worker install warm | not measured | 1.467706 | 68157696 | 1 | 384405504 |
| Original MODEL-0 cold, single/cache uncontrolled | 3.363676 | 8.795047 | 0 | 1 | 384327680 |
| Task-owned MODEL-0 warm, three repetitions | not measured | 1.446625 | 4031007060 | 1 | 383508480 |
| Independent two 32 MiB JSONL shard preparation | 5.076521 | 3.415149 | 0 | 2 | 908754944 |
| Isolated 64 MiB copy, buffered → native | 0.130750 | 0.123528 | 0 | 1 | 377782272 |

Corrected-source real warm range: 1.417440–1.456875 s. Real cross-filesystem
transfer/prime: 12.130999 s, 385167360 B peak tree RSS, buffered fallback and
exact frozen array SHA. Frozen-stage native/buffered medians
1.945728/1.962605 s and dispatch-create 2.000095/2.002844 s have overlapping
ranges: the earlier no-end-to-end-native-speedup conclusion remains.

Independent current-source `sparselab experiment lock`, warm `explain` and
`explain --cold-verify` returned identical scientific/plan digests. A separate
instrumented CLI interpreter observed no warm `.npy` SHA reads, and exactly
train/validation reads when cold. Signed-proof mutation, stale implementation,
corruption, crash recovery and private-copy isolation gates passed in the full
suite. Historical MODEL-0 lock/source authority was intentionally not rewritten;
real immutable payload verification and new current-source CLI locks were
exercised instead. Physical I/O and accelerator observations remain unavailable,
not inferred from wall time.

Corrected raw references under the task-owned external root:
- `after-small-corrected.json`: `c002d546efd5aa385bb58be67504e0de490c755f52ffd0a94f4c06f5e726c8a3`
- `after-corrected.json`: `33e08d402e5dc9d665cddc1e8d44cc0d4c6445075f8bba5903b67a6f55fa1c9b`
- `final-cli-corrected.json`: `33d2df99fb7ff3b0873dac2e0385f40b90812eb0216a97141de684ee75d7d2e4`
- `recovery-final-smoke.json`: `a756cf39bed1c740f8472f368552b14e7f4abd2c45197cbffaca2e0abe2a1616`

The retained measurements above used an opt-in internal observer. Current
`data prepare`, `stage` and Campaign commands expose `--observations-output`;
see [native diagnostic interfaces](../using-sparselab.md#native-diagnostic-interfaces)
for the supported commands and output boundaries.
Without an observer, these phase samples are unavailable.
Records remain operational; endpoint RSS is not a lifetime/phase peak and
missing accelerator probes produce unknown, not an accelerator-bound claim.

## Retired campaign-specific measurement shortcut

The DevMind-only committed-evidence flags and trust implementation were removed
in the lifecycle cleanup. Current measurement uses cold release/tokenizer
verification, with an explicit tokenizer-origin release when reusing a retained
tokenizer. The earlier proof and measurements remain unchanged in the
[history archive](https://github.com/supernovae/tiny-sparse-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/README.md); their existence
does not authorize replay or replace verification of current inputs.

## Generation-panel integration history

The existing DevMind v5 MODEL-0 panel format is supported without modifying that
research campaign or claiming a new scientific result. CPU regression fixtures
exercise this integration; accelerator execution needs its own hardware evidence.

## Completed FineWeb-Edu micro study

The `engram-ffn-substitution-v1` micro study uses the pinned [FineWeb-Edu sample-10BT revision](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9). Run data is capped at 2,000,000 model tokens and validation at 65,536 tokens, with an 8,192-entry tokenizer. Before the BPE vocabulary exists, tokenizer fitting conservatively caps its UTF-8 input at 2,000,000 bytes; it used that full byte budget across 464 training documents. Each prepared memory-mode cache contains 2,000,000 train tokens from 1,571 documents and 65,536 validation tokens from 48 documents; the none and lexical caches have identical token-content hashes. The byte limit bounds input pieces, not final BPE tokens.

The 18-run FFN-width × lookup matrix ran on Apple Silicon through PyTorch MPS. It uses 320 hidden dimensions, six layers, four heads, and 128-token training sequences. It varies dense FFN width (5,120 / 2,560 / 1,280), the repository's lexical `ngram` memory ([Engram reference](../engram.md), off / on), and seed (17 / 41 / 73). The lexical variant uses a three-token address, an 8,191-row table, and a 32-dimensional latent value. Each arm has a 262,144-target-token ceiling (1,024 steps at the configured effective batch); seven declared contrasts are repeated across three matched seeds (21 descriptive pairings), without preregistered thresholds or inferential claims. The study also reports the fixed alias-retention, held-out-recall, and context-override cards separately as out-of-domain stress tests, not FineWeb quality metrics. Lookup adds 272,672 parameters at each width, so these are not parameter-matched controls. The micro study does not establish transfer to a 1B model.

The external [Pythia-70M-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped) observer scored 62,318 FineWeb-Edu validation targets from 60 documents: mean next-token loss 11.0523 at `step0` (commit `c913ae980de9355947d0bf73f9f10d580eb79301`), 4.0516 at `step10000` (`c890c8f6d8f86c36b2af66c3012a14ef1d35d3f3`), and 4.0215 at `step143000` (`9a7c847e93250c8f24d4b7e7134dbf369e8fc9cb`). The ignored local artifact `artifacts/phase-g/pythia-trajectory.json` records snapshot checksums and evaluation identities. This is an external training-trajectory observation, not a controlled comparison with SparseLab models or tokenizers.

Nemotron-CC was excluded: NVIDIA's [agreement](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2/blob/main/LICENSE.md) permits internal AI-solution training only, bars transfer or distribution of the dataset and any use that would place it under open-source terms, and grants no rights to underlying copyrighted material. FineWeb-Edu is released under the [ODC-BY-1.0 license](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and is subject to [Common Crawl's terms](https://commoncrawl.org/terms-of-use); no blanket rights clearance for individual web documents is claimed. No raw corpus text will be included in the local weight package.

#### Completed endpoints and outcomes

All 18 arms completed 1,024 steps and 262,144 training targets on MPS/fp32. The validation cache contains 65,536 tokens, but the configured six-batch checkpoint evaluation scored 1,536 targets per run; the cache size is not the evaluated target count.

Terminal next-token validation loss by width, memory mode, and seed:

| FFN width | Memory | Seed 17 | Seed 41 | Seed 73 |
|---|---|---:|---:|---:|
| 5,120 (4x) | none | 6.704315 | 6.685425 | 6.686797 |
| 5,120 (4x) | lexical | 6.704510 | 6.681582 | 6.679570 |
| 2,560 (2x) | none | 6.740580 | 6.731210 | 6.699564 |
| 2,560 (2x) | lexical | 6.739928 | 6.746710 | 6.703064 |
| 1,280 (1x) | none | 6.771940 | 6.786267 | 6.731881 |
| 1,280 (1x) | lexical | 6.740157 | 6.763137 | 6.746589 |

Descriptive paired validation-loss deltas (variant minus baseline; lower is better):

| Contrast | Mean delta | Seed 17 | Seed 41 | Seed 73 |
|---|---:|---:|---:|---:|
| lexical − none, 4x | −0.003625 | +0.000195 | −0.003843 | −0.007227 |
| lexical − none, 2x | +0.006116 | −0.000652 | +0.015500 | +0.003500 |
| lexical − none, 1x | −0.013402 | −0.031783 | −0.023131 | +0.014709 |
| 2x − 4x, none | +0.031606 | +0.036265 | +0.045785 | +0.012768 |
| 1x − 4x, none | +0.071184 | +0.067626 | +0.100842 | +0.045084 |
| 2x − 4x, lexical | +0.041347 | +0.035418 | +0.065128 | +0.023494 |
| 1x − 4x, lexical | +0.061407 | +0.035647 | +0.081554 | +0.067019 |

The narrower FFNs had higher validation loss than 4x in every matched seed. The lexical-minus-none differences changed sign across seeds or widths; they are not a consistent benefit, and lexical memory adds 272,672 trainable parameters. All three fixed stress-card scores were 0.0 for every terminal arm. These cards are out-of-domain checks, not FineWeb quality metrics. Three seeds and 1,536 evaluated targets per arm support descriptive observations only, not statistical or scaling-law claims.

Collection and report artifacts are local under `artifacts/phase-g/study-budgeted/`: terminal evidence `reports/architecture-24ec6aa377b2-5c5109b2fb96e1fcd53164ae35f387a6b385f717b99be9f2010f8e8a3fbdf222.json`; static report bundle `research-reports/31131bff510bfde30c1e2a169909870904eb5b469bab949a530a7d5a700abf1e/`. Collection explicitly selected each run's `latest.json`; intermediate checkpoints were not included in the card comparisons.

The optional local reference package is `artifacts/phase-g/micro-reference/`: three 4x/none fp32 safetensors weights for seeds 17, 41, and 73, the tokenizer, portable architecture metadata, source checkpoint manifests, and provenance hashes. The 414,909,288 weight bytes and all package hashes were verified; each source checkpoint passed `sparselab checkpoint verify`. No raw corpus text, token arrays, or optimizer states are included. Publishing remains unavailable because no external destination was supplied.

## Early controlled comparisons

### Completed multi-seed studies

- [Context/Engram study](../context-engram-study.md#execution-results--2026-09-22): 24 endpoints spanning seeds 17/41/73, two exact target budgets, dense/backbone and dense-total comparisons, and collision/address-order diagnostics. Every untouched override endpoint remained 0/8. Added memory capacity and observed bucket collisions are not evidence of a generalization advantage.
- [Domain adaptation](../path-domain-corpus.md#2026-09-22-execution-record): three pretraining/adaptation pairs with frozen supervision, provenance, semantic leakage checks, per-case results, and static-retention measurements. Training-case acquisition improved, but held-out reliability remained poor and retention worsened sharply.

[Independent acceptance](https://github.com/supernovae/tiny-sparse-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/acceptance/scientific_studies_2026_09_22.json) binds the input inventories, endpoint identities, stored responses, paired deltas, and retention observations. These studies retain all declared seeds/endpoints and their negative outcomes; they are not a general model-selection or significance framework.

### Historical controlled dense runs

These recorded runs used the pinned TinyStories revision, the same 8192-token tokenizer, seed, optimizer, 128-token sequence length, and 204,800 target-token budget on MPS. Validation was standalone evaluation over the retained validation split. They predate the current integrity-bound report format; retain them as historical loss observations, not newly verified capability evidence.

| Run | Parameters | Validation loss | Validation perplexity |
|---|---:|---:|---:|
| `scale-micro-3m` | 3,344,064 | 5.1614 | 174.42 |
| `scale-dense-7m` | 6,917,376 | 4.9942 | 147.55 |
| `scale-dense-10m` | 10,244,160 | 4.8790 | 131.49 |
| `scale-dense-25m` | 29,893,120 | 4.5751 | 97.04 |
| `scale-dense-50m` | 50,274,752 | 4.2770 | 72.03 |

These observations suggest lower validation loss at this fixed budget as dense parameter count increases. They do not establish an optimal scaling law, chat ability, or an Engram benefit. Previously published throughput values are withdrawn: the trainer divided one update's tokens by cumulative run time. New measurements use synchronized update duration; rerun controlled pairs before making a performance claim.
The smoke configurations prove CPU wiring only. For a meaningful comparison, train separate unique run IDs with matching source, tokenizer, device, token budget, sequence length, optimizer, and seed. Store the resulting run directory and SQLite metrics; compare observed loss or throughput only at matching recorded budgets. Do not convert unmatched runs into an aggregate quality score.

Dense attention, sliding-window attention, MLA, MoE, and byte memory alter different resource boundaries. Attribute an observed difference only after a controlled ablation; a combined run is a compatibility check, not evidence that its mechanisms compound beneficially.

## Partial recovery declaration example

The [DevMind v4 recovery declaration](https://github.com/supernovae/tiny-sparse-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/devmind-pretrain-v4/recovery.yaml) is intentionally partial: its historical release SHA is expected only. Missing remote inputs/rights and MODEL-0 tokenizer, architecture/budget, runtime and evaluation declarations prevent any claim of recovered v4 model weights.

## Task-scoped ancestry replay

The DevMind coordinator is deliberately task-scoped, **not** a declarative
inheritance DSL. Its committed lineage fixes four historical producers and
gates; each invocation attempts only the named next stage, re-verifies parents,
and preserves the first failure without automatic retry:

The original invocation is retained with the
[ancestry replay record](https://github.com/supernovae/tiny-sparse-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/experiments/research/history/devmind-pretrain-v4/ancestry-replay-report.md).
It is a historical recovery procedure, not a general native lab workflow.
Current `corpus_release` recovery steps support
[declared snapshot inheritance](lifecycle-recovery.md#declared-snapshot-inheritance)
from authenticated parent replay evidence. Use that native contract for new
workflows; it does not turn this historical coordinator into a general runner.

The coordinator enforces pre-launch byte/inode reservations plus a 25% free
floor; it does not itself monitor live storage growth. Execution must provide
that monitoring. The historical coordinator does not prove historical release
recovery. Its proposed compact durable identity metadata closure was a follow-up
proposal, not a guarantee of the recorded execution; this record does not
establish the proposal's current implementation status. Current inheritance
behavior is defined by the native contract linked above.
