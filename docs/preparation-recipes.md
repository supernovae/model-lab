# Portable preparation recipes

`experiment recipe preparation inspect RECIPE --bindings LOCAL --json` compiles
an external preparation recipe without running it. This is a separate
`portable_preparation_version: 1` contract; the existing matrix and prepared-input
workflow descriptor formats, identities and commands are unchanged.

The descriptor has `id`, `entrypoint`, `contracts` and `files`. Contracts must be
`{campaign: 1, run_config: 2, tokenizer_config: 1}`. Each file has a canonical
recipe-root-relative `path`, SHA-256 of its bytes, and a role: `campaign`,
`run_config`, `tokenizer_config`, `source_lock` or `metadata`. The entrypoint is the sole Campaign
member. All declaration references use the inventory root, never CWD or Git
fallback. The existing 64-document, 1 MiB per-document and 8 MiB total limits apply.

The entrypoint is a native CampaignPlan restricted to `tokenizer_train`,
`data_prepare`, native `dataset_snapshot` and external tokenizer/snapshot
`artifact_reference` stages. CampaignPlan
remains the sole graph validator/order authority; `compile_preparation` produces
its existing immutable operation projection. No second DAG or temporary YAML is
created. Snapshot edges use the existing typed preparation ports; acquisition
uses the native Campaign snapshot stage with the separate admission below. Other
stage kinds are refused before their validators can import computation. Two supported compositions are:

- `tokenizer_train → data_prepare` with an unresolved future tokenizer output.
- `artifact_reference(tokenizer) → data_prepare` with a declared, unverified identity.

The separate current local document has `artifacts`
(stage ID to location for every external tokenizer producer), and `config_paths`
(config reference to a mapping of native field names to locations). For example,
a training graph's bindings can be:

```json
{
  "artifacts": {},
  "config_paths": {
    "tokenizer.json": {"output_dir": "payload/tok", "dataset.cache_dir": "payload/cache"},
    "run.json": {"tokenizer.path": "payload/tok/tokenizer.json", "dataset.cache_dir": "payload/cache", "logging.root_dir": "payload/runs"}
  }
}
```

Bindings must name exactly the compiled config references and every present
native typed Path field. Relative locations anchor to the local binding document;
canonical absolute locations are allowed. Traversal and symlinks are refused.
Native declaration path placeholders must also be canonical relative paths.
The tokenizer consumer location must match its compiled producer's output or
external binding. Opaque metadata fields named `path` are not interpreted.

Inspection snapshots and verifies inventoried declaration bytes, validates native
configs, and reports operations, typed edges, config byte digests and local paths.
It neither opens payloads nor claims their availability. Future outputs have no
fabricated content identity. Runtime capability is unknown, committed provenance
is unverified, and execution is unavailable. The separate recipe digest uses the
`klab-portable-preparation-v1` domain and sorted inventory; local bindings
have their own byte digest. No historical hash algorithm or seed acceptance changes.

## Iteration and operation boundaries

Read-only iteration can now reconstruct a published prepared-input workflow's
intent from its authenticated availability sidecar and revalidated descriptor,
bindings and inventory. Valid context no longer fails because the native entrypoint
omits descriptor-supplied artifacts. It instead reports the explicit
`WORKFLOW_EXECUTION_UNAVAILABLE` gate. Invalid declarations and changed identities
still report errors. The profile-presence guard in CLI/Campaign dispatch remains.
The informational `execution` annotation does not authorize anything.

Inventory integrity is not trusted authorship. Acquisition, preparation, lock
publication, provenance verification, model runtime acceptance and execution
admission remain distinct operations. The explicit preparation route below uses
CampaignEngine for a bounded local CPU subset. Preparation inspection itself never
dispatches model work. The separate [prepared-input workflow](workflow-recipes.md)
now supports its documented CPU execution/evaluation/report subset with explicit
admission; broader composition remains unavailable. Neither inspection nor lock
publication grants that execution context.

## Proof and change-impact map

| Boundary | Regression coverage |
| --- | --- |
| Descriptor/context → native graph | Train/reuse graphs equal native compilation, ordering, references, role/contract checks, cycle/missing/wrong producer refusals |
| Bound config → native adapter | Typed request construction and native pack/verifier spies; producer-location mismatch refusal; existing CampaignEngine dispatch |
| Integrity and locations | Relocation, sorted inventory, immutable snapshots, changed bytes, binding overlap, traversal/symlink, opaque metadata |
| Inspection → effects | Fresh-process CLI/help import, payload, credentials, write, network/process and runtime fences |
| Iteration → admission | Explicit gate even without other missing readiness evidence; malformed/changed source retains errors; original CLI/shared Campaign refusal tests retained |
| Compatibility | Existing recipe v1 and campaign digest goldens, seed coercion, native iteration and verifier-authority regressions, normal safe/fast/tiny CPU lab-loop CI |

These are bounded software contracts, not actual TinyStories portability or GPU
qualification.

## Guard coverage after review

The preparation-only operation requirement, exact config-reference binding set,
external-tokenizer-only profile and unsupported native path allowlist have direct
loader regressions. The allowlist case uses a valid native `local_token_mixture`
config with all required paths and provenance fields; it fails at the portable
profile boundary without reading mixture payloads. Other cases validate their
native Campaign schema first and use config-read spies where appropriate.

The workflow `declaration_hashes`, declaration source and declaration root checks
are redundant consistency checks alongside the authenticated descriptor/binding
and inventory identities. Direct helper tests vary only those redundant fields,
after a successful normal context load. These tests do not claim altered sidecars
pass `open_lock`, nor grant committed provenance or execution authority.

The private `_config` helper explicitly refuses missing role metadata or config
bindings. Normal loading already establishes both through inventory/reference and
exact binding-set checks. Direct helper regressions protect those diagnostics and
prove undeclared references are still refused first, before config reads. This is
defence in depth for future callers/reordering, not a new admission mechanism.

## Explicit local CPU preparation

The owner-reviewed execution subset adds:

```sh
klab experiment recipe preparation apply RECIPE --bindings LOCAL \
  --admission ADMISSION.json --work-dir /absolute/work/root --json
klab experiment recipe preparation resume RECIPE --bindings LOCAL \
  --admission ADMISSION.json --work-dir /absolute/work/root --json
```

`inspect` still performs no execution admission or payload verification. `apply`
and `resume` require an explicit local admission document with all these fields:

```json
{
  "reviewed_commit": "<full 40-character recipe Git commit>",
  "recipe_sha256": "<recipe digest from inspection>",
  "binding_sha256": "<binding byte digest from inspection>",
  "cpu_python": "/absolute/selected/environment/bin/python",
  "max_documents": 32,
  "max_tokens": 1024,
  "max_vocab_size": 512,
  "min_free_bytes": 268435456,
  "min_free_inodes": 128
}
```

Digest placeholders above must be replaced with real hexadecimal identities.
The selected Python must be the interpreter running the command; the command
never provisions or launches another environment. When selecting `reviewed_commit`,
the recipe descriptor and every inventoried declaration must be clean and committed in their
own repository: `reviewed_commit` must equal the recipe repository's current
HEAD. Any later commit, including an unrelated change elsewhere in that repository,
invalidates the admission. Selecting that commit asserts that the owner reviewed the recipe.
Git proves committed bytes, **not trusted authorship** or review. No uncommitted
or model-execution override is accepted on this route. Legacy Campaign overrides
and readers retain their existing behavior for legacy records. Alternatively, select
exact descriptor content with `reviewed_descriptor_sha256` and omit `reviewed_commit`;
see [content admission](portable-recipes.md#explicit-content-admission-outside-git).
All other admission limits and artifact checks remain required.

For the synthetic route, the total train plus
validation document and token limits of each native config must fit the explicit
admission caps; tokenizer document and vocabulary limits must also fit. These are
admission ceilings, never rewritten scientific settings. Native tokenizer storage,
packing storage, host checks and artifact verifiers still run. The additional
free-space/inode and permission check is repeated before dispatch. Synthetic preparation performs no corpus acquisition, remote source, model
training, GPU allocation or model-runtime qualification. RunConfig runtime/model fields retain their native meaning;
this CPU preparation admission does not qualify that later model workload.

All output paths must lie below the explicit absolute work root outside Git.
Tokenizer output and cache directories must not overlap or occupy native operational
store directories. Native preparation receipt/scratch publication uses that same
work root; process location settings are restored after dispatch. External tokenizers
remain explicit bound inputs, verified through the native artifact verifier.

The engine receives the inventoried native plan and bound native configs directly.
It retains the sole existing scheduler, stage-input hashes, preparation adapters,
artifact verification and immutable receipts. One context supplies availability,
requests, pre/post-dispatch provenance and replay. Receipts record recipe/binding
identity, declaration closure and commit, locations, admission limits, installed lab
source digest, Python identity and dependency versions as operational provenance.
Scientific declaration/stage/artifact hash algorithms are unchanged. An existing
store refuses a changed context (including a differently formatted bindings file),
even when a caller explicitly selects its new digest. Use a fresh work root for a
new context; cross-binding resume and migration are future work.

Native committed receipts recover after a crash without calling the adapter again.
Completed outputs are reverified, including before receipt commit; changed outputs
are refused. An interruption without a committed receipt retains its native attempt
and interruption observation and refuses automatic rerun. Existing unreceipted
outputs are never overwritten or adopted. This conservative refusal is deliberate:
partial preparation recovery needs a separate validated contract. It does not remove
legacy cancellation, recovery or resume behavior.

| Execution boundary | Focused regression evidence |
| --- | --- |
| Selected context → admission | Wrong commit/digests/interpreter, dirty index, finite caps, environment drift, no work-store creation on refusal |
| Native plan → two tiny graphs | Train then pack and verified external tokenizer then pack both complete using native adapters |
| Dispatch → receipt | Binding change immediately before adapter or after dispatch, output change before commit, immutable receipt checks |
| Receipt → replay/recovery | No redispatch on replay, crash after immutable receipt before state save, cross-binding and legacy-route refusal |
| Interrupted work → next action | Preserved interrupted attempt, no automatic overwrite/rerun, collision retains existing bytes |
| CLI → effects | Public lightweight entry route, explicit local work-root precedence, fresh-process apply/resume help without importing execution or heavy optional packages |

The profiled lifecycle tests are marked slow and explicitly included in the existing
`lab-loop` CI job; metadata and refusal checks remain in `fast`. No new workflow or
manual workload gate is introduced.

These tiny synthetic checks validate the wiring. They do not establish
portability for real datasets, accelerator execution or model quality.

## Verified local dataset snapshots

Portable preparation can consume an already materialized native dataset snapshot.
This version-2 consumption profile does not authorize acquisition. Use an
`artifact_reference` of kind `dataset_snapshot` (corpus scope),
distinct from the existing corpus/Forge `source_snapshot` artifact:

```yaml
- id: input
  kind: artifact_reference
  scope: corpus
  artifact:
    kind: dataset_snapshot
    version: 1
    producer: native dataset snapshot
    identifier: <source lock_sha256>
    sha256: <snapshot manifest_sha256>
    path: snapshot/manifest.json
- id: tokenizer
  kind: tokenizer_train
  scope: tokenizer
  requires: [input]
  snapshot: input
  config: tokenizer.json
- id: packed
  kind: data_prepare
  scope: model
  requires: [input, tokenizer]
  snapshot: input
  tokenizer: tokenizer
  config: run.json
```

These are native Campaign stages; the compiled preparation projection owns their
ports. The descriptor remains version 1; bindings use the single current model. Inventory the Campaign and
config files as usual. Bind `artifacts.input` to the local `manifest.json`.
In each config, declare `dataset.source: snapshot` and canonical relative
`dataset.source_manifest_path`, `dataset.train_path`, and
`dataset.validation_path`; bind these exact typed fields in that config's
`config_paths`. The manifest must be the input producer's location, and splits
must be its sibling `train.jsonl` and `validation.jsonl`. Output locations remain
under the explicit work root and may not overlap snapshot inputs in either
direction. Snapshot inputs also cannot overlap the native operational namespaces
`campaigns`, `scratch`, `execution`, or `ownership`. Retained inputs in a distinct
work-root directory such as `inputs` are allowed; admission does not initialize
operational storage.

### Snapshot capability limits

The current admission requires both explicit snapshot limits when the compiled
graph consumes or acquires snapshots. Synthetic-only graphs may omit both:

```json
{
  "reviewed_commit": "<full recipe commit SHA>",
  "recipe_sha256": "<preparation inspect recipe_sha256>",
  "binding_sha256": "<preparation inspect binding_sha256>",
  "cpu_python": "/absolute/provisioned/bin/python",
  "max_documents": 32,
  "max_tokens": 1024,
  "max_vocab_size": 512,
  "min_free_bytes": 1073741824,
  "min_free_inodes": 128,
  "max_snapshot_bytes": 1048576,
  "max_snapshot_records": 40
}
```

All numeric limits are strict positive integers (vocabulary minimum 260).
Snapshot ceilings are cumulative across declared snapshot references, counting
repeated references again. `max_snapshot_bytes` counts every regular file in
each flat snapshot directory, including manifest, events and exclusions, before
parsing or cold verification. Nested directories, symlinks and nonregular files
are refused. `max_snapshot_records` bounds the sum of **declared per-split
`max_source_records` ceilings**, not just retained document counts. For example,
a two-split source declaring 20 records per split requires at least 40. Existing
document/token/vocabulary and free-storage limits still apply. These are input
and configuration admission checks, not kernel quotas; use the native monitor
for wall time, process-tree RSS, scratch and workspace growth.

The native verifier authenticates the lock/manifest identities, split contents,
exclusions and event history. It checks each consumer's revision, license,
dataset configuration and split locations. The embedded source lock retains
source attribution and licensing provenance; successful verification does not
grant rights or establish redistributability. This profile requires
`klab-dataset-snapshot-v1`; legacy imported snapshots remain readable through
their existing native readers but are not newly admitted here.

Use the existing public commands with this admission:

```sh
klab experiment recipe preparation inspect recipe.json --bindings local.json --json
klab experiment recipe preparation apply recipe.json --bindings local.json --admission admission.json --work-dir /absolute/work --json
klab experiment recipe preparation resume recipe.json --bindings local.json --admission admission.json --work-dir /absolute/work --json
```

Inspection/help never read snapshot payloads or launch preparation. Apply/resume
revalidate input integrity, admission, committed inventory, interpreter/source
identity and receipt bindings. Missing or changed payloads refuse without network
fallback. Completed receipts replay without retraining tokenizers or repacking.
The artifact memo/proof key includes sibling payloads, not just manifest bytes.
Native snapshot/config/tokenizer/prepared hashes and legacy seed coercion are
unchanged. No model execution capability is added.

The historical [`test_portable_snapshot_preparation.py`](https://github.com/supernovae/model-lab/blob/17446cfa9dc4fd8a528206e00b58de88527367c5/tests/test_portable_snapshot_preparation.py) exercised the real native
snapshot/tokenizer/packing/verifier and public CLI. Snapshot fixture construction
substitutes only Hub metadata/transport with local rows; it is not real-data
acquisition evidence. See [portable experiment ownership](portable-experiment-migration.md)
for where recipes and Lab code belong.

Current [`test_verifier_authority_current.py`](../tests/test_verifier_authority_current.py)
checks registered authority closures and exclusion audits. The historical
[`test_snapshot_warm_reuse.py`](https://github.com/supernovae/model-lab/blob/17446cfa9dc4fd8a528206e00b58de88527367c5/tests/test_snapshot_warm_reuse.py), [`test_snapshot_authority.py`](https://github.com/supernovae/model-lab/blob/17446cfa9dc4fd8a528206e00b58de88527367c5/tests/test_snapshot_authority.py) and
[`test_verification_reuse.py`](https://github.com/supernovae/model-lab/blob/17446cfa9dc4fd8a528206e00b58de88527367c5/tests/test_verification_reuse.py) documented additional snapshot and prepared-inventory
reuse coverage before test consolidation; these retired files are evidence of
that earlier coverage, not tests run by current CI. The new artifact
branch uses cold native verification and does not add a persistent authority
kind. Existing dispatch-owner exclusion pins were explicitly re-audited for
this addition; historical content hashes and records are not rewritten.

Admission boundary tests cover both output/input overlap directions before work
creation, and preserve existing retained input bytes on refusal. They exercise
both source/compiled-port mismatch directions (including omission of snapshot limits),
all three snapshot locations, and the per-split record ceiling at 39/40. Direct
artifact-domain tests verify the envelope and refuse a genuinely verified legacy
import without relying on verifier-authority pins. Malformed manifest metadata
gets a descriptive admission error; unrelated internal KeyErrors remain visible.

## Bounded Hub acquisition

An inventoried native `source.lock.json` with role `source_lock` may now feed a
native `dataset_snapshot` stage. Its `lock`, `output` and `cache_dir` placeholders
are canonical recipe-relative references. `lock` resolves inside the inventory;
local output/cache locations come exclusively from the current bindings:

```yaml
artifacts: {}
acquisitions:
  input:
    output: /absolute/work/snapshot
    cache_dir: /absolute/work/acquisition-cache
config_paths: # the same exact typed config bindings as above
  # ...
```

Use native stage `kind: dataset_snapshot`, `id: input`, `scope: corpus`,
`lock: source.lock.json`, `output: snapshot`, `cache_dir: acquisition-cache`.
Consumers declare `snapshot: input` and `requires: [input]` as above. CampaignPlan
still owns ordering, and CampaignEngine still owns snapshot journals, immutable
receipts and recovery. No Forge Project, release workflow, plugin registry or
second graph is introduced. `stage.resume` must remain false; use public
`experiment recipe preparation resume` for explicit continuation.

The current preparation admission requires `acquisitions`, keyed exactly by
snapshot acquisition stage ID, when the graph acquires snapshots. A graph without
acquisition stages refuses nonempty grants. Each entry requires:

| Field | Contract |
| --- | --- |
| `acquisition_version`, `provider`, `decoder_profile` | `1`, `huggingface`, `scalar-parquet-v1` |
| `repo_id`, `revision`, `lock_sha256` | Exact native lock authority; immutable 40-hex revision and 64-hex lock digest |
| `allow_network` | Explicit `true`; inspection never uses this authority |
| `max_source_body_bytes`, `max_transfers`, `max_retries_per_shard` | Full pinned file size reserved durably before a request, including each retry |
| `max_metadata_body_bytes`, `max_requests` | Separate response-body reservations for redirects/errors and request count |
| `max_wall_seconds` | Persistent deadline from ledger initialization, not reset on resume |
| `max_cache_bytes`, `max_cache_files` | Private payload reservation ceilings, including retained partial files |
| `max_scanned_rows` | Cumulative conservative row reservations before decoding, including replayed prefixes |
| `max_verification_bytes` | Cumulative logical input-byte reservations for file authentication and each native snapshot verification call |
| `max_observed_owned_bytes`, `max_observed_owned_inodes` | Aggregate work-root observations with stop-on-exceed behavior, **not a hard filesystem quota** |

Integers are strict and positive except retries, which may be zero. Optional
`require_hard_disk_bytes`, `require_hard_decoder_memory_bytes` and
`require_expanded_bytes` are unavailable: setting any refuses before network or
output. Parquet batch size is bounded by rows; that is not a bound on decoded
bytes or peak decoder memory. Directory observations can miss transient peaks.
Verification reservations count logical file lengths, not physical I/O bytes.

The source lock must already contain reviewed ordered file inventories and
loader options. Lock-time Hub discovery is outside this invocation. Only pinned
whole Parquet files with scalar/text columns and supported scalar `Value`
features are admitted; projection, filters, feature casts, media/external-file
features, malformed-file skipping and unknown loader options are refused.
Embedded feature metadata is checked too. Native selected-row ordering,
deduplication, normalization, rights provenance and snapshot hashes are retained.
Git blob IDs authenticate the Git blob header plus content; LFS SHA-256 values
authenticate content directly. No global SDK cache or SDK streaming is used.
Requests use HTTPS with explicit Hub delivery-host redirects; credentials are
removed across origins, and transfer responses must have exact Content-Length
and identity encoding.

Operational state lives at `WORK/execution/acquisition/<binding>/transport.sqlite`
with a `guard`, optional `CANCEL`, and diagnostic `receipt.json`. The cache has
an exclusive `.owner` binding. Missing ledgers with existing payloads/journals,
foreign caches and symlinks fail closed. The operation holds a native file lock;
this is not a sandbox against malicious concurrent filesystem replacement.
Native verifier databases and process scratch stay under the accounted work
root. Snapshot record/byte limits cover declared inputs cumulatively; source
journal limits remain distinct from downloaded cache allowances.

Receipts distinguish `source_charged`/`metadata_charged` reservations from observed
`source_actual`/`metadata_actual`. Observations may be incomplete after a crash;
reservations are never refunded. `scanned_rows` is charged work and
`scanned_rows_observed` is decoded rows observed. No wire-byte claim is made.
Published files are authenticated on reuse, and repeated verification consumes
the original allowance. Exhausting an allowance or the persistent deadline can
therefore refuse replay. Cancellation retains both native journals and spent
reservations; remove the owned `CANCEL` only when explicitly resuming. Successful
snapshot publication followed by interrupted Campaign receipt commit reuses the
published snapshot without another transfer.

Version-1/2 admission and binding serialization, native source/snapshot/config
hash algorithms, legacy seeds and historical readers remain unchanged. Direct
native acquisition retains its existing loader capabilities. This new admission
selects a deliberately narrower supported decoder contract.

| Change boundary | Focused evidence |
| --- | --- |
| HTTP → authenticated private file | `test_bounded_acquisition`: loopback framing, typed Git/LFS hashes, auth stripping, redirect/error accounting, pre-request full-file/cache refusals |
| Persistent admission → resume | Crash reservations, retry/request/deadline/cancel ceilings, missing/foreign ledger/cache, verification and scan exhaustion |
| Recipe → native Campaign | `test_portable_acquisition`: inspect fences, authority/overlap/version refusals, public apply/resume, cancellation, interrupted publication commit |
| Native scientific identity | Bounded HTTP and historical local Parquet decoder produce equal native manifests on the same locked bytes |
| Legacy component boundaries | Existing preparation recipe/local-snapshot, registered verifier authority, source lifecycle and typed-plan regressions; normal safe/fast/lab-loop CI |

This is fixture wiring and compatibility evidence. Real TinyStories acquisition
needs its own immutable inventory, network/cache/storage admission and decoder
resource qualification. Its first whole train+validation files require roughly
259 MB of transfer/cache capacity; the full inventory is roughly 1 GB. The prior
256 MiB snapshot-work allowance is not a download budget. Paired
training/comparison, continuation, evaluation and GPU qualification require
their own verification and are not established by these tests.

### Acquisition boundary tests

[Historical transport guard tests](https://github.com/supernovae/model-lab/blob/17446cfa9dc4fd8a528206e00b58de88527367c5/tests/test_acquisition_guards.py) covered HTTP and
credential boundaries, explicit network permission, cache/storage integrity,
decoder limits and cumulative verification accounting.
[Historical authority tests](https://github.com/supernovae/model-lab/blob/17446cfa9dc4fd8a528206e00b58de88527367c5/tests/test_acquisition_authority.py) covered admission paths/stages
and exclusive invocation ownership. Those HTTP fixtures used an in-memory opener
and fake token, with no external network or credential-store access. These files
were retired during test consolidation. Current
[corpus acquisition tests](../tests/test_corpus_acquisition.py) exercise pinned
source selection, bounded reads, offline reuse and mutation refusal;
[dataset source tests](../tests/test_dataset_sources.py) cover snapshot bindings,
resource caps and interrupted replay. These links do not claim one-for-one
coverage of every retired guard.

Known limitation: each 64 KiB read and decoded batch calls `observe()`, which walks
the work root at O(entries) cost.


## Current contract and migration

There is one `PreparationAdmission` parser/writer and one `PreparationBindings`
parser/writer. Their checked-in schemas are
[`preparation-admission.schema.json`](../schemas/preparation-admission.schema.json)
and [`preparation-bindings.schema.json`](../schemas/preparation-bindings.schema.json).
Graph capabilities, rather than a version selector, determine which limits and
acquisition authorities are required. The nested native acquisition authority
still has its single `acquisition_version: 1` transport contract.

Documents containing `preparation_admission_version` (formerly 1/2/3) or
`preparation_bindings_version` (formerly 1/2) fail with regeneration guidance
before execution. Regenerate bindings for the current typed config paths and
acquisition stages, inspect the committed recipe with those bindings, then author
a fresh admission using the returned recipe/binding digests and reviewed commit.
Do not just strip a marker from old authorization: review capability ceilings,
network grants and locations again. Preserve archived declarations, receipts,
failed attempts and payloads unchanged; the current executor does not resume
obsolete admissions or rewrite their identities. No compatibility execution
branch is retained. Native scientific hash algorithms and seed acceptance are
unchanged.

The following table records the **historical regression layout** before test
consolidation. Its pinned files are not part of current CI. Current connected
preparation coverage lives in [the preparation sequence tests](../tests/test_preparation_sequence.py),
including native phase supervision, declaration refusal, reviewed admission,
split freezing and inventory completion reuse.

| Boundary | Historical regression | Original coverage |
| --- | --- | --- |
| Current models/schema | [`test_preparation_contract.py`](https://github.com/supernovae/model-lab/blob/17446cfa9dc4fd8a528206e00b58de88527367c5/tests/test_preparation_contract.py) | One round-trip/schema per document; obsolete input refuses before context/compilation; paired strict snapshot ceilings |
| Typed graph/local binding | [`test_preparation_recipe.py`](https://github.com/supernovae/model-lab/blob/17446cfa9dc4fd8a528206e00b58de88527367c5/tests/test_preparation_recipe.py) | Exact paths/ports, metadata-only inspection, no interpretation of opaque plugin paths |
| Synthetic execution | [`test_preparation_execution.py`](https://github.com/supernovae/model-lab/blob/17446cfa9dc4fd8a528206e00b58de88527367c5/tests/test_preparation_execution.py) | Native CPU prep, bounds, provenance, replay, no-clobber and source-change refusal |
| Snapshot consumption | [`test_portable_snapshot_preparation.py`](https://github.com/supernovae/model-lab/blob/17446cfa9dc4fd8a528206e00b58de88527367c5/tests/test_portable_snapshot_preparation.py) | Required caps, cold integrity, exact split/source bindings and replay |
| Acquisition | [`test_portable_acquisition.py`](https://github.com/supernovae/model-lab/blob/17446cfa9dc4fd8a528206e00b58de88527367c5/tests/test_portable_acquisition.py), [`test_acquisition_authority.py`](https://github.com/supernovae/model-lab/blob/17446cfa9dc4fd8a528206e00b58de88527367c5/tests/test_acquisition_authority.py), [`test_acquisition_guards.py`](https://github.com/supernovae/model-lab/blob/17446cfa9dc4fd8a528206e00b58de88527367c5/tests/test_acquisition_guards.py) | Explicit network authority, lock identity, reservations, cancellation and recovery |

This consolidation changes operational admission/binding identity, not computation,
Campaign ordering or historical content identity. It proves fixture-scale contract
behavior only. Recipe-specific run/replay/repeat plans belong with the external
experiment repository. Full acceptance still requires independently bounded
snapshot-backed controlled pairs, acquisition/reuse, interrupted continuation and
current CPU/GPU runtime evidence; synthetic fixture success does not establish them.
