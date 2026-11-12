# Declare, pin and reuse datasets

Commands below use the [activated, provisioned environment](../README.md#first-run).

A dataset is an input to an experiment, not a training function. A source
declaration describes where text comes from and how it is selected; a lock pins
the source; a snapshot retains the selected records and exclusion accounting.
RunConfig references that immutable snapshot through `dataset.source: snapshot`.
The [TinyStories walkthrough](tinystories-microlab.md) is one reference declaration
of this general pattern.

## Source kinds

`dataset.source` names how data is read, never which dataset it is. A named
dataset is a value in your recipe, so a new one needs no Lab change.

| `dataset.source` | Reads | Required fields |
| --- | --- | --- |
| `local_text` | your JSONL files, one `{"text": ...}` document per line | `train_path`, `validation_path`, `license` |
| `local_chat` | your JSONL conversations | `train_path`, `validation_path`, `license` |
| `local_token_mixture` | a verified [token mixture](token-store.md#mixture-output) | mixture declaration and output paths |
| `snapshot` | a pinned, retained [snapshot](#source-declaration-and-lock) | `revision`, `license`, split and manifest paths |
| `huggingface` | a Hugging Face dataset, streamed at a pinned revision | `dataset_id`, `revision`, `license` |
| `synthetic` | a built-in offline generator | none (`synthetic_task` picks one) |

A `huggingface` source also accepts `dataset_config`, `text_field` (default
`text`), `train_split` (default `train`) and `validation_split`. Without a
validation split, validation is the documents that follow the first
`train_max_documents` documents of the training split. Streaming reads records
in source order and can cut the final document at the token budget; use a
`snapshot` when you need whole documents, deduplication and retained bytes.

```yaml
dataset:
  source: huggingface
  dataset_id: HuggingFaceFW/fineweb-edu
  dataset_config: sample-10BT
  revision: 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
  license: ODC-By-1.0; Common Crawl Terms of Use
```

`synthetic_task` selects `stories` (the default smoke text),
`instruction_reference`, `chat_recall`, `engram_recall` or `withheld_facts`.
These are fixtures for wiring checks and the probe battery, not corpora.

The earlier named sources (`tinystories`, `fineweb_edu`, `cosmopedia`,
`local_stories`, and the four generators as source names) are no longer
accepted. Rewrite such a config with the kinds above; its prepared data and
run identities are new.

## Source declaration and lock

Author a source declaration in your own recipe directory. The optional
[Workshop example](tinystories-microlab.md) supplies one. Declare the repository,
exact commit revision, configuration, logical split mapping and text field for
your selected standard Hub text dataset.
Keep attribution and terms explicit. No callback or repository Python loader is
part of the interface. Unsupported formats fail rather than executing code.

`selection.mode: bounded` names per-split retained-document targets.
`selection.mode: exhaustion` consumes the declared source splits to their end.
Both have resource limits. Reaching a safety limit is not evidence of complete
source coverage. Source revisions must be immutable; moving names such as `main`
are rejected. An explicit exact-text dedup policy can process validation first,
remove within-split repeats and exclude training overlap. Selection and policy
are part of scientific identity, not invisible cleaning defaults.

```sh
klab data lock source.yaml --output source.lock.json --json
klab data snapshot source.lock.json --output snapshot --json
```

Use new output paths. A snapshot retains its source lock, selected records,
source positions, exclusion inventory, stop reasons and content digests.
Do not edit it to add data. Increasing bounds, changing the source revision or
selecting another split creates a new lock/snapshot and new downstream identities.
Model-width, seed or training-budget comparisons can reuse the unchanged inputs.

For an interrupted acquisition, repeat the snapshot command with `--resume`,
using the original lock and output path. Without that flag, an existing adjacent
work path refuses with an explicit resume diagnostic and leaves its journal
untouched. Resume still refuses a busy journal or a different lock identity.
If initialization failed before a regular journal became observable, inspect
retained evidence and the original failure instead: the diagnostic cannot promise
resume is available. An initializer may still be active. A missing journal is
not permission to delete the work path or clear its guard.
Its adjacent work journal is authenticated before reuse; immutable sources can
be replayed to verify the completed prefix when seeking is unavailable. Atomic
publication prevents a partial snapshot from masquerading as complete input.
Snapshot events and the deduplication index commit together with SQLite FULL
synchronization, in groups of at most 256 records or 8 MiB of estimated growth.
A 250 ms cooperative check also ends a group between records; it cannot interrupt
an upstream read or a filesystem operation. A permitted record larger than the
group byte allowance is admitted separately and committed alone. Only committed
events are recovery progress: interruption discards the pending group, which is
re-read on resume. Split boundaries commit before export.
Capacity scans admit conservative growth allowances and check live journal usage
before and after commits. Export consumes byte allowances and retains file fsync,
cold verification and atomic publication. These are observed safety stops, not
filesystem quotas or reservations against other writers; concurrent disk usage
can change between observations. SQLite readers may still briefly contend with
commit or cache-spill locks. The batching does not change source/scan transport
accounting, selection, deduplication, snapshot identity, or publication contents.
Disk-backed indexes bound deduplication memory; resource limits and storage
headroom still need admission on the actual filesystems. Hub caches may consume
more than the selected text, so small document bounds are not a download-size
guarantee.
The record-byte ceiling is checked after the upstream loader decodes a record;
it does not cap that loader's transient allocations. Pilot the actual format and
resource policy before attempting a full split.

Corpus Forge's explicit files and checksummed shards remain available. Use
`kind: forge_files` to select verified Forge snapshot members with declared
per-split file bindings. This route serves sources needing explicit file control;
it does not remove release/export requirements from a Forge training contract.

Bounded Forge acquisition receipts record per-shard `resources.expanded_bytes`:
the JSONL bytes returned by the decompressor for the consumed prefix, including
any rejected overlong line prefix. This is not the full expanded shard size.
Parquet reports this reading as unavailable (`null`); declaring a required
`max_decompressed_bytes` cap for Parquet fails before acquisition. Raw bounded
Git blobs also have no expansion reading. `peak_staging_bytes` counts logical
regular-file bytes and `peak_staging_inodes` counts file/directory entries,
including the private source staging root, through extraction. These readings
exclude caches, the transport ledger, sealed-manifest overhead, filesystem block
allocation and parser memory. Missing required staging readings fail closed.
The transport ledger retains interrupted transfer readings across resume.
Historical receipts without these fields remain unchanged: missing measurements
are never inferred, backfilled or grounds for reacquisition.
Wikimedia dump success receipts report the same staging peaks and XML bytes
returned to the parser (which can include parser read-ahead beyond selected
pages). Wikimedia retains operational per-attempt readings in the source
snapshot parent's `.resource-receipts/attempt-*.json`, outside disposable
staging. Each receipt binds the source declaration and staging name;
failed/interrupted attempts survive cleanup and later attempts receive separate
receipts. A `started` receipt after an abrupt process kill is unfinished evidence,
not proof of safe shutdown. Operational journals do not change snapshot identity;
new Wikimedia success measurements join its existing retrieval identity, while
historical receipts and verified reuse remain unchanged.
Bounded HF acquisition without a transport ledger uses the same retained
operational receipts, one per shard attempt, additionally bound to the shard
path. Budgeted HF/Git keep their readings in the transport ledger.

An immutable Forge release can serve several experiments. Export it once for a
particular view, base configuration and vocabulary, then retain the export's
exact `dataset` and tokenizer declarations in each consumer configuration. Run
settings such as seed and optimizer budget may differ. `klab corpus
verify-export RUN.yaml` cold-verifies that configuration's release/export
closure and reports its content binding without fitting, preparing or copying
data. The declared `dataset.cache_dir` remains the export's prepared directory;
an independent cache override is not authenticated reuse. `experiment
bind-inputs` can authenticate compatible prepared inputs without a dataset
copy. `stage --prepared-inputs` has a different, full-config binding and copies
assets into a new stage. Token-mixture datasets authenticate their own release,
family inventory and tokenizer closure instead of claiming an LM export.

Prospective Forge releases may opt in to `release.normalizer:
normalizer-structure-v3`. The older normalizer remains the absent-field default
and keeps historical release identities. The opt-in cleaner records retained
line spans, removed-span hashes/reasons, structural blocks and ambiguity flags
with each normalized document. It excludes recognized metadata-only front
matter from LM views while retaining the source row and rights evidence in the
release; meaningful YAML configuration, headings, lists, links, fenced code,
indentation and equations remain text. Only explicit Gutenberg production
credits and marker-delimited wrappers are removed, with their original bytes
replayable through the pinned snapshot. A changed cleaning rule needs a new
normalizer identity and release; never reinterpret an old release in place.

## Preparation and exposure

Run/tokenizer configurations name snapshot `train_path`, `validation_path`,
`source_manifest_path`, revision and license. Native verification matches those
paths and identities before fitting or preparing. Tokenizer fitting remains
train-only; its input byte/document bounds are separate from training exposure.
Whole-document packing fails on insufficient token bounds rather than silently
truncating selected text. Preparation uses disk-backed resumable chunks.

```sh
klab tokenizer train tokenizer.yaml
PREPARED=$(klab data prepare run.yaml)
klab data coverage --prepared-root "$PREPARED" --config run.yaml --json
klab data budget --prepared-root "$PREPARED" --config run.yaml \
  --passes 1 --no-require-full --output one-pass.yaml --json
```

### Share a tokenizer across snapshot datasets

For a controlled lab comparison, each run may retain the same tokenizer while
its top-level `dataset` selects a different training snapshot. Add an explicit
`tokenizer.origin` to each run configuration:

```yaml
tokenizer:
  path: ../shared-tokenizer/tokenizer.json
  origin:
    sha256: <64-hex tokenizer file digest>
    manifest_sha256: <64-hex tokenizer_manifest.json file digest>
    dataset:
      # Copy the complete dataset declaration from the original tokenizer fit.
      source: snapshot
      revision: <original pinned Hub revision>
      dataset_config: default
      license: <original declared license>
      source_manifest_path: ../origin-snapshot/manifest.json
      train_path: ../origin-snapshot/train.jsonl
      validation_path: ../origin-snapshot/validation.jsonl
      cache_dir: ../origin-cache
      train_max_documents: 500000
      validation_max_documents: 10000
      train_max_tokens: 1000000000
      validation_max_tokens: 10000000
```

The bounds above are illustrative: retain the original fit declaration's actual
values. Obtain the two digests from `klab tokenizer verify ORIGINAL_RUN.yaml
--json` before changing the consuming dataset. Relative paths are anchored to the
new run YAML, including the nested origin paths. Keep the original tokenizer and
manifest bytes unchanged; do not relabel their source or refit the tokenizer.

Run `klab tokenizer verify RUN.yaml --json`, then `klab data prepare
RUN.yaml`. Verification authenticates the pinned tokenizer bytes and manifest
against the original snapshot; preparation independently verifies the consuming
snapshot, including on cache reuse. The origin snapshot must remain available.
The origin binding participates in run configuration identity; moving its files
does not change that identity. An omitted origin preserves the existing binding
to the run's own dataset and serialization.

This opt-in currently supports lab runs consuming a `snapshot`
with a tokenizer origin that is also a frozen snapshot. It does not change
Forge release/export, token-mixture, or release ExperimentPlan artifact binding
contracts. ExperimentPlan resolution rejects `tokenizer.origin` in the base run,
axis choices, or phase patches before verifying input artifacts; use a tokenizer
bound to the consuming dataset for those plans. Cross-source release locking
remains unsupported. Reusing an origin
does not establish train/evaluation disjointness across datasets: audit those
splits separately before making a comparison.

Coverage distinguishes source exhaustion, exclusions, retained documents,
truncation, packed tokens, complete usable blocks, tails and supervised targets.
Passes are over usable prepared blocks; they are not raw-document epochs.
`--require-full` is the budget command's default and rejects incomplete or
unproven full-source coverage. `--no-require-full` explicitly permits a bounded
selection. Objective masks are counted; unsupported objectives fail.

Budget authoring writes a new config with absolute input paths and proposed
step/target limits. It preserves optimizer settings, so review the fresh schedule
before training. It does not derive cumulative continuation budgets or approve
a run. Check measured memory fit and disk growth with native inspection,
preflight and staging.

For another pilot with authenticated materialized inputs, use
`stage CONFIG --prepared-inputs PRIOR_STAGE/prepared --through warmup --output NEW_STAGE`.
This is the sealed prepared-input bundle, not the array cache returned by
`data prepare`. The adapter verifies the existing inventory and requested config;
missing or mismatched inputs fail without downloading or repacking. Select the
registered runtime explicitly when using an accelerator.

The Python `stage(..., prepared_inputs=...)` API enforces the same input checks,
including for `through="inspect"`. Existing-input staging requires a new output;
the destination is checked again under the stage lock and published without
replacing a competing writer. Generic `allow_runtime_drift` is not accepted in
this mode. Ordinary staging without supplied inputs retains verified output reuse.
Validated stage receipts record the input bundle's producer, the prepared data's
production source, and the current verification source separately. These fields
describe provenance; they do not grant source compatibility or change the sealed
input bytes. Older input bundles remain readable without these additional fields.

## Campaign and existing-input reuse

Campaign `dataset_snapshot`, `tokenizer_train` and `data_prepare` stages call
these native operations. `experiment_plan` with `mode: bind` authenticates their
outputs and binds a phase template before locking. Runtime, run, collection,
evaluation and generation stages consume the resulting identities.

Already prepared inputs can also enter through typed artifact references and
standalone `experiment bind-inputs`. Direct snapshots do not need a fabricated
Corpus Forge release. Forge-backed inputs retain their release/export gates.
Missing inputs, inconsistent tokenizer/snapshot provenance and un-ingested
parents block dependent work. See [Campaigns](campaigns.md).

Generic snapshot verification reuses successful payload hashing and SQLite replay
only within one `data prepare`, `stage`, or Campaign
`plan|status|next|explain|apply|resume|approve` operation. Every invocation still
rereads the manifest and embedded lock, binds the supplied dataset configuration,
and checks the exact sibling inventory. Complete tree fingerprints invalidate
reuse after edits, inode replacement, permission changes, or symlinks. The first
verification in every new command is cold; no persistent proof receipt extends
this scope. Imported historical snapshots and acquisition publication remain cold.

Pass `--cold-verify` to any of those commands to replay every verification.
Direct Python `verify_snapshot` calls remain cold unless enclosed in
`snapshot_verification_operation()`; direct Campaign operations own that same
exception-safe scope. `snapshot_verification_statistics()` exposes actual calls,
cold verifications, reuse hits, invalidations and monotonic verifier durations
while the operation is active, and returns `None` afterward. These are operational
observations, not changed dataset identities or model-quality evidence.

## Prepared content identity and replicas

Ordinary `data prepare` caches bind the logical dataset declaration and verified
input digests, complete tokenizer serialization, packing version and sidecar
settings, plus a conservative preparation implementation fingerprint. Model
width/depth and optimizer settings are not preparation inputs. Models of different
sizes can share these arrays when their tokenizer, data and packing requirements
match; a changed byte-memory layout still requires a different cache.

The preparation fingerprint includes all top-level `data/*.py` modules and
`config/models.py`, except `data/snapshot_writer.py`. That module only acquires and
publishes snapshots; no prepared-data reader calls it. Selection rules, snapshot
verification and iteration remain in fingerprinted `sources.py`. A writer change
that changes published content changes the verified snapshot manifest and hence
the prepared identity. Unknown new data modules remain included by default.
Other acquisition code still co-located with readers is conservatively included;
this is not a promise that every implementation-only change avoids invalidation.

| Binding | Purpose |
| --- | --- |
| `cache_identity.source_identity_sha256` | Preparation implementation compatibility |
| Top-level manifest `source_identity_sha256` | Producer package provenance, retained unchanged on reuse |
| Dataset/snapshot and tokenizer digests, packing settings | Logical input and output requirements |
| Batch sizes, worker limits, telemetry | Execution controls, outside cache identity |
| Cache/input/tokenizer paths | Physical locations, excluded from ordinary logical identity |

Copy an intact completed cache directory (including its manifest and all arrays)
to another cache root to replicate storage. Point the current declaration at the
replica and its verified input/tokenizer locations. Native loading still verifies
identity, inventory, metadata and hashes; a directory name or filesystem copy is
not proof. Interrupted staging binds physical paths separately and is not a
portable completed cache. Sealed `stage --prepared-inputs` bundles keep their
existing full-config binding; an ordinary array cache is not such a bundle.
Allocation-backed artifacts retain their package-bound attestation requirements.

This writer/reader separation intentionally creates a new preparation fingerprint
once. Caches produced before the separation are not renamed, rewritten or
implicitly accepted under the new identity. Prepare a new current cache once,
retain historical evidence unchanged, then reuse/replicate that current cache.
No historical compatibility adapter or automatic hash substitution is provided.
