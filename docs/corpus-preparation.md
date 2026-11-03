# Corpus-to-prepared-bundle public surface

## Iterating on a mixture

Existing-snapshot acquisition finalization (`corpus acquire PROJECT
--require-existing`) prints a bounded result with `lock_path`, `project_id`,
`source_count` and `complete`. The acquisition lock remains the artifact. The
optional `--summary` spelling is also accepted. Finalization rebuilds a stale
project lock from verified saved snapshots, without fetching missing sources.

Source normalization is cached by the source definition, snapshot/file identity,
normalizer, resolved rights and that file's verified admission decisions. The
build and complete admission identities remain provenance; adding an unrelated
source does not invalidate an unchanged normalized shard. Family assignment,
rights verification and held-out isolation still run under the current plan.

Mixture encoding uses a persistent tokenizer/content cache, independent of
quotas, ordering and output names. Completed batches survive interruption; valid
encodings are committed even when a supply floor cannot be met. No mixture is
published until all declared quotas and unique-content floors pass. Retained
single-exposure schedules avoid the additional temporary encoded spool.

Mixture source reads use an ordered prefetch queue (128 MiB of serialized source
or cached-token payloads, with a record cap of 256 or twice the reader count,
whichever is larger), followed by bounded tokenizer batches and one cache
writer. Ordinary preflight verifies per-record rights before cache hits can use
its authenticated metadata to avoid schedule-stage source-body reads. Retained
references lack those rights fields, so retained candidates still read and check
each source record, including on a warm token cache. A bounded LRU retains up
to 64 source handles (also limited by the process descriptor budget). Each
active read duplicates its handle and uses positional I/O, so concurrent reads
and cache eviction cannot change another reader's position or close its file.
A single tokenizer subprocess owns its Rayon pool; the measured CPU/RAM plan
splits capacity between source readers and encoding instead of nesting pools.
`SPARSELAB_MIXTURE_WORKERS=N` caps that operation's combined worker plan; omitted,
it uses available physical cores, affinity and measured memory. Ordinary text
preparation retains its existing default. These byte budgets bound queued
payloads, not total RSS: decoded objects, the current record, tokenizer native
memory, SQLite and final receipt storage are additional.

An interrupted materialization resumes automatically at the last authenticated
assembly frontier for the same output. Immutable segments live beside it in
`.OUTPUT.assembly`; preserve that directory and the output parent's `cache`.
Commit cadence is independent of the 256-document/8 MiB tokenizer batches.
By default a commit becomes due after 4,096 candidates or emitted rows, 64 MiB
of pending segment data, or 30 seconds, checked at the next batch boundary;
a completed stratum always commits. `SPARSELAB_MIXTURE_COMMIT_CANDIDATES=N`
selects a positive candidate/row interval (use `1` for eager commits). Byte and
time guards still apply. Pending index deltas stream to disk, not a growing RAM
buffer. This amortizes proof/fsync overhead at the cost of replaying uncommitted
work after a stop; a long individual batch can exceed the time interval.
Each committed segment binds source position, dedup/index deltas, counters,
exclusions and emitted rows to the exact declaration, sources, tokenizer and
producer. Restart restores SQL lookup structures from deltas and concatenates
sealed output segments; it does not fetch, encode, select or emit committed
candidates again. Ordinary source inventory indexes also have authenticated
preflight reuse. Input verification still runs; restoration and final output
copying require sequential I/O. Multiple-exposure output has its own committed
ordinal. A worker-count change is operational and can resume the same frontier.
Changed identities or corrupt frontiers fail closed; choose a new output for a
changed request. Only one writer may own an output. Uncommitted directories are
retained for diagnosis, and completed segments currently remain after publication,
so budget extra disk space for them and any multi-exposure encoded spool.

`.OUTPUT.assembly/progress.json` records per-stratum candidates, selected and
committed work, cache lookups, read queue peaks, stage elapsed/worker time and
parent/child RSS. It is operational telemetry, separate from the scientific
receipt. Selected counts nonempty deduplicated eligible candidates, including
supply beyond the output quota; committed target tokens describe output.
Source-worker utilization is busy wall time divided by available worker time,
not a CPU-utilization measurement. Peak simultaneous source reads, encoder
Rayon bounds, observed child thread counts and encoder CPU utilization make
small concurrency comparisons inspectable. A child that exits during sampling
records an incomplete observation instead of aborting materialization; unknown
encoder utilization is reported as null. Three workers allocate one reader and
two encoder threads, while four workers allocate two of each. Full supply and family accounting
continues after output quotas fill because those counts remain receipt semantics.

For a small standalone comparison, export `tools/benchmark_mixture.py` from the
candidate and run that **same file** with each pinned Lab interpreter (no pytest,
installation, downloads or training). For example:

```sh
/path/to/pinned-lab/.venv/bin/python /path/to/benchmark_mixture.py \
  --root /srv/sparselab/synthetic-mixture-UNIQUE --workers 1 \
  --documents 512 --text-bytes 512
```

The root must be new. The harness pins affinity, runs cold and warm content-cache
cases, includes candidate checkpoint costs, and writes `benchmark.json` with code
revision, fixture/output hashes, complete receipts, timings and scratch usage.
It uses explicitly synthetic retained-source evidence and exercises index/schedule
machinery, **not** the release verifier. Compare hashes and semantic receipts
before timings. Warm means the token cache is warm; it does not claim cold OS or
ZFS caches for the first case. Use each host's own baseline; cross-host timings
alone do not establish NVMe causality or a corpus-scale completion estimate.

Ordinary `corpus verify-mixture DECLARATION --output OUTPUT` authenticates native
completion and current inputs without re-running the schedule. An unchanged
materialization request returns that verified completion. Missing or changed
completion evidence fails closed; `--audit` explicitly reproduces the schedule.
An audit re-encodes without the token cache, compares the complete receipt, and
establishes a current verification record for unchanged output bytes. It does
not rewrite the original receipt or claim that the original execution was new.
Mixture commands print bounded status/counts and artifact paths; the complete
receipt remains in `receipt.json`. Completion uses Lab's owned verification
storage, independently of supervisor logs. Ordinary mixtures keep their owned
verification key under the output parent's cache, so publication does not require
a writable user configuration directory. Preserve this cache with local artifacts.

Packed data depends on preparation implementation rather than unrelated package
code or documentation. Run provenance still records the package revision.
Token-mixture target packing commits native chunks containing both target IDs
and supervision masks. Restart verifies the chunks and input stream, skips
committed output records, and rebuilds only unfinished packing. Raw chunks are
retained until the complete prepared bundle publishes; validation preparation
still restarts if interrupted. Run-owned data uses the existing authenticated
copy path, allowing reflink/clone support where available and falling back to
independent copies elsewhere.

## Reusing completed local preparation

`corpus freeze-splits PROJECT --inventory INVENTORY --clusters CLUSTERS
--output SPLITS --reuse-completed-preparation` selects bounded external sorting
and streaming joins for an already completed, parentless inventory. It requires
the existing completed-execution adoption evidence and the exact reviewed v2
admission digest. Source revisions, inventory-to-admission snapshot bindings,
basic source-level rights eligibility and every qualify/exclude/quarantine
decision remain enforced. Reviewed family
merges, seeded integer split assignment and cross-split exact-content isolation
are unchanged; no documents or categories are removed or reassigned.

This explicit local reuse policy defers acquisition-lock/snapshot verification,
detailed row/file admission verification (coverage, source hashes, per-file
rights exceptions and manual-review requirements), repeated source-body hashing
and per-record provenance-envelope reconstruction during freezing. It checks
snapshot agreement between the retained inventory and admission, rather than
revalidating those identities against the acquisition lock. The receipt
records that scope and retains the original completed-execution references. It
does not claim those checks ran again, prove original-seed independence, grant
redistribution rights or certify a public release. The ordinary verification
policy remains available without this selection. The ordinary downstream build
still invokes acquisition and detailed admission verification and creates its
own document index; this freeze selection does not suppress those later costs.

The producer commits immutable sort runs and phase completion before proceeding.
Native verification proofs authenticate resumable completion against exact
producer/input/output identities, retaining the existing transitive verifier
authority as well as the streaming producer files; a checksum chain alone is
not authority.
Changed/incomplete work is recomputed, preserving prior artifacts. Export chunks
record source byte offsets so a restart resumes after completed chunks. State
lives under the corpus work root rather than in an uncommitted SQLite transaction;
the old failed scratch is never imported. The 32 MiB sort buffer bounds working
buffers, not dataset size or run time. Phase records retain measured work and
elapsed time; no corpus-scale speed claim follows from fixture tests.

Frozen assignments retain their canonical JSONL format. Consumers use an
operation-owned compact key/offset/split lookup instead of copying assignment
JSON payloads into a relational index. Lookup maps close before scratch cleanup,
and repeated consumers in one operation share the same validated lookup.

Current experiment definitions live in [Kybern Workshop](https://github.com/supernovae/kybern-workshop).
Lab maintains a [synthetic native preparation fixture](../tests/fixtures/preparation/README.md) and a connected Workshop binder-to-stage regression.
Use one `preparation.json` with native `attempt phase-command` for inspection and
`attempt run-phase` for one supervised zero-model phase. Public `corpus
render-declaration`, `attempt init`, `attempt bind-artifact`, corpus verifiers and
`data prepared-inputs publish/verify` remain the underlying operations.

## Freezing a completed local payload without copying it

After a successful completed-local build, use its printed build path:

```sh
sparselab --work-dir WORK corpus freeze BUILD --reuse-completed-preparation
```

This explicit selection writes `manifest.json` and `release-publication.json`
beside the existing payload. It returns that same directory as the release
path. Its content-derived release ID is unchanged; consumers take the ID from
the authenticated manifest rather than the directory name. Payload files are
not copied, relinked, moved or modified, and the original build manifest and
completion proofs remain intact. The ordinary freeze selection retains its
independent-copy behavior.

The native reader normally authenticates the original signed publication and
producer identity, plus unchanged input/output bindings. If that signature is
absent, an explicit local executor-evidence declaration can instead establish
that the same build completed and its owner-private executor reports recorded
successful cleanup. This unsigned evidence is not an HMAC proof or a fresh
source re-audit. A cache miss by itself is never enough.

Place the declaration at `WORK/corpora/PROJECT/build-executions/BUILD_ID.json`.
It contains `kind: completed_local_build_execution`, `producer` (`revision`,
`runtime_sha256`, `image`), `project` (`path`, `sha256`), `build` (`path`,
`build_id`, `manifest_sha256`), and an `evidence` mapping. Evidence has exact
`path` and `sha256` references for `job_spec`, `job_report`, `session_spec`,
`session_report`, `container_report`, `stdout`, `stderr`, and `events`. Control
records must be bounded owner-private files under the work root. The reader
compares these with the checkpoint, the recorded producer-file hashes, the
scientific inputs, and the output stamps. Its `declared_revision` field is the
revision in the executor records; it is not an independent Git verification.

The publication record labels either path as adopted execution, not a fresh
build. Default build-cache refusal after verifier changes is unchanged. Missing,
corrupt or changed evidence refuses reuse instead of triggering corpus processing.

Retain the whole returned directory, original source dependencies and native
completion storage for the release lifetime. For executor-evidence adoption,
also retain the original producer checkout and every declared evidence file;
future verification needs their original bytes. Build and release share one local
payload: later mutation or deletion is refused by native consumers. This is a
local continuation path, not an independently portable archive or a change to
source-rights and scientific split policy.

## Interface inventory and migration

| Previous surface | Current surface / disposition |
| --- | --- |
| Separate v1/v2 preparation path maps + copied operator/test argv | `current/preparation.json`, one compiler and native `run-phase` adapter |
| `attempt phase-paths --leaf-name ...` | `attempt phase-command --plan ... --label ...`; output names cannot diverge from the plan |
| `project-pre-freeze-reuse-v2` / `project-build-reuse-v2` live templates | Unversioned current templates; source/split/release/transform slots are generated by the plan |
| Fixture-written project, review, release, mixture, contract and run templates | Actual current templates with explicit tiny-fixture semantic substitutions |
| Writable `AttemptBudget.create/reserve/run`, old module CLI | Removed; public `attempt init/run/status` and contract v1 only |
| Historical SQLite v1 ledgers | Retained raw evidence only; no execution or replay compatibility; current ledger schema 2 required |
| Release normalizers v1/v2/v3, historical admission review v1 | Truthful normalizer identities retained for current inputs; unbound review-v1 acceptance removed; current packet requires inspection-bound review-v2 |
| DevMind-only committed evidence flags and trust shortcut | Removed; current native release/tokenizer paths use explicit evidence and tokenizer-origin binding |
| Old dense/TinyStories surface-study import modes | Removed; native triage import remains |
| Experiment-specific Python/shell launchers and old CI guard | Removed; native attempt dispatch and monitors own execution/accounting/shutdown |
| Historical versioned research packet files | Preserved byte-for-byte at cited paths, excluded from the current plan; not runnable compatibility wrappers |

Release schemas, normalizer identities and ledger schemas are provenance, not
competing public command versions. Renaming them would misrepresent retained
artifacts. The current contract is v1 and durable ledger storage is schema 2.
This finite refactor does not redesign serving, architecture, evaluation science
or archived experiments.

## Inventory completion and split recovery

Native `corpus split-inventory` records a host-owned authenticated completion
cache after all source processing finishes. `corpus freeze-splits` can reuse that
inventory while its source declarations, normalizer, family fields, acquired
snapshot dependencies, output and producer authority remain unchanged. Missing,
corrupt or unavailable optional cache state never becomes evidence of success.

A retained inventory that predates those completion records can select an
`inventory_completion: {path, sha256}` reference in its reviewed cluster
declaration. This explicitly adopts successful local executor evidence: exact
producer revision, Python and container identities, native command, bound
scientific inputs, native output digest and counts, completed session and cleanup.
Control receipts must be bounded owned files inside the work root. Adoption is
reported as `adopted_execution`, distinct from `native_production` or a fresh
`cold_verification`; neither a checksum alone nor a failed or partial execution
is a completion proof. Acquisition and admission verification remain required
before frozen splits publish. Historical timers and resource allowances are not
execution policy for the adopting run.

The historical signed native-completion fallback also needs the original
producer checkout at the declaration's source path. It checks the recorded bytes
of `src/sparselab/corpus/split_inventory_adoption.py` and
`src/sparselab/corpus/completed_preparation.py`, alongside the authenticated
completion evidence and the remaining producer authority. Preserve that checkout
and evidence while relying on the handoff: changing or removing those files can
refuse a later verification that needs this fallback. An unchanged cached
adoption can return without repeating that checkout check; a successful first
adoption is not a general guarantee that the checkout is no longer needed.
This is an inventory-completion contract, separate from retained-release
executor adoption and from any supported release relocation.

Adoption preserves output from the explicitly selected producer revision; it
does not run that historical code or claim current-code production equivalence.
Native completion reuse instead binds current producer and transitive corpus
verifier authority, including normalization and row-family qualification. If a
producer's final unchanged-input check fails, its output is not completed work
and must be discarded rather than reused.

The optional `merge_provenance: {path, sha256}` reference binds reviewed
exact-content component evidence to the inventory, admission, source strata,
seed, weights and merge membership. Missing known merges or stale review inputs
are reported before expensive source processing. This diagnostic does not replace
the full exact-content leakage check. Component IDs are the SHA-256 of canonical
JSON containing the stratum and sorted member hints, prefixed
`exact-content-family:`. Applying a reviewed component does not remove documents
or move them to another stratum.

## Test coverage and pruning

| Guarantee | Retained coverage |
| --- | --- |
| Baseline embedded identity, no ledger before validation | `test_operational_monitor_cli_baseline`, `test_attempt_packet_dispatch`, connected preparation |
| Zero-network retained snapshot reuse | `test_corpus_source_effects`, connected preparation's network tripwire and offline contract |
| Draft/sample/reviewer identity and coverage | `test_corpus_admission_inspection`, connected preparation |
| Normalizer/inventory and protected families | Connected preparation rejection tests and `test_corpus_split_freeze` |
| Leaf/receipt/monitor output collisions | `test_attempt_contract`, connected admission-draft collision |
| Accounting, exclusive creation, rollback, shutdown | Canonical `test_attempt_contract`, contract binding/CLI tests |
| Actual operator argv and production template closure | `test_preparation_commands`, connected public `run-phase` through cold prepared bundle |
| Current Workshop model declaration and native staged inputs | `test_workshop_preparation_integration` |

The duplicated static template-closure test and v1/v2 executable path-map tests
are removed in favor of current compiler/connected coverage. Obsolete budget
writer tests are replaced by canonical accounting and obsolete-input rejection
coverage. Old KML validators, arbitrary-Python Card02 fixtures and the experimental
hosted guard are removed. Device UUID/sensor errors and storage accounting have
focused native monitor regressions. Old model-science assertions are retired,
not represented as passing or replaced by zero-model tests. Historical results
and their limitations remain in the research archive.

The remaining `test_kml_card03_context_check.py` and
`test_kml_card03_review_receipts.py` modules were retired after review found that
they imported deleted, research-specific scripts. Their historical allowlist is
archived unchanged. Native `test_kml_card03_items` retains held-out family/source
binding, exact chunk identity, control structure, immutable output and incomplete
denominator checks. It does not replace the old token-context/phrase-overlap
screen or independent gold-blind evaluation-review receipt protocol. Those old
evaluation-specific guarantees are retired, not claimed as preparation coverage;
current corpus admission review remains inspection-bound and independently tested.

The live research lifecycle now points directly to archived TinyStories evidence
while retaining its exact digest and size. The public lifecycle-reader regression
checks availability; no historical replay or path alias layer was introduced.
Tokenizer probe validation reads its unchanged archived fixture. The source-effects
regression loads `current/project-acquire.yaml`, retaining the three reuse-only
sources and single permitted Wikimedia source declaration.

The connected fixture keeps complete `reuse_only` bindings to verified tiny
origin snapshots through pre-freeze/build and cold prepared-bundle verification.
It rejects a missing retained origin even when a valid candidate copy exists.
All four fixture sources are retained; real Wikimedia acquisition is not exercised.
The independent `test_corpus_source_effects` public-CLI tests cover the mixed
production effect policy using an explicit mocked Wikimedia transport and network
tripwires. Neither fixture result admits a real corpus or qualifies real transfer.

## Split bundle commit and explicit recovery

`corpus freeze-splits` publishes family candidates, split YAML, then the schema-1
receipt as the commit record. Consumers require all three regular files and
matching split/candidate digests. Publication uses exclusive hard links, file
fsync, and directory fsync through the existing work root (including newly
created output directories, but never ancestors above the work root). Receipt publication follows durable data publication.
An error after any link, including a directory-sync error after the receipt,
means the operation's durability is uncertain: keep all evidence. A complete
matching visible receipt is readable, but a failed call does not establish
successful durable completion.

For an interrupted operation on an authorized work area, repeat the identical
`corpus freeze-splits PROJECT --inventory INVENTORY --clusters CLUSTERS --output
SPLITS --recover` command. Recovery is a separately authorized operator step
**outside attempt contracts**: their reviewed option allowlist refuses `--recover`.
An interrupted budgeted attempt must retain its failure/accounting evidence;
operator recovery does not replay or reset that attempt. Adding recovery to the
attempt protocol requires a separately reviewed policy change.

The operator command revalidates acquisition, admission and reviewed clusters,
reuses authenticated completed inventory or verifies it through native production,
recomputes deterministic bytes, checks **every**
existing member before writing, and exclusively publishes only missing members.
Existing identical members are fsynced again without replacement; any conflicting
file or symlink stops recovery. Ordinary retries still refuse existing outputs.
Never delete conflicting evidence or use recovery to repair historical real runs.
Concurrent writers may cause an exclusive-publication failure; preserve outputs
and explicitly revalidate instead of assuming the other writer completed.

New YAML includes a deterministic comment identifying a frozen bundle; this
requires a receipt even when both companions are missing. Assignments and parsed
scientific identity are unchanged, JSONL serialization remains unchanged, and the
receipt binds the new exact YAML bytes. Complete pre-marker schema-1 bundles
remain readable without migration. Legacy unmarked YAML with neither companion
is indistinguishable from a standalone hand-authored split declaration, which
remains supported; do not strip companions from legacy bundles. Explicit bundle
verification and family finalization always require the receipt. Recovery does
not rewrite old unmarked YAML into the new format; such byte differences are
conflicts, preserving historical evidence.


### Large-build rights header limit

The large builder reads file-level rights evidence from at most the first 30
LF-delimited lines within 64 KiB (65,536 bytes), allowing one additional byte
only to detect overflow. CRLF and a final line without a newline retain the
rights resolver's existing semantics. If 30 complete lines or EOF cannot be
reached within that ceiling, the build fails before rights classification or
prepared-shard reuse; a partial header never grants admission. JSON, JSONL and
Parquet names with the resolver's exact lowercase suffixes need no header read.

This limit only removes the redundant full-file rights read. Full-file admission
screening, acquisition budgets and normalizer identities are unchanged. Fallback
preparation still materializes complete inputs (including JSON arrays, Parquet
and monolithic text normalization); bounding that memory remains separate LAB-52
work. The rights-header fix does not make all fallback formats memory-bounded.

## Reviewed split ratios and unique-target exports

The reviewed family-cluster declaration consumed by `corpus freeze-splits` may
include `split_weights`, with exactly `train`, `validation`, and `test` positive
integer weights (sum at most 10000). For fresh EXP1 preparation the unexecuted
configuration is:

```json
"split_weights": {"train": 98, "validation": 1, "test": 1}
```

The existing cluster digest binds these weights, source-to-stratum mapping, seed,
and reviewed family merges. The receipt records requested weights and realized
family counts. Each source stratum needs at least the sum of the weights in
eligible families (100 for 98/1/1); smaller inventories refuse rather than lose a
held-out split through rounding. Largest-remainder allocation is deterministic.
These are **family ratios**, not promises of exact byte or token ratios. Group
repositories, duplicate clusters and synthetic seed families before freezing.
Source strata use nonempty lowercase identifiers, so fresh real-English,
synthetic and code groups need not borrow historical domain names. Existing
cross-stratum family and cross-split content refusals remain in force. Each
source's stratum must also appear in its declared `domains`: the mixture verifier
checks this relationship. Split freezing currently validates identifier syntax,
not that downstream domain relationship; review it before publishing splits.

Omitting weights preserves the existing 8/1/1 allocation and receipt shape.
Changed weights require new declarations and outputs; no historical assignments,
receipts or protected family identities are rewritten.

Native mixture declarations accept integer `max_exposures: 1` or `2`. Use `1`
when preparing a unique-target inventory: insufficient unique positions refuse
publication and report the stratum, available unique positions, requested quota,
exposure limit and shortfall. No extra exposure is silently added. Successful
receipts report actual and unique scheduled positions separately and retain the
existing cold replay verification. Boolean, string and floating-point exposure
limits are refused. Existing explicit `2` declarations retain their behavior.

One-exposure **preparation** does not authorize model training or prescribe its
later epoch count. Counts include the existing segment-EOS semantics; only the
native packing verification establishes supervised target counts. Do not call a
repeated-position export a larger unique corpus. Sequence length remains the
native RunConfig setting (`training.seq_len: 2048` for the approved fresh plan),
and token storage remains native int32.

These controls are preparation prerequisites, not an executable EXP1 allocation.
Train-only byte sampling/audits and the first **4 GiB**
feasibility report within the cumulative **100 GiB** transfer ledger require the
remaining reviewed preparation integration. No data transfer is approved by these
example declarations themselves.

## Declared HF row qualification

A bounded-shard Hugging Face source may declare `acquisition.qualification`.
The declaration digest binds the profile and exact original-row field paths;
the acquisition envelope still binds revision, shard, row index and row hash.
Legacy Gutenberg/Wikimedia sources refuse qualification overrides, preserving
one consistent admission/family policy. For example, this
unexecuted code-policy fragment assumes these fields have been verified in the
pinned source schema:

```yaml
qualification:
  profile: code
  family_fields: [repo_name]
  language_field: language
  allowed_languages: [Python, JavaScript, TypeScript, Go, Rust, Shell, SQL, Markdown]
  repository_field: repo_name
  path_field: path
  license_field: license
  license_labels:
    mit: MIT
    apache-2.0: Apache-2.0
    bsd-2-clause: BSD-2-Clause
    bsd-3-clause: BSD-3-Clause
    isc: ISC
```

Code families group the entire repository; adding path fields to split one
repository into smaller families is refused. Repository labels only narrow the
candidate pool. Missing or incompatible labels, disallowed languages, unsafe or
vendor/generated paths, generated notices, detected credentials/private
identifiers and conflicting file-level notices quarantine rows. Other detected
notice flags require review. The existing notice patterns are screening, not a
complete license or secret detector. Verified repository/path/license/language
attribution, including the mapped SPDX expression, remains in normalized document
metadata. Repository family values are casefolded for GitHub identity grouping;
original attribution is retained.

`profile: synthetic_text` requires `family_fields` and
`synthetic_provenance_fields` naming retained original seed/provenance fields.
Only the declared acquisition `text_field` becomes training text; provenance is
hashed into document metadata. Group shared seed families before split freezing.
`profile: web_text` requires declared family fields and optionally a language
field with exact allowed values. Dotted paths such as `metadata.url` and
`metadata.language` traverse dictionaries only; empty, reserved, overlong or
whitespace-padded path segments refuse. A FineWeb-shaped policy can group on
`metadata.url` without treating each dump as a different family. Code-only fields
on other profiles and synthetic provenance fields on other profiles refuse,
rather than imply an unused screen. Whitespace-only provenance is missing.
Cosmopedia `seed_data` is a dataset label, not an individual seed identity: it
cannot alone establish seed families. Actual per-seed provenance and reviewed
family grouping remain a source-qualification gate; no seed IDs are invented.
Family hints are conservative inputs to the
reviewed cluster declaration, not a replacement for duplicate/decontamination
review. Missing family or synthetic provenance is a material conflict.

Use native `corpus admission-draft` and the existing inspection/review/build
sequence. Metadata inventory can enumerate unadmitted rows, marking unresolved
provenance; enumeration is not admission. An unfrozen build without a reviewed
record-admission manifest does not apply these screens. Freeze requires reviewed
admission and refuses unresolved families. These declarations never grant source
rights or owner approval.
The current admission receipt supports **one selected HF shard per source
entry**; declare separate pinned entries for additional shards and map them to
the same source stratum. Verified repository/seed values can group across those
entries. Real source schema, rights and provenance qualification remain gates;
these synthetic fixtures do not establish the suitability of any remote corpus.

## One native tokenizer with TRAIN byte sampling

For a newly verified LM export, copy its tokenizer configuration to a new
reviewed declaration, retaining its dataset/export bindings, and add:

```yaml
vocab_size: 32768
sampling:
  source_strata:
    fineweb_shard_000: real_english
    cosmopedia_shard_000: synthetic
    code_shard_000: code
  byte_weights: {real_english: 60, synthetic: 20, code: 20}
  max_fit_bytes: 536870912
  audit_documents_per_stratum: 1000
  max_audit_bytes: 67108864
```

This is an **unexecuted fragment**: source IDs must cover the actual frozen
release exactly, and the export must already specify vocabulary 32768. Run the
existing `tokenizer train CONFIG` command. The native preparation dispatcher supports this leaf with declared sampling,
authenticated export, a new output inside the attempt root, and any required
release acceptance binding. The existing ByteLevel BPE trainer is used once, with no tokenizer normalizer.

Selection visits eligible documents in immutable release order, restricted to
the authenticated LM TRAIN lineage. It takes whole UTF-8 documents without
repeating content until each stratum's byte cap is reached or its bounded scan
is exhausted. Largest-remainder integer byte caps implement the requested
weights; whole-document granularity can leave unused bytes. This is a
deterministic prefix within each stratum, **not a representative random sample**.
Native release rows are sorted by digest-based `document_id`, so this is hash
order rather than source/acquisition order.
`max_documents` bounds inspected TRAIN documents. The manifest records requested
caps, realized bytes/documents, unfilled bytes, order and selected identity hash.
An empty selected stratum refuses training. No validation/test document is used
for fit or audit, and no duplicate is added to fill a quota.

The audit uses different TRAIN content hashes, up to 1000 documents per stratum
by default and a finite shared byte cap. It reports exact round-trip failures,
unknown tokens and bytes/token; fixed identifier, numeric, multilingual and code
whitespace probes are reported separately. Audits encode literal special-token
spellings such as `<unk>` as ordinary text on an isolated tokenizer copy; the
saved tokenizer's production encoding behavior is unchanged. The report records
this audit encoding policy and cold replay applies it identically. The audit
population is the remainder after fit selection and can favor documents that
were too large for the remaining fit caps; its bytes/token is descriptive for
that population, not an unbiased corpus estimate. Missing audit documents are explicit
shortfalls, not passes or claims that enough unseen data was available. Unknown
tokens or failed round-trips refuse artifact publication. Cold tokenizer
verification replays selection and audit against the authenticated export; a
changed declaration requires a new output location. Existing unsampled tokenizer
configurations keep their prior serialization and behavior.

The 512 MiB cap bounds selected fit bytes, not process RSS. The existing trainer
and release verifier may use more memory; the 16 GiB operational target still
requires runtime observation and the feasibility gate. The current sampled fit
makes four scans of the document prefix (statistics, fit, audit re-selection,
audit); cold verification makes three. Each scan JSON-decodes rows, including
held-out rows that are then excluded from fit/scoring. The TRAIN-lineage set and
selected-content hashes also consume memory. These costs need measurement in the
pilot; a 512 MiB fit budget does not bound total I/O. This does not authorize
fitting on a real corpus or running a model workload.

## Initial transfer feasibility gate

A new transport budget may specify `initial_body_bytes` below
`max_total_body_bytes`. Both caps combine source and metadata **reservations**;
failed/interrupted transfers and retries retain their full conservative charge.
For the unexecuted fresh EXP1 plan these are 4 GiB and 100 GiB, respectively.
Separate source/metadata caps, retry limits and disk checks apply in either mode.
Enforced ledgers also enforce the original wall deadline; observational ledgers
retain and report it as a target. Omitted fields preserve existing ledger
bindings.

Before a reservation crosses the initial cap, acquisition refuses and retains
all completed snapshots and failure evidence. Save `corpus budget-status PROJECT`
as the `ledger_receipt` member of a report with explicit `findings`. Findings must
report `status: ready` to continue; blocked findings refuse. A separately
reviewed JSON decision requires `decision: continue`, the exact report
`report_sha256`, and an actual nonempty `approval_reference`. With real owner
authorization, use `corpus budget-release-gate PROJECT --report REPORT --review
REVIEW`. This action is intentionally excluded from automatic attempt dispatch.
The ledger receipt must match exactly. Catchable interrupts mark reservations
interrupted and re-raise. After an uncatchable stop, a reservation can still say
`started`: before release, the owner must independently verify that its process
and descendants have stopped, then add `stopped_transfers` to the reviewed
JSON decision. This maps every started transfer key to a nonempty stop-evidence
reference. Missing, extra or blank entries refuse. An unchanged receipt alone
is not proof of shutdown. Release retains the pre-reconciliation receipt hash,
records those references and marks only those entries interrupted; all charged
bytes, retry history, deadline and scope remain unchanged. Partial or zero-byte
pilot results must be disclosed in feasibility findings, never called completed
preparation. A retry consumes another full reservation from the same total.
Approval text or stop evidence in a file never creates authority or bypasses
another hold. Never reconcile a live transfer.

Concrete preparation recipes belong in user-owned directories or repositories.
The historical EXP1 declarations moved to the DevMind project in Workshop
(recipe DevMind 340M Base); the [disposition record](exp1-retirement.json)
binds their original bytes to the destination commit. Lab supplies the native
commands described here; Workshop and an internal runner are not prerequisites.
A feasibility gate is not a new transfer allocation; never initialize a replacement
ledger to evade a cap or change the full source scope after starting.

Historical fixed evaluation selection/slice commands still require their three
historical strata. Fresh `real_english`/`synthetic`/`code` preparation does not
make those evaluation protocols compatible; a separately reviewed evaluation
change is required before using fresh strata there.

Tokenizer dispatch cold-verifies the export before reserving an attempt; the
trainer verifies it again. This adds a full verification pass that must be
included in the pilot's observed I/O and wall-time costs.

## Preview acquired JSONL without copying a corpus

`corpus preview DECLARATION --input-root INPUTS --output NEW_DIRECTORY`
reads explicitly pinned snapshot manifests and acquired JSONL in place. The JSON
`sparselab-corpus-preview-v1` declaration specifies a seed, source/snapshot IDs,
relative manifest/data paths with exact SHA256 and byte lengths, aggregate read
and record limits, sample/excerpt/output bounds and wall seconds.

Only HF-dataset acquisition envelopes with `source_row_index` and
`source_row_sha256` are supported; Wikimedia envelopes are unsupported. Every
row must have a string in its source's declared top-level `text_field` (default
`text`). For example, set `"text_field": "code"` on a source whose acquired rows
contain `code`. The selector controls excerpts only: provenance hashes still cover
the entire original row without renaming or copying fields. The receipt records
the selector. Missing or non-string selected fields refuse; there is no fallback.
One data file per snapshot is supported in a declaration:
manifest and data paths must both be unique. Resolve symlinked root/ancestor
paths first (including macOS `/tmp` and `/var` aliases).

This unexecuted example shows the declaration structure. Replace the symbolic
pins and byte sizes with verified values from your own acquired snapshot;
placeholders deliberately fail validation:

```json
{
  "format": "sparselab-corpus-preview-v1",
  "seed": "preview-1",
  "sources": [{
    "source_id": "example",
    "snapshot_sha256": "SNAPSHOT_SHA256",
    "manifest": {
      "path": "snapshots/SNAPSHOT_SHA256/manifest.json",
      "sha256": "MANIFEST_SHA256", "bytes": 1234
    },
    "data": {
      "path": "snapshots/SNAPSHOT_SHA256/files/sample.jsonl",
      "sha256": "DATA_SHA256", "bytes": 5678
    }
  }],
  "max_read_bytes": 1048576,
  "max_record_bytes": 65536,
  "samples_per_source": 3,
  "max_excerpt_bytes": 256,
  "max_output_bytes": 65536,
  "max_wall_seconds": 60
}
```

Selection keeps the lowest seeded SHA256 ranks per source and records physical
line, original row index/hash and clipped text. It checks each acquired row's
self-contained provenance envelope and both whole-file hashes in one pass;
blank lines count toward bytes but not rows. Manifest bytes are hashed without
loading the manifest's row array. Snapshot identities are supplied by the reviewed
binding: this preview does not reconstruct acquisition metadata or substitute for
native cold acquisition verification or admission review. Its receipt explicitly
states that narrower verification scope and never grants admission.

The read counter includes declaration, manifests and data; preflight rejects pins
that cannot fit one pass. Every data read is charged, records are bounded, and
only the requested excerpts survive processing. Output is exclusive and includes
`started.json` plus either `report.json` or `failure.json`; abrupt termination may
leave only the started receipt. Keep supervisor exit/cleanup evidence too. Output
bytes include started plus either report or failure, with 4096 bytes reserved
for bounded failure evidence. Failure evidence records the completed source
count and SHA256 of their ordered ID list, and a UTF-8 byte-clipped error. A
report that exceeds the output cap refuses after scanning; choose an output
budget that covers the declaration, source metadata and selected excerpts. Wall
checks are cooperative between reads; use a supervisor for process cleanup,
measured RSS and retained logs. Network isolation is also a supervisor property;
the preview itself performs no network operation. No files are copied, and an
existing output directory refuses. Current admission inspection limits are
unchanged: a preview is evidence for review, not an admission inspection receipt.

### Record attribution under an explicit application policy

For acquired HF sources with declared `acquisition.qualification`, an admission
application-template source may include `row_use_policy`:

```json
{
  "mode": "record_attribution",
  "restricted_rows": {
    "<exact-original-row-sha256>": {
      "basis": "The actual explicit restriction on the intended training use",
      "evidence": "Pointer to the retained source notice or restriction text"
    }
  }
}
```

The placeholder above must be replaced with an acquired original-row digest;
this is not an executable declaration. An empty `restricted_rows` records no
explicit row restrictions in that policy; it does not prove that no restrictions
exist. Source-wide `rights.training_restriction` checks remain in force. This
policy is provenance and a declared processing choice, not new owner authority.

`record_attribution` retains unknown, mixed and unmapped code license labels and
SPDX/copyright notice lines instead of using a permissive-label allowlist as a
training gate. It makes no legal-clearance claim and does not infer explicit
non-use terms from a license label. Bind known explicit restrictions to original
row digests with their basis and evidence. Such rows are excluded, and unused
restriction digests refuse verification. Credentials, absent family/source
provenance, out-of-policy languages, unsafe paths and generated/vendor code
continue to fail their separate quality checks.

The native draft writes `row_evidence`; cold verification recomputes it from the
pinned original row and refuses missing/substituted evidence. The full original
row remains the authoritative source for license text and notices; extracted
notice lines are not an exhaustive terms detector. Inspection carries the
record, and normalized documents preserve it as canonical JSON in scalar
metadata `row_attribution_json`, including explicit unknown values. Existing
source declaration bytes and acquisition identities remain unchanged. This mode
is not supported for undeclared historical HF profiles or Git-file admission.

Inspection gathers offsets and cleaned lengths during native admission validation,
then selects samples from disk-backed metadata using the exact median, rank,
strata and fallback rules. It reads only selected records again for display;
whole source text is not retained. The declared text field includes `code`.
HF integrity hashing is folded into its physical JSONL validation pass, including
blank lines. Cold snapshot authentication remains a separate native check.

`max_input_bytes` bounds the snapshot payload size once, and its declared-source-
maxima precheck uses the same units. `input_bytes` records that size; `read_bytes`
records the source-validation plus selected-record reads used to construct the
inspection. These construction observations remain stable when completed work is
reused; they are not a claim that the current invocation reread those bytes.
The command's `work` summary records current full source passes, validated rows,
selected records/bytes, cache hits, and distinct upstream snapshot cold checks
or proof hits. Separate acquisition/integrity reads are not included in the
inspection `read_bytes` counter. No elapsed-time or throughput improvement is
implied by the tiny fixture pass-count tests.

Each HF JSONL record in drafting, verification and inspection, and each Git file
in inspection, must be at most 8 MiB; an oversized item refuses the source rather
than being skipped. Inspection retains its 64 MiB sample-pool bound, small
excerpt/output bounds and explicit input budget. These bounds do not establish
capacity or model-training readiness for a whole production corpus.

Notice excerpts retain LF physical line numbers, at most 64 matching lines and
1024 characters per line, with `notice_lines_truncated` recording omissions.
Original rows, including full notices, remain authoritative.

Declared-row credential screening now additionally refuses `password=`, `api_key=`
and `secret_key=` assignments with values of at least 24 characters, in every
declared profile (including without `row_use_policy`). Previously qualified rows
containing these patterns can now fail cold verification.

Receipt verification compares the complete recorded inspection with the expected
result from current validated inputs. It may reuse an authenticated completed
inspection; it never accepts an edited public receipt merely because inputs
still hash correctly. Regenerate an incompatible inspection and its bound review
through the current flow, reusing an unchanged valid draft. A draft whose own
attribution evidence is outdated still needs native regeneration. Preserve old
evidence and approvals as history; there is no compatibility adapter.

#### Inspection reuse and restart

Reissue the same `corpus inspect-admission` declaration after an interruption.
Completed, authenticated source results survive; incomplete source work is
discarded and recomputed. A matching completed output is verified and returned
without overwriting it. Different, corrupt or incomplete existing output refuses;
use a new output path to preserve that evidence. Publication uses a temporary
file and an exclusive final link, so a new interruption cannot publish a partial
receipt. The command does not automatically retry a failed operation.

Reuse uses Lab's existing host-authenticated operational proof store. It binds
source/admission identities, selection and normalizer settings, verifier code,
and the exact input inventories and file fingerprints, including ctime. A changed
source decision or source-specific selection recomputes that source; changed
shared settings or code invalidate the affected completed work. A missing,
corrupt, unsigned or stale cache falls back to native validation. Unavailable
cache authority or cache storage also falls back; the `work` summary reports
cache availability without treating the native result as a cache hit. Public receipt
contents are still compared in full against the authenticated expected result.
Source hashes alone do not establish that sampling or validation ran correctly.

Keep the same owned work root, service UID and persistent `XDG_CONFIG_HOME`
across process/container restarts. The configuration directory holds the existing
private verification key; keep it outside published artifacts and logs. Lost or
foreign keys and unsafe ownership/permissions prevent warm reuse rather than
relaxing checks. The proof-store trust boundary assumes immutable service-owned
inputs; it does not protect against a compromised same-UID owner.

The `work` summary counts completed full validation passes for this invocation,
not partial work from a failed attempt. Source indexing and exact-median queries
use disposable disk metadata; only successful bounded inspection results are
retained for reuse. Warm reuse is an operational optimization, not new review
authority or evidence that the corpus is scientifically suitable for training.


### Bounded admission metadata and split inventory

`corpus admission-draft`, `corpus split-inventory`, and `corpus freeze-splits`
verify acquired snapshots through `verification_options(root)`. In a trusted
work root, matching authenticated proofs can reuse verification of unchanged
dependencies and verifier authority. Missing or stale proofs fall back to cold
verification; an untrusted work root uses cold verification directly. The first
run after a relevant authority change may therefore reread snapshot bytes.
Proof reuse does not waive integrity checks: changed snapshot bytes that no
longer match their manifest refuse rather than becoming a new trusted snapshot.

`corpus admission-draft`, admission inspection/reverification, and split inventory
use disposable SQLite storage for allowlisted JSON inventory arrays (positions listed below)
and row-index lookups during their operation. Original JSON files remain the
source of truth: native acquisition/admission checks still authenticate them,
and canonical draft bytes and document identities are preserved. No new artifact
schema, review authority, source identity, or training eligibility is introduced.

Storage lives in `WORK_DIR/corpora/PROJECT/metadata-scratch/.metadata-*` and is
removed when the operation exits normally or raises. Abrupt process termination
can leave owned scratch; it is never reused as evidence. Native phases use owned scratch without a fixed database/file quota. Non-row JSON
remains limited to 4 MiB and individual streamed metadata rows to 8 MiB. Explicitly
capped MetadataStore API contexts retain their database and input limits.
Duplicate JSON keys, nonfinite numbers, nesting deeper than 64, and symlinked
metadata inputs/scratch paths refuse. These are refusal limits, not measured
capacity guarantees. SQLite's cache setting is not a hard RSS cap or an aggregate
filesystem limit; the existing supervisor must still account for all storage,
time and sampled memory. Nested native verification shares the same owned database and any explicit caller cap.

Draft publication streams canonical JSON to a temporary file, fsyncs it,
then publishes without replacing an
existing destination. A failure removes only its own temporary file. HF JSONL
split inventory streams one acquired row at a time, verifies every receipt even
beyond parser selection limits, and preserves the whole-shard digest and nonblank
row ordinal used by the document parser. Other split-inventory file formats, including current `wikimedia_dump` JSONL,
refuse above 8 MiB instead of reading an unbounded file.

This removes full HF metadata-array and source-shard loads from these phases.
It does not establish that the complete release-to-bundle path fits a retained
production corpus within its existing limits. Inspection still
uses compact numeric median metadata and bounded sample pools. Validate the
connected tiny fixture path and then measured resource fit before preparing a
large retained corpus; a fixture pass does not establish production readiness.


Disk-backed collections are private to these operations. Use their explicit
iteration/index interfaces; copying or concatenating them refuses instead of
silently returning empty builtin storage. Do not pass the internal list/dict
subclasses directly to Pydantic or other C-level container consumers: those can
inspect their empty builtin backing. A future public schema boundary must use a
non-subclass representation or deliberate bounded conversion first. Cold snapshot
verification does validate source declarations with Pydantic. Declarations remain
plain header-bounded containers: spilling is limited to top-level inventory
arrays, admission `sources[*].files`/`records`, and snapshot/lock retrieval
`shards[*].selected_rows`. Arrays nested under declarations or unrelated keys
never spill. Repeated reads of unchanged metadata within one operation reuse the
parsed row collections after checking the exact input bytes and file identity.
They hash the file again, but do not parse or insert its rows again. Parser-limit
changes invalidate reuse. Headers are fresh copies, and owning-store appends
detach shared rows before modification. Scoped row release invalidates affected
materializations while preserving older collections. The bounded header cache evicts or skips
entries rather than refusing a valid workload; its budget is not a hard RSS cap.
This is operation-local derived storage, not an authenticated execution receipt
or a persistent restart cache. Native provenance/source verification still runs;
closing the operation discards the materialization. No full-corpus speedup is
claimed from tiny regression fixtures.


### v2 disk builds

Reviewed v2 LM-only builds now use the existing disk builder when all sources are
v2 HF/Git declarations, record admission is pinned, chat is unselected, and no
fraction selection is requested. Other transform combinations retain native
dispatch. Admission is verified before dispatch and its exact bytes are copied
into the immutable build. Admitted normalization stays in the owning process;
SQLite-backed decisions are never passed to process workers. Admitted HF rows and admitted
Git files refuse above 8 MiB. Cold JSONL replay applies that same cap only to
files with verified record admission. Structure-only replay without admission
retains the builder's existing row-size behavior; this change does not introduce a
universal row cap. V2 keeps its original representative ordering,
cross-split duplicate refusal and incompatible-rights refusal; it does not adopt
v3 canonical-origin grouping or held-out promotion. Audits retain family overlap
warnings and policy-unit refusals.

Build and cold-readback admission coverage use SQL joins rather than full Python
identity sets. Cold coverage uses the operation's existing capped metadata
database, including its cleanup and storage accounting. Frozen split and cold evidence storage are described below. Tiny fixture parity
does not establish capacity for the retained real corpus. The existing inspection/draft limits and
external aggregate storage, time and memory supervision still apply unchanged.


### Small derivatives of retained HF snapshots

For an explicitly approved small profile, an HF source may declare
`acquisition.retained_subset` with an absolute `snapshot_path`, full
`snapshot_sha256` and `manifest_sha256`, a nonempty `profile_id`, integer `seed`,
and optional `max_record_bytes` (at most 262144). Keep the parent's source
identity/policy fields and bounded upstream shard declaration unchanged; only
`max_rows`, `max_bytes`, and this derivative binding differ. This produces a new
source declaration and snapshot identity. It never relabels the original snapshot.

`corpus acquire PROJECT` creates the derivative locally. All project sources must
use retained selection, with no transport budget or source effects, and aggregate
maxima at most 30000 rows and 100 MiB. `corpus acquire PROJECT --offline` verifies
an already-published acquisition; it does not create one. The retained path cannot
fall back to Hub acquisition. Use the existing supervisor's network isolation and
resource accounting as well.

Selection is exactly SHA256 of canonical JSON
`[profile_id, seed, parent_source_id, parent_snapshot_sha256, original_row_index,
original_row_sha256]`, interpreted as an integer modulo 100 equal to zero. Selected
physical rows above the declared record limit are excluded and counted; other
selected bytes are retained unchanged in parent order. Empty selection or row/byte
overflow refuses without publishing a partial source. This samples rows, not a
statistically certified representative population. Long-row exclusions and actual
counts must accompany any result. Family splitting, admission, tokenizer fitting
and training authorization remain separate native stages.

A supported parent has one JSONL shard of at most 8 GiB, a manifest of at most
256 MiB and physical rows of at most 8 MiB. These are refusal limits; selected
output rows still have the separate 256 KiB ceiling.

Cold snapshot verification replays the pinned parent and selection, including
unselected rows, and checks exact output bytes and row membership. Nested subsets
are unsupported. Parent mutation, policy drift, symlinks, incomplete ancestry and
resealed omissions refuse. Parent reads can be repeated during downstream cold
verification; budget all passes and temporary files. Parent metadata scratch is
released after replay while preserving the existing operation database cap.
Plain acquisition places scratch under its owned staging directory; direct cold
verification uses the child snapshot's parent directory. An enclosing native
metadata operation keeps its existing owned scratch location. No fallback to a
system temporary directory is used for this replay. A
successful tiny fixture establishes correctness, not real-data capacity.

The native preparation supervisor accepts creation under an
`offline_retained_sources_only` contract only when the acquisition project is
bound both by its file digest (`acquisition_project_sha256`) and by the existing
resolved acquisition identity (`preparation_acquisition_identity_sha256`, paired
with `preparation_normalizer`). The resolved identity is the native acquisition
lock's project identity: project ID plus every complete source declaration, not
just the bytes of the top-level YAML. A missing identity or changed source file
refuses before reservation. Every source must declare a retained subset, with no transport
allocation or source-effect plan is present, and aggregate selection caps fit.
Mixed live/retained projects refuse before reserving work. Parent bytes still
undergo native cold verification; a declaration or approval reference is not a
verification receipt.

The absolute parent path is part of the source declaration and child snapshot
identity. Moving the parent changes that identity and prevents cold replay at the
old location; matching parent bytes alone do not make a child snapshot portable.
Publish a native prepared-input bundle for a portable training handoff rather
than treating the derivative snapshot as a standalone transferable artifact.

Create validation and test splits through the native family split of one subset,
or explicitly verify overlap against other data. Different selection seeds or
profiles are not disjoint splits and do not inherit the parent's held-out split.
The fixed one-percent selector and named Lab ceilings deliberately refuse large
selections; the modulus is not an adjustable sampling or automatic retry knob.


### Mixture preparation memory and scratch

Mixture materialization and cold replay index document offsets, family/content
constraints, deterministic ranks and encoded-token offsets in disposable SQLite
storage. Ordinary and multi-exposure schedules spool encoded documents on disk
and replay them in the same stratum, rank and exposure order. Retained
single-exposure schedules write directly to the output stream without that
additional encoded spool; committed assembly segments still consume disk.
Token JSONL bytes and schema-1 receipt meanings are
unchanged. The existing contiguous-EOS packer writes train and validation arrays
through disk spools and opens the completed arrays read-only with memory mapping;
it does not allocate arrays for the entire scheduled training quota in RAM.

This bounds retained Python data by individual records and finite metadata, not
by total corpus tokens. Individual mixture source/encoded JSONL records are limited
to 64 MiB, receipts to 256 MiB, and exclusion entries to 128 MiB of canonical JSON.
Raw validation JSONL follows the local-text packer instead: at the default
1 MiB tokenizer source batch, its per-row cap is 6,292,480 bytes
(`6 * source_batch_bytes + 1024`), not the mixture's 64 MiB cap. An
oversized record or receipt refuses publication without dropping evidence from a
successful receipt. SQLite uses a 4 MiB page cache and disk temporary storage;
these are not hard RSS enforcement. Tokenizer internals and release verification
have their own resource requirements.

Scratch is real storage: document indexes, encoded candidates where spooling is
used, retained assembly segments, output schedules, cold-replay output, raw array
spools and completed arrays can coexist. Keep native workspace/resource monitoring and cumulative accounting;
cleanup is not a refund. This implementation does not reduce acquisition
reservations, authorize a larger disk budget or establish that a full corpus fits.

### Observe local corpus storage

For an owner-authorized local workflow without a storage allowance, select the
policy once on the existing ledger while no transfer is active:

```sh
sparselab --work-dir WORK corpus budget-observe-storage PROJECT --owner OWNER --approval-reference APPROVAL_REFERENCE --min-free-bytes FREE_FLOOR
```

This records the original disk target and the actual free-space floor as
provenance. It preserves full project/source identity, transfer/retry history and
every network reservation/charge. It changes neither time policy nor training
scope. Other ledgers and dataset acquisition keep their enforced defaults.
The command is intentionally excluded from automatic attempt dispatch.

With observational storage, source steps omit `--max-new-bytes` and retain
`--min-free-bytes` at least as large as the recorded floor. Native path/peak
projections remain reported observations; they do not admit/refuse by a storage
allowance or predicted occupancy. Actual free space is checked before transfer
and at staging growth observations. The outer supervisor must keep the floor on
both the actual corpus mount and its Windows backing volume. These checks are
sampled stops, not filesystem quotas. Existing file integrity, bounded traversal,
memory and cancellation controls remain active. This policy grants no new
downloads, cloud spending or authority beyond the approved corpus.

### Observe total transport time

Transport ledgers enforce `max_wall_seconds` by default. For an owner-authorized
local corpus workflow whose elapsed time is observational, select that policy once
on the existing ledger, while no shard transfer is active:

```sh
sparselab --work-dir WORK corpus budget-observe-time PROJECT --owner OWNER --approval-reference APPROVAL_REFERENCE
```

This command is intentionally excluded from automatic attempt dispatch. Run it
through the approved operator workflow against the existing corpus ledger.

This changes only the overall elapsed-time policy. The original start and planned
deadline, project/source binding, all completed and failed transfers, cumulative
byte charges, retry limits and storage limits remain intact. Later acquisition
steps inherit the recorded policy; there is no repeated renewal. The reference
records provenance and does not create owner authority.

The original `deadline` is a reported target in observational mode. Each HTTP
connect/read still has a maximum 30-second operation timeout, and cancellation,
resource stops and verification remain active. Other corpus ledgers and bounded
dataset-acquisition workflows retain their existing enforced defaults. Harness
reports session start/end timestamps and elapsed duration independently.

For a large project, initialize its transport ledger once with `corpus budget-init`
and keep the complete project declaration unchanged. An explicit step selects one
exact source ID and all bounded shards in that declaration. It does not create a
smaller project, change sampling or mixture, reset the ledger, or retry a failure.
For a project with one declared shard per source, this is one shard per step.

```sh
sparselab --work-dir WORK corpus acquire PROJECT --source-id SOURCE_ID \
  --max-new-bytes REMAINING_NATIVE_BYTES --min-free-bytes FREE_FLOOR_BYTES --plan
sparselab --work-dir WORK corpus acquire PROJECT --source-id SOURCE_ID \
  --max-new-bytes REMAINING_NATIVE_BYTES --min-free-bytes FREE_FLOOR_BYTES
```

Within a Lab supervised attempt, `corpus acquire` accepts this step form only
when `--source-id`, `--max-new-bytes` and `--min-free-bytes` are all present;
`--plan` is optional, and `--offline` cannot be combined with step limits. The
attempt classifier rejects partial or invalid numeric limits before dispatch.

Both byte limits are required integers. `--plan` cold-verifies retained origins,
existing selected snapshots, the full-project ledger binding and admission without
network or writes. Execution repeats admission. Use an exclusive supervised work
root; the command is a single acquisition operation, not a scheduler or authority
to acquire data. The caller must supply the actual remaining cumulative allowance
after charging earlier work and reserving external logs, receipts and baseline
files. There is no inference of approval from these numeric options.

Admission reports `native_new_path_bytes` (a conservative bound on added logical
path bytes) separately from `native_peak_added_bytes` (peak added logical occupancy).
The former includes selected compressed bodies, expanded output and source
manifest at both staging and published paths, plus native ledger overhead. The
latter uses one expanded copy and the largest compressed shard. Current project
bytes plus peak addition must fit the unchanged transport disk cap; filesystem
free space must cover peak addition plus the requested floor. This does not bound
physical allocated blocks, other volumes, total bytes rewritten to an existing
file, or concurrent unrelated disk use. A supervisor must enforce those separate
host limits and cumulative accounting. Deletion does not refund previous charges.

The step is tested on Linux with the existing Python 3.14 runtime; native Windows
is unsupported (POSIX filesystem statistics are required). It supports immutable
bounded-HF declarations, including verified
`reuse_only` sources. It requires an existing full-project transport ledger,
cold verification and no proof-store or metadata-scratch writes. Source manifests
are capped at 256 MiB before writing. Transport state is capped at 16 MiB JSON in
a 32 MiB SQLite database, with a stable rollback journal reserved at 64 MiB;
PERSIST journaling preserves rollback semantics without creating a new journal
inode on every counter update. These are native metadata ceilings, not RSS caps.
Unsafe paths, a hot journal needing recovery, missing pins, changed bindings or
insufficient allowance refuse before transport. Metadata and response bodies
remain subject to the existing transport counters, deadlines and retry limits.

A successful step returns `status: source_verified`, `complete: false`, the exact
snapshot path/digest, full-project identity, transport binding and admission.
Snapshots and the ledger retain partial progress; there is no partial active
`acquisition.json`. A later explicit step reuses a cold-verified completed source
without downloading or copying it again. Failures preserve earlier snapshots and
consumed charges. Changing the selected ID never weakens source permissions or
publishes an incomplete corpus as complete.

After every declared source is present, ordinary `corpus acquire PROJECT` verifies
all declarations and publishes the complete acquisition lock, reusing the saved
snapshots. Its existing full-lock and verification memory costs still apply;
source-step admission does not certify downstream build, tokenizer, packing or
training capacity. Keep every step's JSON/stdout/stderr and supervisor receipts.
No real corpus acquisition or model run is part of the offline fixture evidence.

To finalize eligible existing snapshots without fetching or reconstructing a
missing source, use `corpus acquire PROJECT --require-existing`. Eligibility
depends on the source and lock state:

| Source path | Existing matching lock | Stale or absent lock |
| --- | --- | --- |
| Ordinary pinned `git`, `huggingface_dataset`, or `wikimedia_dump` | Reverify the referenced snapshot and its declaration/adapter compatibility. | Discover verified compatible snapshots under the project's source snapshot directory and rebuild the lock; refuse if no eligible snapshot exists. A stale reference can also fall through to discovery. |
| Ordinary `local` or `http_document` | Excluded from old-lock reuse; refuse without reacquiring. | Excluded from immutable snapshot discovery; refuse without reacquiring. |
| Declared `reuse_only` source effect | Verify retained origin rights/permissions and bytes, and the matching snapshot already in the destination project. | The same checks apply independently of the lock. A valid origin alone is insufficient: a missing destination snapshot refuses instead of being copied. |
| HF `retained_subset` project | An existing lock takes the dedicated verification path; an invalid or stale lock refuses instead of entering ordinary discovery. | With no lock, existing compatible snapshots can be discovered; missing snapshots refuse. Retained-subset project restrictions still apply. |

Rejected sources are recorded as rejected rather than requiring an acquired
snapshot. A replacement full lock publishes only after all applicable source
checks succeed; failure preserves the prior active lock. Do not delete a live
lock to work around a refusal. This command can write metadata and the completed
lock, so it is not a read-only check. Its bounded output reports `lock_path`,
`project_id`, `source_count`, and `complete`; `--summary` remains an alias for
that output.

The flag does not replace snapshot integrity checks and cannot be combined with
`--source-id`, `--plan`, or `--offline`. Use `--offline` to verify an already
published lock rather than finalize one.


### Local network observation

For an already authorized local corpus assignment, select the policy on the
existing ledger instead of replacing it or resetting counters:

```sh
sparselab --work-dir WORK corpus budget-observe-network PROJECT \
  --owner OWNER --approval-reference REFERENCE
```

This is provenance of an actual owner decision, not new download authority. It
records the original planned network limits and retains all historical byte and
transfer records. Cumulative source/metadata/combined byte quotas, the initial
byte gate, and numerical retry/transfer allowances become observations. A retry
is still an explicit operator action; no retry loop is added. Known completed
transfers and terminal integrity failures are not silently replayed.

New observed transfers record start, last byte progress, end, elapsed time and
measured body-byte throughput. Historical records are not assigned invented
timing. Immutable source scope, content hashes, per-response/per-file bounds,
network inactivity timeouts, metadata storage bounds and actual resource checks
remain. Other ledgers keep their declared budgets. Time and storage observation
remain explicit separate selections; neither implies this network choice.

A sampled free-space-floor stop is an environmental interruption. After space is
restored and cleanup is verified, an explicit retry uses the same ledger and keeps
both attempts and all observed bytes. Integrity failures remain terminal. Transfer
duration/throughput measurements describe a normal same-boot start and finish;
recovery preserves the old transfer and starts a new record, rather than inferring
comparable historical or cross-boot throughput.

### Frozen assignment companions and cold evidence indexes

`corpus freeze-splits` now writes a compact document-split declaration with empty
`assignments` and `assignment_file: {sha256: DIGEST, rows: COUNT}`. Its companion is
`assignments-DIGEST.jsonl` beside the declaration. Each canonical row contains
`document_id` and `split`, in strictly increasing document-ID order. Hand-authored
inline assignments remain supported; the schema requires exactly one representation.
The file representation is only valid for document splits. Keep the companion
with the declaration when moving a recipe. The digest and row count are bound to
the native build identity, and builds copy the exact companion into their closure.
Cold build/release verification checks every emitted document against it.

Inventory, family grouping, prior-family/content protection, quota assignment and
family finalization now use operation-owned SQLite joins. Assignment choices keep
the same seed ordering, integer quotas, prior held-out protection and duplicate
refusals. Family candidates and assignments stream to owned temporary files.
Publication writes the family candidates, immutable assignment companion, compact
split YAML, then receipt. An existing identical digest-named companion may be
shared by a second bundle. Existing owned bundle members require explicit
`--recover`; recovery revalidates inputs and all retained members before completing
publication. Conflicts refuse without replacement. All four publication stages are
covered by interruption/recovery tests. Regenerate new declarations and receipts
through the public flow; do not rewrite retained receipts or active attempt pins.

Cold document/span/lineage and derived-record lookups use private SQLite indexes;
selected-view comparisons and stage identity hashing stream rows. These private
collections never cross Pydantic/public schema boundaries. Frozen declaration and
receipt headers refuse above 4 MiB; frozen assignment and inventory rows refuse
above 8 MiB. Derived evidence streams one row at a time without adding a new
row-size limit to non-admitted builds. Admitted raw-source replay retains its
8 MiB limit. Index tables share operation-owned disk storage, including nested operations.
Temporary publication files and SQLite sort scratch also consume real storage;
monitor actual free space and memory throughout the operation. Explicit caller
quotas remain supported. No production capacity or throughput claim follows from
the offline fixture tests.

Before real preparation, validate unchanged admission/inspection limits and the
complete corpus-to-bundle path under its approved storage/time/memory envelope.
Other downstream inventory consumers still require capacity assessment; this
change does not authorize acquisition, execution, cap increases or repinning.

### Native metadata scratch and capacity

Native admission, split, build and cold-release operations stream finite pinned
metadata through disposable SQLite storage. They do not impose a fixed 1 GiB
scratch quota, a 256 MiB metadata-file quota or a 128 MiB admission-draft quota.
Readers use the current regular file size as their input bound and refuse growth
beyond it. Record, header, nesting and path-integrity limits remain; explicit
capped MetadataStore API contexts retain their input/database limits, including
when a caller supplies an enclosing context. Inspection keeps its explicit
max-input-bytes budget and small output/excerpt bounds, without a 2 GiB ceiling
on a caller's streaming input budget. Choose that explicit input budget for the
declared corpus; it does not allocate an array of that size.

Scratch and outputs consume real disk space. For large local preparation, retain
supervision of actual free space on the corpus filesystem and its backing volume,
memory sampling and cancellation. SQLite cache settings are not hard RSS limits.
I/O failure retains source evidence and removes only owned unpublished scratch;
no complete artifact is published on failure. Tiny fixture tests establish
semantics and streaming, not full-corpus capacity or scientific qualification.


### Full acquisition closure memory

The full-project `corpus acquire PROJECT` CLI keeps manifest and receipt row
inventories in operation-owned disk scratch through cold verification, atomic
lock publication and stdout serialization. It streams the existing JSON shape;
canonical acquisition-lock bytes and source identities are unchanged. The
receipt can still be large on disk, so retain stdout with appropriate capture
capacity and supervise actual free space. Per-source acquisition keeps its
compact result and does not claim a complete project lock. This change does not
alter source selection, retry policy or scientific preparation settings.


### Reproducible split families from retained fields

The existing split declaration accepts `family_fields_by_source`, mapping a
qualified HF source ID to original row field names. For Cosmopedia preparation:

```yaml
family_fields_by_source:
  cosmopedia_shard: [prompt]
```

This changes split grouping without editing immutable acquisition declarations
or reacquiring their bytes. Inventory generation applies the declared fields;
freeze persists the same map in the resulting split YAML. Cold inventory replay
therefore uses the identical rule. Unknown sources, missing qualification,
invalid field paths and overrides that weaken repository grouping refuse.
Missing original row values produce unresolved families and cannot be admitted
to frozen training splits.

The native family ID is `synthetic_text-family:` followed by SHA-256 of the
canonical UTF-8 JSON array of the selected original values. With `[prompt]`,
that array contains the exact retained prompt: no trimming, case folding,
Unicode normalization or extraction of quoted seed text. Identical prompts
group together across shards. Different prompts may still share an underlying
seed; repeatable prompt families are not proof of independent original seeds.
Cosmopedia's retained `seed_data` values can be dataset labels, and quoted seed
excerpts in prompts are not authenticated original seed IDs.

To reproduce a split, retain the recipe/code commit, upstream revision, shard
and snapshot digests, acquisition lock, admitted inventory digest, explicit
family-field map, reviewed merges, split seed and integer weights. Native
freezing ranks families by the hash of `[seed, stratum, family]`, with the family
name as the tie-breaker, and assigns deterministic integer split counts. Keep
the frozen YAML, digest-bound assignment companion, family inventory and commit
receipt together. A clean clone can verify those artifacts and regenerate the
same memberships from the same bytes; mutable upstream availability is separate.

### Local build from completed preparation

For an explicitly approved local continuation after completed acquisition,
admission, inspection and inventory, use the streaming freeze's completed evidence
and opt into its matching native build path:

```sh
sparselab --work-dir WORK corpus build PROJECT --reuse-completed-preparation
sparselab --work-dir WORK corpus freeze BUILD_DIRECTORY
```

The build requires the frozen split receipt's explicit completed-inventory
declaration. It authenticates that historical execution through the existing
adoption path, requires unchanged source declarations and reviewed admission, and
records this policy in the build/release identity. It renders each admitted HF
JSONL shard with the native renderer and checks the acquired sample hash during
that necessary read. It does **not** replay the acquisition/snapshot audit or
reconstruct every row-admission/provenance envelope. Those deferred checks are
listed in the identity; this local continuation is not a new full source audit or
a redistribution grant. Ordinary builds retain their existing verification.

Compact references, external sorts and an eight-byte-per-document parent array
apply the same native v2 transitive duplicate policy and minimum-ID representative.
Cross-split exact-content and incompatible-rights duplicates still refuse.
Frozen candidate identities/content/strata are joined to rendered documents and
family isolation is checked before native output publication. The final kept
family inventory is included in the build; `finalize-family-inventory` can export
it without reconstructing the document index. The descriptive lineage census is
explicitly deferred, not reported as performed. Documents, spans, lineage, LM
views and rights/audit formats remain native.

Completed source rendering, compact-index phases and final publication use the
existing native completion store plus append-only phase records. Unchanged
authenticated phases are reused after interruption. Changed inputs, producer
code, configuration or outputs invalidate reuse; a hash alone does not establish
execution. Local build/release consumers require the recorded completion and
its trusted work-root dependencies. Missing completion authority refuses instead
of silently starting a full audit. Keep the work root and its completion records
with the outputs. This is local resumability, not a portable execution attestation.
Unfinished phases can still require recomputation. Native output/copy hashing and
actual free-space monitoring remain; tiny tests make no full-corpus throughput or
capacity claim.

Within a trusted private work root, group write is eligible when the group is
the owner's user-private group as checked by `writable_by_others`; shared-group
or world write does not qualify an immutable artifact as trusted. The separate
mutable-declaration path can bind eligible owner-group-writable declarations
under its ownership and path checks. Inspection cache proofs bind
their captured fingerprints in the signed
closure and recheck them on load, without changing filesystem permissions. A
same-size input change still invalidates reuse. World-writable, foreign-owned,
linked or unsafe mutable dependencies remain ineligible.

For native split/build phases, an owned atomic completion record can also recover
a checkpoint whose optional cache proof was never published. The current request
identity, completion digest and cached result must agree; all outputs must retain
their recorded stamps and trusted owned paths. User-private group write is
eligible; shared-group and world write are refused. This is local native
completion evidence, reported as `owned_native_completion`, not a newly invented
execution signature. Corrupt present proofs and incomplete or changed native
records refuse reuse. A present proof rejected for changed verifier authority is
not adopted by this fallback; it requires recomputation under the new authority.
Historical records and proofs are never rewritten.

For a mixture from this release, normal materialization executes the schedule
once and seals its actual completion. Later packing/verification consumers check
that completion, exact declaration/tokenizer/family/release paths and unchanged
outputs instead of tokenizing the whole corpus again. Other mixtures keep their
ordinary cold replay. The optional `unique_train_content_floors` map requires
positive counts for exactly the declared strata; it counts all eligible unique
TRAIN content IDs **excluding EOS**, during the same encoding pass. It does not
change target quotas, ordering, exposure limits or the training stop. Supply
counts and floors are recorded separately from scheduled supervised targets.

Mixture encoding reuses cached token IDs by tokenizer digest and text. Cache
misses are sent in order-preserving CPU batches of at most 256 documents and
8 MiB of source text to the existing bounded encoder (up to four workers).
Larger individual documents and tokenizers with padding enabled use the existing
single-document encoder to preserve exact token IDs. Cache
writes and mixture ordering remain in the coordinator; completed cache entries
survive retry. The reference sort, duplicate checks and ordered output remain
host-side work, so tokenizer batching does not parallelize the whole pipeline.

A retained in-place release may explicitly set `tokenizer_observation` on its
mixture declaration to reuse a tokenizer observed after a completed local job.
Set `method: post_run_observation`, `observed_at_utc` to the actual UTC
observation time, and `files` to `{path, sha256}` entries for `job_spec`,
`session_spec`, `job_report`, `session_report`, `container_report`, `stdout`,
`stderr`, `tokenizer_config`, `tokenizer`, `tokenizer_manifest` and `export`.
The consumer checks those small files, the closed job/session, and the retained
release and export metadata before encoding and again before publication.
The receipt labels the result `post_run_observation`: current tokenizer digests
are not claimed to have been attested by the earlier job. The tokenizer sample
selection, disjoint audit, and export split/license recomputation are not
replayed. This explicit choice relies on the owner's local observation and the
existing retained-release trust boundary. It is not a cold-verification proof.
Keep all observed files at their recorded paths, unchanged, for the life of the
mixture; later consumers check them again and use the bound export's recorded
validation bounds and license rather than repeating those corpus scans.
Omit the field for the ordinary cold tokenizer verification path.


### Explicit retained-render adoption after an assessed verifier change

Ordinary resume still refuses every present proof rejected by current authority.
If a reviewed bookkeeping-only correction prevents reuse of exact retained
renders, an operator can bind a one-invocation adoption decision. Assess the
specific old/new code first: matching partial producer identities alone cannot
establish equivalence of arbitrary transitive changes.

```sh
sparselab --work-dir WORK corpus adopt-retained-renders PROJECT \
  --source-stages WORK/corpora/ID/source-build-stages \
  --from-source OLD_COMMIT --to-source REVIEWED_COMMIT \
  --rationale 'Reviewed cache bookkeeping change; retain exact prior execution' \
  --output WORK/retained-renders.json
sparselab --work-dir WORK corpus build PROJECT --offline \
  --reuse-completed-preparation \
  --retained-render-adoption WORK/retained-renders.json \
  --retained-render-adoption-sha256 PRINTED_SHA256
```

Declaration creation reads only bounded completion/cache/proof metadata and
fingerprints. It authenticates each historical proof using the existing store,
requires an unchanged native producer and every non-authority binding, and
records the exact old proof, old/current binding and authority hashes. It does
not re-sign old execution, claim a new cold render, or read corpus bodies.
`--from-source` and `--to-source` are operator assertions with distinct
40-character hexadecimal values; this path does not resolve Git commits or
verify that a checkout matches either label. The operator's assessment remains
necessary.
The decision binds the parsed project and a fixed set of source-stage requests.
Within the declared source-stage root, changed requests, inputs, outputs, proofs
or current authority refuse continuation instead of silently rerendering.
Directories outside that root follow ordinary cache behavior; the option is not
a global no-rerender guarantee. Original proofs and completion records remain
untouched. Each adopted render appends a durable `retained-renders.uses-*.jsonl`
entry next to the declaration; keep it with the run evidence. This journal records
actual adoptions, not proof that every declared entry was consumed. The option applies
only to that build invocation and does not change future default cache behavior.

### Add a separately prepared release

An incremental streaming split may bind the retained baseline through the
family-cluster declaration's `prior_family_inventory` object. Set `path` and
`sha256` to the baseline release's `family-inventory.jsonl`, `release_path`
to that retained release, and `family_clusters` to an object containing the
original cluster declaration's `path` and `sha256`. The prior cluster digest
must match the authenticated original split receipt. Use completed preparation
reuse; the ordinary cold split path retains its existing contract.

The streaming path indexes the retained family and content metadata without
reading the old document bodies. Existing family splits stay fixed. A new
family containing retained content is anchored to the retained split when the
relationship is unambiguous; incompatible strata, conflicting prior splits,
or quotas that cannot be met without moving old families are refused. Exact
content crossing new families is still checked by the normal split logic.

For mixture materialization, `additional_releases` may contain one object with a
`release_path` for a separately built, retained release in the same work root.
Multiple additional releases are refused until their family splits can be
coordinated. The additional release must carry a completed split bound to the baseline family inventory;
source IDs must be disjoint and all sources mapped in `source_strata`. The
tokenizer remains bound to the baseline release. The mixture records the
ordered release identities and reads authenticated retained offsets instead of
rebuilding an index from every old document body. Its existing cross-release
split and rights checks still apply. Ranked reference checkpoints can be
reused, but an interrupted token encoding spool is not yet resumable.


### Relocate a retained release

For a copied retained release, run `sparselab corpus attest-release-migration`
on the original host with the release path, `--verifier-checkout` and the exact
`--verifier-revision` of a clean checkout that can verify that release. Copy the
resulting source observation to the corresponding private work root on the new
host, then run `sparselab corpus adopt-release-migration` with the relocated
release path and `--source-observation`. Both hosts must use the same existing
private verification key; neither command creates a key or changes historical
receipts. Preserve the original verifier checkout and execution evidence.

The source observation records native release verification and the source to
target relative-path mapping. It reuses digests bound to verified snapshot
manifests and hashes remaining inputs; it does not claim to have reread the
snapshot bytes. The target hashes the mapped input and output files once, checks
them against those digests, and records new target fingerprints. Signed
per-file progress supports retry without rereading unchanged files. Missing
evidence, changed bytes, or a changed inventory are refused. This transfer can
be large; its size depends on the release. The receipt uses the existing local
key and trusts its service user, who can create or modify local evidence; it is
not an independent third-party signature or a replacement historical proof.
The signed completed receipt records the adoption authority and remains usable
across compatible Lab updates. A changed target fingerprint, including a device
identity change, requires recovery: these commands do not automatically replace
an existing target receipt or regenerate a source observation for changed
adoption code. Keep the original host and evidence for that recovery.


### Reuse retained mixture stages after relocation

The release migration above does not cover the separate stage files needed for
retained-mixture indexing. With the original host and its evidence still intact,
run `sparselab corpus attest-mixture-stages RELEASE` there. Copy the resulting
`RELEASE_ID.stages.source.json` to the corresponding target work root, then run
`sparselab corpus adopt-mixture-stages RELEASE --source-observation FILE` on the
target. Both commands use the existing private verification key. The target also
requires the previously completed release migration and its source observation.

The source observation identifies whether each global stage used a historical
proof or the existing owned native-completion fallback. Native completion is
local executor evidence, not a historical HMAC proof. Render stages require their
existing authenticated evidence. Preserve the original receipts, source files,
producer checkout and key; neither command creates a historical proof. The source
reuses recorded digests and unchanged source fingerprints instead of rereading
stage bodies. The target hashes each distinct required stage file and records
separate target fingerprints; signed per-file progress supports retry. Changed or
missing bytes or evidence are refused. This is an owner-private local trust
boundary, not an independent third-party attestation.


### Reuse a tokenizer after relocation

After the release has a verified migration receipt, make a new tokenizer
configuration whose paths map the original work root to the verified target
work root; retain the original tokenizer configuration and job receipts.
In the mixture's existing `tokenizer_observation`, keep the original file paths
and digests and add `relocation` with `source_receipts_root` and
`target_receipts_root` for the separate job-receipt directory. The verifier
obtains the corpus roots from the authenticated release migration receipt,
maps the two roots only, checks the original evidence bytes, and requires the
new configuration to match the original apart from that path mapping. It
records this as a relocated post-run observation; it does not claim that the
original job attested tokenizer output digests or repeat tokenizer fitting.
