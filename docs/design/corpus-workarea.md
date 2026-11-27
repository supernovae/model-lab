# Corpus work area and knowledge compilation

Corpus prepares reusable training information from pinned sources. The same
work area serves conventional natural-language pretraining and experiments
with relational, hierarchical, symbolic, and procedural representations.
An ordinary text recipe can still go from acquisition to training without
any structured extraction. Small datasets can still use the
[local data path](../datasets.md#prepare).

The original increments were reviewed at Lab `b53022b2` and Workshop `4164998`.
The recovery entries below were refreshed at Lab `fa567164`. The interfaces
in the design sections describe future work. Current commands
are in [Corpus preparation](../corpus-preparation.md) and
[Token store](../token-store.md). The issues are
[#459](https://github.com/supernovae/model-lab/issues/459) for the design,
[#450](https://github.com/supernovae/model-lab/issues/450) for bounded build and
verification, and [#485](https://github.com/supernovae/model-lab/issues/485)
for recovery. The [rolling backlog](https://github.com/supernovae/model-lab/issues/497)
records the decisions from [the original design](https://github.com/supernovae/model-lab/pull/493).
The [D1–D12 decision table](https://github.com/supernovae/model-lab/blob/b53022b2/docs/design/corpus-workarea.md#9-open-decisions-for-the-owner) preserves the labels used in that backlog.

## Current implementation

| Original increment | Evidence on main | Remaining work |
| --- | --- | --- |
| 1. Shared shard runner and verification | #494 and #495 landed the range runner. #576 bounds its worker window by CPU and available RAM, limits native threads, and verifies cached JSON and file-backed results within the same window. | Extend bounded scratch admission and receipted reductions to the remaining long passes. |
| 2. Near-duplicate screen | #491 and #496 landed checkpointing, sharded scans, and template-aware classes. | Reuse the held-out index and template evidence as artifacts. Add passage-level grouping and measure classification precision. |
| 3. Tokenizer sampling | #494 and #499 landed range scans and seeded-hash sampling. Workshop #112 pins the sample seed. | Keep this path. Track fit and cold sample-verification costs separately from sample selection. |
| 4. Acquisition parts | #456 landed the verified transport cache. Bounded Git, Hugging Face and PDF reuse verified parts. Archives and wiki dumps reuse pinned inputs and completed derivations. Indexed Wikimedia commits planning and extraction in groups. #601 adds exact-generation staging inspection and removal under the adapter lock. | Measure representative archive and partial-body replay, wider Wikimedia runs, and recovery on slower storage. |
| 5. Narrow build | #486 checkpoints complete disk-build index and dedup stages. #488 landed the canary. #579 resumes native disk LM view emission from verified append boundaries. The native disk index references prepared payloads and checkpoints groups of prepared files. | Measure out-of-cache emission and the largest input group. Bound compact index assembly, dedup replay, report measurement and manifest hashing. Extend emission recovery to completed preparation. |
| 6. Freeze and verification | #500–#502 landed signed receipt reuse, freeze by verified rename, and resumable describe measurement. Cold verification still performs full checks. Reviewed, fractional and chat releases index file ranges and compact metadata for replay. | Extend checkpointing and measure out-of-cache replay for reviewed releases. Separate integrity from transformation replay and sampled audits. |
| 7. Shared work area | `StageStore`, normalized shards, and token caches exist. Most source and build artifacts are still project-scoped. | Share immutable outputs across projects, narrow identities, and protect reference closure during explicit pruning. |
| 8–9. Operators and sidecars | #448/#451 moved choice into recorded release policy. Exact deduplication and the approximate screen exist. The runner declares stages but there is no general typed operator registry or sidecar store. | Extract reusable evidence and make selection consume sidecars without rewriting source text. |
| 10. Reference releases and mixtures | #503–#507 improved quota checks, mixture indexing, and the ordered writer. #519–#523 and #557 improved reuse and verification keys. Mixture checkpoints bind prefixes of the files that will be published. The sealed cache publishes bounded int32 or opt-in uint16 segments keyed on exact content, tokenizer behavior and physical encoding. Reference schedules bind exact segment dependencies. An opt-in direct loader reads those segments. Reference-only bundles own the schedule, records, validation and masks. Offline dispatch supports whole sealed segments or selected ranges with fidelity checks, and transfer inspection compares their physical bytes. | Measure accelerator use and batch wait at production sizes. Compare loader modes before choosing a default. Measure the transfer and replay costs of selected ranges on deployed workers. |
| 11–12. Planner and external operators | Existing CLI, journals, preflight and dashboard supply entry points. | Add dependency plans, bounded inspection and explicitly authorized command operators. |
| 13. Training preemption | #533 added opt-in device-layout changes under runtime optimization. #535/#537 added finished-run data cleanup and reuse. | Trainer recovery stays with #485 and #516. Corpus tracks data and schedule identity so those changes can proceed independently. |

Workshop's [foundational corpus](https://github.com/supernovae/kybern-workshop/pull/112)
uses source-family splits, seeded tokenizer sampling, and mixture declarations.
Its [additive release](https://github.com/supernovae/kybern-workshop/issues/109)
can proceed through the existing workflow, with independent rights, split,
contamination, and quota checks. Retain the screening and reporting decisions
for the existing base release. Keep active training on its pinned checkout.

The [long-stage recovery inventory](../corpus-preparation.md#long-steps-and-what-a-failure-costs)
records the committed units, restart work and measured costs for the current
Corpus and training paths.

## Architectural decisions

- Keep one content-addressed work area under `WORK/store`, immutable source and
  release artifacts, atomic publication, separate annotations, explicit
  policies, and reuse by reference. Keep the optional transport cache on its
  own configured storage. Resolve rights for each project from retained
  evidence. A shared artifact supplies data and provenance, while a project
  supplies the decision to use it.
- Keep documents and passages as typed artifacts. Add source observations and
  structured records alongside them. Let normalization preserve the source
  syntax and locator needed by typed adapters. Acquisition, normalization,
  split assignment, and release assembly remain owned Lab stages. An adapter
  may parse pinned source data. Authored code and rule systems enter as data
  unless a user explicitly authorizes execution.
- Replace the five-kind ceiling with typed operator ports and declared
  capabilities. Annotation, selection, deduplication, extraction, analysis,
  compilation, tokenization, and mixture construction have distinct output
  contracts. Keep a small built-in registry and Lab-owned stage ordering.
  Dependency planning follows manifests through that order. There is no
  recipe-defined workflow scheduler.
- Replace the independent map and small-merge restriction with explicit
  execution patterns. Native multithreading, bounded cross-partition work,
  and documented sequential stages belong in the same recovery contract.
- Treat structural redundancy as research evidence. Approximate similarity
  and heuristic co-occurrence can propose candidates. They require independent
  justification before claiming equivalence, causal relationships, or
  suitability for automatic removal.
- Retain the decisions to prefer reference releases, seeded-hash sampling,
  explicit prune, and smaller token IDs when the vocabulary and measurements
  justify them. Keep cold verification and full replay available. Stage the
  transition to routine integrity verification plus seeded replay samples
  with distinct results and measured audit coverage. The old target of
  landing layout changes before the next large Workshop run is historical.
  Use a verified cutover before building a new release that needs the layout.

## Artifacts and dependencies

An artifact manifest carries the payload type and schema, computation key,
payload sizes and digests, producer identity, resolved semantic options,
typed input references, provenance, and receipt references. Keep time,
resource use, progress, and reporting fields outside the computation key.
Use Lab's canonical digest and file hashing functions. The three identities
have separate purposes:

| Identity | Meaning |
| --- | --- |
| Computation key | Reuse decision. Bind the producer and its output-affecting implementation dependencies, resolved options, input identities, and partition descriptor. |
| Payload digest | Byte integrity. Hash exactly what was published. Re-encoding a table changes its byte digest even when its records are equivalent. |
| Record ID | Stable reference within a typed contract. Source occurrence IDs retain a pinned location, input digest and parser/normalizer identity. Semantic IDs follow the typed identity rules and retain links to all supporting or conflicting occurrences. Dictionary ordinals stay local to their index. |

Native LM builds now share normalized source partitions by computation key
within one Lab work root. The typed receipt binds the pinned source,
resolved rights and admission, renderer options and three payload digests.
Projects resolve eligibility before lookup. The shared cache is outside
project-specific stage retention. `klab corpus plan PROJECT` reads the pinned
inventory and resolves the native build ID. It reports an existing build or
consumed release, or predicts normalization work and index and deduplication
reuse from the ordered input receipts. Those candidates carry byte counts
for payloads the build will verify. Missing receipts defer dependent keys.
The planner and builder share identity construction. Updating those modules
changes the normalized artifact producer and the native build key. New
builds at this cutover re-normalize from their existing source snapshots
and derive fresh index and deduplication keys. Existing releases remain
addressable by their recorded IDs.
With retained index and deduplication receipts, the native planner can
identify a view checkpoint and report the candidate verified prefix and
uncommitted tail from metadata. Build authenticates the checkpoint chain
before reuse. The shared view identity implementation changes the native
build key. New builds can reuse verified normalization, index and deduplication
artifacts across that change. Fresh-stage scratch estimates, reference
publication and closure-aware pruning remain in #567.
Native storage inspection reports the filesystems' available bytes, published
normalized candidate sizes and the flat span output and append bytes when
their inputs are known. Builds journal logical output footprints for sealed
index, deduplication and checkpointed view payloads. Planning reports the
same-build observations separately from prospective bytes.
`klab corpus plan --build BUILD` previews ordinary freeze from existing
build metadata. It reports the release identity, verification and placement
dependencies, shared store growth and duplicate staging bytes. Optional
verification inspection checks signed receipts, available scratch space and
matching row-index checkpoints. A valid complete receipt forecasts zero
verification scratch. The verifier journals observed logical and allocated
row-scratch peaks, and inspection reports recent observations for the same
artifact. A byte-verification plan still leaves total peak demand unestimated.
`klab corpus artifacts` reports retained bytes, computation keys and staging
counts from shared-store metadata. Its paginated inspection is advisory.
Release sample, review, lineage, source, audit and description queries use the
manifest-bound release reader and keep verification in scope until the query
finishes. Export creation, cold export verification and tokenizer bakeoff
also use this reader. Export and bakeoff check the verified generation before
publishing their results. Sampled tokenizer fitting resolves the members
through its authenticated export proof and keeps their recorded input
generations through fit and audit. Native and additive mixture preparation,
tokenizer observation, the training validation view, publication metadata,
family inventory derivation and finalization, prior split preparation,
protected-lineage audits and source-token measurement also resolve their
release payloads through the shared contract. Ordinary freeze can opt in to
digest references for transform outputs, released documents, evidence spans
and LM text views. The manifest binds their content and placement. LM
lineage stays with each release, and export config paths resolve the shared
view for the tokenizer and trainer.
Release and export verification receipts track their file generations.
Shared placement records the verified generation of a large payload once
it moves to its retained digest path. Later freezes can reuse that signed
check. The move checks the original verified staging generation before
signing the stored file.
Corpus artifact inventory reports unique stored bytes and published or
in-flight references for each store. Its unreferenced preview withholds
candidates when the metadata scan is incomplete. Releases still render a
project-scoped document inventory. The release payload stores support
explicit digest pruning. The apply path walks all published and staged
release roots under the work-area publication lease and uses the same
per-digest locks as freeze. These roots protect the payloads reached by
exports, prepared bundles and runs through their release manifests.
Ordinary native builds catalog normalized evidence span partitions in
acquisition order. Freeze can bind the ordered partition keys and receipts
directly to the release after verifying that they reproduce the logical
span digest. Published releases then retain the referenced normalized spans
and receipts. Normalized inventory can inspect published and staged span
references for one page of partitions using the same guarded manifest scan
as the digest stores. It can include unconsumed builds and sealed native view
staging in the same scan. Consumed build markers hand off retention to release
manifests, and unsealed native views make the combined counts unknown. A native
build can opt in to a verified span projection before view emission and carry
it through freeze. The seal binds the ordered
input hashes and exact file generations, so a view retry can reuse it while
the original partitions are available. Normalized pruning accepts explicit
partition IDs and checks the combined release and build roots, plus durable
pins for interrupted native preparation. Builders hold a shared normalized
lease, and apply takes that lease exclusively along with the release lease
and selected writer locks. Apply retires each directory by rename before
deleting its members. Released documents carry selection,
split and duplicate decisions, so references to normalized document payloads
require a separate overlay and ordering contract. That contract and retention
for other artifact families are tracked in #567.

Start with typed document, passage, annotation, duplicate-group, selection,
entity, assertion, evidence, edge, index, and compiled sequence payloads.
An assertion carries subject, predicate, typed value or object, qualifiers,
validity where present, and evidence links. Keep observations, formal rules,
constraints, and derivations in their own tables as consumers arrive.
Preserve source-specific claims and conflicting values. An extraction result
identifies explicit source statements, derived results, and heuristic
candidates. Unsupported data keeps its original source locator and receives
a diagnostic.

Store immutable text and serialized sequences in sharded files. Use Arrow
batches and Parquet for narrow tables with the existing PyArrow dependency.
Keep bulk joins sequential within partitions and spill sorts to configured
scratch. Measure an out-of-core engine such as DuckDB on the same workload
before introducing it. Use compact integer dictionaries and partitioned
edge tables. Add memory-mapped lookup or CSR/CSC adjacency where measured
query patterns justify it. CPU execution is the baseline. No full graph or
large Python object per edge is required.

| Changed input | Reuse and rebuild |
| --- | --- |
| Selection or project eligibility policy | Reuse acquisition, normalization and unrelated sidecars. Refresh the decisions and dependent release. |
| New source or changed partition | Reuse unaffected partition outputs. Recompute scope-wide duplicate decisions or indexes if their semantics depend on the combined scope. |
| Compiler options | Reuse canonical records and unrelated indexes. Rebuild affected representations and their downstream artifacts. |
| Serialization | Reuse canonical records. Encode only affected serialized sequences again. |
| Tokenizer or encoding behavior | Reuse extraction, indexes and serialized content. Re-encode affected bytes. |
| Mixture weights or sequence length | Reuse token segments. Rebuild the schedule or prepared view as required. |

The proposed `klab corpus plan` traverses these dependencies and reports
reuse, rebuild or missing inputs, the reason, affected partitions, and
estimated work and scratch in text and JSON. Show when estimates lack a
measured baseline. Filesystem paths are locations for artifacts, not
computation identities.

## Operators and execution

Each built-in operator declares its input and output payload types, typed
parameters, implementation revision and output-affecting dependencies,
partition or global scope, execution pattern, determinism, and resource
envelope. Pin output-affecting seeds and libraries in its identity. Record
the Lab revision for traceability and exercise deterministic outputs in CI
to catch an unrecorded semantic change. Keep format versions separate from
named policy options.

Extend the shared runner (`klab.corpus.shards.run_sharded` and `run_tasks`)
and `StageStore` together. Start with one partitioned
recipe path from normalization to a sealed release. Add file-backed maps,
partitioned shuffle and external sort, indexed joins, and bounded iterative
passes as operators require them. Workers pass small manifests and counts.
Cap working memory across processes and native threads, bound completed
results awaiting ordered publication, and reserve or monitor assigned
scratch. Oversized work is repartitioned or refused with its estimated
cost. Each multi-pass operator must identify its receipted boundaries and
largest unit of replay.

Stage output on the destination filesystem, validate it, then durably
publish by atomic rename under a per-key lock. Stop launching work on
cancellation. Completed units remain reusable, staging output remains
ineligible, and aggregation resumes at a committed boundary. Report
processed and reused units, rates, resource bounds and replay cost. Test
that deterministic stages produce the same payload digests after
interruption and across worker counts.

The proposed external command operator binds a pinned executable and
output-affecting dependencies, validated parameters, explicit input and
output contracts, execution context and determinism. Require
`--allow-command-operators` at execution. Planning and inspection never
run commands. Checked-out paths convey no trust. Launch with staged
inputs and outputs and validate the outputs before ingestion. A plain
subprocess runs with the caller's filesystem and network authority.
Advertise only isolation and resource limits actually enforced. Require
explicit execution again for a requested replay of external code.

## Text refinement and compilation

- **Sidecars and policy.** Join sidecars by document ID within the identified
  normalized artifact. Text-only annotation can memoize by exact content
  digest. Metadata- and corpus-scoped annotation binds the dependencies it
  reads. Start with language, privacy, source statistics, quality indicators,
  exact and normalized duplicate groups, and reusable template evidence.
  Put selection in recorded release policy over typed columns. Build can
  run missing annotators or adopt their verified outputs through the same
  path. Preserve duplicate evidence, representative decisions, and the
  original material. Approximate document and passage matches default to
  reporting. Continue explicit exact-overlap policy for held-out data.
  The text-statistics annotator publishes fixed-width counts keyed to one
  verified normalized partition. It binds the exact document order and
  IDs, supports bounded inspection, and reuses completed evidence across
  release view changes. Release policy can adopt this artifact in a later
  increment.
- **Coherent packing first.** Compare seeded random, source-coherent,
  lexical-topic, explicit-entity and link-neighborhood schedules using
  identical natural-language bytes and exposure budgets. Assign semantic
  families to splits before building indexes or neighborhoods. Preserve
  boundaries, attention, and supervision masks through the normal loader.
  Source-coherent ordering now derives a schedule from a verified baseline
  selection. The paired arm keeps exact record and range membership.
  Lexical-topic ordering uses a reusable TRAIN-only index of bounded
  per-document word features. It uses the same schedule writer and records
  grouping coverage while retaining the original token segments.
  Source-link ordering ingests digest-pinned directed edges against the
  verified TRAIN family scope. The retained index supports bounded
  outgoing-link queries and one-hop grouping through the same writer.
  Entity ordering uses explicit namespaced memberships and the same scoped
  index lifecycle. Its bounded entity-to-document queries also cover
  entities beyond the packing anchor cap.
  Add shared-mechanism neighborhoods once mechanisms have typed outputs.
- **Structured sources and compilation.** Start with pinned Wikipedia
  links and categories, bounded Wikidata statements with qualifiers and
  references, and deterministic miniature worlds. Compile triples and
  seeded graph traversals first. Add infoboxes, explicit linguistic
  relationships, scientific observations, software ASTs and formal
  definitions through separate source adapters as evidence warrants.
  Use hierarchy and shared-property factorization before motif discovery
  or description-length experiments. Add bounded symbolic derivations,
  procedural trajectories, constraints, counterfactuals and mixed
  curricula as separate targets. Use the same canonical inputs when
  comparing representations. Verify lossless transformations by round trip
  or semantic equivalence. A lossy compiler records omitted information
  and reason while evidence and qualifiers remain in canonical artifacts.
- **Provenance.** Record known human-authored material, structured sources,
  deterministic compilations, formal derivations, procedural simulations,
  teacher-generated data, and unknown or mixed upstream origins. Recipes
  can select by those classes. The baseline compilers use parsers,
  deterministic algorithms, specified seeds, solvers, or simulations
  without a teacher model. Upstream provenance remains a separate claim.
- **Training.** Seal reusable token segments through the existing content
  encoder. Key the encoding on exact serialized bytes and all relevant
  tokenizer behavior and options. Schedule and pack references, attaching
  boundaries and masks where required. Train with the normal next-token
  objective. Keep contiguous materialization as an option for the same
  schedule. Record packing, representation and exposure in run lineage.
  Evaluate uint16 separately with a vocabulary-boundary and decoding check.

## Verification and performance

Report byte integrity, lineage, structural validation, independently
checked semantic correctness, transformation fidelity, sampled semantic
audit, and learning outcomes separately. Continue to use parallel hash
manifests and host-local verification receipts. A stat match permits reuse
only within the current verifier's trust rules. Keep full audit available.
For new typed artifacts, add a seeded sample with published scope and
findings before changing routine replay defaults.

Make reference releases immutable through sealed manifests and verified
payloads. Roots include retained releases, prepared bundles, retained runs,
and in-flight publication. Prune walks the full dependency closure, previews
reclaimable paths and bytes, and uses the publisher's locks before removal.
Keep acquired source data outside automatic cleanup. Validate the closure
even when original project pointers have moved.
Native build and release receipts already include the parent snapshot
required to replay each declared retained subset. The dependency comes from
the pinned source declaration in the build identity.
Streaming replay checks lineage, stage and view coverage over the verified
document row IDs with a single reusable file-backed bitmap. The source and
document indexes still provide the semantic checks and eligibility counts.
Measured coverage scratch and sampled full-audit peaks are recorded in the
[Corpus guide](../corpus-preparation.md#streaming-replay-coverage-measurements).
The streaming verifier carries compact duplicate keys and representatives
from the document scan into one grouped check. The checked index keeps the
document references for subsequent replay. The grouping input is temporary.
[Streaming duplicate verification measurements](../corpus-preparation.md#streaming-duplicate-verification-measurements)
cover the isolated grouping and the full cold audit.
The span and document checkpoints bind pinned file rights, source identities
and exact row generations. Raw replay reads those checked facts and the
original source bytes. It uses the document and span payloads during index
construction. [Raw replay measurements](../corpus-preparation.md#raw-replay-from-checked-index-facts)
report full audit costs and the small increase in sampled named scratch.
The checked document index clusters raw replay facts by source position.
The build sorts narrow temporary records once, and repeated replay scans
the resulting table in storage order. [Source-ordered replay measurements](../corpus-preparation.md#source-ordered-replay-index-measurements)
include the cold scan, index construction and full audit trade-offs.
Reviewed admission, fractional releases and chat use disposable indexes of
file ranges and compact document metadata. Their source-order join sorts
locators, then reads the verified payloads from their original members.
Point lookups retain the full evidence semantics for generated records and
citations. [Reviewed release range index measurements](../corpus-preparation.md#reviewed-release-range-index-measurements)
cover cold and warm audits of two file-admitted release sizes.

| Workload | Evidence |
| --- | --- |
| Workshop HDD incident, 57 GB and 6.7M documents | #450 reports a 4h40m freeze. Audit was stopped after 1h45m with 160 GB scratch and about 1.6 TB read. Indexing measured 3.4k docs/s on HDD and 19.8k on NVMe. |
| Later Workshop NVMe workload, approximately 12B tokens | #493 reports a 374 GB, 92-minute freeze copy, a 227 GB wide index, and about 1.3 TB peak-scale storage accounting. Its 30–45 minute and roughly 0.4 TB end-state figures are estimates. |
| Receipt and freeze changes | #500 reports roughly 1 second for a warm verification of the real release. #501 reports synthetic 13.2 GB freeze improving from 158.6 to 72.3 seconds. |
| Subsequent profiling | #499 measures seeded sampling on a warm synthetic 11.9 GB, 2M-document release. #502 measures synthetic 5M-row describe at 119.9 seconds before and 55.1 seconds after, without a tokenizer. #506 replaces the mixture-selection SQLite snapshot with mapped columns, but its merge and checkpoint footprint need scale checks. #521 reports a 94-minute real mixture audit and cheaper re-sealing with a narrower guarantee. |

Sources: [#450](https://github.com/supernovae/model-lab/issues/450),
[#493](https://github.com/supernovae/model-lab/pull/493),
[#499](https://github.com/supernovae/model-lab/pull/499),
[#500](https://github.com/supernovae/model-lab/pull/500),
[#501](https://github.com/supernovae/model-lab/pull/501),
[#502](https://github.com/supernovae/model-lab/pull/502),
[#506](https://github.com/supernovae/model-lab/pull/506),
[#521](https://github.com/supernovae/model-lab/pull/521).

Use the connected canary for correctness, a repeatable approximately
1M-document workload for profiling, and the historical Workshop workloads
for scale where the source data and host are available. Separate cold,
warm, one-partition append, and resumed runs. Record wall and CPU time,
throughput, peak RSS, peak scratch, I/O amplification, worker scaling,
reuse and recovery cost. Include storage and hardware details and compare
NVMe with slower storage where practical. Record inferred costs separately
from measurements. Require bounded memory and scratch, no release-sized
copy at reference freeze, and zero re-encoding after a schedule-only
change. Before a loader default changes, compare both modes on the same
schedule for batch waits, access locality, host memory, step throughput,
and accelerator utilization.

### Acquisition replay measurements

The [replay benchmark](../../tools/benchmark_acquisition_replay.py) injects
an interruption after the adapter finishes deriving its output and before
the staged snapshot verifier starts. It retries through the ordinary
acquisition entry point and times through publication. Run it in the Lab
development environment with the command below. Use `--adapter git` to
run one case or `--scratch-dir` to test another filesystem. Each output line
is a JSON record with timings, request and subprocess counts, payload identity
and staging bytes where the adapter records them.

```sh
uv run --locked --no-sync python tools/benchmark_acquisition_replay.py
```

The baseline table reports the median of three retries for each configuration
at Lab `1b4e31a4` on tmpfs with Python 3.14.8 and a 24-core AMD Threadripper PRO
9965WX.
Payload digests were stable across repetitions and across cache settings
for the archive and dumps. Those fixtures contain 256 selected text records
of 64 KiB each. The text is
hexadecimal output from Python's `Random(485).randbytes`. The gzip and
bzip2 fixtures use compression levels 1 and 3 respectively. The other file
fixtures use 64 files or rows of 256 KiB each. The HTTP fixture combines
them in one document.
The HTTP responses and the Hub file lookup are local fixtures. Git fetches
from a local pinned upstream. The measured times cover local replay and
verification.

| Adapter | Input / output | Retry without Lab transport cache | Retry with warm Lab transport cache |
| --- | --- | --- | --- |
| Archive, gzip tar | 9.37 MiB compressed, 16.00 MiB emitted | 0.400 s, one body request | 0.399 s, zero requests |
| MediaWiki, gzip XML | 9.38 MiB compressed, 16.07 MiB emitted | 0.761 s, one body request | 0.748 s, zero requests |
| Wikimedia, bzip2 XML | 8.17 MiB compressed, 16.08 MiB emitted | 1.117 s, body and checksum requests | 1.116 s, zero requests |
| Git include | 16.00 MiB emitted | 0.880 s, one local fetch and all selected files extracted | 0.292 s, zero fetches and all selected files extracted |
| Hugging Face include | 16.00 MiB emitted | 0.063 s, one Hub client call against the fixture and all selected files validated and copied | Outside the Lab transport cache |
| Local files | 16.00 MiB emitted | 0.059 s, all files copied | Outside the Lab transport cache |
| Pinned HTTP document | 16.00 MiB emitted | 0.038 s, one body request | 0.037 s, zero requests |

The same harness compared Git at `1b4e31a4` with the batched reader on
tmpfs. For 64 files, each retry went from 128 `git cat-file` subprocesses
to one. In three trials per configuration, the median went from 0.897 s to
0.773 s without a cache and from 0.287 s to 0.171 s with a warm cache. The
payload inventory digest matched across builds and cache settings. The
cached Git reader computes the blob hash while copying. On a Wikimedia
cache miss, the SHA-1 check likewise shares the copy pass with SHA-256.

Archive and dump retries wrote the complete derived payload again in both
cache configurations. Their peak staged file bytes were 25.37 MiB for the
archive, 25.45 MiB for MediaWiki and 24.25 MiB for Wikimedia. Those figures
include the raw body and output as they coexist in staging. Network
latency, NVMe and HDD replay, and retained-subset replay time were outside
this measurement. The Workshop base recipe at `4164998` has 15 Git sources
and 18 Hugging Face sources. The dump and archive figures describe other
adapter paths and should be measured again with a representative source
before setting a scale target.

The retained Workshop acquisition lock is **4,120,395,181 bytes**, measured
from file metadata. The former loop read it once per consulted source.
Across 33 consultations, that is about 136 GB of calculated logical input.
Acquisition now reads the advisory lock at most once and falls back to
verified immutable snapshots if that read fails, including on memory
exhaustion. Acquisition then reads and verifies the candidate lock before
atomically publishing it. Independent audits verify the published lock. The
peak RSS, physical I/O and time of reading the production lock remain
unmeasured. Keep wide metadata and production measurements tracked in
[#450](https://github.com/supernovae/model-lab/issues/450).

Retained copies now use the final candidate-lock check for the origin
recheck before publication. On ext4 over local NVMe, copying a new retained
Git snapshot in cold verification mode read one fewer full payload than
`11b1b9a1`. The medians below are from three runs per revision and fixture
with the origin already acquired. Reads are Linux process `rchar` counters.

| Synthetic payload | Logical reads before / after | Elapsed before / after | Peak RSS before / after |
| --- | --- | --- | --- |
| 128 MiB in one file | 768.50 / 640.34 MiB | 0.411 / 0.354 s | 72.5 / 72.5 MiB |
| 128 MiB across 512 files | 769.00 / 640.76 MiB | 1.194 / 1.091 s | 77.4 / 77.4 MiB |

Snapshot identities matched across revisions. Origin hashing went from
384 to 256 MiB for both fixtures. The copy still hashes 256 MiB across
staging and final validation. Separate cold audits continue to read the
origin and copy. Production-scale and rotational-disk timing remain
unmeasured. [#485](https://github.com/supernovae/model-lab/issues/485)

Archive and MediaWiki acquisition now retain the verified body and seal the
derived output before publication. Using the same fixture on tmpfs, the
median retry without a transport cache was approximately 0.14 s for the
archive and 0.08 s for MediaWiki across three trials each. Both retried
without requests, and both matched the baseline payload inventory digests.
The timed retry verifies staged output and publishes it. A stop inside a
compressed stream still replays parsing from the retained body. The
[source recovery inventory](../corpus-preparation.md#recovery-by-source-adapter)
records the current replay boundary for every adapter. [#485](https://github.com/supernovae/model-lab/issues/485)

On the same 16 MiB archive fixture, a source-step retry from a finished
derivation reduced payload bytes hashed from 48 MiB to 16 MiB. A retry from
the body reduced that count from 50.74 MiB to 25.37 MiB. These counters cover
SHA-256 reads in one source-step call. They exclude extraction reads and
later verification in a separate command.

Wikimedia uses the same recovery boundary. A retry also checks a completed
download that was interrupted before promotion. On the 256-page bzip2
fixture on tmpfs, the follow-up to `b89e2ca1` reduced the three-trial median
after a halfway XML stop from 1.085 s to 0.744 s. Retry from a finished
derivation went from 0.051 s to 0.045 s. The extraction shortcut skips the
per-letter language count for ASCII pages. Output hashing shares the write
pass, and cold verification checks file and page digests in one read. The
payload digest matched across builds. Both interruption points retried
without requests and yielded identical snapshots within each build, with
and without the transport cache. Pass
`--adapter wikimedia_dump --interrupt-at mid-xml` to rerun the halfway stop.

Mid-stream replay still decompresses and parses the full input.
[#587](https://github.com/supernovae/model-lab/issues/587) tracks verified
compressed ranges, ordered checkpoints, and worker scaling on a
representative dump. Wikimedia's index marks compressed streams within
one XML document, so range extraction must preserve the shared XML
context. [Wikimedia multistream format](https://meta.wikimedia.org/wiki/Data_dumps/Dump_format#Multistream_dumps)

The [Wikimedia range planner](../corpus-preparation.md#verify-wikimedia-multistream-ranges)
validates pinned local dump and index bytes, physical bzip2 member boundaries,
shared XML context and ordered page IDs. Its metadata is proportional to the
number of compressed members. The planner emits no decoded payload. The
indexed acquisition path uses those offsets to checkpoint selected JSONL
in source order. A retry verifies the saved range plan and member outputs.
Only unfinished or damaged members need to be decoded again.
On tmpfs, three runs of the planner on 256 synthetic pages with separate
header and footer members took a median 0.75 s for 16 MiB of page text
and 1.98 s for 64 MiB. Peak RSS was about 81 MiB in both cases, including
CLI startup. These are serial planning measurements on synthetic input.

With an index declared, the 16 MiB text fixture's three-trial cold median
was 1.184 s, compared with 0.750 s for serial acquisition of the same
multistream body. Cold indexed acquisition also verifies the range plan.
After interruption following the first 100-page checkpoint, the median
retry was 0.495 s with zero requests. It decoded the remaining 156 pages,
returned 9.77 MiB of expanded data to the extractor, and peaked at
40.36 MiB of staged files while assembling the snapshot. Output digests
matched serial acquisition.

Indexed acquisition can now overlap bzip2 decoding with ordered XML
selection and commits. It uses the shared shard runner, an explicit
extra buffer budget, and the existing member checkpoints. The cold range
planner is still serial. Speculative workers write no stage files. Changing
the worker count leaves the acquisition identity unchanged.

Three trials per setting on tmpfs used a 64 MiB prefetch allowance for
three workers. The resumed run started after one of three page members
committed, with no transport cache configured. Times include range
verification on a cold acquisition. Peak staging and decoded bytes are
from the acquisition receipts.

| Synthetic page text | Cold, 1 / 3 workers | Resumed, 1 / 3 workers | Retry expanded | Peak staging, both |
| --- | --- | --- | --- | --- |
| 16 MiB | 1.192 / 0.947 s | 0.494 / 0.405 s | 9.77 MiB | 40.36 MiB |
| 64 MiB | 4.708 / 3.773 s | 1.957 / 1.608 s | 39.02 MiB | 160.83 MiB |

Both worker counts produced identical payload and snapshot digests with
zero retry requests. The benchmark process peaked at 174 / 193 MiB RSS
for the 16 MiB fixture and 458 / 514 MiB for 64 MiB, including in-memory
fixture construction and transport. Run `tools/benchmark_acquisition_replay.py`
with `--adapter wikimedia_dump --interrupt-at member-checkpoint` and
compare `--workers 1` with `--workers 3 --wikimedia-prefetch-bytes 67108864`.

The first dated English Wikipedia shard (`p1p41242`, 2026-11-01,
287.04 MiB compressed) was also replayed from pinned local files on
NVMe ext4. Each run used a fresh work root or resumed its own hard stop.
The operating system page cache was left in place. The recipe capped
selection at 5,000 pages and scanning at 24,000. The selected prefix used
82 page members and produced 83.96 MiB of JSONL. The interrupted runs
received `SIGKILL` after the 25th member commit. [Dated dump inventory](https://dumps.wikimedia.org/enwiki/20261101/)

| Real shard, one run per setting | 1 worker | 3 workers, 64 MiB prefetch |
| --- | --- | --- |
| Cold acquisition | 56.67 s, 185 MiB peak RSS | 46.26 s, 199 MiB peak RSS |
| Time to hard stop, including startup | 35.09 s, 154 MiB peak RSS | 35.29 s, 164 MiB peak RSS |
| Resumed acquisition | 19.42 s, 174 MiB peak RSS | 13.53 s, 193 MiB peak RSS |
| Retry expanded bytes | 237.92 MiB | 245.63 MiB |
| Peak staging, completed attempts | 455.89 MiB | 455.89 MiB |

All four completed attempts had identical snapshot and payload digests
and the same ordered page and revision IDs. Both resumed without
transport requests. Three workers expanded about 7.7 MiB of additional
input speculatively near the selection boundary. Cold range verification
and per-member commits remain part of the elapsed time. Parallel
decoding stays opt-in because the improvement and memory cost depend on
the input and storage. Slower storage and wider ranges remain unmeasured.

The #590 review measured 3.3 to 3.5 s in serial XML selection on a
160-member synthetic dump. Three workers reduced total time from about
7.4 to 6.1 s, and eight workers gave no further measured gain. Bounded
process-based XML selection and shared commits for consecutive extraction
members remain on the resumability checklist. [Performance review](https://github.com/supernovae/model-lab/pull/590#discussion_r4688320386) · [Resumability follow-ups](https://github.com/supernovae/model-lab/issues/485)

Indexed acquisition now checkpoints whole-dump verification every 64
physical members. Each receipt binds the dump and index digests, range
bounds, and XML context at the group boundary. The temporary payloads
are range metadata. On retry, acquisition verifies the dump pins in one
pass. The planner reuses that check if the file fingerprint still matches,
and reads the entire index. It can restore the parser with the bounded
XML header and skip decoding verified groups. The public range planner
continues to run read-only.

On the same 287 MiB official shard, a single-worker acquisition took
50.38 s from a fresh work root. A retry after `SIGKILL` following two
64-member verification commits took 37.38 s. It reused two of five
groups, avoided 539.88 MiB of decompression, and made zero transport
requests. Both runs used about 185 MiB peak RSS and 455.96 MiB of logical
staging. Snapshot, payload, and ordered page and revision IDs matched.
The plan digest also matched the earlier continuous scan. The files were
local on NVMe with the operating system page cache left in place.

On tmpfs, three trials on 16 MiB of synthetic page text stopped after
the first group of a 130-member dump. An 8-page selection took a median
0.422 s cold and 0.234 s on retry. Selecting all 256 pages took 1.160 s
cold and 0.974 s on retry. Both selections matched their uninterrupted
snapshot digests with zero retry requests. Run
`tools/benchmark_acquisition_replay.py --adapter wikimedia_dump --interrupt-at plan-group`
with `--selected-pages 8` or `--selected-pages 256` to reproduce the fixtures.

Across five retries of the 8-page fixture, passing the checked dump to the
planner reduced full-body hash reads from 17,163,290 to 8,581,645 bytes
per attempt. Plan and payload digests matched the two-pass baseline.

On a cold indexed acquisition, Lab now stages the selected prefix as
verification groups commit. A later verification failure prevents
snapshot publication. An interruption after staging a member preserves
that output even if the full range plan is unfinished. Selection and
verification batches still run in order. The whole-dump validation cost
therefore remains part of the successful acquisition time.

Three paired trials per setting on the 16 MiB synthetic fixture measured
the wall time to the first committed extraction member and to a completed
acquisition. The baseline completes the range plan before extraction.

| Selection | Workers | First member, baseline / prefix | Total cold, baseline / prefix |
| --- | ---: | ---: | ---: |
| 8 pages | 1 | 0.397 / 0.203 s | 0.418 / 0.417 s |
| 256 pages | 1 | 0.393 / 0.202 s | 1.159 / 1.171 s |
| 8 pages | 3 | 0.396 / 0.203 s | 0.413 / 0.411 s |
| 256 pages | 3 | 0.399 / 0.204 s | 0.839 / 0.846 s |

The committed prefix was available about twice as early. Total cold
times differed by at most 12 ms. Payload digests matched at both worker
counts. The full-selection fixture used 29,425 more logical staging
bytes for the member range metadata in the receipts. The replay tool's
`median_cold_first_member_seconds` reports this checkpoint latency.

On the pinned 287 MiB English Wikipedia shard, time to the first durable
extraction member was 26.57 s on the baseline and 7.19 s with prefix
staging. Total acquisition was 50.40 and 48.68 s. Those are single runs
with one worker and local inputs on NVMe. Peak RSS was 193 and 195 MiB.
Plan digest, payload bytes, and ordered page and revision IDs matched.

Indexed acquisition can also send verified members to the process runner
for XML parsing. `--wikimedia-prefetch-bytes` bounds decoded lookahead.
`--wikimedia-parse-bytes` bounds candidate JSONL across the active worker
window. The coordinator applies the remaining limits in order from row
positions and writes the selected member receipts. Candidates stay in
memory. An exhausted candidate budget or a speculative parser error leads
to ordered serial extraction if that member is still required. The shared
runner's process workers exit when the coordinator dies.

Three trials per selection on the 16 MiB synthetic fixture used three
workers and a 64 MiB decode budget, with a 64 MiB candidate budget for
process parsing. The replay tool stopped after the first verification group.

| Selection | Cold, threaded / process | Retry, threaded / process | Peak process-tree RSS, threaded / process |
| --- | ---: | ---: | ---: |
| 8 pages | 0.412 / 0.645 s | 0.223 / 0.465 s | 157 / 317 MiB |
| 256 pages | 0.871 / 1.226 s | 0.674 / 1.024 s | 157 / 328 MiB |

RSS is the sampled sum across the benchmark and its children, including
fixture construction and shared pages. Snapshot digests and staging
bytes matched. The short batches took longer with process startup and
candidate transfer. Run `tools/benchmark_acquisition_replay.py` with
`--adapter wikimedia_dump --interrupt-at plan-group --workers 3`,
`--wikimedia-prefetch-bytes 67108864`, and compare
`--wikimedia-parse-bytes 0` and `--wikimedia-parse-bytes 67108864`.

On the pinned 287 MiB English Wikipedia shard, three process workers
finished the cold acquisition in 40.63 s versus 43.73 s with threaded
decode. The first extraction checkpoint took 11.57 s versus 7.52 s.
Sampled process-tree RSS was 590 versus 205 MiB. Both runs used 455.97 MiB
of peak logical staging and produced the same snapshot, payload, and
ordered page and revision IDs. These are single runs on NVMe with local
inputs. Process parsing stays opt-in because startup and candidate transfer
costs vary with member size, and the larger run used more host memory.

Indexed extraction can group consecutive members into a single JSONL
payload and receipt. A retry verifies the payload hash while copying it
to the staged snapshot. A failed receipt discards the provisional copy
before rebuilding its group. This leaves the hash check and atomic
publication in the existing stage store.

On the 287 MiB English Wikipedia shard, one cold NVMe run per setting
compared single-member commits with groups of up to eight. Acquisition
made 1,217 and 720 `fsync` calls and used 357 and 73 peak staging inodes.
The first durable extraction commit took 11.31 and 13.41 s. Total times
were 54.85 and 51.33 s. Snapshot, payload, and ordered page and revision
IDs matched. The single-member default retains the finer recovery unit.

On tmpfs, three retries per setting on the 16 MiB synthetic fixture
started after the same eight extracted members were committed. Median
retry time was 0.934 and 0.901 s, with 876 and 148 `fsync` calls. In both
settings, verification shared the copy of 1,053,781 committed payload
bytes. These timings do not characterize rotational storage.

A separate hard stop before publishing the second eight-member group
kept the first group, rebuilt eight uncommitted members, and decoded one
new member. The retry made zero transport requests and produced the same
snapshot as a fresh single-member run. The replay tool accepts
`--wikimedia-checkpoint-members` for grouped extraction measurements.

To reproduce the adapter-upgrade recovery case in
[#591](https://github.com/supernovae/model-lab/issues/591), an indexed fixture
was stopped with SIGKILL after the body was committed and before extraction.
Each retry used a new adapter identity. The fixture had 32 MiB of generated
text, a 17,121,236-byte compressed body, and local HTTP fixtures without a
transport cache. On NVMe, the baseline at `5ef6af68` made three requests.
The retry that moved the verified body made only the index request. At the
start of extraction, combined compressed bodies across staging generations
occupied 34,242,472 bytes on the baseline and 17,121,236 bytes with reuse.
One run per case took 1.466 and 1.345 s respectively. A separate run with an
altered older body fetched the dump again and left that older body intact.
All three snapshots had the same digest. These timings use local fixtures.

## Relational Knowledge Microlab

Keep named recipes and study outputs in Workshop. Start with deterministic
miniature worlds with known entities, rules and answers. Add bounded
astronomy data and software metadata or AST-derived dependency and type
relationships to test transfer. Use the same small architecture, paired
seeds and fixed evaluation inputs. Run separate studies matching training
tokens, source-information coverage and measured compute. Lab comparisons
record declared training-data changes and require identical held-out
inputs and evaluation protocols.

Use ordinary natural language with random packing as the baseline and
the same bytes with coherent packing as the first comparison. Then add
relational serialization, graph traversal, hierarchy and factorization,
executable mechanisms and trajectories, and a mixed curriculum as their
operators land. Evaluate direct and inverse recall, held-out entities and
combinations, multi-hop composition, rule application, counterfactuals,
overrides, boundary conditions, calibration, transfer, expression, and
forgetting. Use exact answers, executable outcomes or formal checks where
available. Keep semantic families intact across training and evaluation,
and keep an untouched final evaluation for a chosen representation.
Report learning effectiveness per parameter, token and compute with
negative results. Use existing instrumentation for interference or gradient
studies when it supplies a valid measurement. Promote a representation to
a foundational Workshop recipe after replicated evidence and measured
pipeline costs.

## Delivery order and cutover

1. Bound and resume long stages under [#485](https://github.com/supernovae/model-lab/issues/485),
   then remove wide metadata and verifier paths under [#450](https://github.com/supernovae/model-lab/issues/450).
2. Introduce shared artifacts and reference releases with planning,
   retention and source/lineage inspection. [#567](https://github.com/supernovae/model-lab/issues/567)
3. Add annotation, selection and duplicate evidence, and seal token segments
   for referenced and contiguous loading. [Sidecars #568](https://github.com/supernovae/model-lab/issues/568) ·
   [Schedules #569](https://github.com/supernovae/model-lab/issues/569)
4. Ship coherent packing, then structured adapters and relational
   compilation. Start the microlab with the packing comparison.
   [Packing #570](https://github.com/supernovae/model-lab/issues/570) ·
   [Relational compiler #571](https://github.com/supernovae/model-lab/issues/571) ·
   [Workshop #118](https://github.com/supernovae/kybern-workshop/issues/118)
5. Add hierarchy and factorization, then bounded symbolic and procedural
   compilation. Expand microlab arms as each lands. After sidecars and
   structured records exercise the operator contracts, add the authorized
   command extension independently. [Factorization #572](https://github.com/supernovae/model-lab/issues/572) ·
   [Procedural compilation #574](https://github.com/supernovae/model-lab/issues/574) ·
   [Command extension #573](https://github.com/supernovae/model-lab/issues/573)

Land machine-readable inspection and current documentation in every
increment. Extend the existing dashboard with source and storage use,
lineage, annotation and duplicate summaries, bounded or paginated
relationship queries, representation and compression counts, and links
from run results to the recipe and artifact identities.

Each format change switches all readers and writers together and names
the artifacts to rebuild. Reuse verified transport and snapshots where
their contracts still match. Keep existing runs on their pinned
revisions. Plan one coordinated rebuild of normalized artifacts and
releases when the shared reference layout changes. Refit a tokenizer only
when its sample or fit inputs change, and rebuild token segments when
serialized bytes or encoding behavior change. Rebuild schedules and
prepared views when their contracts change. Report the rebuild cost
before switching Workshop recipes.

Use the existing connected Corpus recipe test and the smallest focused
tests for drift, corrupt data, budget refusal, concurrent publication,
protected pruning, worker-count determinism and recovery. Every delivery
must preserve ordinary Corpus preparation and small local training.
