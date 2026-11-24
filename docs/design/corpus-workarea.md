# Corpus work area and knowledge compilation

Corpus prepares reusable training information from pinned sources. The same
work area serves conventional natural-language pretraining and experiments
with relational, hierarchical, symbolic, and procedural representations.
An ordinary text recipe can still go from acquisition to training without
any structured extraction. Small datasets can still use the
[local data path](../datasets.md#prepare).

The inventory below is based on Lab `b53022b2` and Workshop `4164998`.
The interfaces in the design sections describe future work. Current commands
are in [Corpus preparation](../corpus-preparation.md) and
[Token store](../token-store.md). The issues are
[#459](https://github.com/supernovae/model-lab/issues/459) for the design,
[#450](https://github.com/supernovae/model-lab/issues/450) for bounded build and
verification, and [#485](https://github.com/supernovae/model-lab/issues/485)
for recovery. The [rolling backlog](https://github.com/supernovae/model-lab/issues/497)
records the decisions from [the original design](https://github.com/supernovae/model-lab/pull/493).

## Current implementation

| Original increment | Evidence on main | Remaining work |
| --- | --- | --- |
| 1. ShardRunner and verification | #494 and #495 landed the shared range runner and early consumers. It bounds submitted work, can checkpoint JSON results, and supports ordered output. | Admit work by memory and scratch, bound buffered results, and checkpoint large payloads and reductions. |
| 2. Near-duplicate screen | #491 and #496 landed checkpointing, sharded scans, and template-aware classes. | Reuse the held-out index and template evidence as artifacts. Add passage-level grouping and measure classification precision. |
| 3. Tokenizer sampling | #494 and #499 landed range scans and seeded-hash sampling. Workshop #112 pins the sample seed. | Keep this path. Track fit and cold sample-verification costs separately from sample selection. |
| 4. Acquisition parts | #456 landed the verified transport cache. Supported bodies can be reused after a source declaration changes. | Publish independently receipted parts for eligible acquisition paths so a stop can resume within a source. |
| 5. Narrow build | #486 checkpoints complete disk-build index and dedup stages. #488 landed the canary. The disk builder still stores payloads and spans in SQLite and can replay the entire view-emission pass after a stop. | Replace the wide tables with narrow metadata and references. Checkpoint view segments. |
| 6. Freeze and verification | #500–#502 landed signed receipt reuse, freeze by verified rename, and resumable describe measurement. Cold verification still performs full checks. Some LM and chat releases still use the row-store verifier. | Bound the remaining row-store paths, including reviewed admission and fractional releases. Separate integrity from transformation replay and sampled audits. |
| 7. Shared work area | `StageStore`, normalized shards, and token caches exist. Most source and build artifacts are still project-scoped. | Share immutable outputs across projects, narrow identities, and protect reference closure during explicit pruning. |
| 8–9. Operators and sidecars | #448/#451 moved choice into recorded release policy. Exact deduplication and the approximate screen exist. The runner declares stages but there is no general typed operator registry or sidecar store. | Extract reusable evidence and make selection consume sidecars without rewriting source text. |
| 10. Reference releases and mixtures | #503–#507 improved quota checks, mixture indexing, and the ordered writer. #519–#523 and #557 improved reuse and verification keys. `ContentTokens` holds reusable int32 encodings but is a recoverable append-only cache. Mixture preparation still makes a checked contiguous copy. | Make releases reference immutable artifacts, seal reusable token segments, and train from schedules over those segments. |
| 11–12. Planner and external operators | Existing CLI, journals, preflight and dashboard supply entry points. | Add dependency plans, bounded inspection and explicitly authorized command operators. |
| 13. Training preemption | #533 added opt-in device-layout changes under runtime optimization. #535/#537 added finished-run data cleanup and reuse. | Trainer recovery stays with #485 and #516. Corpus tracks data and schedule identity so those changes can proceed independently. |

Workshop's [foundational corpus](https://github.com/supernovae/kybern-workshop/pull/112)
uses source-family splits, seeded tokenizer sampling, and mixture declarations.
Its [additive release](https://github.com/supernovae/kybern-workshop/issues/109)
can proceed through the existing workflow, with independent rights, split,
contamination, and quota checks. Retain the screening and reporting decisions
for the existing base release. Keep active training on its pinned checkout.

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

Extend `ShardRunner` and `StageStore` together. Start with one partitioned
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
- **Coherent packing first.** Compare seeded random, source-coherent,
  lexical-topic, explicit-entity and link-neighborhood schedules using
  identical natural-language bytes and exposure budgets. Assign semantic
  families to splits before building indexes or neighborhoods. Preserve
  boundaries, attention, and supervision masks through the normal loader.
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
