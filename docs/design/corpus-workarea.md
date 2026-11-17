# Design: the corpus work area

Status: proposal for owner review. Nothing here is implemented. Issues:
[#459](https://github.com/supernovae/model-lab/issues/459) (scalable corpus
pipeline), [#485](https://github.com/supernovae/model-lab/issues/485)
(resumability contract), [#450](https://github.com/supernovae/model-lab/issues/450),
[#448](https://github.com/supernovae/model-lab/issues/448). Workshop
[#109](https://github.com/supernovae/kybern-workshop/issues/109) is the second
user of the machinery.

Marking: **Decided** means the evidence is strong enough that the design
commits to it unless the owner objects. **Owner decides** points to section 9.
Numbers tagged **[M]** were measured in a 12B-token run (6.7M, then 20.9M
documents, 48 cores, NVMe, hashing about 3.5 GB/s). Numbers tagged **[E]** are
estimates from those measurements and from reading the code; they carry roughly
a factor of two of uncertainty and are the first thing each PR must replace
with a measurement.

## 0. Summary

1. Lab has good pieces and the wrong shape. Stage outputs are already
   content-keyed and receipted (`build_stages.py`, normalized shards, the
   transport cache, the content token store), but they live under
   `corpora/<project-id>/`, so a second project cannot see them; the release
   is a 374 GB copy of a 365 GB build that is in turn derived from 108 GB of
   prepared shards plus a 227 GB index; and every long stage that is not a
   shard-per-file loop runs on about one core.
2. The design is one shared, content-addressed **store** under the work root
   (`WORK/store/`) holding every expensive layer, keyed by exactly its inputs.
   A project is a recipe plus named pointers into the store. Releases and
   mixtures become small manifests (references plus keep-lists and
   schedules), not copies.
3. Work is split into **shards** and every stage runs through one
   **`ShardRunner`** primitive. A shard is both the unit of parallelism and
   the unit of checkpoint, so resumability (#485) and 48-core use come from the
   same mechanism. A stage that stays single-core must say why in its
   declaration; a test enforces it.
4. Five operator kinds (`annotate`, `select`, `dedupe`, `mix`, `tokenize`) with
   named, registered implementations and declared parameters. Their resolved
   options and identity are recorded in what they produce. A user script joins
   through a declared-inputs/declared-outputs `command` operator. No DAG
   engine, no schema per model type.
5. Annotations are sidecars; boilerplate is annotated and content similarity is
   scored on text with the boilerplate removed, so well-run projects keep their
   issue templates and license headers while a held-out copy is still caught.
6. Expected effect at 12B-token scale **[E]**: a build that takes 4.5 h plus 92
   min freeze plus 41 min audit plus 162 min near-duplicate screen plus 114
   min tokenizer scan (about 11 h of wall time, and about 1.3 TB of disk in total) becomes roughly 30 to 45 minutes of wall time and about 0.4 TB. A
   second project over the same data costs minutes and almost no disk.

## 1. The physical model

### 1.1 What is wrong today (evidence)

- **Layers are project-private.** Snapshots are
  `corpora/<project.config.id>/snapshots/<source>/<snapshot_sha256>`
  (`acquisition.py`), prepared shards are
  `corpora/<id>/builds/prepared/normalized-source-v2/`, stage outputs are
  `corpora/<id>/builds/stages/`. The only cross-project mechanism is a snapshot
  alias (a symlink with a cold readback) and `retained_subset`. A recipe that
  uses 10 of the run's 33 sources under another project id re-acquires,
  re-verifies and re-normalizes them.
- **Identities are stronger than the reuse they allow, and in one place
  stronger than necessary.** The snapshot key
  (`declaration_sha256` over id, kind, URI, revision and the acquisition block)
  is exactly right and project-independent. The normalized-shard key
  (`_receipt_identity` in `large_build.py`) includes the **whole** source
  declaration (`source.model_dump`, so `notes` and unrelated rights text) and
  the SHA-256 of six implementation files, among them the 3,000-line
  `pipeline.py`. Any edit to those files, in any Lab upgrade, invalidates every
  shard: the work area does not survive a Lab update, which is the opposite of
  "a durable work area to build from". The index key adds four more file
  digests.
- **Policy leaks into expensive layers.** `documents.exclusions:
  research_privacy` and `empty_text` act inside normalize, so changing the
  privacy rule re-normalizes everything; it should be an annotation plus a
  selection rule (#459 step 2).
- **Document identity binds the normalizer.** `document_id =
  digest([snapshot_id, location, raw_sha, normalizer.identity])`
  (`pipeline.py`). This answers #459's open question (section 5).
- **The release is a copy of a build, and the build holds the text three
  times.** Disk at 12B-token scale **[M]**: acquired 108 GB, prepared 108 GB,
  index 227 GB, build 365 GB (`documents.jsonl` 188 GB plus `lm/*.jsonl` views, spans and lineage; the
  split of the remaining 177 GB is inferred from the writer, not measured), release 374 GB, mixture 48 GB, per-run prepared
  copy about 48 GB. About 1.3 TB for 108 GB of source information. The release
  copy exists because a link "would let a later write to the build change the
  release" and because receipts refuse multiply linked files. `freeze` takes 92
  minutes **[M]** to move bytes that were already hashed.
- **Precedent for not copying exists.** `freeze_retained`
  (`completed_release.py`) publishes a release in place, and `copy_hashed`
  already uses reflinks where the filesystem has them. On ext4 NVMe (this
  host) neither helps the default path.
- **Mixtures copy.** `materialize-mixture` writes `train.tokens.ids`
  (12B tokens x 4 bytes = 48 GB). The content token store
  (`ContentTokens`, keyed by tokenizer SHA and document content SHA) is already
  the right reusable layer; the mixture should be a schedule over it.

### 1.2 The layers

One rule governs every layer: **the key is the digest of exactly the inputs
that determine the bytes, and nothing else**; the directory holds the data and
a `receipt.json` (inputs, file sizes and SHA-256, result values); it appears by
one atomic rename; it is trusted only after its receipt verifies; names and
modification times prove nothing. This is `StageStore` today, promoted from
"stage of one build" to "layer of the store".

```text
WORK/
  store/                               shared by every project in this work root
    transport/                         L0  (the existing KLAB_TRANSPORT_CACHE layout)
      bodies/<aa>/<sha256>  keys/<digest>.json  git/<digest>.git
    snapshots/<source_id>/<snapshot_sha256>/                         L1
    normalized/<source_id>/<shard_key>/                              L2
      docs.jsonl.zst  spans.jsonl.zst  rejected.jsonl  narrow.arrow  receipt.json
    annotations/<annotator>/<annotation_key>/<shard_key>/            L3
      part.parquet  receipt.json
    dedupe/<dedupe_key>/        groups.parquet  decisions.parquet  receipt.json   L4
    selections/<selection_key>/ keep-<shard_key>.arrow  receipt.json              L5
    tokenizers/<tokenizer_sha256>/                                   L7
    tokens/<tokenizer_sha256>/  tokens.ids  index.bin                L8 (ContentTokens)
    tmp/                         staging, never an input
  corpora/<project_id>/                a project is names and pointers
    lock.json                  acquisition lock (points at L1 digests)
    releases/<release_id>/     manifest.json  report.json  keep lists by reference   L6
    mixtures/<mixture_id>/     schedule.arrow  receipt.json                           L9
    runs-prepared/             per-run packed arrays (disposable)                     L10
  scratch/                     verification and sort scratch; deleted on exit
```

| Layer | Key: exactly what it depends on | Reusable by | Retention | Invalidated by |
| --- | --- | --- | --- | --- |
| L0 transport body | request (kind, URL, revision, pin) -> body SHA-256 | any project on the host; copyable between hosts (nothing in it is trusted) | cold storage; `cache-prune` removes only unverifiable | nothing (re-hashed at use) |
| L1 snapshot | `snapshot_sha256`: id, kind, canonical URI, revision, acquisition block, file digests | any project declaring the same source | kept while any release references it; the bulk of "what you paid to fetch" | an acquisition-field change makes a new snapshot; the old stays until pruned |
| L2 normalized shard | one input file range (`file_sha256`, byte range) + normalizer options digest + the per-file **resolved** rights decision + the document-policy fields that change text (`markdown_kind`, `row_files`) + operator `impl_version` of `normalize` | any project with the same normalizer and rights decision | kept; second largest layer | a normalizer option, a changed rights decision for that file, or a bump of `impl_version` |
| L3 annotation shard | L2 `shard_key` + annotator name + annotator `impl_version` + resolved parameters (+ keys of annotations it reads) | any project; text-scoped annotators also memoized by content (section 5) | kept; small (about 5 GB for the 12B-token set **[E]**) | a new L2 key, parameters, `impl_version` |
| L4 dedupe decisions | the sorted set of L2 keys **in scope** + dedupe operator + parameters + L3 keys it reads + split declaration + source ranks | same scope only (decisions are relative to the corpus) | newest N per scope; tiny (a few bytes per document) | any change of scope, parameters, splits or ranks; costs minutes (section 6) |
| L5 selection | L4 key + `select` operators + parameters + L3 keys they read | same scope | with its release | as L4 |
| L6 release | manifest digest over L5 key, split declaration, source declarations (rights, licenses), normalizer identity, publication mode, resolved operator list | the project; others read it by id | kept while any mixture or run references it | never mutated; a new input is a new release |
| L7 tokenizer | tokenizer file SHA-256; its training record names the sampled documents, config and release | any project that binds the SHA | kept | never mutated |
| L8 token arrays | (tokenizer SHA-256, document content SHA-256) per document (exists today) | any project, any mixture, any sequence length | evict by last use; the large reusable layer | tokenizer change only |
| L9 mixture | release id(s) + tokenizer SHA + quotas/weights + seed + exposures + packing | the project | with the runs that cite it | any of its inputs; **a schedule, megabytes, no token copy** |
| L10 prepared run data | L9 key + `seq_len` + packing | one run | disposable | any change |

**Never copied; referenced instead.** A release does not contain document
text. It contains, per L2 shard it uses, the shard key and a keep-list (row
ranges plus drop reasons from L4/L5), plus the report, the license report and
the recorded resolved options. Readers (`documents`, `sample`, `lineage`,
mixture, tokenizer sampling, near-duplicates) iterate "shard, then keep-list"
through one `ReleaseReader`. Because the manifest is the only file a release
owns, the "single hard link" receipt rule is satisfied, a release is a few
hundred megabytes at most, and `freeze` becomes hashing a manifest plus
verifying the referenced shard receipts (seconds warm; about 35 s cold for
108 GB at 3.5 GB/s). The same is true for a mixture: a schedule of
(document reference, repeat) rows. Tokens are gathered from L8 when a run's
data is prepared (L10), which stays a per-run copy because the trainer wants
one contiguous memory-mapped array; if the owner wants that gone too, the
loader needs an indirection and is a separate decision (D9).

This is the biggest change in the document and the one with the largest
refactor cost (section 8, PR 7). A cheaper intermediate is **decided** as the
first step: *freeze in place* (rename the verified build directory into
`releases/<id>`, mark it read-only, hash in parallel once), which removes the
374 GB copy and the 92 minutes without touching any reader. By-reference
releases (D1) are then a question of when, not whether, and they are the only
way to drop the 188 GB text duplicate and the 227 GB wide index.

### 1.3 Sharing safely

- **Concurrency.** Two projects computing the same key at once: each builds in
  `store/tmp`, then `rename_noreplace`; the loser verifies the winner's receipt
  and discards its own work (`StageStore.build` already does this). A per-key
  lock file lets the second project wait instead of duplicating an hour of
  work; waiting is the default and is shown in progress.
- **Trust.** Nothing in the store is trusted without its receipt (same stance
  as the transport cache). The proof store (`verification_proofs.py`, an HMAC
  key under `XDG_CONFIG_HOME`, stat fingerprints) remains host-local and is
  only a cache of "I hashed this already". A copied store is verified cold
  once: parallel hashing makes 116 GB (normalized plus annotations) about 35 s
  **[E]**.
- **Rights.** Snapshots hold bytes, not permission. Eligibility is decided per
  file and baked into the L2 key, and applied to a release from the project's
  own declarations, so a second project with a different license reading never
  silently inherits another project's decision. A gated or private source in
  the store is as readable as directory permissions make it; the store is never
  published.
- **Relocation.** `store/` is a directory of verified objects: copy or rsync
  it to a leased machine. What to ship is a money question; section 6.4.

### 1.4 What "second project" looks like

The 10-source variant: 10 of the 33 source files, `dedupe: component_ranked` plus MinHash,
a 16k tokenizer, different stratum weights, in the same work root:

| Step | Work | Cost **[E]** |
| --- | --- | --- |
| acquire | same source declarations -> same L1 keys -> hit; no network, no hashing beyond receipts | seconds |
| normalize | same normalizer, same rights decisions -> L2 hit | seconds |
| annotate | L3 hit for annotators with equal parameters; a new annotator runs only over the 10 sources | minutes if new |
| dedupe | new scope -> new L4 key; narrow keys only, in memory | under 1 min |
| select / release | keep-lists plus manifest | seconds |
| tokenizer | new SHA; sampling is a seeded-hash pass over the narrow table | about 5 min fit |
| tokens / mixture | L8 is keyed by tokenizer: miss for the 16k tokenizer, encode the selected documents once (1B tokens: minutes on 48 cores; the 12B-token set about 15 to 25 min); the weights change nothing in L8 | minutes |
| disk | about 3 to 5 GB of new bytes except the new token arrays | |

**Scale up** (add sources): `corpus acquire` fetches only the source whose
snapshot key is absent; normalize and annotate run for its shards only;
dedupe is re-run over all narrow keys (under a minute); the new release is a
new manifest. Splits: `prior_family_inventory` keeps existing family
assignments fixed, as today. **Scale down** (a 1B-token subset): a `select`
operator `sample` with `unit: token_budget`, `by: stratum`, and a seeded hash
of the document content SHA as rank, so a smaller subset is a strict prefix of
a larger one (**nested**): results at two scales are comparable and L8 token
arrays for the small subset are a subset of the large one's. Acquisition is
never re-run to scale down. `acquisition.retained_subset` and `hash_modulus`
remain for the case where the snapshot itself should be small (a held-out
source), and the recipe doc says which to use: shrink the snapshot only when
the full snapshot would be unwanted storage; shrink the selection otherwise.

### 1.5 The small-dataset test

A small dataset uses `klab data snapshot` + `tokenizer` + `data prepare` (3.5 GB
snapshot, 1.6 GB tokens, 93 minutes of training), not the corpus pipeline. The
design must not make that heavier, so:

- **Decided:** the corpus work area is *entered by choice*. `klab data
  snapshot` and the token store keep their current, flat layout; the tokens are
  already content-keyed (`TokenStore.open(source_sha256, tokenizer_sha256)`).
  `store/` is created on first use of a corpus command; nothing creates it for
  a `try`.
- **Decided:** every operator list has a default that equals today's policy
  (no annotators, `rights` plus `exact` dedupe), and `ShardRunner` runs inline
  below a size threshold (`PARALLEL_MIN_BYTES`, as `run_tasks` does), so a
  small corpus pays no pool start-up.
- A small-dataset user who later wants a mixture can declare the snapshot as a
  `kind: local` source; the design adds no step for that.
- Check: a 3.5 GB corpus runs through the work area in one shard-sized pass;
  receipts cost one JSON per layer. Disk overhead is the narrow table and
  receipts, below 1%.

## 2. Parallel by construction

**Rule (decided).** Every stage's unit of work is a *shard* that can run in
any order on a process pool. A shard has a key, writes its output to staging,
and is committed by a receipt. The scheduler skips shards whose receipt
verifies. Hence parallelism and resumability are one mechanism, and a stage
that cannot be sharded must say why.

### 2.1 The shared primitive: `ShardRunner`

One module (`klab/corpus/shards.py`), replacing the ad hoc pools that exist
today (`release_verify.run_tasks`, the `ProcessPoolExecutor` in
`large_build.py`, `preparation_workers.py`, `mixture_pipeline.ordered_fetch`,
`EncoderPool`): those differ in what they bound and none checkpoints.

```python
plan = ShardPlan(
    stage="annotate/language",
    inputs={...},                      # stage key inputs (operator, params, upstream keys)
    shards=[Shard(key=..., unit=<file|byte range|doc range|source shard>, est_bytes=...)],
    parallelism="shardable",           # or Single("reason: ...")
)
result = run_sharded(plan, worker, store=layer, merge=merge_fn, workers="auto")
```

Behavior, all in one place:

1. **Plan.** Shards are sized for balance, not per file: files above a
   threshold (e.g. 256 MB) are cut at line boundaries (`chunk_ranges` and
   `line_boundaries` in `release_verify.py` already do this for verification);
   shards are submitted largest-first. This removes the straggler where a
   single multi-GB JSONL is one shard (the build today only parallelizes files
   of at least 32 MiB, one worker per file, and that build sat at a load
   average of 2.5 to 6).
2. **Skip.** A shard whose receipt verifies (key = stage inputs + shard
   descriptor) is reused and counted as `reused`.
3. **Bounded pool.** `workers = min(cores, memory_budget / shard_memory)`;
   at most 2 tasks per worker in flight (the `run_tasks` rule), spawn context
   and the `__main__`-hiding workaround kept; any failure cancels the rest and
   leaves completed shards in place.
4. **Commit.** Each shard output is written to `store/tmp`, fsynced, receipted
   and renamed; a kill leaves only `.incomplete-*`.
5. **Merge.** A stage declares `merge` as one of: none (outputs stay
   separate, readers iterate), `concat` (ordered by shard key, bytes
   recorded), or `reduce(fn)` over small shard results (counts, accumulators).
   Merges are themselves receipted shards, so a crash in the merge repeats the
   merge, not the stage.
6. **Progress.** One progress record format (the `BuildProgress` journal):
   shards `done/reused/total`, `cores_used` (workers active, sampled), bytes/s,
   ETA, and the per-stage line "reused 612/640". Preflight reads it.
7. **Fault injection.** `KLAB_FAULT=<stage>:<n>` (test only) raises after the
   n-th shard commit. One parametrized test runs every registered stage on a
   tiny fixture: run with the fault, rerun, assert that the second run reuses
   n shards, executes the rest, and that the final output digest equals an
   uninterrupted run. A stage that registers cannot opt out (section 3).

### 2.2 The single-core rule

A stage registers `parallelism="shardable"` or `Single(reason)`. A test fails
for a `Single` without a reason string and for a stage missing from the table
in `docs/corpus-preparation.md`. Accepted reasons are narrow: an external
tool that is already multithreaded (BPE training), a sequential dependency
over a *small* artifact (union-find over 6 MB of duplicate groups), network
serialization imposed by a server. "The code was written that way" is not a
reason.

### 2.3 Per stage: unit, speedup, bottleneck, target

Targets assume 48 cores and NVMe at about 3.5 GB/s. "Today" is measured where
tagged **[M]**.

| Stage | Today | Shard (unit and checkpoint) | Cores achievable | Shared bottleneck | Target **[E]** |
| --- | --- | --- | --- | --- | --- |
| download body | network | one request/body; verified atomic cache entry | N transfers, budget-bounded | network, ledger | network-bound |
| acquisition row selection (HF qualification) | 73 min, 1 core **[M]** | one parquet file (44 for that run) or row group; part file + receipt | about 44 | pyarrow decode, Python row logic (vectorize with Arrow compute); read of 44 GB cache is 15 s | 3 to 6 min |
| snapshot verify | 15 min cold **[M]** | one snapshot file range | 48 | disk read | cold 108 GB: 35 s; warm: receipt, about 1 s |
| normalize | 25 to 40 min **[M]** | file range (cut at 256 MB) -> L2 shard + narrow rows | about 40 | CPU (cleaning) then NVMe write of 108 GB | 4 to 8 min |
| index (wide SQLite) | 26 min **[M]**, 227 GB | **removed**: the narrow row (id, split, rank, kind, flags, rights signature, raw/normalized/origin hashes, shard, offset) is emitted by normalize | n/a | n/a | merge 2.4 GB of narrow rows: under 1 min, memory-bound |
| dedupe | minutes after #490 **[M]** | hash bucket (256 prefix buckets of the key hash); union-find of groups is `Single("sequential over a few MB of groups")` | 48 for grouping | memory (2.4 GB) | under 1 min |
| annotate | n/a | L2 shard | 48 | CPU for model-based annotators; disk for lexical | lexical 1 min; fastText-class 2 to 5 min; MinHash 4 to 6 min |
| select / keep-lists | in views | L2 shard (apply decisions, write row ranges) | 48 | disk write of about 3 GB | under 1 min |
| views / release write | 25 min, 1 core **[M]** | with by-reference release: none. With materialized export: one output segment per L2 shard, concat | 48 | NVMe write: 188 GB at 2 to 3 GB/s | by reference: seconds; export: 3 to 5 min |
| freeze | 92 min **[M]** | in place: one file/shard hash | 48 | disk read 3.5 GB/s | 365 GB: about 105 s (in place); by reference: seconds |
| audit / verify cold | 41 min **[M]** | file range for hashes; shard for sampled replay | 48 | disk | hash 105 s + sampled replay 2 to 4 min; warm: about 0 |
| export | 16 min **[M]** (cause unknown, section 3.2) | one pass over `documents`-level data | 48 | | seconds with receipt |
| tokenizer sampling | 114 min **[M]**, linear scan of 188 GB | stratum x L2 shard: candidates by seeded hash threshold on the narrow table, then seek to offsets | 48 | random reads of about 0.5 GB | 1 to 2 min |
| tokenizer fit | included above | the BPE fit itself: `Single("HF tokenizers trainer is internally multithreaded; sequential merge selection")` | native threads | memory | few minutes (unmeasured, section 3.2) |
| near-duplicate screen | 162 min, 1 core **[M]** | a train byte range; held-out band table is built once, saved as a receipted artifact and memory-mapped by every worker (not 48 copies of a 6 GB index) | about 40 | CPU (MinHash is numpy); page cache | 4 to 7 min |
| token encode (L8) | existing pool | document batch (committed segments exist) | physical cores | CPU | about 1B tokens per few minutes **[E]**; 12B in 15 to 25 min |
| mixture schedule | copies 48 GB | the schedule is computed over the narrow table: no text read | n/a (seconds) | | seconds; the gather is part of `data prepare` |
| `data prepare` | receipted chunks | chunk | per-host plan | NVMe | disk-bound |
| describe | 40 min **[M]** | L2 shard stats memoized by (shard key, tokenizer) | 48 | | seconds warm |
| train | step checkpoints | checkpoint generation | devices | | see 3.1 |

These stage rows are the table the doc page keeps; the declared `parallelism`
and the table must agree (tested).

Why not simply "add `--workers`"? Because five of the singleton stages
(near-duplicates, tokenizer scan, acquisition selection, index, views) have a
global structure (one running tally, one SQLite writer, one accumulator, one
file offset), so parallelizing them needs their output to be a *set of
mergeable shards* first. The sharded form is the checkpoint form; adding
checkpoints to the singleton form (as #491 did for the near-duplicate screen)
is useful but leaves 162 minutes on one core.

## 3. The resumability contract applied

### 3.1 Contract, restated for this model

(From #485; **decided**.) (1) Every layer output is keyed by its inputs and
receipted, published atomically. (2) A failure resumes at the last verified
shard of the last unfinished stage. (3) A resumed run's result is byte
identical to an uninterrupted one, which holds when shard outputs are
deterministic and merges are ordered by key, not by completion. (4) Nothing
verifies unchanged inputs without cause (section 7). (5) One fault-injection
harness covers all stages (2.1). (6) Progress states reused vs rebuilt shards.
(7) Retention bounds checkpoint disk: incomplete directories are removed on the
next start; superseded L4/L5 outputs keep the newest N.

**Fail-fast corollary.** The three late failures came from recipe or policy
values (empty source after indexing, 93 min; module mix-up, 20 min;
rights-conflict duplicate at dedupe, 2 h). A policy check runs on the
cheapest artifact that can decide it: empty-source and admission checks on
narrow rows right after normalize; rights conflicts on narrow rows (the rights
signature is already a narrow column); and `klab corpus plan` (section 7)
runs the canary and these checks before any long stage is paid for.

### 3.2 Status table

| Stage | Unit today | Resumable today | Gap and the fix | Failure at the end costs now |
| --- | --- | --- | --- | --- |
| acquire (download) | source/transfer | yes (snapshots, transport cache) | none | the unfinished source |
| acquisition row selection | whole source | **no**: 73 min **[M]** from the 44 GB cache | per-parquet-file parts (PR 4) | the source, 73 min |
| snapshot verify | snapshot | receipts (#487) | first cold pass still 15 min: parallel hash (PR 1) | one cold pass |
| normalize | source file | yes (receipted shards) | re-hashes every shard on start; big files are one shard; shard key too broad (1.1); file-range sharding (PR 5, 7) | the files in flight |
| index | whole stage | yes as a stage (#486), 26 min | replaced by narrow rows emitted in normalize (PR 5) | the stage |
| dedupe | whole stage | yes (#486, #490) | bucketed, still cheap (PR 5) | the stage, minutes |
| views and publish | one pass | **no**; #491 argues against mid-pass resume | disappears with keep-lists / per-shard segments (PR 5); a pass that has shards is resumable for free | 25 min |
| freeze | copy | **no**; copy restarts | in place, then by reference (PR 6, 7) | 92 min |
| audit / readers | cold pass then receipt | partly | receipt-first; sampled replay; parallel (PR 6) | one verification |
| describe | verification + measurement | **no** | likely not re-verification: it calls `verify_release` (receipt-aware when the proof store is active) and then `measure_views`, which loads every lineage row into a temporary SQLite table and tokenizes every view row in one process. This contradicts the performance rule and is the probable source of the 40 min; **to be profiled** before the fix. Fix: memoized per-shard stats (PR 6) | 40 min |
| export | single pass | staged, atomic | 16 min **[M]**: not yet explained; if it is verification, receipts fix it; **to be investigated** | the export |
| tokenizer sampling scan | one linear scan | **no** | seeded-hash sampling over narrow rows; the receipt caches the selected sample (PR 3) | 114 min |
| tokenizer fit | one fit | **no** | `Single` with reason; minutes once the scan is gone | the fit |
| near-duplicates | one pass | yes since #491 (checkpoint every 200k docs), still one core | sharded form (PR 2) | docs since checkpoint |
| encode tokens / materialize mixture | segments | yes (committed segments) | becomes schedule + L8 (PR 10) | a segment |
| verify-mixture | one pass | n/a (a check) | receipt | one pass |
| `data prepare` | chunk | yes (receipted chunks) | none | a chunk |
| stage warmup | pilot | no | minutes; leave | the pilot |
| **training** | generation | yes (`--recover`, `--resume`, data cursor, `--spot-policy`) | **not verified for leased hardware**: (a) resume on a different device count (the resume refuses a changed "batch shape"; whether device count changes it needs a test), (b) nothing relaunches after a preemption (`--recover` is manual), (c) a run copies its prepared data (about 48 GB) into the run directory, a cost on a rented disk (PR 13) | up to the spot-policy bound |

The #485 acceptance ("no restarts from zero for a pretraining stage") is met
when the rows marked **no** are closed; PR 13 closes the training row by test
and a documented supervisor loop.

## 4. Pluggable operators without a schema zoo

### 4.1 Five kinds, no more

| Kind | Reads | Writes | Replaces / absorbs today |
| --- | --- | --- | --- |
| `annotate` | an L2 shard (and earlier annotations) | an L3 sidecar shard | `documents.exclusions`, the language label, boilerplate and MinHash computation |
| `select` | narrow rows + annotations | keep/drop with reason, per document | rights `training_use`, `empty_text`, privacy exclusion, held-out rules, `sample` for scale-down |
| `dedupe` | narrow rows (+ annotations) | groups and drop decisions | `release.policy.duplicates` (`exact`, `component_ranked`) and the near-duplicate screen |
| `tokenize` | selected documents | tokenizer (L7) and token arrays (L8) | `tokenizer train`, the content token store |
| `mix` | a release (or several) + L8 | a schedule (L9) | `materialize-mixture`, `additional_releases` |

Acquire, normalize, split, and release assembly are *fixed* stages, not
operators: their behavior is already chosen by named options
(`release.normalizer`, `release.policy`), and making them swappable buys
variability that threatens identity. `normalize` is parameterized; it is not a
plugin point.

Why these five and not a "schema per model type": none of them mentions a
model. Chat and tool views stay what they are today (transforms of a release,
`lm`/`chat` views); the operator set describes data, and models appear only at
batch construction (#459 principle 5).

### 4.2 Registration and recording

```python
@operator(
    kind="annotate",
    name="boilerplate",
    impl_version=1,
    scope="text",
    params=BoilerplateParams,
)
def boilerplate(shard: Shard, params: BoilerplateParams) -> Arrow: ...
```

- A **registry in the repo** (`klab/corpus/operators/`), one module per
  operator, with a Pydantic parameter model: unknown or missing parameters are
  refused ("every option is required", as for `release.policy` today).
- **Identity** of an output is `digest({kind, name, impl_version, resolved
  params, input keys})`. The resolved parameter object (defaults filled in) is
  written into the receipt and into the release manifest, exactly as
  `release.policy` is recorded in `build.json`, the manifest, `report.json`
  and `license-report.json` today.
- `impl_version` is an integer *in the operator*, bumped when output bytes for
  the same input and parameters change. It replaces "SHA-256 of six source
  files" (1.1). A golden-output test per operator on a fixture corpus fails if
  the output digest changes without a bump, so a forgotten bump is caught in CI
  rather than discovered as silently stale reuse. This is **not** a schema
  version: it versions an implementation, never a file format, and bundles no
  unrelated choices. (D3.)
- A recipe lists operators; options live next to the operator:

```yaml
release:
  normalizer: {structure_cleaning: true, front_matter_extra_keys: {}, gutenberg_boilerplate: []}
  annotate:
    - {op: language,    params: {min_confidence: 0.6}}
    - {op: privacy,     params: {profile: research_privacy}}
    - {op: boilerplate, params: {min_document_frequency: 50, min_line_chars: 12}}
    - {op: minhash,     params: {shingle_words: 3, content_only: true, bands: 16, rows: 4}}
  select:
    - {op: rights,   params: {training_use: allowed_unless_explicitly_prohibited}}
    - {op: exclude,  where: "privacy.flags != []", reason: privacy}
    - {op: exclude,  where: "language.code != 'en' and source.kind == 'web'", reason: language}
  dedupe:
    - {op: exact,   params: {keys: [raw, normalized], heldout_overlap: drop_train_copy,
                             resolution: chain_smallest_id, rights_conflict: error}}
    - {op: minhash, params: {content_jaccard: 0.8, heldout_overlap: report}}
```

`where` is a small expression over annotation columns (comparison, `and`,
`or`, `not`, `in`, null tests) evaluated vectorized on Arrow; there is no
embedded Python. Anything more elaborate is an annotator, which keeps the
selection reviewable as data. The existing `release.policy` sections become
the parameters of the `rights`, `exact` and `exclude` operators with the same
names and values (no alias is kept; recipes are edited in the same PR and
downstream recipes follow).

### 4.3 Swapping and re-parameterizing

Change `content_jaccard: 0.8` to `0.7` and re-run: the annotation
(`minhash`, L3) is a hit because its key does not include the threshold; only
the join in `dedupe` (L4, under a minute) and the selection rerun. Replace
`exact` by `component_ranked`: same. Add a new filter: one new expression in
`select`. A new *annotator* runs over the L2 shards in parallel, once, for
every project that ever uses it.

**Rule for authors:** put expensive, parameter-independent computation in an
`annotate` and cheap, parameter-dependent decisions in `select`/`dedupe`. The
operator playbook says so, and `klab corpus plan` shows which layer a
parameter change invalidates.

### 4.4 Custom operators: the `command` contract

```yaml
annotate:
  - op: command
    name: my_quality_score
    command: ["python", "tools/score.py", "--in", "{input}", "--out", "{output}"]
    executable_sha256: <digest of tools/score.py>     # pinned
    reads: {documents: [id, text]}                    # declared inputs
    writes: {schema: {document_id: string, score: float32}}   # declared output columns
    scope: text
    params: {model: v3}
```

Lab hands the command a read-only input shard (Arrow/JSONL), expects a
sidecar shard with exactly the declared columns and one row per input document,
validates it (row count, key alignment, types), hashes it and ingests it as
an ordinary L3 shard whose identity includes the command line, the executable
digest and the params. Same command on the same shard is a cache hit and
resumable per shard like any other annotator. A command that is nondeterministic
is the author's problem; the receipt records it and `klab corpus verify` can
re-run one shard cold to tell (`deterministic: false` is a recorded property,
not an error).

### 4.5 Extension mechanism (D4)

| Option | For | Against |
| --- | --- | --- |
| Registry in the repo | reviewed, testable, golden-output guarded, no install-time code execution; matches "use what Lab provides" | users must send a PR for a reusable operator |
| Python entry points | reusable plugins in a separate package | executes arbitrary code at import in every `klab` process; identity of plugin code is not captured by Lab; version skew; supply-chain surface |
| Recipe-declared `command` | the escape hatch #459 asks for; code identity pinned by digest; runs in a subprocess with declared I/O | still arbitrary code |

**Recommendation:** the in-repo registry for everything Lab ships, plus the
`command` operator for user code; no entry points now. Security stance for
`command`: it runs with the user's authority, like any script the user chooses
to run, and Lab does not pretend to sandbox it. Lab (a) refuses it unless
`--allow-command-operators` is passed or the recipe is under the user's own
checked-out recipe root, (b) refuses when `executable_sha256` does not match,
(c) runs it in `scratch/` with the input read-only, a resource envelope, and
no network unless declared, (d) never lets it write outside its output
directory, and (e) records the command, digest and environment in the
receipt. A recipe from someone else is untrusted code: read it first. The cost
of being wrong about entry points is low: they can be added later behind the
same `@operator` interface.

### 4.6 What not to build

- No general workflow engine, no user-defined DAG, no scheduler beyond
  `ShardRunner`. Order is fixed: normalize, annotate, select/dedupe, release,
  tokenize, mix. (#459 step 5 asked whether a stage graph is needed once more
  than one stage depends on another's output ordering; with annotations
  declaring the annotations they read, the dependency is a list inside the
  fixed order, not a graph.)
- No per-model-type schema; no operator that mentions a model.
- No operator version negotiation, no compatibility reader (AGENTS.md).
- No plugin discovery by directory scan.

## 5. Annotations and selection (#459)

### 5.1 Format and alignment

- **Format.** Parquet (pyarrow is already a dependency), one part per L2 shard,
  sorted in the shard's row order, columns: `document_id`, `content_sha256`,
  then the operator's declared columns. Columnar so a `select` over one column
  reads kilobytes, not the 108 GB (the performance rule). A receipt carries
  the producer, parameters, input keys and part hashes; the release manifest
  lists the annotation keys it used.
- **Alignment key (answers the #459 open question).** The join key is
  `document_id`, which already encodes `(snapshot, location, raw_sha,
  normalizer identity)`. A change of normalizer changes every ID, so sidecars
  are bound to a normalizer and are *not* silently valid across it; that is
  correct for any annotator that sees text. To avoid recomputing expensive
  annotators after a harmless normalizer change, annotators declare a
  `scope`:
  - `text`: a pure function of the normalized text (language, MinHash, a
    quality classifier). Keyed per document by `content_sha256` in a memo
    (`store/memo/<annotator>/<params>/`), so a normalizer change recomputes
    only documents whose text changed. The shard sidecar is then assembled from
    memo hits.
  - `document`: needs metadata (path, source, license), keyed by `document_id`.
  - `corpus`: relational (duplicate clusters, document frequency of lines), keyed
    by the scope. These are `dedupe`/two-pass annotators and are never memoized
    across scopes.
  So: *document id for the join, plus content hash for the memo*. A memo is
  optional in PR 8 and added when a measured annotator is expensive enough.
- **Who produces them.** Separate, resumable operator runs (`klab corpus
  annotate RECIPE [--op NAME]`), driven by `ShardRunner`, not inside `build`.
  `build` requires the annotations the recipe's `select` and `dedupe` read and
  runs missing ones, so a plain `build` still works end to end. Answer to
  "inside `build` or separate": both, because separately invokable is the same
  code path.
- **Selection** references columns by `annotator.column`; the release manifest
  stores the expression text and the annotation keys, so a release states
  exactly which annotations and options produced it (#459 acceptance).

### 5.2 The first annotations

| Annotation | Scope | Columns | Notes |
| --- | --- | --- | --- |
| `language` | text | `code`, `confidence` | replaces the "predominantly non-Latin" heuristic for MediaWiki |
| `privacy` | text | `flags` (list), `reasons` | today's `_excluded_research_record`, moved out of normalize; the exclusion becomes `select: exclude where privacy.flags != []` |
| `duplicate` | corpus | `cluster_id`, `representative`, `method` | output of `dedupe`, queryable by `select` and mixtures |
| `boilerplate` | corpus (document frequency of line hashes) | `template_share`, `content_bytes` | section 5.3 |
| `minhash` | text | `bands` (16 x uint64), `content_bands` | input to dedupe and to the cross-split screen; about 2.7 GB for 20.9M documents **[E]** |
| `source_quality` | document | `source_id`, `rank`, `license`, `kind`, repository stars/age when the source supplies them | from existing fields, so `select` and the mixture can use them without reading text |

### 5.3 Template-aware near-duplicates

The problem (README of the Workshop screen, `near-duplicate-screen.md`): the
cross-split screen on that run lists "thousands of GitHub threads that share
issue templates and bot comments", and the reviewer's remedy is a hand-written
`default: kept`. Dropping them would delete the best-maintained projects'
material; keeping all of them hides real leakage in the same list.

Design: **annotate boilerplate and score content similarity separately.**

1. `boilerplate` computes, in two parallel passes, a document-frequency table
   of normalized line hashes (counted by hash bucket in parallel: sort-merge on
   disk, or a count-min sketch of about 4 GB if exact counting is too large;
   the choice is measured, not assumed). A line appearing in at least
   `min_document_frequency` distinct documents of at least two different
   projects/families is *template*. Per document: `template_share` = bytes in
   template lines / bytes, `content_bytes` = the rest. Nothing is dropped.
2. `minhash` computes signatures twice: over all shingles (`bands`) and over
   shingles of **non-template lines only** (`content_bands`).
3. The cross-split screen joins held-out against train on both. A pair is
   classed:
   - `leak`: content-Jaccard at least `content_jaccard` (default 0.65, as
     today) - copied files, forks, re-posted pages. Needs a decision, and with
     `heldout_overlap_near: drop_train_copy` the train copy is dropped like an
     exact held-out overlap.
   - `template_only`: raw Jaccard at or above the threshold but content-Jaccard
     below it. Recorded as machine-decided `kept`, with both scores as the
     evidence, and no human review. Train and held-out documents both stay.
   - `content_too_short`: fewer content words than a shingle count can judge;
     matched by exact content text only (the rule for documents without words
     today).
4. Training selection may use `template_share` (for example to drop a document
   that is entirely a bot template), but the default selects nothing on it.
   Exact whole-document duplicates are still collapsed; that loses no content.

What protects held-out data is unchanged in strength for real leakage and
better in coverage: the old screen's templated pairs are indistinguishable
from leaks at the pair level, so reviewers had to accept them wholesale; the
new classes make the human decision about `leak` pairs only. **Uncertainty:**
the quality of the frequent-line criterion (short boilerplate lines vs
distinctive but common code lines such as `import numpy as np`) is not
measured. PR 9 starts with a measurement on the existing report of that run
(owner has it): sample 200 pairs per class, hand label, report precision of
`template_only` (false "template" calls would silently keep a leak). The
`min_line_chars` and multi-family conditions exist for that reason, and the
default of this feature stays `report`, not `drop`, until measured (D7).

## 6. Verification and cost

### 6.1 One verification model (decided)

- A **manifest of digests** per layer output (the receipt). Verification
  hashes files named by the manifest, in parallel by file range (3.5 GB/s
  measured), then checks references (a release's shard keys, an annotation's
  input keys) by their own receipts. Per-row checks are *sampled* (a seeded
  sample of shards re-derived cold from snapshots, `--audit` for all), not a
  load of every row into a database (D5 asks the owner to accept sampled replay
  as the default; today's full replay is 41 min).
- **Receipts are reused.** Local proof store receipts (stat fingerprint,
  host-local) answer "unchanged since I hashed it". The rule: **no step
  re-verifies an unchanged input without cause**; causes are `--cold`, a
  fingerprint mismatch, a receipt from another host. Readers that today verify
  separately (`audit`, `describe`, `sources`, `sample`, `export`, `materialize`,
  `near-duplicates`) go through `verify_release` once per command and print
  "verified by receipt" or "verified cold, N s".
- **Performance rule (decided).** No per-row single-threaded work; no random
  I/O against a database on wide rows. Wide payload is never in a database; a
  mutable column never lives next to text. State that must be joined lives in
  Arrow/Parquet narrow tables read sequentially (or memory-mapped) with
  DuckDB or pyarrow for out-of-core joins when needed (#450 item 6). A CI
  check greps the corpus package for `sqlite3` on payload tables and for stages
  without a `parallelism` declaration.

### 6.2 Expected wall time on 48 cores **[E]**

| Stage | Today **[M]** | Expected | Bound by |
| --- | --- | --- | --- |
| verify snapshots 108 GB (cold / warm) | 15 min | 35 s / 1 s | NVMe 3.5 GB/s / receipt |
| acquire from cache (row selection) | 73 min | 3 to 6 min | pyarrow decode |
| normalize | 25 to 40 min | 4 to 8 min | CPU, then NVMe write |
| narrow merge + dedupe | 26 min + minutes | 1 to 2 min | memory |
| annotate (language, privacy, boilerplate, minhash) | n/a | 8 to 15 min total, once per shard set | CPU |
| selection + release | about 25 min views + 92 min freeze | under 2 min (by reference); about 5 min (freeze in place) | disk hash 365 GB: 105 s |
| audit cold / warm | 41 min | 3 to 6 min / 0 | disk hash + sampled replay |
| tokenizer sampling + fit | 114 min | under 10 min | random reads, native threads |
| near-duplicate screen | 162 min | 4 to 7 min (with minhash annotation cached: 1 min join) | CPU |
| encode 12B selected tokens | existing pool | 15 to 25 min | CPU |
| **cold build to a tokenized mixture** | about 11 h wall | **about 45 to 70 min** | |
| **warm (everything cached), new mixture weights** | | **under 3 min** | |

### 6.3 Disk accounting (12B-token scale)

| Layer | Today **[M]** | Proposed **[E]** |
| --- | --- | --- |
| L0 transport | 44 GB | 44 GB |
| L1 snapshots | 108 GB | 108 GB |
| L2 normalized | 108 GB | 108 GB + about 2.5 GB narrow rows |
| wide index | 227 GB | 0 |
| L3 annotations | | about 5 GB |
| L4/L5 | in the index | about 3 GB |
| build dir | 365 GB | 0 (merged into the above) |
| L6 release | 374 GB | below 1 GB (by reference) |
| L8 token arrays | | up to 48 GB (int32; 24 GB as uint16, D9) |
| L9 mixture | 48 GB | megabytes |
| L10 prepared run copy | about 48 GB | about 48 GB |
| **total** | **about 1.3 TB** | **about 0.37 TB (0.74 TB after only freeze-in-place)** |
| **peak during build** | index + build + prepared together | prepared + annotations + sort scratch |

### 6.4 Retention and the command

`klab corpus workarea report [--work-dir WORK]` prints, per layer: bytes, number
of objects, last used, and which releases, mixtures and runs reference each
(the reverse index is the receipts' input lists). `klab corpus workarea prune
--dry-run` lists what would go; without `--dry-run` it deletes only:
incomplete directories, superseded L4/L5 beyond N, L10 for runs that are
finished, and objects unreferenced for `--older-than`. It never deletes L0 or
L1, anything a release or run references, or what it did not create (AGENTS.md
section 5).

**Shipping to a leased host** (money): ship L2 + L3 + L8 for the selected
documents (about 160 GB) rather than L0 + L1 (152 GB) when the goal is to
train; ship L0 + L1 only when you will change the normalizer. The report
prints both totals with transfer times at a stated bandwidth.

## 7. Guidance: steer to good, performant choices

- **`klab corpus plan RECIPE`** (new, read-only): resolves the recipe, runs the
  canary and the cheap recipe checks (empty source, rights conflicts on narrow
  rows when they exist, unknown strata), then prints, per stage, `reuse N/M
  shards` or `build`, the layer that a changed parameter invalidates, expected
  wall time on this host's cores and disk (from `ShardRunner` throughput records
  of earlier runs, with a stated fallback table), and disk needed against free
  space. Paying hours should be preceded by reading this. The late-failure
  incidents are the acceptance test.
- **`klab preflight`** gains a `workarea` check: layers verified or reused,
  single-core stages (WARN with their reason), disk headroom including the
  per-run copy, the checkpoint policy against the leased disk.
- **`klab inspect` advisories:** a mixture stratum repeated more than its
  declared exposure sensibly allows, a sampling order of `release_prefix` for a
  tokenizer fit, `heldout_overlap: error` with an annotation-driven selection
  (the combination that fails late), an annotator configured with a threshold
  that selects nothing.
- **A playbook per operator** in one page, `docs/corpus-operators.md`, linked
  from the AGENTS.md section 10 table: what it protects against, parameters,
  the layer a change invalidates, when to use `annotate` instead of `select`,
  measured cost per GB, and the value for that run used and why. This is how the
  boilerplate rule teaches that good projects' templates are kept.
- **Defaults are guidance:** `build.strategy: disk` default, workers `auto`,
  `--scratch-dir` on the fastest volume with a printed warning when it is on
  spinning disk, `sampling.order: seeded_hash`.

## 8. Migration and plan

### 8.1 What must be rebuilt

AGENTS.md forbids readers for old formats. So, per change in section 8.2,
the PR description names what to rebuild. The expensive rebuilds are
acquisition's parts (PR 4; the transport cache absorbs the download) and
normalization (PR 7 changes the shard key and layout, once). The run's current
run keeps working on the Lab revision it started on; the PRs below are merged
in an order such that the layout change (PR 7) lands **before** the next
12B-token-scale run, so that run is built once in its final shape (D6).

### 8.2 PR series

Each PR is independently green, small, ordered by value for the next
12B-token-scale run. Efforts are agent-days **[E]**, test and docs included.

| # | PR | Delivers | Effort | Rebuild |
| --- | --- | --- | --- | --- |
| 1 | `ShardRunner`: shard plan, skip-by-receipt, bounded pool, merge, progress with cores used, fault-injection harness, `parallelism` declaration test; port `hash_inventory` and `run_tasks` onto it | the primitive; cold snapshot and release verification at disk speed (35 s for 108 GB instead of 15 min) | 5 | none |
| 2 | Near-duplicate screen sharded: held-out band table as a receipted memory-mapped artifact, train byte ranges as shards | 162 min -> 4 to 7 min; benefits any release | 4 | none |
| 3 | Tokenizer sampling: parallel byte-range candidate pass with `sampling.order: seeded_hash` (option), receipt keyed by (release, sampling spec) | 114 min -> under 10 min; sample is no longer biased toward the first sources in release order | 3 | tokenizer re-fit if the option changes (D8) |
| 4 | Acquisition row selection sharded per parquet file with part receipts | 73 min -> 3 to 6 min; resumable | 4 | re-run acquire (cache, cheap) |
| 5 | Narrow rows emitted in normalize (Arrow), wide SQLite index removed, dedupe by hash bucket, views written as per-shard segments in parallel and concatenated | index 26 min and views 25 min gone; 227 GB gone; build resumable per shard | 9 | rebuild builds and releases |
| 6 | Freeze in place + receipt-first audit/describe/export; sampled replay (option); `describe` measurement per-shard and memoized | 92 + 41 + 40 min -> minutes; 374 GB copy gone | 6 | refreeze |
| 7 | `WORK/store/`: project-independent layers; narrowed shard key; `impl_version` replaces file hashes; `workarea report/prune`; a second project reuses | multi-project reuse; the work area survives Lab upgrades | 8 | one re-normalize in the new layout (4 to 8 min after PR 4/5) |
| 8 | Operator registry; annotation sidecars; `language`, `privacy`, `source_quality`; `select` with `where`; `release.policy` mapped to operators; docs/operators page | pluggable filters; privacy rule changes cost seconds | 9 | recipes edited; releases rebuilt |
| 9 | `dedupe` operators (exact, minhash), `boilerplate` annotator, template-aware cross-split classes (default `report`) after the measurement | boilerplate kept, real leaks caught; screen becomes a join | 9 | none beyond annotations |
| 10 | Release by reference + `ReleaseReader`; mixture as schedule over L8; `data prepare` gathers | 374 GB and 48 GB copies gone; one text copy | 10 | rebuild releases and mixtures |
| 11 | `klab corpus plan`; preflight `workarea` check; inspect advisories | guidance; no late failures from recipe values | 4 | none |
| 12 | `command` operator | user scripts without Lab modelling them | 5 | none |
| 13 | Training under preemption: resume on a different device count (test), documented supervisor loop (`--recover` retry), prepared data by reference instead of per-run copy | #485 closed for training | 4 | none |

About 80 agent-days; PRs 1 to 6 (roughly 31 days) fix every measured
multi-hour singleton and the copy; they stand alone if the owner stops there.
PR 7 onward is the "durable work area" the owner asked for; PR 7 should not
wait past 6. PRs 2, 3 and 4 are independent of 1 in principle but should use it
(the harness is worth more than the three speedups together), and may start in
parallel after its interface is fixed.

### 8.3 Risk register

| Risk | Likelihood / impact | Mitigation |
| --- | --- | --- |
| A forgotten `impl_version` bump gives silently stale reuse | medium / high | golden-output test per operator in CI; receipt records the Lab revision for forensics; `--cold` and `corpus verify --sample` |
| Release by reference touches every reader (mixture, sampling, measurement, near-duplicates, verifier) | high / medium | PR 6 first (freeze in place) delivers most of the time and disk without it; `ReleaseReader` introduced behind one interface; PR 10 last |
| Sampled replay weakens an existing guarantee | medium / medium | `--audit` full replay remains; sample seeded and recorded; the owner decides (D5) |
| Parallel shards change output order and break byte identity | medium / high | merges ordered by shard key; the harness asserts digest equality with the unfaulted run |
| Memory blow-up with 48 workers (held-out band table, large documents) | medium / medium | memory-mapped shared tables; `workers = min(cores, memory_budget / shard_memory)`; resource envelope |
| The template criterion mislabels real leaks | medium / high | measure before shipping; default `report`; `min_line_chars`, multi-family document frequency; keep exact held-out overlap rule unchanged |
| Two projects on one store race or corrupt each other | low / high | atomic rename, per-key lock, receipts, concurrency test |
| Parallel speedups disappoint (NVMe or Python overhead) | medium / low | every PR replaces its **[E]** with a measurement in its description; the plan, not the promise, is the deliverable |
| `command` operator becomes an injection path | low / high | explicit allow flag, pinned digest, envelope, no network by default, docs say "read the recipe" |
| A user on a spinning disk gets worse random I/O than before | low / medium | the design removes random SQLite I/O; scratch placement warning |
| Estimates are off by more than 2x | medium / medium | PR 1 adds throughput records so `plan` uses measured rates |

### 8.4 What we are not doing

A DAG or workflow engine; user-defined stage ordering; a schema or version per
model type; new `schema_version` values to carry policy (AGENTS.md); entry
point plugins now; reading old layouts or aliases; deduplicating by
paraphrase/embedding (an annotator someone may write, not core); making
Small-dataset users use the work area; automatic deletion of acquired data;
changing the license/rights model; unifying `klab data snapshot` with the
corpus pipeline; hosted/shared multi-user stores with access control (the store
is single-user per host); sandboxing user code.

## 9. Open decisions for the owner

| # | Decision | Recommendation | Cost of being wrong |
| --- | --- | --- | --- |
| D1 | Release by reference (one text copy) vs a frozen in-place directory | Freeze in place first (PR 6); by reference in PR 10 | If reference is skipped: 188 GB text duplicate and 227 GB-class intermediates remain at 0.74 TB. If done wrong: a refactor of every release reader (about 10 days) |
| D2 | One shared `WORK/store` vs per-project layers with aliases | Shared store | Per-project: every second project re-normalizes (minutes after PR 4/5, but disk doubles at 108 GB per copy). Shared: a rights mistake spans projects (mitigated: rights are in the key) |
| D3 | Operator identity by declared `impl_version` + golden test vs source-file digests | Declared version | File digests: every Lab upgrade invalidates the work area (a re-normalize each time, 4 to 8 min compute but also the downstream annotations). Declared: a missed bump yields stale reuse until the golden test catches it |
| D4 | Extension: repo registry + `command`; no entry points | As stated | Later adding entry points is cheap. Wrong the other way: a plugin ecosystem with unpinned code identities |
| D5 | Default verification: sampled replay (full with `--audit`) vs full replay | Sampled replay, seeded and recorded | Full: 3 to 6 min extra per audit, acceptable. Sampled: a rare corrupted row escapes until `--audit`; mitigated by the shard hash |
| D6 | Land PR 5 to 7 before the next 12B-token-scale run? | Yes; the run is then built once in the final layout | If not: a second rebuild later (the 4.5 h cost of today, or 45 min after the PRs) |
| D7 | Template classes: default `report` vs `drop_train_copy` for `leak`; thresholds | `report` until the measurement in PR 9 | `drop` too early: lost good material (the owner's stated worry). `report` too long: manual review remains |
| D8 | Tokenizer sampling `seeded_hash` by default (a prefix of release order is biased toward early sources) | Yes | The run's tokenizer changes; the 32k tokenizer was fitted on the prefix, so any model trained on it needs the comparison noted |
| D9 | uint16 token arrays when the vocabulary is at most 65,536; per-run prepared copy by indirection | uint16 yes (halves L8 and L10); indirection later | Format change to L8/L10 (rebuild); indirection risks loader throughput on rented disks |
| D10 | Retention: explicit `prune` only, never automatic beyond incomplete/superseded | Explicit | Automatic deletion could destroy an hours-long build; none leaves disks full (report warns) |
| D11 | Single-core exceptions: allowed with a written reason, reviewed | Yes | Strict "never" would force bad designs for BPE fit |
| D12 | Keep small datasets on `klab data` and out of the work area | Yes | Unifying doubles the surface for little gain |

## 10. Evidence index

Code read: `corpus/build_stages.py` (`StageStore`, receipts, prune),
`corpus/large_build.py` (`_receipt_identity`, `_stage_inputs`, `build_large`,
the `ProcessPoolExecutor` for files of at least 32 MiB), `corpus/pipeline.py`
(document identity, `research_privacy`), `corpus/release.py` (`freeze`,
`verify_release`, `describe`), `corpus/release_copy.py` (`copy_hashed`,
reflink fallback), `corpus/release_verify.py` (`run_tasks`, `chunk_ranges`),
`corpus/completed_release.py` (in-place freeze), `corpus/measurement.py`
(`_lineage_index`, `measure_views`), `corpus/acquisition.py` (project-scoped
snapshot paths, alias), `corpus/near_duplicates.py`, `data/tokenizer_sampling.py`
(prefix selection in release order), `data/token_store.py` (`TokenStore`,
`ContentTokens`), `verification_proofs.py`, `docs/checkpointing.md`. Workshop:
Workshop recipes. Measurements are the
owner's 12B-token run as quoted in the task; nothing here was run against that
data.
