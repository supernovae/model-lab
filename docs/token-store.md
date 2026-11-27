# Reusable token store

Tokenizing a text file with a tokenizer gives the same token IDs every time, so
Lab does it once and keeps the result. Preparing data for another model, a
larger token budget or a newer Lab revision then copies stored tokens instead of
tokenizing again.

It is used automatically by `klab data prepare` (and by `klab train` when it
prepares data) for `dataset.source: local_text` with a byte-level BPE tokenizer
and no byte-addressed memory. Other sources use the chunked path described in
[runtime](runtime.md#resource-envelopes). The prepared arrays and
manifest are byte-identical either way.

## Where it lives and what it holds

The store sits beside the dataset cache directory: a config with
`dataset.cache_dir: /data/cache` keeps tokens under `/data/cache-tokens`. Runs
that share a cache directory share its tokens.

```text
/data/cache-tokens/<key>/store.json             what the store holds
/data/cache-tokens/<key>/shard-000000.ids       int32 token IDs, each document then EOS
/data/cache-tokens/<key>/shard-000000.ends.npy  token count after each document
```

`<key>` is derived from the source file's SHA-256, the tokenizer's digest and
the store format version. It does not include the model, the token budget or
Lab's source code, which is what makes a store reusable:

| Change | What happens |
|---|---|
| Another model, same tokenizer and data | Tokens are copied; nothing is tokenized |
| A larger `train_max_documents` | Only the additional documents are tokenized |
| A smaller selection | A prefix of the stored tokens is copied |
| A Lab upgrade that changes the prepared-cache key | The prepared cache is rewritten from stored tokens |
| A different tokenizer or an edited source file | A new store is built |

Documents are stored in source order, 256 MiB of tokens per shard.

```sh
klab data tokens RUN.yaml          # list the stores beside this config's cache
klab data tokens RUN.yaml --json
```

There is no database: one small `store.json` per store is the only record.

## Speed

One worker process per core parses and tokenizes whole runs of JSONL lines and
returns raw arrays, so the parent only reads bytes and writes shards. Measured on
a 24-core Threadripper PRO 9965WX with an 8k byte-level BPE on 1.6 GB of text:

| Operation | Rate |
|---|---:|
| Build, 24 workers | 27.5M tokens/s |
| Build, 46 workers | 31.4M tokens/s |
| Copy out of the store, verifying every shard | 270M tokens/s (1.0 GiB/s) |

End to end through `prepare_data` on the same host (a 512-entry test tokenizer,
so rates are not comparable with the table above, only with each other):

| Preparation of 732M tokens | Time |
|---|---:|
| Chunked path | 50.9 s |
| Token store, nothing stored yet | 17.7 s |
| Token store, tokens already stored | 7.3 s |

Worker count follows measured CPU and memory capacity; an envelope's
`max_workers` overrides it.

## Failure and integrity

- A shard is listed in `store.json` only after it has been synced and renamed
  into place. An interrupted build leaves at most `*.tmp` files inside the store
  directory; the next build removes them and continues after the last sealed
  shard. Nothing already tokenized is redone.
- The staging directory for the prepared arrays is removed if preparation fails,
  so a failed run leaves no partial cache.
- Each shard's SHA-256 is recorded when it is sealed and checked whenever the
  shard is read. A shard that changed is refused, never copied.
- Free space is checked before building, from the bytes still to be read.

To discard a store, delete its `<key>` directory; it is rebuilt on demand.

## Limits

- The positional store covers `local_text` JSONL only (mixtures use the content
  store below). Snapshots, chat data and
  byte-addressed memory still use their existing paths.
- Token IDs are stored as int32 whatever the vocabulary size.
- The check that train and validation documents are disjoint still reads both
  source files once per preparation.

## One encoder

Every preparation path tokenizes through the same pool,
`klab.data.encoding.EncoderPool`: single-threaded `klab.token_encode_worker`
processes that receive text and return int32 token IDs as length-prefixed
binary frames. No JSON and no Python integer list crosses a process boundary.
The token store sends workers whole runs of JSONL lines; the streaming
collector sends batches of texts through `PreparationEncoder`, which spreads
each batch over the workers and returns the results in input order. A corpus
mixture sends a batch as the byte ranges of its rows in `documents.jsonl`, so
the worker reads and parses the rows itself and the text never reaches the
coordinator; each batch goes to one free worker and the coordinator applies the
results in order. Documents are encoded independently, so the split
does not change any token.

## Tokens by content

A corpus mixture selects documents by content, not by position in a file, so it
keeps its reusable encodings in `ContentTokens`: the same raw int32 IDs, in two
append-only files per tokenizer under the corpus work root
(`cache/source-tokens/<key>/`):

| File | Holds |
| --- | --- |
| `tokens.ids` | int32 token IDs of each document, without EOS |
| `index.bin` | one 76-byte record per document: content SHA-256, token offset, token count, SHA-256 of its IDs |

The index is read into memory once, so a lookup is one positioned read; there
is no database. Appends are sequential, which is what a spinning disk does
well. It is a cache: nothing is synced, IDs are written before their index
record, and after a crash the index is cut back to the documents whose IDs are
present. Every read checks the stored digest. One build holds a store at a
time; a second build on the same tokenizer and corpus is refused with a clear
message instead of waiting.

### Sealed content segments

A mixture declaration can set `token_cache: sealed`. This uses the same
encoder and defaults to little-endian int32 for the active and sealed segments.
The writer seals by rename before exceeding 67,108,864 tokens or 65,536
documents and at successful completion.
A document that exceeds the token limit occupies its own segment. Sealing
syncs the active files and publishes them with `segment.json`, without
copying their token bytes.

The store namespace binds the exact tokenizer artifact SHA-256, installed
tokenizers version, exact UTF-8 input, physical token encoding and the
choice to omit special tokens and EOS. Each index row binds the source
content SHA-256 to the location, count and SHA-256 of its token IDs. The
segment path is the digest of its canonical manifest, which binds both binary
files by size and digest.

Sealed caches live under `cache/source-tokens/sealed` in the Corpus work
root. An on-disk SQLite index locates documents using an 8 MiB page cache;
the writer keeps one active segment's index in memory and opens at most 16
sealed segments at a time. The locator can be rebuilt from the sealed
indexes. A retry registers a segment that was published before the locator
transaction finished. A lookup checks the sealed index and the digest of
the requested token range. The mixture assembly's `progress.json` reports
`sealed_token_cache_hits`, `published_token_segments` and `sealed_token_dtype`.

The `append` cache is the default. To try the `sealed` mode, use a new
mixture output directory. The first run fills a separate cache for that
encoding. Later sealed runs reuse content across mixture seeds, quotas and
projects within the same work root. Either cache mode can produce contiguous
output. The sealed cache can also back a reference schedule.

Set `segment_dtype: uint16` alongside `token_cache: sealed` to use two-byte
IDs in the segment token files. The tokenizer's complete vocabulary must
use IDs in 0–65,535, and the writer refuses any emitted ID outside that range.
The physical encoding is part of the segment identity, so the first uint16
run fills its own cache. Contiguous mixture output, prepared arrays and
decoded reference streams still use int32. Reference-only training and
offline worker dispatch read the physical encoding from the sealed manifest.

On an NVMe synthetic retained release with two workers and about 2 KiB of
text per document, the median schedule times across three trials were:

| Documents | Append cold | Sealed cold | Append warm | Sealed warm |
| --- | ---: | ---: | ---: | ---: |
| 2,048 | 0.612 s | 0.629 s | 0.249 s | 0.253 s |
| 32,768 | 6.570 s | 6.721 s | 1.536 s | 1.924 s |

The output receipt and file digests matched across modes. The larger
fixture used 1.45 MB more logical space across the cold and warm builds,
including the SQLite locator. A separate 65,536-document trial published
two segments and reused both on the warm run. Production-scale and
media-cold random access remain unmeasured.

The retained-mixture benchmark compares widths with `--segment-dtype`,
`--loader-blocks 8192` and `--portable-segments`. The figures below are
medians of three paired trials on ext4 over NVMe. Each pair uses the same
fixture, quotas and shuffled blocks of 1,024 targets. Decoded target,
record, prepared token and batch digests all matched. The first loader
pass followed Linux file-page eviction advice for the token files.

| Documents | Sealed token bytes, int32 → uint16 | First pass, int32 → uint16 | Warm pass, int32 → uint16 |
| --- | ---: | ---: | ---: |
| 8,192 | 65.03 → 32.52 MB | 0.251 → 0.175 s | 0.056 → 0.056 s |
| 32,768 | 260.15 → 130.07 MB | 0.633 → 0.455 s | 0.057 → 0.057 s |
| 65,536 | 268.43 → 134.21 MB | 0.719 → 0.566 s | 0.058 → 0.058 s |

On the largest fixture, first-pass storage reads went from 91.33 to
60.99 MB. The unchanged index was 2.49 MB. Cold mixture scheduling took
13.431 s with int32 and 13.161 s with uint16. Warm scheduling took 3.204 s
and 3.021 s. Reference-only packing took 0.404 s and 0.412 s. Offline
segment copy time went from 0.659 s to 0.338 s. These timings exclude
network transfer, model execution and accelerator transfers.

## Mixture output

A mixture defaults to `token_output: contiguous`. Its output directory holds:

| File | Holds |
| --- | --- |
| `train.tokens.ids` | every scheduled target, in order, as raw little-endian int32 (each record ends with EOS) |
| `train.records.jsonl` | one small row per scheduled record: document, family, source, stratum, exposure and its token count |
| `receipt.json` | schema version 2; binds both files by SHA-256 and size |

Mixture checkpoints sign prefixes of the staged token and record files. A
retry verifies each committed byte once, discards any uncommitted tail, and
appends to the same files. The schedule still journals index events separately.

`dataset.train_path` of a `local_token_mixture` run names `train.tokens.ids`.
Preparing it for training is a checked copy: the packer reads the record rows
for counts and strata, then copies the IDs in 64 MiB blocks while checking the
digest, the vocabulary range and the EOS that closes every record. No token is
ever parsed from text.

A stratum with one exposure (`max_exposures: 1`, or its own
`strata.NAME.max_exposures: 1`) writes each document as soon as it is encoded.
Only a repeated stratum makes the scratch index retain its encoded documents,
and that retained copy is still JSON. A `document_start` mixture adds no bytes
to these files; packing inserts the masked filler.

### Reference schedules

Set `token_cache: sealed` and `token_output: references` in the mixture
declaration to publish `train.schedule.bin`. Each 16-byte little-endian
entry names a segment ordinal, a token offset and the number of targets,
including EOS. Quota truncation reads the selected prefix and appends EOS.
`train.records.jsonl` records the matching content digest, provenance and
exposure. The receipt binds the schedule digest, exact encoding identity
and the ordered list of sealed segment digests. Its `train_tokens_bytes`
field gives the logical int32 size.

The CLI reports `train_schedule_path`. Use that file as
`dataset.train_path` for `local_token_mixture`. Preparation verifies the
referenced segments and streams their ranges into the usual contiguous
training arrays and supervision mask. The same schedule can be prepared at
another sequence length when using contiguous packing. With
`document_start`, the configured sequence length must match the declared
row length.

`verify-mixture --cold` reads every referenced segment and index. Routine
verification can reuse signed file receipts for their current generations.
`--audit` independently rebuilds the logical targets and verifies both the
decoded token stream and the record bytes. `--reseal` checks the stored
bytes, dependencies and range counts without independently selecting the
training documents.

Keep the mixture receipt, records and sealed store together. The receipt's
segment inventory records the retained dependencies. The existing Corpus
pruning commands exclude sealed token segments.

Three paired two-worker NVMe trials on 32,768 synthetic documents
scheduled 33.55 million targets with identical decoded hashes. Cold
schedule construction took 6.572 s with contiguous output and 6.609 s
with references. Warm construction took 1.700 s and 1.581 s. Reference
dependency verification added 0.130 s cold and 0.003 s warm. A warm
target scan including SHA-256 took 0.062 s for the contiguous file and
0.100 s for references. Across the cold and warm builds, logical working
bytes fell from 643.09 MB to 377.92 MB. These timings exclude packing and
training.

### Source-coherent ordering

Start with a verified reference mixture that uses `packing: contiguous`.
Copy its declaration and add the following option to create a paired arm.

```yaml
sequence_order:
  kind: source
  base_declaration: /work/baseline.yaml
  base_output: /work/baseline-mixture
```

```sh
klab corpus materialize-mixture ORDERED.yaml --output ORDERED_MIXTURE_DIR
```

The declarations must resolve to the same selection settings. Within each
stratum and exposure window, the output groups records by source in order
of first appearance. It preserves the parent's order within each group and
moves each record with its original token range. Selected documents, quota
cuts, exposure labels, sealed segment IDs and the contiguous supervision
mask stay fixed. The original seed sets the baseline order.

The receipt pins the base receipt and schedule digests and reports source
groups and transitions. Audit replays the base selection and derives the
same ordering. Reseal checks the ordering against the verified base output.
Use the new schedule path in the usual `local_token_mixture` config. A
retry before publication rebuilds the ordered metadata from the verified
base while the source segment files remain shared.

Use `verify-mixture --reseal` to refresh a stale base proof before creating
the paired arm, or to recover an ordered output if publication finished
before its local proof was written. Preparing the new order creates a
separate prepared-data identity.

The table reports medians of three synthetic NVMe trials per size, using two
CPUs and 2,048 bytes of source text per document. The reorder timed here runs
after base verification. It reads and writes about twice the selected record
and range bytes through scratch. It uses at most 16 open group files, with
one scratch file per active source and exposure group.

| Input documents | Sources | Selected metadata | Reorder | Source transitions |
| --- | --- | --- | --- | --- |
| 2,048 | 4 | 0.27 MB | 0.006 s | 787 → 3 |
| 32,768 | 8 | 4.41 MB | 0.052 s | 14,241 → 7 |
| 65,536 | 32 | 8.84 MB | 0.172 s | 31,678 → 31 |

On the largest fixture, reading 8,192 identical shuffled block positions
took 0.569 s for the original reference schedule and 0.815 s for the source
order after file-page eviction advice. Storage reads rose from 61.0 MB to
67.0 MB. Warm reads took 0.059 s and 0.060 s. Production storage locality,
large source counts and learning effects need separate measurements.

### Lexical-topic ordering

Use the same baseline declaration and output as for source ordering. Set
the following `sequence_order` and run the same `materialize-mixture`
command for a new output.

```yaml
sequence_order:
  kind: lexical
  base_declaration: /work/baseline.yaml
  base_output: /work/baseline-mixture
  max_topics: 16
```

Each stratum uses eligible, kept TRAIN text from the verified releases.
The index takes up to four case-folded word features from the first 4,096
characters of each document. It ranks a document's features by count,
length, and lexical order. Candidate anchors appear in at least two
documents and at most half the stratum, with a minimum upper bound of
two for tiny fixtures. The most supported anchors are retained up to
`max_topics` (1–32, default 16). Each document uses its highest-ranked
retained anchor. Documents without one stay in baseline order within
their own group. Groups stay within the stratum and exposure windows and
move exact record/range pairs through the same schedule writer.

`sequence_order` in the receipt reports the lexical index identity and
digest, eligible and assigned documents, anchors per stratum, scheduled
records with an anchor, and topic transitions. Shared words are a
heuristic for lexical coherence. The index reads no held-out features.

The index uses a bounded on-disk aggregation and publishes a 72-byte
lookup row per eligible document. The completed index in the work-area
cache can serve different tokenizers, seeds and quotas over the same
releases, family inventory, stratum mapping and options. Release additions
can affect anchor frequencies across a whole stratum and rebuild this
index. An interrupted index build retries its one text scan and reduction.
After the index completes, schedule retries and reseal can reuse it. Audit
independently rebuilds the index and base selection. Token segment IDs,
contiguous supervision and the existing trainer path are shared.

Three synthetic NVMe trials per size used 2,048 bytes of text per document
and eight planted topics. These medians separate index construction from
reordering with the finished index. Release verification is outside both
timings.

| Input documents | Lookup index | Index build | Reorder | Topic transitions |
| --- | --- | --- | --- | --- |
| 2,048 | 0.15 MB | 0.092 s | 0.008 s | 880 → 7 |
| 32,768 | 2.36 MB | 1.392 s | 0.078 s | 14,242 → 7 |
| 65,536 | 4.72 MB | 2.751 s | 0.148 s | 28,540 → 7 |

The largest fixture's reorder read and wrote 21.36 MB each through scratch.
The first pass over 8,192 shuffled reference blocks after file-page eviction
advice took 0.567 s at baseline and 0.818 s with lexical order. Storage reads
rose from 60.4 MB to 71.8 MB. Warm reads took 0.060 s and 0.065 s. A separate
one-million-document test with short generated text built a 72 MB index in
8.63 s. The main scratch database reached 236 MB and peak process RSS was
150 MB through 200,000 warm lookups. These are synthetic operator and loader
measurements. Production I/O and learning effects need separate evaluation.

### Source-link ordering

Use the same verified baseline and pair it with `kind: links`. Supply an
LF-delimited JSON sidecar with exactly `from_document_id` and
`to_document_id` on each row. Use lowercase SHA-256 document IDs from
the baseline's finalized family inventory. The declaration pins the
sidecar's exact bytes. It treats the edges as supplied evidence and
retains their direction.

```yaml
sequence_order:
  kind: links
  base_declaration: /work/baseline.yaml
  base_output: /work/baseline-mixture
  links:
    path: /work/source-links.jsonl
    sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
    max_neighborhoods: 16
```

Every endpoint must belong to the verified TRAIN family inventory.
A missing or held-out endpoint refuses the sidecar before publication.
The index deduplicates identical directed pairs. Within each stratum,
targets with the most distinct incoming links become anchors, with ties
ordered by document ID. `max_neighborhoods` allows 1–32 anchors per
stratum and defaults to 16. Anchors keep their own group. A document that
links to multiple anchors joins the highest-ranked one. Links across
strata and self-links are inspectable and do not form packing groups.
Documents without a group retain the baseline order within a fallback
group. Each exposure is ordered separately with the exact baseline
record/range pairs and supervision mask.

Materialize through the usual mixture command. The receipt records
the evidence SHA-256, index identity, exact edge and input-row counts,
per-stratum anchor coverage, linked scheduled records and neighborhood
transitions. Reseal can reuse the verified index. Full audit rebuilds it
from the pinned sidecar and the authenticated inventories. Changing the
tokenizer, quotas or seed can reuse the same index.
Keep the pinned sidecar available for audits and index reconstruction.

```sh
klab corpus inspect-links ORDERED.yaml --output ORDERED_MIXTURE_DIR --document DOCUMENT_ID --limit 20
```

The query returns outgoing IDs in order and a `next_after` cursor. Pass
the cursor as `--after` for the next page. The limit is 1–1,000. Warm
queries authenticate the index and use a bounded range lookup over its
primary key. Use `verify-mixture` for token and schedule verification.

Three synthetic NVMe trials per size grouped documents around eight linked
anchors. The index stores directed pairs, their scoped endpoints and the
assignments in narrow SQLite tables. These medians exclude release
verification and measure reordering with the finished index.

| Input documents | Directed edges | Index build | Reorder | Neighborhood transitions |
| --- | --- | --- | --- | --- |
| 2,048 | 2,040 | 0.057 s | 0.013 s | 886 → 7 |
| 32,768 | 32,760 | 0.215 s | 0.083 s | 14,252 → 7 |
| 65,536 | 65,528 | 0.418 s | 0.154 s | 28,537 → 7 |

The largest fixture built an 11.05 MB index from 11.27 MB of edge input.
Its build issued 43.96 MB of logical reads, including database operations
and hashing the closed index. The reorder read 24.27 MB and wrote 21.36 MB
of metadata. Reading 8,192 shuffled reference blocks after file-page
eviction advice took 0.570 s at baseline and 0.809 s ordered. Storage reads
rose from 60.4 MB to 72.1 MB, with both warm passes at 0.064 s.

A separate synthetic graph with one million edges built a 121 MB index in
5.95 s at 86 MB peak process RSS. On the warm index, 4,096 pages of 32
outgoing links from a 500,000-link hub took 0.052 s total in the query
kernel. The query timing excludes CLI startup and receipt checks. Link
ordering stays opt-in while production locality and learning are measured.

### Explicit-entity ordering

Set `kind: entities` with a verified reference baseline that uses contiguous
packing. The pinned JSONL sidecar has one `document_id` and one `entity_id`
per row. Document IDs come from the finalized TRAIN family inventory. An
entity ID is a case-sensitive `namespace:identifier`, such as
`wikidata:Q42`. Preserve the IDs assigned by the source adapter so that
references to the same entity agree across documents and partitions.

```yaml
sequence_order:
  kind: entities
  base_declaration: /work/baseline.yaml
  base_output: /work/baseline-mixture
  entities:
    path: /work/entity-memberships.jsonl
    sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
    max_neighborhoods: 16
```

The index refuses a missing or held-out document before publication and
deduplicates exact document/entity pairs. Within each stratum, entities
with at least two distinct member documents compete for up to
`max_neighborhoods` groups (1–32, default 16). Rank by member count and then
entity ID. Documents with several eligible entities join their
highest-ranked group. Documents without an assigned entity retain baseline
order within the fallback group. Both arms use the same selected record/range
pairs, exposure windows, supervision mask and sealed token segments.

The signed receipt records the evidence digest, input rows, unique pairs,
distinct entity IDs, coverage and transitions. The same index can serve
different quotas, seeds and tokenizers across the same inventory, evidence
and grouping options. Full audit independently rebuilds that index and the
baseline selection. Use the usual mixture materialization command.

```sh
klab corpus inspect-entities ORDERED.yaml --output ORDERED_MIXTURE_DIR --entity wikidata:Q42 --limit 20
```

The query returns TRAIN document IDs in ascending order, including members
of entities outside the packing cap. Use `--after` with `next_after` to
continue. Each page contains up to 1,000 IDs. Warm inspection uses the
authenticated edge or membership index with an indexed cursor lookup.

The shared neighborhood producer changes the index computation identity
for previously published link-ordered mixtures. Rematerialize those arms
at new output paths from their original declarations and verified baselines.
Use `verify-mixture --reseal` to refresh stale baseline proofs. The same
release, tokenizer and token segments can serve the new output.

Three synthetic NVMe trials per size used one explicit membership per
document and eight entities. These medians exclude release verification.
Reordering uses the finished index.

| Input documents | Membership index | Build | Reorder | Entity transitions |
| --- | --- | --- | --- | --- |
| 2,048 | 0.32 MB | 0.022 s | 0.008 s | 886 → 7 |
| 32,768 | 4.83 MB | 0.203 s | 0.075 s | 14,252 → 7 |
| 65,536 | 9.61 MB | 0.409 s | 0.150 s | 28,537 → 7 |

On the largest fixture the build issued 25.09 MB of logical reads for
7.60 MB of evidence. After file-page eviction advice, the first pass over
8,192 shuffled reference blocks took 0.568 s at baseline and 0.680 s for
entity order. Warm passes took 0.059 s for both. Link index membership and
schedule digests matched before and after the refactor in all nine paired
trials. On the largest fixture the link build medians were 0.418 s before
and 0.414 s after.

A separate synthetic million-membership test with 500,001 distinct entity
IDs built a 104 MB index in 4.46 s at 78 MB peak process RSS. The warm index
returned 4,096 pages of 32 member IDs in 0.049 s in the query kernel. This
timing excludes CLI startup and receipt checks. Production loader and
learning measurements remain separate.

### Offline transfer inspection

```sh
klab corpus verify-mixture MIXTURE.yaml --output MIXTURE_DIR --inspect-transfer
```

The `transfer` section of the JSON result adds physical byte accounting to
normal verification. `whole_segments.bytes` counts the referenced token,
index and manifest files for a worker with an empty asset cache. The cache
determines which files a worker actually receives.

`selected_ranges.unique_token_bytes` counts each selected physical token
position once. `repeated_token_bytes` counts additional exposures to those
positions. The count uses the actual segment width and excludes the EOS
added by the schedule. `unselected_token_bytes` gives the rest of the
sealed token payload. `logical_target_bytes` gives the scheduled int32
stream size including EOS, before padding or supervision masks.
Projected dispatch adds `selected_ranges.rebased_schedule_bytes` and a
small projection manifest to the selected physical bytes.

After verification, inspection reads the schedule and merges overlapping
and adjacent physical ranges. It uses bounded sort runs under
`--scratch-dir` or the host temporary directory. A new inspection starts a
new sort. It reports the scratch byte bound and the logical bytes written.
The sort reads the schedule and temporary ranges. Verification covers the
token payload separately.

The retained-mixture benchmark accepts `--transfer-plan`. On the uint16
NVMe fixture with 32,768 documents, the plan counted 63.00 MB of unselected
tokens in a 132.48 MB segment inventory. The 65,536-document fixture had
61.47 KB of unselected tokens in 136.70 MB. After verification, the median
inspection times across three trials were 103 ms and 155 ms, respectively.

### Training batch loaders

For an experimental direct loader, set `training.mixture_loader: references`
on a `local_token_mixture` run whose `dataset.train_path` names
`train.schedule.bin`. Run preparation as usual. The existing training
sampler reads its input and target IDs from verified segment ranges and uses
the prepared supervision mask. A sparse position index supports shuffled
batches, quota truncation, EOS and `document_start` filler. The loader keeps
at most 16 segment files open and caches at most 256 pages of range metadata.

By default, preparation retains the contiguous arrays.
`training.mixture_prepared_storage: references` with the direct loader omits
`train.npy` from the prepared cache, local stage inputs and run data. The
bundle owns the schedule, records, supervision mask and validation arrays.
Preparation checks the selected IDs and records the digest the packed
`train.npy` would have had. Evidence readers can open the same logical
training stream through the bundle.

Local continuation uses the run-owned schedule and records and the shared
sealed store. It can resume after the original mixture output moves.
[Offline worker dispatch](workers.md#preparation-and-durable-identity) can also bundle
the referenced sealed segments. The worker verifies their identities and
owns a copy in the run, so continuation can use the run even when the
original store is unavailable.

With `--include-referenced-token-segments`, dispatch ships entire referenced
segments, including tokens outside the selected ranges. A sparse selection
can ship more bytes than contiguous storage. The retained-mixture benchmark's
`--portable-segments` option
measures the owned copy and the subsequent verification using its file
proofs. On ext4 over NVMe, the one-segment 65,536-document fixture packaged
270.91 MB in 0.360 s. The copy issued 541.86 MB of process reads, including
destination authentication. Verification of the resulting closure took
0.95 ms and read 624 bytes. This trial excludes input preparation and
network transfer.

With `--project-referenced-token-segments`, dispatch copies each selected
physical range once and writes a rebased schedule over the compact files.
The original prepared manifest, record bytes, logical token digest and mask
stay bound to the run. Cold verification checks that every projected
position is used and replays the original packed token digest, including BOS,
EOS and filler. Receipt reuse skips the replay after checking all member
bytes or their current-generation file proofs. The rebased schedule costs 16
bytes per record, and the first replay visits each scheduled exposure.

The retained-mixture benchmark compares both policies with
`--portable-segments --project-selected`. On uint16 NVMe fixtures, the
following medians use three trials after source verification. Replay checks
the selected coverage and original packed digest from a fresh projection.

| Documents | Whole segment files | Projected files | Whole copy | Projected packaging | Projected replay |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 8.28 MB | 4.21 MB | 0.022 s | 0.014 s | 0.014 s |
| 32,768 | 132.48 MB | 67.34 MB | 0.169 s | 0.130 s | 0.287 s |
| 65,536 | 136.70 MB | 134.68 MB | 0.185 s | 0.367 s | 0.364 s |

The comparison covers local packaging and replay. Network throughput and
the receiving worker's digest cache determine the transfer savings.

On the same verified reference schedules, the packing benchmark compared
contiguous and reference-only output on ext4 over NVMe. Times are medians of
three paired trials at the first two sizes and one pair at 65,536 documents.
Training files include the mask and either `train.npy` or the schedule and
records. Validation arrays, verification and the shared store are outside this
measurement. The logical token digest and supervision mask matched each time.

| Documents | Contiguous packing | Reference-only packing | Training files |
| --- | ---: | ---: | ---: |
| 2,048 | 0.034 s | 0.021 s | 10.49 → 2.37 MB |
| 32,768 | 0.415 s | 0.216 s | 167.77 → 37.92 MB |
| 65,536 | 0.852 s | 0.429 s | 335.54 → 75.85 MB |

The retained-mixture benchmark accepts `--loader-blocks 8192` and
`--loader-row-tokens 1024` with the reference output. It runs the same
shuffled block order through the two loaders, compares input and label
hashes, and reports batch latency and storage reads. It times the CPU
loader without worker prefetch, model execution or accelerator transfers.

On NVMe, three paired trials read 8,192 shuffled blocks per loader.
Each block held 1,024 targets. The first pass followed Linux file-page
eviction advice for the token files. The next pass reused the same order
with warm pages and the same prepared mask in both modes.

| Documents | Loader | First pass | Warm pass | Warm p95 per four blocks |
| --- | --- | ---: | ---: | ---: |
| 8,192 | Contiguous | 0.218 s | 0.028 s | 14 µs |
| 8,192 | References | 0.252 s | 0.055 s | 33 µs |
| 65,536 | Contiguous | 0.899 s | 0.030 s | 15 µs |
| 65,536 | References | 0.719 s | 0.059 s | 35 µs |

Input and label hashes matched. On the larger fixture, storage reads on
the first pass were 75.4 MB for contiguous and 91.3 MB for references.
Creating the sparse reference index took 70 ms after verification.
Accelerator utilization and production step throughput remain unmeasured.

## Mixture speed

Synthetic release of 40,000 documents, 210M scheduled tokens, RAM-backed
storage, 48 threads, same output tokens in every row:

| Step | Time | Rate |
| --- | --- | --- |
| Cold mixture build | 24.2 s | 8.7M tokens/s |
| Rebuild with a warm content token store | 12.9 s | 16M tokens/s |
| Packing the mixture for training | | 126M tokens/s |

About 6 s of each build is fixed verification work that does not grow with the
corpus.

### Checkpoint I/O

The table compares the former copied checkpoint segments with authenticated
prefixes on ext4 over NVMe. Each synthetic window appends 64 MiB of tokens and
2 MiB of record bytes, then commits. Retry restores the full committed prefix
after an extra uncommitted 1 MiB append. File pages were released before retry.
Times are medians of three trials per size.

| Token payload | Fresh write and commit, before → after | Restore, before → after |
| --- | ---: | ---: |
| 64 MiB | 0.206 s → 0.155 s | 0.112 s → 0.059 s |
| 256 MiB | 0.649 s → 0.440 s | 1.615 s → 0.246 s |
| 1 GiB | 3.703 s → 2.724 s | 3.045 s → 0.958 s |

At 1 GiB, fresh storage writes fell from 2.22 GB to 1.11 GB, and the
staged files and journal used 1.11 GB of logical space instead of 2.21 GB.
Retry read about 1.11 GB from storage in both revisions, and storage writes
fell from 1.11 GB to zero. Output digests matched. The experiment isolates
checkpoint I/O. It excludes tokenization, real index events and training.

On a rotational ZFS pool at 256 MiB of token payload, median fresh
write-and-commit time fell from 4.017 s to 2.458 s and cached restore from
1.817 s to 0.137 s across three trials. File-page advice did not evict the
ZFS cache, so media-cold restore on rotational storage remains unmeasured.
