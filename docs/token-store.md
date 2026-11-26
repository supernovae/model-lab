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
encoder and int32 append format for the active segment. The writer seals by
rename before exceeding 256 MiB of tokens or 65,536 documents and at
successful completion.
A document that exceeds the byte target occupies its own segment. Sealing
syncs the active files and publishes them with `segment.json`, without
copying their token bytes.

The store namespace binds the exact tokenizer artifact SHA-256, installed
tokenizers version, exact UTF-8 input, little-endian int32 encoding and the
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
`sealed_token_cache_hits` and `published_token_segments`.

The `append` cache is the default. To try the `sealed` mode, use a new
mixture output directory. The first run fills a separate cache for that
encoding. Later sealed runs reuse content across mixture seeds, quotas and
projects within the same work root. Either cache mode can produce contiguous
output. The sealed cache can also back a reference schedule.

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

### Training batch loaders

For an experimental direct loader, set `training.mixture_loader: references`
on a `local_token_mixture` run whose `dataset.train_path` names
`train.schedule.bin`. Run preparation as usual. The existing training
sampler reads its input and target IDs from verified segment ranges and uses
the prepared supervision mask. A sparse position index supports shuffled
batches, quota truncation, EOS and `document_start` filler. The loader keeps
at most 16 segment files open and caches at most 256 pages of range metadata.

The run still keeps its contiguous prepared arrays for checkpoint inventory,
validation and evidence readers. The direct loader needs the original
declaration, schedule, records and sealed store on the execution host,
including when resuming from a checkpoint. The default
`training.mixture_loader: contiguous` reads the prepared arrays. Direct
loading with a reference-only prepared bundle is tracked in
[#569](https://github.com/supernovae/model-lab/issues/569).

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
