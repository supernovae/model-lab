# Reusable token store

Tokenizing a text file with a tokenizer gives the same token IDs every time, so
Lab does it once and keeps the result. Preparing data for another model, a
larger token budget or a newer Lab revision then copies stored tokens instead of
tokenizing again.

It is used automatically by `klab data prepare` (and by `klab train` when it
prepares data) for `dataset.source: local_text` with a byte-level BPE tokenizer
and no byte-addressed memory. Other sources use the chunked path described in
[runtime](runtime.md#operational-resource-envelopes). The prepared arrays and
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

- Only `local_text` JSONL is covered. Token mixtures, snapshots, chat data and
  byte-addressed memory still use their existing paths.
- Token IDs are stored as int32 whatever the vocabulary size.
- The check that train and validation documents are disjoint still reads both
  source files once per preparation.
