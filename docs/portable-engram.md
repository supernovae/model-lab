# Portable Engram

A portable Engram package stores only the tokenizer-independent byte-addressed latent table plus a versioned manifest. The manifest records normalization, polynomial hashing identity, N-gram size, table shape, latent width, and the table SHA-256. It intentionally excludes a backbone-specific projection or gate.

Export from a byte-memory checkpoint:

```sh
uv run klab engram export RUN_ID --runs-dir runs --output artifacts/memory.engram
uv run klab engram inspect artifacts/memory.engram
```

`PortableEngramAdapter` freezes the exported table and trains only an output projection and context gate for a target hidden width. Byte-equivalent addressing alone does not prove transfer; use held-out facts and controlled baseline comparisons.

Held-out evaluation records exact greedy completion plus length-normalized expected-value log probability and reciprocal rank among the manifest's held-out candidate values. Candidate values are evaluation labels only; they are never used in source or target training. New reports use the immutable `withheld-v2-<manifest-sha>.json` filename.

The current two-case control remains negative: baseline, random frozen-table, and trained portable-adapter runs each have zero exact matches and reciprocal rank `0.75`. Mean expected-value log probability is `-5.0596`, `-4.9939`, and `-5.1395`, respectively. The trained adapter did not outperform the random-table control under this protocol.

For a configuration-only walkthrough, use the [`portable-engram` lesson](research/lesson-paths.md) with `--memory-package PATH`; it verifies and copies a real package and derives its table fields instead of guessing them. The artifact-only [`engrampack` lesson](research/lesson-paths.md) separately teaches record-pack compilation, inspection, and verification; a records-only pack is not executable model memory.

## Engram packs

An Engram pack is a separate, versioned knowledge artifact. It canonicalizes local JSONL or Parquet records and may include a copied legacy byte-addressed package, supplied semantic vectors, or both. A records-only pack is valid; it does not imply that any model can retrieve or use its facts. The legacy `engram export` and `engram inspect` commands remain unchanged.

Input records use a strict schema. A record must contain either a complete `subject`/`relation`/`value` triple, nonblank `text`, or both, plus a `license` unless one is explicitly supplied to `compile`. `id` is unique within the pack. `query` and `aliases` are authored text; `hard_negative_ids` refer to other record IDs in the same pack. Dates and timezone-aware timestamps are canonicalized to UTC without normalizing authored text. JSONL is one object per line; Parquet is read locally in bounded batches.

```json
{"id":"blorvia-capital","subject":"Blorvia","relation":"capital","value":"Zanther","aliases":["Zanther City"],"license":"CC0-1.0"}
```

Compile, inspect, and verify:

```sh
klab engram pack compile records.jsonl \
  --output artifacts/geography.enpack \
  --name geography --namespace synthetic \
  --created-at 2026-09-23T00:00:00Z
klab engram pack inspect artifacts/geography.enpack
klab engram pack verify artifacts/geography.enpack
```

Use `--license`, `--source-name`, and `--source-revision` only for explicit defaults. A revision default requires a source name and applies only to records inheriting or naming that source. Source paths never become provenance; missing licenses are never guessed. Compilation refuses an existing output and publishes a verified pack with an atomic no-replace rename.

`inspect` validates the bounded manifest and pack identity but reports `verification_status: "not_verified"`; it does not hash or validate payloads. `verify` checks the exact member inventory, content hashes, canonical records, provenance summary, and optional components. `--expected-pack-id` pins the verified revision. The pack ID establishes byte identity and integrity, not source ownership, redistribution permission, semantic correctness, encoder honesty, retrieval quality, or successful model use.

### Optional assets

`--lexical-package FILE` copies an existing portable Engram file byte-for-byte as `lexical.engram`. Its width remains independent of model width, and existing `load_portable_engram` consumers can load the embedded file. Its table slots are not knowledge-record counts.

Supply all three semantic inputs together:

```sh
klab engram pack compile records.jsonl \
  --output artifacts/semantic.enpack \
  --name semantic --namespace synthetic \
  --semantic-keys keys.safetensors \
  --semantic-values values.safetensors \
  --semantic-metadata semantic.json
```

The two safetensors files contain exactly one contiguous finite FP32 matrix each: `keys` with shape `[M, d_key]`, and `values` with shape `[M, d_value]`. `semantic.json` declares format `klab-semantic-assets`, version `1`, ordered `record_ids` for those rows, pinned key/value encoder names, revisions and SHA-256 identities, and `key_normalization` (`none` or `l2`). Record IDs must be unique and present in the pack; vectors may cover a subset. An `l2` declaration is verified, never applied by the compiler. The manifest records the independent key and value widths plus a shared representation-space identity; it contains no model width, tokenizer IDs, or model checkpoint identity.

The portable **byte-addressed lexical-table** path implements export, verified loading, attachment, and target-adapter training. It does not provide semantic retrieval by itself. A separate verified semantic EngramPack retriever and direct `DenseLM` adapter API are implemented; see [Verified semantic memory](research/semantic-memory.md). Encoder identities remain producer declarations, not independent certification that two encoders share meaningful coordinates. The two-case portable-adapter result remains negative and is not evidence of general transfer.
