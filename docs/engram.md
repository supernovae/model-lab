# Engram memory

An Engram memory is a learned lookup table added to the decoder. At each
position the model hashes the most recent tokens (or bytes) into the table,
reads a vector, projects it to the model width and adds it to the hidden state
through a learned gate:

$$h'_t = h_t + \sigma(g(h_t))\,A\,M[a(x_{\leq t})]$$

The address $a$ uses inputs up to position $t$ only, so the table never sees the
token being predicted. The table is ordinary model state: it is trained with the
rest of the model and saved in its checkpoints. Adding documents does not fill
it; training does.

| `model.memory` | Address |
| --- | --- |
| `ngram` | The last `memory_ngram_size` token IDs. |
| `byte` | The last `memory_ngram_size` raw UTF-8 bytes, prepared beside the token IDs. |
| `portable` | A frozen byte-addressed table exported from another run, attached with a new projection and gate. |

```sh
klab train configs/smoke_memory_cpu.yaml --run-id token-memory
klab train configs/smoke_byte_memory_cpu.yaml --run-id byte-memory
```

## Token memory

Set `model.memory: ngram` with:

- `memory_table_size`: rows in the table.
- `memory_ngram_size`: how many recent tokens form an address.
- `memory_dim`: width of a table row, independent of the model width.

Optional fields add lookup streams. `memory_ngram_orders` (for example `[2, 3]`)
looks up several n-gram lengths, and `memory_hash_heads` hashes each one several
ways. Every stream has its own table.

### Combining streams

`memory_variant` chooses how streams combine:

- `hidden_gate` (default): average the projected streams, gate on the hidden
  state alone.
- `context_key`: concatenate the streams, project separate keys and values, and
  gate on the similarity between the normalized hidden state and the key.
  `memory_gate_transform: signed_sqrt` applies a signed square root to that
  similarity before the sigmoid; `identity` (default) does not.

`memory_conv_kernel_size: 4` refines the gated values with a normalized,
depthwise causal convolution whose dilation is the longest n-gram order; `0`
(default) turns it off.

### Where the memory joins the model

`memory_injection` places the memory:

- `final` (default): after the last block and the final norm, just before the
  output head. Memory added here reaches the logits without normalization.
- `embedding`: on the token embeddings.
- `after_block`: after the block named by `memory_block_index`, or after each of
  several blocks with `memory_block_indices` (for example `[1, 14]`). Each site
  has its own tables.

### Addressing

`memory_addressing` chooses the hash. The default hashes a polynomial of the
token IDs into `memory_table_size` rows. `prime_xor_v1` uses seeded XOR
coefficients over prime-sized tables: set `memory_hash_seed` and
`memory_table_sizes`, one distinct prime per stream and site, in site, order,
head order (`memory_table_size` equals the first entry).

A tokenizer can map IDs before hashing, so that tokens differing only in case or
spacing share addresses. `klab tokenizer memory-map` builds the map and writes
it into a config:

```sh
klab tokenizer memory-map "$TOKENIZER" --mode deepseek_safe_v1 \
  --output "$WORK/map.json" --bind-config model.yaml \
  --config-output model-mapped.yaml
```

`identity` keeps every ID distinct. `deepseek_v1` folds case, whitespace and
accents; `deepseek_safe_v1` does the same but keeps special tokens and partial
byte sequences apart. The map affects addresses only: embeddings, labels and
decoding keep the original IDs. Preparation and loading check the map against
the tokenizer's bytes.

`memory_independent_init: true` initializes the memory from its own random
stream, so a model with memory and one without share identical trunk weights at
the same seed.

### Training settings

For AdamW, `optimizer.memory_table_lr_multiplier` (default 1) scales the
tables' learning rate and `optimizer.memory_table_weight_decay` overrides their
weight decay. Tables use dense gradients and optimizer state.

### Add memory to a trained dense model

```sh
klab train target.yaml --promote "$CHECKPOINT_GENERATION" \
  --add-ngram-memory --run-id with-memory
```

Lab verifies the source checkpoint, requires the same tokenizer, backbone and
attention, copies the trunk weights, initializes the new tables and sets the
memory output projections to zero. The child starts with exactly the source
model's outputs and learns to use the memory from there. It gets a fresh
optimizer, schedule and counters, and records its parent. Resume it later with
`--resume` as usual.

To also lengthen the context, add `--extend-context` and set `model.max_seq_len`
and `training.seq_len` in the target. Only the context may change; RoPE keeps
its base and frequencies, so the longer context still has to be trained.

To train only the memory, list every memory tensor (by full name) in
`training.trainable_parameters`, including each output projection.

### Diagnostics

Each update logs, per memory and per stream (`engram/stream_N/*`, and
`engram/port/<block>/*` for multiple sites): lookup count, distinct rows used,
repeated row accesses (`collision_count`, which includes identical keys), row
reuse rate, table utilization, the largest row's share, mean gate value and
vector norms. High reuse can come from a small table or repetitive text; it does
not by itself mean the memory helps. The dashboard's Engram page and
`klab explore` show the same values.

## Byte memory

`model.memory: byte` addresses the table by raw UTF-8 bytes rather than token
IDs. Data preparation writes one address per token beside the token arrays; each
hashes the bytes up to that token's exact byte boundary. Text is not normalized,
case-folded or escaped. EOS gets address zero. When a character spans several
tokens, each token contributes only the bytes it actually covers, and
generation computes addresses the same way.

The hash depends on the table size: sizes that share a factor with 257 use a
mixing hash, others a polynomial over 257. Equal byte spans produce equal
addresses regardless of how a tokenizer splits them.

## Portable tables

A byte-addressed table can be exported from one run and attached to another
model with a different tokenizer or width:

```sh
klab engram export RUN_ID --runs-dir "$RUNS" --output "$WORK/memory.engram"
klab engram inspect "$WORK/memory.engram"
```

The package holds the table and a manifest of its hashing, n-gram size, shape
and digest; it has no projection or gate. A target config with
`model.memory: portable` and `model.memory_package_path` freezes the table and
trains only a new projection and gate.

## Engram packs

An Engram pack is a separate, versioned archive of knowledge records. Each
record has an `id`, a license and either a `subject`/`relation`/`value` triple,
`text`, or both:

```json
{"id":"blorvia-capital","subject":"Blorvia","relation":"capital","value":"Zanther","aliases":["Zanther City"],"license":"CC0-1.0"}
```

```sh
klab engram pack compile records.jsonl --output "$WORK/geography.enpack" \
  --name geography --namespace synthetic --created-at 2026-09-23T00:00:00Z
klab engram pack inspect "$WORK/geography.enpack"
klab engram pack verify "$WORK/geography.enpack"
```

- **`compile`** reads JSONL or Parquet, refuses an existing output and never
  guesses a missing license; `--license`, `--source-name` and
  `--source-revision` supply defaults explicitly.
- **`inspect`** reads the manifest without hashing the payload. **`verify`**
  checks every member, hash and record; `--expected-pack-id` pins a revision.
- **Optional assets:** `--lexical-package FILE` embeds a portable table.
  `--semantic-keys`, `--semantic-values` and `--semantic-metadata` together add
  FP32 key and value matrices for a subset of records, with the encoders named
  in the metadata.

A pack's ID proves its bytes, not that its facts are right or that a model can
use them.

## Learn the mechanism

`klab learn describe lexical-engram`, `byte-engram`, `portable-engram` and
`engrampack` walk through each part with tensor shapes; see
[architecture](architecture.md#explore-a-mechanism-before-training). To use
n-gram repeats to speed up generation instead, see
[speculative decoding](speculative-decoding.md).
