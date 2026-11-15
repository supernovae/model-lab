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

Token n-gram memory uses seeded XOR coefficients over prime-sized tables: set
`memory_hash_seed` (default 0) and `memory_table_sizes`, one distinct prime per
stream and site, in site, order, head order (`memory_table_size` equals the
first entry). Raw-byte memory hashes a base-257 polynomial and refuses a
`memory_table_size` that is a multiple of 257.

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

## Size the tables

A playbook for token n-gram memory. Each recommendation names its basis:
**paper** (the Engram paper, arXiv 2601.07372, and DeepSeek's reference demo),
**measured** (`klab data engram-sizing` on real tokens) or **unmeasured** (a
judgment nothing has tested here).

**When to use it.** Memory pays on text with repeated local patterns and
facts: the paper's gains concentrate on knowledge and reasoning benchmarks, and
suppressing the module at inference collapses factual recall to 29-44% of the
original (paper, Sec. 6.3). It adds parameters, not per-token compute, so the
cost is mainly training memory and optimizer traffic. For a small
model on narrow data, `klab try` the arm with `model.memory: none` first.

**Defaults and why.**

| Setting | Recommended | Basis |
| --- | --- | --- |
| `memory_ngram_orders` | `[2, 3]`; 4-grams are slightly worse at a fixed budget | paper Sec. 6.2 |
| `memory_hash_heads` | paper: 8. At fixed total rows, more heads collide more (proxy text, order 3: one head of 2.1M rows clobbers 26% of occurrences; two of 1M, 32%; four of 0.5M, 41%), so add heads only with their own full-size tables | paper Sec. 2.2 for 8; measured for the trade |
| `memory_variant`, `memory_conv_kernel_size` | `context_key`, `4` (gate on the hidden state, dilated causal convolution) | paper Sec. 2.3-2.4, Table 5 |
| `memory_block_indices` | an early block, then a mid-depth one; one module is within 0.002 validation loss of two at the same memory budget (1.770 against 1.768, split by lowering the dimension) | paper Sec. 6.2 |
| `optimizer.memory_table_lr_multiplier`, `memory_table_weight_decay` | `5`, `0` | paper Sec. 4.1, Table 5 |
| `memory_padding_token_id` | a special token that never occurs in text, such as EOS | paper demo `pad_id`; unmeasured here |
| `memory_dim` | paper: 80 per stream (1280 over 16 streams); 64 is unmeasured | paper Table 5 |
| rows per stream | at least 17.5 per vocabulary entry; size by measurement below | paper Table 5; measured |

**What to measure first.** Table size is the setting that decides whether the
memory can hold anything. The paper trains 2,262,400 rows per stream for a
129,280-token vocabulary (Engram-27B) and 7,239,680 for Engram-40B, and loss
falls as a power law in rows up to 10 million per table (Sec. 3.2). It reports
no collision rates and no table-size ablation beyond that. A third-order
n-gram space over billions of tokens is far larger than any table, so most
rows hold many n-grams; what matters is how much of the traffic a row's
dominant n-grams carry. `klab inspect` prints a note when tables are small for
the vocabulary and says what it cannot know.

```sh
klab data engram-sizing "$PREPARED" --config model.yaml --target-tokens 12000000000
```

The command hashes a sample of the tokens (`--max-tokens`, default 33.5
million, about 40 bytes per token of memory) with the exact functions the model
uses. `PREPARED` is a prepared root (`train.npy`), a token store directory, a
`.npy` file or a raw int32 file. For each site and order it reports:

- distinct n-grams, singletons, top-k mass, the Zipf slope and a Heaps' law
  projection to `--target-tokens`;
- per candidate rows per head (`--sizes`, default powers of two from 16,384 to
  16,777,216): occupancy, the share of occurrences that **share** a row with
  another n-gram next to the random-hash expectation, and the share
  **clobbered**, where other n-grams carry over half of the row's traffic,
  on one head and on every head;
- the parameter cost of the smallest size meeting `--max-clobbered` (default
  25% of occurrences clobbered on all heads) and a `memory_table_sizes` list
  to paste into the config.

Reading it: nearly all occurrences share a row at any affordable size, so
"shared" alone is not informative; clobbered is. Measured shared fractions
match the random-hash expectation, so the hash behaves as designed. Heads
decorrelate less than independence predicts, because rare n-grams lose on every
head. Frequent n-grams, which carry the learnable signal, are what a larger
table protects. The default 25% is a starting point, not a measured optimum:
the paper shows larger is monotonically better, so the real limit is the
budget.

**Budget.** Tables are `rows x memory_dim` parameters per stream. They train
with dense gradients and Adam state, about 16 bytes per parameter in fp32
(`klab inspect` gives the exact estimate), so every million rows at
`memory_dim: 64` adds roughly 1 GiB of training state. Check that against the
device before enlarging, and use `klab inspect` for the whole config.

**Failure modes.**

- Tables near the vocabulary size: most of every row's traffic belongs to other
  n-grams (on 32k-vocabulary proxy text, 65,537 rows clobber 73% of order-3
  occurrences). Whether that costs quality is unmeasured; compare against
  `model.memory: none` with `klab try`.
- Sizing from one corpus and training on another, or from a prefix of an
  ordered corpus: the command samples blocks across the whole input; pass the
  tokens you will train on.
- A padding ID that is a real token merges sequence starts with that token's
  n-grams; training windows restart at `training.seq_len`, which the command
  honors from `--config`.
- Equal primes within a site are refused; use `memory_table_sizes` as printed.
- Tokenizer folding (`klab tokenizer memory-map`) reduces the effective
  vocabulary by 23% for a 128k tokenizer (paper Sec. 2.2); its effect at 32k is
  unmeasured. The paper reports it among the components whose removal costs most.

The sample is a lower bound on collisions for a larger corpus: rerun on the real
tokens, and with half `--max-tokens` to see whether the clobbered share still
moves.

## Byte memory

`model.memory: byte` addresses the table by raw UTF-8 bytes rather than token
IDs. Data preparation writes one address per token beside the token arrays; each
hashes the bytes up to that token's exact byte boundary. Text is not normalized,
case-folded or escaped. EOS gets address zero. When a character spans several
tokens, each token contributes only the bytes it actually covers, and
generation computes addresses the same way.

The hash is a polynomial over 257, so `memory_table_size` must not be a
multiple of 257. Equal byte spans produce equal addresses regardless of how a
tokenizer splits them.

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
