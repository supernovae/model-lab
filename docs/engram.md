# Engram memory

An Engram memory, from DeepSeek's [Conditional Memory via Scalable Lookup](https://arxiv.org/abs/2601.07372),
is a learned lookup table added to the decoder. At each position the model
hashes the most recent tokens into the table, reads a vector, and adds it to the
hidden state through a learned gate. A key projected from the retrieved vector is
compared with the hidden state to set the gate, so the memory decides how much
of what it found to use:

$$h'_t = h_t + \sigma\!\left(\frac{\mathrm{RMSNorm}(h_t)\cdot\mathrm{RMSNorm}(W_K e_t)}{\sqrt{d}}\right) W_V e_t,\qquad e_t = M[a(x_{\leq t})]$$

The address $a$ uses inputs up to position $t$ only, so the table never sees the
token being predicted. The table is ordinary model state: it is trained with the
rest of the model and saved in its checkpoints. Adding documents does not fill
it; training does.

| `model.memory` | Address |
| --- | --- |
| `ngram` | The last `memory_ngram_size` token IDs. |
| `byte` | The last `memory_ngram_size` raw UTF-8 bytes, prepared beside the token IDs. |

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
ways. Every stream has its own table. The streams are concatenated, and one
value projection and one key projection read the result.
`memory_gate_transform: signed_sqrt` applies a signed square root to the
key-hidden similarity before the sigmoid; `identity` (default) does not.

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

## Byte memory

`model.memory: byte` addresses the table by raw UTF-8 bytes rather than token
IDs. It has a single stream, no key projection and a plain linear gate on the
hidden state. Data preparation writes one address per token beside the token arrays; each
hashes the bytes up to that token's exact byte boundary. Text is not normalized,
case-folded or escaped. EOS gets address zero. When a character spans several
tokens, each token contributes only the bytes it actually covers, and
generation computes addresses the same way.

The hash is a polynomial over 257, so `memory_table_size` must not be a
multiple of 257. Equal byte spans produce equal addresses regardless of how a
tokenizer splits them.

## Learn the mechanism

`klab learn describe lexical-engram` and `byte-engram` walk through each part
with tensor shapes; see
[architecture](architecture.md#explore-a-mechanism-before-training). To use
n-gram repeats to speed up generation instead, see
[speculative decoding](speculative-decoding.md).
