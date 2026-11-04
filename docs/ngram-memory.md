# Context-gated token memory and dense-checkpoint adaptation

Token memory remains jointly learned model parameters, not a repository index.
Adding documents does not populate tables without training. The default
`hidden_gate` / legacy addressing configuration retains existing checkpoint
semantics. New fields default away from serialized legacy configurations.

`memory_variant: context_key` concatenates order/head lookups, projects keys and
values separately, gates with RMS-normalized hidden/key similarity, and optionally
refines gated values with depthwise causal convolution. `memory_gate_transform:
identity` selects the paper equation; `signed_sqrt` selects the demo's clamped
signed-square-root transform before sigmoid. `memory_conv_kernel_size: 4` uses
maximum n-gram order as dilation. Projections are biasless in this implementation.

For explicit prime/XOR addressing, set `memory_addressing: prime_xor_v1`,
`memory_hash_seed`, and `memory_table_sizes`: distinct primes in
layer/order/head order. `memory_table_size` must equal the first entry.
`memory_ngram_orders: [2, 3]` and `memory_hash_heads: 2` require four primes per
port. Coefficients use versioned SplitMix64 expansion, with the **original**
model vocabulary bound held fixed across mapping ablations. This differs from
the demo's NumPy RNG and compressed-vocabulary bound; neither seeded addresses
nor the single-stream backbone claim bitwise demo reproduction.

`memory_injection: after_block` accepts either a zero-based `memory_block_index`
or sorted unique `memory_block_indices`. For example `[1, 14]` means after the
second and fifteenth decoder blocks and requires a model with at least fifteen
blocks. Each port has independent tables, layer-seeded coefficients and bounded
request-local convolution history. Existing single-port state names stay intact;
additional ports use `extra_memories.<block>.*`.

## Bind an address-only vocabulary map

```sh
sparselab tokenizer memory-map "$TOKENIZER" --mode deepseek_v1 \
  --output "$WORK/map.json" --bind-config "$WORK/prime-base.yaml" \
  --config-output "$WORK/prime-canonical.yaml"
```

This never fits or downloads a tokenizer. The native derivation embeds the full
validated mapping and its tokenizer/map hashes, anchoring paths through the
existing config derivation path. `identity` is the code-sensitive control;
`deepseek_v1` mirrors the pinned demo normalizer; `deepseek_safe_v1` additionally
isolates special tokens and undecodable byte fragments. Both normalized modes
can merge code-relevant case, whitespace and accents. Input embeddings, output
labels and decoding retain original IDs. Preparation and model loading bind the
map to actual tokenizer bytes. Raw padding IDs are mapped too.

`memory_independent_init: true` isolates memory initialization from the CPU RNG
used for trunk construction. Dense and memory arms then retain identical trunk
tensors and future CPU RNG state at a paired seed. Native construction is CPU
first. This setting is explicit so old initialization behavior remains available.

## Reuse a verified dense checkpoint

```sh
sparselab train "$TARGET" --promote "$IMMUTABLE_GENERATION" \
  --add-ngram-memory --run-id NEW_CHILD_RUN
```

The native loader verifies the source generation and its parent run's artifact
closure. It requires exact tokenizer identity and unchanged backbone/attention,
copies trunk weights, initializes new memory tables normally, and zeroes only the
memory output projections. Initial logits match the source; output weights learn
first, allowing gradients into tables afterward. The source is never edited.
A standalone copied generation still needs its matching verified parent closure;
do not point this command at a mutable latest pointer or invent missing metadata.

This is **weights-only promotion**, with a new optimizer, scheduler, seed, data
cursor and token counters, plus explicit parent checkpoint/step/token lineage.
It is not exact resume under changed architecture/data. After creation, ordinary
`--resume` restores the child's optimizer and counters exactly.

Joint training is the default. To freeze the backbone, explicitly select all
canonical memory tensor names in `training.trainable_parameters`; wildcards are
not supported. Each memory output projection must remain trainable. Frozen
backbone parameters still require backward computation through downstream layers.
Compare dense continuation, joint memory continuation and memory-only adaptation
on the same revised diet, target budget and retention panels before considering
a restart. Existing dense training need not finish before such a separately
scheduled decision.

For PyTorch AdamW, `optimizer.memory_table_lr_multiplier` (default 1) and
`optimizer.memory_table_weight_decay` (default inherited) affect tables only.
Checkpoint verification binds those groups and their scheduled learning rates.
`training.stable_steps` inserts a plateau between warmup and cosine decay;
zero preserves the existing schedule. Existing `optimizer.decay_steps` optionally
sets its horizon; a later floor plateau is distinct from the pre-decay stable
phase. WSD `--extend-budget` is currently unsupported; choose a new declared
promotion branch when changing the policy. A 5x table LR or zero table decay is
an experiment, not a recommended result. Gradients and AdamW states are dense.

## Inspect the actual packed addresses and work outcomes

```sh
sparselab research corpus mine --prepared-train "$PREPARED" \
  --model-config "$TARGET" --memory-block 1 \
  --sequence-length 256 --max-blocks 128 --output "$WORK/addresses.json"
```

A multiport model requires explicit port selection. The report binds the full
model identity, selected port and tokenizer; it distinguishes mapping-equivalent
raw keys from distinct canonical keys sharing a hash bucket. It mines bounded
verified TRAIN input blocks, including EOS/masked context, not a shuffled or
exposure-weighted training schedule. The legacy `collision_count` model metric
means repeated accesses. Multiport aggregate training metrics cover all ports;
`engram/port/<block>/*` exposes individual ports. Explorer and parameter inventory
include every table and mapping buffer. No memory/throughput advantage is implied.

`capability code-work` provides a first optional Inspect Docker function-task
adapter. It records unavailable execution explicitly when Inspect 0.3.287 or the
pinned local image is missing; it never installs or pulls either. Fixture tests
validate classification/provenance, not actual container isolation. Tests are
withheld from prompts but accessible to generated code inside the container, so
this is not an adversarial anti-cheating evaluator. Real integration validation
and repository patch evaluation remain outstanding.

The companion Workshop `experiments/devmind/ngram-memory` contains controlled
fusion/addressing studies, code/docs source review, continuation design and
retention/work panels. No active run is renamed or automatically continued.
