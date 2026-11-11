# Engram Memory

Commands below use the [activated, provisioned environment](../README.md#first-run).

SparseLab's token Engram is a causal N-gram lookup table with a latent value width independent of the backbone hidden width. Each position hashes only its current and earlier token IDs, retrieves a latent table vector, projects it to model width, and mixes it with the hidden state through a sigmoid gate:

$$h'_t = h_t + \sigma(g(h_t))\,A\,M[a(x_{\leq t})].$$

Set `model.memory: ngram` with `memory_table_size`, `memory_ngram_size`, and `memory_dim`. Optional `memory_ngram_orders` and `memory_hash_heads` create independent causal lookup streams whose projected values are averaged before gating. The byte-addressed variant (`memory: byte`) receives causal raw UTF-8 suffix addresses prepared alongside tokens; this is an addressing experiment, not evidence of tokenizer-independent knowledge transfer.

Each forward pass records lookup count, unique buckets, collisions, bucket reuse rate, table utilization, largest-bucket fraction, mean gate activation, retrieved-vector norm, and hidden-state norm. High reuse can arise from a small table or repeated corpus structure; it is not by itself a learned-memory success signal. Inspect language loss and withheld-fact evaluation separately.

```sh
sparselab train --runs-dir sparselab-work/runs configs/smoke_memory_cpu.yaml --run-id token-engram-smoke
sparselab train --runs-dir sparselab-work/runs configs/smoke_byte_memory_cpu.yaml --run-id byte-engram-smoke
```

Multi-order/multi-head streams preserve causal addressing and allocate one table per stream. Per-run metrics persist aggregate lookup count, unique buckets, collisions, reuse rate, table utilization, and gate mean, plus the same six metrics under `engram/stream_N/*` for each stream. The Architecture dashboard renders those persisted series. Portable packages and frozen adapter training are available through `memory: portable`; the withheld-fact control matrix currently yields no transfer success, so it remains a negative experiment result.

## Learn and compare placement

The [`lexical-engram` lesson](research/lesson-paths.md) provides a standalone final-injection configuration and initialized shape/diagnostic probe. Lexical `memory_injection` supports final and embedding placement; token memory also supports the explicit after-block site described below. The [placement study](research/engram-placement.md) remains historical evidence. Separately, verified semantic retrieval and direct `DenseLM` adapters support embedding, after-block, and final sites, including multiple named/multipack attachments; see [Verified semantic memory](research/semantic-memory.md). Lookup diagnostics remain mechanism observations, not score or transfer proof.


## Adding an Engram variant

Treat a new Engram as an explicit architecture and artifact contract, not as a hidden replacement for an existing mode. Define its causal address inputs, table/adapter ownership, checkpoint compatibility, parameter accounting, and persisted diagnostics before adding it to `model.memory`. Preserve `none`, `ngram`, `byte`, and `portable` semantics unchanged so existing runs remain interpretable. A new addressing scheme must state what equality it preserves and what it does **not** establish; address collisions, lookup reuse, or a successful smoke run are not evidence of retrieval quality or knowledge transfer.

## Context-key fusion ablation

For token memory, `memory_variant: context_key` concatenates per-order/head
lookups and projects separate keys and values. Its scalar gate is the sigmoid
of the RMS-normalized hidden/key dot product divided by sqrt(hidden width),
following equations (3)–(4) of [Engram v2](https://arxiv.org/html/2601.07372v2#S2).
`memory_conv_kernel_size: 4` adds equation (5)'s RMS normalization, depthwise
causal convolution (dilation = maximum n-gram order), SiLU and residual; zero
omits refinement. Convolution parameters use PyTorch's Conv1d initialization;
linear/embedding weights retain Lab's normal initialization. All are jointly
trained with the configured optimizer. Convolved memory rejects allocation
masks because masked values must not leak through its temporal history.

Use `memory_injection: after_block` with a zero-based `memory_block_index` for
one token-memory attachment after that decoder block and before the next block
(or the final norm). Full and cached paths use the same site. Cached convolution
history is bounded, request-owned, counted in cache bytes and reset on rollover.
Explorer residual-post measurements include an attachment between blocks.

The default `hidden_gate` keeps averaged projected streams and a hidden-only
gate. Both variants retain Lab's existing token hashes and identity token mapping.
This is a fusion/placement ablation, not an exact paper reproduction: it omits
canonical token compression, the paper's XOR/prime-table construction and
multi-branch backbone. It uses equation (4), not the demo's signed-square-root
transform. Changing those choices requires explicit architecture/address identity.
Standard embeddings allocate dense gradients and optimizer state; one active
row per stream is not the total training-memory cost.

Aggregate occupancy counts independent table namespaces. The historical metric
`collision_count` is repeated bucket accesses, including identical keys; use
[packed train mining](capabilities.md) to separate distinct-key collisions from
repeat accesses. For `context_key`, stream `value_norm` describes raw lookup
vectors and aggregate `value_norm` describes their projected concatenation;
`hidden_gate` stream norms describe projected vectors. Compare gates and losses,
not unlike stream norms.
