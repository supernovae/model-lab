# Sparse attention

Lab has two sparse attention kinds. Use `local_selected` for long context and
for training at scale; `block_sparse` is the per-position reference selector
with native ROCm and Metal kernels.

## Local plus selected blocks

```yaml
model:
  max_seq_len: 32768
  ffn: moe            # or dense: attention and feed-forward kinds are independent
attention:
  kind: local_selected
  block_size: 64      # tokens per block
  window_size: 256    # always-visible previous tokens; a multiple of block_size
  selected_blocks: 8  # further-back blocks chosen per head
  latent_dim: 128     # optional: MLA latent keys and values
  selector: learned   # optional: train the block summaries (default mean_key)
```

Each query attends to its own block (causally), the `window_size / block_size`
blocks before it, and `selected_blocks` earlier blocks. Selection is per batch
element and head, shared by the queries of one block: the block's first query is
scored against the mean key of every block before the local window, and the top
blocks are gathered. Because that first query sees no later token, the layer is
causal at every position. Selection is a top-k choice and carries no gradient of
its own; the query and key projections it uses are trained through attention.

Cost per layer is `T * (window_size + (selected_blocks + 1) * block_size)` score
cells plus one gathered copy of the visible keys and values, and
`(T / block_size)^2` block scores computed in bounded chunks. No `[T, T]` tensor
and no per-position loop exist. When the local window and selected blocks cover
every earlier block, the output equals dense attention (or MLA with
`latent_dim`) with the same weights, which is how the implementation is tested.

### Block selectors

`selector: mean_key` (the default) summarizes a block by its mean key. Nothing is
learned for selection.

`selector: learned` adds a coarse attention branch. Every query also attends to
one learned summary per complete earlier block, and a per-head gate adds that
result to the output. A summary is a learned weighting of the positions in a
block, applied to keys and values. Because the summaries now contribute to the
output, they receive gradient from the loss, and block selection reuses them.
The weighting starts uniform and the gate at zero, so a fresh layer computes
exactly what `mean_key` computes, and a `mean_key` checkpoint can be promoted by
adding the two new tensors (`pool_logits`, `coarse_gate`). The branch scores
`T / block_size` summaries per query. Whether learned selection improves quality
over `mean_key` or approaches dense attention has not been measured.

### Generation

Generation uses an incremental cache. Beside keys and values it stores one
summary per completed block and each block's selection, made when the block's
first token arrives. A new token then reads only its local window and selected
blocks. Cached, prefix-recomputed and draft-and-verify decoding produce the same
tokens. On a 4-layer MoE model with a 16,384-token prompt on CPU, a new token
took 5.7 ms with the cache against 1.32 s for one prefix recomputation.

Limits today:

- Grouped-query heads and the MLX engine are not supported.
- Quality on long-range dependencies is unmeasured (see the comparison below).
  Treat `selected_blocks` and `window_size` as settings to measure, not as tuned
  defaults.
- With training retained for backward, each layer holds its gathered keys and
  values: about twice dense fused attention's memory per token on XPU. Enable
  activation checkpointing for deep models or very long context.
- Measured on one accelerator type (Intel Arc Pro B70, XPU). CUDA is unmeasured.

`configs/smoke_local_selected_cpu.yaml` and
`configs/smoke_moe_sparse_latent_cpu.yaml` are CPU smokes.
`configs/long_context_32k_moe_sparse_latent.yaml` trains a small MoE model with
latent keys and values and the learned selector at 32,768 tokens of context on
CPU. On a 24-core workstation its eight single-threaded updates took about 24 s
each with a 12.3 GiB peak, against a 14.2 GiB estimate from `klab inspect`.
`configs/long_context_32k_dense_sdpa.yaml` is the matched dense baseline for an
accelerator: the same model with fused dense attention.

### Measured against dense fused attention

One Intel Arc Pro B70 (`--runtime b70-xpu`), bf16, the 12.0M-parameter 4-layer
MoE model of the 32k example, synthetic stories, one run per row. Step time is
the median after warmup; memory is the device's peak allocated bytes.

| Context | Tokens per update | Attention | Step | Tokens/s | Peak memory |
| --- | --- | --- | --- | --- | --- |
| 8,192 | 32,768 | dense `sdpa` | 205 ms | 159,700 | 4.31 GiB |
| 8,192 | 32,768 | `local_selected` | 307 ms | 106,900 | 9.11 GiB |
| 8,192 | 32,768 | `local_selected`, latent, learned | 297 ms | 110,400 | 8.69 GiB |
| 32,768 | 32,768 | dense `sdpa` | 525 ms | 62,400 | 4.33 GiB |
| 32,768 | 32,768 | `local_selected` | 308 ms | 106,500 | 9.14 GiB |
| 32,768 | 32,768 | `local_selected`, latent, learned | 314 ms | 104,400 | 9.28 GiB |
| 131,072 | 131,072 | dense `sdpa` | 7.82 s | 16,800 | 16.79 GiB |
| 131,072 | 131,072 | `local_selected`, latent, learned, activation checkpointing | 2.24 s | 58,600 | 11.85 GiB |

What this shows:

- **Time:** `local_selected` costs the same per token at 8k and 32k. Dense is
  1.5 times faster at 8k, 1.7 times slower at 32k and 3.5 times slower at 131k.
  The crossover is between 8k and 32k for this model.
- **Memory:** dense fused attention on XPU keeps no `[T, T]` matrix; its peak
  follows tokens per update, not context. `local_selected` used about twice
  dense's memory per token. At 131k without activation checkpointing the memory
  planner refused the `local_selected` run before training; with it the run
  used less than dense.
- **Estimate:** `klab inspect` planned 5.4 GiB for dense at 32k (4.33 measured)
  and 12.9 and 14.2 GiB for the two `local_selected` rows (9.14 and 9.28
  measured).

### Quality against matched dense runs

Four `klab try` comparisons on the same B70: 8,192-token context, the same
model, seed and token budget, dense `sdpa` as the baseline. Both corpora place
the information a prediction needs further back than the 256-token local window:
*recall* states 48 key-value facts, then filler, then queries; *copy* repeats a
random passage after filler.

| Corpus | Tokens | Dense | `local_selected` | `local_selected`, learned |
| --- | --- | --- | --- | --- |
| recall | 39.3M | 1.411 | 1.314 | 1.314 |
| copy | 26.2M | 1.221 | 1.155 | 1.195 |

Values are held-out loss. Sparse attention was not worse than dense at a matched
budget, with one seed per arm. That is all these runs support. Every arm,
dense included, predicted the tokens that depend on distant content at chance
(1.68 nats per token on recall queries and 1.84 on the copied passage, the
entropy of their random digits), so the loss differences come from the filler
text. The runs do not show that block selection retrieves distant content, and
the difference between selectors is within what one seed can produce. A longer
budget or a real long-document corpus, with paired seeds, is still needed before
claiming `local_selected` matches dense on long-range tasks.

## Block-sparse reference

`attention.kind: block_sparse` is a causal selector. It groups only already-available keys into fixed-size blocks, scores compressed mean-key representations against each query, and selects the highest-scoring blocks. The PyTorch reference gathers original K/V tokens; native HIP attends to selected tokens directly. Neither path forms a dense `[B, H, T, T]` score matrix.

The selector remains a PyTorch reference loop on all devices. On ROCm, selected-block FP32 attention dispatches to a native HIP library JIT-built with `hipcc`: online-softmax forward plus query-owned and key-owned backward consume the shared batch/head membership mask directly, without gathering K/V or forming token-square scores. First use requires `hipcc` and the matching ROCm SDK runtime; head dimensions are limited to 2048. This accelerates the attention operation only; selection remains high-level PyTorch, and no end-to-end speedup is claimed.

## Local HIP component measurement

On the RX 7900 XTX (`gfx1100`), a fixed-mask attention-only forward/backward run at `B=2, H=4, T=128, D=32`, block size 16, and two selected blocks measured 0.829 ms for HIP versus 67.809 ms for the eager PyTorch selected-block reference (one warmup, median of 12 synchronized runs). The measurement excludes block selection and projections, covers one shape, and does not establish end-to-end model speedup.

## Optional MLX implementation

`MLXBlockSparseAttention` preserves the same causal block-selection and
batch/head-union semantics. MLX array operations rank causal mean-key blocks;
three custom Metal kernels perform FP32 online-softmax forward, query
gradients, and key/value gradients. The kernels access original K/V directly:
no gathered K/V copies or token-square softmax intermediates. Queries own
forward/dQ rows; keys own dK/dV rows, so backward needs no atomic accumulation.
The custom rule supports first-order training and block recomputation.

`selected_blocks` limits each batch/head query's choice, not the shared union.
The union can contain up to
`min(ceil(T / block_size), B * H * selected_blocks)` blocks per position.
Disagreement can approach dense attention work. Block-index scores still have
shape `[B,H,T,ceil(T/block_size)]`, and key-owned backward scans the compact
membership mask. This is not a claim of subquadratic total work or superiority
to an optimized dense-attention kernel.

### Measured Metal attention component

Apple M3 Pro, MLX 0.32.2, FP32, batch 2, hidden width 64, 4 heads, block size
16, and 2 selected blocks per query. Identical weights/inputs; output, input
gradients, and all projection gradients checked against the PyTorch MPS loop
reference (`rtol=1e-3`, `atol=1e-4`) before timing. Three warmups and five
synchronized forward/backward samples; medians below exclude upload and
memory-observer overhead.

| Sequence | Native Metal median | MPS loop-reference median | MLX native peak bytes |
| ---: | ---: | ---: | ---: |
| 32 | 1.205 ms | 114.980 ms | 626,108 |
| 128 | 1.435 ms | 432.850 ms | 2,334,452 |
| 512 | 3.184 ms | 1,733.562 ms | 9,813,716 |

These are attention-component measurements, not optimizer, full-training, or
end-to-end generation speedups. Host timing varies: the preceding operator
baseline measured the same MPS loop reference at 1,125.005 ms for length 512.
Do not infer a stable cross-run latency ratio. At that length, the earlier
gathered-operator path used 417,926,100 MLX native peak bytes under the same
measurement procedure. Both MLX peaks use the same allocator high-water
method; MPS sampled peaks are lower bounds and are not directly comparable.

Raw samples, runtime identities, source identities, errors, and limitations:
`artifacts/benchmarks/mlx_sparse_metal_2026_09_22.json` and
`artifacts/benchmarks/mlx_sparse_operator_2026_09_22.json` are the recorded
artifact locations; these payloads are not present in this checkout.
Regressions cover prefix causality, partial blocks, batch/head unions, input
and projection gradients, a 40-channel head spanning the 32-lane SIMD width,
and recomputed sparse blocks. Full MLX checkpoint/inference lifecycle
acceptance is tracked separately from this attention component.

```sh
uv run --locked --extra cpu klab train --runs-dir klab-work/runs configs/smoke_sparse_cpu.yaml --run-id sparse-smoke
```

The [`sparse-attention` lesson](research/lesson-paths.md) scaffolds the block-16/select-2 PyTorch configuration and exposes selector diagnostics without a campaign. The [memory/selection study](research/engram-sparse-budget.md) keeps endpoint comparisons and observed deltas separate from Python timing; batch/head selection unions can exceed a per-head budget.
