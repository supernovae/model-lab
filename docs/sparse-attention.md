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
- No quality comparison with dense attention has been run, for either selector.
  Treat `selected_blocks` and `window_size` as settings to measure, not as tuned
  defaults.
- With training retained for backward, each layer holds its gathered keys and
  values; enable activation checkpointing for deep models at long context.
- No accelerator measurement exists yet.

`configs/smoke_local_selected_cpu.yaml` and
`configs/smoke_moe_sparse_latent_cpu.yaml` are CPU smokes.
`configs/long_context_32k_moe_sparse_latent.yaml` trains a small MoE model with
latent keys and values and the learned selector at 32,768 tokens of context on
CPU. On a 24-core workstation its eight single-threaded updates took about 24 s
each with a 12.3 GiB peak, against a 14.2 GiB estimate from `klab inspect`.

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
