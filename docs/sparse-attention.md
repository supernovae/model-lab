# Block Sparse Attention

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
