# Activation recomputation

Activation recomputation (often called activation checkpointing) saves memory during backward by retaining less of the forward graph and running selected forward work again. It is **not** a durable disk checkpoint, and enabling it does not make an interrupted run resumable without a verified checkpoint generation.

## Current PyTorch behavior

Enable the supported transformer-block strategy explicitly:

```yaml
runtime:
  memory:
    activation_checkpointing:
      enabled: true
      strategy: transformer_block
```

During grad-enabled PyTorch training, each decoder block runs through `torch.utils.checkpoint.checkpoint` with non-reentrant execution and RNG-state preservation. The block returns both hidden state and its differentiable auxiliary loss, so MoE balance loss remains part of backward rather than being read from mutable diagnostic state. Inference and ordinary non-gradient forward calls use the normal path.

Recomputation suppresses diagnostic snapshot writes during backward replay
without changing the mathematical forward. Scalar diagnostics stay detached
on-device until update reporting. Full MoE snapshots include only supervised
positions, not masked prompt/padding rows. `logging.architecture_diagnostics: full`
explicitly requests larger current-microbatch snapshots and must be included in
memory decisions.

`recompute/block_call_ratio` records extra PyTorch block-forward calls per
executed, nonempty microbatch. Wholly masked chunks do not dilute it. This is a
mechanism counter, not a memory or speed result. A resource claim needs matched
observed runs with the same data, weights, runtime, precision, microbatch, and
diagnostics.

## Memory and tradeoff

The conservative estimator changes its retained-activation model when transformer-block recomputation is enabled: it retains block inputs plus an active-block allowance rather than every block's saved intermediates. Attention working allowance remains conservative; block-sparse reference attention does not earn an invented sparse reduction.

Recomputation can reduce retained activation memory while adding backward-time computation. It does not reduce resident weights, gradients, optimizer state, or disk checkpoint size. Combine it with smaller microbatches/accumulation only deliberately: each adjustment changes resource behavior and potentially timing.

## Boundaries

- **Disk checkpoints** serialize verified model/training state for restart; see [checkpointing](checkpointing.md).
- **Gradient accumulation** combines microbatch gradients into a single committed update; see [gradient accumulation](gradient-accumulation.md).
- **Activation offload** copies autograd-saved tensors to host storage rather than rerunning computation; see [offload](offload.md).
- **KV caching** is an inference/decode technique. Its request-local state is not used by the training path or credited to training memory.

MLX uses `mx.checkpoint` on a pure callable whose block parameters and hidden
state are explicit arguments; it does not rely on an implicit mutable-module
closure for parameter gradients. Dense and native sparse recomputation have
numerical gradient/update coverage.

The planner proposes recomputation explicitly; it never silently changes the
requested config. The recorded focused generation/import/recomputation sweep passed
35 tests, and the [actual MPS offload comparison](offload.md#actual-mps-comparison)
verified composition with recomputation and final-parameter parity. The
[host continuation record](https://github.com/supernovae/model-lab/blob/9d4c840ef1f102becf27532c2548c505c2ed35a6/artifacts/acceptance/host_cli_2026_09_22.json)
also exercises recomputation in both engines.
