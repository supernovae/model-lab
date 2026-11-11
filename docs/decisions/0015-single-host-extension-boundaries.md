# ADR 0015: Keep the core local while extending mechanisms explicitly

## Status

Accepted for the per-experiment execution boundary; the original exclusion of remote queues was superseded by the runtime/staging/independent-workers amendment. The decision text below is retained as historical context.

**Current scope:** one host and private optimizer state per experiment, on one device or, with `runtime.secondary_device_indices`, on several same-backend devices of that host through [in-process microbatch replicas](../gradient-accumulation.md#several-devices-on-one-host), with implemented local/SSH scheduling of whole independent experiments, explicit matrices, leases, and verified local aggregation. Native MLX and HIP sparse attention are also implemented with bounded measurements; HIP execution is verified on one RX 7900 XTX. No distributed process group, expert sharding, shared optimizer, or all-to-all training is implemented. Actual CUDA/XPU and real three-host gates remain separately blocked; see [workers](../workers.md), [sparse attention](../sparse-attention.md), and [the current feature roadmap](../../ROADMAP.md#where-work-is-tracked).

## Context

Kybern Model Lab now includes the dense decoder baseline, local MoE, several attention reference paths, token/byte/portable Engram memory, local checkpoint lifecycle, and optional MLX dense training. The project needs a clear public boundary while new Engram variants and scale experiments are added: breadth must not silently turn the reference laboratory into a distributed systems framework or reinterpret prior run artifacts.

## Decision

Keep the supported core to one process on one host using one correctly detected CPU or accelerator. PyTorch remains canonical; MLX is an optional, separate local Metal engine. No remote worker service, queue, distributed process group, expert sharding, all-to-all dispatch, or homogeneous data-parallel training is part of the current runtime.

Extend the laboratory through explicit, concrete configuration choices. A new Engram mechanism must document causal inputs, address identity, trainable/frozen state, adapter behavior, artifact and checkpoint compatibility, parameter accounting, and diagnostics. Existing memory modes retain their serialized semantics. Scale experiments remain explicit configurations with recorded source, tokenizer, device, sequence length, budget, optimizer, and seed; a combined smoke run proves compatibility, not a performance or quality result.

## Consequences

The repository can grow in architectural depth without invalidating earlier evidence or adding implicit scheduling behavior. Native sparse kernels, target-hardware acceptance, and distributed work remain separately visible in the deferred roadmap. Documentation must distinguish implemented reference behavior, measured observations, and deferred capabilities.

## Related decisions

- [ADR 0001](0001-dense-first.md) establishes the decoder baseline.
- [ADR 0002](0002-local-moe-routing.md) keeps MoE routing local.
- [ADR 0003](0003-token-ngram-memory.md) establishes the first Engram contract.
- [ADR 0013](0013-combined-architecture.md) defines explicit mechanism composition.
